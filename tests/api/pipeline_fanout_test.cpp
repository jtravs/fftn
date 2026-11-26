// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Fan-out on row pipelines (fftn/pipeline.hpp, DESIGN.md 12.2; task 5.2e): a pipeline running
// `fanout` inverse transforms on each line's spectrum, copy k with its own middle factor
// (fanout_factor_dist k elements on) and its own output (fanout_out_dist k elements on). Each copy
// is checked against its own inverse: the line transformed once in quadruple precision, times copy
// k's kernel, transformed back exactly, so F separate inverses per line. The copies' outputs lie
// apart with gaps between them and between lines, filled with canaries that must keep their bits;
// the input outside its runs holds NaN, so a read of one would show. Kernels shared or per line,
// fused into the inverse's first pass or in a separate pass, epilogue factors fused or separate,
// accumulation, stored runs (5.2d's, on both sides), strides and copy distances of either sign,
// the <C, C, C> and <T, C, T> kinds and the depth-first pair (forced), on every instruction set in
// both precisions; serial, pieces of the batch, teams (completion::all and own) and an executor
// bitwise alike. Expectations hold on every target: none assumes a target's radices or strategy
// unless the test forces it.

#include "core/pipeline.hpp"
#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

namespace d = fftn::detail;
using fftn::factor_kind;
constexpr factor_kind real = factor_kind::real;
constexpr factor_kind cplx = factor_kind::complex;
constexpr factor_kind none = factor_kind::none;

template <class V> bool same_bits(const V& a, const V& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

struct run_spec {
  std::size_t logical = 0;
  std::size_t stored = 0;
  std::size_t count = 0;
};

struct fan_case {
  const char* name = "";
  std::size_t n = 16;
  std::size_t count = 3;
  std::size_t fanout = 3;
  bool backwards = false;        // copies at negative distances
  bool between = false;          // copies between lines, inside the batch's span (not backwards)
  std::ptrdiff_t factor_gap = 1; // elements between copies' kernels; -1: one kernel for all copies
  std::vector<run_spec> in{};    // empty: the input line stored whole
  std::vector<run_spec> out{};   // empty: the output line stored whole
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t out_stride = 1;
  factor_kind mid = cplx;      // the kernel's kind (none: no kernel)
  std::ptrdiff_t mid_axis = 1; // its stride along the bins
  bool per_row = false;        // a kernel per line
  double mid_scale = 0.0;      // 0: 1/n
  double epi_scale = 1.0;
  bool accumulate = false;
  factor_kind pro = none;           // a prologue factor (stride 1, one for every line)
  factor_kind epi = none;           // an epilogue factor, every copy's
  const char* expect = "fan-out: "; // in describe()
};

std::size_t extent_of(const std::vector<run_spec>& runs, std::size_t n) {
  if (runs.empty()) {
    return n;
  }
  std::size_t e = 0;
  for (const run_spec& r : runs) {
    e = std::max(e, r.stored + r.count);
  }
  return e;
}

// One case's buffers, description and reference; Elem is the pipeline's In and Out type.
template <class T, class Elem> struct fan_run {
  static constexpr bool real_kind = std::is_same_v<Elem, T>;
  using C = std::complex<T>;
  using desc_t = fftn::pipeline_desc<Elem, C, Elem>;

  fan_case c;
  fftn::isa target;
  std::size_t bins = 0;
  std::ptrdiff_t in_dist = 0;
  std::ptrdiff_t out_dist = 0;
  std::ptrdiff_t copy_dist = 0;   // fanout_out_dist
  std::ptrdiff_t factor_dist = 0; // fanout_factor_dist
  std::size_t in_base = 0;
  std::size_t out_base = 0; // element 0 of copy 0's line 0 in the output buffer
  std::ptrdiff_t kbatch = 0;
  double mid_scale = 1.0;
  desc_t desc;
  std::vector<Elem> x;
  std::vector<Elem> init; // the output buffer before execution (canaries, accumulation)
  std::vector<T> kreal;
  std::vector<C> kcomplex;
  std::ptrdiff_t kzero = 0; // copy 0's kernel coordinate 0 in its storage
  std::vector<bool> out_listed;
  std::vector<T> rpro;
  std::vector<C> cpro;
  std::vector<T> repi;
  std::vector<C> cepi;

  fan_run(const fan_case& fc, fftn::isa t, fftn::test::input_source& src)
      : c(fc), target(t), bins(real_kind ? fc.n / 2 + 1 : fc.n) {
    const std::size_t ein = extent_of(c.in, c.n);
    const std::size_t eout = extent_of(c.out, c.n);
    const auto span = [](std::size_t e, std::ptrdiff_t stride) {
      return static_cast<std::ptrdiff_t>(e) * std::abs(stride) + 2; // a gap between lines
    };
    in_dist = span(ein, c.in_stride);
    out_dist = span(eout, c.out_stride);
    // The copies' batches 3 elements apart: canaries between them. Between lines: copy k of line
    // r right after copy k - 1 of it (a gap of 2), the lines fanout copies apart.
    const std::ptrdiff_t batch_span = out_dist * static_cast<std::ptrdiff_t>(c.count);
    copy_dist = (batch_span + 3) * (c.backwards ? -1 : 1);
    if (c.between) {
      copy_dist = out_dist;
      out_dist *= static_cast<std::ptrdiff_t>(c.fanout);
    }
    const std::ptrdiff_t copies = static_cast<std::ptrdiff_t>(c.fanout) - 1;
    in_base = c.in_stride < 0 ? static_cast<std::size_t>(-c.in_stride) * (ein - 1) : 0;
    std::size_t line_base =
        c.out_stride < 0 ? static_cast<std::size_t>(-c.out_stride) * (eout - 1) : 0;
    out_base = line_base + (c.backwards ? static_cast<std::size_t>(-copy_dist * copies) : 0);
    mid_scale = c.mid_scale != 0.0 ? c.mid_scale : 1.0 / static_cast<double>(c.n);

    desc.line = {.n = c.n, .in_stride = c.in_stride, .out_stride = c.out_stride};
    desc.batches = {{.count = c.count, .in_dist = in_dist, .out_dist = out_dist}};
    for (const run_spec& r : c.in) {
      desc.runs.push_back({.axis = 0,
                           .where = fftn::side::in,
                           .logical = r.logical,
                           .stored = r.stored,
                           .count = r.count});
    }
    for (const run_spec& r : c.out) {
      desc.runs.push_back({.axis = 0,
                           .where = fftn::side::out,
                           .logical = r.logical,
                           .stored = r.stored,
                           .count = r.count});
    }
    kbatch = c.per_row ? std::abs(c.mid_axis) * static_cast<std::ptrdiff_t>(bins) : 0;
    desc.middle.scale = mid_scale;
    if (c.mid != none) {
      desc.middle.mul = {.kind = c.mid, .axis_strides = {c.mid_axis}, .batch_strides = {kbatch}};
    }
    desc.epilogue.scale = c.epi_scale;
    desc.epilogue.accumulate = c.accumulate;
    const auto side_factor = [&](factor_kind k, std::vector<T>& r,
                                 std::vector<C>& z) -> fftn::factor {
      if (k == none) {
        return {};
      }
      if (k == real) {
        r = src.reals<T>(c.n);
      } else {
        z = src.complexes<T>(c.n);
      }
      return {.kind = k, .axis_strides = {1}, .batch_strides = {}};
    };
    desc.prologue.mul = side_factor(c.pro, rpro, cpro);
    desc.epilogue.mul = side_factor(c.epi, repi, cepi);

    // The kernels: copy k's kernel factor_dist k elements after copy 0's.
    const std::ptrdiff_t along = c.mid_axis * static_cast<std::ptrdiff_t>(bins - 1);
    const std::ptrdiff_t across = kbatch * static_cast<std::ptrdiff_t>(c.count - 1);
    const std::ptrdiff_t one = std::abs(along) + across + 1;
    factor_dist = c.factor_gap < 0 ? 0 : one + c.factor_gap;
    kzero = -std::min<std::ptrdiff_t>(0, along);
    const auto ksize = static_cast<std::size_t>(one + factor_dist * copies);
    if (c.mid == real) {
      kreal = src.reals<T>(ksize);
    } else if (c.mid == cplx) {
      kcomplex = src.complexes<T>(ksize);
    }
    desc.fanout = c.fanout;
    desc.fanout_out_dist = copy_dist;
    desc.fanout_factor_dist = factor_dist;

    x = values(src, static_cast<std::size_t>(in_dist) * c.count);
    init = values(src, line_base +
                           (c.between ? static_cast<std::size_t>(out_dist) * c.count
                                      : static_cast<std::size_t>(std::abs(copy_dist)) * c.fanout));
    std::vector<bool> in_listed(x.size(), false);
    out_listed.assign(init.size(), false);
    for (std::size_t r = 0; r < c.count; ++r) {
      for (std::size_t j = 0; j < c.n; ++j) {
        if (const std::ptrdiff_t at = stored_of(c.in, j); at >= 0) {
          in_listed[in_at(static_cast<std::size_t>(at), r)] = true;
        }
        if (const std::ptrdiff_t at = stored_of(c.out, j); at >= 0) {
          for (std::size_t k = 0; k < c.fanout; ++k) {
            out_listed[out_at(static_cast<std::size_t>(at), r, k)] = true;
          }
        }
      }
    }
    const T nan = std::numeric_limits<T>::quiet_NaN();
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (!in_listed[i]) {
        x[i] = Elem(nan);
      }
    }
  }

  static std::vector<Elem> values(fftn::test::input_source& src, std::size_t n) {
    if constexpr (real_kind) {
      return src.reals<T>(n);
    } else {
      return src.complexes<T>(n);
    }
  }

  [[nodiscard]] static std::ptrdiff_t stored_of(const std::vector<run_spec>& runs, std::size_t j) {
    if (runs.empty()) {
      return static_cast<std::ptrdiff_t>(j);
    }
    for (const run_spec& r : runs) {
      if (j >= r.logical && j < r.logical + r.count) {
        return static_cast<std::ptrdiff_t>(r.stored + (j - r.logical));
      }
    }
    return -1;
  }

  [[nodiscard]] std::size_t in_at(std::size_t p, std::size_t r) const {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(in_base) +
                                    static_cast<std::ptrdiff_t>(r) * in_dist +
                                    static_cast<std::ptrdiff_t>(p) * c.in_stride);
  }
  [[nodiscard]] std::size_t out_at(std::size_t p, std::size_t r, std::size_t k) const {
    return static_cast<std::size_t>(
        static_cast<std::ptrdiff_t>(out_base) + static_cast<std::ptrdiff_t>(k) * copy_dist +
        static_cast<std::ptrdiff_t>(r) * out_dist + static_cast<std::ptrdiff_t>(p) * c.out_stride);
  }

  [[nodiscard]] const void* kernel() const {
    if (c.mid == real) {
      return kreal.data() + kzero;
    }
    return c.mid == cplx ? static_cast<const void*>(kcomplex.data() + kzero) : nullptr;
  }

  [[nodiscard]] fftn::exec_args args() const {
    const auto address = [](const std::vector<T>& r, const std::vector<C>& z) -> const void* {
      return !r.empty() ? static_cast<const void*>(r.data())
                        : (!z.empty() ? static_cast<const void*>(z.data()) : nullptr);
    };
    return {.in_factor = address(rpro, cpro),
            .mid_factor = kernel(),
            .out_factor = address(repi, cepi)};
  }

  [[nodiscard]] static std::complex<double> side_at(const std::vector<T>& r,
                                                    const std::vector<C>& z, std::size_t j) {
    if (!r.empty()) {
      return {static_cast<double>(r[j]), 0.0};
    }
    return z.empty() ? std::complex<double>(1.0, 0.0) : std::complex<double>(z[j]);
  }

  // Copy k's kernel at bin b of line r (1 without a kernel).
  [[nodiscard]] std::complex<double> kernel_at(std::size_t b, std::size_t r, std::size_t k) const {
    if (c.mid == none) {
      return {1.0, 0.0};
    }
    const auto i = static_cast<std::size_t>(
        kzero + (bins > 1 ? c.mid_axis * static_cast<std::ptrdiff_t>(b) : 0) +
        (c.count > 1 ? kbatch * static_cast<std::ptrdiff_t>(r) : 0) +
        factor_dist * static_cast<std::ptrdiff_t>(k));
    return c.mid == real ? std::complex<double>(kreal[i], 0.0) : std::complex<double>(kcomplex[i]);
  }

  [[nodiscard]] double spread() const {
    double peak = 0.0;
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t k = 0; k < c.fanout; ++k) {
      for (std::size_t r = 0; r < c.count; ++r) {
        for (std::size_t b = 0; b < bins; ++b) {
          const double m = std::abs(kernel_at(b, r, k));
          peak = std::max(peak, m);
          sum += m * m;
          ++count;
        }
      }
    }
    return sum == 0.0 ? 1.0 : peak / std::sqrt(sum / static_cast<double>(count));
  }

  [[nodiscard]] d::pipeline_view view() const {
    return {
        .kind = real_kind ? d::pipeline_kind::real_convolution : d::pipeline_kind::complex_pair,
        .prec = sizeof(T) == 8 ? d::precision::f64 : d::precision::f32,
        .first = desc.first,
        .line = desc.line,
        .batches = desc.batches,
        .runs = desc.runs,
        .prologue = d::view_of(desc.prologue),
        .middle = {.scale = desc.middle.scale,
                   .kind = desc.middle.mul.kind,
                   .axis_strides = desc.middle.mul.axis_strides,
                   .batch_strides = desc.middle.mul.batch_strides},
        .epilogue = d::view_of(desc.epilogue),
        .fanout = desc.fanout,
        .fanout_out_dist = desc.fanout_out_dist,
        .fanout_factor_dist = desc.fanout_factor_dist,
    };
  }

  // Each line's spectrum computed once, exactly; then for each copy its own inverse: times copy
  // k's kernel (rounded to double), transformed back exactly, times the scales and the epilogue's
  // factor (plus the prior output with accumulation), compared on the output's runs. Every element
  // outside the copies' runs keeps its bits.
  void check(const std::vector<Elem>& y, const std::string& describe) const {
    const std::string what = std::string(c.name) + ", n = " + std::to_string(c.n) + ", " +
                             fftn::isa_name(target) + (sizeof(T) == 8 ? " double" : " float") +
                             (real_kind ? ", real" : "");
    SCOPED_TRACE(what);
    const double scale = mid_scale * c.epi_scale;
    const fftn::test::tolerance tol =
        fftn::test::derived_tolerance<T>(c.n, fftn::test::fftn_bound(c.n)) * (1.0 + spread()) *
            (c.epi != none ? 1.8 : 1.0) +
        fftn::test::input_rounding<T>() * 6;
    fftn::test::outcome o;
    for (std::size_t r = 0; r < c.count; ++r) {
      std::vector<std::complex<double>> forward(bins);
      if constexpr (real_kind) {
        std::vector<double> line(c.n, 0.0);
        for (std::size_t j = 0; j < c.n; ++j) {
          if (const std::ptrdiff_t at = stored_of(c.in, j); at >= 0) {
            line[j] = static_cast<double>(x[in_at(static_cast<std::size_t>(at), r)]) *
                      side_at(rpro, cpro, j).real();
          }
        }
        const auto s = fftn::test::reference::r2c(line);
        for (std::size_t b = 0; b < bins; ++b) {
          forward[b] = {s[b].re.hi, s[b].im.hi};
        }
      } else {
        std::vector<std::complex<double>> line(c.n, 0.0);
        for (std::size_t j = 0; j < c.n; ++j) {
          if (const std::ptrdiff_t at = stored_of(c.in, j); at >= 0) {
            line[j] = std::complex<double>(x[in_at(static_cast<std::size_t>(at), r)]) *
                      side_at(rpro, cpro, j);
          }
        }
        const auto s = fftn::test::reference::c2c(-1, line);
        for (std::size_t b = 0; b < bins; ++b) {
          forward[b] = {s[b].re.hi, s[b].im.hi};
        }
      }
      for (std::size_t k = 0; k < c.fanout; ++k) {
        std::vector<std::complex<double>> spectrum(bins);
        for (std::size_t b = 0; b < bins; ++b) {
          spectrum[b] = forward[b] * kernel_at(b, r, k);
        }
        const std::string where =
            what + ", line " + std::to_string(r) + ", copy " + std::to_string(k);
        std::vector<Elem> got;
        if constexpr (real_kind) {
          const auto back = fftn::test::reference::c2r(c.n, spectrum);
          std::vector<fftn::test::exact_real> want;
          for (std::size_t j = 0; j < c.n; ++j) {
            if (const std::ptrdiff_t at = stored_of(c.out, j); at >= 0) {
              const std::size_t i = out_at(static_cast<std::size_t>(at), r, k);
              double v = back[j].hi * side_at(repi, cepi, j).real() * scale;
              if (c.accumulate) {
                v += static_cast<double>(init[i]);
              }
              want.push_back(fftn::test::exact_from(v));
              got.push_back(y[i]);
            }
          }
          o.within(fftn::test::compare<T>(std::span<const T>(got), want), tol, where);
        } else {
          const auto back = fftn::test::reference::c2c(+1, spectrum);
          std::vector<fftn::test::exact_complex> want;
          for (std::size_t j = 0; j < c.n; ++j) {
            if (const std::ptrdiff_t at = stored_of(c.out, j); at >= 0) {
              const std::size_t i = out_at(static_cast<std::size_t>(at), r, k);
              std::complex<double> v = std::complex<double>(back[j].re.hi, back[j].im.hi) *
                                       side_at(repi, cepi, j) * scale;
              if (c.accumulate) {
                v += std::complex<double>(init[i]);
              }
              want.push_back(fftn::test::exact_from(v));
              got.push_back(y[i]);
            }
          }
          o.within(fftn::test::compare<T>(std::span<const C>(got), want), tol, where);
        }
      }
    }
    EXPECT_TRUE(o.passed()) << o.detail() << describe;
    std::size_t touched = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      if (!out_listed[i] && !same_bits(std::vector<Elem>{y[i]}, std::vector<Elem>{init[i]})) {
        ++touched;
      }
    }
    EXPECT_EQ(touched, 0U) << what << ": canaries between or around the copies written\n"
                           << describe;
  }
};

// Built with construction knobs (settings), executed serially (against the reference where
// `reference`), in two pieces of the batch and, with `threads`, by teams of three (completion::all
// and own) and on a pool of three, all bitwise alike; the input and the kernels left as they were.
template <class Run>
std::string run_core(const Run& run, const d::build_settings& settings, bool threads,
                     bool reference = true) {
  const fan_case& fc = run.c;
  const std::shared_ptr<const d::plan_core> core =
      d::make_pipeline_core(run.view(), {.force_isa = run.target}, settings);
  const std::string describe = d::describe(*core);
  EXPECT_NE(describe.find(fc.expect), std::string::npos)
      << fc.name << ": no '" << fc.expect << "' in\n"
      << describe;
  const fftn::exec_args args = run.args();
  fftn::workspace ws(d::workspace_bytes(*core, 3));
  const auto x0 = run.x;
  const auto kr = run.kreal;
  const auto kc = run.kcomplex;
  auto xs = run.x;
  const auto in = [&] { return xs.data() + run.in_base; };

  auto serial = run.init;
  d::execute_all(*core, in(), serial.data() + run.out_base, ws, args);
  if (reference) {
    run.check(serial, describe);
  }
  EXPECT_TRUE(same_bits(xs, x0)) << fc.name << ": input written";
  EXPECT_TRUE(same_bits(run.kreal, kr) && same_bits(run.kcomplex, kc))
      << fc.name << ": kernel written";

  auto pieces = run.init;
  const std::size_t half = fc.count / 2;
  d::execute_core(*core, 0, half, in(), pieces.data() + run.out_base, ws, args);
  d::execute_core(*core, half, fc.count - half, in(), pieces.data() + run.out_base, ws, args);
  EXPECT_TRUE(same_bits(pieces, serial)) << fc.name << ": pieces of the batch differ\n" << describe;
  if (!threads) {
    return describe;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    auto teamed = run.init;
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        d::execute_team(*core, in(), teamed.data() + run.out_base, fftn::team{.rank = r, .size = 3},
                        ws, done, args);
      });
    }
    d::execute_team(*core, in(), teamed.data() + run.out_base, fftn::team{.rank = 0, .size = 3}, ws,
                    done, args);
    for (std::thread& m : members) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << fc.name << ": a team differs\n" << describe;
  }
  const fftn::thread_pool pool(3);
  auto pooled = run.init;
  d::execute_on(*core, in(), pooled.data() + run.out_base, pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << fc.name << ": the executor differs\n" << describe;
  return describe;
}

template <class T, class Elem>
std::string run_case(const fan_case& fc, fftn::isa t, bool threads,
                     const d::build_settings& settings = {}, bool reference = true) {
  fftn::test::input_source src(0x5E2EULL + fc.n * 131 + fc.count * 7 + fc.fanout +
                               fc.out.size() * 3);
  return run_core(fan_run<T, Elem>(fc, t, src), settings, threads, reference);
}

// The copies shared by a team, each on a member's own scratch: the copy writes its output itself.
constexpr const char* shared = "a team shares a line's copies";
// The copies in the workspace block's second region, one after another.
constexpr const char* serial = "a team runs a line's copies one after another";

// Fan-outs on a line of n: the kernel fused (complex, stride 1) or in a separate pass (real or
// strided), shared or per line, copies apart or sharing one kernel, at negative distances, with
// strides of either sign; epilogue factors fused or in a pass, accumulation, scales, prologue
// factors, no kernel at all; and (n at least 8) stored runs on either side.
std::vector<fan_case> fan_sets(std::size_t n, bool real_kind) {
  std::vector<fan_case> sets = {
      {.name = "a kernel per copy", .n = n, .expect = shared},
      {.name = "copies between lines", .n = n, .count = 4, .between = true, .expect = shared},
      {.name = "copies between strided lines, accumulated",
       .n = n,
       .between = true,
       .out_stride = -2,
       .accumulate = true,
       .expect = serial},
      {.name = "a kernel per copy and line", .n = n, .fanout = 2, .per_row = true},
      {.name = "one kernel for every copy", .n = n, .factor_gap = -1, .expect = shared},
      {.name = "a real kernel, copies backwards", .n = n, .backwards = true, .mid = real},
      {.name = "a strided kernel, strided lines",
       .n = n,
       .fanout = 4,
       .in_stride = -2,
       .out_stride = 3,
       .mid_axis = -2,
       .per_row = true},
      {.name = "scaled, a prologue factor",
       .n = n,
       .mid_scale = 0.25,
       .epi_scale = 3.0,
       .pro = real_kind ? real : cplx,
       .expect = shared},
      {.name = "no kernel", .n = n, .fanout = 2, .mid = none, .epi_scale = 2.0},
      {.name = "an epilogue factor",
       .n = n,
       .epi_scale = 0.5,
       .epi = real_kind ? real : cplx,
       .expect = real_kind ? serial : shared},
      {.name = "a real epilogue factor, a real kernel",
       .n = n,
       .mid = real,
       .epi = real,
       .expect = serial},
      {.name = "accumulated, scaled, backwards",
       .n = n,
       .count = 2,
       .backwards = true,
       .epi_scale = 2.0,
       .accumulate = true,
       .expect = serial},
      {.name = "accumulated, no kernel, strided",
       .n = n,
       .fanout = 2,
       .out_stride = -1,
       .mid = none,
       .accumulate = true,
       .expect = serial},
  };
  if (n >= 8) {
    const std::size_t h = n / 2;
    const std::size_t q = n / 4;
    const std::vector<fan_case> runs = {
        {.name = "valid half", .n = n, .in = {{0, 0, h}}, .out = {{0, 0, h}}},
        {.name = "valid half at offsets, a kernel per line",
         .n = n,
         .in = {{0, 3, h}},
         .out = {{0, 5, h}},
         .per_row = true},
        {.name = "centred valid half, scaled",
         .n = n,
         .in = {{0, 0, h}},
         .out = {{q, 0, h}},
         .epi_scale = 2.0,
         .expect = serial},
        {.name = "bands in any order, backwards",
         .n = n,
         .backwards = true,
         .in = {{n - 3, 0, 2}, {1, 4, 2}, {q + 1, 9, q - 1}},
         .out = {{h + 1, 1, 2}, {0, 6, 1}, {q, 3, 2}},
         .expect = serial},
        {.name = "output runs, accumulated, a real kernel",
         .n = n,
         .out = {{1, 0, h - 1}, {h + 2, h, 2}},
         .mid = real,
         .accumulate = true,
         .expect = serial},
        {.name = "input runs only", .n = n, .in = {{2, 1, h}}, .expect = shared},
    };
    sets.insert(sets.end(), runs.begin(), runs.end());
  }
  return sets;
}

std::vector<std::size_t> lengths() {
  return {1, 2, 8, 12, 15, 16, 32, 60, 64, 100, 128, 210, 256, 257};
}

template <class T> class PipelineFanout : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(PipelineFanout, precisions);

TYPED_TEST(PipelineFanout, ComplexOnEveryInstructionSet) {
  using C = std::complex<TypeParam>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths()) {
      for (const fan_case& fc : fan_sets(n, false)) {
        run_case<TypeParam, C>(fc, t, false);
      }
    }
  }
}

TYPED_TEST(PipelineFanout, RealOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths()) {
      for (const fan_case& fc : fan_sets(n, true)) {
        run_case<TypeParam, TypeParam>(fc, t, false);
      }
    }
  }
}

// Teams share the copies (each on a member's own scratch) or run them one after another (in the
// block's second region), bitwise as serial execution, with completion::all and own and on an
// executor; and a long line, whose stages a team splits into chunks.
TYPED_TEST(PipelineFanout, TeamsAndExecutorsAgreeWithSerial) {
  using C = std::complex<TypeParam>;
  // The reference only where it is cheap: the point here is the bits.
  for (const std::size_t n : {16U, 256U, 1000U}) {
    for (fan_case fc : fan_sets(n, false)) {
      fc.count = 7;
      run_case<TypeParam, C>(fc, fftn::isa::best, true, {}, n <= 16);
    }
    for (fan_case fc : fan_sets(n, true)) {
      fc.count = 7;
      run_case<TypeParam, TypeParam>(fc, fftn::isa::best, true, {}, n <= 16);
    }
  }
  constexpr std::size_t big = 131072;
  for (const fan_case& fc : {
           fan_case{.name = "a long line", .n = big, .count = 1, .fanout = 3, .expect = shared},
           fan_case{.name = "a long accumulated line",
                    .n = big,
                    .count = 1,
                    .fanout = 2,
                    .accumulate = true,
                    .expect = serial},
       }) {
    run_case<TypeParam, C>(fc, fftn::isa::best, true, {}, false);
    run_case<TypeParam, TypeParam>(fc, fftn::isa::best, true, {}, false);
  }
}

// A short line whose copies a team splits among its members (the schedule forced to split one
// line: every threshold at its least, a copy per chunk), each copy on a member's own scratch,
// bitwise as serial execution; and the copies in the second region, one chunk.
TYPED_TEST(PipelineFanout, CopiesSplitAmongATeam) {
  using C = std::complex<TypeParam>;
  d::build_settings s;
  s.chunk_points = 1;
  s.profile.inline_multiple = 0;
  s.profile.split_min_points = 1;
  s.profile.split_min_points_mesh = 1;
  s.profile.split_min_points_lines = 1;
  s.profile.remainder_split_min_points = 1;
  s.profile.min_points_per_member = 1;
  s.profile.sync_ns = 0;
  for (const bool accumulate : {false, true}) {
    const fan_case fc{.name = accumulate ? "split copies, accumulated" : "split copies",
                      .n = 64,
                      .count = 1,
                      .fanout = 6,
                      .accumulate = accumulate,
                      .expect = accumulate ? serial : shared};
    fftn::test::input_source src(0x5E2FULL);
    const fan_run<TypeParam, C> run(fc, fftn::isa::best, src);
    const auto core = d::make_pipeline_core(run.view(), {.force_isa = run.target}, s);
    const auto copies = std::ranges::find_if(core->graph.phases,
                                             [](const d::graph_phase& ph) { return ph.nest != 0; });
    ASSERT_NE(copies, core->graph.phases.end()) << d::describe(*core);
    EXPECT_EQ(copies->chunks, accumulate ? 1U : 6U) << d::describe(*core);
    const d::lane_map lanes{.lanes = 3};
    const d::schedule plan = d::make_schedule(core->graph, lanes, core->profile);
    EXPECT_NE(d::share_of(plan, lanes, 0).split, 0U) << "the team does not split the line";
    run_core(run, s, true);
  }
}

// The depth-first pair (forced): the spectrum in the block's second region, each copy's middle in
// a pass into scratch and its backward chain on the block's first region, with the copies one
// after another in a team; with 5.2d's runs, accumulation and factors; against the reference on
// every instruction set, and bitwise under teams on a long line.
TYPED_TEST(PipelineFanout, DepthFirstPairForced) {
  using C = std::complex<TypeParam>;
  struct forced {
    std::size_t n;
    std::vector<std::size_t> tops;
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const forced& f : {forced{1024, {8}}, forced{2048, {8, 2}}}) {
      d::build_settings s;
      s.strategy = d::strategy_choice::depth;
      s.depth_tops = f.tops;
      const std::size_t n = f.n;
      const std::size_t h = n / 2;
      const std::size_t q = n / 4;
      for (const fan_case& fc : {
               fan_case{.name = "depth: kernels", .n = n, .expect = "second region"},
               fan_case{.name = "depth: a real kernel per line, backwards",
                        .n = n,
                        .backwards = true,
                        .mid = real,
                        .per_row = true,
                        .expect = serial},
               fan_case{.name = "depth: valid halves, scaled",
                        .n = n,
                        .in = {{0, 3, h}},
                        .out = {{q, 5, h}},
                        .epi_scale = 2.0,
                        .expect = "the depth-first pair"},
               fan_case{.name = "depth: bands, accumulated",
                        .n = n,
                        .in = {{n - 3, 0, 2}, {1, 4, 2}},
                        .out = {{h + 1, 1, 2}, {0, 6, 1}},
                        .accumulate = true,
                        .expect = "the depth-first pair"},
               fan_case{.name = "depth: factors, no kernel",
                        .n = n,
                        .fanout = 2,
                        .mid = none,
                        .pro = cplx,
                        .epi = real,
                        .expect = "the depth-first pair"},
           }) {
        fan_case two = fc;
        two.count = 2;
        run_case<TypeParam, C>(two, t, false, s);
      }
    }
  }
  d::build_settings s;
  s.strategy = d::strategy_choice::depth;
  s.depth_tops = {16};
  run_case<TypeParam, C>({.name = "depth, long",
                          .n = 65536,
                          .count = 2,
                          .fanout = 2,
                          .expect = "the depth-first pair"},
                         fftn::isa::best, true, s, false);
}

// The public API: a pipeline with fan-out against F separate pipelines of one copy each (copy k's
// kernel, writing where copy k does), within two transforms' tolerance of each other.
TEST(PipelineFanout, AgreesWithSeparatePipelines) {
  using C = std::complex<double>;
  constexpr std::size_t n = 96;
  constexpr std::size_t rows = 4;
  constexpr std::size_t copies = 3;
  fftn::test::input_source src(0xFA0ULL);
  const std::vector<C> x = src.complexes<double>(n * rows);
  const std::vector<C> kernels = src.complexes<double>(n * copies);
  fftn::pipeline_desc desc = fftn::complex_convolution_rows<double>(n, rows);
  desc.middle.scale = 1.0 / static_cast<double>(n);
  desc.fanout = copies;
  desc.fanout_out_dist = static_cast<std::ptrdiff_t>(n * rows);
  desc.fanout_factor_dist = static_cast<std::ptrdiff_t>(n);
  const fftn::pipeline fan(desc);
  fftn::workspace ws(fan);
  std::vector<C> y(n * rows * copies);
  fan.execute(x.data(), y.data(), ws, {.mid_factor = kernels.data()});
  fftn::pipeline_desc one = desc;
  one.fanout = 1;
  one.fanout_out_dist = 0;
  one.fanout_factor_dist = 0;
  const fftn::pipeline single(one);
  fftn::workspace ws1(single);
  std::vector<C> want(n * rows * copies);
  for (std::size_t k = 0; k < copies; ++k) {
    single.execute(x.data(), want.data() + k * n * rows, ws1,
                   {.mid_factor = kernels.data() + k * n});
  }
  double diff = 0;
  double norm = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    diff += std::norm(y[i] - want[i]);
    norm += std::norm(want[i]);
  }
  const auto tol = fftn::test::derived_tolerance<double>(n, fftn::test::fftn_bound(n)) * 4.0;
  EXPECT_LE(std::sqrt(diff / norm), tol.rms) << fan.describe();
}

template <class In, class Mid, class Out>
void expect_refused(const fftn::pipeline_desc<In, Mid, Out>& desc, const fftn::options& opts,
                    const std::string& field) {
  try {
    const fftn::pipeline<In, Mid, Out> p(desc, opts);
    ADD_FAILURE() << "not refused: " << field;
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(field), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("fftn::pipeline: "), std::string::npos) << e.what();
  }
}

TEST(PipelineFanout, RefusalsNameTheField) {
  using C = std::complex<double>;
  fftn::pipeline_desc<C, C, C> good = fftn::complex_convolution_rows<double>(16, 2);
  good.fanout = 3;
  good.fanout_out_dist = 32;
  good.fanout_factor_dist = 16;
  const fftn::pipeline<C, C, C> ok(good);
  EXPECT_NE(ok.describe().find("fan-out: 3 copies"), std::string::npos) << ok.describe();
  auto desc = good;
  desc.fanout = 0;
  expect_refused(desc, {}, "desc.fanout");
  desc = good;
  expect_refused(desc, {.in_place = true}, "desc.fanout");
  // Copies on top of each other, inside the batch, or over the next copy's lines.
  for (const std::ptrdiff_t dist : {0, 1, 16, -16, 31, -31}) {
    desc = good;
    desc.fanout_out_dist = dist;
    expect_refused(desc, {}, "desc.fanout_out_dist");
  }
  desc = good;
  desc.fanout_out_dist = -32;
  const fftn::pipeline<C, C, C> backwards(desc);
  // With stored runs the copies' reach is the output's stored extent.
  desc = good;
  desc.runs = {{.axis = 0, .where = fftn::side::out, .logical = 0, .stored = 0, .count = 8}};
  desc.batches[0].out_dist = 8;
  desc.fanout_out_dist = 16;
  const fftn::pipeline<C, C, C> by_extent(desc);
  desc.fanout_out_dist = 15;
  expect_refused(desc, {}, "desc.fanout_out_dist");
  // Spans that overflow.
  desc = good;
  desc.fanout_out_dist = std::numeric_limits<std::ptrdiff_t>::max() / 8;
  expect_refused(desc, {}, "desc.fanout_out_dist");
  desc = good;
  desc.fanout_factor_dist = std::numeric_limits<std::ptrdiff_t>::max() / 8;
  expect_refused(desc, {}, "desc.fanout_factor_dist");
  fftn::pipeline_desc<double, C, double> r = fftn::convolution_rows<double>(16, 2);
  r.fanout = 2;
  r.fanout_out_dist = 31;
  expect_refused(r, {}, "desc.fanout_out_dist");
  r.fanout_out_dist = 32;
  const fftn::pipeline<double, C, double> real_ok(r);
  EXPECT_NE(real_ok.describe().find(shared), std::string::npos) << real_ok.describe();
}

} // namespace
