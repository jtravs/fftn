// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Stored runs on row pipelines (fftn/pipeline.hpp, DESIGN.md 12.2; task 5.2d): a pipeline whose
// line has runs on either side against the full-length reference, the line built from the input's
// runs with zeros elsewhere, transformed, given the middle and transformed back in quadruple
// precision, and only the output's runs compared (cropped). Elements of the input outside its runs
// hold NaN, so a read of one would show; elements of the output outside its runs are canaries that
// must keep their bits. Valid halves (the zero-padded convolution's case), bands, runs listed in
// any order, stored offsets and strides of either sign, accumulation and scales, the chain's
// pruned first and last passes (forced radices, so that every target builds them), the <C, C, C>
// and <T, C, T> kinds, every instruction set in both precisions; serial, pieces of the batch,
// teams (completion::all and own) and an executor bitwise alike. Expectations hold on every
// target: none assumes a target's radices or strategy unless the test forces it.

#include "core/pipeline.hpp"
#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

template <class V> bool same_bits(const V& a, const V& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

struct run_spec {
  std::size_t logical = 0;
  std::size_t stored = 0;
  std::size_t count = 0;
};

struct runs_case {
  const char* name = "";
  std::size_t n = 16;
  std::size_t count = 3;
  std::vector<run_spec> in{};  // empty: the input line stored whole
  std::vector<run_spec> out{}; // empty: the output line stored whole
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t out_stride = 1;
  factor_kind mid = cplx;      // the kernel's kind
  std::ptrdiff_t mid_axis = 1; // its stride along the bins
  bool per_row = false;        // a kernel per line (batch stride one line of bins)
  double mid_scale = 0.0;      // 0: 1/n
  double epi_scale = 1.0;
  bool accumulate = false;
  // A prologue or epilogue factor (stride 1 along the line, one for every line), with no runs;
  // the real kind takes real ones only.
  factor_kind pro = factor_kind::none;
  factor_kind epi = factor_kind::none;
  const char* expect = ""; // in describe()
};

// The end of a side's last stored run, or n without runs.
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

// One case's buffers, description and reference. Elem is the pipeline's In and Out element type:
// std::complex<T> for <C, C, C>, T for <T, C, T>.
template <class T, class Elem> struct runs_run {
  static constexpr bool real_kind = std::is_same_v<Elem, T>;
  using C = std::complex<T>;
  using desc_t = fftn::pipeline_desc<Elem, C, Elem>;

  runs_case c;
  fftn::isa target;
  std::size_t bins = 0;
  std::ptrdiff_t in_dist = 0;
  std::ptrdiff_t out_dist = 0;
  std::size_t in_base = 0;
  std::size_t out_base = 0;
  double mid_scale = 1.0;
  desc_t desc;
  std::vector<Elem> x;    // the input buffer: NaN outside the runs
  std::vector<Elem> init; // the output buffer before execution (canaries, accumulation)
  std::vector<T> kreal;   // the kernel's storage, real or complex
  std::vector<C> kcomplex;
  std::ptrdiff_t kzero = 0;     // the kernel's coordinate 0 in its storage
  std::vector<bool> in_listed;  // input elements in a run (or a line stored whole)
  std::vector<bool> out_listed; // output elements in a run
  // The prologue's and the epilogue's factors (n values), real or complex.
  std::vector<T> rpro;
  std::vector<C> cpro;
  std::vector<T> repi;
  std::vector<C> cepi;

  runs_run(const runs_case& rc, fftn::isa t, fftn::test::input_source& src)
      : c(rc), target(t), bins(real_kind ? rc.n / 2 + 1 : rc.n) {
    const std::size_t ein = extent_of(c.in, c.n);
    const std::size_t eout = extent_of(c.out, c.n);
    const auto span = [](std::size_t e, std::ptrdiff_t stride) {
      return static_cast<std::ptrdiff_t>(e) * std::abs(stride) + 2; // a gap between lines
    };
    in_dist = span(ein, c.in_stride);
    out_dist = span(eout, c.out_stride);
    in_base = c.in_stride < 0 ? static_cast<std::size_t>(-c.in_stride) * (ein - 1) : 0;
    out_base = c.out_stride < 0 ? static_cast<std::size_t>(-c.out_stride) * (eout - 1) : 0;
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
    const std::ptrdiff_t kbatch =
        c.per_row ? std::abs(c.mid_axis) * static_cast<std::ptrdiff_t>(bins) : 0;
    desc.middle.scale = mid_scale;
    desc.middle.mul = {.kind = c.mid, .axis_strides = {c.mid_axis}, .batch_strides = {kbatch}};
    desc.epilogue.scale = c.epi_scale;
    desc.epilogue.accumulate = c.accumulate;
    const auto side_factor = [&](factor_kind k, std::vector<T>& r,
                                 std::vector<C>& z) -> fftn::factor {
      if (k == factor_kind::none) {
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

    const std::ptrdiff_t along = c.mid_axis * static_cast<std::ptrdiff_t>(bins - 1);
    const std::ptrdiff_t across = kbatch * static_cast<std::ptrdiff_t>(c.count - 1);
    kzero = -std::min<std::ptrdiff_t>(0, along);
    const auto ksize = static_cast<std::size_t>(std::abs(along) + across + 1);
    if (c.mid == real) {
      kreal = src.reals<T>(ksize);
    } else {
      kcomplex = src.complexes<T>(ksize);
    }

    x = values(src, static_cast<std::size_t>(in_dist) * c.count);
    init = values(src, static_cast<std::size_t>(out_dist) * c.count);
    in_listed.assign(x.size(), false);
    out_listed.assign(init.size(), false);
    for (std::size_t r = 0; r < c.count; ++r) {
      for (std::size_t j = 0; j < c.n; ++j) {
        if (const std::ptrdiff_t at = stored_of(c.in, j); at >= 0) {
          in_listed[in_at(static_cast<std::size_t>(at), r)] = true;
        }
        if (const std::ptrdiff_t at = stored_of(c.out, j); at >= 0) {
          out_listed[out_at(static_cast<std::size_t>(at), r)] = true;
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

  // The stored position of logical position j on a side with these runs (j itself without runs),
  // or -1 where the side does not store it.
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
  [[nodiscard]] std::size_t out_at(std::size_t p, std::size_t r) const {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(out_base) +
                                    static_cast<std::ptrdiff_t>(r) * out_dist +
                                    static_cast<std::ptrdiff_t>(p) * c.out_stride);
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

  // A side's factor at logical position j (1 without one).
  [[nodiscard]] static std::complex<double> side_at(const std::vector<T>& r,
                                                    const std::vector<C>& z, std::size_t j) {
    if (!r.empty()) {
      return {static_cast<double>(r[j]), 0.0};
    }
    return z.empty() ? std::complex<double>(1.0, 0.0) : std::complex<double>(z[j]);
  }

  [[nodiscard]] const void* kernel() const {
    return c.mid == real ? static_cast<const void*>(kreal.data() + kzero)
                         : static_cast<const void*>(kcomplex.data() + kzero);
  }

  [[nodiscard]] std::complex<double> kernel_at(std::size_t k, std::size_t r) const {
    const std::ptrdiff_t kbatch =
        c.per_row ? std::abs(c.mid_axis) * static_cast<std::ptrdiff_t>(bins) : 0;
    const auto i = static_cast<std::size_t>(
        kzero + (bins > 1 ? c.mid_axis * static_cast<std::ptrdiff_t>(k) : 0) +
        (c.count > 1 ? kbatch * static_cast<std::ptrdiff_t>(r) : 0));
    return c.mid == real ? std::complex<double>(kreal[i], 0.0) : std::complex<double>(kcomplex[i]);
  }

  // The largest magnitude of the kernel over its root mean square: how much more an error at a
  // bin can weigh than the output's typical size.
  [[nodiscard]] double spread() const {
    double peak = 0.0;
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t r = 0; r < c.count; ++r) {
      for (std::size_t k = 0; k < bins; ++k) {
        const double m = std::abs(kernel_at(k, r));
        peak = std::max(peak, m);
        sum += m * m;
        ++count;
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
    };
  }

  // The full-length reference: each line built from the input's runs with zeros elsewhere,
  // transformed exactly, times the kernel (rounded to double), transformed back exactly, times
  // the scales (plus the prior output with accumulation), compared on the output's runs only.
  // Tolerance: two transforms of n points, the first's error weighed by the kernel's spread,
  // plus a rounding per multiplication and the accumulation. Every element outside the output's
  // runs keeps its bits.
  void check(const std::vector<Elem>& y, const std::string& describe) const {
    const std::string what = std::string(c.name) + ", n = " + std::to_string(c.n) + ", " +
                             fftn::isa_name(target) + (sizeof(T) == 8 ? " double" : " float") +
                             (real_kind ? ", real" : "");
    SCOPED_TRACE(what);
    const double scale = mid_scale * c.epi_scale;
    const fftn::test::tolerance tol =
        fftn::test::derived_tolerance<T>(c.n, fftn::test::fftn_bound(c.n)) * (1.0 + spread()) *
            (c.epi != factor_kind::none ? 1.8 : 1.0) +
        fftn::test::input_rounding<T>() * 6;
    fftn::test::outcome o;
    for (std::size_t r = 0; r < c.count; ++r) {
      std::vector<std::complex<double>> spectrum(bins);
      if constexpr (real_kind) {
        std::vector<double> line(c.n, 0.0);
        for (std::size_t j = 0; j < c.n; ++j) {
          if (const std::ptrdiff_t at = stored_of(c.in, j); at >= 0) {
            line[j] = static_cast<double>(x[in_at(static_cast<std::size_t>(at), r)]) *
                      side_at(rpro, cpro, j).real();
          }
        }
        const auto s = fftn::test::reference::r2c(line);
        for (std::size_t k = 0; k < bins; ++k) {
          spectrum[k] = std::complex<double>(s[k].re.hi, s[k].im.hi) * kernel_at(k, r);
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
        for (std::size_t k = 0; k < bins; ++k) {
          spectrum[k] = std::complex<double>(s[k].re.hi, s[k].im.hi) * kernel_at(k, r);
        }
      }
      std::vector<Elem> got;
      if constexpr (real_kind) {
        const auto back = fftn::test::reference::c2r(c.n, spectrum);
        std::vector<fftn::test::exact_real> want;
        for (std::size_t j = 0; j < c.n; ++j) {
          if (const std::ptrdiff_t at = stored_of(c.out, j); at >= 0) {
            const std::size_t i = out_at(static_cast<std::size_t>(at), r);
            double v = back[j].hi * side_at(repi, cepi, j).real() * scale;
            if (c.accumulate) {
              v += static_cast<double>(init[i]);
            }
            want.push_back(fftn::test::exact_from(v));
            got.push_back(y[i]);
          }
        }
        o.within(fftn::test::compare<T>(std::span<const T>(got), want), tol,
                 what + ", line " + std::to_string(r));
      } else {
        const auto back = fftn::test::reference::c2c(+1, spectrum);
        std::vector<fftn::test::exact_complex> want;
        for (std::size_t j = 0; j < c.n; ++j) {
          if (const std::ptrdiff_t at = stored_of(c.out, j); at >= 0) {
            const std::size_t i = out_at(static_cast<std::size_t>(at), r);
            std::complex<double> v =
                std::complex<double>(back[j].re.hi, back[j].im.hi) * side_at(repi, cepi, j) * scale;
            if (c.accumulate) {
              v += std::complex<double>(init[i]);
            }
            want.push_back(fftn::test::exact_from(v));
            got.push_back(y[i]);
          }
        }
        o.within(fftn::test::compare<T>(std::span<const C>(got), want), tol,
                 what + ", line " + std::to_string(r));
      }
    }
    EXPECT_TRUE(o.passed()) << o.detail() << describe;
    std::size_t touched = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      if (!out_listed[i] && !same_bits(std::vector<Elem>{y[i]}, std::vector<Elem>{init[i]})) {
        ++touched;
      }
    }
    EXPECT_EQ(touched, 0U) << what << ": elements outside the output's runs written\n" << describe;
  }
};

// The pipeline of a run built from its description with construction knobs (settings), executed
// serially (checked against the reference where `reference`), in two pieces of the batch and,
// with `threads`, by teams of three (completion::all and own) and on a pool of three, each from
// the same output buffer, all bitwise alike; the input and the kernel left as they were. Returns
// describe().
template <class Run>
std::string run_core(const Run& run, const d::build_settings& settings, bool threads,
                     bool reference = true, decltype(Run::x)* result = nullptr) {
  const runs_case& rc = run.c;
  const std::shared_ptr<const d::plan_core> core =
      d::make_pipeline_core(run.view(), {.force_isa = run.target}, settings);
  const std::string describe = d::describe(*core);
  EXPECT_NE(describe.find(rc.expect), std::string::npos)
      << rc.name << ": no '" << rc.expect << "' in\n"
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
  if (result != nullptr) {
    *result = serial;
  }
  EXPECT_TRUE(same_bits(xs, x0)) << rc.name << ": input written";
  EXPECT_TRUE(same_bits(run.kreal, kr) && same_bits(run.kcomplex, kc))
      << rc.name << ": kernel written";

  auto pieces = run.init;
  const std::size_t half = rc.count / 2;
  d::execute_core(*core, 0, half, in(), pieces.data() + run.out_base, ws, args);
  d::execute_core(*core, half, rc.count - half, in(), pieces.data() + run.out_base, ws, args);
  EXPECT_TRUE(same_bits(pieces, serial)) << rc.name << ": pieces of the batch differ\n" << describe;
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
    EXPECT_TRUE(same_bits(teamed, serial)) << rc.name << ": a team differs\n" << describe;
  }
  const fftn::thread_pool pool(3);
  auto pooled = run.init;
  d::execute_on(*core, in(), pooled.data() + run.out_base, pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << rc.name << ": the executor differs\n" << describe;
  return describe;
}

template <class T, class Elem>
std::string run_case(const runs_case& rc, fftn::isa t, bool threads,
                     const d::build_settings& settings = {}, bool reference = true) {
  fftn::test::input_source src(0x5D2DULL + rc.n * 131 + rc.count + rc.in.size() * 7 +
                               rc.out.size());
  return run_core(runs_run<T, Elem>(rc, t, src), settings, threads, reference);
}

// Run sets on a line of n (n at least 8): valid halves (stored where the line's are, and at an
// offset), the valid half of the output in the middle of the line (a centred kernel's), bands of
// the input and crops of the output listed out of order, one side only, a scale and accumulation,
// a separate middle pass (a real kernel), a kernel per line, strides of either sign.
std::vector<runs_case> run_sets(std::size_t n) {
  const std::size_t h = n / 2;
  const std::size_t q = n / 4;
  return {
      {.name = "valid half",
       .n = n,
       .in = {{0, 0, h}},
       .out = {{0, 0, h}},
       .expect = "stored runs:"},
      {.name = "valid half at offsets", .n = n, .in = {{0, 3, h}}, .out = {{0, 5, h}}},
      {.name = "centred valid half",
       .n = n,
       .in = {{0, 0, h}},
       .out = {{q, 0, h}},
       .expect = "the rest not written"},
      {.name = "bands in any order",
       .n = n,
       .in = {{n - 3, 0, 2}, {1, 4, 2}, {q + 1, 9, q - 1}},
       .out = {{h + 1, 1, 2}, {0, 6, 1}, {q, 3, 2}},
       .expect = "zeros elsewhere"},
      {.name = "input runs only", .n = n, .in = {{2, 1, h}}},
      {.name = "output runs only, accumulated",
       .n = n,
       .out = {{1, 0, h - 1}, {h + 2, h, 2}},
       .epi_scale = 0.5,
       .accumulate = true,
       .expect = "accumulated"},
      {.name = "valid half, a real kernel in a separate pass, scaled",
       .n = n,
       .in = {{0, 0, h}},
       .out = {{0, 0, h}},
       .mid = real,
       .mid_axis = -1,
       .epi_scale = 3.0},
      {.name = "valid half, a kernel per line",
       .n = n,
       .in = {{0, 0, h}},
       .out = {{0, 0, h}},
       .per_row = true},
      {.name = "strided valid half",
       .n = n,
       .in = {{0, 0, h}},
       .out = {{0, 1, h}},
       .in_stride = 3,
       .out_stride = -2},
      {.name = "strided bands",
       .n = n,
       .in = {{h, 0, 3}, {0, 4, 2}},
       .out = {{q, 2, 3}},
       .in_stride = -2,
       .out_stride = 2,
       .per_row = true},
  };
}

// Lines of a leaf, a pair leaf or a chain, codelet, mixed, odd, generic and Bluestein lengths.
std::vector<std::size_t> lengths() { return {8, 12, 15, 16, 32, 60, 64, 100, 128, 210, 256, 257}; }

template <class T> class PipelineRuns : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(PipelineRuns, precisions);

TYPED_TEST(PipelineRuns, ComplexOnEveryInstructionSet) {
  using C = std::complex<TypeParam>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths()) {
      for (const runs_case& rc : run_sets(n)) {
        run_case<TypeParam, C>(rc, t, false);
      }
    }
  }
}

TYPED_TEST(PipelineRuns, RealOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths()) {
      for (const runs_case& rc : run_sets(n)) {
        run_case<TypeParam, TypeParam>(rc, t, false);
      }
    }
  }
}

TYPED_TEST(PipelineRuns, TeamsAndExecutorsAgreeWithSerial) {
  using C = std::complex<TypeParam>;
  for (const std::size_t n : {16U, 256U, 1000U}) {
    for (runs_case rc : run_sets(n)) {
      rc.count = 7;
      run_case<TypeParam, C>(rc, fftn::isa::best, true);
      run_case<TypeParam, TypeParam>(rc, fftn::isa::best, true);
    }
  }
  // A line split among a team: its stages have many chunks.
  constexpr std::size_t big = 131072;
  run_case<TypeParam, C>({.name = "a long valid half",
                          .n = big,
                          .count = 1,
                          .in = {{0, 0, big / 2}},
                          .out = {{big / 4, 0, big / 2}}},
                         fftn::isa::best, true, {}, false);
  run_case<TypeParam, TypeParam>({.name = "a long real valid half",
                                  .n = big,
                                  .count = 1,
                                  .in = {{0, 0, big / 2}},
                                  .out = {{big / 4, 0, big / 2}}},
                                 fftn::isa::best, true, {}, false);
}

// The chain's pruned first and last passes (task 5.1e) on a pipeline's line: with radices forced
// (an even codelet radix first and last), the valid half is read and written where it is stored
// (no read or store stage), runs within the first half go through the block's half, and runs
// beyond it take the whole line, on every instruction set.
TYPED_TEST(PipelineRuns, PrunedPassesWithForcedRadices) {
  using C = std::complex<TypeParam>;
  struct forced {
    std::size_t n;
    std::vector<std::size_t> radices;
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const forced& f : {forced{64, {4, 4, 4}}, forced{128, {8, 16}}, forced{240, {4, 6, 10}},
                            forced{1024, {16, 4, 16}}}) {
      d::build_settings s;
      s.radices = f.radices;
      s.layout = d::layout_choice::interleaved; // a split run's last pass is never pruned
      const std::size_t n = f.n;
      const std::size_t h = n / 2;
      run_case<TypeParam, C>({.name = "valid half, pruned where stored",
                              .n = n,
                              .in = {{0, 2, h}},
                              .out = {{0, 1, h}},
                              .expect = "read where they are stored by the first pass, pruned"},
                             t, false, s);
      std::string text = run_case<TypeParam, C>(
          {.name = "valid half, pruned, scaled",
           .n = n,
           .in = {{0, 0, h}},
           .out = {{0, 0, h}},
           .epi_scale = 2.0,
           .expect = "written where they are stored by the last pass, pruned"},
          t, false, s);
      EXPECT_NE(text.find("they cost 0 separate passes"), std::string::npos) << text;
      run_case<TypeParam, C>({.name = "bands within the half",
                              .n = n,
                              .in = {{h - 3, 0, 3}, {1, 3, 2}},
                              .out = {{2, 0, 3}, {h - 1, 3, 1}},
                              .expect = "in the line's first half into the workspace block"},
                             t, false, s);
      run_case<TypeParam, C>({.name = "an accumulated half",
                              .n = n,
                              .in = {{0, 0, h}},
                              .out = {{0, 0, h}},
                              .accumulate = true,
                              .expect = "accumulated from the workspace block in a separate pass, "
                                        "the last pass pruned"},
                             t, false, s);
      run_case<TypeParam, C>({.name = "strided halves",
                              .n = n,
                              .in = {{0, 0, h}},
                              .out = {{0, 0, h}},
                              .in_stride = 2,
                              .out_stride = -1,
                              .expect = "in the line's first half into the workspace block"},
                             t, false, s);
      run_case<TypeParam, C>({.name = "beyond the half",
                              .n = n,
                              .in = {{1, 0, h}},
                              .out = {{h, 0, h}},
                              .expect = "the whole transform computed between them"},
                             t, false, s);
    }
  }
}

// The main case through the public API: convolution_rows with the valid halves as runs (a row
// of m values and a kernel of up to m taps, zero-padded to 2m), against the direct linear
// convolution, which is exact in double for small integers.
TYPED_TEST(PipelineRuns, ValidHalfOfAZeroPaddedConvolution) {
  using T = TypeParam;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t m : {4U, 8U, 50U, 128U, 500U}) {
      const std::size_t n = 2 * m;
      const std::size_t rows = 5;
      const std::size_t taps = m / 2 + 1;
      // The first m outputs of the linear convolution of x (m values) with k (taps values).
      const std::vector<fftn::stored_run> runs{
          {.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = m},
          {.axis = 0, .where = fftn::side::out, .logical = 0, .stored = 0, .count = m},
      };
      const fftn::pipeline_desc<T, std::complex<T>, T> desc =
          fftn::convolution_rows<T>(n, rows, runs);
      EXPECT_EQ(desc.batches[0].in_dist, static_cast<std::ptrdiff_t>(m));
      EXPECT_EQ(desc.batches[0].out_dist, static_cast<std::ptrdiff_t>(m));
      const fftn::pipeline<T, std::complex<T>, T> p(desc, {.force_isa = t});
      fftn::test::input_source src(0xC0DEULL + m);
      std::vector<T> x(m * rows);
      for (T& v : x) {
        v = static_cast<T>(src.integer(8));
      }
      std::vector<double> k(n, 0.0);
      for (std::size_t j = 0; j < taps; ++j) {
        k[j] = src.integer(8);
      }
      const auto ks = fftn::test::reference::r2c(k);
      std::vector<std::complex<T>> spectrum(n / 2 + 1);
      for (std::size_t b = 0; b < spectrum.size(); ++b) {
        spectrum[b] = {static_cast<T>(ks[b].re.hi / static_cast<double>(n)),
                       static_cast<T>(ks[b].im.hi / static_cast<double>(n))};
      }
      std::vector<T> y(m * rows, T(7));
      fftn::workspace ws(p);
      p.execute(x.data(), y.data(), ws, {.mid_factor = spectrum.data()});
      double worst = 0.0;
      double peak = 0.0;
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t i = 0; i < m; ++i) {
          double want = 0.0;
          for (std::size_t j = 0; j <= i && j < taps; ++j) {
            want += k[j] * static_cast<double>(x[r * m + i - j]);
          }
          worst = std::max(worst, std::abs(static_cast<double>(y[r * m + i]) - want));
          peak = std::max(peak, std::abs(want));
        }
      }
      // Two transforms of n and the spectrum's rounding, relative to the output's peak; the
      // spectrum's spread is at most taps times its rms.
      const double eps = std::numeric_limits<T>::epsilon();
      EXPECT_LE(worst, peak * eps * 8.0 * (std::log2(static_cast<double>(n)) + 1.0) *
                           std::sqrt(static_cast<double>(taps)))
          << "m = " << m << ", " << fftn::isa_name(t) << "\n"
          << p.describe();
    }
  }
}

// The depth-first pair (task 5.2d), forced with tops every complex count C (at most 8) divides:
// one input run from logical 0 and one output run from a logical start C divides fused into the
// forward first pass and the last top pass; other runs, a prologue or epilogue factor and
// accumulation through the read and store stages; a separate middle on scratch; a kernel per line;
// on every instruction set in both precisions, against the reference.
struct depth_forced {
  std::size_t n;
  std::vector<std::size_t> tops;
};

std::vector<depth_forced> forced_depths() { return {{1024, {8}}, {1536, {8}}, {2048, {8, 2}}}; }

std::vector<runs_case> depth_sets(std::size_t n) {
  const std::size_t h = n / 2;
  const std::size_t q = n / 4;
  return {
      {.name = "depth: the kernel", .n = n, .expect = "the depth-first pair"},
      {.name = "depth: valid half",
       .n = n,
       .in = {{0, 0, h}},
       .out = {{0, 0, h}},
       .expect = "read where it is stored by the forward chain's first pass, zeros beyond"},
      {.name = "depth: centred valid half at offsets, scaled",
       .n = n,
       .in = {{0, 3, h}},
       .out = {{q, 5, h}},
       .epi_scale = 2.0,
       .expect = "written where it is stored by the backward chain's last top pass"},
      {.name = "depth: an output run off the lanes",
       .n = n,
       .in = {{0, 0, h + 1}},
       .out = {{3, 0, h}},
       .expect = "the depth-first pair"},
      {.name = "depth: bands in any order",
       .n = n,
       .in = {{n - 3, 0, 2}, {1, 4, 2}, {q + 1, 9, q - 1}},
       .out = {{h + 1, 1, 2}, {0, 6, 1}, {q, 3, 2}},
       .expect = "read with zeros elsewhere into scratch in a separate pass"},
      {.name = "depth: an accumulated run",
       .n = n,
       .out = {{q, 0, h}},
       .epi_scale = 0.5,
       .accumulate = true,
       .expect = "accumulated from the workspace block in a separate pass"},
      {.name = "depth: a real kernel on scratch, a kernel per line",
       .n = n,
       .in = {{0, 0, h}},
       .out = {{0, 0, h}},
       .mid = real,
       .per_row = true,
       .expect = "in a separate pass on the line in scratch"},
      {.name = "depth: a complex kernel per line, fused, scaled",
       .n = n,
       .in = {{0, 7, h - 1}},
       .out = {{q, 2, h + 3}},
       .per_row = true,
       .epi_scale = 3.0,
       .expect = "the depth-first pair"},
      {.name = "depth: a strided complex kernel, separate on scratch",
       .n = n,
       .in = {{0, 0, h}},
       .out = {{0, 0, h}},
       .mid_axis = 2,
       .expect = "in a separate pass on the line in scratch"},
      {.name = "depth: prologue and epilogue factors, accumulated",
       .n = n,
       .accumulate = true,
       .pro = cplx,
       .epi = real,
       .expect = "2 separate passes over the line"},
  };
}

TYPED_TEST(PipelineRuns, DepthFirstPairForced) {
  using C = std::complex<TypeParam>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const depth_forced& f : forced_depths()) {
      d::build_settings s;
      s.strategy = d::strategy_choice::depth;
      s.depth_tops = f.tops;
      for (runs_case rc : depth_sets(f.n)) {
        rc.count = 2;
        run_case<TypeParam, C>(rc, t, false, s);
      }
    }
  }
}

// The depth-first pair's results against the chain pair's, on a line too long for the reference:
// both within two transforms' tolerance of each other.
template <class T>
void expect_close(const std::vector<std::complex<T>>& got, const std::vector<std::complex<T>>& want,
                  std::size_t n, const std::string& what) {
  double diff = 0;
  double norm = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    diff += std::norm(std::complex<double>(got[i]) - std::complex<double>(want[i]));
    norm += std::norm(std::complex<double>(want[i]));
  }
  const auto tol = fftn::test::derived_tolerance<T>(n, fftn::test::fftn_bound(n)) * 4.0;
  EXPECT_LE(std::sqrt(diff / norm), tol.rms) << what;
}

// Within a line, a team shares the depth-first pair's stages (their chunks) and its sub-row loops
// (lines on each member's own scratch, the owner's block shared), bitwise as serial execution.
TYPED_TEST(PipelineRuns, DepthFirstPairSharedByATeam) {
  using C = std::complex<TypeParam>;
  constexpr std::size_t n = 65536;
  d::build_settings depth;
  depth.strategy = d::strategy_choice::depth;
  depth.depth_tops = {16};
  d::build_settings chain;
  chain.strategy = d::strategy_choice::chain;
  for (const runs_case& rc : {
           runs_case{.name = "depth, long: valid half",
                     .n = n,
                     .count = 2,
                     .in = {{0, 0, n / 2}},
                     .out = {{n / 4, 0, n / 2}},
                     .expect = "the depth-first pair"},
           runs_case{.name = "depth, long: bands, a real kernel",
                     .n = n,
                     .count = 2,
                     .in = {{n / 2, 0, 100}, {7, 100, n / 3}},
                     .out = {{1, 0, n / 2}, {n - 9, n / 2, 9}},
                     .mid = real,
                     .expect = "the depth-first pair"},
       }) {
    std::vector<C> by_depth;
    std::vector<C> by_chain;
    fftn::test::input_source a(0xD3E9ULL);
    fftn::test::input_source b(0xD3E9ULL);
    run_core(runs_run<TypeParam, C>(rc, fftn::isa::best, a), depth, true, false, &by_depth);
    runs_case as_chain = rc;
    as_chain.expect = "the chain pair";
    run_core(runs_run<TypeParam, C>(as_chain, fftn::isa::best, b), chain, false, false, &by_chain);
    expect_close<TypeParam>(by_depth, by_chain, n, rc.name);
  }
}

// By the rules a <C, C, C> line keeps the chain pair even where a Bluestein stage whose
// convolution had the line's length would take the depth-first convolution (X50's rule), with a
// profile the test sets: when the rules take the pair is experiment X67's to decide (task 5.2g).
// Forced there, the pair agrees with the chain pair; a strided line never takes it.
TEST(PipelineRuns, DepthFirstPairByTheRules) {
  using C = std::complex<double>;
  constexpr std::size_t n = 34560; // 2^8 3^3 5: a top of 16 and every complex count split it
  d::build_settings rules;
  rules.rader = 0;
  rules.profile.depth_min_bytes = std::size_t{256} * 1024;
  rules.profile.depth_max_bytes = std::size_t{4} * 1024 * 1024;
  rules.profile.depth_part_bytes = 0;
  rules.profile.depth_top = 16;
  rules.profile.depth_top_alt = 20;
  rules.profile.depth_convolution_max_bytes = std::size_t{4} * 1024 * 1024;
  rules.profile.bluestein_pow2_part_bytes = SIZE_MAX;
  rules.bluestein_length = n;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    // A Bluestein prime whose convolution is n points long.
    const fftn::shape<C, C> row{
        .dir = fftn::direction::forward, .axes = {{.n = 16411}}, .batches = {}};
    const auto plan = d::make_plan_core(d::view_of(row), {.force_isa = t}, rules);
    const bool convolution = d::describe(*plan).find("two depth-first chains") != std::string::npos;
    // X50's rule: in double on 32 vector registers.
    const std::string name = fftn::isa_name(t);
    EXPECT_EQ(convolution, name == "avx512" || name == "neon" || name == "neon_fcma") << name;
    fftn::test::input_source src(0x2D1EULL);
    runs_case rc{.name = "depth by the rules",
                 .n = n,
                 .count = 1,
                 .in = {{0, 0, n / 2}},
                 .out = {{0, 0, n / 2}}};
    rc.expect = "the chain pair";
    std::vector<C> by_rules;
    run_core(runs_run<double, C>(rc, t, src), rules, false, false, &by_rules);
    d::build_settings forced = rules;
    forced.strategy = d::strategy_choice::depth;
    runs_case as_depth = rc;
    as_depth.expect = "the depth-first pair";
    fftn::test::input_source again(0x2D1EULL);
    std::vector<C> by_depth;
    run_core(runs_run<double, C>(as_depth, t, again), forced, false, false, &by_depth);
    expect_close<double>(by_depth, by_rules, n, fftn::isa_name(t));
    runs_case strided = rc;
    strided.in_stride = 2;
    fftn::test::input_source more(0x2D1FULL);
    run_core(runs_run<double, C>(strided, t, more), forced, false, false);
  }
  fftn::test::input_source src(0x2D20ULL);
  run_core(runs_run<float, std::complex<float>>(
               {.name = "float by the rules", .n = n, .count = 1, .expect = "the chain pair"},
               fftn::isa::best, src),
           rules, false, false);
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

TEST(PipelineRuns, RefusalsNameTheField) {
  using C = std::complex<double>;
  const fftn::stored_run half{
      .axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = 8};
  const fftn::pipeline_desc<C, C, C> good = fftn::complex_convolution_rows<double>(16, 2);
  auto desc = good;
  desc.runs = {half};
  desc.runs[0].axis = 1;
  expect_refused(desc, {}, "desc.runs[0].axis");
  desc.runs = {half};
  desc.runs[0].count = 0;
  expect_refused(desc, {}, "desc.runs[0].count");
  desc.runs = {half};
  desc.runs[0].logical = 9;
  expect_refused(desc, {}, "desc.runs[0].logical");
  desc.runs = {half, half};
  desc.runs[1].logical = 8;
  desc.runs[1].stored = 7;
  expect_refused(desc, {}, "desc.runs[1].stored");
  desc.runs = {half, half};
  desc.runs[1].stored = 8;
  desc.runs[1].logical = 4;
  expect_refused(desc, {}, "desc.runs[1].logical");
  desc = good;
  desc.runs = {half};
  desc.batches[0].in_dist = 0;
  expect_refused(desc, {.in_place = true}, "desc.runs");
  desc = good;
  desc.runs = {half};
  desc.prologue.mul = {.kind = cplx, .axis_strides = {1}, .batch_strides = {}};
  expect_refused(desc, {}, "desc.prologue.mul.kind");
  desc = good;
  desc.runs = {half};
  desc.runs[0].where = fftn::side::out;
  desc.epilogue.mul = {.kind = real, .axis_strides = {1}, .batch_strides = {}};
  expect_refused(desc, {}, "desc.epilogue.mul.kind");
  // Overlapping lines by stored extent: the input's runs reach 12 elements, its lines 8 apart.
  desc = good;
  desc.runs = {{.axis = 0, .where = fftn::side::out, .logical = 0, .stored = 4, .count = 8}};
  desc.batches[0].out_dist = 8;
  expect_refused(desc, {}, "desc");
  // A <T, C, T> pipeline's runs reach the n logical positions of its real line.
  fftn::pipeline_desc<double, C, double> r = fftn::convolution_rows<double>(16, 2);
  r.runs = {{.axis = 0, .where = fftn::side::out, .logical = 9, .stored = 0, .count = 8}};
  expect_refused(r, {}, "desc.runs[0].logical");
  r.runs[0].logical = 8;
  r.batches[0].out_dist = 8;
  const fftn::pipeline<double, C, double> ok(r);
  EXPECT_NE(ok.describe().find("stored runs:"), std::string::npos) << ok.describe();
}

} // namespace
