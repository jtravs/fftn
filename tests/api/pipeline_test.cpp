// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Row pipelines (fftn/pipeline.hpp, DESIGN.md 12.2; task 5.2b): the <C, C, C> chain pair against
// "forward transform, the user's multiply, inverse transform", evaluated on the
// quadruple-precision reference within two transforms' derived tolerance plus a rounding per
// vocabulary operation, on every instruction set in both precisions. Lengths from 1 to the
// leaves, then codelet, mixed, generic, Rader and Bluestein lengths; middle factors (the kernel)
// of strides of every sign, broadcast and per row, fused and separate; prologue and epilogue
// factors, scales and accumulation; strided lines with canaries in their gaps; in place; the
// input and the factor arrays left as they were; serial, pieces of the batch, teams (lines
// split, completion::all and own) and an executor bitwise alike; the flat form named for
// leaf-sized lines; and every refusal naming its field. The <T, C, T> real chain pair (task 5.2c)
// likewise against "r2c, multiply on the n/2 + 1 bins, c2r", even and odd lengths, real and
// complex kernels, and convolution_rows against the direct convolution. The edge driver's
// row-block form on a pipeline's program (PR 140's latent finding). Expectations hold on every
// target: none assumes a target's radices or strategy.

#include "core/edge_driver.hpp"
#include "core/pipeline.hpp"
#include "core/plan_core.hpp"
#include "core/program.hpp"
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
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

template <class T> using cvec = std::vector<std::complex<T>>;

template <class V> bool same_bits(const V& a, const V& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

// A factor as a test gives it: kind and strides along the line and the batch.
struct factor_spec {
  fftn::factor_kind kind = fftn::factor_kind::none;
  std::ptrdiff_t axis = 1;
  std::ptrdiff_t batch = 0;
};

using fftn::factor_kind;
constexpr factor_kind real = factor_kind::real;
constexpr factor_kind cplx = factor_kind::complex;

struct pipe_case {
  const char* name = "";
  std::size_t n = 1;
  std::size_t count = 1;
  double options_scale = 1.0;
  double pro_scale = 1.0;
  double mid_scale = 0.0; // 0: 1/n, the round trip's normalisation
  double epi_scale = 1.0;
  factor_spec pro{};
  factor_spec mid{.kind = cplx, .axis = 1, .batch = 0};
  factor_spec epi{};
  bool accumulate = false;
  bool in_place = false;
  std::ptrdiff_t in_stride = 1; // along the line; lines are n * stride + 1 apart (a gap)
  std::ptrdiff_t out_stride = 1;
  const char* expect = ""; // in describe()
};

// A factor array over count x n coordinates with its strides: the storage and the offset of
// coordinate 0 in it.
template <class T> struct factor_array {
  std::vector<T> reals;
  cvec<T> complexes;
  std::ptrdiff_t zero = 0;

  factor_array(const factor_spec& f, std::size_t n, std::size_t count,
               fftn::test::input_source& src) {
    if (f.kind == fftn::factor_kind::none) {
      return;
    }
    const auto last = [](std::ptrdiff_t stride, std::size_t extent) {
      return extent > 1 ? stride * static_cast<std::ptrdiff_t>(extent - 1) : 0;
    };
    const std::ptrdiff_t a = last(f.axis, n);
    const std::ptrdiff_t b = last(f.batch, count);
    const std::ptrdiff_t lo = std::min<std::ptrdiff_t>(0, a) + std::min<std::ptrdiff_t>(0, b);
    const std::ptrdiff_t hi = std::max<std::ptrdiff_t>(0, a) + std::max<std::ptrdiff_t>(0, b);
    zero = -lo;
    const auto size = static_cast<std::size_t>(hi - lo + 1);
    if (f.kind == fftn::factor_kind::real) {
      reals = src.reals<T>(size);
    } else {
      complexes = src.complexes<T>(size);
    }
  }

  [[nodiscard]] const void* address() const {
    if (!reals.empty()) {
      return reals.data() + zero;
    }
    return complexes.empty() ? nullptr : static_cast<const void*>(complexes.data() + zero);
  }

  // The factor at (j, r) in double; 1 without a factor.
  [[nodiscard]] std::complex<double> at(const factor_spec& f, std::size_t j, std::size_t r,
                                        std::size_t n, std::size_t count) const {
    const std::ptrdiff_t i = zero + (n > 1 ? f.axis * static_cast<std::ptrdiff_t>(j) : 0) +
                             (count > 1 ? f.batch * static_cast<std::ptrdiff_t>(r) : 0);
    const auto u = static_cast<std::size_t>(i);
    if (!reals.empty()) {
      return {static_cast<double>(reals[u]), 0.0};
    }
    if (!complexes.empty()) {
      return std::complex<double>(complexes[u]);
    }
    return {1.0, 0.0};
  }

  // The largest magnitude of the array over its root mean square (1 without a factor): how much
  // more an error in the values it multiplies can weigh than the values themselves.
  [[nodiscard]] double spread() const {
    double peak = 0.0;
    double sum = 0.0;
    std::size_t count = 0;
    const auto add = [&](double m) {
      peak = std::max(peak, m);
      sum += m * m;
      ++count;
    };
    for (const T v : reals) {
      add(std::abs(static_cast<double>(v)));
    }
    for (const std::complex<T>& v : complexes) {
      add(std::abs(std::complex<double>(v)));
    }
    return count == 0 || sum == 0.0 ? 1.0 : peak / std::sqrt(sum / static_cast<double>(count));
  }
};

fftn::factor factor_of(const factor_spec& f) {
  if (f.kind == fftn::factor_kind::none) {
    return {};
  }
  return {.kind = f.kind, .axis_strides = {f.axis}, .batch_strides = {f.batch}};
}

template <class T>
using desc_of = fftn::pipeline_desc<std::complex<T>, std::complex<T>, std::complex<T>>;
template <class T>
using pipe_of = fftn::pipeline<std::complex<T>, std::complex<T>, std::complex<T>>;

// The layout of a case's lines in its buffers (of complex values for <C, C, C>, of reals for
// <T, C, T>), its factor arrays (the middle's over `bins` bins per line) and the description's
// fields every kind shares.
template <class T> struct run_layout {
  pipe_case c;
  fftn::isa target;
  std::ptrdiff_t in_dist = 0;
  std::ptrdiff_t out_dist = 0;
  // Where element 0 of line 0 is in each buffer: past the line's other elements when its stride
  // is negative.
  std::size_t in_base = 0;
  std::size_t out_base = 0;
  double mid_scale = 1.0;
  factor_array<T> fpro;
  factor_array<T> fmid;
  factor_array<T> fepi;
  fftn::options opts;

  run_layout(const pipe_case& pc, fftn::isa t, std::size_t bins, fftn::test::input_source& src)
      : c(pc), target(t), fpro(pc.pro, pc.n, pc.count, src), fmid(pc.mid, bins, pc.count, src),
        fepi(pc.epi, pc.n, pc.count, src) {
    const auto reach = [this](std::ptrdiff_t stride) {
      return static_cast<std::ptrdiff_t>(c.n) * std::abs(stride);
    };
    const auto base = [this](std::ptrdiff_t stride) {
      return stride < 0 ? static_cast<std::size_t>(-stride) * (c.n - 1) : std::size_t{0};
    };
    in_dist = reach(c.in_stride) + 1;
    out_dist = c.in_place ? in_dist : reach(c.out_stride) + 1;
    in_base = base(c.in_stride);
    out_base = c.in_place ? in_base : base(c.out_stride);
    mid_scale = c.mid_scale != 0.0 ? c.mid_scale : 1.0 / static_cast<double>(c.n);
    opts = {.in_place = c.in_place, .scale = c.options_scale, .force_isa = t};
  }

  template <class D> void describe_into(D& desc) const {
    desc.line = {
        .n = c.n, .in_stride = c.in_stride, .out_stride = c.in_place ? c.in_stride : c.out_stride};
    desc.batches = {{.count = c.count, .in_dist = in_dist, .out_dist = out_dist}};
    desc.prologue.scale = c.pro_scale;
    desc.prologue.mul = factor_of(c.pro);
    desc.middle.scale = mid_scale;
    desc.middle.mul = factor_of(c.mid);
    desc.epilogue.scale = c.epi_scale;
    desc.epilogue.mul = factor_of(c.epi);
    desc.epilogue.accumulate = c.accumulate;
  }

  [[nodiscard]] fftn::exec_args args() const {
    return {
        .in_factor = fpro.address(), .mid_factor = fmid.address(), .out_factor = fepi.address()};
  }

  [[nodiscard]] std::size_t in_at(std::size_t j, std::size_t r) const {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(in_base) +
                                    static_cast<std::ptrdiff_t>(r) * in_dist +
                                    static_cast<std::ptrdiff_t>(j) * c.in_stride);
  }
  [[nodiscard]] std::size_t out_at(std::size_t k, std::size_t r) const {
    const std::ptrdiff_t stride = c.in_place ? c.in_stride : c.out_stride;
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(out_base) +
                                    static_cast<std::ptrdiff_t>(r) * out_dist +
                                    static_cast<std::ptrdiff_t>(k) * stride);
  }
};

template <class T> struct pipe_run : run_layout<T> {
  using pipe = pipe_of<T>;
  using run_layout<T>::c;
  using run_layout<T>::fpro;
  using run_layout<T>::fmid;
  using run_layout<T>::fepi;
  using run_layout<T>::in_at;
  using run_layout<T>::out_at;
  desc_of<T> desc;
  cvec<T> x;    // the input buffer, gaps included
  cvec<T> init; // the output buffer before execution (accumulation, canaries)

  pipe_run(const pipe_case& pc, fftn::isa t, fftn::test::input_source& src)
      : run_layout<T>(pc, t, pc.n, src) {
    this->describe_into(desc);
    x = src.complexes<T>(static_cast<std::size_t>(this->in_dist) * c.count);
    init = c.in_place ? x : src.complexes<T>(static_cast<std::size_t>(this->out_dist) * c.count);
  }

  // Checks y (the output buffer after execution) against the reference, and every element
  // outside the lines against init.
  //
  // Reference: the exact forward transform of the input times the prologue, rounded to double,
  // times the middle, then the exact backward transform, times the epilogue and the scales. The
  // tolerance: two transforms of n points (each within derived_tolerance), the first's error
  // weighed by the middle factor's spread (an error at a bin of large factor against an output of
  // the factors' typical size) and the output's by the epilogue's, plus a rounding for the
  // input, each multiplication (prologue, middle, epilogue, the scale, the spectrum's rounding to
  // double in the reference) and the accumulation.
  void check(const cvec<T>& y, const std::string& describe) const {
    const std::string what = std::string(c.name) + ", n = " + std::to_string(c.n) + ", " +
                             fftn::isa_name(this->target) + (sizeof(T) == 8 ? " double" : " float");
    SCOPED_TRACE(what);
    const double scale = c.options_scale * c.pro_scale * this->mid_scale * c.epi_scale;
    const fftn::test::tolerance one =
        fftn::test::derived_tolerance<T>(c.n, fftn::test::fftn_bound(c.n));
    const fftn::test::tolerance tol =
        one * ((1.0 + fmid.spread()) * fepi.spread()) + fftn::test::input_rounding<T>() * 7;
    std::vector<bool> described(y.size(), false);
    fftn::test::outcome o;
    for (std::size_t r = 0; r < c.count; ++r) {
      std::vector<std::complex<double>> xin(c.n);
      for (std::size_t j = 0; j < c.n; ++j) {
        xin[j] = std::complex<double>(x[in_at(j, r)]) * fpro.at(c.pro, j, r, c.n, c.count);
      }
      const auto spectrum = fftn::test::reference::c2c(-1, xin);
      std::vector<std::complex<double>> mid(c.n);
      for (std::size_t k = 0; k < c.n; ++k) {
        mid[k] = std::complex<double>(spectrum[k].re.hi, spectrum[k].im.hi) *
                 fmid.at(c.mid, k, r, c.n, c.count);
      }
      const auto back = fftn::test::reference::c2c(+1, mid);
      std::vector<fftn::test::exact_complex> want(c.n);
      cvec<T> got(c.n);
      for (std::size_t j = 0; j < c.n; ++j) {
        std::complex<double> v = std::complex<double>(back[j].re.hi, back[j].im.hi) *
                                 fepi.at(c.epi, j, r, c.n, c.count) * scale;
        if (c.accumulate) {
          v += std::complex<double>(init[out_at(j, r)]);
        }
        want[j] = fftn::test::exact_from(v);
        got[j] = y[out_at(j, r)];
        described[out_at(j, r)] = true;
      }
      o.within(fftn::test::compare<T>(std::span<const std::complex<T>>(got), want), tol,
               what + ", line " + std::to_string(r));
    }
    EXPECT_TRUE(o.passed()) << o.detail() << describe;
    if (!c.in_place) {
      std::size_t touched = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        if (!described[i] && !same_bits(cvec<T>{y[i]}, cvec<T>{init[i]})) {
          ++touched;
        }
      }
      EXPECT_EQ(touched, 0U) << "elements outside the lines written";
    }
  }
};

// Whether the factor arrays of a run hold what they held before execution (`before`).
template <class T> bool same_factors(const factor_array<T>& a, const factor_array<T>& before) {
  return same_bits(a.reals, before.reals) && same_bits(a.complexes, before.complexes);
}

// The pipeline of a run (pipe_run, real_run), executed serially (checked against the reference
// where `reference`), in two pieces of the batch and, with `threads`, by teams of three
// (completion::all and own) and on a pool of three: each from the same output buffer, all bitwise
// alike.
template <class Run> void run_on(const Run& run, bool threads, bool reference) {
  using buffer = decltype(run.x);
  const pipe_case& pc = run.c;
  const typename Run::pipe p(run.desc, run.opts);
  const std::string describe = p.describe();
  EXPECT_NE(describe.find(pc.expect), std::string::npos)
      << pc.name << ": no '" << pc.expect << "' in\n"
      << describe;
  EXPECT_EQ(p.instruction_set(), run.target == fftn::isa::best ? p.instruction_set() : run.target);
  const fftn::exec_args args = run.args();
  const buffer x0 = run.x;
  const auto fpro = run.fpro;
  const auto fmid = run.fmid;
  const auto fepi = run.fepi;

  const auto fresh = [&] { return pc.in_place ? run.x : run.init; };
  // Element 0 of line 0 in the input and the output buffers.
  const auto in_of = [&](buffer& y, const buffer& xs) {
    return (pc.in_place ? y.data() : xs.data()) + run.in_base;
  };
  const auto out_of = [&](buffer& y) { return y.data() + run.out_base; };

  fftn::workspace ws(p, 3);
  buffer xs = run.x;
  buffer serial = fresh();
  p.execute(in_of(serial, xs), out_of(serial), ws, args);
  if (reference) {
    run.check(serial, describe);
  }
  if (!pc.in_place) {
    EXPECT_TRUE(same_bits(xs, x0)) << pc.name << ": input written";
  }
  EXPECT_TRUE(same_factors(run.fpro, fpro) && same_factors(run.fmid, fmid) &&
              same_factors(run.fepi, fepi))
      << pc.name << ": factor written";

  buffer pieces = fresh();
  const std::size_t half = pc.count / 2;
  p.execute_transforms(0, half, in_of(pieces, xs), out_of(pieces), ws, args);
  p.execute_transforms(half, pc.count - half, in_of(pieces, xs), out_of(pieces), ws, args);
  EXPECT_TRUE(same_bits(pieces, serial)) << pc.name << ": pieces of the batch differ";
  if (!threads) {
    return;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    buffer teamed = fresh();
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        p.execute(in_of(teamed, xs), out_of(teamed), fftn::team{.rank = r, .size = 3}, ws, done,
                  args);
      });
    }
    p.execute(in_of(teamed, xs), out_of(teamed), fftn::team{.rank = 0, .size = 3}, ws, done, args);
    for (std::thread& m : members) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << pc.name << ": a team differs";
  }
  // The completion::all form without an explicit completion.
  buffer teamed = fresh();
  std::thread member([&] {
    p.execute(in_of(teamed, xs), out_of(teamed), fftn::team{.rank = 1, .size = 2}, ws, args);
  });
  p.execute(in_of(teamed, xs), out_of(teamed), fftn::team{.rank = 0, .size = 2}, ws, args);
  member.join();
  EXPECT_TRUE(same_bits(teamed, serial)) << pc.name << ": a team of two differs";
  const fftn::thread_pool pool(3);
  buffer pooled = fresh();
  p.execute(in_of(pooled, xs), out_of(pooled), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << pc.name << ": the executor differs";
}

// A <C, C, C> case (pipe_run) through run_on.
template <class T>
void run_case(const pipe_case& pc, fftn::isa t, bool threads, bool reference = true) {
  fftn::test::input_source src(0x919EULL + pc.n * 31 + pc.count);
  run_on(pipe_run<T>(pc, t, src), threads, reference);
}

// The kernel alone (a complex middle factor of stride 1, one for every line): every length from
// one point to the leaves, codelet radices, mixed, generic, Rader and Bluestein lengths.
std::vector<std::size_t> lengths() {
  std::vector<std::size_t> out;
  for (std::size_t n = 1; n <= 20; ++n) {
    out.push_back(n);
  }
  for (const std::size_t n :
       {23U, 25U, 29U, 32U, 37U, 47U, 53U, 60U, 64U, 97U, 100U, 128U, 210U, 257U}) {
    out.push_back(n);
  }
  return out;
}

// Every placement of the vocabulary: fused factors, separate passes for what the passes lack,
// and the scale wherever it goes.
std::vector<pipe_case> cases() {
  return {
      {.name = "kernel per line, fused",
       .n = 64,
       .count = 3,
       .mid = {cplx, 1, 64},
       .expect = "middle: times a complex factor in the forward chain's last pass"},
      {.name = "kernel reversed, per line backwards",
       .n = 48,
       .count = 3,
       .mid = {cplx, -1, -48},
       .expect = "middle: times a complex factor in a separate pass"},
      {.name = "kernel of stride 2, separate",
       .n = 32,
       .count = 2,
       .mid = {cplx, 2, 1},
       .expect = "middle: times a complex factor in a separate pass"},
      {.name = "kernel broadcast along the line, separate",
       .n = 16,
       .count = 4,
       .mid = {cplx, 0, 1},
       .expect = "middle: times a complex factor in a separate pass"},
      {.name = "real kernel, separate",
       .n = 40,
       .count = 3,
       .mid = {real, 1, 40},
       .expect = "middle: times a real factor in a separate pass"},
      {.name = "real kernel reversed, shared",
       .n = 30,
       .count = 2,
       .mid = {real, -1, 0},
       .expect = "middle: times a real factor in a separate pass"},
      {.name = "no kernel, scales only",
       .n = 36,
       .count = 2,
       .options_scale = 0.5,
       .pro_scale = 3.0,
       .mid = {},
       .expect = "middle: none"},
      {.name = "prologue and epilogue fused, scale on the backward input",
       .n = 60,
       .count = 2,
       .pro = {cplx, 1, 60},
       .epi = {cplx, 1, -60},
       .expect = "scales: one product, applied once in the backward chain's first pass"},
      {.name = "prologue fused, scale on the backward output",
       .n = 50,
       .count = 3,
       .pro = {cplx, 1, 0},
       .expect = "prologue: times a complex factor in the forward chain's first pass"},
      {.name = "epilogue fused, prologue none: scale on the forward input",
       .n = 24,
       .count = 3,
       .epi = {cplx, 1, 24},
       .expect = "scales: one product, applied once in the forward chain's first pass"},
      {.name = "real prologue and epilogue, separate",
       .n = 20,
       .count = 3,
       .pro_scale = 2.0,
       .pro = {real, 1, 20},
       .epi = {real, -1, 0},
       .expect = "the vocabulary costs 2 separate passes"},
      {.name = "accumulation, reversed epilogue factor",
       .n = 24,
       .count = 3,
       .pro = {cplx, 1, 0},
       .epi = {cplx, -1, -24},
       .accumulate = true,
       .expect = "epilogue: times a complex factor, accumulated in a separate pass"},
      {.name = "accumulation alone, every scale",
       .n = 18,
       .count = 2,
       .options_scale = 0.25,
       .pro_scale = 0.5,
       .accumulate = true,
       .expect = "epilogue: accumulated"},
      {.name = "in place, fused",
       .n = 128,
       .count = 2,
       .pro = {cplx, 1, 128},
       .mid = {cplx, 1, 0},
       .epi = {cplx, 1, 0},
       .in_place = true,
       .expect = "in place"},
      {.name = "in place, separate",
       .n = 45,
       .count = 2,
       .pro = {real, 1, 0},
       .mid = {cplx, 3, 0},
       .in_place = true,
       .expect = "separate"},
      {.name = "strided lines, fused",
       .n = 20,
       .count = 3,
       .pro = {cplx, 1, 20},
       .epi = {cplx, 1, 20},
       .in_stride = 2,
       .out_stride = 3,
       .expect = "in the forward chain's first pass"},
      {.name = "strided lines, separate, accumulated",
       .n = 12,
       .count = 2,
       .options_scale = 2.0,
       .pro = {real, -1, 12},
       .mid = {real, 1, 0},
       .epi = {cplx, 3, 0},
       .accumulate = true,
       .in_stride = 3,
       .out_stride = 2,
       .expect = "the vocabulary costs 3 separate passes"},
      {.name = "reversed lines",
       .n = 16,
       .count = 2,
       .in_stride = -1,
       .out_stride = -2,
       .expect = "out of place"},
      {.name = "Bluestein line, fused",
       .n = 47,
       .count = 2,
       .pro = {cplx, 1, 47},
       .mid = {cplx, 1, 47},
       .epi = {cplx, 1, 0},
       .expect = "fftn pipeline"},
      {.name = "Bluestein line, separate",
       .n = 59,
       .count = 2,
       .mid = {real, 1, 0},
       .accumulate = true,
       .expect = "separate"},
      {.name = "lines of one point",
       .n = 1,
       .count = 4,
       .pro = {cplx, 7, 1},
       .mid = {real, 5, -1},
       .epi = {cplx, 3, 2},
       .expect = "n = 1"},
      {.name = "a leaf with every factor fused",
       .n = 8,
       .count = 5,
       .options_scale = 0.125,
       .pro = {cplx, 1, 8},
       .mid = {cplx, 1, 8},
       .epi = {cplx, 1, 8},
       .expect = "form: flat"},
  };
}

template <class T> class Pipelines : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(Pipelines, precisions);

TYPED_TEST(Pipelines, EveryLengthOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths()) {
      run_case<TypeParam>({.name = "kernel", .n = n, .count = 3, .expect = "the chain pair"}, t,
                          false);
    }
  }
}

TYPED_TEST(Pipelines, EveryOperationOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const pipe_case& c : cases()) {
      run_case<TypeParam>(c, t, false);
    }
  }
}

// Longer lines on the best instruction set: their programs unrolled or walked.
TYPED_TEST(Pipelines, LongerLines) {
  for (const std::size_t n : {1000U, 1024U}) {
    run_case<TypeParam>({.name = "long kernel",
                         .n = n,
                         .count = 2,
                         .mid = {cplx, 1, -1},
                         .expect = "the chain pair"},
                        fftn::isa::best, false);
  }
  run_case<TypeParam>({.name = "long, every operation",
                       .n = 1536,
                       .count = 1,
                       .pro = {cplx, 1, 0},
                       .mid = {real, -1, 1536},
                       .epi = {cplx, 1, 1},
                       .accumulate = true,
                       .expect = "accumulated"},
                      fftn::isa::best, false);
}

// Teams and executors on lines short and long: a line of 2^17 points has stages of many chunks,
// which a team splits among its members (beyond the scheduler's split thresholds).
TYPED_TEST(Pipelines, TeamsAndExecutorsAgreeWithSerial) {
  for (const pipe_case& c : cases()) {
    run_case<TypeParam>(c, fftn::isa::best, true);
  }
  run_case<TypeParam>({.name = "many lines",
                       .n = 256,
                       .count = 37,
                       .mid = {cplx, 1, 256},
                       .expect = "the chain pair"},
                      fftn::isa::best, true);
  run_case<TypeParam>({.name = "a line split among a team",
                       .n = 131072,
                       .count = 1,
                       .pro = {cplx, 1, 0},
                       .mid = {cplx, 1, 0},
                       .epi = {real, 1, 0},
                       .accumulate = true,
                       .expect = "the chain pair"},
                      fftn::isa::best, true, false);
}

// Leaf-sized lines are a flat body (the forward leaf with the kernel, then the inverse leaf)
// batched by the slots' steps, on every instruction set, with factors or without.
TEST(Pipelines, LeafSizedLinesAreFlat) {
  using C = std::complex<double>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : {2U, 4U, 8U, 16U}) {
      fftn::pipeline_desc<C, C, C> with = fftn::complex_convolution_rows<double>(n, 7);
      const fftn::pipeline<C, C, C> kernel(with, {.force_isa = t});
      EXPECT_NE(kernel.describe().find("form: flat"), std::string::npos) << kernel.describe();
      const fftn::pipeline<C, C, C> plain(
          fftn::spectral_pair_rows<double>(n, 7, fftn::direction::forward), {.force_isa = t});
      EXPECT_NE(plain.describe().find("form: flat"), std::string::npos) << plain.describe();
    }
  }
}

// The helpers' descriptions: tight rows, a shared kernel of stride 1, and with runs the stored
// extents as distances.
TEST(Pipelines, Helpers) {
  using C = std::complex<float>;
  const fftn::pipeline_desc<C, C, C> d = fftn::complex_convolution_rows<float>(12, 5);
  EXPECT_EQ(d.first, fftn::direction::forward);
  EXPECT_EQ(d.line.n, 12U);
  ASSERT_EQ(d.batches.size(), 1U);
  EXPECT_EQ(d.batches[0].count, 5U);
  EXPECT_EQ(d.batches[0].in_dist, 0);
  EXPECT_EQ(d.middle.mul.kind, cplx);
  EXPECT_EQ(d.middle.mul.axis_strides, std::vector<std::ptrdiff_t>{1});
  EXPECT_TRUE(d.middle.mul.batch_strides.empty());
  EXPECT_EQ(d.middle.scale, 1.0);
  const std::vector<fftn::stored_run> runs{
      {.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = 6},
      {.axis = 0, .where = fftn::side::out, .logical = 2, .stored = 1, .count = 3}};
  const fftn::pipeline_desc<C, C, C> r = fftn::complex_convolution_rows<float>(12, 5, runs);
  EXPECT_EQ(r.batches[0].in_dist, 6);
  EXPECT_EQ(r.batches[0].out_dist, 4);
  EXPECT_EQ(r.runs.size(), 2U);
  const auto pair = fftn::spectral_pair_rows<float>(12, 5, fftn::direction::backward);
  EXPECT_EQ(pair.first, fftn::direction::backward);
  EXPECT_EQ(pair.middle.mul.kind, factor_kind::none);
}

// transforms_of deals lines as a plan of the same batch deals transforms; workspace_bytes grows
// with the team; options::effort is ignored, and describe() says so.
TEST(Pipelines, DealingWorkspaceAndPlanning) {
  using C = std::complex<double>;
  const auto d = fftn::complex_convolution_rows<double>(64, 10);
  const fftn::pipeline<C, C, C> p(d);
  const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(64, 10, fftn::direction::forward));
  for (int r = 0; r < 4; ++r) {
    const fftn::team t{.rank = r, .size = 4};
    EXPECT_EQ(p.transforms_of(t).first, plan.transforms_of(t).first);
    EXPECT_EQ(p.transforms_of(t).count, plan.transforms_of(t).count);
  }
  EXPECT_GT(p.workspace_bytes(), 64 * sizeof(C));
  EXPECT_GE(p.workspace_bytes(4), p.workspace_bytes());
  EXPECT_EQ(fftn::workspace(p, 4).size(), p.workspace_bytes(4));
  const fftn::pipeline<C, C, C> measured(d, {.effort = fftn::planning::quick});
  EXPECT_NE(measured.describe().find("options.effort and options.use are ignored"),
            std::string::npos)
      << measured.describe();
  EXPECT_EQ(p.describe().find("planning:"), std::string::npos) << p.describe();
}

template <class In, class Mid, class Out>
void expect_refused(const fftn::pipeline_desc<In, Mid, Out>& d, const fftn::options& opts,
                    const std::string& field) {
  try {
    const fftn::pipeline<In, Mid, Out> p(d, opts);
    ADD_FAILURE() << "not refused: " << field;
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(field), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("fftn::pipeline: "), std::string::npos) << e.what();
  }
}

TEST(Pipelines, RefusalsNameTheField) {
  using C = std::complex<double>;
  using F = std::complex<float>;
  const fftn::pipeline_desc<C, C, C> good = fftn::complex_convolution_rows<double>(16, 2);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto d = good;
  // Not built yet.
  d.first = fftn::direction::backward;
  expect_refused(d, {}, "desc.first");
  d = good;
  d.first = static_cast<fftn::direction>(3);
  expect_refused(d, {}, "desc.first");
  d = good;
  d.fanout = 2;
  expect_refused(d, {}, "desc.fanout");
  d.fanout = 0;
  expect_refused(d, {}, "desc.fanout");
  d = good;
  d.runs = {{.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = 8}};
  expect_refused(d, {}, "desc.runs");
  fftn::pipeline_desc<C, double, C> real_middle;
  real_middle.line = {.n = 16};
  expect_refused(real_middle, {}, "desc");
  // The layout.
  d = good;
  d.line.n = 0;
  expect_refused(d, {}, "desc.line.n");
  d = good;
  d.batches[0].count = 0;
  expect_refused(d, {}, "desc.batches[0].count");
  d = good;
  d.batches[0].out_dist = 8;
  expect_refused(d, {}, "desc describes output elements that overlap");
  d = good;
  d.line.out_stride = std::numeric_limits<std::ptrdiff_t>::min();
  expect_refused(d, {}, "desc.line.out_stride");
  d = good;
  d.line.out_stride = 2;
  d.batches[0].out_dist = 32;
  expect_refused(d, {.in_place = true}, "options.in_place");
  expect_refused(good, {.measure_seconds = -1.0}, "options.measure_seconds");
  // The vocabulary.
  d = good;
  d.prologue.accumulate = true;
  expect_refused(d, {}, "desc.prologue.accumulate");
  d = good;
  d.epilogue.accumulate = true;
  expect_refused(d, {.in_place = true}, "desc.epilogue.accumulate");
  d = good;
  d.epilogue.scale = 2.0;
  expect_refused(d, {.scale = 0.5}, "desc.epilogue.scale");
  d = good;
  d.middle.scale = nan;
  expect_refused(d, {}, "desc.middle.scale");
  d = good;
  d.middle.mul.axis_strides = {1, 1};
  expect_refused(d, {}, "desc.middle.mul.axis_strides");
  d = good;
  d.middle.mul.batch_strides = {1, 2};
  expect_refused(d, {}, "desc.middle.mul.batch_strides");
  d = good;
  d.middle.mul.kind = static_cast<factor_kind>(9);
  expect_refused(d, {}, "desc.middle.mul.kind");
  d = good;
  d.prologue.mul = {.kind = real, .axis_strides = {std::numeric_limits<std::ptrdiff_t>::max()}};
  expect_refused(d, {}, "desc.prologue.mul.axis_strides[0]");
  d = good;
  d.epilogue.mul = {.kind = cplx, .axis_strides = {}};
  expect_refused(d, {}, "desc.epilogue.mul.axis_strides");
  expect_refused(good, {.scale = nan}, "options.scale");
  fftn::pipeline_desc<F, F, F> f = fftn::complex_convolution_rows<float>(16, 2);
  f.prologue.scale = 1e30;
  f.middle.scale = 1e30;
  expect_refused(f, {}, "desc.prologue.scale");
  f.prologue.scale = 1e300;
  expect_refused(f, {}, "desc.prologue.scale");
}

// `execute(in, out, team, ws, {})` is the team form with completion::all.
template <class C>
concept braces_bind_team =
    requires(const fftn::pipeline<C, C, C>& p, const C* in, C* out, fftn::workspace& ws) {
      p.execute(in, out, fftn::team{}, ws, {});
    };
static_assert(braces_bind_team<std::complex<double>>);
// No execute without the execution's data.
template <class C>
concept bare_execute = requires(const fftn::pipeline<C, C, C>& p, const C* in, C* out,
                                fftn::workspace& ws) { p.execute(in, out, ws); };
static_assert(!bare_execute<std::complex<double>>);

// Executions without a factor array the pipeline needs end the program, in every build.
TEST(PipelinesDeathTest, MissingFactorArraysAbort) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using C = std::complex<double>;
  const fftn::pipeline<C, C, C> p(fftn::complex_convolution_rows<double>(32, 2));
  auto separate_desc = fftn::complex_convolution_rows<double>(32, 2);
  separate_desc.middle.mul.kind = real;
  const fftn::pipeline<C, C, C> separate(separate_desc);
  cvec<double> x(64);
  cvec<double> y(64);
  const cvec<double> k(32);
  fftn::workspace ws(p, 2);
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws, {}), "without the factor arrays");
  EXPECT_DEATH(p.execute_transforms(1, 1, x.data(), y.data(), ws, {}), "without the factor arrays");
  EXPECT_DEATH(separate.execute(x.data(), y.data(), ws, {}), "without the factor arrays");
  // The wrong array: the input's given, the middle's needed.
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws, {.in_factor = k.data()}),
               "without the factor arrays");
  EXPECT_DEATH(p.execute(x.data(), y.data(), fftn::team{}, ws, {}), "without the factor arrays");
  EXPECT_DEATH(
      {
        const fftn::thread_pool pool(2);
        p.execute(x.data(), y.data(), pool, ws, {});
      },
      "without the factor arrays");
}

// <T, C, T>, the real chain pair (task 5.2c): real lines of the same layouts, the middle over the
// n/2 + 1 bins of the r2c transform.

template <class T> using real_desc_of = fftn::pipeline_desc<T, std::complex<T>, T>;
template <class T> using real_pipe_of = fftn::pipeline<T, std::complex<T>, T>;

template <class T> struct real_run : run_layout<T> {
  using pipe = real_pipe_of<T>;
  using run_layout<T>::c;
  using run_layout<T>::fpro;
  using run_layout<T>::fmid;
  using run_layout<T>::fepi;
  using run_layout<T>::in_at;
  using run_layout<T>::out_at;
  real_desc_of<T> desc;
  std::vector<T> x;    // the input buffer, gaps included
  std::vector<T> init; // the output buffer before execution (accumulation, canaries)

  real_run(const pipe_case& pc, fftn::isa t, fftn::test::input_source& src)
      : run_layout<T>(pc, t, pc.n / 2 + 1, src) {
    this->describe_into(desc);
    x = src.reals<T>(static_cast<std::size_t>(this->in_dist) * c.count);
    init = c.in_place ? x : src.reals<T>(static_cast<std::size_t>(this->out_dist) * c.count);
  }

  // Checks y as pipe_run::check does, the reference being the exact r2c transform of the input
  // times the prologue, rounded to double, times the middle on the n/2 + 1 bins, then the exact
  // c2r transform (which ignores the imaginary parts of bin 0 and, for even n, of bin n/2, as the
  // pipeline's c2r chain does), times the epilogue and the scales. The tolerance is pipe_run's.
  void check(const std::vector<T>& y, const std::string& describe) const {
    const std::string what = std::string(c.name) + ", n = " + std::to_string(c.n) + ", " +
                             fftn::isa_name(this->target) +
                             (sizeof(T) == 8 ? " double" : " float") + ", real";
    SCOPED_TRACE(what);
    const double scale = c.options_scale * c.pro_scale * this->mid_scale * c.epi_scale;
    const std::size_t bins = c.n / 2 + 1;
    const fftn::test::tolerance one =
        fftn::test::derived_tolerance<T>(c.n, fftn::test::fftn_bound(c.n));
    const fftn::test::tolerance tol =
        one * ((1.0 + fmid.spread()) * fepi.spread()) + fftn::test::input_rounding<T>() * 7;
    std::vector<bool> described(y.size(), false);
    fftn::test::outcome o;
    for (std::size_t r = 0; r < c.count; ++r) {
      std::vector<double> xin(c.n);
      for (std::size_t j = 0; j < c.n; ++j) {
        xin[j] = static_cast<double>(x[in_at(j, r)]) * fpro.at(c.pro, j, r, c.n, c.count).real();
      }
      const auto spectrum = fftn::test::reference::r2c(xin);
      std::vector<std::complex<double>> mid(bins);
      for (std::size_t k = 0; k < bins; ++k) {
        mid[k] = std::complex<double>(spectrum[k].re.hi, spectrum[k].im.hi) *
                 fmid.at(c.mid, k, r, bins, c.count);
      }
      const auto back = fftn::test::reference::c2r(c.n, mid);
      std::vector<fftn::test::exact_real> want(c.n);
      std::vector<T> got(c.n);
      for (std::size_t j = 0; j < c.n; ++j) {
        double v = back[j].hi * fepi.at(c.epi, j, r, c.n, c.count).real() * scale;
        if (c.accumulate) {
          v += static_cast<double>(init[out_at(j, r)]);
        }
        want[j] = fftn::test::exact_from(v);
        got[j] = y[out_at(j, r)];
        described[out_at(j, r)] = true;
      }
      o.within(fftn::test::compare<T>(std::span<const T>(got), want), tol,
               what + ", line " + std::to_string(r));
    }
    EXPECT_TRUE(o.passed()) << o.detail() << describe;
    if (!c.in_place) {
      std::size_t touched = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        if (!described[i] && !same_bits(std::vector<T>{y[i]}, std::vector<T>{init[i]})) {
          ++touched;
        }
      }
      EXPECT_EQ(touched, 0U) << "elements outside the lines written";
    }
  }
};

template <class T>
void run_real_case(const pipe_case& pc, fftn::isa t, bool threads, bool reference = true) {
  fftn::test::input_source src(0x7EA1ULL + pc.n * 31 + pc.count);
  run_on(real_run<T>(pc, t, src), threads, reference);
}

// Even and odd lengths: one point, the pairing alone (2), leaves of the half length, fused and
// separate pairings, odd lengths by the full-length chain, Rader and Bluestein lengths.
std::vector<std::size_t> real_lengths() {
  std::vector<std::size_t> out;
  for (std::size_t n = 1; n <= 34; ++n) {
    out.push_back(n);
  }
  for (const std::size_t n : {38U, 45U, 46U, 47U, 48U, 59U, 64U, 94U, 96U, 100U, 128U, 162U, 210U,
                              256U, 257U, 320U, 514U}) {
    out.push_back(n);
  }
  return out;
}

// Every placement of the vocabulary on real lines (every factor a separate pass on the line in
// the workspace block; the scale on such a pass or on the c2r chain's output).
std::vector<pipe_case> real_cases() {
  constexpr const char* bins_complex =
      "middle: times a complex factor in a separate pass on the line's n/2 + 1 bins";
  constexpr const char* bins_real =
      "middle: times a real factor in a separate pass on the line's n/2 + 1 bins";
  return {
      {.name = "kernel per line",
       .n = 64,
       .count = 3,
       .mid = {cplx, 1, 33},
       .expect = bins_complex},
      {.name = "kernel reversed, per line backwards",
       .n = 48,
       .count = 3,
       .mid = {cplx, -1, -25},
       .expect = bins_complex},
      {.name = "kernel of stride 2",
       .n = 32,
       .count = 2,
       .mid = {cplx, 2, 1},
       .expect = bins_complex},
      {.name = "kernel broadcast along the line",
       .n = 16,
       .count = 4,
       .mid = {cplx, 0, 1},
       .expect = bins_complex},
      {.name = "real kernel", .n = 40, .count = 3, .mid = {real, 1, 21}, .expect = bins_real},
      {.name = "real kernel reversed, shared",
       .n = 30,
       .count = 2,
       .mid = {real, -1, 0},
       .expect = bins_real},
      {.name = "odd line, kernel per line",
       .n = 27,
       .count = 3,
       .mid = {cplx, 1, 14},
       .expect = "full-length complex chain (n odd)"},
      {.name = "no kernel, scales only",
       .n = 36,
       .count = 2,
       .options_scale = 0.5,
       .pro_scale = 3.0,
       .mid = {},
       .expect = "middle: none"},
      {.name = "scale on the c2r chain's output",
       .n = 50,
       .count = 2,
       .mid = {},
       .expect = "scales: one product, applied once in the backward chain's last pass"},
      {.name = "real prologue and epilogue",
       .n = 20,
       .count = 3,
       .pro_scale = 2.0,
       .pro = {real, 1, 20},
       .epi = {real, -1, 0},
       .expect = "the vocabulary costs 3 separate passes"},
      {.name = "accumulation, reversed epilogue factor",
       .n = 24,
       .count = 3,
       .pro = {real, 1, 0},
       .epi = {real, -1, -24},
       .accumulate = true,
       .expect = "epilogue: times a real factor, accumulated in a separate pass"},
      {.name = "accumulation alone, every scale",
       .n = 18,
       .count = 2,
       .options_scale = 0.25,
       .pro_scale = 0.5,
       .accumulate = true,
       .expect = "epilogue: accumulated"},
      {.name = "in place, every factor",
       .n = 128,
       .count = 2,
       .pro = {real, 1, 128},
       .mid = {cplx, 1, 0},
       .epi = {real, 1, 0},
       .in_place = true,
       .expect = "in place"},
      {.name = "in place, odd",
       .n = 45,
       .count = 2,
       .mid = {real, 3, 0},
       .in_place = true,
       .expect = "in place"},
      {.name = "strided lines",
       .n = 20,
       .count = 3,
       .mid = {cplx, 1, 11},
       .in_stride = 2,
       .out_stride = 3,
       .expect = "(strided line)"},
      {.name = "strided lines, every factor, accumulated",
       .n = 12,
       .count = 2,
       .options_scale = 2.0,
       .pro = {real, -1, 12},
       .mid = {real, 1, 0},
       .epi = {real, 3, 0},
       .accumulate = true,
       .in_stride = 3,
       .out_stride = 2,
       .expect = "the vocabulary costs 3 separate passes"},
      {.name = "odd strided lines",
       .n = 15,
       .count = 2,
       .in_stride = 2,
       .out_stride = -1,
       .expect = "full-length complex chain (n odd)"},
      {.name = "reversed lines",
       .n = 16,
       .count = 2,
       .in_stride = -1,
       .out_stride = -2,
       .expect = "out of place"},
      {.name = "Bluestein half length",
       .n = 94,
       .count = 2,
       .mid = {cplx, 1, 48},
       .expect = "the real chain pair"},
      {.name = "Bluestein odd line, accumulated",
       .n = 59,
       .count = 2,
       .mid = {real, 1, 0},
       .accumulate = true,
       .expect = "separate"},
      {.name = "lines of one point",
       .n = 1,
       .count = 4,
       .pro = {real, 7, 1},
       .mid = {cplx, 5, -1},
       .epi = {real, 3, 2},
       .expect = "n = 1"},
      {.name = "lines of two points",
       .n = 2,
       .count = 5,
       .mid = {cplx, 1, 2},
       .expect = "the pairing pass alone"},
      {.name = "a leaf",
       .n = 8,
       .count = 5,
       .options_scale = 0.125,
       .mid = {cplx, 1, 5},
       .expect = "form: flat"},
  };
}

template <class T> class RealPipelines : public ::testing::Test {};
TYPED_TEST_SUITE(RealPipelines, precisions);

TYPED_TEST(RealPipelines, EveryLengthOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : real_lengths()) {
      run_real_case<TypeParam>(
          {.name = "kernel", .n = n, .count = 3, .expect = "fftn pipeline: <t, c, t> r2c then c2r"},
          t, false);
    }
  }
}

TYPED_TEST(RealPipelines, EveryOperationOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const pipe_case& c : real_cases()) {
      run_real_case<TypeParam>(c, t, false);
    }
  }
}

// Longer lines on the best instruction set: their programs unrolled or walked.
TYPED_TEST(RealPipelines, LongerLines) {
  for (const std::size_t n : {999U, 1000U, 1024U, 2048U}) {
    run_real_case<TypeParam>({.name = "long kernel",
                              .n = n,
                              .count = 2,
                              .mid = {cplx, 1, -1},
                              .expect = "the real chain pair"},
                             fftn::isa::best, false);
  }
  run_real_case<TypeParam>({.name = "long, every operation",
                            .n = 1536,
                            .count = 1,
                            .pro = {real, 1, 0},
                            .mid = {real, -1, 769},
                            .epi = {real, 1, 1},
                            .accumulate = true,
                            .expect = "accumulated"},
                           fftn::isa::best, false);
}

// Teams and executors on lines short and long: a line of 2^18 points has stages of many chunks,
// which a team splits among its members.
TYPED_TEST(RealPipelines, TeamsAndExecutorsAgreeWithSerial) {
  for (const pipe_case& c : real_cases()) {
    run_real_case<TypeParam>(c, fftn::isa::best, true);
  }
  run_real_case<TypeParam>({.name = "many lines",
                            .n = 256,
                            .count = 37,
                            .mid = {cplx, 1, 129},
                            .expect = "the real chain pair"},
                           fftn::isa::best, true);
  run_real_case<TypeParam>({.name = "a line split among a team",
                            .n = 262144,
                            .count = 1,
                            .pro = {real, 1, 0},
                            .mid = {cplx, 1, 0},
                            .epi = {real, 1, 0},
                            .accumulate = true,
                            .expect = "the real chain pair"},
                           fftn::isa::best, true, false);
}

// convolution_rows: each row's circular convolution with a kernel, here a linear one of a row of
// m values and a kernel of `taps` padded with zeros to n >= m + taps - 1, against the direct
// convolution. Rows and kernel are small integers, so the direct convolution is exact in double;
// the kernel's spectrum is the exact r2c transform rounded to T. Tolerance: two transforms, the
// first's error weighed by the spectrum's spread, plus the spectrum's rounding and the scale's.
template <class T>
void convolution_case(std::size_t n, std::size_t m, std::size_t taps, std::size_t count,
                      fftn::isa t) {
  const std::string what = "convolution n = " + std::to_string(n) + ", m = " + std::to_string(m) +
                           ", taps = " + std::to_string(taps) + ", " + fftn::isa_name(t) +
                           (sizeof(T) == 8 ? " double" : " float");
  SCOPED_TRACE(what);
  fftn::test::input_source src(0xC0417ULL + n);
  const auto small = [](T v) { return static_cast<T>(std::round(static_cast<double>(v) * 8.0)); };
  std::vector<T> x(n * count, T(0));
  const std::vector<T> xs = src.reals<T>(m * count);
  for (std::size_t r = 0; r < count; ++r) {
    for (std::size_t i = 0; i < m; ++i) {
      x[r * n + i] = small(xs[r * m + i]);
    }
  }
  std::vector<double> h(n, 0.0);
  const std::vector<T> hs = src.reals<T>(taps);
  for (std::size_t i = 0; i < taps; ++i) {
    h[i] = static_cast<double>(small(hs[i]));
  }
  const auto spectrum = fftn::test::reference::r2c(h);
  cvec<T> kernel(n / 2 + 1);
  double peak = 0.0;
  double sum = 0.0;
  for (std::size_t k = 0; k < kernel.size(); ++k) {
    kernel[k] = {static_cast<T>(spectrum[k].re.hi), static_cast<T>(spectrum[k].im.hi)};
    const double a = std::abs(std::complex<double>(kernel[k]));
    peak = std::max(peak, a);
    sum += a * a;
  }
  const double spread =
      sum == 0.0 ? 1.0 : peak / std::sqrt(sum / static_cast<double>(kernel.size()));

  real_desc_of<T> desc = fftn::convolution_rows<T>(n, count);
  desc.middle.scale = 1.0 / static_cast<double>(n);
  const real_pipe_of<T> p(desc, {.force_isa = t});
  const std::string text = p.describe();
  EXPECT_NE(text.find("the real chain pair"), std::string::npos) << text;
  fftn::workspace ws(p);
  std::vector<T> y(n * count, T(7));
  p.execute(x.data(), y.data(), ws, {.mid_factor = kernel.data()});

  const fftn::test::tolerance tol =
      fftn::test::derived_tolerance<T>(n, fftn::test::fftn_bound(n)) * (1.0 + spread) +
      fftn::test::input_rounding<T>() * 3;
  fftn::test::outcome o;
  for (std::size_t r = 0; r < count; ++r) {
    std::vector<fftn::test::exact_real> want(n);
    for (std::size_t j = 0; j < n; ++j) {
      double v = 0.0;
      for (std::size_t i = 0; i <= j && i < m; ++i) {
        if (j - i < taps) {
          v += static_cast<double>(x[r * n + i]) * h[j - i];
        }
      }
      want[j] = fftn::test::exact_from(v);
    }
    o.within(fftn::test::compare<T>(std::span<const T>(y.data() + r * n, n), want), tol,
             what + ", row " + std::to_string(r));
  }
  EXPECT_TRUE(o.passed()) << o.detail() << text;
}

TYPED_TEST(RealPipelines, ConvolutionRowsAgainstTheDirectConvolution) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    convolution_case<TypeParam>(2, 1, 2, 3, t);
    convolution_case<TypeParam>(7, 4, 4, 2, t);
    convolution_case<TypeParam>(16, 9, 8, 4, t);
    convolution_case<TypeParam>(45, 30, 16, 3, t);
    convolution_case<TypeParam>(64, 40, 25, 5, t);
    convolution_case<TypeParam>(100, 60, 41, 2, t);
    convolution_case<TypeParam>(210, 150, 61, 2, t);
  }
}

// Leaf-sized real lines are a flat body batched by the slots' steps, with the kernel or without.
TEST(RealPipelines, LeafSizedLinesAreFlat) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : {2U, 4U, 8U, 16U, 32U}) {
      const real_pipe_of<double> kernel(fftn::convolution_rows<double>(n, 7), {.force_isa = t});
      EXPECT_NE(kernel.describe().find("form: flat"), std::string::npos) << kernel.describe();
      real_desc_of<float> plain_desc = fftn::convolution_rows<float>(n, 7);
      plain_desc.middle.mul = {};
      const real_pipe_of<float> plain(plain_desc, {.force_isa = t});
      EXPECT_NE(plain.describe().find("form: flat"), std::string::npos) << plain.describe();
    }
  }
}

// The helper's description: tight rows, a shared kernel of stride 1 over the bins, and with runs
// the stored extents as distances; a pipeline built from it ignores `first`.
TEST(RealPipelines, Helper) {
  const real_desc_of<double> d = fftn::convolution_rows<double>(12, 5);
  EXPECT_EQ(d.line.n, 12U);
  ASSERT_EQ(d.batches.size(), 1U);
  EXPECT_EQ(d.batches[0].count, 5U);
  EXPECT_EQ(d.batches[0].in_dist, 0);
  EXPECT_EQ(d.batches[0].out_dist, 0);
  EXPECT_EQ(d.middle.mul.kind, cplx);
  EXPECT_EQ(d.middle.mul.axis_strides, std::vector<std::ptrdiff_t>{1});
  EXPECT_TRUE(d.middle.mul.batch_strides.empty());
  EXPECT_EQ(d.middle.scale, 1.0);
  EXPECT_EQ(d.prologue.mul.kind, factor_kind::none);
  EXPECT_EQ(d.epilogue.mul.kind, factor_kind::none);
  const std::vector<fftn::stored_run> runs{
      {.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = 6},
      {.axis = 0, .where = fftn::side::out, .logical = 2, .stored = 1, .count = 3}};
  const real_desc_of<float> r = fftn::convolution_rows<float>(12, 5, runs);
  EXPECT_EQ(r.batches[0].in_dist, 6);
  EXPECT_EQ(r.batches[0].out_dist, 4);
  EXPECT_EQ(r.runs.size(), 2U);
  const real_pipe_of<double> p(d);
  EXPECT_GE(p.workspace_bytes(), 7 * sizeof(std::complex<double>));
  EXPECT_EQ(fftn::workspace(p, 2).size(), p.workspace_bytes(2));
  real_desc_of<double> backward = d;
  backward.first = fftn::direction::backward;
  const real_pipe_of<double> q(backward);
  std::vector<double> in(60);
  for (std::size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<double>((i * 7) % 13) - 6.0;
  }
  const cvec<double> k(7, {0.5, -0.25});
  std::vector<double> y1(60);
  std::vector<double> y2(60);
  fftn::workspace ws(p);
  p.execute(in.data(), y1.data(), ws, {.mid_factor = k.data()});
  q.execute(in.data(), y2.data(), ws, {.mid_factor = k.data()});
  EXPECT_TRUE(same_bits(y1, y2));
}

TEST(RealPipelines, RefusalsNameTheField) {
  const real_desc_of<double> good = fftn::convolution_rows<double>(16, 2);
  auto d = good;
  d.prologue.mul = {.kind = cplx, .axis_strides = {1}};
  expect_refused(d, {}, "desc.prologue.mul.kind");
  d = good;
  d.epilogue.mul = {.kind = cplx, .axis_strides = {1}};
  expect_refused(d, {}, "desc.epilogue.mul.kind");
  d = good;
  d.middle.mul.axis_strides = {std::numeric_limits<std::ptrdiff_t>::max()};
  expect_refused(d, {}, "desc.middle.mul.axis_strides[0]");
  d = good;
  d.middle.mul.batch_strides = {1, 2};
  expect_refused(d, {}, "desc.middle.mul.batch_strides");
  d = good;
  d.middle.mul.kind = static_cast<factor_kind>(9);
  expect_refused(d, {}, "desc.middle.mul.kind");
  d = good;
  d.fanout = 2;
  expect_refused(d, {}, "desc.fanout");
  d = good;
  d.runs = {{.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = 8}};
  expect_refused(d, {}, "desc.runs");
  d = good;
  d.line.n = 0;
  expect_refused(d, {}, "desc.line.n");
  d = good;
  d.batches[0].out_dist = 8;
  expect_refused(d, {}, "desc describes output elements that overlap");
  d = good;
  d.line.out_stride = 2;
  d.batches[0].out_dist = 32;
  expect_refused(d, {.in_place = true}, "options.in_place");
  d = good;
  d.prologue.accumulate = true;
  expect_refused(d, {}, "desc.prologue.accumulate");
  d = good;
  d.epilogue.accumulate = true;
  expect_refused(d, {.in_place = true}, "desc.epilogue.accumulate");
  d = good;
  d.epilogue.scale = 2.0;
  expect_refused(d, {.scale = 0.5}, "desc.epilogue.scale");
  real_desc_of<float> f = fftn::convolution_rows<float>(16, 2);
  f.prologue.scale = 1e30;
  f.middle.scale = 1e30;
  expect_refused(f, {}, "desc.prologue.scale");
  // The middle's reach is over the n/2 + 1 = 9 bins, not the n points: a stride whose reach over
  // 8 steps of 16 bytes fits ptrdiff_t is accepted, one beyond it refused.
  constexpr std::ptrdiff_t most = std::numeric_limits<std::ptrdiff_t>::max() / 16 / 8;
  d = good;
  d.middle.mul.axis_strides = {most};
  EXPECT_NO_THROW(real_pipe_of<double>(d, {}));
  d.middle.mul.axis_strides = {most + 1};
  expect_refused(d, {}, "desc.middle.mul.axis_strides[0]");
}

TEST(RealPipelinesDeathTest, MissingFactorArraysAbort) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const real_pipe_of<double> p(fftn::convolution_rows<double>(32, 2));
  auto edges_desc = fftn::convolution_rows<double>(32, 2);
  edges_desc.prologue.mul = {.kind = real, .axis_strides = {1}};
  const real_pipe_of<double> edges(edges_desc);
  std::vector<double> x(64);
  std::vector<double> y(64);
  const cvec<double> k(17);
  fftn::workspace ws(p, 2);
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws, {}), "without the factor arrays");
  EXPECT_DEATH(edges.execute(x.data(), y.data(), ws, {.mid_factor = k.data()}),
               "without the factor arrays");
  EXPECT_DEATH(p.execute(x.data(), y.data(), fftn::team{}, ws, {}), "without the factor arrays");
}

// PR 140's latent finding: in the edge driver's row-block form (make_edge_forms with a row
// block), a separate middle pass is one call over a block's rows whose factor moves by the batch
// loop's step of the middle's cursor. Pipelines build no row blocks yet (5.2d and 5.2e may): this
// builds them on a pipeline's program, with a workspace block of one line per row, and checks
// them against the flat form, bitwise, with a kernel per line. Without the fix the middle pass ran
// row by row (its name was not an edge pass's), and with the name alone its factor moved by the
// prologue's step.
TEST(Pipelines, RowBlocksMoveTheMiddlesFactor) {
  namespace d = fftn::detail;
  using C = std::complex<double>;
  constexpr std::size_t n = 8;
  constexpr std::size_t count = 7;
  constexpr std::size_t rows = 3;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const factor_kind kind : {real, cplx}) {
      desc_of<double> desc = fftn::complex_convolution_rows<double>(n, count);
      // Stride 2 along the line: a separate middle pass.
      desc.middle.mul = {.kind = kind, .axis_strides = {2}, .batch_strides = {2 * n}};
      const d::pipeline_view view{
          .kind = d::pipeline_kind::complex_pair,
          .prec = d::precision::f64,
          .line = desc.line,
          .batches = desc.batches,
          .runs = desc.runs,
          .prologue = d::view_of(desc.prologue),
          .middle = {.scale = 1.0 / n,
                     .kind = kind,
                     .axis_strides = desc.middle.mul.axis_strides,
                     .batch_strides = desc.middle.mul.batch_strides},
          .epilogue = d::view_of(desc.epilogue),
      };
      const std::shared_ptr<const d::plan_core> core =
          d::make_pipeline_core(view, {.force_isa = t});
      const std::string text = d::describe(*core);
      ASSERT_TRUE(core->prog.flat) << text;
      ASSERT_EQ(core->edge_batched.row_block, 0U) << text;

      fftn::test::input_source src(0x2B10CULL + n);
      const cvec<double> x = src.complexes<double>(n * count);
      const std::vector<double> kr = src.reals<double>(2 * n * count);
      const cvec<double> kc = src.complexes<double>(2 * n * count);
      const fftn::exec_args args{.mid_factor = kind == real ? static_cast<const void*>(kr.data())
                                                            : static_cast<const void*>(kc.data())};
      cvec<double> one(n * count);
      {
        fftn::workspace ws(core->bytes);
        d::execute_all(*core, x.data(), one.data(), ws, args);
      }

      // The same core in the row-block form, with a workspace block of one line per row. Cores
      // never copy (their stages point into their tables), and make_pipeline_core makes a
      // mutable one, so the test changes it in place.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): the core is not const, see above
      d::plan_core& blocked = const_cast<d::plan_core&>(*core);
      const auto cb = static_cast<std::ptrdiff_t>(sizeof(C));
      blocked.edge_batched =
          d::make_edge_forms(blocked.prog, blocked.edge_records, true, rows, n * cb, cb);
      blocked.block_elements = n * rows;
      d::lay_out_workspace(blocked, blocked.scratch_line, sizeof(C), false);
      ASSERT_EQ(blocked.edge_batched.row_block, rows) << text;
      const std::size_t factor_bytes = kind == real ? sizeof(double) : sizeof(C);
      std::size_t middles = 0;
      for (const d::edge_flat_stage& st : blocked.edge_batched.flat) {
        if ((st.moving & d::moving_mid_factor) != 0) {
          ++middles;
          EXPECT_TRUE(st.block_call) << text;
          EXPECT_EQ(st.s.args.dst_qstride, static_cast<std::ptrdiff_t>(2 * n * factor_bytes))
              << text;
        }
      }
      EXPECT_EQ(middles, 1U) << text;
      fftn::workspace ws(blocked.bytes);
      cvec<double> other(n * count);
      d::execute_all(blocked, x.data(), other.data(), ws, args);
      EXPECT_TRUE(same_bits(one, other)) << fftn::isa_name(t) << ": the row blocks differ\n"
                                         << text;
      // Pieces that start and end inside a block of rows.
      cvec<double> pieces(n * count);
      for (const auto& [first, size] :
           {std::pair<std::size_t, std::size_t>{0, 1}, std::pair<std::size_t, std::size_t>{1, 4},
            std::pair<std::size_t, std::size_t>{5, 2}}) {
        d::execute_core(blocked, first, size, x.data(), pieces.data(), ws, args);
      }
      EXPECT_TRUE(same_bits(one, pieces)) << fftn::isa_name(t) << ": pieces differ";
    }
  }
}

} // namespace
