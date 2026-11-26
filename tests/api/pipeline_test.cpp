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
// leaf-sized lines; and every refusal naming its field. Expectations hold on every target: none
// assumes a target's radices or strategy.

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
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

template <class T> using cvec = std::vector<std::complex<T>>;

template <class T> bool same_bits(const cvec<T>& a, const cvec<T>& b) {
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

template <class T> struct pipe_run {
  pipe_case c;
  fftn::isa target;
  desc_of<T> desc;
  fftn::options opts;
  std::ptrdiff_t in_dist = 0;
  std::ptrdiff_t out_dist = 0;
  // Where element 0 of line 0 is in each buffer: past the line's other elements when its stride
  // is negative.
  std::size_t in_base = 0;
  std::size_t out_base = 0;
  double mid_scale = 1.0;
  cvec<T> x;    // the input buffer, gaps included
  cvec<T> init; // the output buffer before execution (accumulation, canaries)
  factor_array<T> fpro;
  factor_array<T> fmid;
  factor_array<T> fepi;

  pipe_run(const pipe_case& pc, fftn::isa t, fftn::test::input_source& src)
      : c(pc), target(t), fpro(pc.pro, pc.n, pc.count, src), fmid(pc.mid, pc.n, pc.count, src),
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
    desc.line = {
        .n = c.n, .in_stride = c.in_stride, .out_stride = c.in_place ? c.in_stride : c.out_stride};
    desc.batches = {{.count = c.count, .in_dist = in_dist, .out_dist = out_dist}};
    desc.prologue = {.scale = c.pro_scale, .mul = factor_of(c.pro), .accumulate = false};
    desc.middle.scale = mid_scale;
    desc.middle.mul = factor_of(c.mid);
    desc.epilogue = {.scale = c.epi_scale, .mul = factor_of(c.epi), .accumulate = c.accumulate};
    opts = {.in_place = c.in_place, .scale = c.options_scale, .force_isa = t};
    x = src.complexes<T>(static_cast<std::size_t>(in_dist) * c.count);
    init = c.in_place ? x : src.complexes<T>(static_cast<std::size_t>(out_dist) * c.count);
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
                             fftn::isa_name(target) + (sizeof(T) == 8 ? " double" : " float");
    SCOPED_TRACE(what);
    const double scale = c.options_scale * c.pro_scale * mid_scale * c.epi_scale;
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

// The pipeline of a case, executed serially (checked against the reference where `reference`),
// in two pieces of the batch and, with `threads`, by teams of three (completion::all and own)
// and on a pool of three: each from the same output buffer, all bitwise alike.
template <class T>
void run_case(const pipe_case& pc, fftn::isa t, bool threads, bool reference = true) {
  fftn::test::input_source src(0x919EULL + pc.n * 31 + pc.count);
  const pipe_run<T> run(pc, t, src);
  const pipe_of<T> p(run.desc, run.opts);
  const std::string describe = p.describe();
  EXPECT_NE(describe.find(pc.expect), std::string::npos)
      << pc.name << ": no '" << pc.expect << "' in\n"
      << describe;
  EXPECT_EQ(p.instruction_set(), t == fftn::isa::best ? p.instruction_set() : t);
  const fftn::exec_args args = run.args();
  const cvec<T> x0 = run.x;
  const std::vector<T> pro_reals = run.fpro.reals;
  const cvec<T> mid_complexes = run.fmid.complexes;
  const std::vector<T> mid_reals = run.fmid.reals;
  const cvec<T> epi_complexes = run.fepi.complexes;

  const auto fresh = [&] { return pc.in_place ? run.x : run.init; };
  // Element 0 of line 0 in the input and the output buffers.
  const auto in_of = [&](cvec<T>& y, const cvec<T>& xs) {
    return (pc.in_place ? y.data() : xs.data()) + run.in_base;
  };
  const auto out_of = [&](cvec<T>& y) { return y.data() + run.out_base; };

  fftn::workspace ws(p, 3);
  cvec<T> xs = run.x;
  cvec<T> serial = fresh();
  p.execute(in_of(serial, xs), out_of(serial), ws, args);
  if (reference) {
    run.check(serial, describe);
  }
  if (!pc.in_place) {
    EXPECT_TRUE(same_bits(xs, x0)) << pc.name << ": input written";
  }
  EXPECT_TRUE(run.fpro.reals == pro_reals && run.fmid.reals == mid_reals &&
              same_bits(run.fmid.complexes, mid_complexes) &&
              same_bits(run.fepi.complexes, epi_complexes))
      << pc.name << ": factor written";

  cvec<T> pieces = fresh();
  const std::size_t half = pc.count / 2;
  p.execute_transforms(0, half, in_of(pieces, xs), out_of(pieces), ws, args);
  p.execute_transforms(half, pc.count - half, in_of(pieces, xs), out_of(pieces), ws, args);
  EXPECT_TRUE(same_bits(pieces, serial)) << pc.name << ": pieces of the batch differ";
  if (!threads) {
    return;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    cvec<T> teamed = fresh();
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
  cvec<T> teamed = fresh();
  std::thread member([&] {
    p.execute(in_of(teamed, xs), out_of(teamed), fftn::team{.rank = 1, .size = 2}, ws, args);
  });
  p.execute(in_of(teamed, xs), out_of(teamed), fftn::team{.rank = 0, .size = 2}, ws, args);
  member.join();
  EXPECT_TRUE(same_bits(teamed, serial)) << pc.name << ": a team of two differs";
  const fftn::thread_pool pool(3);
  cvec<T> pooled = fresh();
  p.execute(in_of(pooled, xs), out_of(pooled), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << pc.name << ": the executor differs";
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
  fftn::pipeline_desc<double, C, double> real_conv;
  real_conv.line = {.n = 16};
  expect_refused(real_conv, {}, "desc");
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

} // namespace
