// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Plan edges (fftn/edges.hpp, DESIGN.md 12.1; task 5.1a): every operation of the vocabulary on
// c2c rows against "transform then separate pass", evaluated on the quadruple-precision
// reference within the derived tolerance plus a rounding per multiplication and addition; the
// fused and separate placements (describe()); factor strides of every sign along the line and
// the batch, broadcasts included; in place; strided rows with canaries in their gaps; the factor
// arrays and the input left as they were; serial, pieces of the batch, teams and an executor
// bitwise alike; scale-only edges bitwise equal to options::scale on every kind; and every
// refusal naming its field.

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

// A factor as a test gives it: kind and strides along the row and the batch.
struct factor_spec {
  fftn::factor_kind kind = fftn::factor_kind::none;
  std::ptrdiff_t axis = 1;
  std::ptrdiff_t batch = 0;
};

struct edge_case {
  const char* name = "";
  std::size_t n = 1;
  std::size_t count = 1;
  fftn::direction dir = fftn::direction::forward;
  double options_scale = 1.0;
  double in_scale = 1.0;
  double out_scale = 1.0;
  factor_spec in{};
  factor_spec out{};
  bool accumulate = false;
  bool in_place = false;
  std::ptrdiff_t in_stride = 1; // along the row; rows are n * stride + 1 apart (a gap)
  std::ptrdiff_t out_stride = 1;
  const char* expect = ""; // in describe()
};

// A factor array over rows x n coordinates with its strides: the storage and the offset of
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
};

fftn::factor factor_of(const factor_spec& f) {
  if (f.kind == fftn::factor_kind::none) {
    return {};
  }
  return {.kind = f.kind, .axis_strides = {f.axis}, .batch_strides = {f.batch}};
}

template <class T> struct edge_run {
  edge_case c;
  fftn::isa target;
  fftn::shape<std::complex<T>, std::complex<T>> shape;
  fftn::plan_edges<std::complex<T>, std::complex<T>> edges;
  fftn::options opts;
  std::ptrdiff_t in_dist = 0;
  std::ptrdiff_t out_dist = 0;
  cvec<T> x;    // the input buffer, gaps included
  cvec<T> init; // the output buffer before execution (accumulation, canaries)
  factor_array<T> fin;
  factor_array<T> fout;

  edge_run(const edge_case& ec, fftn::isa t, fftn::test::input_source& src)
      : c(ec), target(t), fin(ec.in, ec.n, ec.count, src), fout(ec.out, ec.n, ec.count, src) {
    in_dist = static_cast<std::ptrdiff_t>(c.n) * c.in_stride + 1;
    out_dist = c.in_place ? in_dist : static_cast<std::ptrdiff_t>(c.n) * c.out_stride + 1;
    shape.dir = c.dir;
    shape.axes = {{.n = c.n,
                   .in_stride = c.in_stride,
                   .out_stride = c.in_place ? c.in_stride : c.out_stride}};
    shape.batches = {{.count = c.count, .in_dist = in_dist, .out_dist = out_dist}};
    edges.in = {.scale = c.in_scale, .mul = factor_of(c.in), .accumulate = false};
    edges.out = {.scale = c.out_scale, .mul = factor_of(c.out), .accumulate = c.accumulate};
    opts = {.in_place = c.in_place, .scale = c.options_scale, .force_isa = t};
    x = src.complexes<T>(static_cast<std::size_t>(in_dist) * c.count);
    if (c.in_place) {
      init = x;
    } else {
      // Output values (accumulated onto) of about the transform's size, and canaries elsewhere.
      init = src.complexes<T>(static_cast<std::size_t>(out_dist) * c.count);
      for (auto& v : init) {
        v *= static_cast<T>(std::sqrt(static_cast<double>(c.n)));
      }
    }
  }

  [[nodiscard]] fftn::exec_args args() const {
    return {.in_factor = fin.address(), .out_factor = fout.address()};
  }

  [[nodiscard]] std::size_t in_at(std::size_t j, std::size_t r) const {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(r) * in_dist +
                                    static_cast<std::ptrdiff_t>(j) * c.in_stride);
  }
  [[nodiscard]] std::size_t out_at(std::size_t k, std::size_t r) const {
    const std::ptrdiff_t stride = c.in_place ? c.in_stride : c.out_stride;
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(r) * out_dist +
                                    static_cast<std::ptrdiff_t>(k) * stride);
  }

  // Checks y (the output buffer after execution) against the reference, and every element
  // outside the rows against init.
  void check(const cvec<T>& y, const std::string& describe) const {
    const std::string what = std::string(c.name) + ", " + fftn::isa_name(target) +
                             (sizeof(T) == 8 ? " double" : " float");
    SCOPED_TRACE(what);
    const double scale = c.options_scale * c.in_scale * c.out_scale;
    const int sign = static_cast<int>(c.dir);
    std::vector<bool> described(y.size(), false);
    fftn::test::outcome o;
    for (std::size_t r = 0; r < c.count; ++r) {
      std::vector<std::complex<double>> xin(c.n);
      for (std::size_t j = 0; j < c.n; ++j) {
        xin[j] = std::complex<double>(x[in_at(j, r)]) * fin.at(c.in, j, r, c.n, c.count);
      }
      const auto exact = fftn::test::reference::c2c(sign, xin);
      std::vector<fftn::test::exact_complex> want(c.n);
      cvec<T> got(c.n);
      for (std::size_t k = 0; k < c.n; ++k) {
        const std::complex<double> e(exact[k].re.hi, exact[k].im.hi);
        std::complex<double> v = e * fout.at(c.out, k, r, c.n, c.count) * scale;
        if (c.accumulate) {
          v += std::complex<double>(init[out_at(k, r)]);
        }
        want[k] = fftn::test::exact_from(v);
        got[k] = y[out_at(k, r)];
        described[out_at(k, r)] = true;
      }
      // A rounding for each factor's product, the scale, and the accumulation.
      const auto tol = fftn::test::derived_tolerance<T>(c.n, fftn::test::fftn_bound(c.n)) +
                       fftn::test::input_rounding<T>() * 4;
      o.within(fftn::test::compare<T>(std::span<const std::complex<T>>(got), want), tol,
               what + ", row " + std::to_string(r));
    }
    EXPECT_TRUE(o.passed()) << o.detail() << describe;
    if (!c.in_place) {
      std::size_t touched = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        if (!described[i] && !same_bits(cvec<T>{y[i]}, cvec<T>{init[i]})) {
          ++touched;
        }
      }
      EXPECT_EQ(touched, 0U) << "elements outside the rows written";
    }
  }
};

// The plan of a case, executed serially, in two pieces of the batch, by a team of three
// threads and on a pool of three: each from the same output buffer, all bitwise alike.
// reference: whether to check the serial result against the reference (rows too long for it
// are checked against serial execution only).
template <class T>
void run_case(const edge_case& ec, fftn::isa t, bool threads, bool reference = true) {
  fftn::test::input_source src(0xED6E5ULL + ec.n * 31 + ec.count);
  const edge_run<T> run(ec, t, src);
  const fftn::plan<std::complex<T>, std::complex<T>> p(run.shape, run.opts, run.edges);
  const std::string describe = p.describe();
  // The placement the case names is the chain's (fused, or separate passes through the
  // workspace block). Where the rules take another strategy for the row on this target (the
  // lanes for short float rows on x86, say), every factor is a separate pass around it.
  const bool other_strategy = describe.find("the row's strategy") != std::string::npos;
  const char* expect = other_strategy ? "the row's strategy" : ec.expect;
  EXPECT_NE(describe.find(expect), std::string::npos) << ec.name << ": no '" << expect << "' in\n"
                                                      << describe;
  if (other_strategy) {
    EXPECT_EQ(describe.find("strategy: chain (Stockham"), std::string::npos)
        << ec.name << ": separate passes around the chain\n"
        << describe;
  }
  const fftn::exec_args args = run.args();
  const cvec<T> x0 = run.x;
  const std::vector<T> fin_reals = run.fin.reals;
  const cvec<T> fin_complexes = run.fin.complexes;
  const cvec<T> fout_complexes = run.fout.complexes;

  const auto fresh = [&] { return ec.in_place ? run.x : run.init; };
  const auto in_of = [&](cvec<T>& y, const cvec<T>& xs) {
    return ec.in_place ? y.data() : xs.data();
  };

  fftn::workspace ws(p, 3);
  cvec<T> xs = run.x;
  cvec<T> serial = fresh();
  p.execute(in_of(serial, xs), serial.data(), ws, args);
  if (reference) {
    run.check(serial, describe);
  }
  if (!ec.in_place) {
    EXPECT_TRUE(same_bits(xs, x0)) << ec.name << ": input written";
  }
  EXPECT_TRUE(run.fin.reals == fin_reals && same_bits(run.fin.complexes, fin_complexes) &&
              same_bits(run.fout.complexes, fout_complexes))
      << ec.name << ": factor written";

  cvec<T> pieces = fresh();
  const std::size_t half = ec.count / 2;
  p.execute_transforms(0, half, in_of(pieces, xs), pieces.data(), ws, args);
  p.execute_transforms(half, ec.count - half, in_of(pieces, xs), pieces.data(), ws, args);
  EXPECT_TRUE(same_bits(pieces, serial)) << ec.name << ": pieces of the batch differ";
  if (!threads) {
    return;
  }

  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    cvec<T> teamed = fresh();
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        p.execute(in_of(teamed, xs), teamed.data(), fftn::team{.rank = r, .size = 3}, ws, done,
                  args);
      });
    }
    p.execute(in_of(teamed, xs), teamed.data(), fftn::team{.rank = 0, .size = 3}, ws, done, args);
    for (std::thread& m : members) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << ec.name << ": a team differs";
  }
  const fftn::thread_pool pool(3);
  cvec<T> pooled = fresh();
  p.execute(in_of(pooled, xs), pooled.data(), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << ec.name << ": the executor differs";
}

using fftn::factor_kind;
constexpr factor_kind real = factor_kind::real;
constexpr factor_kind cplx = factor_kind::complex;

// Every placement: fused factors, separate passes for what the passes lack, and the scale on
// whichever edge is free.
std::vector<edge_case> cases() {
  return {
      {.name = "input factor per row, fused",
       .n = 64,
       .count = 3,
       .in = {cplx, 1, 64},
       .expect = "the input times a complex factor in the first pass"},
      {.name = "output factor broadcast, scale on the input",
       .n = 48,
       .count = 3,
       .options_scale = 0.5,
       .out = {cplx, 1, 0},
       .expect = "the output times a complex factor in the last pass"},
      {.name = "both fused factors and a scale",
       .n = 60,
       .count = 2,
       .in_scale = 0.25,
       .in = {cplx, 1, 60},
       .out = {cplx, 1, -60},
       .expect = "the output times a complex factor in a separate pass"},
      {.name = "real input factor, separate",
       .n = 40,
       .count = 3,
       .in_scale = 3.0,
       .in = {real, 1, 40},
       .expect = "the input times a real factor in a separate pass"},
      {.name = "real output factor, separate",
       .n = 30,
       .count = 2,
       .out = {real, 1, 0},
       .expect = "the output times a real factor in a separate pass"},
      {.name = "complex output factor of stride 2",
       .n = 32,
       .count = 2,
       .out = {cplx, 2, 1},
       .expect = "in a separate pass from the workspace block"},
      {.name = "input factor broadcast along the row",
       .n = 16,
       .count = 4,
       .in = {cplx, 0, 1},
       .expect = "the input times a complex factor in a separate pass"},
      {.name = "accumulation alone",
       .n = 50,
       .count = 3,
       .out_scale = 0.25,
       .accumulate = true,
       .expect = "the output accumulated in a separate pass"},
      {.name = "accumulation, reversed factor",
       .n = 24,
       .count = 3,
       .in = {cplx, 1, 0},
       .out = {cplx, -1, -24},
       .accumulate = true,
       .expect = "accumulated"},
      {.name = "in place, separate input factor",
       .n = 36,
       .count = 2,
       .in = {real, 1, 36},
       .in_place = true,
       .expect = "the input times a real factor in a separate pass"},
      {.name = "in place, both fused",
       .n = 128,
       .count = 2,
       .dir = fftn::direction::backward,
       .in = {cplx, 1, 0},
       .out = {cplx, 1, 128},
       .in_place = true,
       .expect = "the output times a complex factor in the last pass"},
      {.name = "strided rows, fused factors",
       .n = 20,
       .count = 3,
       .in = {cplx, 1, 20},
       .out = {cplx, 1, 20},
       .in_stride = 2,
       .out_stride = 3,
       .expect = "in the first pass"},
      {.name = "strided rows, separate passes",
       .n = 12,
       .count = 2,
       .options_scale = 2.0,
       .in = {real, -1, 12},
       .out = {cplx, 3, 0},
       .accumulate = true,
       .in_stride = 3,
       .out_stride = 2,
       .expect = "into the workspace block"},
      {.name = "Bluestein row, fused factors",
       .n = 47,
       .count = 2,
       .in = {cplx, 1, 47},
       .out = {cplx, 1, 0},
       .expect = "bluestein"},
      {.name = "Bluestein row, separate passes",
       .n = 59,
       .count = 2,
       .in = {real, 1, 0},
       .accumulate = true,
       .expect = "bluestein"},
      {.name = "Prime row (Rader or Bluestein by target), fused and separate",
       .n = 53,
       .count = 2,
       .in = {cplx, 1, 53},
       .out = {real, 1, 0},
       // The rules take Rader for 53 on some targets and Bluestein on others (avx512):
       // the edges are what this case checks.
       .expect = "edges:"},
      {.name = "rows of one point",
       .n = 1,
       .count = 4,
       .in = {cplx, 7, 1},
       .out = {real, 5, -1},
       .expect = "edges:"},
      {.name = "a leaf with fused factors",
       .n = 8,
       .count = 5,
       .options_scale = 0.125,
       .in = {cplx, 1, 8},
       .expect = "the input times a complex factor in the first pass"},
  };
}

template <class T> class Edges : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(Edges, precisions);

TYPED_TEST(Edges, EveryOperationOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const edge_case& c : cases()) {
      run_case<TypeParam>(c, t, false);
    }
  }
}

// Teams and executors on rows short and long: a row of 2^17 points has stages of many chunks,
// which a team splits among its members (beyond the scheduler's split thresholds, X75).
TYPED_TEST(Edges, TeamsAndExecutorsAgreeWithSerial) {
  for (const edge_case& c : cases()) {
    run_case<TypeParam>(c, fftn::isa::best, true);
  }
  run_case<TypeParam>({.name = "long row, fused and separate",
                       .n = 4096,
                       .count = 1,
                       .in = {cplx, 1, 0},
                       .out = {real, 1, 0},
                       .accumulate = true,
                       .expect = "accumulated"},
                      fftn::isa::best, true);
  run_case<TypeParam>({.name = "long rows, fused",
                       .n = 4096,
                       .count = 2,
                       .out_scale = 0.5,
                       .in = {cplx, 1, 4096},
                       .expect = "in the first pass"},
                      fftn::isa::best, true);
  run_case<TypeParam>({.name = "a row split among a team",
                       .n = 131072,
                       .count = 1,
                       .in = {cplx, 1, 0},
                       .out = {cplx, 1, 0},
                       .accumulate = true,
                       .expect = "in the first pass"},
                      fftn::isa::best, true, false);
}

// Scales alone are one product in options::scale: the same plan, bit for bit, on every kind.
template <class In, class Out>
void expect_scale_only_same(const fftn::shape<In, Out>& s, std::size_t in_count,
                            std::size_t out_count) {
  const fftn::plan<In, Out> shortcut(s, {.scale = 0.375});
  const fftn::plan<In, Out> edged(s, {}, {.in = {.scale = 0.75}, .out = {.scale = 0.5}});
  EXPECT_EQ(shortcut.describe(), edged.describe());
  fftn::test::input_source src(0x5CA1EULL);
  std::vector<In> x(in_count);
  for (auto& v : x) {
    if constexpr (std::is_floating_point_v<In>) {
      v = static_cast<In>(src.uniform());
    } else {
      v = {static_cast<typename In::value_type>(src.uniform()),
           static_cast<typename In::value_type>(src.uniform())};
    }
  }
  std::vector<Out> a(out_count);
  std::vector<Out> b(out_count);
  fftn::workspace ws(std::max(shortcut.workspace_bytes(), edged.workspace_bytes()));
  shortcut.execute(x.data(), a.data(), ws);
  edged.execute(x.data(), b.data(), ws, {});
  EXPECT_TRUE(std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b))))
      << edged.describe();
}

TEST(Edges, ScalesAloneAreTheShortcut) {
  expect_scale_only_same(fftn::rows_c2c<double>(100, 3, fftn::direction::forward), 300, 300);
  expect_scale_only_same(fftn::rows_r2c<float>(64, 2), 128, 66);
  expect_scale_only_same(fftn::rows_c2r<double>(30, 2), 32, 60);
  fftn::shape<std::complex<double>, std::complex<double>> plane;
  plane.axes = {{.n = 12}, {.n = 10}};
  expect_scale_only_same(plane, 120, 120);
}

// Edges with factors take measured planning (task 5.1b-3): the candidates are timed with the
// edges, and the plan kept computes what the rules' plan computes.
TEST(Edges, MeasuredPlanningTakesTheEdges) {
  using C = std::complex<double>;
  const auto shape = fftn::rows_c2c<double>(256, 4, fftn::direction::forward);
  const fftn::plan_edges<C, C> edges{
      .in = {.mul = {.kind = cplx, .axis_strides = {1}, .batch_strides = {0}}},
      .out = {.mul = {.kind = real, .axis_strides = {1}, .batch_strides = {256}},
              .accumulate = true}};
  const fftn::plan<C, C> measured(shape, {.effort = fftn::planning::quick}, edges);
  const fftn::plan<C, C> rules(shape, {}, edges);
  EXPECT_NE(measured.describe().find("planning: measured"), std::string::npos)
      << measured.describe();
  fftn::test::input_source src(0x3EA5ULL);
  const cvec<double> x = src.complexes<double>(1024);
  const cvec<double> fin = src.complexes<double>(256);
  const std::vector<double> fout = src.reals<double>(1024);
  const cvec<double> init = src.complexes<double>(1024);
  const fftn::exec_args args{.in_factor = fin.data(), .out_factor = fout.data()};
  fftn::workspace ws(std::max(measured.workspace_bytes(), rules.workspace_bytes()));
  cvec<double> a = init;
  cvec<double> b = init;
  measured.execute(x.data(), a.data(), ws, args);
  rules.execute(x.data(), b.data(), ws, args);
  std::vector<fftn::test::exact_complex> want;
  for (const C& v : b) {
    want.push_back(fftn::test::exact_from(v));
  }
  fftn::test::outcome o;
  o.within(fftn::test::compare<double>(std::span<const C>(a), want),
           fftn::test::derived_tolerance<double>(256, fftn::test::fftn_bound(256)) * 2,
           "measured against the rules' plan");
  EXPECT_TRUE(o.passed()) << o.detail() << measured.describe();
}

template <class In, class Out>
void expect_refused(const fftn::shape<In, Out>& s, const fftn::options& opts,
                    const fftn::plan_edges<In, Out>& edges, const std::string& field) {
  try {
    const fftn::plan<In, Out> p(s, opts, edges);
    ADD_FAILURE() << "not refused: " << field;
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(field), std::string::npos) << e.what();
  }
}

TEST(Edges, RefusalsNameTheField) {
  using C = std::complex<double>;
  using F = std::complex<float>;
  const auto rows = fftn::rows_c2c<double>(16, 2, fftn::direction::forward);
  const auto frows = fftn::rows_c2c<float>(16, 2, fftn::direction::forward);
  const fftn::factor good{.kind = cplx, .axis_strides = {1}, .batch_strides = {16}};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  expect_refused<C, C>(rows, {}, {.in = {.scale = nan}}, "edges.in.scale");
  expect_refused<C, C>(rows, {}, {.out = {.scale = std::numeric_limits<double>::infinity()}},
                       "edges.out.scale");
  expect_refused<F, F>(frows, {}, {.out = {.scale = 1e300}}, "edges.out.scale");
  expect_refused<F, F>(frows, {}, {.in = {.scale = 1e-300}}, "edges.in.scale");
  expect_refused<F, F>(frows, {}, {.in = {.scale = 1e30}, .out = {.scale = 1e30}},
                       "edges.in.scale");
  expect_refused<C, C>(rows, {.scale = 0.5}, {.out = {.scale = 2.0}}, "edges.out.scale");
  // options::scale is judged even where a separate pass would apply it.
  expect_refused<C, C>(rows, {.scale = nan}, {.out = {.accumulate = true}}, "options.scale");
  expect_refused<C, C>(rows, {}, {.in = {.accumulate = true}}, "edges.in.accumulate");
  expect_refused<C, C>(rows, {}, {.in = {.mul = {.kind = cplx, .batch_strides = {0}}}},
                       "edges.in.mul.axis_strides");
  expect_refused<C, C>(
      rows, {}, {.out = {.mul = {.kind = real, .axis_strides = {1}, .batch_strides = {1, 2}}}},
      "edges.out.mul.batch_strides");
  // Accumulation in place would add to the input (the maintainer, 2026-11-25).
  expect_refused<C, C>(rows, {.in_place = true}, {.out = {.accumulate = true}},
                       "edges.out.accumulate");
  expect_refused<C, C>(rows, {},
                       {.out = {.mul = {.kind = static_cast<fftn::factor_kind>(7),
                                        .axis_strides = {1},
                                        .batch_strides = {0}}}},
                       "edges.out.mul.kind");
  expect_refused<C, C>(rows, {},
                       {.in = {.mul = {.kind = cplx,
                                       .axis_strides = {std::numeric_limits<std::ptrdiff_t>::max()},
                                       .batch_strides = {0}}}},
                       "edges.in.mul.axis_strides[0]");
  expect_refused<C, C>(
      rows, {},
      {.out = {.mul = {.kind = cplx,
                       .axis_strides = {1},
                       .batch_strides = {std::numeric_limits<std::ptrdiff_t>::min()}}}},
      "edges.out.mul.batch_strides[0]");
  // A complex factor would make a real side complex.
  const auto r2c = fftn::rows_r2c<double>(16, 2);
  expect_refused<double, C>(r2c, {}, {.in = {.mul = good}}, "edges.in.mul.kind");
  const auto c2r = fftn::rows_c2r<double>(16, 2);
  expect_refused<C, double>(c2r, {}, {.out = {.mul = good}}, "edges.out.mul.kind");
  // A plane's factor gives a stride per axis.
  fftn::shape<C, C> plane;
  plane.axes = {{.n = 8}, {.n = 8}};
  expect_refused<C, C>(plane, {}, {.out = {.mul = {.kind = real, .axis_strides = {1}}}},
                       "edges.out.mul.axis_strides");
  expect_refused<C, C>(plane, {.in_place = true}, {.out = {.accumulate = true}},
                       "edges.out.accumulate");
  // The shape's own errors come first.
  fftn::shape<C, C> empty;
  expect_refused<C, C>(empty, {}, {.in = {.mul = good}}, "shape.axes");
}

// No batch strides broadcast along every batch dimension, as strides of 0 do: the same plan
// and the same bits.
TEST(Edges, NoBatchStridesBroadcast) {
  using C = std::complex<double>;
  fftn::shape<C, C> s;
  s.axes = {{.n = 24}};
  s.batches = {{.count = 3}, {.count = 2}};
  const fftn::plan<C, C> zeros(
      s, {}, {.in = {.mul = {.kind = cplx, .axis_strides = {1}, .batch_strides = {0, 0}}}});
  const fftn::plan<C, C> none(s, {}, {.in = {.mul = {.kind = cplx, .axis_strides = {1}}}});
  EXPECT_EQ(zeros.describe(), none.describe());
  fftn::test::input_source src(0xB40ADULL);
  const cvec<double> x = src.complexes<double>(24 * 6);
  const cvec<double> f = src.complexes<double>(24);
  cvec<double> a(x.size());
  cvec<double> b(x.size());
  fftn::workspace ws(zeros);
  zeros.execute(x.data(), a.data(), ws, {.in_factor = f.data()});
  none.execute(x.data(), b.data(), ws, {.in_factor = f.data()});
  EXPECT_TRUE(same_bits(a, b));
}

// `execute(in, out, team, ws, {})` must not bind: before the exec_args overload of the team
// execute it meant `completion{}` without the execution's data, silently.
template <class In, class Out>
concept braces_bind_team =
    requires(const fftn::plan<In, Out>& p, const In* in, Out* out, fftn::workspace& ws) {
      p.execute(in, out, fftn::team{}, ws, {});
    };
template <class In, class Out>
concept args_bind_team =
    requires(const fftn::plan<In, Out>& p, const In* in, Out* out, fftn::workspace& ws) {
      p.execute(in, out, fftn::team{}, ws, fftn::exec_args{});
    };
static_assert(!braces_bind_team<std::complex<double>, std::complex<double>>);
static_assert(!braces_bind_team<float, std::complex<float>>);
static_assert(args_bind_team<std::complex<double>, std::complex<double>>);

// Executions that do not pass a factor array the plan needs end the program, in every build.
TEST(EdgesDeathTest, MissingFactorArraysAbort) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using C = std::complex<double>;
  const fftn::plan<C, C> p(fftn::rows_c2c<double>(32, 2, fftn::direction::forward), {},
                           {.out = {.mul = {.kind = cplx, .axis_strides = {1}}}});
  const fftn::plan<C, C> separate(fftn::rows_c2c<double>(32, 2, fftn::direction::forward), {},
                                  {.in = {.mul = {.kind = real, .axis_strides = {1}}}});
  cvec<double> x(64);
  cvec<double> y(64);
  fftn::workspace ws(p, 2);
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws), "without the factor arrays");
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws, {}), "without the factor arrays");
  EXPECT_DEATH(p.execute_transforms(1, 1, x.data(), y.data(), ws), "without the factor arrays");
  EXPECT_DEATH(separate.execute(x.data(), y.data(), ws), "without the factor arrays");
  EXPECT_DEATH(p.execute(x.data(), y.data(), fftn::team{}, ws), "without the factor arrays");
  // The wrong array: the input's given, the output's needed.
  const cvec<double> f(32);
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws, {.in_factor = f.data()}),
               "without the factor arrays");

  // Teams of two (one member present does everything) and an executor, on rows long enough
  // to be shared out, through the plain overloads and through the twins with no array.
  const fftn::plan<C, C> rows(fftn::rows_c2c<double>(16384, 4, fftn::direction::forward), {},
                              {.in = {.mul = {.kind = cplx, .axis_strides = {1}}}});
  cvec<double> lx(16384 * 4);
  cvec<double> ly(16384 * 4);
  fftn::workspace wide(rows, 4);
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    EXPECT_DEATH(rows.execute(lx.data(), ly.data(), fftn::team{.rank = 0, .size = 2}, wide, done),
                 "without the factor arrays");
    EXPECT_DEATH(rows.execute(lx.data(), ly.data(), fftn::team{.rank = 0, .size = 2}, wide, done,
                              fftn::exec_args{}),
                 "without the factor arrays");
  }
  EXPECT_DEATH(
      {
        const fftn::thread_pool pool(4);
        rows.execute(lx.data(), ly.data(), pool, wide);
      },
      "without the factor arrays");
  EXPECT_DEATH(
      {
        const fftn::thread_pool pool(4);
        rows.execute(lx.data(), ly.data(), pool, wide, {});
      },
      "without the factor arrays");
}

// Several batch dimensions with factor strides of both signs, against the reference: a fused
// input factor and a separate, accumulated output factor, serial and by a team of three.
TEST(Edges, SeveralBatchDimensions) {
  using C = std::complex<double>;
  const std::size_t n = 20;
  fftn::shape<C, C> s;
  s.axes = {{.n = n}};
  s.batches = {{.count = 3}, {.count = 3}};
  // Input factor: rows of n along the axis, batch strides -3n and n (zero at the middle).
  const std::ptrdiff_t ni = static_cast<std::ptrdiff_t>(n);
  const std::vector<std::ptrdiff_t> in_batch{-3 * ni, ni};
  // Output factor (real): stride -1 along the axis, batch strides 2 and -7.
  const std::vector<std::ptrdiff_t> out_batch{2, -7};
  const fftn::plan<C, C> p(
      s, {},
      {.in = {.mul = {.kind = cplx, .axis_strides = {1}, .batch_strides = in_batch}},
       .out = {.scale = 0.5,
               .mul = {.kind = real, .axis_strides = {-1}, .batch_strides = out_batch},
               .accumulate = true}});
  fftn::test::input_source src(0x3B47CULL);
  const cvec<double> x = src.complexes<double>(9 * n);
  const cvec<double> init = src.complexes<double>(9 * n);
  // Spans: input offsets from -6n to 2n + n - 1, output from -(n - 1) - 14 to 4.
  const cvec<double> fin = src.complexes<double>(9 * n);
  const std::ptrdiff_t fin_zero = 6 * ni;
  const std::vector<double> fout = src.reals<double>(n + 20);
  const std::ptrdiff_t fout_zero = ni - 1 + 14;
  const fftn::exec_args args{.in_factor = fin.data() + fin_zero,
                             .out_factor = fout.data() + fout_zero};
  fftn::workspace ws(p, 3);
  cvec<double> y = init;
  p.execute(x.data(), y.data(), ws, args);
  fftn::test::outcome o;
  for (std::size_t b0 = 0; b0 < 3; ++b0) {
    for (std::size_t b1 = 0; b1 < 3; ++b1) {
      const std::size_t t = b0 * 3 + b1;
      const auto i0 = static_cast<std::ptrdiff_t>(b0);
      const auto i1 = static_cast<std::ptrdiff_t>(b1);
      std::vector<C> xin(n);
      for (std::size_t j = 0; j < n; ++j) {
        const std::ptrdiff_t at =
            fin_zero + static_cast<std::ptrdiff_t>(j) + in_batch[0] * i0 + in_batch[1] * i1;
        xin[j] = x[t * n + j] * fin[static_cast<std::size_t>(at)];
      }
      const auto exact = fftn::test::reference::c2c(-1, xin);
      std::vector<fftn::test::exact_complex> want(n);
      cvec<double> got(n);
      for (std::size_t k = 0; k < n; ++k) {
        const std::ptrdiff_t at =
            fout_zero - static_cast<std::ptrdiff_t>(k) + out_batch[0] * i0 + out_batch[1] * i1;
        const C e(exact[k].re.hi, exact[k].im.hi);
        want[k] =
            fftn::test::exact_from(init[t * n + k] + e * fout[static_cast<std::size_t>(at)] * 0.5);
        got[k] = y[t * n + k];
      }
      o.within(fftn::test::compare<double>(std::span<const C>(got), want),
               fftn::test::derived_tolerance<double>(n, fftn::test::fftn_bound(n)) +
                   fftn::test::input_rounding<double>() * 4,
               "transform " + std::to_string(t));
    }
  }
  EXPECT_TRUE(o.passed()) << o.detail() << p.describe();
  cvec<double> teamed = init;
  std::vector<std::thread> members;
  for (int r = 1; r < 3; ++r) {
    members.emplace_back(
        [&, r] { p.execute(x.data(), teamed.data(), fftn::team{.rank = r, .size = 3}, ws, args); });
  }
  p.execute(x.data(), teamed.data(), fftn::team{.rank = 0, .size = 3}, ws, args);
  for (std::thread& m : members) {
    m.join();
  }
  EXPECT_TRUE(same_bits(teamed, y));
}

// Factor strides on dimensions of one point are ignored, as the shape's are.
TEST(Edges, StridesOfSinglePointsAreIgnored) {
  const auto one = fftn::rows_c2c<double>(1, 1, fftn::direction::forward);
  const std::ptrdiff_t huge = std::numeric_limits<std::ptrdiff_t>::max();
  const fftn::plan<std::complex<double>, std::complex<double>> p(
      one, {}, {.in = {.mul = {.kind = cplx, .axis_strides = {huge}, .batch_strides = {huge}}}});
  const std::complex<double> f{0.0, 2.0};
  const std::complex<double> x{3.0, 1.0};
  std::complex<double> y;
  fftn::workspace ws(p);
  p.execute(&x, &y, ws, {.in_factor = &f});
  EXPECT_EQ(y, x * f);
}

} // namespace
