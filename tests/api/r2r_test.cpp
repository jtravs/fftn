// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Real-to-real plans (fftn/shape.hpp r2r_kind, DESIGN.md 5.7; task 6.3), so far the DHT: against
// a quadruple-precision direct evaluation (the separable product of the 1-D transforms for several
// axes) within the derived tolerance of the underlying transform plus a rounding per stage
// operation, on every instruction set in both precisions, every n up to 512 and sampled longer
// smooth, prime and Bluestein lengths; the DHT twice gives n times the input; planes and volumes,
// strides of every sign, in place, canaries in the gaps; the edges (scales, real factors,
// accumulation); serial, pieces of the batch, teams and an executor bitwise alike; refusals naming
// the field. Expectations hold on every target: none assumes a target's radices or strategy.

#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using fftn::factor_kind;
using fftn::r2r_kind;

template <class T> bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

// The tolerance of a DHT of `points` points over `phases` axes: the underlying complex
// transforms' derived tolerance (their errors add in quadrature across axes, as for c2c planes),
// times `spread` (an output factor's), plus a rounding for each operation of the read and store
// stages (the pairing's two sums and halvings, the twiddle's product, the combination) per phase,
// and `extra` more (the edges' multiplications, the accumulation, the reference's rounding).
template <class T>
fftn::test::tolerance dht_tolerance(std::size_t points, std::size_t phases, double spread = 1.0,
                                    double extra = 1.0) {
  const fftn::test::tolerance one =
      fftn::test::derived_tolerance<T>(points, fftn::test::fftn_bound(points));
  return one * spread +
         fftn::test::input_rounding<T>() * (6.0 * static_cast<double>(phases) + extra);
}

// a - b with the error of the first difference kept: exact_real is hi + lo.
fftn::test::exact_real difference(fftn::test::exact_real a, fftn::test::exact_real b, double sign) {
  const double bh = sign * b.hi;
  const double s = a.hi - bh;
  const double v = s - a.hi;
  const double err = (a.hi - (s - v)) + (-bh - v);
  const double lo = err + a.lo - sign * b.lo;
  const double hi = s + lo;
  return {.hi = hi, .lo = lo - (hi - s)};
}

// Outputs `ks` of the DHT of x, from the reference r2c's bins: H_k = Re X_k - Im X_k for
// k <= n/2, Re X_(n-k) + Im X_(n-k) above (for long lines, where the direct sum is too slow).
std::vector<fftn::test::exact_real> dht_outputs(std::span<const double> x,
                                                std::span<const std::size_t> ks) {
  const std::size_t n = x.size();
  std::vector<std::size_t> bins;
  for (const std::size_t k : ks) {
    bins.push_back(k <= n / 2 ? k : n - k);
  }
  const auto X = fftn::test::reference::r2c(x, bins);
  std::vector<fftn::test::exact_real> out;
  for (std::size_t i = 0; i < ks.size(); ++i) {
    out.push_back(difference(X[i].re, X[i].im, ks[i] <= n / 2 ? 1.0 : -1.0));
  }
  return out;
}

// --- Rows of every length, tight, on every instruction set -------------------------------------

// Row 0 of an input of n reals (steps of 2^-23, so exact in float and double alike) and its
// reference, computed once per n for every target and precision. Row 1 of the tests is row 0
// circularly reversed (x_(-j mod n)), whose DHT is H_(-k mod n): no second reference.
struct row_reference {
  std::vector<double> x;
  std::vector<fftn::test::exact_real> h;
};

const row_reference& reference_of(std::size_t n) {
  static std::map<std::size_t, row_reference> cache;
  if (const auto it = cache.find(n); it != cache.end()) {
    return it->second;
  }
  fftn::test::input_source src(0xD47ULL + n);
  row_reference r{.x = src.reals<double>(n), .h = {}};
  const std::array<std::size_t, 1> extents{n};
  r.h = fftn::test::reference::dht(extents, r.x);
  return cache.emplace(n, std::move(r)).first->second;
}

template <class T> void check_row(std::size_t n, fftn::isa t, fftn::test::outcome& o) {
  const row_reference& ref = reference_of(n);
  const std::string what = "n = " + std::to_string(n) + ", " + fftn::isa_name(t) +
                           (sizeof(T) == 8 ? " double" : " float");
  const fftn::r2r_plan<T> p(fftn::rows_r2r<T>(n, 2, r2r_kind::dht), {.force_isa = t});
  std::vector<T> x(2 * n);
  for (std::size_t j = 0; j < n; ++j) {
    x[j] = static_cast<T>(ref.x[j]);
    x[n + j] = static_cast<T>(ref.x[(n - j) % n]);
  }
  std::vector<T> y(2 * n);
  fftn::workspace ws(p);
  p.execute(x.data(), y.data(), ws);
  std::vector<fftn::test::exact_real> want(n);
  for (std::size_t k = 0; k < n; ++k) {
    want[k] = ref.h[(n - k) % n];
  }
  const fftn::test::tolerance tol = dht_tolerance<T>(n, 1);
  o.within(fftn::test::compare<T>(std::span<const T>(y.data(), n), ref.h), tol, what + ", row 0");
  o.within(fftn::test::compare<T>(std::span<const T>(y.data() + n, n), want), tol,
           what + ", row 1");
  // The DHT twice: n times the input, within two transforms.
  std::vector<T> back(2 * n);
  p.execute(y.data(), back.data(), ws);
  std::vector<fftn::test::exact_real> scaled(2 * n);
  for (std::size_t j = 0; j < 2 * n; ++j) {
    scaled[j] = fftn::test::exact_from(static_cast<double>(x[j]) * static_cast<double>(n));
  }
  o.within(fftn::test::compare<T>(std::span<const T>(back), scaled), tol * 2.0, what + ", twice");
}

template <class T> class R2r : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(R2r, precisions);

TYPED_TEST(R2r, DhtOfEveryLengthUpTo512OnEveryInstructionSet) {
  fftn::test::outcome o;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (std::size_t n = 1; n <= 512; ++n) {
      check_row<TypeParam>(n, t, o);
    }
  }
  EXPECT_TRUE(o.passed()) << o.detail();
}

// Longer lines: smooth, powers of two, a prime with a Rader or Bluestein stage, a long one with a
// large prime factor; checked at sampled outputs, and whole by the DHT twice.
TYPED_TEST(R2r, LongerLinesOnEveryInstructionSet) {
  using T = TypeParam;
  fftn::test::outcome o;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : {1000U, 1024U, 4099U, 6000U, 10007U, 65536U, 100002U}) {
      const std::string what = "n = " + std::to_string(n) + ", " + fftn::isa_name(t) +
                               (sizeof(T) == 8 ? " double" : " float");
      fftn::test::input_source src(0x10E6ULL + n);
      const std::vector<double> xd = src.reals<double>(n);
      const std::vector<T> x(xd.begin(), xd.end());
      const fftn::r2r_plan<T> p(fftn::rows_r2r<T>(n, 1, r2r_kind::dht), {.force_isa = t});
      fftn::workspace ws(p);
      std::vector<T> y(n);
      p.execute(x.data(), y.data(), ws);
      std::vector<std::size_t> ks{0, 1, 2, n / 2, n - 1, n - 2};
      for (std::size_t k = 3; k < n; k += n / 23) {
        ks.push_back(k);
      }
      std::vector<T> got;
      for (const std::size_t k : ks) {
        got.push_back(y[k]);
      }
      const fftn::test::tolerance tol = dht_tolerance<T>(n, 1);
      o.within(fftn::test::compare<T>(std::span<const T>(got), dht_outputs(xd, ks)), tol,
               what + ", sampled outputs");
      std::vector<T> back(n);
      p.execute(y.data(), back.data(), ws);
      std::vector<fftn::test::exact_real> scaled(n);
      for (std::size_t j = 0; j < n; ++j) {
        scaled[j] = fftn::test::exact_from(xd[j] * static_cast<double>(n));
      }
      o.within(fftn::test::compare<T>(std::span<const T>(back), scaled), tol * 2.0,
               what + ", twice");
    }
  }
  EXPECT_TRUE(o.passed()) << o.detail();
}

// --- General layouts: planes, volumes, strides of every sign, in place, edges, threads ---------

// A side's layout as a test gives it: per axis a sign and, beyond the tight extent, a gap of
// `gap` elements after every line of the axis (so canaries sit between the lines).
struct layout_spec {
  std::vector<int> sign{}; // per axis, empty: all +1
  std::ptrdiff_t gap = 0;
};

// A factor as a test gives it: kind, reversed along the last axis, broadcast along the batch.
struct factor_spec {
  factor_kind kind = factor_kind::none;
  bool reversed = false;
  bool broadcast = false;
};

struct r2r_case {
  const char* name = "";
  std::vector<std::size_t> n; // the extents, slowest first
  std::size_t count = 1;
  layout_spec in{};
  layout_spec out{};
  bool in_place = false;
  double options_scale = 1.0;
  double in_scale = 1.0;
  double out_scale = 1.0;
  factor_spec fin{};
  factor_spec fout{};
  bool accumulate = false;
  const char* expect = ""; // in describe()
};

// One side's strides, batch distance, base offset (element 0 where strides are negative) and
// buffer length.
struct side_layout {
  std::vector<std::ptrdiff_t> stride;
  std::ptrdiff_t dist = 0;
  std::ptrdiff_t base = 0;
  std::size_t size = 0;
};

side_layout lay_out(const std::vector<std::size_t>& n, std::size_t count, const layout_spec& s) {
  side_layout l;
  const std::size_t r = n.size();
  l.stride.assign(r, 0);
  std::ptrdiff_t span = 1;
  for (std::size_t i = r; i > 0; --i) {
    const std::size_t d = i - 1;
    const int sign = s.sign.empty() ? 1 : s.sign[d];
    l.stride[d] = sign * span;
    span = span * static_cast<std::ptrdiff_t>(n[d]) + s.gap;
  }
  l.dist = span + s.gap;
  std::ptrdiff_t lo = 0;
  std::ptrdiff_t hi = 0;
  for (std::size_t d = 0; d < r; ++d) {
    const std::ptrdiff_t reach = l.stride[d] * static_cast<std::ptrdiff_t>(n[d] - 1);
    (reach < 0 ? lo : hi) += reach;
  }
  hi += l.dist * static_cast<std::ptrdiff_t>(count - 1);
  l.base = -lo;
  l.size = static_cast<std::size_t>(hi - lo + 1 + s.gap);
  return l;
}

// The position in a side's buffer of element (j) of transform r.
std::size_t position(const side_layout& l, const std::vector<std::size_t>& j, std::size_t r) {
  std::ptrdiff_t at = l.base + l.dist * static_cast<std::ptrdiff_t>(r);
  for (std::size_t d = 0; d < j.size(); ++d) {
    at += l.stride[d] * static_cast<std::ptrdiff_t>(j[d]);
  }
  return static_cast<std::size_t>(at);
}

// Calls f(j, flat) for every multi-index j of extents n in C order.
void for_each_index(const std::vector<std::size_t>& n,
                    const std::function<void(const std::vector<std::size_t>&, std::size_t)>& f) {
  const std::size_t total =
      std::accumulate(n.begin(), n.end(), std::size_t{1}, std::multiplies<>());
  std::vector<std::size_t> j(n.size(), 0);
  for (std::size_t flat = 0; flat < total; ++flat) {
    f(j, flat);
    for (std::size_t i = n.size(); i > 0; --i) {
      if (++j[i - 1] < n[i - 1]) {
        break;
      }
      j[i - 1] = 0;
    }
  }
}

// A factor array over count transforms of the extents: C-ordered (reversed along the last axis)
// per transform, transforms one after another or all sharing one.
template <class T> struct factor_array {
  std::vector<T> v;
  std::vector<std::ptrdiff_t> axis;
  std::ptrdiff_t batch = 0;
  std::ptrdiff_t zero = 0;

  factor_array(const factor_spec& f, const std::vector<std::size_t>& n, std::size_t count,
               fftn::test::input_source& src) {
    if (f.kind == factor_kind::none) {
      return;
    }
    const std::size_t total =
        std::accumulate(n.begin(), n.end(), std::size_t{1}, std::multiplies<>());
    axis.assign(n.size(), 0);
    std::ptrdiff_t span = 1;
    for (std::size_t i = n.size(); i > 0; --i) {
      axis[i - 1] = span;
      span *= static_cast<std::ptrdiff_t>(n[i - 1]);
    }
    if (f.reversed) {
      axis.back() = -1;
      zero = static_cast<std::ptrdiff_t>(n.back() - 1);
    }
    batch = f.broadcast ? 0 : static_cast<std::ptrdiff_t>(total);
    v = src.reals<T>(total * (f.broadcast ? 1 : count));
  }

  [[nodiscard]] fftn::factor spec(factor_kind kind, bool broadcast) const {
    if (kind == factor_kind::none) {
      return {};
    }
    return {.kind = kind,
            .axis_strides = axis,
            .batch_strides = broadcast ? std::vector<std::ptrdiff_t>{} : std::vector{batch}};
  }
  [[nodiscard]] const void* address() const { return v.empty() ? nullptr : v.data() + zero; }
  [[nodiscard]] double at(const std::vector<std::size_t>& j, std::size_t r) const {
    if (v.empty()) {
      return 1.0;
    }
    std::ptrdiff_t i = zero + batch * static_cast<std::ptrdiff_t>(r);
    for (std::size_t d = 0; d < j.size(); ++d) {
      i += axis[d] * static_cast<std::ptrdiff_t>(j[d]);
    }
    return static_cast<double>(v[static_cast<std::size_t>(i)]);
  }
  // The largest magnitude over the root mean square (1 without a factor).
  [[nodiscard]] double spread() const {
    if (v.empty()) {
      return 1.0;
    }
    double peak = 0.0;
    double sum = 0.0;
    for (const T x : v) {
      peak = std::max(peak, std::abs(static_cast<double>(x)));
      sum += static_cast<double>(x) * static_cast<double>(x);
    }
    return sum == 0.0 ? 1.0 : peak / std::sqrt(sum / static_cast<double>(v.size()));
  }
};

template <class T> struct r2r_run {
  r2r_case c;
  fftn::shape<T, T> shape;
  fftn::options opts;
  fftn::plan_edges<T, T> edges;
  side_layout lin;
  side_layout lout;
  std::vector<T> x;    // the input buffer, gaps included
  std::vector<T> init; // the output buffer before execution
  factor_array<T> fin;
  factor_array<T> fout;

  r2r_run(const r2r_case& rc, fftn::isa t, fftn::test::input_source& src)
      : c(rc), lin(lay_out(rc.n, rc.count, rc.in)),
        lout(rc.in_place ? lin : lay_out(rc.n, rc.count, rc.out)), fin(rc.fin, rc.n, rc.count, src),
        fout(rc.fout, rc.n, rc.count, src) {
    for (std::size_t d = 0; d < c.n.size(); ++d) {
      shape.axes.push_back({.n = c.n[d], .in_stride = lin.stride[d], .out_stride = lout.stride[d]});
    }
    shape.batches = {{.count = c.count, .in_dist = lin.dist, .out_dist = lout.dist}};
    shape.kinds.assign(c.n.size(), r2r_kind::dht);
    opts = {.in_place = c.in_place, .scale = c.options_scale, .force_isa = t};
    edges.in = {.scale = c.in_scale, .mul = fin.spec(c.fin.kind, c.fin.broadcast)};
    edges.out = {.scale = c.out_scale,
                 .mul = fout.spec(c.fout.kind, c.fout.broadcast),
                 .accumulate = c.accumulate};
    x = src.reals<T>(lin.size);
    init = c.in_place ? x : src.reals<T>(lout.size);
  }

  [[nodiscard]] fftn::exec_args args() const {
    return {.in_factor = fin.address(), .out_factor = fout.address()};
  }

  // Checks y against the reference, and every element outside the described output against
  // init.
  void check(const std::vector<T>& y, const std::string& describe, const std::string& what) const {
    SCOPED_TRACE(what);
    const std::size_t points =
        std::accumulate(c.n.begin(), c.n.end(), std::size_t{1}, std::multiplies<>());
    std::size_t phases = 0;
    for (const std::size_t n : c.n) {
      phases += n > 1 ? 1 : 0;
    }
    const double scale = c.options_scale * c.in_scale * c.out_scale;
    const fftn::test::tolerance tol =
        dht_tolerance<T>(points, std::max<std::size_t>(phases, 1), fout.spread(), 5.0);
    std::vector<bool> described(y.size(), false);
    fftn::test::outcome o;
    for (std::size_t r = 0; r < c.count; ++r) {
      std::vector<double> xin(points);
      for_each_index(c.n, [&](const std::vector<std::size_t>& j, std::size_t flat) {
        xin[flat] = static_cast<double>(x[position(lin, j, r)]) * fin.at(j, r);
      });
      const auto h = fftn::test::reference::dht(c.n, xin);
      std::vector<fftn::test::exact_real> want(points);
      std::vector<T> got(points);
      for_each_index(c.n, [&](const std::vector<std::size_t>& j, std::size_t flat) {
        double v = h[flat].hi * fout.at(j, r) * scale;
        const std::size_t at = position(lout, j, r);
        if (c.accumulate) {
          v += static_cast<double>(init[at]);
        }
        want[flat] = fftn::test::exact_from(v);
        got[flat] = y[at];
        described[at] = true;
      });
      o.within(fftn::test::compare<T>(std::span<const T>(got), want), tol,
               what + ", transform " + std::to_string(r));
    }
    EXPECT_TRUE(o.passed()) << o.detail() << describe;
    if (!c.in_place) {
      std::size_t touched = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        if (!described[i] && !same_bits(std::vector<T>{y[i]}, std::vector<T>{init[i]})) {
          ++touched;
        }
      }
      EXPECT_EQ(touched, 0U) << "elements outside the transforms written";
    }
  }
};

bool has_edges(const r2r_case& c) {
  return c.fin.kind != factor_kind::none || c.fout.kind != factor_kind::none || c.accumulate ||
         c.in_scale != 1.0 || c.out_scale != 1.0;
}

// The plan of a case, executed serially (checked against the reference where `reference`; long
// lines are checked by LongerLines), in two pieces of the
// batch and, with `threads`, by teams of three (completion::all and own), a team of two and a pool
// of three: each from the same output buffer, all bitwise alike.
template <class T>
void run_case(const r2r_case& rc, fftn::isa t, bool threads, bool reference = true) {
  fftn::test::input_source src(0x2B2ULL + rc.count * 131 + rc.n.size() * 7 + rc.n.back());
  const r2r_run<T> run(rc, t, src);
  const fftn::r2r_plan<T> p = has_edges(rc) ? fftn::r2r_plan<T>(run.shape, run.opts, run.edges)
                                            : fftn::r2r_plan<T>(run.shape, run.opts);
  const std::string describe = p.describe();
  const std::string what =
      std::string(rc.name) + ", " + fftn::isa_name(t) + (sizeof(T) == 8 ? " double" : " float");
  EXPECT_NE(describe.find("fftn plan: r2r dht"), std::string::npos) << describe;
  EXPECT_NE(describe.find(rc.expect), std::string::npos)
      << what << ": no '" << rc.expect << "' in\n"
      << describe;
  EXPECT_EQ(p.instruction_set(), t == fftn::isa::best ? p.instruction_set() : t);
  const fftn::exec_args args = run.args();
  const std::vector<T> x0 = run.x;
  const std::vector<T> fin0 = run.fin.v;
  const std::vector<T> fout0 = run.fout.v;
  const auto fresh = [&] { return rc.in_place ? run.x : run.init; };
  const auto in_of = [&](std::vector<T>& y, const std::vector<T>& xs) {
    return (rc.in_place ? y.data() : xs.data()) + run.lin.base;
  };
  const auto out_of = [&](std::vector<T>& y) { return y.data() + run.lout.base; };

  fftn::workspace ws(p, 3);
  std::vector<T> xs = run.x;
  std::vector<T> serial = fresh();
  p.execute(in_of(serial, xs), out_of(serial), ws, args);
  if (reference) {
    run.check(serial, describe, what);
  }
  if (!rc.in_place) {
    EXPECT_TRUE(same_bits(xs, x0)) << what << ": input written";
  }
  EXPECT_TRUE(run.fin.v == fin0 && run.fout.v == fout0) << what << ": factor written";

  std::vector<T> pieces = fresh();
  const std::size_t half = rc.count / 2;
  p.execute_transforms(0, half, in_of(pieces, xs), out_of(pieces), ws, args);
  p.execute_transforms(half, rc.count - half, in_of(pieces, xs), out_of(pieces), ws, args);
  EXPECT_TRUE(same_bits(pieces, serial)) << what << ": pieces of the batch differ";
  if (!threads) {
    return;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    std::vector<T> teamed = fresh();
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
    EXPECT_TRUE(same_bits(teamed, serial)) << what << ": a team differs";
  }
  std::vector<T> teamed = fresh();
  std::thread member([&] {
    p.execute(in_of(teamed, xs), out_of(teamed), fftn::team{.rank = 1, .size = 2}, ws, args);
  });
  p.execute(in_of(teamed, xs), out_of(teamed), fftn::team{.rank = 0, .size = 2}, ws, args);
  member.join();
  EXPECT_TRUE(same_bits(teamed, serial)) << what << ": a team of two differs";
  const fftn::thread_pool pool(3);
  std::vector<T> pooled = fresh();
  p.execute(in_of(pooled, xs), out_of(pooled), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << what << ": the executor differs";
  // Without exec_args, a plan without factors gives the same bits.
  if (rc.fin.kind == factor_kind::none && rc.fout.kind == factor_kind::none) {
    std::vector<T> plain = fresh();
    p.execute(in_of(plain, xs), out_of(plain), ws);
    EXPECT_TRUE(same_bits(plain, serial)) << what << ": execute without exec_args differs";
  }
}

constexpr factor_kind real = factor_kind::real;

std::vector<r2r_case> cases() {
  return {
      {.name = "rows with gaps",
       .n = {24},
       .count = 3,
       .in = {.gap = 2},
       .out = {.gap = 3},
       .expect = "r2r store: the r2c pairing and the Hartley combination"},
      {.name = "odd rows reversed on both sides",
       .n = {15},
       .count = 2,
       .in = {.sign = {-1}, .gap = 1},
       .out = {.sign = {-1}},
       .expect = "the Hartley combination"},
      {.name = "rows in place", .n = {32}, .count = 3, .in_place = true, .expect = "in place"},
      {.name = "rows of one point", .n = {1}, .count = 4, .out = {.gap = 1}, .expect = "n = 1"},
      {.name = "rows of two points", .n = {2}, .count = 3, .expect = "n = 2"},
      {.name = "a Bluestein row", .n = {94}, .count = 2, .expect = "n = 94"},
      {.name = "planes", .n = {8, 12}, .count = 2, .expect = "phase 2"},
      {.name = "odd planes with gaps and mixed signs",
       .n = {5, 7},
       .count = 2,
       .in = {.sign = {-1, 1}, .gap = 1},
       .out = {.sign = {1, -1}, .gap = 2},
       .expect = "phase 2"},
      {.name = "planes in place", .n = {6, 10}, .count = 2, .in_place = true, .expect = "phase 2"},
      {.name = "volumes", .n = {4, 6, 5}, .count = 2, .expect = "phase 3"},
      {.name = "a volume with an axis of one point",
       .n = {3, 1, 9},
       .count = 2,
       .expect = "phase 2"},
      {.name = "one point in every axis", .n = {1, 1}, .count = 3, .expect = "phase 1"},
      {.name = "scales",
       .n = {20},
       .count = 2,
       .options_scale = 0.5,
       .in_scale = 3.0,
       .expect = "scales: one product, applied once in the last phase's store stage"},
      {.name = "real factors on both sides",
       .n = {18},
       .count = 3,
       .in_scale = 0.25,
       .fin = {.kind = real},
       .fout = {.kind = real, .reversed = true},
       .expect = "input edge: times a real factor in the first phase's read stage"},
      {.name = "an output factor shared by the batch, accumulated",
       .n = {21},
       .count = 3,
       .out = {.gap = 1},
       .fout = {.kind = real, .broadcast = true},
       .accumulate = true,
       .expect = "output edge: times a real factor, accumulated"},
      {.name = "planes with factors on both sides",
       .n = {6, 8},
       .count = 2,
       .in = {.sign = {1, -1}},
       .fin = {.kind = real, .reversed = true},
       .fout = {.kind = real},
       .expect = "the edges cost no separate pass"},
      {.name = "planes accumulated through the block",
       .n = {5, 6},
       .count = 2,
       .out = {.gap = 1},
       .options_scale = 2.0,
       .accumulate = true,
       .expect = "the phases before the last go through the workspace block"},
      {.name = "volumes with factors, accumulated",
       .n = {3, 4, 5},
       .count = 2,
       .fin = {.kind = real, .broadcast = true},
       .fout = {.kind = real},
       .accumulate = true,
       .expect = "phase 3"},
  };
}

TYPED_TEST(R2r, LayoutsAndEdgesOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const r2r_case& c : cases()) {
      run_case<TypeParam>(c, t, false);
    }
  }
}

TYPED_TEST(R2r, TeamsAndExecutorsAgreeWithSerial) {
  for (const r2r_case& c : cases()) {
    run_case<TypeParam>(c, fftn::isa::best, true);
  }
  // Lines long enough that a team shares one line's stages (rows) or a phase's lines (planes).
  run_case<TypeParam>({.name = "a long row", .n = {65536}, .count = 1, .expect = "65536"},
                      fftn::isa::best, true, false);
  run_case<TypeParam>({.name = "long rows with factors",
                       .n = {8192},
                       .count = 3,
                       .fin = {.kind = real},
                       .fout = {.kind = real},
                       .accumulate = true,
                       .expect = "8192"},
                      fftn::isa::best, true, false);
  run_case<TypeParam>({.name = "a large plane", .n = {96, 128}, .count = 1, .expect = "phase 2"},
                      fftn::isa::best, true);
  run_case<TypeParam>({.name = "large planes with factors, accumulated",
                       .n = {64, 80},
                       .count = 2,
                       .fin = {.kind = real},
                       .fout = {.kind = real},
                       .accumulate = true,
                       .expect = "phase 2"},
                      fftn::isa::best, true);
}

// axes_r2r: dimensions of a strided array, the others a batch, against the same transforms as
// a plan of tight planes.
TEST(R2r, AxesOfAStridedArray) {
  const std::array<fftn::axis, 3> array{{{.n = 6}, {.n = 5}, {.n = 4}}};
  const std::array<std::size_t, 2> dims{0, 2};
  const std::array<r2r_kind, 2> kinds{r2r_kind::dht, r2r_kind::dht};
  const auto s = fftn::axes_r2r<double>(array, dims, kinds);
  ASSERT_EQ(s.kinds.size(), 2U);
  const fftn::r2r_plan<double> p(s);
  fftn::test::input_source src(0xA2E5ULL);
  const std::vector<double> x = src.reals<double>(120);
  std::vector<double> y(120);
  fftn::workspace ws(p);
  p.execute(x.data(), y.data(), ws);
  fftn::test::outcome o;
  for (std::size_t m = 0; m < 5; ++m) {
    std::vector<double> plane(24);
    std::vector<double> got(24);
    for (std::size_t a = 0; a < 6; ++a) {
      for (std::size_t b = 0; b < 4; ++b) {
        plane[a * 4 + b] = x[a * 20 + m * 4 + b];
        got[a * 4 + b] = y[a * 20 + m * 4 + b];
      }
    }
    const std::array<std::size_t, 2> extents{6, 4};
    o.within(fftn::test::compare<double>(std::span<const double>(got),
                                         fftn::test::reference::dht(extents, plane)),
             dht_tolerance<double>(24, 2), "plane " + std::to_string(m));
  }
  EXPECT_TRUE(o.passed()) << o.detail();
  EXPECT_THROW(static_cast<void>(fftn::axes_r2r<double>(array, dims, std::span(kinds).first(1))),
               std::invalid_argument);
}

// The shape helpers' kinds and layouts.
TEST(R2r, Helpers) {
  const auto rows = fftn::rows_r2r<float>(10, 3, r2r_kind::dht);
  EXPECT_EQ(rows.kinds, std::vector<r2r_kind>{r2r_kind::dht});
  const auto planes = fftn::planes_r2r<double>(4, 6, 2, r2r_kind::dht);
  EXPECT_EQ(planes.kinds.size(), 2U);
  EXPECT_EQ(planes.axes.size(), 2U);
  const auto volumes = fftn::volumes_r2r<double>(2, 3, 4, 5, r2r_kind::dht);
  EXPECT_EQ(volumes.kinds.size(), 3U);
  EXPECT_EQ(volumes.batches[0].count, 5U);
  // planes_r2r is the plan of the tight C-ordered shape: the same bits as the explicit one.
  fftn::shape<double, double> explicit_shape;
  explicit_shape.axes = {{.n = 4}, {.n = 6}};
  explicit_shape.batches = {{.count = 2}};
  explicit_shape.kinds = {r2r_kind::dht, r2r_kind::dht};
  const fftn::r2r_plan<double> a(planes);
  const fftn::r2r_plan<double> b(explicit_shape);
  fftn::test::input_source src(0x4E1ULL);
  const std::vector<double> x = src.reals<double>(48);
  std::vector<double> ya(48);
  std::vector<double> yb(48);
  fftn::workspace ws(a);
  a.execute(x.data(), ya.data(), ws);
  b.execute(x.data(), yb.data(), ws);
  EXPECT_TRUE(same_bits(ya, yb));
  EXPECT_EQ(a.describe(), b.describe());
}

// Measured planning and wisdom are not built for real-to-real plans: the rules' plan, which
// describe() says.
TEST(R2r, PlannedByTheRules) {
  const auto s = fftn::rows_r2r<double>(64, 2, r2r_kind::dht);
  const fftn::r2r_plan<double> rules(s);
  const fftn::r2r_plan<double> measured(
      s, {.effort = fftn::planning::measure, .measure_seconds = 0.01});
  EXPECT_NE(measured.describe().find("planning: by the rules"), std::string::npos)
      << measured.describe();
  EXPECT_EQ(rules.describe().find("planning:"), std::string::npos) << rules.describe();
  fftn::wisdom w;
  w.add(rules);
  EXPECT_EQ(w.size(), 1U);
}

template <class F> std::string refusal(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "(nothing thrown)";
}

TEST(R2r, RefusalsNameTheField) {
  using S = fftn::shape<double, double>;
  const auto expect = [](const std::string& message, const char* field, const char* why) {
    EXPECT_NE(message.find(field), std::string::npos) << message;
    EXPECT_NE(message.find(why), std::string::npos) << message;
  };
  {
    S s = fftn::rows_r2r<double>(8, 1, r2r_kind::dht);
    s.kinds.clear();
    expect(refusal([&] { const fftn::r2r_plan<double> p(s); }), "shape.kinds", "one kind per axis");
  }
  {
    S s = fftn::planes_r2r<double>(4, 8, 1, r2r_kind::dht);
    s.kinds.pop_back();
    expect(refusal([&] { const fftn::r2r_plan<double> p(s); }), "shape.kinds", "1 kinds for 2");
  }
  for (const r2r_kind k : {r2r_kind::dct2, r2r_kind::dct3, r2r_kind::dst2, r2r_kind::dst3}) {
    expect(refusal([&] { const fftn::r2r_plan<double> p(fftn::rows_r2r<double>(8, 1, k)); }),
           "shape.kinds[0]", "not built yet (task 6.4)");
  }
  for (const r2r_kind k : {r2r_kind::dct1, r2r_kind::dct4, r2r_kind::dst1, r2r_kind::dst4}) {
    expect(refusal([&] { const fftn::r2r_plan<double> p(fftn::rows_r2r<double>(8, 1, k)); }),
           "shape.kinds[0]", "not built yet (task 6.5)");
  }
  {
    S s = fftn::planes_r2r<double>(4, 1, 1, r2r_kind::dht);
    s.kinds[1] = r2r_kind::dct1;
    expect(refusal([&] { const fftn::r2r_plan<double> p(s); }), "shape.kinds[1]",
           "dct1 on an axis of one point");
  }
  {
    S s = fftn::rows_r2r<double>(8, 1, r2r_kind::dht);
    s.kinds[0] = static_cast<r2r_kind>(42);
    expect(refusal([&] { const fftn::r2r_plan<double> p(s); }), "shape.kinds[0]",
           "is not an r2r_kind");
  }
  {
    auto c = fftn::rows_c2c<double>(8, 1, fftn::direction::forward);
    c.kinds = {r2r_kind::dht};
    expect(refusal([&] { const fftn::c2c_plan<double> p(c); }), "shape.kinds", "must be empty");
    expect(refusal([&] { const fftn::c2c_plan<double> p(c, {.effort = fftn::planning::measure}); }),
           "shape.kinds", "must be empty");
    auto r = fftn::rows_r2c<double>(8, 1);
    r.kinds = {r2r_kind::dht};
    expect(refusal([&] { const fftn::r2c_plan<double> p(r, {}, {}); }), "shape.kinds",
           "must be empty");
  }
  {
    S s = fftn::rows_r2r<double>(8, 1, r2r_kind::dht);
    s.runs = {{.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = 4}};
    expect(refusal([&] { const fftn::r2r_plan<double> p(s); }), "shape.runs",
           "take no stored runs");
  }
  {
    S s = fftn::rows_r2r<double>(8, 2, r2r_kind::dht);
    s.axes[0].n = 0;
    expect(refusal([&] { const fftn::r2r_plan<double> p(s); }), "shape.axes[0].n", "is 0");
  }
  {
    S s = fftn::rows_r2r<double>(8, 2, r2r_kind::dht);
    s.axes[0].in_stride = 2;
    expect(refusal([&] { const fftn::r2r_plan<double> p(s, {.in_place = true}); }),
           "options.in_place", "in_stride == out_stride");
  }
  {
    S s = fftn::rows_r2r<double>(8, 2, r2r_kind::dht);
    s.batches[0].out_dist = 4;
    expect(refusal([&] { const fftn::r2r_plan<double> p(s); }), "shape", "overlap");
  }
  const S rows = fftn::rows_r2r<double>(8, 2, r2r_kind::dht);
  {
    fftn::plan_edges<double, double> e;
    e.in.mul = {.kind = factor_kind::complex, .axis_strides = {1}};
    expect(refusal([&] { const fftn::r2r_plan<double> p(rows, {}, e); }), "edges.in.mul.kind",
           "a real side takes a real factor");
  }
  {
    fftn::plan_edges<double, double> e;
    e.out.mul = {.kind = factor_kind::complex, .axis_strides = {1}};
    expect(refusal([&] { const fftn::r2r_plan<double> p(rows, {}, e); }), "edges.out.mul.kind",
           "a real side takes a real factor");
  }
  {
    fftn::plan_edges<double, double> e;
    e.out.accumulate = true;
    expect(refusal([&] { const fftn::r2r_plan<double> p(rows, {.in_place = true}, e); }),
           "edges.out.accumulate", "in-place");
  }
  {
    fftn::plan_edges<double, double> e;
    e.in.accumulate = true;
    expect(refusal([&] { const fftn::r2r_plan<double> p(rows, {}, e); }), "edges.in.accumulate",
           "output edge only");
  }
  {
    fftn::plan_edges<double, double> e;
    e.out.scale = 2.0;
    expect(refusal([&] { const fftn::r2r_plan<double> p(rows, {.scale = 3.0}, e); }),
           "edges.out.scale", "both given");
  }
  {
    fftn::plan_edges<double, double> e;
    e.in.mul = {.kind = real, .axis_strides = {1, 2}};
    expect(refusal([&] { const fftn::r2r_plan<double> p(rows, {}, e); }),
           "edges.in.mul.axis_strides", "has 2 strides");
  }
  expect(refusal([&] {
           const fftn::r2r_plan<float> p(fftn::rows_r2r<float>(8, 1, r2r_kind::dht),
                                         {.scale = 1e300});
         }),
         "options.scale", "outside the range of float");
}

TEST(R2rDeathTest, MissingFactorArraysAbort) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto s = fftn::planes_r2r<double>(4, 8, 2, r2r_kind::dht);
  fftn::plan_edges<double, double> e;
  e.out.mul = {.kind = real, .axis_strides = {8, 1}, .batch_strides = {32}};
  const fftn::r2r_plan<double> p(s, {}, e);
  std::vector<double> x(64);
  std::vector<double> y(64);
  fftn::workspace ws(p, 2);
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws), "without the factor arrays");
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws, {}), "without the factor arrays");
  EXPECT_DEATH(p.execute(x.data(), y.data(), fftn::team{}, ws, fftn::exec_args{}),
               "without the factor arrays");
}

} // namespace
