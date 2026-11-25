// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Plan edges on real and multi-axis plans (fftn/edges.hpp, DESIGN.md 12.1; task 5.1b): every
// operation of the vocabulary on r2c and c2r rows and on c2c, r2c and c2r planes and volumes,
// against "transform then separate pass" evaluated on the separable quadruple-precision
// reference (rounded to double after each axis) within the derived tolerance plus a rounding
// per multiplication and addition; factor strides of both signs along every axis and the
// batch, broadcasts included; axes listed in any order; in place; padded rows with canaries
// in their gaps; the input and the factor arrays left as they were; serial, pieces of the
// batch, teams and an executor bitwise alike, a team program (plan_core::team) included; and
// the executions without a factor array ending the program.

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
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using fftn::factor_kind;
using fftn::detail::transform_kind;
using cd = std::complex<double>;
using dims_t = std::vector<std::size_t>;
using strides_t = std::vector<std::ptrdiff_t>;

constexpr factor_kind real = factor_kind::real;
constexpr factor_kind cplx = factor_kind::complex;

std::size_t product(const dims_t& d) {
  return std::accumulate(d.begin(), d.end(), std::size_t{1}, std::multiplies<>());
}

// A factor as a case gives it: its kind and, per axis in C order and for the batch, the sign of
// its stride (0 broadcasts), whose magnitude is the tight C-ordered layout's of the edge's side.
struct factor_spec {
  factor_kind kind = factor_kind::none;
  std::vector<int> axes; // empty: +1 along every axis
  int batch = 1;
};

struct shape_case {
  const char* name = "";
  transform_kind kind = transform_kind::c2c;
  dims_t dims;                       // the logical lengths in C order (the last contiguous)
  std::vector<std::size_t> listed{}; // c2c: shape.axes[i] is C axis listed[i]; empty: C order
  std::size_t count = 1;
  fftn::direction dir = fftn::direction::forward; // c2c only
  double options_scale = 1.0;
  double in_scale = 1.0;
  double out_scale = 1.0;
  factor_spec in{};
  factor_spec out{};
  bool accumulate = false;
  bool in_place = false;
  std::ptrdiff_t gap = 0; // out of place: elements after each row and each transform
  const char* expect = "";
  // c2c: the transforms one element apart, each axis's stride the C-ordered one times the
  // count (the lanes over adjacent transforms, DESIGN.md 9.3).
  bool adjacent = false;
};

bool real_in(const shape_case& c) { return c.kind == transform_kind::r2c; }
bool real_out(const shape_case& c) { return c.kind == transform_kind::c2r; }

// The lengths of one side's elements: the complex side of a real transform has n/2 + 1 bins
// along the last axis.
dims_t side_dims(const shape_case& c, bool output) {
  dims_t d = c.dims;
  const bool halved = output ? c.kind == transform_kind::r2c : c.kind == transform_kind::c2r;
  if (halved) {
    d.back() = d.back() / 2 + 1;
  }
  return d;
}

// One side's layout in its own elements (real or complex), C order.
struct layout {
  strides_t stride;
  std::ptrdiff_t dist = 0;
  std::size_t span = 0; // elements from element 0 of transform 0 to one past the last

  [[nodiscard]] std::size_t at(const dims_t& j, std::size_t b) const {
    std::ptrdiff_t i = static_cast<std::ptrdiff_t>(b) * dist;
    for (std::size_t d = 0; d < j.size(); ++d) {
      i += static_cast<std::ptrdiff_t>(j[d]) * stride[d];
    }
    return static_cast<std::size_t>(i);
  }
};

layout tight_layout(const dims_t& d, std::size_t count, std::ptrdiff_t gap) {
  layout l;
  const std::size_t r = d.size();
  l.stride.assign(r, 1);
  for (std::size_t i = r - 1; i > 0; --i) {
    l.stride[i - 1] = l.stride[i] * static_cast<std::ptrdiff_t>(d[i]) + (i == r - 1 ? gap : 0);
  }
  l.dist = l.stride[0] * static_cast<std::ptrdiff_t>(d[0]) + gap;
  l.span = static_cast<std::size_t>(l.dist) * count;
  return l;
}

// Calls f(j) for every coordinate of d in C order.
template <class F> void each(const dims_t& d, F&& f) {
  dims_t j(d.size(), 0);
  for (std::size_t k = 0; k < product(d); ++k) {
    f(j);
    for (std::size_t a = d.size(); a > 0; --a) {
      if (++j[a - 1] < d[a - 1]) {
        break;
      }
      j[a - 1] = 0;
    }
  }
}

// A factor array over a side's coordinates and the batch: the storage, the strides in C order
// and the offset of coordinate 0.
template <class T> struct factor_array {
  factor_kind kind = factor_kind::none;
  strides_t stride;
  std::ptrdiff_t batch = 0;
  std::vector<T> reals; // complex values as pairs
  std::ptrdiff_t zero = 0;

  factor_array(const factor_spec& f, const dims_t& d, std::size_t count,
               fftn::test::input_source& src)
      : kind(f.kind) {
    if (kind == factor_kind::none) {
      return;
    }
    const std::size_t r = d.size();
    stride.assign(r, 0);
    std::ptrdiff_t mag = 1;
    for (std::size_t a = r; a > 0; --a) {
      const int sign = f.axes.empty() ? 1 : f.axes[a - 1];
      stride[a - 1] = sign * mag;
      mag *= static_cast<std::ptrdiff_t>(d[a - 1]);
    }
    batch = f.batch * mag;
    std::ptrdiff_t lo = 0;
    std::ptrdiff_t hi = 0;
    const auto reach = [&](std::ptrdiff_t s, std::size_t extent) {
      const std::ptrdiff_t last = extent > 1 ? s * static_cast<std::ptrdiff_t>(extent - 1) : 0;
      lo += std::min<std::ptrdiff_t>(0, last);
      hi += std::max<std::ptrdiff_t>(0, last);
    };
    for (std::size_t a = 0; a < r; ++a) {
      reach(stride[a], d[a]);
    }
    reach(batch, count);
    zero = -lo;
    const auto size = static_cast<std::size_t>(hi - lo + 1);
    reals = src.reals<T>(kind == cplx ? 2 * size : size);
  }

  [[nodiscard]] fftn::factor spec(const std::vector<std::size_t>& listed) const {
    if (kind == factor_kind::none) {
      return {};
    }
    fftn::factor f{.kind = kind, .axis_strides = {}, .batch_strides = {batch}};
    for (const std::size_t a : listed) {
      f.axis_strides.push_back(stride[a]);
    }
    return f;
  }

  [[nodiscard]] const void* address() const {
    if (kind == factor_kind::none) {
      return nullptr;
    }
    return reals.data() + (kind == cplx ? 2 : 1) * zero;
  }

  [[nodiscard]] cd at(const dims_t& j, std::size_t b) const {
    if (kind == factor_kind::none) {
      return {1.0, 0.0};
    }
    std::ptrdiff_t i = zero + static_cast<std::ptrdiff_t>(b) * batch;
    for (std::size_t a = 0; a < j.size(); ++a) {
      i += static_cast<std::ptrdiff_t>(j[a]) * stride[a];
    }
    const auto u = static_cast<std::size_t>(i);
    if (kind == real) {
      return {static_cast<double>(reals[u]), 0.0};
    }
    return {static_cast<double>(reals[2 * u]), static_cast<double>(reals[2 * u + 1])};
  }
};

// The C-ordered index of j in d.
std::size_t flat(const dims_t& d, const dims_t& j) {
  std::size_t i = 0;
  for (std::size_t a = 0; a < d.size(); ++a) {
    i = i * d[a] + j[a];
  }
  return i;
}

// The DFT along axis a of a C-ordered array of lengths d, from the quadruple-precision
// reference, rounded to double.
void dft_axis(int sign, const dims_t& d, std::vector<cd>& x, std::size_t a) {
  std::size_t inner = 1;
  for (std::size_t e = a + 1; e < d.size(); ++e) {
    inner *= d[e];
  }
  const std::size_t n = d[a];
  const std::size_t outer = x.size() / (n * inner);
  std::vector<cd> line(n);
  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t i = 0; i < inner; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        line[j] = x[(o * n + j) * inner + i];
      }
      const auto y = fftn::test::reference::c2c(sign, line);
      for (std::size_t k = 0; k < n; ++k) {
        x[(o * n + k) * inner + i] = {y[k].re.hi, y[k].im.hi};
      }
    }
  }
}

// The transform of one input (C order over the input side's lengths) as FFTN computes it:
// c2c over every axis; r2c the rows, then the other axes forward; c2r the other axes backward,
// then the rows. Complex values; c2r's are real.
std::vector<cd> reference(const shape_case& c, const std::vector<cd>& x) {
  const std::size_t r = c.dims.size();
  if (c.kind == transform_kind::c2c) {
    std::vector<cd> y = x;
    for (std::size_t a = 0; a < r; ++a) {
      dft_axis(static_cast<int>(c.dir), c.dims, y, a);
    }
    return y;
  }
  const std::size_t n = c.dims.back();
  const std::size_t h = n / 2 + 1;
  const std::size_t rows = product(c.dims) / n;
  if (c.kind == transform_kind::r2c) {
    std::vector<cd> y(rows * h);
    std::vector<double> line(n);
    for (std::size_t q = 0; q < rows; ++q) {
      for (std::size_t j = 0; j < n; ++j) {
        line[j] = x[q * n + j].real();
      }
      const auto z = fftn::test::reference::r2c(line);
      for (std::size_t k = 0; k < h; ++k) {
        y[q * h + k] = {z[k].re.hi, z[k].im.hi};
      }
    }
    const dims_t half = side_dims(c, true);
    for (std::size_t a = 0; a + 1 < r; ++a) {
      dft_axis(-1, half, y, a);
    }
    return y;
  }
  std::vector<cd> z = x;
  const dims_t half = side_dims(c, false);
  for (std::size_t a = 0; a + 1 < r; ++a) {
    dft_axis(+1, half, z, a);
  }
  std::vector<cd> y(rows * n);
  for (std::size_t q = 0; q < rows; ++q) {
    const std::vector<cd> line(z.begin() + static_cast<std::ptrdiff_t>(q * h),
                               z.begin() + static_cast<std::ptrdiff_t>((q + 1) * h));
    const auto v = fftn::test::reference::c2r(n, line);
    for (std::size_t j = 0; j < n; ++j) {
      y[q * n + j] = {v[j].hi, 0.0};
    }
  }
  return y;
}

template <class T> struct shape_run {
  shape_case c;
  dims_t in_dims;
  dims_t out_dims;
  layout lin;
  layout lout;
  std::vector<std::size_t> listed;
  std::vector<T> x;    // the input buffer as reals (complex values as pairs)
  std::vector<T> init; // the output buffer before execution (out of place)
  factor_array<T> fin;
  factor_array<T> fout;
  fftn::options opts;

  shape_run(const shape_case& sc, fftn::isa target, fftn::test::input_source& src)
      : c(sc), in_dims(side_dims(sc, false)), out_dims(side_dims(sc, true)),
        fin(sc.in, in_dims, sc.count, src), fout(sc.out, out_dims, sc.count, src) {
    const std::size_t r = c.dims.size();
    listed = c.listed;
    if (listed.empty()) {
      listed.resize(r);
      std::iota(listed.begin(), listed.end(), std::size_t{0});
    }
    if (c.in_place && c.kind != transform_kind::c2c) {
      // FFTW's padded layout: the complex side tight, the real side's strides twice its.
      const bool r2c = c.kind == transform_kind::r2c;
      const layout complex = tight_layout(r2c ? out_dims : in_dims, c.count, 0);
      layout padded = complex;
      for (std::size_t a = 0; a + 1 < r; ++a) {
        padded.stride[a] *= 2;
      }
      padded.dist *= 2;
      padded.span *= 2;
      lin = r2c ? padded : complex;
      lout = r2c ? complex : padded;
    } else if (c.adjacent) {
      lin = tight_layout(in_dims, 1, 0);
      for (std::ptrdiff_t& st : lin.stride) {
        st *= static_cast<std::ptrdiff_t>(c.count);
      }
      lin.dist = 1;
      lin.span = product(in_dims) * c.count;
      lout = lin;
    } else {
      lin = tight_layout(in_dims, c.count, c.in_place ? 0 : c.gap);
      lout = c.in_place ? lin : tight_layout(out_dims, c.count, c.gap);
    }
    opts = {.in_place = c.in_place, .scale = c.options_scale, .force_isa = target};
    x = src.reals<T>(lin.span * (real_in(c) ? 1 : 2));
    if (!c.in_place) {
      init = src.reals<T>(lout.span * (real_out(c) ? 1 : 2));
      for (T& v : init) {
        v *= static_cast<T>(std::sqrt(static_cast<double>(product(c.dims))));
      }
    }
  }

  template <class In, class Out> [[nodiscard]] fftn::shape<In, Out> shape() const {
    fftn::shape<In, Out> s;
    s.dir = c.dir;
    for (const std::size_t a : listed) {
      s.axes.push_back({.n = c.dims[a], .in_stride = lin.stride[a], .out_stride = lout.stride[a]});
    }
    s.batches = {{.count = c.count, .in_dist = lin.dist, .out_dist = lout.dist}};
    return s;
  }

  template <class In, class Out> [[nodiscard]] fftn::plan_edges<In, Out> edges() const {
    fftn::plan_edges<In, Out> e;
    e.in = {.scale = c.in_scale, .mul = fin.spec(listed), .accumulate = false};
    e.out = {.scale = c.out_scale, .mul = fout.spec(listed), .accumulate = c.accumulate};
    return e;
  }

  [[nodiscard]] fftn::exec_args args() const {
    return {.in_factor = fin.address(), .out_factor = fout.address()};
  }

  [[nodiscard]] std::size_t scalars_out() const { return lout.span * (real_out(c) ? 1 : 2); }

  // The value of output element (j, b) in a buffer of reals.
  [[nodiscard]] cd value(const std::vector<T>& y, const dims_t& j, std::size_t b) const {
    const std::size_t i = lout.at(j, b);
    if (real_out(c)) {
      return {static_cast<double>(y[i]), 0.0};
    }
    return {static_cast<double>(y[2 * i]), static_cast<double>(y[2 * i + 1])};
  }

  // Checks y (the output buffer after execution) against the reference and, out of place,
  // every scalar outside the described elements against init.
  void check(const std::vector<T>& y, const std::string& describe, const std::string& what) const {
    SCOPED_TRACE(what);
    const double scale = c.options_scale * c.in_scale * c.out_scale;
    fftn::test::outcome o;
    std::vector<bool> described(y.size(), false);
    for (std::size_t b = 0; b < c.count; ++b) {
      std::vector<cd> xin(product(in_dims));
      each(in_dims, [&](const dims_t& j) {
        const std::size_t i = lin.at(j, b);
        const cd v = real_in(c)
                         ? cd(static_cast<double>(x[i]), 0.0)
                         : cd(static_cast<double>(x[2 * i]), static_cast<double>(x[2 * i + 1]));
        xin[flat(in_dims, j)] = v * fin.at(j, b);
      });
      const std::vector<cd> exact = reference(c, xin);
      std::vector<fftn::test::exact_complex> want;
      std::vector<std::complex<T>> got;
      each(out_dims, [&](const dims_t& j) {
        cd v = exact[flat(out_dims, j)] * fout.at(j, b) * scale;
        if (c.accumulate) {
          v += value(init, j, b);
        }
        want.push_back(fftn::test::exact_from(v));
        const cd g = value(y, j, b);
        got.emplace_back(static_cast<T>(g.real()), static_cast<T>(g.imag()));
        const std::size_t i = lout.at(j, b);
        if (real_out(c)) {
          described[i] = true;
        } else {
          described[2 * i] = described[2 * i + 1] = true;
        }
      });
      // The transform's bound; the reference's rounding after each axis; a rounding for each
      // factor's product, the scale and the accumulation.
      const std::size_t points = product(c.dims);
      const auto tol = fftn::test::derived_tolerance<T>(points, fftn::test::fftn_bound(points)) +
                       fftn::test::input_rounding<double>() * static_cast<double>(c.dims.size()) +
                       fftn::test::input_rounding<T>() * 4;
      o.within(fftn::test::compare<T>(std::span<const std::complex<T>>(got), want), tol,
               what + ", transform " + std::to_string(b));
    }
    EXPECT_TRUE(o.passed()) << o.detail() << describe;
    if (!c.in_place) {
      std::size_t touched = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        if (!described[i] && std::memcmp(&y[i], &init[i], sizeof(T)) != 0) {
          ++touched;
        }
      }
      EXPECT_EQ(touched, 0U) << "elements outside the transforms written";
    }
  }
};

template <class T> bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

// A plan built through the internal entry with construction knobs (build_settings), executed
// through the internal entries: the strategies forced in turn (AGENTS.md section 5).
struct core_plan {
  std::shared_ptr<const fftn::detail::plan_core> core;

  [[nodiscard]] std::string describe() const { return fftn::detail::describe(*core); }
  [[nodiscard]] std::size_t workspace_bytes(int lanes) const {
    return fftn::detail::workspace_bytes(*core, lanes);
  }
  void execute(const void* in, void* out, const fftn::workspace& ws,
               const fftn::exec_args& a) const {
    fftn::detail::execute_all(*core, in, out, ws, a);
  }
  void execute_transforms(std::size_t first, std::size_t count, const void* in, void* out,
                          const fftn::workspace& ws, const fftn::exec_args& a) const {
    fftn::detail::execute_core(*core, first, count, in, out, ws, a);
  }
  void execute(const void* in, void* out, const fftn::team& t, const fftn::workspace& ws,
               fftn::completion done, const fftn::exec_args& a) const {
    fftn::detail::execute_team(*core, in, out, t, ws, done, a);
  }
  void execute(const void* in, void* out, const fftn::thread_pool& pool, const fftn::workspace& ws,
               const fftn::exec_args& a) const {
    fftn::detail::execute_on(*core, in, out, pool, ws, a);
  }
};

template <class T, transform_kind K, class Plan>
void run_plan(const shape_case& sc, const shape_run<T>& run, const Plan& p, fftn::isa target,
              bool threads, bool reference, const char* strategy);

// The plan of a case, executed serially, in two pieces of the batch, by teams of three threads
// and on a pool of three, each from the same buffers: all bitwise alike, the serial result
// against the reference where `reference` is set. With `settings` the plan is built with
// those knobs and must name `strategy` in describe().
template <class T, transform_kind K>
void run_case(const shape_case& sc, fftn::isa target, bool threads, bool reference = true,
              const fftn::detail::build_settings* settings = nullptr,
              const char* strategy = nullptr) {
  using C = std::complex<T>;
  using In = std::conditional_t<K == transform_kind::r2c, T, C>;
  using Out = std::conditional_t<K == transform_kind::c2r, T, C>;
  ASSERT_EQ(sc.kind, K);
  fftn::test::input_source src(0x5A9E5ULL + product(sc.dims) * 7 + sc.count);
  const shape_run<T> run(sc, target, src);
  if (settings != nullptr) {
    const core_plan p{fftn::detail::make_plan_core_with_edges(
        fftn::detail::view_of(run.template shape<In, Out>()), run.opts,
        fftn::detail::view_of(run.template edges<In, Out>()), *settings)};
    run_plan<T, K>(sc, run, p, target, threads, reference, strategy);
    return;
  }
  const fftn::plan<In, Out> p(run.template shape<In, Out>(), run.opts,
                              run.template edges<In, Out>());
  run_plan<T, K>(sc, run, p, target, threads, reference, strategy);
}

template <class T, transform_kind K, class Plan>
void run_plan(const shape_case& sc, const shape_run<T>& run, const Plan& p, fftn::isa target,
              bool threads, bool reference, const char* strategy) {
  using C = std::complex<T>;
  using In = std::conditional_t<K == transform_kind::r2c, T, C>;
  using Out = std::conditional_t<K == transform_kind::c2r, T, C>;
  const std::string describe = p.describe();
  const std::string what = std::string(sc.name) + ", " + fftn::isa_name(target) +
                           (sizeof(T) == 8 ? " double" : " float");
  EXPECT_NE(describe.find(sc.expect), std::string::npos)
      << what << ": no '" << sc.expect << "' in\n"
      << describe;
  if (strategy != nullptr) {
    EXPECT_NE(describe.find(strategy), std::string::npos)
        << what << ": not the strategy '" << strategy << "':\n"
        << describe;
  }
  const fftn::exec_args args = run.args();
  const std::vector<T> fin = run.fin.reals;
  const std::vector<T> fout = run.fout.reals;

  // Out of place: the input stays in xs, the output in a copy of init; in place: the buffer
  // is a copy of x.
  std::vector<T> xs = run.x;
  const auto fresh = [&] { return sc.in_place ? run.x : run.init; };
  const auto in_of = [&](std::vector<T>& y) {
    return reinterpret_cast<const In*>(sc.in_place ? y.data() : xs.data());
  };
  const auto out_of = [](std::vector<T>& y) { return reinterpret_cast<Out*>(y.data()); };

  fftn::workspace ws(p.workspace_bytes(3));
  std::vector<T> serial = fresh();
  p.execute(in_of(serial), out_of(serial), ws, args);
  if (reference) {
    run.check(serial, describe, what);
  }
  if (!sc.in_place) {
    EXPECT_TRUE(same_bits(xs, run.x)) << what << ": input written";
  }
  EXPECT_TRUE(same_bits(run.fin.reals, fin) && same_bits(run.fout.reals, fout))
      << what << ": factor written";

  std::vector<T> pieces = fresh();
  const std::size_t half = sc.count / 2;
  p.execute_transforms(0, half, in_of(pieces), out_of(pieces), ws, args);
  p.execute_transforms(half, sc.count - half, in_of(pieces), out_of(pieces), ws, args);
  EXPECT_TRUE(same_bits(pieces, serial)) << what << ": pieces of the batch differ";
  if (!threads) {
    return;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    std::vector<T> teamed = fresh();
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        p.execute(in_of(teamed), out_of(teamed), fftn::team{.rank = r, .size = 3}, ws, done, args);
      });
    }
    p.execute(in_of(teamed), out_of(teamed), fftn::team{.rank = 0, .size = 3}, ws, done, args);
    for (std::thread& m : members) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << what << ": a team differs";
  }
  const fftn::thread_pool pool(3);
  std::vector<T> pooled = fresh();
  p.execute(in_of(pooled), out_of(pooled), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << what << ": the executor differs";
}

constexpr const char* before = "in a separate phase before the transform";
constexpr const char* after = "in a separate phase after the transform";

std::vector<shape_case> c2c_cases() {
  using enum transform_kind;
  return {
      {.name = "plane, input factor",
       .kind = c2c,
       .dims = {12, 10},
       .count = 3,
       .in = {cplx, {}, 1},
       .expect = before},
      {.name = "plane, output factor and scale",
       .kind = c2c,
       .dims = {8, 6},
       .count = 2,
       .dir = fftn::direction::backward,
       .options_scale = 0.5,
       .out = {cplx, {1, -1}, -1},
       .expect = after},
      {.name = "volume, real input, reversed complex output, accumulated",
       .kind = c2c,
       .dims = {4, 6, 5},
       .count = 2,
       .in_scale = 2.0,
       .in = {real, {}, 1},
       .out = {cplx, {-1, -1, -1}, 1},
       .accumulate = true,
       .expect = "accumulated in a separate phase after the transform"},
      {.name = "plane in place, both factors",
       .kind = c2c,
       .dims = {16, 8},
       .count = 2,
       .options_scale = 0.25,
       .in = {cplx, {}, 1},
       .out = {real, {1, 1}, 0},
       .in_place = true,
       .expect = before},
      {.name = "plane listed in reverse, padded, accumulated",
       .kind = c2c,
       .dims = {6, 10},
       .listed = {1, 0},
       .count = 2,
       .in = {cplx, {-1, 1}, 1},
       .out = {cplx, {1, -1}, 0},
       .accumulate = true,
       .gap = 3,
       .expect = after},
      {.name = "plane, factors broadcast along an axis and the batch",
       .kind = c2c,
       .dims = {7, 12},
       .count = 3,
       .in = {cplx, {0, 1}, 0},
       .out = {real, {1, 0}, 0},
       .expect = before},
      {.name = "volume, accumulation alone",
       .kind = c2c,
       .dims = {3, 4, 6},
       .count = 2,
       .out_scale = 0.125,
       .accumulate = true,
       .expect = "the output accumulated in a separate phase"},
      {.name = "volume in place, input factor",
       .kind = c2c,
       .dims = {5, 4, 8},
       .count = 1,
       .in = {cplx, {1, -1, 1}, 1},
       .in_place = true,
       .expect = before},
      {.name = "plane with an axis of one point",
       .kind = c2c,
       .dims = {1, 12},
       .count = 2,
       .in = {cplx, {}, 1},
       .out = {cplx, {}, 1},
       .accumulate = true,
       .expect = "edges:"},
      {.name = "plane of columns",
       .kind = c2c,
       .dims = {9, 1},
       .count = 2,
       .in = {real, {}, 1},
       .out = {cplx, {-1, 1}, 1},
       .expect = "edges:"},
      {.name = "plane, lanes over columns",
       .kind = c2c,
       .dims = {32, 64},
       .count = 1,
       .in = {cplx, {}, 1},
       .out = {cplx, {}, 1},
       .accumulate = true,
       .expect = after},
  };
}

std::vector<shape_case> r2c_cases() {
  using enum transform_kind;
  return {
      {.name = "r2c rows, both factors",
       .kind = r2c,
       .dims = {32},
       .count = 3,
       .in = {real, {}, 1},
       .out = {cplx, {1}, -1},
       .expect = before},
      {.name = "r2c odd rows, accumulated",
       .kind = r2c,
       .dims = {15},
       .count = 2,
       .options_scale = 0.5,
       .in = {real, {-1}, 1},
       .accumulate = true,
       .gap = 2,
       .expect = "accumulated"},
      {.name = "r2c rows in place",
       .kind = r2c,
       .dims = {20},
       .count = 2,
       .in = {real, {}, 1},
       .out = {real, {1}, 0},
       .in_place = true,
       .expect = after},
      {.name = "r2c plane, accumulated output factor",
       .kind = r2c,
       .dims = {6, 16},
       .count = 2,
       .in = {real, {1, -1}, 1},
       .out = {cplx, {}, 1},
       .accumulate = true,
       .gap = 2,
       .expect = after},
      {.name = "r2c volume, output real factor",
       .kind = r2c,
       .dims = {4, 5, 8},
       .count = 1,
       .out_scale = 0.25,
       .out = {real, {1, 0, 1}, 1},
       .expect = after},
      {.name = "r2c plane in place",
       .kind = r2c,
       .dims = {8, 12},
       .count = 2,
       .in = {real, {}, 1},
       .out = {cplx, {-1, 1}, 1},
       .in_place = true,
       .expect = before},
      {.name = "r2c rows of two",
       .kind = r2c,
       .dims = {2},
       .count = 4,
       .in = {real, {}, 1},
       .out = {cplx, {}, 1},
       .expect = "edges:"},
  };
}

std::vector<shape_case> c2r_cases() {
  using enum transform_kind;
  return {
      {.name = "c2r rows, both factors",
       .kind = c2r,
       .dims = {32},
       .count = 3,
       .in = {cplx, {}, 1},
       .out = {real, {-1}, 1},
       .expect = before},
      {.name = "c2r odd rows, accumulated",
       .kind = c2r,
       .dims = {9},
       .count = 2,
       .in_scale = 0.5,
       .out_scale = 3.0,
       .out = {real, {}, 1},
       .accumulate = true,
       .gap = 1,
       .expect = "accumulated"},
      {.name = "c2r rows in place",
       .kind = c2r,
       .dims = {16},
       .count = 2,
       .in = {real, {1}, 0},
       .out = {real, {}, 1},
       .in_place = true,
       .expect = before},
      {.name = "c2r plane, accumulated",
       .kind = c2r,
       .dims = {6, 16},
       .count = 2,
       .in = {cplx, {-1, 1}, 1},
       .out = {real, {}, 1},
       .accumulate = true,
       .gap = 3,
       .expect = after},
      {.name = "c2r volume, input factor",
       .kind = c2r,
       .dims = {4, 6, 10},
       .count = 1,
       .options_scale = 2.0,
       .in = {cplx, {}, 1},
       .expect = before},
      {.name = "c2r plane in place",
       .kind = c2r,
       .dims = {8, 10},
       .count = 2,
       .in = {cplx, {1, -1}, 0},
       .out = {real, {}, 1},
       .in_place = true,
       .expect = after},
      {.name = "c2r plane, accumulation alone",
       .kind = c2r,
       .dims = {5, 12},
       .count = 2,
       .accumulate = true,
       .expect = "the output accumulated"},
  };
}

template <class T> class EdgeShapes : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(EdgeShapes, precisions);

TYPED_TEST(EdgeShapes, EveryOperationOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const shape_case& c : c2c_cases()) {
      run_case<TypeParam, transform_kind::c2c>(c, t, false);
    }
    for (const shape_case& c : r2c_cases()) {
      run_case<TypeParam, transform_kind::r2c>(c, t, false);
    }
    for (const shape_case& c : c2r_cases()) {
      run_case<TypeParam, transform_kind::c2r>(c, t, false);
    }
  }
}

// Teams and executors on every case, and on shapes large enough to be shared out: the lines of
// the planes' phases, the edge phases' included, and the stages of a long real row.
TYPED_TEST(EdgeShapes, TeamsAndExecutorsAgreeWithSerial) {
  for (const shape_case& c : c2c_cases()) {
    run_case<TypeParam, transform_kind::c2c>(c, fftn::isa::best, true);
  }
  for (const shape_case& c : r2c_cases()) {
    run_case<TypeParam, transform_kind::r2c>(c, fftn::isa::best, true);
  }
  for (const shape_case& c : c2r_cases()) {
    run_case<TypeParam, transform_kind::c2r>(c, fftn::isa::best, true);
  }
  using enum transform_kind;
  run_case<TypeParam, c2c>({.name = "large plane, both factors, accumulated",
                            .kind = c2c,
                            .dims = {256, 128},
                            .count = 1,
                            .in = {cplx, {}, 1},
                            .out = {real, {}, 1},
                            .accumulate = true,
                            .expect = after},
                           fftn::isa::best, true, false);
  run_case<TypeParam, r2c>({.name = "large r2c plane",
                            .kind = r2c,
                            .dims = {128, 256},
                            .count = 2,
                            .in = {real, {}, 1},
                            .out = {cplx, {}, 1},
                            .expect = after},
                           fftn::isa::best, true, false);
  run_case<TypeParam, c2r>({.name = "large c2r plane, accumulated",
                            .kind = c2r,
                            .dims = {64, 512},
                            .count = 1,
                            .in = {cplx, {}, 1},
                            .out = {real, {}, 1},
                            .accumulate = true,
                            .expect = after},
                           fftn::isa::best, true, false);
  run_case<TypeParam, r2c>({.name = "long r2c row",
                            .kind = r2c,
                            .dims = {65536},
                            .count = 1,
                            .in = {real, {}, 1},
                            .out = {cplx, {}, 1},
                            .accumulate = true,
                            .expect = after},
                           fftn::isa::best, true, false);
  run_case<TypeParam, c2r>({.name = "long c2r row",
                            .kind = c2r,
                            .dims = {32768},
                            .count = 2,
                            .in = {cplx, {}, 1},
                            .out = {real, {}, 1},
                            .expect = after},
                           fftn::isa::best, true, false);
}

// The team program of a plan with edges (plan_core::team; X43): narrower column blocks, or a
// volume's plane order, with its own edge phases and edge records. A team's results equal the
// program's bit for bit.
TEST(EdgeShapes, TeamProgramsCarryTheEdges) {
  using C = std::complex<double>;
  fftn::detail::build_settings settings;
  settings.profile.column_team_bytes = 64;
  settings.profile.line_bytes = 64;
  settings.profile.column_wide_bytes = 0;
  settings.profile.team_volume_planes = true;
  for (const dims_t& dims : {dims_t{128, 128}, dims_t{16, 32, 64}}) {
    fftn::shape<C, C> s;
    for (const std::size_t n : dims) {
      s.axes.push_back({.n = n});
    }
    s.batches = {{.count = 2}};
    const std::size_t points = product(dims);
    strides_t tight(dims.size(), 1);
    for (std::size_t a = dims.size() - 1; a > 0; --a) {
      tight[a - 1] = tight[a] * static_cast<std::ptrdiff_t>(dims[a]);
    }
    const fftn::plan_edges<C, C> edges{
        .in = {.mul = {.kind = cplx,
                       .axis_strides = tight,
                       .batch_strides = {static_cast<std::ptrdiff_t>(points)}}},
        .out = {.scale = 0.5, .mul = {.kind = real, .axis_strides = tight}, .accumulate = true},
    };
    const auto core = fftn::detail::make_plan_core_with_edges(
        fftn::detail::view_of(s), {}, fftn::detail::view_of(edges), settings);
    ASSERT_NE(core->team, nullptr) << fftn::detail::describe(*core);
    EXPECT_TRUE(core->team->needs_args);
    EXPECT_EQ(core->team->edge_records.size(), core->team->prog.records.size());
    fftn::test::input_source src(points);
    const auto x = src.complexes<double>(2 * points);
    const auto fin = src.complexes<double>(2 * points);
    const auto fout = src.reals<double>(points);
    const auto init = src.complexes<double>(2 * points);
    const fftn::exec_args args{.in_factor = fin.data(), .out_factor = fout.data()};
    const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 4));
    std::vector<C> serial = init;
    fftn::detail::execute_core(*core, 0, 2, x.data(), serial.data(), ws, args);
    // The serial result is the transform's (tests above check the program); the team's
    // must equal it bit for bit.
    std::vector<C> teamed = init;
    std::vector<std::thread> members;
    for (int r = 1; r < 4; ++r) {
      members.emplace_back([&, r] {
        fftn::detail::execute_team(*core, x.data(), teamed.data(), fftn::team{.rank = r, .size = 4},
                                   ws, fftn::completion::all, args);
      });
    }
    fftn::detail::execute_team(*core, x.data(), teamed.data(), fftn::team{.rank = 0, .size = 4}, ws,
                               fftn::completion::all, args);
    for (std::thread& m : members) {
      m.join();
    }
    EXPECT_TRUE(teamed == serial) << fftn::detail::describe(*core);
    // And the program agrees with a plan of the same edges without the team program.
    const auto plain = fftn::detail::make_plan_core_with_edges(
        fftn::detail::view_of(s), {}, fftn::detail::view_of(edges), fftn::detail::build_settings{});
    std::vector<C> other = init;
    const fftn::workspace ws1(fftn::detail::workspace_bytes(*plain, 1));
    fftn::detail::execute_core(*plain, 0, 2, x.data(), other.data(), ws1, args);
    fftn::test::outcome o;
    std::vector<fftn::test::exact_complex> want;
    for (const C& v : other) {
      want.push_back(fftn::test::exact_from(v));
    }
    o.within(fftn::test::compare<double>(std::span<const C>(serial), want),
             fftn::test::derived_tolerance<double>(points, fftn::test::fftn_bound(points)) * 2,
             "team program against the plain plan");
    EXPECT_TRUE(o.passed()) << o.detail();
  }
}

// Every row strategy, and the lanes over adjacent transforms, forced with edges (5.1b-2): the
// lanes over rows (c2c, r2c, c2r), the grid (c2c and a real row's half-length transform), the
// depth-first chain (c2c and real), Bluestein's depth-first convolution and batch leaves, with
// separate passes around them (the input edge's into the output, in place, or a real row's
// into the block; accumulation through the output's saved prior values).
template <class T> void forced_strategies(fftn::isa t, bool threads) {
  using enum transform_kind;
  using fftn::detail::build_settings;
  using fftn::detail::strategy_choice;
  build_settings lanes;
  lanes.strategy = strategy_choice::lanes;
  build_settings grid;
  grid.strategy = strategy_choice::grid;
  build_settings depth;
  depth.strategy = strategy_choice::depth;
  build_settings bluestein = depth;
  bluestein.rader = 0;
  bluestein.bluestein_length = 4096;
  build_settings leaves;
  leaves.batch_leaves = fftn::detail::batch_leaf_choice::own;
  build_settings adjacent;
  adjacent.profile.adjacent_lanes = true;
  adjacent.profile.adjacent_min_bytes = 0;
  adjacent.profile.adjacent_team_chain = false;
  adjacent.profile.column_team_bytes = 0;
  adjacent.profile.column_wide_bytes = 0;
  struct forced {
    shape_case c;
    const build_settings* settings;
    const char* strategy;
  };
  const char* row_lanes = "lanes (blocks of";
  const char* c2c_grid = "grid (Cooley-Tukey";
  const char* c2c_depth = "depth-first chain (the sub-rows";
  const char* half_grid = "grid for the half-length";
  const char* half_depth = "depth-first chain for the half-length";
  const std::vector<forced> cases{
      {{.name = "c2c lanes, both factors",
        .kind = c2c,
        .dims = {16},
        .count = 7,
        .in = {cplx, {}, 1},
        .out = {real, {1}, -1},
        .expect = "before the row's strategy"},
       &lanes,
       row_lanes},
      {{.name = "c2c lanes, accumulated",
        .kind = c2c,
        .dims = {12},
        .count = 9,
        .out_scale = 0.5,
        .in = {real, {-1}, 1},
        .accumulate = true,
        .gap = 1,
        .expect = "added to the saved prior values"},
       &lanes,
       row_lanes},
      {{.name = "c2c lanes in place",
        .kind = c2c,
        .dims = {20},
        .count = 6,
        .in = {cplx, {}, 0},
        .out = {cplx, {}, 1},
        .in_place = true,
        .expect = "after the row's strategy"},
       &lanes,
       row_lanes},
      {{.name = "r2c lanes, accumulated",
        .kind = r2c,
        .dims = {32},
        .count = 6,
        .in = {real, {}, 1},
        .out = {cplx, {-1}, 1},
        .accumulate = true,
        .expect = "before the row's strategy"},
       &lanes,
       row_lanes},
      {{.name = "c2r lanes, accumulated",
        .kind = c2r,
        .dims = {24},
        .count = 5,
        .in = {cplx, {}, -1},
        .out = {real, {}, 0},
        .accumulate = true,
        .expect = "before the row's strategy"},
       &lanes,
       row_lanes},
      {{.name = "c2c grid, accumulated",
        .kind = c2c,
        .dims = {1024},
        .count = 2,
        .in = {cplx, {}, 1},
        .out = {cplx, {-1}, 1},
        .accumulate = true,
        .expect = "added to the saved prior values"},
       &grid,
       c2c_grid},
      {{.name = "c2c grid in place",
        .kind = c2c,
        .dims = {2048},
        .count = 1,
        .options_scale = 0.25,
        .in = {real, {}, 1},
        .out = {cplx, {}, 1},
        .in_place = true,
        .expect = "after the row's strategy"},
       &grid,
       c2c_grid},
      {{.name = "c2c depth-first, both factors",
        .kind = c2c,
        .dims = {2048},
        .count = 2,
        .in = {cplx, {-1}, 1},
        .out = {real, {}, 1},
        .expect = "before the row's strategy"},
       &depth,
       c2c_depth},
      {{.name = "c2c depth-first, accumulated",
        .kind = c2c,
        .dims = {4096},
        .count = 1,
        .out = {cplx, {}, 1},
        .accumulate = true,
        .expect = "added to the saved prior values"},
       &depth,
       c2c_depth},
      {{.name = "Bluestein depth-first, accumulated",
        .kind = c2c,
        .dims = {1031},
        .count = 2,
        .in = {cplx, {}, 1},
        .out = {real, {}, 0},
        .accumulate = true,
        .expect = "added to the saved prior values"},
       &bluestein,
       "two depth-first chains"},
      {{.name = "r2c half-length grid, accumulated",
        .kind = r2c,
        .dims = {4096},
        .count = 2,
        .out = {cplx, {}, 1},
        .accumulate = true,
        .expect = "added to the saved prior values"},
       &grid,
       half_grid},
      {{.name = "r2c half-length grid, input factor",
        .kind = r2c,
        .dims = {4096},
        .count = 1,
        .in = {real, {}, 1},
        .expect = "before the row's strategy"},
       &grid,
       half_grid},
      {{.name = "c2r half-length grid, output factor",
        .kind = c2r,
        .dims = {4096},
        .count = 2,
        .out = {real, {}, 1},
        .accumulate = true,
        .expect = "added to the saved prior values"},
       &grid,
       half_grid},
      {{.name = "r2c half-length depth-first, both factors",
        .kind = r2c,
        .dims = {4096},
        .count = 2,
        .in = {real, {-1}, 1},
        .out = {cplx, {}, 1},
        .accumulate = true,
        .expect = "before the row's strategy"},
       &depth,
       half_depth},
      {{.name = "c2r half-length depth-first, output factor",
        .kind = c2r,
        .dims = {4096},
        .count = 1,
        .out = {real, {}, 1},
        .expect = "after the row's strategy"},
       &depth,
       half_depth},
      {{.name = "batch leaves, real factors",
        .kind = c2c,
        .dims = {8},
        .count = 16,
        .in = {real, {}, 1},
        .out = {real, {}, 0},
        .accumulate = true,
        .expect = "accumulated"},
       // A target whose precision has one complex value per register has no batch leaves
       // (scalar, double on NEON); where it has them, describe() says they run one row at a
       // time.
       &leaves,
       nullptr},
      {{.name = "lanes over adjacent transforms, both factors",
        .kind = c2c,
        .dims = {8, 12},
        .count = 8,
        .in = {cplx, {}, 1},
        .out = {real, {-1, 1}, 1},
        .expect = before,
        .adjacent = true},
       &adjacent,
       "adjacent transforms"},
      {{.name = "lanes over adjacent transforms in place",
        .kind = c2c,
        .dims = {6, 10},
        .count = 6,
        .in = {cplx, {1, -1}, 0},
        .out = {cplx, {}, 1},
        .in_place = true,
        .expect = after,
        .adjacent = true},
       &adjacent,
       "adjacent transforms"},
  };
  for (const forced& f : cases) {
    switch (f.c.kind) {
    case c2c:
      run_case<T, c2c>(f.c, t, threads, true, f.settings, f.strategy);
      break;
    case r2c:
      run_case<T, r2c>(f.c, t, threads, true, f.settings, f.strategy);
      break;
    case c2r:
      run_case<T, c2r>(f.c, t, threads, true, f.settings, f.strategy);
      break;
    }
  }
}

TYPED_TEST(EdgeShapes, EveryStrategyForced) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    forced_strategies<TypeParam>(t, false);
  }
  forced_strategies<TypeParam>(fftn::isa::best, true);
}

// Executions without a factor array a real or multi-axis plan needs end the program, in every
// build.
TEST(EdgeShapesDeathTest, MissingFactorArraysAbort) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using C = std::complex<double>;
  fftn::shape<C, C> plane;
  plane.axes = {{.n = 16}, {.n = 16}};
  const fftn::plan<C, C> p(plane, {}, {.out = {.mul = {.kind = real, .axis_strides = {16, 1}}}});
  const fftn::plan<double, C> r(fftn::rows_r2c<double>(32, 2), {},
                                {.in = {.mul = {.kind = real, .axis_strides = {1}}}});
  std::vector<C> x(256);
  std::vector<C> y(256);
  std::vector<double> xr(64);
  fftn::workspace ws(std::max(p.workspace_bytes(), r.workspace_bytes()));
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws), "without the factor arrays");
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws, {}), "without the factor arrays");
  EXPECT_DEATH(r.execute(xr.data(), y.data(), ws), "without the factor arrays");
  const std::vector<double> f(32);
  EXPECT_DEATH(r.execute(xr.data(), y.data(), ws, {.out_factor = f.data()}),
               "without the factor arrays");
}

} // namespace
