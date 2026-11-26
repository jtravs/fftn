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

#include "core/candidates.hpp"
#include "core/plan_core.hpp"
#include "core/wisdom_entry.hpp"
#include "fftn/detail/core.hpp"
#include "isa/kernel_table.hpp"
#include "planner/rules.hpp"
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
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
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
  dims_t dims; // the logical lengths in C order (the last contiguous)
  // shape.axes[i] is C axis listed[i] (a real transform's halved axis stays last); empty: C
  // order.
  std::vector<std::size_t> listed{};
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
  // A second, outer batch dimension of `outer` transforms of the inner `count`, outer_gap
  // elements past the inner ones; and (out of place) every stride and distance times `stride`,
  // the last (halved) axis included.
  std::size_t outer = 1;
  std::ptrdiff_t outer_gap = 0;
  std::ptrdiff_t stride = 1;
  // Out of place, the input's strides and distances times in_stride as well (only the input
  // strided).
  std::ptrdiff_t in_stride = 1;
};

std::size_t transforms(const shape_case& c) { return c.count * c.outer; }

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
  std::size_t span = 0;  // elements from element 0 of transform 0 to one past the last
  std::size_t inner = 1; // transforms of the inner batch dimension (b = o inner + i)
  std::ptrdiff_t outer_dist = 0;

  [[nodiscard]] std::size_t at(const dims_t& j, std::size_t b) const {
    std::ptrdiff_t i = static_cast<std::ptrdiff_t>(b % inner) * dist +
                       static_cast<std::ptrdiff_t>(b / inner) * outer_dist;
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
  std::size_t inner = 1;          // the inner batch dimension's transforms
  std::size_t outer = 1;          // the outer's (1: one batch dimension)
  std::ptrdiff_t outer_batch = 0; // the outer dimension's stride: the inner's times inner
  std::vector<T> reals;           // complex values as pairs
  std::ptrdiff_t zero = 0;

  factor_array(const factor_spec& f, const dims_t& d, std::size_t count,
               fftn::test::input_source& src, std::size_t outer_count = 1)
      : kind(f.kind), inner(count), outer(outer_count) {
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
    outer_batch = batch * static_cast<std::ptrdiff_t>(count);
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
    reach(outer_batch, outer);
    zero = -lo;
    const auto size = static_cast<std::size_t>(hi - lo + 1);
    reals = src.reals<T>(kind == cplx ? 2 * size : size);
  }

  [[nodiscard]] fftn::factor spec(const std::vector<std::size_t>& listed) const {
    if (kind == factor_kind::none) {
      return {};
    }
    fftn::factor f{.kind = kind, .axis_strides = {}, .batch_strides = {batch}};
    if (outer > 1) {
      f.batch_strides = {outer_batch, batch};
    }
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
    std::ptrdiff_t i = zero + static_cast<std::ptrdiff_t>(b % inner) * batch +
                       static_cast<std::ptrdiff_t>(b / inner) * outer_batch;
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
        fin(sc.in, in_dims, sc.count, src, sc.outer),
        fout(sc.out, out_dims, sc.count, src, sc.outer) {
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
      if (!c.in_place) {
        for (layout* l : {&lin, &lout}) {
          const std::ptrdiff_t times = c.stride * (l == &lin ? c.in_stride : 1);
          for (std::ptrdiff_t& st : l->stride) {
            st *= times;
          }
          l->dist *= times;
          l->span *= static_cast<std::size_t>(times);
        }
      }
    }
    // The outer batch dimension: whole inner batches, outer_gap apart (scaled as the rest;
    // in place both sides alike, the real side's gap twice the complex side's).
    for (layout* l : {&lin, &lout}) {
      const bool doubled = c.in_place && c.kind != transform_kind::c2c &&
                           ((l == &lin) == (c.kind == transform_kind::r2c));
      const std::ptrdiff_t gap = (doubled ? 2 : 1) * c.outer_gap * (c.in_place ? 1 : c.stride);
      l->inner = c.count;
      l->outer_dist = static_cast<std::ptrdiff_t>(l->span) + gap;
      l->span = static_cast<std::size_t>(l->outer_dist) * c.outer;
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
    if (c.outer > 1) {
      s.batches.insert(
          s.batches.begin(),
          fftn::batch{.count = c.outer, .in_dist = lin.outer_dist, .out_dist = lout.outer_dist});
    }
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
    for (std::size_t b = 0; b < transforms(c); ++b) {
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
  // A row's edge phases (the chain's) become separate passes where the rules take another
  // strategy for the row on this target (a real row's grid, or the lanes for short float rows
  // on x86): the case's wording for the phase, said of the row's strategy.
  std::string expect = sc.expect;
  if (describe.find("the row's strategy") != std::string::npos) {
    for (const auto& [phase, pass] : {std::pair{"in a separate phase before the transform",
                                                "in a separate pass before the row's strategy"},
                                      std::pair{"in a separate phase after the transform",
                                                "in a separate pass after the row's strategy"}}) {
      if (const std::size_t at = expect.find(phase); at != std::string::npos) {
        expect.replace(at, std::string(phase).size(), pass);
      }
    }
  }
  EXPECT_NE(describe.find(expect), std::string::npos) << what << ": no '" << expect << "' in\n"
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
  const std::size_t half = transforms(sc) / 2;
  p.execute_transforms(0, half, in_of(pieces), out_of(pieces), ws, args);
  p.execute_transforms(half, transforms(sc) - half, in_of(pieces), out_of(pieces), ws, args);
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
      {.name = "c2c volume listed out of order, two batch dimensions",
       .kind = c2c,
       .dims = {3, 4, 5},
       .listed = {2, 0, 1},
       .count = 2,
       .in = {cplx, {1, -1, 1}, 1},
       .out = {real, {}, -1},
       .accumulate = true,
       .gap = 1,
       .expect = after,
       .outer = 2,
       .outer_gap = 2,
       .stride = 2},
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
      {.name = "r2c volume, axes listed out of order",
       .kind = r2c,
       .dims = {4, 6, 8},
       .listed = {1, 0, 2},
       .count = 2,
       .in = {real, {1, -1, 1}, 1},
       .out = {cplx, {-1, 1, 1}, -1},
       .accumulate = true,
       .expect = after},
      {.name = "r2c plane, strided rows",
       .kind = r2c,
       .dims = {6, 16},
       .count = 2,
       .in = {real, {}, 1},
       .out = {cplx, {1, -1}, 1},
       .gap = 1,
       .expect = before,
       .stride = 2},
      {.name = "r2c plane, two batch dimensions",
       .kind = r2c,
       .dims = {5, 12},
       .count = 2,
       .in = {real, {1, -1}, 1},
       .out = {cplx, {}, -1},
       .accumulate = true,
       .gap = 2,
       .expect = after,
       .outer = 3,
       .outer_gap = 4},
      {.name = "r2c rows, two batch dimensions, strided",
       .kind = r2c,
       .dims = {18},
       .count = 3,
       .in = {real, {-1}, 1},
       .out = {cplx, {}, 1},
       .expect = before,
       .outer = 2,
       .outer_gap = 1,
       .stride = 3},
      {.name = "r2c volume in place, two batch dimensions",
       .kind = r2c,
       .dims = {4, 4, 8},
       .count = 2,
       .in = {real, {}, -1},
       .out = {cplx, {1, 1, -1}, 1},
       .in_place = true,
       .expect = before,
       .outer = 2,
       .outer_gap = 3},
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
      {.name = "c2r volume, axes listed out of order",
       .kind = c2r,
       .dims = {4, 6, 10},
       .listed = {1, 0, 2},
       .count = 2,
       .in = {cplx, {-1, 1, 1}, 1},
       .out = {real, {1, -1, 1}, -1},
       .accumulate = true,
       .expect = after},
      {.name = "c2r plane, strided rows",
       .kind = c2r,
       .dims = {6, 16},
       .count = 2,
       .in = {cplx, {}, 1},
       .out = {real, {1, -1}, 1},
       .expect = before,
       .stride = 3},
      {.name = "c2r plane, two batch dimensions",
       .kind = c2r,
       .dims = {6, 10},
       .count = 2,
       .in = {cplx, {1, -1}, -1},
       .out = {real, {}, 1},
       .accumulate = true,
       .gap = 1,
       .expect = after,
       .outer = 2,
       .outer_gap = 5},
      {.name = "c2r volume in place, two batch dimensions",
       .kind = c2r,
       .dims = {4, 4, 6},
       .count = 2,
       .in = {cplx, {}, 1},
       .out = {real, {1, -1, 1}, -1},
       .in_place = true,
       .expect = after,
       .outer = 2,
       .outer_gap = 3},
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

// The edge driver's batched forms (machine_profile::edge_batched_forms, core/edge_driver.hpp;
// task 5.2a, X65's P1): c2c rows of every kind of edge, as batch leaves, fused leaves, chains
// and Bluestein, contiguous, padded, strided, in place and under two batch dimensions, with
// factor strides of both signs along the line and the batch and broadcasts.
std::vector<shape_case> batched_row_cases() {
  using enum transform_kind;
  constexpr const char* any = "edges: the ";
  return {
      {.name = "codelet rows, both factors separate",
       .kind = c2c,
       .dims = {16},
       .count = 37,
       .in = {real, {}, 1},
       .out = {cplx, {-1}, 1},
       .expect = any},
      {.name = "codelet rows, output factor, padded",
       .kind = c2c,
       .dims = {64},
       .count = 21,
       .out = {real, {}, -1},
       .gap = 3,
       .expect = any},
      {.name = "codelet rows, input factor broadcast along the batch",
       .kind = c2c,
       .dims = {32},
       .count = 20,
       .dir = fftn::direction::backward,
       .in = {real, {-1}, 0},
       .expect = any},
      {.name = "codelet rows, accumulated with a factor and a scale",
       .kind = c2c,
       .dims = {8},
       .count = 300,
       .out_scale = 0.25,
       .in = {cplx, {-1}, -1},
       .accumulate = true,
       .gap = 1,
       .expect = any},
      {.name = "leaf rows, fused factor per row",
       .kind = c2c,
       .dims = {16},
       .count = 9,
       .in = {cplx, {}, 1},
       .expect = any},
      {.name = "leaf rows, fused factor broadcast",
       .kind = c2c,
       .dims = {8},
       .count = 11,
       .out = {cplx, {}, 0},
       .gap = 2,
       .expect = any},
      {.name = "leaf rows, fused factor, negative batch stride",
       .kind = c2c,
       .dims = {12},
       .count = 6,
       .out = {cplx, {}, -1},
       .expect = any},
      {.name = "chain rows, separate factors",
       .kind = c2c,
       .dims = {256},
       .count = 9,
       .in = {cplx, {-1}, 1},
       .out = {real, {}, -1},
       .expect = any},
      {.name = "chain rows, fused factors and a scale",
       .kind = c2c,
       .dims = {96},
       .count = 5,
       .options_scale = 0.5,
       .in = {cplx, {}, 1},
       .out = {cplx, {}, -1},
       .expect = any},
      {.name = "chain rows, accumulated, output factor",
       .kind = c2c,
       .dims = {1024},
       .count = 3,
       .out = {real, {1}, 1},
       .accumulate = true,
       .expect = any},
      {.name = "Bluestein rows, input factor",
       .kind = c2c,
       .dims = {67},
       .count = 4,
       .in = {real, {}, 1},
       .expect = any},
      {.name = "strided rows, factors",
       .kind = c2c,
       .dims = {16},
       .count = 10,
       .in = {cplx, {}, 1},
       .out = {real, {-1}, 1},
       .expect = any,
       .stride = 2},
      {.name = "rows in place, factors",
       .kind = c2c,
       .dims = {32},
       .count = 6,
       .in = {real, {}, 1},
       .out = {cplx, {-1}, -1},
       .in_place = true,
       .expect = any},
      {.name = "rows under two batch dimensions",
       .kind = c2c,
       .dims = {20},
       .count = 4,
       .out_scale = 2.0,
       .in = {cplx, {}, -1},
       .out = {real, {}, 1},
       .accumulate = true,
       .gap = 2,
       .expect = any,
       .outer = 3,
       .outer_gap = 5},
  };
}

// Which batched forms the plans of a run took (core/edge_driver.hpp, edge_forms).
struct forms_seen {
  std::size_t flat = 0;
  std::size_t leaf_rows = 0;
  std::size_t calls = 0;
  std::size_t row_blocks = 0;
};

// The plan of a case with the batched forms against the plan without them (the walked form,
// which the tests above check against the reference): both executed serially from the same
// buffers, bitwise alike; the walked plan's own forms empty, the batched one's workspace at
// least as large.
template <class T, transform_kind K>
void forms_agree(const shape_case& sc, fftn::isa target, forms_seen& seen) {
  using C = std::complex<T>;
  using In = std::conditional_t<K == transform_kind::r2c, T, C>;
  using Out = std::conditional_t<K == transform_kind::c2r, T, C>;
  namespace d = fftn::detail;
  ASSERT_EQ(sc.kind, K);
  fftn::test::input_source src(0xB47C5ULL + product(sc.dims) * 7 + sc.count);
  const shape_run<T> run(sc, target, src);
  d::build_settings walked;
  walked.profile.edge_batched_forms = false;
  d::build_settings batched;
  batched.profile.edge_batched_forms = true;
  const auto make = [&](const d::build_settings& s) {
    return d::make_plan_core_with_edges(d::view_of(run.template shape<In, Out>()), run.opts,
                                        d::view_of(run.template edges<In, Out>()), s);
  };
  const auto w = make(walked);
  const auto b = make(batched);
  const std::string what = std::string(sc.name) + ", " + fftn::isa_name(target) +
                           (sizeof(T) == 8 ? " double" : " float");
  EXPECT_TRUE(w->edge_batched.flat.empty() && w->edge_batched.calls.empty() &&
              w->edge_row_block == 0)
      << what;
  EXPECT_GE(b->bytes, w->bytes) << what;
  if (b->needs_args) {
    const d::edge_forms& f = b->edge_batched;
    seen.flat += f.flat.empty() ? 0U : 1U;
    seen.leaf_rows += f.leaf_rows ? 1U : 0U;
    seen.calls += f.calls.empty() ? 0U : 1U;
    seen.row_blocks += f.row_block != 0 ? 1U : 0U;
    // A c2c row's chain with separate passes and one batch dimension takes the row-block form.
    const std::string text = d::describe(*b);
    if (K == transform_kind::c2c && sc.dims.size() == 1 && sc.outer == 1 && sc.count >= 2 &&
        text.find("separate pass into the workspace block") != std::string::npos) {
      EXPECT_NE(f.row_block, 0U) << what << ":\n" << text;
      EXPECT_NE(text.find("the rows in blocks of"), std::string::npos) << what << ":\n" << text;
    }
  }
  std::vector<T> xs = run.x;
  const auto fresh = [&] { return sc.in_place ? run.x : run.init; };
  const auto in_of = [&](std::vector<T>& y) {
    return reinterpret_cast<const In*>(sc.in_place ? y.data() : xs.data());
  };
  const auto out_of = [](std::vector<T>& y) { return reinterpret_cast<Out*>(y.data()); };
  const fftn::exec_args args = run.args();
  fftn::workspace ws(std::max(b->bytes, w->bytes));
  std::vector<T> one = fresh();
  d::execute_all(*w, in_of(one), out_of(one), ws, args);
  std::vector<T> other = fresh();
  d::execute_all(*b, in_of(other), out_of(other), ws, args);
  EXPECT_TRUE(same_bits(one, other)) << what << ": the batched forms differ from the walked form\n"
                                     << d::describe(*b);
  // Pieces of the batch that start and end inside a block of rows.
  if (transforms(sc) > 3) {
    std::vector<T> pieces = fresh();
    const std::size_t n = transforms(sc);
    for (const auto& [first, count] :
         {std::pair{std::size_t{0}, std::size_t{1}}, std::pair{std::size_t{1}, n / 2},
          std::pair{1 + n / 2, n - 1 - n / 2}}) {
      d::execute_core(*b, first, count, in_of(pieces), out_of(pieces), ws, args);
    }
    EXPECT_TRUE(same_bits(other, pieces)) << what << ": pieces of the batch differ";
  }
}

TYPED_TEST(EdgeShapes, BatchedFormsAgreeWithTheWalkedForm) {
  forms_seen seen;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const shape_case& c : batched_row_cases()) {
      forms_agree<TypeParam, transform_kind::c2c>(c, t, seen);
    }
    for (const shape_case& c : c2c_cases()) {
      forms_agree<TypeParam, transform_kind::c2c>(c, t, seen);
    }
    for (const shape_case& c : r2c_cases()) {
      forms_agree<TypeParam, transform_kind::r2c>(c, t, seen);
    }
    for (const shape_case& c : c2r_cases()) {
      forms_agree<TypeParam, transform_kind::c2r>(c, t, seen);
    }
  }
  // Every form was taken somewhere.
  EXPECT_GT(seen.flat, 0U);
  EXPECT_GT(seen.leaf_rows, 0U);
  EXPECT_GT(seen.calls, 0U);
  EXPECT_GT(seen.row_blocks, 0U);
}

// The batched forms against the reference, with canaries in the rows' gaps, and serial, pieces,
// teams and an executor bitwise alike (run_case), on every instruction set for the rows and on
// the best for the planes and volumes.
TYPED_TEST(EdgeShapes, BatchedFormsAgainstTheReference) {
  fftn::detail::build_settings batched;
  batched.profile.edge_batched_forms = true;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const shape_case& c : batched_row_cases()) {
      run_case<TypeParam, transform_kind::c2c>(c, t, true, true, &batched);
    }
  }
  for (const shape_case& c : c2c_cases()) {
    run_case<TypeParam, transform_kind::c2c>(c, fftn::isa::best, true, true, &batched);
  }
  for (const shape_case& c : r2c_cases()) {
    run_case<TypeParam, transform_kind::r2c>(c, fftn::isa::best, true, true, &batched);
  }
  for (const shape_case& c : c2r_cases()) {
    run_case<TypeParam, transform_kind::c2r>(c, fftn::isa::best, true, true, &batched);
  }
}

// Batch leaves with separate edge passes, which the walked form runs a row at a time
// (DescribeNamesTheStrategyAnEdgeForces): in the batched forms the plan keeps the strategy of the
// plan without the edges, and describe() names the row blocks and no forced strategy.
TYPED_TEST(EdgeShapes, BatchedFormsTakeBatchLeaves) {
  namespace d = fftn::detail;
  using C = std::complex<TypeParam>;
  d::build_settings leaves;
  leaves.batch_leaves = d::batch_leaf_choice::own;
  leaves.strategy = d::strategy_choice::chain;
  leaves.profile.edge_batched_forms = true;
  const std::vector<shape_case> cases{
      {.name = "batch leaves, real factors",
       .kind = transform_kind::c2c,
       .dims = {8},
       .count = 40,
       .in = {real, {}, 1},
       .out = {real, {-1}, -1},
       .expect = "edge row blocks: the rows in blocks of"},
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const shape_case& c : cases) {
      fftn::test::input_source src(0x5C2ULL);
      const shape_run<TypeParam> run(c, t, src);
      const fftn::shape<C, C> s = run.template shape<C, C>();
      const auto plain = d::build_plan_core(d::view_of(s), run.opts, leaves);
      if (plain->strategy_name != "the chain on batch leaves") {
        continue; // a target without batch leaves in this precision
      }
      const auto core = d::make_plan_core_with_edges(
          d::view_of(s), run.opts, d::view_of(run.template edges<C, C>()), leaves);
      const std::string text = d::describe(*core);
      EXPECT_EQ(core->strategy_name, plain->strategy_name) << text;
      EXPECT_EQ(text.find("not the strategy"), std::string::npos) << text;
      EXPECT_NE(core->edge_row_block, 0U) << text;
      run_case<TypeParam, transform_kind::c2c>(c, t, true, true, &leaves, "batch leaf radix-8");
    }
  }
}

TEST(EdgeShapesDeathTest, BatchedFormsMissingFactorArraysAbort) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using C = std::complex<double>;
  namespace d = fftn::detail;
  d::build_settings batched;
  batched.profile.edge_batched_forms = true;
  const fftn::shape<C, C> rows = fftn::rows_c2c<double>(16, 8, fftn::direction::forward);
  const fftn::plan_edges<C, C> real_out{.out = {.mul = {.kind = real, .axis_strides = {1}}}};
  const fftn::plan_edges<C, C> fused_in{.in = {.mul = {.kind = cplx, .axis_strides = {1}}}};
  for (const auto* e : {&real_out, &fused_in}) {
    const auto core = d::make_plan_core_with_edges(d::view_of(rows), {}, d::view_of(*e), batched);
    std::vector<C> x(128);
    std::vector<C> y(128);
    fftn::workspace ws(core->bytes);
    EXPECT_DEATH(d::execute_all(*core, x.data(), y.data(), ws, {}), "without the factor arrays");
    EXPECT_DEATH(d::execute_all(*core, x.data(), y.data(), ws), "without the factor arrays");
  }
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
        .dims = {1024},
        .count = 2,
        .in = {cplx, {-1}, 1},
        .out = {real, {}, 1},
        .expect = "before the row's strategy"},
       &depth,
       c2c_depth},
      {{.name = "c2c depth-first, accumulated",
        .kind = c2c,
        .dims = {2048},
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
        .dims = {2048},
        .count = 2,
        .out = {cplx, {}, 1},
        .accumulate = true,
        .expect = "added to the saved prior values"},
       &grid,
       half_grid},
      {{.name = "r2c half-length grid, input factor",
        .kind = r2c,
        .dims = {2048},
        .count = 1,
        .in = {real, {}, 1},
        .expect = "before the row's strategy"},
       &grid,
       half_grid},
      {{.name = "c2r half-length grid, output factor",
        .kind = c2r,
        .dims = {2048},
        .count = 2,
        .out = {real, {}, 1},
        .accumulate = true,
        .expect = "added to the saved prior values"},
       &grid,
       half_grid},
      {{.name = "r2c half-length depth-first, both factors",
        .kind = r2c,
        .dims = {2048},
        .count = 2,
        .in = {real, {-1}, 1},
        .out = {cplx, {}, 1},
        .accumulate = true,
        .expect = "before the row's strategy"},
       &depth,
       half_depth},
      {{.name = "c2r half-length depth-first, output factor",
        .kind = c2r,
        .dims = {2048},
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
      // Rows one element apart along the batch (column-major rows): the lanes over adjacent
      // transforms on a plan of one axis, whose input edge's phase must write the output that
      // the lanes then read.
      // The rows are longer than any target's lanes over rows take (lanes_up_to: 100 on
      // avx512 in float), which would otherwise win.
      {{.name = "rows as adjacent transforms, input factor",
        .kind = c2c,
        .dims = {128},
        .count = 32,
        .in = {real, {}, 1},
        .expect = before,
        .adjacent = true},
       &adjacent,
       "adjacent transforms"},
      {{.name = "rows as adjacent transforms, both factors",
        .kind = c2c,
        .dims = {112},
        .count = 16,
        .in = {cplx, {-1}, -1},
        .out = {cplx, {}, 1},
        .expect = after,
        .adjacent = true},
       &adjacent,
       "adjacent transforms"},
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

// A plan whose edges keep it from the strategy of the plan without them (task 5.1c): the case,
// the knobs that make the plain plan take that strategy, the start of its name
// (plan_core::strategy_name) and the reason describe() must give.
struct forced_line {
  shape_case c;
  fftn::detail::build_settings settings;
  const char* plain;
  const char* reason;
};

// Whether the plain plan of `f` takes the strategy f.plain on target t; if so, describe() of the
// plan with the edges names it, the strategy taken instead and the reason, and the plan computes
// the transform (run_case).
template <class T, transform_kind K> bool forced_line_holds(const forced_line& f, fftn::isa t) {
  namespace d = fftn::detail;
  using C = std::complex<T>;
  using In = std::conditional_t<K == transform_kind::r2c, T, C>;
  using Out = std::conditional_t<K == transform_kind::c2r, T, C>;
  fftn::test::input_source src(0x5C1ULL);
  const shape_run<T> run(f.c, t, src);
  const fftn::shape<In, Out> s = run.template shape<In, Out>();
  const fftn::plan_edges<In, Out> e = run.template edges<In, Out>();
  // The walked form's reasons: in the batched forms, batch leaves take separate passes a block
  // of rows at a time (BatchedFormsTakeBatchLeaves).
  d::build_settings settings = f.settings;
  settings.profile.edge_batched_forms = false;
  const auto plain = d::build_plan_core(d::view_of(s), run.opts, settings);
  if (!plain->strategy_name.starts_with(f.plain)) {
    return false;
  }
  const auto core = d::make_plan_core_with_edges(d::view_of(s), run.opts, d::view_of(e), settings);
  const std::string text = d::describe(*core);
  const std::string what =
      std::string(f.c.name) + ", " + fftn::isa_name(t) + (sizeof(T) == 8 ? " double" : " float");
  EXPECT_NE(core->strategy_name, plain->strategy_name) << what << "\n" << text;
  const std::string line = "\nnot the strategy the rules would choose: " + plain->strategy_name +
                           " without the edges, " + core->strategy_name +
                           " with them: " + f.reason + "\n";
  EXPECT_NE(text.find(line), std::string::npos) << what << ": no '" << line << "' in\n" << text;
  run_case<T, K>(f.c, t, false, true, &settings, f.reason);
  return true;
}

// Each case in which an edge keeps a plan from the strategy the rules (or a test's knobs) give
// the same plan without edges: describe() says which, which it takes instead and why (task
// 5.1c). The strategies come from knobs and explicit profile fields, so the cases hold on every
// target, except batch leaves, which only some targets' precisions have.
TYPED_TEST(EdgeShapes, DescribeNamesTheStrategyAnEdgeForces) {
  using T = TypeParam;
  using enum transform_kind;
  namespace d = fftn::detail;
  d::build_settings adjacent;
  adjacent.profile.adjacent_lanes = true;
  adjacent.profile.adjacent_min_bytes = 0;
  adjacent.profile.adjacent_team_chain = false;
  adjacent.profile.column_team_bytes = 0;
  adjacent.profile.column_wide_bytes = 0;
  d::build_settings grid;
  grid.strategy = d::strategy_choice::grid;
  d::build_settings depth;
  depth.strategy = d::strategy_choice::depth;
  d::build_settings leaves;
  leaves.batch_leaves = d::batch_leaf_choice::own;
  d::build_settings packed;
  packed.c2r_packed = 1;
  const std::vector<forced_line> every_target{
      {{.name = "rows as adjacent transforms, accumulated",
        .kind = c2c,
        .dims = {128},
        .count = 32,
        .in = {real, {}, 1},
        .accumulate = true,
        .adjacent = true},
       adjacent,
       "the lanes over adjacent transforms",
       "accumulation takes no lanes over adjacent transforms"},
      {{.name = "planes as adjacent transforms, accumulated",
        .kind = c2c,
        .dims = {8, 12},
        .count = 8,
        .out = {cplx, {}, 1},
        .accumulate = true,
        .adjacent = true},
       adjacent,
       "the lanes over adjacent transforms",
       "accumulation takes no lanes over adjacent transforms"},
      {{.name = "c2r half-length grid, input factor",
        .kind = c2r,
        .dims = {2048},
        .count = 2,
        .in = {cplx, {}, 1}},
       grid,
       "the half-length grid",
       "a real row's input edge takes no grid here"},
      {{.name = "c2r half-length depth-first, input factor",
        .kind = c2r,
        .dims = {2048},
        .count = 1,
        .in = {real, {-1}, 1},
        .out = {real, {}, 1}},
       depth,
       "the half-length depth-first chain",
       "a c2r row's input edge takes no depth-first chain"},
      {{.name = "c2r plane packed, output factor",
        .kind = c2r,
        .dims = {16, 32},
        .count = 2,
        .out = {real, {1, -1}, 1}},
       packed,
       "each axis in turn, the intermediate packed in the output",
       "an edge phase takes no intermediate packed in the output"},
  };
  // These take batch leaves, which only some targets' precisions have (and the last one also
  // the rules' lanes over rows, x86 in float): each must hold on every target whose tables say
  // it can, so that none goes untested where it applies.
  d::build_settings chain_leaves = leaves;
  chain_leaves.strategy = d::strategy_choice::chain;
  struct batch_line {
    forced_line f;
    bool rules_lanes; // the plain plan is the rules' lanes over rows
  };
  const std::vector<batch_line> batch_targets{
      {{{.name = "batch leaves, real factor",
         .kind = c2c,
         .dims = {8},
         .count = 16,
         .in = {real, {}, 1}},
        leaves,
        "the chain on batch leaves",
        "batch leaves, one row at a time between the edge passes"},
       false},
      {{{.name = "batch leaves, fused complex factor",
         .kind = c2c,
         .dims = {16},
         .count = 16,
         .out = {cplx, {}, 1}},
        leaves,
        "the chain on batch leaves",
        "the chain with a fused factor takes no batch leaves"},
       false},
      // The input strided, the output not: the input edge's pass moves the rows into the
      // output, where batch leaves are found that the plain plan's strided rows cannot take.
      // The move is the cause, not batch leaves (which the plain plan does not have).
      {{{.name = "strided input, batch leaves after the input edge's pass",
         .kind = c2c,
         .dims = {8},
         .count = 16,
         .in = {real, {}, 1},
         .in_stride = 2},
        chain_leaves,
        "the chain",
        "the input edge's pass leaves the rows in the output, where the strategy is chosen in "
        "place"},
       false},
      // Likewise with a fused input factor: the separate passes' plan is then the chain on
      // batch leaves, so the plan is the fused chain, which has no input pass.
      {{{.name = "strided input, lanes, fused input factor",
         .kind = c2c,
         .dims = {16},
         .count = 16,
         .in = {cplx, {}, 1},
         .in_stride = 2},
        leaves,
        "the lanes over rows",
        "the strategy is chosen for the rows where an input edge's pass would leave them, the "
        "chain, which takes the factor fused instead"},
       true},
  };
  const auto holds = [](const forced_line& f, fftn::isa t) {
    switch (f.c.kind) {
    case c2c:
      return forced_line_holds<T, c2c>(f, t);
    case r2c:
      return forced_line_holds<T, r2c>(f, t);
    case c2r:
      break;
    }
    return forced_line_holds<T, c2r>(f, t);
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const forced_line& f : every_target) {
      EXPECT_TRUE(holds(f, t)) << f.c.name << ", " << fftn::isa_name(t)
                               << ": the plain plan takes another strategy";
    }
    const auto target = static_cast<d::target>(static_cast<unsigned char>(t));
    for (const batch_line& b : batch_targets) {
      // The target's own batch leaves of the row's length in this precision (batch_leaves_of).
      const std::size_t n = b.f.c.dims[0];
      const d::batch_leaf_table* table = d::find_batch_leaves(target, n);
      const std::size_t lanes = table != nullptr ? d::of<T>(*table).lanes : 1;
      const bool applies =
          lanes > 1 && (!b.rules_lanes || n <= d::profile_of(target, lanes).lanes_up_to);
      if (applies) {
        EXPECT_TRUE(holds(b.f, t))
            << b.f.c.name << ", " << fftn::isa_name(t) << ": the plain plan takes another strategy";
      }
    }
  }
}

// The separate passes an edge costs, as describe() counts them (task 5.1c): none for a fused
// factor, one per separate factor or accumulation, and one more where accumulation saves the
// output's prior values before a strategy that writes the output.
TYPED_TEST(EdgeShapes, DescribeCountsTheSeparatePasses) {
  using T = TypeParam;
  using enum transform_kind;
  namespace d = fftn::detail;
  // The walked form's placements: in the batched forms, batch leaves (avx2's rules for 64-point
  // rows) take a fusable factor as a separate pass (BatchedFormsTakeBatchLeaves).
  d::build_settings chain;
  chain.strategy = d::strategy_choice::chain;
  chain.profile.edge_batched_forms = false;
  d::build_settings lanes;
  lanes.strategy = d::strategy_choice::lanes;
  lanes.profile.edge_batched_forms = false;
  struct counted {
    shape_case c;
    const d::build_settings* settings;
    const char* cost;
  };
  const std::vector<counted> cases{
      {{.name = "c2c chain, fused factors",
        .kind = c2c,
        .dims = {64},
        .count = 4,
        .in = {cplx, {}, 1},
        .out = {cplx, {}, 0}},
       &chain,
       "; they cost no separate pass\n"},
      {{.name = "c2c chain, real input factor",
        .kind = c2c,
        .dims = {64},
        .count = 4,
        .in = {real, {}, 1}},
       &chain,
       "; they cost 1 separate pass\n"},
      {{.name = "c2c lanes, both factors",
        .kind = c2c,
        .dims = {16},
        .count = 7,
        .in = {cplx, {}, 1},
        .out = {real, {1}, -1}},
       &lanes,
       "; they cost 2 separate passes\n"},
      {{.name = "c2c lanes, input factor, accumulated",
        .kind = c2c,
        .dims = {12},
        .count = 9,
        .in = {real, {-1}, 1},
        .accumulate = true},
       &lanes,
       "the output's prior values saved in a pass before it; they cost 3 separate passes\n"},
      {{.name = "plane, both factors, accumulated",
        .kind = c2c,
        .dims = {12, 10},
        .count = 3,
        .in = {cplx, {}, 1},
        .out = {real, {1, -1}, 0},
        .accumulate = true},
       nullptr,
       "; they cost 2 separate passes\n"},
  };
  d::build_settings rules;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const counted& c : cases) {
      run_case<T, c2c>(c.c, t, false, true, c.settings != nullptr ? c.settings : &rules, c.cost);
    }
  }
}

// A stand-in for measured planning's clock (build_settings::measure_time): the first candidate
// asked for after the rules' choice is twice as fast as every other.
struct first_wins {
  std::string winner;
  std::vector<std::string> asked;
};

double first_wins_time(const std::string& choice, void* context) {
  auto& f = *static_cast<first_wins*>(context);
  f.asked.push_back(choice);
  if (f.winner.empty() && choice != "rules") {
    f.winner = choice;
  }
  return choice == f.winner ? 0.5 : 1.0;
}

// Measured planning and wisdom with edges (task 5.1b-3): the candidates are built with the edges
// (a strategy forced by a candidate takes them as separate passes), the plan kept carries them
// and computes the transform, and its wisdom entry rebuilds the same program, bit for bit,
// for the same edges only.
TEST(EdgeShapes, MeasuredPlanningAndWisdom) {
  using C = std::complex<double>;
  namespace d = fftn::detail;
  const std::size_t n = 4096;
  const auto shape = fftn::rows_c2c<double>(n, 2, fftn::direction::forward);
  const auto ni = static_cast<std::ptrdiff_t>(n);
  const fftn::plan_edges<C, C> edges{
      .in = {.mul = {.kind = real, .axis_strides = {1}, .batch_strides = {ni}}},
      .out = {.scale = 0.5, .mul = {.kind = cplx, .axis_strides = {-1}}, .accumulate = true}};
  first_wins clock;
  d::build_settings settings;
  settings.measure_time = &first_wins_time;
  settings.measure_context = &clock;
  const auto core = d::make_plan_core_with_edges(
      d::view_of(shape), {.effort = fftn::planning::measure}, d::view_of(edges), settings);
  const std::string text = d::describe(*core);
  ASSERT_FALSE(clock.winner.empty()) << text;
  EXPECT_EQ(clock.asked.front(), "rules");
  // The winner, not the rules' choice (its text pinned to the program built: a grid's phases
  // as its factors, say, so not necessarily the text the clock was asked about).
  EXPECT_NE(core->chosen_rank, 0U) << text;
  EXPECT_DOUBLE_EQ(core->rules_over_chosen, 2.0) << text;
  EXPECT_NE(text.find("planning: measured"), std::string::npos) << text;
  EXPECT_NE(text.find("edges: the input times a real factor"), std::string::npos) << text;
  EXPECT_TRUE(core->needs_args);

  // The transform: against the rules' plan of the same edges.
  fftn::test::input_source src(0x3EA5ULL);
  const auto x = src.complexes<double>(2 * n);
  const auto fin = src.reals<double>(2 * n);
  const auto fout_storage = src.complexes<double>(n);
  const C* fout = fout_storage.data() + (n - 1);
  const auto init = src.complexes<double>(2 * n);
  const fftn::exec_args args{.in_factor = fin.data(), .out_factor = fout};
  const auto rules =
      d::make_plan_core_with_edges(d::view_of(shape), {}, d::view_of(edges), d::build_settings{});
  const auto run = [&](const d::plan_core& c) {
    std::vector<C> y = init;
    const fftn::workspace ws(d::workspace_bytes(c, 1));
    d::execute_core(c, 0, 2, x.data(), y.data(), ws, args);
    return y;
  };
  const std::vector<C> measured = run(*core);
  std::vector<fftn::test::exact_complex> want;
  for (const C& v : run(*rules)) {
    want.push_back(fftn::test::exact_from(v));
  }
  fftn::test::outcome o;
  o.within(fftn::test::compare<double>(std::span<const C>(measured), want),
           fftn::test::derived_tolerance<double>(n, fftn::test::fftn_bound(n)) * 2,
           "measured against the rules' plan");
  EXPECT_TRUE(o.passed()) << o.detail() << text;

  // Wisdom: the entry rebuilds the program with the same edges, nothing timed, bit for bit.
  const fftn::wisdom w(d::record_text(d::record_of(*core)) + "\n");
  first_wins unused;
  d::build_settings again = settings;
  again.measure_context = &unused;
  for (const fftn::planning effort : {fftn::planning::rules, fftn::planning::measure}) {
    const auto rebuilt = d::make_plan_core_with_edges(
        d::view_of(shape), {.effort = effort, .use = &w}, d::view_of(edges), again);
    EXPECT_TRUE(unused.asked.empty());
    EXPECT_EQ(d::program_fingerprint(*rebuilt), d::program_fingerprint(*core));
    EXPECT_NE(d::describe(*rebuilt).find("planning: from wisdom"), std::string::npos)
        << d::describe(*rebuilt);
    EXPECT_TRUE(run(*rebuilt) == measured);
  }
  // Not for the plan without edges, nor for other edges.
  const auto plain = d::make_plan_core(d::view_of(shape), {.use = &w}, again);
  EXPECT_EQ(d::describe(*plain).find("from wisdom"), std::string::npos);
  fftn::plan_edges<C, C> other = edges;
  other.in.mul.kind = cplx;
  const auto different =
      d::make_plan_core_with_edges(d::view_of(shape), {.use = &w}, d::view_of(other), again);
  EXPECT_EQ(d::describe(*different).find("from wisdom"), std::string::npos);
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

// Stored runs (shape::runs, DESIGN.md 12.1; task 5.1d). A case: a kind, the logical lengths
// in C order, the runs (their axes C-ordered indices), and the layout's gap after each
// transform; the shape lists the axes in C order with default strides, which span each axis's
// stored extent on a side with runs there, or (`listed` not empty) C axis listed[i] as its axis
// i, with those strides given, so that the plan orders the axes back (order_axes).
struct runs_case {
  const char* name = "";
  transform_kind kind = transform_kind::c2c;
  dims_t dims{};
  std::size_t count = 1;
  std::vector<fftn::stored_run> runs{};
  fftn::direction dir = fftn::direction::forward;
  double scale = 1.0;
  bool accumulate = false;
  std::ptrdiff_t gap = 0;
  std::vector<std::size_t> listed{};
};

// One side of a run case: its logical lengths, stored extents and tight strides by them, and
// which logical positions its runs hold.
struct run_side {
  dims_t logical;
  dims_t stored;
  strides_t stride;
  std::ptrdiff_t dist = 0;
  std::size_t span = 0;
  std::vector<std::vector<fftn::stored_run>> runs; // per axis

  run_side(const runs_case& c, bool output)
      : logical(side_dims(shape_case{.kind = c.kind, .dims = c.dims}, output)) {
    const std::size_t r = logical.size();
    runs.resize(r);
    stored = logical;
    for (const fftn::stored_run& run : c.runs) {
      if ((run.where == fftn::side::out) == output) {
        runs[run.axis].push_back(run);
      }
    }
    for (std::size_t a = 0; a < r; ++a) {
      if (!runs[a].empty()) {
        stored[a] = 0;
        for (const fftn::stored_run& run : runs[a]) {
          stored[a] = std::max(stored[a], run.stored + run.count);
        }
      }
    }
    stride.assign(r, 1);
    for (std::size_t a = r - 1; a > 0; --a) {
      stride[a - 1] = stride[a] * static_cast<std::ptrdiff_t>(stored[a]);
    }
    dist = stride[0] * static_cast<std::ptrdiff_t>(stored[0]) + c.gap;
    span = static_cast<std::size_t>(dist) * c.count;
  }

  // The stored index of logical position j of transform b, or none when a run does not hold it.
  [[nodiscard]] std::optional<std::size_t> at(const dims_t& j, std::size_t b) const {
    std::ptrdiff_t i = static_cast<std::ptrdiff_t>(b) * dist;
    for (std::size_t a = 0; a < j.size(); ++a) {
      std::size_t pos = j[a];
      if (!runs[a].empty()) {
        const auto it = std::ranges::find_if(runs[a], [&](const fftn::stored_run& run) {
          return j[a] >= run.logical && j[a] < run.logical + run.count;
        });
        if (it == runs[a].end()) {
          return std::nullopt;
        }
        pos = it->stored + (j[a] - it->logical);
      }
      i += static_cast<std::ptrdiff_t>(pos) * stride[a];
    }
    return static_cast<std::size_t>(i);
  }
};

// The plan of a run case, executed serially, in two pieces, by teams of three (completion all
// and own) and on a pool of three, from the same buffers: all bitwise alike; the serial result
// against the reference of the logical input (zeros outside the input's runs) at the output's
// runs; every input element outside the runs a NaN that must not be read; no output element
// outside the runs written; the input unwritten.
template <class T, transform_kind K> void run_runs_case(const runs_case& c, fftn::isa target) {
  using C = std::complex<T>;
  using In = std::conditional_t<K == transform_kind::r2c, T, C>;
  using Out = std::conditional_t<K == transform_kind::c2r, T, C>;
  ASSERT_EQ(c.kind, K);
  const std::string what =
      std::string(c.name) + ", " + fftn::isa_name(target) + (sizeof(T) == 8 ? " double" : " float");
  SCOPED_TRACE(what);
  const run_side in(c, false);
  const run_side out(c, true);
  const std::size_t win = c.kind == transform_kind::r2c ? 1 : 2;
  const std::size_t wout = c.kind == transform_kind::c2r ? 1 : 2;
  fftn::test::input_source src(0x9A5ULL + product(c.dims) * 3 + c.runs.size());
  // The input: NaN where no run holds a stored element (never to be read).
  std::vector<T> x(in.span * win, std::numeric_limits<T>::quiet_NaN());
  for (std::size_t b = 0; b < c.count; ++b) {
    each(in.logical, [&](const dims_t& j) {
      if (const auto i = in.at(j, b)) {
        for (std::size_t k = 0; k < win; ++k) {
          x[*i * win + k] = static_cast<T>(src.uniform());
        }
      }
    });
  }
  std::vector<T> init = src.reals<T>(out.span * wout);

  fftn::shape<In, Out> s;
  s.dir = c.dir;
  if (c.listed.empty()) {
    for (const std::size_t n : c.dims) {
      s.axes.push_back({.n = n});
    }
    s.batches = {{.count = c.count,
                  .in_dist = c.gap != 0 ? in.dist : 0,
                  .out_dist = c.gap != 0 ? out.dist : 0}};
    s.runs = c.runs;
  } else {
    for (const std::size_t a : c.listed) {
      s.axes.push_back({.n = c.dims[a], .in_stride = in.stride[a], .out_stride = out.stride[a]});
    }
    s.batches = {{.count = c.count, .in_dist = in.dist, .out_dist = out.dist}};
    for (fftn::stored_run run : c.runs) {
      run.axis = static_cast<std::size_t>(std::ranges::find(c.listed, run.axis) - c.listed.begin());
      s.runs.push_back(run);
    }
  }
  fftn::plan_edges<In, Out> edges;
  edges.out.accumulate = c.accumulate;
  const fftn::plan<In, Out> p(s, {.scale = c.scale, .force_isa = target}, edges);
  const std::string text = p.describe();
  EXPECT_NE(text.find("stored runs:"), std::string::npos) << text;

  const std::vector<T> x0 = x;
  fftn::workspace ws(p.workspace_bytes(3));
  std::vector<T> serial = init;
  p.execute(reinterpret_cast<const In*>(x.data()), reinterpret_cast<Out*>(serial.data()), ws);
  // NaN never equals itself, so the input is compared bit for bit.
  EXPECT_TRUE(same_bits(x, x0)) << "input written";

  fftn::test::outcome o;
  std::vector<bool> described(serial.size(), false);
  const std::size_t points = product(c.dims);
  const shape_case as_case{.kind = c.kind, .dims = c.dims, .dir = c.dir};
  for (std::size_t b = 0; b < c.count; ++b) {
    std::vector<cd> xin(product(in.logical));
    each(in.logical, [&](const dims_t& j) {
      if (const auto i = in.at(j, b)) {
        xin[flat(in.logical, j)] =
            win == 1 ? cd(static_cast<double>(x[*i]), 0.0)
                     : cd(static_cast<double>(x[2 * *i]), static_cast<double>(x[2 * *i + 1]));
      }
    });
    const std::vector<cd> exact = reference(as_case, xin);
    std::vector<fftn::test::exact_complex> want;
    std::vector<C> got;
    each(out.logical, [&](const dims_t& j) {
      const auto i = out.at(j, b);
      if (!i) {
        return;
      }
      cd v = exact[flat(out.logical, j)] * c.scale;
      const cd prior =
          wout == 1 ? cd(static_cast<double>(init[*i]), 0.0)
                    : cd(static_cast<double>(init[2 * *i]), static_cast<double>(init[2 * *i + 1]));
      if (c.accumulate) {
        v += prior;
      }
      want.push_back(fftn::test::exact_from(v));
      const cd g = wout == 1 ? cd(static_cast<double>(serial[*i]), 0.0)
                             : cd(static_cast<double>(serial[2 * *i]),
                                  static_cast<double>(serial[2 * *i + 1]));
      got.emplace_back(static_cast<T>(g.real()), static_cast<T>(g.imag()));
      for (std::size_t k = 0; k < wout; ++k) {
        described[*i * wout + k] = true;
      }
    });
    const auto tol = fftn::test::derived_tolerance<T>(points, fftn::test::fftn_bound(points)) +
                     fftn::test::input_rounding<double>() * static_cast<double>(c.dims.size()) +
                     fftn::test::input_rounding<T>() * 4;
    o.within(fftn::test::compare<T>(std::span<const C>(got), want), tol,
             what + ", transform " + std::to_string(b));
  }
  EXPECT_TRUE(o.passed()) << o.detail() << text;
  std::size_t touched = 0;
  for (std::size_t i = 0; i < serial.size(); ++i) {
    if (!described[i] && std::memcmp(&serial[i], &init[i], sizeof(T)) != 0) {
      ++touched;
    }
  }
  EXPECT_EQ(touched, 0U) << "elements outside the runs written\n" << text;

  const auto in_ptr = reinterpret_cast<const In*>(x.data());
  std::vector<T> pieces = init;
  const std::size_t half = c.count / 2;
  p.execute_transforms(0, half, in_ptr, reinterpret_cast<Out*>(pieces.data()), ws);
  p.execute_transforms(half, c.count - half, in_ptr, reinterpret_cast<Out*>(pieces.data()), ws);
  EXPECT_TRUE(same_bits(pieces, serial)) << "pieces of the batch differ";
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    std::vector<T> teamed = init;
    std::vector<std::thread> members;
    for (int rank = 1; rank < 3; ++rank) {
      members.emplace_back([&, rank] {
        p.execute(in_ptr, reinterpret_cast<Out*>(teamed.data()),
                  fftn::team{.rank = rank, .size = 3}, ws, done);
      });
    }
    p.execute(in_ptr, reinterpret_cast<Out*>(teamed.data()), fftn::team{.rank = 0, .size = 3}, ws,
              done);
    for (std::thread& m : members) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << "a team differs";
  }
  const fftn::thread_pool pool(3);
  std::vector<T> pooled = init;
  p.execute(in_ptr, reinterpret_cast<Out*>(pooled.data()), pool, ws);
  EXPECT_TRUE(same_bits(pooled, serial)) << "the executor differs";
}

// Two runs on one axis and side of a line of `length` on that side: the second logical run
// stored first, a gap between them in both orders.
std::vector<fftn::stored_run> two_runs(std::size_t axis, fftn::side where, std::size_t length) {
  if (length == 1) {
    return {{.axis = axis, .where = where, .logical = 0, .stored = 1, .count = 1}};
  }
  if (length < 6) {
    return {{.axis = axis, .where = where, .logical = 0, .stored = 1, .count = 1},
            {.axis = axis, .where = where, .logical = length - 1, .stored = 0, .count = 1}};
  }
  return {{.axis = axis, .where = where, .logical = 1, .stored = 3, .count = 2},
          {.axis = axis, .where = where, .logical = length - 2, .stored = 0, .count = 2}};
}

std::vector<runs_case> runs_cases() {
  using enum transform_kind;
  std::vector<runs_case> cases;
  // {3, 1, 8}: a run stored past the start of an axis of length 1, which has no loop.
  const std::vector<std::pair<dims_t, std::size_t>> shapes{
      {{16}, 3}, {{6, 10}, 2}, {{4, 6, 8}, 2}, {{3, 1, 8}, 2}};
  for (const transform_kind k : {c2c, r2c, c2r}) {
    for (const auto& [dims, count] : shapes) {
      for (std::size_t a = 0; a < dims.size(); ++a) {
        for (const fftn::side where : {fftn::side::in, fftn::side::out}) {
          const bool output = where == fftn::side::out;
          const std::size_t length = side_dims(shape_case{.kind = k, .dims = dims}, output)[a];
          cases.push_back({.name = "runs on one axis and side",
                           .kind = k,
                           .dims = dims,
                           .count = count,
                           .runs = two_runs(a, where, length)});
        }
      }
      // Runs on every axis of both sides, accumulated, scaled, with gaps between transforms.
      runs_case all{.name = "runs on every axis and side, accumulated",
                    .kind = k,
                    .dims = dims,
                    .count = count,
                    .dir = fftn::direction::backward,
                    .scale = 0.5,
                    .accumulate = true,
                    .gap = 3};
      for (std::size_t a = 0; a < dims.size(); ++a) {
        for (const fftn::side where : {fftn::side::in, fftn::side::out}) {
          const std::size_t length =
              side_dims(shape_case{.kind = k, .dims = dims}, where == fftn::side::out)[a];
          for (const fftn::stored_run& run : two_runs(a, where, length)) {
            all.runs.push_back(run);
          }
        }
      }
      cases.push_back(all);
      // The same, the shape listing the axes out of C order (the halved axis of a real
      // transform staying last), which the plan orders back: runs and pruning by the plan's
      // order of the axes (plan_core.cpp, listed).
      if (dims.size() == 3 || (dims.size() == 2 && k == c2c)) {
        all.name = "runs on every axis and side, accumulated, axes listed out of order";
        all.listed = dims.size() == 2 ? std::vector<std::size_t>{1, 0}
                     : k == c2c       ? std::vector<std::size_t>{2, 0, 1}
                                      : std::vector<std::size_t>{1, 0, 2};
        cases.push_back(all);
      }
    }
    // Larger, so that teams share the phases out rather than run them inline: a band of bins
    // in the middle of a long row's input; runs on two axes of a plane and of a volume.
    const std::size_t row = 2048;
    const std::size_t row_in = k == c2r ? row / 2 + 1 : row;
    cases.push_back({.name = "a band of a long row's input",
                     .kind = k,
                     .dims = {row},
                     .count = 6,
                     .runs = {{.where = fftn::side::in, .logical = 100, .stored = 0, .count = 300},
                              {.where = fftn::side::out, .logical = 0, .stored = 7, .count = 9}}});
    cases.push_back(
        {.name = "a plane, runs on both axes",
         .kind = k,
         .dims = {96, 64},
         .count = 3,
         .runs = {{.axis = 0, .where = fftn::side::in, .logical = 10, .stored = 0, .count = 30},
                  {.axis = 0, .where = fftn::side::in, .logical = 70, .stored = 30, .count = 20},
                  {.axis = 1, .where = fftn::side::out, .logical = 3, .stored = 0, .count = 20}}});
    cases.push_back(
        {.name = "a volume, runs on two axes",
         .kind = k,
         .dims = {20, 12, 16},
         .count = 2,
         .runs = {{.axis = 1, .where = fftn::side::in, .logical = 2, .stored = 5, .count = 5},
                  {.axis = 0, .where = fftn::side::out, .logical = 4, .stored = 0, .count = 9},
                  {.axis = 2,
                   .where = fftn::side::in,
                   .logical = 0,
                   .stored = 0,
                   .count = std::min<std::size_t>(row_in, 16) / 2}}});
  }
  return cases;
}

// Construction refuses stored runs that are not valid, naming the run and its field, and those
// not built yet (in place, beside a factor).
TEST(EdgeShapes, StoredRunsRefused) {
  using C = std::complex<double>;
  using fftn::side;
  const auto refused = [](const std::vector<fftn::stored_run>& runs, const char* field,
                          bool in_place = false) {
    fftn::shape<C, C> s = fftn::planes_c2c<double>(8, 6, 2, fftn::direction::forward);
    s.runs = runs;
    try {
      const fftn::plan<C, C> p(s, {.in_place = in_place});
      ADD_FAILURE() << "not refused: " << field;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find(field), std::string::npos) << e.what();
    }
  };
  refused({{.axis = 2, .count = 1}}, "shape.runs[0].axis");
  refused({{.axis = 0}}, "shape.runs[0].count");
  refused({{.axis = 1, .logical = 4, .count = 3}}, "shape.runs[0].logical");
  refused({{.axis = 0, .logical = 0, .stored = 0, .count = 3},
           {.axis = 0, .logical = 3, .stored = 2, .count = 2}},
          "shape.runs[1].stored");
  refused({{.axis = 0, .logical = 0, .stored = 0, .count = 3},
           {.axis = 0, .logical = 2, .stored = 5, .count = 2}},
          "shape.runs[1].logical");
  refused({{.axis = 0, .where = side::out, .stored = PTRDIFF_MAX, .count = 1}},
          "shape.runs[0].stored");
  refused({{.axis = 0, .count = 2}}, "shape.runs", true);
  // Runs of different sides or axes may share positions.
  fftn::shape<C, C> ok = fftn::planes_c2c<double>(8, 6, 2, fftn::direction::forward);
  ok.runs = {{.axis = 0, .count = 3},
             {.axis = 0, .where = side::out, .count = 3},
             {.axis = 1, .count = 3}};
  EXPECT_NO_THROW((fftn::plan<C, C>(ok, {})));
  // A factor beside runs is not built yet.
  try {
    const fftn::plan<C, C> p(ok, {}, {.in = {.mul = {.kind = real, .axis_strides = {6, 1}}}});
    ADD_FAILURE() << "a factor beside runs not refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("edges.in.mul.kind"), std::string::npos) << e.what();
  }
}

// axes_c2c and axes_r2c with stored runs: the default strides span a dimension's stored extent
// on each side, and the runs come back with their axes as indices into shape::axes. A compact
// (y, x, w) array of nw stored bins of an nt-point line along w (DESIGN.md 12.1) then plans as
// it is, with x's stride nw: by logical extents its lines would overlap.
TEST(EdgeShapes, StoredRunsInAxes) {
  using C = std::complex<double>;
  using fftn::side;
  const std::size_t ny = 3;
  const std::size_t nx = 4;
  const std::size_t nt = 32;
  const std::size_t nw = 10;
  const std::vector<fftn::axis> array{{.n = ny}, {.n = nx}, {.n = nt}};
  const std::vector<std::size_t> along{2};
  // The band: bins 0-5 and the last 4, stored contiguously on both sides.
  const std::vector<fftn::stored_run> band{
      {.axis = 2, .where = side::in, .logical = 0, .stored = 0, .count = 6},
      {.axis = 2, .where = side::in, .logical = nt - 4, .stored = 6, .count = 4},
      {.axis = 2, .where = side::out, .logical = 0, .stored = 0, .count = 6},
      {.axis = 2, .where = side::out, .logical = nt - 4, .stored = 6, .count = 4}};
  const auto s = fftn::axes_c2c<double>(array, along, fftn::direction::backward, band);
  ASSERT_EQ(s.axes.size(), 1U);
  EXPECT_EQ(s.axes[0].in_stride, 1);
  ASSERT_EQ(s.batches.size(), 1U); // y and x nest exactly: one batch dimension
  EXPECT_EQ(s.batches[0].count, ny * nx);
  EXPECT_EQ(s.batches[0].in_dist, static_cast<std::ptrdiff_t>(nw));
  EXPECT_EQ(s.batches[0].out_dist, static_cast<std::ptrdiff_t>(nw));
  ASSERT_EQ(s.runs.size(), band.size());
  for (const fftn::stored_run& r : s.runs) {
    EXPECT_EQ(r.axis, 0U);
  }
  const fftn::plan<C, C> p(s);
  // The transform: each pencil's band, against the logical line by the reference.
  fftn::test::input_source src(0xBA4DULL);
  const auto x = src.complexes<double>(ny * nx * nw);
  std::vector<C> y(ny * nx * nw);
  fftn::workspace ws(p);
  p.execute(x.data(), y.data(), ws);
  fftn::test::outcome o;
  for (std::size_t q = 0; q < ny * nx; ++q) {
    std::vector<C> line(nt);
    for (std::size_t k = 0; k < 6; ++k) {
      line[k] = x[q * nw + k];
    }
    for (std::size_t k = 0; k < 4; ++k) {
      line[nt - 4 + k] = x[q * nw + 6 + k];
    }
    const auto exact = fftn::test::reference::c2c(+1, line);
    std::vector<fftn::test::exact_complex> want;
    std::vector<C> got;
    for (std::size_t k = 0; k < 6; ++k) {
      want.push_back(exact[k]);
      got.push_back(y[q * nw + k]);
    }
    for (std::size_t k = 0; k < 4; ++k) {
      want.push_back(exact[nt - 4 + k]);
      got.push_back(y[q * nw + 6 + k]);
    }
    o.within(fftn::test::compare<double>(std::span<const C>(got), want),
             fftn::test::derived_tolerance<double>(nt, fftn::test::fftn_bound(nt)), "band");
  }
  EXPECT_TRUE(o.passed()) << o.detail() << p.describe();

  // r2c of a real (y, t) array into nw stored bins: the complex side's default strides by them.
  const std::vector<fftn::axis> plane{{.n = ny}, {.n = nt}};
  const std::vector<std::size_t> rows{1};
  const std::vector<fftn::stored_run> bins{
      {.axis = 1, .where = side::out, .logical = 2, .stored = 0, .count = nw}};
  const auto r = fftn::axes_r2c<double>(plane, rows, false, bins);
  ASSERT_EQ(r.batches.size(), 1U);
  EXPECT_EQ(r.batches[0].in_dist, static_cast<std::ptrdiff_t>(nt));
  EXPECT_EQ(r.batches[0].out_dist, static_cast<std::ptrdiff_t>(nw));
  EXPECT_NO_THROW((fftn::plan<double, C>(r, {})));
  // A run on a dimension that is not transformed is refused.
  const std::vector<fftn::stored_run> batch_run{{.axis = 0, .count = 1}};
  EXPECT_THROW(static_cast<void>(fftn::axes_r2c<double>(plane, rows, false, batch_run)),
               std::invalid_argument);
}

// Stored runs along a reversed axis (a negative stride): the passes' offsets from element 0 are
// negative.
TEST(EdgeShapes, StoredRunsAlongAReversedAxis) {
  using C = std::complex<double>;
  using fftn::side;
  const std::size_t n = 16;
  fftn::shape<C, C> s;
  s.axes = {{.n = n, .in_stride = -1, .out_stride = -2}};
  s.batches = {{.count = 2, .in_dist = 8, .out_dist = 20}};
  s.runs = {{.where = side::in, .logical = 0, .stored = 0, .count = 4},
            {.where = side::in, .logical = 12, .stored = 4, .count = 4},
            {.where = side::out, .logical = 3, .stored = 1, .count = 5}};
  const fftn::plan<C, C> p(s);
  fftn::test::input_source src(0x4EFULL);
  std::vector<C> x = src.complexes<double>(16);
  std::vector<C> y = src.complexes<double>(40);
  const std::vector<C> init = y;
  fftn::workspace ws(p);
  // Element 0 of transform 0 is the last of its stored elements on each side.
  p.execute(x.data() + 7, y.data() + 11, ws);
  fftn::test::outcome o;
  std::vector<bool> written(y.size(), false);
  for (std::size_t b = 0; b < 2; ++b) {
    std::vector<C> line(n);
    for (std::size_t j = 0; j < n; ++j) {
      const std::ptrdiff_t stored = j < 4 ? static_cast<std::ptrdiff_t>(j)
                                          : (j >= 12 ? static_cast<std::ptrdiff_t>(j - 8) : -1);
      if (stored >= 0) {
        line[j] = x[static_cast<std::size_t>(7 + 8 * static_cast<std::ptrdiff_t>(b) - stored)];
      }
    }
    const auto exact = fftn::test::reference::c2c(-1, line);
    std::vector<fftn::test::exact_complex> want;
    std::vector<C> got;
    for (std::size_t k = 3; k < 8; ++k) {
      const auto i = static_cast<std::size_t>(11 + 20 * static_cast<std::ptrdiff_t>(b) -
                                              2 * static_cast<std::ptrdiff_t>(k - 2));
      want.push_back(exact[k]);
      got.push_back(y[i]);
      written[i] = true;
    }
    o.within(fftn::test::compare<double>(std::span<const C>(got), want),
             fftn::test::derived_tolerance<double>(n, fftn::test::fftn_bound(n)), "reversed");
  }
  EXPECT_TRUE(o.passed()) << o.detail() << p.describe();
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (!written[i]) {
      EXPECT_EQ(y[i], init[i]) << "element " << i << " outside the runs written";
    }
  }
}

// A wisdom entry of a plan with stored runs serves the same runs only; such plans are planned by
// the rules whatever the effort, and describe() says so; a strategy the runs keep the plan from
// is named.
TEST(EdgeShapes, StoredRunsPlanning) {
  using C = std::complex<double>;
  namespace d = fftn::detail;
  fftn::shape<C, C> s = fftn::rows_c2c<double>(64, 4, fftn::direction::forward);
  s.runs = {{.where = fftn::side::in, .logical = 0, .stored = 0, .count = 8}};
  const fftn::plan<C, C> measured(s, {.effort = fftn::planning::measure});
  EXPECT_NE(measured.describe().find("planning: by the rules: measured planning and wisdom take "
                                     "no stored runs yet"),
            std::string::npos)
      << measured.describe();
  const auto core = d::make_plan_core(d::view_of(s), {});
  const auto plain =
      d::make_plan_core(d::view_of(fftn::rows_c2c<double>(64, 4, fftn::direction::forward)), {});
  EXPECT_NE(core->shape_key, plain->shape_key);
  fftn::shape<C, C> other = s;
  other.runs[0].count = 9;
  EXPECT_NE(d::make_plan_core(d::view_of(other), {})->shape_key, core->shape_key);
  // The lanes the knobs force on these rows, which runs keep the plan from.
  d::build_settings lanes;
  lanes.strategy = d::strategy_choice::lanes;
  const auto forced = d::make_plan_core(d::view_of(s), {}, lanes);
  const auto unforced = d::make_plan_core(
      d::view_of(fftn::rows_c2c<double>(64, 4, fftn::direction::forward)), {}, lanes);
  if (unforced->strategy_name != forced->strategy_name) {
    EXPECT_NE(d::describe(*forced).find(
                  "not the strategy the rules would choose: " + unforced->strategy_name +
                  " without the stored runs, the chain with them: stored "
                  "runs take the chain (task 5.1d)"),
              std::string::npos)
        << d::describe(*forced);
  }
  // Scales alone are in the key, as options::scale's product.
  const fftn::plan_edges<C, C> half{.out = {.scale = 0.5}};
  EXPECT_NE(d::make_plan_core_with_edges(d::view_of(s), {}, d::view_of(half))->shape_key,
            core->shape_key);
  EXPECT_EQ(d::make_plan_core_with_edges(d::view_of(s), {}, d::view_of(half))->shape_key,
            d::make_plan_core(d::view_of(s), {.scale = 0.5})->shape_key);
  // Rows as adjacent transforms take the lanes over them without edges, but not with
  // accumulation: what the runs keep the plan from is the accumulated plan's strategy, never
  // the lanes, which accumulation alone rules out.
  d::build_settings adjacent;
  adjacent.profile.adjacent_lanes = true;
  adjacent.profile.adjacent_min_bytes = 0;
  adjacent.profile.adjacent_team_chain = false;
  adjacent.profile.column_team_bytes = 0;
  adjacent.profile.column_wide_bytes = 0;
  fftn::shape<C, C> adj;
  adj.axes = {{.n = 64, .in_stride = 16, .out_stride = 16}};
  adj.batches = {{.count = 16, .in_dist = 1, .out_dist = 1}};
  const fftn::plan_edges<C, C> accumulate{.out = {.accumulate = true}};
  const auto adj_plain = d::make_plan_core(d::view_of(adj), {}, adjacent);
  ASSERT_EQ(adj_plain->strategy_name, "the lanes over adjacent transforms");
  const auto adj_accumulated =
      d::make_plan_core_with_edges(d::view_of(adj), {}, d::view_of(accumulate), adjacent);
  adj.runs = {{.where = fftn::side::in, .logical = 0, .stored = 0, .count = 8}};
  const auto adj_runs =
      d::make_plan_core_with_edges(d::view_of(adj), {}, d::view_of(accumulate), adjacent);
  const std::string adj_text = d::describe(*adj_runs);
  EXPECT_EQ(adj_text.find("the lanes over adjacent transforms without the stored runs"),
            std::string::npos)
      << adj_text;
  if (adj_accumulated->strategy_name != adj_runs->strategy_name) {
    EXPECT_NE(adj_text.find("not the strategy the rules would choose: " +
                            adj_accumulated->strategy_name + " without the stored runs"),
              std::string::npos)
        << adj_text;
  }
}

TYPED_TEST(EdgeShapes, StoredRunsOnEveryAxisAndSide) {
  using T = TypeParam;
  using enum transform_kind;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const runs_case& c : runs_cases()) {
      switch (c.kind) {
      case c2c:
        run_runs_case<T, c2c>(c, t);
        break;
      case r2c:
        run_runs_case<T, r2c>(c, t);
        break;
      case c2r:
        run_runs_case<T, c2r>(c, t);
        break;
      }
    }
  }
}

} // namespace
