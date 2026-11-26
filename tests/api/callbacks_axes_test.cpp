// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Block callbacks on planes and volumes (fftn/callback.hpp, DESIGN.md 12.3; task 5.3c): c2c, r2c
// and c2r, in and out of place, strided with gaps (canaries), axes listed in any order, a stride-1
// batch (the free-space layout, the lanes over adjacent transforms), a volume plane at a time, and
// the team program's narrower column blocks, against the quadruple-precision reference of "the
// input callback's loop, the factor, the transform, the factor and scale, the output callback's
// loop". The callbacks are nonlinear (cubic) and depend on every coordinate; they check every
// block's geometry (origin, element and line axes and strides within the side, lines along an
// axis or along transforms, never an idle lane, a line whole up to the block size), `member` and
// their scratch, and count every element they are passed (each exactly once). Every mode, beside
// a factor or accumulation (the callback then in that side's edge phase) and scales; the input
// left as it was; serial, pieces of the batch, teams (all and own) and an executor bitwise alike;
// and describe(): the callbacks inside the first and last phases, no edge phase of their own.

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
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
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

namespace d = fftn::detail;
using d::transform_kind;
using fftn::factor_kind;
using cd = std::complex<double>;
using dims_t = std::vector<std::size_t>;
using strides_t = std::vector<std::ptrdiff_t>;

enum class in_kind : unsigned char { none, modify, produce };
enum class out_kind : unsigned char { none, modify, consume };

std::size_t product(const dims_t& v) {
  return std::accumulate(v.begin(), v.end(), std::size_t{1}, std::multiplies<>());
}

struct axes_case {
  const char* name = "";
  transform_kind kind = transform_kind::c2c;
  dims_t dims; // logical lengths in C order (the last contiguous)
  // shape.axes[i] is C axis listed[i] (a real transform's halved axis stays last); empty: C
  // order.
  std::vector<std::size_t> listed{};
  std::size_t count = 1;
  in_kind in = in_kind::none;
  out_kind out = out_kind::none;
  fftn::direction dir = fftn::direction::forward; // c2c only
  double scale = 1.0;                             // options::scale
  factor_kind in_factor = factor_kind::none;      // tight C-ordered strides, per transform
  factor_kind out_factor = factor_kind::none;
  bool accumulate = false;
  bool in_place = false;
  std::ptrdiff_t gap = 0;    // out of place: elements after each row and each transform
  std::ptrdiff_t stride = 1; // out of place: every stride and distance times this
  bool adjacent = false;     // c2c: the transforms one element apart (a stride-1 batch)
  std::size_t scratch = 0;
  const d::build_settings* settings = nullptr; // null: the rules
  bool team_program = false;                   // the plan must have a team program
};

// The side's lengths in C order: the complex side of a real transform has n/2 + 1 bins along the
// last axis.
dims_t side_dims(const axes_case& c, bool output) {
  dims_t v = c.dims;
  if (output ? c.kind == transform_kind::r2c : c.kind == transform_kind::c2r) {
    v.back() = v.back() / 2 + 1;
  }
  return v;
}

struct layout {
  strides_t stride; // C order
  std::ptrdiff_t dist = 0;
  std::size_t span = 0;
  [[nodiscard]] std::size_t at(const dims_t& j, std::size_t b) const {
    std::ptrdiff_t i = static_cast<std::ptrdiff_t>(b) * dist;
    for (std::size_t a = 0; a < j.size(); ++a) {
      i += static_cast<std::ptrdiff_t>(j[a]) * stride[a];
    }
    return static_cast<std::size_t>(i);
  }
};

layout tight_layout(const dims_t& v, std::size_t count, std::ptrdiff_t gap) {
  layout l;
  const std::size_t r = v.size();
  l.stride.assign(r, 1);
  for (std::size_t i = r - 1; i > 0; --i) {
    l.stride[i - 1] = l.stride[i] * static_cast<std::ptrdiff_t>(v[i]) + (i == r - 1 ? gap : 0);
  }
  l.dist = l.stride[0] * static_cast<std::ptrdiff_t>(v[0]) + gap;
  l.span = static_cast<std::size_t>(l.dist) * count;
  return l;
}

template <class F> void each(const dims_t& v, F&& f) {
  dims_t j(v.size(), 0);
  for (std::size_t k = 0; k < product(v); ++k) {
    f(j);
    for (std::size_t a = v.size(); a > 0; --a) {
      if (++j[a - 1] < v[a - 1]) {
        break;
      }
      j[a - 1] = 0;
    }
  }
}

std::size_t flat(const dims_t& v, const dims_t& j) {
  std::size_t i = 0;
  for (std::size_t a = 0; a < v.size(); ++a) {
    i = i * v[a] + j[a];
  }
  return i;
}

// The DFT along axis a of a C-ordered array of lengths v, from the quadruple-precision reference,
// rounded to double.
void dft_axis(int sign, const dims_t& v, std::vector<cd>& x, std::size_t a) {
  std::size_t inner = 1;
  for (std::size_t e = a + 1; e < v.size(); ++e) {
    inner *= v[e];
  }
  const std::size_t n = v[a];
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

// The transform of one input (C order over the input side's lengths): c2c over every axis; r2c
// the rows, then the other axes forward; c2r the other axes backward, then the rows.
std::vector<cd> reference(const axes_case& c, const std::vector<cd>& x) {
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

// The functions the callbacks apply, pure per element: nonlinear, and dependent on every
// coordinate (q: the element's C-ordered index within its transform, t the transform).
template <class R> R gain_in(std::size_t q, std::size_t t) {
  return static_cast<R>(0.05) * static_cast<R>(1 + (q * 7 + t * 3) % 5);
}
template <class R> R gain_out(std::size_t q, std::size_t t) {
  return static_cast<R>(0.02) * static_cast<R>(1 + (q * 5 + t) % 5);
}
template <class R> std::complex<R> cubic_in(std::complex<R> x, std::size_t q, std::size_t t) {
  const R g = R(1) + gain_in<R>(q, t) * (x.real() * x.real() + x.imag() * x.imag());
  return {x.real() * g, x.imag() * g};
}
template <class R> R cubic_in(R x, std::size_t q, std::size_t t) {
  return x * (R(1) + gain_in<R>(q, t) * x * x);
}
template <class R>
std::complex<R> cubic_out(std::complex<R> y, std::size_t q, std::size_t t, std::size_t points) {
  const R g = R(1) + gain_out<R>(q, t) * (y.real() * y.real() + y.imag() * y.imag()) /
                         static_cast<R>(points);
  return {y.real() * g, y.imag() * g};
}
template <class R> R cubic_out(R y, std::size_t q, std::size_t t, std::size_t points) {
  return y * (R(1) + gain_out<R>(q, t) * y * y / static_cast<R>(points));
}
template <class E> E produced(std::size_t q, std::size_t t) {
  const double a = 0.37 * static_cast<double>(q) + 1.1 * static_cast<double>(t);
  if constexpr (std::is_floating_point_v<E>) {
    return static_cast<E>(std::sin(a));
  } else {
    using R = typename E::value_type;
    return {static_cast<R>(std::sin(a)), static_cast<R>(std::cos(0.61 * a - 0.5))};
  }
}

// What a side's callback shares with the test: the side's geometry (C order, and shape::axes'
// order as `listed`), the elements it was passed, consumed blocks, and its findings.
template <class E> struct cb_context {
  dims_t dims; // the side's lengths, C order
  std::vector<std::size_t> listed;
  std::size_t count = 1;
  std::size_t points = 1; // of the whole transform (the output's gain)
  int lanes = 1;
  std::size_t scratch = 0;
  std::size_t block_bytes = 0;
  std::vector<E> captured; // consume: transform t's element q at t product(dims) + q
  std::unique_ptr<std::atomic<int>[]> hits;
  std::atomic<int> errors{0};
  std::atomic<int> first_error_line{0};

  void reset() {
    hits = std::make_unique<std::atomic<int>[]>(product(dims) * count);
    captured.assign(product(dims) * count, E{});
    errors = 0;
  }
  void fail(int line) {
    if (errors.fetch_add(1) == 0) {
      first_error_line = line;
    }
  }
  // Element i of line k of the block: its C-ordered coordinates and transform, false (and a
  // failure) outside the side or the batch.
  bool where(const fftn::block_info& info, std::size_t i, std::size_t k, dims_t& j,
             std::size_t& t) {
    const std::size_t r = dims.size();
    t = info.transform;
    for (std::size_t a = 0; a < r; ++a) {
      std::size_t v = info.origin[a];
      if (a == info.element_axis) {
        v += i;
      }
      if (a == info.line_axis) {
        v += k;
      }
      j[listed[a]] = v;
    }
    if (info.line_axis == fftn::along_transforms) {
      t += k;
    }
    for (std::size_t a = 0; a < r; ++a) {
      if (j[a] >= dims[a]) {
        fail(__LINE__);
        return false;
      }
    }
    if (t >= count) {
      fail(__LINE__);
      return false;
    }
    return true;
  }
  bool check(const fftn::block_info& info) {
    const std::size_t r = dims.size();
    const bool ok = info.lines >= 1 && info.origin.size() == r && info.element_axis < r &&
                    (info.line_axis < r || info.line_axis == fftn::along_transforms) &&
                    info.line_axis != info.element_axis && info.length > 0 && info.member >= 0 &&
                    info.member < lanes && info.copy == 0 && info.group == 1 &&
                    info.group_stride == 0 && info.scratch.size() == scratch &&
                    (info.lines > 1 || info.line_stride == 0);
    if (!ok) {
      fail(__LINE__);
      return false;
    }
    // A line whole up to the block size, as many a call as fit; a longer one in element ranges
    // of at most that size, one line a call.
    const std::size_t line_length = dims[listed[info.element_axis]];
    const std::size_t line_bytes = line_length * sizeof(E);
    const bool cut_right = line_bytes <= block_bytes
                               ? info.origin[info.element_axis] == 0 &&
                                     info.length == line_length &&
                                     info.lines * line_bytes <= block_bytes
                               : info.lines == 1 && info.length * sizeof(E) <= block_bytes;
    if (!cut_right) {
      fail(__LINE__);
    }
    if (scratch != 0 && reinterpret_cast<std::uintptr_t>(info.scratch.data()) % 128 != 0) {
      fail(__LINE__);
    }
    if (scratch != 0) {
      std::ranges::fill(info.scratch, std::byte{0x3C}); // the whole area is the callback's
    }
    return true;
  }
  template <class F> void each(const fftn::block_info& info, E* data, F f) {
    if (!check(info)) {
      return;
    }
    dims_t j(dims.size(), 0);
    std::size_t t = 0;
    for (std::size_t k = 0; k < info.lines; ++k) {
      for (std::size_t i = 0; i < info.length; ++i) {
        if (!where(info, i, k, j, t)) {
          return;
        }
        const std::size_t q = flat(dims, j);
        hits[t * product(dims) + q].fetch_add(1, std::memory_order_relaxed);
        f(data[static_cast<std::ptrdiff_t>(i) * info.element_stride +
               static_cast<std::ptrdiff_t>(k) * info.line_stride],
          q, t);
      }
    }
  }
};

template <class E> void modify_in(void* user, const fftn::block_info& info, E* data) noexcept {
  static_cast<cb_context<E>*>(user)->each(
      info, data, [](E& x, std::size_t q, std::size_t t) { x = cubic_in(x, q, t); });
}
template <class E> void produce_in(void* user, const fftn::block_info& info, E* data) noexcept {
  static_cast<cb_context<E>*>(user)->each(
      info, data, [](E& x, std::size_t q, std::size_t t) { x = produced<E>(q, t); });
}
template <class E> void modify_out(void* user, const fftn::block_info& info, E* data) noexcept {
  auto& ctx = *static_cast<cb_context<E>*>(user);
  const std::size_t points = ctx.points;
  ctx.each(info, data,
           [points](E& y, std::size_t q, std::size_t t) { y = cubic_out(y, q, t, points); });
}
template <class E> void consume_out(void* user, const fftn::block_info& info, E* data) noexcept {
  auto& ctx = *static_cast<cb_context<E>*>(user);
  const std::size_t side = product(ctx.dims);
  ctx.each(info, data,
           [&ctx, side](E& y, std::size_t q, std::size_t t) { ctx.captured[t * side + q] = y; });
}

template <class T> bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

template <class E> cd widen(E v) {
  if constexpr (std::is_floating_point_v<E>) {
    return {static_cast<double>(v), 0.0};
  } else {
    return cd(v);
  }
}

// One case on one target in precision T.
template <class T, transform_kind K> struct axes_run {
  using C = std::complex<T>;
  using In = std::conditional_t<K == transform_kind::r2c, T, C>;
  using Out = std::conditional_t<K == transform_kind::c2r, T, C>;
  static constexpr std::size_t in_width = std::is_same_v<In, T> ? 1 : 2;
  static constexpr std::size_t out_width = std::is_same_v<Out, T> ? 1 : 2;

  axes_case c;
  dims_t in_dims;
  dims_t out_dims;
  std::vector<std::size_t> listed;
  layout lin;
  layout lout;
  fftn::shape<In, Out> shape;
  fftn::plan_edges<In, Out> edges;
  fftn::options opts;
  std::vector<T> x;    // the input buffer (reals), gaps included
  std::vector<T> init; // the output buffer before execution (in place: the input)
  std::vector<T> fin;  // factors over the side's C-ordered elements, per transform
  std::vector<T> fout;
  cb_context<In> in_ctx;
  cb_context<Out> out_ctx;

  axes_run(const axes_case& cc, fftn::isa t, fftn::test::input_source& src)
      : c(cc), in_dims(side_dims(cc, false)), out_dims(side_dims(cc, true)) {
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
          for (std::ptrdiff_t& st : l->stride) {
            st *= c.stride;
          }
          l->dist *= c.stride;
          l->span *= static_cast<std::size_t>(c.stride);
        }
      }
    }
    shape.dir = c.dir;
    for (const std::size_t a : listed) {
      shape.axes.push_back(
          {.n = c.dims[a], .in_stride = lin.stride[a], .out_stride = lout.stride[a]});
    }
    shape.batches = {{.count = c.count, .in_dist = lin.dist, .out_dist = lout.dist}};
    const auto factor = [&](factor_kind k, const dims_t& v) -> fftn::factor {
      if (k == factor_kind::none) {
        return {};
      }
      const layout l = tight_layout(v, c.count, 0);
      fftn::factor f{.kind = k, .axis_strides = {}, .batch_strides = {l.dist}};
      for (const std::size_t a : listed) {
        f.axis_strides.push_back(l.stride[a]);
      }
      return f;
    };
    edges.in = {.mul = factor(c.in_factor, in_dims)};
    edges.out = {.mul = factor(c.out_factor, out_dims), .accumulate = c.accumulate};
    if (c.in == in_kind::modify) {
      edges.in.call = {.fn = &modify_in<In>, .scratch_bytes = c.scratch};
    } else if (c.in == in_kind::produce) {
      edges.in.call = {
          .fn = &produce_in<In>, .mode = fftn::callback_mode::produce, .scratch_bytes = c.scratch};
    }
    if (c.out == out_kind::modify) {
      edges.out.call = {.fn = &modify_out<Out>, .scratch_bytes = c.scratch};
    } else if (c.out == out_kind::consume) {
      edges.out.call = {.fn = &consume_out<Out>,
                        .mode = fftn::callback_mode::consume,
                        .scratch_bytes = c.scratch};
    }
    opts = {.in_place = c.in_place, .scale = c.scale, .force_isa = t};
    const std::size_t in_reals = lin.span * in_width;
    const std::size_t out_reals = lout.span * out_width;
    x = src.reals<T>(c.in_place ? std::max(in_reals, out_reals) : in_reals);
    init = c.in_place ? x : src.reals<T>(out_reals);
    const auto factors = [&](factor_kind k, const dims_t& v) {
      if (k == factor_kind::none) {
        return std::vector<T>{};
      }
      std::vector<T> f = src.reals<T>(product(v) * c.count * (k == factor_kind::complex ? 2 : 1));
      if (k == factor_kind::real) {
        for (T& u : f) {
          u = static_cast<T>(0.5) + static_cast<T>(0.5) * u;
        }
      }
      return f;
    };
    fin = factors(c.in_factor, in_dims);
    fout = factors(c.out_factor, out_dims);
    const std::size_t block = c.settings != nullptr
                                  ? c.settings->profile.callback_block_bytes
                                  : d::build_settings{}.profile.callback_block_bytes;
    in_ctx.dims = in_dims;
    out_ctx.dims = out_dims;
    in_ctx.listed = out_ctx.listed = listed;
    in_ctx.count = out_ctx.count = c.count;
    in_ctx.points = out_ctx.points = product(c.dims);
    in_ctx.scratch = out_ctx.scratch = c.scratch;
    in_ctx.block_bytes = out_ctx.block_bytes = block;
  }

  [[nodiscard]] cd factor_of(const std::vector<T>& f, factor_kind k, const dims_t& v,
                             const dims_t& j, std::size_t b) const {
    const std::size_t i = b * product(v) + flat(v, j);
    if (k == factor_kind::complex) {
      return {static_cast<double>(f[2 * i]), static_cast<double>(f[2 * i + 1])};
    }
    return k == factor_kind::real ? cd(static_cast<double>(f[i]), 0.0) : cd(1.0, 0.0);
  }
  [[nodiscard]] In in_value(const std::vector<T>& buf, std::size_t at) const {
    if constexpr (in_width == 1) {
      return buf[at];
    } else {
      return {buf[2 * at], buf[2 * at + 1]};
    }
  }
  [[nodiscard]] Out out_value(const std::vector<T>& buf, std::size_t at) const {
    if constexpr (out_width == 1) {
      return buf[at];
    } else {
      return {buf[2 * at], buf[2 * at + 1]};
    }
  }
  [[nodiscard]] fftn::exec_args args() {
    return {.in_factor = fin.empty() ? nullptr : fin.data(),
            .out_factor = fout.empty() ? nullptr : fout.data(),
            .in_user = &in_ctx,
            .out_user = &out_ctx};
  }
  void prepare(int lanes) {
    in_ctx.lanes = out_ctx.lanes = lanes;
    in_ctx.reset();
    out_ctx.reset();
  }
  void expect_clean(const std::string& what) const {
    const auto side = [&](const auto& ctx, bool used, const char* name) {
      EXPECT_EQ(ctx.errors.load(), 0) << what << ": a " << name << " callback's check failed (line "
                                      << ctx.first_error_line.load() << ")";
      std::size_t wrong = 0;
      for (std::size_t i = 0; i < product(ctx.dims) * c.count; ++i) {
        if (ctx.hits[i].load() != (used ? 1 : 0)) {
          ++wrong;
        }
      }
      EXPECT_EQ(wrong, 0U) << what << ": " << name << " elements not passed exactly once";
    };
    side(in_ctx, c.in != in_kind::none, "input");
    side(out_ctx, c.out != out_kind::none, "output");
  }

  void check(const std::vector<T>& y, const std::string& what) const {
    const double scale = c.scale;
    fftn::test::outcome o;
    std::vector<bool> described(y.size(), false);
    for (std::size_t b = 0; b < c.count; ++b) {
      std::vector<cd> xin(product(in_dims));
      each(in_dims, [&](const dims_t& j) {
        const std::size_t q = flat(in_dims, j);
        In v{};
        if (c.in == in_kind::produce) {
          v = produced<In>(q, b);
        } else {
          v = in_value(x, lin.at(j, b));
          if (c.in == in_kind::modify) {
            v = cubic_in(v, q, b);
          }
        }
        xin[q] = widen(v) * factor_of(fin, c.in_factor, in_dims, j, b);
      });
      const std::vector<cd> exact = reference(c, xin);
      std::vector<fftn::test::exact_complex> want;
      std::vector<C> got;
      each(out_dims, [&](const dims_t& j) {
        const std::size_t q = flat(out_dims, j);
        cd v = exact[q] * factor_of(fout, c.out_factor, out_dims, j, b) * scale;
        if (c.out == out_kind::modify) {
          if constexpr (out_width == 1) {
            v = {cubic_out(v.real(), q, b, product(c.dims)), 0.0};
          } else {
            v = cubic_out(v, q, b, product(c.dims));
          }
        }
        if (c.out == out_kind::consume) {
          want.push_back(fftn::test::exact_from(v));
          got.push_back(C(widen(out_ctx.captured[b * product(out_dims) + q])));
          return;
        }
        const std::size_t at = lout.at(j, b);
        if (c.accumulate) {
          v += widen(out_value(init, at));
        }
        want.push_back(fftn::test::exact_from(v));
        got.push_back(C(widen(out_value(y, at))));
        for (std::size_t w = 0; w < out_width; ++w) {
          described[at * out_width + w] = true;
        }
      });
      // The transform's bound; the reference's rounding after each axis; a rounding for each
      // factor's product, the scale and the accumulation; the cubic output's slope.
      const std::size_t points = product(c.dims);
      auto tol = fftn::test::derived_tolerance<T>(points, fftn::test::fftn_bound(points)) +
                 fftn::test::input_rounding<double>() * static_cast<double>(c.dims.size()) +
                 fftn::test::input_rounding<T>() * 4;
      if (c.out == out_kind::modify) {
        tol = tol * 3;
      }
      o.within(fftn::test::compare<T>(std::span<const C>(got), want), tol,
               what + ", transform " + std::to_string(b));
    }
    EXPECT_TRUE(o.passed()) << o.detail();
    if (!c.in_place) {
      std::size_t touched = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        if (!described[i] && std::memcmp(&y[i], &init[i], sizeof(T)) != 0) {
          ++touched;
        }
      }
      EXPECT_EQ(touched, 0U) << what << ": elements outside the transforms written";
    }
  }
};

template <class T, transform_kind K> void run_case(const axes_case& cc, fftn::isa t, bool threads) {
  using R = axes_run<T, K>;
  fftn::test::input_source src(0x5C3ULL + product(cc.dims) * 131 + cc.count);
  R run(cc, t, src);
  const d::build_settings settings = cc.settings != nullptr ? *cc.settings : d::build_settings{};
  const auto core = d::make_plan_core_with_edges(d::view_of(run.shape), run.opts,
                                                 d::view_of(run.edges), settings);
  const std::string text = d::describe(*core);
  const std::string what = std::string(cc.name) + ", " + fftn::isa_name(t) +
                           (sizeof(T) == 8 ? " double" : " float") + "\n" + text;
  SCOPED_TRACE(what);
  EXPECT_NE(text.find("\ncallbacks:"), std::string::npos);
  // A callback has no edge phase of its own (DESIGN.md 12.3): inside the first or last phase,
  // or in the edge phase a factor or accumulation of its side has anyway.
  const bool in_alone = cc.in != in_kind::none && cc.in_factor == factor_kind::none;
  const bool out_alone =
      cc.out != out_kind::none && cc.out_factor == factor_kind::none && !cc.accumulate;
  if (in_alone) {
    EXPECT_NE(text.find("the input's (" +
                        std::string(cc.in == in_kind::modify ? "modify" : "produce") +
                        ") in the first phase"),
              std::string::npos);
  }
  if (out_alone) {
    EXPECT_NE(text.find("the output's (" +
                        std::string(cc.out == out_kind::modify ? "modify" : "consume") +
                        ") in the last phase"),
              std::string::npos);
  }
  const bool phases =
      cc.in_factor != factor_kind::none || cc.out_factor != factor_kind::none || cc.accumulate;
  if (!phases) {
    EXPECT_NE(text.find("edges: block callbacks only; they cost no separate pass"),
              std::string::npos);
    EXPECT_EQ(text.find("separate phase"), std::string::npos);
  }
  if (cc.team_program) {
    EXPECT_NE(core->team, nullptr) << "no team program";
  }
  const fftn::exec_args args = run.args();
  const std::vector<T> x0 = run.x;
  const auto fresh = [&] { return cc.in_place ? run.x : run.init; };
  const bool produced_in = cc.in == in_kind::produce;
  const bool consumed_out = cc.out == out_kind::consume;
  // The serial execution passes the arrays (a produced input null out of place); the others pass
  // a produced input and a consumed output as null.
  const auto in_of = [&](std::vector<T>& y, std::vector<T>& xs, bool null_ok) -> const void* {
    if (produced_in && (null_ok || !cc.in_place)) {
      return nullptr;
    }
    return cc.in_place ? y.data() : xs.data();
  };
  const auto out_of = [&](std::vector<T>& y) -> void* { return consumed_out ? nullptr : y.data(); };
  const fftn::workspace ws(d::workspace_bytes(*core, 3));
  std::vector<T> xs = run.x;
  std::vector<T> serial = fresh();
  run.prepare(1);
  d::execute_core(*core, 0, cc.count, in_of(serial, xs, false), serial.data(), ws, args);
  run.expect_clean("serial");
  run.check(serial, "serial");
  if (!cc.in_place) {
    EXPECT_TRUE(same_bits(xs, x0)) << "input written";
  }
  if (consumed_out) {
    EXPECT_TRUE(same_bits(serial, run.init)) << "a consumed output written";
  }
  const auto captured_serial = run.out_ctx.captured;
  const auto agree = [&](const std::vector<T>& y, const std::string& form) {
    run.expect_clean(form);
    if (consumed_out) {
      EXPECT_TRUE(run.out_ctx.captured == captured_serial) << form << " differs";
    } else {
      EXPECT_TRUE(same_bits(y, serial)) << form << " differs";
    }
  };
  std::vector<T> pieces = fresh();
  run.prepare(1);
  const std::size_t half = cc.count / 2;
  d::execute_core(*core, 0, half, in_of(pieces, xs, true), out_of(pieces), ws, args);
  d::execute_core(*core, half, cc.count - half, in_of(pieces, xs, true), out_of(pieces), ws, args);
  agree(pieces, "pieces of the batch");
  if (!threads) {
    return;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    std::vector<T> teamed = fresh();
    run.prepare(3);
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        d::execute_team(*core, in_of(teamed, xs, true), out_of(teamed),
                        fftn::team{.rank = r, .size = 3}, ws, done, args);
      });
    }
    d::execute_team(*core, in_of(teamed, xs, true), out_of(teamed),
                    fftn::team{.rank = 0, .size = 3}, ws, done, args);
    for (std::thread& m : members) {
      m.join();
    }
    agree(teamed, done == fftn::completion::all ? "a team (all)" : "a team (own)");
  }
  const fftn::thread_pool pool(3);
  std::vector<T> pooled = fresh();
  run.prepare(pool.concurrency());
  d::execute_on(*core, in_of(pooled, xs, true), out_of(pooled), pool, ws, args);
  agree(pooled, "the executor");
}

template <class T> void run_any(const axes_case& c, fftn::isa t, bool threads) {
  switch (c.kind) {
  case transform_kind::c2c:
    run_case<T, transform_kind::c2c>(c, t, threads);
    break;
  case transform_kind::r2c:
    run_case<T, transform_kind::r2c>(c, t, threads);
    break;
  case transform_kind::c2r:
    run_case<T, transform_kind::c2r>(c, t, threads);
    break;
  case transform_kind::r2r:
    ADD_FAILURE() << "an r2r case";
    break;
  }
}

constexpr factor_kind real = factor_kind::real;
constexpr factor_kind cplx = factor_kind::complex;
constexpr transform_kind c2c = transform_kind::c2c;
constexpr transform_kind r2c = transform_kind::r2c;
constexpr transform_kind c2r = transform_kind::c2r;

struct knobs {
  d::build_settings adjacent;     // the lanes over adjacent transforms
  d::build_settings planes;       // a volume a plane at a time
  d::build_settings small_blocks; // blocks of 256 bytes: lines cut into element ranges
  d::build_settings team;         // a team program with narrower column blocks
  knobs() {
    adjacent.profile.adjacent_lanes = true;
    adjacent.profile.adjacent_min_bytes = 0;
    adjacent.profile.adjacent_team_chain = false;
    adjacent.profile.column_team_bytes = 0;
    adjacent.profile.column_wide_bytes = 0;
    planes.volume_planes = true;
    small_blocks.profile.callback_block_bytes = 256;
    team.profile.column_team_bytes = 64;
    team.profile.line_bytes = 64;
    team.profile.column_wide_bytes = 0;
    team.profile.team_volume_planes = true;
  }
};

const knobs& knob() {
  static const knobs k;
  return k;
}

std::vector<axes_case> cases() {
  const knobs& k = knob();
  return {
      // c2c planes and volumes: the callbacks in the first phase's rows and the last phase's
      // columns (alone through the chain, or lanes column blocks where the rules take them).
      {.name = "c2c plane, both modify, scale",
       .kind = c2c,
       .dims = {12, 10},
       .count = 3,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .scale = 0.5,
       .scratch = 48},
      {.name = "c2c plane, produce and consume, backward",
       .kind = c2c,
       .dims = {16, 16},
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .dir = fftn::direction::backward},
      {.name = "c2c plane of wide columns, both modify",
       .kind = c2c,
       .dims = {32, 64},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify},
      {.name = "c2c plane, blocks of 256 bytes",
       .kind = c2c,
       .dims = {48, 32},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .settings = &k.small_blocks},
      {.name = "c2c volume, modify and consume",
       .kind = c2c,
       .dims = {6, 5, 8},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::consume,
       .scratch = 16},
      {.name = "c2c volume a plane at a time",
       .kind = c2c,
       .dims = {4, 6, 8},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .settings = &k.planes},
      {.name = "c2c plane in place",
       .kind = c2c,
       .dims = {8, 12},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_place = true},
      {.name = "c2c plane in place, produce only",
       .kind = c2c,
       .dims = {8, 6},
       .count = 2,
       .in = in_kind::produce,
       .in_place = true},
      {.name = "c2c plane, strided with gaps",
       .kind = c2c,
       .dims = {10, 9},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .gap = 3,
       .stride = 2},
      {.name = "c2c plane, axes listed columns first",
       .kind = c2c,
       .dims = {6, 10},
       .listed = {1, 0},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::consume},
      {.name = "c2c volume, axes listed out of order",
       .kind = c2c,
       .dims = {4, 5, 6},
       .listed = {2, 0, 1},
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::modify},
      // The free-space layout: a stride-1 batch, the lanes over adjacent transforms.
      {.name = "c2c plane, a stride-1 batch",
       .kind = c2c,
       .dims = {8, 8},
       .count = 6,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .adjacent = true,
       .scratch = 32,
       .settings = &k.adjacent},
      {.name = "c2c volume, a stride-1 batch, produce",
       .kind = c2c,
       .dims = {4, 4, 6},
       .count = 5,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .adjacent = true,
       .settings = &k.adjacent},
      {.name = "c2c plane, a stride-1 batch in place",
       .kind = c2c,
       .dims = {6, 8},
       .count = 4,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_place = true,
       .adjacent = true,
       .settings = &k.adjacent},
      // Beside a factor or accumulation: the callback in that side's edge phase.
      {.name = "c2c plane, input factor and modify",
       .kind = c2c,
       .dims = {8, 10},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_factor = cplx},
      {.name = "c2c plane, accumulated output with modify, produced input",
       .kind = c2c,
       .dims = {8, 6},
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .out_factor = real,
       .accumulate = true},
      {.name = "c2c plane, a stride-1 batch, input factor",
       .kind = c2c,
       .dims = {6, 6},
       .count = 4,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_factor = real,
       .adjacent = true,
       .settings = &k.adjacent},
      // The team program's narrower column blocks.
      {.name = "c2c plane, a team program",
       .kind = c2c,
       .dims = {32, 64},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .settings = &k.team,
       .team_program = true},
      {.name = "c2c volume, a team program",
       .kind = c2c,
       .dims = {8, 8, 16},
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .settings = &k.team,
       .team_program = true},
      // r2c: the input's on the real rows (a copy in the workspace block out of place), the
      // output's on the complex columns of the last phase.
      {.name = "r2c plane, both modify, scale",
       .kind = r2c,
       .dims = {8, 12},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .scale = 0.25},
      {.name = "r2c plane, odd rows, modify and consume",
       .kind = r2c,
       .dims = {9, 7},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::consume},
      {.name = "r2c plane in place",
       .kind = r2c,
       .dims = {6, 10},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_place = true},
      {.name = "r2c volume, produce and consume",
       .kind = r2c,
       .dims = {4, 6, 10},
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .scratch = 24},
      {.name = "r2c plane, strided with gaps",
       .kind = r2c,
       .dims = {6, 8},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .gap = 2,
       .stride = 3},
      {.name = "r2c plane, odd rows, strided, produce",
       .kind = r2c,
       .dims = {6, 9},
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .gap = 1,
       .stride = 2},
      {.name = "r2c plane of wide columns",
       .kind = r2c,
       .dims = {64, 64},
       .count = 1,
       .in = in_kind::modify,
       .out = out_kind::modify},
      {.name = "r2c plane, input factor",
       .kind = r2c,
       .dims = {6, 8},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_factor = real},
      // c2r: the input's on the complex columns of the first phase, the output's on the real
      // rows (a consumed output's rows in the workspace block beside the half spectrum).
      {.name = "c2r plane, both modify",
       .kind = c2r,
       .dims = {8, 12},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .scale = 0.5},
      {.name = "c2r plane in place",
       .kind = c2r,
       .dims = {6, 10},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_place = true},
      {.name = "c2r volume, produce and consume",
       .kind = c2r,
       .dims = {4, 6, 10},
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::consume},
      {.name = "c2r plane, modify and consume, odd rows",
       .kind = c2r,
       .dims = {6, 9},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::consume},
      {.name = "c2r plane of wide columns",
       .kind = c2r,
       .dims = {64, 64},
       .count = 1,
       .in = in_kind::modify,
       .out = out_kind::modify},
      {.name = "c2r plane, strided with gaps",
       .kind = c2r,
       .dims = {6, 8},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .gap = 1,
       .stride = 2},
      {.name = "c2r plane, accumulated output with modify",
       .kind = c2r,
       .dims = {6, 8},
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .accumulate = true},
  };
}

template <class T> class CallbackAxes : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(CallbackAxes, precisions);

TYPED_TEST(CallbackAxes, EveryCaseOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const axes_case& c : cases()) {
      run_any<TypeParam>(c, t, false);
    }
  }
}

TYPED_TEST(CallbackAxes, TeamsAndExecutorsAgreeWithSerial) {
  for (const axes_case& c : cases()) {
    run_any<TypeParam>(c, fftn::isa::best, true);
  }
}

// Plans without callbacks keep their programs: a plane or volume with a modify callback on each
// side has the program of the plan without them plus the callback stages (and an input copy per
// line out of place), its workspace that of the plan without them plus the callback scratch.
TEST(CallbackAxes, CallbacksAddOnlyTheirStages) {
  const auto none = [](const fftn::block_info&, cd*) noexcept {};
  using F = decltype(none);
  for (const dims_t& dims : {dims_t{16, 16}, dims_t{4, 8, 8}}) {
    fftn::shape<cd, cd> s;
    for (const std::size_t n : dims) {
      s.axes.push_back({.n = n});
    }
    s.batches = {{.count = 2}};
    for (const bool in_place : {false, true}) {
      const fftn::options opts{.in_place = in_place};
      fftn::plan_edges<cd, cd> both;
      both.in.call = {.fn = fftn::callback_of<const F, cd>()};
      both.out.call = {.fn = fftn::callback_of<const F, cd>()};
      const auto plain = d::make_plan_core(d::view_of(s), opts);
      const auto with = d::make_plan_core_with_edges(d::view_of(s), opts, d::view_of(both));
      EXPECT_EQ(with->strategy_name, plain->strategy_name) << d::describe(*with);
      EXPECT_EQ(with->axis_notes, plain->axis_notes) << d::describe(*with);
      EXPECT_EQ(d::workspace_bytes(*with, 1), d::workspace_bytes(*plain, 1)) << d::describe(*with);
      const std::size_t added = in_place ? 2 : 3; // the callbacks, and out of place the copy
      EXPECT_EQ(with->prog.records.size(), plain->prog.records.size() + added)
          << d::describe(*with);
    }
  }
}

void none_real(void*, const fftn::block_info&, double*) noexcept {}

// An r2c plane's or volume's input callback out of place costs no workspace for its copies: each
// real row goes to its own complex row of the output (rows contiguous there) or to a scratch line
// of the member running it (rows strided), never to a whole transform in the workspace block
// (DESIGN.md 12.3, "As built (task 5.3c)"). A strided output's scratch lines may grow to hold a
// row: by at most one row each, per lane.
TEST(CallbackAxes, RealInputCallbacksCostNoTransformOfWorkspace) {
  for (const dims_t& dims : {dims_t{64, 48}, dims_t{16, 12, 20}, dims_t{30, 99}}) {
    for (const std::ptrdiff_t out_stride : {std::ptrdiff_t{1}, std::ptrdiff_t{3}}) {
      fftn::shape<double, cd> s;
      const std::size_t r = dims.size();
      std::ptrdiff_t in_span = 1;
      std::ptrdiff_t out_span = out_stride;
      s.axes.resize(r);
      for (std::size_t a = r; a > 0; --a) {
        const std::size_t n_out = a == r ? dims[a - 1] / 2 + 1 : dims[a - 1];
        s.axes[a - 1] = {.n = dims[a - 1], .in_stride = in_span, .out_stride = out_span};
        in_span *= static_cast<std::ptrdiff_t>(dims[a - 1]);
        out_span *= static_cast<std::ptrdiff_t>(n_out);
      }
      s.batches = {{.count = 2, .in_dist = in_span, .out_dist = out_span}};
      const fftn::plan<double, cd> plain(s, {});
      for (const fftn::callback_mode mode :
           {fftn::callback_mode::modify, fftn::callback_mode::produce}) {
        const fftn::plan<double, cd> with(s, {},
                                          {.in = {.call = {.fn = &none_real, .mode = mode}}});
        const std::size_t row = 2 * (dims.back() * sizeof(double) + 256);
        for (const int lanes : {1, 4}) {
          const std::size_t limit = plain.workspace_bytes(lanes) +
                                    (out_stride == 1 ? 0 : static_cast<std::size_t>(lanes) * row);
          EXPECT_LE(with.workspace_bytes(lanes), limit)
              << dims.size() << " axes, output stride " << out_stride << ", " << lanes << " lanes\n"
              << with.describe();
        }
      }
    }
  }
}

std::atomic<int> planning_calls{0};
void counting(void*, const fftn::block_info&, cd*) noexcept { planning_calls.fetch_add(1); }
void counting_real(void*, const fftn::block_info&, double*) noexcept {
  planning_calls.fetch_add(1);
}

// Measured planning of planes and volumes with callbacks times its candidates with stand-ins,
// never calling the application's function; the plan kept calls it.
TEST(CallbackAxes, MeasuredPlanningCallsNoCallback) {
  planning_calls = 0;
  fftn::shape<cd, cd> plane;
  plane.axes = {{.n = 64}, {.n = 48}};
  plane.batches = {{.count = 2}};
  const fftn::plan<cd, cd> p(
      plane, {.effort = fftn::planning::measure},
      {.in = {.call = {.fn = &counting}}, .out = {.call = {.fn = &counting}}});
  fftn::shape<double, cd> volume;
  volume.axes = {{.n = 8}, {.n = 12}, {.n = 20}};
  const fftn::plan<double, cd> real_volume(
      volume, {.effort = fftn::planning::quick},
      {.in = {.call = {.fn = &counting_real}}, .out = {.call = {.fn = &counting}}});
  EXPECT_EQ(planning_calls.load(), 0);
  std::vector<cd> x(64 * 48 * 2, cd(1.0, 0.0));
  std::vector<cd> y(64 * 48 * 2);
  fftn::workspace ws(p);
  p.execute(x.data(), y.data(), ws, {});
  EXPECT_GT(planning_calls.load(), 0);
}

} // namespace
