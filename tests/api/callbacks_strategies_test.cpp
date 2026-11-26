// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Block callbacks in every row strategy (fftn/callback.hpp, DESIGN.md 12.3; task 5.3b): the lanes
// over rows, the grid, the depth-first chain, Bluestein's depth-first convolution, batch leaves,
// the lanes over adjacent transforms, and real rows (r2c and c2r, even and odd: the chain's edge
// phases, the lanes, the half-length grid and depth-first chain), each forced and by the rules,
// against the quadruple-precision reference of "the input callback's loop, the transform, the
// output callback's loop". The callbacks are nonlinear (cubic) and depend on the coordinates; they
// check every block's geometry (lines along transforms, within the side and the batch, never an
// idle lane, whole rows up to the block size and as many as fit it, a longer row one a call),
// `member` and their scratch size, and count every element they are passed (each exactly
// once). Every mode, beside factors, scales, accumulation and stored runs, in and out of
// place, strided rows with canaries in their gaps, the input left as it was; serial, pieces of the
// batch, teams (all and own) and an executor bitwise alike; describe() per strategy (where, the
// block shape, the sweep beyond a block); measured planning with stand-ins that never call the
// application's function; and no strategy forced by a callback except a consumed output's.

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
#include <memory>
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

enum class in_kind : unsigned char { none, modify, produce };
enum class out_kind : unsigned char { none, modify, consume };

struct run_spec {
  std::size_t logical = 0;
  std::size_t stored = 0;
  std::size_t count = 0;
};

struct cb_case {
  const char* name = "";
  transform_kind kind = transform_kind::c2c;
  std::size_t n = 1;
  std::size_t count = 1;
  in_kind in = in_kind::none;
  out_kind out = out_kind::none;
  fftn::direction dir = fftn::direction::forward;
  double scale = 1.0;                        // options::scale
  double in_scale = 1.0;                     // edges.in.scale
  factor_kind in_factor = factor_kind::none; // stride 1 along the row, per row
  factor_kind out_factor = factor_kind::none;
  bool accumulate = false;
  bool in_place = false;
  std::ptrdiff_t in_stride = 1; // rows are (stored extent) * stride + 1 apart: a gap
  std::ptrdiff_t out_stride = 1;
  bool adjacent = false; // rows one element apart (stride count, distance 1), c2c
  std::vector<run_spec> in_runs{};
  std::vector<run_spec> out_runs{};
  std::size_t scratch = 0;
  const d::build_settings* settings = nullptr; // null: the rules
  const char* strategy = nullptr;              // in describe()'s strategy line, where given
  // Where given, the input callback's calls take at least this many rows at once on the strategy
  // named (a block of several rows cut by rows, not by elements).
  std::size_t min_lines = 0;
};

// The functions the callbacks apply, pure per element: nonlinear, and dependent on the
// coordinates (j along the row, t the transform). On a real side the same on real values.
template <class R> R gain_in(std::size_t j, std::size_t t) {
  return static_cast<R>(0.05) * static_cast<R>(1 + (j * 7 + t * 3) % 5);
}
template <class R> R gain_out(std::size_t k, std::size_t t) {
  return static_cast<R>(0.02) * static_cast<R>(1 + (k * 5 + t) % 5);
}
template <class R> std::complex<R> cubic_in(std::complex<R> x, std::size_t j, std::size_t t) {
  const R g = R(1) + gain_in<R>(j, t) * (x.real() * x.real() + x.imag() * x.imag());
  return {x.real() * g, x.imag() * g};
}
template <class R> R cubic_in(R x, std::size_t j, std::size_t t) {
  return x * (R(1) + gain_in<R>(j, t) * x * x);
}
template <class R>
std::complex<R> cubic_out(std::complex<R> y, std::size_t k, std::size_t t, std::size_t n) {
  const R g =
      R(1) + gain_out<R>(k, t) * (y.real() * y.real() + y.imag() * y.imag()) / static_cast<R>(n);
  return {y.real() * g, y.imag() * g};
}
template <class R> R cubic_out(R y, std::size_t k, std::size_t t, std::size_t n) {
  return y * (R(1) + gain_out<R>(k, t) * y * y / static_cast<R>(n));
}
template <class E> E produced(std::size_t j, std::size_t t) {
  const double a = 0.37 * static_cast<double>(j) + 1.1 * static_cast<double>(t);
  if constexpr (std::is_floating_point_v<E>) {
    return static_cast<E>(std::sin(a));
  } else {
    using R = typename E::value_type;
    return {static_cast<R>(std::sin(a)), static_cast<R>(std::cos(0.61 * a - 0.5))};
  }
}

template <class E> struct real_of {
  using type = E;
};
template <class R> struct real_of<std::complex<R>> {
  using type = R;
};

// What a side's callback shares with the test: the side's geometry, the elements it was passed,
// where consumed blocks are captured, and its findings.
template <class E> struct cb_context {
  std::size_t length = 1; // elements along the side's row
  std::size_t count = 1;
  std::size_t n = 1; // the transform's length (the output's gain)
  int lanes = 1;
  std::size_t scratch = 0;
  std::vector<E> captured; // consume: transform t's element k at t length + k
  std::unique_ptr<std::atomic<int>[]> hits;
  std::atomic<int> errors{0};
  std::atomic<int> first_error_line{0};
  std::atomic<std::size_t> widest{0}; // the most lines a call took
  // The plan's callback_block_bytes, where the side has no stored runs (0: blocks unchecked): a
  // row that fits is passed whole, as many rows a call as fit; a longer one in element ranges
  // of at most that size, one row a call (fftn/callback.hpp, "Coverage").
  std::size_t block_bytes = 0;

  void reset() {
    hits = std::make_unique<std::atomic<int>[]>(length * count);
    captured.assign(length * count, E{});
    errors = 0;
    widest = 0;
  }
  void fail(int line) {
    if (errors.fetch_add(1) == 0) {
      first_error_line = line;
    }
  }
  bool check(const fftn::block_info& info) {
    const bool ok =
        info.lines >= 1 && info.line_axis == fftn::along_transforms && info.element_axis == 0 &&
        info.origin.size() == 1 && info.length > 0 && info.origin[0] + info.length <= length &&
        info.transform < count && info.lines <= count - info.transform && info.member >= 0 &&
        info.member < lanes && info.copy == 0 && info.group == 1 && info.group_stride == 0 &&
        info.scratch.size() == scratch && (info.lines > 1 || info.line_stride == 0);
    if (!ok) {
      fail(__LINE__);
      return false;
    }
    if (block_bytes != 0) {
      const std::size_t row_bytes = length * sizeof(E);
      const bool cut_right = row_bytes <= block_bytes
                                 ? info.origin[0] == 0 && info.length == length &&
                                       info.lines * row_bytes <= block_bytes
                                 : info.lines == 1 && info.length * sizeof(E) <= block_bytes;
      if (!cut_right) {
        fail(__LINE__);
      }
    }
    if (scratch != 0 && reinterpret_cast<std::uintptr_t>(info.scratch.data()) % 128 != 0) {
      fail(__LINE__);
    }
    std::size_t seen = widest.load(std::memory_order_relaxed);
    while (info.lines > seen && !widest.compare_exchange_weak(seen, info.lines)) {
    }
    for (std::size_t k = 0; k < info.lines; ++k) {
      for (std::size_t i = 0; i < info.length; ++i) {
        hits[(info.transform + k) * length + info.origin[0] + i].fetch_add(
            1, std::memory_order_relaxed);
      }
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
    for (std::size_t k = 0; k < info.lines; ++k) {
      for (std::size_t i = 0; i < info.length; ++i) {
        f(data[static_cast<std::ptrdiff_t>(i) * info.element_stride +
               static_cast<std::ptrdiff_t>(k) * info.line_stride],
          info.origin[0] + i, info.transform + k);
      }
    }
  }
};

template <class E> void modify_in(void* user, const fftn::block_info& info, E* data) noexcept {
  static_cast<cb_context<E>*>(user)->each(
      info, data, [](E& x, std::size_t j, std::size_t t) { x = cubic_in(x, j, t); });
}
template <class E> void produce_in(void* user, const fftn::block_info& info, E* data) noexcept {
  static_cast<cb_context<E>*>(user)->each(
      info, data, [](E& x, std::size_t j, std::size_t t) { x = produced<E>(j, t); });
}
template <class E> void modify_out(void* user, const fftn::block_info& info, E* data) noexcept {
  auto& ctx = *static_cast<cb_context<E>*>(user);
  const std::size_t n = ctx.n;
  ctx.each(info, data, [n](E& y, std::size_t k, std::size_t t) { y = cubic_out(y, k, t, n); });
}
template <class E> void consume_out(void* user, const fftn::block_info& info, E* data) noexcept {
  auto& ctx = *static_cast<cb_context<E>*>(user);
  ctx.each(info, data,
           [&ctx](E& y, std::size_t k, std::size_t t) { ctx.captured[t * ctx.length + k] = y; });
}

std::size_t stored_extent(const std::vector<run_spec>& runs, std::size_t n) {
  if (runs.empty()) {
    return n;
  }
  std::size_t e = 0;
  for (const run_spec& r : runs) {
    e = std::max(e, r.stored + r.count);
  }
  return e;
}

bool stored_at(const std::vector<run_spec>& runs, std::size_t j, std::size_t& at) {
  if (runs.empty()) {
    at = j;
    return true;
  }
  for (const run_spec& r : runs) {
    if (j >= r.logical && j < r.logical + r.count) {
      at = r.stored + (j - r.logical);
      return true;
    }
  }
  return false;
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

// One case on one target in precision T: In and Out are the sides' element types.
template <class T, transform_kind K> struct cb_run {
  using C = std::complex<T>;
  using In = std::conditional_t<K == transform_kind::r2c, T, C>;
  using Out = std::conditional_t<K == transform_kind::c2r, T, C>;
  static constexpr std::size_t in_width = std::is_same_v<In, T> ? 1 : 2;
  static constexpr std::size_t out_width = std::is_same_v<Out, T> ? 1 : 2;

  cb_case c;
  fftn::isa target;
  std::size_t in_len = 0; // elements along each side's row (logical)
  std::size_t out_len = 0;
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t out_stride = 1;
  std::ptrdiff_t in_dist = 0;
  std::ptrdiff_t out_dist = 0;
  fftn::shape<In, Out> shape;
  fftn::plan_edges<In, Out> edges;
  fftn::options opts;
  std::vector<T> x;    // the input buffer (reals), gaps included
  std::vector<T> init; // the output buffer before execution (in place: the input)
  std::vector<T> fin;  // factors, side length per row (complex: interleaved)
  std::vector<T> fout;
  cb_context<In> in_ctx;
  cb_context<Out> out_ctx;

  cb_run(const cb_case& cc, fftn::isa t, fftn::test::input_source& src) : c(cc), target(t) {
    const std::size_t half = c.n / 2 + 1;
    in_len = K == transform_kind::c2r ? half : c.n;
    out_len = K == transform_kind::r2c ? half : c.n;
    in_stride = c.adjacent ? static_cast<std::ptrdiff_t>(c.count) : c.in_stride;
    out_stride = c.adjacent ? static_cast<std::ptrdiff_t>(c.count) : c.out_stride;
    if (c.in_place) {
      out_stride = in_stride;
    }
    if (c.adjacent) {
      in_dist = 1;
      out_dist = 1;
    } else if (c.in_place && K != transform_kind::c2c) {
      // The in-place real layout: rows of 2(n/2 + 1) reals, n/2 + 1 complex values.
      in_dist = K == transform_kind::r2c ? static_cast<std::ptrdiff_t>(2 * half)
                                         : static_cast<std::ptrdiff_t>(half);
      out_dist = K == transform_kind::r2c ? static_cast<std::ptrdiff_t>(half)
                                          : static_cast<std::ptrdiff_t>(2 * half);
    } else {
      in_dist = static_cast<std::ptrdiff_t>(stored_extent(c.in_runs, in_len)) * in_stride + 1;
      out_dist =
          c.in_place
              ? in_dist
              : static_cast<std::ptrdiff_t>(stored_extent(c.out_runs, out_len)) * out_stride + 1;
    }
    shape.dir = c.dir;
    shape.axes = {{.n = c.n, .in_stride = in_stride, .out_stride = out_stride}};
    shape.batches = {{.count = c.count, .in_dist = in_dist, .out_dist = out_dist}};
    for (const run_spec& r : c.in_runs) {
      shape.runs.push_back({.axis = 0,
                            .where = fftn::side::in,
                            .logical = r.logical,
                            .stored = r.stored,
                            .count = r.count});
    }
    for (const run_spec& r : c.out_runs) {
      shape.runs.push_back({.axis = 0,
                            .where = fftn::side::out,
                            .logical = r.logical,
                            .stored = r.stored,
                            .count = r.count});
    }
    const auto factor = [&](factor_kind k, std::size_t len) -> fftn::factor {
      if (k == factor_kind::none) {
        return {};
      }
      return {.kind = k, .axis_strides = {1}, .batch_strides = {static_cast<std::ptrdiff_t>(len)}};
    };
    edges.in = {.scale = c.in_scale, .mul = factor(c.in_factor, in_len)};
    edges.out = {.mul = factor(c.out_factor, out_len), .accumulate = c.accumulate};
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
    const std::size_t in_elements =
        static_cast<std::size_t>(in_dist) * (c.count - 1) +
        static_cast<std::size_t>(in_stride) * (stored_extent(c.in_runs, in_len) - 1) + 1;
    const std::size_t out_elements =
        static_cast<std::size_t>(out_dist) * (c.count - 1) +
        static_cast<std::size_t>(out_stride) * (stored_extent(c.out_runs, out_len) - 1) + 1;
    // In place both sides share one buffer, long enough for either.
    const std::size_t in_reals = in_elements * in_width;
    const std::size_t out_reals = out_elements * out_width;
    x = src.reals<T>(c.in_place ? std::max(in_reals, out_reals) : in_reals);
    init = c.in_place ? x : src.reals<T>(out_reals);
    // Factors of magnitude about 1, so that the cubic callbacks keep their slopes small.
    const auto factors = [&](factor_kind k, std::size_t len) {
      std::vector<T> f = src.reals<T>(len * c.count * (k == factor_kind::complex ? 2 : 1));
      if (k == factor_kind::real) {
        for (T& v : f) {
          v = static_cast<T>(0.5) + static_cast<T>(0.5) * v;
        }
      }
      return k == factor_kind::none ? std::vector<T>{} : f;
    };
    fin = factors(c.in_factor, in_len);
    fout = factors(c.out_factor, out_len);
    in_ctx.length = in_len;
    out_ctx.length = out_len;
    in_ctx.count = out_ctx.count = c.count;
    in_ctx.n = out_ctx.n = c.n;
    in_ctx.scratch = out_ctx.scratch = c.scratch;
    const std::size_t block = c.settings != nullptr
                                  ? c.settings->profile.callback_block_bytes
                                  : d::build_settings{}.profile.callback_block_bytes;
    in_ctx.block_bytes = c.in_runs.empty() ? block : 0;
    out_ctx.block_bytes = c.out_runs.empty() ? block : 0;
  }

  [[nodiscard]] std::size_t in_at(std::size_t j, std::size_t r) const {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(r) * in_dist +
                                    static_cast<std::ptrdiff_t>(j) * in_stride);
  }
  [[nodiscard]] std::size_t out_at(std::size_t k, std::size_t r) const {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(r) * out_dist +
                                    static_cast<std::ptrdiff_t>(k) * out_stride);
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
  [[nodiscard]] cd factor_of(const std::vector<T>& f, factor_kind k, std::size_t len, std::size_t j,
                             std::size_t r) const {
    if (k == factor_kind::complex) {
      return {static_cast<double>(f[2 * (r * len + j)]),
              static_cast<double>(f[2 * (r * len + j) + 1])};
    }
    return k == factor_kind::real ? cd(static_cast<double>(f[r * len + j]), 0.0) : cd(1.0, 0.0);
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
    const auto side = [&](const auto& ctx, bool used, const std::vector<run_spec>& runs,
                          std::size_t len, const char* name) {
      EXPECT_EQ(ctx.errors.load(), 0) << what << ": a " << name << " callback's check failed (line "
                                      << ctx.first_error_line.load() << ")";
      std::size_t wrong = 0;
      for (std::size_t r = 0; r < c.count; ++r) {
        for (std::size_t j = 0; j < len; ++j) {
          std::size_t at = 0;
          if (ctx.hits[r * len + j].load() != (used && stored_at(runs, j, at) ? 1 : 0)) {
            ++wrong;
          }
        }
      }
      EXPECT_EQ(wrong, 0U) << what << ": " << name << " elements not passed exactly once";
    };
    side(in_ctx, c.in != in_kind::none, c.in_runs, in_len, "input");
    side(out_ctx, c.out != out_kind::none, c.out_runs, out_len, "output");
  }

  // The reference of row r's input after the input callback's loop, the factor, as complex
  // values of the side.
  [[nodiscard]] std::vector<cd> logical_input(std::size_t r) const {
    std::vector<cd> row(in_len);
    for (std::size_t j = 0; j < in_len; ++j) {
      std::size_t at = 0;
      if (!stored_at(c.in_runs, j, at)) {
        continue;
      }
      In v{};
      if (c.in == in_kind::produce) {
        v = produced<In>(j, r);
      } else {
        v = in_value(x, in_at(at, r));
        if (c.in == in_kind::modify) {
          v = cubic_in(v, j, r);
        }
      }
      row[j] = widen(v) * factor_of(fin, c.in_factor, in_len, j, r);
    }
    return row;
  }

  void check(const std::vector<T>& y, const std::string& what) const {
    const double scale = c.scale * c.in_scale;
    const int sign = static_cast<int>(c.dir);
    std::vector<bool> described(y.size(), false);
    fftn::test::outcome o;
    for (std::size_t r = 0; r < c.count; ++r) {
      const std::vector<cd> row = logical_input(r);
      std::vector<cd> exact(out_len);
      if constexpr (K == transform_kind::c2c) {
        const auto e = fftn::test::reference::c2c(sign, row);
        for (std::size_t k = 0; k < out_len; ++k) {
          exact[k] = {e[k].re.hi, e[k].im.hi};
        }
      } else if constexpr (K == transform_kind::r2c) {
        std::vector<double> re(c.n);
        for (std::size_t j = 0; j < c.n; ++j) {
          re[j] = row[j].real();
        }
        const auto e = fftn::test::reference::r2c(re);
        for (std::size_t k = 0; k < out_len; ++k) {
          exact[k] = {e[k].re.hi, e[k].im.hi};
        }
      } else {
        const auto e = fftn::test::reference::c2r(c.n, row);
        for (std::size_t k = 0; k < out_len; ++k) {
          exact[k] = {e[k].hi, 0.0};
        }
      }
      std::vector<fftn::test::exact_complex> want;
      std::vector<C> got;
      for (std::size_t k = 0; k < out_len; ++k) {
        std::size_t at = 0;
        if (!stored_at(c.out_runs, k, at)) {
          continue;
        }
        cd v = exact[k] * factor_of(fout, c.out_factor, out_len, k, r) * scale;
        if (c.out == out_kind::modify) {
          if constexpr (out_width == 1) {
            v = {cubic_out(v.real(), k, r, c.n), 0.0};
          } else {
            v = cubic_out(v, k, r, c.n);
          }
        }
        if (c.out == out_kind::consume) {
          want.push_back(fftn::test::exact_from(v));
          got.push_back(C(widen(out_ctx.captured[r * out_len + k])));
          continue;
        }
        if (c.accumulate) {
          v += widen(out_value(init, out_at(at, r)));
        }
        want.push_back(fftn::test::exact_from(v));
        got.push_back(C(widen(out_value(y, out_at(at, r)))));
        for (std::size_t q = 0; q < out_width; ++q) {
          described[out_at(at, r) * out_width + q] = true;
        }
      }
      auto tol = fftn::test::derived_tolerance<T>(c.n, fftn::test::fftn_bound(c.n)) +
                 fftn::test::input_rounding<T>() * 4;
      if (c.out == out_kind::modify) {
        tol = tol * 3;
      }
      o.within(fftn::test::compare<T>(std::span<const C>(got), want), tol,
               what + ", row " + std::to_string(r));
    }
    EXPECT_TRUE(o.passed()) << o.detail();
    if (!c.in_place) {
      std::size_t touched = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        if (!described[i] && std::memcmp(&y[i], &init[i], sizeof(T)) != 0) {
          ++touched;
        }
      }
      EXPECT_EQ(touched, 0U) << what << ": elements outside the rows (or the runs) written";
    }
  }
};

template <class T, transform_kind K> void run_case(const cb_case& cc, fftn::isa t, bool threads) {
  using R = cb_run<T, K>;
  fftn::test::input_source src(0x5B3ULL + cc.n * 131 + cc.count);
  R run(cc, t, src);
  const d::build_settings settings = cc.settings != nullptr ? *cc.settings : d::build_settings{};
  const auto core = d::make_plan_core_with_edges(d::view_of(run.shape), run.opts,
                                                 d::view_of(run.edges), settings);
  const std::string text = d::describe(*core);
  const std::string what = std::string(cc.name) + ", " + fftn::isa_name(t) +
                           (sizeof(T) == 8 ? " double" : " float") + "\n" + text;
  SCOPED_TRACE(what);
  EXPECT_NE(text.find("\ncallbacks:"), std::string::npos);
  if (cc.strategy != nullptr) {
    const std::size_t line = text.find("\nstrategy: ");
    EXPECT_NE(text.find(cc.strategy, line), std::string::npos)
        << "not on the strategy " << cc.strategy;
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
  if (cc.min_lines != 0 && cc.strategy != nullptr &&
      text.find(cc.strategy, text.find("\nstrategy: ")) != std::string::npos) {
    EXPECT_GE(run.in_ctx.widest.load(), cc.min_lines) << "blocks of several rows not passed";
  }
  // Rows too long for the quadruple-precision reference are checked by coverage and bitwise
  // agreement of the execution forms only.
  if (cc.n <= 2100) {
    run.check(serial, "serial");
  }
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

template <class T> void run_any(const cb_case& c, fftn::isa t, bool threads) {
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
  case transform_kind::r2r: // real-to-real plans refuse callbacks (callbacks_test.cpp)
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
  d::build_settings lanes;
  d::build_settings grid;
  d::build_settings depth;
  d::build_settings bluestein;
  d::build_settings leaves;
  d::build_settings adjacent;
  d::build_settings chain;
  d::build_settings small_blocks;   // the rules, blocks of 1 KiB: rows of several blocks
  d::build_settings lanes_small;    // the lanes, blocks of 1 KiB: a lanes block in several calls
  d::build_settings adjacent_small; // the adjacent transforms, blocks of 4 KiB
  d::build_settings copy_out;       // the chain's copy into the output (callback_copy_out)
  knobs() {
    lanes.strategy = d::strategy_choice::lanes;
    grid.strategy = d::strategy_choice::grid;
    depth.strategy = d::strategy_choice::depth;
    bluestein = depth;
    bluestein.rader = 0;
    bluestein.bluestein_length = 4096;
    leaves.batch_leaves = d::batch_leaf_choice::own;
    adjacent.profile.adjacent_lanes = true;
    adjacent.profile.adjacent_min_bytes = 0;
    adjacent.profile.adjacent_team_chain = false;
    adjacent.profile.column_team_bytes = 0;
    adjacent.profile.column_wide_bytes = 0;
    chain.strategy = d::strategy_choice::chain;
    chain.batch_leaves = d::batch_leaf_choice::never;
    small_blocks.profile.callback_block_bytes = 1024;
    lanes_small = lanes;
    lanes_small.profile.callback_block_bytes = 1024;
    adjacent_small = adjacent;
    adjacent_small.profile.callback_block_bytes = 4096;
    copy_out.strategy = d::strategy_choice::chain;
    copy_out.batch_leaves = d::batch_leaf_choice::never;
    copy_out.callback_copy_out = true;
    // The walked form, whose blocks are single rows (no row blocks on any profile).
    chain.profile.edge_batched_forms = d::edge_forms_reach::none;
    copy_out.profile.edge_batched_forms = d::edge_forms_reach::none;
  }
};

const knobs& knob() {
  static const knobs k;
  return k;
}

constexpr const char* row_lanes = "lanes (blocks of";
constexpr const char* c2c_grid = "grid (Cooley-Tukey";
constexpr const char* c2c_depth = "depth-first chain (the sub-rows";
constexpr const char* half_grid = "grid for the half-length";
constexpr const char* half_depth = "depth-first chain for the half-length";
constexpr const char* bluestein = "two depth-first chains";
constexpr const char* adjacent = "adjacent transforms";
constexpr const char* chain = "strategy: chain";

std::vector<cb_case> strategy_cases() {
  const knobs& k = knob();
  return {
      // The lanes over rows: the callbacks on the block's rows.
      {.name = "c2c lanes, both modify, scale",
       .n = 16,
       .count = 7,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .scale = 0.5,
       .scratch = 48,
       .settings = &k.lanes,
       .strategy = row_lanes},
      {.name = "c2c lanes, produce and consume",
       .n = 12,
       .count = 9,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .settings = &k.lanes,
       .strategy = row_lanes},
      {.name = "c2c lanes, consume only, strided",
       .n = 20,
       .count = 6,
       .out = out_kind::consume,
       .in_stride = 2,
       .settings = &k.lanes,
       .strategy = row_lanes},
      {.name = "c2c lanes, modify beside factors and accumulation",
       .n = 16,
       .count = 5,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_scale = 0.25,
       .in_factor = cplx,
       .out_factor = real,
       .accumulate = true,
       .settings = &k.lanes,
       .strategy = row_lanes},
      {.name = "c2c lanes in place",
       .n = 20,
       .count = 6,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_factor = real,
       .in_place = true,
       .settings = &k.lanes,
       .strategy = row_lanes},
      {.name = "r2c lanes, both modify",
       .kind = r2c,
       .n = 32,
       .count = 6,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .settings = &k.lanes,
       .strategy = row_lanes},
      {.name = "r2c lanes, consume, input factor",
       .kind = r2c,
       .n = 24,
       .count = 5,
       .out = out_kind::consume,
       .in_factor = real,
       .settings = &k.lanes},
      {.name = "c2r lanes, produce, accumulate",
       .kind = c2r,
       .n = 24,
       .count = 5,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .accumulate = true,
       .settings = &k.lanes,
       .strategy = row_lanes},
      {.name = "c2r lanes, modify, consume",
       .kind = c2r,
       .n = 16,
       .count = 4,
       .in = in_kind::modify,
       .out = out_kind::consume,
       .settings = &k.lanes},
      // The grid and the depth-first chains: a callback stage over the row before or after.
      {.name = "c2c grid, both modify",
       .n = 1024,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .scale = 0.25,
       .scratch = 16,
       .settings = &k.grid,
       .strategy = c2c_grid},
      {.name = "c2c grid, produce, accumulate beside an output factor",
       .n = 2048,
       .count = 1,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .out_factor = cplx,
       .accumulate = true,
       .settings = &k.grid,
       .strategy = c2c_grid},
      {.name = "c2c grid in place, both modify",
       .n = 2048,
       .count = 1,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_place = true,
       .settings = &k.grid,
       .strategy = c2c_grid},
      {.name = "c2c grid, consumed: the chain",
       .n = 1024,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::consume,
       .settings = &k.grid,
       .strategy = chain},
      {.name = "c2c depth-first, both modify, input factor",
       .n = 1024,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_factor = cplx,
       .settings = &k.depth,
       .strategy = c2c_depth},
      {.name = "c2c depth-first in place, produce",
       .n = 2048,
       .count = 1,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .in_place = true,
       .settings = &k.depth,
       .strategy = c2c_depth},
      {.name = "Bluestein depth-first, both modify, accumulate",
       .n = 1031,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .accumulate = true,
       .settings = &k.bluestein,
       .strategy = bluestein},
      {.name = "Bluestein depth-first, produce, backward",
       .n = 1031,
       .count = 1,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .dir = fftn::direction::backward,
       .settings = &k.bluestein,
       .strategy = bluestein},
      {.name = "r2c half-length grid, both modify",
       .kind = r2c,
       .n = 2048,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .settings = &k.grid,
       .strategy = half_grid},
      {.name = "c2r half-length grid, output modify, accumulate",
       .kind = c2r,
       .n = 2048,
       .count = 1,
       .out = out_kind::modify,
       .accumulate = true,
       .settings = &k.grid,
       .strategy = half_grid},
      {.name = "r2c half-length depth-first, produce, output modify",
       .kind = r2c,
       .n = 2048,
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .settings = &k.depth,
       .strategy = half_depth},
      {.name = "c2r half-length depth-first, output modify",
       .kind = c2r,
       .n = 2048,
       .count = 1,
       .out = out_kind::modify,
       .scale = 0.5,
       .settings = &k.depth,
       .strategy = half_depth},
      {.name = "r2c half-length grid, consumed: the chain",
       .kind = r2c,
       .n = 2048,
       .count = 1,
       .out = out_kind::consume,
       .settings = &k.grid,
       .strategy = chain},
      // Batch leaves (where the target's precision has them), one row at a time.
      {.name = "batch leaves, both modify",
       .n = 8,
       .count = 16,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .settings = &k.leaves},
      {.name = "batch leaves, produce and consume",
       .n = 16,
       .count = 12,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .settings = &k.leaves},
      // Rows one element apart along the batch: the lanes over adjacent transforms, the
      // callbacks on the block's transforms.
      {.name = "rows as adjacent transforms, both modify",
       .n = 128,
       .count = 32,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .adjacent = true,
       .scratch = 8,
       .settings = &k.adjacent,
       .strategy = adjacent},
      {.name = "rows as adjacent transforms, produce, output factor",
       .n = 112,
       .count = 16,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .out_factor = cplx,
       .adjacent = true,
       .settings = &k.adjacent,
       .strategy = adjacent},
      {.name = "rows as adjacent transforms in place",
       .n = 120,
       .count = 12,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_place = true,
       .adjacent = true,
       .settings = &k.adjacent,
       .strategy = adjacent},
      // Blocks of several lines over the block size: whole rows, fewer a call, never element
      // ranges of each (rows of 256 or 128 bytes, a block of 1 KiB; 2 KiB or 1 KiB, 4 KiB).
      {.name = "c2c lanes, a block's rows in several calls",
       .n = 16,
       .count = 37,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .settings = &k.lanes_small,
       .strategy = row_lanes,
       .min_lines = 2},
      {.name = "c2c lanes, produce and consume, a block's rows in several calls",
       .n = 8,
       .count = 21,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .settings = &k.lanes_small,
       .strategy = row_lanes,
       .min_lines = 2},
      {.name = "rows as adjacent transforms, a block's transforms in several calls",
       .n = 128,
       .count = 32,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .adjacent = true,
       .settings = &k.adjacent_small,
       .strategy = adjacent,
       .min_lines = 2},
      // Real rows on the chain: the edge phases.
      {.name = "r2c chain, both modify, even",
       .kind = r2c,
       .n = 64,
       .count = 3,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .settings = &k.chain},
      {.name = "r2c chain, odd, produce and consume",
       .kind = r2c,
       .n = 45,
       .count = 3,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .scratch = 24,
       .settings = &k.chain},
      {.name = "r2c chain, factors and accumulation",
       .kind = r2c,
       .n = 40,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_factor = real,
       .out_factor = cplx,
       .accumulate = true,
       .settings = &k.chain},
      {.name = "r2c chain in place, both modify",
       .kind = r2c,
       .n = 30,
       .count = 3,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_place = true,
       .settings = &k.chain},
      {.name = "c2r chain, odd, both modify, scale",
       .kind = c2r,
       .n = 33,
       .count = 3,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .scale = 0.25,
       .settings = &k.chain},
      {.name = "c2r chain, even, produce and consume",
       .kind = c2r,
       .n = 48,
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .settings = &k.chain},
      {.name = "c2r chain in place, input modify, factor",
       .kind = c2r,
       .n = 28,
       .count = 2,
       .in = in_kind::modify,
       .in_factor = real,
       .in_place = true,
       .settings = &k.chain},
      {.name = "c2r chain, strided, output modify",
       .kind = c2r,
       .n = 20,
       .count = 3,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_stride = 2,
       .out_stride = 3,
       .settings = &k.chain},
      {.name = "r2c chain, input runs, modify",
       .kind = r2c,
       .n = 32,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_runs = {{.logical = 0, .stored = 0, .count = 10},
                   {.logical = 20, .stored = 10, .count = 6}}},
      {.name = "c2r chain, input runs on the bins, produce; output runs, consume",
       .kind = c2r,
       .n = 36,
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .in_runs = {{.logical = 2, .stored = 0, .count = 9}},
       .out_runs = {{.logical = 5, .stored = 0, .count = 20}}},
      {.name = "r2c chain, output runs, modify and accumulate",
       .kind = r2c,
       .n = 40,
       .count = 2,
       .out = out_kind::modify,
       .accumulate = true,
       .out_runs = {{.logical = 0, .stored = 0, .count = 6},
                    {.logical = 15, .stored = 6, .count = 5}}},
      // The chain's copy into the output (X66's arm).
      {.name = "c2c chain, the copy into the output",
       .n = 60,
       .count = 3,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_factor = real,
       .settings = &k.copy_out,
       .strategy = chain},
      {.name = "c2c chain, produced into the output, strided",
       .n = 24,
       .count = 3,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .in_stride = 2,
       .out_stride = 3,
       .settings = &k.copy_out,
       .strategy = chain},
      // Small blocks: rows of several blocks (one more sweep), the lanes' rows cut too.
      {.name = "c2c rows of several blocks",
       .n = 300,
       .count = 3,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .settings = &k.small_blocks},
      {.name = "r2c rows of several blocks",
       .kind = r2c,
       .n = 512,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::consume,
       .settings = &k.small_blocks},
  };
}

// The rules' strategies on rows of many sizes and kinds.
std::vector<cb_case> rules_cases() {
  std::vector<cb_case> v;
  for (const transform_kind kind : {c2c, r2c, c2r}) {
    for (const std::size_t n : {std::size_t{8}, std::size_t{30}, std::size_t{256},
                                std::size_t{2048}, std::size_t{1} << 15U}) {
      const std::size_t count = n >= 2048 ? 2 : 9;
      v.push_back({.name = "the rules, both modify",
                   .kind = kind,
                   .n = n,
                   .count = count,
                   .in = in_kind::modify,
                   .out = out_kind::modify});
      v.push_back({.name = "the rules, produce and consume",
                   .kind = kind,
                   .n = n + 1,
                   .count = count,
                   .in = in_kind::produce,
                   .out = out_kind::consume,
                   .scratch = 32});
    }
  }
  return v;
}

template <class T> class CallbackStrategies : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(CallbackStrategies, precisions);

TYPED_TEST(CallbackStrategies, EveryStrategyForcedOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const cb_case& c : strategy_cases()) {
      run_any<TypeParam>(c, t, false);
    }
  }
}

TYPED_TEST(CallbackStrategies, TeamsAndExecutorsAgreeWithSerial) {
  for (const cb_case& c : strategy_cases()) {
    run_any<TypeParam>(c, fftn::isa::best, true);
  }
}

TYPED_TEST(CallbackStrategies, TheRulesStrategies) {
  for (const cb_case& c : rules_cases()) {
    run_any<TypeParam>(c, fftn::isa::best, true);
  }
}

// A callback forces no strategy: the plan with callbacks takes the strategy of the plan without
// them (forced or by the rules), except where the output is consumed and that strategy writes
// the output as its working memory, which describe() says.
TEST(CallbackStrategies, NoStrategyForcedByACallback) {
  using C = cd;
  const knobs& k = knob();
  struct probe {
    fftn::shape<C, C> shape;
    const d::build_settings* settings;
  };
  const auto rows = [](std::size_t n, std::size_t count) {
    return fftn::rows_c2c<double>(n, count, fftn::direction::forward);
  };
  const std::vector<probe> probes{
      {rows(16, 7), &k.lanes},       {rows(1024, 2), &k.grid},      {rows(1024, 2), &k.depth},
      {rows(1031, 2), &k.bluestein}, {rows(8, 16), &k.leaves},      {rows(64, 3), nullptr},
      {rows(4096, 2), nullptr},      {rows(1U << 16U, 1), nullptr},
  };
  const auto none = [](const fftn::block_info&, C*) noexcept {};
  using F = decltype(none);
  fftn::plan_edges<C, C> modify;
  modify.in.call = {.fn = fftn::callback_of<const F, C>()};
  modify.out.call = {.fn = fftn::callback_of<const F, C>()};
  for (const probe& p : probes) {
    const d::build_settings s = p.settings != nullptr ? *p.settings : d::build_settings{};
    const auto plain = d::make_plan_core(d::view_of(p.shape), {}, s);
    const auto with = d::make_plan_core_with_edges(d::view_of(p.shape), {}, d::view_of(modify), s);
    // Batch leaves with callbacks take a row at a time between the callback stages, as with
    // the edges' separate passes, where the profile gives such plans no row blocks (the walked
    // form): still batch leaves, which describe() says.
    const bool leaves_one_row =
        plain->strategy_name == "the chain on batch leaves" &&
        with->strategy_name == "the chain, its batch leaves one row at a time";
    if (leaves_one_row) {
      EXPECT_NE(d::describe(*with).find("batch leaves, one row at a time between the edge passes"),
                std::string::npos)
          << d::describe(*with);
      continue;
    }
    EXPECT_EQ(with->strategy_name, plain->strategy_name) << d::describe(*with);
    EXPECT_EQ(d::describe(*with).find("not the strategy the rules would choose"), std::string::npos)
        << d::describe(*with);
  }
  // A consumed output on the grid: the chain, and describe() says why.
  fftn::plan_edges<C, C> consume;
  consume.out.call = {.fn = fftn::callback_of<const F, C>(), .mode = fftn::callback_mode::consume};
  const auto grid_rows = rows(1024, 2);
  const auto with =
      d::make_plan_core_with_edges(d::view_of(grid_rows), {}, d::view_of(consume), k.grid);
  EXPECT_NE(d::describe(*with).find("a consumed output takes no grid, which writes the output as "
                                    "its working memory"),
            std::string::npos)
      << d::describe(*with);
}

// describe() per strategy: where the callbacks run, the block shape, the copy, and the sweep
// beyond a block.
TEST(CallbackStrategies, DescribeNamesWhereAndTheBlocks) {
  using C = cd;
  const knobs& k = knob();
  const auto none = [](const fftn::block_info&, C*) noexcept {};
  using F = decltype(none);
  fftn::plan_edges<C, C> both;
  both.in.call = {.fn = fftn::callback_of<const F, C>()};
  both.out.call = {.fn = fftn::callback_of<const F, C>()};
  struct expect {
    fftn::shape<C, C> shape;
    d::build_settings settings;
    std::vector<const char*> words;
  };
  d::build_settings small = k.grid;
  small.profile.callback_block_bytes = 2048;
  d::build_settings lanes = k.lanes;
  lanes.profile.callback_block_bytes = 1U << 20U;
  const std::vector<expect> cases{
      {fftn::rows_c2c<double>(16, 7, fftn::direction::forward),
       lanes,
       {"the input's (modify) on each row in the output, which the strategy then transforms in "
        "place, after a copy of the input, before the lanes, blocks of up to",
        "rows together, whole",
        "the output's (modify) on each row in the output, after the lanes"}},
      {fftn::rows_c2c<double>(1024, 2, fftn::direction::forward),
       small,
       {"before the row's strategy, blocks of up to 128 elements of one row, one more sweep over "
        "each row (longer than a block)",
        "after the row's strategy"}},
      {fftn::rows_c2c<double>(64, 3, fftn::direction::forward),
       k.chain,
       {"the input's (modify) on each row in the workspace block, after a copy of the input, "
        "blocks of one whole row"}},
      {fftn::rows_c2c<double>(64, 3, fftn::direction::forward),
       k.copy_out,
       {"the input's (modify) on each row in the output, which the chain then transforms in "
        "place, after a copy of the input"}},
  };
  for (const expect& e : cases) {
    const auto core =
        d::make_plan_core_with_edges(d::view_of(e.shape), {}, d::view_of(both), e.settings);
    const std::string text = d::describe(*core);
    for (const char* w : e.words) {
      EXPECT_NE(text.find(w), std::string::npos) << "no '" << w << "' in\n" << text;
    }
  }
  // A lanes block over the block size is cut by rows (rows of 256 bytes, a block of 1 KiB): up
  // to 4 whole rows a call, and no sweep, which only a row longer than a block takes.
  const auto lanes_rows = fftn::rows_c2c<double>(16, 64, fftn::direction::forward);
  const auto core =
      d::make_plan_core_with_edges(d::view_of(lanes_rows), {}, d::view_of(both), k.lanes_small);
  const std::string text = d::describe(*core);
  if (text.find(row_lanes, text.find("\nstrategy: ")) != std::string::npos) {
    EXPECT_NE(text.find("before the lanes, blocks of up to 4 rows together, whole"),
              std::string::npos)
        << text;
  }
  EXPECT_EQ(text.find("one more sweep"), std::string::npos) << text;
}

// Measured planning times its candidates with stand-ins that do nothing: the application's
// function is never called during construction, and the plan kept calls it on execution.
std::atomic<int> planning_calls{0};
void counting(void* /*user*/, const fftn::block_info& /*info*/, cd* /*data*/) noexcept {
  planning_calls.fetch_add(1);
}
void counting_real(void* /*user*/, const fftn::block_info& /*info*/, double* /*data*/) noexcept {
  planning_calls.fetch_add(1);
}

TEST(CallbackStrategies, MeasuredPlanningCallsNoCallback) {
  using C = cd;
  planning_calls = 0;
  // Measure's own budget, room for more than one candidate anywhere (planning_test.cpp,
  // BudgetDefaultsAndValidationDoNotDependOnTheEffort): the stand-ins run in every candidate timed.
  const fftn::plan<C, C> p(fftn::rows_c2c<double>(2048, 4, fftn::direction::forward),
                           {.effort = fftn::planning::measure},
                           {.in = {.call = {.fn = &counting}}, .out = {.call = {.fn = &counting}}});
  const fftn::plan<double, C> real_rows(
      fftn::rows_r2c<double>(1000, 3), {.effort = fftn::planning::quick},
      {.in = {.call = {.fn = &counting_real}}, .out = {.call = {.fn = &counting}}});
  EXPECT_EQ(planning_calls.load(), 0);
  // More than one candidate timed: the rules' plan and at least one other, all with stand-ins.
  const std::string text = p.describe();
  const std::string measured = "planning: measured (";
  const std::size_t at = text.find(measured);
  ASSERT_NE(at, std::string::npos) << text;
  EXPECT_GE(std::stoul(text.substr(at + measured.size())), 2UL) << text;
  std::vector<C> x(2048 * 4, C(1.0, 0.0));
  std::vector<C> y(2048 * 4);
  fftn::workspace ws(p);
  p.execute(x.data(), y.data(), ws, {});
  EXPECT_GT(planning_calls.load(), 0);
}

// Wisdom is keyed by the callbacks' positions, modes and scratch, not their functions: a plan
// with another function finds the entry, one with another mode does not.
TEST(CallbackStrategies, WisdomKeyedByTheCallbacks) {
  using C = cd;
  const auto rows = fftn::rows_c2c<double>(4096, 2, fftn::direction::forward);
  fftn::wisdom w;
  const fftn::plan<C, C> first(rows, {.effort = fftn::planning::quick, .measure_seconds = 0.1},
                               {.out = {.call = {.fn = &counting}}});
  w.add(first);
  const auto other = [](const fftn::block_info&, C*) noexcept {};
  using F = decltype(other);
  const fftn::plan<C, C> again(rows, {.effort = fftn::planning::quick, .use = &w},
                               {.out = {.call = {.fn = fftn::callback_of<const F, C>()}}});
  EXPECT_NE(again.describe().find("planning: from wisdom"), std::string::npos) << again.describe();
  const fftn::plan<C, C> consumed(
      rows, {.effort = fftn::planning::quick, .measure_seconds = 0.1, .use = &w},
      {.out = {.call = {.fn = &counting, .mode = fftn::callback_mode::consume}}});
  EXPECT_EQ(consumed.describe().find("planning: from wisdom"), std::string::npos)
      << consumed.describe();
}

} // namespace
