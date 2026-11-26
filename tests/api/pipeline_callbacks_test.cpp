// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Block callbacks on row pipelines (fftn/pipeline.hpp, fftn/callback.hpp, DESIGN.md 12.3; task
// 5.3d): the prologue's (modify, produce), the middle's (modify) and the epilogue's (modify,
// consume) against the forward transform, then the user's callback, then the inverse transform,
// the transforms in quadruple precision and the callbacks' functions in double between them,
// within the derived tolerance. The callbacks are nonlinear (cubic) and depend on the coordinates
// and, on the epilogue, on the copy. They check every block's geometry (the middle's is the whole
// line), the element's address where the block is the caller's output, `member` and their
// scratch, and count every element they are passed: each input and output element (each stored
// one, with runs) exactly once per line and copy, each bin of the spectrum exactly once per line.
// A consumed output receives each line's copies in copy order. Beside factors (fused and in
// passes), the two scale products around a middle callback, accumulation, stored runs, strides,
// in place, fan-out and the depth-first pair (forced), for <C, C, C> and <T, C, T> on every
// instruction set in both precisions; serial, pieces of the batch, teams (completion::all and own)
// and an executor bitwise alike. A convolution nested in a middle, on the callback's scratch,
// bitwise under teams and an executor. Spectral pairs inverse first (task 5.3e): <C, C, C> with
// the backward transform first and <C, T, C> (c2r, a real middle, r2c), with and without
// callbacks, bands of the half spectrum, fan-out and the depth-first pair (forced), likewise.
// Expectations hold on every target: none assumes a target's radices or strategy unless the test
// forces it.

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
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

enum class pro_cb : unsigned char { off, modify, produce };
enum class epi_cb : unsigned char { off, modify, consume };

struct run_spec {
  std::size_t logical = 0;
  std::size_t stored = 0;
  std::size_t count = 0;
};

struct pcb_case {
  const char* name = "";
  std::size_t n = 16;
  std::size_t count = 3;
  std::size_t fanout = 1;
  pro_cb pro = pro_cb::off;
  bool mid = true; // a modify callback on the middle
  epi_cb epi = epi_cb::off;
  double pro_scale = 1.0;
  double mid_scale = 0.0; // 0: 1/n
  double epi_scale = 1.0;
  double options_scale = 1.0;
  factor_kind pro_f = none; // stride 1, one for every line
  factor_kind mid_f = cplx; // the kernel: stride mid_axis, one for every line, one per copy
  std::ptrdiff_t mid_axis = 1;
  factor_kind epi_f = none; // stride 1, one for every line and copy
  bool accumulate = false;
  bool in_place = false;
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t out_stride = 1;
  std::vector<run_spec> in{};
  std::vector<run_spec> out{};
  std::size_t scratch = 0;
  bool backward = false; // <C, C, C>: the inverse transform first (task 5.3e)
  bool per_row = false; // a kernel per line (the middle's batch stride)
  // With a consumed fan-out: fanout_out_dist this value, which a store would refuse (0, or copies
  // overlapping lines), where `odd_copies` is set (DESIGN.md 18.2, "Callbacks" 14).
  bool odd_copies = false;
  std::ptrdiff_t consumed_out_dist = 0;
};

// The callbacks' functions, in the precision they run in: pure per element (the middle's per
// bin), nonlinear and dependent on the coordinates (j or the bin b along the line, t the line, k
// the copy).
template <class R> R gain_in(std::size_t j, std::size_t t) {
  return static_cast<R>(0.05) * static_cast<R>(1 + (j * 7 + t * 3) % 5);
}
template <class R> std::complex<R> cubic_in(std::complex<R> x, std::size_t j, std::size_t t) {
  const R g = R(1) + gain_in<R>(j, t) * (x.real() * x.real() + x.imag() * x.imag());
  return {x.real() * g, x.imag() * g};
}
template <class R> R cubic_in(R x, std::size_t j, std::size_t t) {
  return x * (R(1) + gain_in<R>(j, t) * x * x);
}
template <class E> E produced(std::size_t j, std::size_t t) {
  // Never zero at j = t = 0, so that a line of one point has a reference to compare with.
  const double a = 0.37 * static_cast<double>(j) + 1.1 * static_cast<double>(t) + 0.3;
  if constexpr (std::is_floating_point_v<E>) {
    return static_cast<E>(std::sin(a));
  } else {
    using R = typename E::value_type;
    return {static_cast<R>(std::sin(a)), static_cast<R>(std::cos(0.61 * a - 0.5))};
  }
}
// The middle's: Y (1 + c |Y|^2 / n), c from 0.01 to 0.05, a slope below 2.5 for the spectra of
// inputs of magnitude 1.
template <class R>
std::complex<R> cubic_mid(std::complex<R> y, std::size_t b, std::size_t t, std::size_t n) {
  const R c = static_cast<R>(0.01) * static_cast<R>(1 + (b * 3 + t * 5) % 5);
  const R g = R(1) + c * (y.real() * y.real() + y.imag() * y.imag()) / static_cast<R>(n);
  return {y.real() * g, y.imag() * g};
}
// A real middle's (<C, T, C>, task 5.3e): y (1 + c y^2 / n).
template <class R> R cubic_mid(R y, std::size_t b, std::size_t t, std::size_t n) {
  const R c = static_cast<R>(0.01) * static_cast<R>(1 + (b * 3 + t * 5) % 5);
  return y * (R(1) + c * y * y / static_cast<R>(n));
}
template <class R> R gain_out(std::size_t j, std::size_t t, std::size_t k) {
  return static_cast<R>(0.02) * static_cast<R>(1 + (j * 5 + t + k * 3) % 5);
}
template <class R>
std::complex<R> cubic_out(std::complex<R> y, std::size_t j, std::size_t t, std::size_t k) {
  const R g = R(1) + gain_out<R>(j, t, k) * (y.real() * y.real() + y.imag() * y.imag());
  return {y.real() * g, y.imag() * g};
}
template <class R> R cubic_out(R y, std::size_t j, std::size_t t, std::size_t k) {
  return y * (R(1) + gain_out<R>(j, t, k) * y * y);
}

// What one side's callback shares with the test: the geometry to check, the elements it was
// passed (per line, copy and position), the consumed blocks, the copies' order, and findings.
template <class E> struct side_context {
  std::size_t length = 0; // the line's elements (the middle's: its bins)
  std::size_t n = 0;      // the line's logical length (the middle's normalisation)
  std::size_t count = 1;
  std::size_t copies = 1;
  int lanes = 1;
  std::size_t scratch = 0;
  bool whole = false; // the middle: the whole line in one call
  // The caller's output, where a direct epilogue's block must be: element j of line t, copy k
  // at base + t dist + k copy_dist + j stride; null elsewhere.
  const E* base = nullptr;
  std::ptrdiff_t stride = 1;
  std::ptrdiff_t dist = 0;
  std::ptrdiff_t copy_dist = 0;
  E* captured = nullptr; // consume: line t, copy k, element j at (t copies + k) length + j
  std::unique_ptr<std::atomic<int>[]> hits;
  std::unique_ptr<std::atomic<std::size_t>[]> last_copy;
  std::atomic<int> errors{0};
  std::atomic<int> first_error_line{0};
  // The last mark each member left in its callback scratch: shared by the callbacks of one
  // pipeline, whose scratch is one area per member (one member never runs two at once).
  std::array<std::atomic<std::uint32_t>, 8>* marks = nullptr;

  void reset(int members) {
    lanes = members;
    hits = std::make_unique<std::atomic<int>[]>(length * count * copies);
    last_copy = std::make_unique<std::atomic<std::size_t>[]>(count);
    for (auto& m : *marks) {
      m = 0;
    }
    errors = 0;
  }
  void fail(int line) {
    if (errors.fetch_add(1) == 0) {
      first_error_line = line;
    }
  }
  bool check(const fftn::block_info& info, const E* data) {
    const bool ok = info.lines == 1 && info.line_axis == fftn::along_transforms &&
                    info.element_axis == 0 && info.origin.size() == 1 && info.length > 0 &&
                    info.origin[0] + info.length <= length && info.transform < count &&
                    info.copy < copies && info.member >= 0 && info.member < lanes &&
                    info.group == 1 && info.group_stride == 0 && info.line_stride == 0 &&
                    info.scratch.size() == scratch &&
                    (!whole || (info.origin[0] == 0 && info.length == length));
    if (!ok) {
      fail(__LINE__);
      return false;
    }
    if (scratch != 0 && reinterpret_cast<std::uintptr_t>(info.scratch.data()) % 128 != 0) {
      fail(__LINE__);
    }
    if (base != nullptr && (data != base + static_cast<std::ptrdiff_t>(info.transform) * dist +
                                        static_cast<std::ptrdiff_t>(info.copy) * copy_dist +
                                        static_cast<std::ptrdiff_t>(info.origin[0]) * stride ||
                            (info.length > 1 && info.element_stride != stride))) {
      fail(__LINE__);
    }
    for (std::size_t i = 0; i < info.length; ++i) {
      hits[(info.transform * copies + info.copy) * length + info.origin[0] + i].fetch_add(
          1, std::memory_order_relaxed);
    }
    // A consumed output receives each line's copies in copy order (DESIGN.md 12.3).
    std::size_t seen = last_copy[info.transform].load();
    if (captured != nullptr && info.copy < seen) {
      fail(__LINE__);
    }
    while (seen < info.copy && !last_copy[info.transform].compare_exchange_weak(seen, info.copy)) {
    }
    // The member's scratch is its own: the mark it left on its last call is still there (FFTN
    // never writes the area and no other member uses it), then a new one, the area written.
    if (scratch >= sizeof(std::uint32_t)) {
      const auto m = static_cast<std::size_t>(info.member);
      std::uint32_t found = 0;
      std::memcpy(&found, info.scratch.data(), sizeof found);
      const std::uint32_t last = (*marks)[m].load();
      if (last != 0 && found != last) {
        fail(__LINE__);
      }
      std::ranges::fill(info.scratch, std::byte{0x5A});
      const std::uint32_t mark = (static_cast<std::uint32_t>(m + 1) << 24U) |
                                 ((static_cast<std::uint32_t>(info.transform) & 0xFFFU) << 12U) |
                                 (static_cast<std::uint32_t>(info.origin[0]) & 0xFFFU);
      std::memcpy(info.scratch.data(), &mark, sizeof mark);
      (*marks)[m] = mark;
    }
    return true;
  }
  template <class F> void each(const fftn::block_info& info, E* data, F f) {
    if (!check(info, data)) {
      return;
    }
    for (std::size_t i = 0; i < info.length; ++i) {
      f(data[static_cast<std::ptrdiff_t>(i) * info.element_stride], info.origin[0] + i,
        info.transform, info.copy);
    }
  }
};

template <class E> void modify_in(void* user, const fftn::block_info& info, E* data) noexcept {
  static_cast<side_context<E>*>(user)->each(
      info, data,
      [](E& x, std::size_t j, std::size_t t, std::size_t /*k*/) { x = cubic_in(x, j, t); });
}
template <class E> void produce_in(void* user, const fftn::block_info& info, E* data) noexcept {
  static_cast<side_context<E>*>(user)->each(
      info, data,
      [](E& x, std::size_t j, std::size_t t, std::size_t /*k*/) { x = produced<E>(j, t); });
}
template <class C> void modify_mid(void* user, const fftn::block_info& info, C* data) noexcept {
  auto& ctx = *static_cast<side_context<C>*>(user);
  const std::size_t n = ctx.n;
  ctx.each(info, data, [n](C& y, std::size_t b, std::size_t t, std::size_t /*k*/) {
    y = cubic_mid(y, b, t, n);
  });
}
template <class E> void modify_out(void* user, const fftn::block_info& info, E* data) noexcept {
  static_cast<side_context<E>*>(user)->each(
      info, data,
      [](E& y, std::size_t j, std::size_t t, std::size_t k) { y = cubic_out(y, j, t, k); });
}
template <class E> void consume_out(void* user, const fftn::block_info& info, E* data) noexcept {
  auto& ctx = *static_cast<side_context<E>*>(user);
  ctx.each(info, data, [&ctx](E& y, std::size_t j, std::size_t t, std::size_t k) {
    ctx.captured[(t * ctx.copies + k) * ctx.length + j] = y;
  });
}

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

// Where logical position j of a side with these runs is stored, or -1.
std::ptrdiff_t stored_of(const std::vector<run_spec>& runs, std::size_t j) {
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

// One case's buffers, description, callbacks' contexts and reference; E is the pipeline's In
// and Out type and M its Mid: <C, C, C>, <T, C, T> (E = T) and <C, T, C> (M = T, task 5.3e).
template <class T, class E, class M = std::complex<T>> struct pcb_run {
  static constexpr bool real_side = std::is_same_v<E, T>;
  static constexpr bool real_mid = std::is_same_v<M, T>;
  using C = std::complex<T>;
  using desc_t = fftn::pipeline_desc<E, M, E>;

  pcb_case c;
  fftn::isa target;
  std::size_t side_n = 0; // the sides' logical elements: n, or the n/2 + 1 bins of <C, T, C>
  std::size_t bins = 0;   // the middle's: n, the n/2 + 1 bins of <T, C, T>, or n reals
  std::ptrdiff_t in_dist = 0;
  std::ptrdiff_t out_dist = 0;
  std::ptrdiff_t copy_dist = 0;
  std::ptrdiff_t factor_dist = 0;
  std::ptrdiff_t kbatch = 0; // the kernel's batch stride (per_row)
  double mid_scale = 1.0;
  desc_t desc;
  std::vector<E> x;
  std::vector<E> init; // the output buffer before execution (canaries, accumulation)
  std::vector<T> rpro;
  std::vector<C> cpro;
  std::vector<T> repi;
  std::vector<C> cepi;
  std::vector<T> kreal;
  std::vector<C> kcomplex;
  std::vector<bool> out_listed;
  std::vector<E> captured;
  std::array<std::atomic<std::uint32_t>, 8> marks{};
  side_context<E> pro_ctx;
  side_context<M> mid_ctx;
  side_context<E> epi_ctx;

  pcb_run(const pcb_case& pc, fftn::isa t, fftn::test::input_source& src)
      : c(pc), target(t), side_n(real_mid ? pc.n / 2 + 1 : pc.n),
        bins(real_side ? pc.n / 2 + 1 : pc.n) {
    // A real middle takes real factors only.
    if (real_mid && c.mid_f == cplx) {
      c.mid_f = real;
    }
    const std::size_t ein = extent_of(c.in, side_n);
    const std::size_t eout = extent_of(c.out, side_n);
    in_dist = static_cast<std::ptrdiff_t>(ein) * c.in_stride + 2;
    out_dist = c.in_place ? in_dist : static_cast<std::ptrdiff_t>(eout) * c.out_stride + 2;
    copy_dist = out_dist * static_cast<std::ptrdiff_t>(c.count) + 3;
    mid_scale = c.mid_scale != 0.0 ? c.mid_scale : 1.0 / static_cast<double>(c.n);

    desc.first = c.backward ? fftn::direction::backward : fftn::direction::forward;
    desc.line = {
        .n = c.n, .in_stride = c.in_stride, .out_stride = c.in_place ? c.in_stride : c.out_stride};
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
    const auto side_factor = [&](factor_kind k, std::vector<T>& r,
                                 std::vector<C>& z) -> fftn::factor {
      if (k == none) {
        return {};
      }
      if (k == real) {
        r = src.reals<T>(side_n);
      } else {
        z = src.complexes<T>(side_n);
      }
      return {.kind = k, .axis_strides = {1}, .batch_strides = {}};
    };
    desc.prologue.scale = c.pro_scale;
    desc.prologue.mul = side_factor(c.pro_f, rpro, cpro);
    desc.epilogue.scale = c.epi_scale;
    desc.epilogue.mul = side_factor(c.epi_f, repi, cepi);
    desc.epilogue.accumulate = c.accumulate;
    desc.middle.scale = mid_scale;
    const std::size_t sb = c.scratch;
    if (c.pro == pro_cb::modify) {
      desc.prologue.call = {.fn = &modify_in<E>, .scratch_bytes = sb};
    } else if (c.pro == pro_cb::produce) {
      desc.prologue.call = {
          .fn = &produce_in<E>, .mode = fftn::callback_mode::produce, .scratch_bytes = sb};
    }
    if (c.mid) {
      desc.middle.call = {.fn = &modify_mid<M>, .scratch_bytes = sb};
    }
    if (c.epi == epi_cb::modify) {
      desc.epilogue.call = {.fn = &modify_out<E>, .scratch_bytes = sb};
    } else if (c.epi == epi_cb::consume) {
      desc.epilogue.call = {
          .fn = &consume_out<E>, .mode = fftn::callback_mode::consume, .scratch_bytes = sb};
    }
    // The kernels: copy k's factor_dist k elements after copy 0's, along the bins mid_axis apart.
    // With per_row, line r's kernel kbatch r elements after line 0's.
    kbatch = c.per_row ? c.mid_axis * static_cast<std::ptrdiff_t>(bins) : 0;
    const std::ptrdiff_t one = c.mid_axis * static_cast<std::ptrdiff_t>(bins - 1) + 1 +
                               kbatch * static_cast<std::ptrdiff_t>(c.count - 1);
    factor_dist = one + 1;
    const auto ksize =
        static_cast<std::size_t>(one + factor_dist * static_cast<std::ptrdiff_t>(c.fanout - 1));
    if (c.mid_f != none) {
      desc.middle.mul = {.kind = c.mid_f, .axis_strides = {c.mid_axis}, .batch_strides = {}};
      if (c.per_row) {
        desc.middle.mul.batch_strides = {kbatch};
      }
      if (c.mid_f == real) {
        kreal = src.reals<T>(ksize);
      } else {
        kcomplex = src.complexes<T>(ksize);
      }
    }
    desc.fanout = c.fanout;
    if (c.fanout > 1) {
      desc.fanout_out_dist = copy_dist;
      desc.fanout_factor_dist = factor_dist;
      if (c.odd_copies) {
        desc.fanout_out_dist = c.consumed_out_dist;
      }
    }

    x = values(src, static_cast<std::size_t>(in_dist) * c.count);
    const std::size_t out_size =
        static_cast<std::size_t>(out_dist) * c.count +
        (c.fanout > 1 ? static_cast<std::size_t>(copy_dist) * (c.fanout - 1) : 0);
    init = c.in_place ? x : values(src, out_size);
    std::vector<bool> in_listed(x.size(), false);
    out_listed.assign(init.size(), false);
    for (std::size_t r = 0; r < c.count; ++r) {
      for (std::size_t j = 0; j < side_n; ++j) {
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
    // An input element outside the runs (or the lines) is NaN, so that a read of one shows.
    if (!c.in_place) {
      const T nan = std::numeric_limits<T>::quiet_NaN();
      for (std::size_t i = 0; i < x.size(); ++i) {
        if (!in_listed[i]) {
          x[i] = E(nan);
        }
      }
    }
    captured.assign(c.count * c.fanout * side_n, E{});
    pro_ctx.length = side_n;
    mid_ctx.length = bins;
    epi_ctx.length = side_n;
    for (auto* ctx : {&pro_ctx, &epi_ctx}) {
      ctx->n = c.n;
      ctx->count = c.count;
      ctx->scratch = sb;
    }
    mid_ctx.n = c.n;
    mid_ctx.count = c.count;
    mid_ctx.scratch = sb;
    mid_ctx.whole = true;
    pro_ctx.marks = &marks;
    mid_ctx.marks = &marks;
    epi_ctx.marks = &marks;
    epi_ctx.copies = c.fanout;
    epi_ctx.stride = desc.line.out_stride;
    epi_ctx.dist = out_dist;
    epi_ctx.copy_dist = copy_dist;
    if (c.epi == epi_cb::consume) {
      epi_ctx.captured = captured.data();
    }
  }

  static std::vector<E> values(fftn::test::input_source& src, std::size_t n) {
    if constexpr (real_side) {
      return src.reals<T>(n);
    } else {
      return src.complexes<T>(n);
    }
  }

  [[nodiscard]] std::size_t in_at(std::size_t p, std::size_t r) const {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(r) * in_dist +
                                    static_cast<std::ptrdiff_t>(p) * c.in_stride);
  }
  [[nodiscard]] std::size_t out_at(std::size_t p, std::size_t r, std::size_t k) const {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(k) * copy_dist +
                                    static_cast<std::ptrdiff_t>(r) * out_dist +
                                    static_cast<std::ptrdiff_t>(p) * desc.line.out_stride);
  }
  [[nodiscard]] bool epilogue_direct() const {
    return c.epi == epi_cb::modify && !c.accumulate && c.out.empty();
  }
  [[nodiscard]] bool any_callback() const {
    return c.pro != pro_cb::off || c.mid || c.epi != epi_cb::off;
  }

  [[nodiscard]] fftn::exec_args args() {
    const auto address = [](const std::vector<T>& r, const std::vector<C>& z) -> const void* {
      return !r.empty() ? static_cast<const void*>(r.data())
                        : (!z.empty() ? static_cast<const void*>(z.data()) : nullptr);
    };
    return {.in_factor = address(rpro, cpro),
            .mid_factor = address(kreal, kcomplex),
            .out_factor = address(repi, cepi),
            .in_user = &pro_ctx,
            .mid_user = &mid_ctx,
            .out_user = &epi_ctx};
  }

  // Points the contexts at an execution with `lanes` members writing the output buffer y.
  void prepare(int lanes, const std::vector<E>& y) {
    pro_ctx.reset(lanes);
    mid_ctx.reset(lanes);
    epi_ctx.reset(lanes);
    epi_ctx.base = epilogue_direct() ? y.data() : nullptr;
    std::ranges::fill(captured, E{});
  }

  // Every element passed exactly once (each stored one, with runs; each bin once per line; each
  // output element once per copy), and no finding.
  void expect_clean(const std::string& what) const {
    const auto side = [&](const auto& ctx, bool used, const std::vector<run_spec>& runs,
                          std::size_t copies, const char* name) {
      EXPECT_EQ(ctx.errors.load(), 0) << what << ": the " << name << "'s check failed (line "
                                      << ctx.first_error_line.load() << ")";
      std::size_t wrong = 0;
      for (std::size_t r = 0; r < c.count; ++r) {
        for (std::size_t k = 0; k < copies; ++k) {
          for (std::size_t j = 0; j < ctx.length; ++j) {
            const int want = used && stored_of(runs, j) >= 0 ? 1 : 0;
            if (ctx.hits[(r * copies + k) * ctx.length + j].load() != want) {
              ++wrong;
            }
          }
        }
      }
      EXPECT_EQ(wrong, 0U) << what << ": the " << name
                           << "'s elements not passed exactly once each";
    };
    side(pro_ctx, c.pro != pro_cb::off, c.in, 1, "prologue");
    side(mid_ctx, c.mid, {}, 1, "middle");
    side(epi_ctx, c.epi != epi_cb::off, c.out, c.fanout, "epilogue");
  }

  [[nodiscard]] std::complex<double> side_at(const std::vector<T>& r, const std::vector<C>& z,
                                             std::size_t j) const {
    if (!r.empty()) {
      return {static_cast<double>(r[j]), 0.0};
    }
    return z.empty() ? std::complex<double>(1.0, 0.0) : std::complex<double>(z[j]);
  }
  [[nodiscard]] std::complex<double> kernel_at(std::size_t b, std::size_t r, std::size_t k) const {
    if (c.mid_f == none) {
      return {1.0, 0.0};
    }
    const auto i = static_cast<std::size_t>(c.mid_axis * static_cast<std::ptrdiff_t>(b) +
                                            kbatch * static_cast<std::ptrdiff_t>(r) +
                                            factor_dist * static_cast<std::ptrdiff_t>(k));
    return c.mid_f == real ? std::complex<double>(kreal[i], 0.0)
                           : std::complex<double>(kcomplex[i]);
  }
  [[nodiscard]] double spread() const {
    double peak = 0.0;
    double sum = 0.0;
    for (std::size_t k = 0; k < c.fanout; ++k) {
      for (std::size_t r = 0; r < c.count; ++r) {
        for (std::size_t b = 0; b < bins; ++b) {
          const double m = std::abs(kernel_at(b, r, k));
          peak = std::max(peak, m);
          sum += m * m;
        }
      }
    }
    const auto count = static_cast<double>(c.fanout * c.count * bins);
    return sum == 0.0 ? 1.0 : peak / std::sqrt(sum / count);
  }

  // Line t's input as the user's loop would give the transform: the stored runs read into the
  // logical line with zeros elsewhere (or produced there), the prologue's callback in the
  // pipeline's precision, then its factor.
  [[nodiscard]] std::vector<std::complex<double>> logical_input(std::size_t r) const {
    std::vector<std::complex<double>> line(side_n, 0.0);
    for (std::size_t j = 0; j < side_n; ++j) {
      const std::ptrdiff_t at = stored_of(c.in, j);
      if (at < 0) {
        continue;
      }
      E v{};
      if (c.pro == pro_cb::produce) {
        v = produced<E>(j, r);
      } else {
        v = x[in_at(static_cast<std::size_t>(at), r)];
        if (c.pro == pro_cb::modify) {
          v = cubic_in(v, j, r);
        }
      }
      line[j] = std::complex<double>(v) * side_at(rpro, cpro, j);
    }
    return line;
  }

  // The first transform exactly (<C, C, C> in `first`'s direction, r2c, or c2r for <C, T, C>),
  // then (rounded to double) the prologue's scale and the middle's callback, then for each copy
  // its kernel and the scales after the callback, the second transform exactly, the epilogue's
  // factor and callback (in double), and the store, the accumulation or the capture, compared on
  // the output's runs. Every element outside the copies' runs keeps its bits.
  void check(const std::vector<E>& y, const std::string& describe) const {
    const std::string what = std::string(c.name) + ", n = " + std::to_string(c.n) + ", " +
                             fftn::isa_name(target) + (sizeof(T) == 8 ? " double" : " float") +
                             (real_side ? ", <t, c, t>" : (real_mid ? ", <c, t, c>" : "")) +
                             (c.backward ? ", backward first" : "");
    SCOPED_TRACE(what);
    const int first = c.backward ? +1 : -1;
    const double pre = c.mid ? c.pro_scale : 1.0;
    const double post = c.options_scale * mid_scale * c.epi_scale * (c.mid ? 1.0 : c.pro_scale);
    fftn::test::tolerance tol = fftn::test::derived_tolerance<T>(c.n, fftn::test::fftn_bound(c.n)) *
                                    (1.0 + spread()) * (c.epi_f != none ? 1.8 : 1.0) +
                                fftn::test::input_rounding<T>() * 6;
    // The callbacks' slopes (below 2.5) on the errors before them.
    if (c.mid) {
      tol = tol * 3;
    }
    if (c.epi == epi_cb::modify) {
      tol = tol * 3;
    }
    const auto rounded = [](const auto& v) {
      std::vector<std::complex<double>> r(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) {
        if constexpr (std::is_same_v<std::decay_t<decltype(v[i])>, fftn::test::exact_real>) {
          r[i] = v[i].hi;
        } else {
          r[i] = {v[i].re.hi, v[i].im.hi};
        }
      }
      return r;
    };
    fftn::test::outcome o;
    for (std::size_t r = 0; r < c.count; ++r) {
      const std::vector<std::complex<double>> line = logical_input(r);
      std::vector<std::complex<double>> spectrum;
      if constexpr (real_side) {
        std::vector<double> re(c.n);
        for (std::size_t j = 0; j < c.n; ++j) {
          re[j] = line[j].real();
        }
        spectrum = rounded(fftn::test::reference::r2c(re));
      } else if constexpr (real_mid) {
        spectrum = rounded(fftn::test::reference::c2r(c.n, line));
      } else {
        spectrum = rounded(fftn::test::reference::c2c(first, line));
      }
      for (std::size_t b = 0; b < bins; ++b) {
        spectrum[b] *= pre;
        if (c.mid) {
          if constexpr (real_mid) {
            spectrum[b] = cubic_mid(spectrum[b].real(), b, r, c.n);
          } else {
            spectrum[b] = cubic_mid(spectrum[b], b, r, c.n);
          }
        }
      }
      for (std::size_t k = 0; k < c.fanout; ++k) {
        std::vector<std::complex<double>> z(bins);
        for (std::size_t b = 0; b < bins; ++b) {
          z[b] = spectrum[b] * kernel_at(b, r, k) * post;
        }
        std::vector<std::complex<double>> back;
        if constexpr (real_side) {
          back = rounded(fftn::test::reference::c2r(c.n, z));
        } else if constexpr (real_mid) {
          std::vector<double> re(c.n);
          for (std::size_t j = 0; j < c.n; ++j) {
            re[j] = z[j].real();
          }
          back = rounded(fftn::test::reference::r2c(re));
        } else {
          back = rounded(fftn::test::reference::c2c(-first, z));
        }
        std::vector<E> got;
        std::vector<std::complex<double>> want;
        for (std::size_t j = 0; j < side_n; ++j) {
          const std::ptrdiff_t at = stored_of(c.out, j);
          if (at < 0) {
            continue;
          }
          std::complex<double> v = back[j] * side_at(repi, cepi, j);
          if constexpr (real_side) {
            v = v.real();
          }
          if (c.epi == epi_cb::modify) {
            if constexpr (real_side) {
              v = cubic_out(v.real(), j, r, k);
            } else {
              v = cubic_out(v, j, r, k);
            }
          }
          if (c.epi == epi_cb::consume) {
            got.push_back(captured[(r * c.fanout + k) * side_n + j]);
          } else {
            const std::size_t i = out_at(static_cast<std::size_t>(at), r, k);
            if (c.accumulate) {
              v += std::complex<double>(init[i]);
            }
            got.push_back(y[i]);
          }
          want.push_back(v);
        }
        const std::string where =
            what + ", line " + std::to_string(r) + ", copy " + std::to_string(k);
        if constexpr (real_side) {
          std::vector<fftn::test::exact_real> w;
          for (const auto& v : want) {
            w.push_back(fftn::test::exact_from(v.real()));
          }
          o.within(fftn::test::compare<T>(std::span<const T>(got), w), tol, where);
        } else {
          std::vector<fftn::test::exact_complex> w;
          for (const auto& v : want) {
            w.push_back(fftn::test::exact_from(v));
          }
          o.within(fftn::test::compare<T>(std::span<const C>(got), w), tol, where);
        }
      }
    }
    EXPECT_TRUE(o.passed()) << o.detail() << describe;
    if (!c.in_place) {
      std::size_t touched = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        const bool written = out_listed[i] && c.epi != epi_cb::consume;
        if (!written && !same_bits(std::vector<E>{y[i]}, std::vector<E>{init[i]})) {
          ++touched;
        }
      }
      EXPECT_EQ(touched, 0U) << what << ": elements outside the copies' runs written\n" << describe;
    }
  }

  [[nodiscard]] d::pipeline_view view() const {
    return {
        .kind = real_side  ? d::pipeline_kind::real_convolution
                : real_mid ? d::pipeline_kind::real_middle
                           : d::pipeline_kind::complex_pair,
        .prec = sizeof(T) == 8 ? d::precision::f64 : d::precision::f32,
        .first = desc.first,
        .line = desc.line,
        .batches = desc.batches,
        .runs = desc.runs,
        .prologue = d::view_of(desc.prologue),
        .middle = {.scale = desc.middle.scale,
                   .kind = desc.middle.mul.kind,
                   .axis_strides = desc.middle.mul.axis_strides,
                   .batch_strides = desc.middle.mul.batch_strides,
                   .call = reinterpret_cast<d::erased_block_fn>(desc.middle.call.fn),
                   .call_mode = desc.middle.call.mode,
                   .call_scratch = desc.middle.call.scratch_bytes},
        .epilogue = d::view_of(desc.epilogue),
        .fanout = desc.fanout,
        .fanout_out_dist = desc.fanout_out_dist,
        .fanout_factor_dist = desc.fanout_factor_dist,
    };
  }
};

// Built with construction knobs (settings), executed serially (against the reference where
// `reference`), in two pieces of the batch and, with `threads`, by teams of three (completion::all
// and own) and on a pool of three: every execution bitwise alike (the output, and a consumed
// output's captured blocks), every element passed exactly once, the input and the factors left as
// they were. A produced input and a consumed output are passed as null, except serially, where
// the output is checked unwritten.
template <class T, class E, class M>
std::string run_core(pcb_run<T, E, M>& run, const d::build_settings& settings, bool threads,
                     bool reference = true, const char* expect = nullptr) {
  const pcb_case& pc = run.c;
  if (expect == nullptr) {
    expect = run.any_callback() ? "callbacks:" : "fftn pipeline:";
  }
  const std::shared_ptr<const d::plan_core> core = d::make_pipeline_core(
      run.view(), {.in_place = pc.in_place, .scale = pc.options_scale, .force_isa = run.target},
      settings);
  const std::string describe = d::describe(*core);
  const std::string what = std::string(pc.name) + ", n = " + std::to_string(pc.n);
  EXPECT_NE(describe.find(expect), std::string::npos) << what << ": no '" << expect << "' in\n"
                                                      << describe;
  const fftn::exec_args args = run.args();
  fftn::workspace ws(d::workspace_bytes(*core, 3));
  const auto x0 = run.x;
  const auto kr = run.kreal;
  const auto kc = run.kcomplex;
  auto xs = run.x;
  const bool produce = pc.pro == pro_cb::produce;
  const bool consume = pc.epi == epi_cb::consume;
  const auto fresh = [&] { return pc.in_place ? run.x : run.init; };
  const auto in_of = [&](std::vector<E>& y) -> const E* {
    if (pc.in_place) {
      return y.data();
    }
    return produce ? nullptr : xs.data();
  };

  auto serial = fresh();
  run.prepare(1, serial);
  d::execute_all(*core, in_of(serial), serial.data(), ws, args);
  run.expect_clean(what + ", serial");
  if (reference) {
    run.check(serial, describe);
  }
  const auto captured = run.captured;
  EXPECT_TRUE(same_bits(xs, x0)) << what << ": input written";
  EXPECT_TRUE(same_bits(run.kreal, kr) && same_bits(run.kcomplex, kc))
      << what << ": kernel written";
  const auto agrees = [&](const std::vector<E>& y, const std::string& form) {
    if (!consume) {
      EXPECT_TRUE(same_bits(y, serial)) << what << ": " << form << " differs\n" << describe;
    }
    EXPECT_TRUE(same_bits(run.captured, captured))
        << what << ": " << form << " consumed other values\n"
        << describe;
    run.expect_clean(what + ", " + form);
  };
  const auto out_of = [&](std::vector<E>& y) { return consume ? nullptr : y.data(); };

  auto pieces = fresh();
  run.prepare(1, pieces);
  const std::size_t half = pc.count / 2;
  d::execute_core(*core, 0, half, in_of(pieces), out_of(pieces), ws, args);
  d::execute_core(*core, half, pc.count - half, in_of(pieces), out_of(pieces), ws, args);
  agrees(pieces, "pieces of the batch");
  if (!threads) {
    return describe;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    auto teamed = fresh();
    run.prepare(3, teamed);
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        d::execute_team(*core, in_of(teamed), out_of(teamed), fftn::team{.rank = r, .size = 3}, ws,
                        done, args);
      });
    }
    d::execute_team(*core, in_of(teamed), out_of(teamed), fftn::team{.rank = 0, .size = 3}, ws,
                    done, args);
    for (std::thread& m : members) {
      m.join();
    }
    agrees(teamed, done == fftn::completion::all ? "a team" : "a team (own)");
  }
  const fftn::thread_pool pool(3);
  auto pooled = fresh();
  run.prepare(3, pooled);
  d::execute_on(*core, in_of(pooled), out_of(pooled), pool, ws, args);
  agrees(pooled, "the executor");
  return describe;
}

template <class T, class E, class M = std::complex<T>>
std::string run_case(const pcb_case& pc, fftn::isa t, bool threads,
                     const d::build_settings& settings = {}, bool reference = true,
                     const char* expect = nullptr) {
  fftn::test::input_source src(0x53D0ULL + pc.n * 131 + pc.count * 7 + pc.fanout +
                               pc.out.size() * 3);
  pcb_run<T, E, M> run(pc, t, src);
  return run_core(run, settings, threads, reference, expect);
}

// Callbacks on a line of n: the middle's alone, beside kernels fused (complex, stride 1) and in a
// pass (real or strided), and none; the two scale products around it; prologue and epilogue
// callbacks in every mode, beside factors, scales and accumulation, strided and in place; with
// fan-out (the middle once per line, the epilogue per copy, a consumed output in copy order); with
// scratch; and (n at least 8) stored runs on either side.
std::vector<pcb_case> sets(std::size_t n, bool real_kind, std::size_t m = 0) {
  const factor_kind side = real_kind ? real : cplx;
  // The sides' logical length, where the runs lie: n, or <C, T, C>'s n/2 + 1 bins.
  m = m == 0 ? n : m;
  std::vector<pcb_case> s = {
      {.name = "the middle's alone", .n = n},
      {.name = "the middle's, no kernel, scaled", .n = n, .mid_scale = 0.5, .mid_f = none},
      {.name = "two products around the middle's",
       .n = n,
       .pro_scale = 2.0,
       .epi_scale = 0.75,
       .options_scale = 1.0},
      {.name = "two products, factors on both sides",
       .n = n,
       .pro_scale = 0.5,
       .epi_scale = 3.0,
       .pro_f = side,
       .epi_f = side},
      {.name = "the middle's beside a strided kernel", .n = n, .pro_scale = 2.0, .mid_axis = 2},
      {.name = "the middle's beside a real kernel", .n = n, .mid_f = real},
      {.name = "prologue and epilogue modify, no middle",
       .n = n,
       .pro = pro_cb::modify,
       .mid = false,
       .epi = epi_cb::modify,
       .pro_scale = 2.0,
       .epi_scale = 0.5},
      {.name = "produced and consumed", .n = n, .pro = pro_cb::produce, .epi = epi_cb::consume},
      {.name = "all three beside factors",
       .n = n,
       .pro = pro_cb::modify,
       .epi = epi_cb::modify,
       .pro_scale = 0.5,
       .epi_scale = 3.0,
       .pro_f = side,
       .epi_f = side},
      {.name = "all three, strided, options.scale",
       .n = n,
       .pro = pro_cb::modify,
       .epi = epi_cb::modify,
       .options_scale = 2.0,
       .in_stride = 2,
       .out_stride = 3},
      {.name = "the epilogue's modify, accumulated",
       .n = n,
       .epi = epi_cb::modify,
       .epi_scale = 2.0,
       .accumulate = true},
      {.name = "the epilogue's consume beside a factor",
       .n = n,
       .mid = false,
       .epi = epi_cb::consume,
       .epi_f = side},
      {.name = "in place, all three",
       .n = n,
       .pro = pro_cb::modify,
       .epi = epi_cb::modify,
       .pro_scale = 2.0,
       .in_place = true},
      {.name = "in place, produced", .n = n, .pro = pro_cb::produce, .in_place = true},
      {.name = "with scratch", .n = n, .pro = pro_cb::modify, .epi = epi_cb::modify, .scratch = 96},
      {.name = "fan-out: the epilogue's per copy",
       .n = n,
       .fanout = 3,
       .epi = epi_cb::modify,
       .pro_scale = 2.0},
      {.name = "fan-out: consumed in copy order",
       .n = n,
       .fanout = 3,
       .pro = pro_cb::modify,
       .epi = epi_cb::consume},
      {.name = "fan-out: produced, accumulated, two products",
       .n = n,
       .fanout = 2,
       .pro = pro_cb::produce,
       .epi = epi_cb::modify,
       .pro_scale = 2.0,
       .epi_scale = 0.5,
       .accumulate = true},
      {.name = "fan-out: a kernel in a pass, an epilogue factor",
       .n = n,
       .fanout = 3,
       .epi = epi_cb::modify,
       .mid_axis = 2,
       .epi_f = side},
      {.name = "fan-out: consumed, fanout_out_dist 0 (a store would refuse it)",
       .n = n,
       .fanout = 3,
       .epi = epi_cb::consume,
       .odd_copies = true,
       .consumed_out_dist = 0},
      {.name = "fan-out: consumed, copies overlapping lines (a store would refuse them)",
       .n = n,
       .count = 4,
       .fanout = 2,
       .epi = epi_cb::consume,
       .odd_copies = true,
       .consumed_out_dist = 1},
      {.name = "a kernel per line after the middle's (the inverse's first pass)",
       .n = n,
       .pro_scale = 2.0,
       .per_row = true},
      {.name = "a strided kernel per line, scaled", .n = n, .mid_axis = 2, .per_row = true},
      {.name = "fan-out: no middle callback",
       .n = n,
       .fanout = 2,
       .pro = pro_cb::modify,
       .mid = false,
       .epi = epi_cb::modify,
       .epi_scale = 2.0},
  };
  if (m >= 8) {
    const std::size_t h = m / 2;
    const std::size_t q = m / 4;
    const std::vector<pcb_case> runs = {
        {.name = "runs: valid half, all three",
         .n = n,
         .pro = pro_cb::modify,
         .epi = epi_cb::modify,
         .in = {{0, 0, h}},
         .out = {{0, 0, h}}},
        {.name = "runs: produced bands, consumed",
         .n = n,
         .pro = pro_cb::produce,
         .epi = epi_cb::consume,
         .in = {{1, 2, 2}, {h, 6, q}},
         .out = {{q, 0, h}}},
        {.name = "runs: fan-out, accumulated, scaled",
         .n = n,
         .fanout = 2,
         .pro = pro_cb::modify,
         .epi = epi_cb::modify,
         .epi_scale = 2.0,
         .accumulate = true,
         .out = {{1, 0, h - 1}, {h + 2, h, 2}}},
    };
    s.insert(s.end(), runs.begin(), runs.end());
  }
  return s;
}

// Spectral pairs inverse first and <C, T, C> (task 5.3e): sets' cases, and 5.2b's without
// callbacks: kernels fused and in passes, factors, accumulation, fan-out, strides, in place, and
// (m at least 8) bands of the sides at their edges and in their middle.
std::vector<pcb_case> spectral_sets(std::size_t n, bool real_kind, std::size_t m, bool backward) {
  const factor_kind side = real_kind ? real : cplx;
  std::vector<pcb_case> s = {
      {.name = "plain", .n = n, .mid = false},
      {.name = "plain, a strided kernel, factors, scaled",
       .n = n,
       .mid = false,
       .pro_scale = 2.0,
       .epi_scale = 0.5,
       .pro_f = side,
       .mid_axis = 2,
       .epi_f = side},
      {.name = "plain, accumulated, strided",
       .n = n,
       .mid = false,
       .accumulate = true,
       .in_stride = 2,
       .out_stride = 3},
      {.name = "plain, in place", .n = n, .mid = false, .in_place = true},
      {.name = "plain, fan-out", .n = n, .fanout = 3, .mid = false},
      {.name = "plain, fan-out, an epilogue factor, accumulated",
       .n = n,
       .fanout = 2,
       .mid = false,
       .epi_f = side,
       .accumulate = true},
  };
  if (m >= 8) {
    const std::size_t third = m / 3;
    const std::vector<pcb_case> bands = {
        {.name = "plain, bands at the edges",
         .n = n,
         .mid = false,
         .in = {{0, 0, 3}, {m - 3, 3, 3}},
         .out = {{m - 2, 0, 2}, {0, 2, 2}}},
        {.name = "plain, a band in the middle, fan-out",
         .n = n,
         .fanout = 2,
         .mid = false,
         .in = {{third, 1, third}},
         .out = {{third, 0, third + 1}}},
        {.name = "bands, all three callbacks, accumulated",
         .n = n,
         .pro = pro_cb::modify,
         .epi = epi_cb::modify,
         .accumulate = true,
         .in = {{1, 0, third}},
         .out = {{m - third, 1, third}}},
    };
    s.insert(s.end(), bands.begin(), bands.end());
  }
  for (const pcb_case& pc : sets(n, real_kind, m)) {
    s.push_back(pc);
  }
  for (pcb_case& pc : s) {
    pc.backward = backward;
  }
  return s;
}

std::vector<std::size_t> lengths() { return {1, 2, 8, 12, 15, 16, 60, 64, 100, 128, 257}; }

template <class T> class PipelineCallbacks : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(PipelineCallbacks, precisions);

TYPED_TEST(PipelineCallbacks, ComplexOnEveryInstructionSet) {
  using C = std::complex<TypeParam>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths()) {
      for (const pcb_case& pc : sets(n, false)) {
        run_case<TypeParam, C>(pc, t, false);
      }
    }
  }
}

TYPED_TEST(PipelineCallbacks, RealOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths()) {
      for (const pcb_case& pc : sets(n, true)) {
        run_case<TypeParam, TypeParam>(pc, t, false);
      }
    }
  }
}

// Teams and executors bitwise as serial execution with pure callbacks, on short lines (whole
// lines a task) and longer ones; and a long line, whose callback stages a team splits into
// blocks, the middle's whole.
TYPED_TEST(PipelineCallbacks, TeamsAndExecutorsAgreeWithSerial) {
  using C = std::complex<TypeParam>;
  for (const std::size_t n : {16U, 256U, 1000U}) {
    for (pcb_case pc : sets(n, false)) {
      pc.count = 7;
      run_case<TypeParam, C>(pc, fftn::isa::best, true, {}, n <= 16);
    }
    for (pcb_case pc : sets(n, true)) {
      pc.count = 7;
      run_case<TypeParam, TypeParam>(pc, fftn::isa::best, true, {}, n <= 16);
    }
  }
  constexpr std::size_t big = 131072;
  for (const pcb_case& pc : {
           pcb_case{.name = "a long line, all three",
                    .n = big,
                    .count = 1,
                    .pro = pro_cb::modify,
                    .epi = epi_cb::modify,
                    .pro_scale = 2.0},
           pcb_case{.name = "a long line, fan-out, consumed",
                    .n = big,
                    .count = 1,
                    .fanout = 2,
                    .pro = pro_cb::produce,
                    .epi = epi_cb::consume},
       }) {
    run_case<TypeParam, C>(pc, fftn::isa::best, true, {}, false);
    run_case<TypeParam, TypeParam>(pc, fftn::isa::best, true, {}, false);
  }
}

// The depth-first pair (forced): the prologue's callback on scratch, the middle's on the spectrum
// in scratch (in the block's second region with fan-out), the epilogue's on the output or the
// block; against the reference on every instruction set, and bitwise under teams.
TYPED_TEST(PipelineCallbacks, DepthFirstPairForced) {
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
      for (const pcb_case& pc : {
               pcb_case{.name = "depth: the middle's, two products",
                        .n = n,
                        .count = 2,
                        .pro_scale = 2.0,
                        .epi_scale = 0.5},
               pcb_case{.name = "depth: all three, factors",
                        .n = n,
                        .count = 2,
                        .pro = pro_cb::modify,
                        .epi = epi_cb::modify,
                        .pro_f = cplx,
                        .epi_f = real},
               pcb_case{.name = "depth: produced, consumed, runs",
                        .n = n,
                        .count = 2,
                        .pro = pro_cb::produce,
                        .epi = epi_cb::consume,
                        .in = {{0, 3, h}},
                        .out = {{0, 1, h}}},
               pcb_case{.name = "depth: fan-out, two products, accumulated",
                        .n = n,
                        .count = 2,
                        .fanout = 2,
                        .pro = pro_cb::modify,
                        .epi = epi_cb::modify,
                        .pro_scale = 2.0,
                        .epi_scale = 0.5,
                        .accumulate = true},
               pcb_case{.name = "depth: fan-out, consumed",
                        .n = n,
                        .count = 2,
                        .fanout = 3,
                        .epi = epi_cb::consume,
                        .pro_scale = 0.5},
           }) {
        run_case<TypeParam, C>(pc, t, false, s, true, "the depth-first pair");
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
                          .pro = pro_cb::modify,
                          .epi = epi_cb::modify,
                          .pro_scale = 2.0},
                         fftn::isa::best, true, s, false, "the depth-first pair");
}

// Spectral pairs inverse first (task 5.3e): <C, C, C> with the backward transform first, and the
// kind <C, T, C> (c2r, a real middle of n points, r2c), even and odd n, with and without
// callbacks, bands of the half spectrum at its edges and in its middle, against the reference on
// every instruction set.
TYPED_TEST(PipelineCallbacks, InverseFirstOnEveryInstructionSet) {
  using C = std::complex<TypeParam>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths()) {
      for (const pcb_case& pc : spectral_sets(n, false, n, true)) {
        run_case<TypeParam, C>(pc, t, false);
      }
    }
  }
}

TYPED_TEST(PipelineCallbacks, RealMiddleOnEveryInstructionSet) {
  using C = std::complex<TypeParam>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths()) {
      for (const pcb_case& pc : spectral_sets(n, false, n / 2 + 1, false)) {
        run_case<TypeParam, C, TypeParam>(pc, t, false);
      }
    }
  }
}

// Both kinds inverse first under teams and an executor, bitwise as serial execution.
TYPED_TEST(PipelineCallbacks, InverseFirstTeamsAgreeWithSerial) {
  using C = std::complex<TypeParam>;
  for (const std::size_t n : {16U, 256U, 1000U}) {
    for (pcb_case pc : spectral_sets(n, false, n, true)) {
      pc.count = 7;
      run_case<TypeParam, C>(pc, fftn::isa::best, true, {}, n <= 16);
    }
    for (pcb_case pc : spectral_sets(n, false, n / 2 + 1, false)) {
      pc.count = 7;
      run_case<TypeParam, C, TypeParam>(pc, fftn::isa::best, true, {}, n <= 16);
    }
  }
}

// The depth-first pair (forced) with the backward chain first: its runs, fan-out and callbacks.
TYPED_TEST(PipelineCallbacks, DepthFirstPairInverseFirst) {
  using C = std::complex<TypeParam>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    d::build_settings s;
    s.strategy = d::strategy_choice::depth;
    s.depth_tops = {8};
    const std::size_t n = 1024;
    const std::size_t h = n / 2;
    const std::size_t q = n / 4;
    for (const pcb_case& pc : {
             pcb_case{.name = "depth inverse first: a kernel", .n = n, .count = 2, .mid = false},
             pcb_case{.name = "depth inverse first: valid halves, scaled",
                      .n = n,
                      .count = 2,
                      .mid = false,
                      .epi_scale = 2.0,
                      .in = {{0, 3, h}},
                      .out = {{q, 5, h}}},
             pcb_case{.name = "depth inverse first: fan-out, factors",
                      .n = n,
                      .count = 2,
                      .fanout = 2,
                      .mid = false,
                      .pro_f = cplx,
                      .epi_f = real},
             pcb_case{.name = "depth inverse first: all three callbacks",
                      .n = n,
                      .count = 2,
                      .pro = pro_cb::modify,
                      .epi = epi_cb::modify,
                      .pro_scale = 2.0},
         }) {
      pcb_case back = pc;
      back.backward = true;
      run_case<TypeParam, C>(back, t, false, s, true, "the depth-first pair");
    }
  }
}

// A consumed fan-out's copies are not placed (DESIGN.md 18.2, "Callbacks" 14): a fanout_out_dist
// that a store refuses is taken with consume (the cases "fan-out: consumed, ..." above check its
// results).
TYPED_TEST(PipelineCallbacks, ConsumedCopiesAreNotPlaced) {
  using C = std::complex<TypeParam>;
  using pipe_t = fftn::pipeline<C, C, C>;
  for (const std::ptrdiff_t dist : {0, 1}) {
    fftn::pipeline_desc<C, C, C> desc = fftn::complex_convolution_rows<TypeParam>(16, 2);
    desc.fanout = 3;
    desc.fanout_out_dist = dist;
    desc.epilogue.call = {.fn = &modify_out<C>};
    EXPECT_THROW((void)pipe_t(desc), std::invalid_argument) << dist;
    desc.epilogue.call = {.fn = &consume_out<C>, .mode = fftn::callback_mode::consume};
    EXPECT_NO_THROW((void)pipe_t(desc)) << dist;
  }
}

// describe() says where each callback runs, and the two products of the scales.
TYPED_TEST(PipelineCallbacks, DescribeNamesTheCallbacks) {
  using C = std::complex<TypeParam>;
  fftn::pipeline_desc<C, C, C> desc = fftn::complex_convolution_rows<TypeParam>(64, 4);
  desc.prologue.scale = 2.0;
  desc.middle.scale = 1.0 / 64;
  desc.middle.call = {.fn = &modify_mid<C>};
  desc.epilogue.call = {.fn = &consume_out<C>, .mode = fftn::callback_mode::consume};
  const std::string text = fftn::pipeline(desc).describe();
  for (const char* want : {"callbacks:", "the middle's (modify) on each line's spectrum",
                           "the whole line in one block", "the epilogue's (consume)",
                           "the output not written", "two products around the middle's",
                           "in the backward chain's first pass, after the middle's callback"}) {
    EXPECT_NE(text.find(want), std::string::npos) << "no '" << want << "' in\n" << text;
  }
  // Inverse first, and <C, T, C> (task 5.3e).
  fftn::pipeline_desc<C, C, C> pair =
      fftn::spectral_pair_rows<TypeParam>(64, 4, fftn::direction::backward);
  const std::string back = fftn::pipeline(pair).describe();
  for (const char* want : {"<c, c, c> backward then forward", "): backward "}) {
    EXPECT_NE(back.find(want), std::string::npos) << "no '" << want << "' in\n" << back;
  }
  fftn::pipeline_desc<C, TypeParam, C> real_middle;
  real_middle.line = {.n = 63};
  real_middle.batches = {{.count = 3}};
  real_middle.middle.mul = {.kind = real, .axis_strides = {1}};
  const std::string rm = fftn::pipeline(real_middle).describe();
  for (const char* want : {"<c, t, c> c2r then r2c", "the middle there on its n reals",
                           "in a separate pass on the line's n reals"}) {
    EXPECT_NE(rm.find(want), std::string::npos) << "no '" << want << "' in\n" << rm;
  }
  // Without callbacks, nothing of them.
  const std::string plain =
      fftn::pipeline(fftn::complex_convolution_rows<TypeParam>(64, 4)).describe();
  EXPECT_EQ(plain.find("callback"), std::string::npos) << plain;
}

// A convolution nested in a middle (DESIGN.md 12.3, "Nested execution"): the middle's callback
// executes another pipeline on the line's spectrum, in place, through its serial overload, with
// its workspace on the callback's scratch; against the reference, and bitwise under teams and an
// executor.
template <class T> struct nested_context {
  const fftn::pipeline<std::complex<T>, std::complex<T>, std::complex<T>>* inner = nullptr;
  const std::complex<T>* kernel = nullptr;
  std::atomic<int> calls{0};
  std::atomic<int> errors{0};
  void operator()(const fftn::block_info& info, std::complex<T>* data) noexcept {
    calls.fetch_add(1);
    if (info.origin[0] != 0 || info.length != 64 ||
        info.scratch.size() < inner->workspace_bytes()) {
      errors.fetch_add(1);
      return;
    }
    fftn::workspace ws{info.scratch};
    inner->execute(data, data, ws, {.mid_factor = kernel});
  }
};

TYPED_TEST(PipelineCallbacks, ConvolutionNestedInAMiddle) {
  using T = TypeParam;
  using C = std::complex<T>;
  constexpr std::size_t n = 64;
  constexpr std::size_t count = 6;
  fftn::test::input_source src(0x53D1ULL);
  const std::vector<C> x = src.complexes<T>(n * count);
  const std::vector<C> k_outer = src.complexes<T>(n);
  const std::vector<C> k_inner = src.complexes<T>(n);

  fftn::pipeline_desc<C, C, C> inner_desc = fftn::complex_convolution_rows<T>(n, 1);
  inner_desc.middle.scale = 1.0 / static_cast<double>(n);
  const fftn::pipeline inner(inner_desc, {.in_place = true});
  nested_context<T> ctx{.inner = &inner, .kernel = k_inner.data()};

  fftn::pipeline_desc<C, C, C> desc = fftn::complex_convolution_rows<T>(n, count);
  desc.middle.scale = 1.0 / static_cast<double>(n);
  desc.middle.call = {.fn = fftn::callback_of<nested_context<T>, C>(),
                      .scratch_bytes = inner.workspace_bytes()};
  const fftn::pipeline outer(desc);
  const fftn::exec_args args{.mid_factor = k_outer.data(), .mid_user = &ctx};
  fftn::workspace ws(outer, 3);

  std::vector<C> serial(n * count);
  outer.execute(x.data(), serial.data(), ws, args);
  EXPECT_EQ(ctx.calls.load(), static_cast<int>(count));
  EXPECT_EQ(ctx.errors.load(), 0);

  // The reference: forward, the nested convolution of the spectrum, the outer kernel, backward.
  const auto tol = fftn::test::derived_tolerance<T>(n, fftn::test::fftn_bound(n)) * 12 +
                   fftn::test::input_rounding<T>() * 6;
  fftn::test::outcome o;
  const auto exact = [](const std::vector<fftn::test::exact_complex>& v) {
    std::vector<std::complex<double>> r(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      r[i] = {v[i].re.hi, v[i].im.hi};
    }
    return r;
  };
  for (std::size_t r = 0; r < count; ++r) {
    std::vector<std::complex<double>> line(x.begin() + static_cast<std::ptrdiff_t>(r * n),
                                           x.begin() + static_cast<std::ptrdiff_t>((r + 1) * n));
    std::vector<std::complex<double>> y = exact(fftn::test::reference::c2c(-1, line));
    std::vector<std::complex<double>> z = exact(fftn::test::reference::c2c(-1, y));
    for (std::size_t b = 0; b < n; ++b) {
      z[b] *= std::complex<double>(k_inner[b]) / static_cast<double>(n);
    }
    y = exact(fftn::test::reference::c2c(+1, z));
    for (std::size_t b = 0; b < n; ++b) {
      y[b] *= std::complex<double>(k_outer[b]) / static_cast<double>(n);
    }
    const auto back = fftn::test::reference::c2c(+1, y);
    o.within(fftn::test::compare<T>(
                 std::span<const C>(serial.data() + static_cast<std::ptrdiff_t>(r * n), n), back),
             tol, "line " + std::to_string(r));
  }
  EXPECT_TRUE(o.passed()) << o.detail();

  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    std::vector<C> teamed(n * count);
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        outer.execute(x.data(), teamed.data(), fftn::team{.rank = r, .size = 3}, ws, done, args);
      });
    }
    outer.execute(x.data(), teamed.data(), fftn::team{.rank = 0, .size = 3}, ws, done, args);
    for (std::thread& m : members) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << "a team differs";
  }
  const fftn::thread_pool pool(3);
  std::vector<C> pooled(n * count);
  outer.execute(x.data(), pooled.data(), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << "the executor differs";
  EXPECT_EQ(ctx.errors.load(), 0);
}

} // namespace
