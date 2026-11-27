// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Block callbacks (fftn/callback.hpp, DESIGN.md 12.3; task 5.3a) on c2c rows, against
// "transform then the same function applied in a separate loop": bitwise against the plain plan
// on the chain with the user's loops around it wherever no vocabulary is fused beside the
// callback, and always against the quadruple-precision reference within the derived tolerance.
// The callbacks are nonlinear (cubic) and depend on the coordinates; they check every block's
// geometry, the element's address where the block is the caller's array, `member`, and their
// scratch (a mark each member leaves and finds again), and count every element they are passed
// (each exactly once). Every mode and both sides, beside factors, scales, accumulation and stored
// runs, in and out of place, strided rows with canaries in their gaps, on every instruction set
// in both precisions; serial, pieces of the batch, teams (all and own) and an executor bitwise
// alike; a plan executed inside a callback on its scratch, under teams and an executor; refusals
// naming their fields; callback_of and its refusal of throwing callables; plans with callbacks
// executed without exec_args end the program.

#include "core/plan_core.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/precisions.hpp"
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
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

template <class T> using cvec = std::vector<std::complex<T>>;

template <class T> bool same_bits(const cvec<T>& a, const cvec<T>& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

enum class in_kind : unsigned char { none, modify, produce };
enum class out_kind : unsigned char { none, modify, consume };

struct run_spec {
  std::size_t logical = 0;
  std::size_t stored = 0;
  std::size_t count = 0;
};

struct cb_case {
  const char* name = "";
  std::size_t n = 1;
  std::size_t count = 1;
  fftn::direction dir = fftn::direction::forward;
  in_kind in = in_kind::none;
  out_kind out = out_kind::none;
  double options_scale = 1.0;
  double in_scale = 1.0;
  double out_scale = 1.0;
  fftn::factor_kind in_factor = fftn::factor_kind::none; // stride 1 along the row, per row
  fftn::factor_kind out_factor = fftn::factor_kind::none;
  bool accumulate = false;
  bool in_place = false;
  std::ptrdiff_t in_stride = 1; // rows are (stored extent) * stride + 1 apart: a gap
  std::ptrdiff_t out_stride = 1;
  std::vector<run_spec> in_runs{};
  std::vector<run_spec> out_runs{};
  std::size_t scratch = 0; // each callback's scratch_bytes
};

// The functions the callbacks apply, in the plan's precision, and the test's loops with them.
// Pure per element: nonlinear, and dependent on the coordinates (j along the row, t the
// transform).
template <class T> std::complex<T> cubic_in(std::complex<T> x, std::size_t j, std::size_t t) {
  const T c = static_cast<T>(0.05) * static_cast<T>(1 + (j * 7 + t * 3) % 5);
  const T m = x.real() * x.real() + x.imag() * x.imag();
  const T g = T(1) + c * m;
  return {x.real() * g, x.imag() * g};
}
template <class T> std::complex<T> produced(std::size_t j, std::size_t t) {
  const double a = 0.37 * static_cast<double>(j) + 1.1 * static_cast<double>(t);
  return {static_cast<T>(std::sin(a)), static_cast<T>(std::cos(0.61 * a - 0.5))};
}
// The output's: y (1 + c |y|^2 / n), c from 0.02 to 0.1, so that its slope stays below 2.5
// for the transforms of inputs of magnitude 1 (|y|^2 / n about 2/3, a few times that at most).
template <class R>
std::complex<R> cubic_out(std::complex<R> y, std::size_t k, std::size_t t, std::size_t n) {
  const R c = static_cast<R>(0.02) * static_cast<R>(1 + (k * 5 + t) % 5);
  const R m = y.real() * y.real() + y.imag() * y.imag();
  const R g = R(1) + c * m / static_cast<R>(n);
  return {y.real() * g, y.imag() * g};
}

// The last mark each member left in its callback scratch, and its calls: shared by the callbacks
// of one plan, whose scratch is one area per member (one member never runs two at once).
struct member_marks {
  std::array<std::uint32_t, 8> marks{};
  std::array<std::uint32_t, 8> calls{};
};

// What the callbacks share with the test: the row's geometry, what to check, the elements they
// were passed, where a consumed block is captured, and their findings.
template <class T> struct cb_context {
  std::size_t n = 1;
  std::size_t count = 1;
  int lanes = 1;
  std::size_t scratch = 0;
  // Where the block must be (the caller's array), or null where it is in the workspace.
  const std::complex<T>* base = nullptr;
  std::ptrdiff_t stride = 1;
  std::ptrdiff_t dist = 0;
  std::complex<T>* captured = nullptr; // consume: transform t's element k at t n + k
  std::unique_ptr<std::atomic<int>[]> hits;
  member_marks* marks = nullptr;
  std::atomic<int> errors{0};
  std::atomic<int> first_error_line{0};

  void reset() {
    hits = std::make_unique<std::atomic<int>[]>(n * count);
    *marks = {};
    errors = 0;
  }
  void fail(int line) {
    if (errors.fetch_add(1) == 0) {
      first_error_line = line;
    }
  }
  // Checks common to every block, counts its elements, and checks and moves the member's mark.
  // A block may hold several rows (the edge driver's row blocks, task 5.3b): line k is transform
  // `transform + k`, line_stride elements on.
  bool check(const fftn::block_info& info, const std::complex<T>* data) {
    const bool ok = info.lines >= 1 && info.line_axis == fftn::along_transforms &&
                    info.element_axis == 0 && info.origin.size() == 1 && info.length > 0 &&
                    info.origin[0] + info.length <= n && info.transform < count &&
                    info.lines <= count - info.transform && info.member >= 0 &&
                    info.member < lanes && info.copy == 0 && info.group == 1 &&
                    info.scratch.size() == scratch && (info.lines > 1 || info.line_stride == 0);
    if (!ok) {
      fail(__LINE__);
      return false;
    }
    if (scratch != 0 && reinterpret_cast<std::uintptr_t>(info.scratch.data()) % 128 != 0) {
      fail(__LINE__);
    }
    for (std::size_t k = 0; k < info.lines; ++k) {
      const std::size_t t = info.transform + k;
      // The stride of an axis of one point is the shape's to ignore.
      const std::complex<T>* line = data + static_cast<std::ptrdiff_t>(k) * info.line_stride;
      if (base != nullptr && (line != base + static_cast<std::ptrdiff_t>(t) * dist +
                                          static_cast<std::ptrdiff_t>(info.origin[0]) * stride ||
                              (n > 1 && info.element_stride != stride))) {
        fail(__LINE__);
      }
      for (std::size_t i = 0; i < info.length; ++i) {
        hits[t * n + info.origin[0] + i].fetch_add(1, std::memory_order_relaxed);
      }
    }
    // The member's scratch is its own: the mark it left on its last call is still there (FFTN
    // never writes the area and no other member uses it), then a new one, the whole area written.
    if (scratch >= sizeof(std::uint32_t)) {
      const auto m = static_cast<std::size_t>(info.member);
      std::uint32_t seen = 0;
      std::memcpy(&seen, info.scratch.data(), sizeof seen);
      if (marks->marks[m] != 0 && seen != marks->marks[m]) {
        fail(__LINE__);
      }
      std::ranges::fill(info.scratch, std::byte{0x5A});
      const std::uint32_t mark = (static_cast<std::uint32_t>(m + 1) << 24U) | ++marks->calls[m];
      std::memcpy(info.scratch.data(), &mark, sizeof mark);
      marks->marks[m] = mark;
    }
    return true;
  }
  // f(element, j, t) on each element of the block, after check().
  template <class F> void each(const fftn::block_info& info, std::complex<T>* data, F f) {
    if (!check(info, data)) {
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

template <class T>
void modify_in(void* user, const fftn::block_info& info, std::complex<T>* data) noexcept {
  static_cast<cb_context<T>*>(user)->each(
      info, data, [](std::complex<T>& x, std::size_t j, std::size_t t) { x = cubic_in(x, j, t); });
}

template <class T>
void produce_in(void* user, const fftn::block_info& info, std::complex<T>* data) noexcept {
  static_cast<cb_context<T>*>(user)->each(
      info, data, [](std::complex<T>& x, std::size_t j, std::size_t t) { x = produced<T>(j, t); });
}

template <class T>
void modify_out(void* user, const fftn::block_info& info, std::complex<T>* data) noexcept {
  auto& ctx = *static_cast<cb_context<T>*>(user);
  const std::size_t n = ctx.n;
  ctx.each(info, data,
           [n](std::complex<T>& y, std::size_t k, std::size_t t) { y = cubic_out(y, k, t, n); });
}

template <class T>
void consume_out(void* user, const fftn::block_info& info, std::complex<T>* data) noexcept {
  auto& ctx = *static_cast<cb_context<T>*>(user);
  ctx.each(info, data, [&ctx](std::complex<T>& y, std::size_t k, std::size_t t) {
    ctx.captured[t * ctx.n + k] = y;
  });
}

// The stored extent of a side along the row: the end of its last run, or n.
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

// Whether logical position j of a side with these runs is stored, and where.
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

template <class T> struct cb_run {
  using C = std::complex<T>;
  cb_case c;
  fftn::isa target;
  fftn::shape<C, C> shape;
  fftn::plan_edges<C, C> edges;
  fftn::options opts;
  std::ptrdiff_t in_dist = 0;
  std::ptrdiff_t out_dist = 0;
  cvec<T> x;    // the input buffer, gaps included
  cvec<T> init; // the output buffer before execution (accumulation, canaries)
  cvec<T> fin;  // factors, n per row
  std::vector<T> fin_r;
  cvec<T> fout;
  std::vector<T> fout_r;
  member_marks marks;
  cb_context<T> in_ctx;
  cb_context<T> out_ctx;
  cvec<T> captured;

  cb_run(const cb_case& cc, fftn::isa t, fftn::test::input_source& src) : c(cc), target(t) {
    const std::size_t in_extent = stored_extent(c.in_runs, c.n);
    const std::size_t out_extent = stored_extent(c.out_runs, c.n);
    in_dist = static_cast<std::ptrdiff_t>(in_extent) * c.in_stride + 1;
    out_dist = c.in_place ? in_dist : static_cast<std::ptrdiff_t>(out_extent) * c.out_stride + 1;
    shape.dir = c.dir;
    shape.axes = {{.n = c.n,
                   .in_stride = c.in_stride,
                   .out_stride = c.in_place ? c.in_stride : c.out_stride}};
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
    const auto factor = [&](fftn::factor_kind k) -> fftn::factor {
      if (k == fftn::factor_kind::none) {
        return {};
      }
      return {.kind = k, .axis_strides = {1}, .batch_strides = {static_cast<std::ptrdiff_t>(c.n)}};
    };
    edges.in = {.scale = c.in_scale, .mul = factor(c.in_factor)};
    edges.out = {.scale = c.out_scale, .mul = factor(c.out_factor), .accumulate = c.accumulate};
    const std::size_t sb = c.scratch;
    if (c.in == in_kind::modify) {
      edges.in.call = {.fn = &modify_in<T>, .scratch_bytes = sb};
    } else if (c.in == in_kind::produce) {
      edges.in.call = {
          .fn = &produce_in<T>, .mode = fftn::callback_mode::produce, .scratch_bytes = sb};
    }
    if (c.out == out_kind::modify) {
      edges.out.call = {.fn = &modify_out<T>, .scratch_bytes = sb};
    } else if (c.out == out_kind::consume) {
      edges.out.call = {
          .fn = &consume_out<T>, .mode = fftn::callback_mode::consume, .scratch_bytes = sb};
    }
    opts = {.in_place = c.in_place, .scale = c.options_scale, .force_isa = t};
    x = src.complexes<T>(static_cast<std::size_t>(in_dist) * c.count);
    if (c.in_place) {
      init = x;
    } else {
      init = src.complexes<T>(static_cast<std::size_t>(out_dist) * c.count);
    }
    const std::size_t factors = c.n * c.count;
    if (c.in_factor == fftn::factor_kind::complex) {
      fin = src.complexes<T>(factors);
    } else if (c.in_factor == fftn::factor_kind::real) {
      fin_r = src.reals<T>(factors);
    }
    if (c.out_factor == fftn::factor_kind::complex) {
      fout = src.complexes<T>(factors);
    } else if (c.out_factor == fftn::factor_kind::real) {
      fout_r = src.reals<T>(factors);
    }
    captured.assign(c.n * c.count, C{});
    for (cb_context<T>* ctx : {&in_ctx, &out_ctx}) {
      ctx->n = c.n;
      ctx->count = c.count;
      ctx->scratch = sb;
      ctx->marks = &marks;
    }
    out_ctx.captured = captured.data();
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
  [[nodiscard]] std::complex<double> in_factor(std::size_t j, std::size_t r) const {
    if (!fin.empty()) {
      return std::complex<double>(fin[r * c.n + j]);
    }
    return fin_r.empty() ? 1.0 : static_cast<double>(fin_r[r * c.n + j]);
  }
  [[nodiscard]] std::complex<double> out_factor(std::size_t k, std::size_t r) const {
    if (!fout.empty()) {
      return std::complex<double>(fout[r * c.n + k]);
    }
    return fout_r.empty() ? 1.0 : static_cast<double>(fout_r[r * c.n + k]);
  }
  [[nodiscard]] fftn::exec_args args() {
    return {.in_factor = !fin.empty() ? static_cast<const void*>(fin.data())
                                      : (fin_r.empty() ? nullptr : fin_r.data()),
            .out_factor = !fout.empty() ? static_cast<const void*>(fout.data())
                                        : (fout_r.empty() ? nullptr : fout_r.data()),
            .in_user = &in_ctx,
            .out_user = &out_ctx};
  }
  // Point the contexts at an execution with `lanes` members and output buffer y (the block's
  // address is checked where the callback works on the caller's array).
  void prepare(int lanes, cvec<T>& y) {
    for (cb_context<T>* ctx : {&in_ctx, &out_ctx}) {
      ctx->lanes = lanes;
      ctx->reset();
    }
    in_ctx.base = c.in_place && c.in == in_kind::modify ? y.data() : nullptr;
    in_ctx.stride = c.in_stride;
    in_ctx.dist = in_dist;
    const bool direct = c.out == out_kind::modify && !c.accumulate && c.out_runs.empty();
    out_ctx.base = direct ? y.data() : nullptr;
    out_ctx.stride = c.in_place ? c.in_stride : c.out_stride;
    out_ctx.dist = out_dist;
    std::ranges::fill(captured, std::complex<T>{});
  }
  // Every element passed exactly once (each stored element, with runs), and no finding.
  void expect_clean(const std::string& what) const {
    for (const cb_context<T>* ctx : {&in_ctx, &out_ctx}) {
      const bool used = ctx == &in_ctx ? c.in != in_kind::none : c.out != out_kind::none;
      const std::vector<run_spec>& runs = ctx == &in_ctx ? c.in_runs : c.out_runs;
      EXPECT_EQ(ctx->errors.load(), 0)
          << what << ": a callback's check failed (line " << ctx->first_error_line.load() << ")";
      std::size_t wrong = 0;
      for (std::size_t r = 0; r < c.count; ++r) {
        for (std::size_t j = 0; j < c.n; ++j) {
          std::size_t at = 0;
          const int want = used && stored_at(runs, j, at) ? 1 : 0;
          if (ctx->hits[r * c.n + j].load() != want) {
            ++wrong;
          }
        }
      }
      EXPECT_EQ(wrong, 0U) << what << ": elements not passed exactly once ("
                           << (ctx == &in_ctx ? "input" : "output") << ")";
    }
  }

  // The transform's input in the plan's precision, as the user's loop before the transform
  // would leave it: the stored runs read into the logical row with zeros elsewhere, then the
  // callback.
  [[nodiscard]] cvec<T> logical_input(std::size_t r) const {
    cvec<T> row(c.n);
    for (std::size_t j = 0; j < c.n; ++j) {
      std::size_t at = 0;
      if (!stored_at(c.in_runs, j, at)) {
        continue;
      }
      if (c.in == in_kind::produce) {
        row[j] = produced<T>(j, r);
      } else {
        row[j] = x[in_at(at, r)];
        if (c.in == in_kind::modify) {
          row[j] = cubic_in(row[j], j, r);
        }
      }
    }
    return row;
  }

  // Checks the output buffer y (and, consumed, the captured blocks) against the reference.
  void check(const cvec<T>& y, const std::string& describe) const {
    const std::string what = std::string(c.name) + ", " + fftn::isa_name(target) +
                             (sizeof(T) == 8 ? " double" : " float");
    SCOPED_TRACE(what);
    const double scale = c.options_scale * c.in_scale * c.out_scale;
    const int sign = static_cast<int>(c.dir);
    std::vector<bool> described(y.size(), false);
    fftn::test::outcome o;
    for (std::size_t r = 0; r < c.count; ++r) {
      const cvec<T> row = logical_input(r);
      std::vector<std::complex<double>> xin(c.n);
      for (std::size_t j = 0; j < c.n; ++j) {
        xin[j] = std::complex<double>(row[j]) * in_factor(j, r);
      }
      const auto exact = fftn::test::reference::c2c(sign, xin);
      std::vector<fftn::test::exact_complex> want;
      cvec<T> got;
      for (std::size_t k = 0; k < c.n; ++k) {
        // Only the output's runs are stored, or consumed.
        std::size_t at = 0;
        if (!stored_at(c.out_runs, k, at)) {
          continue;
        }
        std::complex<double> v =
            std::complex<double>(exact[k].re.hi, exact[k].im.hi) * out_factor(k, r) * scale;
        if (c.out == out_kind::modify) {
          v = cubic_out(v, k, r, c.n);
        }
        if (c.out == out_kind::consume) {
          want.push_back(fftn::test::exact_from(v));
          got.push_back(captured[r * c.n + k]);
          continue;
        }
        if (c.accumulate) {
          v += std::complex<double>(init[out_at(at, r)]);
        }
        want.push_back(fftn::test::exact_from(v));
        got.push_back(y[out_at(at, r)]);
        described[out_at(at, r)] = true;
      }
      // The transform's bound, a rounding for each factor's product, the scale and the
      // accumulation; times the output callback's slope (below 2.5, cubic_out) where it has one.
      auto tol = fftn::test::derived_tolerance<T>(c.n, fftn::test::fftn_bound(c.n)) +
                 fftn::test::input_rounding<T>() * 4;
      if (c.out == out_kind::modify) {
        tol = tol * 3;
      }
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
      EXPECT_EQ(touched, 0U) << "elements outside the rows (or the runs) written";
    }
  }

  // The same work as the user's loops would do it, bitwise: the plain plan on the chain (for the
  // plans the rules put on the chain) over the logical input after the input callback's loop,
  // then the output callback's loop and the store, accumulation or capture. The plain plan reads
  // and writes the layouts the plan's chain does (the workspace block's contiguous row where the
  // plan copies or ends there, else the caller's strided rows, in place for in place), since
  // the chain's radices depend on them. Valid where no factor is beside a callback.
  [[nodiscard]] cvec<T> separate_loops(cvec<T>& consumed) const {
    const bool direct = c.out == out_kind::modify && !c.accumulate && c.out_runs.empty();
    const bool in_block = (c.in != in_kind::none && !c.in_place) || !c.in_runs.empty();
    const bool out_block = c.out == out_kind::none ? c.accumulate || !c.out_runs.empty() : !direct;
    const auto n = static_cast<std::ptrdiff_t>(c.n);
    fftn::shape<C, C> plain;
    plain.dir = c.dir;
    fftn::options plain_opts{.scale = c.options_scale * c.in_scale * c.out_scale,
                             .force_isa = target};
    if (c.in_place) {
      plain.axes = {{.n = c.n, .in_stride = c.in_stride, .out_stride = c.in_stride}};
      plain.batches = {{.count = c.count, .in_dist = in_dist, .out_dist = in_dist}};
      plain_opts.in_place = true;
    } else {
      plain.axes = {{.n = c.n,
                     .in_stride = in_block ? 1 : c.in_stride,
                     .out_stride = out_block ? 1 : c.out_stride}};
      plain.batches = {{.count = c.count,
                        .in_dist = in_block ? n : in_dist,
                        .out_dist = out_block ? n : out_dist}};
      plain_opts.in_place = in_block && out_block;
    }
    fftn::detail::build_settings chain;
    chain.strategy = fftn::detail::strategy_choice::chain;
    chain.batch_leaves = fftn::detail::batch_leaf_choice::never;
    const auto core = fftn::detail::make_plan_core(fftn::detail::view_of(plain), plain_opts, chain);
    fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
    // The transform's input where the plain plan reads it.
    cvec<T> xin = c.in_place ? x : (in_block ? cvec<T>(c.n * c.count) : x);
    for (std::size_t r = 0; r < c.count; ++r) {
      const cvec<T> row = logical_input(r);
      for (std::size_t j = 0; j < c.n; ++j) {
        xin[in_block ? r * c.n + j : in_at(j, r)] = row[j];
      }
    }
    cvec<T> y = init;
    if (c.in_place) {
      fftn::detail::execute_all(*core, xin.data(), xin.data(), ws);
      y = xin;
    } else if (plain_opts.in_place) {
      fftn::detail::execute_all(*core, xin.data(), xin.data(), ws);
    } else if (out_block) {
      cvec<T> yt(c.n * c.count);
      fftn::detail::execute_all(*core, xin.data(), yt.data(), ws);
      xin = yt;
    } else {
      fftn::detail::execute_all(*core, xin.data(), y.data(), ws);
    }
    for (std::size_t r = 0; r < c.count; ++r) {
      for (std::size_t k = 0; k < c.n; ++k) {
        std::size_t at = 0;
        if (!stored_at(c.out_runs, k, at)) {
          continue;
        }
        std::complex<T> v = out_block ? xin[r * c.n + k] : y[out_at(at, r)];
        if (c.out == out_kind::modify) {
          v = cubic_out(v, k, r, c.n);
        }
        if (c.out == out_kind::consume) {
          consumed[r * c.n + k] = v;
          continue;
        }
        if (c.accumulate) {
          v = init[out_at(at, r)] + v;
        }
        y[out_at(at, r)] = v;
      }
    }
    return y;
  }
};

// The case's plan, checked against the reference (and bitwise against the separate loops where
// no factor is beside the callbacks), executed serially, in two pieces of the batch, by teams of
// three (all and own) and on a pool of three: every execution bitwise alike, every element passed
// once, the input and the factors left as they were. `threads`: the teams and the pool too.
// reference: whether to check against the quadruple-precision reference (rows too long for it
// are checked bitwise against the separate loops only).
template <class T>
void run_case(const cb_case& cc, fftn::isa t, bool threads, bool reference = true) {
  fftn::test::input_source src(0xCA11BULL + cc.n * 31 + cc.count);
  cb_run<T> run(cc, t, src);
  const fftn::plan<std::complex<T>, std::complex<T>> p(run.shape, run.opts, run.edges);
  const std::string describe = p.describe();
  const std::string what =
      std::string(cc.name) + ", " + fftn::isa_name(t) + (sizeof(T) == 8 ? " double" : " float");
  EXPECT_NE(describe.find("callbacks:"), std::string::npos) << what << "\n" << describe;
  const fftn::exec_args args = run.args();
  const cvec<T> x0 = run.x;
  const cvec<T> fin0 = run.fin;
  const std::vector<T> fin_r0 = run.fin_r;
  const auto fresh = [&] { return cc.in_place ? run.x : run.init; };
  // The serial execution passes the arrays (a produced input null out of place), and checks
  // that a consumed output is not written; the others pass a produced input and a consumed
  // output as null, which they may be, in place too: out of place with both, both null.
  const auto in_of = [&](cvec<T>& y, cvec<T>& xs, bool null_ok) -> const std::complex<T>* {
    if (cc.in == in_kind::produce && (null_ok || !cc.in_place)) {
      return nullptr;
    }
    return cc.in_place ? y.data() : xs.data();
  };
  const auto out_of = [&](cvec<T>& y) -> std::complex<T>* {
    return cc.out == out_kind::consume ? nullptr : y.data();
  };

  fftn::workspace ws(p, 3);
  cvec<T> xs = run.x;
  cvec<T> serial = fresh();
  run.prepare(1, serial);
  p.execute(in_of(serial, xs, false), serial.data(), ws, args);
  run.expect_clean(what + ", serial");
  if (reference) {
    run.check(serial, describe);
  }
  const cvec<T> captured_serial = run.captured;
  if (!cc.in_place) {
    EXPECT_TRUE(same_bits(xs, x0)) << what << ": input written";
  }
  EXPECT_TRUE(same_bits(run.fin, fin0) && run.fin_r == fin_r0) << what << ": factor written";
  // The separate loops run the plain plan on the chain: comparable bitwise only where the plan
  // with callbacks is on the chain too (the rules may choose another strategy, task 5.3b).
  const bool on_chain = describe.find("\nstrategy: chain (") != std::string::npos &&
                        describe.find("batch leaves") == std::string::npos;
  if (on_chain && cc.in_factor == fftn::factor_kind::none &&
      cc.out_factor == fftn::factor_kind::none) {
    cvec<T> consumed(cc.n * cc.count);
    const cvec<T> loops = run.separate_loops(consumed);
    if (cc.out == out_kind::consume) {
      EXPECT_TRUE(same_bits(captured_serial, consumed)) << what << ": not the separate loops'"
                                                        << "\n"
                                                        << describe;
    } else {
      EXPECT_TRUE(same_bits(serial, loops)) << what << ": not the separate loops'\n" << describe;
    }
  }

  const auto agree = [&](const cvec<T>& y, const std::string& form) {
    run.expect_clean(what + ", " + form);
    if (cc.out == out_kind::consume) {
      EXPECT_TRUE(same_bits(run.captured, captured_serial)) << what << ": " << form << " differs";
    } else {
      EXPECT_TRUE(same_bits(y, serial)) << what << ": " << form << " differs";
    }
  };
  cvec<T> pieces = fresh();
  run.prepare(1, pieces);
  const std::size_t half = cc.count / 2;
  p.execute_transforms(0, half, in_of(pieces, xs, true), out_of(pieces), ws, args);
  p.execute_transforms(half, cc.count - half, in_of(pieces, xs, true), out_of(pieces), ws, args);
  agree(pieces, "pieces of the batch");
  if (!threads) {
    return;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    cvec<T> teamed = fresh();
    run.prepare(3, teamed);
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        p.execute(in_of(teamed, xs, true), out_of(teamed), fftn::team{.rank = r, .size = 3}, ws,
                  done, args);
      });
    }
    p.execute(in_of(teamed, xs, true), out_of(teamed), fftn::team{.rank = 0, .size = 3}, ws, done,
              args);
    for (std::thread& m : members) {
      m.join();
    }
    agree(teamed, done == fftn::completion::all ? "a team (all)" : "a team (own)");
  }
  const fftn::thread_pool pool(3);
  cvec<T> pooled = fresh();
  run.prepare(pool.concurrency(), pooled);
  p.execute(in_of(pooled, xs, true), out_of(pooled), pool, ws, args);
  agree(pooled, "the executor");
}

using fftn::factor_kind;
constexpr factor_kind real = factor_kind::real;
constexpr factor_kind cplx = factor_kind::complex;

std::vector<cb_case> cases() {
  return {
      {.name = "modify the input", .n = 64, .count = 3, .in = in_kind::modify},
      {.name = "produce the input", .n = 48, .count = 3, .in = in_kind::produce, .scratch = 64},
      {.name = "modify the output", .n = 60, .count = 2, .out = out_kind::modify},
      {.name = "consume the output",
       .n = 40,
       .count = 3,
       .dir = fftn::direction::backward,
       .out = out_kind::consume,
       .scratch = 200},
      {.name = "both sides, scales",
       .n = 32,
       .count = 4,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .options_scale = 0.5,
       .in_scale = 0.25},
      {.name = "produce and consume",
       .n = 30,
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::consume},
      {.name = "modify beside a fused input factor",
       .n = 64,
       .count = 2,
       .in = in_kind::modify,
       .in_factor = cplx},
      {.name = "modify beside a real input factor and an output factor",
       .n = 36,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .out_scale = 2.0,
       .in_factor = real,
       .out_factor = cplx},
      {.name = "consume beside an output factor and a scale",
       .n = 24,
       .count = 3,
       .out = out_kind::consume,
       .options_scale = 0.125,
       .out_factor = real},
      {.name = "modify with accumulation",
       .n = 50,
       .count = 2,
       .out = out_kind::modify,
       .accumulate = true},
      {.name = "modify with accumulation and a scale",
       .n = 20,
       .count = 3,
       .in = in_kind::produce,
       .out = out_kind::modify,
       .out_scale = 0.5,
       .accumulate = true},
      {.name = "in place, both modify",
       .n = 128,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_place = true,
       .scratch = 8},
      {.name = "in place, produce", .n = 16, .count = 3, .in = in_kind::produce, .in_place = true},
      {.name = "in place, output only",
       .n = 45,
       .count = 2,
       .out = out_kind::modify,
       .in_place = true},
      // No workspace but the callbacks' scratch.
      {.name = "a leaf in place, output only, scratch",
       .n = 8,
       .count = 3,
       .out = out_kind::modify,
       .in_place = true,
       .scratch = 40},
      {.name = "strided rows",
       .n = 20,
       .count = 3,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .in_stride = 2,
       .out_stride = 3},
      {.name = "strided rows, produce and consume",
       .n = 12,
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::consume,
       .in_stride = 3,
       .out_stride = 2},
      {.name = "input runs, modify",
       .n = 32,
       .count = 2,
       .in = in_kind::modify,
       .in_runs = {{.logical = 0, .stored = 0, .count = 10},
                   {.logical = 20, .stored = 10, .count = 6}}},
      {.name = "input runs, produce",
       .n = 24,
       .count = 3,
       .in = in_kind::produce,
       .in_runs = {{.logical = 4, .stored = 0, .count = 8}}},
      {.name = "output runs, modify",
       .n = 40,
       .count = 2,
       .out = out_kind::modify,
       .out_runs = {{.logical = 0, .stored = 0, .count = 5},
                    {.logical = 30, .stored = 5, .count = 10}}},
      {.name = "output runs, consume",
       .n = 36,
       .count = 2,
       .out = out_kind::consume,
       .out_runs = {{.logical = 10, .stored = 0, .count = 12}}},
      {.name = "output runs, modify and accumulate",
       .n = 28,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify,
       .accumulate = true,
       .out_runs = {{.logical = 2, .stored = 0, .count = 7}}},
      {.name = "Bluestein row",
       .n = 47,
       .count = 2,
       .in = in_kind::modify,
       .out = out_kind::modify},
      {.name = "Bluestein row, consume",
       .n = 59,
       .count = 2,
       .in = in_kind::produce,
       .out = out_kind::consume},
      {.name = "rows of one point",
       .n = 1,
       .count = 4,
       .in = in_kind::modify,
       .out = out_kind::modify},
      {.name = "a leaf", .n = 8, .count = 5, .in = in_kind::modify, .out = out_kind::consume},
  };
}

template <class T> class Callbacks : public ::testing::Test {};
TYPED_TEST_SUITE(Callbacks, fftn::test::precisions, fftn::test::precision_names);

TYPED_TEST(Callbacks, EveryModeOnEveryInstructionSet) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const cb_case& c : cases()) {
      run_case<TypeParam>(c, t, false);
    }
  }
}

// Teams and executors on rows short and long: rows of 2^13 and 2^17 points have callback
// stages of several chunks (blocks), which a team splits among its members.
TYPED_TEST(Callbacks, TeamsAndExecutorsAgreeWithSerial) {
  for (const cb_case& c : cases()) {
    run_case<TypeParam>(c, fftn::isa::best, true);
  }
  run_case<TypeParam>({.name = "long rows, both sides",
                       .n = 8192,
                       .count = 3,
                       .in = in_kind::modify,
                       .out = out_kind::modify,
                       .scratch = 256},
                      fftn::isa::best, true, false);
  run_case<TypeParam>({.name = "a row split among a team",
                       .n = 131072,
                       .count = 1,
                       .in = in_kind::produce,
                       .out = out_kind::consume,
                       .scratch = 64},
                      fftn::isa::best, true, false);
}

// A callback that executes another plan on its scratch (a serial overload, the nested
// workspace borrowed from info.scratch): per element, the inner transform of a vector made from
// the element, whose bin 3 is added to it. Pure per element, so serial, teams and an executor
// agree bitwise, and with the test's own loop.
struct nested_context {
  const fftn::plan<std::complex<double>, std::complex<double>>* inner = nullptr;
  std::atomic<int> errors{0};
};
constexpr std::size_t inner_n = 97; // a prime: a plan with scratch

std::complex<double>
nested_value(std::complex<double> y,
             const fftn::plan<std::complex<double>, std::complex<double>>& inner,
             fftn::workspace& ws) {
  std::array<std::complex<double>, inner_n> v{};
  std::array<std::complex<double>, inner_n> f{};
  std::complex<double> p = y;
  for (std::complex<double>& e : v) {
    e = p;
    p = p * std::complex<double>(0.5, 0.25) + 0.125;
  }
  inner.execute(v.data(), f.data(), ws);
  return y + 0.001 * f[3];
}

void nested_out(void* user, const fftn::block_info& info, std::complex<double>* data) noexcept {
  auto& ctx = *static_cast<nested_context*>(user);
  if (info.scratch.size() < ctx.inner->workspace_bytes()) {
    ctx.errors.fetch_add(1);
    return;
  }
  fftn::workspace ws{info.scratch};
  for (std::size_t k = 0; k < info.lines; ++k) {
    for (std::size_t i = 0; i < info.length; ++i) {
      std::complex<double>& y = data[static_cast<std::ptrdiff_t>(i) * info.element_stride +
                                     static_cast<std::ptrdiff_t>(k) * info.line_stride];
      y = nested_value(y, *ctx.inner, ws);
    }
  }
}

TEST(Callbacks, NestedSerialExecutionOnScratch) {
  using C = std::complex<double>;
  const fftn::plan<C, C> inner(fftn::rows_c2c<double>(inner_n, 1, fftn::direction::forward));
  ASSERT_GT(inner.workspace_bytes(), 0U) << inner.describe();
  const std::size_t n = 4096;
  const std::size_t count = 4;
  const auto shape = fftn::rows_c2c<double>(n, count, fftn::direction::backward);
  const fftn::plan<C, C> p(
      shape, {}, {.out = {.call = {.fn = &nested_out, .scratch_bytes = inner.workspace_bytes()}}});
  fftn::test::input_source src(0x4E57ULL);
  const cvec<double> x = src.complexes<double>(n * count);
  nested_context ctx{.inner = &inner};
  const fftn::exec_args args{.out_user = &ctx};
  fftn::workspace ws(p, 3);
  cvec<double> serial(n * count);
  p.execute(x.data(), serial.data(), ws, args);
  EXPECT_EQ(ctx.errors.load(), 0);
  // The test's loop: the plain plan's output, then the same function.
  const fftn::plan<C, C> plain(shape);
  cvec<double> loops(n * count);
  fftn::workspace pws(plain);
  plain.execute(x.data(), loops.data(), pws);
  fftn::workspace iws(inner);
  for (C& y : loops) {
    y = nested_value(y, inner, iws);
  }
  fftn::test::outcome o;
  std::vector<fftn::test::exact_complex> want;
  for (const C& v : loops) {
    want.push_back(fftn::test::exact_from(v));
  }
  o.within(fftn::test::compare<double>(std::span<const C>(serial), want),
           fftn::test::derived_tolerance<double>(n, fftn::test::fftn_bound(n)) * 2,
           "nested against the loop");
  EXPECT_TRUE(o.passed()) << o.detail() << p.describe();
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    cvec<double> teamed(n * count);
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        p.execute(x.data(), teamed.data(), fftn::team{.rank = r, .size = 3}, ws, done, args);
      });
    }
    p.execute(x.data(), teamed.data(), fftn::team{.rank = 0, .size = 3}, ws, done, args);
    for (std::thread& m : members) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << "a team differs";
  }
  const fftn::thread_pool pool(3);
  cvec<double> pooled(n * count);
  p.execute(x.data(), pooled.data(), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << "the executor differs";
  EXPECT_EQ(ctx.errors.load(), 0);
}

// callback_of: a lambda (const or not) as a callback, its address the user pointer.
TEST(Callbacks, CallbackOfALambda) {
  using C = std::complex<float>;
  const std::size_t n = 32;
  const auto twice = [](const fftn::block_info& info, C* data) noexcept {
    for (std::size_t k = 0; k < info.lines; ++k) {
      for (std::size_t i = 0; i < info.length; ++i) {
        data[static_cast<std::ptrdiff_t>(i) * info.element_stride +
             static_cast<std::ptrdiff_t>(k) * info.line_stride] *= 2.0F;
      }
    }
  };
  const auto shape = fftn::rows_c2c<float>(n, 2, fftn::direction::forward);
  const fftn::plan<C, C> p(
      shape, {}, {.out = {.call = {.fn = fftn::callback_of<const decltype(twice), C>()}}});
  const fftn::plan<C, C> doubled(shape, {.scale = 2.0});
  fftn::test::input_source src(0x1A3BULL);
  const cvec<float> x = src.complexes<float>(2 * n);
  cvec<float> a(2 * n);
  cvec<float> b(2 * n);
  fftn::workspace ws(std::max(p.workspace_bytes(), doubled.workspace_bytes()));
  p.execute(x.data(), a.data(), ws,
            {.out_user = const_cast<void*>(static_cast<const void*>(&twice))});
  doubled.execute(x.data(), b.data(), ws);
  std::vector<fftn::test::exact_complex> want;
  for (const C& v : b) {
    want.push_back(fftn::test::exact_from(std::complex<double>(v)));
  }
  fftn::test::outcome o;
  o.within(fftn::test::compare<float>(std::span<const C>(a), want),
           fftn::test::derived_tolerance<float>(n, fftn::test::fftn_bound(n)) * 2, "twice");
  EXPECT_TRUE(o.passed()) << o.detail() << p.describe();
}

// A callable that may throw does not convert (a compile-time refusal, tested as a concept).
template <class F, class T>
concept makes_callback = requires { fftn::callback_of<F, T>(); };
using cd = std::complex<double>;
using nothrow_lambda = decltype([](const fftn::block_info&, cd*) noexcept {});
using throwing_lambda = decltype([](const fftn::block_info&, cd*) {});
using wrong_type = decltype([](const fftn::block_info&, double*) noexcept {});
static_assert(makes_callback<nothrow_lambda, cd>);
static_assert(makes_callback<const nothrow_lambda, cd>);
static_assert(!makes_callback<throwing_lambda, cd>);
static_assert(!makes_callback<wrong_type, cd>);
// A function that may throw does not convert to block_fn either.
using may_throw = void (*)(void*, const fftn::block_info&, cd*);
static_assert(!std::is_convertible_v<may_throw, fftn::block_fn<cd>>);

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

void some_fn(void*, const fftn::block_info&, cd*) noexcept {}
void some_real_fn(void*, const fftn::block_info&, double*) noexcept {}

TEST(Callbacks, RefusalsNameTheField) {
  using C = cd;
  const auto rows = fftn::rows_c2c<double>(16, 2, fftn::direction::forward);
  using fftn::callback_mode;
  expect_refused<C, C>(rows, {}, {.in = {.call = {.fn = &some_fn, .mode = callback_mode::consume}}},
                       "edges.in.call.mode");
  expect_refused<C, C>(rows, {},
                       {.out = {.call = {.fn = &some_fn, .mode = callback_mode::produce}}},
                       "edges.out.call.mode");
  expect_refused<C, C>(
      rows, {},
      {.out = {.accumulate = true, .call = {.fn = &some_fn, .mode = callback_mode::consume}}},
      "edges.out.call.mode");
  expect_refused<C, C>(rows, {.in_place = true},
                       {.out = {.call = {.fn = &some_fn, .mode = callback_mode::consume}}},
                       "edges.out.call.mode");
  expect_refused<C, C>(rows, {},
                       {.in = {.call = {.fn = &some_fn, .mode = static_cast<callback_mode>(9)}}},
                       "edges.in.call.mode");
  // A null fn is no callback: its other fields keep their defaults.
  expect_refused<C, C>(rows, {}, {.in = {.call = {.mode = callback_mode::produce}}},
                       "edges.in.call.mode");
  expect_refused<C, C>(rows, {}, {.out = {.call = {.scratch_bytes = 64}}},
                       "edges.out.call.scratch_bytes");
  expect_refused<C, C>(rows, {}, {.out = {.call = {.fn = &some_fn, .scratch_bytes = SIZE_MAX / 2}}},
                       "edges.out.call.scratch_bytes");
  // Real rows take callbacks (task 5.3b); a consumed real output in place is refused.
  expect_refused<C, double>(
      fftn::rows_c2r<double>(16, 2), {.in_place = true},
      {.out = {.call = {.fn = &some_real_fn, .mode = callback_mode::consume}}},
      "edges.out.call.mode");
  // Not built yet on real-to-real plans (planes and volumes take callbacks from task 5.3c, beside
  // stored runs too: callbacks_axes_test.cpp).
  expect_refused<double, double>(fftn::rows_r2r<double>(16, 2, fftn::r2r_kind::dht), {},
                                 {.in = {.call = {.fn = &some_real_fn}}}, "edges.in.call.fn");
}

// describe() says where each callback runs and its blocks, the plan taking the strategy the
// rules choose (task 5.3b; each strategy's line: callbacks_strategies_test.cpp), measured
// planning included.
TEST(Callbacks, DescribeNamesTheCallbacks) {
  using C = cd;
  for (const std::size_t n : {std::size_t{8}, std::size_t{64}, std::size_t{1} << 16U}) {
    const auto rows = fftn::rows_c2c<double>(n, 16, fftn::direction::forward);
    const fftn::plan<C, C> p(
        rows, {.effort = fftn::planning::quick},
        {.in = {.call = {.fn = &some_fn}},
         .out = {.call = {.fn = &some_fn, .mode = fftn::callback_mode::consume}}});
    const std::string d = p.describe();
    EXPECT_NE(d.find("callbacks: the input's (modify) on each row in the "), std::string::npos)
        << d;
    EXPECT_NE(d.find("the output's (consume) on each row in the workspace block"),
              std::string::npos)
        << d;
    EXPECT_NE(d.find("planning: "), std::string::npos) << d;
    EXPECT_EQ(d.find("callbacks take the chain"), std::string::npos) << d;
  }
}

// The workspace counts each lane's callback scratch; plans without callbacks keep theirs.
TEST(Callbacks, WorkspaceCountsTheScratch) {
  using C = cd;
  const auto rows = fftn::rows_c2c<double>(256, 4, fftn::direction::forward);
  const fftn::plan<C, C> none(rows, {}, {.out = {.call = {.fn = &some_fn}}});
  const fftn::plan<C, C> some(rows, {},
                              {.in = {.call = {.fn = &some_fn, .scratch_bytes = 1000}},
                               .out = {.call = {.fn = &some_fn, .scratch_bytes = 3000}}});
  EXPECT_GE(some.workspace_bytes(), none.workspace_bytes() + 3000);
  EXPECT_GE(some.workspace_bytes(4), none.workspace_bytes(4) + 4 * 3000);
}

// A plan with callbacks executed through an overload without exec_args ends the program.
TEST(CallbacksDeathTest, WithoutExecArgsAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using C = cd;
  const fftn::plan<C, C> p(fftn::rows_c2c<double>(32, 2, fftn::direction::forward), {},
                           {.out = {.call = {.fn = &some_fn}}});
  cvec<double> x(64);
  cvec<double> y(64);
  fftn::workspace ws(p, 2);
  EXPECT_DEATH(p.execute(x.data(), y.data(), ws), "without exec_args");
  EXPECT_DEATH(p.execute_transforms(1, 1, x.data(), y.data(), ws), "without exec_args");
  EXPECT_DEATH(p.execute(x.data(), y.data(), fftn::team{}, ws), "without exec_args");
  EXPECT_DEATH(
      {
        const fftn::thread_pool pool(2);
        p.execute(x.data(), y.data(), pool, ws);
      },
      "without exec_args");
}

} // namespace
