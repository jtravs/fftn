// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Plan construction: validation, the default layouts, and the program of a transform of
// any rank up to three (DESIGN.md 9.3): each transform is finished before the next, one
// axis at a time, one line at a time, every line through the chain strategy. The layout
// rules are stated for users in docs/conventions.md; this file is their implementation.

#include "core/plan_core.hpp"

#include "core/callbacks.hpp"
#include "core/candidates.hpp"
#include "core/drive.hpp"
#include "core/driver.hpp"
#include "core/edge_driver.hpp"
#include "core/edges.hpp"
#include "core/pipeline.hpp"
#include "core/program.hpp"
#include "core/runs.hpp"
#include "core/wisdom_entry.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/edges.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/workspace.hpp"
#include "gen/codelets.hpp"
#include "isa/arch.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"
#include "platform/aligned_allocator.hpp"
#include "platform/cpu_features.hpp"
#include "strategies/chain.hpp"
#include "strategies/grid.hpp"
#include "threading/schedule.hpp"
#include "threading/team_state.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

// Transforms of one to three axes (DESIGN.md 5.6); any number of batch dimensions.
constexpr std::size_t max_axes = 3;

[[noreturn]] void reject(const std::string& argument, const std::string& why) {
  throw std::invalid_argument("fftn::plan: " + argument + " " + why);
}

std::string axis_name(std::size_t d) { return "shape.axes[" + std::to_string(d) + "]"; }
std::string batch_name(std::size_t b) { return "shape.batches[" + std::to_string(b) + "]"; }

std::ptrdiff_t checked_mul(std::ptrdiff_t a, std::ptrdiff_t b, const std::string& what) {
  std::ptrdiff_t r = 0;
  if (__builtin_mul_overflow(a, b, &r)) {
    reject(what, "makes the array too large to address");
  }
  return r;
}

std::ptrdiff_t checked_add(std::ptrdiff_t a, std::ptrdiff_t b, const std::string& what) {
  std::ptrdiff_t r = 0;
  if (__builtin_add_overflow(a, b, &r)) {
    reject(what, "makes the array too large to address");
  }
  return r;
}

std::ptrdiff_t to_signed(std::size_t v, const std::string& what) {
  if (v > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    reject(what, "is too large");
  }
  return static_cast<std::ptrdiff_t>(v);
}

// Only for strides and distances, which resolve() keeps above PTRDIFF_MIN.
std::ptrdiff_t magnitude(std::ptrdiff_t v) { return v < 0 ? -v : v; }

// A given stride or distance: its magnitude must be representable, so PTRDIFF_MIN is refused.
// Everything derived from it at construction or in the kernels (reaches, byte steps, the
// doubled strides of complex values read as reals) is then bounded by check_reach.
std::ptrdiff_t given_step(std::ptrdiff_t v, const std::string& what) {
  if (v == std::numeric_limits<std::ptrdiff_t>::min()) {
    reject(what, "is too large in magnitude");
  }
  return v;
}

// One side of a transform with every default resolved, in elements of the side's type.
struct side {
  std::vector<std::size_t> extent; // per axis (the halved one included)
  // Per axis, in the shape's order: the elements the layout spans along it, the extent or, on an
  // axis with stored runs on this side, its stored extent (task 5.1d). The layout checks use it.
  std::vector<std::size_t> reach;
  std::vector<std::ptrdiff_t> stride; // per axis
  std::vector<std::ptrdiff_t> dist;   // per batch dimension
  std::ptrdiff_t element_bytes = 0;
  bool complex = true;
};

// The C-ordered defaults: the last axis contiguous (padded to `last_physical`), each axis
// spanning the ones after it, the innermost batch dimension one whole array apart and each
// outer one spanning the inner ones.
//
// An axis of extent 1 and a batch dimension of count 1 never step, so whatever stride or
// distance was given for them is replaced by 0: check_reach does not see them (they add
// (1 - 1) x stride to the reach), so nothing computed from them could be proven in range.
//
// An axis with stored runs on the side (`runs`, task 5.1d) spans its stored extent instead, in the
// defaults and in the reach.
side resolve(const shape_view& s, const std::vector<std::size_t>& extent, std::size_t last_physical,
             bool input, bool complex, std::ptrdiff_t element_bytes, const side_runs& runs) {
  const std::size_t r = extent.size();
  side out;
  out.extent = extent;
  out.complex = complex;
  out.element_bytes = element_bytes;
  out.stride.assign(r, 0);
  out.reach.assign(r, 0);
  std::ptrdiff_t span = 1;
  for (std::size_t i = r; i > 0; --i) {
    const std::size_t d = i - 1;
    out.reach[d] = runs.stored_extent(d, extent[d]);
    const std::ptrdiff_t given = input ? s.axes[d].in_stride : s.axes[d].out_stride;
    if (extent[d] > 1 || out.reach[d] > 1) {
      out.stride[d] = given != 0
                          ? given_step(given, axis_name(d) + (input ? ".in_stride" : ".out_stride"))
                          : span;
    }
    const std::size_t physical = runs.stored_extent(d, d + 1 == r ? last_physical : extent[d]);
    span = checked_mul(span, to_signed(physical, axis_name(d) + ".n"), axis_name(d) + ".n");
  }
  out.dist.assign(s.batches.size(), 0);
  for (std::size_t i = s.batches.size(); i > 0; --i) {
    const std::size_t b = i - 1;
    const std::ptrdiff_t given = input ? s.batches[b].in_dist : s.batches[b].out_dist;
    if (s.batches[b].count > 1) {
      out.dist[b] =
          given != 0 ? given_step(given, batch_name(b) + (input ? ".in_dist" : ".out_dist")) : span;
    }
    span = checked_mul(span, to_signed(s.batches[b].count, batch_name(b) + ".count"),
                       batch_name(b) + ".count");
  }
  return out;
}

// The farthest element from element 0, in bytes, must be addressable; so must every step.
void check_reach(const shape_view& s, const side& v, bool input) {
  const std::string what = input ? "shape (input layout)" : "shape (output layout)";
  std::ptrdiff_t reach = 0;
  for (std::size_t d = 0; d < v.reach.size(); ++d) {
    reach = checked_add(
        reach, checked_mul(to_signed(v.reach[d] - 1, what), magnitude(v.stride[d]), what), what);
  }
  for (std::size_t b = 0; b < v.dist.size(); ++b) {
    reach = checked_add(
        reach, checked_mul(to_signed(s.batches[b].count - 1, what), magnitude(v.dist[b]), what),
        what);
  }
  static_cast<void>(checked_mul(checked_add(reach, 1, what), v.element_bytes, what));
}

// The bytes a side's elements span relative to element 0, [lo, hi): strides may be negative.
std::pair<std::ptrdiff_t, std::ptrdiff_t> span_of(const shape_view& s, const side& v) {
  std::ptrdiff_t lo = 0;
  std::ptrdiff_t hi = 0;
  const auto step = [&](std::size_t count, std::ptrdiff_t stride) {
    const std::ptrdiff_t reach = static_cast<std::ptrdiff_t>(count - 1) * stride;
    (reach < 0 ? lo : hi) += reach;
  };
  for (std::size_t d = 0; d < v.reach.size(); ++d) {
    step(v.reach[d], v.stride[d]);
  }
  for (std::size_t b = 0; b < v.dist.size(); ++b) {
    step(s.batches[b].count, v.dist[b]);
  }
  return {lo * v.element_bytes, (hi + 1) * v.element_bytes};
}

// No two output elements may share an address. Ordered by size of stride, each dimension's
// stride must exceed the reach of all smaller ones: a sufficient condition that admits
// every nested layout (rows, columns, planes, padding, reversed strides).
void check_no_overlap(const shape_view& s, const side& v) {
  std::vector<std::pair<std::ptrdiff_t, std::size_t>> dims; // |stride|, extent
  for (std::size_t d = 0; d < v.reach.size(); ++d) {
    if (v.reach[d] > 1) {
      dims.emplace_back(magnitude(v.stride[d]), v.reach[d]);
    }
  }
  for (std::size_t b = 0; b < v.dist.size(); ++b) {
    if (s.batches[b].count > 1) {
      dims.emplace_back(magnitude(v.dist[b]), s.batches[b].count);
    }
  }
  std::ranges::sort(dims);
  std::ptrdiff_t reach = 0;
  for (const auto& [stride, extent] : dims) {
    if (stride <= reach) {
      reject("shape", "describes output elements that overlap (out_stride and out_dist)");
    }
    reach += static_cast<std::ptrdiff_t>(extent - 1) * stride;
  }
}

// In place, every transform must read and write the same memory: identical layouts for
// c2c; for the real kinds FFTW's padded layout, the last axis contiguous on both sides and
// every other stride and distance on the real side twice the complex side's.
void check_in_place(const shape_view& s, const side& in, const side& out) {
  const std::size_t r = in.extent.size();
  if (s.kind == transform_kind::c2c) {
    for (std::size_t d = 0; d < r; ++d) {
      if (in.extent[d] > 1 && in.stride[d] != out.stride[d]) {
        reject("options.in_place", "needs " + axis_name(d) + ".in_stride == out_stride");
      }
    }
    for (std::size_t b = 0; b < in.dist.size(); ++b) {
      if (s.batches[b].count > 1 && in.dist[b] != out.dist[b]) {
        reject("options.in_place", "needs " + batch_name(b) + ".in_dist == out_dist");
      }
    }
    return;
  }
  const side& real = in.complex ? out : in;
  const side& cplx = in.complex ? in : out;
  // A shape from axes_r2c or axes_c2r with `padded` false is laid out for an out-of-place plan.
  const std::string padded_hint =
      " (FFTW's padded layout; axes_r2c and axes_c2r give it with padded = true)";
  if (s.axes[r - 1].n > 1 && (real.stride[r - 1] != 1 || cplx.stride[r - 1] != 1)) {
    reject("options.in_place", "needs the last axis contiguous on both sides (" + axis_name(r - 1) +
                                   " strides 1) for a real transform" + padded_hint);
  }
  for (std::size_t d = 0; d + 1 < r; ++d) {
    if (s.axes[d].n > 1 && real.stride[d] != 2 * cplx.stride[d]) {
      reject("options.in_place", "needs " + axis_name(d) +
                                     "'s real-side stride twice its complex-side stride" +
                                     padded_hint);
    }
  }
  for (std::size_t b = 0; b < real.dist.size(); ++b) {
    if (s.batches[b].count > 1 && real.dist[b] != 2 * cplx.dist[b]) {
      reject("options.in_place", "needs " + batch_name(b) +
                                     "'s real-side distance twice its complex-side distance" +
                                     padded_hint);
    }
  }
}

// The order the axes run in (DESIGN.md 9.3), whatever order the shape lists them in: widest
// stride first, as a C-ordered array lists its dimensions, so that the contiguous axis runs
// as rows and the others as columns over it. The transform does not depend on the order
// (c2c, and the complex axes of a real transform; a real transform's halved axis stays last).
// Axes of extent 1 never step and go first, out of the way. Ties keep the shape's order.
void order_axes(const shape_view& s, side& in, side& out, plan_core& c) {
  const std::size_t r = s.axes.size();
  std::vector<std::size_t> order(r);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const auto key = [&](std::size_t d) {
    const bool unit = c.lengths[d] == 1;
    return std::tuple(unit ? 1 : 0, magnitude(out.stride[d]), magnitude(in.stride[d]));
  };
  const auto end =
      order.begin() + static_cast<std::ptrdiff_t>(s.kind == transform_kind::c2c ? r : r - 1);
  std::stable_sort(order.begin(), end,
                   [&](std::size_t a, std::size_t b) { return key(a) > key(b); });
  if (std::ranges::is_sorted(order)) {
    return;
  }
  const auto arrange = [&](auto& v) {
    auto w = v;
    for (std::size_t i = 0; i < r; ++i) {
      w[i] = v[order[i]];
    }
    v = std::move(w);
  };
  arrange(in.extent);
  arrange(in.stride);
  arrange(out.extent);
  arrange(out.stride);
  arrange(c.lengths);
  c.axis_of = std::move(order);
}

template <class T> plan_tables<T>& tables_of(plan_core& c) {
  if constexpr (std::is_same_v<T, double>) {
    return c.tables_f64;
  } else {
    return c.tables_f32;
  }
}

} // namespace

// The target of the plan: the fastest the processor and the build support, or the one
// options::force_isa names. Every target has the passes.
target plan_target_of(const options& opts) {
  std::optional<target> forced;
  if (opts.force_isa != isa::best) {
    forced = static_cast<target>(static_cast<unsigned char>(opts.force_isa));
  }
  try {
    return choose_target(detect_cpu_features(), forced);
  } catch (const std::invalid_argument& e) {
    reject("options.force_isa", std::string("cannot be used: ") + e.what());
  }
}

namespace {

// A phase's view of one slot: where it is and its strides per axis, in elements.
struct view {
  slot where = slot::in;
  std::vector<std::ptrdiff_t> stride;
  std::ptrdiff_t element_bytes = 0;
};

// The bytes a lines loop of an edge phase (build, edge_phase) moves the input's and the
// output's factor cursors per iteration: the edge driver's records of a plan with edges
// (mark_cursors, edge_record::factor_step). Batch loops take theirs from the batch strides.
struct edge_loop {
  std::size_t record = 0;
  factor_steps step{};
};

// The bytes `elements` steps along axis e move each of the views' slots, as a loop's step
// or start holds them (in, out, block, twiddles).
std::array<std::ptrdiff_t, moving_slots> slot_bytes_along(std::size_t e, std::ptrdiff_t elements,
                                                          const view& src, const view& dst) {
  std::array<std::ptrdiff_t, moving_slots> step{};
  for (const view* v : {&src, &dst}) {
    const std::ptrdiff_t bytes = elements * v->stride[e] * v->element_bytes;
    switch (v->where) {
    case slot::in:
      step[0] = bytes;
      break;
    case slot::out:
      step[1] = bytes;
      break;
    case slot::block:
      step[2] = bytes;
      break;
    case slot::a:
    case slot::b:
      break;
    }
  }
  return step;
}

// Appends the loops over the lines of every axis but `axis` and `skip` (outermost first),
// returning their records' indices for close_loops.
std::vector<std::size_t> open_loops(program& p, std::size_t axis, std::size_t skip,
                                    const std::vector<std::size_t>& extent, const view& src,
                                    const view& dst) {
  std::vector<std::size_t> open;
  for (std::size_t e = 0; e < extent.size(); ++e) {
    if (e == axis || e == skip || extent[e] == 1) {
      continue;
    }
    open.push_back(p.records.size());
    p.records.emplace_back(loop{
        .kind = loop_kind::lines,
        .count = extent[e],
        .step = slot_bytes_along(e, 1, src, dst),
    });
  }
  return open;
}

void close_loops(program& p, const std::vector<std::size_t>& open) {
  for (const std::size_t i : open) {
    std::get<loop>(p.records[i]).body_end = static_cast<std::uint32_t>(p.records.size());
  }
}

// Whether neighbouring lines of a column phase write the same cache lines: lines of
// `line_bytes` of adjacent elements along the last axis, in rows `row_bytes` apart, unless
// both are whole cache lines. A real plane's rows of n/2 + 1 complex values are rarely whole
// cache lines (never for n a multiple of 4, where n/2 + 1 is odd), so its columns share them in
// most rows, and columns alone through the chain always do (X59: at 8 lanes on the Xeon 6240 these
// phases took 2.5-6.9 times their one-thread work). Planes only: on the M5 Max the fair runs this
// leads to cost real 48^3 volumes 1.2 of the time.
bool lines_share_cache_lines(std::size_t line_bytes, std::size_t row_bytes,
                             std::size_t cache_line) noexcept {
  return cache_line != 0 && (line_bytes % cache_line != 0 || row_bytes % cache_line != 0);
}

// Appends the loops over the lines of every axis but `axis` (outermost first), the line's
// stages inside them, and closes the loops.
void add_phase(program& p, std::size_t axis, const std::vector<std::size_t>& extent,
               const view& src, const view& dst, std::vector<stage>&& line) {
  const std::vector<std::size_t> open = open_loops(p, axis, axis, extent, src, dst);
  for (stage& st : line) {
    p.records.emplace_back(std::move(st));
  }
  close_loops(p, open);
}

// The edge operations of the c2c phases: the input edge belongs to the first phase, the
// output edge to the last (DESIGN.md 12.1). The scale goes on the output edge, where it is
// exactly a scaling of the result, unless that edge has a factor (or a test asks).
line_edges c2c_edges(const options& opts, const build_settings& settings, bool first_phase,
                     bool last_phase) {
  line_edges e;
  e.scale = opts.scale;
  if (first_phase && settings.in_factor != nullptr) {
    e.in = edge_op::factor;
    e.in_factor = settings.in_factor;
  }
  if (last_phase && settings.out_factor != nullptr) {
    e.out = edge_op::factor;
    e.out_factor = settings.out_factor;
  }
  if (opts.scale != 1.0) {
    const bool on_input = settings.scale_on_input || settings.out_factor != nullptr;
    if (on_input && first_phase) {
      e.in = edge_op::scale;
    } else if (!on_input && last_phase) {
      e.out = edge_op::scale;
    }
  }
  return e;
}

// Whether a complex row of n of one axis is a grid (DESIGN.md 8.5, experiment X12), and
// its split: beyond the profile's grid_min_bytes (and, for a length that is not a power of
// two, outside its chain_part_bytes and chain_mixed_bytes, X29) or when forced, if the row
// is contiguous on both sides, carries no edge factor (the grid applies only the scale, on its
// output or, with build_settings::scale_on_input, its input), has no forced radices for its whole
// length and has a split whose phases are codelet chains. The row is a c2c row or (real) the
// half-length complex transform of a real row of 2n (DESIGN.md 9.1). A c2c row of a power of
// two stays a chain up to the profile's pow2_chain_bytes (X32), and so does the half-length
// transform of a real row (X33). kind: the plan's (c2c, or the real row's); in_place: whether
// the plan is in place (X39's rules).
template <class T>
grid_split plan_grid(std::size_t n, const side& in, const side& out, const build_settings& settings,
                     const chain_builder<T>& chain, std::size_t lanes, transform_kind kind,
                     bool in_place) {
  if (n < 2 || settings.strategy == strategy_choice::chain ||
      settings.strategy == strategy_choice::lanes || settings.in_factor != nullptr ||
      settings.out_factor != nullptr || in.stride[0] != 1 || out.stride[0] != 1) {
    return {};
  }
  if (settings.strategy == strategy_choice::rules) {
    // A length that is not a power of two stays a chain longer (machine_profile, X29).
    const std::size_t bytes = n * 2 * sizeof(T);
    std::size_t part = 1;
    while ((n / part) % 2 == 0) {
      part *= 2;
    }
    const bool mixed = part != n;
    // A c2c power of two, or an r2c row's half-length one, may be a grid below grid_min_bytes
    // (X39; c2r 65536 lost 1.13 as a grid out of place).
    const bool pow2_grid = kind != transform_kind::c2r && !mixed &&
                           settings.profile.pow2_grid_bytes != 0 &&
                           bytes > settings.profile.pow2_grid_bytes;
    if ((bytes <= settings.profile.grid_min_bytes && !pow2_grid) ||
        (mixed && ((part * 2 * sizeof(T) <= settings.profile.chain_part_bytes &&
                    bytes <= settings.profile.chain_part_limit_bytes) ||
                   bytes <= settings.profile.chain_mixed_bytes)) ||
        (!mixed && bytes <= settings.profile.pow2_chain_bytes)) {
      return {};
    }
  }
  if (!settings.radices.empty() && checked_product(settings.radices) == n) {
    return {};
  }
  // Three phases beyond grid_three_bytes, unless the odd part of n (X34) or its small
  // power-of-two part (X35) keeps two.
  std::size_t odd = n;
  while (odd % 2 == 0) {
    odd /= 2;
  }
  const std::size_t part_bytes = n / odd * 2 * sizeof(T);
  // A c2c power of two out of place stays in two phases beyond grid_two_pow2_bytes (X39).
  const bool two_pow2 = kind == transform_kind::c2c && !in_place && odd == 1 &&
                        settings.profile.grid_two_pow2_bytes != 0 &&
                        n * 2 * sizeof(T) > settings.profile.grid_two_pow2_bytes;
  const bool three = settings.grid_phases != 0
                         ? settings.grid_phases == 3
                         : n * 2 * sizeof(T) > settings.profile.grid_three_bytes && !two_pow2 &&
                               (settings.profile.grid_two_odd_part == 0 ||
                                odd < settings.profile.grid_two_odd_part) &&
                               (odd == 1 || part_bytes > settings.profile.grid_two_part_bytes);
  const std::size_t block_bytes = !three && settings.profile.grid_two_block_bytes != 0
                                      ? settings.profile.grid_two_block_bytes
                                      : settings.profile.grid_block_bytes;
  grid_split g = choose_grid(
      n, lanes, block_bytes / (4 * sizeof(T)), three, settings.profile.grid_block,
      settings.grid_factors,
      settings.profile.grid_three_block != 0 ? settings.profile.grid_three_block : max_grid_block);
  g.pad = grid_pad(g, 2 * sizeof(T), settings.profile.grid_pad_period);
  for (std::size_t i = 0; i < g.phases; ++i) {
    for (const std::size_t radix : chain.radices(g.factors.at(i))) {
      if (!has_codelet(radix)) {
        return {}; // forced radices for a phase's length that need the generic pass
      }
    }
  }
  return g;
}

// The depth-first chain's top factors for a c2c row of n of one axis (DESIGN.md 8.5,
// strategies/chain.hpp, c2c_depth; experiment X46), or none: for a row contiguous on both
// sides (in place through the workspace block), with no factor edge and the scale (if any) on
// its output, without forced radices for its whole length, whose split fits. The given
// factors (build_settings::depth_tops), when not empty, replace the default ones: wherever
// they fit when the strategy is forced, and where the rules take the chain otherwise (none
// where they do not fit). Forced, by default: one codelet radix, the smallest whose sub-row
// of n / F1 fits the L1 data cache (profile.grid_block_bytes), else the one with the
// shortest sub-row. By the rules (X46 arms A and C, X51): a row of [depth_min_bytes,
// depth_max_bytes] (powers of two only to depth_pow2_max_bytes) whose power-of-two part has
// more than depth_part_bytes, in double (in float to depth_float_max_bytes), with one
// top factor, the profile's depth_top, else its depth_top_alt, else its depth_top_last, the
// first whose split fits (C divides the sub-row; X51: the top need not be a multiple of C,
// so 25000 on 4 lanes takes 10 x 2500); on a target with 32 vector registers (`avx512`,
// NEON), or tops up to 8 on `avx2` (tops of 16 lost 1.2-1.7 on its 16, X46); other targets
// keep their plans (unmeasured). On the Xeon 6240's counters screen one top of 16 (20 where
// C does not divide n / 16) was the best or within 0.05 of the best of the splits tried (one
// or two tops of 4 to 32) at every length from 16384 to 262144 points; on the M5 Max a top
// of 4 won from 3072 to 16384 points where the power-of-two part had 2048 points or more and
// lost (1.05-1.09) on 10000, 12000 and 16000.
// The tops for a contiguous line of n with neither edge factor (depth_tops_of's choice
// after its conditions on the row; also a Bluestein convolution's, bluestein_tops_of). By the
// rules from `least` where that is below depth_min_bytes (a real row's half length, X75).
template <class T>
std::vector<std::size_t>
depth_tops_for(std::size_t n, const build_settings& settings, const chain_builder<T>& chain,
               target isa, bool convolution = false,
               std::size_t least = std::numeric_limits<std::size_t>::max()) {
  // A convolution's fused chirp passes need whole blocks of C copies in the first pass
  // (chain_builder::bluestein_depth_fits; X51 lets a row's top leave a partial block).
  const auto fits = [&](const std::vector<std::size_t>& tops) {
    return convolution ? chain.bluestein_depth_fits(n, tops) : chain.depth_fits(n, tops);
  };
  if (settings.strategy == strategy_choice::rules) {
    const std::size_t bytes = n * 2 * sizeof(T);
    const machine_profile& m = settings.profile;
    const bool wide = isa == target::avx512 || isa == target::neon || isa == target::neon_fcma;
    // A Bluestein convolution (X50): in double, on 32 vector registers, up to the profile's
    // depth_convolution_max_bytes, the convolutions measured; rows by X46's and X51's fields.
    const bool pow2 = (n & (n - 1)) == 0;
    const std::size_t top = convolution ? m.depth_convolution_max_bytes : m.depth_max_bytes;
    // NOLINTNEXTLINE(misc-const-correctness): float's branch below narrows it
    std::size_t most = pow2 ? std::min(top, m.depth_pow2_max_bytes) : top;
    if constexpr (!std::is_same_v<T, double>) {
      most = convolution ? 0 : std::min(most, m.depth_float_max_bytes);
    }
    if ((convolution && !wide) || top == 0 || most == 0 ||
        bytes < std::min(m.depth_min_bytes, least) || bytes > most ||
        (n & (~n + 1)) * 2 * sizeof(T) <= m.depth_part_bytes) {
      return {};
    }
    if (!settings.depth_tops.empty()) {
      return fits(settings.depth_tops) ? settings.depth_tops : std::vector<std::size_t>{};
    }
    for (const std::size_t r : {m.depth_top, m.depth_top_alt, m.depth_top_last}) {
      const bool fits_registers = wide || (r <= 8 && isa == target::avx2);
      if (r != 0 && fits_registers && fits({r})) {
        return {r};
      }
    }
    return {};
  }
  if (!settings.depth_tops.empty()) {
    return fits(settings.depth_tops) ? settings.depth_tops : std::vector<std::size_t>{};
  }
  std::vector<std::size_t> best;
  for (const std::size_t r : gen::radices) {
    if (!fits({r})) {
      continue;
    }
    best = {r};
    if (n / r * 2 * sizeof(T) <= settings.profile.grid_block_bytes) {
      break;
    }
  }
  return best;
}

template <class T>
std::vector<std::size_t> depth_tops_of(std::size_t n, const side& in, const side& out,
                                       const build_settings& settings,
                                       const chain_builder<T>& chain, target isa) {
  if ((settings.strategy != strategy_choice::depth &&
       settings.strategy != strategy_choice::rules) ||
      settings.in_factor != nullptr || settings.out_factor != nullptr || settings.scale_on_input ||
      in.stride[0] != 1 || out.stride[0] != 1 ||
      (!settings.radices.empty() && checked_product(settings.radices) == n)) {
    return {};
  }
  return depth_tops_for<T>(n, settings, chain, isa);
}

// The depth-first chain's top factors for the half-length complex transform of an even real
// row of n of one axis (experiment X72), or none: out of place, contiguous on both sides,
// without forced radices for the half length, by depth_tops_for for h = n / 2 where h's bytes
// are within the profile's depth_real_max_bytes, from the smaller of depth_min_bytes and
// depth_real_min_bytes (or wherever the split fits when the strategy is forced to depth).
template <class T>
std::vector<std::size_t> real_depth_tops_of(std::size_t n, const side& in, const side& out,
                                            bool in_place, const build_settings& settings,
                                            const chain_builder<T>& chain, target isa) {
  const std::size_t h = n / 2;
  if ((settings.strategy != strategy_choice::depth &&
       settings.strategy != strategy_choice::rules) ||
      n % 2 != 0 || in_place || in.stride[0] != 1 || out.stride[0] != 1 ||
      (!settings.radices.empty() && checked_product(settings.radices) == h) ||
      (settings.strategy == strategy_choice::rules &&
       h * 2 * sizeof(T) > settings.profile.depth_real_max_bytes)) {
    return {};
  }
  return depth_tops_for<T>(h, settings, chain, isa, false, settings.profile.depth_real_min_bytes);
}

// The top factors of the depth-first convolution of a Bluestein stage (chain_builder::
// bluestein_depth; experiment X50) for a c2c row of one axis of n whose chain is that one
// stage (bluestein_single: a prime, one copy, no Rader root), or none: on the conditions of
// depth_tops_of for the row, and with the tops a row of the convolution's length M would
// take (depth_tops_for: by the rules where the profile's depth-first range, with
// depth_convolution_max_bytes as its upper limit, holds M, or forced with strategy depth). The
// convolution's two chains of M are the stage's cost: on the Xeon 6240 a breadth-first chain of
// 2^18 points took 8.2 ns per point against 4.4 depth-first, of 2073600 points 17.0 against 9.5
// (X50's screen).
template <class T>
std::vector<std::size_t> bluestein_tops_of(std::size_t n, const side& in, const side& out,
                                           const build_settings& settings,
                                           const chain_builder<T>& chain, target isa) {
  if ((settings.strategy != strategy_choice::depth &&
       settings.strategy != strategy_choice::rules) ||
      settings.in_factor != nullptr || settings.out_factor != nullptr || settings.scale_on_input ||
      in.stride[0] != 1 || out.stride[0] != 1 || !settings.radices.empty()) {
    return {};
  }
  const std::size_t q = chain.bluestein_single(n);
  if (q == 0) {
    return {};
  }
  return depth_tops_for<T>(chain.bluestein_length_of(q), settings, chain, isa, true);
}

// The batch leaves the rows of a c2c plan take (build_settings::batch_leaves, experiment
// X53; strategies/chain.hpp, use_batch_leaves), or null: only for one axis of contiguous
// rows in one batch dimension of at least two rows, the driver's one call of the leaf over
// the batch (driver.hpp, run_leaf_rows); a precision with one complex number per register
// has none. By the rules, the profile's (target_profile::batch_rows, batch_max_rows,
// batch_ragged_rows, batch_narrower, batch_own_rows).
template <class T>
const batch_leaf_set<T>* batch_leaves_of(const shape_view& s, const side& in, const side& out,
                                         const build_settings& settings,
                                         const target_profile& profile, target t) {
  if (settings.batch_leaves == batch_leaf_choice::never || s.kind != transform_kind::c2c ||
      s.axes.size() != 1 || s.batches.size() != 1 || s.batches[0].count < 2 || in.stride[0] != 1 ||
      out.stride[0] != 1) {
    return nullptr;
  }
  const std::size_t n = s.axes[0].n;
  const batch_leaf_table* table = nullptr;
  switch (settings.batch_leaves) {
  case batch_leaf_choice::own:
    table = find_batch_leaves(t, n);
    break;
  case batch_leaf_choice::narrower:
    table = find_narrower_batch_leaves(t, n);
    if (table == nullptr) {
      table = find_batch_leaves(t, n);
    }
    break;
  case batch_leaf_choice::rules:
    if (const std::size_t slot = codelet_slot(n);
        slot < profile.batch_rows.size() && profile.batch_rows[slot] != 0 &&
        s.batches[0].count >= profile.batch_rows[slot] &&
        (slot >= profile.batch_max_rows.size() || profile.batch_max_rows[slot] == 0 ||
         s.batches[0].count <= profile.batch_max_rows[slot]) &&
        (s.batches[0].count % profile.lanes == 0 || slot >= profile.batch_ragged_rows.size() ||
         s.batches[0].count >= profile.batch_ragged_rows[slot]) &&
        (n != 12 || t != target::avx512 || settings.profile.batch_leaf_12)) {
      const std::size_t rows = s.batches[0].count;
      const bool own = !profile.batch_narrower ||
                       (slot < profile.batch_own_rows.size() && profile.batch_own_rows[slot] != 0 &&
                        rows >= profile.batch_own_rows[slot]);
      table = own ? find_batch_leaves(t, n) : find_narrower_batch_leaves(t, n);
    }
    break;
  case batch_leaf_choice::never:
    break;
  }
  if (table == nullptr || of<T>(*table).lanes == 1) {
    return nullptr;
  }
  return &of<T>(*table);
}

// The most rows in a block of the lanes strategy by default (lanes_block). Experiment X7.
constexpr std::size_t max_lane_block = 64;

// The block (rows per block) of the lanes strategy for a plan of one axis (DESIGN.md 8.5,
// strategies/lanes.cpp, experiment X7), or 0 when the plan is not lanes: forced, or by the
// rules for rows up to the profile's lanes_up_to in a batch of at least C rows. The block
// is the largest power of two times C up to max_lane_block and the batch whose two
// scratch tiles fit half the L1 data cache (profile.grid_block_bytes), or the forced block
// rounded up to a multiple of C.
template <class T>
std::size_t lanes_block(const shape_view& s, const side& in, const side& out,
                        const build_settings& settings, const target_profile& profile,
                        const chain_builder<T>& chain, std::size_t c) {
  if (s.axes.size() != 1 || s.batches.empty() || settings.strategy == strategy_choice::chain ||
      settings.strategy == strategy_choice::grid || settings.strategy == strategy_choice::depth ||
      settings.in_factor != nullptr || settings.out_factor != nullptr || settings.scale_on_input) {
    return 0;
  }
  const std::size_t rows = s.batches.back().count;
  const std::size_t n = s.axes[0].n;
  const bool real = s.kind != transform_kind::c2c;
  if (rows < 2 || (real && (s.kind == transform_kind::r2c ? in.stride[0] : out.stride[0]) != 1)) {
    return 0;
  }
  // Radices a test forces for the row (experiments X2 and X8 time chains with them) keep
  // the rules' chain; forced lanes take them too.
  // A real row whose complex side is strided gets the contiguous row's bits from the
  // lanes but other bits from the chain's strided route, and the two targets of a pair
  // (avx2, avx512) have different thresholds: the rules keep such rows on the chain, so
  // the pair agree (DESIGN.md 6.2). Forced lanes take them.
  const bool strided_complex =
      real && (s.kind == transform_kind::r2c ? out.stride[0] : in.stride[0]) != 1;
  if (settings.strategy == strategy_choice::rules &&
      (n > profile.lanes_up_to || rows < c || chain.forces(real ? n / 2 : n) || strided_complex)) {
    return 0;
  }
  if (!chain.lanes_fit(n, real)) {
    return 0;
  }
  if (settings.lane_block != 0) {
    return (settings.lane_block + c - 1) / c * c;
  }
  const std::size_t elements = real ? n / 2 + 1 : n;
  const std::size_t most = std::max(c, rows / c * c);
  std::size_t block = c;
  while (2 * block <= max_lane_block && 2 * block <= most &&
         2 * (2 * block) * elements * 2 * sizeof(T) <= settings.profile.grid_block_bytes / 2) {
    block *= 2;
  }
  return block;
}

// The block (adjacent columns per block) of a lanes column phase (DESIGN.md 9.3, experiment
// X40): axis `axis` of a c2c plan of two or three axes, not the last, run as the lanes
// strategy over blocks of adjacent columns, which needs the last axis contiguous in the
// phase's source and destination (a run of C columns is then one vector access), a chain of
// codelet lane passes for the column (lanes_fit) and no input edge or factor; else 0 (the
// columns one at a time through the chain). The rule, from X40's sweeps of 4-32 columns:
// 256 bytes of columns (16 complex doubles), at least C and a line, at most the columns
// there are (rounded up to C). Where a column's elements lie a page or more apart, at least
// four L1 lines of columns (32 complex doubles on Apple's 128-byte lines, where 1024^2 and
// 36 x 2048 took 0.46 and 0.38 of main's time with 32 against 0.50 and 0.47 with 16, while
// 100^2 lost 1.03 with 32): wide blocks won where every element of a column is on its own
// page (2048^2: EPYC 0.126 of main's time with 16 columns, 0.183 with 4). Where they lie
// less than a page apart, halved while the block's two tiles exceed a quarter of L2
// (machine_profile::column_tile_bytes), down to a line of columns, and if they still exceed
// it the columns run alone (0): there a block whose tiles left L2 lost (Xeon 2048 x 36: 4
// columns 0.79 of main, 16 1.09; EPYC 1.04 with any block, the chain's one column in L2
// being as fast). A forced block applies alone, rounded up to a multiple of C.
//
// The complex axes of a real plan (`real`; X41) are rows of n/2 + 1 columns, whose stride is
// an odd multiple of a complex value for even n, so a column alone never crowds an L1 set the
// way X40's power-of-two c2c columns did: the lanes pay there only for the copies a column
// costs where C > 1 (arm R: the M5 Max, C = 1, lost 1.04 on the real planes; the EPYC lost
// 1.09-1.11 at 1024 x 1024, whose 16-column tiles of 512 KiB left L2, and 1.04 on the 5 x 7
// planes; arm R2, 4 columns there, still 1.05-1.06). So: only where C > 1, for columns of at
// least 32 points, and blocks of at least two L1 lines of columns whose tiles fit
// column_tile_bytes; failing that, X40's block where every column element has its own page
// and the column alone exceeds half of L1 (the chain's line leaves L1 too: 2048 x 2048 0.80-
// 0.87 of the chain's time on the EPYC), else the chain.
template <class T>
std::size_t column_block(std::size_t axis, const std::vector<std::size_t>& extent, const view& src,
                         const view& dst, const line_edges& edges, const build_settings& settings,
                         const chain_builder<T>& chain, std::size_t c, bool real = false) {
  const std::size_t last = extent.size() - 1;
  if (!settings.lanes_columns || axis == last || extent[last] < 2 || src.stride[last] != 1 ||
      dst.stride[last] != 1 || edges.in != edge_op::none || edges.out == edge_op::factor ||
      !chain.lanes_fit(extent[axis], false)) {
    return 0;
  }
  const auto up = [c](std::size_t v) { return (v + c - 1) / c * c; };
  if (settings.column_block != 0) {
    return up(settings.column_block);
  }
  constexpr std::size_t real_min_column = 32;
  if (real && (c == 1 || extent[axis] < real_min_column)) {
    return 0;
  }
  constexpr std::size_t block_bytes = 256;
  constexpr std::size_t lines_per_block = 4;
  const std::size_t n = extent[axis];
  const std::size_t complex_bytes = 2 * sizeof(T);
  const std::size_t line =
      up(std::max<std::size_t>(settings.profile.line_bytes / complex_bytes, 1));
  std::size_t block = std::max(up(block_bytes / complex_bytes), line);
  const auto stride =
      static_cast<std::size_t>(std::max(magnitude(src.stride[axis]) * src.element_bytes,
                                        magnitude(dst.stride[axis]) * dst.element_bytes));
  if (stride >= settings.profile.page_bytes) {
    block = std::max(block, up(lines_per_block * line));
  }
  block = std::min(block, up(extent[last]));
  if (real) {
    const std::size_t wide = block;
    const auto tiles = [&](std::size_t b) { return 2 * n * b * complex_bytes; };
    while (block > line && tiles(block) > settings.profile.column_tile_bytes) {
      block = up(block / 2);
    }
    if (block >= 2 * line && tiles(block) <= settings.profile.column_tile_bytes) {
      return block;
    }
    return stride >= settings.profile.page_bytes &&
                   n * complex_bytes > settings.profile.grid_block_bytes / 2
               ? wide
               : 0;
  }
  if (stride < settings.profile.page_bytes) {
    const auto tiles = [&](std::size_t b) { return 2 * n * b * complex_bytes; };
    // Down to 64 bytes of columns: an x86 line, half of Apple's 128-byte line, where blocks
    // of 4 columns won at 2048 x 36 (X41 arm L1).
    constexpr std::size_t narrowest_bytes = 64;
    const std::size_t narrowest = std::min(line, up(narrowest_bytes / complex_bytes));
    const std::size_t budget =
        settings.profile.column_near_tile_bytes != 0 && 4 * stride <= settings.profile.page_bytes
            ? settings.profile.column_near_tile_bytes
            : settings.profile.column_tile_bytes;
    while (block > narrowest && tiles(block) > budget) {
      block = up(block / 2);
    }
    if (tiles(block) > budget) {
      // A team program runs such columns as blocks of a line rather than alone through the
      // chain (X43 arm B2): members running neighbouring columns alone would write the same
      // cache lines. complex_columns gives them the chain's radices, so the bits agree.
      return settings.team_program && settings.profile.column_team_bytes != 0 ? line : 0;
    }
  }
  // Wider blocks where their tiles fit column_tile_bytes, in the one-thread program only
  // (machine_profile::column_wide_bytes; X52 arm W4b: teams keep X40's blocks, with which the
  // Xeon's 8 lanes ran up to 1.46 times faster than with the wide ones).
  if (const std::size_t wide = up(settings.profile.column_wide_bytes / complex_bytes);
      settings.profile.column_wide_bytes != 0 && !settings.team_program && wide > block &&
      2 * n * wide * complex_bytes <= settings.profile.column_tile_bytes) {
    block = std::min(wide, up(extent[last]));
  }
  // A team program's blocks (X43 arm B1; machine_profile::column_team_bytes).
  if (settings.team_program && settings.profile.column_team_bytes != 0) {
    block = std::min(block, std::max(line, up(settings.profile.column_team_bytes / complex_bytes)));
  }
  return block;
}

// The most of r legs `apart` bytes apart that fall into one set of the L1 data cache, whose
// set period is `period` bytes and whose lines are `line` bytes.
std::size_t legs_per_set(std::size_t r, std::size_t apart, std::size_t period, std::size_t line) {
  std::vector<std::size_t> sets(r);
  for (std::size_t j = 0; j < r; ++j) {
    sets[j] = j * (apart % period) % period / line;
  }
  std::ranges::sort(sets);
  std::size_t most = 0;
  for (std::size_t j = 0; j < r;) {
    std::size_t k = j;
    while (k < r && sets[k] == sets[j]) {
      ++k;
    }
    most = std::max(most, k - j);
    j = k;
  }
  return most;
}

// The radices of a lanes column phase's chain for columns of n, `stride` bytes between a
// column's elements: the row's (chain.radices), unless the last pass, which writes its legs
// to the array s = n / R columns' elements apart, would put more of them into one L1 set
// than the L1 has ways; then the cheapest sequence (ranked_radices) whose last pass does
// not, of codelet radices with lane passes. Experiment X40: on the M5 Max a last pass of
// radix 16 whose legs shared a set made 128^2 and 256^2 planes 2.2 times as long as the
// same chain ending in radix 8 (reads in the first pass did not matter). Forced radices win.
template <class T>
std::vector<std::size_t>
column_radices(std::size_t n, std::ptrdiff_t stride, const build_settings& settings,
               const target_profile& profile, const chain_builder<T>& chain,
               const kernels_of<T>& kernels, bool split_leaves) {
  const auto lane_codelets = [&](const std::vector<std::size_t>& r) {
    return std::ranges::all_of(r, [&](std::size_t x) {
      return has_codelet(x) && kernels.passes.codelet[codelet_slot(x)]->lane[0] != nullptr;
    });
  };
  if (const std::vector<std::size_t>& forced = settings.column_radices;
      !forced.empty() && lane_codelets(forced) &&
      std::accumulate(forced.begin(), forced.end(), std::size_t{1}, std::multiplies<>()) == n) {
    return forced;
  }
  std::vector<std::size_t> rad = chain.radices(n);
  if (chain.forces(n)) {
    return rad;
  }
  constexpr std::size_t single_pass_column = 32;
  const std::size_t ways = std::max<std::size_t>(settings.profile.l1_ways, 1);
  const std::size_t period = std::max<std::size_t>(settings.profile.grid_block_bytes / ways, 1);
  const std::size_t line = std::clamp<std::size_t>(settings.profile.line_bytes, 1, period);
  const auto bytes = static_cast<std::size_t>(stride < 0 ? -stride : stride);
  const auto fits = [&](const std::vector<std::size_t>& r) {
    return legs_per_set(r.back(), n / r.back() * bytes, period, line) <= ways;
  };
  // Columns of up to 32 points with a codelet of their length run as that one lane pass in
  // double, where its legs fit the L1's ways (X44): the row's chain of 32 on x86 is a pair
  // leaf in registers, which the lanes cannot take, so the columns got its radices 2 x 16 as
  // two passes over the tile; one radix-32 pass took the EPYC's 32 x 32 plane to 0.69 of
  // that time (4 x 8: 0.87).
  if (const std::vector<std::size_t> one{n}; std::is_same_v<T, double> && n <= single_pass_column &&
                                             n > 16 && lane_codelets(one) && fits(one)) {
    return one;
  }
  if (rad.empty() || fits(rad)) {
    return rad;
  }
  // A leaf (one pass of radix n) whose legs crowd one L1 set: ranked_radices has nothing
  // else for a leaf length, so two lane passes a x b = n, the most balanced first (X52 arm
  // L: the M5 Max's 16 x 256 x 256, whose outer leaf of 16 has its legs 1 MiB apart, took
  // 0.67 of the leaf's time as 4 x 4 and 0.82 as 8 x 2). Not on avx512 (split_leaves false),
  // where the Xeon 6240 gained too little (0.966 over the volumes it changes).
  if (split_leaves && rad.size() == 1) {
    std::vector<std::vector<std::size_t>> two;
    for (std::size_t a = 2; a < n; ++a) {
      if (n % a == 0) {
        two.push_back({a, n / a});
      }
    }
    const auto skew = [](const std::vector<std::size_t>& r) {
      return r[0] > r[1] ? r[0] / r[1] : r[1] / r[0];
    };
    std::ranges::stable_sort(two, [&](const auto& x, const auto& y) { return skew(x) < skew(y); });
    for (const std::vector<std::size_t>& r : two) {
      if (lane_codelets(r) && fits(r)) {
        return r;
      }
    }
  }
  constexpr std::size_t candidates = 64;
  for (const std::vector<std::size_t>& r : ranked_radices(n, profile, candidates)) {
    if (lane_codelets(r) && fits(r)) {
      return r;
    }
  }
  return rad;
}

// Appends a lanes column phase (column_block) and returns the scratch it needs: the loops
// over the other axes, then over the whole blocks of `block` adjacent columns along the
// last axis, each block's stages inside (strategies/lanes.cpp, pass_args::rows the block's
// columns); then the same nest once for the columns left over, as one narrower block rounded
// up to C whose idle lanes read nothing and are never written.
template <class T>
std::size_t add_column_phase(program& p, chain_builder<T>& chain, std::size_t axis,
                             const std::vector<std::size_t>& extent, const view& src,
                             const view& dst, int sign, const line_edges& edges, std::size_t block,
                             std::size_t c, const std::vector<std::size_t>& rad,
                             std::size_t cache_line, bool fair) {
  const std::size_t last = extent.size() - 1;
  const std::size_t whole = extent[last] / block;
  const std::size_t left = extent[last] % block;
  std::size_t need = 0;
  const auto nest = [&](std::size_t count, std::size_t width, std::size_t rows,
                        std::size_t first_column) {
    const std::size_t outermost = p.records.size();
    std::vector<std::size_t> open = open_loops(p, axis, last, extent, src, dst);
    open.push_back(p.records.size());
    p.records.emplace_back(loop{
        .kind = loop_kind::lines,
        .count = count,
        .step = slot_bytes_along(last, static_cast<std::ptrdiff_t>(width), src, dst),
        .start = slot_bytes_along(last, static_cast<std::ptrdiff_t>(first_column), src, dst),
    });
    std::vector<stage> st;
    need = std::max(need, chain.c2c_lanes(extent[axis], sign, {src.where, src.stride[axis]},
                                          src.stride[last], {dst.where, dst.stride[axis]},
                                          dst.stride[last], edges, width, st, true, rad));
    for (stage& x : st) {
      x.args.rows = static_cast<std::uint32_t>(rows);
      p.records.emplace_back(std::move(x));
    }
    close_loops(p, open);
    // The narrower block's lines are disjoint from the whole blocks': one phase for threads.
    loop& first = std::get<loop>(p.records[outermost]);
    first.joins = first_column != 0;
    first.shares_lines =
        fair && extent.size() == 2 &&
        lines_share_cache_lines(
            width * static_cast<std::size_t>(magnitude(dst.stride[last]) * dst.element_bytes),
            static_cast<std::size_t>(magnitude(dst.stride[axis]) * dst.element_bytes), cache_line);
  };
  if (whole != 0) {
    nest(whole, block, block, 0);
  }
  if (left != 0) {
    nest(1, (left + c - 1) / c * c, left, whole * block);
  }
  return need;
}

// A volume's two inner axes plane at a time (DESIGN.md 9.3, task 4.2): a loop over the
// planes along axis 0 whose body, appended by `phases(plane, src, dst)` with the views of
// one plane (axes 1 and 2 of the volume's), finishes each plane while it is in cache. The
// loop's iterations are whole planes, so the work graph makes it one phase of whole planes
// per member (threading/work_graph.cpp). Every line computes what it computes when the
// axes run one by one over the whole volume: only the order of the lines changes.
template <class Phases>
void add_volume_planes(program& p, const std::vector<std::size_t>& extent, const view& src,
                       const view& dst, const Phases& phases) {
  const auto inner = [](const view& v) {
    return view{
        .where = v.where,
        .stride = {v.stride[1], v.stride[2]},
        .element_bytes = v.element_bytes,
    };
  };
  const std::size_t planes = p.records.size();
  p.records.emplace_back(loop{
      .kind = loop_kind::lines,
      .count = extent[0],
      .step = slot_bytes_along(0, 1, src, dst),
  });
  phases(std::vector<std::size_t>{extent[1], extent[2]}, inner(src), inner(dst));
  std::get<loop>(p.records[planes]).body_end = static_cast<std::uint32_t>(p.records.size());
}

// The axes' own lines say how each axis runs.
constexpr const char* volume_note = "the two inner axes plane at a time, each plane finished "
                                    "before the next, then the outer axis; each transform "
                                    "finished before the next";

// When the pairing of an even real line fuses (strategies/chain.hpp, DESIGN.md 9.1). The
// rule: radix 8 (4 on avx2, whose 16 registers two radix-8 butterflies overfill), r2c only
// within L1 (the grid's threshold). A forced radix limit applies alone (tests).
real_fusion fusion_of(const build_settings& settings, target t) {
  if (settings.fuse_real_radix != 0) {
    return {.radix = settings.fuse_real_radix, .min_lanes = 0, .r2c_bytes = SIZE_MAX};
  }
  return {
      .radix = t == target::avx2 ? std::size_t{4} : std::size_t{8},
      .min_lanes = 4,
      .r2c_bytes = settings.profile.grid_block_bytes, // the L1 data cache (task 2.1, X6)
  };
}

// Whether a grid's gathers and scatters prefetch the rows they copy next (passes/grid.inl,
// experiment X26): not on avx2 on AMD's cores, where the EPYC 7763 gained without it (long rows
// 0.920); everywhere else, the Xeon 6240 under avx2 included (it lost 1.14-1.20 without it at
// 2^20 and 2^22), so every other vendor's avx2 plans keep it (DESIGN.md 13.5).
bool copies_prefetch(const build_settings& settings, target t) {
  return t != target::avx2 || settings.profile.vendor != cpu_vendor::amd;
}

// Whether an r2c grid pairs its bins in its last phase where the phase allows (X33): not on
// avx2, where the EPYC 7763 gained nothing from it (0.98-1.02), the separate pass after the
// grid elsewhere. A forced setting applies alone (tests).
bool r2c_pairs_in_grid(const build_settings& settings, target t) {
  if (settings.r2c_pair_last >= 0) {
    return settings.r2c_pair_last != 0;
  }
  return t != target::avx2;
}

// Whether a c2r plan builds its intermediate packed in the output (experiments X60, X63): a
// plane (two axes; volumes keep the block, where X60's teams lost), out of place, an even real
// axis contiguous on the output and an even outer output stride, so that each output row holds
// its h = n/2 complex values (pass_args::packed); by the rules where one transform's half
// spectrum exceeds the profile's c2r_packed_bytes on the target it was measured on
// (c2r_packed_isa). Otherwise the intermediate is that half spectrum in the workspace block.
// build_settings::c2r_packed 0 keeps the block, 1 packs wherever the layout allows (tests).
bool c2r_packed(const shape_view& s, const side& in, const side& out, bool in_place,
                const build_settings& settings, target t) {
  const std::size_t r = s.axes.size();
  if (s.kind != transform_kind::c2r || r != 2 || in_place || settings.c2r_packed == 0 ||
      s.axes[r - 1].n % 2 != 0 || out.stride[r - 1] != 1 ||
      !std::all_of(out.stride.begin(), out.stride.end() - 1,
                   [](std::ptrdiff_t v) { return v % 2 == 0; })) {
    return false;
  }
  if (settings.c2r_packed > 0) {
    return true;
  }
  const fftn::isa measured = settings.profile.c2r_packed_isa;
  if (measured != fftn::isa::best &&
      static_cast<unsigned char>(measured) != static_cast<unsigned char>(t)) {
    return false;
  }
  std::size_t bytes = 2 * sizeof(double);
  for (const std::size_t e : in.extent) {
    bytes = e != 0 && bytes > SIZE_MAX / e ? SIZE_MAX : bytes * e;
  }
  if (s.prec == precision::f32) {
    bytes /= 2;
  }
  return bytes > settings.profile.c2r_packed_bytes;
}

// Makes the records from `first` on read the output slot where they read the input: the
// grid of a c2r row runs in place on the output, which the pairing pass has written (the
// grid builder reads its input from the input slot, and with in_place set touches it only
// through phase 1's gather and that phase's loop).
void read_output_for_input(program& p, std::size_t first) {
  for (std::size_t i = first; i < p.records.size(); ++i) {
    if (auto* st = std::get_if<stage>(&p.records[i])) {
      if (st->src == slot::in) {
        st->src = slot::out;
      }
      continue;
    }
    loop& l = std::get<loop>(p.records[i]);
    if (l.step[0] != 0) {
      l.step[1] = l.step[0];
      l.step[0] = 0;
    }
  }
}

// Makes the records [first, end) read slot `to` (the output or the workspace block) where they
// read the input: a row's strategy after the input edge's pass (build, row_strategy). The
// loops move `to` by the bytes they moved the input; a loop that moved both is a contradiction.
void retarget_input(std::vector<record>& records, std::size_t first, slot to) {
  const std::size_t k = to == slot::out ? 1 : 2;
  for (std::size_t i = first; i < records.size(); ++i) {
    if (auto* st = std::get_if<stage>(&records[i])) {
      if (st->src == slot::in) {
        st->src = to;
      }
      continue;
    }
    loop& l = std::get<loop>(records[i]);
    if (l.step[0] != 0) {
      if (l.step[k] != 0 && l.step[k] != l.step[0]) {
        throw std::logic_error("fftn: internal: a loop moves the input and its new slot apart");
      }
      l.step[k] = l.step[0];
      l.step[0] = 0;
    }
    if (l.start[0] != 0) {
      if (l.start[k] != 0 && l.start[k] != l.start[0]) {
        throw std::logic_error("fftn: internal: a loop starts the input and its new slot apart");
      }
      l.start[k] = l.start[0];
      l.start[0] = 0;
    }
  }
}

// The strategies (plan_core::strategy_name) and reasons (plan_core::edge_fallback) that
// build_with_edges and name_forced_strategy compare across builds (task 5.1c).
constexpr std::string_view batch_leaves_name = "the chain on batch leaves";
constexpr std::string_view batch_one_row_name = "the chain, its batch leaves one row at a time";
constexpr std::string_view c2c_input_pass_reason =
    "the input edge's pass leaves the rows in the output, where the strategy is chosen in place";
// A fused chain's in place of c2c_input_pass_reason: the separate passes' plan chose the chain
// for the rows in the output, and the fused chain, which has no input pass, takes over.
constexpr std::string_view c2c_fused_input_reason =
    "the strategy is chosen for the rows where an input edge's pass would leave them, the "
    "chain, which takes the factor fused instead";
constexpr std::string_view batch_one_row_reason =
    "batch leaves, one row at a time between the edge passes";
// The points of a block of rows in the edge driver's row-block form (task 5.2a): the block's
// lines in the workspace block, 32 KiB in double, stay in L1 between the edge passes and the
// rows' transforms; chunk_points' 2048, unmeasured otherwise.
constexpr std::size_t edge_row_block_points = 2048;

} // namespace

void lay_out_workspace(plan_core& c, std::size_t scratch, std::size_t complex_bytes, bool stagger_b,
                       std::size_t callback_scratch) {
  constexpr std::size_t line_alignment = 64;
  const auto round_up = [](std::size_t v) {
    return (v + line_alignment - 1) / line_alignment * line_alignment;
  };
  c.scratch_line = scratch;
  const std::size_t line_bytes = round_up(scratch * complex_bytes);
  // A grid's scratch line b starts an odd number of 64-byte lines (seven) after a whole
  // multiple of a's size, so that the passes ping-ponging between them do not read and
  // write the same cache sets (DESIGN.md 10, rule 4): a grid's scratch lines are a block of
  // a power-of-two size for most lengths. Experiment X4 measured 8-9 % at 2^22 on the Mac
  // and no case slower; the chain's lines were not part of it.
  const std::size_t stagger = stagger_b ? 7 * 64 : 0;
  c.offset_b = line_bytes + stagger;
  c.offset_block = 2 * line_bytes + stagger;
  std::size_t used = c.offset_block + c.block_elements * complex_bytes;
  // Each slot ends with the callback area (task 5.3a), from a 128-byte boundary: the largest of
  // the callbacks' scratch, one member never running two at once, plus the 64 bytes that align
  // it where the slot starts on a 64-byte boundary only (serial execution; core/callbacks.cpp,
  // callback_area). Plans without callback scratch keep their layout.
  if (callback_scratch != 0) {
    constexpr std::size_t area_alignment = 128;
    c.offset_callback = (used + area_alignment - 1) / area_alignment * area_alignment;
    c.callback_bytes = callback_scratch;
    // scratch_bytes is bounded at validation (core/edges.cpp), so nothing here overflows.
    used = c.offset_callback + line_alignment + callback_scratch;
  }
  c.bytes = used == 0 ? 0 : used + line_alignment;
  c.slot_bytes = (used + sync_line - 1) / sync_line * sync_line;
}

target_profile plan_profile(const plan_core& c, std::size_t complex_lanes,
                            const build_settings& settings) {
  target_profile profile = profile_of(c.isa_target, complex_lanes, settings.profile.vendor);
  if (settings.leaf_limit != 0) {
    profile.leaf_limit = settings.leaf_limit;
  }
  if (settings.layout != layout_choice::rules) {
    profile.split = settings.layout == layout_choice::split;
  }
  // X69: on the 2 MiB-L2 key the fit limits serve that key's table, not the target's.
  if (settings.profile.fit_large_l2) {
    profile.fit_overrides = large_l2_fit_overrides(c.isa_target, complex_lanes);
  }
  if (settings.rader == 0) {
    profile.rader_below = 0;
  } else if (settings.rader > 0) {
    profile.rader_from = 0;
    profile.rader_below = std::numeric_limits<std::size_t>::max();
  }
  return profile;
}

template <class T>
chain_builder<T> plan_chain(plan_core& c, const target_profile& profile,
                            const build_settings& settings) {
  const kernels_of<T>& kernels = of<T>(*find_kernels(c.isa_target));
  chain_builder<T> chain(kernels, profile, tables_of<T>(c), settings.chunk_points, settings.radices,
                         settings.bluestein_above, fusion_of(settings, c.isa_target));
  // X32: only on the machines whose profile keeps long powers of two as chains, the ones
  // the radix sequences were measured on (the Xeon lost with chains of either).
  if (settings.profile.pow2_chain_bytes != 0) {
    chain.use_long_rows(settings.profile.grid_min_bytes, settings.profile.pow2_radix8_bytes);
  }
  // X36: a convolution is always a chain, so a long one takes the long-row radices beyond
  // the profile's long_convolution_bytes, radix 8 throughout where the profile sets no
  // limit; a Bluestein length's power-of-two part follows the profile.
  chain.use_long_convolutions(settings.profile.long_convolution_bytes,
                              settings.profile.pow2_chain_bytes != 0
                                  ? settings.profile.pow2_radix8_bytes
                                  : std::numeric_limits<std::size_t>::max(),
                              settings.profile.bluestein_pow2_part_bytes);
  chain.use_whole_lanes(settings.profile.bluestein_whole_lanes);
  chain.force_convolution_length(settings.bluestein_length);
  if (const narrow_leaf_table* narrow = find_narrow_leaves(c.isa_target)) {
    chain.use_narrow_leaves(&of<T>(*narrow));
  }
  if (const pair_leaf_table* pairs = find_pair_leaves(c.isa_target);
      pairs != nullptr && settings.pair_leaves) {
    // Another target's leaves in double only (X37): in float they would also change the
    // lanes' chains (strategies/lanes.cpp), which nothing has measured.
    const pair_leaf_table* more =
        std::is_same_v<T, double> ? find_more_pair_leaves(c.isa_target) : nullptr;
    chain.use_pair_leaves(&of<T>(*pairs), more != nullptr ? &of<T>(*more) : nullptr);
  }
  return chain;
}

template chain_builder<double> plan_chain<double>(plan_core& c, const target_profile& profile,
                                                  const build_settings& settings);
template chain_builder<float> plan_chain<float>(plan_core& c, const target_profile& profile,
                                                const build_settings& settings);

namespace {

// A row's stored runs (none where the side has none).
const std::vector<line_run>& row_runs(const side_runs& runs) {
  static const std::vector<line_run> none;
  return runs.any() ? runs.of(0) : none;
}

// The longest block a callback stage over a row of n elements passes (core/callbacks.cpp: one
// chunk of about chunk_points elements, the row whole up to twice that).
std::size_t callback_block_elements(std::size_t n, std::size_t chunk_points) {
  const std::size_t chunks = std::max<std::size_t>(n / std::max<std::size_t>(chunk_points, 1), 1);
  return (n + chunks - 1) / chunks;
}

// The stages of a c2c row's input edge with a callback (task 5.3a), appended to `st`, read from
// `from`: where the chain then reads. The callback runs where the row is in the side's natural
// order: in place on the input itself, else in the workspace block, which the callback fills
// (produce: with zeros between the stored runs) or which a copy of the input fills first (modify:
// the run read stage, or a copy, so that the plan's input is not changed). The side's factor and
// scale follow in a separate pass, unless the factor is fused into the chain's first pass, so
// that the callback sees the values before any vocabulary (DESIGN.md 12.3).
endpoint row_input_with_callback(precision prec, const edge_side& in, std::size_t n, endpoint from,
                                 bool in_place, std::size_t chunk_points,
                                 std::vector<std::unique_ptr<callback_site>>& sites,
                                 std::vector<stage>& st) {
  const endpoint block{.where = slot::block, .stride = 1};
  const std::ptrdiff_t bytes =
      static_cast<std::ptrdiff_t>(2 * (prec == precision::f64 ? sizeof(double) : sizeof(float)));
  const std::vector<line_run>& runs = row_runs(in.runs);
  if (in.call.any()) {
    // An in-place plan's input is its output's memory, which the plan may write (runs are
    // refused in place, core/runs.cpp).
    const endpoint target = in_place ? endpoint{.where = slot::out, .stride = from.stride} : block;
    if (in.call.mode == callback_mode::produce) {
      std::size_t next = 0;
      for (const line_run& run : runs) {
        if (run.logical > next) {
          run_zero_line(
              prec, run.logical - next,
              {.dst = slot::block, .dst_offset = static_cast<std::ptrdiff_t>(next) * bytes},
              chunk_points, false, st);
        }
        next = run.logical + run.count;
      }
      if (!runs.empty() && next < n) {
        run_zero_line(prec, n - next,
                      {.dst = slot::block, .dst_offset = static_cast<std::ptrdiff_t>(next) * bytes},
                      chunk_points, false, st);
      }
    } else if (!in_place) {
      if (!runs.empty()) {
        // The runs read without the scale, which the vocabulary's pass applies after the
        // callback.
        edge_side read;
        read.runs = in.runs;
        run_line_stages(prec, read, false, n, runs, {.src = from.where, .src_stride = from.stride},
                        chunk_points, false, st);
      } else {
        stage copy = edge_copy_stage(prec, n, from.where, from.stride, slot::block, 1, chunk_points,
                                     false, edge_rows{});
        copy.name = "edge in: the input copied for its callback";
        st.push_back(std::move(copy));
      }
    }
    if (runs.empty()) {
      st.push_back(callback_stage(prec, false, in.call, false, n, 0, target.where, target.stride, 0,
                                  chunk_points, sites));
    } else {
      // A block never crosses a gap between runs; coordinates are logical.
      for (const line_run& run : runs) {
        st.push_back(
            callback_stage(prec, false, in.call, false, run.count, run.logical, slot::block, 1,
                           static_cast<std::ptrdiff_t>(run.logical) * bytes, chunk_points, sites));
      }
    }
    from = target;
  } else if (!runs.empty()) {
    run_line_stages(prec, in, false, n, runs, {.src = from.where, .src_stride = from.stride},
                    chunk_points, false, st);
    return block;
  }
  // The vocabulary: a factor that is not fused, or the scale where resolve_edges put it on this
  // side (with the edges' separate passes).
  if ((in.separate && in.kind != factor_kind::none) || in.scale != 1.0) {
    edge_side vocabulary = in;
    vocabulary.runs = {};
    st.push_back(edge_pass_stage(prec, vocabulary, false, n, from.where, from.stride, slot::block,
                                 1, chunk_points, false, in.axis_stride));
    return block;
  }
  return from;
}

// Where row_input_with_callback runs the input's callback, for describe().
std::string row_input_where(const edge_side& in, bool in_place) {
  if (in_place) {
    return "in the input, in place";
  }
  if (in.call.mode == callback_mode::produce) {
    return "in the workspace block";
  }
  return in.runs.any() ? "in the workspace block, the runs read there first"
                       : "in the workspace block, after a copy of the input";
}

// The stages of a c2c row's output edge with a callback after the chain (task 5.3a), appended to
// `st`. direct: a modify callback without accumulation or runs, on the output itself, which the
// chain (or the side's separate pass from the block) has written. Otherwise the chain has ended
// in the workspace block: the side's factor and scale there in a pass of their own, the callback,
// then (unless it consumes the row) the store of the row or its runs, or the accumulation, with
// nothing else applied (DESIGN.md 12.3).
void row_output_with_callback(precision prec, const edge_side& out, std::size_t n, endpoint to,
                              bool direct, std::size_t chunk_points,
                              std::vector<std::unique_ptr<callback_site>>& sites,
                              std::vector<stage>& st) {
  const std::ptrdiff_t bytes =
      static_cast<std::ptrdiff_t>(2 * (prec == precision::f64 ? sizeof(double) : sizeof(float)));
  const std::vector<line_run>& runs = row_runs(out.runs);
  edge_side vocabulary = out;
  vocabulary.runs = {};
  vocabulary.accumulate = false;
  if (direct) {
    if (out.separate) {
      st.push_back(edge_pass_stage(prec, vocabulary, true, n, slot::block, 1, to.where, to.stride,
                                   chunk_points, false, out.axis_stride));
    }
    st.push_back(callback_stage(prec, false, out.call, true, n, 0, to.where, to.stride, 0,
                                chunk_points, sites));
    return;
  }
  if ((out.separate && out.kind != factor_kind::none) || out.scale != 1.0) {
    st.push_back(edge_pass_stage(prec, vocabulary, true, n, slot::block, 1, slot::block, 1,
                                 chunk_points, false, out.axis_stride));
  }
  if (runs.empty()) {
    st.push_back(
        callback_stage(prec, false, out.call, true, n, 0, slot::block, 1, 0, chunk_points, sites));
  } else {
    for (const line_run& run : runs) {
      st.push_back(callback_stage(prec, false, out.call, true, run.count, run.logical, slot::block,
                                  1, static_cast<std::ptrdiff_t>(run.logical) * bytes, chunk_points,
                                  sites));
    }
  }
  if (out.call.mode == callback_mode::consume) {
    return;
  }
  edge_side store;
  store.accumulate = out.accumulate;
  store.runs = out.runs;
  if (!runs.empty()) {
    run_line_stages(prec, store, true, n, runs,
                    {.src = slot::block, .dst = to.where, .dst_stride = to.stride}, chunk_points,
                    false, st);
  } else {
    st.push_back(edge_pass_stage(prec, store, true, n, slot::block, 1, to.where, to.stride,
                                 chunk_points, false, 0));
  }
}

template <class T>
void build(plan_core& c, const shape_view& s, const side& in, const side& out, const options& opts,
           const build_settings& settings, const edge_plan* edge_ops,
           std::vector<edge_loop>& edge_loops, std::vector<edge_loop>& tail_loops) {
  c.isa_target = plan_target_of(opts);
  const kernels_of<T>& kernels = of<T>(*find_kernels(c.isa_target));
  const target_profile profile = plan_profile(c, kernels.complex_lanes, settings);
  chain_builder<T> chain = plan_chain<T>(c, profile, settings);
  program& p = c.prog;
  const std::size_t r = s.axes.size();
  const auto complex_bytes = static_cast<std::ptrdiff_t>(2 * sizeof(T));

  p.transforms = 1;
  // An input a callback produces and an output a callback consumes are never addressed, and may
  // be null (task 5.3a): no loop moves them, so no offset is applied to a null pointer.
  const bool in_unread = edge_ops != nullptr && edge_ops->in.call.any() &&
                         edge_ops->in.call.mode == callback_mode::produce;
  const bool out_unwritten = edge_ops != nullptr && edge_ops->out.call.any() &&
                             edge_ops->out.call.mode == callback_mode::consume;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    p.records.emplace_back(loop{
        .kind = loop_kind::batch,
        .count = s.batches[b].count,
        .step = {in_unread ? 0 : in.dist[b] * in.element_bytes,
                 out_unwritten ? 0 : out.dist[b] * out.element_bytes, 0},
    });
    p.transforms *= s.batches[b].count;
  }
  p.batch_loops = s.batches.size();
  // X57: a two-pass row plan reads 12-16 streams at once, which pays only while the batch's
  // rows and their output stay in the cache (machine_profile::fit_row_bytes; in place also
  // fit_in_place_bytes, X57 amendment 2). X69: on the 2 MiB-L2 key, 20-32 streams within L2.
  // X70: out of place, rows longer than a page also within fit_long_row_bytes. X80: out of
  // place, rows and output also within fit_out_of_place_bytes.
  if (s.kind == transform_kind::c2c && r == 1 && settings.profile.fit_row_bytes != 0 &&
      in.stride[0] == 1 && out.stride[0] == 1) {
    const machine_profile& m = settings.profile;
    const std::size_t sides = c.in_place ? 1 : 2;
    const std::size_t row_bytes =
        checked_product(std::array{c.lengths[0], std::size_t{2} * sizeof(T)});
    const std::size_t touched = checked_product(std::array{row_bytes, p.transforms, sides});
    constexpr std::size_t page_bytes = 4096;
    const std::size_t limit =
        c.in_place ? std::min(m.fit_row_bytes, m.fit_in_place_bytes)
                   : std::min({m.fit_row_bytes, m.fit_out_of_place_bytes,
                               row_bytes > page_bytes ? m.fit_long_row_bytes
                                                      : std::numeric_limits<std::size_t>::max()});
    chain.use_row_fit(c.lengths[0], touched != 0 && touched <= limit);
  }

  const view in_view{.where = slot::in, .stride = in.stride, .element_bytes = in.element_bytes};
  const view out_view{.where = slot::out, .stride = out.stride, .element_bytes = out.element_bytes};
  std::size_t scratch = 0;
  const auto lines = [&](std::size_t axis, const std::vector<std::size_t>& extent, const view& src,
                         const view& dst, std::size_t line_scratch, std::vector<stage>&& stages) {
    scratch = std::max(scratch, line_scratch);
    const std::size_t outermost = p.records.size();
    add_phase(p, axis, extent, src, dst, std::move(stages));
    // Columns alone through the chain (not the rows of the last axis).
    const std::size_t last = extent.size() - 1;
    if (settings.profile.fair_line_runs && extent.size() == 2 && axis != last &&
        outermost < p.records.size()) {
      if (loop* l = std::get_if<loop>(&p.records[outermost])) {
        l->shares_lines = lines_share_cache_lines(
            static_cast<std::size_t>(magnitude(dst.stride[last]) * dst.element_bytes),
            static_cast<std::size_t>(magnitude(dst.stride[axis]) * dst.element_bytes),
            settings.profile.line_bytes);
      }
    }
  };
  const auto at = [](const view& v, std::size_t axis) {
    return endpoint{.where = v.where, .stride = v.stride[axis]};
  };
  const int sign = static_cast<int>(s.dir);
  // Axis d of the plan as the shape lists it (plan_core::axis_of), for describe().
  const auto listed = [&c](std::size_t d) { return c.axis_of.empty() ? d : c.axis_of[d]; };

  // The edges of a multi-axis or real plan (core/edges.hpp; c2c rows place theirs in the row's
  // chain, c2c_axis below): every factor and accumulation is an edge phase, a phase of its own
  // over the rows of the edge's side (the last axis), the input's before the transform and the
  // output's after it, whose loops move the factor cursors by the factor's strides along the
  // other axes (edge_loops). The input edge's phase writes the output where the transform can
  // run in place there (c2c without accumulation), else the workspace block, which then holds
  // one transform: the input side's elements, real rows padded to 2(n/2 + 1) values as an
  // in-place real transform's are. With accumulation the transform ends in the block (the
  // output side's elements, in the same layout), and the output edge's phase adds it to the
  // output; else the output edge's phase runs in place on the output. One block serves both,
  // as an in-place transform's array does.
  const bool edge_phases = edge_ops != nullptr && (r > 1 || s.kind != transform_kind::c2c);
  view src0 = in_view;  // where the transform's first phase reads
  view dst0 = out_view; // where its phases write
  view block_complex{.where = slot::block, .stride = {}, .element_bytes = complex_bytes};
  view block_real{
      .where = slot::block, .stride = {}, .element_bytes = static_cast<std::ptrdiff_t>(sizeof(T))};
  std::size_t edge_block = 0;
  if (edge_phases) {
    const std::size_t last_n =
        s.kind == transform_kind::c2c ? c.lengths[r - 1] : c.lengths[r - 1] / 2 + 1;
    block_complex.stride.assign(r, 0);
    block_real.stride.assign(r, 0);
    std::ptrdiff_t span = 1;
    for (std::size_t i = r; i > 0; --i) {
      block_complex.stride[i - 1] = span;
      block_real.stride[i - 1] = i == r ? 1 : 2 * span;
      span = checked_mul(span, to_signed(i == r ? last_n : c.lengths[i - 1], "shape"), "shape");
    }
    static_cast<void>(checked_mul(span, complex_bytes, "shape"));
    // Stored runs on the output (task 5.1d) leave the output short of the whole transform: it
    // ends in the block, as with accumulation, and the output's phase stores the runs.
    const bool to_block = edge_ops->out.accumulate || edge_ops->out.runs.any();
    if (edge_ops->in.separate) {
      src0 = s.kind == transform_kind::c2c && !to_block
                 ? out_view
                 : (s.kind == transform_kind::r2c ? block_real : block_complex);
    }
    if (to_block) {
      dst0 = s.kind == transform_kind::c2r ? block_real : block_complex;
    }
    if (src0.where == slot::block || dst0.where == slot::block) {
      edge_block = static_cast<std::size_t>(span);
    }
  }
  // The edge phase of a side with stored runs (task 5.1d), from `from` to `to` over lines of
  // `extent` along the last axis: the input's reads the stored runs into the whole transform with
  // zeros elsewhere, the output's stores (or accumulates) only the runs. Each axis but the last is
  // cut into pieces, its runs and (input only) the gaps between them, and every combination of
  // pieces is a nest of its own over the pieces' extents, its lines' stages offset to the
  // pieces' first elements: a gap's lines are zeros, the others' lines run_line_stages. The
  // factor is none (build_with_runs), so no cursor moves.
  const auto run_phase = [&](program& q, bool output, const std::vector<std::size_t>& extent,
                             const view& from, const view& to, bool real) {
    const edge_side& e = output ? edge_ops->out : edge_ops->in;
    const std::size_t row = r - 1;
    const view& stored = output ? to : from;
    const view& logical = output ? from : to;
    struct piece {
      std::size_t logical = 0;
      std::size_t stored = 0;
      std::size_t count = 0;
      bool gap = false;
    };
    std::vector<std::vector<piece>> pieces(r);
    for (std::size_t d = 0; d + 1 < r; ++d) {
      const std::vector<line_run>& runs = e.runs.of(listed(d));
      if (runs.empty()) {
        pieces[d].push_back({.count = extent[d]});
        continue;
      }
      std::size_t next = 0;
      for (const line_run& run : runs) {
        if (!output && run.logical > next) {
          pieces[d].push_back({.logical = next, .count = run.logical - next, .gap = true});
        }
        pieces[d].push_back({.logical = run.logical, .stored = run.stored, .count = run.count});
        next = run.logical + run.count;
      }
      if (!output && next < extent[d]) {
        pieces[d].push_back({.logical = next, .count = extent[d] - next, .gap = true});
      }
    }
    pieces[row].push_back({.count = extent[row]});
    std::vector<std::size_t> choice(r, 0);
    for (;;) {
      std::vector<std::size_t> sub = extent;
      std::ptrdiff_t stored_offset = 0;
      std::ptrdiff_t logical_offset = 0;
      bool gap = false;
      for (std::size_t d = 0; d + 1 < r; ++d) {
        const piece& pc = pieces[d][choice[d]];
        sub[d] = pc.count;
        gap = gap || pc.gap;
        stored_offset += static_cast<std::ptrdiff_t>(pc.stored) * stored.stride[d];
        logical_offset += static_cast<std::ptrdiff_t>(pc.logical) * logical.stride[d];
      }
      stored_offset *= stored.element_bytes;
      logical_offset *= logical.element_bytes;
      const std::vector<std::size_t> open = open_loops(q, row, row, sub, from, to);
      std::vector<stage> line;
      const run_line line_at{
          .src = from.where,
          .src_stride = from.stride[row],
          .src_offset = output ? logical_offset : stored_offset,
          .dst = to.where,
          .dst_stride = to.stride[row],
          .dst_offset = output ? stored_offset : logical_offset,
      };
      if (gap) {
        run_zero_line(s.prec, extent[row], line_at, settings.chunk_points, real, line);
      } else {
        run_line_stages(s.prec, e, output, extent[row], e.runs.of(listed(row)), line_at,
                        settings.chunk_points, real, line);
      }
      for (stage& st : line) {
        q.records.emplace_back(std::move(st));
      }
      close_loops(q, open);
      // The next combination of pieces, the last axis' pieces fastest.
      std::size_t d = r - 1;
      while (d > 0) {
        --d;
        if (++choice[d] < pieces[d].size()) {
          break;
        }
        choice[d] = 0;
        if (d == 0) {
          return;
        }
      }
      if (r == 1) {
        return;
      }
    }
  };
  // Pruning by restricted loops (DESIGN.md 12.1; task 5.1d): the phase of axis d, which
  // `build` emits over a given extent from src to dst, run only over the lines it needs. Along an
  // axis e not transformed yet (done[e] false) whose input has stored runs, the lines outside them
  // are zeros, and stay zeros, in a phase that works in place (the input's phase wrote zeros
  // there and every phase before kept them); along an axis already transformed whose output has
  // stored runs, the lines outside them are not wanted. Each combination of the pieces that
  // remain is the phase over the pieces' extents inside a loop of one iteration whose start moves
  // the slots to the pieces' first elements. Without runs, or with nothing to restrict, `build`
  // runs once over the whole extent.
  const auto pruned = [&](std::size_t d, const std::vector<std::size_t>& extent, const view& src,
                          const view& dst, const std::vector<bool>& done, const auto& build) {
    const bool runs = edge_ops != nullptr && (edge_ops->in.runs.any() || edge_ops->out.runs.any());
    const bool in_place_phase = src.where == dst.where && src.stride == dst.stride;
    std::vector<std::vector<line_run>> pieces(r);
    bool restricted = false;
    for (std::size_t e = 0; runs && e < r; ++e) {
      if (e == d || extent[e] == 1) {
        continue;
      }
      const side_runs& held = done[e] ? edge_ops->out.runs : edge_ops->in.runs;
      if (held.any() && !held.of(listed(e)).empty() && (done[e] || in_place_phase)) {
        pieces[e] = held.of(listed(e));
        restricted = true;
      }
    }
    // The same slot moved by both views by different amounts cannot be one start.
    if (restricted && src.where == dst.where && !in_place_phase) {
      restricted = false;
    }
    if (!restricted) {
      build(extent);
      return;
    }
    for (std::size_t e = 0; e < r; ++e) {
      if (pieces[e].empty()) {
        pieces[e].push_back({.logical = 0, .stored = 0, .count = extent[e]});
      }
    }
    std::size_t kept_notes = SIZE_MAX;
    std::vector<std::size_t> choice(r, 0);
    for (;;) {
      std::vector<std::size_t> sub = extent;
      std::array<std::ptrdiff_t, moving_slots> start{};
      for (std::size_t e = 0; e < r; ++e) {
        const line_run& pc = pieces[e][choice[e]];
        sub[e] = pc.count;
        const auto moved_by =
            slot_bytes_along(e, static_cast<std::ptrdiff_t>(pc.logical), src, dst);
        for (std::size_t k = 0; k < moving_slots; ++k) {
          start[k] += moved_by[k];
        }
      }
      const std::size_t wrapper = p.records.size();
      p.records.emplace_back(loop{.kind = loop_kind::lines, .count = 1, .start = start});
      build(sub);
      std::get<loop>(p.records[wrapper]).body_end = static_cast<std::uint32_t>(p.records.size());
      // One combination's notes describe the axis.
      if (kept_notes == SIZE_MAX) {
        kept_notes = c.axis_notes.size();
      } else {
        c.axis_notes.resize(kept_notes);
      }
      std::size_t e = r;
      while (e > 0) {
        --e;
        if (++choice[e] < pieces[e].size()) {
          break;
        }
        choice[e] = 0;
        if (e == 0) {
          return;
        }
      }
    }
  };
  // An edge phase of program q from `from` to `to` over lines of `extent` along the last axis,
  // its loops' factor steps into `loops`; `rows`: a lanes block's lines (the lanes over adjacent
  // transforms).
  const auto edge_phase = [&](program& q, std::vector<edge_loop>& loops, bool output,
                              const std::vector<std::size_t>& extent, const view& from,
                              const view& to, const edge_rows& rows) {
    const edge_side& e = output ? edge_ops->out : edge_ops->in;
    const bool real = output ? s.kind == transform_kind::c2r : s.kind == transform_kind::r2c;
    const std::size_t row = r - 1;
    if (e.runs.any()) {
      run_phase(q, output, extent, from, to, real);
      return;
    }
    const std::vector<std::size_t> open = open_loops(q, row, row, extent, from, to);
    q.records.emplace_back(
        edge_pass_stage(s.prec, e, output, extent[row], from.where, from.stride[row], to.where,
                        to.stride[row], settings.chunk_points, real,
                        e.kind != factor_kind::none ? e.axis_strides.at(listed(row)) : 0,
                        e.accumulate ? edge_mode::accumulate : edge_mode::store, rows));
    close_loops(q, open);
    std::size_t k = 0;
    for (std::size_t d = 0; d < extent.size(); ++d) {
      if (d == row || extent[d] == 1) {
        continue; // open_loops opens no loop for these
      }
      edge_loop l{.record = open.at(k++)};
      if (e.kind != factor_kind::none) {
        l.step[output ? 1 : 0] =
            e.axis_strides.at(listed(d)) * static_cast<std::ptrdiff_t>(e.factor_bytes);
      }
      loops.push_back(l);
    }
  };
  // A strided complex axis d as the lanes over blocks of adjacent columns where they fit
  // (column_block; DESIGN.md 9.3, experiments X40 and X41): returns whether it was built.
  // `axis` names d in describe() when `extent` is not the plan's (a volume's plane). `real`:
  // column_block's rule for the complex axes of a real plan (rows of n/2 + 1 columns), by
  // default where the plan is real.
  const auto complex_columns = [&](std::size_t d, const std::vector<std::size_t>& extent,
                                   const view& src, const view& dst, const line_edges& edges,
                                   int dir, const char* what, std::size_t axis = SIZE_MAX,
                                   int real_rule = -1) {
    const bool real = real_rule < 0 ? s.kind != transform_kind::c2c : real_rule != 0;
    const std::size_t block =
        column_block<T>(d, extent, src, dst, edges, settings, chain, kernels.complex_lanes, real);
    if (block == 0) {
      return false;
    }
    const std::size_t n = extent[d];
    // A team program's block where the program runs the columns alone through the chain
    // (column_block, X43 arm B2) takes the chain's radices: the same bits as the program.
    build_settings alone = settings;
    alone.team_program = false;
    const bool chain_columns =
        settings.team_program &&
        column_block<T>(d, extent, src, dst, edges, alone, chain, kernels.complex_lanes, real) == 0;
    const std::vector<std::size_t> rad =
        chain_columns ? chain.radices(n)
                      : column_radices<T>(n, dst.stride[d] * dst.element_bytes, settings, profile,
                                          chain, kernels, c.isa_target != target::avx512);
    scratch = std::max(scratch,
                       add_column_phase<T>(p, chain, d, extent, src, dst, dir, edges, block,
                                           kernels.complex_lanes, rad, settings.profile.line_bytes,
                                           settings.profile.fair_line_runs));
    c.axis_notes.push_back("axis " + std::to_string(listed(axis == SIZE_MAX ? d : axis)) +
                           " (n = " + std::to_string(n) + "): " + what +
                           (what[0] != '\0' ? ", " : "") + "lanes over blocks of " +
                           std::to_string(block) + " adjacent columns, radices " +
                           chain.radix_text(n, rad));
    c.strategy_note = "chain for the last axis' rows, the lanes over blocks of adjacent columns "
                      "for the others; each transform finished before the next";
    return true;
  };

  // Rows with edges whose factors are not fused (5.1b-2): the strategy the rules choose, with
  // separate passes around it (row_edges below). An input edge's pass writes the output, where
  // the strategy then runs in place (c2c out of place: in_eff, in_place_build), or the input's
  // own memory in place, or (a real row out of place) the workspace block, from which the
  // strategy reads (input_slot). The strategies are chosen for the layout they then read.
  const bool row_edges = edge_ops != nullptr && r == 1 && !any_fused(*edge_ops);
  // Why an edge keeps the plan from the strategy the rules would choose, for describe().
  std::string edge_fallback;
  // How a c2c row's chain takes each side's stored runs (task 5.1e), for describe().
  run_pruning in_pruning = run_pruning::none;
  run_pruning out_pruning = run_pruning::none;
  const bool row_input = row_edges && edge_ops->in.separate;
  side in_eff = in;
  bool in_place_build = c.in_place;
  view in_view_eff = in_view;
  slot input_slot = slot::in;
  // The input side's elements of a row: n (c2c, r2c) or n/2 + 1 (c2r).
  const std::size_t n_in = s.kind == transform_kind::c2r ? c.lengths[0] / 2 + 1 : c.lengths[0];
  const std::size_t n_out = s.kind == transform_kind::r2c ? c.lengths[0] / 2 + 1 : c.lengths[0];
  if (row_input && !c.in_place) {
    // Said only where the strategy then differs from the plan's without the edges, and
    // replaced by a reason below that names a strategy the input edge rules out.
    edge_fallback = s.kind == transform_kind::c2c
                        ? std::string(c2c_input_pass_reason)
                        : "the input edge's pass leaves the rows in the workspace block, where "
                          "the strategy is chosen for contiguous rows";
    if (s.kind == transform_kind::c2c) {
      in_eff = out;
      in_place_build = true;
      input_slot = slot::out;
      in_view_eff = out_view;
    } else {
      in_eff.stride = {1};
      in_eff.dist.assign(in.dist.size(), to_signed(n_in, "shape"));
      input_slot = slot::block;
      in_view_eff = view{.where = slot::block, .stride = {1}, .element_bytes = in.element_bytes};
    }
  }
  // Batch leaves a test or fbench forces take the rows from the lanes the rules would choose
  // (avx2 and avx512 in float); the rules' own only where the rules choose no lanes.
  const batch_leaf_set<T>* batch =
      batch_leaves_of<T>(s, in_eff, out, settings, profile, c.isa_target);
  const bool forced_batch = batch != nullptr && settings.batch_leaves != batch_leaf_choice::rules &&
                            settings.strategy == strategy_choice::rules &&
                            !chain.forces(s.axes[0].n);
  const std::size_t lanes = forced_batch ? 0
                                         : lanes_block<T>(s, in_eff, out, settings, profile, chain,
                                                          kernels.complex_lanes);
  // The axes with more than one point, in the order they run (`order_axes`: the last first).
  std::vector<std::size_t> active;
  for (std::size_t i = r; i > 0; --i) {
    if (c.lengths[i - 1] > 1) {
      active.push_back(i - 1);
    }
  }
  // Blocks of adjacent transforms (add_adjacent_transforms): the innermost batch dimension's
  // transforms one element apart on both sides, every axis strided. Each axis's column_block
  // for columns of its length, the transforms being the columns; the narrowest of them. The
  // team program takes these blocks too: the one-thread rule, so that the bits agree.
  // The radices of each axis's lanes over adjacent transforms (column_radices), and whether they
  // are the chain's for every axis: only then can a team run the chain with the program's bits.
  std::vector<std::vector<std::size_t>> adjacent_rad(r);
  bool chain_radices = true;
  if (s.kind == transform_kind::c2c && !s.batches.empty() && in.dist.back() == 1) {
    for (const std::size_t d : active) {
      adjacent_rad[d] = column_radices<T>(c.lengths[d], out.stride[d] * complex_bytes, settings,
                                          profile, chain, kernels, c.isa_target != target::avx512);
      chain_radices = chain_radices && adjacent_rad[d] == chain.radices(c.lengths[d]);
    }
  }
  const bool team_chain = settings.profile.adjacent_team_chain && chain_radices;
  const std::size_t adjacent = [&]() -> std::size_t {
    // Where the profile says and the radices agree, a team program runs the chain
    // (machine_profile::adjacent_team_chain).
    if (lanes != 0 || s.kind != transform_kind::c2c || s.batches.empty() || active.empty() ||
        !settings.lanes_columns || !settings.profile.adjacent_lanes ||
        (settings.team_program && team_chain) ||
        (settings.profile.adjacent_team_chain && !chain_radices) ||
        (settings.strategy != strategy_choice::rules &&
         settings.strategy != strategy_choice::lanes) ||
        settings.in_factor != nullptr || settings.out_factor != nullptr ||
        settings.scale_on_input || in.dist.back() != 1 || out.dist.back() != 1 ||
        s.batches.back().count < 2) {
      return 0;
    }
    // Within L1 (machine_profile::adjacent_min_bytes) the chain keeps a plan with one complex
    // value per register, which the lanes cannot widen, and with more a plan whose axes are all
    // of leaf lengths (X61).
    constexpr std::size_t leaf_axis = 16;
    std::size_t bytes = p.transforms * static_cast<std::size_t>(complex_bytes);
    bool short_axes = true;
    for (const std::size_t d : active) {
      bytes *= c.lengths[d];
      short_axes = short_axes && c.lengths[d] <= leaf_axis;
    }
    if (bytes <= settings.profile.adjacent_min_bytes &&
        (kernels.complex_lanes == 1 || short_axes)) {
      return 0;
    }
    build_settings alone = settings;
    alone.team_program = false;
    std::size_t block = std::numeric_limits<std::size_t>::max();
    for (const std::size_t d : active) {
      if (magnitude(in.stride[d]) == 1 || magnitude(out.stride[d]) == 1) {
        return 0;
      }
      const std::vector<std::size_t> extent{c.lengths[d], s.batches.back().count};
      const auto two = [&](const side& v, slot where) {
        return view{.where = where, .stride = {v.stride[d], 1}, .element_bytes = complex_bytes};
      };
      const std::size_t b = column_block<T>(0, extent, two(in, slot::in), two(out, slot::out),
                                            line_edges(), alone, chain, kernels.complex_lanes);
      if (b == 0) {
        return 0;
      }
      block = std::min(block, b);
    }
    // Accumulation's edge phases end the transform in the workspace block, which a block of
    // adjacent transforms would need laid out by the block (not built).
    if (edge_ops != nullptr && edge_ops->out.accumulate) {
      edge_fallback = "accumulation takes no lanes over adjacent transforms";
      return 0;
    }
    return block;
  }();
  grid_split grid_plan = s.kind == transform_kind::c2c && r == 1
                             ? plan_grid<T>(c.lengths[0], in_eff, out, settings, chain,
                                            kernels.complex_lanes, s.kind, in_place_build)
                             : grid_split{};
  // A real row of one axis whose half-length complex transform is a grid (DESIGN.md 9.1):
  // the grid writes the output and the pairing runs after it in place (r2c), or the pairing
  // writes the output and the grid runs in place there (c2r).
  grid_split real_grid = s.kind != transform_kind::c2c && r == 1 && c.lengths[0] % 2 == 0
                             ? plan_grid<T>(c.lengths[0] / 2, in_eff, out, settings, chain,
                                            kernels.complex_lanes, s.kind, in_place_build)
                             : grid_split{};
  std::vector<std::size_t> depth_tops =
      s.kind == transform_kind::c2c && r == 1 && lanes == 0
          ? depth_tops_of<T>(c.lengths[0], in_eff, out, settings, chain, c.isa_target)
          : std::vector<std::size_t>{};
  std::vector<std::size_t> bluestein_tops =
      s.kind == transform_kind::c2c && r == 1 && lanes == 0 && depth_tops.empty()
          ? bluestein_tops_of<T>(c.lengths[0], in_eff, out, settings, chain, c.isa_target)
          : std::vector<std::size_t>{};
  if (!depth_tops.empty() || !bluestein_tops.empty()) {
    grid_plan = grid_split{};
  }
  std::vector<std::size_t> real_depth_tops =
      s.kind != transform_kind::c2c && r == 1 && lanes == 0
          ? real_depth_tops_of<T>(c.lengths[0], in_eff, out, in_place_build, settings, chain,
                                  c.isa_target)
          : std::vector<std::size_t>{};
  if (!real_depth_tops.empty()) {
    real_grid = grid_split{};
  }
  // Only an intermediate the plan keeps in the workspace block anyway is padded: moving a
  // two-phase plan's intermediate out of the output into the block cost 1-9 % (X26b).
  if (!grid_builder<T>::uses_block(grid_plan, in_place_build)) {
    grid_plan.pad = 0;
  }
  // A c2r row out of place in two phases pairs the bins in phase 1's gather, from the
  // input, which it only reads: the grid then runs from the input to the output like a
  // c2c row, without the workspace block. In place, or in three phases, the block stays
  // and the gather's second read of the input cost more than the separate pass; on the
  // EPYC it lost even out of place (the profile's c2r_paired_gather; X29).
  const bool paired_gather = settings.profile.c2r_paired_gather && s.kind == transform_kind::c2r &&
                             !in_place_build && real_grid.phases == 2;
  const bool real_in_place = in_place_build || (s.kind == transform_kind::c2r && !paired_gather);
  // A real row's input in the workspace block: a strategy that uses the block itself cannot
  // read it there (its own block is the region the input would occupy), nor can the c2r grid
  // and depth-first chain, which read the input in their first gathers of the pairing (not
  // retargeted); such a row is a chain (describe() says so).
  if (input_slot == slot::block) {
    if (real_grid.phases != 0 &&
        (s.kind == transform_kind::c2r || grid_builder<T>::uses_block(real_grid, real_in_place))) {
      real_grid = grid_split{};
      edge_fallback = "a real row's input edge takes no grid here";
    }
    if (!real_depth_tops.empty() && s.kind == transform_kind::c2r) {
      real_depth_tops.clear();
      edge_fallback = "a c2r row's input edge takes no depth-first chain";
    }
  }
  if (!grid_builder<T>::uses_block(real_grid, real_in_place)) {
    real_grid.pad = 0;
  }
  if (lanes == 0 && grid_plan.phases == 0 && depth_tops.empty()) {
    chain.use_batch_leaves(batch);
  }
  // A row's strategy other than the chain with edges: separate passes around it (row_pre,
  // row_post). Otherwise a row's edges are the chain's (c2c rows, c2c_axis) or edge phases.
  const bool row_strategy =
      row_edges && (lanes != 0 || grid_plan.phases != 0 || !depth_tops.empty() ||
                    !bluestein_tops.empty() || real_grid.phases != 0 || !real_depth_tops.empty());
  const bool use_edge_phases = edge_phases && !row_strategy && adjacent == 0;
  // The edge driver's batched forms (machine_profile::edge_batched_forms, task 5.2a): every plan
  // with edges where the profile's reach is all; where it is leaf_rows, a plan of c2c rows whose
  // line's transform is one leaf, which c2c_axis decides (X65's P1, narrowed).
  const edge_forms_reach forms_reach = settings.profile.edge_batched_forms;
  // Callbacks (task 5.3a) keep the walked form: their stages run from the edge records only.
  c.edge_batched_on =
      edge_ops != nullptr && forms_reach == edge_forms_reach::all && !any_callback(*edge_ops);
  // The row-block form: a c2c plan of rows of one batch dimension whose separate edge passes
  // include a factor (so that it runs on the edge driver) takes its rows in blocks of about
  // edge_row_block_points points, each row with a line of its own in the workspace block
  // (c2c_axis), the edge passes and the leaves once per block.
  if (row_edges && !row_strategy && adjacent == 0 && s.kind == transform_kind::c2c &&
      s.batches.size() == 1 && forms_reach != edge_forms_reach::none && !any_callback(*edge_ops) &&
      (edge_ops->in.kind != factor_kind::none || edge_ops->out.kind != factor_kind::none) &&
      !edge_ops->in.runs.any() && !edge_ops->out.runs.any()) {
    const std::size_t rows =
        std::min(s.batches[0].count, std::max<std::size_t>(edge_row_block_points / n_in, 1));
    c.edge_row_block = rows >= 2 ? rows : 0;
  }
  const bool accumulate = edge_ops != nullptr && edge_ops->out.accumulate;
  // The output's prior values for accumulation into a row's strategy, which writes the output:
  // saved in the workspace block past the strategy's own region and the input's, then added to
  // the result (edge_mode::add_prior). The offset is known once the strategy is built.
  std::size_t save_record = SIZE_MAX;
  const bool real_in = s.kind == transform_kind::r2c;
  const bool real_out = s.kind == transform_kind::c2r;
  // The block's complex elements that hold `count` elements of a side (real values in pairs).
  const auto block_units = [](std::size_t count, bool real) {
    return real ? (count + 1) / 2 : count;
  };
  // The input edge's pass of a row strategy (rows: a lanes block's).
  const auto row_input_pass = [&](const edge_rows& rows_of) {
    const slot to = c.in_place ? slot::out : input_slot;
    const std::ptrdiff_t to_stride =
        c.in_place ? in.stride[0] : (input_slot == slot::out ? out.stride[0] : 1);
    return edge_pass_stage(s.prec, edge_ops->in, false, n_in, slot::in, in.stride[0], to, to_stride,
                           settings.chunk_points, real_in, edge_ops->in.axis_strides.at(0),
                           edge_mode::store, rows_of);
  };
  if (use_edge_phases) {
    // Plans with edges whose strategy takes no separate passes are built on the axes' own
    // lines.
    if (lanes != 0 || adjacent != 0 || real_grid.phases != 0 || !real_depth_tops.empty() ||
        grid_plan.phases != 0 || !depth_tops.empty() || !bluestein_tops.empty()) {
      throw std::logic_error("fftn: internal: an edge phase with a strategy that takes none");
    }
    c.block_elements = edge_block;
    if (edge_ops->in.separate) {
      edge_phase(p, edge_loops, false, in.extent, in_view, src0, edge_rows{});
    }
  }
  if (row_strategy && lanes == 0) {
    if (accumulate) {
      save_record = p.records.size();
      p.records.emplace_back(edge_copy_stage(s.prec, n_out, slot::out, out.stride[0], slot::block,
                                             1, settings.chunk_points, real_out, edge_rows{}));
    }
    if (row_input) {
      p.records.emplace_back(row_input_pass(edge_rows{}));
    }
  }
  const std::size_t strategy_first = p.records.size();
  // A c2r plane out of place may keep its intermediate packed in the output (below), but not
  // with edge phases, whose transform ends where the output edge's phase reads it.
  const bool packed_plane = c2r_packed(s, in, out, c.in_place, settings, c.isa_target);
  if (packed_plane && edge_phases) {
    edge_fallback = "an edge phase takes no intermediate packed in the output";
  }
  if (lanes != 0) {
    // The lanes strategy (DESIGN.md 8.5): blocks of `lanes` rows of the innermost batch
    // dimension through the body, the rest in blocks of C through the tail.
    const std::size_t b = s.batches.size() - 1;
    const std::size_t n = c.lengths[0];
    const std::size_t unit = kernels.complex_lanes;
    // With edges (row_strategy), each block of rows: the output's prior values saved into the
    // block past the input's region (accumulation), the input edge's pass, the lanes' stages
    // from where it wrote (in_view_eff), then the output edge's pass (on the output in place,
    // or adding the result to the saved values). Their lines are the block's rows.
    const std::size_t region_in =
        input_slot == slot::block ? block_units(lanes * n_in, real_in) : 0;
    const std::size_t save_offset = region_in * static_cast<std::size_t>(complex_bytes);
    const auto lane_stages = [&](std::size_t block, std::vector<stage>& st) {
      switch (s.kind) {
      case transform_kind::c2c:
        return chain.c2c_lanes(n, sign, at(in_view_eff, 0), in_eff.dist[b], at(out_view, 0),
                               out.dist[b], c2c_edges(opts, settings, true, true), block, st);
      case transform_kind::r2c:
        return chain.r2c_lanes(n, at(in_view_eff, 0), in_eff.dist[b], at(out_view, 0), out.dist[b],
                               opts.scale, block, st);
      case transform_kind::c2r:
        break;
      }
      return chain.c2r_lanes(n, at(in_view_eff, 0), in_eff.dist[b], at(out_view, 0), out.dist[b],
                             opts.scale, block, st);
    };
    const auto block_stages = [&](std::size_t block, std::vector<stage>& st) {
      if (!row_strategy) {
        return lane_stages(block, st);
      }
      const auto n_out_signed = to_signed(n_out, "shape");
      if (accumulate) {
        st.push_back(edge_copy_stage(
            s.prec, n_out, slot::out, out.stride[0], slot::block, 1, settings.chunk_points,
            real_out,
            {.src_dist = out.dist[b], .dst_dist = n_out_signed, .dst_offset = save_offset}));
      }
      if (row_input) {
        st.push_back(row_input_pass({
            .src_dist = in.dist[b],
            .dst_dist = c.in_place ? in.dist[b] : in_eff.dist[b],
            .factor_step = edge_ops->in.batch_steps.at(b),
        }));
      }
      std::vector<stage> core;
      const std::size_t need = lane_stages(block, core);
      for (stage& x : core) {
        st.push_back(std::move(x));
      }
      const edge_side& e = edge_ops->out;
      const std::ptrdiff_t fs = e.kind != factor_kind::none ? e.axis_strides.at(0) : 0;
      if (accumulate) {
        st.push_back(
            edge_pass_stage(s.prec, e, true, n_out, slot::block, 1, slot::out, out.stride[0],
                            settings.chunk_points, real_out, fs, edge_mode::add_prior,
                            {.src_dist = n_out_signed,
                             .dst_dist = out.dist[b],
                             .factor_step = e.kind != factor_kind::none ? e.batch_steps.at(b) : 0,
                             .src_offset = save_offset}));
      } else if (e.separate) {
        st.push_back(edge_pass_stage(s.prec, e, true, n_out, slot::out, out.stride[0], slot::out,
                                     out.stride[0], settings.chunk_points, real_out, fs,
                                     edge_mode::store,
                                     {.src_dist = out.dist[b],
                                      .dst_dist = out.dist[b],
                                      .factor_step = e.batch_steps.at(b)}));
      }
      return need;
    };
    std::vector<stage> body;
    std::vector<stage> tail;
    scratch = std::max(block_stages(lanes, body), block_stages(unit, tail));
    if (row_strategy) {
      c.block_elements = region_in + (accumulate ? block_units(lanes * n_out, real_out) : 0);
    }
    for (stage& st : body) {
      p.records.emplace_back(std::move(st));
    }
    p.lanes = lanes;
    p.lane_unit = unit;
    p.lane_tail.assign(std::make_move_iterator(tail.begin()), std::make_move_iterator(tail.end()));
    c.built = strategy_choice::lanes;
    const bool real = s.kind != transform_kind::c2c;
    c.axis_notes.push_back("axis 0 (n = " + std::to_string(n) + "): on the lanes' tile, " +
                           (real ? chain.real_text(n, s.kind == transform_kind::r2c, false)
                                 : "radices " + chain.radix_text(n)));
    c.strategy_note = "lanes (blocks of " + std::to_string(lanes) +
                      " rows of the innermost batch dimension, one row per vector lane, then "
                      "blocks of " +
                      std::to_string(unit) +
                      "; each row's own chain on a tile of the block's rows)";
  } else if (adjacent != 0) {
    // The lanes strategy over blocks of `adjacent` adjacent transforms (DESIGN.md 9.3, task
    // 4.3): every axis of the block's transforms runs as a lanes phase, its lines one per
    // transform of the block, whose C neighbours are one vector access. A body of those phases
    // for whole blocks and a tail for blocks of C, as for the lanes over rows.
    const std::size_t unit = kernels.complex_lanes;
    const std::vector<std::vector<std::size_t>>& rad = adjacent_rad;
    c.adjacent_team_chain = team_chain;
    // With edges (no accumulation here: the adjacent lambda), the edge phases inside each
    // block's records, their lines the block's transforms: the input edge's into the output
    // (src0), where the axes then run in place, the output edge's in place on the output.
    // The input edge's phase writes the output (c2c, no accumulation: in place too), where the
    // first axis then reads; src0 is the output only for edge phases (r > 1), not for rows.
    const bool adjacent_input = edge_ops != nullptr && edge_ops->in.separate;
    const view& first_src = adjacent_input ? out_view : in_view;
    const auto phases = [&](program& q, std::size_t block) {
      std::vector<edge_loop>& loops = &q == &p ? edge_loops : tail_loops;
      if (adjacent_input) {
        edge_phase(q, loops, false, in.extent, in_view, first_src,
                   {.src_dist = in.dist.back(),
                    .dst_dist = out.dist.back(),
                    .factor_step = edge_ops->in.batch_steps.back()});
      }
      for (std::size_t i = 0; i < active.size(); ++i) {
        const std::size_t d = active[i];
        const view& src = i == 0 ? first_src : out_view;
        std::vector<stage> st;
        scratch = std::max(
            scratch, chain.c2c_lanes(c.lengths[d], sign, at(src, d), 1, at(out_view, d), 1,
                                     c2c_edges(opts, settings, i == 0, i + 1 == active.size()),
                                     block, st, true, rad[d]));
        add_phase(q, d, in.extent, src, out_view, std::move(st));
      }
      if (edge_ops != nullptr && edge_ops->out.separate) {
        edge_phase(q, loops, true, out.extent, out_view, out_view,
                   {.src_dist = out.dist.back(),
                    .dst_dist = out.dist.back(),
                    .factor_step = edge_ops->out.batch_steps.back()});
      }
    };
    phases(p, adjacent);
    program tail;
    phases(tail, unit);
    p.lanes = adjacent;
    p.lane_unit = unit;
    p.lane_tail = std::move(tail.records);
    c.built = strategy_choice::lanes;
    for (const std::size_t d : active) {
      c.axis_notes.push_back("axis " + std::to_string(listed(d)) +
                             " (n = " + std::to_string(c.lengths[d]) + "): lanes, radices " +
                             chain.radix_text(c.lengths[d], rad[d]));
    }
    c.strategy_note = "lanes over blocks of " + std::to_string(adjacent) +
                      " adjacent transforms, one element apart, every axis a lanes phase, " +
                      "then blocks of " + std::to_string(unit);
  } else if (real_grid.phases != 0) {
    const std::size_t n = c.lengths[0];
    const bool r2c = s.kind == transform_kind::r2c;
    c.built = strategy_choice::grid;
    for (std::size_t i = 0; i < real_grid.phases; ++i) {
      c.grid_factors.push_back(real_grid.factors.at(i));
    }
    grid_builder<T> grid(kernels, chain, tables_of<T>(c),
                         settings.grid_fused &&
                             (n / 2) * 2 * sizeof(T) <= settings.profile.grid_fuse_bytes,
                         settings.grid_phase_radix != 0 ? settings.grid_phase_radix
                                                        : settings.profile.grid_phase_radix,
                         copies_prefetch(settings, c.isa_target));
    const bool in_place = real_in_place;
    if (grid_builder<T>::uses_block(real_grid, in_place)) {
      c.block_elements = grid_builder<T>::block_elements(real_grid);
    }
    std::vector<stage> pairing;
    // r2c: the pairing in the grid's last phase, over pairs of mirror column sets, where the
    // last phase's columns allow (X33); else a separate pass after the grid.
    const bool paired_last =
        r2c && r2c_pairs_in_grid(settings, c.isa_target) && grid.pairs_last(real_grid);
    if (paired_last) {
      scratch = grid.c2c_row(real_grid, -1, in_place, opts.scale, false, p, 0, n);
    } else if (r2c) {
      scratch = grid.c2c_row(real_grid, -1, in_place, 1.0, false, p);
      chain.add_pairing(true, n, {slot::out, 1}, {slot::out, 1}, opts.scale, pairing);
      p.records.emplace_back(std::move(pairing.front()));
    } else if (paired_gather) {
      scratch = grid.c2c_row(real_grid, +1, false, opts.scale, false, p, n);
    } else {
      chain.add_pairing(false, n, {slot::in, 1}, {slot::out, 1}, 1.0, pairing);
      p.records.emplace_back(std::move(pairing.front()));
      const std::size_t first = p.records.size();
      scratch = grid.c2c_row(real_grid, +1, true, opts.scale, false, p);
      read_output_for_input(p, first);
    }
    const char* pairing_note = ", then a pairing pass";
    if (paired_last) {
      pairing_note = ", the pairing in its last phase";
    } else if (!r2c) {
      pairing_note = paired_gather ? ", the pairing in its first gather" : ", after a pairing pass";
    }
    c.axis_notes.push_back("axis 0 (n = " + std::to_string(n) + "): " + (r2c ? "r2c" : "c2r") +
                           " through a half-length complex " + grid.note(real_grid) + pairing_note);
    c.strategy_note =
        std::string("grid for the half-length complex transform (") +
        std::to_string(real_grid.phases) + " phases, each depth-first over blocks of columns" +
        (c.block_elements != 0 ? "; an intermediate array in the workspace block" : "") +
        "), each transform finished before the next";
  } else if (!depth_tops.empty()) {
    // The depth-first chain (strategies/chain.hpp, c2c_depth): the sub-rows' first pass, a
    // loop over the sub-rows, each block's chain inside it, then the top passes (the inner
    // one in a loop over the first top factor's runs).
    const std::size_t n = c.lengths[0];
    const std::size_t subs = checked_product(depth_tops);
    const std::size_t m = n / subs;
    c.built = strategy_choice::depth;
    c.depth_tops = depth_tops;
    // In place the blocks go to the workspace block: the first pass reads the whole input.
    const slot work = in_place_build ? slot::block : slot::out;
    if (in_place_build) {
      c.block_elements = n;
    }
    typename chain_builder<T>::depth_stages parts =
        chain.c2c_depth(n, depth_tops, sign, opts.scale, work, input_slot);
    const auto looped = [&](std::size_t count, std::size_t step, std::vector<stage>& body) {
      const std::size_t first = p.records.size();
      loop l{.kind = loop_kind::lines, .count = count};
      l.step[in_place_build ? 2 : 1] = static_cast<std::ptrdiff_t>(step) * complex_bytes;
      p.records.emplace_back(l);
      for (stage& st : body) {
        p.records.emplace_back(std::move(st));
      }
      std::get<loop>(p.records[first]).body_end = static_cast<std::uint32_t>(p.records.size());
    };
    for (stage& st : parts.first) {
      p.records.emplace_back(std::move(st));
    }
    looped(subs, m, parts.sub);
    if (!parts.mid.empty()) {
      looped(depth_tops[0], n / depth_tops[0], parts.mid);
    }
    for (stage& st : parts.top) {
      p.records.emplace_back(std::move(st));
    }
    scratch = parts.scratch;
    c.axis_notes.push_back("axis 0 (n = " + std::to_string(n) +
                           "): " + chain.depth_text(n, depth_tops));
    c.strategy_note = std::string("depth-first chain (the sub-rows' first pass over the row, "
                                  "each sub-row's chain finished on its block, "
                                  "decimation-in-time top passes") +
                      (in_place_build ? "; the blocks in the workspace block" : " in place") +
                      "), each transform finished before the next";
  } else if (!real_depth_tops.empty()) {
    // An even real row out of place whose half-length transform is a depth-first chain
    // (experiment X72). r2c: the chain reads the reals as h complex values and leaves its
    // blocks and its result in the output, then the pairing runs there in place. c2r: the
    // chain's first pass forms the h complex values from the input's bins as it loads them
    // (experiment X78) where the profile's depth_c2r_fused range holds the half length and every
    // run of the pass is whole (c2r_depth_fits); otherwise the pairing writes them into the
    // workspace block (the owner's in a team, which no other transform's lines touch), and the
    // chain reads them from there into the output.
    const std::size_t n = c.lengths[0];
    const std::size_t h = n / 2;
    const bool r2c = s.kind == transform_kind::r2c;
    const std::size_t half_bytes = h * 2 * sizeof(T);
    const bool fused = !r2c && half_bytes >= settings.profile.depth_c2r_fused_min_bytes &&
                       half_bytes <= settings.profile.depth_c2r_fused_max_bytes &&
                       chain.c2r_depth_fits(h, real_depth_tops);
    const std::size_t subs = checked_product(real_depth_tops);
    const std::size_t m = h / subs;
    c.built = strategy_choice::depth;
    c.depth_tops = real_depth_tops;
    std::vector<stage> pairing;
    if (!r2c && !fused) {
      c.block_elements = h;
      chain.add_pairing(false, n, {slot::in, 1}, {slot::block, 1}, 1.0, pairing);
      p.records.emplace_back(std::move(pairing.front()));
    }
    typename chain_builder<T>::depth_stages parts =
        fused ? chain.c2r_depth(n, real_depth_tops, opts.scale)
              : chain.c2c_depth(h, real_depth_tops, r2c ? -1 : +1, r2c ? 1.0 : opts.scale,
                                slot::out, r2c ? input_slot : slot::block, slot::out);
    const auto looped = [&](std::size_t count, std::size_t step, std::vector<stage>& body) {
      const std::size_t first = p.records.size();
      loop l{.kind = loop_kind::lines, .count = count};
      l.step[1] = static_cast<std::ptrdiff_t>(step) * complex_bytes;
      p.records.emplace_back(l);
      for (stage& st : body) {
        p.records.emplace_back(std::move(st));
      }
      std::get<loop>(p.records[first]).body_end = static_cast<std::uint32_t>(p.records.size());
    };
    for (stage& st : parts.first) {
      p.records.emplace_back(std::move(st));
    }
    looped(subs, m, parts.sub);
    if (!parts.mid.empty()) {
      looped(real_depth_tops[0], h / real_depth_tops[0], parts.mid);
    }
    for (stage& st : parts.top) {
      p.records.emplace_back(std::move(st));
    }
    if (r2c) {
      chain.add_pairing(true, n, {slot::out, 1}, {slot::out, 1}, opts.scale, pairing);
      p.records.emplace_back(std::move(pairing.front()));
    }
    scratch = parts.scratch;
    c.axis_notes.push_back("axis 0 (n = " + std::to_string(n) + "): " + (r2c ? "r2c" : "c2r") +
                           " through a half-length complex " +
                           chain.depth_text(h, real_depth_tops) +
                           (r2c     ? ", then a pairing pass"
                            : fused ? ", the pairing fused into its first pass"
                                    : ", after a pairing pass into the workspace block"));
    c.strategy_note = "depth-first chain for the half-length complex transform (the sub-rows' "
                      "first pass over the row, each sub-row's chain finished on its block, "
                      "decimation-in-time top passes in place), each transform finished before "
                      "the next";
  } else if (!bluestein_tops.empty()) {
    // A Bluestein stage whose convolution is two depth-first chains (chain_builder::
    // bluestein_depth, experiment X50): each chain's first pass (the forward one with the
    // chirp pass in), a loop over its sub-rows on their blocks of the workspace block (and
    // with two tops a loop over the inner top pass's runs), its top passes (the backward
    // one's last with the chirp pass out).
    const std::size_t n = c.lengths[0];
    const std::size_t m = chain.bluestein_length_of(n);
    const std::size_t subs = checked_product(bluestein_tops);
    c.built = strategy_choice::depth;
    c.depth_tops = bluestein_tops;
    c.block_elements = m;
    typename chain_builder<T>::bluestein_depth_stages parts = chain.bluestein_depth(
        n, bluestein_tops, sign, {.where = input_slot, .stride = 1},
        {.where = slot::out, .stride = 1}, c2c_edges(opts, settings, true, true));
    const auto looped = [&](std::size_t count, std::size_t step, std::vector<stage>& body) {
      const std::size_t first = p.records.size();
      loop l{.kind = loop_kind::lines, .count = count};
      l.step[2] = static_cast<std::ptrdiff_t>(step) * complex_bytes;
      p.records.emplace_back(l);
      for (stage& st : body) {
        p.records.emplace_back(std::move(st));
      }
      std::get<loop>(p.records[first]).body_end = static_cast<std::uint32_t>(p.records.size());
    };
    for (typename chain_builder<T>::depth_stages* d : {&parts.forward, &parts.backward}) {
      for (stage& st : d->first) {
        p.records.emplace_back(std::move(st));
      }
      looped(subs, m / subs, d->sub);
      if (!d->mid.empty()) {
        looped(bluestein_tops[0], m / bluestein_tops[0], d->mid);
      }
      for (stage& st : d->top) {
        p.records.emplace_back(std::move(st));
      }
    }
    scratch = parts.scratch;
    c.axis_notes.push_back("axis 0 (n = " + std::to_string(n) +
                           "): " + chain.bluestein_depth_text(n, bluestein_tops));
    c.strategy_note = "chain (one Bluestein stage whose convolution runs as two depth-first "
                      "chains, their blocks in the workspace block), each transform finished "
                      "before the next";
  } else if (grid_plan.phases != 0) {
    c.built = strategy_choice::grid;
    for (std::size_t i = 0; i < grid_plan.phases; ++i) {
      c.grid_factors.push_back(grid_plan.factors.at(i));
    }
    grid_builder<T> grid(kernels, chain, tables_of<T>(c),
                         settings.grid_fused &&
                             c.lengths[0] * 2 * sizeof(T) <= settings.profile.grid_fuse_bytes,
                         settings.grid_phase_radix != 0 ? settings.grid_phase_radix
                                                        : settings.profile.grid_phase_radix,
                         copies_prefetch(settings, c.isa_target));
    if (grid_builder<T>::uses_block(grid_plan, in_place_build)) {
      c.block_elements = grid_builder<T>::block_elements(grid_plan);
    }
    scratch = grid.c2c_row(grid_plan, sign, in_place_build, opts.scale, settings.scale_on_input, p);
    c.axis_notes.push_back("axis 0 (n = " + std::to_string(c.lengths[0]) +
                           "): " + grid.note(grid_plan));
    c.strategy_note =
        "grid (Cooley-Tukey in " + std::to_string(grid_plan.phases) +
        " phases, each depth-first over blocks of columns" +
        (c.block_elements != 0 ? "; an intermediate array in the workspace block" : "") +
        "), each transform finished before the next";
  } else if (s.kind == transform_kind::c2c) {
    // Axis d of the plan as axis e of `extent`: the whole transform's, or a volume's plane.
    const auto c2c_axis = [&](std::size_t d, std::size_t e, const std::vector<std::size_t>& extent,
                              const view& src, const view& dst) {
      const bool aliased = src.where == dst.where || c.in_place;
      line_edges line = c2c_edges(opts, settings, d + 1 == r, d == 0);
      std::vector<stage> st;
      std::size_t need = 0;
      if (edge_ops != nullptr && r == 1 &&
          (edge_ops->in.separate || edge_ops->out.separate || any_callback(*edge_ops))) {
        // The edges' separate passes (core/edges.hpp) through the workspace block: the input's
        // into it before the line's chain, which then starts there; the output's out of it
        // after the chain, which then ends there.
        const std::size_t n = c.lengths[0];
        const endpoint block{.where = slot::block, .stride = 1};
        endpoint from = at(src, e);
        const endpoint to = at(dst, e);
        bool uses_block = false;
        // Whether the line's transform is one leaf (the leaf_rows reach below); a line with
        // callbacks leaves it false, so that it keeps the walked form (core/callbacks.cpp).
        bool one_leaf = false;
        if (any_callback(*edge_ops)) {
          // Block callbacks (task 5.3a) in DESIGN.md 12.3's order: read or produce, modify,
          // then the vocabulary; the vocabulary, modify, then store, accumulate or consume.
          from = row_input_with_callback(s.prec, edge_ops->in, n, from, c.in_place,
                                         settings.chunk_points, c.callback_sites, st);
          const bool direct = edge_ops->out.call.mode == callback_mode::modify &&
                              !edge_ops->out.accumulate && !edge_ops->out.runs.any();
          const endpoint line_to = edge_ops->out.call.any()
                                       ? (direct && !edge_ops->out.separate ? to : block)
                                       : (edge_ops->out.separate ? block : to);
          // In place, the input and output slots are one array.
          const bool one_array =
              from.where == line_to.where ||
              (c.in_place && from.where != slot::block && line_to.where != slot::block);
          need = chain.c2c_line(n, sign, from, line_to, one_array, line, st);
          if (edge_ops->out.call.any()) {
            row_output_with_callback(s.prec, edge_ops->out, n, to, direct, settings.chunk_points,
                                     c.callback_sites, st);
          } else if (edge_ops->out.runs.any()) {
            run_line_stages(s.prec, edge_ops->out, true, n, edge_ops->out.runs.of(0),
                            {.src = slot::block, .dst = to.where, .dst_stride = to.stride},
                            settings.chunk_points, false, st);
          } else if (edge_ops->out.separate) {
            st.push_back(edge_pass_stage(s.prec, edge_ops->out, true, n, slot::block, 1, to.where,
                                         to.stride, settings.chunk_points, false,
                                         edge_ops->out.axis_stride));
          }
          uses_block = std::ranges::any_of(
              st, [](const stage& x) { return x.src == slot::block || x.dst == slot::block; });
          c.callbacks_note = callbacks_text(
              edge_ops->in.call, edge_ops->out.call, row_input_where(edge_ops->in, c.in_place),
              line_to.where == slot::block ? "in the workspace block" : "in the output",
              callback_block_elements(n, settings.chunk_points));
        } else {
          // Pruning by the edge passes (task 5.1e): where a side's runs lie in the line's first
          // half and the chain's first (input) or last (output) pass has a radix with pruned
          // codelets, that pass reads or writes the first half only. One run of exactly that
          // half, stored with stride 1, is read or written where it is stored, with no separate
          // pass (run_pruning::direct, unless the output accumulates); otherwise the separate
          // pass moves the runs through the half in the workspace block (run_pruning::half).
          const auto within_half = [n](const std::vector<line_run>& runs) {
            return std::ranges::all_of(
                runs, [n](const line_run& run) { return run.logical + run.count <= n / 2; });
          };
          const auto direct = [n](const std::vector<line_run>& runs, const endpoint& stored) {
            return runs.size() == 1 && runs[0].logical == 0 && 2 * runs[0].count == n &&
                   stored.stride == 1;
          };
          const std::size_t cb = 2 * (s.prec == precision::f64 ? sizeof(double) : sizeof(float));
          // Stored runs (task 5.1d): the input's pass reads the runs with zeros elsewhere, the
          // output's stores only the runs.
          if (edge_ops->in.runs.any()) {
            const std::vector<line_run>& runs = edge_ops->in.runs.of(0);
            line.in_head = line.in == edge_op::none && within_half(runs) && chain.head_first(n);
            if (line.in_head && direct(runs, from)) {
              line.in_offset = runs[0].stored * cb;
              in_pruning = run_pruning::direct;
            } else {
              run_line_stages(s.prec, edge_ops->in, false, line.in_head ? n / 2 : n, runs,
                              {.src = from.where, .src_stride = from.stride}, settings.chunk_points,
                              false, st);
              from = block;
              in_pruning = line.in_head ? run_pruning::half : run_pruning::none;
            }
          } else if (edge_ops->in.separate) {
            st.push_back(edge_pass_stage(s.prec, edge_ops->in, false, n, from.where, from.stride,
                                         slot::block, 1, settings.chunk_points, false,
                                         edge_ops->in.axis_stride));
            from = block;
          }
          endpoint line_to = edge_ops->out.separate ? block : to;
          bool out_pass = edge_ops->out.separate;
          if (edge_ops->out.runs.any()) {
            const std::vector<line_run>& runs = edge_ops->out.runs.of(0);
            line.out_head = line.out != edge_op::factor && within_half(runs) && chain.head_last(n);
            if (line.out_head && !edge_ops->out.accumulate && direct(runs, to)) {
              line.out_offset = runs[0].stored * cb;
              line_to = to;
              out_pass = false;
              out_pruning = run_pruning::direct;
            } else {
              out_pruning = line.out_head ? run_pruning::half : run_pruning::none;
            }
          }
          const std::size_t line_first = st.size();
          need = chain.c2c_line(n, sign, from, line_to, from.where == line_to.where, line, st);
          one_leaf = st.size() == line_first + 1 && st[line_first].leaf != nullptr;
          if (edge_ops->out.runs.any() && out_pass) {
            run_line_stages(s.prec, edge_ops->out, true, n, edge_ops->out.runs.of(0),
                            {.src = slot::block, .dst = to.where, .dst_stride = to.stride},
                            settings.chunk_points, false, st);
          } else if (out_pass) {
            st.push_back(edge_pass_stage(s.prec, edge_ops->out, true, n, slot::block, 1, to.where,
                                         to.stride, settings.chunk_points, false,
                                         edge_ops->out.axis_stride));
          }
          uses_block = from.where == slot::block || line_to.where == slot::block;
        }
        // The row-block form runs flat programs only (make_edge_forms): a line of more stages
        // (a Bluestein length) runs a transform at a time, unrolled.
        if (st.size() > flat_stages_max ||
            std::ranges::any_of(st, [](const stage& x) { return x.moving_twiddles; })) {
          c.edge_row_block = 0;
        }
        // Where the reach is leaf_rows, a line whose transform is more than one leaf keeps the
        // walked form (X65's P1, narrowed: chains of 256-1024 points moved 0.90-1.01).
        if (forms_reach == edge_forms_reach::leaf_rows) {
          c.edge_batched_on = one_leaf;
          if (!one_leaf) {
            c.edge_row_block = 0;
          }
        }
        // Only where the line starts or ends in the block: a side whose runs the chain's pruned
        // first or last pass reads or writes where they are stored needs none (task 5.1e).
        if (uses_block) {
          c.block_elements =
              std::max(c.block_elements, n * std::max<std::size_t>(c.edge_row_block, 1));
        }
      } else {
        if (complex_columns(e, extent, src, dst, line, sign, "", d)) {
          return;
        }
        need = chain.c2c_line(c.lengths[d], sign, at(src, e), at(dst, e), aliased, line, st);
        // A row's chain with fused factors, where the reach is leaf_rows: the batched forms for
        // a line of one leaf (as above).
        if (edge_ops != nullptr && r == 1 && forms_reach == edge_forms_reach::leaf_rows) {
          c.edge_batched_on = st.size() == 1 && st[0].leaf != nullptr;
        }
      }
      lines(e, extent, src, dst, need, std::move(st));
      c.axis_notes.push_back("axis " + std::to_string(listed(d)) +
                             " (n = " + std::to_string(c.lengths[d]) + "): radices " +
                             chain.radix_text(c.lengths[d]));
    };
    // A team program takes plane order only for volumes of at least team_min_planes planes:
    // its plane phase has one line per plane, so fewer planes than members leave members
    // idle (X43b measured 8 lanes on volumes of 16 planes and more; 2 x 1024 x 1024 and
    // 4 x 512 x 512 keep the axes one by one, whose phases share out by lines).
    constexpr std::size_t team_min_planes = 8;
    if (r == 3 && in.extent[0] > 1 &&
        (settings.volume_planes || (settings.team_program && settings.profile.team_volume_planes &&
                                    in.extent[0] >= team_min_planes))) {
      add_volume_planes(
          p, in.extent, src0, dst0,
          [&](const std::vector<std::size_t>& plane, const view& src, const view& dst) {
            c2c_axis(2, 1, plane, src, dst);
            c2c_axis(1, 0, plane, dst, dst);
          });
      c2c_axis(0, 0, in.extent, dst0, dst0);
      c.strategy_note = volume_note;
    } else {
      for (std::size_t i = r; i > 0; --i) {
        const std::size_t d = i - 1;
        const view& src = d + 1 == r ? src0 : dst0;
        std::vector<bool> done(r, false);
        for (std::size_t e = d + 1; e < r; ++e) {
          done[e] = true;
        }
        pruned(d, in.extent, src, dst0, done,
               [&](const std::vector<std::size_t>& extent) { c2c_axis(d, d, extent, src, dst0); });
      }
    }
  } else if (s.kind == transform_kind::r2c) {
    // src0 and dst0: the caller's arrays, or with edge phases the workspace block (where the
    // rows then run in place when both are, as an in-place plan's do).
    std::vector<stage> st;
    const std::size_t need =
        chain.r2c_line(c.lengths[r - 1], at(src0, r - 1), at(dst0, r - 1),
                       src0.where == slot::in ? c.in_place : src0.where == dst0.where,
                       r == 1 ? opts.scale : 1.0, st);
    lines(r - 1, in.extent, src0, dst0, need, std::move(st));
    c.axis_notes.push_back("axis " + std::to_string(listed(r - 1)) +
                           " (n = " + std::to_string(c.lengths[r - 1]) + "): " +
                           chain.real_text(c.lengths[r - 1], true,
                                           src0.stride[r - 1] != 1 || dst0.stride[r - 1] != 1));
    for (std::size_t i = r - 1; i > 0; --i) {
      const std::size_t d = i - 1;
      const std::size_t n = c.lengths[d];
      const line_edges edges = c2c_edges(opts, settings, false, d == 0);
      std::vector<bool> done(r, false);
      for (std::size_t e = d + 1; e < r; ++e) {
        done[e] = true;
      }
      pruned(d, out.extent, dst0, dst0, done, [&](const std::vector<std::size_t>& extent) {
        if (complex_columns(d, extent, dst0, dst0, edges, -1, "c2c forward")) {
          return;
        }
        std::vector<stage> cs;
        const std::size_t cneed = chain.c2c_line(n, -1, at(dst0, d), at(dst0, d), true, edges, cs);
        lines(d, extent, dst0, dst0, cneed, std::move(cs));
        c.axis_notes.push_back("axis " + std::to_string(listed(d)) + " (n = " + std::to_string(n) +
                               "): c2c forward, radices " + chain.radix_text(n));
      });
    }
  } else if (!edge_phases && packed_plane) {
    // A c2r plane out of place, its intermediate packed in the output (DESIGN.md 9.1,
    // experiments X60, X63): each output row's n reals hold its h = n/2 complex values after
    // the column axis, X_0 .. X_h-1 with the imaginary place of X_0 holding
    // Re X_h (pass_args::packed), which is all the row's c2r reads. The complex axes run on
    // columns 0 .. h-1 from the input into the output; column h, whose real part alone is
    // needed, runs through the complex axes from the input into the block (one value per
    // row), as one more line of the last column phase where that phase has no other tail;
    // each row then takes its Re X_h from the block into the imaginary place of its column 0,
    // which that column's own transform wrote, and runs its c2r in place.
    const std::size_t n = c.lengths[r - 1];
    const std::size_t h = n / 2;
    view packed{.where = slot::out, .stride = out.stride, .element_bytes = complex_bytes};
    for (std::size_t d = 0; d + 1 < r; ++d) {
      packed.stride[d] = out.stride[d] / 2;
    }
    packed.stride[r - 1] = 1;
    std::vector<std::size_t> columns = in.extent;
    columns[r - 1] = h;
    std::size_t last_phase = p.records.size();
    for (std::size_t i = r - 1; i > 0; --i) {
      const std::size_t d = i - 1;
      const view& src = d + 2 == r ? in_view : packed;
      last_phase = p.records.size();
      // The output's rows are those of a c2c plane (n complex values, a power of two for
      // most n): c2c columns' rule (column_block, column_radices), not that of the odd rows
      // of n/2 + 1 columns. With the real rule the M5 Max ran these columns alone through
      // the chain, their elements a power of two apart, and lost 1.6-2.6 from 128^2 up.
      if (complex_columns(d, columns, src, packed, line_edges(), +1,
                          "c2c backward into the output, packed", SIZE_MAX, 0)) {
        continue;
      }
      std::vector<stage> cs;
      const std::size_t cneed = chain.c2c_line(c.lengths[d], +1, at(src, d), at(packed, d),
                                               src.where == slot::out, line_edges(), cs);
      lines(d, columns, src, packed, cneed, std::move(cs));
      c.axis_notes.push_back(
          "axis " + std::to_string(listed(d)) + " (n = " + std::to_string(c.lengths[d]) +
          "): c2c backward into the output, packed, radices " + chain.radix_text(c.lengths[d]));
    }
    // Column h: one line of a loop that starts there on the input. It joins the last column
    // phase where that phase is one nest of lines (no narrower block of its own): its line
    // reads only the input and writes only the block.
    std::size_t nests = 0;
    for (std::size_t i = last_phase; i < p.records.size(); ++nests) {
      const loop* l = std::get_if<loop>(&p.records[i]);
      if (l == nullptr) {
        nests = 2;
        break;
      }
      i = l->body_end;
    }
    view block_view{
        .where = slot::block,
        .stride = std::vector<std::ptrdiff_t>(r, 0),
        .element_bytes = complex_bytes,
    };
    std::vector<std::size_t> edge = in.extent;
    edge[r - 1] = 1;
    std::ptrdiff_t span = 1;
    for (std::size_t i = r - 1; i > 0; --i) {
      block_view.stride[i - 1] = span;
      span = checked_mul(span, to_signed(in.extent[i - 1], "shape"), "shape");
    }
    c.block_elements = static_cast<std::size_t>(span);
    const std::size_t wrapper = p.records.size();
    p.records.emplace_back(loop{
        .kind = loop_kind::lines,
        .count = 1,
        .start = {to_signed(h, "shape") * in.stride[r - 1] * in.element_bytes, 0, 0, 0},
        .joins = nests == 1,
    });
    for (std::size_t i = r - 1; i > 0; --i) {
      const std::size_t d = i - 1;
      const view& src = d + 2 == r ? in_view : block_view;
      std::vector<stage> cs;
      const std::size_t cneed = chain.c2c_line(c.lengths[d], +1, at(src, d), at(block_view, d),
                                               src.where == slot::block, line_edges(), cs);
      scratch = std::max(scratch, cneed);
      add_phase(p, d, edge, src, block_view, std::move(cs));
    }
    std::get<loop>(p.records[wrapper]).body_end = static_cast<std::uint32_t>(p.records.size());
    c.axis_notes.push_back("column " + std::to_string(h) + " through the complex axes into the " +
                           "block" + (nests == 1 ? ", one more line of the column phase" : "") +
                           "; each row takes its real part in place of its column 0's imaginary "
                           "part");
    std::vector<stage> st;
    {
      stage ny;
      ny.run = kernels.passes.c2r_nyquist;
      ny.src = slot::block;
      ny.dst = slot::out;
      ny.units = 1;
      ny.name = "Re X_" + std::to_string(h) + " from the block into the packed row";
      st.push_back(std::move(ny));
    }
    const std::size_t need =
        chain.c2r_line(n, {slot::out, 1}, {slot::out, 1}, opts.scale, st, true);
    const std::size_t rows_first = p.records.size();
    lines(r - 1, out.extent, out_view, out_view, need, std::move(st));
    // The rows' loops move the block too, a value per row (lines() moves only the out slot).
    std::size_t at_loop = rows_first;
    for (std::size_t e = 0; e + 1 < r; ++e) {
      if (out.extent[e] == 1) {
        continue;
      }
      std::get<loop>(p.records[at_loop++]).step[2] = block_view.stride[e] * complex_bytes;
    }
    c.axis_notes.push_back("axis " + std::to_string(listed(r - 1)) + " (n = " + std::to_string(n) +
                           "): in place on the packed rows, " + chain.real_text(n, false, false));
  } else {
    // c2r: the complex axes first, then the real one, which must come last because the
    // real output cannot hold a complex intermediate. Out of place the input is preserved,
    // so the intermediate goes to the block: one transform's half spectrum (DESIGN.md 9.1).
    view block_view{
        .where = slot::block,
        .stride = std::vector<std::ptrdiff_t>(r, 0),
        .element_bytes = complex_bytes,
    };
    if (r > 1 && !c.in_place) {
      // The block is tight and complex. The checks of the two sides do not bound its
      // bytes (the input may overlap itself; the real output has half the bytes per
      // element), so its size is checked here.
      std::ptrdiff_t span = 1;
      for (std::size_t i = r; i > 0; --i) {
        block_view.stride[i - 1] = span;
        span = checked_mul(span, to_signed(in.extent[i - 1], "shape"), "shape");
      }
      static_cast<void>(checked_mul(span, complex_bytes, "shape"));
      c.block_elements = std::max(c.block_elements, static_cast<std::size_t>(span));
    }
    // An input edge phase has written the half spectrum into the block, in its layout
    // (block_complex is block_view's): the transform runs there, out of place or in place.
    const bool in_block = edge_phases && edge_ops->in.separate;
    if (in_block) {
      block_view = block_complex;
    }
    // In place the complex phases work on the output slot, which is the input's memory,
    // through the input's (complex) strides.
    const view work =
        c.in_place && !in_block
            ? view{.where = slot::out, .stride = in.stride, .element_bytes = complex_bytes}
            : block_view;
    for (std::size_t i = r - 1; i > 0; --i) {
      const std::size_t d = i - 1;
      const view& src = d + 2 == r ? (in_block ? block_view : (c.in_place ? work : in_view)) : work;
      std::vector<bool> done(r, false);
      for (std::size_t e = d + 1; e + 1 < r; ++e) {
        done[e] = true;
      }
      pruned(d, in.extent, src, work, done, [&](const std::vector<std::size_t>& extent) {
        if (complex_columns(d, extent, src, work, line_edges(), +1, "c2c backward")) {
          return;
        }
        std::vector<stage> cs;
        const std::size_t n = c.lengths[d];
        const std::size_t cneed = chain.c2c_line(n, +1, at(src, d), at(work, d),
                                                 src.where == work.where, line_edges(), cs);
        lines(d, extent, src, work, cneed, std::move(cs));
        c.axis_notes.push_back("axis " + std::to_string(listed(d)) + " (n = " + std::to_string(n) +
                               "): c2c backward, radices " + chain.radix_text(n));
      });
    }
    // The rows from the half spectrum to dst0: the output, or with accumulation the block, in
    // place there when the half spectrum is in it (c2r never writes its source before reading
    // it, as an in-place plan's rows show).
    const view& src = in_block || (r > 1 && !c.in_place) ? block_view : in_view;
    std::vector<bool> done(r, true);
    done[r - 1] = false;
    pruned(r - 1, out.extent, src, dst0, done, [&](const std::vector<std::size_t>& extent) {
      std::vector<stage> st;
      const std::size_t need =
          chain.c2r_line(c.lengths[r - 1], at(src, r - 1), at(dst0, r - 1), opts.scale, st);
      lines(r - 1, extent, src, dst0, need, std::move(st));
      c.axis_notes.push_back("axis " + std::to_string(listed(r - 1)) +
                             " (n = " + std::to_string(c.lengths[r - 1]) + "): " +
                             chain.real_text(c.lengths[r - 1], false,
                                             src.stride[r - 1] != 1 || dst0.stride[r - 1] != 1));
    });
  }
  if (use_edge_phases && edge_ops->out.separate) {
    // The output edge's phase: from dst0 (the output itself, in place, or the block) to the
    // output.
    edge_phase(p, edge_loops, true, out.extent, dst0, out_view, edge_rows{});
  }
  if (row_strategy && lanes == 0) {
    // The strategy read the input slot; after the input edge's pass the input is in the output
    // or the block (input_slot), with the input side's layout there: stride 1 (a grid, a
    // depth-first chain and the real rows' strategies read contiguous rows), so the loops'
    // bytes are the same.
    if (input_slot != slot::in) {
      retarget_input(p.records, strategy_first, input_slot);
    }
    // The input's region of the block (a real row out of place) is the strategy's whole block:
    // a strategy that uses the block takes no input there (the conflicts above).
    const std::size_t region_in = input_slot == slot::block ? block_units(n_in, real_in) : 0;
    const std::size_t own = std::max(c.block_elements, region_in);
    const edge_side& e = edge_ops->out;
    const std::ptrdiff_t fs = e.kind != factor_kind::none ? e.axis_strides.at(0) : 0;
    if (accumulate) {
      const std::size_t offset = own * static_cast<std::size_t>(complex_bytes);
      std::get<stage>(p.records.at(save_record)).args.dst_cols = offset;
      p.records.emplace_back(edge_pass_stage(s.prec, e, true, n_out, slot::block, 1, slot::out,
                                             out.stride[0], settings.chunk_points, real_out, fs,
                                             edge_mode::add_prior, {.src_offset = offset}));
      c.block_elements = own + block_units(n_out, real_out);
    } else {
      if (e.separate) {
        p.records.emplace_back(edge_pass_stage(s.prec, e, true, n_out, slot::out, out.stride[0],
                                               slot::out, out.stride[0], settings.chunk_points,
                                               real_out, fs, edge_mode::store, edge_rows{}));
      }
      c.block_elements = own;
    }
  }
  if (edge_ops != nullptr) {
    // The lanes over adjacent transforms take edge phases whatever the rank.
    const edge_layout layout =
        s.kind == transform_kind::c2c && r == 1 && !row_strategy && adjacent == 0
            ? edge_layout::row_chain
            : (row_strategy ? edge_layout::row_strategy : edge_layout::phases);
    if (edge_ops->in.runs.any() || edge_ops->out.runs.any()) {
      // Stored runs (task 5.1d), with at most accumulation and scales beside them: one pass per
      // side that has runs or accumulates, except a side whose runs the chain's pruned first or
      // last pass reads or writes where they are stored (task 5.1e).
      const bool to_out = use_edge_phases && src0.where == slot::out;
      const bool in_pass = edge_ops->in.runs.any() && in_pruning != run_pruning::direct;
      const bool out_pass =
          edge_ops->out.runs.any() ? out_pruning != run_pruning::direct : edge_ops->out.accumulate;
      const std::size_t passes = (in_pass ? 1U : 0U) + (out_pass ? 1U : 0U);
      c.edges_note =
          runs_text({.in = edge_ops->in.runs, .out = edge_ops->out.runs}, edge_ops->out.accumulate,
                    to_out ? "the output" : "the workspace block", passes, in_pruning, out_pruning);
    } else {
      c.edges_note = edges_text(*edge_ops, layout, edge_passes(p));
      if (c.edge_row_block != 0) {
        c.edges_note += "\nedge row blocks: the rows in blocks of " +
                        std::to_string(c.edge_row_block) +
                        ", each separate pass, and a leaf, once per block";
      }
    }
    // Batch leaves take a batch of rows in one call of the leaf (driver.hpp, run_leaf_rows),
    // which separate edge passes between the rows break up: each row's leaf runs alone, except
    // in the row-block form, which c2c_axis may have dropped (so decided here, after it). Not
    // over a reason already given: where the input edge's pass has moved the rows (above), the
    // plan without the edges may have no batch leaves at all, and the move is the cause;
    // name_forced_strategy gives this reason where that plan has them.
    if (row_edges && !row_strategy && batch != nullptr && s.kind == transform_kind::c2c &&
        c.edge_row_block == 0 && edge_fallback.empty()) {
      edge_fallback = batch_one_row_reason;
    }
    c.edge_fallback = edge_fallback;
    if (any_callback(*edge_ops)) {
      // Task 5.3a: callbacks on the chain only (core/edges.cpp refuses them elsewhere); the
      // other strategies take them in 5.3b.
      c.edge_fallback = "callbacks take the chain until task 5.3b";
      c.edges_note += "\n" + c.callbacks_note;
    }
  }
  // The strategy in a few words, in the order of the branches above, for build_with_edges'
  // comparison with the plan without edges.
  c.strategy_name = [&]() -> std::string {
    if (lanes != 0) {
      return "the lanes over rows";
    }
    if (adjacent != 0) {
      return "the lanes over adjacent transforms";
    }
    if (real_grid.phases != 0) {
      return "the half-length grid in " + std::to_string(real_grid.phases) + " phases";
    }
    if (!depth_tops.empty()) {
      return "the depth-first chain";
    }
    if (!real_depth_tops.empty()) {
      return "the half-length depth-first chain";
    }
    if (!bluestein_tops.empty()) {
      return "Bluestein's depth-first convolution";
    }
    if (grid_plan.phases != 0) {
      return "the grid in " + std::to_string(grid_plan.phases) + " phases";
    }
    if (r > 1) {
      if (c.strategy_note == volume_note) {
        return "each plane of the volume, then the outer axis";
      }
      return !edge_phases && packed_plane
                 ? "each axis in turn, the intermediate packed in the output"
                 : "each axis in turn";
    }
    if (batch != nullptr && s.kind == transform_kind::c2c) {
      return std::string(row_edges && c.edge_row_block == 0 ? batch_one_row_name
                                                            : batch_leaves_name);
    }
    return "the chain";
  }();
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    std::get<loop>(p.records[b]).body_end = static_cast<std::uint32_t>(p.records.size());
  }

  const std::size_t callback_scratch =
      edge_ops != nullptr
          ? std::max(edge_ops->in.call.any() ? edge_ops->in.call.scratch_bytes : 0,
                     edge_ops->out.call.any() ? edge_ops->out.call.scratch_bytes : 0)
          : 0;
  lay_out_workspace(c, scratch, 2 * sizeof(T), grid_plan.phases != 0, callback_scratch);
}

// The shape's counts and direction, which everything else depends on (std::invalid_argument
// naming the field).
void check_counts(const shape_view& s) {
  const std::size_t r = s.axes.size();
  if (r == 0) {
    reject("shape.axes", "must hold at least one axis");
  }
  if (r > max_axes) {
    reject("shape.axes", "holds " + std::to_string(r) + " axes; at most " +
                             std::to_string(max_axes) + " are transformed");
  }
  for (std::size_t d = 0; d < r; ++d) {
    if (s.axes[d].n == 0) {
      reject(axis_name(d) + ".n", "is 0; every axis needs its length (`.axes{2}` makes two "
                                  "axes of length 0; write `.axes = {{2}}`)");
    }
  }
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    if (s.batches[b].count == 0) {
      reject(batch_name(b) + ".count",
             "is 0; every batch dimension needs its count (`.batches{8}` makes eight empty "
             "batch dimensions; write `.batches = {{8}}`)");
    }
  }
  if (s.kind == transform_kind::c2c && s.dir != direction::forward &&
      s.dir != direction::backward) {
    reject("shape.dir", "must be direction::forward or direction::backward");
  }
}

// The tags build_plan_core puts in build_settings::in_factor and out_factor for the edges'
// fused factors: their addresses only, which nothing reads (one array, so that the two are
// distinct however the toolchain merges constants).
const std::array<unsigned char, 2> factor_tags{};
const void* const in_factor_tag = factor_tags.data();
const void* const out_factor_tag = factor_tags.data() + 1;

// The edge driver's records of a plan with edges (core/edge_driver.hpp): the stages that read a
// factor through its cursor, those built with a tag (build_plan_core) and the separate edge
// passes (which edge_pass_stage marks), keep their kernels there and carry guards; the batch
// loops move the cursors by the factors' batch strides (build makes one batch loop per batch
// dimension, in the shape's order, so loop b is dimension b), and the lines loops of the edge
// phases by the factors' axis strides (`loops`, from build). Other lines loops enclose no stage
// that reads a cursor, so they need not move it.
// `records`: a program's records or its lanes tail (program::lane_tail, no batch loops).
std::vector<edge_record> mark_cursors(std::vector<record>& records, std::size_t batch_loops,
                                      const edge_plan& edges, const std::vector<edge_loop>& loops) {
  std::vector<edge_record> e(records.size());
  for (std::size_t i = 0; i < records.size(); ++i) {
    stage* st = std::get_if<stage>(&records[i]);
    if (st == nullptr) {
      continue;
    }
    if (st->args.in_factor == in_factor_tag) {
      st->args.in_factor = nullptr;
      st->moving_factors |= moving_in_factor;
    }
    if (st->args.out_factor == out_factor_tag) {
      st->args.out_factor = nullptr;
      st->moving_factors |= moving_out_factor;
    }
    if ((st->moving_factors & moving_callback) != 0) {
      // A callback stage (core/callbacks.cpp): its kernel is already a guard, its site in roots.
      e[i].call = static_cast<const callback_site*>(st->args.roots);
      continue;
    }
    if (st->moving_factors != 0) {
      e[i].run = st->run;
      e[i].leaf = st->leaf;
      st->run = st->run != nullptr ? &factor_guard_pass : nullptr;
      st->leaf = st->leaf != nullptr ? &factor_guard_leaf : nullptr;
    }
  }
  for (std::size_t b = 0; b < batch_loops; ++b) {
    if (edges.in.kind != factor_kind::none) {
      e[b].factor_step[0] = edges.in.batch_steps.at(b);
    }
    if (edges.out.kind != factor_kind::none) {
      e[b].factor_step[1] = edges.out.batch_steps.at(b);
    }
  }
  for (const edge_loop& l : loops) {
    if (!std::holds_alternative<loop>(records.at(l.record))) {
      throw std::logic_error("fftn: internal: an edge phase's loop is not a loop");
    }
    e[l.record].factor_step = l.step;
  }
  return e;
}

const char* kind_name(transform_kind k) {
  switch (k) {
  case transform_kind::c2c:
    break;
  case transform_kind::r2c:
    return "r2c";
  case transform_kind::c2r:
    return "c2r";
  }
  return "c2c";
}

const char* slot_name(slot s) {
  switch (s) {
  case slot::in:
    return "in";
  case slot::out:
    return "out";
  case slot::a:
    return "a";
  case slot::b:
    return "b";
  case slot::block:
    break;
  }
  return "block";
}

// build_plan_core, with its team core where `team_core`: name_forced_strategy's plan without
// the edges goes without, its strategy being all it is built for.
std::shared_ptr<plan_core> build_core(const shape_view& s, const options& opts,
                                      const build_settings& given, const edge_plan* edges,
                                      bool team_core) {
  // A plan whose edges have a factor fused into a c2c row's passes (core/edges.hpp,
  // resolve_edges) is a chain whatever the rules or a test would choose: the fused factor goes
  // through the settings' factor fields, which every other strategy's rule refuses, as a tag
  // that the stages built from it are found by afterwards (mark_cursors). Edges whose factors
  // are all separate passes take the strategy the rules (or a test) choose.
  // Callbacks (task 5.3a) likewise take the chain, without batch leaves, until 5.3b.
  build_settings tagged;
  const bool with_edges = edges != nullptr && edges->any;
  const bool fused = with_edges && any_fused(*edges);
  const bool calls = with_edges && any_callback(*edges);
  if (fused || calls) {
    tagged = given;
    tagged.strategy = strategy_choice::chain;
    tagged.batch_leaves = batch_leaf_choice::never;
  }
  if (fused) {
    tagged.in_factor = edges->in.fused ? in_factor_tag : nullptr;
    tagged.out_factor = edges->out.fused ? out_factor_tag : nullptr;
    tagged.scale_on_input = edges->scale_on_input;
  }
  const build_settings& settings = fused || calls ? tagged : given;
  check_counts(s);
  const std::size_t r = s.axes.size();

  auto c = std::make_shared<plan_core>();
  c->kind = s.kind;
  c->prec = s.prec;
  c->dir = s.dir;
  if (s.kind != transform_kind::c2c) {
    c->dir = s.kind == transform_kind::r2c ? direction::forward : direction::backward;
  }
  c->in_place = opts.in_place;
  for (const axis& a : s.axes) {
    c->lengths.push_back(a.n);
  }
  for (const batch& b : s.batches) {
    c->counts.push_back(b.count);
  }

  // Stored runs (task 5.1d) come as the edges' (build_with_runs): each side's, in the shape's
  // order, which the layout's defaults and checks take.
  const side_runs no_runs;
  const side_runs& in_runs = with_edges ? edges->in.runs : no_runs;
  const side_runs& out_runs = with_edges ? edges->out.runs : no_runs;
  if (!s.runs.empty() && !in_runs.any() && !out_runs.any()) {
    throw std::logic_error("fftn: internal: a shape with stored runs built without them");
  }

  const std::size_t real_bytes = s.prec == precision::f64 ? sizeof(double) : sizeof(float);
  const std::size_t half = s.axes[r - 1].n / 2 + 1;
  std::vector<std::size_t> full = c->lengths;
  std::vector<std::size_t> halved = c->lengths;
  halved[r - 1] = half;
  // The padded real row of an in-place real transform: the 2(n/2 + 1) reals of its complex
  // row.
  const std::size_t padded = opts.in_place ? 2 * half : s.axes[r - 1].n;
  const auto rb = static_cast<std::ptrdiff_t>(real_bytes);
  const auto cb = static_cast<std::ptrdiff_t>(2 * real_bytes);
  side in;
  side out;
  switch (s.kind) {
  case transform_kind::c2c:
    in = resolve(s, full, full[r - 1], true, true, cb, in_runs);
    out = resolve(s, full, full[r - 1], false, true, cb, out_runs);
    break;
  case transform_kind::r2c:
    in = resolve(s, full, padded, true, false, rb, in_runs);
    out = resolve(s, halved, half, false, true, cb, out_runs);
    break;
  case transform_kind::c2r:
    in = resolve(s, halved, half, true, true, cb, in_runs);
    out = resolve(s, full, padded, false, false, rb, out_runs);
    break;
  }
  check_reach(s, in, true);
  check_reach(s, out, false);
  check_no_overlap(s, out);
  if (opts.in_place) {
    check_in_place(s, in, out);
  }
  std::tie(c->in_lo, c->in_hi) = span_of(s, in);
  std::tie(c->out_lo, c->out_hi) = span_of(s, out);
  order_axes(s, in, out, *c);

  // The scale is applied in the plan's precision: it must be finite there, and a non-zero
  // scale must not vanish. (Converting a double beyond float's range to float is undefined,
  // so the range is compared, not converted.)
  if (!std::isfinite(opts.scale)) {
    reject("options.scale", "must be finite");
  }
  if (s.prec == precision::f32) {
    const double magnitude_of_scale = std::abs(opts.scale);
    if (magnitude_of_scale > static_cast<double>(std::numeric_limits<float>::max())) {
      reject("options.scale", "is beyond the range of float, the plan's precision");
    }
    if (opts.scale != 0.0 &&
        magnitude_of_scale < static_cast<double>(std::numeric_limits<float>::denorm_min())) {
      reject("options.scale", "is below the smallest float, the plan's precision");
    }
  }
  const bool factors = settings.in_factor != nullptr || settings.out_factor != nullptr;
  if ((factors || settings.scale_on_input) && (s.kind != transform_kind::c2c || r != 1)) {
    reject("build_settings", "edge factors and an input scale are for c2c rows only");
  }
  if (settings.in_factor != nullptr && opts.scale != 1.0 &&
      (settings.scale_on_input || settings.out_factor != nullptr)) {
    reject("build_settings", "puts a scale and a factor on the input edge");
  }

  std::vector<edge_loop> edge_loops;
  std::vector<edge_loop> tail_loops;
  if (s.prec == precision::f64) {
    build<double>(*c, s, in, out, opts, settings, with_edges ? edges : nullptr, edge_loops,
                  tail_loops);
  } else {
    build<float>(*c, s, in, out, opts, settings, with_edges ? edges : nullptr, edge_loops,
                 tail_loops);
  }
  if (with_edges) {
    c->edge_records = mark_cursors(c->prog.records, c->prog.batch_loops, *edges, edge_loops);
    c->edge_tail_records = mark_cursors(c->prog.lane_tail, 0, *edges, tail_loops);
    c->factor_args =
        static_cast<std::uint8_t>((edges->in.kind != factor_kind::none ? moving_in_factor : 0) |
                                  (edges->out.kind != factor_kind::none ? moving_out_factor : 0));
    c->needs_args = c->factor_args != 0 || !c->callback_sites.empty();
    c->in_unread = edges->in.call.any() && edges->in.call.mode == callback_mode::produce;
    c->out_unwritten = edges->out.call.any() && edges->out.call.mode == callback_mode::consume;
    if (!c->callback_sites.empty() && c->prog.lanes != 0) {
      throw std::logic_error("fftn: internal: a callback stage in a lanes program");
    }
  }
  std::size_t points = 1;
  for (const std::size_t n : c->lengths) {
    points *= n; // the layouts' reach checks have bounded the product
  }
  // A real plane or volume is split by its complex points (the last axis n / 2 + 1 long), the
  // work its phases share, where the profile says (X52 arm S1: at 8 lanes real planes of
  // 96^2-100^2 took 1.1-1.3 times their one-thread time on the EPYC when split by their real
  // points). Rows are unchanged.
  if (r > 1 && s.kind != transform_kind::c2c && settings.profile.split_complex_points) {
    points = points / c->lengths.back() * (c->lengths.back() / 2 + 1);
  }
  // Only multi-axis programs are unrolled (X41): a 1-D grid without moving twiddles would
  // otherwise change its execution path, which nothing measured.
  mark_flat(c->prog, settings.unroll_calls && r > 1);
  // The edge driver's batched forms (task 5.2a): rows too are unrolled there, the walked form
  // being what X65 measured them against.
  if (c->needs_args && c->edge_batched_on) {
    const auto complex_bytes = static_cast<std::ptrdiff_t>(2 * real_bytes);
    c->edge_batched =
        make_edge_forms(c->prog, c->edge_records, settings.unroll_calls, c->edge_row_block,
                        static_cast<std::ptrdiff_t>(c->lengths[0]) * complex_bytes, complex_bytes);
    // describe() and the workspace layout name the row blocks that build() planned.
    if (c->edge_batched.row_block != c->edge_row_block) {
      throw std::logic_error("fftn: internal: the row blocks planned are not the form built");
    }
    // The other forms, which describe() names too, so that a run's notes show which plans take
    // them (the row blocks' line comes from build()).
    const edge_forms& f = c->edge_batched;
    if (f.row_block == 0 && (!f.flat.empty() || !f.calls.empty())) {
      c->edges_note += std::string("\nedge forms: ") +
                       (f.flat.empty() ? "the unrolled calls" : "the flat form") +
                       " with the factor cursors";
    }
  }
  c->graph = build_work_graph(c->prog, points, settings.chunk_points);
  // A real row on a chain's stages likewise, where the profile says (X83: on the Xeon 6240 teams
  // of 4 and 8 ran real rows of 6400-10240 points 1.32-2.69 times slower than one thread). Not a
  // row whose graph has phases of lines (the depth-first chain, the grid): their own threshold,
  // split_min_points_lines, was set on real points (X75's 32768 gains at 4 lanes).
  if (r == 1 && s.kind != transform_kind::c2c && settings.profile.split_complex_rows &&
      std::ranges::none_of(c->graph.phases, [](const graph_phase& ph) { return ph.nest != 0; })) {
    c->graph.points = c->lengths.back() / 2 + 1;
  }
  c->profile = settings.profile;
  c->runs_inline = runs_inline(c->graph, c->profile);
  c->radices_forced = !settings.radices.empty();
  c->shape_key = shape_key(s, opts);
  const plan_choice built{
      .strategy = settings.strategy,
      .radices = settings.radices,
      .grid_factors = settings.grid_factors,
      .grid_phases = settings.grid_phases,
      .rader = settings.rader,
      .depth_tops = settings.depth_tops,
  };
  c->program_text = choice_text(pinned_choice(built, *c));
  // The team program (X43 arms B1, B2), kept where it differs from the program: its column
  // blocks or (X43b arm VT) a volume's order.
  // X43b arm VT: a team core also where only a volume's order would differ.
  const bool team_order = r == 3 && settings.profile.team_volume_planes;
  if (team_core && s.kind == transform_kind::c2c &&
      ((r > 1 && (settings.profile.column_team_bytes != 0 || team_order ||
                  settings.profile.column_wide_bytes != 0)) ||
       c->adjacent_team_chain) &&
      !settings.team_program) {
    // A plan with edges passes them on: its team core has its own edge phases, tags and edge
    // records (mark_cursors), and is built from the settings the edges tagged.
    build_settings team = settings;
    team.team_program = true;
    std::shared_ptr<const plan_core> t =
        build_plan_core(s, opts, team, with_edges ? edges : nullptr);
    if (t->axis_notes != c->axis_notes || t->strategy_note != c->strategy_note) {
      c->team = std::move(t);
    }
  }
  return c;
}

} // namespace

std::shared_ptr<plan_core> build_plan_core(const shape_view& s, const options& opts,
                                           const build_settings& given, const edge_plan* edges) {
  return build_core(s, opts, given, edges, true);
}

row_layout resolve_row_layout(const shape_view& s, bool in_place, bool real) {
  check_counts(s);
  if (s.kind != transform_kind::c2c || s.axes.size() != 1 || !s.runs.empty()) {
    throw std::logic_error("fftn: internal: resolve_row_layout takes c2c rows without runs");
  }
  const std::size_t real_bytes = s.prec == precision::f64 ? sizeof(double) : sizeof(float);
  const auto cb = static_cast<std::ptrdiff_t>(real ? real_bytes : 2 * real_bytes);
  const std::vector<std::size_t> full{s.axes[0].n};
  const side_runs none;
  const side in = resolve(s, full, full[0], true, true, cb, none);
  const side out = resolve(s, full, full[0], false, true, cb, none);
  check_reach(s, in, true);
  check_reach(s, out, false);
  check_no_overlap(s, out);
  if (in_place) {
    check_in_place(s, in, out);
  }
  row_layout l{
      .in_stride = in.stride[0],
      .out_stride = out.stride[0],
      .in_dist = in.dist,
      .out_dist = out.dist,
  };
  std::tie(l.in_lo, l.in_hi) = span_of(s, in);
  std::tie(l.out_lo, l.out_hi) = span_of(s, out);
  return l;
}

namespace {

// A plan whose shape has stored runs (task 5.1d), with its edges (null: none): built as a plan
// with edges whose sides with runs are separate passes (core/runs.hpp), on the chain, by the
// rules whatever options::effort and options::use say; describe() names the strategy the plan
// without the runs would take where it differs, and says how it was planned.
std::shared_ptr<const plan_core> build_with_runs(const shape_view& shape, const options& opts,
                                                 const edges_view* given,
                                                 const build_settings& settings) {
  check_counts(shape);
  if (!std::isfinite(opts.measure_seconds) || opts.measure_seconds < 0) {
    reject("options.measure_seconds", "must be finite and not negative");
  }
  const run_plan runs = resolve_runs(shape, opts);
  const edges_view none{};
  const edges_view& edges = given != nullptr ? *given : none;
  for (const auto& [e, name] :
       {std::pair{&edges.in, "edges.in.mul.kind"}, std::pair{&edges.out, "edges.out.mul.kind"}}) {
    if (e->kind != factor_kind::none) {
      reject(name, "is a factor, which a shape with stored runs does not take yet (task 5.1d)");
    }
  }
  resolved_edges r = resolve_edges(shape, opts, edges, false);
  edge_plan& p = r.plan;
  p.any = true;
  p.in.runs = runs.in;
  p.out.runs = runs.out;
  p.in.separate = p.in.separate || runs.in.any();
  p.out.separate = p.out.separate || runs.out.any();
  // The chain: the strategies that read or write the caller's arrays themselves (the lanes,
  // the grid, depth-first chains, batch leaves) take no runs yet.
  build_settings chain = settings;
  chain.strategy = strategy_choice::chain;
  chain.batch_leaves = batch_leaf_choice::never;
  // A volume a plane at a time nests its planes' phases in a loop over the planes, which the
  // pruning by restricted loops does not cut (pruned, build_plan_core).
  chain.volume_planes = false;
  chain.profile.team_volume_planes = false;
  const std::shared_ptr<plan_core> c = build_plan_core(shape, r.opts, chain, &p);
  // The plan without the runs but with the same edges, where its layout is valid by logical
  // extents: its strategy, so that what the edges alone change (accumulation's, named by
  // build_with_edges) is not put down to the runs. Scales alone are options::scale's product, as
  // in any plan of scales only. Neither plan needs its team core here.
  shape_view whole = shape;
  whole.runs = {};
  options whole_opts = opts;
  whole_opts.scale = opts.scale * edges.in.scale * edges.out.scale;
  const bool edges_alone = resolve_edges(whole, opts, edges).plan.any;
  try {
    const std::shared_ptr<const plan_core> plain =
        edges_alone ? build_with_edges(whole, opts, edges, settings, false)
                    : build_core(whole, whole_opts, settings, nullptr, false);
    if (plain->strategy_name != c->strategy_name) {
      c->edges_note +=
          "\n" + forced_text(plain->strategy_name, c->strategy_name,
                             "stored runs take the chain (task 5.1d)", "the stored runs");
    }
  } catch (const std::invalid_argument&) {
    // Overlapping or unaddressable without the runs: no plan to compare with.
  }
  if (opts.effort != planning::rules || opts.use != nullptr) {
    c->planning_note = "by the rules: measured planning and wisdom take no stored runs yet";
  }
  // Complete although a plan with runs looks no wisdom up: scales alone (no edges_key) as
  // options::scale's product, otherwise the edges' key, which carries every scale.
  c->shape_key =
      edges_alone ? shape_key_with_edges(shape, opts, edges) : shape_key(shape, whole_opts);
  return c;
}

} // namespace

std::shared_ptr<const plan_core> make_plan_core(const shape_view& shape, const options& opts,
                                                const build_settings& settings) {
  if (!shape.runs.empty()) {
    return build_with_runs(shape, opts, nullptr, settings);
  }
  // Refused whatever the effort, so that an option that takes effect only with measured
  // planning is not valid or invalid depending on another option.
  if (!std::isfinite(opts.measure_seconds) || opts.measure_seconds < 0) {
    reject("options.measure_seconds", "must be finite and not negative");
  }
  if (opts.use == nullptr && opts.effort == planning::rules) {
    return build_plan_core(shape, opts, settings);
  }
  return plan_with_effort(shape, opts, settings);
}

std::shared_ptr<const plan_core> make_plan_core(const shape_view& shape, const options& opts) {
  return make_plan_core(shape, opts, build_settings{});
}

std::string shape_key_with_edges(const shape_view& shape, const options& opts,
                                 const edges_view& edges) {
  return shape_key(shape, opts) + edges_key(edges, opts.scale);
}

std::shared_ptr<plan_core> build_with_edges(const shape_view& shape, const options& opts,
                                            const edges_view& edges, const build_settings& settings,
                                            bool name_forced) {
  const resolved_edges r = resolve_edges(shape, opts, edges);
  if (!r.plan.any) {
    return build_plan_core(shape, r.opts, settings);
  }
  // The strategy the rules (or the settings) choose with every factor a separate pass; where
  // that is the chain and a c2c row's factor can be fused into its passes, the chain with the
  // fused factor (5.1a), which saves the separate pass. X65 (task 5.1f) is to weigh a fused
  // chain against another strategy with separate passes.
  const resolved_edges separate = resolve_edges(shape, opts, edges, false);
  std::shared_ptr<plan_core> c = build_plan_core(shape, separate.opts, settings, &separate.plan);
  if (any_fused(r.plan) && c->built == strategy_choice::chain) {
    const std::string fallback = c->edge_fallback;
    c = build_plan_core(shape, r.opts, settings, &r.plan);
    // What kept the separate passes' plan from the plain plan's strategy keeps this one too,
    // but the fused chain has no input pass: the strategy is the one chosen for the rows where
    // that pass would have left them (name_forced_strategy adds batch leaves' reason).
    c->edge_fallback =
        fallback == c2c_input_pass_reason ? std::string(c2c_fused_input_reason) : fallback;
  }
  if (name_forced) {
    name_forced_strategy(*c, shape, opts, edges, settings);
  }
  // Wisdom records the plan under its edges, so that no plan without them matches its entry,
  // with the caller's options (not the scale a placement fuses), so that the key does not
  // depend on which of the two builds above was kept.
  c->shape_key = shape_key_with_edges(shape, opts, edges);
  return c;
}

void name_forced_strategy(plan_core& c, const shape_view& shape, const options& opts,
                          const edges_view& edges, const build_settings& settings) {
  if (!resolve_edges(shape, opts, edges).plan.any) {
    return;
  }
  // The plan without the edges (their scales folded into options::scale, as a plan of scales
  // alone has them), without its team core: the strategy the rules (or the settings) choose,
  // which describe() names where the edges keep the plan from it (task 5.1c).
  options plain_opts = opts;
  plain_opts.scale = opts.scale * edges.in.scale * edges.out.scale;
  const std::shared_ptr<const plan_core> plain =
      build_core(shape, plain_opts, settings, nullptr, false);
  if (c.strategy_name == plain->strategy_name) {
    return;
  }
  std::string fallback = c.edge_fallback;
  // Where the plan without the edges is on batch leaves, the edges' separate passes take a
  // row at a time, and a fused factor (the chain the edges keep, the only other plan they
  // build there) takes none, whatever else build() noted. Callbacks give their own reason.
  if (plain->strategy_name == batch_leaves_name && c.callback_sites.empty()) {
    if (c.strategy_name == batch_one_row_name) {
      fallback = batch_one_row_reason;
    } else if (c.strategy_name == "the chain") {
      fallback = "the chain with a fused factor takes no batch leaves";
    }
  }
  c.edges_note += "\n" + forced_text(plain->strategy_name, c.strategy_name, fallback);
}

std::shared_ptr<const plan_core> make_plan_core_with_edges(const shape_view& shape,
                                                           const options& opts,
                                                           const edges_view& edges,
                                                           const build_settings& settings) {
  if (!shape.runs.empty()) {
    return build_with_runs(shape, opts, &edges, settings);
  }
  check_counts(shape);
  const resolved_edges r = resolve_edges(shape, opts, edges);
  if (!r.plan.any) {
    // Scales only: one product in options::scale, planned as any plan.
    return make_plan_core(shape, r.opts, settings);
  }
  if (!std::isfinite(opts.measure_seconds) || opts.measure_seconds < 0) {
    reject("options.measure_seconds", "must be finite and not negative");
  }
  if (opts.use == nullptr && opts.effort == planning::rules) {
    return build_with_edges(shape, opts, edges, settings);
  }
  if (any_callback(edges)) {
    // Task 5.3a: measured planning's stand-ins for callbacks come with 5.3b.
    const std::shared_ptr<plan_core> c = build_with_edges(shape, opts, edges, settings);
    c->planning_note = "by the rules: measured planning and wisdom take no callbacks yet (task "
                       "5.3b)";
    return c;
  }
  // Measured planning and wisdom with the edges: every candidate is built and timed with them
  // (core/measure.cpp).
  return plan_with_effort(shape, opts, settings, &edges);
}

std::shared_ptr<const plan_core>
make_plan_core_with_edges(const shape_view& shape, const options& opts, const edges_view& edges) {
  return make_plan_core_with_edges(shape, opts, edges, build_settings{});
}

namespace {

// A serial execution leaves the workspace ready for the next team execution to set its
// state up afresh (team_state.hpp): the scratch below overlaps that state, and a team
// that skipped an execution must not carry its state into the next one
// (TeamExecute.ResetAfterASkippedExecution, which a plan of leaf rows failed until X28b).
[[gnu::always_inline]] inline void reset_team_state(const workspace& ws) noexcept {
  const sync_word team = workspace_access::team_word(ws);
  if (team.load(std::memory_order_relaxed) != 0) {
    team.store(0, std::memory_order_relaxed);
  }
}

// execute_core's path for every program but leaf rows (X37), on a driver (core/drive.hpp):
// out of line on AArch64, where a frame shared with the path of leaf rows cost single rows of
// 2-6 points 6-9 % on the M5 Max; inline elsewhere, since out of line it slowed single rows of
// 31-300 points by 3-13 % on the EPYC 7763 (GCC 16) while its gain on leaf rows did not
// survive without the leaves' byte distances (passes/access.inl).
// State: the driver's state (edge_drive: the factor cursors), none for plain_drive, so that
// its instantiation takes main's parameters.
template <class Drive, class... State>
#if FFTN_ARCH_AARCH64
[[gnu::noinline]]
#else
[[gnu::always_inline]] inline
#endif
void execute_program(const plan_core& core, std::size_t first, std::size_t count, const void* in,
                     void* out, const workspace& ws, const State&... state) noexcept {
  assert(first <= core.prog.transforms && count <= core.prog.transforms - first &&
         "fftn: execute_transforms beyond the batch");
  assert(in_place_pointers(core, in, out) && "fftn: in-place plan executed with in != out");
  assert(out_of_place_pointers(core, in, out) && "fftn: out-of-place plan executed with in == out");
  if (count == 0) {
    return;
  }
  std::byte* a = nullptr;
  std::byte* b = nullptr;
  std::byte* block = nullptr;
  if (core.bytes != 0) {
    // The workspace may have any alignment; its size includes the slack for this.
    std::byte* raw = ws.memory().data();
    const auto address = reinterpret_cast<std::uintptr_t>(raw);
    const std::size_t skip = (64 - address % 64) % 64;
    std::byte* base = raw + skip;
    a = base;
    b = base + core.offset_b;
    block = core.block_elements != 0 ? base + core.offset_block : nullptr;
  }
  if constexpr (Drive::flat_shortcut) {
    if (core.prog.flat && first == 0 && count == 1) {
      // One transform, the whole batch or its first (X28): no slot_addresses.
      run_flat_first(core.prog, in, out, a, b, block);
      return;
    }
  }
  const slot_addresses at{
      .in = static_cast<const std::byte*>(in),
      .out = static_cast<std::byte*>(out),
      .block = block,
      .a = a,
      .b = b,
  };
  Drive::run(core, first, count, at, state...);
}

} // namespace

void execute_core(const plan_core& core, std::size_t first, std::size_t count, const void* in,
                  void* out, const workspace& ws) noexcept {
  assert(ws.size() >= core.bytes && "fftn: workspace smaller than plan::workspace_bytes()");
  reset_team_state(ws);
  if (core.prog.leaf_rows) {
    // Rows of leaves use no scratch and need no slot_addresses (driver.hpp).
    assert(first <= core.prog.transforms && count <= core.prog.transforms - first &&
           "fftn: execute_transforms beyond the batch");
    assert((!core.in_place || in == out) && "fftn: in-place plan executed with in != out");
    assert((core.in_place || in != out) && "fftn: out-of-place plan executed with in == out");
    if (count != 0) {
      run_leaf_rows(core.prog, first, count, in, out);
    }
    return;
  }
  execute_program<plain_drive>(core, first, count, in, out, ws);
}

void execute_all(const plan_core& core, const void* in, void* out, const workspace& ws) noexcept {
  execute_core(core, 0, core.prog.transforms, in, out, ws);
}

void execute_core(const plan_core& core, std::size_t first, std::size_t count, const void* in,
                  void* out, const workspace& ws, const exec_args& args) noexcept {
  if (!core.needs_args) {
    execute_core(core, first, count, in, out, ws);
    return;
  }
  // A plan whose edges have factors runs on the edge driver (core/drive.hpp, edge_drive), with
  // the factor cursors starting at the arrays: execute_core's path above without its shortcuts
  // (programs with factor edges are walked).
  execute_core_cursors(core, first, count, in, out, ws, edge_state_of(args, core.factor_args));
}

void execute_core_cursors(const plan_core& core, std::size_t first, std::size_t count,
                          const void* in, void* out, const workspace& ws,
                          const edge_state& f) noexcept {
  assert(ws.size() >= core.bytes && "fftn: workspace smaller than plan::workspace_bytes()");
  reset_team_state(ws);
  execute_program<edge_drive>(core, first, count, in, out, ws, f);
}

void execute_all(const plan_core& core, const void* in, void* out, const workspace& ws,
                 const exec_args& args) noexcept {
  execute_core(core, 0, core.prog.transforms, in, out, ws, args);
}

std::size_t transform_count(const plan_core& core) noexcept { return core.prog.transforms; }

isa instruction_set(const plan_core& core) noexcept {
  return static_cast<isa>(static_cast<unsigned char>(core.isa_target));
}

std::string describe(const plan_core& core) {
  std::ostringstream o;
  if (!core.pipeline_title.empty()) {
    o << core.pipeline_title << "\n";
  } else {
    o << "fftn plan: " << kind_name(core.kind);
    if (core.kind == transform_kind::c2c) {
      o << (core.dir == direction::forward ? " forward" : " backward");
    }
    o << ", " << (core.prec == precision::f64 ? "double" : "float") << ", n = ";
    // The lengths as the shape lists its axes (plan_core::axis_of).
    std::vector<std::size_t> listed = core.lengths;
    for (std::size_t d = 0; d < core.axis_of.size(); ++d) {
      listed[core.axis_of[d]] = core.lengths[d];
    }
    for (std::size_t d = 0; d < listed.size(); ++d) {
      o << (d == 0 ? "" : " x ") << listed[d];
    }
    o << ", " << core.prog.transforms << (core.prog.transforms == 1 ? " transform" : " transforms")
      << (core.in_place ? ", in place" : ", out of place") << "\n";
  }
  o << "instruction set: " << isa_name(instruction_set(core)) << "\n";
  // Some avx2 rules apply to one vendor's processors only (DESIGN.md 13.5).
  o << "processor vendor: " << cpu_vendor_name(core.profile.vendor) << "\n";
  // How the program was chosen, when not by the rules alone (measured planning, wisdom).
  if (!core.planning_note.empty()) {
    o << "planning: " << core.planning_note << "\n";
  }
  const char* rule = core.built == strategy_choice::grid ? "the grid's phase rule" : "planner v1";
  if (!core.strategy_note.empty()) {
    o << "strategy: " << core.strategy_note << "; radices from ";
  } else {
    o << "strategy: chain (Stockham autosort, breadth-first), one line at a time; each "
         "transform finished before the next; radices from ";
  }
  if (core.radices_forced) {
    o << (core.planning_note.empty() ? "build_settings.radices (a test knob)"
                                     : "the chosen candidate")
      << " where they multiply to a line's length, otherwise from " << rule << "\n";
  } else {
    o << rule << "\n";
  }
  for (const std::string& note : core.axis_notes) {
    o << note << "\n";
  }
  if (!core.edges_note.empty()) {
    o << core.edges_note << "\n";
  }
  if (core.team != nullptr) {
    if (core.team->strategy_note != core.strategy_note) {
      o << "with a team of two or more: " << core.team->strategy_note << "\n";
    }
    for (std::size_t i = 0; i < core.team->axis_notes.size(); ++i) {
      if (i >= core.axis_notes.size() || core.team->axis_notes[i] != core.axis_notes[i]) {
        o << "with a team of two or more: " << core.team->axis_notes[i] << "\n";
      }
    }
  }
  o << "workspace: " << core.bytes << " bytes (scratch lines a, b of " << core.scratch_line
    << " complex each; block of " << core.block_elements << " complex)\n";
  const auto list = [&o](const std::vector<record>& records) {
    std::vector<std::size_t> ends;
    for (std::size_t i = 0; i < records.size(); ++i) {
      while (!ends.empty() && ends.back() == i) {
        ends.pop_back();
      }
      const std::string indent(2 * (ends.size() + 1), ' ');
      const record& rec = records[i];
      if (const loop* l = std::get_if<loop>(&rec)) {
        o << indent << "loop " << (l->kind == loop_kind::batch ? "batch" : "lines") << " x"
          << l->count << " (steps in bytes: in " << l->step[0] << ", out " << l->step[1]
          << ", block " << l->step[2]
          << (l->step[3] != 0 ? ", twiddles " + std::to_string(l->step[3]) : "") << ")";
        if (l->start != std::array<std::ptrdiff_t, moving_slots>{}) {
          o << " from (in " << l->start[0] << ", out " << l->start[1] << ", block " << l->start[2]
            << ")";
        }
        o << "\n";
        ends.push_back(l->body_end);
        continue;
      }
      const auto& st = std::get<stage>(rec);
      o << indent << st.name << ": " << slot_name(st.src) << " -> " << slot_name(st.dst) << ", "
        << st.units << " units in " << st.chunks << (st.chunks == 1 ? " chunk" : " chunks") << "\n";
    }
  };
  o << "program:\n";
  list(core.prog.records);
  if (core.prog.lanes != 0) {
    o << "lanes tail (blocks of " << core.prog.lane_unit << " rows, the last padded):\n";
    list(core.prog.lane_tail);
  }
  return o.str();
}

} // namespace fftn::inline v0::detail
