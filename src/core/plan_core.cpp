// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Plan construction: validation, the default layouts, and the program of a transform of
// any rank up to three (DESIGN.md 9.3): each transform is finished before the next, one
// axis at a time, one line at a time, every line through the chain strategy. The layout
// rules are stated for users in docs/conventions.md; this file is their implementation.

#include "core/plan_core.hpp"

#include "core/driver.hpp"
#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/workspace.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"
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
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
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
  std::vector<std::size_t> extent;    // per axis (the halved one included)
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
side resolve(const shape_view& s, const std::vector<std::size_t>& extent, std::size_t last_physical,
             bool input, bool complex, std::ptrdiff_t element_bytes) {
  const std::size_t r = extent.size();
  side out;
  out.extent = extent;
  out.complex = complex;
  out.element_bytes = element_bytes;
  out.stride.assign(r, 0);
  std::ptrdiff_t span = 1;
  for (std::size_t i = r; i > 0; --i) {
    const std::size_t d = i - 1;
    const std::ptrdiff_t given = input ? s.axes[d].in_stride : s.axes[d].out_stride;
    if (extent[d] > 1) {
      out.stride[d] = given != 0
                          ? given_step(given, axis_name(d) + (input ? ".in_stride" : ".out_stride"))
                          : span;
    }
    span = checked_mul(span, to_signed(d + 1 == r ? last_physical : extent[d], axis_name(d) + ".n"),
                       axis_name(d) + ".n");
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
  for (std::size_t d = 0; d < v.extent.size(); ++d) {
    reach = checked_add(
        reach, checked_mul(to_signed(v.extent[d] - 1, what), magnitude(v.stride[d]), what), what);
  }
  for (std::size_t b = 0; b < v.dist.size(); ++b) {
    reach = checked_add(
        reach, checked_mul(to_signed(s.batches[b].count - 1, what), magnitude(v.dist[b]), what),
        what);
  }
  static_cast<void>(checked_mul(checked_add(reach, 1, what), v.element_bytes, what));
}

// No two output elements may share an address. Ordered by size of stride, each dimension's
// stride must exceed the reach of all smaller ones: a sufficient condition that admits
// every nested layout (rows, columns, planes, padding, reversed strides).
void check_no_overlap(const shape_view& s, const side& v) {
  std::vector<std::pair<std::ptrdiff_t, std::size_t>> dims; // |stride|, extent
  for (std::size_t d = 0; d < v.extent.size(); ++d) {
    if (v.extent[d] > 1) {
      dims.emplace_back(magnitude(v.stride[d]), v.extent[d]);
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
  if (s.axes[r - 1].n > 1 && (real.stride[r - 1] != 1 || cplx.stride[r - 1] != 1)) {
    reject("options.in_place", "needs the last axis contiguous on both sides (" + axis_name(r - 1) +
                                   " strides 1) for a real transform");
  }
  for (std::size_t d = 0; d + 1 < r; ++d) {
    if (s.axes[d].n > 1 && real.stride[d] != 2 * cplx.stride[d]) {
      reject("options.in_place",
             "needs " + axis_name(d) + "'s real-side stride twice its complex-side stride");
    }
  }
  for (std::size_t b = 0; b < real.dist.size(); ++b) {
    if (s.batches[b].count > 1 && real.dist[b] != 2 * cplx.dist[b]) {
      reject("options.in_place",
             "needs " + batch_name(b) + "'s real-side distance twice its complex-side distance");
    }
  }
}

template <class T> std::vector<std::vector<T>>& tables_of(plan_core& c) {
  if constexpr (std::is_same_v<T, double>) {
    return c.tables_f64;
  } else {
    return c.tables_f32;
  }
}

// The target of the plan: the fastest the processor and the build support, or the one
// options::force_isa names. Every target has the passes.
target plan_target(const options& opts) {
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

// A phase's view of one slot: where it is and its strides per axis, in elements.
struct view {
  slot where = slot::in;
  std::vector<std::ptrdiff_t> stride;
  std::ptrdiff_t element_bytes = 0;
};

// Appends the loops over the lines of every axis but `axis` (outermost first), the line's
// stages inside them, and closes the loops.
void add_phase(program& p, std::size_t axis, const std::vector<std::size_t>& extent,
               const view& src, const view& dst, std::vector<stage>&& line) {
  std::vector<std::size_t> open;
  for (std::size_t e = 0; e < extent.size(); ++e) {
    if (e == axis || extent[e] == 1) {
      continue;
    }
    loop l{.kind = loop_kind::lines, .count = extent[e]};
    for (const view* v : {&src, &dst}) {
      const std::ptrdiff_t bytes = v->stride[e] * v->element_bytes;
      switch (v->where) {
      case slot::in:
        l.step[0] = bytes;
        break;
      case slot::out:
        l.step[1] = bytes;
        break;
      case slot::block:
        l.step[2] = bytes;
        break;
      case slot::a:
      case slot::b:
        break;
      }
    }
    open.push_back(p.records.size());
    p.records.emplace_back(l);
  }
  for (stage& st : line) {
    p.records.emplace_back(std::move(st));
  }
  for (const std::size_t i : open) {
    std::get<loop>(p.records[i]).body_end = static_cast<std::uint32_t>(p.records.size());
  }
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
// its split: beyond the profile's grid_min_bytes or when forced, if the row is contiguous on
// both sides, carries no edge factor (the grid applies only the scale, on its output or,
// with build_settings::scale_on_input, its input), has no forced radices for its whole
// length and has a split whose phases are codelet chains. The row is a c2c row or the
// half-length complex transform of a real row of 2n (DESIGN.md 9.1).
template <class T>
grid_split plan_grid(std::size_t n, const side& in, const side& out, const build_settings& settings,
                     const chain_builder<T>& chain, std::size_t lanes) {
  if (n < 2 || settings.strategy == strategy_choice::chain || settings.in_factor != nullptr ||
      settings.out_factor != nullptr || in.stride[0] != 1 || out.stride[0] != 1) {
    return {};
  }
  if (settings.strategy == strategy_choice::rules &&
      n * 2 * sizeof(T) <= settings.profile.grid_min_bytes) {
    return {};
  }
  std::size_t forced = 1;
  for (const std::size_t f : settings.radices) {
    forced *= f;
  }
  if (!settings.radices.empty() && forced == n) {
    return {};
  }
  const grid_split g = choose_grid(n, lanes, settings.profile.grid_block_bytes / (4 * sizeof(T)),
                                   n * 2 * sizeof(T) > settings.profile.grid_three_bytes,
                                   settings.profile.grid_block, settings.grid_factors);
  for (std::size_t i = 0; i < g.phases; ++i) {
    for (const std::size_t radix : chain.radices(g.factors.at(i))) {
      if (!has_codelet(radix)) {
        return {}; // forced radices for a phase's length that need the generic pass
      }
    }
  }
  return g;
}

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

// Makes the records from `first` on read the output slot where they read the input: the
// grid of a c2r row runs in place on the output, which the pairing pass has written (the
// grid builder reads its input from the input slot, and with in_place set touches it only
// through phase 1's gather and that phase's loop).
void read_output_for_input(program& p, std::size_t first) {
  for (std::size_t i = first; i < p.records.size(); ++i) {
    if (stage* st = std::get_if<stage>(&p.records[i])) {
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

template <class T>
void build(plan_core& c, const shape_view& s, const side& in, const side& out, const options& opts,
           const build_settings& settings) {
  c.isa_target = plan_target(opts);
  const kernels_of<T>& kernels = of<T>(*find_kernels(c.isa_target));
  target_profile profile = profile_of(c.isa_target, kernels.complex_lanes);
  if (settings.leaf_limit != 0) {
    profile.leaf_limit = settings.leaf_limit;
  }
  if (settings.layout != layout_choice::rules) {
    profile.split = settings.layout == layout_choice::split;
  }
  chain_builder<T> chain(kernels, profile, tables_of<T>(c), settings.chunk_points,
                         settings.radices, settings.bluestein_above,
                         fusion_of(settings, c.isa_target));
  program& p = c.prog;
  const std::size_t r = s.axes.size();
  const auto complex_bytes = static_cast<std::ptrdiff_t>(2 * sizeof(T));

  p.transforms = 1;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    p.records.emplace_back(loop{
        .kind = loop_kind::batch,
        .count = s.batches[b].count,
        .step = {in.dist[b] * in.element_bytes, out.dist[b] * out.element_bytes, 0},
    });
    p.transforms *= s.batches[b].count;
  }
  p.batch_loops = s.batches.size();

  const view in_view{.where = slot::in, .stride = in.stride, .element_bytes = in.element_bytes};
  const view out_view{.where = slot::out, .stride = out.stride, .element_bytes = out.element_bytes};
  std::size_t scratch = 0;
  const auto lines = [&](std::size_t axis, const std::vector<std::size_t>& extent, const view& src,
                         const view& dst, std::size_t line_scratch, std::vector<stage>&& stages) {
    scratch = std::max(scratch, line_scratch);
    add_phase(p, axis, extent, src, dst, std::move(stages));
  };
  const auto at = [](const view& v, std::size_t axis) {
    return endpoint{.where = v.where, .stride = v.stride[axis]};
  };
  const int sign = static_cast<int>(s.dir);

  const grid_split grid_plan =
      s.kind == transform_kind::c2c && r == 1
          ? plan_grid<T>(c.lengths[0], in, out, settings, chain, kernels.complex_lanes)
          : grid_split{};
  // A real row of one axis whose half-length complex transform is a grid (DESIGN.md 9.1):
  // the grid writes the output and the pairing runs after it in place (r2c), or the pairing
  // writes the output and the grid runs in place there (c2r).
  const grid_split real_grid =
      s.kind != transform_kind::c2c && r == 1 && c.lengths[0] % 2 == 0
          ? plan_grid<T>(c.lengths[0] / 2, in, out, settings, chain, kernels.complex_lanes)
          : grid_split{};
  if (real_grid.phases != 0) {
    const std::size_t n = c.lengths[0];
    const bool r2c = s.kind == transform_kind::r2c;
    grid_builder<T> grid(kernels, chain, tables_of<T>(c),
                         settings.grid_fused &&
                             (n / 2) * 2 * sizeof(T) <= settings.profile.grid_fuse_bytes,
                         settings.grid_phase_radix);
    const bool in_place = c.in_place || !r2c;
    if (grid_builder<T>::uses_block(real_grid, in_place)) {
      c.block_elements = n / 2;
    }
    std::vector<stage> pairing;
    if (r2c) {
      scratch = grid.c2c_row(real_grid, -1, in_place, 1.0, false, p);
      chain.add_pairing(true, n, {slot::out, 1}, {slot::out, 1}, opts.scale, pairing);
      p.records.emplace_back(std::move(pairing.front()));
    } else {
      chain.add_pairing(false, n, {slot::in, 1}, {slot::out, 1}, 1.0, pairing);
      p.records.emplace_back(std::move(pairing.front()));
      const std::size_t first = p.records.size();
      scratch = grid.c2c_row(real_grid, +1, true, opts.scale, false, p);
      read_output_for_input(p, first);
    }
    c.axis_notes.push_back("axis 0 (n = " + std::to_string(n) + "): " + (r2c ? "r2c" : "c2r") +
                           " through a half-length complex " + grid.note(real_grid) +
                           (r2c ? ", then a pairing pass" : ", after a pairing pass"));
    c.strategy_note = std::string("grid for the half-length complex transform (") +
                      std::to_string(real_grid.phases) +
                      " phases, each depth-first over blocks of columns" +
                      (c.block_elements != 0 ? "; an intermediate array in the workspace block" : "") +
                      "), each transform finished before the next";
  } else if (grid_plan.phases != 0) {
    grid_builder<T> grid(kernels, chain, tables_of<T>(c),
                         settings.grid_fused &&
                             c.lengths[0] * 2 * sizeof(T) <= settings.profile.grid_fuse_bytes,
                         settings.grid_phase_radix);
    if (grid_builder<T>::uses_block(grid_plan, c.in_place)) {
      c.block_elements = c.lengths[0];
    }
    scratch = grid.c2c_row(grid_plan, sign, c.in_place, opts.scale, settings.scale_on_input, p);
    c.axis_notes.push_back("axis 0 (n = " + std::to_string(c.lengths[0]) +
                           "): " + grid.note(grid_plan));
    c.strategy_note =
        "grid (Cooley-Tukey in " + std::to_string(grid_plan.phases) +
        " phases, each depth-first over blocks of columns" +
        (c.block_elements != 0 ? "; an intermediate array in the workspace block" : "") +
        "), each transform finished before the next";
  } else if (s.kind == transform_kind::c2c) {
    for (std::size_t i = r; i > 0; --i) {
      const std::size_t d = i - 1;
      const view& src = d + 1 == r ? in_view : out_view;
      const bool aliased = src.where == slot::out || c.in_place;
      std::vector<stage> st;
      const std::size_t need =
          chain.c2c_line(c.lengths[d], sign, at(src, d), at(out_view, d), aliased,
                         c2c_edges(opts, settings, d + 1 == r, d == 0), st);
      lines(d, in.extent, src, out_view, need, std::move(st));
      c.axis_notes.push_back("axis " + std::to_string(d) + " (n = " + std::to_string(c.lengths[d]) +
                             "): radices " + chain.radix_text(c.lengths[d]));
    }
  } else if (s.kind == transform_kind::r2c) {
    std::vector<stage> st;
    const std::size_t need =
        chain.r2c_line(c.lengths[r - 1], at(in_view, r - 1), at(out_view, r - 1), c.in_place,
                       r == 1 ? opts.scale : 1.0, st);
    lines(r - 1, in.extent, in_view, out_view, need, std::move(st));
    c.axis_notes.push_back("axis " + std::to_string(r - 1) +
                           " (n = " + std::to_string(c.lengths[r - 1]) +
                           "): " +
                           chain.real_text(c.lengths[r - 1], true,
                                           in.stride[r - 1] != 1 || out.stride[r - 1] != 1));
    for (std::size_t i = r - 1; i > 0; --i) {
      const std::size_t d = i - 1;
      std::vector<stage> cs;
      const std::size_t n = c.lengths[d];
      const std::size_t cneed = chain.c2c_line(n, -1, at(out_view, d), at(out_view, d), true,
                                               c2c_edges(opts, settings, false, d == 0), cs);
      lines(d, out.extent, out_view, out_view, cneed, std::move(cs));
      c.axis_notes.push_back("axis " + std::to_string(d) + " (n = " + std::to_string(n) +
                             "): c2c forward, radices " + chain.radix_text(n));
    }
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
      c.block_elements = static_cast<std::size_t>(span);
    }
    // In place the complex phases work on the output slot, which is the input's memory,
    // through the input's (complex) strides.
    const view work =
        c.in_place ? view{.where = slot::out, .stride = in.stride, .element_bytes = complex_bytes}
                   : block_view;
    for (std::size_t i = r - 1; i > 0; --i) {
      const std::size_t d = i - 1;
      const view& src = d + 2 == r && !c.in_place ? in_view : work;
      std::vector<stage> cs;
      const std::size_t n = c.lengths[d];
      const std::size_t cneed =
          chain.c2c_line(n, +1, at(src, d), at(work, d), src.where == work.where, line_edges(), cs);
      lines(d, in.extent, src, work, cneed, std::move(cs));
      c.axis_notes.push_back("axis " + std::to_string(d) + " (n = " + std::to_string(n) +
                             "): c2c backward, radices " + chain.radix_text(n));
    }
    const view& src = r > 1 && !c.in_place ? block_view : in_view;
    std::vector<stage> st;
    const std::size_t need =
        chain.c2r_line(c.lengths[r - 1], at(src, r - 1), at(out_view, r - 1), opts.scale, st);
    lines(r - 1, out.extent, src, out_view, need, std::move(st));
    c.axis_notes.push_back("axis " + std::to_string(r - 1) +
                           " (n = " + std::to_string(c.lengths[r - 1]) +
                           "): " +
                           chain.real_text(c.lengths[r - 1], false,
                                           src.stride[r - 1] != 1 || out.stride[r - 1] != 1));
  }
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    std::get<loop>(p.records[b]).body_end = static_cast<std::uint32_t>(p.records.size());
  }

  // Workspace layout.
  constexpr std::size_t line_alignment = 64;
  const auto round_up = [](std::size_t v) {
    return (v + line_alignment - 1) / line_alignment * line_alignment;
  };
  c.scratch_line = scratch;
  const std::size_t line_bytes = round_up(scratch * 2 * sizeof(T));
  // A grid's scratch line b starts an odd number of 64-byte lines (seven) after a whole
  // multiple of a's size, so that the passes ping-ponging between them do not read and
  // write the same cache sets (DESIGN.md 10, rule 4): a grid's scratch lines are a block of
  // a power-of-two size for most lengths. Experiment X4 measured 8-9 % at 2^22 on the Mac
  // and no case slower; the chain's lines were not part of it.
  const std::size_t stagger = grid_plan.phases != 0 ? 7 * 64 : 0;
  c.offset_b = line_bytes + stagger;
  c.offset_block = 2 * line_bytes + stagger;
  const std::size_t used = c.offset_block + c.block_elements * 2 * sizeof(T);
  c.bytes = used == 0 ? 0 : used + line_alignment;
  c.slot_bytes = (used + sync_line - 1) / sync_line * sync_line;
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

} // namespace

std::shared_ptr<const plan_core> make_plan_core(const shape_view& s, const options& opts,
                                                const build_settings& settings) {
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
    in = resolve(s, full, full[r - 1], true, true, cb);
    out = resolve(s, full, full[r - 1], false, true, cb);
    break;
  case transform_kind::r2c:
    in = resolve(s, full, padded, true, false, rb);
    out = resolve(s, halved, half, false, true, cb);
    break;
  case transform_kind::c2r:
    in = resolve(s, halved, half, true, true, cb);
    out = resolve(s, full, padded, false, false, rb);
    break;
  }
  check_reach(s, in, true);
  check_reach(s, out, false);
  check_no_overlap(s, out);
  if (opts.in_place) {
    check_in_place(s, in, out);
  }

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

  if (s.prec == precision::f64) {
    build<double>(*c, s, in, out, opts, settings);
  } else {
    build<float>(*c, s, in, out, opts, settings);
  }
  std::size_t points = 1;
  for (const std::size_t n : c->lengths) {
    points *= n; // the layouts' reach checks have bounded the product
  }
  mark_flat(c->prog);
  c->graph = build_work_graph(c->prog, points, settings.chunk_points);
  c->profile = settings.profile;
  c->runs_inline = runs_inline(c->graph, c->profile);
  c->radices_forced = !settings.radices.empty();
  return c;
}

std::shared_ptr<const plan_core> make_plan_core(const shape_view& shape, const options& opts) {
  return make_plan_core(shape, opts, build_settings{});
}

void execute_core(const plan_core& core, std::size_t first, std::size_t count, const void* in,
                  void* out, const workspace& ws) noexcept {
  assert(ws.size() >= core.bytes && "fftn: workspace smaller than plan::workspace_bytes()");
  // The scratch below overlaps any team state the workspace holds: the next team execution
  // sets it up afresh (team_state.hpp).
  const sync_word team = workspace_access::team_word(ws);
  if (team.load(std::memory_order_relaxed) != 0) {
    team.store(0, std::memory_order_relaxed);
  }
  assert(first <= core.prog.transforms && count <= core.prog.transforms - first &&
         "fftn: execute_transforms beyond the batch");
  assert((!core.in_place || in == out) && "fftn: in-place plan executed with in != out");
  assert((core.in_place || in != out) && "fftn: out-of-place plan executed with in == out");
  if (core.prog.leaf_rows) {
    // Rows of leaves use no scratch and need no slot_addresses (driver.hpp).
    run_leaf_rows(core.prog, first, count, in, out);
    return;
  }
  slot_addresses at{.in = static_cast<const std::byte*>(in), .out = static_cast<std::byte*>(out)};
  if (core.bytes != 0) {
    // The workspace may have any alignment; its size includes the slack for this.
    std::byte* raw = ws.memory().data();
    const auto address = reinterpret_cast<std::uintptr_t>(raw);
    const std::size_t skip = (64 - address % 64) % 64;
    std::byte* base = raw + skip;
    at.a = base;
    at.b = base + core.offset_b;
    at.block = core.block_elements != 0 ? base + core.offset_block : nullptr;
  }
  run_transforms(core.prog, first, count, at);
}

std::size_t transform_count(const plan_core& core) noexcept { return core.prog.transforms; }

isa instruction_set(const plan_core& core) noexcept {
  return static_cast<isa>(static_cast<unsigned char>(core.isa_target));
}

std::string describe(const plan_core& core) {
  std::ostringstream o;
  o << "fftn plan: " << kind_name(core.kind);
  if (core.kind == transform_kind::c2c) {
    o << (core.dir == direction::forward ? " forward" : " backward");
  }
  o << ", " << (core.prec == precision::f64 ? "double" : "float") << ", n = ";
  for (std::size_t d = 0; d < core.lengths.size(); ++d) {
    o << (d == 0 ? "" : " x ") << core.lengths[d];
  }
  o << ", " << core.prog.transforms << (core.prog.transforms == 1 ? " transform" : " transforms")
    << (core.in_place ? ", in place" : ", out of place") << "\n";
  o << "instruction set: " << isa_name(instruction_set(core)) << "\n";
  const char* rule = core.strategy_note.empty() ? "planner v1" : "the grid's phase rule";
  if (!core.strategy_note.empty()) {
    o << "strategy: " << core.strategy_note << "; radices from ";
  } else {
    o << "strategy: chain (Stockham autosort, breadth-first), one line at a time; each "
         "transform finished before the next; radices from ";
  }
  if (core.radices_forced) {
    o << "build_settings.radices (a test knob) where they multiply to a line's length, "
         "otherwise from "
      << rule << "\n";
  } else {
    o << rule << "\n";
  }
  for (const std::string& note : core.axis_notes) {
    o << note << "\n";
  }
  o << "workspace: " << core.bytes << " bytes (scratch lines a, b of " << core.scratch_line
    << " complex each; block of " << core.block_elements << " complex)\n";
  o << "program:\n";
  std::vector<std::size_t> ends;
  for (std::size_t i = 0; i < core.prog.records.size(); ++i) {
    while (!ends.empty() && ends.back() == i) {
      ends.pop_back();
    }
    const std::string indent(2 * (ends.size() + 1), ' ');
    const record& rec = core.prog.records[i];
    if (const loop* l = std::get_if<loop>(&rec)) {
      o << indent << "loop " << (l->kind == loop_kind::batch ? "batch" : "lines") << " x"
        << l->count << " (steps in bytes: in " << l->step[0] << ", out " << l->step[1] << ", block "
        << l->step[2] << (l->step[3] != 0 ? ", twiddles " + std::to_string(l->step[3]) : "")
        << ")\n";
      ends.push_back(l->body_end);
      continue;
    }
    const auto& st = std::get<stage>(rec);
    o << indent << st.name << ": " << slot_name(st.src) << " -> " << slot_name(st.dst) << ", "
      << st.units << " units in " << st.chunks << (st.chunks == 1 ? " chunk" : " chunks") << "\n";
  }
  return o.str();
}

} // namespace fftn::inline v0::detail
