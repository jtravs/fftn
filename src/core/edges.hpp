// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Plan edges (DESIGN.md 12.1, fftn/edges.hpp): their validation, where each operation of the
// vocabulary goes in the program, and the separate edge pass.
//
// The scales of both edges and options::scale are one product applied once (the transform is
// linear, so where it applies changes only the rounding). A complex factor of stride 1 along
// the line is fused into the line's first pass (input) or last pass (output): the edge
// variants of every pass kind (passes/pass_args.hpp, edge_op::factor), reading the array
// through the factor cursor (core/edge_driver.hpp, stage::moving_factors). Everything else (a real
// factor, a complex factor of another stride, accumulation, and a scale when both factors are
// fused) is a separate edge pass: the input's from the input into the workspace block before
// the line's chain, the output's from the block into the output after it. Experiment X65
// (task 5.1f) is to decide which of these pay as fused variants.
#pragma once

#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

// One edge as the plan applies it.
struct edge_side {
  factor_kind kind = factor_kind::none;
  std::size_t factor_bytes = 0;   // of one element of the array
  std::ptrdiff_t axis_stride = 0; // in the array's elements, along the line (one axis)
  // In the array's elements, per axis of the shape (in the shape's order): 0 along an axis of
  // one point on the edge's side. Multi-axis and real plans read the factor through these
  // (the edge phases, plan_core.cpp).
  std::vector<std::ptrdiff_t> axis_strides;
  std::vector<std::ptrdiff_t> batch_steps; // bytes per batch dimension, in the shape's order
  bool fused = false;                      // the factor in the line's first or last pass
  bool separate = false;                   // a separate edge pass through the workspace block
  bool accumulate = false;                 // output only, always in the separate pass
  double scale = 1.0;                      // applied by the separate pass
};

struct edge_plan {
  edge_side in;
  edge_side out;
  // A factor or accumulation: the plan is built with these edges (the chain, by the rules).
  // Otherwise the edges are scales only, folded into options::scale.
  bool any = false;
  // The fused scale (options::scale of the resolved options) applies on the input edge
  // (build_settings::scale_on_input): when the output edge's factor is fused and the input
  // edge has none.
  bool scale_on_input = false;
};

struct resolved_edges {
  options opts; // the caller's, with scale the product the plan fuses (1 when a pass takes it)
  edge_plan plan;
};

// Validates `e` against the shape and options, throwing std::invalid_argument naming the
// offending field (edges.in.scale, edges.out.mul.axis_strides, ...), and places each operation.
// fuse: a c2c row's complex factors of stride 1 fuse into its chain's passes (the plan is then
// a chain); otherwise every factor is a separate pass, which any strategy takes.
[[nodiscard]] resolved_edges resolve_edges(const shape_view& s, const options& opts,
                                           const edges_view& e, bool fuse = true);

// Whether either edge's factor is fused into the passes (fuse above).
[[nodiscard]] inline bool any_fused(const edge_plan& p) noexcept {
  return p.in.fused || p.out.fused;
}

// The separate edge pass of one side over a line of n complex values (or, `real`, n real
// values: the real side of a real transform) of precision `prec`: read at src (stride
// src_stride), multiplied by the side's factor (through the factor cursor of that side, with
// factor_stride elements of the array between the line's elements) and scale, and stored or
// added at dst (stride dst_stride). Its units are the line's elements, in chunks of about
// chunk_points.
[[nodiscard]] stage edge_pass_stage(precision prec, const edge_side& side, bool output,
                                    std::size_t n, slot src, std::ptrdiff_t src_stride, slot dst,
                                    std::ptrdiff_t dst_stride, std::size_t chunk_points, bool real,
                                    std::ptrdiff_t factor_stride);

// How an edge pass writes: stores its value, adds it to the destination's (the output edge's
// accumulation), or adds it to a prior value read at the source, the value itself being read
// at the destination (accumulation into a strategy that writes the output: the output's prior
// values saved in the workspace block before the transform, edge_copy_stage).
enum class edge_mode : std::uint8_t { store, accumulate, add_prior };

// The geometry of an edge pass beyond one line: a lanes block's lines (pass_args::rows, set by
// the driver per block) src_dist and dst_dist elements apart and their factors factor_step
// bytes apart; and bytes added to the source and the destination (a region of the workspace
// block past the strategy's own).
struct edge_rows {
  std::ptrdiff_t src_dist = 0;
  std::ptrdiff_t dst_dist = 0;
  std::ptrdiff_t factor_step = 0;
  std::size_t src_offset = 0;
  std::size_t dst_offset = 0;
};

// edge_pass_stage with a mode and a geometry.
[[nodiscard]] stage edge_pass_stage(precision prec, const edge_side& side, bool output,
                                    std::size_t n, slot src, std::ptrdiff_t src_stride, slot dst,
                                    std::ptrdiff_t dst_stride, std::size_t chunk_points, bool real,
                                    std::ptrdiff_t factor_stride, edge_mode mode,
                                    const edge_rows& rows);

// A copy of a line of n elements (real or complex), with the geometry of edge_rows: the
// output's prior values saved before a transform that writes the output, for accumulation.
[[nodiscard]] stage edge_copy_stage(precision prec, std::size_t n, slot src,
                                    std::ptrdiff_t src_stride, slot dst, std::ptrdiff_t dst_stride,
                                    std::size_t chunk_points, bool real, const edge_rows& rows);

// Where a plan's edges went: a c2c row's chain (fused, or separate passes through the workspace
// block), passes around a row's other strategy, or edge phases (multi-axis plans, real rows'
// chains).
enum class edge_layout : std::uint8_t { row_chain, row_strategy, phases };

// describe()'s line for a plan with edges; `fallback`: why an edge keeps the plan from the
// strategy the rules would choose (empty: it does not).
[[nodiscard]] std::string edges_text(const edge_plan& p, edge_layout layout,
                                     const std::string& fallback);

// The elements of a valid edge's factor array its strides reach over the shape, relative to
// the address passed (exec_args): [lo, hi], in the array's elements; {0, 0} without a factor.
// Measured planning times candidates on scratch arrays of this span (core/measure.cpp).
[[nodiscard]] std::pair<std::ptrdiff_t, std::ptrdiff_t>
factor_span(const shape_view& s, const edge_view& e, bool output);

// The edges and options::scale as a wisdom key adds them (core/wisdom.cpp, shape_key): empty
// for scales only.
[[nodiscard]] std::string edges_key(const edges_view& e, double options_scale);

// Ends the program with `what` on stderr: a broken precondition that execution, which cannot
// throw, detects.
[[noreturn]] void contract_failure(const char* what) noexcept;

} // namespace fftn::inline v0::detail
