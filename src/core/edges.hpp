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
// through the factor cursor (program.hpp, stage::moving_factors). Everything else (a real
// factor, a complex factor of another stride, accumulation, and a scale when both factors are
// fused) is a separate edge pass: the input's from the input into the workspace block before
// the line's chain, the output's from the block into the output after it. Experiment X65
// (task 5.1f) is to decide which of these pay as fused variants.
#pragma once

#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {

// One edge as the plan applies it.
struct edge_side {
  factor_kind kind = factor_kind::none;
  std::size_t factor_bytes = 0;            // of one element of the array
  std::ptrdiff_t axis_stride = 0;          // in the array's elements, along the line (one axis)
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
[[nodiscard]] resolved_edges resolve_edges(const shape_view& s, const options& opts,
                                           const edges_view& e);

// The separate edge pass of one side over a line of n complex values of precision `prec`:
// read at src (stride src_stride), multiplied by the side's factor (through the factor cursor
// of that side) and scale, and stored or added at dst (stride dst_stride). Its units are the
// line's elements, in chunks of about chunk_points.
[[nodiscard]] stage edge_pass_stage(precision prec, const edge_side& side, bool output,
                                    std::size_t n, slot src, std::ptrdiff_t src_stride, slot dst,
                                    std::ptrdiff_t dst_stride, std::size_t chunk_points);

// describe()'s line for a plan with edges.
[[nodiscard]] std::string edges_text(const edge_plan& p);

// The edges as a wisdom key adds them (core/wisdom.cpp, shape_key): empty for scales only.
[[nodiscard]] std::string edges_key(const edges_view& e);

} // namespace fftn::inline v0::detail
