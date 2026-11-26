// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Real-to-real plans (DESIGN.md 5.7, fftn/shape.hpp r2r_kind; task 6.3): a program (8.3) of one
// phase per axis over the axis's lines, each line computed in scratch by a read stage, a complex
// transform the plan already has, and a store stage. Built so far: the DHT.
//
// The DHT of a line of n reals is the r2c of n followed by H_k = Re X_k - Im X_k and
// H_(n-k) = Re X_k + Im X_k. The line never leaves the scratch lines of the member running it:
//
//   read stage: the line (any stride, the input edge's real factor and scale) packed into scratch
//     line a as the complex line of the r2c (n even: h = n/2 values x_2j + i x_2j+1; n odd: the n
//     values x_j + 0i);
//   the forward complex chain of h (or n) in scratch (chain_builder::scratch_transform);
//   store stage: the r2c's pairing of bins k and h - k (n even) and the Hartley combination in
//     one pass, written to the line's destination (any stride), with the output edge's real
//     factor, the scales' one product and accumulation.
//
// So the r2c's separate pairing pass and its half spectrum never exist: the store stage reads the
// half-length transform. Every stage's units are independent (the store's unit k writes H_k and
// H_(n-k)), so a team may share one line's stages out as it shares a c2c row's. Several axes are
// a phase each, the line's stages inside loops over the other axes (5.1b's multi-axis phases),
// from the input into the output, then in place on the output; with accumulation the
// intermediate phases go through the workspace block (a whole transform of reals), so that the
// output's prior values survive until the last phase adds to them. Every phase runs on its
// members' scratch lines, which is why the block is not the line's home here as it is in a
// pipeline (12.2): the lines of one phase run on several members at once, and the block is shared
// by the task.
#pragma once

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace fftn::inline v0::detail {

// The layout of a real-to-real shape with every default resolved and checked as a plan checks
// its layout (src/core/plan_core.cpp; std::invalid_argument naming the shape's fields, and the
// counts checked first): strides per axis and distances per batch dimension in reals, and the
// bytes each side spans [lo, hi).
struct r2r_layout {
  std::vector<std::ptrdiff_t> in_stride;
  std::vector<std::ptrdiff_t> out_stride;
  std::vector<std::ptrdiff_t> in_dist;
  std::vector<std::ptrdiff_t> out_dist;
  std::ptrdiff_t in_lo = 0;
  std::ptrdiff_t in_hi = 0;
  std::ptrdiff_t out_lo = 0;
  std::ptrdiff_t out_hi = 0;
};
[[nodiscard]] r2r_layout resolve_r2r_layout(const shape_view& s, bool in_place);

// The plan of a real-to-real shape (shape_view::kind r2r) with its edges (null: none), by the
// rules whatever options::effort and options::use say. Throws std::invalid_argument naming the
// offending field: shape.kinds of the wrong size or a kind not built yet, stored runs, the
// layout, the edges (a complex factor among them) and the options.
[[nodiscard]] std::shared_ptr<const plan_core> make_r2r_core(const shape_view& shape,
                                                             const options& opts,
                                                             const edges_view* edges,
                                                             const build_settings& settings);

} // namespace fftn::inline v0::detail
