// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Row pipelines (DESIGN.md 12.2, fftn/pipeline.hpp; tasks 5.2b, 5.2c): a transform, a middle and
// the inverse transform over one axis, built as a program (8.3) over a scratch-resident line, and
// the parts of plan construction (core/plan_core.cpp) they share with plans.
//
// The <C, C, C> chain pair: batch loops, then the stages of one line,
//
//   [prologue pass] -> forward chain into the workspace block -> [middle pass] ->
//   inverse chain out of the block -> [epilogue pass]
//
// built by chain_builder::c2c_line (the forward from the input to the block, the inverse from
// the block to the output; scratch lines a and b in between). The block is the line's home: the
// batch loops never move it, so the spectrum stays in the member's cache. The prologue's factor
// fuses into the forward chain's first pass, the middle's into its last and the epilogue's into
// the inverse chain's last, through the factor cursors of the edge driver (the middle's is the
// third cursor, moving_mid_out_factor). What cannot fuse (a real factor, a factor of another
// stride along the line, accumulation) is an edge pass on the line in the block (core/edges.hpp,
// edge_pass_stage), which costs no sweep of memory. The scales are one product, applied once.
// A pipeline with factors runs on the edge driver with its batched forms (core/edge_driver.hpp,
// make_edge_forms) whatever machine_profile::edge_batched_forms says: leaf-sized lines are a flat
// body of two leaves, longer ones up to calls_max calls per line unrolled (X65 measured plans
// with edges losing these forms).
//
// The <T, C, T> real chain pair (task 5.2c) has the same shape with chain_builder::r2c_line and
// c2r_line, the middle on the spectrum's n/2 + 1 bins, and every factor in a separate pass
// (pipeline.cpp, build_real_pair).
#pragma once

#include "core/plan_core.hpp"
#include "core/runs.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"
#include "planner/rules.hpp"
#include "strategies/chain.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace fftn::inline v0::detail {

// The layout of a c2c shape of one axis (a pipeline's line under its batch), with every default
// resolved and checked as a plan checks it (std::invalid_argument naming the shape's fields):
// strides and distances in elements, and the bytes each side spans [lo, hi). real: the elements
// are reals, not complex values (a <T, C, T> pipeline's input and output, whose sides have the
// same logical extents and, in place, the same layout). runs: the line's stored runs
// (resolve_runs; task 5.2d), whose stored extents set the defaults and the reach, as a plan's.
struct row_layout {
  std::ptrdiff_t in_stride = 0;
  std::ptrdiff_t out_stride = 0;
  std::vector<std::ptrdiff_t> in_dist;
  std::vector<std::ptrdiff_t> out_dist;
  std::ptrdiff_t in_lo = 0;
  std::ptrdiff_t in_hi = 0;
  std::ptrdiff_t out_lo = 0;
  std::ptrdiff_t out_hi = 0;
};
[[nodiscard]] row_layout resolve_row_layout(const shape_view& s, bool in_place, bool real = false,
                                            const run_plan& runs = {});

// The target profile of c's instruction set (c.isa_target, set by the caller) with the settings'
// overrides, and the chain builder on it with the settings' rules, its tables in c's: what
// build() builds every line with.
[[nodiscard]] target_profile plan_profile(const plan_core& c, std::size_t complex_lanes,
                                          const build_settings& settings);
template <class T>
[[nodiscard]] chain_builder<T> plan_chain(plan_core& c, const target_profile& profile,
                                          const build_settings& settings);

// The top factors of the depth-first pair (DESIGN.md 12.2, task 5.2d) for a line of n, or none:
// with the rules, where X50's rule would make a Bluestein convolution of length n depth-first
// (depth_tops_for with a convolution's fit: C divides the tops' product, so that the first
// pass's blocks are whole); forced (build_settings::strategy depth), wherever such a split
// fits; never with forced radices for n or another strategy forced.
template <class T>
[[nodiscard]] std::vector<std::size_t> convolution_depth_tops(std::size_t n,
                                                              const build_settings& settings,
                                                              const chain_builder<T>& chain,
                                                              target isa);

// c's workspace layout from its scratch lines of `scratch` complex values of complex_bytes each
// and its block (c.block_elements); stagger_b: scratch line b starts seven 64-byte lines further
// (a grid's, X4); callback_scratch: the bytes of each slot's callback area (task 5.3a; 0: none,
// as for pipelines until 5.3d).
void lay_out_workspace(plan_core& c, std::size_t scratch, std::size_t complex_bytes, bool stagger_b,
                       std::size_t callback_scratch);

// make_pipeline_core (fftn/detail/core.hpp) with construction knobs (tests): the settings' chain
// rules (radices, leaf limit, layout, Bluestein and Rader) and profile apply to both chains.
[[nodiscard]] std::shared_ptr<const plan_core>
make_pipeline_core(const pipeline_view& desc, const options& opts, const build_settings& settings);

} // namespace fftn::inline v0::detail
