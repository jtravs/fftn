// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The type-erased plan behind plan<In, Out> (include/fftn/detail/core.hpp): the validated
// layout, the program, the tables it points into, and the workspace layout.
#pragma once

#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"
#include "isa/target.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {

// Which strategy builds a c2c row (DESIGN.md 8.5): the planner's choice, or one forced.
enum class strategy_choice : std::uint8_t { rules, chain, grid };

// Construction knobs for tests; plans built through the public API use the defaults. They
// force the paths the rules would not take for the lengths at hand (AGENTS.md section 5).
struct build_settings {
  // About this many points of work per chunk of a stage (program.hpp) or of a run of
  // lines (work_graph.hpp). Experiment T5 sets the value; here it only has to give long
  // rows several chunks.
  std::size_t chunk_points = 2048;
  // The radices, in pass order, of every complex line whose length they multiply to,
  // instead of the rules' choice (strategies/chain.hpp). Any radix above 16 goes through
  // the generic pass, except trailing radices above the Bluestein threshold, which form
  // one Bluestein stage.
  std::vector<std::size_t> radices;
  // The strategy of a c2c plan of one axis (rows): the rules' choice (a grid beyond
  // profile.grid_min_bytes), or the chain or the grid forced. A grid is built only where
  // the row has a split (strategies/grid.hpp) and no edge factor; elsewhere, forced or not,
  // the row is a chain, and describe() says which was built. Experiment X12 compares them.
  strategy_choice strategy = strategy_choice::rules;
  // The grid's factors in phase order (two or three, multiplying to the row's length),
  // where they can be a grid; empty: the rules' (strategies/grid.hpp).
  std::vector<std::size_t> grid_factors;
  // The grid's phases may read and write the arrays in their first and last passes (true:
  // where the rules say, rows up to profile.grid_fuse_bytes and phases of small radices),
  // or always go through a gather and a scatter around a chain on scratch (X12).
  bool grid_fused = true;
  // The Bluestein threshold instead of the profile's when not 0 (strategies/chain.hpp):
  // 16 sends every prime factor above 16 through Bluestein, SIZE_MAX none.
  std::size_t bluestein_above = 0;
  // The factor edge operation (DESIGN.md 12.1), which the public API does not offer yet:
  // n complex values of the plan's precision multiplying the input (before the transform)
  // or the output (after it) of every row of a c2c plan of one axis. The arrays must
  // outlive the plan.
  const void* in_factor = nullptr;
  const void* out_factor = nullptr;
  // Puts options::scale on the input edge instead of the output edge (c2c only).
  bool scale_on_input = false;
  // The machine the plan's team executions are scheduled for (schedule.hpp).
  machine_profile profile = default_machine_profile();
};

class plan_core {
public:
  plan_core() = default;
  // Stages point into the tables, so a core never moves or copies.
  plan_core(const plan_core&) = delete;
  plan_core& operator=(const plan_core&) = delete;
  plan_core(plan_core&&) = delete;
  plan_core& operator=(plan_core&&) = delete;
  ~plan_core() = default;

  transform_kind kind = transform_kind::c2c;
  precision prec = precision::f64;
  direction dir = direction::forward;
  bool in_place = false;
  std::vector<std::size_t> lengths; // per axis
  std::vector<std::size_t> counts;  // per batch dimension
  target isa_target = target::scalar;

  program prog;
  work_graph graph;         // the program's shareable work (DESIGN.md 13.2)
  machine_profile profile;  // what the scheduler assumes of the machine
  bool runs_inline = false; // rule 1 for the whole job (schedule.hpp): serial on any team
  std::vector<std::vector<double>> tables_f64;
  std::vector<std::vector<float>> tables_f32;
  std::vector<std::string> axis_notes; // one line per axis for describe()
  bool radices_forced = false;         // build_settings.radices was given (describe says so)
  std::string strategy_note;           // describe()'s strategy line

  // Workspace: scratch lines a and b of scratch_line complex values each, then the block,
  // each starting on a 64-byte boundary of the aligned workspace; bytes includes the slack
  // for aligning an arbitrary workspace address.
  std::size_t scratch_line = 0;
  std::size_t block_elements = 0;
  std::size_t offset_b = 0;
  std::size_t offset_block = 0;
  std::size_t bytes = 0;
  // One lane's scratch in a team's workspace: the layout above, in whole 128-byte lines.
  std::size_t slot_bytes = 0;
};

[[nodiscard]] std::shared_ptr<const plan_core>
make_plan_core(const shape_view& shape, const options& opts, const build_settings& settings);

} // namespace fftn::inline v0::detail
