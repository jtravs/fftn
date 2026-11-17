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
#include "platform/aligned_allocator.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {

// Which strategy builds a c2c row (DESIGN.md 8.5): the planner's choice, or one forced.
enum class strategy_choice : std::uint8_t { rules, chain, grid, lanes };

// The arithmetic of a chain's lane passes (DESIGN.md 6.3): the profile's choice, or
// interleaved or split forced.
enum class layout_choice : std::uint8_t { rules, interleaved, split };

// Construction knobs for tests; plans built through the public API use the defaults. They
// force the paths the rules would not take for the lengths at hand (AGENTS.md section 5).
// The fields stay grouped by meaning, the order designated initializers follow; one is built
// per plan, so its padding costs nothing.
struct build_settings { // NOLINT(clang-analyzer-optin.performance.Padding)
  // About this many points of work per chunk of a stage (program.hpp) or of a run of
  // lines (work_graph.hpp). Experiment T5 (S4, runs 20261114-184234-4a75 and
  // 20261114-190017-fd3d) kept 2048: free at one thread, and best at 8 lanes, where smaller
  // chunks cut planes' lines too fine (512: 256^2 planes 2.7 times slower) and larger ones
  // gave rows of 8192-16384 points too few members (8192: 1.5-2.6).
  std::size_t chunk_points = 2048;
  // The radices, in pass order, of every complex line whose length they multiply to,
  // instead of the rules' choice (strategies/chain.hpp). Any radix above 16 goes through
  // the generic pass, except trailing radices above the Bluestein threshold, which form
  // one Bluestein stage.
  std::vector<std::size_t> radices;
  // The planner's leaf limit (planner/rules.hpp, target_profile::leaf_limit) instead of the
  // profile's; 0 keeps the profile's. Experiment X8 compares leaves with chains through it.
  std::size_t leaf_limit = 0;
  // Pair leaves (strategies/chain.hpp, use_pair_leaves; experiment X28b) where the target
  // has them; false: the chain of the rules' radices instead (tests).
  bool pair_leaves = true;
  // The strategy of a plan of one axis (rows): the rules' choice (the lanes for a batch of
  // rows up to the profile's lanes_up_to, a c2c grid beyond profile.grid_min_bytes, else
  // the chain), or the chain, the grid or the lanes forced. A grid is built only for a c2c
  // row with a split (strategies/grid.hpp) and no edge factor, the lanes only for a batch
  // (the innermost batch dimension's count at least 2) of rows every pass of whose chain is
  // a codelet's, without input edge operations, real rows contiguous on the real side
  // (strategies/lanes.cpp); elsewhere, forced or not, the row is a chain, and describe()
  // says which was built. Experiments X12 and X7 compare them.
  strategy_choice strategy = strategy_choice::rules;
  // The lanes strategy's block, in rows (rounded up to a multiple of the target's complex
  // count C), instead of the rule's when not 0 (plan_core.cpp, lane_block; experiment X7).
  std::size_t lane_block = 0;
  // The strided axes of a c2c plan of two or three axes (DESIGN.md 9.3, experiment X40):
  // true, the rules: lanes over blocks of adjacent columns wherever the last axis is
  // contiguous on both sides of the phase and the column's chain fits the lanes; false: every
  // column alone through the chain. column_block: the block in columns (rounded up to a
  // multiple of C) instead of the rule's (plan_core.cpp, column_block) when not 0.
  bool lanes_columns = true;
  std::size_t column_block = 0;
  // The grid's factors in phase order (two or three, multiplying to the row's length),
  // where they can be a grid; empty: the rules' (strategies/grid.hpp).
  std::vector<std::size_t> grid_factors;
  // Two or three grid phases instead of the rule's (grid_three_bytes, grid_two_odd_part) when
  // not 0; grid_factors take precedence. Measured planning (core/candidates.cpp) times both.
  std::size_t grid_phases = 0;
  // The grid's phases may read and write the arrays in their first and last passes (true:
  // where the rules say, rows up to profile.grid_fuse_bytes and phases of small radices),
  // or always go through a gather and a scatter around a chain on scratch (X12).
  bool grid_fused = true;
  // The largest radix the grid's phase rule uses (strategies/grid.hpp), instead of
  // default_grid_phase_radix when not 0. Experiment X3 prices radix 32 and 64 phases.
  std::size_t grid_phase_radix = 0;
  // Interleaved or split lane passes in every chain of the plan instead of the profile's
  // choice (target_profile::split, planner/rules.hpp); where a chain has no split run
  // (strategies/chain.hpp) it stays interleaved. Experiment X1 compares the two.
  layout_choice layout = layout_choice::rules;
  // The Bluestein threshold instead of the profile's when not 0 (strategies/chain.hpp):
  // 16 sends every prime factor above 16 through Bluestein, SIZE_MAX none.
  std::size_t bluestein_above = 0;
  // Rader stages (planner/rules.hpp, rader_root): -1 the profile's range, 0 never (Bluestein
  // instead), 1 for every prime factor whose q - 1 is 13-smooth (experiment X27's rule
  // without its range). Measured planning times both where they differ.
  int rader = -1;
  // The factor edge operation (DESIGN.md 12.1), which the public API does not offer yet:
  // n complex values of the plan's precision multiplying the input (before the transform)
  // or the output (after it) of every row of a c2c plan of one axis. The arrays must
  // outlive the plan.
  const void* in_factor = nullptr;
  const void* out_factor = nullptr;
  // Puts options::scale on the input edge instead of the output edge (c2c only).
  bool scale_on_input = false;
  // The largest radix of the pass the pairing of an even real line fuses into
  // (strategies/chain.hpp, DESIGN.md 9.1): 0 the rule (plan_core.cpp, fusion_of), 1 never
  // (a separate pass), 16 every codelet pass whatever the line's size.
  std::size_t fuse_real_radix = 0;
  // The r2c pairing in a grid's last phase (strategies/grid.hpp, pairs_last; experiment
  // X33): -1 the rule (plan_core.cpp, r2c_pairs_in_grid), 0 never (a separate pass after
  // the grid), 1 wherever the last phase allows.
  int r2c_pair_last = -1;
  // The machine the plan's team executions are scheduled for (schedule.hpp).
  machine_profile profile = default_machine_profile();
  // Measured planning (core/measure.cpp) asks this function for a candidate's seconds per
  // execution, given its choice_text, instead of timing it, when it is set (tests).
  double (*measure_time)(const std::string& choice, void* context) = nullptr;
  void* measure_context = nullptr;
};

// The fields stay grouped by meaning; one object per plan, so its padding costs nothing.
class plan_core { // NOLINT(clang-analyzer-optin.performance.Padding)
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
  plan_tables<double> tables_f64;
  plan_tables<float> tables_f32;
  std::vector<std::string> axis_notes; // one line per axis for describe()
  bool radices_forced = false;         // build_settings.radices was given (describe says so)
  std::string strategy_note;           // describe()'s strategy line
  // What the rows were built as (one axis only; multi-axis plans are chains): measured
  // planning enumerates the alternatives (core/candidates.cpp).
  strategy_choice built = strategy_choice::chain;
  std::vector<std::size_t> grid_factors; // a grid's factors as built, in phase order
  // The candidate the plan was built from, as text (core/candidates.hpp, choice_text):
  // "rules" for the rules' plan. Wisdom records it (core/wisdom.cpp).
  std::string program_text = "rules";
  // How the program was chosen, for describe() (empty: by the rules): measured planning or
  // a wisdom entry.
  std::string planning_note;
  // The bytes each side's elements span, relative to element 0 (strides may be negative):
  // [lo, hi). Measured planning allocates its scratch data from them.
  std::ptrdiff_t in_lo = 0;
  std::ptrdiff_t in_hi = 0;
  std::ptrdiff_t out_lo = 0;
  std::ptrdiff_t out_hi = 0;
  // The shape and options as wisdom keys them (core/wisdom_entry.hpp, shape_key).
  std::string shape_key;
  // Measured planning's account (0 for other plans): its time, the distinct candidates it
  // timed, the rules' choice's time over the chosen program's, and the chosen one's rank
  // (fbench reports them).
  double planning_seconds = 0;
  std::size_t candidates_timed = 0;
  double rules_over_chosen = 0;
  std::size_t chosen_rank = 0; // among the distinct candidates; 0: the rules' choice

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

// A plan as the options ask: built by the rules (build_plan_core), or through measured
// planning or a wisdom entry (plan_with_effort, core/measure.cpp).
[[nodiscard]] std::shared_ptr<const plan_core>
make_plan_core(const shape_view& shape, const options& opts, const build_settings& settings);

// The instruction set a plan with these options uses (options::force_isa, else the best
// the processor and the build support). Throws std::invalid_argument naming
// options.force_isa for a target that cannot be used.
[[nodiscard]] target plan_target_of(const options& opts);

// The plan the settings describe, ignoring options::effort and options::use.
[[nodiscard]] std::shared_ptr<plan_core>
build_plan_core(const shape_view& shape, const options& opts, const build_settings& settings);

// Measured planning and wisdom (core/measure.cpp): options::use, then options::effort.
[[nodiscard]] std::shared_ptr<const plan_core>
plan_with_effort(const shape_view& shape, const options& opts, const build_settings& settings);

} // namespace fftn::inline v0::detail
