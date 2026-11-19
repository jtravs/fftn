// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The work graph (DESIGN.md 13.2): what a plan's program offers to share among threads,
// read off the program at plan construction. It depends on the plan alone, never on a team.
//
// The tasks are the transforms of the batch. Each runs whole, as serial execution runs it,
// or split into phases that run one after another, each phase made of chunks that may run
// on different threads at once:
//
// - a stage of the transform's body (a pass over the one line of a 1-D transform) is a
//   phase whose chunks are the stage's own (program.hpp); the chunks share the task's
//   scratch lines a and b;
// - a loop nest over the lines of one axis (2-D and 3-D transforms) is a phase whose
//   chunks are runs of whole lines; each line runs all its stages on the scratch lines of
//   the thread that runs it, and the task's block (if any) is shared.
//
// Either way a chunk runs exactly the code serial execution runs for it, on the same
// elements, so the output does not depend on who runs which chunk (DESIGN.md 15.5).
#pragma once

#include "core/program.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fftn::inline v0::detail {

struct graph_phase {
  std::uint32_t record = 0;   // the stage, or the outermost loop of the nest
  std::uint32_t nest = 0;     // loops in the nest; 0 for a stage
  std::uint32_t body_end = 0; // lines: one past the nest's last record
  std::size_t units = 0;      // the stage's units, or the nest's lines
  std::uint32_t chunks = 1;
  std::uint64_t first = 0; // the index of the phase's first chunk among the task's chunks
  double work = 0;         // the phase's work in the cost model's units (schedule.hpp)
  // A second nest whose lines follow the first's (loop::joins; X41): lines [0, head) of the
  // phase are the first nest's, [head, units) this one's. tail_nest 0: none.
  std::uint32_t tail_record = 0;
  std::uint32_t tail_nest = 0;
  std::uint32_t tail_body_end = 0;
  std::size_t head = 0;
  // Neighbouring lines write the same cache lines (loop::shares_lines): claimed in fair runs.
  bool fair_runs = false;
};

// The largest fair share (chunks per member) that goes in fair runs (chunk_run). X59: phases of
// more chunks per member need the guided runs' shrinking tail: 2048^2, 8 blocks per member at 8
// lanes, took 1.05-1.15 of the guided time with fair runs, capped at 4 chunks or not.
inline constexpr std::uint32_t max_fair_run = 3;

// How many chunks of phase ph a member claims at once, `left` of them being unclaimed: a phase
// of few chunks per member whose neighbouring lines share cache lines goes in runs of its fair
// share (chunks / cap), so that few runs border each other (X59); any other in a share of what
// is left, left / runs (X52), at least one. Never more than `left`: a run stays within its phase.
[[nodiscard]] inline std::uint32_t chunk_run(const graph_phase& ph, std::uint32_t left,
                                             std::uint32_t cap, std::uint32_t runs) noexcept {
  if (const std::uint32_t fair = ph.chunks / cap; ph.fair_runs && fair <= max_fair_run) {
    return std::min(left, std::max(fair, 1U));
  }
  return std::max(left / runs, 1U);
}

struct work_graph {
  std::vector<graph_phase> phases;
  std::size_t tasks = 1;
  std::uint64_t chunks = 0; // per task, over all phases
  std::uint32_t widest = 1; // the most chunks in one phase
  double work = 0;          // of one task
  std::size_t points = 1;   // elements of one transform: what the split thresholds compare
};

// The work graph of p. points is the transform's size (the product of its lengths); a
// phase of lines gets runs of about chunk_points points of line per chunk.
[[nodiscard]] work_graph build_work_graph(const program& p, std::size_t points,
                                          std::size_t chunk_points);

} // namespace fftn::inline v0::detail
