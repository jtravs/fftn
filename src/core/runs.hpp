// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Stored runs (shape::runs, DESIGN.md 12.1; task 5.1d): their validation, the stored extents
// that default strides and the layout checks use, and the text describe() and wisdom give
// them. A plan with runs is built as a plan with edges (core/edges.hpp): the runs of a side
// make its edge separate, and its pass (an edge phase, or a c2c row's pass through the
// workspace block) reads the stored runs into the full logical line with zeros elsewhere
// (input), or stores only the runs (output). The transform between them computes the whole
// line, except where a c2c row's first or last pass takes the pruned codelets (task 5.1e,
// run_pruning).
#pragma once

#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {

// One stored run of an axis: stored positions [stored, stored + count) hold logical positions
// [logical, logical + count).
struct line_run {
  std::size_t logical = 0;
  std::size_t stored = 0;
  std::size_t count = 0;
};

// The runs of one side, per axis in the shape's order, each axis's sorted by logical position
// (empty for an axis stored whole); `axes` is empty when the side has no runs at all.
struct side_runs {
  std::vector<std::vector<line_run>> axes;

  [[nodiscard]] bool any() const noexcept { return !axes.empty(); }
  // The runs of axis d (the shape's order), empty when it is stored whole.
  [[nodiscard]] const std::vector<line_run>& of(std::size_t d) const;
  // The axis's stored extent: the end of its last stored run, or `whole` without runs.
  [[nodiscard]] std::size_t stored_extent(std::size_t d, std::size_t whole) const;
  // The logical positions its runs hold, of an axis with runs.
  [[nodiscard]] std::size_t positions(std::size_t d) const;
};

struct run_plan {
  side_runs in;
  side_runs out;

  [[nodiscard]] bool any() const noexcept { return in.any() || out.any(); }
};

// Validates shape.runs against the shape and options, throwing std::invalid_argument naming the
// offending run and field (shape.runs[2].logical, ...), and sorts them per axis and side.
[[nodiscard]] run_plan resolve_runs(const shape_view& s, const options& opts);

// The runs as a wisdom key adds them (core/wisdom.cpp, shape_key): empty without runs.
[[nodiscard]] std::string runs_key(const shape_view& s);

// How a c2c row's chain takes one side's stored runs (task 5.1e): whole, between the side's
// separate pass and the transform (none); by the chain's first or last pass on the pruned
// codelets reading or writing the line's first half, n/2 elements, where the one run of n/2
// is stored (direct: no separate pass); or by those passes through the workspace block, the
// separate pass then moving the first half only (half).
enum class run_pruning : std::uint8_t { none, direct, half };

// describe()'s line for the runs: each side's axes with runs (the shape's indices), where the
// input's are read to (`in_to`: "the output", "the workspace block"), whether the output's are
// accumulated, how each side is pruned, and the `passes` they cost.
[[nodiscard]] std::string runs_text(const run_plan& runs, bool accumulate, const char* in_to,
                                    std::size_t passes, run_pruning in = run_pruning::none,
                                    run_pruning out = run_pruning::none);

} // namespace fftn::inline v0::detail
