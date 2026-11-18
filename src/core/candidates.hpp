// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The candidate programs of measured planning (DESIGN.md 11.2, task 2.4): the rules' choice
// and the alternatives the rules decide between by thresholds or cost tables, each a few
// build_settings choices (the knobs the tests and fbench force), so that a candidate is
// built by the same code as the rules' plan and can be written down as text (wisdom).
#pragma once

#include "core/plan_core.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::inline v0::detail {

// One candidate: the build_settings fields it sets. The default is the rules' plan.
struct plan_choice {
  strategy_choice strategy = strategy_choice::rules;
  // The {} is not redundant: designated initializers that leave these out would otherwise
  // warn (-Wmissing-field-initializers, Clang and GCC).
  // NOLINTBEGIN(readability-redundant-member-init)
  std::vector<std::size_t> radices{};      // build_settings::radices
  std::vector<std::size_t> grid_factors{}; // build_settings::grid_factors
  // NOLINTEND(readability-redundant-member-init)
  std::size_t grid_phases = 0; // build_settings::grid_phases
  int rader = -1;              // build_settings::rader
  // NOLINTNEXTLINE(readability-redundant-member-init): as radices above
  std::vector<std::size_t> depth_tops{}; // build_settings::depth_tops (experiment X46)

  bool operator==(const plan_choice&) const = default;
};

// The choice as text: "rules" for the default, else space-separated fields among
// "strategy=chain|grid|lanes|depth", "radices=4.8.8", "grid=64.64.16", "phases=2|3",
// "rader=0|1", "tops=8.4" (one or two depth-first top factors), in that order. parse_choice reads
// it back (nullopt for text that is not a choice).
[[nodiscard]] std::string choice_text(const plan_choice& c);
[[nodiscard]] std::optional<plan_choice> parse_choice(std::string_view text);

// Sets the fields the choice decides (all of them: a test's forced radices do not survive).
void apply_choice(const plan_choice& c, build_settings& s);

// The choice to record for a plan built from c, in a canonical form that names only what
// took effect: a grid's factors and a depth-first chain's top factors written out (so that
// the record does not depend on the grid's phase rule or the depth-first default), and grid
// or depth-first fields without that strategy built, lanes not built, radices whose product
// is no line's length and a Rader setting without a prime factor to take it dropped.
[[nodiscard]] plan_choice pinned_choice(const plan_choice& c, const plan_core& built);

// The candidates for the shape of `rules_plan` (the plan built by the rules from `settings`),
// in the order measured planning times them, the rules' choice first:
// - one axis, rows longer than the L1 data cache (profile.grid_block_bytes) as complex lines
//   (c2c rows, the half-length transform of even real rows): the chain with the rules'
//   radices, the grid in two phases and, beyond profile.grid_min_bytes, in three (the
//   thresholds of X12b, X29, X32, X33 and X34); for powers of two FFTL.jl's long-row radices
//   in radix 8 and in radix 4 (X32);
// - batches of rows: the lanes strategy and the chain (X7);
// - every complex line length: the profile's next cheapest radix sequences (X2's tables), up
//   to `radix_choices` in all (fewer for lines beyond profile.grid_min_bytes);
// - a prime factor without a codelet: Rader stages off and on wherever q - 1 is 13-smooth.
// Many of them build the same program as another; measured planning drops those.
[[nodiscard]] std::vector<plan_choice> enumerate_candidates(const plan_core& rules_plan,
                                                            const build_settings& settings,
                                                            std::size_t radix_choices);

// A hash of the program a plan runs: its describe() text without the lines that say how the
// program was chosen. Two plans with the same fingerprint run the same stages on the same
// tables, so they give bitwise identical results.
[[nodiscard]] std::uint64_t program_fingerprint(const plan_core& c);

} // namespace fftn::inline v0::detail
