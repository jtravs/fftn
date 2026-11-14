// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Planner v1 (DESIGN.md 11.2): the radices of a chain and their order, the sequence of
// codelet radices with the lowest cost in a per-profile table of measured pass costs
// (experiment X2), plus a few measured exceptions. Deterministic and cheap: a shortest path
// over the divisors of the length, a pure function of the length and the profile. Phase 2's
// cost model generalises it (more pass kinds, memory levels, calibration per machine).
#pragma once

#include "isa/target.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fftn::inline v0::detail {

// Pass kinds the costs distinguish: the first pass (grouped, twiddled), a middle pass
// (lanes, twiddled) and the last pass (lanes, no twiddles).
inline constexpr std::size_t cost_kinds = 3;
inline constexpr std::size_t cost_radices = 15; // codelet radices 2 to 16

// A length whose radices are fixed because the costs miss its measured best.
struct radix_override {
  std::size_t n;
  std::array<std::size_t, 4> radices; // in pass order, 0 after the last
};

// One profile's pass costs, in arbitrary units: unit[kind][R - 2] per unit of C butterflies
// of radix R, tail[kind][R - 2] more per unit that is padded (a tail on a tile), and `pass`
// per pass.
struct pass_costs {
  std::array<std::array<std::uint32_t, cost_radices>, cost_kinds> unit;
  std::array<std::array<std::uint32_t, cost_radices>, cost_kinds> tail;
  std::uint32_t pass;
  std::span<const radix_override> overrides;
};

// What the planner reads of a target and precision.
struct target_profile {
  std::size_t lanes = 1;             // C: complex numbers per vector register
  const pass_costs* costs = nullptr; // never null in a profile from profile_of
};

// The profile of target t for a precision with `lanes` complex lanes. Only AVX2's tables
// are measured (X2 on an EPYC 7763): C = 2 (double) and C = 4 (float); every other target
// takes the table of the nearest C, the C = 2 table for C = 1 and the C = 4 table for
// C = 8 (unmeasured there).
[[nodiscard]] target_profile profile_of(target t, std::size_t lanes) noexcept;

// The radices of a chain for n, in pass order: empty for n = 1, {n} for n <= 16 (a leaf).
// Prime factors above 16 are generic passes, ascending, after the codelet passes; radices
// above 16 are such primes.
[[nodiscard]] std::vector<std::size_t> rule_radices(std::size_t n, const target_profile& p);

} // namespace fftn::inline v0::detail
