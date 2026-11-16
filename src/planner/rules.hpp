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
// Codelet radices the costs can price: 2 to 16, then 32 and 64 (experiment X3), in that
// order (cost_column).
inline constexpr std::size_t cost_radices = 17;
inline constexpr std::array<std::size_t, 2> big_cost_radices{32, 64};

// The column of radix r in a cost table, or cost_radices if none.
[[nodiscard]] constexpr std::size_t cost_column(std::size_t r) noexcept {
  if (r >= 2 && r <= 16) {
    return r - 2;
  }
  for (std::size_t i = 0; i < big_cost_radices.size(); ++i) {
    if (big_cost_radices[i] == r) {
      return 15 + i;
    }
  }
  return cost_radices;
}

// A length whose radices are fixed because the costs miss its measured best.
struct radix_override {
  std::size_t n;
  std::array<std::size_t, 5> radices; // in pass order, 0 after the last
};

// One profile's pass costs, in arbitrary units: unit[kind][cost_column(R)] per unit of C
// butterflies of radix R, tail[kind][cost_column(R)] more per unit that is padded (a tail on
// a tile), and `pass` per pass. A radix above 16 whose unit costs are 0 is not priced: the
// search never chooses it (only tables measured with it price 32 and 64).
struct pass_costs {
  std::array<std::array<std::uint32_t, cost_radices>, cost_kinds> unit;
  std::array<std::array<std::uint32_t, cost_radices>, cost_kinds> tail;
  std::uint32_t pass;
  std::span<const radix_override> overrides;
};

// The default leaf limit: every length up to it is one leaf (gen::radices has every radix
// 2 to 16).
inline constexpr std::size_t default_leaf_limit = 16;

// What the planner reads of a target and precision.
struct target_profile {
  std::size_t lanes = 1;             // C: complex numbers per vector register
  const pass_costs* costs = nullptr; // never null in a profile from profile_of
  // A length n <= leaf_limit with a codelet of radix n is one leaf, a whole transform in
  // registers (DESIGN.md 8.2), instead of a chain. Lengths up to 16 are leaves whatever the
  // value. From experiment X8 (dev/experiments/X8-leaves.md): 32 on NEON in double (C = 1),
  // 16 elsewhere.
  std::size_t leaf_limit = default_leaf_limit;
  // Prime factors above this go through one Bluestein stage at the end of the chain;
  // those from 17 up to it are generic passes (DESIGN.md 11.2).
  std::size_t bluestein_above = 0;
  // A chain's last lane passes run in split arithmetic on the BL layout (DESIGN.md 6.3,
  // strategies/chain.hpp) rather than on interleaved registers. Experiment X1 sets it per
  // target and precision.
  bool split = false;
  // The lanes strategy (DESIGN.md 8.5, strategies/lanes.cpp) runs a batch of rows of one
  // axis whose length is at most this, one row per lane, instead of row by row (0: never).
  // Experiment X7 sets it per target and precision.
  std::size_t lanes_up_to = 0;
  // A prime factor q without a codelet takes a Rader stage instead of a Bluestein stage when
  // rader_from <= q < rader_below and q - 1 is 13-smooth (rader_root). Experiment X27 sets
  // them per target and precision.
  std::size_t rader_from = 0;
  std::size_t rader_below = 0;
};

// The profile of target t for a precision with `lanes` complex lanes: the table measured
// for (t, lanes) if there is one, else AVX2's table of the nearest C. Only AVX2's tables
// are measured (X2 on an EPYC 7763): C = 2 (double) and C = 4 (float); so every other
// target takes the C = 2 table for C = 1 and the C = 4 table for C = 8 (unmeasured there).
[[nodiscard]] target_profile profile_of(target t, std::size_t lanes) noexcept;

// The radices of a chain for n, in pass order: empty for n = 1, {n} for n <= 16 and for a
// codelet radix n <= p.leaf_limit (a leaf).
// Prime factors above 16 follow the cost-priced passes, ascending: a codelet pass for 17, 19
// and 23 (experiment X27), otherwise a generic pass or Bluestein (chain_builder::radices);
// radices above 16 are such primes.
[[nodiscard]] std::vector<std::size_t> rule_radices(std::size_t n, const target_profile& p);

// The primitive root (the smallest) of the Rader stage that a prime factor q above the
// codelets takes instead of a Bluestein stage, or 0 for Bluestein: Rader when q lies in the
// profile's range and q - 1 is 13-smooth (experiment X27), so that its convolution of q - 1
// points is a chain of codelet passes and its kernel spectrum a transform of small primes.
[[nodiscard]] std::size_t rader_root(std::size_t q, const target_profile& p);

// The convolution length of a Bluestein stage for a transform of q points: the 7-smooth
// M >= 2q - 1, at most the next power of two, whose chain (rule_radices) plus the two
// chirp passes cost least; the smallest on a tie. Precondition: q >= 2.
[[nodiscard]] std::size_t bluestein_length(std::size_t q, const target_profile& p);

} // namespace fftn::inline v0::detail
