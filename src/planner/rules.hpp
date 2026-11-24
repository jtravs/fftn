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
#include "platform/cpu_features.hpp"

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
  // A c2c batch of at least batch_rows[codelet_slot(n)] contiguous rows of a codelet length n
  // (one axis, one batch dimension) runs as batch leaves, C rows per register
  // (passes/batch_leaf.inl); 0, or no entry: never. batch_narrower: the leaves of the narrower
  // target that rounds alike (avx2's, two rows per register in double, in an avx512 plan).
  // Experiment X53 sets them per target and precision.
  std::span<const std::uint16_t> batch_rows;
  bool batch_narrower = false;
  // Where batch_narrower: from a batch of batch_own_rows[codelet_slot(n)] rows (0, or no
  // entry: never) the target's own leaves instead, C rows per register (X53: avx512's four
  // rows per register won from four rows at 11 and 13-25 points).
  std::span<const std::uint16_t> batch_own_rows;
  // The most rows of a batch that still takes batch leaves, batch_max_rows[codelet_slot(n)]
  // (0, or no entry: no limit). Experiment X62: avx2's leaf of 64 in double lost from some
  // thousands of rows.
  std::span<const std::uint16_t> batch_max_rows;
  // A batch whose row count C does not divide takes batch leaves only from
  // batch_ragged_rows[codelet_slot(n)] rows (0, or no entry: as any other batch). Experiment
  // X62: avx2's leaf of 64 runs the last row of an odd batch as a whole pair, which three rows
  // did not repay.
  std::span<const std::uint16_t> batch_ragged_rows;
  // Lengths whose radices depend on whether a plan's rows fit the cache (fitting_radices;
  // experiment X57).
  std::span<const radix_override> fit_overrides;
};

// The profile of target t for a precision with `lanes` complex lanes: the table measured
// for (t, lanes) if there is one, else AVX2's table of the nearest C. Only AVX2's tables
// are measured (X2 on an EPYC 7763): C = 2 (double) and C = 4 (float); so every other
// target takes the C = 2 table for C = 1 and the C = 4 table for C = 8 (unmeasured there).
// vendor: the processor's maker (machine_profile::vendor). On `avx2` the rules measured only
// on AMD's cores (the EPYC 7763) apply only for amd; for any other vendor, unknown included,
// `avx2` keeps the plans it had before them (DESIGN.md 13.5; the maintainer, 2026-11-24): the
// radix overrides of X30 (512), X42 (500-25000) and X54 (768; float 512-4096) give way to
// what was planned before, and there are no batch leaves (X53, X62). Every other target, and
// every field the vendor does not change, is the same for all vendors. The default, unknown,
// keeps those earlier plans, as profile_for's does: a caller planning for a machine passes its
// vendor (machine_profile::vendor), and amd gives the tables as measured on the EPYC.
[[nodiscard]] target_profile profile_of(target t, std::size_t lanes,
                                        cpu_vendor vendor = cpu_vendor::unknown) noexcept;

// The radices of a chain for n, in pass order: empty for n = 1, {n} for n <= 16 and for a
// codelet radix n <= p.leaf_limit (a leaf).
// Prime factors above 16 follow the cost-priced passes, ascending: a codelet pass for 17, 19
// and 23 (experiment X27), otherwise a generic pass or Bluestein (chain_builder::radices);
// radices above 16 are such primes. overrides: false skips the profile's measured overrides
// (the costs' sequence; chain_builder::real_radices, experiment X39).
[[nodiscard]] std::vector<std::size_t> rule_radices(std::size_t n, const target_profile& p,
                                                    bool overrides = true);

// The radices of the rows of a one-axis c2c plan of n points where they depend on the plan's
// data (experiment X57), in pass order, or empty where they do not (rule_radices then
// decides): for a length of the profile's fit_overrides, its two-pass sequence while the
// plan's data fit (machine_profile::fit_row_bytes, core/plan_core.cpp) and otherwise the
// costs' sequence without the overrides. A pure function like rule_radices.
[[nodiscard]] std::vector<std::size_t> fitting_radices(std::size_t n, const target_profile& p,
                                                       bool fits);

// Up to k radix sequences for n, in pass order: rule_radices' first, then the cheapest
// others by the profile's pass costs (cost, then lexicographic order), without repeats.
// Measured planning (core/candidates.cpp) times them; a pure function like rule_radices.
[[nodiscard]] std::vector<std::vector<std::size_t>>
ranked_radices(std::size_t n, const target_profile& p, std::size_t k);

// The product of radices (or grid factors), or 0 if it overflows std::size_t: a forced
// sequence matches a length only when its product equals it, and a forged one (a wisdom
// entry, a test knob) must not wrap around to a small length.
[[nodiscard]] constexpr std::size_t checked_product(std::span<const std::size_t> factors) noexcept {
  std::size_t product = 1;
  for (const std::size_t f : factors) {
    if (f != 0 && product > SIZE_MAX / f) {
      return 0;
    }
    product *= f;
  }
  return product;
}

// The planner's estimate of one row of n points on profile p, in the cost tables' units: the
// rules' radices priced pass by pass (pass_cost), a 17, 19 or 23 codelet pass as a radix-16
// pass scaled by r log2 r, a Rader or Bluestein stage as its convolution's two transforms
// and two elementwise passes per copy. An in-cache estimate: memory beyond L1 is not priced
// (experiment P3). Precondition: n <= 2^40.
[[nodiscard]] double row_cost(std::size_t n, const target_profile& p);

// row_cost per point of n over row_cost per point of the power of two at or above n: about
// 1 for lengths as cheap per point as a power of two, more for slower ones; 1 for n <= 1 and
// for a length the planner runs as one codelet (rule_radices(n, p) == {n} with a codelet: n
// <= 16, 17, 19, 23, and 32 or 64 within the leaf limit; not 20, which is always a chain).
// Precondition: n <= max_length_query.
[[nodiscard]] double relative_row_cost(std::size_t n, const target_profile& p);

// Task 2.5: a fast length is one whose relative_row_cost on the reference profile is at most
// fast_tolerance. The reference is a frozen copy of AVX2's double table (X2's measured one;
// rules.cpp, reference_costs), not the planner's tuned table, so that a length chosen on one
// machine is the same on another and a refit never moves the fast set (DESIGN.md 9.2).
// Lengths with a prime factor above 23, and composites with a factor 11, 17, 19 or 23
// (experiment P4), never are. The tolerance was checked by measurement (P4): it is a
// property of the table, not a guarantee on every machine.
inline constexpr double fast_tolerance = 1.25;
// The longest length the length queries accept (fftn/lengths.hpp throws above it): row_cost
// is exact up to here, and next_fast_length's search stays within [n, 2n].
inline constexpr std::size_t max_length_query = std::size_t{1} << 40U;
[[nodiscard]] bool is_fast_length(std::size_t n);
// The smallest fast m >= n; 1 for n = 0. With real: the smallest even m >= n with m / 2 fast,
// so 2 for n <= 2. A pure function of the library version. It tests only the candidates in
// [n, 2n] (products of 2, 3, 5, 7 and 13, and the lone 11 to 23). Precondition: n <=
// max_length_query.
[[nodiscard]] std::size_t next_fast_length(std::size_t n, bool real);

// FFTL.jl's radices for a long line of 2^k points (its src/factor.jl; MIT, the maintainer's
// own; experiment X32): radix 8 with one or two 4s first, or radix 4 with an 8 first for an
// odd k. k >= 2.
[[nodiscard]] std::vector<std::size_t> long_pow2_radices(std::size_t k, bool radix8);

// The primitive root (the smallest) of the Rader stage that a prime factor q above the
// codelets takes instead of a Bluestein stage, or 0 for Bluestein: Rader when q lies in the
// profile's range and q - 1 is 13-smooth (experiment X27), so that its convolution of q - 1
// points is a chain of codelet passes and its kernel spectrum a transform of small primes.
[[nodiscard]] std::size_t rader_root(std::size_t q, const target_profile& p);

// The convolution length of a Bluestein stage for a transform of q points: the 7-smooth
// M >= 2q - 1 whose chain (rule_radices) plus the two chirp passes cost least, the smallest
// on a tie; M is at most the next power of two, or at most 2 (2q - 1) when that power of
// two is above pow2_part_limit, and its power-of-two part is at most pow2_part_limit
// (experiment X36: chains of a long power of two, or of a length with a large power-of-two
// part, ran 1.4-2 times as long per point as those of a 7-smooth length nearby, legs a power
// of two apart conflicting in cache, which the cost table does not see). Precondition:
// q >= 2.
[[nodiscard]] std::size_t bluestein_length(std::size_t q, const target_profile& p,
                                           std::size_t pow2_part_limit = SIZE_MAX);

} // namespace fftn::inline v0::detail
