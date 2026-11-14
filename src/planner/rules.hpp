// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Rules planner v0 (DESIGN.md 11.2): the radices of a chain and their order, from fixed
// rules that DESIGN.md 11.2 specifies (taken from FFTL.jl, T. Knopp, MIT licence, which
// derived them from brute-force searches and measurements; this code implements the
// specification). Deterministic and cheap: a pure function of the length and of a few
// properties of the target. Phase 2's cost model replaces the choice
// (it enumerates candidate radix sequences and ranks them); these rules then remain one
// source of candidates and the fallback for unknown machines.
#pragma once

#include "isa/target.hpp"

#include <cstddef>
#include <vector>

namespace fftn::inline v0::detail {

// What the rules read of a target.
struct target_profile {
  std::size_t registers = 16; // vector registers: 16 (AVX2) or 32 (AArch64, AVX-512)
  bool merge = false;         // merged radices 6, 10 and 12 for single lines (AArch64)
};

// The profile of target t. The scalar target takes the build architecture's registers
// (32 floating-point registers on AArch64, 16 on x86-64) and merging rule.
[[nodiscard]] target_profile profile_of(target t) noexcept;

// The radices of a chain for n, in pass order: empty for n = 1, {n} for n <= 16 (a leaf)
// and for a prime n (one generic pass). Radices above 16 are primes, for the generic pass.
[[nodiscard]] std::vector<std::size_t> rule_radices(std::size_t n, const target_profile& p);

} // namespace fftn::inline v0::detail
