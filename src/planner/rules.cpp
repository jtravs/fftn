// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/rules.hpp"

#include "isa/arch.hpp"
#include "isa/target.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

// The largest radix that is a single codelet (src/gen/codelets.hpp); above it a line is
// split into passes.
constexpr std::size_t largest_leaf = 16;

bool power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

std::size_t log2_of(std::size_t n) {
  std::size_t k = 0;
  while ((std::size_t{1} << k) < n) {
    ++k;
  }
  return k;
}

// For n >= 2.
bool prime(std::size_t n) {
  for (std::size_t d = 2; d <= n / d; ++d) {
    if (n % d == 0) {
      return false;
    }
  }
  return true;
}

// Removes the factors p of n and returns how many there were.
std::size_t take(std::size_t& n, std::size_t p) {
  std::size_t e = 0;
  while (n % p == 0) {
    n /= p;
    ++e;
  }
  return e;
}

} // namespace

target_profile profile_of(target t) noexcept {
  switch (t) {
  case target::avx512:
  case target::neon:
  case target::neon_fcma:
    return {.registers = 32, .merge = t != target::avx512};
  case target::scalar:
    return {.registers = FFTN_ARCH_AARCH64 ? 32U : 16U, .merge = FFTN_ARCH_AARCH64 != 0};
  case target::x86_v2:
  case target::avx:
  case target::avx2:
    break;
  }
  return {.registers = 16, .merge = false};
}

// The specification is DESIGN.md 11.2, rule by rule; the numbers below are its rules.
std::vector<std::size_t> rule_radices(std::size_t n, const target_profile& p) {
  if (n <= 1) { // rule 1
    return {};
  }
  if (power_of_two(n)) {
    const std::size_t bits = log2_of(n);
    if (n > largest_leaf && n <= 64) { // rule 2
      return {4, n / 4};
    }
    if (p.registers >= 32 && n > 64 && n <= 16384) { // rule 3
      const std::size_t leading_fours = bits % 3 == 0 ? 0 : 3 - bits % 3;
      std::vector<std::size_t> out(leading_fours, 4);
      out.insert(out.end(), (bits - 2 * leading_fours) / 3, 8);
      return out;
    }
  }
  if (n <= largest_leaf || prime(n)) { // rule 4
    return {n};
  }
  // Rule 5.
  std::size_t rest = n;
  const std::size_t twos = take(rest, 2);
  std::size_t threes = take(rest, 3);
  std::size_t fives = take(rest, 5);
  std::size_t radix8 = 0;
  std::size_t radix2 = 0;
  if (twos % 2 == 1) {
    const bool eight = twos >= 3 && threes == 0 && fives == 0 && (p.merge || n % 7 != 0);
    (eight ? radix8 : radix2) = 1;
  }
  std::size_t radix4 = (twos - 3 * radix8 - radix2) / 2;
  std::vector<std::size_t> others;
  // Pairs as many of `factor` with as many of `with` as both have, into `merged`.
  const auto pair_up = [&](std::size_t& factor, std::size_t& with, std::size_t merged) {
    const std::size_t pairs = std::min(factor, with);
    factor -= pairs;
    with -= pairs;
    others.insert(others.end(), pairs, merged);
  };
  if (p.merge) {
    pair_up(threes, radix4, 12);
    pair_up(threes, radix2, 6);
    pair_up(fives, radix2, 10);
  }
  others.insert(others.end(), threes / 2, 9);
  others.insert(others.end(), threes % 2, 3);
  others.insert(others.end(), fives, 5);
  for (std::size_t q = 7; q <= rest / q; q += 2) {
    const std::size_t count = take(rest, q);
    others.insert(others.end(), count, q);
  }
  if (rest > 1) {
    others.push_back(rest);
  }
  std::ranges::sort(others);
  std::vector<std::size_t> out(radix8, 8);
  out.insert(out.end(), radix4, 4);
  out.insert(out.end(), radix2, 2);
  out.insert(out.end(), others.begin(), others.end());
  return out;
}

} // namespace fftn::inline v0::detail
