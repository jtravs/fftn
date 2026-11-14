// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/rules.hpp"

#include "isa/target.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

// The largest radix that is a single codelet (src/gen/codelets.hpp); above it a line is
// split into passes, and a prime factor above it is a generic pass.
constexpr std::size_t largest_leaf = 16;

// Experiment X2 (dev/experiments/X2-radix-sets.md), EPYC 7763, AVX2, GCC 16, one thread: the
// pass costs fitted to the screening and refinement timings of every candidate sequence of
// the 7-smooth lengths up to 1024, batches 1 and 64 together. No such length has 11 or 13,
// so their columns are radix 10's and 12's scaled by the codelets' operation counts.
// The overrides are the lengths where the costs' choice was more than 2 % slower than
// planner v0 in the refinement; each is the measured winner, which no batch of that
// precision measured slower than v0.
constexpr std::array<radix_override, 4> avx2_double_overrides{{
    {.n = 42, .radices = {2, 3, 7, 0}},
    {.n = 70, .radices = {2, 5, 7, 0}},
    {.n = 350, .radices = {2, 5, 5, 7}},
    {.n = 490, .radices = {2, 7, 5, 7}},
}};
constexpr std::array<radix_override, 2> avx2_float_overrides{{
    {.n = 140, .radices = {4, 5, 7, 0}},
    {.n = 700, .radices = {4, 7, 5, 5}},
}};

constexpr pass_costs avx2_double{
    .unit = {{
        {28, 195, 114, 305, 235, 460, 387, 665, 557, 830, 684, 1238, 995, 1231, 1273},
        {88, 147, 190, 286, 324, 468, 486, 696, 658, 982, 758, 1371, 1096, 1200, 1255},
        {46, 58, 79, 116, 143, 236, 221, 364, 323, 504, 326, 639, 636, 667, 652},
    }},
    .tail = {{
        {391, 748, 914, 1102, 1467, 1552, 1879, 2354, 2535, 3780, 3073, 5560, 3464, 3140, 3836},
        {377, 536, 702, 884, 1061, 1348, 1354, 1733, 1749, 2608, 2073, 3752, 2522, 2810, 3152},
        {19, 1635, 518, 2086, 859, 2215, 1308, 3223, 1804, 2819, 2381, 4671, 2853, 3738, 3002},
    }},
    .pass = 436,
    .overrides = avx2_double_overrides,
};

constexpr pass_costs avx2_float{
    .unit = {{
        {28, 197, 76, 132, 205, 452, 298, 534, 538, 803, 525, 950, 856, 1070, 924},
        {101, 146, 190, 293, 343, 462, 525, 781, 700, 1044, 827, 1496, 1159, 1311, 1459},
        {42, 59, 91, 116, 138, 223, 213, 389, 293, 458, 367, 720, 622, 570, 674},
    }},
    .tail = {{
        {644, 628, 1125, 1110, 1612, 1655, 2285, 2601, 3009, 4487, 3539, 6405, 4310, 4163, 4957},
        {540, 855, 1057, 1304, 1659, 1840, 2129, 2511, 2738, 4082, 3295, 5963, 3767, 4056, 4186},
        {282, 304, 951, 925, 1575, 1680, 2291, 2497, 3013, 4708, 3797, 7448, 4305, 4096, 4695},
    }},
    .pass = 406,
    .overrides = avx2_float_overrides,
};

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// The cost of the pass of radix r after the passes whose radices multiply to s, in a line
// of n points (DESIGN.md 11.2). Costs stay far below 2^64: at most about 2^13 per point
// per pass.
std::uint64_t pass_cost(const target_profile& p, std::size_t n, std::size_t s, std::size_t r,
                        std::size_t kind) {
  const std::size_t c = p.lanes;
  const std::size_t m = n / (r * s);
  // The first pass runs over m in blocks of C, the others over s, once per p < m.
  const std::size_t units = kind == 0 ? ceil_div(m, c) : m * ceil_div(s, c);
  const std::size_t tails = kind == 0 ? (m % c != 0 ? 1 : 0) : (s % c != 0 ? m : 0);
  return std::uint64_t{p.costs->unit[kind][r - 2]} * units +
         std::uint64_t{p.costs->tail[kind][r - 2]} * tails + p.costs->pass;
}

} // namespace

target_profile profile_of(target t, std::size_t lanes) noexcept {
  (void)t; // one table per C for now; a target measured on its own gets its own tables
  return {.lanes = lanes, .costs = lanes <= 2 ? &avx2_double : &avx2_float};
}

// The specification is DESIGN.md 11.2, rule by rule.
std::vector<std::size_t> rule_radices(std::size_t n, const target_profile& p) {
  if (n <= 1) {
    return {};
  }
  if (n <= largest_leaf) {
    return {n};
  }
  for (const radix_override& o : p.costs->overrides) {
    if (o.n == n) {
      std::vector<std::size_t> out;
      for (const std::size_t r : o.radices) {
        if (r != 0) {
          out.push_back(r);
        }
      }
      return out;
    }
  }
  // n = smooth x the prime factors above 16, which become generic passes at the end.
  std::size_t smooth = 1;
  std::size_t rest = n;
  std::vector<std::size_t> big;
  for (std::size_t q = 2; q <= rest / q; ++q) {
    while (rest % q == 0) {
      if (q <= largest_leaf) {
        smooth *= q;
      } else {
        big.push_back(q);
      }
      rest /= q;
    }
  }
  if (rest > 1) {
    if (rest <= largest_leaf) {
      smooth *= rest;
    } else {
      big.push_back(rest);
    }
  }
  std::ranges::sort(big);
  if (smooth == 1) {
    return big;
  }
  // The divisors of `smooth`, ascending: every prefix product of a sequence is one.
  std::vector<std::size_t> divisors{1};
  {
    std::size_t left = smooth;
    for (const std::size_t q : {2U, 3U, 5U, 7U, 11U, 13U}) {
      const std::size_t count = divisors.size();
      std::size_t power = 1;
      while (left % q == 0) {
        left /= q;
        power *= q;
        for (std::size_t i = 0; i < count; ++i) {
          divisors.push_back(divisors[i] * power);
        }
      }
    }
  }
  std::ranges::sort(divisors);
  const auto index = [&](std::size_t d) {
    return static_cast<std::size_t>(std::ranges::lower_bound(divisors, d) - divisors.begin());
  };
  const auto kind = [&](std::size_t s, std::size_t next) -> std::size_t {
    if (s == 1) {
      return 0;
    }
    return next == smooth && big.empty() ? 2 : 1;
  };
  // best[i]: the lowest cost of the passes from prefix product divisors[i] to the end.
  std::vector<std::uint64_t> best(divisors.size(), 0);
  for (std::size_t i = divisors.size() - 1; i-- > 0;) {
    const std::size_t s = divisors[i];
    bool found = false;
    for (std::size_t r = 2; r <= largest_leaf; ++r) {
      if ((smooth / s) % r != 0) {
        continue;
      }
      const std::uint64_t c = pass_cost(p, n, s, r, kind(s, s * r)) + best[index(s * r)];
      if (!found || c < best[i]) {
        best[i] = c;
        found = true;
      }
    }
  }
  // Ties go to the smallest radix at each step, so the sequence is the lexicographically
  // first of the cheapest.
  std::vector<std::size_t> out;
  for (std::size_t s = 1; s != smooth;) {
    for (std::size_t r = 2; r <= largest_leaf; ++r) {
      if ((smooth / s) % r == 0 &&
          pass_cost(p, n, s, r, kind(s, s * r)) + best[index(s * r)] == best[index(s)]) {
        out.push_back(r);
        s *= r;
        break;
      }
    }
  }
  out.insert(out.end(), big.begin(), big.end());
  return out;
}

} // namespace fftn::inline v0::detail
