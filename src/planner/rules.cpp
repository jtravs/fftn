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

// NEON with FCMA, fitted by bench/x2/fit.py to X2's campaign on an Apple M5 Max
// (campaign mac-smooth, 7-smooth lengths up to 1024, batches 1 and 64).
// fit.py --name mac-smooth: ISA neon_fcma, 532 cases
// neon_fcma_double: C = 1, constant per transform 0
//   n = 56: 4.14 +4.1 % against the baseline; override 8.7
//   n = 63: 9.7 +3.9 % against the baseline; override 7.9
//   n = 200: 5.4.10 +2.6 % against the baseline; override 4.5.10
//   n = 252: 6.6.7 +2.5 % against the baseline; override 3.6.14
//   n = 288: 6.4.12 +2.6 % against the baseline; override 4.6.12
//   n = 336: 6.4.14 +2.9 % against the baseline; override 8.6.7
//   n = 420: 6.5.14 +9.2 % against the baseline; override 5.6.14
//   n = 480: 6.8.10 +3.1 % against the baseline; override 4.12.10
//   n = 504: 6.6.14 +5.5 % against the baseline; override 12.6.7
//   n = 560: 5.8.14 +15.6 % against the baseline; override 4.10.14
//   n = 567: 9.9.7 +4.3 % against the baseline; override 7.9.9
//   n = 576: 6.8.12 +3.2 % against the baseline; override 4.12.12
//   n = 588: 6.14.7 +2.2 % against the baseline; override 3.14.14
//   n = 720: 6.10.12 +5.6 % against the baseline; override 5.12.12
//   n = 756: 6.9.14 +4.1 % against the baseline; override 14.6.9
// neon_fcma_double: 202 of 248 cases reproduced (81.5 %; b1-op 98/124, b64-op 104/124)
//   miss c2c-35-b1-op: 5.7 4.0 % behind 7.5
//   miss c2c-36-b1-op: 3.12 2.2 % behind 6.6
//   miss c2c-48-b1-op: 4.12 2.9 % behind 6.8
//   miss c2c-98-b1-op: 14.7 3.7 % behind 7.14
//   miss c2c-98-b64-op: 14.7 3.4 % behind 7.14
//   miss c2c-162-b64-op: 3.6.9 2.1 % behind 6.3.9
//   miss c2c-210-b64-op: 14.15 2.6 % behind 15.14
//   miss c2c-280-b1-op: 5.4.14 3.4 % behind 8.5.7
//   miss c2c-280-b64-op: 5.4.14 4.6 % behind 4.10.7
//   miss c2c-315-b1-op: 3.7.15 3.1 % behind 3.15.7
//   ... 36 more (--verbose)
// rules.cpp avx2_double on these cases: 120 of 248 cases reproduced (48.4 %; b1-op 62/124, b64-op 58/124)
//   miss c2c-18-b1-op: 2.9 7.4 % behind 3.6
//   miss c2c-18-b64-op: 2.9 3.8 % behind 3.6
//   miss c2c-20-b1-op: 2.10 2.2 % behind 4.5
//   miss c2c-21-b1-op: 7.3 6.9 % behind 3.7
//   miss c2c-21-b64-op: 7.3 4.5 % behind 3.7
//   miss c2c-24-b1-op: 2.12 2.7 % behind 4.6
//   miss c2c-24-b64-op: 2.12 2.6 % behind 4.6
//   miss c2c-27-b1-op: 9.3 4.8 % behind 3.9
//   miss c2c-27-b64-op: 9.3 8.5 % behind 3.9
//   miss c2c-28-b1-op: 2.14 3.0 % behind 4.7
//   ... 118 more (--verbose)
//   same choice as avx2_double for 46 of 124 lengths
// neon_fcma_float: C = 2, constant per transform 407
//   n = 20: 2.10 +2.2 % against the baseline; override 4.5
//   n = 21: 7.3 +2.2 % against the baseline; override 3.7
//   n = 28: 2.14 +8.0 % against the baseline; override 4.7
//   n = 45: 3.15 +2.3 % against the baseline; override 5.9
//   n = 70: 10.7 +3.1 % against the baseline; override 7.10
//   n = 280: 4.5.14 +3.8 % against the baseline; override 4.7.10
//   n = 480: 4.12.10 +5.1 % against the baseline; override 4.10.12
// neon_fcma_float: 206 of 248 cases reproduced (83.1 %; b1-op-f32 101/124, b64-op-f32 105/124)
//   miss c2c-18-b64-op-f32: 6.3 7.6 % behind 2.9
//   miss c2c-20-b64-op-f32: 4.5 4.9 % behind 2.10
//   miss c2c-24-b1-op-f32: 2.12 12.4 % behind 4.6
//   miss c2c-24-b64-op-f32: 2.12 5.4 % behind 4.6
//   miss c2c-45-b1-op-f32: 5.9 6.0 % behind 3.15
//   miss c2c-70-b1-op-f32: 7.10 8.7 % behind 10.7
//   miss c2c-98-b1-op-f32: 14.7 3.6 % behind 7.14
//   miss c2c-98-b64-op-f32: 14.7 6.4 % behind 7.14
//   miss c2c-240-b1-op-f32: 15.16 9.0 % behind 16.15
//   miss c2c-245-b1-op-f32: 7.5.7 2.0 % behind 7.7.5
//   ... 32 more (--verbose)
// rules.cpp avx2_float on these cases: 160 of 248 cases reproduced (64.5 %; b1-op-f32 79/124, b64-op-f32 81/124)
//   miss c2c-18-b1-op-f32: 9.2 20.8 % behind 6.3
//   miss c2c-18-b64-op-f32: 9.2 21.2 % behind 2.9
//   miss c2c-20-b1-op-f32: 2.10 2.2 % behind 4.5
//   miss c2c-21-b64-op-f32: 7.3 2.2 % behind 3.7
//   miss c2c-28-b1-op-f32: 2.14 8.0 % behind 4.7
//   miss c2c-28-b64-op-f32: 2.14 5.6 % behind 4.7
//   miss c2c-30-b1-op-f32: 2.15 8.5 % behind 6.5
//   miss c2c-30-b64-op-f32: 2.15 3.9 % behind 6.5
//   miss c2c-35-b1-op-f32: 5.7 3.1 % behind 7.5
//   miss c2c-35-b64-op-f32: 5.7 5.3 % behind 7.5
//   ... 78 more (--verbose)
//   same choice as avx2_float for 68 of 124 lengths

constexpr std::array<radix_override, 15> neon_fcma_double_overrides{{
    {.n = 56, .radices = {8, 7, 0, 0}},
    {.n = 63, .radices = {7, 9, 0, 0}},
    {.n = 200, .radices = {4, 5, 10, 0}},
    {.n = 252, .radices = {3, 6, 14, 0}},
    {.n = 288, .radices = {4, 6, 12, 0}},
    {.n = 336, .radices = {8, 6, 7, 0}},
    {.n = 420, .radices = {5, 6, 14, 0}},
    {.n = 480, .radices = {4, 12, 10, 0}},
    {.n = 504, .radices = {12, 6, 7, 0}},
    {.n = 560, .radices = {4, 10, 14, 0}},
    {.n = 567, .radices = {7, 9, 9, 0}},
    {.n = 576, .radices = {4, 12, 12, 0}},
    {.n = 588, .radices = {3, 14, 14, 0}},
    {.n = 720, .radices = {5, 12, 12, 0}},
    {.n = 756, .radices = {14, 6, 9, 0}},
}};

constexpr pass_costs neon_fcma_double{
    .unit = {{
        {97, 156, 201, 317, 345, 661, 491, 865, 773, 1153, 891, 1612, 1453, 1498, 1386},
        {161, 170, 190, 307, 354, 624, 470, 856, 723, 1078, 876, 1585, 1363, 1406, 1285},
        {200, 199, 209, 276, 317, 522, 415, 741, 624, 975, 698, 1370, 1183, 1187, 1062},
    }},
    .tail = {{
        {10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 10, 18, 10, 10, 10},
        {10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 10, 18, 10, 10, 10},
        {10, 10, 10, 10, 10, 10, 10, 10, 10, 16, 10, 20, 10, 10, 10},
    }},
    .pass = 3142,
    .overrides = neon_fcma_double_overrides,
};

constexpr std::array<radix_override, 7> neon_fcma_float_overrides{{
    {.n = 20, .radices = {4, 5, 0, 0}},
    {.n = 21, .radices = {3, 7, 0, 0}},
    {.n = 28, .radices = {4, 7, 0, 0}},
    {.n = 45, .radices = {5, 9, 0, 0}},
    {.n = 70, .radices = {7, 10, 0, 0}},
    {.n = 280, .radices = {4, 7, 10, 0}},
    {.n = 480, .radices = {4, 10, 12, 0}},
}};

constexpr pass_costs neon_fcma_float{
    .unit = {{
        {28, 65, 98, 157, 204, 365, 322, 495, 489, 730, 590, 1067, 914, 867, 993},
        {112, 135, 190, 276, 335, 508, 501, 687, 689, 1028, 833, 1507, 1201, 1214, 1338},
        {84, 77, 96, 116, 148, 254, 232, 366, 321, 502, 408, 801, 675, 626, 700},
    }},
    .tail = {{
        {142, 1869, 0, 1686, 0, 1745, 35, 1121, 0, 0, 25, 45, 192, 1429, 0},
        {19, 37, 46, 4, 50, 56, 39, 8, 178, 266, 13, 23, 29, 36, 0},
        {29, 799, 289, 787, 442, 1217, 391, 398, 604, 944, 529, 1037, 633, 263, 290},
    }},
    .pass = 913,
    .overrides = neon_fcma_float_overrides,
};

// measured_tables rows: {target::neon_fcma, 1, &neon_fcma_double}, {target::neon_fcma, 2, &neon_fcma_float}

// The measured tables, by target and C. A target measured on its own adds its rows here
// (bench/x2/fit.py writes them); the search does not change.
struct measured_table {
  target t;
  std::size_t lanes;
  const pass_costs* costs;
};
// Plain `neon` (no FCMA) takes `neon_fcma`'s tables: the same registers and C, the nearest
// measured costs (unmeasured on a plain NEON core).
constexpr std::array<measured_table, 6> measured_tables{{
    {.t = target::avx2, .lanes = 2, .costs = &avx2_double},
    {.t = target::avx2, .lanes = 4, .costs = &avx2_float},
    {.t = target::neon_fcma, .lanes = 1, .costs = &neon_fcma_double},
    {.t = target::neon_fcma, .lanes = 2, .costs = &neon_fcma_float},
    {.t = target::neon, .lanes = 1, .costs = &neon_fcma_double},
    {.t = target::neon, .lanes = 2, .costs = &neon_fcma_float},
}};

// Where a target and C have no table of their own: AVX2's table of the nearest C, the
// smaller on a tie (so C = 1 takes C = 2's and C = 8 takes C = 4's).
constexpr target fallback_target = target::avx2;

std::size_t lanes_distance(std::size_t a, std::size_t b) { return a > b ? a - b : b - a; }

} // namespace

target_profile profile_of(target t, std::size_t lanes) noexcept {
  const measured_table* nearest = nullptr;
  for (const measured_table& m : measured_tables) {
    if (m.t == t && m.lanes == lanes) {
      return {.lanes = lanes, .costs = m.costs};
    }
    if (m.t == fallback_target &&
        (nearest == nullptr ||
         lanes_distance(m.lanes, lanes) < lanes_distance(nearest->lanes, lanes))) {
      nearest = &m;
    }
  }
  return {.lanes = lanes, .costs = nearest->costs};
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
