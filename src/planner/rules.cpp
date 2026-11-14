// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/rules.hpp"

#include "isa/target.hpp"
#include "passes/pass_args.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

// Every radix up to this one has a codelet (src/gen/codelets.hpp): lengths up to it are
// leaves, it is the largest radix the pass costs price, and a prime factor above it is a
// generic pass.
constexpr std::size_t largest_leaf = 16;

// The radices a pass may have, ascending (rules.hpp, cost_column).
constexpr std::array<std::size_t, cost_radices> pass_radices{2,  3,  4,  5,  6,  7,  8,  9, 10,
                                                             11, 12, 13, 14, 15, 16, 32, 64};

// Experiment X2 (dev/experiments/X2-radix-sets.md), EPYC 7763, AVX2, GCC 16, one thread: the
// pass costs fitted to the screening and refinement timings of every candidate sequence of
// the 7-smooth lengths up to 1024, batches 1 and 64 together. No such length has 11 or 13,
// so their columns are radix 10's and 12's scaled by the codelets' operation counts.
// The overrides are the lengths where the costs' choice was more than 2 % slower than
// planner v0 in the refinement; each is the measured winner, which no batch of that
// precision measured slower than v0. 4096 and 16384 lie beyond the fit, where its radix-16
// passes (18 live values in 16 registers) lose: experiment X18 measured 8.8.8.8 at 0.75 of
// the fitted 4.8.8.16 and 8.8.16.16 at 0.84 of 4.16.16.16 (EPYC 7763, one row), until X2's
// campaign over the longer lengths replaces them.
constexpr std::array<radix_override, 6> avx2_double_overrides{{
    {.n = 42, .radices = {2, 3, 7, 0}},
    {.n = 70, .radices = {2, 5, 7, 0}},
    {.n = 350, .radices = {2, 5, 5, 7}},
    {.n = 490, .radices = {2, 7, 5, 7}},
    {.n = 4096, .radices = {8, 8, 8, 8}},
    {.n = 16384, .radices = {8, 8, 16, 16}},
}};
constexpr std::array<radix_override, 2> avx2_float_overrides{{
    {.n = 140, .radices = {4, 5, 7, 0}},
    {.n = 700, .radices = {4, 7, 5, 5}},
}};

constexpr pass_costs avx2_double{
    .unit = {{
        {28, 195, 114, 305, 235, 460, 387, 665, 557, 830, 684, 1238, 995, 1231, 1273, 0, 0},
        {88, 147, 190, 286, 324, 468, 486, 696, 658, 982, 758, 1371, 1096, 1200, 1255, 0, 0},
        {46, 58, 79, 116, 143, 236, 221, 364, 323, 504, 326, 639, 636, 667, 652, 0, 0},
    }},
    .tail = {{
        {391, 748, 914, 1102, 1467, 1552, 1879, 2354, 2535, 3780, 3073, 5560, 3464, 3140, 3836, 0, 0},
        {377, 536, 702, 884, 1061, 1348, 1354, 1733, 1749, 2608, 2073, 3752, 2522, 2810, 3152, 0, 0},
        {19, 1635, 518, 2086, 859, 2215, 1308, 3223, 1804, 2819, 2381, 4671, 2853, 3738, 3002, 0, 0},
    }},
    .pass = 436,
    .overrides = avx2_double_overrides,
};

constexpr pass_costs avx2_float{
    .unit = {{
        {28, 197, 76, 132, 205, 452, 298, 534, 538, 803, 525, 950, 856, 1070, 924, 0, 0},
        {101, 146, 190, 293, 343, 462, 525, 781, 700, 1044, 827, 1496, 1159, 1311, 1459, 0, 0},
        {42, 59, 91, 116, 138, 223, 213, 389, 293, 458, 367, 720, 622, 570, 674, 0, 0},
    }},
    .tail = {{
        {644, 628, 1125, 1110, 1612, 1655, 2285, 2601, 3009, 4487, 3539, 6405, 4310, 4163, 4957, 0, 0},
        {540, 855, 1057, 1304, 1659, 1840, 2129, 2511, 2738, 4082, 3295, 5963, 3767, 4056, 4186, 0, 0},
        {282, 304, 951, 925, 1575, 1680, 2291, 2497, 3013, 4708, 3797, 7448, 4305, 4096, 4695, 0, 0},
    }},
    .pass = 406,
    .overrides = avx2_float_overrides,
};

// Experiment X3 (dev/experiments/X3-big-radices.md), Xeon Gold 6240, AVX-512, GCC 14, one
// thread, float (C = 8): avx2_float with radix 32 and 64 columns fitted to an X2-style search
// over every multiple of 32 from 64 to 2048 whose candidates were the sequences containing
// 32 or 64 (bench/x2/fit.py --base f32=avx2_float); the costs up to radix 16 are
// avx2_float's, so wherever the choice has neither 32 nor 64 it is avx2_float's at C = 8. The
// overrides are avx2_float's plus the lengths where the choice was more than 2 % behind
// the baseline in some case or behind the measured winner in every case (256, 512: 32.8 and
// 32.16 beat the table's 8.32 and 16.32 by 9 to 15 %). Beyond 2048 the radix 32 and 64
// columns are extrapolated. AVX-512 in double took no table: its big chains tied (X3).
constexpr std::array<radix_override, 7> avx512_float_overrides{{
    {.n = 140, .radices = {4, 5, 7, 0}},
    {.n = 256, .radices = {32, 8, 0, 0}},
    {.n = 512, .radices = {32, 16, 0, 0}},
    {.n = 700, .radices = {4, 7, 5, 5}},
    {.n = 768, .radices = {8, 3, 32, 0}},
    {.n = 1536, .radices = {8, 12, 16, 0}},
    {.n = 2048, .radices = {8, 4, 64, 0}},
}};

constexpr pass_costs avx512_float{
    .unit = {{
        {28, 197, 76, 132, 205, 452, 298, 534, 538, 803, 525, 950, 856, 1070, 924, 2134, 8695},
        {101, 146, 190, 293, 343, 462, 525, 781, 700, 1044, 827, 1496, 1159, 1311, 1459, 3244, 8235},
        {42, 59, 91, 116, 138, 223, 213, 389, 293, 458, 367, 720, 622, 570, 674, 1605, 5336},
    }},
    .tail = {{
        {644, 628, 1125, 1110, 1612, 1655, 2285, 2601, 3009, 4487, 3539, 6405, 4310, 4163, 4957, 9397, 18667},
        {540, 855, 1057, 1304, 1659, 1840, 2129, 2511, 2738, 4082, 3295, 5963, 3767, 4056, 4186, 8168, 16780},
        {282, 304, 951, 925, 1575, 1680, 2291, 2497, 3013, 4708, 3797, 7448, 4305, 4096, 4695, 13892, 23943},
    }},
    .pass = 406,
    .overrides = avx512_float_overrides,
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
  const std::size_t col = cost_column(r);
  return std::uint64_t{p.costs->unit[kind][col]} * units +
         std::uint64_t{p.costs->tail[kind][col]} * tails + p.costs->pass;
}

// NEON with FCMA, fitted by bench/x2/fit.py to X2's campaign on an Apple M5 Max
// (campaign mac-smooth, 7-smooth lengths up to 1024, batches 1 and 64). Three double
// overrides (490, 686, 882) come from the A/B of these tables against the AVX2 ones (run
// 20261114-101523-3d56): there the fitted choice was 1.25-1.40 times slower.
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

constexpr std::array<radix_override, 18> neon_fcma_double_overrides{{
    {.n = 56, .radices = {8, 7, 0, 0}},
    {.n = 63, .radices = {7, 9, 0, 0}},
    {.n = 200, .radices = {4, 5, 10, 0}},
    {.n = 252, .radices = {3, 6, 14, 0}},
    {.n = 288, .radices = {4, 6, 12, 0}},
    {.n = 336, .radices = {8, 6, 7, 0}},
    {.n = 420, .radices = {5, 6, 14, 0}},
    {.n = 480, .radices = {4, 12, 10, 0}},
    {.n = 490, .radices = {2, 7, 5, 7}},
    {.n = 504, .radices = {12, 6, 7, 0}},
    {.n = 560, .radices = {4, 10, 14, 0}},
    {.n = 567, .radices = {7, 9, 9, 0}},
    {.n = 576, .radices = {4, 12, 12, 0}},
    {.n = 588, .radices = {3, 14, 14, 0}},
    {.n = 686, .radices = {14, 7, 7, 0}},
    {.n = 720, .radices = {5, 12, 12, 0}},
    {.n = 756, .radices = {14, 6, 9, 0}},
    {.n = 882, .radices = {14, 7, 9, 0}},
}};

constexpr pass_costs neon_fcma_double{
    .unit = {{
        {97, 156, 201, 317, 345, 661, 491, 865, 773, 1153, 891, 1612, 1453, 1498, 1386, 0, 0},
        {161, 170, 190, 307, 354, 624, 470, 856, 723, 1078, 876, 1585, 1363, 1406, 1285, 0, 0},
        {200, 199, 209, 276, 317, 522, 415, 741, 624, 975, 698, 1370, 1183, 1187, 1062, 0, 0},
    }},
    .tail = {{
        {10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 10, 18, 10, 10, 10, 0, 0},
        {10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 10, 18, 10, 10, 10, 0, 0},
        {10, 10, 10, 10, 10, 10, 10, 10, 10, 16, 10, 20, 10, 10, 10, 0, 0},
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
        {28, 65, 98, 157, 204, 365, 322, 495, 489, 730, 590, 1067, 914, 867, 993, 0, 0},
        {112, 135, 190, 276, 335, 508, 501, 687, 689, 1028, 833, 1507, 1201, 1214, 1338, 0, 0},
        {84, 77, 96, 116, 148, 254, 232, 366, 321, 502, 408, 801, 675, 626, 700, 0, 0},
    }},
    .tail = {{
        {142, 1869, 0, 1686, 0, 1745, 35, 1121, 0, 0, 25, 45, 192, 1429, 0, 0, 0},
        {19, 37, 46, 4, 50, 56, 39, 8, 178, 266, 13, 23, 29, 36, 0, 0, 0},
        {29, 799, 289, 787, 442, 1217, 391, 398, 604, 944, 529, 1037, 633, 263, 290, 0, 0},
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
constexpr std::array<measured_table, 7> measured_tables{{
    {.t = target::avx2, .lanes = 2, .costs = &avx2_double},
    {.t = target::avx2, .lanes = 4, .costs = &avx2_float},
    {.t = target::avx512, .lanes = 8, .costs = &avx512_float},
    {.t = target::neon_fcma, .lanes = 1, .costs = &neon_fcma_double},
    {.t = target::neon_fcma, .lanes = 2, .costs = &neon_fcma_float},
    {.t = target::neon, .lanes = 1, .costs = &neon_fcma_double},
    {.t = target::neon, .lanes = 2, .costs = &neon_fcma_float},
}};

// Where a target and C have no table of their own: AVX2's table of the nearest C, the
// smaller on a tie (so C = 1 takes C = 2's and C = 8 takes C = 4's).
constexpr target fallback_target = target::avx2;

std::size_t lanes_distance(std::size_t a, std::size_t b) { return a > b ? a - b : b - a; }

// Prime factors above this go through Bluestein (DESIGN.md 11.2). Measured on the Mac
// (Apple M5 Max, neon_fcma, dev/experiments/X17-bluestein-threshold.md): Bluestein beat the
// generic pass at every prime from 17 to 257, by 2.5 to 43 times in double and 7 to 135
// times in float, and at composites with many copies (17 x 64, 11 x 13 x 17) by 2 to 3
// times. So every prime factor above the codelets goes through Bluestein; the generic pass
// remains for radices a test forces. Unmeasured on the other targets, which take the same
// value until a measurement there replaces it.
constexpr std::size_t default_bluestein_above = 16;

// Experiment X8 (dev/experiments/X8-leaves.md): a whole-transform leaf of 32 beat planner
// v1's chain by 16 to 22 % on NEON in double (C = 1, neon and neon_fcma, M5 Max) and lost
// on every target with C > 1 and at 64 everywhere (radix 64 spills; a leaf idles C - 1
// lanes). So the leaf limit is 32 on NEON in double and 16 elsewhere.
std::size_t leaf_limit_of(target t, std::size_t lanes) {
  return (t == target::neon || t == target::neon_fcma) && lanes == 1 ? 32 : default_leaf_limit;
}

struct sequence {
  std::uint64_t cost = 0;
  std::vector<std::size_t> radices;
};

// The cheapest sequence of codelet radices for the smooth part `smooth` of a line of n
// points, followed by generic passes when `more` (rule 4 of DESIGN.md 11.2).
sequence cheapest(std::size_t n, std::size_t smooth, bool more, const target_profile& p) {
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
    return next == smooth && !more ? 2 : 1;
  };
  // Radix r may be the pass after prefix product s: it divides what is left and the table
  // prices it (every radix up to 16 is priced; 32 and 64 where their unit cost is not 0).
  const auto allowed = [&](std::size_t s, std::size_t r) {
    return (smooth / s) % r == 0 &&
           (r <= largest_leaf || p.costs->unit[kind(s, s * r)][cost_column(r)] != 0);
  };
  // best[i]: the lowest cost of the passes from prefix product divisors[i] to the end.
  std::vector<std::uint64_t> best(divisors.size(), 0);
  for (std::size_t i = divisors.size() - 1; i-- > 0;) {
    const std::size_t s = divisors[i];
    bool found = false;
    for (const std::size_t r : pass_radices) {
      if (!allowed(s, r)) {
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
  sequence out{.cost = best[0], .radices = {}};
  for (std::size_t s = 1; s != smooth;) {
    for (const std::size_t r : pass_radices) {
      if (allowed(s, r) &&
          pass_cost(p, n, s, r, kind(s, s * r)) + best[index(s * r)] == best[index(s)]) {
        out.radices.push_back(r);
        s *= r;
        break;
      }
    }
  }
  return out;
}

} // namespace

target_profile profile_of(target t, std::size_t lanes) noexcept {
  const measured_table* nearest = nullptr;
  for (const measured_table& m : measured_tables) {
    if (m.t == t && m.lanes == lanes) {
      return {.lanes = lanes,
              .costs = m.costs,
              .leaf_limit = leaf_limit_of(t, lanes),
              .bluestein_above = default_bluestein_above};
    }
    if (m.t == fallback_target &&
        (nearest == nullptr ||
         lanes_distance(m.lanes, lanes) < lanes_distance(nearest->lanes, lanes))) {
      nearest = &m;
    }
  }
  return {.lanes = lanes,
          .costs = nearest->costs,
          .leaf_limit = leaf_limit_of(t, lanes),
          .bluestein_above = default_bluestein_above};
}

// The specification is DESIGN.md 11.2, rule by rule.
std::vector<std::size_t> rule_radices(std::size_t n, const target_profile& p) {
  if (n <= 1) {
    return {};
  }
  if (n <= largest_leaf || (n <= p.leaf_limit && has_codelet(n))) {
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
  std::vector<std::size_t> out = cheapest(n, smooth, !big.empty(), p).radices;
  out.insert(out.end(), big.begin(), big.end());
  return out;
}

std::size_t bluestein_length(std::size_t q, const target_profile& p) {
  const std::size_t least = 2 * q - 1;
  std::size_t power = 1;
  while (power < least) {
    power *= 2;
  }
  // The two chirp passes touch M points each, priced like a last radix-2 pass per point.
  const std::uint64_t per_point = p.costs->unit[2][0] / (2 * p.lanes) + 1;
  // The 7-smooth lengths in [least, power], ascending.
  std::vector<std::size_t> candidates;
  for (std::size_t a = 1; a <= power; a *= 2) {
    for (std::size_t b = a; b <= power; b *= 3) {
      for (std::size_t c = b; c <= power; c *= 5) {
        for (std::size_t d = c; d <= power; d *= 7) {
          if (d >= least) {
            candidates.push_back(d);
          }
        }
      }
    }
  }
  std::ranges::sort(candidates);
  std::size_t best = power;
  std::uint64_t best_cost = 0;
  bool found = false;
  for (const std::size_t m : candidates) {
    const std::uint64_t c = 2 * cheapest(m, m, false, p).cost + 2 * per_point * m;
    if (!found || c < best_cost) {
      best = m;
      best_cost = c;
      found = true;
    }
  }
  return best;
}

} // namespace fftn::inline v0::detail
