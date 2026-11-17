// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/rules.hpp"

#include "isa/target.hpp"
#include "passes/pass_args.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

// Every radix up to this one has a codelet (src/gen/codelets.hpp): lengths up to it are
// leaves, and it is the largest radix the pass costs price. A prime factor above it is a
// pass after the priced ones: a codelet pass for 17, 19 and 23 (experiment X27), else the
// generic pass or, above the profile's threshold, a Bluestein stage.
constexpr std::size_t largest_leaf = 16;

// The radices a pass may have, ascending (rules.hpp, cost_column).
constexpr std::array<std::size_t, cost_radices> pass_radices{
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64,
};

// Experiment X2 (dev/experiments/X2-radix-sets.md), EPYC 7763, AVX2, GCC 16, one thread: the
// pass costs fitted to the screening and refinement timings of every candidate sequence of
// the 7-smooth lengths up to 1024, batches 1 and 64 together. No such length has 11 or 13,
// so their columns are radix 10's and 12's scaled by the codelets' operation counts.
// The overrides are the lengths where the costs' choice was more than 2 % slower than
// planner v0 in the refinement; each is the measured winner, which no batch of that
// precision measured slower than v0. 4096 lies beyond the fit, where its radix-16 passes
// (18 live values in 16 registers) lose: experiment X18 measured 8.8.8.8 at 0.75 of the
// fitted 4.8.8.16 (EPYC 7763, one row).
// The planner refresh (X2, "Refit after X1, X3, X18 and X19") kept these costs and added
// overrides from the campaigns epyc-avx2-smooth and epyc-avx2-long (EPYC 7763, split runs
// on): the lengths where the costs' choice was more than 2 % behind the measured winner in
// every case (fit.py --base), less those whose override was more than 2 % slower than the
// baseline in a case of the Xeon 6240's AVX2 campaigns (xeon-avx2-smooth2, -long), and less
// 512 (2.16.16): 0.92-0.95 as a row, but the r2c of 1024 x 1024 runs it as 16.16.2 with the
// pairing fused into the radix-2 pass, 1.05 times as long (EPYC tracking 20261115-041212-cdc2).
// 256 and 512 are experiment X30's: two passes with a radix-32 codelet, which the costs (no
// radix 32 or 64 column, X3) cannot choose, where the chain ran three
// (dev/experiments/X30-x86-pass-loops.md).
// 384 and 20000 are experiment X38's: X2's search driver re-run on the EPYC 7763 with
// radix 32 and 64 candidates (campaign x38-epyc-avx2), the lengths where one sequence was at
// most 0.97 of the planner's at batch 1 and 64; 80, 160, 240, 640, 2000, 3000 and 3600 take
// a radix-20 pass by the same rule (campaign x38-r20), as FFTW's plans do there
// (dev/experiments/X38-epyc-mid.md). 500, 3125, 10000 and 15625 take a radix-25 pass by
// that rule (experiment X42, campaign x42-r25f-epyc: dev/experiments/X42-codelets.md); 5000
// and 25000 by X42's arm B3: one sequence at most 0.97 of the planner's at both batches,
// though the two batches' winners differ.
// clang-format off
// NOLINTBEGIN(readability-trailing-comma): bench/x2/fit.py writes and parses this layout
constexpr std::array<radix_override, 42> avx2_double_overrides{{
    {.n = 27, .radices = {3, 9, 0, 0}},
    {.n = 42, .radices = {2, 3, 7, 0}},
    {.n = 54, .radices = {2, 3, 9, 0}},
    {.n = 63, .radices = {7, 9, 0, 0}},
    {.n = 70, .radices = {2, 5, 7, 0}},
    {.n = 80, .radices = {4, 20, 0, 0}},
    {.n = 160, .radices = {8, 20, 0, 0}},
    {.n = 168, .radices = {12, 14, 0, 0}},
    {.n = 216, .radices = {2, 12, 9, 0}},
    {.n = 240, .radices = {12, 20, 0, 0}},
    {.n = 256, .radices = {8, 32, 0, 0}},
    {.n = 350, .radices = {2, 5, 5, 7}},
    {.n = 384, .radices = {12, 32, 0, 0}},
    {.n = 448, .radices = {2, 16, 14, 0}},
    {.n = 490, .radices = {2, 7, 5, 7}},
    {.n = 500, .radices = {20, 25, 0, 0}},
    {.n = 512, .radices = {16, 32, 0, 0}},
    {.n = 567, .radices = {9, 9, 7, 0}},
    {.n = 588, .radices = {6, 7, 14, 0}},
    {.n = 640, .radices = {20, 32, 0, 0}},
    {.n = 675, .radices = {15, 5, 9, 0}},
    {.n = 700, .radices = {10, 5, 14, 0}},
    {.n = 768, .radices = {4, 16, 12, 0}},
    {.n = 896, .radices = {4, 16, 14, 0}},
    {.n = 1536, .radices = {6, 16, 16, 0}},
    {.n = 1920, .radices = {10, 16, 12, 0}},
    {.n = 2000, .radices = {10, 10, 20, 0}},
    {.n = 2400, .radices = {10, 15, 16, 0}},
    {.n = 3000, .radices = {10, 15, 20, 0}},
    {.n = 3125, .radices = {5, 25, 25, 0}},
    {.n = 3600, .radices = {12, 15, 20, 0}},
    {.n = 4096, .radices = {8, 8, 8, 8}},
    {.n = 4800, .radices = {8, 12, 5, 10}},
    {.n = 5000, .radices = {10, 20, 25, 0}},
    {.n = 6000, .radices = {8, 10, 5, 15}},
    {.n = 8192, .radices = {8, 8, 16, 8}},
    {.n = 10000, .radices = {20, 20, 25, 0}},
    {.n = 12000, .radices = {8, 10, 15, 10}},
    {.n = 15625, .radices = {25, 25, 25, 0}},
    {.n = 16384, .radices = {8, 16, 16, 8}},
    {.n = 20000, .radices = {8, 5, 10, 5, 10}},
    {.n = 25000, .radices = {10, 5, 20, 25, 0}},
}};
constexpr std::array<radix_override, 13> avx2_float_overrides{{
    {.n = 90, .radices = {9, 10, 0, 0}},
    {.n = 140, .radices = {4, 5, 7, 0}},
    {.n = 240, .radices = {4, 6, 10, 0}},
    {.n = 336, .radices = {4, 12, 7, 0}},
    {.n = 486, .radices = {9, 9, 6, 0}},
    {.n = 560, .radices = {4, 10, 14, 0}},
    {.n = 567, .radices = {9, 9, 7, 0}},
    {.n = 700, .radices = {4, 5, 5, 7}},
    {.n = 3600, .radices = {16, 15, 15, 0}},
    {.n = 4800, .radices = {8, 10, 4, 15}},
    {.n = 7200, .radices = {12, 4, 10, 15}},
    {.n = 8192, .radices = {8, 8, 16, 8}},
    {.n = 12000, .radices = {16, 5, 15, 10}},
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
// NOLINTEND(readability-trailing-comma)
// clang-format on

// The planner refresh (dev/experiments/X2-radix-sets.md, "Refit after X1, X3, X18 and X19"),
// Xeon Gold 6240, AVX-512, GCC 14, one thread, split runs on (X1).
// Double (C = 4): avx2_float's costs (the nearest C, as before) with the overrides where its
// choice was more than 2 % behind the measured winner in every case of the length
// (fit.py --base f64=avx2_float on campaign xeon-avx512-smooth2): mostly a first radix of 8,
// after which a split run (block 8) starts. Not 576 (8.12.6, 0.92-0.94 as a row): as the
// Bluestein convolution of 257 it ran 1.06 times as long (Xeon tracking 20261115-041226-994e).
// Float (C = 8): fitted by bench/x2/fit.py to the campaigns xeon-avx512-smooth2 (every
// candidate of the 7-smooth lengths up to 1024), xeon-avx512-big2 (the sequences with radix
// 32 or 64, multiples of 32 up to 2048) and xeon-avx512-long (the cost estimate's best 20
// per catalogue length 1200-32768, radices up to 64). The overrides (fit.py --cover) are the
// measured lengths where the costs' choice was more than 2 % slower than the campaigns'
// baseline (X3's table) in some case, more than 2 % behind the winner in every case, or
// never timed (above 1024 the costs extrapolate from 20 candidates per length): each is the
// sequence least behind the winner at worst and no slower than the baseline. 21504 keeps
// X3's 8.7.12.32: it is the Bluestein convolution of 10403 = 101 x 103, where the costs'
// unmeasured five-pass choice ran 1.08 times as long (run 20261115-040151-b716).
// Double, experiment X39 arm D (after X38's radix-20 codelet): 300-25000 with 2^2 x 5 to spare
// take a radix-20 first pass, which the costs (no radix-20 column) cannot choose: 0.82-0.92 of
// the costs' sequences in user-mode cycles on the Xeon 6240 (screens 16183, 16184); a radix 20
// last lost (1.04-1.22). 1000 keeps its X2 override: as 20.10.5 its single rows took 0.90
// but planes and batches of 1000 1.02-1.04 (run 20261117-011743-1a01)
// (dev/experiments/X39-xeon-long.md). Experiment X42: 100 and 500 end in a radix-25 pass,
// one sequence at most 0.97 of the planner's at batch 1 and 64 (X2's search driver,
// campaign x42-r25f-xeon; dev/experiments/X42-codelets.md); 3125, 10000, 15625 and 25000 by
// X42's arm B3: one sequence at most 0.97 of the planner's at both batches, though the two
// batches' winners differ.
// clang-format off
// NOLINTBEGIN(readability-trailing-comma): bench/x2/fit.py writes and parses this layout
constexpr std::array<radix_override, 31> avx512_double_overrides{{
    {.n = 24, .radices = {4, 6, 0, 0}},
    {.n = 27, .radices = {3, 9, 0, 0}},
    {.n = 100, .radices = {4, 25, 0, 0}},
    {.n = 140, .radices = {4, 5, 7, 0}},
    {.n = 168, .radices = {4, 6, 7, 0}},
    {.n = 288, .radices = {8, 12, 3, 0}},
    {.n = 300, .radices = {20, 15, 0, 0}},
    {.n = 320, .radices = {8, 10, 4, 0}},
    {.n = 480, .radices = {8, 12, 5, 0}},
    {.n = 500, .radices = {20, 25, 0, 0}},
    {.n = 640, .radices = {8, 10, 8, 0}},
    {.n = 648, .radices = {4, 6, 9, 3}},
    {.n = 672, .radices = {8, 12, 7, 0}},
    {.n = 700, .radices = {4, 7, 5, 5}},
    {.n = 768, .radices = {8, 12, 8, 0}},
    {.n = 864, .radices = {8, 9, 12, 0}},
    {.n = 896, .radices = {8, 14, 8, 0}},
    {.n = 1000, .radices = {4, 10, 5, 5}},
    {.n = 2000, .radices = {20, 10, 10, 0}},
    {.n = 3000, .radices = {20, 10, 15, 0}},
    {.n = 3125, .radices = {25, 5, 25, 0}},
    {.n = 3600, .radices = {12, 15, 20, 0}},
    {.n = 4000, .radices = {20, 20, 10, 0}},
    {.n = 4800, .radices = {20, 12, 20, 0}},
    {.n = 5000, .radices = {20, 10, 5, 5}},
    {.n = 6000, .radices = {20, 15, 20, 0}},
    {.n = 8000, .radices = {20, 20, 20, 0}},
    {.n = 10000, .radices = {20, 20, 25, 0}},
    {.n = 15625, .radices = {5, 5, 25, 25}},
    {.n = 20000, .radices = {20, 20, 5, 10}},
    {.n = 25000, .radices = {4, 25, 10, 25, 0}},
}};

constexpr pass_costs avx512_double{
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
    .overrides = avx512_double_overrides,
};

constexpr std::array<radix_override, 35> avx512_float_overrides{{
    {.n = 24, .radices = {3, 8, 0, 0}},
    {.n = 36, .radices = {6, 6, 0, 0}},
    {.n = 96, .radices = {8, 3, 4, 0}},
    {.n = 128, .radices = {8, 16, 0, 0}},
    {.n = 256, .radices = {32, 8, 0, 0}},
    {.n = 300, .radices = {12, 5, 5, 0}},
    {.n = 384, .radices = {16, 4, 6, 0}},
    {.n = 432, .radices = {8, 6, 9, 0}},
    {.n = 504, .radices = {8, 7, 9, 0}},
    {.n = 512, .radices = {32, 16, 0, 0}},
    {.n = 576, .radices = {8, 2, 12, 3}},
    {.n = 600, .radices = {8, 5, 15, 0}},
    {.n = 648, .radices = {8, 9, 9, 0}},
    {.n = 672, .radices = {8, 7, 12, 0}},
    {.n = 768, .radices = {16, 12, 4, 0}},
    {.n = 784, .radices = {16, 7, 7, 0}},
    {.n = 800, .radices = {16, 10, 5, 0}},
    {.n = 864, .radices = {16, 9, 6, 0}},
    {.n = 896, .radices = {16, 14, 4, 0}},
    {.n = 960, .radices = {8, 2, 4, 15}},
    {.n = 1008, .radices = {8, 2, 9, 7}},
    {.n = 1024, .radices = {16, 16, 4, 0}},
    {.n = 1200, .radices = {16, 15, 5, 0}},
    {.n = 1440, .radices = {16, 9, 10, 0}},
    {.n = 1920, .radices = {16, 12, 10, 0}},
    {.n = 2048, .radices = {16, 8, 16, 0}},
    {.n = 2400, .radices = {16, 10, 15, 0}},
    {.n = 3000, .radices = {8, 5, 5, 15}},
    {.n = 4096, .radices = {32, 8, 16, 0}},
    {.n = 4800, .radices = {8, 4, 15, 10}},
    {.n = 7200, .radices = {32, 15, 15, 0}},
    {.n = 8192, .radices = {16, 16, 32, 0}},
    {.n = 16384, .radices = {16, 4, 16, 16}},
    {.n = 21504, .radices = {8, 7, 12, 32}},
    {.n = 32768, .radices = {8, 16, 16, 16}},
}};

constexpr pass_costs avx512_float{
    .unit = {{
        {28, 590, 666, 1612, 1420, 1704, 282, 772, 1068, 1593, 1390, 2516, 2256, 2687, 508, 2521, 9838},
        {55, 121, 190, 372, 460, 582, 587, 801, 918, 1369, 1063, 1924, 1733, 1567, 1563, 4790, 9876},
        {24, 54, 79, 179, 157, 331, 253, 501, 477, 745, 542, 1063, 855, 1044, 900, 2535, 7670},
    }},
    .tail = {{
        {739, 122, 648, 359, 702, 1264, 2219, 2027, 2544, 3794, 2884, 5219, 3228, 3140, 4341, 6569, 15857},
        {541, 789, 862, 1097, 1259, 1551, 1541, 1661, 2017, 3007, 2372, 4293, 2500, 3071, 2743, 4824, 12253},
        {391, 508, 807, 916, 1224, 1858, 2086, 959, 1844, 2882, 2334, 4578, 3332, 4041, 2982, 5514, 15468},
    }},
    .pass = 345,
    .overrides = avx512_float_overrides,
};
// NOLINTEND(readability-trailing-comma)
// clang-format on

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
  std::size_t tails = 0;
  if (kind == 0) {
    tails = m % c != 0 ? 1 : 0;
  } else {
    tails = s % c != 0 ? m : 0;
  }
  const std::size_t col = cost_column(r);
  return std::uint64_t{p.costs->unit[kind][col]} * units +
         std::uint64_t{p.costs->tail[kind][col]} * tails + p.costs->pass;
}

// NEON with FCMA, fitted by bench/x2/fit.py to X2's campaign on an Apple M5 Max
// (campaign mac-smooth, 7-smooth lengths up to 1024, batches 1 and 64). Three double
// overrides (490, 686, 882) come from the A/B of these tables against the AVX2 ones (run
// 20261114-101523-3d56): there the fitted choice was 1.25-1.40 times slower. The planner
// refresh (X2, "Refit after X1, X3, X18 and X19") kept the costs and added overrides from
// campaign mac-neon-smooth2 where the current choice (490 included) was more than 2 % behind
// the measured winner in every case (fit.py --base), less 768 (both precisions) and 448
// (float), which the A/B (run 20261115-035010-f3d4) measured 2-5 % slower; the notes below
// are the first fit's.
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
// rules.cpp avx2_double on these cases: 120 of 248 cases reproduced (48.4 %; b1-op 62/124, b64-op
// 58/124)
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
// rules.cpp avx2_float on these cases: 160 of 248 cases reproduced (64.5 %; b1-op-f32 79/124,
// b64-op-f32 81/124)
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

// clang-format off
// NOLINTBEGIN(readability-trailing-comma): bench/x2/fit.py writes and parses this layout
constexpr std::array<radix_override, 31> neon_fcma_double_overrides{{
    {.n = 48, .radices = {6, 8, 0, 0}},
    {.n = 56, .radices = {8, 7, 0, 0}},
    {.n = 63, .radices = {7, 9, 0, 0}},
    {.n = 98, .radices = {7, 14, 0, 0}},
    {.n = 200, .radices = {4, 5, 10, 0}},
    {.n = 252, .radices = {3, 6, 14, 0}},
    {.n = 288, .radices = {4, 6, 12, 0}},
    {.n = 300, .radices = {5, 6, 10, 0}},
    {.n = 336, .radices = {8, 6, 7, 0}},
    {.n = 384, .radices = {6, 8, 8, 0}},
    {.n = 420, .radices = {5, 6, 14, 0}},
    {.n = 448, .radices = {8, 8, 7, 0}},
    {.n = 480, .radices = {4, 12, 10, 0}},
    {.n = 490, .radices = {7, 7, 10, 0}},
    {.n = 504, .radices = {12, 6, 7, 0}},
    {.n = 560, .radices = {7, 8, 10, 0}},
    {.n = 567, .radices = {7, 9, 9, 0}},
    {.n = 576, .radices = {4, 12, 12, 0}},
    {.n = 588, .radices = {12, 7, 7, 0}},
    {.n = 675, .radices = {9, 5, 15, 0}},
    {.n = 686, .radices = {14, 7, 7, 0}},
    {.n = 700, .radices = {10, 7, 10, 0}},
    {.n = 720, .radices = {12, 10, 6, 0}},
    {.n = 756, .radices = {9, 12, 7, 0}},
    {.n = 784, .radices = {8, 7, 14, 0}},
    {.n = 800, .radices = {10, 8, 10, 0}},
    {.n = 840, .radices = {10, 7, 12, 0}},
    {.n = 882, .radices = {14, 7, 9, 0}},
    {.n = 900, .radices = {5, 12, 15, 0}},
    {.n = 980, .radices = {14, 10, 7, 0}},
    {.n = 1008, .radices = {12, 6, 14, 0}},
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

constexpr std::array<radix_override, 10> neon_fcma_float_overrides{{
    {.n = 20, .radices = {4, 5, 0, 0}},
    {.n = 21, .radices = {7, 3, 0, 0}},
    {.n = 28, .radices = {4, 7, 0, 0}},
    {.n = 45, .radices = {5, 9, 0, 0}},
    {.n = 70, .radices = {10, 7, 0, 0}},
    {.n = 224, .radices = {16, 14, 0, 0}},
    {.n = 280, .radices = {4, 5, 14, 0}},
    {.n = 480, .radices = {4, 10, 12, 0}},
    {.n = 720, .radices = {12, 12, 5, 0}},
    {.n = 800, .radices = {10, 8, 10, 0}},
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
// NOLINTEND(readability-trailing-comma)
// clang-format on

// measured_tables rows: {target::neon_fcma, 1, &neon_fcma_double}, {target::neon_fcma, 2,
// &neon_fcma_float}

// The measured tables, by target and C. A target measured on its own adds its rows here
// (bench/x2/fit.py writes them); the search does not change.
struct measured_table {
  target t;
  std::size_t lanes;
  const pass_costs* costs;
};
// Plain `neon` (no FCMA) takes `neon_fcma`'s tables: the same registers and C, the nearest
// measured costs (unmeasured on a plain NEON core).
constexpr std::array<measured_table, 8> measured_tables{
    {
        {.t = target::avx2, .lanes = 2, .costs = &avx2_double},
        {.t = target::avx2, .lanes = 4, .costs = &avx2_float},
        {.t = target::avx512, .lanes = 4, .costs = &avx512_double},
        {.t = target::avx512, .lanes = 8, .costs = &avx512_float},
        {.t = target::neon_fcma, .lanes = 1, .costs = &neon_fcma_double},
        {.t = target::neon_fcma, .lanes = 2, .costs = &neon_fcma_float},
        {.t = target::neon, .lanes = 1, .costs = &neon_fcma_double},
        {.t = target::neon, .lanes = 2, .costs = &neon_fcma_float},
    },
};

// Where a target and C have no table of their own: AVX2's table of the nearest C, the
// smaller on a tie (so C = 1 takes C = 2's and C = 8 takes C = 4's).
constexpr target fallback_target = target::avx2;
static_assert(measured_tables.front().t == fallback_target,
              "profile_of starts its search for the nearest table at the first");

std::size_t lanes_distance(std::size_t a, std::size_t b) { return a > b ? a - b : b - a; }

// Prime factors above this go through Bluestein (DESIGN.md 11.2). Measured on the Mac
// (Apple M5 Max, neon_fcma, dev/experiments/X17-bluestein-threshold.md): Bluestein beat the
// generic pass at every prime from 17 to 257, by 2.5 to 43 times in double and 7 to 135
// times in float, and at composites with many copies (17 x 64, 11 x 13 x 17) by 2 to 3
// times. Confirmed on x86 (X17's x86 section; fbench 20261115-075530-ddbd Xeon 6240 avx512,
// 20261115-075549-ba92 Xeon avx2, 20261115-075537-5a01 EPYC 7763 avx2): Bluestein faster at
// every prime from 17 to 257 in double and float, by 12 to 500 times. Only AVX-512 float
// composites with 64 or more copies of 17 ran up to 13 % faster in the generic pass, too
// narrow for a rule. So every prime factor above the codelets goes through Bluestein; the
// generic pass remains for radices a test forces. AArch64 Linux takes the same value
// unmeasured. The primes with a codelet (17, 19, 23, experiment X27) are never Bluestein's:
// the threshold applies to factors without one (chain_builder::radices).
constexpr std::size_t default_bluestein_above = 16;

// Experiment X8 (dev/experiments/X8-leaves.md): a whole-transform leaf of 32 beat planner
// v1's chain by 16 to 22 % on NEON in double (C = 1, neon and neon_fcma, M5 Max) and lost
// on every target with C > 1 and at 64 everywhere (radix 64 spills; a leaf idles C - 1
// lanes). So the leaf limit is 32 on NEON in double and 16 elsewhere.
std::size_t leaf_limit_of(target t, std::size_t lanes) {
  return (t == target::neon || t == target::neon_fcma) && lanes == 1 ? 32 : default_leaf_limit;
}

// Whether chains on target t end in split passes (DESIGN.md 6.3, strategies/chain.hpp),
// from experiment X1 (dev/experiments/X1-split-lowering.md): on x86 (avx2, avx512; double
// and float) split runs were 7-23 % faster than the same passes on interleaved registers
// on the Xeon 6240 and the EPYC 7763; on the Mac (neon_fcma) 10-25 % slower, so NEON and
// the scalar target stay interleaved.
bool split_of(target t, std::size_t /*lanes*/) { return t == target::avx2 || t == target::avx512; }

// The longest row the lanes strategy takes (target_profile::lanes_up_to), from experiment X7
// (dev/experiments/X7-lanes.md) against the chain with X20's batched leaves and flat
// programs: per target and complex count, the longest catalogue length up to which no case
// (c2c, r2c, c2r; batches of 64 to 4096) ran slower than the chain, on every machine
// measured. Only float on x86 qualifies: AVX-512 100 (C = 8; the Xeon), AVX2 32 (C = 4; the
// Xeon forced, the EPYC alone allowed 48). In double and on NEON the chain's batched leaf
// already runs short rows as fast as the lanes (8 points: lanes 1.09 of the chain on the
// M5, 1.3 on the Xeon, 1.6 on the EPYC), so the rule gives 0. X24
// (dev/experiments/X24-lanes-x86.md) re-tested double on x86 with a cheaper lanes driver and
// with edge passes that need no gather: even the best arm is slower than the chain at 8
// points (batch 64: 1.04 on the Xeon's AVX-512, 1.08 on the EPYC; neither change was kept),
// so double stays 0. The scalar target is not measured.
std::size_t lanes_up_to_of(target t, std::size_t lanes) {
  switch (t) {
  case target::avx512:
    return lanes == 8 ? 100 : 0;
  case target::avx2:
    return lanes == 4 ? 32 : 0;
  default:
    return 0;
  }
}

// The Rader range (target_profile::rader_from, rader_below), from experiment X27
// (dev/experiments/X27-prime-codelets.md), Rader against the Bluestein stage on single prime
// rows. Lower end: a single row runs Rader's gather and scatter in scalar code while the
// Bluestein stage's chirp passes fill the C lanes, so with C >= 4 small primes lose: 29 and
// 61 on AVX2 in float (1.27, 1.22; EPYC 7763, run 20261115-233441-5bdc), 29 on AVX-512 in
// double (1.11) and 29 to 97 in float (1.33-1.97; Xeon 6240, run 20261116-002300-c123),
// while from 97 (C = 4) and 257 (C = 8) up every prime measured won (0.45-0.91, except
// 8191 in AVX-512 float, 1.08); with C <= 2 every prime from 29 won (0.17-0.85: the M5 Max,
// runs 20261115-200226-b1eb and 20261115-203357-1ff2; the EPYC in double). So Rader starts
// at 16 C for C >= 4. Upper end: at 65537 Rader lost on the M5 Max (1.21: its gather and
// scatter wander over a megabyte) and won on the EPYC (0.62 double, 0.83 float) and the Xeon
// (0.62, 0.56), so NEON stops below 2^16 and x86 below 2^24, where the index table's floats
// stop being exact (passes/rader.inl). The scalar target takes NEON's values, unmeasured.
std::size_t rader_from_of(std::size_t lanes) { return lanes >= 4 ? 16 * lanes : 0; }
std::size_t rader_below_of(target t) {
  return t == target::avx2 || t == target::avx512 ? std::size_t{1} << 24U : std::size_t{1} << 16U;
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
    const std::size_t left = smooth / s;
    bool found = false;
    for (const std::size_t r : pass_radices) {
      if (left % r != 0 || !allowed(s, r)) {
        continue;
      }
      // s r is a divisor above s: search only past i.
      const auto next = std::ranges::lower_bound(
          divisors.begin() + static_cast<std::ptrdiff_t>(i) + 1, divisors.end(), s * r);
      const std::uint64_t c = pass_cost(p, n, s, r, kind(s, s * r)) +
                              best[static_cast<std::size_t>(next - divisors.begin())];
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
  const measured_table* nearest = &measured_tables.front();
  for (const measured_table& m : measured_tables) {
    if (m.t == t && m.lanes == lanes) {
      return {
          .lanes = lanes,
          .costs = m.costs,
          .leaf_limit = leaf_limit_of(t, lanes),
          .bluestein_above = default_bluestein_above,
          .split = split_of(t, lanes),
          .lanes_up_to = lanes_up_to_of(t, lanes),
          .rader_from = rader_from_of(lanes),
          .rader_below = rader_below_of(t),
      };
    }
    if (m.t == fallback_target &&
        lanes_distance(m.lanes, lanes) < lanes_distance(nearest->lanes, lanes)) {
      nearest = &m;
    }
  }
  return {
      .lanes = lanes,
      .costs = nearest->costs,
      .leaf_limit = leaf_limit_of(t, lanes),
      .bluestein_above = default_bluestein_above,
      .split = split_of(t, lanes),
      .lanes_up_to = lanes_up_to_of(t, lanes),
      .rader_from = rader_from_of(lanes),
      .rader_below = rader_below_of(t),
  };
}

namespace {

// n = smooth x the prime factors above 16 (ascending), which become passes after the
// priced ones: a codelet's (17, 19, 23), the generic pass's or one Bluestein stage
// (chain_builder::radices).
std::size_t split_smooth(std::size_t n, std::vector<std::size_t>& big) {
  std::size_t smooth = 1;
  std::size_t rest = n;
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
  return smooth;
}

std::vector<std::size_t> override_of(std::size_t n, const target_profile& p) {
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
  return {};
}

// The k cheapest sequences of codelet radices for the smooth part of a line of n points
// (cheapest's search keeping the k best continuations of every prefix product instead of
// one), by cost and then lexicographically, so that the first is cheapest's choice.
std::vector<sequence> ranked_sequences(std::size_t n, std::size_t smooth, bool more,
                                       const target_profile& p, std::size_t k) {
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
  const auto allowed = [&](std::size_t s, std::size_t r) {
    return (smooth / s) % r == 0 &&
           (r <= largest_leaf || p.costs->unit[kind(s, s * r)][cost_column(r)] != 0);
  };
  const auto before = [](const sequence& a, const sequence& b) {
    return a.cost != b.cost ? a.cost < b.cost : a.radices < b.radices;
  };
  // top[i]: the k cheapest continuations from prefix product divisors[i] to the end.
  std::vector<std::vector<sequence>> top(divisors.size());
  top.back().push_back({});
  for (std::size_t i = divisors.size() - 1; i-- > 0;) {
    const std::size_t s = divisors[i];
    std::vector<sequence> options;
    for (const std::size_t r : pass_radices) {
      if (!allowed(s, r)) {
        continue;
      }
      const std::uint64_t c = pass_cost(p, n, s, r, kind(s, s * r));
      for (const sequence& rest : top[index(s * r)]) {
        sequence q{.cost = c + rest.cost, .radices = {r}};
        q.radices.insert(q.radices.end(), rest.radices.begin(), rest.radices.end());
        options.push_back(std::move(q));
      }
    }
    std::ranges::sort(options, before);
    if (options.size() > k) {
      options.resize(k);
    }
    top[i] = std::move(options);
  }
  return std::move(top.front());
}

} // namespace

// The specification is DESIGN.md 11.2, rule by rule.
std::vector<std::size_t> rule_radices(std::size_t n, const target_profile& p, bool overrides) {
  if (n <= 1) {
    return {};
  }
  // Whole-transform leaves above 16: the powers of two (X8) and the primes (X27). Radices 20
  // and 25 are pass radices only (experiments X38, X42): their leaves were never measured.
  if (n <= largest_leaf || (n <= p.leaf_limit && has_codelet(n) && n != 20 && n != 25)) {
    return {n};
  }
  if (std::vector<std::size_t> o = overrides ? override_of(n, p) : std::vector<std::size_t>{};
      !o.empty()) {
    return o;
  }
  std::vector<std::size_t> big;
  const std::size_t smooth = split_smooth(n, big);
  if (smooth == 1) {
    return big;
  }
  std::vector<std::size_t> out = cheapest(n, smooth, !big.empty(), p).radices;
  out.insert(out.end(), big.begin(), big.end());
  return out;
}

std::vector<std::vector<std::size_t>> ranked_radices(std::size_t n, const target_profile& p,
                                                     std::size_t k) {
  std::vector<std::vector<std::size_t>> out;
  if (k == 0 || n <= 1) {
    return out;
  }
  const std::vector<std::size_t> first = rule_radices(n, p);
  out.push_back(first);
  if (first.size() <= 1 && n <= largest_leaf) {
    return out; // a leaf: nothing to rank
  }
  std::vector<std::size_t> big;
  const std::size_t smooth = split_smooth(n, big);
  if (smooth == 1) {
    return out;
  }
  for (sequence& q : ranked_sequences(n, smooth, !big.empty(), p, k)) {
    q.radices.insert(q.radices.end(), big.begin(), big.end());
    if (out.size() < k && std::ranges::find(out, q.radices) == out.end()) {
      out.push_back(std::move(q.radices));
    }
  }
  return out;
}

std::vector<std::size_t> long_pow2_radices(std::size_t k, bool radix8) {
  std::vector<std::size_t> rad;
  if (radix8) {
    std::size_t fours = 0; // k = 3 j + 2 takes one 4, k = 3 j + 1 two
    if (k % 3 != 0) {
      fours = k % 3 == 2 ? 1 : 2;
    }
    rad.assign(fours, 4);
    rad.insert(rad.end(), (k - 2 * fours) / 3, 8);
  } else {
    if (k % 2 == 1) {
      rad.push_back(8);
    }
    rad.insert(rad.end(), (k - 3 * (k % 2)) / 2, 4);
  }
  return rad;
}

namespace {

std::uint64_t pow_mod(std::uint64_t b, std::uint64_t e, std::uint64_t q) {
  std::uint64_t r = 1;
  b %= q;
  for (; e != 0; e >>= 1U) {
    if ((e & 1U) != 0) {
      r = r * b % q;
    }
    b = b * b % q;
  }
  return r;
}

} // namespace

std::size_t rader_root(std::size_t q, const target_profile& p) {
  if (q < 5 || q < p.rader_from || q >= p.rader_below || has_codelet(q)) {
    return 0;
  }
  for (std::size_t d = 2; d <= q / d; ++d) {
    if (q % d == 0) {
      return 0;
    }
  }
  std::size_t left = q - 1;
  for (const std::size_t f : {2U, 3U, 5U, 7U, 11U, 13U}) {
    while (left % f == 0) {
      left /= f;
    }
  }
  // The convolution needs two codelet passes or more (as a Bluestein stage's does).
  if (left != 1 || rule_radices(q - 1, p).size() < 2) {
    return 0;
  }
  for (std::uint64_t g = 2; g < q; ++g) {
    bool primitive = true;
    for (const std::uint64_t f : {2U, 3U, 5U, 7U, 11U, 13U}) {
      if ((q - 1) % f == 0 && pow_mod(g, (q - 1) / f, q) == 1) {
        primitive = false;
        break;
      }
    }
    if (primitive) {
      return static_cast<std::size_t>(g);
    }
  }
  return 0; // unreachable: every prime has a primitive root
}

double row_cost(std::size_t n, const target_profile& p) {
  if (n <= 1) {
    return 0;
  }
  // The table's own cheapest sequence for the smooth part, leaves included (a leaf is a first
  // pass with m = 1), so that every length is priced alike; the overrides are left out, since
  // they use radices the table does not price (32 and 64 on AVX2). Then the prime factors
  // above 16: a codelet's (17, 19, 23) priced as a middle radix-16 pass scaled by r log2 r,
  // the rest merged into one Rader or Bluestein stage at the end (chain_builder::radices).
  std::vector<std::size_t> big;
  const std::size_t smooth = split_smooth(n, big);
  std::size_t rest = 1;
  std::vector<std::size_t> codelet_primes;
  for (const std::size_t q : big) {
    if (has_codelet(q)) {
      codelet_primes.push_back(q);
    } else {
      rest *= q;
    }
  }
  double cost = smooth > 1 ? static_cast<double>(cheapest(n, smooth, !big.empty(), p).cost) : 0;
  for (const std::size_t r : codelet_primes) {
    // n / r butterflies of radix r, C to a unit, at a middle radix-16 unit's cost scaled by
    // the operations' ratio, r log2 r over 16 log2 16.
    const double scale = static_cast<double>(r) * std::log2(static_cast<double>(r)) / 64.0;
    const auto units = static_cast<double>(ceil_div(n / r, p.lanes));
    cost += (static_cast<double>(p.costs->unit[1][cost_column(16)]) * units +
             static_cast<double>(p.costs->pass)) *
            scale;
  }
  if (rest > 1) {
    // The stage runs n / rest copies of a convolution of L points: two transforms of L and
    // two elementwise passes (bluestein_length's pricing).
    const std::size_t len =
        rader_root(rest, p) != 0
            ? rest - 1
            : bluestein_length(rest, p, std::numeric_limits<std::size_t>::max());
    // Whole units per point (integer division, as bluestein_length prices them).
    const std::size_t units = p.costs->unit[2][0] / (2 * p.lanes) + 1;
    const auto per_point = static_cast<double>(units);
    const double conv = 2 * static_cast<double>(cheapest(len, len, false, p).cost) +
                        2 * per_point * static_cast<double>(len);
    const std::size_t copies = n / rest; // exact: rest divides n
    cost += conv * static_cast<double>(copies);
  }
  return cost;
}

double relative_row_cost(std::size_t n, const target_profile& p) {
  if (n <= 1) {
    return 1;
  }
  // A transform the planner runs as one codelet (a leaf, rule 1, or a lone 17, 19 or 23) is
  // as fast per point as a leaf of a power of two: the tables' first-pass costs at m = 1 do
  // not price leaves (X8 measured them apart), so they are not compared. Every other length
  // is priced as the chain the planner builds; 20 and 25 are (rule_radices: X38 and X42
  // measured no leaf of either).
  if ((n <= largest_leaf || has_codelet(n)) && rule_radices(n, p).size() == 1) {
    return 1;
  }
  const std::size_t power = std::bit_ceil(n);
  return (row_cost(n, p) / static_cast<double>(n)) /
         (row_cost(power, p) / static_cast<double>(power));
}

namespace {

// Whether n may be fast: every prime factor has a codelet (at most 23; a length with a
// larger one takes a Rader or Bluestein stage, which the reference prices at 1.49 times a
// power of two per point or more up to 20000, so it is never fast), and n is not a composite
// with a factor 11, 17, 19 or 23. Experiment P4: lengths with them missed 1.3 times the
// power of two above on two machines each (132 on the EPYC and the Xeon, 1.40-1.68; 1020 on
// the Mac and the Xeon, 1.32-1.38; 150144 and 250240 on the EPYC, 1.47-1.56; then 10051 =
// 19 x 23^2 on the Mac and the EPYC, 1.41-1.66), the reference's scaled columns being
// optimistic for them. Their single codelets stay.
bool may_be_fast(std::size_t n) {
  if (n == 11 || n == 17 || n == 19 || n == 23) {
    return true;
  }
  for (const std::size_t q : {2U, 3U, 5U, 7U, 13U}) {
    while (n % q == 0) {
      n /= q;
    }
  }
  return n == 1;
}

// The reference table of the fast lengths (task 2.5, DESIGN.md 9.2): a frozen copy of X2's
// AVX2 double table (avx2_double as it was at task 2.5) without its overrides, which
// row_cost does not read. It is not the planner's table: a refit of avx2_double changes
// plans, never the fast lengths, so users' grids move only when this copy is changed on
// purpose (tests/core/lengths_test.cpp pins the fast set up to 2^20).
// clang-format off
// NOLINTBEGIN(readability-trailing-comma): the layout of the tables above, which this copies
constexpr pass_costs reference_costs{
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
    .overrides = {},
};
// NOLINTEND(readability-trailing-comma)
// clang-format on

// AVX2's double profile over the frozen table, its other values written out for the same
// reason (C = 2, leaf limit 16, AVX2's Rader range).
constexpr target_profile reference_profile{
    .lanes = 2,
    .costs = &reference_costs,
    .leaf_limit = default_leaf_limit,
    .bluestein_above = 16,
    .split = true,
    .lanes_up_to = 0,
    .rader_from = 0,
    .rader_below = std::size_t{1} << 24U,
};

// The lengths in [lo, hi] that may_be_fast accepts, ascending: the products of 2, 3, 5, 7
// and 13 there, and the lone 11, 17, 19 and 23. A few thousand at most up to 2^41.
std::vector<std::size_t> fast_candidates(std::size_t lo, std::size_t hi) {
  std::vector<std::size_t> out;
  for (const std::size_t q : {11U, 17U, 19U, 23U}) {
    if (lo <= q && q <= hi) {
      out.push_back(q);
    }
  }
  for (std::size_t a = 1; a <= hi; a *= 2) {
    for (std::size_t b = a; b <= hi; b *= 3) {
      for (std::size_t c = b; c <= hi; c *= 5) {
        for (std::size_t d = c; d <= hi; d *= 7) {
          for (std::size_t e = d; e <= hi; e *= 13) {
            if (e >= lo) {
              out.push_back(e);
            }
          }
        }
      }
    }
  }
  std::ranges::sort(out);
  return out;
}

// The smallest fast length in [lo, 2 lo], lo >= 1: there is one, the power of two there.
std::size_t first_fast_from(std::size_t lo) {
  for (const std::size_t m : fast_candidates(lo, 2 * lo)) {
    if (m <= 2 || relative_row_cost(m, reference_profile) <= fast_tolerance) {
      return m;
    }
  }
  return std::bit_ceil(lo); // unreachable: the power of two is a candidate and fast
}

} // namespace

bool is_fast_length(std::size_t n) {
  if (n <= 2) {
    return true;
  }
  if (!may_be_fast(n)) {
    return false;
  }
  return relative_row_cost(n, reference_profile) <= fast_tolerance;
}

std::size_t next_fast_length(std::size_t n, bool real) {
  // Every candidate tested is a product of 2, 3, 5, 7 and 13 (or a lone 11 to 23) in
  // [n, 2n]: the call costs microseconds up to 2^40 (DESIGN.md 9.2), not a trial of every
  // integer above n.
  if (real) {
    return 2 * first_fast_from(n <= 2 ? 1 : (n + 1) / 2);
  }
  return n <= 2 ? std::max<std::size_t>(n, 1) : first_fast_from(n);
}

std::size_t bluestein_length(std::size_t q, const target_profile& p, std::size_t pow2_part_limit) {
  const std::size_t least = 2 * q - 1;
  std::size_t power = 1;
  while (power < least) {
    power *= 2;
  }
  const std::size_t top = power > pow2_part_limit ? 2 * least : power;
  // The two chirp passes touch M points each, priced like a last radix-2 pass per point.
  const std::uint64_t per_point = p.costs->unit[2][0] / (2 * p.lanes) + 1;
  // The 7-smooth lengths in [least, top], ascending, without those whose power-of-two part
  // is above pow2_part_limit.
  std::vector<std::size_t> candidates;
  for (std::size_t a = 1; a <= top; a *= 2) {
    for (std::size_t b = a; b <= top; b *= 3) {
      for (std::size_t c = b; c <= top; c *= 5) {
        for (std::size_t d = c; d <= top; d *= 7) {
          if (d >= least && a <= pow2_part_limit) {
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
