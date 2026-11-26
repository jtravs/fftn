// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The grid strategy (DESIGN.md 8.5, src/strategies/grid.hpp, experiment X12) forced on short
// rows so that the quadruple-precision reference stays cheap: two and three phases,
// power-of-two and smooth composite factors, every instruction set the machine runs, double
// and float, both directions, out of place and in place, batches with padded distances, and
// the scale fused into the last phase (or the first, on the input); forced blocks, narrow
// and not powers of two; the grid against the chain at the lengths the rules give it. And the
// planner's choice: a grid beyond the profile's threshold, three phases beyond the second, the
// chain elsewhere and where the row cannot be a grid; the split rules of choose_grid.

#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "isa/kernel_table.hpp"
#include "reference/reference_dft.hpp"
#include "strategies/grid.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/precisions.hpp"
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;
using fftn::detail::plan_core;
using fftn::detail::strategy_choice;

template <class T> using cvec = std::vector<std::complex<T>>;

std::string described(const plan_core& core) { return fftn::detail::describe(core); }

bool is_grid(const plan_core& core) {
  return described(core).find("strategy: grid") != std::string::npos;
}

struct grid_case {
  std::size_t n;
  std::vector<std::size_t> factors; // empty: the rules' split
  std::size_t phases;               // expected
  std::size_t block = 0;            // forced (0: the rules')
};

// A batch of `count` rows of n, `dist` apart (dist >= n; the gaps are canaries), through a
// grid plan; every row against the reference, and every element outside the rows untouched.
template <class T>
void check_grid(fftn::isa target, const grid_case& gc, fftn::direction dir, bool in_place,
                double scale, std::size_t count, std::size_t dist, bool scale_on_input = false,
                bool fused = true) {
  const std::size_t n = gc.n;
  const std::string what =
      std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
      ", n = " + std::to_string(n) + (dir == fftn::direction::forward ? " forward" : " backward") +
      (in_place ? ", in place" : "") + ", scale " + std::to_string(scale) + ", " +
      std::to_string(count) + " rows " + std::to_string(dist) + " apart";
  SCOPED_TRACE(what);
  build_settings settings;
  settings.strategy = strategy_choice::grid;
  settings.grid_factors = gc.factors;
  settings.profile.grid_block = gc.block;
  settings.scale_on_input = scale_on_input;
  settings.grid_fused = fused;
  const fftn::shape<std::complex<T>, std::complex<T>> shape{
      .dir = dir,
      .axes = {{.n = n}},
      .batches =
          {
              {
                  .count = count,
                  .in_dist = static_cast<std::ptrdiff_t>(dist),
                  .out_dist = static_cast<std::ptrdiff_t>(dist),
              },
          },
  };
  const auto core =
      make_plan_core(fftn::detail::view_of(shape),
                     {.in_place = in_place, .scale = scale, .force_isa = target}, settings);
  const std::string text = described(*core);
  const std::size_t lanes =
      fftn::detail::of<T>(*fftn::detail::find_kernels(core->isa_target)).complex_lanes;
  if (gc.block % lanes != 0) {
    EXPECT_FALSE(is_grid(*core)) << text; // a block must be whole registers
    return;
  }
  ASSERT_TRUE(is_grid(*core)) << text;
  if (gc.block != 0) {
    EXPECT_NE(text.find("in blocks of " + std::to_string(gc.block) + " columns"), std::string::npos)
        << text;
  }
  EXPECT_NE(text.find("in " + std::to_string(gc.phases) + " phases"), std::string::npos) << text;

  fftn::test::input_source src(0x6121DULL + n + count);
  const std::size_t total = dist * (count - 1) + n;
  const cvec<T> x = src.complexes<T>(total);
  const std::complex<T> canary(T(-7.25), T(3.5));
  cvec<T> y(total, canary);
  if (in_place) {
    y = x;
  }
  const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  fftn::detail::execute_core(*core, 0, count, in_place ? y.data() : x.data(), y.data(), ws);

  const auto tol = fftn::test::derived_tolerance<T>(n, fftn::test::stable_fft_bound);
  for (std::size_t b = 0; b < count; ++b) {
    std::vector<std::complex<double>> xin(n);
    for (std::size_t j = 0; j < n; ++j) {
      xin[j] = std::complex<double>(x[b * dist + j]);
    }
    const auto exact = fftn::test::reference::c2c(static_cast<int>(dir), xin);
    std::vector<fftn::test::exact_complex> want(n);
    for (std::size_t k = 0; k < n; ++k) {
      want[k] =
          fftn::test::exact_from(std::complex<double>(exact[k].re.hi, exact[k].im.hi) * scale);
    }
    fftn::test::outcome o;
    o.within(fftn::test::compare<T>(std::span<const std::complex<T>>(y.data() + b * dist, n), want),
             tol, what + ", row " + std::to_string(b));
    EXPECT_TRUE(o.passed()) << o.detail() << text;
    if (b + 1 < count) {
      for (std::size_t j = n; j < dist; ++j) {
        const std::complex<T> expect = in_place ? x[b * dist + j] : canary;
        EXPECT_EQ(y[b * dist + j], expect) << "gap element " << j << " of row " << b;
      }
    }
  }
}

// Splits that every target's complex count (1 to 8) admits: factors multiples of 8 where the
// block must divide them.
const std::vector<grid_case>& forced_cases() {
  static const std::vector<grid_case> cases{
      {.n = 256, .factors = {16, 16}, .phases = 2},
      {.n = 512, .factors = {8, 64}, .phases = 2},
      {.n = 960, .factors = {24, 40}, .phases = 2}, // 3 and 5 in the phases' chains
      {.n = 896, .factors = {56, 16}, .phases = 2}, // 7, the longer phase first
      {.n = 1024, .factors = {8, 8, 16}, .phases = 3},
      {.n = 576, .factors = {8, 6, 12}, .phases = 3}, // later factors need not be multiples
      {.n = 768, .factors = {}, .phases = 2},         // the rules' split
      // Phase 2 of three with two blocks of l (L / block = 2) at each of 8 columns c.
      {.n = 1024, .factors = {16, 8, 8}, .phases = 3, .block = 8},
      // Narrow blocks (several per phase), and blocks that are no power of two, which only
      // targets whose complex count divides them can run.
      {.n = 1024, .factors = {16, 4, 16}, .phases = 3, .block = 4},
      {.n = 512, .factors = {16, 32}, .phases = 2, .block = 2},
      {.n = 216, .factors = {6, 36}, .phases = 2, .block = 6},
      {.n = 720, .factors = {12, 6, 10}, .phases = 3, .block = 3},
      {.n = 720, .factors = {12, 6, 10}, .phases = 3, .block = 6},
  };
  return cases;
}

// Splits whose last factor is a single pass of radix at most 8 (fused at both ends).
const std::vector<grid_case>& last_small_cases() {
  static const std::vector<grid_case> cases{
      {.n = 1024, .factors = {16, 8, 8}, .phases = 3},
      {.n = 512, .factors = {64, 8}, .phases = 2},
      {.n = 1024, .factors = {8, 16, 8}, .phases = 3},
  };
  return cases;
}

template <class T> void every_case(fftn::isa target) {
  for (const grid_case& gc : forced_cases()) {
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_grid<T>(target, gc, dir, false, 1.0, 1, gc.n);
      // Unfused phases (build_settings::grid_fused off): FusedEqualsUnfused, bit for bit.
    }
  }
  // In place, a scale (on the output, and on the input), and a batch with gaps, on two and
  // three phases.
  for (const grid_case& gc : {forced_cases()[1], forced_cases()[4]}) {
    check_grid<T>(target, gc, fftn::direction::forward, true, 1.0, 1, gc.n);
    check_grid<T>(target, gc, fftn::direction::backward, true, 0.25, 2, gc.n + 3);
    check_grid<T>(target, gc, fftn::direction::forward, false, 1.0 / 3, 3, gc.n + 5);
    check_grid<T>(target, gc, fftn::direction::forward, false, 1.0 / 3, 1, gc.n, true);
  }
  // A last phase that is one pass of radix <= 8, which reads and writes the rows and so
  // carries the output scale itself (a regression: it once took the input edge's scale).
  for (const grid_case& gc : last_small_cases()) {
    check_grid<T>(target, gc, fftn::direction::forward, true, 0.5, 2, gc.n + 3);
    check_grid<T>(target, gc, fftn::direction::backward, false, 0.5, 1, gc.n);
  }
}

template <class T> class Grid : public ::testing::Test {};
TYPED_TEST_SUITE(Grid, fftn::test::precisions, fftn::test::precision_names);

TYPED_TEST(Grid, EveryTargetDirectionAndPlacement) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    every_case<TypeParam>(t);
  }
}

// The scale costs no pass of its own: the last phase's scatter applies it.
TYPED_TEST(Grid, ScaleIsFusedIntoTheLastPhase) {
  build_settings settings;
  settings.strategy = strategy_choice::grid;
  const auto plan = [&](double scale) {
    return make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<TypeParam>(1024, 1, fftn::direction::forward)),
        {.scale = scale}, settings);
  };
  const auto count = [](const plan_core& core, const char* part) {
    std::size_t stages = 0;
    std::size_t named = 0;
    for (const auto& rec : core.prog.records) {
      if (const auto* st = std::get_if<fftn::detail::stage>(&rec); st != nullptr) {
        ++stages;
        named += st->name.find(part) != std::string::npos ? 1U : 0U;
      }
    }
    return std::pair{stages, named};
  };
  const auto scaled = count(*plan(0.5), "out: scale");
  const auto plain = count(*plan(1.0), "out: scale");
  EXPECT_EQ(scaled.first, plain.first);
  EXPECT_EQ(scaled.second, 1U);
  EXPECT_EQ(plain.second, 0U);
}

// The rules: a grid beyond grid_min_bytes, three phases beyond grid_three_bytes, the chain
// below and wherever the row cannot be a grid.
TEST(Grid, PlannerChoosesByTheProfile) {
  build_settings s;
  s.profile.depth_max_bytes = 0; // the grid and chain rules alone (no depth-first chain, X46)
  s.profile.grid_min_bytes = 8192;
  s.profile.grid_three_bytes = 32768;
  s.profile.grid_block_bytes = 16384;
  s.profile.chain_part_bytes = 0;
  s.profile.chain_mixed_bytes = 0;
  s.profile.pow2_chain_bytes = 0;
  s.profile.grid_three_block = 0; // X39's fields, which the running machine may set
  s.profile.grid_two_block_bytes = 0;
  s.profile.pow2_grid_bytes = 0;
  s.profile.grid_two_pow2_bytes = 0;
  const auto plan = [&](std::size_t n, const build_settings& b) {
    return make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<double>(n, 1, fftn::direction::forward)),
        fftn::options(), b);
  };
  EXPECT_FALSE(is_grid(*plan(512, s))); // 8192 bytes: not beyond
  EXPECT_TRUE(is_grid(*plan(1024, s))); // 16384
  const std::string two = described(*plan(2048, s));
  EXPECT_NE(two.find("in 2 phases"), std::string::npos) << two;
  const std::string three = described(*plan(4096, s));
  EXPECT_NE(three.find("in 3 phases"), std::string::npos) << three;
  EXPECT_FALSE(is_grid(*plan(1031, s)));                   // prime
  EXPECT_FALSE(is_grid(*plan(std::size_t{1024} * 17, s))); // a factor above 13
  build_settings chain = s;
  chain.strategy = strategy_choice::chain;
  EXPECT_FALSE(is_grid(*plan(4096, chain)));
  // Forced radices for the whole row keep it a chain.
  build_settings radices = s;
  radices.radices = {16, 16, 16};
  EXPECT_FALSE(is_grid(*plan(4096, radices)));
  // Forced factors that do not split the row: a chain, not the rules' split.
  build_settings bad = s;
  bad.strategy = strategy_choice::grid;
  bad.grid_factors = {3, 1024 / 3};
  EXPECT_FALSE(is_grid(*plan(1024, bad)));
  // A strided row and an edge factor are chains.
  const fftn::shape<std::complex<double>, std::complex<double>> strided{
      .dir = fftn::direction::forward,
      .axes = {{.n = 4096, .in_stride = 2}},
      .batches = {},
  };
  EXPECT_FALSE(is_grid(*make_plan_core(fftn::detail::view_of(strided), {}, s)));
  const cvec<double> factor(4096, 1.0);
  build_settings factors = s;
  factors.out_factor = factor.data();
  EXPECT_FALSE(is_grid(*plan(4096, factors)));
  // Phases of radices up to 8 read and write rows in their end passes up to
  // grid_fuse_bytes, and copy them through gathers and scatters beyond.
  build_settings fuse = s;
  fuse.grid_factors = {64, 64};
  fuse.profile.grid_fuse_bytes = std::size_t{4096} * 16;
  const std::string fused = described(*plan(4096, fuse));
  EXPECT_NE(fused.find("reads rows"), std::string::npos) << fused;
  EXPECT_EQ(fused.find("gather"), std::string::npos) << fused;
  fuse.profile.grid_fuse_bytes = 4096 * 16 - 1;
  const std::string copied = described(*plan(4096, fuse));
  EXPECT_EQ(copied.find("reads rows"), std::string::npos) << copied;
  EXPECT_NE(copied.find("gather"), std::string::npos) << copied;
}

// A length that is not a power of two stays a chain beyond grid_min_bytes when its
// power-of-two part's bytes are within chain_part_bytes, or its own within
// chain_mixed_bytes (X29), the former up to chain_part_limit_bytes; powers of two are
// unaffected.
TEST(Grid, ChainKeepsLengthsThatAreNotPowersOfTwo) {
  build_settings s;
  s.profile.depth_max_bytes = 0; // the grid and chain rules alone (no depth-first chain, X46)
  s.profile.grid_min_bytes = 8192;
  s.profile.grid_three_bytes = std::size_t{1} << 30U;
  s.profile.grid_block_bytes = 16384;
  s.profile.chain_part_bytes = std::size_t{64} * 16; // power-of-two parts up to 64 points
  s.profile.chain_mixed_bytes = 0;
  s.profile.pow2_chain_bytes = 0;
  const auto plan = [&](std::size_t n) {
    return make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<double>(n, 1, fftn::direction::forward)),
        fftn::options(), s);
  };
  EXPECT_FALSE(is_grid(*plan(std::size_t{64} * 45))); // part 64
  EXPECT_TRUE(is_grid(*plan(std::size_t{128} * 45))); // part 128
  EXPECT_TRUE(is_grid(*plan(4096)));                  // a power of two
  s.profile.chain_mixed_bytes = std::size_t{128} * 45 * 16;
  EXPECT_FALSE(is_grid(*plan(std::size_t{128} * 45))); // within chain_mixed_bytes
  EXPECT_TRUE(is_grid(*plan(std::size_t{256} * 45)));
  EXPECT_TRUE(is_grid(*plan(8192)));
  s.profile.chain_mixed_bytes = 0;
  s.profile.chain_part_limit_bytes = std::size_t{64} * 45 * 16 - 1; // part 64, but beyond the limit
  EXPECT_TRUE(is_grid(*plan(std::size_t{64} * 45)));
}

// Beyond grid_three_bytes a row whose length is not a power of two and whose power-of-two
// part's bytes are within grid_two_part_bytes stays in two phases (X35); powers of two and
// rows with a larger power-of-two part take three.
TEST(Grid, SmallPowerOfTwoPartKeepsTwoPhases) {
  build_settings s;
  s.profile.depth_max_bytes = 0; // the grid and chain rules alone (no depth-first chain, X46)
  s.profile.grid_min_bytes = 8192;
  s.profile.grid_three_bytes = 32768;
  s.profile.grid_block_bytes = 16384;
  s.profile.chain_part_bytes = 0;
  s.profile.chain_mixed_bytes = 0;
  s.profile.pow2_chain_bytes = 0;
  s.profile.grid_two_part_bytes = std::size_t{16} * 16; // power-of-two parts up to 16 points
  const auto phases = [&](std::size_t n) {
    const std::string text = described(*make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<double>(n, 1, fftn::direction::forward)),
        fftn::options(), s));
    if (text.find("in 2 phases") != std::string::npos) {
      return 2;
    }
    return text.find("in 3 phases") != std::string::npos ? 3 : 0;
  };
  // The rule keeps two phases where the planner would otherwise take three, and changes
  // nothing for a larger power-of-two part or a power of two (whose phase count choose_grid
  // decides by the target's radices).
  constexpr std::size_t odd = 729;
  const int part16 = phases(16 * odd); // part 16: 186624 bytes, beyond grid_three_bytes
  const int part32 = phases(32 * odd); // part 32
  const int pow2 = phases(8192);
  EXPECT_EQ(part16, 2);
  s.profile.grid_two_part_bytes = 0;
  EXPECT_EQ(phases(32 * odd), part32);
  EXPECT_EQ(phases(8192), pow2);
  EXPECT_EQ(pow2, 3);
  EXPECT_GE(phases(16 * odd), 2);
}

// A c2c row of a power of two stays a chain beyond grid_min_bytes up to pow2_chain_bytes,
// with FFTL.jl's radices beyond grid_min_bytes: 8s up to pow2_radix8_bytes, 4s beyond
// (X32), as does the half-length transform of a real row (X33); lengths that are not
// powers of two keep their rules.
TEST(Grid, PowersOfTwoStayChainsUpToTheProfilesLimit) {
  build_settings s;
  s.profile.depth_max_bytes = 0; // the grid and chain rules alone (no depth-first chain, X46)
  s.profile.grid_min_bytes = 8192;
  s.profile.grid_three_bytes = std::size_t{1} << 30U;
  s.profile.grid_block_bytes = 16384;
  s.profile.chain_part_bytes = 0;
  s.profile.chain_mixed_bytes = 0;
  s.profile.pow2_chain_bytes = std::size_t{4096} * 16;
  s.profile.pow2_radix8_bytes = std::size_t{2048} * 16;
  const auto plan = [&](std::size_t n) {
    return make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<double>(n, 1, fftn::direction::forward)),
        fftn::options(), s);
  };
  const auto axis = [](const plan_core& core) {
    const std::string text = described(core);
    const std::size_t at = text.find("axis 0");
    return at == std::string::npos ? text : text.substr(at, text.find('\n', at) - at);
  };
  const std::string eights = axis(*plan(2048));
  EXPECT_FALSE(is_grid(*plan(2048)));
  EXPECT_NE(eights.find("4 x 8 x 8 x 8"), std::string::npos) << eights; // 2^11: one 4, three 8s
  const std::string fours = axis(*plan(4096));
  EXPECT_FALSE(is_grid(*plan(4096)));
  EXPECT_NE(fours.find("4 x 4 x 4 x 4 x 4 x 4"), std::string::npos) << fours; // 2^12: six 4s
  EXPECT_TRUE(is_grid(*plan(8192)));
  EXPECT_TRUE(is_grid(*plan(std::size_t{45} * 128))); // not a power of two
  s.profile.pow2_radix8_bytes = 0;
  const std::string odd = axis(*plan(2048));
  EXPECT_NE(odd.find("8 x 4 x 4 x 4 x 4"), std::string::npos) << odd; // 2^11: an 8, four 4s
  // Forced radices win.
  build_settings forced = s;
  forced.radices = {16, 16, 16};
  const std::string three = axis(*make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<double>(4096, 1, fftn::direction::forward)),
      fftn::options(), forced));
  EXPECT_NE(three.find("16 x 16 x 16"), std::string::npos) << three;
  // A real row's half-length transform follows the same rule (X33): 8192 reals (half length
  // 4096) a chain, 16384 a grid.
  const auto real =
      make_plan_core(fftn::detail::view_of(fftn::rows_r2c<double>(8192, 1)), fftn::options(), s);
  EXPECT_FALSE(is_grid(*real)) << described(*real);
  const auto longer =
      make_plan_core(fftn::detail::view_of(fftn::rows_r2c<double>(16384, 1)), fftn::options(), s);
  EXPECT_TRUE(is_grid(*longer)) << described(*longer);
}

// The Xeon profile's powers of two (X39): a c2c row of a power-of-two length is a grid beyond
// pow2_grid_bytes, below grid_min_bytes; other lengths and real rows keep grid_min_bytes. A
// two-phase grid's block fills grid_two_block_bytes.
TEST(Grid, PowersOfTwoBecomeGridsBeyondTheProfilesLimit) {
  build_settings s;
  s.profile.depth_max_bytes = 0; // the grid and chain rules alone (no depth-first chain, X46)
  s.profile.grid_min_bytes = std::size_t{512} * 1024;
  s.profile.grid_three_bytes = std::size_t{1} << 20U;
  s.profile.grid_block_bytes = std::size_t{32} * 1024;
  s.profile.chain_part_bytes = 0;
  s.profile.chain_mixed_bytes = 0;
  s.profile.pow2_chain_bytes = 0;
  s.profile.pow2_grid_bytes = std::size_t{256} * 1024;
  s.profile.grid_two_block_bytes = 0;
  s.profile.grid_two_pow2_bytes = 0;
  const auto plan = [&](std::size_t n) {
    return make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<double>(n, 1, fftn::direction::forward)),
        fftn::options(), s);
  };
  EXPECT_FALSE(is_grid(*plan(16384)));
  EXPECT_FALSE(is_grid(*plan(25000)));
  const auto g = plan(32768);
  EXPECT_TRUE(is_grid(*g)) << described(*g);
  // An r2c row's half length follows the rule (X39 arm D), a c2r row's does not.
  const auto r2c =
      make_plan_core(fftn::detail::view_of(fftn::rows_r2c<double>(65536, 1)), fftn::options(), s);
  EXPECT_TRUE(is_grid(*r2c)) << described(*r2c);
  const auto c2r =
      make_plan_core(fftn::detail::view_of(fftn::rows_c2r<double>(65536, 1)), fftn::options(), s);
  EXPECT_FALSE(is_grid(*c2r)) << described(*c2r);
  // Beyond grid_two_pow2_bytes a c2c power of two out of place stays in two phases, in place
  // three (X39 arm D).
  s.profile.grid_two_pow2_bytes = std::size_t{8} * 1024 * 1024;
  const std::string far = described(*plan(std::size_t{1} << 20U));
  EXPECT_NE(far.find("in 2 phases"), std::string::npos) << far;
  const std::string near = described(*plan(std::size_t{1} << 19U));
  EXPECT_NE(near.find("in 3 phases"), std::string::npos) << near;
  const std::string inplace =
      described(*make_plan_core(fftn::detail::view_of(fftn::rows_c2c<double>(
                                    std::size_t{1} << 20U, 1, fftn::direction::forward)),
                                {.in_place = true}, s));
  EXPECT_NE(inplace.find("in 3 phases"), std::string::npos) << inplace;
  s.profile.grid_two_pow2_bytes = 0;
  // Blocks of 4 columns for 256-long phases within L1, 8 within twice L1 (complex lanes 4 or
  // fewer: every target's double).
  const std::string narrow = described(*g);
  s.profile.grid_two_block_bytes = std::size_t{64} * 1024;
  const std::string wide = described(*plan(32768));
  if (narrow.find("blocks of 4 columns") != std::string::npos) {
    EXPECT_NE(wide.find("blocks of 8 columns"), std::string::npos) << wide;
  }
}

// build_settings::scale_on_input puts the scale on phase 1's input: its first pass, or its
// gather when the phases copy their blocks.
TYPED_TEST(Grid, ScaleOnInputIsPhaseOnesInput) {
  for (const bool fused : {true, false}) {
    build_settings settings;
    settings.strategy = strategy_choice::grid;
    settings.scale_on_input = true;
    settings.grid_fused = fused;
    settings.grid_factors = {8, 128}; // phase 1 a single radix-8 pass, which reads the rows
    const auto core = make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<TypeParam>(1024, 1, fftn::direction::forward)),
        {.scale = 0.5}, settings);
    std::size_t in = 0;
    std::size_t out = 0;
    std::string first;
    for (const auto& rec : core->prog.records) {
      if (const auto* st = std::get_if<fftn::detail::stage>(&rec); st != nullptr) {
        if (first.empty()) {
          first = st->name;
        }
        in += st->name.find("in: scale") != std::string::npos ? 1U : 0U;
        out += st->name.find("out: scale") != std::string::npos ? 1U : 0U;
      }
    }
    EXPECT_EQ(in, 1U);
    EXPECT_EQ(out, 0U);
    EXPECT_NE(first.find("in: scale"), std::string::npos) << first;
    EXPECT_EQ(first.find("gather") != std::string::npos, !fused) << first;
  }
}

// At the lengths the rules make grids (2^15 to 2^18, two and three phases), the grid
// agrees with the chain to within both their error bounds: too long for the
// quadruple-precision reference, which checks the chain in passes_test.
// phase_radix: the grid's phase radix (build_settings); three_block: the profile's
// three-phase block (X39).
template <class T>
void grid_agrees_with_chain(std::size_t n, bool three, std::size_t phase_radix = 0,
                            std::size_t three_block = 0) {
  SCOPED_TRACE((sizeof(T) == 8 ? "double, n = " : "float, n = ") + std::to_string(n) +
               ", phase radix " + std::to_string(phase_radix));
  build_settings grid;
  grid.strategy = strategy_choice::grid;
  grid.profile.grid_three_bytes = three ? 1 : std::size_t{1} << 40U;
  grid.grid_phase_radix = phase_radix;
  grid.profile.grid_three_block = three_block;
  build_settings chain;
  chain.strategy = strategy_choice::chain;
  const auto shape = fftn::rows_c2c<T>(n, 1, fftn::direction::forward);
  const auto g = make_plan_core(fftn::detail::view_of(shape), {}, grid);
  const auto c = make_plan_core(fftn::detail::view_of(shape), {}, chain);
  const std::string text = described(*g);
  ASSERT_TRUE(is_grid(*g)) << text;
  EXPECT_NE(text.find(three ? "in 3 phases" : "in 2 phases"), std::string::npos) << text;
  fftn::test::input_source src(0xC4A1ULL + n);
  const cvec<T> x = src.complexes<T>(n);
  cvec<T> yg(n);
  cvec<T> yc(n);
  const fftn::workspace wg(fftn::detail::workspace_bytes(*g, 1));
  const fftn::workspace wc(fftn::detail::workspace_bytes(*c, 1));
  fftn::detail::execute_core(*g, 0, 1, x.data(), yg.data(), wg);
  fftn::detail::execute_core(*c, 0, 1, x.data(), yc.data(), wc);
  double diff = 0;
  double norm = 0;
  for (std::size_t k = 0; k < n; ++k) {
    diff += std::norm(std::complex<double>(yg[k]) - std::complex<double>(yc[k]));
    norm += std::norm(std::complex<double>(yc[k]));
  }
  const auto tol = fftn::test::derived_tolerance<T>(n, fftn::test::stable_fft_bound) * 2.0;
  EXPECT_LE(std::sqrt(diff / norm), tol.rms);
}

TYPED_TEST(Grid, AgreesWithTheChainAtLengthsTheRulesTake) {
  for (const auto& [n, three] : {
           std::pair{std::size_t{32768}, false},
           std::pair{std::size_t{131072}, false},
           std::pair{std::size_t{262144}, true},
       }) {
    grid_agrees_with_chain<TypeParam>(n, three);
  }
  // Radix-32 and radix-64 phases (X3, X39's arm B) and the Xeon profile's three-phase blocks
  // of 16 (X39).
  for (const auto& [n, three] : {
           std::pair{std::size_t{131072}, true},
           std::pair{std::size_t{262144}, true},
           std::pair{std::size_t{131072}, false},
       }) {
    grid_agrees_with_chain<TypeParam>(n, three, 64, 16);
  }
}

// Two forced grids of one case that differ only in how they move data (fusion, padding)
// give the same bits: every case, both directions, the scale on the output and on the
// input, in and out of place, batches with gaps. variants[i] adjusts the settings of arm
// i; `expect` (when not empty) must appear in some arm's describe() text.
template <class T>
void equal_bits(fftn::isa target, const std::vector<grid_case>& cases,
                const std::vector<std::function<void(build_settings&)>>& variants,
                const std::string& expect = "") {
  bool seen = expect.empty();
  for (const grid_case& gc : cases) {
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      for (const int scaled : {0, 1, 2}) { // none, on the output, on the input
        for (const bool in_place : {false, true}) {
          const std::size_t n = gc.n;
          const std::size_t dist = n + 3;
          const std::string what =
              std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
              ", n = " + std::to_string(n) + ", block " + std::to_string(gc.block) +
              (dir == fftn::direction::forward ? " forward" : " backward") + ", scale " +
              std::to_string(scaled) + (in_place ? ", in place" : "");
          SCOPED_TRACE(what);
          const fftn::shape<std::complex<T>, std::complex<T>> shape{
              .dir = dir,
              .axes = {{.n = n}},
              .batches =
                  {
                      {
                          .count = 2,
                          .in_dist = static_cast<std::ptrdiff_t>(dist),
                          .out_dist = static_cast<std::ptrdiff_t>(dist),
                      },
                  },
          };
          fftn::test::input_source src(0xF05EULL + n);
          const cvec<T> x = src.complexes<T>(dist + n);
          std::vector<cvec<T>> out;
          for (const auto& variant : variants) {
            build_settings s;
            s.strategy = strategy_choice::grid;
            s.grid_factors = gc.factors;
            s.profile.grid_block = gc.block;
            s.scale_on_input = scaled == 2;
            variant(s);
            const auto core = make_plan_core(
                fftn::detail::view_of(shape),
                {.in_place = in_place, .scale = scaled != 0 ? 0.5 : 1.0, .force_isa = target}, s);
            if (!is_grid(*core)) {
              break; // a block this target's registers do not divide (check_grid covers it)
            }
            seen = seen || described(*core).find(expect) != std::string::npos;
            cvec<T> y = in_place ? x : cvec<T>(dist + n, std::complex<T>(T(-1), T(2)));
            const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
            fftn::detail::execute_core(*core, 0, 2, in_place ? y.data() : x.data(), y.data(), ws);
            out.push_back(std::move(y));
          }
          for (std::size_t i = 1; i < out.size(); ++i) {
            EXPECT_EQ(std::memcmp(out[0].data(), out[i].data(), out[0].size() * sizeof(out[0][0])),
                      0)
                << "arm " << i;
          }
        }
      }
    }
  }
  EXPECT_TRUE(seen) << "no plan says: " << expect;
}

// Fusing a phase's copies into its end passes moves data, not arithmetic.
template <class T> void fused_equals_unfused(fftn::isa target) {
  std::vector<grid_case> cases = forced_cases();
  cases.insert(cases.end(), last_small_cases().begin(), last_small_cases().end());
  equal_bits<T>(target, cases,
                {
                    [](build_settings& s) { s.grid_fused = true; },
                    [](build_settings& s) { s.grid_fused = false; },
                });
}

TYPED_TEST(Grid, FusedEqualsUnfused) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    fused_equals_unfused<TypeParam>(t);
  }
}

// Padding the intermediate (grid_pad, experiment X26b) moves data, not arithmetic: padded
// grids (the x86 machines' 4 KiB period, and a shorter one that pads more of these short
// rows) give the bits of unpadded ones, fused and not, two phases in place and three.
template <class T> void padding_keeps_bits(fftn::isa target) {
  std::vector<grid_case> cases = forced_cases();
  cases.insert(cases.end(), last_small_cases().begin(), last_small_cases().end());
  cases.push_back({.n = 4096, .factors = {64, 64}, .phases = 2});
  cases.push_back({.n = 8192, .factors = {16, 16, 32}, .phases = 3});
  constexpr std::size_t never = std::numeric_limits<std::size_t>::max();
  for (const bool fused : {true, false}) {
    equal_bits<T>(target, cases,
                  {
                      [&](build_settings& s) {
                        s.grid_fused = fused;
                        s.profile.grid_pad_period = never;
                      },
                      [&](build_settings& s) {
                        s.grid_fused = fused;
                        s.profile.grid_pad_period = 4096;
                      },
                      [&](build_settings& s) {
                        s.grid_fused = fused;
                        s.profile.grid_pad_period = 512;
                      },
                  },
                  "rows padded by");
  }
}

TYPED_TEST(Grid, PaddingKeepsBits) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    padding_keeps_bits<TypeParam>(t);
  }
}

// The pad: a multiple of the block that spreads phase 2's rows over the period's sets.
TEST(Grid, PadRule) {
  using fftn::detail::grid_pad;
  using fftn::detail::grid_split;
  constexpr std::size_t never = std::numeric_limits<std::size_t>::max();
  // The EPYC's 2^20 (64 x 128 x 128, blocks of 16): rows 128 KiB apart, one block of pad.
  const grid_split g20{.phases = 3, .factors = {64, 128, 128}, .block = 16};
  EXPECT_EQ(grid_pad(g20, 16, 4096), 16U);
  EXPECT_EQ(grid_pad(g20, 8, 4096), 16U);
  EXPECT_EQ(grid_pad(g20, 16, never), 0U);
  // The Xeon's two-phase 65536 (256 x 256, blocks of 4): rows exactly 4 KiB apart.
  const grid_split g16{.phases = 2, .factors = {256, 256, 0}, .block = 4};
  EXPECT_EQ(grid_pad(g16, 16, 4096), 4U);
  // A block that does not divide the last factor would straddle two padded rows.
  EXPECT_EQ(grid_pad({.phases = 3, .factors = {32, 36, 40}, .block = 32}, 16, 4096), 0U);
  // Rows already spread: no pad.
  EXPECT_EQ(grid_pad({.phases = 2, .factors = {8, 64, 0}, .block = 8}, 16, 4096), 0U);
  EXPECT_EQ(grid_pad({}, 16, 4096), 0U);
}

// The machine profile sets the plan's padding, and a padded grid says so.
TEST(Grid, PlannerPadsByTheProfile) {
  build_settings s;
  s.profile.depth_max_bytes = 0; // the grid and chain rules alone (no depth-first chain, X46)
  s.strategy = strategy_choice::grid;
  s.grid_factors = {64, 64};
  s.profile.grid_block = 8;
  s.profile.grid_pad_period = 4096;
  const auto shape = fftn::rows_c2c<double>(4096, 1, fftn::direction::forward);
  const auto padded =
      make_plan_core(fftn::detail::view_of(shape), {.in_place = true}, s); // in the block
  EXPECT_NE(described(*padded).find("rows padded by"), std::string::npos) << described(*padded);
  EXPECT_GT(padded->block_elements, std::size_t{4096});
  s.profile.grid_pad_period = std::numeric_limits<std::size_t>::max();
  const auto plain = make_plan_core(fftn::detail::view_of(shape), {.in_place = true}, s);
  EXPECT_EQ(described(*plain).find("rows padded by"), std::string::npos) << described(*plain);
  EXPECT_EQ(plain->block_elements, std::size_t{4096});
  // Two phases out of place keep the intermediate in the output, unpadded.
  s.profile.grid_pad_period = 4096;
  const auto out_of_place = make_plan_core(fftn::detail::view_of(shape), fftn::options(), s);
  EXPECT_EQ(described(*out_of_place).find("rows padded by"), std::string::npos);
  EXPECT_EQ(out_of_place->block_elements, 0U);
}

// A grid's scratch line b is staggered from a by seven 64-byte lines (DESIGN.md 10, rule 4;
// X4); a chain's is not.
TEST(Grid, ScratchLinesAreStaggered) {
  build_settings grid;
  grid.strategy = strategy_choice::grid;
  build_settings chain;
  chain.strategy = strategy_choice::chain;
  const auto shape = fftn::rows_c2c<double>(4096, 1, fftn::direction::forward);
  const auto g = make_plan_core(fftn::detail::view_of(shape), fftn::options(), grid);
  const auto c = make_plan_core(fftn::detail::view_of(shape), fftn::options(), chain);
  EXPECT_EQ(g->offset_b, g->scratch_line * 2 * sizeof(double) + std::size_t{7} * 64);
  EXPECT_EQ(c->offset_b, c->scratch_line * 2 * sizeof(double));
}

TEST(Grid, SplitRules) {
  using fftn::detail::choose_grid;
  // Two phases near sqrt(n), the shorter first, block widened within the budget.
  auto g = choose_grid(std::size_t{1} << 16U, 1, 4096, false, 0, {});
  EXPECT_EQ(g.phases, 2U);
  EXPECT_EQ(g.factors[0], 256U);
  EXPECT_EQ(g.factors[1], 256U);
  EXPECT_EQ(g.block, 16U);
  g = choose_grid(std::size_t{1} << 17U, 1, 4096, false, 0, {});
  EXPECT_EQ(g.factors[0], 256U);
  EXPECT_EQ(g.factors[1], 512U);
  EXPECT_EQ(g.block, 8U);
  // Three phases, as even as possible, the block filling twice the budget.
  g = choose_grid(std::size_t{1} << 22U, 1, 4096, true, 0, {});
  EXPECT_EQ(g.phases, 3U);
  EXPECT_EQ(g.factors[0], 128U);
  EXPECT_EQ(g.factors[1], 128U);
  EXPECT_EQ(g.factors[2], 256U);
  EXPECT_EQ(g.block, 32U);
  // A profile may cap the three-phase block (X39): the Xeon's 16 columns at 2^18.
  g = choose_grid(std::size_t{1} << 18U, 4, 1024, true, 0, {});
  EXPECT_EQ(g.block, 32U);
  g = choose_grid(std::size_t{1} << 18U, 4, 1024, true, 0, {}, 16);
  EXPECT_EQ(g.phases, 3U);
  EXPECT_EQ(g.block, 16U);
  // The block is a multiple of the complex count and at least two.
  g = choose_grid(64, 8, 4096, false, 0, {});
  EXPECT_EQ(g.factors[0], 8U);
  EXPECT_EQ(g.block, 8U);
  EXPECT_EQ(choose_grid(32, 8, 4096, false, 0, {}).phases, 0U);
  EXPECT_EQ(choose_grid(std::size_t{2} * 1031, 1, 4096, false, 0, {}).phases, 0U);
  // Forced factors and blocks are taken as given, or refused.
  g = choose_grid(1024, 2, 4096, false, 4, {32, 32});
  EXPECT_EQ(g.block, 4U);
  EXPECT_EQ(choose_grid(1024, 2, 4096, false, 3, {32, 32}).phases, 0U);
  EXPECT_EQ(choose_grid(1024, 2, 4096, false, 0, {32, 16}).phases, 0U);
  EXPECT_EQ(choose_grid(1024, 2, 4096, false, 0, {1024}).phases, 0U);
  // A forced block is kept for a rules' split into three as well.
  g = choose_grid(std::size_t{1} << 22U, 1, 4096, true, 8, {});
  EXPECT_EQ(g.phases, 3U);
  EXPECT_EQ(g.block, 8U);
}

} // namespace
