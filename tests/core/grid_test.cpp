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
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
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
  fftn::shape<std::complex<T>, std::complex<T>> shape{
      .dir = dir,
      .axes = {{.n = n}},
      .batches = {{.count = count,
                   .in_dist = static_cast<std::ptrdiff_t>(dist),
                   .out_dist = static_cast<std::ptrdiff_t>(dist)}},
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

TEST(Grid, EveryTargetPrecisionDirectionAndPlacement) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    every_case<double>(t);
    every_case<float>(t);
  }
}

// The scale costs no pass of its own: the last phase's scatter applies it.
TEST(Grid, ScaleIsFusedIntoTheLastPhase) {
  build_settings settings;
  settings.strategy = strategy_choice::grid;
  const auto plan = [&](double scale) {
    return make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<double>(1024, 1, fftn::direction::forward)),
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
  s.profile.grid_min_bytes = 8192;
  s.profile.grid_three_bytes = 32768;
  s.profile.grid_block_bytes = 16384;
  const auto plan = [&](std::size_t n, const build_settings& b) {
    return make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<double>(n, 1, fftn::direction::forward)), {}, b);
  };
  EXPECT_FALSE(is_grid(*plan(512, s))); // 8192 bytes: not beyond
  EXPECT_TRUE(is_grid(*plan(1024, s))); // 16384
  const std::string two = described(*plan(2048, s));
  EXPECT_NE(two.find("in 2 phases"), std::string::npos) << two;
  const std::string three = described(*plan(4096, s));
  EXPECT_NE(three.find("in 3 phases"), std::string::npos) << three;
  EXPECT_FALSE(is_grid(*plan(1031, s)));      // prime
  EXPECT_FALSE(is_grid(*plan(1024 * 17, s))); // a factor above 13
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
  fftn::shape<std::complex<double>, std::complex<double>> strided{
      .dir = fftn::direction::forward, .axes = {{.n = 4096, .in_stride = 2}}, .batches = {}};
  EXPECT_FALSE(is_grid(*make_plan_core(fftn::detail::view_of(strided), {}, s)));
  const cvec<double> factor(4096, 1.0);
  build_settings factors = s;
  factors.out_factor = factor.data();
  EXPECT_FALSE(is_grid(*plan(4096, factors)));
  // Phases of radices up to 8 read and write rows in their end passes up to
  // grid_fuse_bytes, and copy them through gathers and scatters beyond.
  build_settings fuse = s;
  fuse.grid_factors = {64, 64};
  fuse.profile.grid_fuse_bytes = 4096 * 16;
  const std::string fused = described(*plan(4096, fuse));
  EXPECT_NE(fused.find("reads rows"), std::string::npos) << fused;
  EXPECT_EQ(fused.find("gather"), std::string::npos) << fused;
  fuse.profile.grid_fuse_bytes = 4096 * 16 - 1;
  const std::string copied = described(*plan(4096, fuse));
  EXPECT_EQ(copied.find("reads rows"), std::string::npos) << copied;
  EXPECT_NE(copied.find("gather"), std::string::npos) << copied;
}

// build_settings::scale_on_input puts the scale on phase 1's input: its first pass, or its
// gather when the phases copy their blocks.
TEST(Grid, ScaleOnInputIsPhaseOnesInput) {
  for (const bool fused : {true, false}) {
    build_settings settings;
    settings.strategy = strategy_choice::grid;
    settings.scale_on_input = true;
    settings.grid_fused = fused;
    settings.grid_factors = {8, 128}; // phase 1 a single radix-8 pass, which reads the rows
    const auto core = make_plan_core(
        fftn::detail::view_of(fftn::rows_c2c<double>(1024, 1, fftn::direction::forward)),
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
template <class T> void grid_agrees_with_chain(std::size_t n, bool three) {
  SCOPED_TRACE((sizeof(T) == 8 ? "double, n = " : "float, n = ") + std::to_string(n));
  build_settings grid;
  grid.strategy = strategy_choice::grid;
  grid.profile.grid_three_bytes = three ? 1 : std::size_t{1} << 40;
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

TEST(Grid, AgreesWithTheChainAtLengthsTheRulesTake) {
  for (const auto& [n, three] :
       {std::pair{std::size_t{32768}, false}, std::pair{std::size_t{131072}, false},
        std::pair{std::size_t{262144}, true}}) {
    grid_agrees_with_chain<double>(n, three);
    grid_agrees_with_chain<float>(n, three);
  }
}

// Padding the intermediate array (X4) moves data, not arithmetic: the result is bitwise
// the unpadded one, with phase 2 reading its padded rows through a gather or directly.
template <class T>
void padding_keeps_bits(std::size_t n, const std::vector<std::size_t>& factors, bool in_place) {
  SCOPED_TRACE((sizeof(T) == 8 ? "double, n = " : "float, n = ") + std::to_string(n) +
               (in_place ? ", in place" : ""));
  const auto shape = fftn::rows_c2c<T>(n, 2, fftn::direction::forward);
  fftn::test::input_source src(0x9ADULL + n);
  const cvec<T> x = src.complexes<T>(2 * n);
  const auto run = [&](std::size_t pad, bool fuse_padded) {
    build_settings s;
    s.strategy = strategy_choice::grid;
    s.grid_factors = factors;
    s.grid_pad = pad;
    s.grid_fuse_padded = fuse_padded;
    const auto core = make_plan_core(fftn::detail::view_of(shape), {.in_place = in_place}, s);
    EXPECT_TRUE(is_grid(*core)) << described(*core);
    cvec<T> y = in_place ? x : cvec<T>(2 * n);
    const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
    fftn::detail::execute_core(*core, 0, 2, in_place ? y.data() : x.data(), y.data(), ws);
    return y;
  };
  const cvec<T> plain = run(0, false);
  for (const std::size_t pad : {std::size_t{1}, std::size_t{8}, std::size_t{1027}}) {
    for (const bool fuse_padded : {false, true}) {
      const cvec<T> padded = run(pad, fuse_padded);
      EXPECT_EQ(std::memcmp(padded.data(), plain.data(), plain.size() * sizeof(plain[0])), 0)
          << "pad " << pad << (fuse_padded ? ", fused" : "");
    }
  }
}

// Fusing a phase's copies into its end passes moves data, not arithmetic: fused and
// unfused plans give the same bits for every forced case, both directions, the scale on
// the output and on the input, in and out of place, batches with gaps.
template <class T> void fused_equals_unfused(fftn::isa target) {
  std::vector<grid_case> cases = forced_cases();
  cases.insert(cases.end(), last_small_cases().begin(), last_small_cases().end());
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
          fftn::shape<std::complex<T>, std::complex<T>> shape{
              .dir = dir,
              .axes = {{.n = n}},
              .batches = {{.count = 2,
                           .in_dist = static_cast<std::ptrdiff_t>(dist),
                           .out_dist = static_cast<std::ptrdiff_t>(dist)}},
          };
          fftn::test::input_source src(0xF05EULL + n);
          const cvec<T> x = src.complexes<T>(dist + n);
          std::vector<cvec<T>> out;
          for (const bool fused : {true, false}) {
            build_settings s;
            s.strategy = strategy_choice::grid;
            s.grid_factors = gc.factors;
            s.profile.grid_block = gc.block;
            s.grid_fused = fused;
            s.scale_on_input = scaled == 2;
            const auto core = make_plan_core(
                fftn::detail::view_of(shape),
                {.in_place = in_place, .scale = scaled != 0 ? 0.5 : 1.0, .force_isa = target}, s);
            if (!is_grid(*core)) {
              break; // a block this target's registers do not divide (check_grid covers it)
            }
            cvec<T> y = in_place ? x : cvec<T>(dist + n, std::complex<T>(T(-1), T(2)));
            const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
            fftn::detail::execute_core(*core, 0, 2, in_place ? y.data() : x.data(), y.data(), ws);
            out.push_back(std::move(y));
          }
          if (out.size() == 2) {
            EXPECT_EQ(std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(out[0][0])),
                      0);
          }
        }
      }
    }
  }
}

TEST(Grid, PaddingKeepsBits) {
  for (const bool in_place : {false, true}) {
    padding_keeps_bits<double>(4096, {64, 64}, in_place);
    padding_keeps_bits<double>(8192, {16, 32, 16}, in_place);
    padding_keeps_bits<float>(8192, {16, 32, 16}, in_place);
    padding_keeps_bits<double>(4096, {16, 16, 16}, in_place);
    padding_keeps_bits<double>(16384, {}, in_place); // the rules' split
  }
}

TEST(Grid, FusedEqualsUnfused) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    fused_equals_unfused<double>(t);
    fused_equals_unfused<float>(t);
  }
}

TEST(Grid, SplitRules) {
  using fftn::detail::choose_grid;
  // Two phases near sqrt(n), the shorter first, block widened within the budget.
  auto g = choose_grid(1 << 16, 1, 4096, false, 0, {});
  EXPECT_EQ(g.phases, 2U);
  EXPECT_EQ(g.factors[0], 256U);
  EXPECT_EQ(g.factors[1], 256U);
  EXPECT_EQ(g.block, 16U);
  g = choose_grid(1 << 17, 1, 4096, false, 0, {});
  EXPECT_EQ(g.factors[0], 256U);
  EXPECT_EQ(g.factors[1], 512U);
  EXPECT_EQ(g.block, 8U);
  // Three phases, as even as possible, the block filling twice the budget.
  g = choose_grid(1 << 22, 1, 4096, true, 0, {});
  EXPECT_EQ(g.phases, 3U);
  EXPECT_EQ(g.factors[0], 128U);
  EXPECT_EQ(g.factors[1], 128U);
  EXPECT_EQ(g.factors[2], 256U);
  EXPECT_EQ(g.block, 32U);
  // The block is a multiple of the complex count and at least two.
  g = choose_grid(64, 8, 4096, false, 0, {});
  EXPECT_EQ(g.factors[0], 8U);
  EXPECT_EQ(g.block, 8U);
  EXPECT_EQ(choose_grid(32, 8, 4096, false, 0, {}).phases, 0U);
  EXPECT_EQ(choose_grid(2 * 1031, 1, 4096, false, 0, {}).phases, 0U);
  // Forced factors and blocks are taken as given, or refused.
  g = choose_grid(1024, 2, 4096, false, 4, {32, 32});
  EXPECT_EQ(g.block, 4U);
  EXPECT_EQ(choose_grid(1024, 2, 4096, false, 3, {32, 32}).phases, 0U);
  EXPECT_EQ(choose_grid(1024, 2, 4096, false, 0, {32, 16}).phases, 0U);
  EXPECT_EQ(choose_grid(1024, 2, 4096, false, 0, {1024}).phases, 0U);
  // A forced block is kept for a rules' split into three as well.
  g = choose_grid(1 << 22, 1, 4096, true, 8, {});
  EXPECT_EQ(g.phases, 3U);
  EXPECT_EQ(g.block, 8U);
}

} // namespace
