// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The depth-first chain (DESIGN.md 8.5, src/strategies/chain.hpp c2c_depth, src/passes/
// depth.inl; experiment X46) forced on short rows so that the quadruple-precision reference
// stays cheap: one and two top factors, power-of-two and smooth sub-rows, sub-row chains of
// two to four passes (the copy back, the alternation with scratch a, the detour through b),
// every instruction set the machine runs, double and float, both directions, the scale on
// the top pass, batches with padded distances and in place (through the workspace block);
// and where it cannot run (complex counts that do not divide the factors) the row stays a
// chain.

#include "core/candidates.hpp"
#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "isa/kernel_table.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;
using fftn::detail::strategy_choice;

template <class T> using cvec = std::vector<std::complex<T>>;

struct depth_case {
  std::size_t n;
  std::vector<std::size_t> tops;
  std::vector<std::size_t> sub; // the sub-rows' radices (empty: the rules')
};

// A batch of `count` rows of n, `dist` apart (the gaps are canaries), through the forced
// depth-first chain where it fits; every row against the reference, every gap untouched.
template <class T>
void check_depth(fftn::isa target, const depth_case& dc, fftn::direction dir, double scale,
                 std::size_t count, std::size_t dist, bool in_place = false) {
  const std::size_t n = dc.n;
  std::string tops;
  for (const std::size_t f : dc.tops) {
    tops += std::to_string(f) + " x ";
  }
  const std::string what =
      std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
      ", n = " + std::to_string(n) + " = " + tops + "..." +
      (dir == fftn::direction::forward ? " forward" : " backward") +
      (in_place ? ", in place" : "") + ", scale " + std::to_string(scale) + ", " +
      std::to_string(count) + " rows " + std::to_string(dist) + " apart";
  SCOPED_TRACE(what);
  build_settings settings;
  settings.strategy = strategy_choice::depth;
  settings.depth_tops = dc.tops;
  settings.radices = dc.sub;
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
  const std::string text = fftn::detail::describe(*core);
  const std::size_t lanes =
      fftn::detail::of<T>(*fftn::detail::find_kernels(core->isa_target)).complex_lanes;
  std::size_t subs = 1;
  for (const std::size_t f : dc.tops) {
    subs *= f;
  }
  // With one top the first pass's last block of sub-rows may be partial (X51); C divides M.
  const bool fits = (dc.tops.size() == 1 || dc.tops[0] % lanes == 0) && (n / subs) % lanes == 0;
  EXPECT_EQ(text.find("strategy: depth-first chain") != std::string::npos, fits) << text;

  fftn::test::input_source src(0xDE97ULL + n + count);
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

// Top factors that every target's complex count (1 to 8) divides, except where a case says.
const std::vector<depth_case>& cases() {
  static const std::vector<depth_case> all{
      {.n = 512, .tops = {8}, .sub = {}},
      {.n = 2048, .tops = {16}, .sub = {}},
      {.n = 960, .tops = {8}, .sub = {}},            // 3 and 5 in the sub-rows' chains
      {.n = 1024, .tops = {8}, .sub = {8, 16}},      // one pass after the first: a copy back
      {.n = 1024, .tops = {8}, .sub = {4, 4, 8}},    // two: with scratch a
      {.n = 2048, .tops = {8}, .sub = {4, 4, 4, 4}}, // three: once through b
      {.n = 1024, .tops = {8}, .sub = {4, 32}},      // a radix-32 pass in the sub-rows
      {.n = 2048, .tops = {8, 4}, .sub = {}},        // two top factors
      {.n = 1920, .tops = {8, 5}, .sub = {}},        // an odd inner top factor
      {.n = 1280, .tops = {16, 2}, .sub = {5, 8}},   // a first radix C does not divide
      {.n = 1200, .tops = {4}, .sub = {}},           // too narrow for 8 lanes
      {.n = 2000, .tops = {20}, .sub = {}},          // radix 20 at the top
      // One top that C does not divide: the first pass's last block of sub-rows is partial
      // (X51), whole blocks and one, two or three sub-rows left over.
      {.n = 1000, .tops = {5}, .sub = {}},
      {.n = 1000, .tops = {5}, .sub = {5, 8, 5}}, // a first radix C does not divide
      {.n = 1200, .tops = {6}, .sub = {}},
  };
  return all;
}

template <class T> void every_case(fftn::isa target) {
  for (const depth_case& dc : cases()) {
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_depth<T>(target, dc, dir, 1.0, 1, dc.n);
    }
  }
  // The scale on the top pass and a batch with gaps; one and two tops.
  for (const depth_case& dc : {cases()[1], cases()[7]}) {
    check_depth<T>(target, dc, fftn::direction::backward, 0.25, 2, dc.n + 3);
    check_depth<T>(target, dc, fftn::direction::forward, 1.0 / 3, 3, dc.n + 5);
  }
  // In place through the workspace block, with a scale and a batch; one and two tops.
  check_depth<T>(target, cases()[0], fftn::direction::forward, 1.0, 1, cases()[0].n, true);
  check_depth<T>(target, cases()[7], fftn::direction::backward, 0.5, 2, cases()[7].n + 3, true);
  // A partial first-pass block with the scale, a batch with gaps and in place.
  check_depth<T>(target, cases()[12], fftn::direction::backward, 0.25, 3, cases()[12].n + 7);
  check_depth<T>(target, cases()[13], fftn::direction::forward, 0.5, 2, cases()[13].n + 1, true);
}

TEST(Depth, EveryTargetPrecisionDirectionAndSplit) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    every_case<double>(t);
    every_case<float>(t);
  }
}

// The rules (experiment X46 arms A and C): in double on targets with 32 vector registers,
// rows of [depth_min_bytes, depth_max_bytes] (profile fields, off by default) whose
// power-of-two part has more than depth_part_bytes, one top of depth_top, else
// depth_top_alt; not in float, not on other targets, not outside the range, never where
// neither top fits.
TEST(Depth, RulesTakeItByTheProfile) {
  const auto planned = [](std::size_t n, fftn::isa isa, bool on, bool in_place = false) {
    build_settings settings;
    settings.profile.depth_min_bytes = on ? std::size_t{256} * 1024 : 0;
    settings.profile.depth_max_bytes = on ? std::size_t{4} * 1024 * 1024 : 0;
    settings.profile.depth_top_last = 0; // the host profile may set X51's fields
    settings.profile.depth_pow2_max_bytes = std::numeric_limits<std::size_t>::max();
    settings.profile.depth_float_max_bytes = 0;
    settings.profile.depth_part_bytes = 0;
    settings.profile.depth_top = 16;
    settings.profile.depth_top_alt = 20;
    const fftn::shape<std::complex<double>, std::complex<double>> shape{
        .dir = fftn::direction::forward,
        .axes = {{.n = n}},
        .batches = {},
    };
    return fftn::detail::describe(*make_plan_core(
        fftn::detail::view_of(shape), {.in_place = in_place, .force_isa = isa}, settings));
  };
  const auto depth = [](const std::string& text) {
    return text.find("strategy: depth-first chain") != std::string::npos;
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(t));
    const std::string name = fftn::isa_name(t);
    // 32 vector registers: avx512 and NEON. With C = 1 (NEON double) 25000 = 20 x 1250 fits.
    const bool wide = name == "avx512" || name == "neon" || name == "neon_fcma";
    EXPECT_FALSE(depth(planned(16384, t, false)));
    EXPECT_EQ(depth(planned(16384, t, true)), wide);
    EXPECT_EQ(depth(planned(65536, t, true, true)), wide); // in place
    EXPECT_FALSE(depth(planned(8192, t, true)));           // below the range
    EXPECT_FALSE(depth(planned(524288, t, true)));         // beyond it
    if (name == "avx512") {
      EXPECT_FALSE(depth(planned(25000, t, true))); // neither 16 nor 20 fits on 4 lanes
      EXPECT_NE(planned(32768, t, true).find("depth-first 16 x 2048"), std::string::npos);
      EXPECT_NE(planned(20000, t, true).find("depth-first 20 x 1000"), std::string::npos);
    }
  }
  // A third top (X51): 25000 takes 10 x 2500 on avx512 (a partial block of sub-rows in the
  // first pass), where neither 16 nor 20 fits; rows that 16 or 20 fit keep them.
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    if (std::string(fftn::isa_name(t)) != "avx512") {
      continue;
    }
    build_settings settings;
    settings.profile.depth_min_bytes = std::size_t{256} * 1024;
    settings.profile.depth_max_bytes = std::size_t{4} * 1024 * 1024;
    settings.profile.depth_pow2_max_bytes = std::numeric_limits<std::size_t>::max();
    settings.profile.depth_float_max_bytes = 0;
    settings.profile.depth_part_bytes = 0;
    settings.profile.depth_top = 16;
    settings.profile.depth_top_alt = 20;
    settings.profile.depth_top_last = 10;
    const auto text = [&](std::size_t n) {
      const fftn::shape<std::complex<double>, std::complex<double>> shape{
          .dir = fftn::direction::forward,
          .axes = {{.n = n}},
          .batches = {},
      };
      return fftn::detail::describe(
          *make_plan_core(fftn::detail::view_of(shape), {.force_isa = t}, settings));
    };
    EXPECT_NE(text(25000).find("depth-first 10 x 2500"), std::string::npos);
    EXPECT_NE(text(20000).find("depth-first 20 x 1000"), std::string::npos);
    EXPECT_NE(text(32768).find("depth-first 16 x 2048"), std::string::npos);
  }
  // A power-of-two part of at most depth_part_bytes keeps the old plan (the Apple profile).
  {
    build_settings settings;
    settings.profile.depth_min_bytes = std::size_t{64} * 1024;
    settings.profile.depth_max_bytes = std::size_t{256} * 1024;
    settings.profile.depth_top_last = 0; // the host profile may set X51's fields
    settings.profile.depth_pow2_max_bytes = std::numeric_limits<std::size_t>::max();
    settings.profile.depth_float_max_bytes = 0;
    settings.profile.depth_top_alt = 0;
    settings.profile.depth_part_bytes = std::size_t{16} * 1024;
    settings.profile.depth_top = 4;
    const auto planned_on = [&](std::size_t n, fftn::isa isa) {
      const fftn::shape<std::complex<double>, std::complex<double>> shape{
          .dir = fftn::direction::forward,
          .axes = {{.n = n}},
          .batches = {},
      };
      return depth(fftn::detail::describe(
          *make_plan_core(fftn::detail::view_of(shape), {.force_isa = isa}, settings)));
    };
    for (const fftn::isa t : fftn::test::runnable_isas()) {
      const std::string name = fftn::isa_name(t);
      // A top of 4 is taken on `avx2` too (16 registers suffice up to 8, X51).
      const bool taken =
          name == "avx512" || name == "neon" || name == "neon_fcma" || name == "avx2";
      SCOPED_TRACE(name);
      EXPECT_EQ(planned_on(8192, t), taken);  // part 8192 points: 128 KiB
      EXPECT_EQ(planned_on(12288, t), taken); // part 4096
      EXPECT_FALSE(planned_on(10000, t));     // part 16
      EXPECT_FALSE(planned_on(9216, t));      // part 1024: 16 KiB, not above
    }
  }
  // X51's limits: powers of two only up to depth_pow2_max_bytes; tops up to 8 on `avx2` too
  // (16 registers), larger tops only with 32; float up to depth_float_max_bytes.
  {
    build_settings settings;
    settings.profile.depth_min_bytes = std::size_t{128} * 1024;
    settings.profile.depth_max_bytes = std::size_t{2} * 1024 * 1024;
    settings.profile.depth_top_last = 0; // the host profile may set X51's fields
    settings.profile.depth_float_max_bytes = 0;
    settings.profile.depth_part_bytes = 0;
    settings.profile.depth_top_alt = 0;
    settings.profile.depth_pow2_max_bytes = std::size_t{256} * 1024;
    settings.profile.depth_top = 4;
    // With `forced`, the chain forced with the profile's top: whether the split fits there
    // (C divides the sub-row, and the sub-row's first radix has a depth-first pass: on `avx2`
    // 4096 starts with radix 64, which has none).
    const auto build = [&](std::size_t n, fftn::isa isa, bool single, bool forced) {
      build_settings s = settings;
      if (forced) {
        s.strategy = strategy_choice::depth;
        s.depth_tops = {s.profile.depth_top};
      }
      const fftn::options opts{.force_isa = isa};
      if (single) {
        const fftn::shape<std::complex<float>, std::complex<float>> shape{
            .dir = fftn::direction::forward,
            .axes = {{.n = n}},
            .batches = {},
        };
        return depth(
            fftn::detail::describe(*make_plan_core(fftn::detail::view_of(shape), opts, s)));
      }
      const fftn::shape<std::complex<double>, std::complex<double>> shape{
          .dir = fftn::direction::forward,
          .axes = {{.n = n}},
          .batches = {},
      };
      return depth(fftn::detail::describe(*make_plan_core(fftn::detail::view_of(shape), opts, s)));
    };
    const auto taken = [&](std::size_t n, fftn::isa isa, bool single = false) {
      return build(n, isa, single, false);
    };
    const auto fits = [&](std::size_t n, fftn::isa isa, bool single = false) {
      return build(n, isa, single, true);
    };
    for (const fftn::isa t : fftn::test::runnable_isas()) {
      const std::string name = fftn::isa_name(t);
      SCOPED_TRACE(name);
      const bool measured =
          name == "avx512" || name == "avx2" || name == "neon" || name == "neon_fcma";
      // A power of two within its limit (256 KiB), beyond it; not a power of two (768 KiB).
      EXPECT_EQ(taken(16384, t), measured && fits(16384, t));
      EXPECT_FALSE(taken(32768, t));
      EXPECT_EQ(taken(49152, t), measured && fits(49152, t));
      EXPECT_FALSE(taken(49152, t, true)); // float: off
      settings.profile.depth_float_max_bytes = std::size_t{2} * 1024 * 1024;
      EXPECT_EQ(taken(98304, t, true), measured && fits(98304, t, true)); // 768 KiB of floats
      settings.profile.depth_float_max_bytes = std::size_t{512} * 1024;   // float's own cap
      EXPECT_FALSE(taken(98304, t, true));
      EXPECT_EQ(taken(49152, t, true), measured && fits(49152, t, true)); // 384 KiB: within it
      settings.profile.depth_float_max_bytes = 0;
      settings.profile.depth_top = 16; // 16 needs 32 registers
      const bool wide = name == "avx512" || name == "neon" || name == "neon_fcma";
      EXPECT_EQ(taken(49152, t), wide && fits(49152, t));
      settings.profile.depth_top = 4;
    }
  }
  // Float keeps the old plans (unmeasured).
  build_settings settings;
  settings.profile.depth_min_bytes = std::size_t{128} * 1024;
  settings.profile.depth_max_bytes = std::size_t{4} * 1024 * 1024;
  settings.profile.depth_top_last = 0; // the host profile may set X51's fields
  settings.profile.depth_pow2_max_bytes = std::numeric_limits<std::size_t>::max();
  settings.profile.depth_float_max_bytes = 0;
  settings.profile.depth_part_bytes = 0;
  settings.profile.depth_top_alt = 0;
  const fftn::shape<std::complex<float>, std::complex<float>> shape{
      .dir = fftn::direction::forward,
      .axes = {{.n = 32768}},
      .batches = {},
  };
  EXPECT_EQ(fftn::detail::describe(*make_plan_core(fftn::detail::view_of(shape), {}, settings))
                .find("strategy: depth-first chain"),
            std::string::npos);
}

// At the lengths the rules make depth-first chains (16384 and 65536 on the Xeon's profile),
// too long for the quadruple-precision reference, the chain built with the default tops
// (forced, no tops given: the scan over codelet radices) and with the rules' tops (on targets
// with 32 vector registers) agrees with the plain chain within both their error bounds, in
// both directions, in and out of place; the top pass's derived twiddles then step through
// many blocks of k.
template <class T>
void depth_agrees_with_chain(fftn::isa target, std::size_t n, fftn::direction dir, bool in_place,
                             bool rules, const std::vector<std::size_t>& tops = {}) {
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
               ", n = " + std::to_string(n) +
               (dir == fftn::direction::forward ? " forward" : " backward") +
               (in_place ? ", in place" : "") + (rules ? ", the rules' tops" : ", default tops"));
  build_settings depth;
  if (rules) {
    depth.profile.depth_min_bytes = std::size_t{256} * 1024;
    depth.profile.depth_max_bytes = std::size_t{4} * 1024 * 1024;
    depth.profile.depth_top_last = 0; // the host profile may set X51's fields
    depth.profile.depth_pow2_max_bytes = std::numeric_limits<std::size_t>::max();
    depth.profile.depth_float_max_bytes = 0;
    depth.profile.depth_part_bytes = 0;
    depth.profile.depth_top = 16;
    depth.profile.depth_top_alt = 20;
  } else {
    depth.strategy = strategy_choice::depth;
    depth.depth_tops = tops;
  }
  build_settings chain;
  chain.strategy = strategy_choice::chain;
  const fftn::shape<std::complex<T>, std::complex<T>> shape{
      .dir = dir,
      .axes = {{.n = n}},
      .batches = {},
  };
  const fftn::options opts{.in_place = in_place, .force_isa = target};
  const auto dc = make_plan_core(fftn::detail::view_of(shape), opts, depth);
  const auto cc = make_plan_core(fftn::detail::view_of(shape), opts, chain);
  const std::string text = fftn::detail::describe(*dc);
  ASSERT_NE(text.find("strategy: depth-first chain"), std::string::npos) << text;
  fftn::test::input_source src(0xDEEDULL + n);
  const cvec<T> x = src.complexes<T>(n);
  cvec<T> yd = in_place ? x : cvec<T>(n);
  cvec<T> yc = in_place ? x : cvec<T>(n);
  const fftn::workspace wd(fftn::detail::workspace_bytes(*dc, 1));
  const fftn::workspace wc(fftn::detail::workspace_bytes(*cc, 1));
  fftn::detail::execute_core(*dc, 0, 1, in_place ? yd.data() : x.data(), yd.data(), wd);
  fftn::detail::execute_core(*cc, 0, 1, in_place ? yc.data() : x.data(), yc.data(), wc);
  double diff = 0;
  double norm = 0;
  for (std::size_t k = 0; k < n; ++k) {
    diff += std::norm(std::complex<double>(yd[k]) - std::complex<double>(yc[k]));
    norm += std::norm(std::complex<double>(yc[k]));
  }
  const auto tol = fftn::test::derived_tolerance<T>(n, fftn::test::stable_fft_bound) * 2.0;
  EXPECT_LE(std::sqrt(diff / norm), tol.rms) << text;
}

TEST(Depth, AgreesWithTheChainAtLengthsTheRulesTake) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    const std::string name = fftn::isa_name(t);
    const bool wide = name == "avx512" || name == "neon" || name == "neon_fcma";
    for (const std::size_t n : {std::size_t{16384}, std::size_t{65536}}) {
      for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
        for (const bool in_place : {false, true}) {
          depth_agrees_with_chain<double>(t, n, dir, in_place, false);
          if (wide) {
            depth_agrees_with_chain<double>(t, n, dir, in_place, true);
          }
        }
      }
      depth_agrees_with_chain<float>(t, n, fftn::direction::forward, false, false);
    }
    // Tops that C does not divide (X51): 25000 = 10 x 2500 and 5 x 5000 in double, 20000 =
    // 10 x 2000 in float (C divides the sub-rows on every target).
    for (const std::size_t top : {std::size_t{10}, std::size_t{5}}) {
      depth_agrees_with_chain<double>(t, 25000, fftn::direction::forward, false, false, {top});
    }
    depth_agrees_with_chain<double>(t, 25000, fftn::direction::backward, true, false, {10});
    depth_agrees_with_chain<float>(t, 20000, fftn::direction::forward, false, false, {10});
  }
}

// A plan records its tops (plan_core::program_text, measured planning's pinned choice), and
// replaying the record builds the same program: forced tops, and tops the rules chose.
TEST(Depth, PinnedChoiceCarriesTheTops) {
  const fftn::shape<std::complex<double>, std::complex<double>> shape{
      .dir = fftn::direction::forward,
      .axes = {{.n = 4096}},
      .batches = {},
  };
  build_settings forced;
  forced.strategy = strategy_choice::depth;
  forced.depth_tops = {8, 4};
  const auto core = make_plan_core(fftn::detail::view_of(shape), fftn::options(), forced);
  EXPECT_EQ(core->program_text, "strategy=depth tops=8.4");
  const std::optional<fftn::detail::plan_choice> back =
      fftn::detail::parse_choice(core->program_text);
  if (!back.has_value()) {
    FAIL() << "the record does not parse: " << core->program_text;
  }
  build_settings replay;
  fftn::detail::apply_choice(*back, replay);
  const auto again = make_plan_core(fftn::detail::view_of(shape), fftn::options(), replay);
  EXPECT_EQ(fftn::detail::describe(*again), fftn::detail::describe(*core));
  // The rules' choice, recorded with its tops, and the given tops replacing the rules' own.
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    const std::string name = fftn::isa_name(t);
    if (name != "avx512" && name != "neon" && name != "neon_fcma") {
      continue;
    }
    SCOPED_TRACE(name);
    build_settings rules;
    rules.profile.depth_min_bytes = std::size_t{256} * 1024;
    rules.profile.depth_max_bytes = std::size_t{4} * 1024 * 1024;
    rules.profile.depth_top_last = 0; // the host profile may set X51's fields
    rules.profile.depth_pow2_max_bytes = std::numeric_limits<std::size_t>::max();
    rules.profile.depth_float_max_bytes = 0;
    rules.profile.depth_part_bytes = 0;
    rules.profile.depth_top = 16;
    rules.profile.depth_top_alt = 20;
    const auto longer = fftn::shape<std::complex<double>, std::complex<double>>{
        .dir = fftn::direction::forward,
        .axes = {{.n = 32768}},
        .batches = {},
    };
    const auto r = make_plan_core(fftn::detail::view_of(longer), {.force_isa = t}, rules);
    EXPECT_EQ(r->program_text, "strategy=depth tops=16");
    rules.depth_tops = {8, 8};
    const auto given = make_plan_core(fftn::detail::view_of(longer), {.force_isa = t}, rules);
    EXPECT_NE(fftn::detail::describe(*given).find("depth-first 8 x 8 x 512"), std::string::npos);
    EXPECT_EQ(given->program_text, "strategy=depth tops=8.8");
  }
}

} // namespace
