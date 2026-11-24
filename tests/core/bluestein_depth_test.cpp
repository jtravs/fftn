// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A Bluestein stage whose convolution runs as two depth-first chains (DESIGN.md 8.5,
// src/strategies/chain.hpp bluestein_depth; experiment X50), forced on short primes so that
// the quadruple-precision reference stays cheap: one and two top factors, every instruction
// set the machine runs, double and float, both directions, the scale on the chirp pass out,
// batches with padded distances, and in place; the rules take it where a row of the
// convolution's length would be a depth-first chain; and where it cannot run (strides, a
// Rader prime, copies) the row keeps its Bluestein chain.

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
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;
using fftn::detail::strategy_choice;

template <class T> using cvec = std::vector<std::complex<T>>;

bool depth_convolution(const std::string& text) {
  return text.find("two depth-first chains") != std::string::npos;
}

// A batch of `count` rows of the prime q, `dist` apart (the gaps are canaries), through the
// forced depth-first convolution with the given tops (empty: the default scan) where it
// fits; every row against the reference, every gap untouched. length: the convolution's
// length (build_settings::bluestein_length; 0: the rule's). Returns whether it was built;
// lanes, when given, receives the plan's complex count C.
template <class T>
bool check_prime(fftn::isa target, std::size_t q, const std::vector<std::size_t>& tops,
                 fftn::direction dir, double scale, std::size_t count, std::size_t dist,
                 bool in_place = false, std::size_t length = 0, std::size_t* lanes = nullptr) {
  const std::string what =
      std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
      ", q = " + std::to_string(q) + ", " + std::to_string(tops.size()) + " tops given" +
      (dir == fftn::direction::forward ? ", forward" : ", backward") +
      (in_place ? ", in place" : "") + ", scale " + std::to_string(scale) + ", " +
      std::to_string(count) + " rows " + std::to_string(dist) + " apart";
  SCOPED_TRACE(what);
  build_settings settings;
  settings.strategy = strategy_choice::depth;
  settings.depth_tops = tops;
  settings.rader = 0; // Bluestein for every prime
  settings.bluestein_length = length;
  const fftn::shape<std::complex<T>, std::complex<T>> shape{
      .dir = dir,
      .axes = {{.n = q}},
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
  EXPECT_NE(text.find("bluestein " + std::to_string(q)), std::string::npos) << text;
  if (lanes != nullptr) {
    *lanes = fftn::detail::of<T>(*fftn::detail::find_kernels(core->isa_target)).complex_lanes;
  }

  fftn::test::input_source src(0xB1D3ULL + q + count);
  const std::size_t total = dist * (count - 1) + q;
  const cvec<T> x = src.complexes<T>(total);
  const std::complex<T> canary(T(-7.25), T(3.5));
  cvec<T> y(total, canary);
  if (in_place) {
    y = x;
  }
  const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  fftn::detail::execute_core(*core, 0, count, in_place ? y.data() : x.data(), y.data(), ws);

  // A Bluestein stage's error is that of two transforms of M and the chirps around them.
  const auto tol = fftn::test::derived_tolerance<T>(4 * q, fftn::test::stable_fft_bound);
  for (std::size_t b = 0; b < count; ++b) {
    std::vector<std::complex<double>> xin(q);
    for (std::size_t j = 0; j < q; ++j) {
      xin[j] = std::complex<double>(x[b * dist + j]);
    }
    const auto exact = fftn::test::reference::c2c(static_cast<int>(dir), xin);
    std::vector<fftn::test::exact_complex> want(q);
    for (std::size_t k = 0; k < q; ++k) {
      want[k] =
          fftn::test::exact_from(std::complex<double>(exact[k].re.hi, exact[k].im.hi) * scale);
    }
    fftn::test::outcome o;
    o.within(fftn::test::compare<T>(std::span<const std::complex<T>>(y.data() + b * dist, q), want),
             tol, what + ", row " + std::to_string(b));
    EXPECT_TRUE(o.passed()) << o.detail() << text;
    if (b + 1 < count) {
      for (std::size_t j = q; j < dist; ++j) {
        const std::complex<T> expect = in_place ? x[b * dist + j] : canary;
        EXPECT_EQ(y[b * dist + j], expect) << "gap element " << j << " of row " << b;
      }
    }
  }
  return depth_convolution(text);
}

template <class T> void every_case(fftn::isa target) {
  // Both directions on convolutions of 2^12 and 2^13 points (forced), which every complex
  // count splits, so that the depth-first path runs; then the profile's own lengths, whose
  // splits depend on the target (the stage is right whichever path it takes).
  for (const std::size_t q : {std::size_t{1031}, std::size_t{2053}}) {
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      EXPECT_TRUE(check_prime<T>(target, q, {}, dir, 1.0, 1, q, false, q < 2048 ? 4096 : 8192));
    }
    check_prime<T>(target, q, {}, fftn::direction::forward, 1.0, 1, q);
  }
  // The scale on the chirp pass out, batches with gaps, in place: on a convolution of 8192
  // points, which every complex count splits, so that they run the depth-first path.
  EXPECT_TRUE(
      check_prime<T>(target, 2053, {}, fftn::direction::backward, 0.25, 2, 2053 + 3, false, 8192));
  EXPECT_TRUE(
      check_prime<T>(target, 2053, {}, fftn::direction::forward, 1.0 / 3, 3, 2053 + 5, true, 8192));
}

TEST(BluesteinDepth, EveryTargetPrecisionAndDirection) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    every_case<double>(t);
    every_case<float>(t);
  }
}

// Tops given: one and two factors, on convolutions (forced) of 2^13 (tops of 8 and 16) and 10240
// (tops of 20: the radix-20 top pass with the kernel-spectrum factor forward and the fused
// chirp backward), with the scale and in place.
TEST(BluesteinDepth, GivenTops) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::vector<std::size_t>& tops : {
             std::vector<std::size_t>{16},
             std::vector<std::size_t>{8, 4},
             std::vector<std::size_t>{8, 8},
         }) {
      EXPECT_TRUE(
          check_prime<double>(t, 2053, tops, fftn::direction::forward, 1.0, 1, 2053, false, 8192));
      EXPECT_TRUE(
          check_prime<float>(t, 2053, tops, fftn::direction::backward, 0.5, 1, 2053, true, 8192));
    }
    // Tops of 20 run where C divides 20: the fused chirp passes read whole blocks of C
    // copies (chain_builder::bluestein_depth_fits: C divides the tops' product) and two tops
    // need C to divide the first (depth_fits). Every target but 8 lanes.
    for (const std::vector<std::size_t>& tops :
         {std::vector<std::size_t>{20}, std::vector<std::size_t>{20, 4}}) {
      std::size_t lanes = 1;
      const bool d = check_prime<double>(t, 2053, tops, fftn::direction::forward, 1.0, 1, 2053,
                                         false, 10240, &lanes);
      EXPECT_EQ(d, tops[0] % lanes == 0);
      const bool db = check_prime<double>(t, 2053, tops, fftn::direction::backward, 0.25, 2,
                                          2053 + 3, true, 10240, &lanes);
      EXPECT_EQ(db, tops[0] % lanes == 0);
      const bool f = check_prime<float>(t, 2053, tops, fftn::direction::backward, 0.5, 1, 2053,
                                        false, 10240, &lanes);
      EXPECT_EQ(f, tops[0] % lanes == 0);
    }
  }
}

// A plan the rules chose (M of at least 16384 points, in the Xeon profile's range, in double
// on targets with 32 vector registers; forced elsewhere), too long for the quadruple-precision
// reference, agrees with the breadth-first Bluestein stage within both their error bounds,
// both directions, in and out of place, with a scale.
template <class T>
void rules_agree_with_chain(fftn::isa target, std::size_t q, fftn::direction dir, bool in_place,
                            double scale) {
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
               ", q = " + std::to_string(q) +
               (dir == fftn::direction::forward ? " forward" : " backward") +
               (in_place ? ", in place" : "") + ", scale " + std::to_string(scale));
  const std::string name = fftn::isa_name(target);
  const bool wide = name == "avx512" || name == "neon" || name == "neon_fcma";
  build_settings depth;
  depth.rader = 0;
  depth.profile.depth_min_bytes = std::size_t{256} * 1024;
  depth.profile.depth_max_bytes = std::size_t{4} * 1024 * 1024;
  depth.profile.depth_part_bytes = 0;
  depth.profile.depth_top = 16;
  depth.profile.depth_top_alt = 20;
  depth.profile.depth_convolution_max_bytes = std::size_t{4} * 1024 * 1024;
  depth.profile.bluestein_pow2_part_bytes = SIZE_MAX;
  // 34560 = 2^8 3^3 5 points (540 KiB in double), which a top of 16 and every complex count
  // split; the rule's own length may be odd (no top fits).
  depth.bluestein_length = 34560;
  if (!wide || sizeof(T) != 8) {
    depth.strategy = strategy_choice::depth; // the rules take it in double on 32 registers
  }
  build_settings chain = depth;
  chain.strategy = strategy_choice::chain;
  const fftn::shape<std::complex<T>, std::complex<T>> shape{
      .dir = dir,
      .axes = {{.n = q}},
      .batches = {},
  };
  const fftn::options opts{.in_place = in_place, .scale = scale, .force_isa = target};
  const auto dc = make_plan_core(fftn::detail::view_of(shape), opts, depth);
  const auto cc = make_plan_core(fftn::detail::view_of(shape), opts, chain);
  const std::string text = fftn::detail::describe(*dc);
  ASSERT_TRUE(depth_convolution(text)) << text;
  ASSERT_FALSE(depth_convolution(fftn::detail::describe(*cc)));
  fftn::test::input_source src(0xB1D4ULL + q);
  const cvec<T> x = src.complexes<T>(q);
  cvec<T> yd = in_place ? x : cvec<T>(q);
  cvec<T> yc = in_place ? x : cvec<T>(q);
  const fftn::workspace wd(fftn::detail::workspace_bytes(*dc, 1));
  const fftn::workspace wc(fftn::detail::workspace_bytes(*cc, 1));
  fftn::detail::execute_core(*dc, 0, 1, in_place ? yd.data() : x.data(), yd.data(), wd);
  fftn::detail::execute_core(*cc, 0, 1, in_place ? yc.data() : x.data(), yc.data(), wc);
  double diff = 0;
  double norm = 0;
  for (std::size_t k = 0; k < q; ++k) {
    diff += std::norm(std::complex<double>(yd[k]) - std::complex<double>(yc[k]));
    norm += std::norm(std::complex<double>(yc[k]));
  }
  const auto tol = fftn::test::derived_tolerance<T>(4 * q, fftn::test::stable_fft_bound) * 2.0;
  EXPECT_LE(std::sqrt(diff / norm), tol.rms) << text;
}

TEST(BluesteinDepth, RulesPlanAgreesWithTheChain) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      for (const bool in_place : {false, true}) {
        rules_agree_with_chain<double>(t, 16411, dir, in_place, in_place ? 0.5 : 1.0);
      }
    }
    rules_agree_with_chain<float>(t, 16411, fftn::direction::forward, false, 1.0);
  }
}

// The rules: where the profile's depth-first range, with depth_convolution_max_bytes as its
// upper limit, holds the convolution's length (and the rest of depth_tops_of's conditions on a
// row of that length hold), not elsewhere; never with
// the chain forced, with a stride, or for a Rader prime.
TEST(BluesteinDepth, RulesTakeItWhereARowOfMWould) {
  const auto planned = [](std::size_t q, fftn::isa isa, bool on, strategy_choice strategy,
                          std::ptrdiff_t stride = 1, int rader = 0, bool beyond = false) {
    build_settings settings;
    settings.strategy = strategy;
    settings.rader = rader;
    settings.profile.depth_min_bytes = on ? std::size_t{256} * 1024 : 0;
    settings.profile.depth_max_bytes = on ? std::size_t{4} * 1024 * 1024 : 0;
    settings.profile.depth_part_bytes = 0;
    settings.profile.depth_top = 16;
    settings.profile.depth_top_alt = 20;
    const std::size_t convolutions = beyond ? SIZE_MAX : std::size_t{4} * 1024 * 1024;
    settings.profile.depth_convolution_max_bytes = on ? convolutions : 0;
    settings.profile.bluestein_pow2_part_bytes = SIZE_MAX;
    const fftn::shape<std::complex<double>, std::complex<double>> shape{
        .dir = fftn::direction::forward,
        .axes = {{.n = q, .in_stride = stride, .out_stride = stride}},
        .batches = {},
    };
    return fftn::detail::describe(
        *make_plan_core(fftn::detail::view_of(shape), {.force_isa = isa}, settings));
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(t));
    const std::string name = fftn::isa_name(t);
    const bool wide = name == "avx512" || name == "neon" || name == "neon_fcma";
    const std::string on = planned(16411, t, true, strategy_choice::rules);
    EXPECT_EQ(depth_convolution(on), wide) << on;
    EXPECT_FALSE(depth_convolution(planned(16411, t, false, strategy_choice::rules)));
    EXPECT_FALSE(depth_convolution(planned(16411, t, true, strategy_choice::chain)));
    EXPECT_FALSE(depth_convolution(planned(16411, t, true, strategy_choice::rules, 2)));
    EXPECT_FALSE(depth_convolution(planned(1031, t, true, strategy_choice::rules))); // M short
    // Beyond 4 MiB (M at least 2^19 points, 8 MiB) only with a larger convolutions' limit.
    EXPECT_FALSE(depth_convolution(planned(400009, t, true, strategy_choice::rules)));
    EXPECT_EQ(depth_convolution(planned(400009, t, true, strategy_choice::rules, 1, 0, true)),
              wide);
    // 65537 = 2^16 + 1 has a Rader root on every target where Rader is on.
    EXPECT_FALSE(depth_convolution(planned(65537, t, true, strategy_choice::rules, 1, 1)));
  }
}

} // namespace
