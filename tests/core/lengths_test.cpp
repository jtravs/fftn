// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Task 2.5: next_fast_length and relative_cost (include/fftn/lengths.hpp). The fast lengths
// are a pure function of the library version (a golden list), next_fast_length is the
// smallest fast length at or above n, monotone and idempotent; the real variant is even with
// a fast half; relative_cost is 1 for powers of two and low for fast lengths on every target
// and precision, high for large primes; and the planner computes every fast length with
// codelet passes only (no Rader or Bluestein stage).

#include "fftn/fftn.hpp"
#include "fftn/lengths.hpp"
#include "planner/rules.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace d = fftn::detail;

TEST(NextFastLength, FastLengthsArePinnedForTheVersion) {
  // The reference is fixed per library version (DESIGN.md 9.2): a change to this list is a
  // change of every caller's grid sizes and belongs in the decision log.
  // Composites with 11, 17, 19 or 23 left the set after experiment P4.
  const std::vector<std::size_t> expected{
      1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
      17,  19,  20,  23,  24,  28,  32,  36,  40,  48,  56,  60,  64,  72,  80,  84,
      96,  100, 112, 120, 128, 130, 140, 144, 150, 156, 160, 162, 168, 180, 182, 192,
      196, 200, 208, 210, 216, 224, 234, 240, 252, 256, 260, 270, 280, 288, 294, 300};
  std::vector<std::size_t> fast;
  for (std::size_t n = 1; n <= 300; ++n) {
    if (d::is_fast_length(n)) {
      fast.push_back(n);
    }
  }
  EXPECT_EQ(fast, expected);
}

TEST(NextFastLength, SmallestFastLengthAtOrAbove) {
  std::size_t previous = 1;
  for (std::size_t n = 0; n <= 20000; ++n) {
    const std::size_t m = fftn::next_fast_length(n);
    ASSERT_GE(m, std::max<std::size_t>(n, 1)) << n;
    ASSERT_LE(m, std::bit_ceil(std::max<std::size_t>(n, 1))) << n;
    ASSERT_TRUE(d::is_fast_length(m)) << n;
    ASSERT_EQ(fftn::next_fast_length(m), m) << n;
    ASSERT_GE(m, previous) << n;
    previous = m;
    for (std::size_t k = std::max<std::size_t>(n, 1); k < m; ++k) {
      ASSERT_FALSE(d::is_fast_length(k)) << k;
    }
  }
}

TEST(NextFastLength, Examples) {
  EXPECT_EQ(fftn::next_fast_length(0), 1U);
  EXPECT_EQ(fftn::next_fast_length(1001), 1008U);
  EXPECT_EQ(fftn::next_fast_length(4093), 4096U);
  EXPECT_EQ(fftn::next_fast_length(99991), 100000U);
  for (std::size_t k = 0; k <= 40; ++k) {
    EXPECT_EQ(fftn::next_fast_length(std::size_t{1} << k), std::size_t{1} << k);
  }
  const std::size_t huge = (std::size_t{1} << 40U) + 1;
  EXPECT_EQ(fftn::next_fast_length(huge), huge); // beyond the search: n itself
}

TEST(NextFastLength, RealLengthsAreEvenWithAFastHalf) {
  for (std::size_t n = 3; n <= 20000; ++n) {
    const std::size_t m = fftn::next_fast_length(n, true);
    ASSERT_GE(m, n);
    ASSERT_EQ(m % 2, 0U) << n;
    ASSERT_TRUE(d::is_fast_length(m / 2)) << n;
    for (std::size_t k = n + n % 2; k < m; k += 2) {
      ASSERT_FALSE(d::is_fast_length(k / 2)) << k;
    }
  }
  EXPECT_EQ(fftn::next_fast_length(2, true), 2U);
}

std::vector<fftn::isa> every_target() {
  return {fftn::isa::scalar, fftn::isa::x86_v2, fftn::isa::avx,       fftn::isa::avx2,
          fftn::isa::avx512, fftn::isa::neon,   fftn::isa::neon_fcma, fftn::isa::best};
}

template <class T> void check_relative_costs() {
  for (const fftn::isa t : every_target()) {
    SCOPED_TRACE(std::string(fftn::isa_name(t)) + (sizeof(T) == 8 ? " double" : " float"));
    for (std::size_t k = 1; k <= 22; ++k) {
      EXPECT_DOUBLE_EQ(fftn::relative_cost<T>(std::size_t{1} << k, t), 1.0);
    }
    // Fast lengths are cheap on every target's table, not only the reference's.
    std::vector<double> fast;
    for (std::size_t n = 32; n <= 4096; ++n) {
      if (d::is_fast_length(n)) {
        fast.push_back(fftn::relative_cost<T>(n, t));
      }
    }
    std::ranges::sort(fast);
    // AVX-512 in float has eight complex values per register: lengths that 8 does not divide
    // pay for partial tiles (X16: 1.20-1.23), and its table prices radix 32 and 64, so its
    // powers of two are cheap. The reference's fast lengths are machine-independent by
    // design (DESIGN.md 9.2); relative_cost is the query that shows such a target's view.
    const bool wide = t == fftn::isa::avx512 && sizeof(T) == 4;
    EXPECT_LE(fast[fast.size() / 2], wide ? 2.5 : 1.35);
    // Large primes take a Rader or Bluestein stage and are slower per point. (Short ones are
    // not compared: the tables' first-pass costs at a few units do not price them well.)
    for (const std::size_t q : {101U, 1009U, 4093U, 65537U}) {
      EXPECT_GT(fftn::relative_cost<T>(q, t), 1.3) << q;
    }
  }
}

TEST(RelativeCost, PowersOfTwoAreOneFastLengthsLowPrimesHighOnEveryTarget) {
  check_relative_costs<double>();
  check_relative_costs<float>();
}

TEST(RelativeCost, RefusesAValueThatIsNotATarget) {
  EXPECT_THROW(static_cast<void>(fftn::relative_cost<double>(64, static_cast<fftn::isa>(200))),
               std::invalid_argument);
}

// The planner computes every fast length with codelet passes: no Rader or Bluestein stage.
TEST(NextFastLength, ThePlannerUsesCodeletPassesOnly) {
  std::vector<std::size_t> lengths;
  for (std::size_t n = 17; n <= 4096; ++n) {
    if (d::is_fast_length(n)) {
      lengths.push_back(n);
    }
  }
  for (const std::size_t n : {46080U, 100000U, 1000000U}) {
    lengths.push_back(fftn::next_fast_length(n));
  }
  for (const std::size_t n : lengths) {
    for (const bool single : {false, true}) {
      const std::string text =
          single ? fftn::c2c_plan<float>(fftn::rows_c2c<float>(n, 1, fftn::direction::forward))
                       .describe()
                 : fftn::c2c_plan<double>(fftn::rows_c2c<double>(n, 1, fftn::direction::forward))
                       .describe();
      ASSERT_EQ(text.find("bluestein"), std::string::npos) << text;
      ASSERT_EQ(text.find("rader"), std::string::npos) << text;
      ASSERT_EQ(text.find("generic"), std::string::npos) << text;
    }
  }
}

} // namespace
