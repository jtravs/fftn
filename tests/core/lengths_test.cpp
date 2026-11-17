// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Task 2.5: next_fast_length and relative_cost (include/fftn/lengths.hpp). The fast lengths
// are a pure function of the library version (a golden list to 300 and a checksum to 2^20),
// next_fast_length is the smallest fast length at or above n, monotone and idempotent, and
// cheap up to 2^40, above which both queries throw; the real variant is even with a fast
// half; relative_cost is 1 for powers of two and low for fast lengths on every target and
// precision, high for large primes; and the planner computes every fast length with codelet
// passes only (no Rader or Bluestein stage).

#include "fftn/fftn.hpp"
#include "fftn/lengths.hpp"
#include "planner/rules.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
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
      196, 200, 208, 210, 216, 224, 234, 240, 252, 256, 260, 270, 280, 288, 294, 300,
  };
  std::vector<std::size_t> fast;
  for (std::size_t n = 1; n <= 300; ++n) {
    if (d::is_fast_length(n)) {
      fast.push_back(n);
    }
  }
  EXPECT_EQ(fast, expected);
}

TEST(NextFastLength, FastSetUpTo2To20IsPinned) {
  // The whole fast set up to 2^20 as a count and an FNV-1a hash of the lengths (8 bytes each,
  // little-endian first): the reference table is a frozen copy (planner/rules.cpp,
  // reference_costs), so a refit of the planner's tables cannot change it; a change here is a
  // change of users' grids and belongs in the decision log.
  std::uint64_t hash = 14695981039346656037ULL;
  std::size_t count = 0;
  for (std::size_t n = 1; n <= (std::size_t{1} << 20U); ++n) {
    if (d::is_fast_length(n)) {
      ++count;
      for (unsigned byte = 0; byte < 8; ++byte) {
        hash ^= (static_cast<std::uint64_t>(n) >> (8 * byte)) & 0xFFU;
        hash *= 1099511628211ULL;
      }
    }
  }
  EXPECT_EQ(count, 1879U);
  EXPECT_EQ(hash, 0xb972c9ac0cea111fULL);
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
  // The largest length accepted is 2^40, itself fast; above it both queries throw.
  const std::size_t top = std::size_t{1} << 40U;
  EXPECT_EQ(fftn::next_fast_length(top - 1), top);
  EXPECT_EQ(fftn::next_fast_length(top - 1, true), top);
  EXPECT_EQ(fftn::next_fast_length(top, true), top);
  EXPECT_THROW(static_cast<void>(fftn::next_fast_length(top + 1)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(fftn::next_fast_length(top + 1, true)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(fftn::next_fast_length(SIZE_MAX)), std::invalid_argument);
}

// Far from the golden range, the candidate search agrees with a trial of every length.
TEST(NextFastLength, LargeLengthsMatchATrialOfEveryLength) {
  for (const std::size_t n : {1000003U, 16777259U, 123456789U, 1000000007U, 1162261468U}) {
    const std::size_t m = fftn::next_fast_length(n);
    ASSERT_TRUE(d::is_fast_length(m)) << n;
    for (std::size_t k = n; k < m; ++k) {
      ASSERT_FALSE(d::is_fast_length(k)) << k;
    }
    const std::size_t r = fftn::next_fast_length(n, true);
    ASSERT_EQ(r % 2, 0U) << n;
    ASSERT_TRUE(d::is_fast_length(r / 2)) << n;
    for (std::size_t k = n + n % 2; k < r; k += 2) {
      ASSERT_FALSE(d::is_fast_length(k / 2)) << k;
    }
  }
}

// DESIGN.md 9.2: a call costs microseconds to a fraction of a millisecond up to 2^40 (it
// tests only products of 2, 3, 5, 7 and 13 in [n, 2n]). A trial of every length from n took
// 1.8 s at 2^39 + 1 and 118 ms at 10^11; the bounds below leave sanitizers and slow runners
// a wide margin.
TEST(NextFastLength, CallsAreCheapUpTo2To40) {
  using clock = std::chrono::steady_clock;
  std::vector<std::size_t> lengths{
      (std::size_t{1} << 39U) + 1,
      100000000003ULL,
      (std::size_t{1} << 40U) - 1,
  };
  for (std::size_t n = 1000; n <= (std::size_t{1} << 40U); n = n / 2 * 3 + 1) {
    lengths.push_back(n);
  }
  const clock::time_point start = clock::now();
  double slowest = 0;
  for (const std::size_t n : lengths) {
    for (const bool real : {false, true}) {
      const clock::time_point t0 = clock::now();
      static_cast<void>(fftn::next_fast_length(n, real));
      slowest = std::max(slowest, std::chrono::duration<double>(clock::now() - t0).count());
    }
  }
  const double total = std::chrono::duration<double>(clock::now() - start).count();
  EXPECT_LT(slowest, 0.25);
  EXPECT_LT(total, 2.0) << lengths.size() << " lengths";
}

TEST(NextFastLength, RealLengthsAreEvenWithAFastHalf) {
  for (std::size_t n = 0; n <= 20000; ++n) {
    const std::size_t m = fftn::next_fast_length(n, true);
    ASSERT_GE(m, n);
    ASSERT_EQ(m % 2, 0U) << n;
    ASSERT_TRUE(d::is_fast_length(m / 2)) << n;
    for (std::size_t k = std::max<std::size_t>(n + n % 2, 2); k < m; k += 2) {
      ASSERT_FALSE(d::is_fast_length(k / 2)) << k;
    }
  }
  EXPECT_EQ(fftn::next_fast_length(0, true), 2U);
  EXPECT_EQ(fftn::next_fast_length(1, true), 2U);
  EXPECT_EQ(fftn::next_fast_length(2, true), 2U);
  EXPECT_EQ(fftn::next_fast_length(3, true), 4U);
}

std::vector<fftn::isa> every_target() {
  return {
      fftn::isa::scalar, fftn::isa::x86_v2, fftn::isa::avx,       fftn::isa::avx2,
      fftn::isa::avx512, fftn::isa::neon,   fftn::isa::neon_fcma, fftn::isa::best,
  };
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
    // isa::best is the target a plan uses here (AVX-512 on some CI runners).
    const fftn::isa resolved = t == fftn::isa::best
                                   ? fftn::plan<std::complex<T>, std::complex<T>>(
                                         fftn::rows_c2c<T>(8, 1, fftn::direction::forward))
                                         .instruction_set()
                                   : t;
    const bool wide = resolved == fftn::isa::avx512 && sizeof(T) == 4;
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
  // An out-of-range value on purpose.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_THROW(static_cast<void>(fftn::relative_cost<double>(64, static_cast<fftn::isa>(200))),
               std::invalid_argument);
}

TEST(RelativeCost, RefusesLengthsAbove2To40) {
  const std::size_t top = std::size_t{1} << 40U;
  EXPECT_DOUBLE_EQ(fftn::relative_cost<double>(top), 1.0);
  EXPECT_THROW(static_cast<void>(fftn::relative_cost<double>(top + 1)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(fftn::relative_cost<float>(SIZE_MAX, fftn::isa::avx2)),
               std::invalid_argument);
}

// Lengths the planner runs as one codelet cost 1; 20 is a pass radix only (X38), so it is
// priced as the chain the planner builds for it, not as a leaf.
TEST(RelativeCost, OnlyPlannerLeavesCostOneByConstruction) {
  for (const std::size_t n : {3U, 7U, 16U, 17U, 19U, 23U}) {
    EXPECT_DOUBLE_EQ(fftn::relative_cost<double>(n, fftn::isa::avx2), 1.0) << n;
  }
  const double twenty = fftn::relative_cost<double>(20, fftn::isa::avx2);
  EXPECT_NE(twenty, 1.0);
  EXPECT_GT(twenty, 1.0);
  EXPECT_TRUE(d::is_fast_length(20)); // 1.22 on the reference table
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
