// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Exact values, error metrics, tolerances, inputs and lengths.

#include "support/exact.hpp"
#include "support/inputs.hpp"
#include "support/lengths.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

namespace fftn::test {
namespace {

TEST(Exact, RoundsToFloatOnceWithLoBreakingTies) {
  const double f = 1.0;
  const double half_ulp = std::ldexp(1.0, -24); // between 1 and its float successor
  // hi exactly halfway: ties to even (1.0) unless lo says otherwise.
  EXPECT_EQ(round_to<float>(exact_real{f + half_ulp, 0.0}), 1.0F);
  EXPECT_EQ(round_to<float>(exact_real{f + half_ulp, 1e-30}), std::nextafter(1.0F, 2.0F));
  EXPECT_EQ(round_to<float>(exact_real{f + half_ulp, -1e-30}), 1.0F);
  // Halfway below 1 (the ulp halves there).
  const double below = 1.0 - std::ldexp(1.0, -25);
  EXPECT_EQ(round_to<float>(exact_real{below, -1e-30}), std::nextafter(1.0F, 0.0F));
  EXPECT_EQ(round_to<float>(exact_real{below, 1e-30}), 1.0F);
  // Not a tie: lo cannot matter.
  EXPECT_EQ(round_to<float>(exact_real{0.1, 1e-18}), 0.1F);
  EXPECT_EQ(round_to<double>(exact_real{0.1, 1e-18}), 0.1);
  EXPECT_EQ(round_to<float>(exact_complex{{0.5, 0.0}, {-0.25, 0.0}}),
            std::complex<float>(0.5F, -0.25F));
}

TEST(Exact, ErrorAndUlp) {
  EXPECT_EQ(error_of(1.0, {1.0, 1e-20}), -1e-20);
  EXPECT_EQ(ulp_of<double>(1.0), std::numeric_limits<double>::epsilon());
  EXPECT_EQ(ulp_of<double>(0.75), std::numeric_limits<double>::epsilon() / 2);
  EXPECT_EQ(ulp_of<float>(1.0), static_cast<double>(std::numeric_limits<float>::epsilon()));
  EXPECT_EQ(ulp_of<float>(-3.0), 2 * static_cast<double>(std::numeric_limits<float>::epsilon()));
  EXPECT_EQ(ulp_of<double>(0.0), std::numeric_limits<double>::denorm_min());
}

TEST(Metrics, RelativeRmsAndMax) {
  const std::vector<std::complex<double>> got{{1.0, 0.0}, {0.0, 2.0}};
  const std::vector<exact_complex> want{exact_from({1.0, 0.0}), exact_from({0.0, 2.5})};
  const error_stats e = compare<double>(got, want);
  EXPECT_DOUBLE_EQ(e.rms, 0.5 / std::sqrt(1.0 + 6.25));
  EXPECT_DOUBLE_EQ(e.max, 0.5 / 2.5);
  // NaN and size mismatches can never pass.
  const std::vector<std::complex<double>> nan{{std::nan(""), 0.0}, {0.0, 2.5}};
  EXPECT_TRUE(std::isinf(compare<double>(nan, want).rms));
  EXPECT_TRUE(std::isinf(compare<double>(std::vector<std::complex<double>>(1), want).max));
}

TEST(Outcome, RecordsFailuresAndRatios) {
  outcome o;
  o.within({.rms = 1e-16, .max = 1e-16}, {.rms = 1e-15, .max = 1e-15}, "fine");
  EXPECT_TRUE(o.passed());
  EXPECT_DOUBLE_EQ(o.worst_ratio(), 0.1);
  o.within({.rms = 1e-14, .max = 0.0}, {.rms = 1e-15, .max = 1e-15}, "too big");
  EXPECT_FALSE(o.passed());
  EXPECT_NE(o.detail().find("too big"), std::string::npos);
  outcome p;
  p.merge(o);
  EXPECT_FALSE(p.passed());
  p.require(false, "also");
  EXPECT_NE(p.detail().find("also"), std::string::npos);
}

TEST(Tolerance, GrowsWithLengthAndPrecision) {
  const tolerance d1 = derived_tolerance<double>(1, stable_fft_bound);
  const tolerance d2 = derived_tolerance<double>(2, stable_fft_bound);
  const tolerance d1024 = derived_tolerance<double>(1024, stable_fft_bound);
  const tolerance f1024 = derived_tolerance<float>(1024, stable_fft_bound);
  const double eps = std::numeric_limits<double>::epsilon();
  EXPECT_DOUBLE_EQ(d1.rms, stable_fft_bound.rms * eps);
  EXPECT_DOUBLE_EQ(d1.rms, d2.rms);
  EXPECT_DOUBLE_EQ(d1024.rms, stable_fft_bound.rms * eps * std::sqrt(10.0));
  EXPECT_DOUBLE_EQ(d1024.max, stable_fft_bound.max * eps * 10.0);
  EXPECT_DOUBLE_EQ(f1024.rms / d1024.rms,
                   static_cast<double>(std::numeric_limits<float>::epsilon()) / eps);
  const tolerance sum = d1 + d2 * 2.0;
  EXPECT_DOUBLE_EQ(sum.rms, 3 * d1.rms);
}

TEST(Inputs, AreDeterministicAndExactInFloat) {
  input_source a(42);
  input_source b(42);
  const auto x = a.complexes<double>(100);
  const auto y = b.complexes<float>(100);
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(x[i], std::complex<double>(y[i]));
    EXPECT_GE(x[i].real(), -1.0);
    EXPECT_LT(x[i].real(), 1.0);
  }
  input_source c(42);
  EXPECT_EQ(c.uniform(), input_source(42).uniform());
  const auto ints = c.integer_complexes<double>(1000, 3);
  for (const auto& v : ints) {
    EXPECT_LE(std::abs(v.real()), 3.0);
    EXPECT_EQ(v.real(), std::round(v.real()));
  }
}

TEST(Lengths, CoverEverySmallLength) {
  const auto naive = naive_lengths();
  const auto fast = fast_lengths();
  const std::size_t naive_all = under_thread_sanitizer ? 32 : 64;
  const std::size_t fast_all = under_thread_sanitizer ? 64 : 512;
  for (std::size_t n = 1; n <= naive_all; ++n) {
    EXPECT_EQ(naive[n - 1], n);
  }
  for (std::size_t n = 1; n <= fast_all; ++n) {
    EXPECT_EQ(fast[n - 1], n);
  }
  EXPECT_FALSE(plane_shapes().empty());
}

} // namespace
} // namespace fftn::test
