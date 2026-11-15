// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The Bluestein stage (DESIGN.md 8.5, 9.2) at the lengths that take it, beyond the suites'
// every n <= 512 (fast_lengths):
//
// - every n <= 4096 with a prime factor above 16, c2c in both directions, in double and
//   float, against FFTW (skipped without it): on the instruction set plans choose by
//   default; with FFTN_TEST_LONG=1 (lengths.hpp) also on every other instruction set the
//   machine runs, and r2c and c2r too. Plan construction dominates the cost (each plan
//   computes its chirp, kernel spectrum and twiddles in double-double), which is why the
//   gate's run is the default one.
// - the large prime 65537 against the quadruple-precision reference at sampled bins, and
//   2 x 3 x 65537 against FFTW; with FFTN_TEST_LONG=1 both against the reference, on every
//   instruction set.
//
// The tolerances are Bluestein's (support/tolerance.hpp), and every run checks the memory
// contracts (harness.hpp).

#include "fftn/isa.hpp"
#include "fftw/fftw_checks.hpp"
#include "reference/reference_checks.hpp"
#include "support/fftn_subject.hpp"
#include "support/fftw_subject.hpp"
#include "support/lengths.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"
#include "support/transform_under_test.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace fftn::test {
namespace {

// Under ThreadSanitizer the lengths up to 4096 would take many minutes (lengths.hpp); there
// the first 512 do.
std::size_t last_length() { return under_thread_sanitizer ? 512 : 4096; }

std::vector<fftn::isa> isas() {
  return long_tests() ? runnable_isas() : std::vector<fftn::isa>{fftn::isa::best};
}

template <class T> void every_length_against_fftw(fftn::isa target, std::size_t first,
                                                  std::size_t last) {
  const subject<T> s = fftn_subject<T>(target);
  std::size_t failures = 0;
  for (std::size_t n = first; n <= last; ++n) {
    if (!has_large_prime_factor(n)) {
      continue;
    }
    for (const auto& check : fftw_checks::row_checks<T>()) {
      if (check.k != kind::c2c && !long_tests()) {
        continue;
      }
      const outcome o = check.run(s, n);
      if (!o.passed() && ++failures <= 4) {
        ADD_FAILURE() << s.name << (sizeof(T) == 8 ? " double " : " float ") << check.name
                      << ", n = " << n << ":\n"
                      << o.detail();
      }
    }
  }
  EXPECT_EQ(failures, 0U) << s.name;
}

// The lengths in eight ranges, so that CTest runs them in parallel.
class BluesteinLengths : public ::testing::TestWithParam<std::size_t> {};

TEST_P(BluesteinLengths, AgainstFftw) {
  if (!fftw_subject<double>()) {
    GTEST_SKIP() << "this build has no FFTW";
  }
  const std::size_t part = last_length() / 8;
  const std::size_t first = GetParam() * part + 1;
  const std::size_t last = first + part - 1;
  for (const fftn::isa t : isas()) {
    every_length_against_fftw<double>(t, first, last);
    every_length_against_fftw<float>(t, first, last);
  }
}

INSTANTIATE_TEST_SUITE_P(EveryLengthUpTo4096, BluesteinLengths,
                         ::testing::Range(std::size_t{0}, std::size_t{8}));

template <class T> void large_lengths(fftn::isa target) {
  const subject<T> s = fftn_subject<T>(target);
  const char* precision = sizeof(T) == 8 ? " double" : " float";
  const outcome prime = reference_checks::c2c(s, 65537);
  EXPECT_TRUE(prime.passed()) << s.name << precision << ", n = 65537:\n" << prime.detail();
  constexpr std::size_t composite = std::size_t{2} * 3 * 65537;
  if (long_tests() || !fftw_subject<T>()) {
    const outcome o = reference_checks::c2c(s, composite);
    EXPECT_TRUE(o.passed()) << s.name << precision << ", n = " << composite << ":\n" << o.detail();
  } else {
    const outcome o = fftw_checks::c2c(s, composite);
    EXPECT_TRUE(o.passed()) << s.name << precision << ", n = " << composite << ":\n" << o.detail();
  }
}

TEST(Bluestein, LargePrimes) {
  for (const fftn::isa t : isas()) {
    large_lengths<double>(t);
    large_lengths<float>(t);
  }
}

} // namespace
} // namespace fftn::test
