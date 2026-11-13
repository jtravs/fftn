// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The FFTW comparison (fftw_checks.hpp) on every subject but FFTW itself, in double and
// float: FFTN at fast_lengths() (every n <= 512 and a sample of larger lengths) and the
// naive DFT at naive_lengths() (support/lengths.hpp), which checks the comparison itself.
// Skipped when the build has no FFTW (FFTN_TEST_WITH_FFTW=OFF).

#include "fftw/fftw_checks.hpp"
#include "support/checks.hpp"
#include "support/fftw_subject.hpp"
#include "support/subjects.hpp"
#include "support/suite_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace fftn::test {
namespace {

template <class T> struct fftw_suite {
  static std::vector<row_check<T>> rows() { return fftw_checks::row_checks<T>(); }
  static std::vector<plane_check<T>> planes() { return fftw_checks::plane_checks<T>(); }
};

// Comparing FFTW with itself would check nothing.
std::vector<suite_case> cases() {
  auto all = make_cases<fftw_suite>(&subject_lengths);
  std::erase_if(all, [](const suite_case& c) { return c.name.starts_with("fftw_"); });
  return all;
}

class FftwSuite : public ::testing::TestWithParam<suite_case> {};

TEST_P(FftwSuite, Agrees) {
  if (!fftw_subject<double>()) {
    GTEST_SKIP() << "this build has no FFTW";
  }
  GetParam().body();
}

INSTANTIATE_TEST_SUITE_P(Fftw, FftwSuite, ::testing::ValuesIn(cases()),
                         [](const ::testing::TestParamInfo<suite_case>& case_info) {
                           return case_info.param.name;
                         });

} // namespace
} // namespace fftn::test
