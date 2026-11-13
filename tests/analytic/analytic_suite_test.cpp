// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The analytic oracle suites (oracles.hpp) on every subject, in double and float: the naive
// DFT at naive_lengths() and FFTW at fast_lengths() (support/lengths.hpp). A test per
// subject, precision and check, named <subject>_<precision>_<check>.

#include "analytic/oracles.hpp"
#include "support/checks.hpp"
#include "support/subjects.hpp"
#include "support/suite_runner.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace fftn::test {
namespace {

template <class T> struct analytic_suite {
  static std::vector<row_check<T>> rows() { return analytic::row_checks<T>(); }
  static std::vector<plane_check<T>> planes() { return analytic::plane_checks<T>(); }
};

class AnalyticSuite : public ::testing::TestWithParam<suite_case> {};

TEST_P(AnalyticSuite, Holds) { GetParam().body(); }

INSTANTIATE_TEST_SUITE_P(Oracles, AnalyticSuite,
                         ::testing::ValuesIn(make_cases<analytic_suite>(&subject_lengths)),
                         [](const ::testing::TestParamInfo<suite_case>& case_info) {
                           return case_info.param.name;
                         });

} // namespace
} // namespace fftn::test
