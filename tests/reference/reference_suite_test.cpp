// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The reference-comparison suite (reference_checks.hpp) on every subject, in double and
// float. Its FFTW cases are the check of FFTW against the quadruple-precision reference;
// they are skipped when the build has no FFTW (FFTN_TEST_WITH_FFTW=OFF).

#include "reference/reference_checks.hpp"
#include "support/checks.hpp"
#include "support/suite_runner.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <string_view>
#include <vector>

namespace fftn::test {
namespace {

template <class T> struct reference_suite {
  static std::vector<row_check<T>> rows() { return reference_checks::row_checks<T>(); }
  static std::vector<plane_check<T>> planes() { return reference_checks::plane_checks<T>(); }
};

std::vector<std::size_t> suite_lengths(std::string_view subject_name) {
  return reference_checks::lengths(subject_name != "naive");
}

class ReferenceSuite : public ::testing::TestWithParam<suite_case> {};

TEST_P(ReferenceSuite, Agrees) { GetParam().body(); }

INSTANTIATE_TEST_SUITE_P(Reference, ReferenceSuite,
                         ::testing::ValuesIn(make_cases<reference_suite>(&suite_lengths)),
                         [](const ::testing::TestParamInfo<suite_case>& case_info) {
                           return case_info.param.name;
                         });

} // namespace
} // namespace fftn::test
