// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Turns lists of checks into GoogleTest cases: one case per subject, precision and check,
// each looping over that subject's lengths (or plane shapes). A case reports every length
// that fails, and prints the worst error/tolerance ratio it saw, which shows how much room
// the tolerances leave (run the test binary directly, or ctest -V, to see it).

#pragma once

#include "support/checks.hpp"
#include "support/subjects.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fftn::test {

struct suite_case {
  std::string name;
  std::function<void()> body;
};

inline void PrintTo(const suite_case& c, std::ostream* os) { *os << c.name; }

template <class T> std::string_view precision_name() {
  return sizeof(T) == sizeof(double) ? "double" : "float";
}

namespace detail {

// At most this many failing lengths are reported in full per case.
constexpr int reported_failures = 4;

inline void report(const std::string& label, std::size_t failures, double worst,
                   const std::string& worst_at) {
  EXPECT_EQ(failures, 0U) << label << ": " << failures << " length(s) failed";
  std::cout << "[ worst    ] " << label << ": error/tolerance " << worst << " at " << worst_at
            << "\n";
}

} // namespace detail

template <class T>
void run_row_check(std::string_view subject_name, const row_check<T>& check,
                   const std::vector<std::size_t>& lengths) {
  const auto s = make_subject<T>(subject_name);
  if (!s) {
    GTEST_SKIP() << "subject " << subject_name << " is not in this build";
  }
  if (!supports(*s, check.k)) {
    GTEST_SKIP() << "subject " << subject_name << " has no such transform";
  }
  std::size_t failures = 0;
  double worst = 0.0;
  std::string worst_at = "-";
  for (const std::size_t n : lengths) {
    const outcome o = check.run(*s, n);
    if (!o.passed() && ++failures <= detail::reported_failures) {
      ADD_FAILURE() << check.name << " fails at n = " << n << ":\n" << o.detail();
    }
    if (o.worst_ratio() > worst) {
      worst = o.worst_ratio();
      worst_at = "n = " + std::to_string(n);
    }
  }
  detail::report(std::string(check.name), failures, worst, worst_at);
}

template <class T>
void run_plane_check(std::string_view subject_name, const plane_check<T>& check,
                     const std::vector<std::pair<std::size_t, std::size_t>>& shapes) {
  const auto s = make_subject<T>(subject_name);
  if (!s) {
    GTEST_SKIP() << "subject " << subject_name << " is not in this build";
  }
  if (!s->c2c_planes) {
    GTEST_SKIP() << "subject " << subject_name << " has no planes";
  }
  std::size_t failures = 0;
  double worst = 0.0;
  std::string worst_at = "-";
  for (const auto& [ny, nx] : shapes) {
    const outcome o = check.run(*s, ny, nx);
    if (!o.passed() && ++failures <= detail::reported_failures) {
      ADD_FAILURE() << check.name << " fails at " << ny << " x " << nx << ":\n" << o.detail();
    }
    if (o.worst_ratio() > worst) {
      worst = o.worst_ratio();
      worst_at = std::to_string(ny) + " x " + std::to_string(nx);
    }
  }
  detail::report(std::string(check.name), failures, worst, worst_at);
}

// Cases for every subject, both precisions and every check in the given lists.
// `row_lengths(subject name)` gives the lengths for a subject.
template <template <class> class Checks>
std::vector<suite_case>
make_cases(const std::function<std::vector<std::size_t>(std::string_view)>& row_lengths) {
  std::vector<suite_case> cases;
  const auto add = [&]<class T>(T /*tag*/) {
    for (const std::string& subject_name : subject_names()) {
      const std::string prefix =
          std::string(subject_name) + "_" + std::string(precision_name<T>()) + "_";
      for (const auto& check : Checks<T>::rows()) {
        cases.push_back({
            prefix + std::string(check.name),
            [subject_name, check, row_lengths] {
              run_row_check<T>(subject_name, check, row_lengths(subject_name));
            },
        });
      }
      for (const auto& check : Checks<T>::planes()) {
        cases.push_back({
            prefix + std::string(check.name),
            [subject_name, check] { run_plane_check<T>(subject_name, check, plane_shapes()); },
        });
      }
    }
  };
  add(0.0);
  add(0.0F);
  return cases;
}

} // namespace fftn::test
