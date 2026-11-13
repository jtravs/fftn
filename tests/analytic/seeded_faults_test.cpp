// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The acceptance demonstration of task 0.4: each seeded fault of the naive DFT
// (support/naive_dft.hpp) is caught, at every length tested, by the analytic suite and,
// where the fault shows in outputs of tight layouts, by the reference suite. "Caught" means
// a check reports failure; each catching check is then run on the correct naive DFT at the
// same length and must pass, so the failure is the fault's and not the check's.
//
// Lengths: faults that change values by order one are tried at small and moderate n. A
// sign error cannot show at n <= 2 (the DFTs of both signs agree there), an output shift
// or a scaling not at n = 1, and gaps exist only in rows of two or more elements. The
// twiddle-generation error grows with n (tolerance.hpp): its RMS effect is about
// 0.29 eps n, measured at 2.4 times the RMS tolerance at n = 64 and 110 times at n = 4096,
// while at a dozen points it is within rounding and no tolerance can see it. It is tried
// from n = 64 up to n = 4096 (the largest with c2c only, to keep the O(n^2) subject fast
// under the sanitizers). The single-entry twiddle error is invisible in outputs at any n;
// the table check below catches it.

#include "analytic/oracles.hpp"
#include "reference/reference_checks.hpp"
#include "support/checks.hpp"
#include "support/lengths.hpp"
#include "support/naive_dft.hpp"
#include "support/outcome.hpp"
#include "support/suite_runner.hpp"
#include "support/transform_under_test.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fftn::test {
namespace {

struct fault_case {
  fault f;
  std::vector<std::size_t> lengths; // tried with every kind in `kinds`
  std::vector<kind> kinds;
  bool planes;                          // also tried on 2-D planes
  bool reference;                       // also shows in the reference suite (tight layouts)
  std::vector<std::size_t> c2c_lengths; // further lengths, tried with c2c only

  // A constructor rather than an aggregate, so that the table below reads as a table.
  fault_case(fault f_, std::vector<std::size_t> lengths_, std::vector<kind> kinds_, bool planes_,
             bool reference_, std::vector<std::size_t> c2c_lengths_)
      : f(f_), lengths(std::move(lengths_)), kinds(std::move(kinds_)), planes(planes_),
        reference(reference_), c2c_lengths(std::move(c2c_lengths_)) {}
};

std::vector<fault_case> fault_cases() {
  const std::vector<kind> all{kind::c2c, kind::r2c, kind::c2r};
  const std::vector<kind> c2r_only{kind::c2r};
  // clang-format off
  return {
      {fault::wrong_sign,       {3, 4, 5, 7, 8, 16, 17, 31, 64, 100, 128, 1000}, all,      true,  true,  {}},
      {fault::bin_shift,        {2, 3, 4, 5, 8, 13, 64, 100, 128, 1000},         all,      true,  true,  {}},
      {fault::scaled,           {2, 3, 4, 8, 9, 64, 1000},                       all,      true,  true,  {}},
      {fault::dropped_element,  {1, 2, 3, 4, 8, 9, 64, 1000},                    all,      true,  true,  {}},
      {fault::twiddle_base_ulp, {64, 256, 1000},                                 all,      false, true,  {4096}},
      {fault::c2r_uses_dc_imag, {1, 2, 3, 4, 7, 8, 64, 1000},                    c2r_only, false, true,  {}},
      {fault::touches_gaps,     {2, 3, 8, 64},                                   all,      true,  false, {}},
      {fault::destroys_input,   {1, 2, 3, 8, 64},                                all,      true,  true,  {}},
  };
  // clang-format on
}

void PrintTo(const fault_case& c, std::ostream* os) { *os << fault_name(c.f); }

std::vector<std::pair<std::size_t, std::size_t>> fault_plane_shapes() {
  return {{4, 4}, {3, 5}, {8, 6}};
}

// The name of the first check of kind k that reports the fault, if any.
template <class Check, class T, class... Args>
std::optional<std::string_view> first_catch(const std::vector<Check>& checks,
                                            const subject<T>& faulty, const subject<T>& correct,
                                            kind k, Args... args) {
  for (const auto& check : checks) {
    if constexpr (requires { check.k; }) {
      if (check.k != k) {
        continue;
      }
    }
    const outcome bad = check.run(faulty, args...);
    if (!bad.passed()) {
      const outcome good = check.run(correct, args...);
      EXPECT_TRUE(good.passed()) << check.name << " fails on the correct naive DFT too:\n"
                                 << good.detail();
      return check.name;
    }
  }
  return std::nullopt;
}

// Expects the catches and prints which checks made them (seen with ctest -V or by running
// the binary).
void report(const std::string& label, std::optional<std::string_view> analytic,
            std::optional<std::string_view> reference, bool reference_expected) {
  EXPECT_TRUE(analytic) << label << ": not caught by the analytic suite";
  if (reference_expected) {
    EXPECT_TRUE(reference) << label << ": not caught by the reference suite";
  }
  std::cout << "[ caught   ] " << label << ": " << analytic.value_or("-") << ", "
            << reference.value_or("-") << "\n";
}

template <class T> void demonstrate(const fault_case& c) {
  const subject<T> faulty = naive_dft<T>(c.f).as_subject();
  const subject<T> correct = naive_dft<T>().as_subject();
  const auto analytic_rows = analytic::row_checks<T>();
  const auto reference_rows = reference_checks::row_checks<T>();
  std::vector<std::pair<std::size_t, kind>> runs;
  for (const std::size_t n : c.lengths) {
    for (const kind k : c.kinds) {
      runs.emplace_back(n, k);
    }
  }
  for (const std::size_t n : c.c2c_lengths) {
    runs.emplace_back(n, kind::c2c);
  }
  for (const auto& [n, k] : runs) {
    report(describe(fault_name(c.f), ", ", precision_name<T>(), ", ", name_of(k), ", n = ", n),
           first_catch(analytic_rows, faulty, correct, k, n),
           c.reference ? first_catch(reference_rows, faulty, correct, k, n) : std::nullopt,
           c.reference);
  }
  if (c.planes) {
    const auto analytic_planes = analytic::plane_checks<T>();
    const auto reference_planes = reference_checks::plane_checks<T>();
    for (const auto& [ny, nx] : fault_plane_shapes()) {
      report(describe(fault_name(c.f), ", ", precision_name<T>(), ", planes ", ny, " x ", nx),
             first_catch(analytic_planes, faulty, correct, kind::c2c, ny, nx),
             c.reference ? first_catch(reference_planes, faulty, correct, kind::c2c, ny, nx)
                         : std::nullopt,
             c.reference);
    }
  }
}

class SeededFault : public ::testing::TestWithParam<fault_case> {};

TEST_P(SeededFault, IsCaughtInDouble) { demonstrate<double>(GetParam()); }
TEST_P(SeededFault, IsCaughtInFloat) { demonstrate<float>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Naive, SeededFault, ::testing::ValuesIn(fault_cases()),
                         [](const ::testing::TestParamInfo<fault_case>& case_info) {
                           return std::string(fault_name(case_info.param.f));
                         });

// A twiddle table entry one ulp off is caught by the entry-by-entry table check, and the
// correct table passes it, at every naive length.
template <class T> void twiddle_entry_demo() {
  const naive_dft<T> faulty(fault::twiddle_entry_ulp);
  const naive_dft<T> correct;
  for (const std::size_t n : naive_lengths()) {
    for (const direction d : {direction::forward, direction::backward}) {
      const auto good = correct.twiddles(d, n);
      const outcome ok = reference_checks::twiddle_table<T>(good, sign_of(d), n);
      EXPECT_TRUE(ok.passed()) << "correct table, n = " << n << ":\n" << ok.detail();
      if (naive_dft<T>::perturbed_entry(sign_of(d), n)) {
        const auto bad = faulty.twiddles(d, n);
        const outcome caught = reference_checks::twiddle_table<T>(bad, sign_of(d), n);
        EXPECT_FALSE(caught.passed()) << "perturbed table not caught, n = " << n;
      }
    }
  }
  // Every length from 3 on, except 4, has an entry to perturb.
  EXPECT_TRUE(naive_dft<T>::perturbed_entry(-1, 3));
  EXPECT_FALSE(naive_dft<T>::perturbed_entry(-1, 4));
}

TEST(SeededFault, TwiddleEntryUlpIsCaughtByTheTableCheckInDouble) { twiddle_entry_demo<double>(); }
TEST(SeededFault, TwiddleEntryUlpIsCaughtByTheTableCheckInFloat) { twiddle_entry_demo<float>(); }

} // namespace
} // namespace fftn::test
