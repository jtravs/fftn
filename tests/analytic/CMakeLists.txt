# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The analytic oracle suites (DESIGN.md 15.1, oracle 1) and the seeded-fault demonstration
# of task 0.4. The naive DFT they run is O(n^2), so these sources are optimised in every
# build type to keep the Debug (coverage) run short.

fftn_add_test(fftn_analytic_tests
  SOURCES analytic_suite_test.cpp seeded_faults_test.cpp
  LIBRARIES fftn_test_support)
fftn_always_optimise(analytic_suite_test.cpp seeded_faults_test.cpp)
