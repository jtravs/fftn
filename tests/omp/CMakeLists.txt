# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The optional OpenMP adapters, fftn/omp.hpp (task 3.3): this_team() inside regions and the
# executor, bitwise against serial execution, their nesting and width rules. This executable
# links the compiler's OpenMP runtime and nothing else that brings one (cmake/FftnTestDeps.cmake
# says when it is built). The allocation hook replaces operator new in this executable only.

if(FFTN_TEST_WITH_OPENMP AND TARGET OpenMP::OpenMP_CXX)
  # LSAN_OPTIONS names the shared suppressions (libomp's own allocations; the file says why):
  # a test's ENVIRONMENT replaces the gate's LSAN_OPTIONS rather than adding to it.
  fftn_add_test(fftn_omp_tests SOURCES omp_test.cpp
    LIBRARIES fftn_test_support fftn_test_alloc_hook
    ENVIRONMENT "LSAN_OPTIONS=suppressions=${PROJECT_SOURCE_DIR}/tests/support/lsan.supp")
  fftn_link_openmp(fftn_omp_tests)
endif()
