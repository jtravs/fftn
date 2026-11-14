# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The optional OpenMP adapters, fftn/omp.hpp (task 3.3): this_team() inside regions and the
# executor, bitwise against serial execution, their nesting and width rules. This executable
# links the compiler's OpenMP runtime and nothing else that brings one (cmake/FftnTestDeps.cmake
# says when it is built). The allocation hook replaces operator new in this executable only.

if(FFTN_TEST_WITH_OPENMP AND TARGET OpenMP::OpenMP_CXX)
  # libomp's start-up (__kmp_register_library_startup) at times leaks the string it registers
  # the runtime under, before any FFTN code runs in the process: LeakSanitizer reported 512
  # bytes from it once in about ninety processes under the gate's parallel load (Homebrew
  # LLVM 23), never alone. Leaks under that one libomp frame are suppressed for these tests.
  fftn_add_test(fftn_omp_tests SOURCES omp_test.cpp
    LIBRARIES fftn_test_support fftn_test_alloc_hook
    ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp")
  fftn_link_openmp(fftn_omp_tests)
endif()
