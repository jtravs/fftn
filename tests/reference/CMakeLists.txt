# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The quadruple-precision reference DFT (DESIGN.md 15.1, oracle 2), its own tests and the
# reference-comparison suite. Boost.Multiprecision is confined to reference_dft.cpp, which
# is optimised in every build type: unoptimised 113-bit arithmetic would make the Debug
# (coverage) suites many times slower.

add_library(fftn_test_reference STATIC reference_dft.cpp)
fftn_configure_target(fftn_test_reference)
fftn_always_optimise(reference_dft.cpp)
target_include_directories(fftn_test_reference PUBLIC ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(fftn_test_reference PRIVATE fftn_test_boost_mp)
# clang-tidy's static analyzer follows the 113-bit arithmetic into Boost.Multiprecision's
# cpp_int internals and reports paths there that Boost rules out (a shift by the full limb
# width in left_shift_generic, which eval_left_shift never sends it; an access past a limb
# array in divide.hpp). The reports are located in Boost's headers, where no NOLINT can
# reach, and nearly every path in this file runs through Boost. So this one-file target
# runs every clang-tidy check except the analyzer's.
if(FFTN_CLANG_TIDY)
  get_target_property(_tidy fftn_test_reference CXX_CLANG_TIDY)
  list(APPEND _tidy "--checks=-clang-analyzer-*")
  set_target_properties(fftn_test_reference PROPERTIES CXX_CLANG_TIDY "${_tidy}")
endif()

fftn_add_test(fftn_reference_tests
  SOURCES reference_dft_test.cpp reference_suite_test.cpp
  LIBRARIES fftn_test_support)
fftn_always_optimise(reference_suite_test.cpp)
