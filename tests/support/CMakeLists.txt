# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Shared test support (DESIGN.md 15): the transform-under-test interface, the naive and
# FFTW subjects, derived tolerances, the harness with its canary pages, the allocation hook
# and input preservation. Headers are included as "support/...".

add_library(fftn_test_support STATIC canary.cpp)
fftn_configure_target(fftn_test_support)
target_include_directories(fftn_test_support PUBLIC ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(fftn_test_support PUBLIC fftn_test_reference fftn_test_fftw)

# Replaces the global operator new and delete in every executable that links it.
add_library(fftn_test_alloc_hook OBJECT alloc_hook.cpp)
fftn_configure_target(fftn_test_alloc_hook)
# The standard library declares operator new and delete with reserved parameter names
# (__sz, __p, ...), which a replacement cannot use; clang-tidy reports the mismatch at the
# standard library's declarations, where no NOLINT can reach. Only this one-file target
# turns the check off.
if(FFTN_CLANG_TIDY)
  get_target_property(_tidy fftn_test_alloc_hook CXX_CLANG_TIDY)
  list(APPEND _tidy "--checks=-readability-inconsistent-declaration-parameter-name")
  set_target_properties(fftn_test_alloc_hook PROPERTIES CXX_CLANG_TIDY "${_tidy}")
endif()
target_include_directories(fftn_test_alloc_hook PUBLIC ${PROJECT_SOURCE_DIR}/tests)

fftn_add_test(fftn_support_tests
  SOURCES alloc_hook_test.cpp canary_test.cpp exact_test.cpp harness_test.cpp
  LIBRARIES fftn_test_support fftn_test_alloc_hook)
