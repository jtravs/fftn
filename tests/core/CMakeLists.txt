# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The plan core's internals: twiddle tables against the reference, the planner's radices,
# chunking (task 0.5); every pass kind and edge variant forced on every instruction set,
# and the fused scale (task 1.2); the grid strategy (task 1.7); the routes of real transforms
# (task 2.1); the lanes strategy (task 2.2); partial runs in every shape (experiment X23); the
# chain builder's narrow and pair leaf tables on every target (X28, X28b); the lanes column
# phase of planes and volumes (X40); volumes plane at a time and their scratch bound (task
# 4.2); the depth-first chain (X46). The quadruple-precision reference makes the checks
# heavy, so the tests are optimised in every build type.

fftn_add_test(fftn_core_tests SOURCES core_test.cpp passes_test.cpp grid_test.cpp
  real_test.cpp lanes_test.cpp tails_test.cpp planning_test.cpp chain_test.cpp columns_test.cpp
  lengths_test.cpp volumes_test.cpp depth_test.cpp
  LIBRARIES fftn_test_support)
fftn_always_optimise(core_test.cpp passes_test.cpp grid_test.cpp real_test.cpp lanes_test.cpp
  tails_test.cpp planning_test.cpp chain_test.cpp columns_test.cpp
  lengths_test.cpp volumes_test.cpp depth_test.cpp)
# The planner's recorded radices (data/rule_radices.txt).
target_compile_definitions(fftn_core_tests PRIVATE FFTN_CORE_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
