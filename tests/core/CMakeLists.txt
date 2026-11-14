# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The plan core's internals: twiddle tables against the reference, the planner's radices,
# chunking (task 0.5); every pass kind and edge variant forced on every instruction set,
# and the fused scale (task 1.2). The quadruple-precision reference makes the checks
# heavy, so the tests are optimised in every build type.

fftn_add_test(fftn_core_tests SOURCES core_test.cpp passes_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(core_test.cpp passes_test.cpp)
# The planner's recorded radices (data/rule_radices.txt).
target_compile_definitions(fftn_core_tests PRIVATE FFTN_CORE_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
