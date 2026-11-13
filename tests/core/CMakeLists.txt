# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The plan core's internals (task 0.5): twiddle tables against the reference, the chain's
# radices, chunking. The quadruple-precision reference makes the twiddle checks heavy, so
# the test is optimised in every build type.

fftn_add_test(fftn_core_tests SOURCES core_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(core_test.cpp)
