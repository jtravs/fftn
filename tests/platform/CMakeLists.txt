# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Platform layer tests (task 0.2).

fftn_add_test(fftn_platform_tests
  SOURCES
    cpu_list_test.cpp
    sysctl_values_test.cpp)
