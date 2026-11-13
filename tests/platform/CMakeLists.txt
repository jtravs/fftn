# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Platform layer tests (task 0.2). The readers run against recorded and synthetic fixtures
# in data/ on every OS.

set(platform_data "${CMAKE_CURRENT_SOURCE_DIR}/data")

fftn_add_test(fftn_platform_tests
  SOURCES
    cpu_features_test.cpp
    cpu_list_test.cpp
    platform_text.cpp
    sysctl_values_test.cpp
    topology_fixture_test.cpp
    topology_test.cpp)
target_compile_definitions(fftn_platform_tests PRIVATE FFTN_PLATFORM_DATA="${platform_data}")
