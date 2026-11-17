# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Platform layer tests (task 0.2). The readers run against recorded and synthetic fixtures
# in data/ on every OS; fftn_platform_dump prints what the running machine reports.

set(platform_data "${CMAKE_CURRENT_SOURCE_DIR}/data")

fftn_add_test(fftn_platform_tests
  SOURCES
    aligned_allocator_test.cpp
    aligned_buffer_test.cpp
    cpu_features_test.cpp
    cpu_list_test.cpp
    cpu_model_test.cpp
    page_memory_test.cpp
    environment_test.cpp
    file_source_test.cpp
    platform_text.cpp
    sysctl_values_test.cpp
    timer_test.cpp
    topology_fixture_test.cpp
    topology_test.cpp)
target_compile_definitions(fftn_platform_tests PRIVATE FFTN_PLATFORM_DATA="${platform_data}")

add_executable(fftn_platform_dump platform_dump.cpp platform_text.cpp)
fftn_configure_target(fftn_platform_dump)
target_include_directories(fftn_platform_dump PRIVATE ${PROJECT_SOURCE_DIR}/src
                                                      ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(fftn_platform_dump PRIVATE fftn::fftn)
# Its runs below exercise system_topology() and the FFTN_TOPOLOGY override, which need a
# fresh process each; they count towards coverage like the unit tests.
set_property(GLOBAL APPEND PROPERTY FFTN_TEST_EXECUTABLES "$<TARGET_FILE:fftn_platform_dump>")

add_test(NAME platform.dump.system COMMAND fftn_platform_dump)
set_tests_properties(platform.dump.system PROPERTIES
  ENVIRONMENT "FFTN_TOPOLOGY="
  PASS_REGULAR_EXPRESSION "best_target [a-z0-9_]+\nclock_resolution_ns [0-9]+\n.*logical_cpus [1-9]")

add_test(NAME platform.dump.override_sysfs COMMAND fftn_platform_dump)
set_tests_properties(platform.dump.override_sysfs PROPERTIES
  ENVIRONMENT "FFTN_TOPOLOGY=${platform_data}/linux-synthetic-2socket-smt.sysfs"
  PASS_REGULAR_EXPRESSION "source linux-sysfs\n.*logical_cpus 8: 0-7\n")

add_test(NAME platform.dump.override_sysctl COMMAND fftn_platform_dump)
set_tests_properties(platform.dump.override_sysctl PROPERTIES
  ENVIRONMENT "FFTN_TOPOLOGY=${platform_data}/macos-apple-m5-max.sysctl"
  PASS_REGULAR_EXPRESSION "source macos-sysctl\n.*logical_cpus 18: 0-17\n")

add_test(NAME platform.dump.override_missing COMMAND fftn_platform_dump)
set_tests_properties(platform.dump.override_missing PROPERTIES
  ENVIRONMENT "FFTN_TOPOLOGY=${platform_data}/no-such-fixture"
  PASS_REGULAR_EXPRESSION "fftn_platform_dump: fftn: cannot read the topology file")

add_test(NAME platform.dump.path
         COMMAND fftn_platform_dump ${platform_data}/linux-synthetic-big-little.sysfs)
set_tests_properties(platform.dump.path PROPERTIES
  PASS_REGULAR_EXPRESSION "capacity=1024 cores=4 logical_cpus=4 cpus=4-7\n")

add_test(NAME platform.dump.usage COMMAND fftn_platform_dump a b)
set_tests_properties(platform.dump.usage PROPERTIES
  PASS_REGULAR_EXPRESSION "usage: fftn_platform_dump")
