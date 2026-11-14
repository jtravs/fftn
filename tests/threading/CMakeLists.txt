# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Threading (tasks 3.1 to 3.3): team executions on real threads against serial execution,
# the scheduler's rules, the waiting policy, the claiming engine under a deterministic
# simulator with adversarial interleavings and virtual time, executions on executors, and
# the thread pool. The allocation hook replaces operator new in this executable only.

fftn_add_test(fftn_threading_tests
  SOURCES
    executor_test.cpp
    pool_test.cpp
    schedule_test.cpp
    sim_test.cpp
    team_test.cpp
    wait_test.cpp
  LIBRARIES fftn_test_support fftn_test_alloc_hook)
# The cache-grouping tests read the platform tests' recorded topologies.
target_compile_definitions(fftn_threading_tests PRIVATE
  FFTN_PLATFORM_DATA="${PROJECT_SOURCE_DIR}/tests/platform/data")
