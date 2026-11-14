# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Threading (tasks 3.1 and 3.2): team executions on real threads against serial execution,
# the work graph, the scheduler's rules and the waiting policy. The allocation hook replaces
# operator new in this executable only.

fftn_add_test(fftn_threading_tests
  SOURCES schedule_test.cpp team_test.cpp wait_test.cpp
  LIBRARIES fftn_test_support fftn_test_alloc_hook)
