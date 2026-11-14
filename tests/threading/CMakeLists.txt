# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Threading (tasks 3.1 and 3.2): the work graph and the scheduler's rules.

fftn_add_test(fftn_threading_tests
  SOURCES schedule_test.cpp
  LIBRARIES fftn_test_support)
