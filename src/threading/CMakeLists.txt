# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Threading (phase 3): the work graph read off a plan's program, the scheduler, the claiming
# engine behind the team overload of plan::execute, the executor overload, and the opt-in
# thread pool.

target_sources(fftn PRIVATE
  cache_groups.cpp
  executor_execute.cpp
  schedule.cpp
  team_execute.cpp
  thread_pool.cpp
  work_graph.cpp)
