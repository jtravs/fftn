# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Threading (phase 3): the work graph read off a plan's program, the scheduler, and the
# claiming engine behind the team overload of plan::execute.

target_sources(fftn PRIVATE schedule.cpp team_execute.cpp work_graph.cpp)
