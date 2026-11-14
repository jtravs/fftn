# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Threading (phase 3): the work graph read off a plan's program and the scheduler.

target_sources(fftn PRIVATE schedule.cpp work_graph.cpp)
