# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Strategies: program builders (DESIGN.md 8.5): the chain (with the plan-time tables of its
# Bluestein stage), the grid and the lanes.

target_sources(fftn PRIVATE bluestein.cpp chain.cpp grid.cpp lanes.cpp)
