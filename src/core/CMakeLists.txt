# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The plan core: validation and layouts, programs and their driver, twiddle generation and
# the out-of-line public API (task 0.5).

target_sources(fftn PRIVATE api.cpp candidates.cpp driver.cpp edges.cpp lengths.cpp measure.cpp plan_core.cpp twiddles.cpp
  wisdom.cpp)
