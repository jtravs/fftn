# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The convenience layer (fftn/easy.hpp, step 6.6): numpy's functions over the public plan API,
# with each thread's cache of plans and scratch.

target_sources(fftn PRIVATE easy.cpp)
