// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The members per cache group of a team, estimated from the topology (fftn::cache_group_size
// in include/fftn/team.hpp): the grouping the OpenMP adapters, the thread pool and callers'
// own teams put in fftn::team when the threads' placement is not known (DESIGN.md 13.5,
// 18.2). A pure function of the topology, so that the fixtures test it on any machine.
#pragma once

#include "platform/topology.hpp"

namespace fftn::inline v0::detail {

// Logical CPUs per cache group: the smallest group of the last level that is shared by more
// than one core (L3, else L2); 0 when no level is.
[[nodiscard]] int cpus_per_cache_group(const topology& t) noexcept;

// Members per cache group of a team of `members` threads packed onto consecutive cores (one
// per core while there are enough cores, then every logical CPU); 0 for one group.
[[nodiscard]] int cache_group_size(const topology& t, int members) noexcept;

} // namespace fftn::inline v0::detail
