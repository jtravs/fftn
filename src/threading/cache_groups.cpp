// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "threading/cache_groups.hpp"

#include "fftn/team.hpp"
#include "platform/topology.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

std::size_t smallest(const std::vector<cpu_group>& groups) noexcept {
  std::size_t least = std::numeric_limits<std::size_t>::max();
  for (const cpu_group& g : groups) {
    least = std::min(least, g.cpus.size());
  }
  return least;
}

} // namespace

int cpus_per_cache_group(const topology& t) noexcept {
  // A level whose smallest group holds no more CPUs than a core does is private to cores
  // (an L2 on most x86 processors): sharing a transform there buys nothing over any other
  // pair of cores.
  const std::size_t core = t.cores.empty() ? 1 : smallest(t.cores);
  for (const std::vector<cpu_group>* level : {&t.l3_groups, &t.l2_groups}) {
    if (level->empty()) {
      continue;
    }
    const std::size_t g = smallest(*level);
    if (g > core) {
      return static_cast<int>(std::min<std::size_t>(g, std::numeric_limits<int>::max()));
    }
  }
  return 0;
}

namespace {

// What the estimate needs of a topology.
struct grouping {
  int group = 0; // logical CPUs per cache group; 0: unknown
  std::size_t cores = 0;
  std::size_t cpus = 0;
};

grouping grouping_of(const topology& t) noexcept {
  return {.group = cpus_per_cache_group(t), .cores = t.cores.size(), .cpus = t.cpus.size()};
}

int members_per_group(const grouping& m, int members) noexcept {
  int group = m.group;
  if (members < 2 || group == 0) {
    return 0;
  }
  // One thread per core while the team has no more members than the machine has cores: a
  // group then holds as many members as it has cores. (Integer logical CPUs per core: a
  // hybrid processor whose small cores lack SMT counts as having none.)
  if (m.cores != 0 && std::cmp_less_equal(members, m.cores)) {
    const std::size_t smt = std::max<std::size_t>(m.cpus / m.cores, 1);
    group = std::max(group / static_cast<int>(smt), 1);
  }
  return group >= members ? 0 : group;
}

} // namespace

int cache_group_size(const topology& t, int members) noexcept {
  return members_per_group(grouping_of(t), members);
}

} // namespace fftn::inline v0::detail

namespace fftn::inline v0 {

int cache_group_size(int members) noexcept {
  // Computed once (the topology is a constant too), so that the members of a team can call
  // this at every execution.
  static const detail::grouping machine = []() noexcept {
    try {
      return detail::grouping_of(detail::system_topology());
    } catch (...) {
      // Unknown: a topology that cannot be read (memory, an unreadable FFTN_TOPOLOGY
      // recording) gives no grouping, as missing data does.
      return detail::grouping{};
    }
  }();
  return detail::members_per_group(machine, members);
}

} // namespace fftn::inline v0
