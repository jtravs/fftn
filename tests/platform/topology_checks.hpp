// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Structural invariants every topology must satisfy, whatever its source.
#pragma once

#include "platform/topology.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace fftn::test {

// Each CPU is in exactly one group of `groups`, the one its index names (or in none, when
// the index is no_index and `optional` allows that).
inline void expect_partition(const fftn::detail::topology& t,
                             const std::vector<fftn::detail::cpu_group>& groups,
                             unsigned fftn::detail::logical_cpu::* index, bool optional,
                             const char* what) {
  std::vector<unsigned> seen(t.cpus.size(), 0);
  for (const auto& g : groups) {
    EXPECT_FALSE(g.cpus.empty()) << what;
    EXPECT_TRUE(std::ranges::is_sorted(g.cpus)) << what;
    for (const unsigned id : g.cpus) {
      const auto it = std::ranges::find(t.cpus, id, &fftn::detail::logical_cpu::id);
      ASSERT_NE(it, t.cpus.end()) << what << ": CPU " << id << " is not a recorded CPU";
      ++seen[static_cast<std::size_t>(it - t.cpus.begin())];
    }
  }
  for (std::size_t i = 0; i < t.cpus.size(); ++i) {
    const unsigned g = t.cpus[i].*index;
    if (g == fftn::detail::no_index) {
      EXPECT_TRUE(optional) << what << ": CPU " << t.cpus[i].id << " has no group";
      EXPECT_EQ(seen[i], 0U) << what;
      continue;
    }
    EXPECT_EQ(seen[i], 1U) << what << ": CPU " << t.cpus[i].id;
    ASSERT_LT(g, groups.size()) << what;
    EXPECT_TRUE(std::ranges::binary_search(groups[g].cpus, t.cpus[i].id)) << what;
  }
}

inline void expect_consistent(const fftn::detail::topology& t) {
  using fftn::detail::logical_cpu;
  ASSERT_FALSE(t.cpus.empty());
  EXPECT_TRUE(std::ranges::is_sorted(t.cpus, {}, &logical_cpu::id));
  EXPECT_GT(t.page_size, 0U);
  expect_partition(t, t.cores, &logical_cpu::core, false, "cores");
  expect_partition(t, t.packages, &logical_cpu::package, false, "packages");
  expect_partition(t, t.numa_nodes, &logical_cpu::node, false, "numa_nodes");
  expect_partition(t, t.l2_groups, &logical_cpu::l2_group, true, "l2_groups");
  expect_partition(t, t.l3_groups, &logical_cpu::l3_group, true, "l3_groups");
  std::size_t cores = 0;
  std::size_t logical = 0;
  for (const auto& c : t.classes) {
    cores += c.cores;
    logical += c.logical_cpus;
  }
  EXPECT_EQ(cores, t.cores.size());
  EXPECT_EQ(logical, t.cpus.size());
  for (const auto& cpu : t.cpus) {
    ASSERT_LT(cpu.core_class, t.classes.size());
    // SMT siblings share a class.
    for (const unsigned sibling : t.cores[cpu.core].cpus) {
      const auto it = std::ranges::find(t.cpus, sibling, &logical_cpu::id);
      EXPECT_EQ(it->core_class, cpu.core_class);
    }
  }
}

} // namespace fftn::test
