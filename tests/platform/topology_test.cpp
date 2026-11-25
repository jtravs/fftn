// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/topology.hpp"

#include "platform/cpu_features.hpp"
#include "platform/file_source.hpp"
#include "platform/sysctl_values.hpp"
#include "platform/topology_checks.hpp"
#include "platform/topology_sources.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using fftn::detail::assemble_topology;
using fftn::detail::cpu_record;
using fftn::detail::cpu_vendor;
using fftn::detail::no_index;
using fftn::detail::topology;
using fftn::detail::topology_parts;
using cpus = std::vector<unsigned>;

cpu_record record(unsigned id, cpus siblings, cpus l2, cpus l3) {
  cpu_record r;
  r.id = id;
  r.smt_siblings = std::move(siblings);
  r.l2_sharing = std::move(l2);
  r.l3_sharing = std::move(l3);
  return r;
}

TEST(AssembleTopology, DropsUnknownCpusAndAddsSelf) {
  topology_parts parts;
  parts.page_size = 4096;
  // CPU 1 is offline: lists that name it lose it; CPU 2's sibling list forgot CPU 2.
  parts.cpus.push_back(record(3, {2, 3}, {0, 1, 2, 3}, {}));
  parts.cpus.push_back(record(0, {0, 1}, {0, 1, 2, 3}, {}));
  parts.cpus.push_back(record(2, {3}, {0, 1, 2, 3}, {}));
  parts.cpus.push_back(record(0, {0, 1}, {}, {})); // a duplicate record is ignored
  const topology t = assemble_topology(parts);
  fftn::test::expect_consistent(t);
  ASSERT_EQ(t.cpus.size(), 3U);
  EXPECT_EQ(t.cpus[0].id, 0U);
  ASSERT_EQ(t.cores.size(), 2U);
  EXPECT_EQ(t.cores[0].cpus, cpus{0});
  EXPECT_EQ(t.cores[1].cpus, (cpus{2, 3}));
  ASSERT_EQ(t.l2_groups.size(), 1U);
  EXPECT_EQ(t.l2_groups[0].cpus, (cpus{0, 2, 3}));
  EXPECT_TRUE(t.l3_groups.empty());
  EXPECT_EQ(t.cpus[1].l3_group, no_index);
  ASSERT_EQ(t.classes.size(), 1U);
  EXPECT_EQ(t.classes[0].cores, 2U);
  EXPECT_EQ(t.classes[0].logical_cpus, 3U);
}

TEST(AssembleTopology, AddsClassesTheSourceDidNotDescribe) {
  topology_parts parts;
  parts.page_size = 4096;
  parts.cpus.push_back(record(0, {}, {}, {}));
  parts.cpus.push_back(record(1, {}, {}, {}));
  parts.cpus[1].core_class = 2;
  parts.cpus[1].package = 7;
  parts.cpus[1].node = 3;
  const topology t = assemble_topology(parts);
  fftn::test::expect_consistent(t);
  ASSERT_EQ(t.classes.size(), 3U);
  EXPECT_EQ(t.classes[1].logical_cpus, 0U);
  EXPECT_EQ(t.classes[2].logical_cpus, 1U);
  ASSERT_EQ(t.packages.size(), 2U);
  EXPECT_EQ(t.packages[1].id, 7U);
  EXPECT_EQ(t.cpus[1].package, 1U);
  EXPECT_EQ(t.numa_nodes[1].id, 3U);
  EXPECT_EQ(t.cpus[1].node, 1U);
}

TEST(FallbackTopology, IsCpusOnly) {
  const topology t = fftn::detail::fallback_topology(4);
  fftn::test::expect_consistent(t);
  EXPECT_EQ(t.source, "fallback");
  EXPECT_EQ(t.cpus.size(), 4U);
  EXPECT_EQ(t.cores.size(), 4U);
  EXPECT_TRUE(t.l2_groups.empty());
  // hardware_concurrency() may return 0; there is always at least one CPU.
  EXPECT_EQ(fftn::detail::fallback_topology(0).cpus.size(), 1U);
}

TEST(SysctlTopology, LevelsWithoutACacheHaveNoGroups) {
  const topology t = fftn::detail::topology_from_sysctl(fftn::detail::parse_sysctl_text(
      "hw.nperflevels: 1\nhw.perflevel0.logicalcpu: 4\nhw.perflevel0.physicalcpu: 4\n"
      "hw.perflevel0.cpusperl2: 4\nhw.pagesize: 16384\n"));
  fftn::test::expect_consistent(t);
  EXPECT_EQ(t.cpus.size(), 4U);
  EXPECT_TRUE(t.l2_groups.empty()); // cpusperl2 without an L2 size describes nothing
  EXPECT_EQ(t.classes[0].l2.size, 0U);
}

TEST(SysctlTopology, IntelMacWithoutCacheConfig) {
  const topology t = fftn::detail::topology_from_sysctl(fftn::detail::parse_sysctl_text(
      "hw.logicalcpu: 4\nhw.physicalcpu: 2\nhw.l2cachesize: 262144\nhw.pagesize: 4096\n"));
  fftn::test::expect_consistent(t);
  EXPECT_EQ(t.cores.size(), 2U);
  EXPECT_TRUE(t.l2_groups.empty()); // the sharing is unknown
  EXPECT_EQ(t.classes[0].l2.size, 262144U);
  EXPECT_EQ(t.classes[0].l2.shared_by, 0U);
}

// A recording may state its processor's vendor (file_source.hpp); without it, none is known
// and the host's stands (planning_vendor).
TEST(RecordedVendor, ReadFromASnapshotEntry) {
  const auto snapshot = [](const std::string& entry) {
    return fftn::detail::topology_from_sysfs(
        fftn::detail::snapshot_files("# fftn-sysfs-snapshot 1\n" + entry +
                                     "sys/devices/system/cpu/online\t0\n"),
        4096);
  };
  EXPECT_EQ(snapshot("").vendor, std::nullopt);
  EXPECT_EQ(snapshot("fftn/cpu_vendor\tintel\n").vendor, cpu_vendor::intel);
  EXPECT_EQ(snapshot("fftn/cpu_vendor\tamd\\n\n").vendor, cpu_vendor::amd); // a final newline
  EXPECT_EQ(snapshot("fftn/cpu_vendor\tarm_other\n").vendor, cpu_vendor::arm_other);
  EXPECT_EQ(snapshot("fftn/cpu_vendor\tunknown\n").vendor, cpu_vendor::unknown);
  EXPECT_THROW((void)snapshot("fftn/cpu_vendor\tGenuineIntel\n"), std::runtime_error);
  EXPECT_THROW((void)snapshot("fftn/cpu_vendor\t\n"), std::runtime_error);
}

TEST(RecordedVendor, ReadFromASysctlKey) {
  const auto sysctl = [](const std::string& text) {
    return fftn::detail::topology_from_sysctl(
        fftn::detail::parse_sysctl_text(text + "hw.logicalcpu: 2\nhw.physicalcpu: 2\n"));
  };
  EXPECT_EQ(sysctl("").vendor, std::nullopt);
  EXPECT_EQ(sysctl("fftn.cpu_vendor: apple\n").vendor, cpu_vendor::apple);
  EXPECT_THROW((void)sysctl("fftn.cpu_vendor: arm\n"), std::runtime_error);
}

TEST(RecordedVendor, OverridesTheHostsOnly) {
  topology recorded;
  recorded.vendor = cpu_vendor::intel;
  const topology unrecorded;
  for (const cpu_vendor host : {cpu_vendor::unknown, cpu_vendor::intel, cpu_vendor::amd,
                                cpu_vendor::apple, cpu_vendor::arm_other}) {
    EXPECT_EQ(fftn::detail::planning_vendor(recorded, host), cpu_vendor::intel);
    EXPECT_EQ(fftn::detail::planning_vendor(unrecorded, host), host);
  }
}

TEST(Topology, PageSizeIsAPowerOfTwo) {
  const std::size_t page = fftn::detail::system_page_size();
  EXPECT_GE(page, 4096U);
  EXPECT_EQ(page & (page - 1), 0U);
}

TEST(Topology, LoadRejectsAMissingPath) {
  EXPECT_THROW((void)fftn::detail::load_topology(FFTN_PLATFORM_DATA "/no-such-fixture"),
               std::runtime_error);
}

TEST(Topology, ReadersForOtherSystemsAreEmptyHere) {
#ifndef _WIN32
  EXPECT_TRUE(fftn::detail::topology_from_windows().cpus.empty());
#endif
#ifndef __APPLE__
  EXPECT_TRUE(fftn::detail::read_topology_sysctl().empty());
#endif
}

// The running machine: whatever it is, the result is consistent, and its CPUs are those
// the C++ library sees.
TEST(TopologyLive, DetectsTheRunningMachine) {
  const topology t = fftn::detail::detect_topology();
  fftn::test::expect_consistent(t);
  if (std::thread::hardware_concurrency() > 0) {
    EXPECT_EQ(t.cpus.size(), std::thread::hardware_concurrency());
  }
  EXPECT_EQ(t.page_size, fftn::detail::system_page_size());
#ifdef __APPLE__
  EXPECT_EQ(t.source, "macos-sysctl");
  const auto values = fftn::detail::read_topology_sysctl();
  EXPECT_EQ(t.cores.size(), fftn::detail::sysctl_integer(values, "hw.physicalcpu"));
#ifdef __aarch64__
  // Apple silicon: performance levels, 128-byte lines, an L2 per cluster, no SMT.
  EXPECT_EQ(t.classes.size(), fftn::detail::sysctl_integer(values, "hw.nperflevels"));
  EXPECT_EQ(t.classes[0].l1d.line_size, 128U);
  EXPECT_GT(t.classes[0].l2.size, 0U);
  EXPECT_EQ(t.cores.size(), t.cpus.size());
#endif
#elif defined(__linux__)
  EXPECT_EQ(t.source, "linux-sysfs");
#endif
}

TEST(TopologyLive, SystemTopologyIsComputedOnce) {
  const topology& a = fftn::detail::system_topology();
  const topology& b = fftn::detail::system_topology();
  EXPECT_EQ(&a, &b);
  fftn::test::expect_consistent(a);
}

} // namespace
