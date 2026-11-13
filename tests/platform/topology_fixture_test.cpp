// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The Linux and macOS readers against the fixtures in data/: machines recorded with
// tools/capture_platform.py (named by CPU model: the Apple M5 Max) and synthetic sysfs snapshots
// for the awkward cases (SMT off, offline CPUs, big.LITTLE, Intel hybrid, no L3, missing and
// malformed files, cgroup v1 and v2 quotas). Each fixture's known topology is spelled out
// below.

#include "platform/topology.hpp"

#include "platform/cpu_list.hpp"
#include "platform/file_source.hpp"
#include "platform/platform_text.hpp"
#include "platform/sysctl_values.hpp"
#include "platform/topology_checks.hpp"
#include "platform/topology_sources.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using fftn::detail::cache_geometry;
using fftn::detail::no_index;
using fftn::detail::topology;
using cpus = std::vector<unsigned>;

constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = 1024 * KiB;

struct class_expectation {
  const char* name = "";
  unsigned capacity = 0;
  std::size_t cores = 0;
  std::size_t logical_cpus = 0;
  cache_geometry l1d{};
  cache_geometry l2{};
  cache_geometry l3{};
};

constexpr cache_geometry cache(std::size_t size, std::size_t line, unsigned ways,
                               unsigned shared_by) {
  return {.size = size, .line_size = line, .ways = ways, .shared_by = shared_by};
}

struct fixture {
  const char* path = nullptr; // under data/: a .sysfs snapshot or a .sysctl file
  bool os_cpu_ids = true;
  std::size_t page_size = 0;
  std::size_t logical_cpus = 0;
  std::size_t cores = 0;
  std::size_t packages = 0;
  std::size_t numa_nodes = 0;
  std::size_t l2_groups = 0;
  std::size_t l3_groups = 0;
  std::size_t allowed_cpus = 0;
  double cpu_quota = 0;
  std::vector<class_expectation> classes;
};

// Linux fixtures carry no page size (sysfs does not hold it); the tests pass the one the
// recorded machine has.
std::string read_fixture(const char* name) {
  const std::ifstream file(std::filesystem::path(FFTN_PLATFORM_DATA) / name);
  std::ostringstream text;
  text << file.rdbuf();
  return std::move(text).str();
}

topology load(const char* name, std::size_t page_size = 0) {
  const std::string text = read_fixture(name);
  if (fftn::detail::is_sysfs_snapshot(text)) {
    return fftn::detail::topology_from_sysfs(fftn::detail::snapshot_files(text), page_size);
  }
  return fftn::detail::topology_from_sysctl(fftn::detail::parse_sysctl_text(text));
}

std::vector<fixture> all_fixtures() {
  return {
      {
          .path = "macos-apple-m5-max.sysctl",
          .os_cpu_ids = false,
          .page_size = 16 * KiB,
          .logical_cpus = 18,
          .cores = 18,
          .packages = 1,
          .numa_nodes = 1,
          .l2_groups = 3,
          .classes =
              {
                  {
                      .name = "Super",
                      .cores = 6,
                      .logical_cpus = 6,
                      .l1d = cache(128 * KiB, 128, 0, 1),
                      .l2 = cache(16 * MiB, 128, 0, 6),
                  },
                  {
                      .name = "Performance",
                      .cores = 12,
                      .logical_cpus = 12,
                      .l1d = cache(64 * KiB, 128, 0, 1),
                      .l2 = cache(8 * MiB, 128, 0, 6),
                  },
              },
      },
      {
          .path = "macos-synthetic-intel.sysctl",
          .os_cpu_ids = false,
          .page_size = 4 * KiB,
          .logical_cpus = 16,
          .cores = 8,
          .packages = 1,
          .numa_nodes = 1,
          .l2_groups = 8,
          .l3_groups = 1,
          .classes =
              {
                  {
                      .cores = 8,
                      .logical_cpus = 16,
                      .l1d = cache(32 * KiB, 64, 0, 2),
                      .l2 = cache(256 * KiB, 64, 0, 2),
                      .l3 = cache(16 * MiB, 64, 0, 16),
                  },
              },
      },
      {
          .path = "linux-synthetic-2socket-smt.sysfs",
          .page_size = 4 * KiB,
          .logical_cpus = 8,
          .cores = 4,
          .packages = 2,
          .numa_nodes = 2,
          .l2_groups = 4,
          .l3_groups = 2,
          .allowed_cpus = 8,
          .classes =
              {
                  {
                      .cores = 4,
                      .logical_cpus = 8,
                      .l1d = cache(32 * KiB, 64, 8, 2),
                      .l2 = cache(1 * MiB, 64, 16, 2),
                      .l3 = cache(25344 * KiB, 64, 11, 4),
                  },
              },
      },
      // SMT off: the second threads are offline, yet the sibling and sharing lists name them.
      {
          .path = "linux-synthetic-smt-off.sysfs",
          .page_size = 4 * KiB,
          .logical_cpus = 4,
          .cores = 4,
          .packages = 2,
          .numa_nodes = 2,
          .l2_groups = 4,
          .l3_groups = 2,
          .allowed_cpus = 4,
          .classes =
              {
                  {
                      .cores = 4,
                      .logical_cpus = 4,
                      .l1d = cache(32 * KiB, 64, 8, 1),
                      .l2 = cache(1 * MiB, 64, 16, 1),
                      .l3 = cache(25344 * KiB, 64, 11, 2),
                  },
              },
      },
      // Arm big.LITTLE by cpu_capacity, no L3, the old sibling file name, package id -1.
      {
          .path = "linux-synthetic-big-little.sysfs",
          .page_size = 4 * KiB,
          .logical_cpus = 8,
          .cores = 8,
          .packages = 1,
          .numa_nodes = 1,
          .l2_groups = 5,
          .classes =
              {
                  {
                      .capacity = 1024,
                      .cores = 4,
                      .logical_cpus = 4,
                      .l1d = cache(64 * KiB, 64, 4, 1),
                      .l2 = cache(1 * MiB, 64, 8, 1),
                  },
                  {
                      .capacity = 446,
                      .cores = 4,
                      .logical_cpus = 4,
                      .l1d = cache(32 * KiB, 64, 4, 1),
                      .l2 = cache(512 * KiB, 64, 16, 4),
                  },
              },
      },
      // Intel hybrid: P-cores with two threads, four E-cores sharing an L2.
      {
          .path = "linux-synthetic-hybrid-intel.sysfs",
          .page_size = 4 * KiB,
          .logical_cpus = 8,
          .cores = 6,
          .packages = 1,
          .numa_nodes = 1,
          .l2_groups = 3,
          .l3_groups = 1,
          .classes =
              {
                  {
                      .name = "core",
                      .cores = 2,
                      .logical_cpus = 4,
                      .l1d = cache(48 * KiB, 64, 12, 2),
                      .l2 = cache(1280 * KiB, 64, 10, 2),
                      .l3 = cache(12 * MiB, 64, 12, 8),
                  },
                  {
                      .name = "atom",
                      .cores = 4,
                      .logical_cpus = 4,
                      .l1d = cache(32 * KiB, 64, 8, 1),
                      .l2 = cache(2 * MiB, 64, 16, 4),
                      .l3 = cache(12 * MiB, 64, 12, 8),
                  },
              },
      },
      // No online file, topology, NUMA or proc. A malformed size reads as unknown (0) and
      // keeps the fields that did parse; a cache without a type is skipped.
      {
          .path = "linux-synthetic-missing.sysfs",
          .page_size = 4 * KiB,
          .logical_cpus = 3,
          .cores = 3,
          .packages = 1,
          .numa_nodes = 1,
          .classes = {{.cores = 3, .logical_cpus = 3, .l1d = cache(0, 64, 0, 0)}},
      },
      // cgroup v2: the tightest quota along the path (2.5 CPUs); a narrower affinity mask.
      {
          .path = "linux-synthetic-cgroup-v2.sysfs",
          .page_size = 64 * KiB,
          .logical_cpus = 4,
          .cores = 4,
          .packages = 1,
          .numa_nodes = 1,
          .allowed_cpus = 2,
          .cpu_quota = 2.5,
          .classes = {{.cores = 4, .logical_cpus = 4}},
      },
      // cgroup v1: 1.5 CPUs; an unrelated controller and a malformed line are skipped.
      {
          .path = "linux-synthetic-cgroup-v1.sysfs",
          .page_size = 4 * KiB,
          .logical_cpus = 2,
          .cores = 2,
          .packages = 1,
          .numa_nodes = 1,
          .cpu_quota = 1.5,
          .classes = {{.cores = 2, .logical_cpus = 2}},
      },
  };
}

class TopologyFixture : public testing::TestWithParam<fixture> {};

TEST_P(TopologyFixture, MatchesTheKnownMachine) {
  const fixture& f = GetParam();
  const topology t = load(f.path, f.page_size);
  fftn::test::expect_consistent(t);
  EXPECT_EQ(t.os_cpu_ids, f.os_cpu_ids);
  EXPECT_EQ(t.page_size, f.page_size);
  EXPECT_EQ(t.cpus.size(), f.logical_cpus);
  EXPECT_EQ(t.cores.size(), f.cores);
  EXPECT_EQ(t.packages.size(), f.packages);
  EXPECT_EQ(t.numa_nodes.size(), f.numa_nodes);
  EXPECT_EQ(t.l2_groups.size(), f.l2_groups);
  EXPECT_EQ(t.l3_groups.size(), f.l3_groups);
  EXPECT_EQ(t.allowed_cpus.size(), f.allowed_cpus);
  EXPECT_EQ(t.cpu_quota, f.cpu_quota);
  ASSERT_EQ(t.classes.size(), f.classes.size());
  for (std::size_t i = 0; i < f.classes.size(); ++i) {
    const auto& got = t.classes[i];
    const auto& want = f.classes[i];
    EXPECT_EQ(got.name, std::string(want.name)) << "class " << i;
    EXPECT_EQ(got.capacity, want.capacity) << "class " << i;
    EXPECT_EQ(got.cores, want.cores) << "class " << i;
    EXPECT_EQ(got.logical_cpus, want.logical_cpus) << "class " << i;
    EXPECT_EQ(got.l1d, want.l1d) << "class " << i << " l1d";
    EXPECT_EQ(got.l2, want.l2) << "class " << i << " l2";
    EXPECT_EQ(got.l3, want.l3) << "class " << i << " l3";
  }
}

TEST_P(TopologyFixture, LoadsThroughTheOverridePath) {
  const fixture& f = GetParam();
  topology t = fftn::detail::load_topology(std::filesystem::path(FFTN_PLATFORM_DATA) / f.path);
  // load_topology gives a sysfs snapshot this process's page size; the rest is the same.
  t.page_size = f.page_size;
  EXPECT_EQ(t, load(f.path, f.page_size));
}

std::string fixture_name(const testing::TestParamInfo<fixture>& info) {
  std::string name;
  for (const char c : std::string(info.param.path)) {
    name += (c == '-' || c == '.') ? '_' : c;
  }
  return name;
}

INSTANTIATE_TEST_SUITE_P(Fixtures, TopologyFixture, testing::ValuesIn(all_fixtures()),
                         fixture_name);

// Details the table above does not capture.

// A snapshot written out as a directory tree reads the same through the live-system reader.
TEST(TopologyFixtureDetail, DirectoryAndSnapshotAgree) {
  const char* name = "linux-synthetic-cgroup-v2.sysfs";
  const fftn::detail::snapshot_files snapshot(read_fixture(name));
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "fftn-platform-test-cgroup-v2";
  std::filesystem::remove_all(root);
  std::vector<std::string> pending = {""};
  while (!pending.empty()) {
    const std::string dir = pending.back();
    pending.pop_back();
    for (const std::string& entry : snapshot.list(dir)) {
      std::string path = dir;
      if (!path.empty()) {
        path += '/';
      }
      path += entry;
      if (const auto content = snapshot.read(path)) {
        std::filesystem::create_directories((root / path).parent_path());
        std::ofstream(root / path) << *content << '\n';
      } else {
        pending.push_back(path);
      }
    }
  }
  topology from_tree = fftn::detail::load_topology(root);
  from_tree.page_size = 64 * KiB;
  EXPECT_EQ(from_tree, load(name, 64 * KiB));
  EXPECT_EQ(from_tree.cpu_quota, 2.5);
  std::filesystem::remove_all(root);
}

TEST(TopologyFixtureDetail, TwoSocketGroupsFollowIntelNumbering) {
  const topology t = load("linux-synthetic-2socket-smt.sysfs", 4096);
  EXPECT_EQ(t.source, "linux-sysfs");
  EXPECT_EQ(t.allowed_cpus, (cpus{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(t.cores[0].cpus, (cpus{0, 4}));
  EXPECT_EQ(t.l3_groups[0].cpus, (cpus{0, 1, 4, 5}));
  EXPECT_EQ(t.l3_groups[1].cpus, (cpus{2, 3, 6, 7}));
  EXPECT_EQ(t.numa_nodes[1].cpus, (cpus{2, 3, 6, 7}));
  EXPECT_EQ(t.packages[1].cpus, (cpus{2, 3, 6, 7}));
  EXPECT_EQ(t.cpus[6].package, 1U);
  EXPECT_EQ(t.cpus[6].l3_group, 1U);
}

TEST(TopologyFixtureDetail, SmtOffDropsOfflineSiblings) {
  const topology t = load("linux-synthetic-smt-off.sysfs", 4096);
  EXPECT_EQ(t.cores[0].cpus, cpus{0});
  EXPECT_EQ(t.l3_groups[0].cpus, (cpus{0, 1}));
  EXPECT_EQ(t.numa_nodes[0].cpus, (cpus{0, 1}));
}

TEST(TopologyFixtureDetail, BigLittleOrdersClassesFastestFirst) {
  const topology t = load("linux-synthetic-big-little.sysfs", 4096);
  EXPECT_EQ(t.cpus[0].core_class, 1U); // a little core
  EXPECT_EQ(t.cpus[7].core_class, 0U);
  EXPECT_EQ(t.cpus[0].l3_group, no_index);
  EXPECT_EQ(t.l2_groups[0].cpus, (cpus{0, 1, 2, 3}));
  EXPECT_EQ(t.packages[0].id, 0U); // physical_package_id -1
}

TEST(TopologyFixtureDetail, IntelHybridSeparatesCoreAndAtom) {
  const topology t = load("linux-synthetic-hybrid-intel.sysfs", 4096);
  EXPECT_EQ(t.cpus[1].core_class, 0U);
  EXPECT_EQ(t.cpus[4].core_class, 1U);
  EXPECT_EQ(t.l2_groups[2].cpus, (cpus{4, 5, 6, 7}));
}

TEST(TopologyFixtureDetail, MacNumbersCpusInClassOrder) {
  const topology t = load("macos-apple-m5-max.sysctl");
  EXPECT_EQ(t.source, "macos-sysctl");
  EXPECT_EQ(t.cpus[5].core_class, 0U);
  EXPECT_EQ(t.cpus[6].core_class, 1U);
  EXPECT_EQ(t.l2_groups[0].cpus, (cpus{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(t.l2_groups[2].cpus, (cpus{12, 13, 14, 15, 16, 17}));
}

TEST(TopologyFixtureDetail, IntelMacPairsSmtThreads) {
  const topology t = load("macos-synthetic-intel.sysctl");
  EXPECT_EQ(t.cores[3].cpus, (cpus{6, 7}));
  EXPECT_EQ(t.l2_groups[3].cpus, (cpus{6, 7}));
  EXPECT_EQ(t.l3_groups[0].cpus.size(), 16U);
}

// The dump's text form is what gets recorded per machine; it must not drift silently.
TEST(TopologyFixtureDetail, TextFormIsStable) {
  const topology t = load("linux-synthetic-cgroup-v1.sysfs", 4096);
  EXPECT_EQ(fftn::test::topology_text(t), R"(source linux-sysfs
os_cpu_ids 1
page_size 4096
logical_cpus 2: 0-1
allowed_cpus 0
cpu_quota 1.5
classes 1
  0: name="" capacity=0 cores=2 logical_cpus=2 cpus=0-1
    l1d size=0 line=0 ways=0 shared_by=0
    l2 size=0 line=0 ways=0 shared_by=0
    l3 size=0 line=0 ways=0 shared_by=0
packages 1
  0: 0-1
numa_nodes 1
  0: 0-1
l3_groups 0
l2_groups 0
cores 2
  0: 0
  1: 1
)");
}

} // namespace
