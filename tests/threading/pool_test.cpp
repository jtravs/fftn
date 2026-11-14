// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fftn::thread_pool (task 3.3): every task of a call runs exactly once and before the call
// returns, on pools of 1 to 8 lanes, for calls of 0 to thousands of tasks, back to back and
// with pauses long enough for idle threads to block; calls from two threads at once and from
// inside a task; shutdown. And the cache grouping it and the OpenMP adapters use
// (fftn::cache_group_size), on synthetic and recorded topologies.

#include "platform/topology.hpp"
#include "threading/cache_groups.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <thread>
#include <vector>

namespace {

struct counts {
  std::vector<std::atomic<int>> runs;

  explicit counts(int n) : runs(static_cast<std::size_t>(n)) {}
};

void count_task(void* arg, int i) {
  static_cast<counts*>(arg)->runs[static_cast<std::size_t>(i)].fetch_add(1);
}

bool each_ran_once(const counts& c) {
  return std::ranges::all_of(c.runs, [](const std::atomic<int>& r) { return r.load() == 1; });
}

TEST(ThreadPool, RunsEveryTaskOnceBeforeReturning) {
  for (const int lanes : {1, 2, 3, 8}) {
    const fftn::thread_pool pool(lanes);
    EXPECT_EQ(pool.concurrency(), lanes);
    EXPECT_EQ(pool.group_size(), fftn::cache_group_size(lanes));
    for (const int n : {0, 1, 2, 7, 8, 9, 100, 5000}) {
      for (int round = 0; round < 3; ++round) {
        counts c(n);
        pool.bulk(n, count_task, &c);
        EXPECT_TRUE(each_ran_once(c)) << lanes << " lanes, " << n << " tasks";
      }
    }
    // Long enough between calls for the idle threads to stop spinning and block.
    for (int round = 0; round < 3; ++round) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      counts c(64);
      pool.bulk(64, count_task, &c);
      EXPECT_TRUE(each_ran_once(c)) << lanes << " lanes, after a pause";
    }
  }
}

TEST(ThreadPool, DefaultLanesAndNegativeCounts) {
  const fftn::thread_pool pool;
  EXPECT_GE(pool.concurrency(), 1);
  counts c(0);
  pool.bulk(-3, count_task, &c);
  counts d(40);
  pool.bulk(40, count_task, &d);
  EXPECT_TRUE(each_ran_once(d));
}

// Calls from inside a task run serially, see concurrency 1, and may use another pool.
TEST(ThreadPool, NestedCallsRunSerially) {
  const fftn::thread_pool pool(4);
  const fftn::thread_pool other(2);
  struct context {
    const fftn::thread_pool* pool;
    const fftn::thread_pool* other;
    std::vector<counts> inner;
    std::vector<counts> single;
    std::atomic<int> wide{0};
  } ctx{.pool = &pool, .other = &other, .inner = {}, .single = {}, .wide = {}};
  ctx.inner.reserve(12);
  ctx.single.reserve(12);
  for (int k = 0; k < 12; ++k) {
    ctx.inner.emplace_back(5);
    ctx.single.emplace_back(1);
  }
  pool.bulk(
      12,
      [](void* arg, int k) {
        auto& x = *static_cast<context*>(arg);
        if (x.pool->concurrency() != 1 || x.other->concurrency() != 1) {
          x.wide.fetch_add(1);
        }
        const fftn::thread_pool& use = k % 2 == 0 ? *x.pool : *x.other;
        use.bulk(5, count_task, &x.inner[static_cast<std::size_t>(k)]);
        use.bulk(1, count_task, &x.single[static_cast<std::size_t>(k)]);
      },
      &ctx);
  EXPECT_EQ(ctx.wide.load(), 0);
  for (std::size_t k = 0; k < ctx.inner.size(); ++k) {
    EXPECT_TRUE(each_ran_once(ctx.inner[k])) << k;
    EXPECT_TRUE(each_ran_once(ctx.single[k])) << k;
  }
  EXPECT_EQ(pool.concurrency(), 4);
}

// Two threads calling one pool at once: one job holds the pool, the other runs serially on
// its own thread; both complete.
TEST(ThreadPool, ConcurrentCallersBothComplete) {
  const fftn::thread_pool pool(4);
  for (int round = 0; round < 20; ++round) {
    counts a(300);
    counts b(300);
    std::thread other([&] { pool.bulk(300, count_task, &b); });
    pool.bulk(300, count_task, &a);
    other.join();
    EXPECT_TRUE(each_ran_once(a)) << round;
    EXPECT_TRUE(each_ran_once(b)) << round;
  }
}

// Pools come and go, idle or just after work.
TEST(ThreadPool, ShutsDownCleanly) {
  for (int k = 0; k < 20; ++k) {
    const fftn::thread_pool pool(1 + k % 6);
    if (k % 2 == 0) {
      counts c(50);
      pool.bulk(50, count_task, &c);
      EXPECT_TRUE(each_ran_once(c));
    }
  }
}

fftn::detail::cpu_group group_of(unsigned first, unsigned count) {
  fftn::detail::cpu_group g;
  for (unsigned c = first; c < first + count; ++c) {
    g.cpus.push_back(c);
  }
  return g;
}

// A machine of `groups` cache groups of `cores_per_group` cores with `smt` CPUs each; the
// cache level is L3, or L2 when `l2` is set.
fftn::detail::topology machine(unsigned groups, unsigned cores_per_group, unsigned smt, bool l2) {
  fftn::detail::topology t;
  const unsigned cores = groups * cores_per_group;
  t.cpus.resize(std::size_t{cores} * smt);
  for (unsigned c = 0; c < cores; ++c) {
    t.cores.push_back(group_of(c * smt, smt));
  }
  for (unsigned g = 0; g < groups; ++g) {
    (l2 ? t.l2_groups : t.l3_groups)
        .push_back(group_of(g * cores_per_group * smt, cores_per_group * smt));
  }
  return t;
}

TEST(CacheGroups, FromSyntheticTopologies) {
  using fftn::detail::cache_group_size;
  using fftn::detail::cpus_per_cache_group;
  // 8 groups of 8 cores with SMT 2, as a 64-core processor of CCXs.
  const auto ccx = machine(8, 8, 2, false);
  EXPECT_EQ(cpus_per_cache_group(ccx), 16);
  EXPECT_EQ(cache_group_size(ccx, 64), 8);   // one thread per core
  EXPECT_EQ(cache_group_size(ccx, 128), 16); // every logical CPU
  EXPECT_EQ(cache_group_size(ccx, 9), 8);
  EXPECT_EQ(cache_group_size(ccx, 8), 0); // one group holds them all
  EXPECT_EQ(cache_group_size(ccx, 1), 0);
  EXPECT_EQ(cache_group_size(ccx, 0), 0);
  // Clusters sharing an L2, no SMT, as Apple's processors.
  const auto clusters = machine(3, 6, 1, true);
  EXPECT_EQ(cpus_per_cache_group(clusters), 6);
  EXPECT_EQ(cache_group_size(clusters, 18), 6);
  EXPECT_EQ(cache_group_size(clusters, 4), 0);
  // Groups of different sizes: the smallest.
  auto mixed = clusters;
  mixed.l2_groups.push_back(group_of(18, 4));
  mixed.cpus.resize(22);
  for (unsigned c = 18; c < 22; ++c) {
    mixed.cores.push_back(group_of(c, 1));
  }
  EXPECT_EQ(cpus_per_cache_group(mixed), 4);
  EXPECT_EQ(cache_group_size(mixed, 22), 4);
  // An L2 per core and one L3: the L3 counts. Only private caches, or nothing known: none.
  auto l3_over_l2 = machine(1, 18, 2, false);
  for (const auto& core : l3_over_l2.cores) {
    l3_over_l2.l2_groups.push_back(core);
  }
  EXPECT_EQ(cpus_per_cache_group(l3_over_l2), 36);
  EXPECT_EQ(cache_group_size(l3_over_l2, 72), 36);
  EXPECT_EQ(cache_group_size(l3_over_l2, 36), 0);
  auto private_only = l3_over_l2;
  private_only.l3_groups.clear();
  EXPECT_EQ(cpus_per_cache_group(private_only), 0);
  EXPECT_EQ(cache_group_size(private_only, 36), 0);
  EXPECT_EQ(cpus_per_cache_group(fftn::detail::topology{}), 0);
  // No core information: logical CPUs only.
  auto no_cores = clusters;
  no_cores.cores.clear();
  EXPECT_EQ(cache_group_size(no_cores, 12), 6);
}

TEST(CacheGroups, FromRecordedMachines) {
  const std::filesystem::path data(FFTN_PLATFORM_DATA);
  const auto m5 = fftn::detail::load_topology(data / "macos-apple-m5-max.sysctl");
  EXPECT_EQ(fftn::detail::cpus_per_cache_group(m5), 6);
  EXPECT_EQ(fftn::detail::cache_group_size(m5, 18), 6);
  const auto epyc = fftn::detail::load_topology(data / "linux-amd-epyc-7763.sysfs");
  // SMT off: a CCX is 8 CPUs.
  EXPECT_EQ(fftn::detail::cpus_per_cache_group(epyc), 8);
  EXPECT_EQ(fftn::detail::cache_group_size(epyc, 128), 8);
  EXPECT_EQ(fftn::detail::cache_group_size(epyc, 8), 0);
}

TEST(CacheGroups, OfThisMachine) {
  EXPECT_EQ(fftn::cache_group_size(1), 0);
  EXPECT_EQ(fftn::cache_group_size(-4), 0);
  for (const int members : {2, 4, 16, 64, 256}) {
    const int g = fftn::cache_group_size(members);
    EXPECT_TRUE(g == 0 || (g >= 1 && g < members)) << members << ": " << g;
  }
}

} // namespace
