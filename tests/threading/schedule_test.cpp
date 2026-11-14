// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler's rules (DESIGN.md 13.4), one by one, on work graphs read off real plans,
// and the work graph itself. The schedule is a pure function: the same inputs give the same
// schedule, and it never depends on anything but its arguments.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

using fftn::detail::group_share;
using fftn::detail::lane_map;
using fftn::detail::machine_profile;
using fftn::detail::make_schedule;
using fftn::detail::schedule;
using fftn::detail::share_of;
using fftn::detail::work_graph;

std::shared_ptr<const fftn::detail::plan_core> rows(std::size_t n, std::size_t count) {
  const fftn::options opts;
  const fftn::detail::build_settings settings;
  return fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<double>(n, count, fftn::direction::forward)), opts,
      settings);
}

machine_profile profile() {
  machine_profile p; // the provisional defaults, with a machine of 64 CPUs
  p.cpus = 64;
  return p;
}

// The shares of every group cover the batch once, in order.
void expect_partition(const schedule& s, const lane_map& lanes) {
  std::size_t next = 0;
  for (int g = 0; g < lanes.groups(); ++g) {
    const group_share sh = share_of(s, lanes, g);
    if (sh.count != 0) {
      EXPECT_EQ(sh.first, next);
    }
    EXPECT_EQ(sh.whole + sh.split, sh.count);
    EXPECT_LT(sh.split, static_cast<std::size_t>(lanes.size_of(g)) + (sh.count == 0 ? 1 : 0));
    next += sh.count;
  }
  EXPECT_EQ(next, s.tasks);
}

TEST(WorkGraph, PhasesOfRowsAndPlanes) {
  // A 1-D row: one phase per stage, chunks as the stages have them.
  const auto r = rows(16384, 2);
  const work_graph& g = r->graph;
  EXPECT_EQ(g.tasks, 2U);
  EXPECT_EQ(g.points, 16384U);
  std::uint64_t first = 0;
  for (const auto& ph : g.phases) {
    EXPECT_EQ(ph.nest, 0U);
    EXPECT_EQ(ph.first, first);
    first += ph.chunks;
  }
  EXPECT_EQ(g.chunks, first);
  EXPECT_GT(g.widest, 1U);
  // Every codelet pass moves the 16384 points once, whatever its unit (C > 1 blocks).
  EXPECT_EQ(g.work, 16384.0 * static_cast<double>(g.phases.size()));
  // The same in float, where a neon_fcma or avx2 pass's unit is a block of 2 or 4 copies.
  const auto f = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<float>(16384, 1, fftn::direction::forward)),
      fftn::options(), fftn::detail::build_settings());
  EXPECT_EQ(f->graph.work, 16384.0 * static_cast<double>(f->graph.phases.size()));
  // A generic pass of radix r sums r terms per point: 2 x 17 is a grouped radix-2 pass (34
  // points) and a generic radix-17 pass (34 x 17).
  EXPECT_EQ(rows(34, 1)->graph.work, 34.0 + 34.0 * 17);
  // A copy or real edge pass touches two points per unit: n = 1 is one copy of one element.
  EXPECT_EQ(rows(1, 1)->graph.work, 2.0);
  // Planes: one phase of lines per axis.
  const fftn::options opts;
  fftn::detail::build_settings fine;
  fine.chunk_points = 256;
  const auto p = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::planes_c2c<double>(64, 32, 1, fftn::direction::forward)), opts,
      fine);
  ASSERT_EQ(p->graph.phases.size(), 2U);
  EXPECT_EQ(p->graph.phases[0].nest, 1U);
  EXPECT_EQ(p->graph.phases[0].units, 64U); // lines of 32 along axis 1
  EXPECT_EQ(p->graph.phases[0].chunks, 8U); // 256 / 32 lines per chunk
  EXPECT_EQ(p->graph.phases[1].units, 32U);
  // A volume: the two loops over the other axes are one phase of their product.
  fftn::shape<std::complex<double>, std::complex<double>> v;
  v.axes = {{.n = 4}, {.n = 5}, {.n = 6}};
  const auto vol =
      fftn::detail::make_plan_core(fftn::detail::view_of(v), opts, fftn::detail::build_settings());
  ASSERT_EQ(vol->graph.phases.size(), 3U);
  EXPECT_EQ(vol->graph.phases[0].nest, 2U);
  EXPECT_EQ(vol->graph.phases[0].units, 20U);
  EXPECT_EQ(vol->graph.phases[2].units, 30U);
}

// Rule 1: a job smaller than a few dispatch costs runs inline.
TEST(Schedule, InlineWhenTheJobIsSmall) {
  const lane_map lanes{.lanes = 8};
  const schedule small = make_schedule(rows(1728, 1)->graph, lanes, profile());
  EXPECT_TRUE(small.inline_all);
  const group_share sh = share_of(small, lanes, 0);
  EXPECT_EQ(sh.count, 1U);
  EXPECT_EQ(sh.split, 0U);
  EXPECT_FALSE(make_schedule(rows(1728, 64)->graph, lanes, profile()).inline_all);
  // A team of one is always inline.
  EXPECT_TRUE(make_schedule(rows(65536, 8)->graph, {.lanes = 1}, profile()).inline_all);
}

// Rule 2: with at least as many rows as members, rows go whole; leftovers are split when
// they are long enough.
TEST(Schedule, WholeRowsThenLongLeftoversSplit) {
  const lane_map lanes{.lanes = 8};
  const schedule even = make_schedule(rows(65536, 16)->graph, lanes, profile());
  EXPECT_EQ(share_of(even, lanes, 0).split, 0U);
  const schedule nine = make_schedule(rows(65536, 9)->graph, lanes, profile());
  const group_share sh = share_of(nine, lanes, 0);
  EXPECT_EQ(sh.whole, 8U);
  EXPECT_EQ(sh.split, 1U);
  EXPECT_EQ(sh.cap, 8);
  const schedule short_rows = make_schedule(rows(3072, 9)->graph, lanes, profile());
  EXPECT_EQ(share_of(short_rows, lanes, 0).split, 0U);
  const schedule eleven = make_schedule(rows(65536, 11)->graph, lanes, profile());
  EXPECT_EQ(share_of(eleven, lanes, 0).split, 3U);
  EXPECT_EQ(share_of(eleven, lanes, 0).cap, 2); // 8 members / 3 leftovers
}

// Rule 3: split only inside a group, with enough points per member, and on a large
// mesh-connected group only from about 8000 points.
TEST(Schedule, SplitThresholds) {
  // 3072 points: worth splitting in a small group, not on an 18-lane mesh.
  const schedule ccx = make_schedule(rows(3072, 1)->graph, {.lanes = 8}, profile());
  EXPECT_EQ(share_of(ccx, {.lanes = 8}, 0).split, 1U);
  EXPECT_LE(share_of(ccx, {.lanes = 8}, 0).cap, 6); // 3072 / 512 points per member
  const schedule mesh = make_schedule(rows(3072, 1)->graph, {.lanes = 18}, profile());
  EXPECT_EQ(share_of(mesh, {.lanes = 18}, 0).split, 0U);
  const schedule long_mesh = make_schedule(rows(16384, 1)->graph, {.lanes = 18}, profile());
  EXPECT_EQ(share_of(long_mesh, {.lanes = 18}, 0).split, 1U);
  // Never more members than the row has chunks in its widest phase, nor than the group has.
  const lane_map grouped{.lanes = 32, .group_size = 4};
  const schedule g = make_schedule(rows(65536, 1)->graph, grouped, profile());
  EXPECT_EQ(g.groups, 1);
  EXPECT_LE(share_of(g, grouped, 0).cap, 4);
  // When synchronisation costs more than splitting saves, the row stays whole.
  machine_profile slow_sync = profile();
  slow_sync.sync_ns = 1e9;
  const schedule costly = make_schedule(rows(65536, 1)->graph, {.lanes = 8}, slow_sync);
  EXPECT_EQ(share_of(costly, {.lanes = 8}, 0).split, 0U);
  // A row too short to give two members 512 points each.
  machine_profile low = profile();
  low.split_min_points = 1;
  low.dispatch_ns = 0;
  const schedule tiny = make_schedule(rows(1000, 1)->graph, {.lanes = 8}, low);
  EXPECT_EQ(share_of(tiny, {.lanes = 8}, 0).split, 0U);
}

// Rules 4 and 5: rows are dealt to groups in proportion to their members; a batch smaller
// than a group stays in one group.
TEST(Schedule, DealsToGroups) {
  const lane_map lanes{.lanes = 18, .group_size = 8}; // groups of 8, 8 and 2
  const schedule many = make_schedule(rows(1024, 1800)->graph, lanes, profile());
  EXPECT_EQ(many.groups, 3);
  expect_partition(many, lanes);
  EXPECT_EQ(share_of(many, lanes, 0).count, 800U);
  EXPECT_EQ(share_of(many, lanes, 2).count, 200U);
  const schedule few = make_schedule(rows(65536, 4)->graph, lanes, profile());
  EXPECT_EQ(few.groups, 1);
  expect_partition(few, lanes);
  EXPECT_EQ(share_of(few, lanes, 1).count, 0U);
  EXPECT_EQ(share_of(few, lanes, 0).split, 4U);
  EXPECT_EQ(share_of(few, lanes, 0).cap, 2);
  // Twelve rows: two groups' worth.
  const schedule twelve = make_schedule(rows(65536, 12)->graph, lanes, profile());
  EXPECT_EQ(twelve.groups, 2);
  expect_partition(twelve, lanes);
}

// An oversubscribed team (more members than CPUs) splits nothing.
TEST(Schedule, OversubscribedSplitsNothing) {
  machine_profile p = profile();
  p.cpus = 4;
  const lane_map lanes{.lanes = 8};
  const schedule s = make_schedule(rows(65536, 1)->graph, lanes, p);
  EXPECT_TRUE(s.oversubscribed);
  EXPECT_EQ(share_of(s, lanes, 0).split, 0U);
  p.cpus = 0; // unknown: never oversubscribed
  EXPECT_FALSE(make_schedule(rows(65536, 1)->graph, lanes, p).oversubscribed);
}

// A pure function: equal inputs, equal schedules; the default profile knows the machine.
TEST(Schedule, PureAndDefaults) {
  const auto r = rows(65536, 3);
  const lane_map lanes{.lanes = 12, .group_size = 4};
  const schedule a = make_schedule(r->graph, lanes, profile());
  const schedule b = make_schedule(r->graph, lanes, profile());
  for (int g = 0; g < lanes.groups(); ++g) {
    const group_share x = share_of(a, lanes, g);
    const group_share y = share_of(b, lanes, g);
    EXPECT_EQ(x.first, y.first);
    EXPECT_EQ(x.count, y.count);
    EXPECT_EQ(x.split, y.split);
    EXPECT_EQ(x.cap, y.cap);
  }
  EXPECT_GT(fftn::detail::default_machine_profile().cpus, 0);
  // Batches beyond the 24-bit counters are claimed in units of several transforms, which
  // still cover the batch once; teams beyond them run inline.
  work_graph huge = r->graph;
  huge.tasks = (std::size_t{1} << 40U) + 12345;
  const schedule h = make_schedule(huge, lanes, profile());
  EXPECT_FALSE(h.inline_all);
  EXPECT_GT(h.grain, 1U);
  expect_partition(h, lanes);
  std::size_t units = 0;
  for (int g = 0; g < lanes.groups(); ++g) {
    const group_share sh = share_of(h, lanes, g);
    EXPECT_LE(fftn::detail::whole_units(h, sh), fftn::detail::max_counted);
    units += fftn::detail::whole_units(h, sh) + sh.split;
  }
  EXPECT_EQ(fftn::detail::done_target(h, lanes), units);
  EXPECT_TRUE(make_schedule(r->graph, {.lanes = 4 * 1024 * 1024}, profile()).inline_all);
}

TEST(Schedule, LaneMaps) {
  const lane_map one{.lanes = 5};
  EXPECT_EQ(one.groups(), 1);
  EXPECT_EQ(one.size_of(0), 5);
  const lane_map big{.lanes = 5, .group_size = 9};
  EXPECT_EQ(big.groups(), 1);
  const lane_map split{.lanes = 10, .group_size = 4};
  EXPECT_EQ(split.groups(), 3);
  EXPECT_EQ(split.group_of(9), 2);
  EXPECT_EQ(split.base(2), 8);
  EXPECT_EQ(split.size_of(2), 2);
}

TEST(Schedule, OwnedTasksPartition) {
  // Fewer transforms than members: the first members own one each, the rest none.
  for (int l = 0; l < 4; ++l) {
    const auto r = fftn::detail::owned_tasks(2, 4, l);
    EXPECT_EQ(r.count, l < 2 ? 1U : 0U) << l;
    EXPECT_EQ(r.first, l < 2 ? static_cast<std::size_t>(l) : 2U) << l;
  }
  for (const std::size_t tasks :
       {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{64}, std::size_t{1000003}}) {
    for (const int lanes : {1, 2, 3, 8, 13, 64}) {
      std::size_t next = 0;
      for (int l = 0; l < lanes; ++l) {
        const auto r = fftn::detail::owned_tasks(tasks, lanes, l);
        EXPECT_EQ(r.first, next);
        next += r.count;
      }
      EXPECT_EQ(next, tasks);
    }
  }
}

} // namespace
