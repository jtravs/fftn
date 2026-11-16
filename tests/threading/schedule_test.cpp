// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler's rules (DESIGN.md 13.4), one by one, on work graphs read off real plans,
// and the work graph itself. The schedule is a pure function: the same inputs give the same
// schedule, and it never depends on anything but its arguments.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "platform/topology.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
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

// The thresholds below were set on planner v0's plans. Its 3072-point plan had five passes
// (4 x 4 x 4 x 4 x 12 on 32 registers); planner v1 needs three, a third less work, which
// moves that row across rule 3's thresholds. The rules are tested on the plan they were
// written for, forced as the tests force any sequence.
// Chains: the rules below are stated for a chain's graph (a phase per stage), which long
// rows would not have on a machine whose caches make them grids (strategies/grid.hpp).
std::shared_ptr<const fftn::detail::plan_core> rows(std::size_t n, std::size_t count) {
  const fftn::options opts;
  fftn::detail::build_settings settings;
  settings.strategy = fftn::detail::strategy_choice::chain;
  if (n == 3072) {
    settings.radices = {4, 4, 4, 4, 12};
  }
  // The work of a generic pass (below): its prime factor is not a Bluestein stage.
  if (n == 58) {
    settings.bluestein_above = static_cast<std::size_t>(-1);
  }
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
  fftn::detail::build_settings chain;
  chain.strategy = fftn::detail::strategy_choice::chain;
  const auto f = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<float>(16384, 1, fftn::direction::forward)),
      fftn::options(), chain);
  EXPECT_EQ(f->graph.work, 16384.0 * static_cast<double>(f->graph.phases.size()));
  // A grid: one phase per phase of the grid, each a loop nest over its blocks of columns
  // (two loops, over columns c and blocks, after the first phase of three).
  fftn::detail::build_settings grid;
  grid.strategy = fftn::detail::strategy_choice::grid;
  grid.grid_factors = {16, 32, 32};
  grid.profile.grid_block = 8;
  grid.profile.grid_pad_period = std::numeric_limits<std::size_t>::max();
  const auto gr = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<double>(16384, 2, fftn::direction::forward)),
      fftn::options(), grid);
  ASSERT_EQ(gr->graph.phases.size(), 3U);
  EXPECT_EQ(gr->graph.phases[0].nest, 1U);
  EXPECT_EQ(gr->graph.phases[0].units, 1024U / 8); // blocks of the 1024 columns of phase 1
  EXPECT_EQ(gr->graph.phases[1].nest, 2U);
  EXPECT_EQ(gr->graph.phases[1].units, 32U * (16 / 8)); // columns c x blocks of L = 16
  EXPECT_EQ(gr->graph.phases[2].nest, 1U);
  EXPECT_EQ(gr->graph.phases[2].units, 512U / 8); // blocks of L = 512, the last phase
  // A padded intermediate (X26b): phase 1 loops over phase 2's 32 rows, then their blocks.
  grid.profile.grid_pad_period = 4096;
  const auto padded = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<double>(16384, 2, fftn::direction::forward)),
      fftn::options(), grid);
  ASSERT_EQ(padded->graph.phases.size(), 3U);
  EXPECT_EQ(padded->graph.phases[0].nest, 2U);
  EXPECT_EQ(padded->graph.phases[0].units, 32U * (32 / 8));
  EXPECT_EQ(padded->graph.phases[1].units, 32U * (16 / 8));
  // A generic pass of radix r sums r terms per point: 2 x 29 is a grouped radix-2 pass (58
  // points) and a generic radix-29 pass (58 x 29).
  EXPECT_EQ(rows(58, 1)->graph.work, 58.0 + 58.0 * 29);
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
// mesh-connected group only from longer rows (T5).
TEST(Schedule, SplitThresholds) {
  // 6144 points: worth splitting in a small group, not on an 18-lane mesh.
  const schedule ccx = make_schedule(rows(6144, 1)->graph, {.lanes = 8}, profile());
  EXPECT_EQ(share_of(ccx, {.lanes = 8}, 0).split, 1U);
  const schedule mesh = make_schedule(rows(6144, 1)->graph, {.lanes = 18}, profile());
  EXPECT_EQ(share_of(mesh, {.lanes = 18}, 0).split, 0U);
  const schedule long_mesh = make_schedule(rows(32768, 1)->graph, {.lanes = 18}, profile());
  EXPECT_EQ(share_of(long_mesh, {.lanes = 18}, 0).split, 1U);
  // Below the threshold of a small group the row stays whole.
  const schedule short_row = make_schedule(rows(4096, 1)->graph, {.lanes = 8}, profile());
  EXPECT_EQ(share_of(short_row, {.lanes = 8}, 0).split, 0U);
  // At most points / 512 members.
  machine_profile any_length = profile();
  any_length.split_min_points = 1;
  any_length.sync_ns = 0;
  const schedule capped = make_schedule(rows(3072, 1)->graph, {.lanes = 8}, any_length);
  EXPECT_EQ(share_of(capped, {.lanes = 8}, 0).split, 1U);
  EXPECT_LE(share_of(capped, {.lanes = 8}, 0).cap, 6); // 3072 / 512 points per member
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

// Rule 4 under an executor: with nothing split, no more members are dispatched than there
// are tasks, and a team of that many members gets the same (unsplit) division.
TEST(Schedule, DispatchesNoMoreMembersThanTasksWhenNothingSplits) {
  machine_profile never = profile();
  never.split_min_points = std::numeric_limits<std::size_t>::max();
  never.split_min_points_mesh = std::numeric_limits<std::size_t>::max();
  const lane_map group8{.lanes = 8};
  for (const std::size_t k : {std::size_t{1}, std::size_t{3}, std::size_t{7}}) {
    const auto r = rows(65536, k);
    const schedule s = make_schedule(r->graph, group8, never);
    EXPECT_FALSE(s.inline_all);
    EXPECT_EQ(s.lanes_dealt, 8);
    EXPECT_EQ(s.dispatch, static_cast<int>(k)) << k;
    const lane_map team{.lanes = s.dispatch, .threads = 8};
    const schedule t = make_schedule(r->graph, team, never);
    EXPECT_EQ(t.lanes_dealt, s.dispatch);
    EXPECT_EQ(t.dispatch, s.dispatch);
    EXPECT_EQ(share_of(t, team, 0).split, 0U);
    EXPECT_EQ(share_of(t, team, 0).count, k);
    // Splitting allowed (at least two members per task): the whole group shares them.
    if (k <= 4) {
      const schedule split = make_schedule(r->graph, group8, profile());
      EXPECT_GT(share_of(split, group8, 0).split, 0U);
      EXPECT_EQ(split.dispatch, 8);
    }
  }
  // Several groups, nothing split: nine short rows on 64 lanes in groups of 8 take 9
  // members, a whole group and one member of the next.
  const lane_map wide{.lanes = 64, .group_size = 8};
  const auto nine = rows(1024, 9);
  machine_profile eager_dispatch = never;
  eager_dispatch.dispatch_ns = 0;
  const schedule s = make_schedule(nine->graph, wide, eager_dispatch);
  EXPECT_EQ(s.groups, 2);
  EXPECT_EQ(s.lanes_dealt, 16);
  EXPECT_EQ(s.dispatch, 9);
  const lane_map team{.lanes = 9, .group_size = 8, .threads = 64};
  const schedule t = make_schedule(nine->graph, team, eager_dispatch);
  expect_partition(t, team);
  EXPECT_EQ(share_of(t, team, 0).count, 8U);
  EXPECT_EQ(share_of(t, team, 1).count, 1U);
  EXPECT_EQ(share_of(t, team, 0).split + share_of(t, team, 1).split, 0U);
  // At least as many tasks as dealt lanes: everyone is dispatched.
  EXPECT_EQ(make_schedule(rows(1024, 64)->graph, wide, eager_dispatch).dispatch, 64);
}

// Rule 1 is decided once per plan, from the profile alone.
TEST(Schedule, RunsInlineAtConstruction) {
  EXPECT_TRUE(rows(64, 1)->runs_inline);
  EXPECT_TRUE(fftn::detail::runs_inline(rows(1728, 1)->graph, profile()));
  EXPECT_FALSE(rows(65536, 8)->runs_inline);
  const auto r = rows(1728, 64);
  EXPECT_EQ(fftn::detail::runs_inline(r->graph, profile()),
            make_schedule(r->graph, {.lanes = 8}, profile()).inline_all);
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

// A team dispatched on an executor of more threads than CPUs is oversubscribed though it has
// fewer members than CPUs: the executor's other threads compete for them too.
TEST(Schedule, OversubscribedByTheRuntimesThreads) {
  machine_profile p = profile();
  p.cpus = 4;
  const lane_map alone{.lanes = 4};
  EXPECT_FALSE(make_schedule(rows(65536, 1)->graph, alone, p).oversubscribed);
  EXPECT_GT(share_of(make_schedule(rows(65536, 1)->graph, alone, p), alone, 0).split, 0U);
  const lane_map dispatched{.lanes = 4, .threads = 16};
  const schedule s = make_schedule(rows(65536, 1)->graph, dispatched, p);
  EXPECT_TRUE(s.oversubscribed);
  EXPECT_EQ(share_of(s, dispatched, 0).split, 0U);
  const lane_map fewer{.lanes = 4, .threads = 2}; // never fewer threads than members
  EXPECT_FALSE(make_schedule(rows(65536, 1)->graph, fewer, p).oversubscribed);
}

// The profile follows the machine's kind (T1, T5): an Apple performance cluster splits from
// longer rows than a Zen 3 CCX; a mesh's threshold is the rules' (groups of more than
// mesh_group_lanes members), whatever the machine.
TEST(Schedule, ProfileForRecordedMachines) {
  const std::filesystem::path data(FFTN_PLATFORM_DATA);
  const machine_profile m5 =
      fftn::detail::profile_for(fftn::detail::load_topology(data / "macos-apple-m5-max.sysctl"));
  const machine_profile epyc =
      fftn::detail::profile_for(fftn::detail::load_topology(data / "linux-amd-epyc-7763.sysfs"));
  const machine_profile xeon = fftn::detail::profile_for(
      fftn::detail::load_topology(data / "linux-intel-xeon-gold-6240.sysfs"));
  const machine_profile ccx{};
  EXPECT_EQ(m5.split_min_points, 8192U);
  EXPECT_EQ(epyc.split_min_points, ccx.split_min_points);
  EXPECT_EQ(xeon.split_min_points, ccx.split_min_points);
  EXPECT_EQ(xeon.split_min_points_mesh, ccx.split_min_points_mesh);
  // T1's measured costs were adopted on the CCX only.
  EXPECT_EQ(epyc.sync_ns, ccx.sync_ns);
  EXPECT_EQ(epyc.claim_divisor, ccx.claim_divisor);
  for (const machine_profile& other : {m5, xeon}) {
    EXPECT_EQ(other.sync_ns, 700);
    EXPECT_EQ(other.claim_divisor, 2);
  }
  // The grid's sizes come from the caches (X12b: a private L2 starts the grid at half of it,
  // the EPYC's 512 KiB at 256 KiB; the grid's blocks stay within L1).
  EXPECT_EQ(epyc.grid_min_bytes, std::size_t{256} * 1024);
  EXPECT_EQ(epyc.grid_block_bytes, std::size_t{32} * 1024);
  // An Intel Mac (per-core L2s) is not an Apple processor: macOS alone does not decide.
  const machine_profile intel_mac =
      fftn::detail::profile_for(fftn::detail::load_topology(data / "macos-synthetic-intel.sysctl"));
  EXPECT_EQ(intel_mac.split_min_points, ccx.split_min_points);
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

// The grid's sizes from the recorded machines' caches (experiment X12b): a private L2
// (Xeon, EPYC) keeps the chain to half of L2 and two phases to L2, and with 2 MiB or more of
// L3 per CPU (EPYC) lengths whose power-of-two part is small to half of L3 (X29); an L2
// shared by a cluster (Apple) gets the grid beyond L1 and three phases beyond an eighth of
// L2.
TEST(Schedule, GridSizesFromCaches) {
  const std::filesystem::path data(FFTN_PLATFORM_DATA);
  const auto sizes = [&](const char* file) {
    machine_profile p;
    fftn::detail::set_grid_sizes(p, fftn::detail::load_topology(data / file).classes.at(0));
    return p;
  };
  constexpr std::size_t kib = 1024;
  const machine_profile xeon = sizes("linux-intel-xeon-gold-6240.sysfs"); // 32 KiB, 1 MiB
  EXPECT_EQ(xeon.grid_min_bytes, 512 * kib);
  EXPECT_EQ(xeon.grid_three_bytes, 1024 * kib);
  EXPECT_EQ(xeon.grid_block_bytes, 32 * kib);
  EXPECT_EQ(xeon.grid_fuse_bytes, 256 * kib);
  const machine_profile epyc = sizes("linux-amd-epyc-7763.sysfs"); // 32 KiB, 512 KiB
  EXPECT_EQ(epyc.grid_min_bytes, 256 * kib);
  EXPECT_EQ(epyc.grid_three_bytes, 512 * kib);
  EXPECT_EQ(epyc.grid_block_bytes, 32 * kib);
  const machine_profile m5 = sizes("macos-apple-m5-max.sysctl"); // 128 KiB, 16 MiB for 6
  EXPECT_EQ(m5.grid_min_bytes, 128 * kib);
  EXPECT_EQ(m5.grid_three_bytes, 2048 * kib);
  EXPECT_EQ(m5.grid_block_bytes, 128 * kib);
  EXPECT_EQ(m5.grid_fuse_bytes, 4096 * kib);
  // The grid pads its intermediate where the L2 is shared by a cluster (X26b).
  EXPECT_EQ(xeon.grid_pad_period, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(epyc.grid_pad_period, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(m5.grid_pad_period, 4 * kib);
  // Lengths that are not powers of two stay chains longer where the L2 is shared (X29).
  EXPECT_EQ(m5.chain_part_bytes, 16 * kib);
  EXPECT_EQ(m5.chain_mixed_bytes, 256 * kib);
  EXPECT_EQ(m5.chain_part_limit_bytes, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(xeon.chain_part_bytes, 0U);
  EXPECT_EQ(epyc.chain_mixed_bytes, 0U);
  // With 4 MiB of L3 per CPU the EPYC keeps them chains to half of its L3 (X29).
  EXPECT_EQ(epyc.chain_part_bytes, 16 * kib);
  EXPECT_EQ(epyc.chain_part_limit_bytes, 16 * 1024 * kib);
  // The c2r grid pairs in its first gather only where the L2 is shared (X29).
  EXPECT_TRUE(m5.c2r_paired_gather);
  EXPECT_FALSE(epyc.c2r_paired_gather);
  EXPECT_FALSE(xeon.c2r_paired_gather);
  // Power-of-two rows stay chains to four times a shared L2, radix 8 up to an eighth of it;
  // to twice L3 with radix 8 on the EPYC; never on the Xeon (X32).
  EXPECT_EQ(m5.pow2_chain_bytes, 64 * 1024 * kib);
  EXPECT_EQ(m5.pow2_radix8_bytes, 2048 * kib);
  EXPECT_EQ(epyc.pow2_chain_bytes, 64 * 1024 * kib);
  EXPECT_EQ(epyc.pow2_radix8_bytes, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(xeon.pow2_chain_bytes, 0U);
  // Unknown sizes keep the defaults; an unknown sharing counts as a shared L2.
  const machine_profile defaults;
  machine_profile unknown;
  fftn::detail::set_grid_sizes(unknown, fftn::detail::core_class{});
  EXPECT_EQ(unknown.grid_min_bytes, defaults.grid_min_bytes);
  EXPECT_EQ(unknown.grid_three_bytes, defaults.grid_three_bytes);
  fftn::detail::core_class no_sharing;
  no_sharing.l1d.size = 32 * kib;
  no_sharing.l2.size = 1024 * kib;
  machine_profile shared;
  fftn::detail::set_grid_sizes(shared, no_sharing);
  EXPECT_EQ(shared.grid_min_bytes, 32 * kib);
  EXPECT_EQ(shared.grid_three_bytes, 128 * kib);
}

} // namespace
