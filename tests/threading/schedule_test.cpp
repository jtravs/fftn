// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler's rules (DESIGN.md 13.4), one by one, on work graphs read off real plans,
// and the work graph itself. The schedule is a pure function: the same inputs give the same
// schedule, and it never depends on anything but its arguments.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "platform/cpu_features.hpp"
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
#include <utility>
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

// X59: a phase whose neighbouring lines write the same cache lines goes in fair runs.
TEST(WorkGraph, FairRunsWhereNeighbouringLinesShareCacheLines) {
  namespace d = fftn::detail;
  const fftn::options opts;
  d::build_settings s;
  s.chunk_points = 256;
  s.column_block = 8;
  s.profile.line_bytes = 64;
  s.profile.fair_line_runs = true;
  const auto plane = [&](std::size_t n0, std::size_t n1) {
    return d::make_plan_core(
        d::view_of(fftn::planes_c2c<double>(n0, n1, 1, fftn::direction::forward)), opts, s);
  };
  // Blocks of 8 complex doubles and rows of 32: both whole cache lines.
  const auto whole = plane(64, 32);
  ASSERT_EQ(whole->graph.phases.size(), 2U);
  EXPECT_FALSE(whole->graph.phases[0].fair_runs); // the rows
  EXPECT_FALSE(whole->graph.phases[1].fair_runs);
  // Rows of 30 complex doubles are 480 bytes: blocks meet inside cache lines.
  EXPECT_TRUE(plane(64, 30)->graph.phases[1].fair_runs);
  // A real plane's rows hold 33 complex doubles.
  fftn::shape<double, std::complex<double>> real;
  real.axes = {{.n = 64}, {.n = 64}};
  real.batches = {{.count = 1}};
  const auto r2c = d::make_plan_core(d::view_of(real), opts, s);
  EXPECT_FALSE(r2c->graph.phases[0].fair_runs);
  EXPECT_TRUE(r2c->graph.phases[1].fair_runs);
  // Columns alone through the chain are 16 bytes wide.
  s.lanes_columns = false;
  EXPECT_TRUE(plane(64, 32)->graph.phases[1].fair_runs);
  // A profile without fair runs (the Xeon 6240's) flags nothing.
  s.profile.fair_line_runs = false;
  EXPECT_FALSE(plane(64, 32)->graph.phases[1].fair_runs);
  s.profile.fair_line_runs = true;
  // Volumes keep the guided runs, whatever their lines share.
  fftn::shape<double, std::complex<double>> volume;
  volume.axes = {{.n = 16}, {.n = 16}, {.n = 16}};
  volume.batches = {{.count = 1}};
  // The plan is held: a range over a member of a temporary would outlive the temporary.
  const auto vol = d::make_plan_core(d::view_of(volume), opts, s);
  for (const d::graph_phase& phase : vol->graph.phases) {
    EXPECT_FALSE(phase.fair_runs);
  }

  d::graph_phase ph;
  ph.chunks = 16;
  ph.chunk_points = 4096;
  ph.fair_runs = true;
  EXPECT_EQ(d::chunk_run(ph, 16, 8, 16), 2U); // a fair share, 16 / 8
  EXPECT_EQ(d::chunk_run(ph, 1, 8, 16), 1U);  // never more than is left
  ph.chunks = 5;
  EXPECT_EQ(d::chunk_run(ph, 5, 8, 16), 1U); // fewer chunks than members
  ph.chunks = 64;
  ph.chunk_points = d::max_fair_points / 8;
  EXPECT_EQ(d::chunk_run(ph, 64, 8, 16), 8U); // a share of max_fair_points
  ph.chunk_points = d::max_fair_points / 4;   // larger: guided, for the balance of its end
  EXPECT_EQ(d::chunk_run(ph, 64, 8, 16), 4U);
  EXPECT_EQ(d::chunk_run(ph, 20, 8, 16), 1U);
  ph.fair_runs = false;
  ph.chunk_points = 4096;
  EXPECT_EQ(d::chunk_run(ph, 64, 8, 16), 4U); // a share of what is left
  EXPECT_EQ(d::chunk_run(ph, 3, 8, 16), 1U);
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
  // Planes: one phase of lines per axis; the columns alone through the chain, or in blocks of
  // adjacent columns through the lanes (X40), each block a line of the phase.
  const fftn::options opts;
  fftn::detail::build_settings fine;
  fine.chunk_points = 256;
  fine.lanes_columns = false;
  const auto p = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::planes_c2c<double>(64, 32, 1, fftn::direction::forward)), opts,
      fine);
  ASSERT_EQ(p->graph.phases.size(), 2U);
  EXPECT_EQ(p->graph.phases[0].nest, 1U);
  EXPECT_EQ(p->graph.phases[0].units, 64U); // lines of 32 along axis 1
  EXPECT_EQ(p->graph.phases[0].chunks, 8U); // 256 / 32 lines per chunk
  EXPECT_EQ(p->graph.phases[1].units, 32U);
  fine.lanes_columns = true;
  fine.column_block = 8;
  const auto blocks = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::planes_c2c<double>(64, 32, 1, fftn::direction::forward)), opts,
      fine);
  ASSERT_EQ(blocks->graph.phases.size(), 2U);
  EXPECT_EQ(blocks->graph.phases[1].nest, 1U);
  EXPECT_EQ(blocks->graph.phases[1].units, 4U);  // blocks of 8 of the 32 columns
  EXPECT_EQ(blocks->graph.phases[1].chunks, 4U); // a block holds 64 x 8 points
  // 30 columns: three blocks of 8 and a narrower one of 6 (rounded up to C), whose nest
  // joins the whole blocks' phase (loop::joins, X41) rather than running alone after it.
  const auto tail = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::planes_c2c<double>(64, 30, 1, fftn::direction::forward)), opts,
      fine);
  ASSERT_EQ(tail->graph.phases.size(), 2U);
  EXPECT_EQ(tail->graph.phases[1].units, 4U);
  EXPECT_EQ(tail->graph.phases[1].head, 3U);
  EXPECT_EQ(tail->graph.phases[1].tail_nest, 1U);
  EXPECT_EQ(tail->graph.phases[1].chunks, 4U);
  // Coarse chunks: 40 columns are two blocks of 16 and a tail; two lines per chunk, so the second
  // chunk holds the last whole block and the tail (team_env::run_chunk splits it at head).
  fftn::detail::build_settings coarse = fine;
  coarse.chunk_points = 2048;
  coarse.column_block = 16;
  const auto straddle = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::planes_c2c<double>(64, 40, 1, fftn::direction::forward)), opts,
      coarse);
  ASSERT_EQ(straddle->graph.phases.size(), 2U);
  EXPECT_EQ(straddle->graph.phases[1].units, 3U);
  EXPECT_EQ(straddle->graph.phases[1].head, 2U);
  EXPECT_EQ(straddle->graph.phases[1].chunks, 2U);
  // A volume (task 4.2): its two inner axes plane at a time, one phase whose lines are the
  // 4 whole planes (the loop over them holds each plane's own loops), then the outer axis,
  // whose two loops over the other axes are one phase of their product.
  fftn::shape<std::complex<double>, std::complex<double>> v;
  v.axes = {{.n = 4}, {.n = 5}, {.n = 6}};
  fftn::detail::build_settings alone;
  alone.lanes_columns = false;
  alone.volume_planes = true;
  const auto vol = fftn::detail::make_plan_core(fftn::detail::view_of(v), opts, alone);
  ASSERT_EQ(vol->graph.phases.size(), 2U);
  EXPECT_EQ(vol->graph.phases[0].nest, 1U);
  EXPECT_EQ(vol->graph.phases[0].units, 4U);
  EXPECT_EQ(vol->graph.phases[1].nest, 2U);
  EXPECT_EQ(vol->graph.phases[1].units, 30U);
  // With the lanes, the outer axis takes one block of the 6 adjacent pencils (rounded up to
  // C, the idle lanes never written) per line of axis 1.
  fftn::detail::build_settings six;
  six.column_block = 6;
  six.volume_planes = true;
  const auto lanes_vol = fftn::detail::make_plan_core(fftn::detail::view_of(v), opts, six);
  ASSERT_EQ(lanes_vol->graph.phases.size(), 2U);
  EXPECT_EQ(lanes_vol->graph.phases[0].units, 4U);
  EXPECT_EQ(lanes_vol->graph.phases[1].nest, 2U);
  EXPECT_EQ(lanes_vol->graph.phases[1].units, 5U);
  // A volume's planes are cut by a plane's points: 64 x 64 x 64 at 2048 points per chunk
  // has 64 chunks of one plane of 4096 points each.
  fftn::shape<std::complex<double>, std::complex<double>> cube;
  cube.axes = {{.n = 64}, {.n = 64}, {.n = 64}};
  fftn::detail::build_settings planes;
  planes.volume_planes = true;
  const auto big = fftn::detail::make_plan_core(fftn::detail::view_of(cube), opts, planes);
  EXPECT_EQ(big->graph.phases[0].units, 64U);
  EXPECT_EQ(big->graph.phases[0].chunks, 64U);
  // Axis by axis (the profile's default): one phase per axis, the two loops over the other
  // axes one phase of their product.
  alone.volume_planes = false;
  const auto axes = fftn::detail::make_plan_core(fftn::detail::view_of(v), opts, alone);
  ASSERT_EQ(axes->graph.phases.size(), 3U);
  EXPECT_EQ(axes->graph.phases[0].nest, 2U);
  EXPECT_EQ(axes->graph.phases[0].units, 20U);
  EXPECT_EQ(axes->graph.phases[2].units, 30U);
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

// Planes and volumes (X52 arm S): a real plane is split by its complex points where the
// profile says (a private L2), and on a mesh a plane is split only from
// split_min_points_lines points.
TEST(Schedule, PlaneSplitThresholds) {
  const auto plane = [](bool real, const machine_profile& p) {
    fftn::detail::build_settings settings;
    settings.profile = p;
    return real ? fftn::detail::make_plan_core(
                      fftn::detail::view_of(fftn::planes_r2c<double>(96, 96, 1)), fftn::options(),
                      settings)
                : fftn::detail::make_plan_core(fftn::detail::view_of(fftn::planes_c2c<double>(
                                                   96, 96, 1, fftn::direction::forward)),
                                               fftn::options(), settings);
  };
  const lane_map lanes{.lanes = 8};
  const auto split = [&](bool real, const machine_profile& p) {
    const schedule s = make_schedule(plane(real, p)->graph, lanes, p);
    return share_of(s, lanes, 0).split;
  };
  machine_profile p = profile();
  EXPECT_EQ(split(false, p), 1U); // 9216 points
  EXPECT_EQ(split(true, p), 1U);  // 9216 real points
  p.split_complex_points = true;  // 96 x 49 complex points: below 6144
  EXPECT_EQ(split(false, p), 1U);
  EXPECT_EQ(split(true, p), 0U);
  machine_profile mesh = profile();
  mesh.split_min_points_lines = 32768;
  EXPECT_EQ(split(false, mesh), 0U);
  // Rows are unaffected by either.
  mesh.split_complex_points = true;
  const schedule row = make_schedule(rows(65536, 1)->graph, lanes, mesh);
  EXPECT_EQ(share_of(row, lanes, 0).split, 1U);
  // The recorded machines: the Xeon's mesh and both x86 private L2s, not the M5 Max.
  const std::filesystem::path data(FFTN_PLATFORM_DATA);
  const machine_profile m5 =
      fftn::detail::profile_for(fftn::detail::load_topology(data / "macos-apple-m5-max.sysctl"));
  const machine_profile epyc =
      fftn::detail::profile_for(fftn::detail::load_topology(data / "linux-amd-epyc-7763.sysfs"));
  const machine_profile xeon = fftn::detail::profile_for(
      fftn::detail::load_topology(data / "linux-intel-xeon-gold-6240.sysfs"));
  EXPECT_EQ(xeon.split_min_points_lines, 32768U);
  EXPECT_EQ(epyc.split_min_points_lines, 0U);
  EXPECT_EQ(m5.split_min_points_lines, 0U);
  EXPECT_TRUE(xeon.split_complex_points);
  EXPECT_TRUE(epyc.split_complex_points);
  EXPECT_FALSE(m5.split_complex_points);
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
  // The architecture is the recorded machine's, never the host's (X62: the Bluestein limit of
  // Zen 3's class is set on x86-64 only), and so is the vendor (the Xeon 6240's class is Intel's
  // only, DESIGN.md 13.5).
  const auto sizes = [&](const char* file, bool x86_64,
                         fftn::detail::cpu_vendor vendor = fftn::detail::cpu_vendor::unknown) {
    machine_profile p;
    p.vendor = vendor;
    fftn::detail::set_grid_sizes(p, fftn::detail::load_topology(data / file).classes.at(0), x86_64);
    return p;
  };
  constexpr std::size_t kib = 1024;
  const machine_profile xeon = sizes("linux-intel-xeon-gold-6240.sysfs", true,
                                     fftn::detail::cpu_vendor::intel); // 32 KiB, 1 MiB
  EXPECT_EQ(xeon.grid_min_bytes, 512 * kib);
  EXPECT_EQ(xeon.grid_three_bytes, 1024 * kib);
  EXPECT_EQ(xeon.grid_block_bytes, 32 * kib);
  EXPECT_EQ(xeon.grid_fuse_bytes, 256 * kib);
  const machine_profile epyc =
      sizes("linux-amd-epyc-7763.sysfs", true, fftn::detail::cpu_vendor::amd); // 32 KiB, 512 KiB
  EXPECT_EQ(epyc.grid_min_bytes, 256 * kib);
  EXPECT_EQ(epyc.grid_three_bytes, 512 * kib);
  EXPECT_EQ(epyc.grid_block_bytes, 32 * kib);
  const machine_profile m5 = sizes("macos-apple-m5-max.sysctl", false); // 128 KiB, 16 MiB for 6
  EXPECT_EQ(m5.grid_min_bytes, 128 * kib);
  EXPECT_EQ(m5.grid_three_bytes, 2048 * kib);
  EXPECT_EQ(m5.grid_block_bytes, 128 * kib);
  EXPECT_EQ(m5.grid_fuse_bytes, 4096 * kib);
  // The grid pads its intermediate where the L2 is shared by a cluster (X26b).
  EXPECT_EQ(xeon.grid_pad_period, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(epyc.grid_pad_period, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(m5.grid_pad_period, 4 * kib);
  // Bluestein lengths avoid large powers of two there too (X36).
  EXPECT_EQ(m5.bluestein_pow2_part_bytes, 32 * kib);
  EXPECT_EQ(xeon.bluestein_pow2_part_bytes, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(epyc.bluestein_pow2_part_bytes, 16 * kib); // half the L1 on Zen 3's class (X62)
  // Lengths that are not powers of two stay chains longer where the L2 is shared (X29).
  EXPECT_EQ(m5.chain_part_bytes, 16 * kib);
  EXPECT_EQ(m5.chain_mixed_bytes, 256 * kib);
  EXPECT_EQ(m5.chain_part_limit_bytes, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(xeon.chain_part_bytes, 0U);
  EXPECT_EQ(epyc.chain_mixed_bytes, 0U);
  // With 4 MiB of L3 per CPU the EPYC keeps them chains to half of its L3 (X29).
  EXPECT_EQ(epyc.chain_part_bytes, 16 * kib);
  EXPECT_EQ(epyc.chain_part_limit_bytes, std::size_t{16} * 1024 * kib);
  // Without that L3 the Xeon keeps rows with an odd part of 15 or more in two phases (X34).
  EXPECT_EQ(xeon.grid_two_odd_part, 15U);
  EXPECT_EQ(epyc.grid_two_odd_part, 0U);
  EXPECT_EQ(m5.grid_two_odd_part, 0U);
  // Team programs (X43, X43b), as an x86-64 build sets them: the EPYC's teams run narrower
  // column blocks, both machines' teams run volumes plane at a time; the M5 Max neither.
  // The recorded machines' vendors (DESIGN.md 13.5: X63's key on the Xeon 6240's class reads it).
  const auto x86 = [&](const char* file,
                       fftn::detail::cpu_vendor vendor = fftn::detail::cpu_vendor::unknown) {
    machine_profile p;
    p.vendor = vendor;
    fftn::detail::set_grid_sizes(p, fftn::detail::load_topology(data / file).classes.at(0), true);
    return p;
  };
  const machine_profile epyc86 = x86("linux-amd-epyc-7763.sysfs", fftn::detail::cpu_vendor::amd);
  const machine_profile xeon86 =
      x86("linux-intel-xeon-gold-6240.sysfs", fftn::detail::cpu_vendor::intel);
  EXPECT_EQ(epyc86.column_team_bytes, 128U);
  EXPECT_TRUE(epyc86.team_volume_planes);
  EXPECT_EQ(xeon86.column_team_bytes, 0U);
  EXPECT_TRUE(xeon86.team_volume_planes);
  EXPECT_EQ(m5.column_team_bytes, 0U);
  EXPECT_FALSE(m5.team_volume_planes);
  // Fair runs for column phases whose lines share cache lines (X59): where measured to win, the
  // EPYC 7763 (Zen 3) and the M5 Max; not the Xeon 6240 (lost), nor Zen 5 (not measured).
  EXPECT_TRUE(epyc86.fair_line_runs);
  EXPECT_FALSE(xeon86.fair_line_runs);
  EXPECT_FALSE(x86("linux-gcp-c4d-zen5.sysfs").fair_line_runs);
  machine_profile apple;
  fftn::detail::set_grid_sizes(
      apple, fftn::detail::load_topology(data / "macos-apple-m5-max.sysctl").classes.at(0), false);
  EXPECT_TRUE(apple.fair_line_runs);
  // Whole transforms anchored to members only where measured: Zen 3's class (X73) and Apple
  // (X74).
  EXPECT_TRUE(epyc86.anchored_runs);
  EXPECT_FALSE(xeon86.anchored_runs);
  EXPECT_TRUE(apple.anchored_runs);
  for (const auto& [file, x86_64] : {std::pair{"linux-gcp-c4a-axion.sysfs", false},
                                     std::pair{"linux-gcp-n2-ice-lake.sysfs", true},
                                     std::pair{"linux-gcp-c4-granite-rapids.sysfs", true}}) {
    SCOPED_TRACE(file);
    EXPECT_FALSE(
        fftn::detail::profile_for(fftn::detail::load_topology(data / file), x86_64).anchored_runs);
  }
  EXPECT_FALSE(x86("linux-gcp-c4d-zen5.sysfs").anchored_runs);
  // The team state on pages of its own on the Xeon 6240's class only (X79, narrowed to it by the
  // maintainer): the EPYC 7763, the M5 Max, Granite Rapids, the Ice Lake VM, Zen 5 and Axion keep
  // it on its 128-byte lines.
  EXPECT_EQ(xeon86.team_state_page, 4096U);
  EXPECT_EQ(epyc86.team_state_page, 0U);
  EXPECT_EQ(apple.team_state_page, 0U);
  EXPECT_EQ(
      x86("linux-gcp-c4-granite-rapids.sysfs", fftn::detail::cpu_vendor::intel).team_state_page,
      0U);
  EXPECT_EQ(x86("linux-gcp-n2-ice-lake.sysfs", fftn::detail::cpu_vendor::intel).team_state_page,
            0U);
  EXPECT_EQ(x86("linux-gcp-c4d-zen5.sysfs", fftn::detail::cpu_vendor::amd).team_state_page, 0U);
  EXPECT_EQ(
      fftn::detail::profile_for(fftn::detail::load_topology(data / "linux-gcp-c4a-axion.sysfs"),
                                false, fftn::detail::cpu_vendor::arm_other)
          .team_state_page,
      0U);
  // Rows' two-pass radices while the data fit three quarters of the CCX's L3 (X57).
  EXPECT_EQ(epyc86.fit_row_bytes, std::size_t{24} * 1024 * kib);
  EXPECT_EQ(xeon86.fit_row_bytes, 0U);
  EXPECT_EQ(m5.fit_row_bytes, 0U);
  // In place only to the footprint measured (X57 amendment 2): 1024 rows of 1024, half the L3.
  EXPECT_EQ(epyc86.fit_in_place_bytes, std::size_t{16} * 1024 * kib);
  EXPECT_EQ(xeon86.fit_in_place_bytes, std::numeric_limits<std::size_t>::max());
  // A Bluestein stage over C copies keeps the whole line on Zen 3's class only (X62).
  EXPECT_TRUE(epyc86.bluestein_whole_lanes);
  EXPECT_FALSE(xeon86.bluestein_whole_lanes);
  EXPECT_FALSE(m5.bluestein_whole_lanes);
  EXPECT_FALSE(x86("linux-gcp-c4d-zen5.sysfs").bluestein_whole_lanes);
  EXPECT_EQ(x86("linux-gcp-c4d-zen5.sysfs").bluestein_pow2_part_bytes, 48 * kib);
  // Real rows' half-length depth-first chain on Zen 3's class (X72): the rows' range.
  EXPECT_EQ(epyc86.depth_real_max_bytes, epyc86.depth_max_bytes);
  EXPECT_EQ(epyc86.depth_real_max_bytes, std::size_t{2048} * kib);
  // And on the Xeon 6240's class, over its rows' range (X75).
  EXPECT_EQ(xeon86.depth_real_max_bytes, xeon86.depth_max_bytes);
  EXPECT_EQ(xeon86.depth_real_max_bytes, std::size_t{64} * 1024 * kib);
  // Above L2/8 there, the c2c rows' L2/4 elsewhere (X75 arm B, narrowed).
  EXPECT_EQ(xeon86.depth_real_min_bytes, std::size_t{128} * kib + 1);
  EXPECT_EQ(epyc86.depth_real_min_bytes, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(m5.depth_real_min_bytes, std::numeric_limits<std::size_t>::max());
  // X78: the fused c2r pairing on the Xeon 6240's class only, L2/4 to 16 L2.
  EXPECT_EQ(xeon86.depth_c2r_fused_min_bytes, std::size_t{256} * kib);
  EXPECT_EQ(xeon86.depth_c2r_fused_max_bytes, std::size_t{16} * 1024 * kib);
  EXPECT_EQ(epyc86.depth_c2r_fused_max_bytes, 0U);
  EXPECT_EQ(m5.depth_c2r_fused_max_bytes, 0U);
  EXPECT_EQ(m5.depth_real_max_bytes, 0U);
  EXPECT_EQ(x86("linux-gcp-c4d-zen5.sysfs").depth_real_max_bytes, 0U);
  EXPECT_EQ(x86("linux-gcp-n2-ice-lake.sysfs").depth_real_max_bytes, 0U);
  EXPECT_EQ(x86("linux-gcp-c4-granite-rapids.sysfs").depth_real_max_bytes, 0U);
  // The c2r grid pairs in its first gather only where the L2 is shared (X29).
  EXPECT_TRUE(m5.c2r_paired_gather);
  EXPECT_FALSE(epyc.c2r_paired_gather);
  EXPECT_FALSE(xeon.c2r_paired_gather);
  // A c2r plane packs its intermediate in the output only on the classes measured (X60, X63):
  // beyond the Xeon 6240's L2 on avx512, beyond half the EPYC's CCX L3 on avx2; the block on
  // Apple, Zen 5 and elsewhere.
  EXPECT_EQ(xeon86.c2r_packed_bytes, 1024 * kib);
  EXPECT_EQ(xeon86.c2r_packed_isa, fftn::isa::avx512);
  // On Intel's cores only (X64): an AMD core the same key reaches (Zen 4c with SMT, a Zen 4 VM
  // reporting a small L3 per CPU: 1 MiB of private L2, under 2 MiB of L3 per CPU) keeps the
  // block, and so does an unknown vendor; Zen 3's avx2 branch does not read the vendor.
  {
    fftn::detail::core_class zen4c;
    zen4c.l1d = {.size = 32 * kib, .line_size = 64, .ways = 8, .shared_by = 2};
    zen4c.l2 = {.size = 1024 * kib, .line_size = 64, .ways = 8, .shared_by = 2};
    zen4c.l3 = {.size = 16 * 1024 * kib, .line_size = 64, .ways = 16, .shared_by = 16};
    const auto packed = [&](fftn::detail::cpu_vendor v) {
      machine_profile p;
      p.vendor = v;
      fftn::detail::set_grid_sizes(p, zen4c, true);
      return std::pair{p.c2r_packed_bytes, p.c2r_packed_isa};
    };
    EXPECT_EQ(packed(fftn::detail::cpu_vendor::intel),
              (std::pair{std::size_t{1024} * kib, fftn::isa::avx512}));
    for (const auto v : {fftn::detail::cpu_vendor::amd, fftn::detail::cpu_vendor::unknown}) {
      EXPECT_EQ(packed(v), (std::pair{std::numeric_limits<std::size_t>::max(), fftn::isa::best}));
    }
    EXPECT_EQ(x86("linux-intel-xeon-gold-6240.sysfs").c2r_packed_bytes,
              std::numeric_limits<std::size_t>::max());
    EXPECT_EQ(x86("linux-amd-epyc-7763.sysfs").c2r_packed_bytes, std::size_t{16} * 1024 * kib);
    EXPECT_EQ(fftn::detail::profile_for(
                  fftn::detail::load_topology(data / "linux-intel-xeon-gold-6240.sysfs"), true,
                  fftn::detail::cpu_vendor::intel)
                  .c2r_packed_bytes,
              1024 * kib);
  }
  EXPECT_EQ(epyc86.c2r_packed_bytes, std::size_t{16} * 1024 * kib);
  EXPECT_EQ(epyc86.c2r_packed_isa, fftn::isa::avx2);
  EXPECT_EQ(apple.c2r_packed_bytes, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(x86("linux-gcp-c4d-zen5.sysfs").c2r_packed_bytes,
            std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(machine_profile{}.c2r_packed_bytes, std::numeric_limits<std::size_t>::max());
  // Power-of-two rows stay chains to four times a shared L2, radix 8 up to an eighth of it;
  // to twice L3 with radix 8 on the EPYC; never on the Xeon (X32).
  EXPECT_EQ(m5.pow2_chain_bytes, std::size_t{64} * 1024 * kib);
  EXPECT_EQ(m5.pow2_radix8_bytes, 2048 * kib);
  EXPECT_EQ(epyc.pow2_chain_bytes, std::size_t{64} * 1024 * kib);
  EXPECT_EQ(epyc.pow2_radix8_bytes, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(xeon.pow2_chain_bytes, 0U);
  EXPECT_EQ(xeon.grid_two_part_bytes, 0U);
  EXPECT_EQ(epyc.grid_two_part_bytes, 0U);
  EXPECT_EQ(m5.grid_two_part_bytes, 0U);
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

// Experiment X35: the planner's fields of the profile of every recorded machine. The M5 Max,
// the EPYC 7763, the Zen 5 VM (Google Cloud C4D, 8 cores) and the Xeon Gold 6240 keep the
// values they had before X35; on x86-64 a private L2 of 2 MiB or more (the Granite Rapids
// VM, Google Cloud C4) takes grids for long powers of two, two phases where the power-of-two
// part is at most 2048 complex doubles, the grid and three phases beyond
// half of L2, whatever its L3 per CPU (63 MiB in that VM). The Axion VM (C4A, AArch64,
// also 2 MiB of private L2) keeps the profile it had: it lost with X35's. X36's two fields
// follow: Granite Rapids takes the Xeon's rules (no Bluestein length limit, long
// convolutions beyond 4 L2), the Axion VM the Zen ones; neither measured. X55: Granite Rapids
// takes the Xeon's depth-first chain from L2/4 to 64 L2 in double; X64, X50's convolutions on
// that chain up to 32 MiB.
struct planner_fields {
  std::size_t grid_min_bytes;
  std::size_t grid_three_bytes;
  std::size_t grid_block_bytes;
  std::size_t grid_fuse_bytes;
  std::size_t grid_pad_period;
  std::size_t grid_two_odd_part;
  std::size_t grid_two_part_bytes;
  std::size_t chain_part_bytes;
  std::size_t chain_mixed_bytes;
  std::size_t chain_part_limit_bytes;
  std::size_t pow2_chain_bytes;
  std::size_t pow2_radix8_bytes;
  bool c2r_paired_gather;
  std::size_t grid_three_block;            // X39
  std::size_t grid_two_block_bytes;        // X39
  std::size_t pow2_grid_bytes;             // X39
  std::size_t bluestein_pow2_part_bytes;   // X36
  std::size_t long_convolution_bytes;      // X36
  std::size_t grid_phase_radix;            // X39 (arm D)
  std::size_t grid_two_pow2_bytes;         // X39 (arm D)
  std::size_t depth_min_bytes;             // X46
  std::size_t depth_max_bytes;             // X46
  std::size_t depth_part_bytes;            // X46 (arm C)
  std::size_t depth_top;                   // X46
  std::size_t depth_top_alt;               // X46
  std::size_t depth_top_last;              // X51
  std::size_t depth_pow2_max_bytes;        // X51
  std::size_t depth_float_max_bytes;       // X51
  std::size_t depth_convolution_max_bytes; // X50

  bool operator==(const planner_fields&) const = default;
};

planner_fields fields_of(const machine_profile& p) {
  return {
      .grid_min_bytes = p.grid_min_bytes,
      .grid_three_bytes = p.grid_three_bytes,
      .grid_block_bytes = p.grid_block_bytes,
      .grid_fuse_bytes = p.grid_fuse_bytes,
      .grid_pad_period = p.grid_pad_period,
      .grid_two_odd_part = p.grid_two_odd_part,
      .grid_two_part_bytes = p.grid_two_part_bytes,
      .chain_part_bytes = p.chain_part_bytes,
      .chain_mixed_bytes = p.chain_mixed_bytes,
      .chain_part_limit_bytes = p.chain_part_limit_bytes,
      .pow2_chain_bytes = p.pow2_chain_bytes,
      .pow2_radix8_bytes = p.pow2_radix8_bytes,
      .c2r_paired_gather = p.c2r_paired_gather,
      .grid_three_block = p.grid_three_block,
      .grid_two_block_bytes = p.grid_two_block_bytes,
      .pow2_grid_bytes = p.pow2_grid_bytes,
      .bluestein_pow2_part_bytes = p.bluestein_pow2_part_bytes,
      .long_convolution_bytes = p.long_convolution_bytes,
      .grid_phase_radix = p.grid_phase_radix,
      .grid_two_pow2_bytes = p.grid_two_pow2_bytes,
      .depth_min_bytes = p.depth_min_bytes,
      .depth_max_bytes = p.depth_max_bytes,
      .depth_part_bytes = p.depth_part_bytes,
      .depth_top = p.depth_top,
      .depth_top_alt = p.depth_top_alt,
      .depth_top_last = p.depth_top_last,
      .depth_pow2_max_bytes = p.depth_pow2_max_bytes,
      .depth_float_max_bytes = p.depth_float_max_bytes,
      .depth_convolution_max_bytes = p.depth_convolution_max_bytes,
  };
}

TEST(Schedule, ProfilesOfRecordedMachines) {
  const std::filesystem::path data(FFTN_PLATFORM_DATA);
  using fftn::detail::cpu_vendor;
  // Each recording with its processor's vendor (the Intel-only keys read it, DESIGN.md 13.5).
  const auto fields = [&](const char* file, bool x86_64, cpu_vendor vendor) {
    return fields_of(
        fftn::detail::profile_for(fftn::detail::load_topology(data / file), x86_64, vendor));
  };
  constexpr std::size_t kib = 1024;
  constexpr std::size_t mib = 1024 * kib;
  constexpr std::size_t never = std::numeric_limits<std::size_t>::max();
  // The values before X35 (main at 2eea943).
  EXPECT_EQ(
      fields("macos-apple-m5-max.sysctl", false, cpu_vendor::apple),
      (planner_fields{128 * kib, 2 * mib,  128 * kib, 4 * mib, 4 * kib,  0, 0, 16 * kib, 256 * kib,
                      never,     64 * mib, 2 * mib,   true,    0,        0, 0, 32 * kib, 128 * kib,
                      0,         0,        128 * kib, 2 * mib, 16 * kib, 4, 0, 0,        256 * kib,
                      0,         256 * kib}));
  // X62: the EPYC 7763's Bluestein power-of-two limit is half its L1.
  EXPECT_EQ(fields("linux-amd-epyc-7763.sysfs", true, cpu_vendor::amd),
            (planner_fields{256 * kib, 512 * kib, 32 * kib,  128 * kib, never,    0,
                            0,         16 * kib,  0,         16 * mib,  64 * mib, never,
                            false,     0,         0,         0,         16 * kib, 256 * kib,
                            0,         0,         128 * kib, 2 * mib,   8 * kib,  4,
                            0,         0,         0,         0,         0}));
  EXPECT_EQ(fields("linux-gcp-c4d-zen5.sysfs", true, cpu_vendor::amd),
            (planner_fields{512 * kib, 1 * mib,   48 * kib, 256 * kib, never, 0, 0, 16 * kib,
                            0,         16 * mib,  64 * mib, never,     false, 0, 0, 0,
                            48 * kib,  512 * kib, 0,        0,         0,     0, 0, 0,
                            0,         0,         never,    0,         0}));
  EXPECT_EQ(fields("linux-intel-xeon-gold-6240.sysfs", true, cpu_vendor::intel),
            (planner_fields{512 * kib, 1 * mib,     32 * kib,  256 * kib, never,   15,
                            0,         0,           0,         never,     0,       0,
                            false,     16,          64 * kib,  256 * kib, never,   4 * mib,
                            20,        12672 * kib, 256 * kib, 64 * mib,  0,       16,
                            20,        10,          never,     8 * mib,   32 * mib}));
  const planner_fields axion{
      .grid_min_bytes = 1 * mib,
      .grid_three_bytes = 2 * mib,
      .grid_block_bytes = 64 * kib,
      .grid_fuse_bytes = 512 * kib,
      .grid_pad_period = never,
      .grid_two_odd_part = 0,
      .grid_two_part_bytes = 0,
      .chain_part_bytes = 16 * kib,
      .chain_mixed_bytes = 0,
      .chain_part_limit_bytes = 40 * mib,
      .pow2_chain_bytes = 160 * mib,
      .pow2_radix8_bytes = never,
      .c2r_paired_gather = false,
      .grid_three_block = 0,
      .grid_two_block_bytes = 0,
      .pow2_grid_bytes = 0,
      .bluestein_pow2_part_bytes = 64 * kib,
      .long_convolution_bytes = 1 * mib,
      .grid_phase_radix = 0,
      .grid_two_pow2_bytes = 0,
      .depth_min_bytes = 0,
      .depth_max_bytes = 0,
      .depth_part_bytes = 0,
      .depth_top = 0,
      .depth_top_alt = 0,
      .depth_top_last = 0,
      .depth_pow2_max_bytes = never,
      .depth_float_max_bytes = 0,
      .depth_convolution_max_bytes = 0,
  };
  EXPECT_EQ(fields("linux-gcp-c4a-axion.sysfs", false, cpu_vendor::arm_other), axion);
  // A private L2 of 2 MiB on x86-64.
  const planner_fields large_l2{
      .grid_min_bytes = 1 * mib,
      .grid_three_bytes = 1 * mib,
      .grid_block_bytes = 48 * kib,
      .grid_fuse_bytes = 512 * kib,
      .grid_pad_period = never,
      .grid_two_odd_part = 0,
      .grid_two_part_bytes = 32 * kib,
      .chain_part_bytes = 0,
      .chain_mixed_bytes = 0,
      .chain_part_limit_bytes = never,
      .pow2_chain_bytes = 0,
      .pow2_radix8_bytes = 0,
      .c2r_paired_gather = false,
      .grid_three_block = 0,
      .grid_two_block_bytes = 0,
      .pow2_grid_bytes = 0,
      .bluestein_pow2_part_bytes = never,
      .long_convolution_bytes = 8 * mib,
      .grid_phase_radix = 0,
      .grid_two_pow2_bytes = 0,
      .depth_min_bytes = 512 * kib, // X55
      .depth_max_bytes = 128 * mib,
      .depth_part_bytes = 0,
      .depth_top = 16,
      .depth_top_alt = 20,
      .depth_top_last = 10,
      .depth_pow2_max_bytes = never,
      .depth_float_max_bytes = 0,
      .depth_convolution_max_bytes = 32 * mib, // X50 on the key (X64)
  };
  EXPECT_EQ(fields("linux-gcp-c4-granite-rapids.sysfs", true, cpu_vendor::intel), large_l2);
  // Sapphire Rapids (c3-standard-8) and Emerald Rapids (c4-standard-16) VMs: the same L1 and L2,
  // so the same fields (X64, X69, X70 were judged there).
  EXPECT_EQ(fields("linux-gcp-c3-sapphire-rapids.sysfs", true, cpu_vendor::intel), large_l2);
  EXPECT_EQ(fields("linux-gcp-c4-emerald-rapids.sysfs", true, cpu_vendor::intel), large_l2);
  // X58: a private L2 of 2 MiB leaves out the batch leaf of 12; the other machines keep it.
  const auto batch_leaf_12 = [&](const char* file, cpu_vendor vendor) {
    return fftn::detail::profile_for(fftn::detail::load_topology(data / file), true, vendor)
        .batch_leaf_12;
  };
  EXPECT_FALSE(batch_leaf_12("linux-gcp-c4-granite-rapids.sysfs", cpu_vendor::intel));
  EXPECT_FALSE(batch_leaf_12("linux-gcp-c3-sapphire-rapids.sysfs", cpu_vendor::intel));
  EXPECT_FALSE(batch_leaf_12("linux-gcp-c4-emerald-rapids.sysfs", cpu_vendor::intel));
  EXPECT_TRUE(batch_leaf_12("linux-intel-xeon-gold-6240.sysfs", cpu_vendor::intel));
  EXPECT_TRUE(batch_leaf_12("linux-gcp-c4d-zen5.sysfs", cpu_vendor::amd));
  // X58: an Ice Lake VM (1.25 MiB of private L2, much L3 per CPU) takes the depth-first chain.
  const planner_fields ice_lake = fields("linux-gcp-n2-ice-lake.sysfs", true, cpu_vendor::intel);
  EXPECT_EQ(ice_lake.depth_min_bytes, 320 * kib);
  EXPECT_EQ(ice_lake.depth_max_bytes, 80 * mib);
  EXPECT_EQ(ice_lake.depth_top, 16U);
  EXPECT_EQ(ice_lake.depth_top_alt, 20U);
  EXPECT_EQ(ice_lake.depth_top_last, 10U);
  EXPECT_EQ(ice_lake.depth_float_max_bytes, 0U);
  EXPECT_TRUE(batch_leaf_12("linux-gcp-n2-ice-lake.sysfs", cpu_vendor::intel));
  // The same on bare metal with SMT, where Granite Rapids' L3 per CPU is under 2 MiB
  // (504 MiB for 128 cores, 256 CPUs): the key is the L2, not the L3 per CPU.
  fftn::detail::core_class metal;
  metal.l1d = {.size = 48 * kib, .line_size = 64, .ways = 12, .shared_by = 2};
  metal.l2 = {.size = 2 * mib, .line_size = 64, .ways = 16, .shared_by = 2};
  metal.l3 = {.size = 504 * mib, .line_size = 64, .ways = 16, .shared_by = 256};
  machine_profile p;
  p.vendor = cpu_vendor::intel;
  fftn::detail::set_grid_sizes(p, metal, true);
  EXPECT_EQ(fields_of(p), large_l2);
  // Off x86-64 the rule does not apply: those caches take the large-L3 profile (X29, X32).
  EXPECT_EQ(
      fields("linux-gcp-c4-granite-rapids.sysfs", false, cpu_vendor::arm_other).pow2_chain_bytes,
      2 * std::size_t{516096} * kib);
}

// X69: the 2 MiB-L2 key's fit table (machine_profile::fit_large_l2) with its limits, on that key
// alone. X80: no limit in place, 64 times the L2 for rows and output out of place, the L2 for rows
// longer than a page out of place (X70). The other recorded classes keep their fit fields as
// before: X57's on the EPYC's class (and, unmeasured, on Zen 4 and Zen 5, whose `avx512` plans
// have no fit table without the key), none on the Xeon 6240's or Apple's, and no out-of-place or
// long-row limit on any of them.
TEST(Schedule, FitTableKeys) {
  const std::filesystem::path data(FFTN_PLATFORM_DATA);
  constexpr std::size_t kib = 1024;
  constexpr std::size_t mib = 1024 * kib;
  constexpr std::size_t never = std::numeric_limits<std::size_t>::max();
  struct fit_fields {
    bool large_l2;
    std::size_t row_bytes;
    std::size_t in_place_bytes;
    std::size_t long_row_bytes;
    std::size_t out_of_place_bytes;
    bool operator==(const fit_fields&) const = default;
  };
  const auto of = [](const machine_profile& p) {
    return fit_fields{p.fit_large_l2, p.fit_row_bytes, p.fit_in_place_bytes, p.fit_long_row_bytes,
                      p.fit_out_of_place_bytes};
  };
  using fftn::detail::cpu_vendor;
  const auto recorded = [&](const char* file, bool x86_64, cpu_vendor vendor) {
    return of(fftn::detail::profile_for(fftn::detail::load_topology(data / file), x86_64, vendor));
  };
  for (const char* file :
       {"linux-gcp-c4-granite-rapids.sysfs", "linux-gcp-c3-sapphire-rapids.sysfs",
        "linux-gcp-c4-emerald-rapids.sysfs"}) {
    SCOPED_TRACE(file);
    EXPECT_EQ(recorded(file, true, cpu_vendor::intel),
              (fit_fields{true, never, never, 2 * mib, 128 * mib}));
  }
  EXPECT_EQ(recorded("linux-amd-epyc-7763.sysfs", true, cpu_vendor::amd),
            (fit_fields{false, 24 * mib, 16 * mib, never, never}));
  EXPECT_EQ(recorded("linux-intel-xeon-gold-6240.sysfs", true, cpu_vendor::intel),
            (fit_fields{false, 0, never, never, never}));
  EXPECT_EQ(recorded("macos-apple-m5-max.sysctl", false, cpu_vendor::apple),
            (fit_fields{false, 0, never, never, never}));
  // AArch64 with a 2 MiB private L2 (Axion) is not on the key.
  EXPECT_FALSE(recorded("linux-gcp-c4a-axion.sysfs", false, cpu_vendor::arm_other).large_l2);
  // Zen 5 and Ice Lake VMs (a large L3 per CPU): X57's large-L3 fields, without the key.
  for (const auto& [file, vendor] : {std::pair{"linux-gcp-c4d-zen5.sysfs", cpu_vendor::amd},
                                     std::pair{"linux-gcp-n2-ice-lake.sysfs", cpu_vendor::intel}}) {
    SCOPED_TRACE(file);
    const fftn::detail::core_class c = fftn::detail::load_topology(data / file).classes.at(0);
    EXPECT_EQ(recorded(file, true, vendor),
              (fit_fields{false, c.l3.size / 4 * 3, c.l3.size / 2, never, never}));
  }
  // Zen 4 (1 MiB of private L2) with a large L3 per CPU (a VM) and without (Zen 4c with SMT).
  fftn::detail::core_class zen4;
  zen4.l1d = {.size = 32 * kib, .line_size = 64, .ways = 8, .shared_by = 1};
  zen4.l2 = {.size = 1 * mib, .line_size = 64, .ways = 8, .shared_by = 1};
  zen4.l3 = {.size = 32 * mib, .line_size = 64, .ways = 16, .shared_by = 8};
  machine_profile p;
  fftn::detail::set_grid_sizes(p, zen4, true);
  EXPECT_EQ(of(p), (fit_fields{false, 24 * mib, 16 * mib, never, never}));
  zen4.l1d.shared_by = 2;
  zen4.l2.shared_by = 2;
  zen4.l3 = {.size = 16 * mib, .line_size = 64, .ways = 16, .shared_by = 16};
  p = machine_profile{};
  fftn::detail::set_grid_sizes(p, zen4, true);
  EXPECT_EQ(of(p), (fit_fields{false, 0, never, never, never}));
  // Bare-metal Granite Rapids with SMT (under 2 MiB of L3 per CPU): the key is the L2.
  fftn::detail::core_class metal;
  metal.l1d = {.size = 48 * kib, .line_size = 64, .ways = 12, .shared_by = 2};
  metal.l2 = {.size = 2 * mib, .line_size = 64, .ways = 16, .shared_by = 2};
  metal.l3 = {.size = 504 * mib, .line_size = 64, .ways = 16, .shared_by = 256};
  p = machine_profile{};
  p.vendor = cpu_vendor::intel;
  fftn::detail::set_grid_sizes(p, metal, true);
  EXPECT_EQ(of(p), (fit_fields{true, never, never, 2 * mib, 128 * mib}));
}

// A rule measured on one vendor's processors only is keyed to that vendor (the maintainer,
// 2026-11-25; DESIGN.md 13.5). The Xeon 6240's class (X34, X39, X46, X50, X51, X58, X61's x86
// half, X63, X75, X78, X79) and the 2 MiB-L2 key (X35, X55, X58, X64, X69, X70, X80) were
// measured on Intel's cores only, so they read the vendor: every Intel recording keeps the profile
// it had (the literals here and in the tests above are main's values before the vendor was read,
// with X79's and X80's fields since), and AMD's cores with the same caches, or an unknown vendor,
// keep the fields' defaults there.
struct intel_keyed {
  std::size_t grid_two_odd_part;
  std::size_t grid_three_block;
  std::size_t grid_phase_radix;
  std::size_t grid_two_block_bytes;
  std::size_t pow2_grid_bytes;
  std::size_t grid_two_pow2_bytes;
  std::size_t grid_two_part_bytes;
  std::size_t depth_min_bytes;
  std::size_t depth_max_bytes;
  std::size_t depth_top;
  std::size_t depth_top_alt;
  std::size_t depth_top_last;
  std::size_t depth_float_max_bytes;
  std::size_t depth_convolution_max_bytes;
  std::size_t depth_real_max_bytes;
  std::size_t depth_real_min_bytes;
  std::size_t depth_c2r_fused_min_bytes;
  std::size_t depth_c2r_fused_max_bytes;
  bool adjacent_lanes;
  bool adjacent_team_chain;
  bool batch_leaf_12;
  bool fit_large_l2;
  std::size_t fit_long_row_bytes;
  std::size_t fit_out_of_place_bytes;
  std::size_t c2r_packed_bytes;
  fftn::isa c2r_packed_isa;
  std::size_t team_state_page;

  bool operator==(const intel_keyed&) const = default;
};

intel_keyed keyed_of(const machine_profile& p) {
  return {
      .grid_two_odd_part = p.grid_two_odd_part,
      .grid_three_block = p.grid_three_block,
      .grid_phase_radix = p.grid_phase_radix,
      .grid_two_block_bytes = p.grid_two_block_bytes,
      .pow2_grid_bytes = p.pow2_grid_bytes,
      .grid_two_pow2_bytes = p.grid_two_pow2_bytes,
      .grid_two_part_bytes = p.grid_two_part_bytes,
      .depth_min_bytes = p.depth_min_bytes,
      .depth_max_bytes = p.depth_max_bytes,
      .depth_top = p.depth_top,
      .depth_top_alt = p.depth_top_alt,
      .depth_top_last = p.depth_top_last,
      .depth_float_max_bytes = p.depth_float_max_bytes,
      .depth_convolution_max_bytes = p.depth_convolution_max_bytes,
      .depth_real_max_bytes = p.depth_real_max_bytes,
      .depth_real_min_bytes = p.depth_real_min_bytes,
      .depth_c2r_fused_min_bytes = p.depth_c2r_fused_min_bytes,
      .depth_c2r_fused_max_bytes = p.depth_c2r_fused_max_bytes,
      .adjacent_lanes = p.adjacent_lanes,
      .adjacent_team_chain = p.adjacent_team_chain,
      .batch_leaf_12 = p.batch_leaf_12,
      .fit_large_l2 = p.fit_large_l2,
      .fit_long_row_bytes = p.fit_long_row_bytes,
      .fit_out_of_place_bytes = p.fit_out_of_place_bytes,
      .c2r_packed_bytes = p.c2r_packed_bytes,
      .c2r_packed_isa = p.c2r_packed_isa,
      .team_state_page = p.team_state_page,
  };
}

TEST(Schedule, IntelOnlyKeys) {
  const std::filesystem::path data(FFTN_PLATFORM_DATA);
  using fftn::detail::cpu_vendor;
  constexpr std::size_t kib = 1024;
  constexpr std::size_t mib = 1024 * kib;
  constexpr std::size_t never = std::numeric_limits<std::size_t>::max();
  const auto recorded = [&](const char* file, cpu_vendor vendor) {
    return fftn::detail::profile_for(fftn::detail::load_topology(data / file), true, vendor);
  };
  const auto with_caches = [](const fftn::detail::core_class& c, cpu_vendor vendor) {
    machine_profile p;
    p.vendor = vendor;
    fftn::detail::set_grid_sizes(p, c, true);
    return p;
  };
  const intel_keyed defaults = keyed_of(machine_profile{});
  // Every Intel recording keeps its profile.
  const intel_keyed xeon{
      .grid_two_odd_part = 15,
      .grid_three_block = 16,
      .grid_phase_radix = 20,
      .grid_two_block_bytes = 64 * kib,
      .pow2_grid_bytes = 256 * kib,
      .grid_two_pow2_bytes = 12672 * kib,
      .grid_two_part_bytes = 0,
      .depth_min_bytes = 256 * kib,
      .depth_max_bytes = 64 * mib,
      .depth_top = 16,
      .depth_top_alt = 20,
      .depth_top_last = 10,
      .depth_float_max_bytes = 8 * mib,
      .depth_convolution_max_bytes = 32 * mib,
      .depth_real_max_bytes = 64 * mib,
      .depth_real_min_bytes = 128 * kib + 1,
      .depth_c2r_fused_min_bytes = 256 * kib,
      .depth_c2r_fused_max_bytes = 16 * mib,
      .adjacent_lanes = true,
      .adjacent_team_chain = true,
      .batch_leaf_12 = true,
      .fit_large_l2 = false,
      .fit_long_row_bytes = never,
      .fit_out_of_place_bytes = never,
      .c2r_packed_bytes = 1 * mib,
      .c2r_packed_isa = fftn::isa::avx512,
      .team_state_page = 4096, // X79 (not main's value before the vendor was read: added after)
  };
  EXPECT_EQ(keyed_of(recorded("linux-intel-xeon-gold-6240.sysfs", cpu_vendor::intel)), xeon);
  intel_keyed ice_lake = defaults; // X58's depth-first chain only
  ice_lake.depth_min_bytes = 320 * kib;
  ice_lake.depth_max_bytes = 80 * mib;
  ice_lake.depth_top = 16;
  ice_lake.depth_top_alt = 20;
  ice_lake.depth_top_last = 10;
  EXPECT_EQ(keyed_of(recorded("linux-gcp-n2-ice-lake.sysfs", cpu_vendor::intel)), ice_lake);
  intel_keyed large_l2 = defaults; // X35, X55, X58, X64, X69, X70 and X80
  large_l2.grid_two_part_bytes = 32 * kib;
  large_l2.depth_min_bytes = 512 * kib;
  large_l2.depth_max_bytes = 128 * mib;
  large_l2.depth_top = 16;
  large_l2.depth_top_alt = 20;
  large_l2.depth_top_last = 10;
  large_l2.depth_convolution_max_bytes = 32 * mib;
  large_l2.batch_leaf_12 = false;
  large_l2.fit_large_l2 = true;
  large_l2.fit_long_row_bytes = 2 * mib;
  large_l2.fit_out_of_place_bytes = 128 * mib; // X80
  const std::vector<const char*> large_l2_files{"linux-gcp-c4-granite-rapids.sysfs",
                                                "linux-gcp-c3-sapphire-rapids.sysfs",
                                                "linux-gcp-c4-emerald-rapids.sysfs"};
  for (const char* file : large_l2_files) {
    SCOPED_TRACE(file);
    const machine_profile p = recorded(file, cpu_vendor::intel);
    EXPECT_EQ(keyed_of(p), large_l2);
    EXPECT_EQ(p.grid_three_bytes, 1 * mib); // L2/2 on the key (X35)
  }
  // Unknown behaves as AMD: off. Zen 4c-like caches (1 MiB of private L2 shared by two SMT
  // threads, 16 MiB of L3 for 16 CPUs) fall into the Xeon 6240's key and keep the defaults.
  fftn::detail::core_class zen4c;
  zen4c.l1d = {.size = 32 * kib, .line_size = 64, .ways = 8, .shared_by = 2};
  zen4c.l2 = {.size = 1 * mib, .line_size = 64, .ways = 8, .shared_by = 2};
  zen4c.l3 = {.size = 16 * mib, .line_size = 64, .ways = 16, .shared_by = 16};
  // The Zen 5 recording with its L3 cut to 1 MiB per CPU (Zen 5c's): the same key.
  fftn::detail::core_class zen5c =
      fftn::detail::load_topology(data / "linux-gcp-c4d-zen5.sysfs").classes.at(0);
  zen5c.l3.size = zen5c.l3.shared_by * mib;
  for (const cpu_vendor vendor : {cpu_vendor::amd, cpu_vendor::unknown}) {
    SCOPED_TRACE(fftn::detail::cpu_vendor_name(vendor));
    for (const fftn::detail::core_class& c : {zen4c, zen5c}) {
      const machine_profile p = with_caches(c, vendor);
      EXPECT_EQ(keyed_of(p), defaults);
      EXPECT_TRUE(p.team_volume_planes); // X43b: measured on AMD's cores and Intel's
      // On Intel's cores the same caches take the Xeon 6240's fields.
      EXPECT_EQ(with_caches(c, cpu_vendor::intel).depth_top, 16U);
      EXPECT_TRUE(with_caches(c, cpu_vendor::intel).adjacent_lanes);
    }
    // The Intel recordings' caches with another vendor: the defaults on those keys too.
    EXPECT_EQ(keyed_of(recorded("linux-intel-xeon-gold-6240.sysfs", vendor)), defaults);
    EXPECT_EQ(keyed_of(recorded("linux-gcp-n2-ice-lake.sysfs", vendor)), defaults);
    for (const char* file : large_l2_files) {
      SCOPED_TRACE(file);
      const machine_profile p = recorded(file, vendor);
      EXPECT_EQ(keyed_of(p), defaults);
      // A 2 MiB L2 off the key takes neither the large-L3 branch (X29, X32, X57) nor the key's
      // L2/2: the sizes every key starts from.
      EXPECT_EQ(p.grid_three_bytes, 2 * mib);
      EXPECT_EQ(p.chain_part_bytes, 0U);
      EXPECT_EQ(p.pow2_chain_bytes, 0U);
      EXPECT_EQ(p.fit_row_bytes, 0U);
      EXPECT_EQ(p.fit_in_place_bytes, never);
    }
  }
  // The AMD recordings do not read the vendor (Zen 3's and the large-L3 branch's rules).
  for (const char* file : {"linux-amd-epyc-7763.sysfs", "linux-gcp-c4d-zen5.sysfs"}) {
    SCOPED_TRACE(file);
    EXPECT_EQ(fields_of(recorded(file, cpu_vendor::amd)),
              fields_of(recorded(file, cpu_vendor::intel)));
    EXPECT_EQ(keyed_of(recorded(file, cpu_vendor::amd)),
              keyed_of(recorded(file, cpu_vendor::unknown)));
  }
}

} // namespace
