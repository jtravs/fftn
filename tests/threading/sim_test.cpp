// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The claiming engine under the deterministic simulator (task 3.2, DESIGN.md 13.7).
//
// Interleavings: seeded adversarial orders of the members' every access to shared state,
// with members that arrive together, late, one at a time, only one, or never, over several
// executions on one workspace (so that members of one execution meet members of the next).
// Every task must run exactly once, phases in order, every member must return only after
// the whole output is written, nothing may deadlock, and the output must equal serial
// execution bit for bit.
//
// Pathologies: the prototype's measured failures, reproduced in virtual time with a model
// of its team execution (static per-member schedules, a barrier after every pass of a
// shared row), and the same plans run by FFTN's scheduler and engine in the same model.

#include "threading/sim_env.hpp"
#include "threading/simulator.hpp"
#include "threading/team_cases.hpp"

#include "core/plan_core.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using fftn::test::team_case;
namespace sim = fftn::test::sim;

// ---- Interleavings.

const std::vector<team_case>& interleaving_cases() {
  static const std::vector<team_case> cases = [] {
    fftn::detail::build_settings settings = fftn::test::eager_settings();
    settings.chunk_points = 256; // fewer switch points than the real-thread tests' 64
    std::vector<team_case> all;
    fftn::test::add_cases<double>(all, settings, "");
    std::vector<team_case> chosen;
    for (team_case& c : all) {
      for (const char* name : {
               "c2c_2000x1",
               "c2c_360x7",
               "c2c_inplace_1024x3",
               "r2c_1000x5",
               "c2r_45x9",
               "c2c_planes_24x20x3",
               "c2r_planes_12x16x2",
               "c2c_volume_8x6x10x2",
               "c2c_many_64x100",
           }) {
        if (c.name == name) {
          chosen.push_back(c);
        }
      }
    }
    return chosen;
  }();
  return cases;
}

enum class arrival : std::uint8_t {
  together,
  all_but_one_late,
  last_never,
  only_first,
  random_steps,
};

std::vector<sim::member_spec> arrivals(arrival how, int lanes, std::uint64_t seed) {
  std::vector<sim::member_spec> m(static_cast<std::size_t>(lanes));
  std::mt19937_64 rng(seed);
  for (std::size_t i = 0; i < m.size(); ++i) {
    switch (how) {
    case arrival::together:
      break;
    case arrival::all_but_one_late:
      m[i].arrive_step = i == 0 ? 0 : 400;
      break;
    case arrival::last_never:
      m[i].arrives = i + 1 < m.size();
      break;
    case arrival::only_first:
      m[i].arrives = i == 0;
      break;
    case arrival::random_steps:
      m[i].arrive_step = static_cast<std::size_t>(rng() % 600);
      break;
    }
  }
  return m;
}

class Interleavings : public ::testing::TestWithParam<std::size_t> {};

TEST_P(Interleavings, EveryOrderCompletesWithSerialBits) {
  const team_case& c = interleaving_cases()[GetParam()];
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  for (std::uint64_t seed = 0; seed < 3; ++seed) {
    for (const int lanes : {2, 3, 8}) {
      for (const int group : {0, 2}) {
        for (const arrival how : {
                 arrival::together,
                 arrival::all_but_one_late,
                 arrival::last_never,
                 arrival::only_first,
                 arrival::random_steps,
             }) {
          const sim::sim_setup setup{
              .lanes = lanes,
              .group_size = group,
              .how = sim::order::random,
              .seed = seed * 7919 + static_cast<std::uint64_t>(lanes),
              .costs = sim::cost_model(),
              .members = arrivals(how, lanes, seed),
              .executions = 3,
          };
          const sim::sim_result r = sim::simulate_engine(c, setup);
          const std::string where = c.name + ", seed " + std::to_string(seed) + ", " +
                                    std::to_string(lanes) + " lanes, groups of " +
                                    std::to_string(group) + ", arrival " +
                                    std::to_string(static_cast<int>(how));
          ASSERT_FALSE(r.deadlocked) << where;
          ASSERT_TRUE(r.problems.empty()) << where << ": " << r.problems.front();
          for (const auto& out : r.outputs) {
            EXPECT_EQ(out, serial) << where;
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Cases, Interleavings,
                         ::testing::Range(std::size_t{0}, interleaving_cases().size()),
                         [](const ::testing::TestParamInfo<std::size_t>& info) {
                           return interleaving_cases()[info.param].name;
                         });

// The simulator really interleaves: different seeds give different orders, and the split
// path shares chunks among members.
TEST(Interleavings, SeedsChangeTheOrder) {
  const team_case& c = interleaving_cases()[0];
  std::vector<std::vector<std::size_t>> shares;
  for (std::uint64_t seed = 0; seed < 4; ++seed) {
    const sim::sim_setup setup{
        .lanes = 4,
        .group_size = 0,
        .how = sim::order::random,
        .seed = seed,
        .costs = sim::cost_model(),
        .members = std::vector<sim::member_spec>(),
    };
    const sim::sim_result r = sim::simulate_engine(c, setup);
    ASSERT_TRUE(r.problems.empty());
    std::printf("[ sim      ] seed %d: chunks by member", static_cast<int>(seed));
    for (const std::size_t n : r.chunks_by_member) {
      std::printf(" %zu", n);
    }
    std::printf(", %zu steps\n", r.steps);
    EXPECT_GE(std::ranges::count_if(r.chunks_by_member, [](std::size_t n) { return n != 0; }), 2)
        << "seed " << seed;
    shares.push_back(r.chunks_by_member);
  }
  EXPECT_FALSE(shares[0] == shares[1] && shares[1] == shares[2] && shares[2] == shares[3]);
}

// ---- Pathologies, in virtual time.

// Plans with the library's default chunking and machine profile, as users get them.
team_case default_rows(std::size_t n, std::size_t count) {
  return fftn::test::make_case("c2c_" + std::to_string(n) + "x" + std::to_string(count),
                               fftn::rows_c2c<double>(n, count, fftn::direction::forward),
                               n * count, n * count, false, fftn::detail::build_settings{});
}

// The virtual times, for the log: the prototype's, FFTN's and serial execution's.
void report(const char* what, double proto, double ours, double serial) {
  std::printf("[ sim      ] %s: prototype %.1f us, FFTN %.1f us, serial %.1f us\n", what,
              proto / 1000, ours / 1000, serial / 1000);
}

double serial_ns(const team_case& c, const sim::cost_model& costs) {
  return static_cast<double>(c.core->prog.transforms) * c.core->graph.work * costs.ns_per_work;
}

// The provisional defaults, on a machine with a CPU for every lane: the simulations must not
// depend on the machine that runs them.
const fftn::detail::machine_profile& defaults() {
  static const fftn::detail::machine_profile p;
  return p;
}

sim::sim_result fftn_timed(const team_case& c, int lanes, int group,
                           const std::vector<sim::member_spec>& members,
                           const sim::cost_model& costs) {
  const sim::sim_setup setup{
      .lanes = lanes,
      .group_size = group,
      .how = sim::order::timed,
      .seed = 0,
      .costs = costs,
      .members = members,
      .executions = 1,
      .profile = &defaults(),
  };
  sim::sim_result r = sim::simulate_engine(c, setup);
  EXPECT_FALSE(r.deadlocked) << c.name;
  EXPECT_TRUE(r.problems.empty()) << c.name;
  EXPECT_EQ(r.outputs.front(), fftn::test::serial_result(c)) << c.name;
  return r;
}

// The prototype shares every transform of the batch across the whole team, in turn.
sim::sim_result prototype_shared(const team_case& c, int lanes,
                                 const std::vector<sim::member_spec>& members,
                                 const sim::cost_model& costs) {
  sim::static_schedule s(static_cast<std::size_t>(lanes));
  std::size_t barriers = 0;
  sim::share_in_turn(s, c.core->graph, 0, c.core->prog.transforms, 0, lanes, barriers);
  return sim::simulate_static(c, s, barriers, costs, members);
}

std::vector<sim::member_spec> waking(int lanes, double every_ns) {
  std::vector<sim::member_spec> m(static_cast<std::size_t>(lanes));
  for (std::size_t i = 0; i < m.size(); ++i) {
    m[i].arrive_ns = static_cast<double>(i) * every_ns;
  }
  return m;
}

// A short job threaded across lanes that wake one after another: a library that splits it
// regardless waits for the last to wake (FFTW took c2c 1728 x 1 from 3.9 to 74 us at 8
// threads). Rule 1 runs it inline on the first member to arrive.
TEST(Pathologies, ShortJobOnLanesThatWakeLate) {
  const sim::cost_model costs;
  const team_case c = default_rows(1728, 1);
  const auto members = waking(8, 10000);
  const double serial = serial_ns(c, costs);
  const sim::sim_result proto = prototype_shared(c, 8, members, costs);
  const sim::sim_result ours = fftn_timed(c, 8, 0, members, costs);
  report("1728 x 1 on 8 lanes waking 10 us apart", proto.done_at, ours.done_at, serial);
  EXPECT_GT(proto.done_at, 5 * serial) << "not reproduced";
  EXPECT_LT(ours.done_at, 1.2 * serial);
}

// Fewer transforms than members, each shared by the whole group in turn: 5120 x 16 on 18
// lanes of one mesh-connected socket ran 5.15x slower than whole rows. Rules 2 and 3 run
// the rows whole, side by side.
TEST(Pathologies, GroupSharingEachRowInTurn) {
  const sim::cost_model costs;
  const team_case c = default_rows(5120, 16);
  const sim::sim_result proto = prototype_shared(c, 18, {}, costs);
  const sim::sim_result ours = fftn_timed(c, 18, 0, {}, costs);
  const double row = serial_ns(c, costs) / 16;
  report("5120 x 16 on 18 lanes", proto.done_at, ours.done_at, serial_ns(c, costs));
  EXPECT_GT(proto.done_at, 3 * ours.done_at) << "not reproduced";
  EXPECT_LT(ours.done_at, 1.3 * row);
}

// A short single row split across a large group: 3072 points on 18 lanes took 13 us against
// 5.7 us serial. Rule 3 (at least 512 points per member, 8000 on a mesh) keeps it whole.
TEST(Pathologies, ShortRowSplitAcrossManyLanes) {
  const sim::cost_model costs;
  const team_case c = default_rows(3072, 1);
  const double serial = serial_ns(c, costs);
  const sim::sim_result proto = prototype_shared(c, 18, {}, costs);
  const sim::sim_result ours = fftn_timed(c, 18, 0, {}, costs);
  report("3072 x 1 on 18 lanes", proto.done_at, ours.done_at, serial);
  EXPECT_GT(proto.done_at, 1.3 * serial) << "not reproduced";
  EXPECT_LT(ours.done_at, 1.1 * serial);
}

// A row split across two cache groups: one 10240-point row took 13 us inside a CCX and
// 29-87 us across CCXs. Rule 3 splits a row only among the members of one group.
TEST(Pathologies, RowSplitAcrossCacheGroups) {
  sim::cost_model costs;
  costs.physical_group = 8;
  fftn::detail::build_settings fine;
  fine.chunk_points = 640; // 16 chunks per pass: one for each lane of the prototype's team
  const team_case c = fftn::test::make_case(
      "c2c_10240x1", fftn::rows_c2c<double>(10240, 1, fftn::direction::forward), 10240, 10240,
      false, fine);
  const double serial = serial_ns(c, costs);
  const sim::sim_result proto = prototype_shared(c, 16, {}, costs);
  const sim::sim_result ours = fftn_timed(c, 16, 8, {}, costs);
  report("10240 x 1 on 16 lanes in two groups", proto.done_at, ours.done_at, serial);
  EXPECT_GT(proto.done_at, 1.4 * ours.done_at) << "not reproduced";
  EXPECT_LT(ours.done_at, 0.5 * serial);
}

// Leftover rows after the whole ones: shared by sub-teams, 3072-point rows lost 4-21 %,
// rows of 8192 points and more gained 10-29 %. Rule 2 splits leftovers only when long.
TEST(Pathologies, LeftoverRowsSplitOnlyWhenLong) {
  const sim::cost_model costs;
  for (const std::size_t n : {std::size_t{3072}, std::size_t{65536}}) {
    const team_case c = default_rows(n, 9);
    // The prototype's rule before the fix: rows 0-7 whole, row 8 shared by all eight.
    sim::static_schedule split(8);
    std::size_t barriers = 0;
    for (std::size_t k = 0; k < 8; ++k) {
      split[k].push_back({.what = sim::static_step::kind::whole, .task = k});
    }
    sim::share_in_turn(split, c.core->graph, 8, 1, 0, 8, barriers);
    // And whole rows only: one member takes two.
    sim::static_schedule whole(8);
    for (std::size_t k = 0; k < 8; ++k) {
      whole[k].push_back({.what = sim::static_step::kind::whole, .task = k});
    }
    whole[0].push_back({.what = sim::static_step::kind::whole, .task = 8});
    const sim::sim_result proto_split = sim::simulate_static(c, split, barriers, costs, {});
    const sim::sim_result proto_whole = sim::simulate_static(c, whole, 0, costs, {});
    const sim::sim_result ours = fftn_timed(c, 8, 0, {}, costs);
    report(n < 8192 ? "3072 x 9 on 8 lanes, leftover split"
                    : "65536 x 9 on 8 lanes, leftover split",
           proto_split.done_at, ours.done_at, serial_ns(c, costs));
    report(n < 8192 ? "3072 x 9 on 8 lanes, whole rows" : "65536 x 9 on 8 lanes, whole rows",
           proto_whole.done_at, ours.done_at, serial_ns(c, costs));
    if (n < 8192) {
      EXPECT_GT(proto_split.done_at, proto_whole.done_at) << "not reproduced";
      EXPECT_LE(ours.done_at, 1.02 * proto_whole.done_at);
    } else {
      EXPECT_LT(proto_split.done_at, proto_whole.done_at) << "not reproduced";
      EXPECT_LT(ours.done_at, 0.8 * proto_whole.done_at);
    }
  }
}

// One long row on 64 lanes that wake one after another: the prototype's static schedule
// waited for every lane (65536 points: 79 us on 8 lanes, 133 us on 64). Rules 4 and 5 deal
// the row to one group, and the members claim: nobody waits for a lane that is not there.
TEST(Pathologies, WakingEveryLaneForOneRow) {
  sim::cost_model costs;
  costs.physical_group = 8;
  const team_case c = default_rows(65536, 1);
  const auto members = waking(64, 800);
  const double serial = serial_ns(c, costs);
  const sim::sim_result proto = prototype_shared(c, 64, members, costs);
  const sim::sim_result ours = fftn_timed(c, 64, 8, members, costs);
  report("65536 x 1 on 64 lanes waking 0.8 us apart", proto.done_at, ours.done_at, serial);
  EXPECT_GT(proto.done_at, 1.5 * ours.done_at) << "not reproduced";
  EXPECT_LT(ours.done_at, 0.4 * serial);
}

// A member that never arrives: barriers deadlock (the simulator detects it); the claiming
// engine completes with the members that came.
TEST(Pathologies, MemberThatNeverArrives) {
  const sim::cost_model costs;
  const team_case c = default_rows(10240, 1);
  std::vector<sim::member_spec> members(4);
  members[3].arrives = false;
  const sim::sim_result proto = prototype_shared(c, 4, members, costs);
  EXPECT_TRUE(proto.deadlocked) << "not reproduced";
  const sim::sim_result ours = fftn_timed(c, 4, 0, members, costs);
  EXPECT_FALSE(ours.deadlocked);
}

} // namespace
