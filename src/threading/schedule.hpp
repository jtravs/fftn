// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The scheduler (DESIGN.md 13.4): a pure function from (work graph, lane map, machine
// profile) to a schedule, plain data that the claiming engine (engine.hpp) follows. It
// decides where whole transforms go and which transforms are split, never what a chunk
// computes, so no schedule can change a result.
//
// A schedule is computed by every member at the start of every team execution: it is a few
// dozen arithmetic operations, allocates nothing, and every member gets the same one.
#pragma once

#include "fftn/team.hpp"
#include "threading/work_graph.hpp"

#include <cstddef>
#include <cstdint>

namespace fftn::inline v0::detail {

// The constants the rules use, per machine. The values here are provisional: they come from
// the prototype's measurements (cited per field) and will be replaced by calibration
// (experiments T1 and T5, task 3.4). Nothing in the rules is a literal.
struct machine_profile {
  // The cost model: nanoseconds per point through a pass (work_graph.hpp). Provisional:
  // the walking skeleton's scalar passes; the phase 1 kernels will be several times faster.
  double ns_per_work = 0.5;
  // Rule 1: run inline when the whole job takes less than inline_multiple x dispatch_ns.
  // A team call's fixed cost is 0.5-10 us of runtime dispatch and completion waits
  // (fork/join measured at 0.5-9.8 us on the prototype's machines).
  double dispatch_ns = 2000;
  double inline_multiple = 3;
  // Rules 2 and 3: the cost of one phase boundary inside a cache group (a done counter and
  // a wake-up): 0.4 us on an Apple cluster, 0.7 us on a Zen 3 CCX, 1.1 us on an 18-core
  // Xeon mesh, measured for the prototype's barrier.
  double sync_ns = 700;
  // Rule 3: at least this many points per member of a split transform (3072 points on 18
  // lanes ran in 13 us against 5.7 us serial).
  std::size_t min_points_per_member = 512;
  // Rule 3: split a transform only from this size: about 2048 points inside a Zen 3 CCX,
  // about 8000 on a large mesh-connected group (more than mesh_group_lanes members).
  std::size_t split_min_points = 2048;
  std::size_t split_min_points_mesh = 8192;
  int mesh_group_lanes = 8;
  // Rule 2: leftover transforms after the whole ones are split only from this size
  // (remainder sub-teams gained 0.71-0.90 for 8192 points and more, lost 1.04-1.21 at 3072).
  std::size_t remainder_split_min_points = 8192;
  // The logical CPUs the process may use (0: unknown). A team with more members is
  // oversubscribed: no transform is split, since a preempted member would stall its
  // partners for a whole time slice, and waiting members do not spin.
  int cpus = 0;
  // Waiting (wait.hpp): spin this long, then yield this long, then block. Provisional:
  // the study's 5-100 us spin range, near its middle.
  double spin_ns = 20000;
  double yield_ns = 50000;
  // Whole transforms are claimed in runs of (left / (claim_divisor x members)), at least
  // one: long runs while much is left (contiguous rows, few atomic operations), single
  // transforms at the end (balance).
  int claim_divisor = 2;
};

// The profile of the running machine: the defaults above with its CPU count.
[[nodiscard]] machine_profile default_machine_profile() noexcept;

// Which lanes (team members) share a cache: contiguous runs of group_size ranks (fftn::team).
struct lane_map {
  int lanes = 1;
  int group_size = 0; // 0, or lanes or more: one group
  // The threads of the runtime the team runs on, when it has more than the team (0: as
  // many as lanes). An executor dispatches only the lanes dealt work, but all of its
  // threads compete for the CPUs (a pool's idle threads linger, an OpenMP region keeps its
  // width): oversubscription is judged by them.
  int threads = 0;

  [[nodiscard]] int per_group() const noexcept {
    return group_size <= 0 || group_size > lanes ? lanes : group_size;
  }
  [[nodiscard]] int groups() const noexcept { return (lanes + per_group() - 1) / per_group(); }
  [[nodiscard]] int group_of(int lane) const noexcept { return lane / per_group(); }
  [[nodiscard]] int base(int group) const noexcept { return group * per_group(); }
  [[nodiscard]] int size_of(int group) const noexcept {
    const int rest = lanes - base(group);
    return rest < per_group() ? rest : per_group();
  }
};

struct schedule {
  std::size_t tasks = 0;
  // Rule 1: the first member to arrive takes every task in one claim.
  bool inline_all = false;
  bool oversubscribed = false;
  // Rules 4 and 5: tasks are dealt to groups [0, groups), in proportion to their members:
  // as many groups as the batch has a group's worth of tasks for, so a batch smaller than
  // a group stays in one (full-width teams lost at 64 lanes: 1.19 against 0.83 at 8).
  int groups = 1;
  int lanes_dealt = 1; // members in those groups
  // The members an executor dispatches (rule 4): lanes_dealt, or, when no group splits a
  // task, only as many as there are tasks (whole tasks give a member each at most). A team
  // of that many members gets the same split decisions, none: each group's share then
  // equals its members.
  int dispatch = 1;
  // Whole tasks are counted, claimed and finished in units of `grain` consecutive tasks, so
  // that a batch of any size fits the 24-bit counters (team_state.hpp); 1 unless the batch
  // has millions of transforms.
  std::size_t grain = 1;
  // What share_of needs to decide splits per group.
  bool splittable = false;
  double task_ns = 0;
  std::size_t phases = 0;
  std::uint32_t widest = 1;
  std::size_t points = 1;
  machine_profile profile;
};

// One group's tasks: [first, first + count) in batch order; the first `whole` run whole
// (claimed in runs), the other `split` are split among at most `cap` of its members each.
struct group_share {
  std::size_t first = 0;
  std::size_t count = 0;
  std::size_t whole = 0;
  std::size_t split = 0;
  int cap = 0;
};

// The counters of a team execution have 24 bits (team_state.hpp). A split task's chunks and
// a group's whole units stay below this; teams of more lanes run inline.
inline constexpr std::uint64_t max_counted = std::uint64_t{1} << 22U;

[[nodiscard]] schedule make_schedule(const work_graph& g, const lane_map& lanes,
                                     const machine_profile& profile) noexcept;

[[nodiscard]] group_share share_of(const schedule& s, const lane_map& lanes, int group) noexcept;

// The units a group's whole tasks count as.
[[nodiscard]] std::size_t whole_units(const schedule& s, const group_share& share) noexcept;

// What the execution's done counter reaches when every task is done: 1 inline, else every
// group's whole units and split tasks.
[[nodiscard]] std::uint32_t done_target(const schedule& s, const lane_map& lanes) noexcept;

// completion::own: the static deal of `tasks` over `lanes` (fftn::plan::transforms_of).
// With at least as many tasks as lanes, contiguous ranges differing by at most one; with
// fewer, the first `tasks` lanes one each, so that a small batch stays on few members.
[[nodiscard]] transform_range owned_tasks(std::size_t tasks, int lanes, int lane) noexcept;

} // namespace fftn::inline v0::detail
