// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// A member of a team execution (DESIGN.md 5.3, 13.3), shared by the team overload of
// plan::execute (team_execute.cpp), the executor overload (executor_execute.cpp) and their
// twins for plans whose edges have factors (edge_execute.cpp): the drivers differ
// (core/drive.hpp), the deal, the team state and the engine do not.
#pragma once

#include "core/drive.hpp"
#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"
#include "threading/engine.hpp"
#include "threading/schedule.hpp"
#include "threading/team_env.hpp"
#include "threading/team_state.hpp"
#include "threading/wait.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace fftn::inline v0::detail {

void execute_member(const plan_core& plan, const void* in, void* out, const team& t,
                    const workspace& ws, completion done, int threads) noexcept;

inline wait_budget budget_of(const schedule& s) noexcept {
  const auto ns = [](double v) {
    return std::chrono::nanoseconds(static_cast<std::chrono::nanoseconds::rep>(v));
  };
  // An oversubscribed team's partners may not be running: spinning would only delay them.
  return {
      .spin = s.oversubscribed ? std::chrono::nanoseconds(0) : ns(s.profile.spin_ns),
      .yield = ns(s.profile.yield_ns),
  };
}

// execute_member on a driver: plain_drive for execute_member itself (no State), edge_drive for
// plans whose edges have factors (State: the factor cursors). Inlined into each, so that
// execute_member compiles as it did before the edge driver shared it.
template <class Drive, class... State>
[[gnu::always_inline]] inline void execute_member_as(const plan_core& plan, const void* in,
                                                     void* out, const team& t, const workspace& ws,
                                                     completion done, int threads,
                                                     const State&... drive_state) noexcept {
  assert(t.size >= 1 && t.rank >= 0 && t.rank < t.size && t.group_size >= 0 &&
         "fftn: invalid team");
  if (t.size == 1) {
    Drive::serial(plan, 0, plan.prog.transforms, in, out, ws, drive_state...);
    return;
  }
  assert(ws.size() >= workspace_bytes(plan, t.size) &&
         "fftn: workspace smaller than plan::workspace_bytes(team size)");
  // Every member of a team of two or more runs the team program where there is one (X43),
  // whatever the schedule: its results are the program's, bit for bit.
  const plan_core& core = team_core(plan);
  assert(in_place_pointers(core, in, out) && "fftn: in-place plan executed with in != out");
  const team_layout layout{static_cast<std::size_t>(t.size)};
  const std::size_t page = core.profile.team_state_page;
  std::byte* state = team_state_in(ws.memory().data(), page);
  std::byte* slots = state + layout.span(page);
  if (done == completion::own) {
    // A static deal: nothing is shared. The members' slots lie beyond the state of a team of
    // this size, but inside that of a larger one, which this execution may overwrite: the
    // workspace then no longer holds valid state for any other size.
    const sync_word flag = workspace_access::team_word(ws);
    const std::uint64_t seen = flag.load(std::memory_order_relaxed);
    if (seen != 0 && seen != team_flag(layout.lanes, team_state_unit(page), team_ready)) {
      flag.store(0, std::memory_order_relaxed);
    }
    const transform_range mine = owned_tasks(core.prog.transforms, t.size, t.rank);
    if (mine.count != 0) {
      basic_team_env<Drive>(core, in, out, slots, {}, drive_state...)
          .run_whole(mine.first, mine.count, t.rank);
    }
    return;
  }
  const lane_map lanes{.lanes = t.size, .group_size = t.group_size, .threads = threads};
  const team_job job{
      .graph = &core.graph,
      .plan = make_schedule(core.graph, lanes, core.profile),
      .lanes = lanes,
      .state = state,
      .flag = workspace_access::team_word(ws),
      .rank = t.rank,
      .unit = team_state_unit(page),
  };
  basic_team_env<Drive> env(core, in, out, slots, budget_of(job.plan), drive_state...);
  team_engine<basic_team_env<Drive>>(job, env).run();
}

} // namespace fftn::inline v0::detail
