// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The executor overload of plan::execute on a driver (core/drive.hpp), shared by
// executor_execute.cpp (plain_drive) and its twin for plans whose edges have factors
// (edge_execute.cpp, edge_drive). See executor_execute.cpp for how the team is chosen.
#pragma once

#include "core/drive.hpp"
#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/executor.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"
#include "threading/schedule.hpp"
#include "threading/team_state.hpp"

#include <cassert>

namespace fftn::inline v0::detail {

// What each dispatched member needs; State: the driver's state, none for plain_drive, so that
// the plain instantiation is the code before edges existed (GCC allocated registers
// differently even for an empty state).
template <class... State> struct dispatched_team {
  const plan_core* core = nullptr;
  const void* in = nullptr;
  void* out = nullptr;
  const workspace* ws = nullptr;
  int size = 0;
  int group_size = 0;
  int threads = 0; // the executor's width
};
template <class State> struct dispatched_team<State> {
  const plan_core* core = nullptr;
  const void* in = nullptr;
  void* out = nullptr;
  const workspace* ws = nullptr;
  int size = 0;
  int group_size = 0;
  int threads = 0;
  State state{};
};

// NOLINTNEXTLINE(misc-const-correctness): the signature is fftn_task's
template <class Drive, class... State> void run_dispatched_member(void* arg, int rank) {
  const auto& t = *static_cast<const dispatched_team<State...>*>(arg);
  if constexpr (sizeof...(State) == 0) {
    Drive::member(*t.core, t.in, t.out, {.rank = rank, .size = t.size, .group_size = t.group_size},
                  *t.ws, completion::all, t.threads);
  } else {
    Drive::member(*t.core, t.in, t.out, {.rank = rank, .size = t.size, .group_size = t.group_size},
                  *t.ws, completion::all, t.threads, t.state);
  }
}

template <class Drive, class... State>
[[gnu::always_inline]] inline void execute_on_as(const plan_core& core, const void* in, void* out,
                                                 const executor_ref& ex, const workspace& ws,
                                                 const State&... state) {
  // Rule 1, decided at plan construction: a small job never asks the executor anything.
  const int width = core.runs_inline ? 1 : ex.concurrency();
  if (width > 1) {
    const int group_size = ex.group_size();
    const lane_map lanes{.lanes = width, .group_size = group_size < 0 ? 0 : group_size};
    // The team that runs it runs the team program where there is one (execute_member).
    const plan_core& runner = team_core(core);
    const schedule plan = make_schedule(runner.graph, lanes, runner.profile);
    if (!plan.inline_all && plan.dispatch > 1) {
      assert(ws.size() >= workspace_bytes(core, width) &&
             "fftn: workspace smaller than plan::workspace_bytes(executor concurrency)");
      dispatched_team<State...> team{
          .core = &core,
          .in = in,
          .out = out,
          .ws = &ws,
          .size = plan.dispatch,
          .group_size = lanes.group_size,
          .threads = width,
      };
      ((team.state = state), ...);
      try {
        ex.bulk(team.size, &run_dispatched_member<Drive, State...>, &team);
      } catch (...) {
        // Some members may not have run: they would count their next execution wrong. The
        // executor's bulk has returned, so no member is inside FFTN and the state can be
        // dropped; the next team execution sets it up afresh.
        workspace_access::team_word(ws).store(0);
        throw;
      }
      return;
    }
  }
  Drive::serial(core, 0, core.prog.transforms, in, out, ws, state...);
}

} // namespace fftn::inline v0::detail
