// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The executor overload of plan::execute (DESIGN.md 5.3, 13.6): FFTN dispatches the members
// of a team itself, so it decides the team's size, and every member it dispatches runs.
//
// The scheduler is asked once with the executor's whole concurrency. If the job is too small
// to share (rule 1) or one lane would have everything, it runs serially on the caller and the
// executor is never called. Otherwise only the lanes that have work are dispatched (rule 4):
// a team of schedule::dispatch members, the leading groups the scheduler deals tasks to, and
// no more members than tasks when nothing is split. A team of that size gets the same groups
// and, like the wider team, no splits, with one member per task (the per-group shares can
// differ: 9 tasks in groups of 8 become 8 and 1 rather than the wide deal's 4 and 5), so
// nothing is lost by not dispatching the rest. The one thing the dealt lanes do
// not show is how many threads compete for the CPUs: all of the executor's (a pool's idle threads
// linger, an OpenMP region keeps its width), so the members judge oversubscription by its width. An
// OpenMP executor still opens its region at
// its constant width, its surplus threads returning at once (fftn/omp.hpp).
//
// Since every dispatched member runs exactly once, no member skips an execution (the
// sequence precondition of team execution) whatever the executor does with the order and
// concurrency of its tasks, and bulk's return orders each rank's calls.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/executor.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"
#include "threading/schedule.hpp"
#include "threading/team_execute.hpp"
#include "threading/team_state.hpp"

#include <cassert>

namespace fftn::inline v0::detail {
namespace {

struct dispatched_team {
  const plan_core* core = nullptr;
  const void* in = nullptr;
  void* out = nullptr;
  const workspace* ws = nullptr;
  int size = 0;
  int group_size = 0;
  int threads = 0; // the executor's width
};

// NOLINTNEXTLINE(misc-const-correctness): the signature is fftn_task's
void run_member(void* arg, int rank) {
  const auto& t = *static_cast<const dispatched_team*>(arg);
  execute_member(*t.core, t.in, t.out, {.rank = rank, .size = t.size, .group_size = t.group_size},
                 *t.ws, completion::all, t.threads);
}

} // namespace

void execute_on(const plan_core& core, const void* in, void* out, const executor_ref& ex,
                const workspace& ws) {
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
      dispatched_team team{
          .core = &core,
          .in = in,
          .out = out,
          .ws = &ws,
          .size = plan.dispatch,
          .group_size = lanes.group_size,
          .threads = width,
      };
      try {
        ex.bulk(team.size, &run_member, &team);
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
  execute_core(core, 0, core.prog.transforms, in, out, ws);
}

} // namespace fftn::inline v0::detail
