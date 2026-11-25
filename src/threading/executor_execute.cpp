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

#include "core/drive.hpp"
#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/executor.hpp"
#include "fftn/workspace.hpp"
#include "threading/executor_dispatch.hpp"

namespace fftn::inline v0::detail {

void execute_on(const plan_core& core, const void* in, void* out, const executor_ref& ex,
                const workspace& ws) {
  execute_on_as<plain_drive>(core, in, out, ex, ws);
}

} // namespace fftn::inline v0::detail
