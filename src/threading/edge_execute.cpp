// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The team and executor overloads of plan::execute with exec_args (fftn/edges.hpp): the code of
// team_execute.hpp and executor_dispatch.hpp instantiated on the edge driver (core/drive.hpp,
// edge_drive) for plans whose edges have factors. In a unit of its own, so that the plain
// instantiations in team_execute.cpp and executor_execute.cpp compile as before edges existed.
// A plan without factor edges goes to those at once.

#include "core/drive.hpp"
#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/edges.hpp"
#include "fftn/executor.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"
#include "threading/executor_dispatch.hpp"
#include "threading/team_execute.hpp"

namespace fftn::inline v0::detail {

void edge_drive::member(const plan_core& c, const void* in, void* out, const team& t,
                        const workspace& ws, completion done, int threads,
                        const state& f) noexcept {
  execute_member_as<edge_drive>(c, in, out, t, ws, done, threads, f);
}

void execute_team(const plan_core& core, const void* in, void* out, const team& t,
                  const workspace& ws, completion done, const exec_args& args) noexcept {
  if (!core.needs_args) {
    execute_team(core, in, out, t, ws, done);
    return;
  }
  // A missing array ends the program here (cursors_of).
  execute_member_as<edge_drive>(core, in, out, t, ws, done, 0, cursors_of(args, core.factor_args));
}

void execute_on(const plan_core& core, const void* in, void* out, const executor_ref& ex,
                const workspace& ws, const exec_args& args) {
  if (!core.needs_args) {
    execute_on(core, in, out, ex, ws);
    return;
  }
  // On the caller, before any member runs.
  execute_on_as<edge_drive>(core, in, out, ex, ws, cursors_of(args, core.factor_args));
}

} // namespace fftn::inline v0::detail
