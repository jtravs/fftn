// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// One member's part in a team execution (team_execute.cpp), for callers inside the library
// that know more about the team than fftn::team says: the executor overload
// (executor_execute.cpp) runs its dispatched members on an executor of more threads.
#pragma once

#include "core/plan_core.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"

namespace fftn::inline v0::detail {

// The team overload of plan::execute, with the threads of the runtime the team runs on
// (lane_map::threads; 0: as many as the team has members).
void execute_member(const plan_core& plan, const void* in, void* out, const team& t,
                    const workspace& ws, completion done, int threads) noexcept;

} // namespace fftn::inline v0::detail
