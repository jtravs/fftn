// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The team overload of plan::execute (DESIGN.md 5.3, 13.3): the workspace layout of a team
// (team_env.hpp), the static deal of completion::own, and the claiming engine for
// completion::all.

#include "threading/team_execute.hpp"
#include "core/driver.hpp"
#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"
#include "threading/engine.hpp"
#include "threading/schedule.hpp"
#include "threading/team_env.hpp"
#include "threading/team_state.hpp"
#include "threading/wait.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <variant>

namespace fftn::inline v0::detail {
// The environment of plans without factor edges, instantiated here as its members were
// defined here before the edge driver shared them (team_env.hpp).
template class basic_team_env<plain_drive>;

std::size_t workspace_bytes(const plan_core& core, int lanes) noexcept {
  if (lanes <= 1) {
    return core.bytes;
  }
  const auto l = static_cast<std::size_t>(lanes);
  // The slack aligns the state to its 128-byte lines at any workspace address or, where the
  // profile gives it pages of its own, puts it on a page boundary, the state then filling whole
  // pages (team_state.hpp, machine_profile::team_state_page; X76, X79). A size that does not
  // fit saturates, so that allocating it fails rather than wrapping to a small one.
  // A team runs the team program where there is one (execute_member), whose slots may be
  // smaller than the program's scratch; but an executor that hands back one lane and a
  // schedule that dispatches only one member run the program serially on the same workspace
  // (executor_execute.cpp), so the bytes are never fewer than a serial execution's.
  const std::size_t page = team_core(core).profile.team_state_page;
  const std::size_t slack = page != 0 ? page - 1 : sync_line;
  std::size_t slots = 0;
  std::size_t total = 0;
  if (__builtin_mul_overflow(l, team_core(core).slot_bytes, &slots) ||
      __builtin_add_overflow(slots, slack + team_layout{l}.span(page), &total)) {
    return std::numeric_limits<std::size_t>::max();
  }
  return std::max(total, core.bytes);
}

transform_range transforms_of(const plan_core& core, const team& t) noexcept {
  return owned_tasks(core.prog.transforms, t.size, t.rank);
}

void execute_team(const plan_core& core, const void* in, void* out, const team& t,
                  const workspace& ws, completion done) noexcept {
  execute_member(core, in, out, t, ws, done, 0);
}

void execute_member(const plan_core& plan, const void* in, void* out, const team& t,
                    const workspace& ws, completion done, int threads) noexcept {
  execute_member_as<plain_drive>(plan, in, out, t, ws, done, threads);
}

} // namespace fftn::inline v0::detail
