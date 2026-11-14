// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The team overload of plan::execute (DESIGN.md 5.3, 13.3): the workspace layout of a team
// (team_env.hpp), the static deal of completion::own, and the claiming engine for
// completion::all.

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

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <variant>

namespace fftn::inline v0::detail {
namespace {

std::byte* align_up(std::byte* p, std::size_t to) noexcept {
  const auto address = reinterpret_cast<std::uintptr_t>(p);
  return p + (to - address % to) % to;
}

wait_budget budget_of(const schedule& s) noexcept {
  const auto ns = [](double v) {
    return std::chrono::nanoseconds(static_cast<std::chrono::nanoseconds::rep>(v));
  };
  // An oversubscribed team's partners may not be running: spinning would only delay them.
  return {
      .spin = s.oversubscribed ? std::chrono::nanoseconds(0) : ns(s.profile.spin_ns),
      .yield = ns(s.profile.yield_ns),
  };
}

} // namespace

team_env::team_env(const plan_core& core, const void* in, void* out, std::byte* slots,
                   const wait_budget& budget) noexcept
    : core_(core),
      io_{.in = static_cast<const std::byte*>(in), .out = static_cast<std::byte*>(out)},
      slots_(slots), budget_(budget) {}

slot_addresses team_env::scratch(int lane) const noexcept {
  slot_addresses at = io_;
  if (core_.bytes != 0) {
    std::byte* slot = slots_ + static_cast<std::size_t>(lane) * core_.slot_bytes;
    at.a = slot;
    at.b = slot + core_.offset_b;
    at.block = core_.block_elements != 0 ? slot + core_.offset_block : nullptr;
  }
  return at;
}

void team_env::run_whole(std::size_t first, std::size_t count, int lane) const noexcept {
  run_transforms(core_.prog, first, count, scratch(lane));
}

void team_env::run_chunk(std::size_t task, std::size_t phase, std::uint32_t chunk, int owner,
                         int lane) const noexcept {
  const graph_phase& ph = core_.graph.phases[phase];
  slot_addresses at = transform_addresses(core_.prog, task, scratch(owner));
  if (ph.nest == 0) {
    run_stage_chunks(*std::get_if<stage>(&core_.prog.records[ph.record]), at, chunk, chunk + 1);
    return;
  }
  // A line runs all its stages on the scratch lines of the member running it.
  const slot_addresses mine = scratch(lane);
  at.a = mine.a;
  at.b = mine.b;
  run_lines(core_.prog, ph.record, ph.nest, ph.body_end, chunk_start(ph.units, ph.chunks, chunk),
            chunk_start(ph.units, ph.chunks, std::size_t{chunk} + 1), at);
}

std::size_t workspace_bytes(const plan_core& core, int lanes) noexcept {
  if (lanes <= 1) {
    return core.bytes;
  }
  const auto l = static_cast<std::size_t>(lanes);
  // The slack aligns the state to its 128-byte lines at any workspace address. A size that
  // does not fit saturates, so that allocating it fails rather than wrapping to a small one.
  std::size_t slots = 0;
  std::size_t total = 0;
  if (__builtin_mul_overflow(l, core.slot_bytes, &slots) ||
      __builtin_add_overflow(slots, sync_line + team_layout{l}.bytes(), &total)) {
    return std::numeric_limits<std::size_t>::max();
  }
  return total;
}

transform_range transforms_of(const plan_core& core, const team& t) noexcept {
  return owned_tasks(core.prog.transforms, t.size, t.rank);
}

void execute_team(const plan_core& core, const void* in, void* out, const team& t,
                  const workspace& ws, completion done) noexcept {
  assert(t.size >= 1 && t.rank >= 0 && t.rank < t.size && t.group_size >= 0 &&
         "fftn: invalid team");
  if (t.size == 1) {
    execute_core(core, 0, core.prog.transforms, in, out, ws);
    return;
  }
  assert(ws.size() >= workspace_bytes(core, t.size) &&
         "fftn: workspace smaller than plan::workspace_bytes(team size)");
  assert((!core.in_place || in == out) && "fftn: in-place plan executed with in != out");
  const team_layout layout{static_cast<std::size_t>(t.size)};
  std::byte* state = align_up(ws.memory().data(), sync_line);
  std::byte* slots = state + layout.bytes();
  if (done == completion::own) {
    // A static deal: nothing is shared. The members' slots lie beyond the state of a team of
    // this size, but inside that of a larger one, which this execution may overwrite: the
    // workspace then no longer holds valid state for any other size.
    const sync_word flag = workspace_access::team_word(ws);
    const std::uint64_t seen = flag.load(std::memory_order_relaxed);
    if (seen != 0 && seen != layout.lanes * 4 + team_ready) {
      flag.store(0, std::memory_order_relaxed);
    }
    const transform_range mine = owned_tasks(core.prog.transforms, t.size, t.rank);
    if (mine.count != 0) {
      team_env(core, in, out, slots, {}).run_whole(mine.first, mine.count, t.rank);
    }
    return;
  }
  const lane_map lanes{.lanes = t.size, .group_size = t.group_size};
  const team_job job{
      .graph = &core.graph,
      .plan = make_schedule(core.graph, lanes, core.profile),
      .lanes = lanes,
      .state = state,
      .flag = workspace_access::team_word(ws),
      .rank = t.rank,
  };
  team_env env(core, in, out, slots, budget_of(job.plan));
  team_engine<team_env>(job, env).run();
}

} // namespace fftn::inline v0::detail
