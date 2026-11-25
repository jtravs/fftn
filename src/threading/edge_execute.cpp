// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The team and executor overloads of plan::execute with exec_args (fftn/edges.hpp): those of
// team_execute.cpp and executor_execute.cpp, for plans whose edges have factors, on the edge
// driver (core/edge_driver.hpp). Kept apart from them, in a unit of their own, so that every
// other plan's team and executor executions are compiled exactly as before edges existed (one
// driver for both cost plans without edges 2-6 % on the EPYC 7763). A plan without factor
// edges goes to those overloads at once.

#include "core/edge_driver.hpp"
#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/edges.hpp"
#include "fftn/executor.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"
#include "threading/engine.hpp"
#include "threading/schedule.hpp"
#include "threading/team_state.hpp"
#include "threading/wait.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace fftn::inline v0::detail {
namespace {

wait_budget budget_of(const schedule& s) noexcept {
  const auto ns = [](double v) {
    return std::chrono::nanoseconds(static_cast<std::chrono::nanoseconds::rep>(v));
  };
  return {
      .spin = s.oversubscribed ? std::chrono::nanoseconds(0) : ns(s.profile.spin_ns),
      .yield = ns(s.profile.yield_ns),
  };
}

// team_env (team_env.hpp) with the factor cursors, on the edge driver.
class edge_team_env {
public:
  edge_team_env(const plan_core& core, const void* in, void* out, std::byte* slots,
                const wait_budget& budget, factor_cursors cursors) noexcept
      : core_(core),
        io_{.in = static_cast<const std::byte*>(in), .out = static_cast<std::byte*>(out)},
        slots_(slots), budget_(budget), cursors_(cursors) {}

  void point() const noexcept {}

  template <class Ready>
  void wait(sync_word w, const sync_word* blocked, Ready ready) const noexcept {
    wait_for(w, blocked, ready, budget_);
  }
  static void wake(sync_word w, const sync_word* blocked) noexcept { detail::wake(w, blocked); }

  void run_whole(std::size_t first, std::size_t count, int lane) const noexcept {
    run_edge_transforms(core_.prog, core_.edge_records, first, count, scratch(lane), cursors_);
  }

  void run_chunk(std::size_t task, std::size_t phase, std::uint32_t chunk, int owner,
                 int lane) const noexcept {
    const graph_phase& ph = core_.graph.phases[phase];
    slot_addresses at = transform_addresses(core_.prog, task, scratch(owner));
    const factor_cursors f = edge_transform_cursors(core_.prog, core_.edge_records, task, cursors_);
    if (ph.nest == 0) {
      run_edge_stage_chunks(core_.prog, core_.edge_records, ph.record, at, f, chunk, chunk + 1);
      return;
    }
    const slot_addresses mine = scratch(lane);
    at.a = mine.a;
    at.b = mine.b;
    const std::size_t lo = chunk_start(ph.units, ph.chunks, chunk);
    const std::size_t hi = chunk_start(ph.units, ph.chunks, std::size_t{chunk} + 1);
    if (ph.tail_nest == 0) {
      run_edge_lines(core_.prog, core_.edge_records, ph.record, ph.nest, ph.body_end, lo, hi, at,
                     f);
      return;
    }
    if (lo < ph.head) {
      run_edge_lines(core_.prog, core_.edge_records, ph.record, ph.nest, ph.body_end, lo,
                     std::min(hi, ph.head), at, f);
    }
    if (hi > ph.head) {
      run_edge_lines(core_.prog, core_.edge_records, ph.tail_record, ph.tail_nest, ph.tail_body_end,
                     std::max(lo, ph.head) - ph.head, hi - ph.head, at, f);
    }
  }

private:
  [[nodiscard]] slot_addresses scratch(int lane) const noexcept {
    slot_addresses at = io_;
    if (core_.bytes != 0) {
      std::byte* slot = slots_ + static_cast<std::size_t>(lane) * core_.slot_bytes;
      at.a = slot;
      at.b = slot + core_.offset_b;
      at.block = core_.block_elements != 0 ? slot + core_.offset_block : nullptr;
    }
    return at;
  }

  const plan_core& core_;
  slot_addresses io_;
  std::byte* slots_;
  wait_budget budget_;
  factor_cursors cursors_;
};

// execute_member (team_execute.cpp) for a plan whose edges have factors.
void execute_member_with(const plan_core& plan, const void* in, void* out, const team& t,
                         const workspace& ws, completion done, int threads,
                         const exec_args& args) noexcept {
  assert(t.size >= 1 && t.rank >= 0 && t.rank < t.size && t.group_size >= 0 &&
         "fftn: invalid team");
  if (t.size == 1) {
    execute_core(plan, 0, plan.prog.transforms, in, out, ws, args);
    return;
  }
  assert(ws.size() >= workspace_bytes(plan, t.size) &&
         "fftn: workspace smaller than plan::workspace_bytes(team size)");
  // Plans with edges are rows (task 5.1a), which have no team core; team_core keeps it general.
  const plan_core& core = team_core(plan);
  assert((!core.in_place || in == out) && "fftn: in-place plan executed with in != out");
  const factor_cursors cursors = cursors_of(args, core.factor_args);
  const team_layout layout{static_cast<std::size_t>(t.size)};
  const std::size_t page = core.profile.team_state_page;
  std::byte* state = team_state_in(ws.memory().data(), page);
  std::byte* slots = state + layout.span(page);
  if (done == completion::own) {
    const sync_word flag = workspace_access::team_word(ws);
    const std::uint64_t seen = flag.load(std::memory_order_relaxed);
    if (seen != 0 && seen != team_flag(layout.lanes, team_state_unit(page), team_ready)) {
      flag.store(0, std::memory_order_relaxed);
    }
    const transform_range mine = owned_tasks(core.prog.transforms, t.size, t.rank);
    if (mine.count != 0) {
      edge_team_env(core, in, out, slots, {}, cursors).run_whole(mine.first, mine.count, t.rank);
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
  edge_team_env env(core, in, out, slots, budget_of(job.plan), cursors);
  team_engine<edge_team_env>(job, env).run();
}

struct dispatched_team {
  const plan_core* core = nullptr;
  const void* in = nullptr;
  void* out = nullptr;
  const workspace* ws = nullptr;
  const exec_args* args = nullptr;
  int size = 0;
  int group_size = 0;
  int threads = 0; // the executor's width
};

// NOLINTNEXTLINE(misc-const-correctness): the signature is fftn_task's
void run_member(void* arg, int rank) {
  const auto& t = *static_cast<const dispatched_team*>(arg);
  execute_member_with(*t.core, t.in, t.out,
                      {.rank = rank, .size = t.size, .group_size = t.group_size}, *t.ws,
                      completion::all, t.threads, *t.args);
}

} // namespace

void execute_team(const plan_core& core, const void* in, void* out, const team& t,
                  const workspace& ws, completion done, const exec_args& args) noexcept {
  if (!core.needs_args) {
    execute_team(core, in, out, t, ws, done);
    return;
  }
  execute_member_with(core, in, out, t, ws, done, 0, args);
}

// execute_on (executor_execute.cpp) for a plan whose edges have factors.
void execute_on(const plan_core& core, const void* in, void* out, const executor_ref& ex,
                const workspace& ws, const exec_args& args) {
  if (!core.needs_args) {
    execute_on(core, in, out, ex, ws);
    return;
  }
  const int width = core.runs_inline ? 1 : ex.concurrency();
  if (width > 1) {
    const int group_size = ex.group_size();
    const lane_map lanes{.lanes = width, .group_size = group_size < 0 ? 0 : group_size};
    const plan_core& runner = team_core(core);
    const schedule plan = make_schedule(runner.graph, lanes, runner.profile);
    if (!plan.inline_all && plan.dispatch > 1) {
      assert(ws.size() >= workspace_bytes(core, width) &&
             "fftn: workspace smaller than plan::workspace_bytes(executor concurrency)");
      // A missing array ends the program here, on the caller, rather than on a member.
      (void)cursors_of(args, core.factor_args);
      dispatched_team team{
          .core = &core,
          .in = in,
          .out = out,
          .ws = &ws,
          .args = &args,
          .size = plan.dispatch,
          .group_size = lanes.group_size,
          .threads = width,
      };
      try {
        ex.bulk(team.size, &run_member, &team);
      } catch (...) {
        workspace_access::team_word(ws).store(0);
        throw;
      }
      return;
    }
  }
  execute_core(core, 0, core.prog.transforms, in, out, ws, args);
}

} // namespace fftn::inline v0::detail
