// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The claiming engine's environment under the simulator (simulator.hpp): every switch point
// and wait goes through the controller, and the work runs for real, through the library's
// own environment, so that outputs can be compared bit for bit, while a recorder checks
// what ran: every task once, whole or as all its chunks, each chunk once, phases in order.
//
// Also a model of the prototype's team execution (a fixed per-member schedule with spin
// barriers between the passes of a shared row), run by the same simulator so that its
// pathological cases can be reproduced and set beside FFTN's handling of them.

#pragma once

#include "threading/simulator.hpp"
#include "threading/team_cases.hpp"

#include "core/plan_core.hpp"
#include "threading/engine.hpp"
#include "threading/schedule.hpp"
#include "threading/team_env.hpp"
#include "threading/team_state.hpp"
#include "threading/work_graph.hpp"

#include <fftn/fftn.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace fftn::test::sim {

class recorder {
public:
  recorder(const fftn::detail::work_graph& g, std::size_t tasks)
      : graph_(&g), whole_(tasks, 0), chunks_(tasks * g.chunks, 0), left_(tasks, g.chunks),
        tasks_(tasks) {}

  void whole(std::size_t first, std::size_t count, double at) {
    for (std::size_t t = first; t < first + count; ++t) {
      if (whole_[t]++ != 0 || left_[t] != graph_->chunks) {
        problems_.push_back("task " + std::to_string(t) + " ran again whole");
      }
      completed(at);
    }
  }

  void chunk(std::size_t task, std::size_t phase, std::uint32_t c, double at, int member) {
    if (static_cast<std::size_t>(member) >= by_member_.size()) {
      by_member_.resize(static_cast<std::size_t>(member) + 1, 0);
    }
    ++by_member_[static_cast<std::size_t>(member)];
    const auto& ph = graph_->phases[phase];
    const std::size_t index = task * graph_->chunks + ph.first + c;
    if (whole_[task] != 0 || chunks_[index]++ != 0) {
      problems_.push_back("chunk " + std::to_string(ph.first + c) + " of task " +
                          std::to_string(task) + " ran twice");
    }
    // Every chunk of the earlier phases must be done.
    for (std::uint64_t k = 0; k < ph.first; ++k) {
      if (chunks_[task * graph_->chunks + k] == 0) {
        problems_.push_back("chunk " + std::to_string(ph.first + c) + " of task " +
                            std::to_string(task) + " ran before chunk " + std::to_string(k));
        break;
      }
    }
    if (--left_[task] == 0) {
      completed(at);
    }
  }

  // completion::all: a member may return only once everything is done.
  void member_returned(int m) {
    if (done_ != tasks_) {
      problems_.push_back("member " + std::to_string(m) + " returned before the end");
    }
  }

  [[nodiscard]] bool complete() const { return done_ == tasks_; }
  // How many chunks each member ran.
  [[nodiscard]] const std::vector<std::size_t>& chunks_by_member() const { return by_member_; }
  [[nodiscard]] double done_at() const { return done_at_; }
  [[nodiscard]] const std::vector<std::string>& problems() const { return problems_; }

private:
  void completed(double at) {
    ++done_;
    done_at_ = std::max(done_at_, at);
  }

  const fftn::detail::work_graph* graph_;
  std::vector<int> whole_;
  std::vector<int> chunks_;
  std::vector<std::uint64_t> left_;
  std::size_t tasks_;
  std::size_t done_ = 0;
  double done_at_ = -1;
  std::vector<std::string> problems_;
  std::vector<std::size_t> by_member_;
};

// The workspace arranged as execute_team arranges it.
struct team_space {
  team_space(const fftn::detail::plan_core& core, int lanes)
      : memory(fftn::detail::workspace_bytes(core, lanes)) {
    const auto address = reinterpret_cast<std::uintptr_t>(memory.memory().data());
    const std::size_t line = fftn::detail::sync_line;
    state = memory.memory().data() + (line - address % line) % line;
    slots = state + fftn::detail::team_layout{static_cast<std::size_t>(lanes)}.bytes();
  }
  fftn::workspace memory;
  std::byte* state = nullptr;
  std::byte* slots = nullptr;
};

class sim_env {
public:
  sim_env(controller& ctl, int member, recorder& rec, const fftn::detail::team_env& real,
          const fftn::detail::work_graph& g)
      : ctl_(&ctl), member_(member), rec_(&rec), real_(&real), graph_(&g) {}

  // The engine calls its environment from noexcept functions: a failure in the simulator
  // ends the test process with a message.
  void point() noexcept {
    guarded([&] { ctl_->point(member_); });
  }

  template <class Ready>
  void wait(fftn::detail::sync_word w, const fftn::detail::sync_word* /*blocked*/,
            Ready ready) noexcept {
    guarded([&] {
      if (!try_wait(w, ready)) {
        std::fputs("simulator: the claiming engine deadlocked\n", stderr);
        std::abort();
      }
    });
  }

  // For the prototype model: false when the execution deadlocked.
  template <class Ready> bool try_wait(fftn::detail::sync_word w, Ready ready) {
    bool blocked = false;
    for (;;) {
      if (ready(w.load())) {
        if (blocked) {
          ctl_->unblocked(member_);
        }
        return true;
      }
      blocked = true;
      if (!ctl_->block(member_)) {
        return false;
      }
    }
  }

  static void wake(fftn::detail::sync_word /*w*/,
                   const fftn::detail::sync_word* /*blocked*/) noexcept {}

  void run_whole(std::size_t first, std::size_t count, int lane) noexcept {
    guarded([&] {
      real_->run_whole(first, count, lane);
      ctl_->advance(member_, static_cast<double>(count) * graph_->work * ctl_->costs().ns_per_work);
      rec_->whole(first, count, ctl_->clock(member_));
    });
  }

  void run_chunk(std::size_t task, std::size_t phase, std::uint32_t chunk, int owner,
                 int lane) noexcept {
    guarded([&] { chunk_now(task, phase, chunk, owner, lane); });
  }

  // The prototype's barrier: arrive, wait for `members`, pay its cost.
  bool barrier(std::atomic<std::uint64_t>& counter, int members) {
    point();
    const fftn::detail::sync_word w(&counter);
    w.increment();
    const auto target = static_cast<std::uint64_t>(members);
    if (!try_wait(w, [target](std::uint64_t v) { return v >= target; })) {
      return false;
    }
    const cost_model& c = ctl_->costs();
    ctl_->advance(member_, c.barrier_base_ns + c.barrier_per_member_ns * members);
    return true;
  }

private:
  template <class F> static void guarded(const F& f) noexcept {
    try {
      f();
    } catch (...) {
      std::fputs("simulator: an exception in a member\n", stderr);
      std::abort();
    }
  }

  void chunk_now(std::size_t task, std::size_t phase, std::uint32_t chunk, int owner, int lane) {
    real_->run_chunk(task, phase, chunk, owner, lane);
    const cost_model& c = ctl_->costs();
    const auto& ph = graph_->phases[phase];
    double ns = ph.work / static_cast<double>(ph.chunks) * c.ns_per_work;
    if (owner != lane) {
      ns *= c.remote_factor;
    }
    if (c.physical_group > 0 && owner / c.physical_group != lane / c.physical_group) {
      ns *= c.cross_factor;
    }
    ctl_->advance(member_, ns);
    rec_->chunk(task, phase, chunk, ctl_->clock(member_), member_);
  }

  controller* ctl_;
  int member_;
  recorder* rec_;
  const fftn::detail::team_env* real_;
  const fftn::detail::work_graph* graph_;
};

// One simulated team execution, or several in sequence on one workspace.
struct sim_result {
  bool deadlocked = false;
  std::vector<std::string> problems;
  std::vector<std::vector<std::byte>> outputs; // per execution
  double done_at = 0;                          // when the last execution's output was complete
  std::size_t steps = 0;
  std::vector<std::size_t> chunks_by_member; // over all executions
};

struct sim_setup {
  int lanes = 2;
  int group_size = 0;
  order how = order::random;
  std::uint64_t seed = 0;
  cost_model costs{};
  std::vector<member_spec> members; // empty: everyone at once
  int executions = 1;
  const fftn::detail::machine_profile* profile = nullptr; // null: the plan's
};

inline sim_result simulate_engine(const team_case& c, const sim_setup& setup) {
  const fftn::detail::plan_core& core = *c.core;
  team_space space(core, setup.lanes);
  const fftn::detail::lane_map lanes{.lanes = setup.lanes, .group_size = setup.group_size};
  const fftn::detail::schedule plan = fftn::detail::make_schedule(
      core.graph, lanes, setup.profile != nullptr ? *setup.profile : core.profile);
  std::vector<case_buffers> buffers;
  std::vector<fftn::detail::team_env> real;
  std::vector<recorder> records;
  const auto executions = static_cast<std::size_t>(setup.executions);
  buffers.reserve(executions);
  real.reserve(executions);
  records.reserve(executions);
  for (int k = 0; k < setup.executions; ++k) {
    buffers.emplace_back(c);
  }
  for (int k = 0; k < setup.executions; ++k) {
    case_buffers& b = buffers[static_cast<std::size_t>(k)];
    real.emplace_back(core, b.src(), b.dst(), space.slots, fftn::detail::wait_budget{});
    records.emplace_back(core.graph, core.prog.transforms);
  }
  std::vector<member_spec> members = setup.members;
  members.resize(static_cast<std::size_t>(setup.lanes));
  controller ctl(setup.how, setup.seed, setup.costs);
  std::atomic<std::uint64_t> flag{0};
  const bool ok = ctl.run(members, [&](int m) {
    for (int k = 0; k < setup.executions; ++k) {
      const auto i = static_cast<std::size_t>(k);
      const fftn::detail::team_job job{
          .graph = &core.graph,
          .plan = plan,
          .lanes = lanes,
          .state = space.state,
          .flag = fftn::detail::sync_word(&flag),
          .rank = m,
      };
      sim_env env(ctl, m, records[i], real[i], core.graph);
      fftn::detail::team_engine<sim_env>(job, env).run();
      records[i].member_returned(m);
    }
  });
  sim_result r;
  r.deadlocked = !ok;
  r.steps = ctl.steps();
  for (int k = 0; k < setup.executions; ++k) {
    const auto i = static_cast<std::size_t>(k);
    for (const std::string& p : records[i].problems()) {
      r.problems.push_back("execution " + std::to_string(k) + ": " + p);
    }
    if (!records[i].complete()) {
      r.problems.push_back("execution " + std::to_string(k) + " did not complete");
    }
    r.outputs.push_back(buffers[i].result());
    r.done_at = records[i].done_at();
    const auto& by = records[i].chunks_by_member();
    r.chunks_by_member.resize(std::max(r.chunks_by_member.size(), by.size()), 0);
    for (std::size_t m = 0; m < by.size(); ++m) {
      r.chunks_by_member[m] += by[m];
    }
  }
  return r;
}

// The prototype's team execution, as a fixed schedule per member.
struct static_step {
  enum class kind : unsigned char { whole, chunks, barrier } what = kind::whole;
  std::size_t task = 0;
  std::size_t count = 1; // whole: transforms from task
  std::size_t phase = 0; // chunks: chunks first, first + stride, ... of this phase
  std::uint32_t first = 0;
  std::uint32_t stride = 1;
  int owner = 0;           // chunks: whose cache holds the task
  std::size_t barrier = 0; // barrier: its index
  int members = 0;         // barrier: how many meet there
};

using static_schedule = std::vector<std::vector<static_step>>; // per member

// The prototype's way of sharing transforms [first, first + count) among `team` (lanes
// [base, base + team)): every pass of each in turn split round-robin over the team, with a
// barrier after each. Appends to s, numbering barriers from `barriers`.
inline void share_in_turn(static_schedule& s, const fftn::detail::work_graph& g, std::size_t first,
                          std::size_t count, int base, int team, std::size_t& barriers) {
  for (std::size_t t = first; t < first + count; ++t) {
    for (std::size_t p = 0; p < g.phases.size(); ++p) {
      for (int k = 0; k < team; ++k) {
        const std::size_t lane = static_cast<std::size_t>(base) + static_cast<std::size_t>(k);
        s[lane].push_back({
            .what = static_step::kind::chunks,
            .task = t,
            .phase = p,
            .first = static_cast<std::uint32_t>(k),
            .stride = static_cast<std::uint32_t>(team),
            .owner = base,
        });
        s[lane].push_back(
            {.what = static_step::kind::barrier, .barrier = barriers, .members = team});
      }
      ++barriers;
    }
  }
}

inline sim_result simulate_static(const team_case& c, const static_schedule& schedule,
                                  std::size_t barriers, const cost_model& costs,
                                  const std::vector<member_spec>& members_in) {
  const fftn::detail::plan_core& core = *c.core;
  const int lanes = static_cast<int>(schedule.size());
  const team_space space(core, lanes);
  case_buffers b(c);
  const fftn::detail::team_env real(core, b.src(), b.dst(), space.slots,
                                    fftn::detail::wait_budget{});
  recorder rec(core.graph, core.prog.transforms);
  std::vector<std::atomic<std::uint64_t>> counters(barriers);
  std::vector<member_spec> members = members_in;
  members.resize(static_cast<std::size_t>(lanes));
  controller ctl(order::timed, 0, costs);
  const bool ok = ctl.run(members, [&](int m) {
    sim_env env(ctl, m, rec, real, core.graph);
    for (const static_step& st : schedule[static_cast<std::size_t>(m)]) {
      switch (st.what) {
      case static_step::kind::whole:
        env.run_whole(st.task, st.count, m);
        break;
      case static_step::kind::chunks: {
        const auto& ph = core.graph.phases[st.phase];
        for (std::uint32_t k = st.first; k < ph.chunks; k += st.stride) {
          env.run_chunk(st.task, st.phase, k, st.owner, m);
        }
        break;
      }
      case static_step::kind::barrier:
        if (!env.barrier(counters[st.barrier], st.members)) {
          return; // deadlocked: unwind without touching anything shared
        }
        break;
      }
    }
  });
  sim_result r;
  r.deadlocked = !ok;
  r.problems = rec.problems();
  r.outputs.push_back(b.result());
  r.done_at = rec.done_at();
  return r;
}

} // namespace fftn::test::sim
