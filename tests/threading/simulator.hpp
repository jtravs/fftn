// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A deterministic simulator for team executions (DESIGN.md 13.7, task 3.2). Each member runs
// on a thread of its own, but only one runs at a time: the controller hands a baton from
// member to member at every point where the engine touches shared state (env.point()) and
// whenever a member has to wait. Which member runs next is the controller's choice:
//
// - order::random picks among the members that can run, from a seeded generator: an
//   adversarial interleaving, the same for the same seed on every platform;
// - order::timed keeps a virtual clock per member, advanced by a cost model, and always
//   runs the member that is furthest behind: a discrete-event simulation of the execution's
//   timing, used to reproduce the prototype's pathological cases.
//
// A member that waits on a condition that is false gives up the baton and looks again after
// each later access to shared state by another member, at that access's time. When no member can
// run but some are waiting, the execution is deadlocked: the controller says so and releases the
// waiters (the engine's environment then aborts the test process; a test-only model may unwind).

#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace fftn::test::sim {

enum class order : unsigned char { random, timed };

// Virtual time, in nanoseconds (order::timed only).
struct cost_model {
  double ns_per_work = 0.5;     // whole tasks and chunks: work_graph units
  double point_ns = 50;         // an atomic operation on a shared line
  double handoff_ns = 200;      // a wait ended by another member's write: the line moves
  double spin_ns = 20000;       // a wait longer than this blocked: it pays wake_ns
  double wake_ns = 5000;        // a blocked thread's wake-up
  double remote_factor = 1.5;   // a split task's chunk run by a member that is not its owner
  double cross_factor = 3.0;    // ... by a member of another physical cache group
  int physical_group = 0;       // members per physical cache group; 0: one group
  double barrier_base_ns = 200; // the prototype's barrier: base + per member
  double barrier_per_member_ns = 50;
};

struct member_spec {
  bool arrives = true;
  double arrive_ns = 0;        // order::timed
  std::size_t arrive_step = 0; // order::random: after this many steps of the others
};

class controller {
public:
  controller(order o, std::uint64_t seed, const cost_model& costs)
      : order_(o), engine_(seed), costs_(costs) {}

  // Runs body(m) on member m's thread for every member that arrives, one at a time.
  // Returns false when the execution deadlocked.
  bool run(const std::vector<member_spec>& members, const std::function<void(int)>& body) {
    const std::size_t n = members.size();
    state_.assign(n, member_state{});
    current_ = -1;
    events_.clear();
    steps_ = 0;
    for (std::size_t m = 0; m < n; ++m) {
      state_[m].spec = members[m];
      state_[m].clock = members[m].arrive_ns;
      state_[m].now = members[m].arrives ? status::waiting_to_start : status::absent;
    }
    deadlocked_ = false;
    std::vector<std::thread> threads;
    for (std::size_t m = 0; m < n; ++m) {
      if (members[m].arrives) {
        threads.emplace_back([this, m, &body] {
          const int me = static_cast<int>(m);
          take_turn(me);
          body(me);
          finish(me);
        });
      }
    }
    {
      const std::scoped_lock lock(mutex_);
      current_ = pick();
      turn_.notify_all();
    }
    for (std::thread& t : threads) {
      t.join();
    }
    return !deadlocked_;
  }

  // A switch point, before an access to shared state: costs point_ns and may hand the baton
  // to another member. When m has it back, it makes its access: an event, at m's time,
  // after which waiting members look again.
  void point(int m) {
    std::unique_lock lock(mutex_);
    member_state& s = state_[idx(m)];
    s.clock += costs_.point_ns;
    ++steps_;
    pass_on(m, lock);
    if (!deadlocked_) {
      events_.push_back(s.clock);
    }
  }

  // Work done by m: advances its clock.
  void advance(int m, double ns) {
    const std::scoped_lock lock(mutex_);
    state_[idx(m)].clock += ns;
  }

  // m's wait found its condition false. Returns false once the execution is deadlocked.
  bool block(int m) {
    std::unique_lock lock(mutex_);
    member_state& s = state_[idx(m)];
    if (s.now != status::blocked) {
      // It looks again after each later event, in turn, until its condition holds.
      s.now = status::blocked;
      s.wait_start = s.clock;
      s.seen = events_.size();
    }
    pass_on(m, lock);
    return !deadlocked_;
  }

  // m's wait found its condition true after blocking: the hand-off, and a wake-up if it
  // waited longer than the spin budget.
  void unblocked(int m) {
    const std::scoped_lock lock(mutex_);
    member_state& s = state_[idx(m)];
    s.now = status::running;
    s.clock += costs_.handoff_ns + (s.clock - s.wait_start > costs_.spin_ns ? costs_.wake_ns : 0);
  }

  [[nodiscard]] double clock(int m) const {
    const std::scoped_lock lock(mutex_);
    return state_[idx(m)].clock;
  }
  [[nodiscard]] const cost_model& costs() const { return costs_; }
  [[nodiscard]] bool deadlocked() const { return deadlocked_; }
  [[nodiscard]] std::size_t steps() const { return steps_; }

private:
  enum class status : unsigned char { absent, waiting_to_start, running, blocked, finished };
  struct member_state {
    member_spec spec;
    status now = status::absent;
    double clock = 0;
    double wait_start = 0;
    std::size_t seen = 0; // blocked: the next event to look again after
  };

  static std::size_t idx(int m) { return static_cast<std::size_t>(m); }

  [[nodiscard]] bool can_run(const member_state& s) const {
    switch (s.now) {
    case status::waiting_to_start:
      return order_ == order::timed || steps_ >= s.spec.arrive_step;
    case status::running:
      return true;
    case status::blocked:
      return s.seen < events_.size();
    case status::absent:
    case status::finished:
      break;
    }
    return false;
  }

  // The member to run next; -1 when none can.
  int pick() {
    std::vector<int> ready;
    for (std::size_t m = 0; m < state_.size(); ++m) {
      if (can_run(state_[m])) {
        ready.push_back(static_cast<int>(m));
      }
    }
    if (ready.empty()) {
      // Late arrivals are forced in before anything is called a deadlock.
      for (std::size_t m = 0; m < state_.size(); ++m) {
        if (state_[m].now == status::waiting_to_start) {
          ready.push_back(static_cast<int>(m));
        }
      }
    }
    if (ready.empty()) {
      return -1;
    }
    if (order_ == order::random) {
      return ready[static_cast<std::size_t>(engine_() % ready.size())];
    }
    const auto when = [this](int m) {
      const member_state& s = state_[idx(m)];
      return s.now == status::blocked ? std::max(s.clock, events_[s.seen]) : s.clock;
    };
    return *std::ranges::min_element(ready, [&](int a, int b) { return when(a) < when(b); });
  }

  // Hands the baton on and returns when it comes back to m.
  void pass_on(int m, std::unique_lock<std::mutex>& lock) {
    current_ = pick();
    if (current_ < 0) {
      release_all();
      return;
    }
    turn_.notify_all();
    wait_turn(m, lock);
  }

  void wait_turn(int m, std::unique_lock<std::mutex>& lock) {
    turn_.wait(lock, [&] { return current_ == m || deadlocked_; });
    member_state& s = state_[idx(m)];
    if (s.now == status::waiting_to_start) {
      s.now = status::running;
      s.clock = std::max(s.clock, order_ == order::timed ? s.spec.arrive_ns : 0.0);
    }
    if (s.now == status::blocked && !deadlocked_) {
      s.clock = std::max(s.clock, events_[s.seen]);
      ++s.seen;
    }
  }

  void take_turn(int m) {
    std::unique_lock lock(mutex_);
    wait_turn(m, lock);
  }

  void finish(int m) {
    const std::scoped_lock lock(mutex_);
    state_[idx(m)].now = status::finished;
    current_ = pick();
    if (current_ < 0) {
      release_all();
    }
    turn_.notify_all();
  }

  // Nothing can run: if anyone is still waiting, the execution is deadlocked.
  void release_all() {
    for (const member_state& s : state_) {
      if (s.now == status::blocked || s.now == status::running) {
        deadlocked_ = true;
      }
    }
    turn_.notify_all();
  }

  order order_;
  std::mt19937_64 engine_;
  cost_model costs_;
  mutable std::mutex mutex_;
  std::condition_variable turn_;
  std::vector<member_state> state_;
  int current_ = -1;
  std::vector<double> events_; // the times of the accesses to shared state, in order
  std::size_t steps_ = 0;
  bool deadlocked_ = false;
};

} // namespace fftn::test::sim
