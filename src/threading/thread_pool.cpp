// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// fftn::thread_pool (DESIGN.md 13.6): persistent std::threads that claim the tasks of one
// bulk call at a time, the caller among them.
//
// A job is published by storing its task, argument and size, then the count of tasks left
// (release), then bumping the epoch that idle threads wait on. A thread claims a task by
// decrementing `left` while it is positive, and only then reads the job's task and argument:
// a successful claim means the job is not over (it ends when every task has run), so what it
// reads is that job's, whichever epoch woke the thread. A thread that wakes late finds `left`
// at zero, or claims a task of a later job, as a member of that job: there is no per-job
// handshake, so a job's end never waits for an idle thread. The caller returns once `done`
// reaches the job's size; nothing of the job is touched after that.
//
// Waiting uses the engine's policy (wait.hpp): spin, then yield, then block in
// std::atomic::wait, with a notification only when a waiter has registered as blocked.

#include "fftn/thread_pool.hpp"

#include "fftn/executor.hpp"
#include "fftn/team.hpp"
#include "platform/topology.hpp"
#include "threading/team_state.hpp"
#include "threading/wait.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace fftn::inline v0::detail {

struct pool_state {
  // The words a job's members touch, each on its own line (team_state.hpp says why 128).
  alignas(sync_line) std::atomic<std::uint64_t> epoch{0};    // bumped per job and at stop
  alignas(sync_line) std::atomic<std::uint64_t> sleepers{0}; // threads blocked on epoch
  alignas(sync_line) std::atomic<std::uint64_t> left{0};     // tasks not yet claimed
  alignas(sync_line) std::atomic<std::uint64_t> done{0};     // tasks finished
  alignas(sync_line) std::atomic<std::uint64_t> caller_blocked{0};
  // The job: read only after a successful claim. Atomics, so that a thread may read them
  // while the caller writes the next job's (it then fails to claim and ignores them).
  alignas(sync_line) std::atomic<task_function> task{nullptr};
  std::atomic<void*> arg{nullptr};
  std::atomic<std::uint64_t> size{0};
  std::atomic<bool> busy{false}; // a bulk call is running
  std::atomic<bool> stop{false};

  int lanes = 1;
  int group_size = 0;
  // Idle threads stay ready this long after a job: the engine's provisional spin and yield
  // budgets (schedule.hpp), so that back-to-back executions do not pay for a wake-up.
  wait_budget linger{.spin = std::chrono::microseconds(20), .yield = std::chrono::microseconds(50)};
  std::vector<std::thread> threads;
};

namespace {

// Set while this thread runs a pool task (a pool thread, always): calls from it run serially.
thread_local bool in_pool_task = false;

// Claims and runs tasks of the current job until none is left.
void take_part(pool_state& s) noexcept {
  std::uint64_t left = s.left.load(std::memory_order_relaxed);
  for (;;) {
    while (left != 0 && !s.left.compare_exchange_weak(left, left - 1, std::memory_order_acquire,
                                                      std::memory_order_relaxed)) {
    }
    if (left == 0) {
      return;
    }
    const task_function task = s.task.load(std::memory_order_relaxed);
    void* const arg = s.arg.load(std::memory_order_relaxed);
    const std::uint64_t size = s.size.load(std::memory_order_relaxed);
    task(arg, static_cast<int>(left - 1));
    if (s.done.fetch_add(1) + 1 == size) {
      const sync_word blocked(&s.caller_blocked);
      wake(sync_word(&s.done), &blocked);
    }
    left = s.left.load(std::memory_order_relaxed);
  }
}

void work(pool_state& s) noexcept {
  in_pool_task = true;
  std::uint64_t seen = 0;
  const sync_word epoch(&s.epoch);
  const sync_word sleepers(&s.sleepers);
  for (;;) {
    wait_for(epoch, &sleepers, [seen](std::uint64_t e) { return e != seen; }, s.linger);
    seen = s.epoch.load(std::memory_order_acquire);
    if (s.stop.load(std::memory_order_acquire)) {
      return;
    }
    take_part(s);
  }
}

void stop_threads(pool_state& s) noexcept {
  s.stop.store(true, std::memory_order_release);
  s.epoch.fetch_add(1);
  s.epoch.notify_all();
  for (std::thread& t : s.threads) {
    t.join();
  }
  s.threads.clear();
}

// One lane per logical CPU the process may use: its allowed CPUs and CPU quota where the
// operating system reports them.
int default_lanes() {
  std::size_t cpus = 0;
  try {
    const topology& t = system_topology();
    cpus = t.allowed_cpus.empty() ? t.cpus.size() : t.allowed_cpus.size();
    if (t.cpu_quota > 0) {
      cpus = std::min(cpus, static_cast<std::size_t>(std::ceil(t.cpu_quota)));
    }
  } catch (...) {
    // A topology that cannot be read (an unreadable FFTN_TOPOLOGY recording): the standard
    // library's count.
    cpus = std::thread::hardware_concurrency();
  }
  return static_cast<int>(std::clamp<std::size_t>(cpus, 1, std::size_t{1} << 16U));
}

} // namespace

} // namespace fftn::inline v0::detail

namespace fftn::inline v0 {

thread_pool::thread_pool(int threads) : state_(std::make_unique<detail::pool_state>()) {
  detail::pool_state& s = *state_;
  s.lanes = threads > 0 ? threads : detail::default_lanes();
  s.group_size = cache_group_size(s.lanes);
  s.threads.reserve(static_cast<std::size_t>(s.lanes - 1));
  try {
    for (int i = 1; i < s.lanes; ++i) {
      s.threads.emplace_back([&s] { detail::work(s); });
    }
  } catch (...) {
    detail::stop_threads(s);
    throw;
  }
}

thread_pool::~thread_pool() { detail::stop_threads(*state_); }

int thread_pool::concurrency() const noexcept { return detail::in_pool_task ? 1 : state_->lanes; }

int thread_pool::group_size() const noexcept { return state_->group_size; }

void thread_pool::bulk(int n, task_function task, void* arg) const noexcept {
  detail::pool_state& s = *state_;
  if (n <= 0) {
    return;
  }
  const bool nested = detail::in_pool_task;
  if (nested || n == 1 || s.threads.empty() || s.busy.exchange(true, std::memory_order_acquire)) {
    // Serially, here: inside a pool task, or while another thread's job holds the pool.
    detail::in_pool_task = true;
    for (int i = 0; i < n; ++i) {
      task(arg, i);
    }
    detail::in_pool_task = nested;
    return;
  }
  const auto size = static_cast<std::uint64_t>(n);
  s.task.store(task, std::memory_order_relaxed);
  s.arg.store(arg, std::memory_order_relaxed);
  s.size.store(size, std::memory_order_relaxed);
  s.done.store(0, std::memory_order_relaxed);
  s.left.store(size, std::memory_order_release);
  s.epoch.fetch_add(1);
  const detail::sync_word sleepers(&s.sleepers);
  detail::wake(detail::sync_word(&s.epoch), &sleepers);

  detail::in_pool_task = true;
  detail::take_part(s);
  detail::in_pool_task = false;
  const detail::sync_word blocked(&s.caller_blocked);
  detail::wait_for(
      detail::sync_word(&s.done), &blocked, [size](std::uint64_t d) { return d == size; },
      s.linger);
  s.busy.store(false, std::memory_order_release);
}

} // namespace fftn::inline v0
