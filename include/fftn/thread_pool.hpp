// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// FFTN's own thread pool, for programs that have no threading runtime of their own.
#pragma once

#include "fftn/config.hpp"
#include "fftn/executor.hpp"

#include <memory>

namespace fftn::inline v0 {

namespace detail {
struct pool_state;
} // namespace detail

/// A fixed set of `std::thread`s that runs tasks for FFTN, and for the program too: an
/// executor (`fftn::executor`) for the executor overload of `plan::execute`.
///
/// FFTN never creates one by itself: threads exist only while the program holds a
/// `thread_pool` it constructed, and the program decides how many. The thread that calls
/// `bulk` takes part as one of the pool's lanes, so a pool of `threads` lanes starts
/// `threads - 1` threads. Idle threads spin briefly after a job (tens of microseconds),
/// then yield, then block until the next one; they are not bound to CPUs.
///
/// Lanes are grouped by cache with `fftn::cache_group_size`, as if the pool's threads ran
/// on consecutive cores (`group_size()`).
///
/// Guarantees: `bulk` never allocates or locks; a call from inside one of the pool's
/// tasks runs its tasks serially on the calling thread (and `concurrency()` there is 1), as
/// does a call made while another thread's call is running on the pool. Destruction joins
/// every thread. Not copyable or movable.
class thread_pool {
public:
  /// Starts a pool of `threads` lanes: `threads - 1` threads, the caller of `bulk` being the
  /// last lane. With `threads` 0 or less, one lane per logical CPU the process may use (its
  /// allowed CPUs and CPU quota, where the operating system reports them).
  ///
  /// Throws `std::system_error` when a thread cannot be started (the threads already started
  /// are joined first), `std::bad_alloc` when memory runs out.
  explicit thread_pool(int threads = 0);

  /// Stops and joins every thread. Precondition: no call of `bulk` is running.
  ~thread_pool();

  /// Not copyable: a pool owns its threads.
  thread_pool(const thread_pool&) = delete;

  /// Not copyable: a pool owns its threads.
  thread_pool& operator=(const thread_pool&) = delete;

  /// Not movable: its threads refer to it.
  thread_pool(thread_pool&&) = delete;

  /// Not movable: its threads refer to it.
  thread_pool& operator=(thread_pool&&) = delete;

  /// The pool's lanes; 1 when called from inside a task of any `fftn::thread_pool`, so that
  /// a nested execution runs serially.
  [[nodiscard]] int concurrency() const noexcept;

  /// Lanes per cache group: `fftn::cache_group_size` of the pool's lanes, fixed at
  /// construction.
  [[nodiscard]] int group_size() const noexcept;

  /// Runs `task(arg, i)` once for every `i` in `[0, n)` on the pool's threads and the
  /// calling thread, and returns when all have returned; nothing for `n` 0 or less.
  ///
  /// Preconditions: `task` does not throw (a task that throws terminates the program). It
  /// may call `bulk` again, on this pool or another: that call runs serially.
  ///
  /// Guarantees: never allocates or locks. Tasks are claimed dynamically, in no particular
  /// order; a thread that wakes late finds the work done. Another thread's call made while
  /// this one runs does not wait: it runs its tasks serially on its own thread.
  void bulk(int n, task_function task, void* arg) const noexcept;

private:
  std::unique_ptr<detail::pool_state> state_;
};

} // namespace fftn::inline v0
