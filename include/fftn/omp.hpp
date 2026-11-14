// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// The OpenMP adapters: the threads of a parallel region as a team, and an executor that
/// opens a region.
///
/// Optional and header-only. `fftn.hpp` does not include it, and it compiles only with the
/// compiler's OpenMP option: it is compiled in the caller's translation units, against the
/// caller's OpenMP runtime, so the process has one OpenMP runtime (the library links none,
/// and two runtimes in one process crash or crawl). Usage is in docs/threading.md.
///
/// Neither adapter changes the runtime's settings (thread counts, wait policy, binding).
#pragma once

#include "fftn/config.hpp"
#include "fftn/executor.hpp"
#include "fftn/team.hpp"

#ifndef _OPENMP
#error "fftn/omp.hpp needs OpenMP: compile with the compiler's OpenMP option (-fopenmp)."
#endif

#include <omp.h>

namespace fftn::inline v0::omp {

/// The calling thread as a member of the team of the innermost enclosing OpenMP parallel
/// region: rank `omp_get_thread_num()`, size `omp_get_num_threads()`, and
/// `fftn::cache_group_size(size)` members per cache group. Outside any region it is a team
/// of one, whose execution is serial.
///
/// For the team overload of `plan::execute`: every thread of the region calls
/// `plan.execute(in, out, fftn::omp::this_team(), ws)` with one shared workspace. Every
/// thread of the team must make every such call, so call it from the region's body itself,
/// not from inside `single`, `masked`, `critical`, a task or a worksharing loop, where only
/// some threads run (there, use the serial overloads). In a nested region it describes the
/// inner team, whose threads then share one workspace among themselves.
///
/// Guarantees: never changes the runtime's settings. The first call in the process
/// detects the machine's topology (`fftn::cache_group_size`), which may allocate; later
/// calls do not.
[[nodiscard]] inline team this_team() noexcept {
  const int size = omp_get_num_threads();
  return {.rank = omp_get_thread_num(), .size = size, .group_size = cache_group_size(size)};
}

/// An executor (`fftn::executor`) that runs FFTN's tasks in an OpenMP parallel region it
/// opens, for the executor overload of `plan::execute` called outside any region.
///
/// Every region it opens has the same width, the executor's `concurrency()`: an execution
/// that has work for fewer lanes still opens a region of that width, whose surplus threads
/// return at once, because some runtimes (libgomp) end and restart threads when the width
/// of successive regions changes. A job too small to share, and every job at width 1, runs
/// on the calling thread without opening a region (`plan::execute` decides that). Called
/// from inside an active parallel region, `concurrency()` is 1, so the execution runs
/// serially on the calling thread instead of nesting a region.
///
/// Guarantees: never changes the runtime's settings: the width goes into each region's
/// `num_threads` clause. Copyable; holds only the width.
class executor {
public:
  /// An executor of `omp_get_max_threads()` lanes, the width an unadorned
  /// `#pragma omp parallel` would have here. Construct it outside parallel regions.
  executor() noexcept : executor(omp_get_max_threads()) {}

  /// An executor of `threads` lanes (at least 1).
  explicit executor(int threads) noexcept : width_(threads < 1 ? 1 : threads) {}

  /// The width of the regions it opens, or 1 inside an active parallel region.
  [[nodiscard]] int concurrency() const noexcept { return omp_in_parallel() != 0 ? 1 : width_; }

  /// Lanes per cache group: `fftn::cache_group_size` of the width.
  [[nodiscard]] int group_size() const noexcept { return cache_group_size(width_); }

  /// Runs `task(arg, i)` once for every `i` in `[0, n)` in a parallel region of the
  /// executor's width, thread t running tasks t, t + w, ... (w the region's actual number
  /// of threads); threads with no task return at once. Inside an active parallel region, or
  /// for a single task, the tasks run on the calling thread instead.
  ///
  /// Preconditions: `task` does not throw (an exception cannot leave a parallel region).
  void bulk(int n, task_function task, void* arg) const noexcept {
    if (n <= 0) {
      return;
    }
    if (n == 1 || width_ == 1 || omp_in_parallel() != 0) {
      for (int i = 0; i < n; ++i) {
        task(arg, i);
      }
      return;
    }
#pragma omp parallel num_threads(width_)
    {
      // The runtime may give the region fewer threads than asked for (a thread limit, dynamic
      // adjustment): the stride covers every task whatever it gives.
      const int threads = omp_get_num_threads();
      for (int i = omp_get_thread_num(); i < n; i += threads) {
        task(arg, i);
      }
    }
  }

private:
  int width_;
};

} // namespace fftn::inline v0::omp
