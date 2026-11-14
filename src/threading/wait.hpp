// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// How a member with nothing to claim waits on a counter (DESIGN.md 13.3): spin for a time
// budget, then yield for another, then block in std::atomic_ref::wait (a futex on Linux,
// __ulock_wait on macOS, WaitOnAddress on Windows).
//
// The budgets are times, not iteration counts: a pause instruction costs about 140 cycles
// since Skylake-SP and a few elsewhere, so counts mean nothing across processors. Spinning
// is bounded because an idle team that spins steals cores from the threads doing the work
// (an application measured 334 s against 13.5 s when two teams alternated and the idle one
// spun).
//
// A notifier wakes only when some waiter has registered as blocked, so the common case makes
// no system call (libc++'s notify_all always makes one).
#pragma once

#include "threading/team_state.hpp"

#include <chrono>
#include <cstdint>
#include <thread>

namespace fftn::inline v0::detail {

struct wait_budget {
  std::chrono::nanoseconds spin{0};
  std::chrono::nanoseconds yield{0};
};

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) && defined(__clang__)
  // isb rather than yield, which is close to a no-op on AArch64 cores.
  __builtin_arm_isb(15);
#elif defined(__aarch64__)
  __asm__ __volatile__("isb" ::: "memory");
#endif
}

// Returns once ready(value of w) holds. `blocked`, when not null, counts the members blocked
// on w; it must be the word the notifier passes to wake().
template <class Ready>
void wait_for(sync_word w, const sync_word* blocked, Ready ready,
              const wait_budget& budget) noexcept {
  if (ready(w.load(std::memory_order_acquire))) {
    return;
  }
  using clock = std::chrono::steady_clock;
  const clock::time_point start = clock::now();
  clock::duration waited{0};
  // Spin, reading the clock only every 64 rounds: it costs tens of nanoseconds.
  constexpr unsigned rounds_per_check = 64;
  for (unsigned round = 1;; ++round) {
    cpu_relax();
    if (ready(w.load(std::memory_order_acquire))) {
      return;
    }
    if (round % rounds_per_check == 0) {
      waited = clock::now() - start;
      if (waited >= budget.spin) {
        break;
      }
    }
  }
  while (waited < budget.spin + budget.yield) {
    std::this_thread::yield();
    if (ready(w.load(std::memory_order_acquire))) {
      return;
    }
    waited = clock::now() - start;
  }
  if (blocked != nullptr) {
    blocked->increment();
  }
  for (;;) {
    const std::uint64_t seen = w.load();
    if (ready(seen)) {
      break;
    }
    w.wait(seen);
  }
  if (blocked != nullptr) {
    blocked->decrement();
  }
}

// Wakes the members blocked on w, after a change to w made with a sequentially consistent
// operation.
inline void wake(sync_word w, const sync_word* blocked) noexcept {
  if (blocked == nullptr || blocked->load() != 0) {
    w.notify_all();
  }
}

} // namespace fftn::inline v0::detail
