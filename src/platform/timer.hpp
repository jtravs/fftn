// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The clock for planning measurements (task 2.4). std::chrono::steady_clock: monotonic on
// every supported platform and fine-grained enough (on macOS it reads the 24 MHz system
// counter, about 42 ns per tick; Linux clock_gettime(CLOCK_MONOTONIC) and Windows
// QueryPerformanceCounter are similar or finer). Planning times whole executions of at
// least microseconds, so a cycle counter would add per-CPU skew and frequency questions for
// no gain.
#pragma once

#include <chrono>

namespace fftn::inline v0::detail {

using planning_clock = std::chrono::steady_clock;
static_assert(planning_clock::is_steady);

// The smallest step the clock was seen to advance by, from a short busy sample (about a
// millisecond at most).
[[nodiscard]] std::chrono::nanoseconds measured_clock_resolution() noexcept;

} // namespace fftn::inline v0::detail
