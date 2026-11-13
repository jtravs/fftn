// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/timer.hpp"

#include <algorithm>
#include <chrono>

namespace fftn::inline v0::detail {

std::chrono::nanoseconds measured_clock_resolution() noexcept {
  using std::chrono::nanoseconds;
  // Bounded by samples, time and reads, so a clock that never advances cannot hang it (the
  // result is then nanoseconds::max()).
  constexpr int samples = 64;
  constexpr long max_reads = 16'777'216;
  constexpr auto budget = std::chrono::milliseconds(1);
  nanoseconds smallest = nanoseconds::max();
  const auto start = planning_clock::now();
  auto previous = start;
  int seen = 0;
  for (long reads = 0; seen < samples && reads < max_reads; ++reads) {
    const auto now = planning_clock::now();
    if (now != previous) {
      smallest = std::min(smallest, std::chrono::duration_cast<nanoseconds>(now - previous));
      previous = now;
      ++seen;
    }
    if (now - start >= budget) {
      break;
    }
  }
  return smallest;
}

} // namespace fftn::inline v0::detail
