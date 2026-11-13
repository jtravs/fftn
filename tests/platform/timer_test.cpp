// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/timer.hpp"

#include <gtest/gtest.h>

#include <chrono>

namespace {

using fftn::detail::planning_clock;

TEST(Timer, IsMonotonic) {
  auto previous = planning_clock::now();
  for (int i = 0; i < 1000000; ++i) {
    const auto now = planning_clock::now();
    ASSERT_GE(now, previous);
    previous = now;
  }
}

TEST(Timer, ResolutionIsFineEnoughForPlanning) {
  const auto resolution = fftn::detail::measured_clock_resolution();
  EXPECT_GT(resolution.count(), 0);
  // Planning times executions of microseconds and more; a coarser clock would need longer
  // measurements than the planner budgets for.
  EXPECT_LE(resolution, std::chrono::microseconds(1));
}

} // namespace
