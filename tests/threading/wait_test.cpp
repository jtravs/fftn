// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The waiting policy (src/threading/wait.hpp): a waiter spins, then yields, then blocks,
// and a notifier wakes blocked waiters; a waiter registered as blocked is never missed.

#include "threading/team_state.hpp"
#include "threading/wait.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace {

using fftn::detail::sync_word;
using fftn::detail::wait_budget;

// Waits with `budget` for a word another thread sets after `delay`; returns how many
// waiters were registered as blocked when the setter looked.
int wait_for_setter(const wait_budget& budget, std::chrono::microseconds delay, bool use_blocked) {
  std::atomic<std::uint64_t> word{0};
  std::atomic<std::uint64_t> blocked{0};
  const sync_word w(&word);
  const sync_word b(&blocked);
  std::atomic<int> seen{-1};
  std::thread setter([&] {
    std::this_thread::sleep_for(delay);
    w.store(1);
    seen.store(static_cast<int>(b.load()));
    fftn::detail::wake(w, use_blocked ? &b : nullptr);
  });
  fftn::detail::wait_for(
      w, use_blocked ? &b : nullptr, [](std::uint64_t v) { return v == 1; }, budget);
  setter.join();
  EXPECT_EQ(w.load(), 1U);
  EXPECT_EQ(b.load(), 0U); // the waiter deregistered
  return seen.load();
}

TEST(Wait, AlreadyTrueReturnsAtOnce) {
  std::atomic<std::uint64_t> word{5};
  const sync_word w(&word);
  fftn::detail::wait_for(w, nullptr, [](std::uint64_t v) { return v == 5; }, {});
  SUCCEED();
}

TEST(Wait, SpinsThenYieldsThenBlocks) {
  const std::chrono::seconds second(1);
  const std::chrono::seconds none(0);
  const std::chrono::microseconds soon(100);
  const std::chrono::microseconds later(20000);
  // Satisfied while spinning.
  EXPECT_EQ(wait_for_setter({.spin = second, .yield = none}, soon, true), 0);
  // Satisfied while yielding.
  EXPECT_EQ(wait_for_setter({.spin = none, .yield = second}, soon, true), 0);
  // Blocked: the setter saw the registration and woke it.
  EXPECT_EQ(wait_for_setter({.spin = none, .yield = none}, later, true), 1);
  // Blocked with no registration word: the notifier always wakes.
  EXPECT_EQ(wait_for_setter({.spin = none, .yield = none}, later, false), 0);
}

TEST(Wait, TaggedWords) {
  using fftn::detail::count_in;
  using fftn::detail::tag_of;
  using fftn::detail::tag_of_epoch;
  using fftn::detail::tagged;
  const std::uint64_t w = tagged(7, 41);
  EXPECT_EQ(tag_of(w), 7U);
  EXPECT_EQ(count_in(w, 7), 41U);
  EXPECT_EQ(count_in(w, 8), 0U); // another execution's: reads as zero
  EXPECT_EQ(count_in(w, 6), 0U);
  // Tags are the low 40 bits of the epoch, with the full 24-bit count beside them.
  const std::uint64_t big = (std::uint64_t{1} << 40U) + 3;
  EXPECT_EQ(tag_of_epoch(big), 3U);
  const std::uint64_t full =
      tagged(tag_of_epoch((std::uint64_t{1} << 40U) - 1), fftn::detail::count_mask);
  EXPECT_EQ(tag_of(full), (std::uint64_t{1} << 40U) - 1);
  EXPECT_EQ(count_in(full, (std::uint64_t{1} << 40U) - 1), fftn::detail::count_mask);
}

} // namespace
