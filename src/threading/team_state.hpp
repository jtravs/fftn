// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The synchronisation state of team executions, kept in the workspace's memory (DESIGN.md
// 5.3, 13.3): std::atomic<std::uint64_t> words, constructed there when the state is set up,
// each hot word on a cache line of its own.
//
// Executions are numbered. Every member counts the team executions it has made on the
// workspace (its `calls` word, written by that member alone); since every member makes
// every execution in the same order, the count names the execution, its epoch, the same for
// every member of it. Each counter carries the epoch that last wrote it in its high 32
// bits: a counter from an earlier epoch reads as zero, so counters reset themselves on
// first use and nothing clears them between executions, and a member that finds a later
// epoch knows that its own execution is over (a later one starts only after an earlier one
// completed). This is what lets a member arrive late, after the others have moved on.
//
// The workspace object holds one word besides its memory (workspace::team_) saying whether
// its memory holds valid state and for how many lanes; the first member to find it unset
// constructs the state's words, all zero, while the others wait (team_engine::prepare in
// engine.hpp). Serial executions clear the word, since their scratch overlaps the state.
#pragma once

#include "fftn/workspace.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace fftn::inline v0::detail {

// 128 bytes, not 64: adjacent-line prefetchers fetch 64-byte lines in pairs, so counters of
// different teams 64 bytes apart still contend (the prototype's suspected cause of 2-3x
// swings between runs).
inline constexpr std::size_t sync_line = 128;

class sync_word {
public:
  explicit sync_word(std::atomic<std::uint64_t>* word) noexcept : word_(word) {}

  [[nodiscard]] std::uint64_t
  load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
    return word_->load(order);
  }
  void store(std::uint64_t value,
             std::memory_order order = std::memory_order_seq_cst) const noexcept {
    word_->store(value, order);
  }
  // Sequentially consistent, so that a waiter's registration (wait.hpp) and a notifier's
  // check of it cannot both miss each other.
  bool compare_exchange(std::uint64_t& expected, std::uint64_t desired) const noexcept {
    return word_->compare_exchange_weak(expected, desired, std::memory_order_seq_cst);
  }
  void increment() const noexcept { word_->fetch_add(1); }
  void decrement() const noexcept { word_->fetch_sub(1); }
  void wait(std::uint64_t old) const noexcept { word_->wait(old, std::memory_order_seq_cst); }
  void notify_all() const noexcept { word_->notify_all(); }

private:
  std::atomic<std::uint64_t>* word_;
};

using team_word_t = std::atomic<std::uint64_t>;
static_assert(sizeof(team_word_t) == 8 && alignof(team_word_t) <= 8,
              "team state words are 8 bytes, 8-byte aligned");
static_assert(team_word_t::is_always_lock_free,
              "team state needs lock-free 64-bit atomics: execute takes no locks");

// Tagged counters: epoch in the high half, count in the low half.
[[nodiscard]] constexpr std::uint64_t tagged(std::uint32_t epoch, std::uint32_t count) noexcept {
  return (std::uint64_t{epoch} << 32U) | count;
}
[[nodiscard]] constexpr std::uint32_t epoch_of(std::uint64_t word) noexcept {
  return static_cast<std::uint32_t>(word >> 32U);
}
// Epochs compare modulo 2^32, so the numbering may wrap.
[[nodiscard]] constexpr bool is_later(std::uint64_t word, std::uint32_t epoch) noexcept {
  return static_cast<std::int32_t>(epoch_of(word) - epoch) > 0;
}
// The count in epoch `epoch`: zero for a word from an earlier one.
[[nodiscard]] constexpr std::uint32_t count_in(std::uint64_t word, std::uint32_t epoch) noexcept {
  return epoch_of(word) == epoch ? static_cast<std::uint32_t>(word) : 0;
}

// Where each word lives, in bytes from the (128-byte aligned) start of the state. The
// layout depends on the lane count alone, so a workspace serves any plan.
//
//   line 0         tasks done (tagged)
//   line 1         members blocked waiting for it
//   calls          one word per lane, 16 to a line: written once per execution by its owner
//   groups         per group (at most one per lane), two lines: whole tasks claimed
//                  (tagged); members arrived (tagged)
//   split tasks    per split task (at most one per lane), two lines: chunks claimed, members
//                  joined, owner (tagged, one line); chunks done (tagged), members blocked
struct team_layout {
  std::size_t lanes = 0;

  [[nodiscard]] std::size_t calls_lines() const noexcept {
    return (lanes * sizeof(std::uint64_t) + sync_line - 1) / sync_line;
  }
  [[nodiscard]] static std::size_t done() noexcept { return 0; }
  [[nodiscard]] static std::size_t done_blocked() noexcept { return sync_line; }
  [[nodiscard]] static std::size_t calls(std::size_t lane) noexcept {
    return 2 * sync_line + lane * sizeof(std::uint64_t);
  }
  [[nodiscard]] std::size_t whole(std::size_t group) const noexcept {
    return (2 + calls_lines() + 2 * group) * sync_line;
  }
  [[nodiscard]] std::size_t arrived(std::size_t group) const noexcept {
    return whole(group) + sync_line;
  }
  [[nodiscard]] std::size_t claim(std::size_t task) const noexcept {
    return (2 + calls_lines() + 2 * lanes + 2 * task) * sync_line;
  }
  [[nodiscard]] std::size_t members(std::size_t task) const noexcept {
    return claim(task) + sizeof(std::uint64_t);
  }
  [[nodiscard]] std::size_t owner(std::size_t task) const noexcept {
    return claim(task) + 2 * sizeof(std::uint64_t);
  }
  [[nodiscard]] std::size_t chunks_done(std::size_t task) const noexcept {
    return claim(task) + sync_line;
  }
  [[nodiscard]] std::size_t task_blocked(std::size_t task) const noexcept {
    return chunks_done(task) + sizeof(std::uint64_t);
  }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return (2 + calls_lines() + 4 * lanes) * sync_line;
  }
};

// The workspace's word saying whether its memory holds valid state: 0, or lanes x 4 plus
// one of these.
inline constexpr std::uint64_t team_preparing = 1;
inline constexpr std::uint64_t team_ready = 2;

struct workspace_access {
  [[nodiscard]] static sync_word team_word(const workspace& ws) noexcept {
    return sync_word(&ws.team_);
  }
};

} // namespace fftn::inline v0::detail
