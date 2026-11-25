// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The synchronisation state of team executions, kept in the workspace's memory (DESIGN.md
// 5.3, 13.3): std::atomic<std::uint64_t> words, constructed there when the state is set up,
// each hot word on a cache line of its own.
//
// Executions are numbered. Every member counts the team executions it has made on the
// workspace (its 64-bit `calls` word, written by that member alone); since every member makes
// every execution in the same order, the count names the execution, its epoch, the same for
// every member of it. Each counter carries the low 40 bits of the epoch that last wrote it
// above a 24-bit count: a counter whose tag is not the current epoch's reads as zero, so
// counters reset themselves on first use and nothing clears them between executions. Tags
// are only ever compared for equality. Whether an execution is over is a separate,
// untagged word, `finished` (the highest epoch a member has returned from, never wrapping):
// a member that finds a word with another epoch's tag reads `finished` after it, and if its
// own epoch is there, its execution is over. This is what lets a member arrive late, after
// the others have moved on.
//
// The limit: a tag repeats every 2^40 executions, so a counter left unwritten for exactly a
// multiple of 2^40 executions would read as current. That is 12.7 days of back-to-back
// executions of one microsecond each, and the word must then be reused at exactly that
// execution (DESIGN.md 18.2).
//
// The workspace object holds one word besides its memory (workspace::team_) saying whether
// its memory holds valid state and for how many lanes; the first member to find it unset
// constructs the state's words, all zero, while the others wait (team_engine::prepare in
// engine.hpp). Serial executions clear the word, since their scratch overlaps the state.
#pragma once

#include "fftn/workspace.hpp"

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace fftn::inline v0::detail {

// 128 bytes, not 64: adjacent-line prefetchers fetch 64-byte lines in pairs, so counters of
// different teams 64 bytes apart still contend (the prototype's suspected cause of 2-3x
// swings between runs).
inline constexpr std::size_t sync_line = 128;

// Where the state sits in a team's workspace: from its first boundary of team_state_unit(page)
// bytes, filling whole units, where page is machine_profile::team_state_page. 0 keeps the state
// on its 128-byte lines, right before the members' slots; a page size gives the state pages of
// its own, which nothing else shares: neither what the heap put before the workspace nor the
// first member's slot after it (X76, X79).
[[nodiscard]] constexpr std::size_t team_state_unit(std::size_t page) noexcept {
  return page != 0 ? page : sync_line;
}

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

// Tagged counters: the epoch's low 40 bits above a 24-bit count.
inline constexpr unsigned count_bits = 24;
inline constexpr std::uint64_t tag_mask = (std::uint64_t{1} << 40U) - 1;
inline constexpr std::uint32_t count_mask = (1U << count_bits) - 1;

// The tag of epoch e. Epochs whose tag would be 0 are skipped (engine.hpp, begin), so a
// zeroed word never carries a live tag.
[[nodiscard]] constexpr std::uint64_t tag_of_epoch(std::uint64_t e) noexcept {
  return e & tag_mask;
}
[[nodiscard]] constexpr std::uint64_t tagged(std::uint64_t tag, std::uint32_t count) noexcept {
  return (tag << count_bits) | count;
}
[[nodiscard]] constexpr std::uint64_t tag_of(std::uint64_t word) noexcept {
  return word >> count_bits;
}
// The count under tag `tag`: zero for a word with another tag.
[[nodiscard]] constexpr std::uint32_t count_in(std::uint64_t word, std::uint64_t tag) noexcept {
  return tag_of(word) == tag ? static_cast<std::uint32_t>(word & count_mask) : 0;
}

// Where each word lives, in bytes from the (128-byte or page aligned) start of the state. The
// layout depends on the lane count alone; where the state starts depends on its unit too, which
// the workspace's flag records (team_flag), so a workspace serves any plan.
//
//   line 0         tasks done (tagged)
//   line 1         members blocked waiting for it
//   line 2         finished: the highest epoch a member has returned from (untagged)
//   calls          one word per lane, 16 to a line: written once per execution by its owner
//   groups         per group (at most one per lane), two lines: whole tasks claimed
//                  (tagged); members arrived (tagged)
//   split tasks    per split task (at most one per lane), two lines: chunks claimed, members
//                  joined, owner (tagged, one line); chunks done (tagged), members blocked
//   anchored       per lane, one line: its range of its group's whole tasks claimed (tagged)
struct team_layout {
  std::size_t lanes = 0;

  [[nodiscard]] std::size_t calls_lines() const noexcept {
    return (lanes * sizeof(std::uint64_t) + sync_line - 1) / sync_line;
  }
  [[nodiscard]] static std::size_t done() noexcept { return 0; }
  [[nodiscard]] static std::size_t done_blocked() noexcept { return sync_line; }
  [[nodiscard]] static std::size_t finished() noexcept { return 2 * sync_line; }
  [[nodiscard]] static std::size_t calls(std::size_t lane) noexcept {
    return 3 * sync_line + lane * sizeof(std::uint64_t);
  }
  [[nodiscard]] std::size_t whole(std::size_t group) const noexcept {
    return (3 + calls_lines() + 2 * group) * sync_line;
  }
  [[nodiscard]] std::size_t arrived(std::size_t group) const noexcept {
    return whole(group) + sync_line;
  }
  [[nodiscard]] std::size_t claim(std::size_t task) const noexcept {
    return (3 + calls_lines() + 2 * lanes + 2 * task) * sync_line;
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
  // A member's range of its group's whole tasks (machine_profile::anchored_runs): claimed
  // (tagged), one line per lane.
  [[nodiscard]] std::size_t anchored(std::size_t lane) const noexcept {
    return (3 + calls_lines() + 4 * lanes + lane) * sync_line;
  }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return (3 + calls_lines() + 5 * lanes) * sync_line;
  }
  // The state's whole units of team_state_unit(page) bytes: the members' slots start after
  // them. bytes() for page 0, whose unit is the state's own line.
  [[nodiscard]] std::size_t span(std::size_t page) const noexcept {
    const std::size_t unit = team_state_unit(page);
    return (bytes() + unit - 1) / unit * unit;
  }
};

// Where a team's state starts in a workspace's memory: at its first boundary of
// team_state_unit(page) bytes. workspace_bytes(lanes) leaves the slack for this.
[[nodiscard]] inline std::byte* team_state_in(std::byte* memory, std::size_t page) noexcept {
  const std::size_t unit = team_state_unit(page);
  const auto address = reinterpret_cast<std::uintptr_t>(memory);
  return memory + (unit - address % unit) % unit;
}

// The workspace's word saying whether its memory holds valid state: 0, or team_flag(lanes,
// unit, status) with one of these statuses.
inline constexpr std::uint64_t team_preparing = 1;
inline constexpr std::uint64_t team_ready = 2;

// The flag's value for a state of `lanes` lanes on units of `unit` bytes (team_state_unit: a
// power of two) with `status` in its low two bits. The state's place in the workspace depends
// on both, so a plan whose profile lays it out on other units (machine_profile::team_state_page)
// finds the flag not ready and prepares its own state, rather than reading another plan's
// words at another address as its own (X79's second read).
[[nodiscard]] constexpr std::uint64_t team_flag(std::size_t lanes, std::size_t unit,
                                                std::uint64_t status) noexcept {
  return ((static_cast<std::uint64_t>(lanes) * 64 +
           static_cast<std::uint64_t>(std::countr_zero(unit)))
          << 2U) |
         status;
}

struct workspace_access {
  [[nodiscard]] static sync_word team_word(const workspace& ws) noexcept {
    return sync_word(&ws.team_);
  }
};

} // namespace fftn::inline v0::detail
