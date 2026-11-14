// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Workspaces: the scratch memory of one execution.
#pragma once

#include "fftn/config.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace fftn::inline v0 {

template <class In, class Out> class plan;

namespace detail {
struct workspace_access;
} // namespace detail

/// Scratch memory for executing plans, owned by the workspace or borrowed from the caller.
///
/// A plan holds no scratch itself: every execute call is handed a workspace, so one plan
/// can be executed by several threads at once, each with its own workspace. A workspace
/// serves any plan whose `workspace_bytes()` is at most `size()`, one execution at a time.
///
/// A team of threads executing one plan together (`fftn::team`) shares one workspace,
/// sized for the team with `plan::workspace_bytes(lanes)`: it then also holds the team's
/// synchronisation state, which persists from one team execution to the next. The
/// workspace sets that state up itself on the first team execution, and again after any
/// serial execution, a change of team size, or a `completion::own` execution by a smaller
/// team, so its memory never needs initialising.
///
/// Guarantees: memory is allocated, if at all, only when the workspace is constructed;
/// executing a plan never allocates. The contents are scratch: they need no
/// initialisation and are not preserved between executions. A workspace is movable, not
/// copyable.
class workspace {
public:
  /// A workspace with no memory, enough for plans whose `workspace_bytes()` is 0.
  workspace() noexcept = default;

  /// A workspace that owns `bytes` bytes of uninitialised memory.
  ///
  /// Throws `std::bad_alloc` when the memory cannot be allocated.
  explicit workspace(std::size_t bytes);

  /// A workspace sized for executions of `p` by teams of up to `lanes` members (1: serial
  /// execution only): it owns `p.workspace_bytes(lanes)` bytes.
  ///
  /// Throws `std::bad_alloc` when the memory cannot be allocated.
  template <class In, class Out>
  explicit workspace(const plan<In, Out>& p, int lanes = 1) : workspace(p.workspace_bytes(lanes)) {}

  /// A workspace over memory that the caller owns, of any alignment. The memory must stay
  /// valid while the workspace is in use; the workspace never frees it.
  explicit workspace(std::span<std::byte> memory) noexcept : memory_(memory) {}

  /// Takes over `other`'s memory; `other` is left with none.
  workspace(workspace&& other) noexcept
      : owned_(std::move(other.owned_)), memory_(std::exchange(other.memory_, {})),
        team_(other.team_.exchange(0, std::memory_order_relaxed)) {}

  /// Releases this workspace's memory and takes over `other`'s; `other` is left with none.
  workspace& operator=(workspace&& other) noexcept {
    owned_ = std::move(other.owned_);
    memory_ = std::exchange(other.memory_, {});
    team_.store(other.team_.exchange(0, std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }

  /// Not copyable: two executions must not share scratch by accident.
  workspace(const workspace&) = delete;

  /// Not copyable: two executions must not share scratch by accident.
  workspace& operator=(const workspace&) = delete;

  /// Frees the memory if the workspace owns it.
  ~workspace() = default;

  /// The usable size in bytes.
  [[nodiscard]] std::size_t size() const noexcept { return memory_.size(); }

  /// The memory itself, for the library's use during an execution.
  [[nodiscard]] std::span<std::byte> memory() const noexcept { return memory_; }

private:
  friend struct detail::workspace_access;

  std::unique_ptr<std::byte[]> owned_; // NOLINT(modernize-avoid-c-arrays): an owned byte array
  std::span<std::byte> memory_;
  // Whether the memory holds valid team synchronisation state, and for how many lanes
  // (src/threading/team_state.hpp). Moves carry it with the memory; no execution overlaps
  // a move.
  mutable std::atomic<std::uint64_t> team_{0};
};

} // namespace fftn::inline v0
