// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Workspaces: the scratch memory of one execution.
#pragma once

#include "fftn/config.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

namespace fftn::inline v0 {

template <class In, class Out> class plan;

/// Scratch memory for executing plans, owned by the workspace or borrowed from the caller.
///
/// A plan holds no scratch itself: every execute call is handed a workspace, so one plan
/// can be executed by several threads at once, each with its own workspace. A workspace
/// serves any plan whose `workspace_bytes()` is at most `size()`, one execution at a time.
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

  /// A workspace sized for `p`: it owns `p.workspace_bytes()` bytes.
  ///
  /// Throws `std::bad_alloc` when the memory cannot be allocated.
  template <class In, class Out>
  explicit workspace(const plan<In, Out>& p) : workspace(p.workspace_bytes()) {}

  /// A workspace over memory that the caller owns, of any alignment. The memory must stay
  /// valid while the workspace is in use; the workspace never frees it.
  explicit workspace(std::span<std::byte> memory) noexcept : memory_(memory) {}

  /// Takes over `other`'s memory; `other` is left with none.
  workspace(workspace&& other) noexcept
      : owned_(std::move(other.owned_)), memory_(std::exchange(other.memory_, {})) {}

  /// Releases this workspace's memory and takes over `other`'s; `other` is left with none.
  workspace& operator=(workspace&& other) noexcept {
    owned_ = std::move(other.owned_);
    memory_ = std::exchange(other.memory_, {});
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
  std::unique_ptr<std::byte[]> owned_; // NOLINT(modernize-avoid-c-arrays): an owned byte array
  std::span<std::byte> memory_;
};

} // namespace fftn::inline v0
