// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The claiming engine's environment in the library (engine.hpp): it runs whole tasks and
// chunks through the program driver on the workspace's scratch slots, and waits with the
// spin, yield and block policy of wait.hpp.
//
// The workspace of a team of L lanes, from its first page boundary (team_page):
//
//   team state (team_state.hpp), whole pages | slot 0 | slot 1 | ... | slot L - 1
//
// Each slot is one serial execution's scratch (lines a and b, then the block), rounded up to
// whole 128-byte lines. A member runs whole tasks in its own slot. A split task's chunks use
// its owner's slot: stage chunks all of it, line chunks only its block, with the scratch
// lines of the member running the line.
#pragma once

#include "core/driver.hpp"
#include "core/plan_core.hpp"
#include "threading/team_state.hpp"
#include "threading/wait.hpp"

#include <cstddef>
#include <cstdint>

namespace fftn::inline v0::detail {

class team_env {
public:
  team_env(const plan_core& core, const void* in, void* out, std::byte* slots,
           const wait_budget& budget) noexcept;

  void point() const noexcept {}

  template <class Ready>
  void wait(sync_word w, const sync_word* blocked, Ready ready) const noexcept {
    wait_for(w, blocked, ready, budget_);
  }
  static void wake(sync_word w, const sync_word* blocked) noexcept { detail::wake(w, blocked); }

  // Transforms [first, first + count) whole, on lane's slot.
  void run_whole(std::size_t first, std::size_t count, int lane) const noexcept;

  // Chunk `chunk` of phase `phase` of transform `task`, the task's shared scratch being
  // owner's slot, run by lane.
  void run_chunk(std::size_t task, std::size_t phase, std::uint32_t chunk, int owner,
                 int lane) const noexcept;

private:
  [[nodiscard]] slot_addresses scratch(int lane) const noexcept;

  const plan_core& core_;
  slot_addresses io_;
  std::byte* slots_;
  wait_budget budget_;
};

} // namespace fftn::inline v0::detail
