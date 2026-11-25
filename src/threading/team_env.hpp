// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The claiming engine's environment in the library (engine.hpp): it runs whole tasks and
// chunks through the program driver on the workspace's scratch slots, and waits with the
// spin, yield and block policy of wait.hpp.
//
// The workspace of a team of L lanes, from its first 128-byte boundary, or page boundary where
// the profile gives the state pages of its own (machine_profile::team_state_page):
//
//   team state (team_state.hpp), whole lines or pages | slot 0 | slot 1 | ... | slot L - 1
//
// Each slot is one serial execution's scratch (lines a and b, then the block), rounded up to
// whole 128-byte lines. A member runs whole tasks in its own slot. A split task's chunks use
// its owner's slot: stage chunks all of it, line chunks only its block, with the scratch
// lines of the member running the line.
#pragma once

#include "core/drive.hpp"
#include "core/driver.hpp"
#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "threading/team_state.hpp"
#include "threading/wait.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace fftn::inline v0::detail {

// Drive: the driver (core/drive.hpp), plain_drive for every plan but those whose edges have
// factors (edge_drive); both share this code.
template <class Drive> class basic_team_env {
public:
  basic_team_env(const plan_core& core, const void* in, void* out, std::byte* slots,
                 const wait_budget& budget) noexcept;
  // With the driver's state (edge_drive: the execution's factor cursors).
  basic_team_env(const plan_core& core, const void* in, void* out, std::byte* slots,
                 const wait_budget& budget, const typename Drive::state& state) noexcept;

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
  [[no_unique_address]] typename Drive::state state_{};
};

using team_env = basic_team_env<plain_drive>;

template <class Drive>
basic_team_env<Drive>::basic_team_env(const plan_core& core, const void* in, void* out,
                                      std::byte* slots, const wait_budget& budget) noexcept
    : core_(core),
      io_{.in = static_cast<const std::byte*>(in), .out = static_cast<std::byte*>(out)},
      slots_(slots), budget_(budget) {}

template <class Drive>
basic_team_env<Drive>::basic_team_env(const plan_core& core, const void* in, void* out,
                                      std::byte* slots, const wait_budget& budget,
                                      const typename Drive::state& state) noexcept
    : core_(core),
      io_{.in = static_cast<const std::byte*>(in), .out = static_cast<std::byte*>(out)},
      slots_(slots), budget_(budget), state_(state) {}

template <class Drive> slot_addresses basic_team_env<Drive>::scratch(int lane) const noexcept {
  slot_addresses at = io_;
  if (core_.bytes != 0) {
    std::byte* slot = slots_ + static_cast<std::size_t>(lane) * core_.slot_bytes;
    at.a = slot;
    at.b = slot + core_.offset_b;
    at.block = core_.block_elements != 0 ? slot + core_.offset_block : nullptr;
  }
  return at;
}

template <class Drive>
void basic_team_env<Drive>::run_whole(std::size_t first, std::size_t count,
                                      int lane) const noexcept {
  Drive::run(core_, first, count, scratch(lane), state_);
}

template <class Drive>
void basic_team_env<Drive>::run_chunk(std::size_t task, std::size_t phase, std::uint32_t chunk,
                                      int owner, int lane) const noexcept {
  const graph_phase& ph = core_.graph.phases[phase];
  slot_addresses at = transform_addresses(core_.prog, task, scratch(owner));
  const typename Drive::task_cursors f = Drive::at_task(core_, task, state_);
  if (ph.nest == 0) {
    Drive::stage(core_, f, ph.record, at, chunk, chunk + 1);
    return;
  }
  // A line runs all its stages on the scratch lines of the member running it.
  const slot_addresses mine = scratch(lane);
  at.a = mine.a;
  at.b = mine.b;
  const std::size_t lo = chunk_start(ph.units, ph.chunks, chunk);
  const std::size_t hi = chunk_start(ph.units, ph.chunks, std::size_t{chunk} + 1);
  if (ph.tail_nest == 0) {
    Drive::lines(core_, f, ph.record, ph.nest, ph.body_end, lo, hi, at);
    return;
  }
  // Lines [0, head) are the first nest's, the rest the joining nest's (work_graph.hpp).
  if (lo < ph.head) {
    Drive::lines(core_, f, ph.record, ph.nest, ph.body_end, lo, std::min(hi, ph.head), at);
  }
  if (hi > ph.head) {
    Drive::lines(core_, f, ph.tail_record, ph.tail_nest, ph.tail_body_end,
                 std::max(lo, ph.head) - ph.head, hi - ph.head, at);
  }
}

} // namespace fftn::inline v0::detail
