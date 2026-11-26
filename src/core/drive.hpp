// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The two drivers as policies of the code they share: the serial entry (plan_core.cpp,
// execute_program), the team environment (threading/team_env.hpp), the team member
// (threading/team_execute.hpp) and the executor dispatch (threading/executor_dispatch.hpp).
// plain_drive is driver.hpp, for every plan whose edges have no factors or callbacks; edge_drive
// is the edge driver (core/edge_driver.hpp), whose state is the execution's factor cursors and
// callback pointers.
//
// The per-task entries (run for whole transforms, at_task for a task's pieces) take the lane
// running them and the team's slots: a callback stage passes the lane to the callback as
// `member`, with that lane's own callback area (DESIGN.md 12.3; the owner's slot cannot serve, as
// two members may run blocks of one callback stage at once). plain_drive ignores both.
//
// The policies are static and plain_drive's state is empty, so that the plain instantiations
// take no parameter and make no call that the code made before edges existed, and compile to
// the same code: plans without edges lost 2-6 % on the EPYC 7763 when one driver moved factor
// cursors for every plan.
#pragma once

#include "core/driver.hpp"
#include "core/edge_driver.hpp"
#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/edges.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"

#include <cstddef>
#include <variant>

namespace fftn::inline v0::detail {

// The team overload's member (threading/team_execute.cpp) for plans without factor edges.
void execute_member(const plan_core& plan, const void* in, void* out, const team& t,
                    const workspace& ws, completion done, int threads) noexcept;

// execute_core's path on the edge driver from its state for the execution (plan_core.cpp).
void execute_core_cursors(const plan_core& core, std::size_t first, std::size_t count,
                          const void* in, void* out, const workspace& ws,
                          const edge_state& f) noexcept;

struct plain_drive {
  // Whether the serial entry takes its shortcut for one transform of a flat program (X28).
  static constexpr bool flat_shortcut = true;
  struct state {};
  struct task_cursors {};

  static void run(const plan_core& c, std::size_t first, std::size_t count,
                  const slot_addresses& at) noexcept {
    run_transforms(c.prog, first, count, at);
  }
  static void run(const plan_core& c, std::size_t first, std::size_t count,
                  const slot_addresses& at, const state& /*s*/, int /*lane*/) noexcept {
    run_transforms(c.prog, first, count, at);
  }
  [[nodiscard]] static task_cursors at_task(const plan_core& /*c*/, std::size_t /*t*/,
                                            const state& /*s*/, int /*lane*/,
                                            std::byte* /*slots*/) noexcept {
    return {};
  }
  static void stage(const plan_core& c, task_cursors /*f*/, std::size_t index,
                    const slot_addresses& at, std::size_t first, std::size_t last) noexcept {
    run_stage_chunks(*std::get_if<detail::stage>(&c.prog.records[index]), at, first, last);
  }
  static void lines(const plan_core& c, task_cursors /*f*/, std::size_t begin, std::size_t nest,
                    std::size_t body_end, std::size_t first, std::size_t last,
                    const slot_addresses& at) noexcept {
    run_lines(c.prog, begin, nest, body_end, first, last, at);
  }
  // A whole execution serially, from the plan's entry.
  static void serial(const plan_core& c, std::size_t first, std::size_t count, const void* in,
                     void* out, const workspace& ws) noexcept {
    execute_core(c, first, count, in, out, ws);
  }
  // A member of an executor's team (threading/executor_dispatch.hpp).
  static void member(const plan_core& c, const void* in, void* out, const team& t,
                     const workspace& ws, completion done, int threads) noexcept {
    execute_member(c, in, out, t, ws, done, threads);
  }
};

struct edge_drive {
  static constexpr bool flat_shortcut = false;
  using state = edge_state;
  using task_cursors = edge_task;

  // Whole transforms on the slot whose scratch line a starts it (serially the workspace's,
  // lane 0).
  static void run(const plan_core& c, std::size_t first, std::size_t count,
                  const slot_addresses& at, const state& s, int lane = 0) noexcept {
    run_edge_transforms(c.prog, c.edge_records, c.edge_tail_records, c.edge_batched, first, count,
                        at, s.f,
                        {.user = s.user,
                         .member = lane,
                         .scratch = callback_area(at.a, c.offset_callback, c.callback_bytes),
                         .stand_in = s.stand_in});
  }
  [[nodiscard]] static task_cursors at_task(const plan_core& c, std::size_t t, const state& s,
                                            int lane, std::byte* slots) noexcept {
    std::byte* slot =
        c.bytes != 0 ? slots + static_cast<std::size_t>(lane) * c.slot_bytes : nullptr;
    return {.f = edge_transform_cursors(c.prog, c.edge_records, t, s.f),
            .env = {.user = s.user,
                    .transform = t,
                    .member = lane,
                    .scratch = callback_area(slot, c.offset_callback, c.callback_bytes),
                    .stand_in = s.stand_in}};
  }
  static void stage(const plan_core& c, const task_cursors& f, std::size_t index,
                    const slot_addresses& at, std::size_t first, std::size_t last) noexcept {
    run_edge_stage_chunks(c.prog, c.edge_records, index, at, f.f, f.env, first, last);
  }
  static void lines(const plan_core& c, const task_cursors& f, std::size_t begin, std::size_t nest,
                    std::size_t body_end, std::size_t first, std::size_t last,
                    const slot_addresses& at) noexcept {
    run_edge_lines(c.prog, c.edge_records, begin, nest, body_end, first, last, at, f.f, f.env);
  }
  static void serial(const plan_core& c, std::size_t first, std::size_t count, const void* in,
                     void* out, const workspace& ws, const state& f) noexcept {
    execute_core_cursors(c, first, count, in, out, ws, f);
  }
  // Defined with the edge instantiations (threading/edge_execute.cpp).
  static void member(const plan_core& c, const void* in, void* out, const team& t,
                     const workspace& ws, completion done, int threads, const state& f) noexcept;
};

} // namespace fftn::inline v0::detail
