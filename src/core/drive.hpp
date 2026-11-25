// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The two drivers as policies of the code they share: the serial entry (plan_core.cpp,
// execute_program), the team environment (threading/team_env.hpp), the team member
// (threading/team_execute.hpp) and the executor dispatch (threading/executor_dispatch.hpp).
// plain_drive is driver.hpp, for every plan whose edges have no factors; edge_drive is the edge
// driver (core/edge_driver.hpp), whose state is the execution's factor cursors.
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

// execute_core's path on the edge driver from the cursors' starting values (plan_core.cpp).
void execute_core_cursors(const plan_core& core, std::size_t first, std::size_t count,
                          const void* in, void* out, const workspace& ws,
                          const factor_cursors& f) noexcept;

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
                  const slot_addresses& at, const state& /*s*/) noexcept {
    run_transforms(c.prog, first, count, at);
  }
  [[nodiscard]] static task_cursors at_task(const plan_core& /*c*/, std::size_t /*t*/,
                                            const state& /*s*/) noexcept {
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
  using state = factor_cursors;
  using task_cursors = factor_cursors;

  static void run(const plan_core& c, std::size_t first, std::size_t count,
                  const slot_addresses& at, const state& f) noexcept {
    run_edge_transforms(c.prog, c.edge_records, c.edge_tail_records, first, count, at, f);
  }
  [[nodiscard]] static task_cursors at_task(const plan_core& c, std::size_t t,
                                            const state& f) noexcept {
    return edge_transform_cursors(c.prog, c.edge_records, t, f);
  }
  static void stage(const plan_core& c, task_cursors f, std::size_t index, const slot_addresses& at,
                    std::size_t first, std::size_t last) noexcept {
    run_edge_stage_chunks(c.prog, c.edge_records, index, at, f, first, last);
  }
  static void lines(const plan_core& c, task_cursors f, std::size_t begin, std::size_t nest,
                    std::size_t body_end, std::size_t first, std::size_t last,
                    const slot_addresses& at) noexcept {
    run_edge_lines(c.prog, c.edge_records, begin, nest, body_end, first, last, at, f);
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
