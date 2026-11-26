// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The driver of plans whose edges have factors (core/edges.hpp, DESIGN.md 12.1): the program
// interpreter of driver.hpp plus the cursors of the factor arrays, which arrive with each
// execution (exec_args). It is a separate driver so that every other plan runs on driver.hpp
// exactly as before edges existed: plans without edges were 2-6 % slower on the EPYC 7763 when
// one driver moved the cursors for every plan (rows of 8, 100 x 1024, real 2048 x 36).
//
// In such a plan a stage that reads a cursor (stage::moving_factors) carries guards as its
// kernels, and its edge_record the kernels themselves: executed through driver.hpp (an overload
// without exec_args) the plan ends the program (contract_failure) instead of reading through a
// null pointer. The walked form of the program always runs here (a team's pieces); a plan whose
// profile takes the batched forms (machine_profile::edge_batched_forms, task 5.2a) runs whole
// executions through edge_forms below: the flat form, the unrolled calls and rows of one leaf
// with the cursors moved per transform, and a c2c plan of rows with separate edge passes in
// blocks of rows. Each runs the same kernels on the same arguments as the walked form, one row's
// results not depending on the rows a call takes, so the bits are the same.
#pragma once

#include "core/callbacks.hpp"
#include "core/driver.hpp"
#include "core/program.hpp"
#include "fftn/edges.hpp"
#include "passes/pass_args.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fftn::inline v0::detail {

// Bytes per cursor, in factor_cursors' order: the input's, the output's and a pipeline's
// middle's (DESIGN.md 12.2).
using factor_steps = std::array<std::ptrdiff_t, 3>;

// What the edge driver keeps per record of a plan whose edges have factors.
struct edge_record {
  // A stage that reads a factor cursor: its kernel (the stage's own are guards).
  pass_fn run = nullptr;
  leaf_fn leaf = nullptr;
  // A loop: the bytes it moves the input's, the output's and the middle's factor cursors per
  // iteration.
  factor_steps factor_step{};
  // A callback stage (stage::moving_factors' moving_callback; task 5.3a): what it calls, owned
  // by the plan (plan_core::callback_sites).
  const callback_site* call = nullptr;
};

// The cursors of the edges' factor arrays, and of a pipeline's middle factor (task 5.2b):
// addresses as integers, so that moving one that is 0 (an edge without a factor) needs no test.
struct factor_cursors {
  std::uintptr_t in = 0;
  std::uintptr_t out = 0;
  std::uintptr_t mid = 0;
};

// The edge driver's state for one execution (core/drive.hpp, edge_drive): the cursors' starting
// values and the callbacks' user pointers (exec_args).
struct edge_state {
  factor_cursors f;
  std::array<void*, 2> user{};
  bool stand_in = false; // measured planning's (callback_env::stand_in)
};

// The edge driver's per-task entry: the task's cursors, and what its callback stages need (the
// transform, the lane running it and that lane's callback area).
struct edge_task {
  factor_cursors f;
  callback_env env;
};

// A stage of a flat program (program::flat) as the edge driver's batched forms run it: its
// kernels (the edge_record's where it reads a cursor), slots and arguments, and which cursors it
// reads. In the row-block form a stage either runs once over a block's rows (`block_call`: an
// edge pass, its arguments' rows and distances set for the block, a leaf over the rows
// src_dist and dst_dist bytes apart, or a callback stage over the rows dst_dist elements apart)
// or once per row. A callback stage (task 5.3b) runs through `call`, its stage (whose chunks are
// the blocks) at `callback`, which the plan owns.
struct edge_flat_stage {
  flat_stage s;
  std::uint8_t moving = 0;
  bool block_call = false;
  std::ptrdiff_t src_dist = 0;
  std::ptrdiff_t dst_dist = 0;
  const callback_site* call = nullptr;
  const stage* callback = nullptr;
};

// One kernel call of an unrolled program (program.hpp, call) on the edge driver: its kernel the
// edge_record's where it reads a cursor, and the bytes the loops have moved the cursors.
struct edge_call {
  call c;
  factor_steps factor_off{};
  std::uint8_t moving = 0;
};

// The batched forms of a plan on the edge driver (task 5.2a), empty where the plan does not take
// them: at most one of `flat` (the body of a flat program; with `leaf_rows`, one leaf between the
// caller's arrays) and `calls` (a walked program of at most calls_max calls per transform,
// unrolled). row_block: rows per block of the row-block form (0: none), a flat program of one
// batch loop whose stages pass each row through its own line of the workspace block,
// block_row_bytes apart.
struct edge_forms {
  std::vector<edge_flat_stage> flat;
  bool leaf_rows = false;
  std::vector<edge_call> calls;
  std::vector<pass_args> call_args;
  std::size_t row_block = 0;
  std::ptrdiff_t block_row_bytes = 0;
};

// The batched forms of program p with edge records e: the flat body where p is flat, else its
// calls where `unroll` and it unrolls; and the row-block form where row_block is at least 2 and
// p is flat with one batch loop that leaves the block in place. element_bytes: of a c2c row's
// complex values, for the edge passes' distances in elements.
[[nodiscard]] edge_forms make_edge_forms(const program& p, const std::vector<edge_record>& e,
                                         bool unroll, std::size_t row_block,
                                         std::ptrdiff_t block_row_bytes,
                                         std::ptrdiff_t element_bytes);

// The kernels a stage that reads a factor cursor carries for driver.hpp: they end the program
// (contract_failure).
void factor_guard_pass(const pass_args* args, const void* src, void* dst, std::size_t first,
                       std::size_t last);
void factor_guard_leaf(const pass_args* args, const void* src, void* dst, std::ptrdiff_t src_dist,
                       std::ptrdiff_t dst_dist, std::size_t count);

// The cursors' starting values from an execution's data: `needed` holds moving_in_factor,
// moving_out_factor and moving_mid_factor's bits for the arrays the plan reads (exec_args'
// in_factor, out_factor and mid_factor); a needed array that is null ends the program.
[[nodiscard]] factor_cursors cursors_of(const exec_args& args, std::uint8_t needed) noexcept;

// The edge driver's state of an execution: cursors_of's cursors and the callbacks' pointers.
[[nodiscard]] edge_state edge_state_of(const exec_args& args, std::uint8_t needed) noexcept;

// The cursors of transform t: f moved by the batch loops.
[[nodiscard]] factor_cursors edge_transform_cursors(const program& p,
                                                    const std::vector<edge_record>& e,
                                                    std::size_t t, factor_cursors f) noexcept;

// As run_transforms, run_stage_chunks and run_lines (driver.hpp), with the cursors: base's
// slots and f's cursors are those of transform 0 (run_edge_transforms) or of the transform the
// piece belongs to (the others). `tail`: the records of a lanes program's tail
// (program::lane_tail), one per record.
// forms: the plan's batched forms (make_edge_forms), which it runs where it has them.
// env: the callback stages' (run_edge_transforms sets its transform for each transform it runs,
// or each block's first; the others take the piece's).
void run_edge_transforms(const program& p, const std::vector<edge_record>& e,
                         const std::vector<edge_record>& tail, const edge_forms& forms,
                         std::size_t first, std::size_t count, const slot_addresses& base,
                         factor_cursors f, const callback_env& env) noexcept;
void run_edge_stage_chunks(const program& p, const std::vector<edge_record>& e, std::size_t index,
                           const slot_addresses& at, factor_cursors f, const callback_env& env,
                           std::size_t first, std::size_t last) noexcept;
void run_edge_lines(const program& p, const std::vector<edge_record>& e, std::size_t begin,
                    std::size_t nest, std::size_t body_end, std::size_t first, std::size_t last,
                    const slot_addresses& at, factor_cursors f, const callback_env& env) noexcept;

} // namespace fftn::inline v0::detail
