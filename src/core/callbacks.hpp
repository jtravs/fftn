// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Block callbacks (fftn/callback.hpp, DESIGN.md 12.3; task 5.3a): the callback stage, a stage of
// a plan's program that calls the application's function on blocks of a line between two of
// FFTN's passes. Baseline code: it is compiled in no ISA region, and a kernel has returned
// through the ordinary ABI before it runs, so no vector state or ISA attribute reaches user code.
//
// A callback stage is a stage like an edge pass: its units are the line's elements, cut into
// chunks fixed by the plan (about chunk_points elements each), and each chunk is one block. A
// team shares the chunks out as it does any stage's (work_graph.hpp), so every element is passed
// once, whoever runs it. The stage carries a guard as its kernel (driver.hpp aborts on it: an
// overload without exec_args); the edge driver (core/edge_driver.cpp) recognises it by
// stage::moving_factors' moving_callback bit and calls run_callback_stage with its edge_record's
// site, the slot address of its data, and the task's callback_env: the user pointers, the
// transform, the member and the member's scratch.
#pragma once

#include "core/program.hpp"
#include "fftn/callback.hpp"
#include "fftn/detail/core.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {

// An edge's callback as the plan applies it (edge_side::call): none where fn is null.
struct side_callback {
  erased_block_fn fn = nullptr;
  callback_mode mode = callback_mode::modify;
  std::size_t scratch_bytes = 0;
  [[nodiscard]] bool any() const noexcept { return fn != nullptr; }
};

// What a callback stage calls, fixed at construction: the function, converted back to the
// block_fn of its element type by `invoke` (one instantiation per element type), which user
// pointer of exec_args it receives, and its block's geometry beyond the stage's own (the stage's
// pass_args::dst_stride is element_stride; its dst slot and dst_cols the data's address).
struct callback_site {
  erased_block_fn fn = nullptr;
  void (*invoke)(erased_block_fn fn, void* user, const block_info& info,
                 std::byte* data) noexcept = nullptr;
  std::uint8_t user = 0;   // 0: exec_args::in_user, 1: out_user
  std::size_t rank = 1;    // the transformed axes, origin's entries
  std::size_t axis = 0;    // element_axis
  std::size_t logical = 0; // the logical position along `axis` of the stage's element 0
  std::size_t element_bytes = 0;
  std::size_t scratch_bytes = 0; // the callback's own (block_info::scratch's size)
};

// What a callback stage needs of the execution and of the task it runs in.
struct callback_env {
  std::array<void*, 2> user{};  // exec_args::in_user, out_user
  std::size_t transform = 0;    // the task's transform, numbered from the batch's first
  int member = 0;               // the lane running the stage
  std::byte* scratch = nullptr; // that lane's callback area (128-byte aligned), or null
};

// The callback area of the scratch slot that starts at `slot` (a 64-byte boundary serially, a
// 128-byte one in a team; core/plan_core.cpp lays it out with room for the alignment): null
// where the plan's callbacks take no scratch.
[[nodiscard]] std::byte* callback_area(std::byte* slot, std::size_t offset) noexcept;

// The guard kernel of a callback stage, for driver.hpp: it ends the program (contract_failure).
void callback_guard(const pass_args* args, const void* src, void* dst, std::size_t first,
                    std::size_t last);

// A callback stage over `count` elements of a line of the side's precision (real: of real
// values, else complex) at slot `where`, `stride` elements apart, starting `offset` bytes from
// the slot's cursor, whose element 0 is at logical position `logical` along the line: chunks of
// about chunk_points elements. Its site is added to `sites` (which owns it; the stage's edge
// record points to it, mark_cursors). `output`: the output edge's callback (out_user).
[[nodiscard]] stage callback_stage(precision prec, bool real, const side_callback& call,
                                   bool output, std::size_t count, std::size_t logical, slot where,
                                   std::ptrdiff_t stride, std::ptrdiff_t offset,
                                   std::size_t chunk_points,
                                   std::vector<std::unique_ptr<callback_site>>& sites);

// Chunks [first, last) of callback stage st with site `site` (each chunk one block), its data at
// `data` (the stage's dst slot address; dst_cols is added here).
void run_callback_stage(const stage& st, const callback_site& site, std::byte* data,
                        const callback_env& env, std::size_t first, std::size_t last) noexcept;

// describe()'s line for a plan's callbacks: where each runs and the block shape.
// in_where, out_where: where each side's blocks lie ("the workspace block", ...).
[[nodiscard]] std::string callbacks_text(const side_callback& in, const side_callback& out,
                                         const std::string& in_where, const std::string& out_where,
                                         std::size_t block_elements);

// The name of a callback mode.
[[nodiscard]] const char* mode_name(callback_mode m) noexcept;

} // namespace fftn::inline v0::detail
