// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Block callbacks (fftn/callback.hpp, DESIGN.md 12.3; task 5.3a): the callback stage, a stage of
// a plan's program that calls the application's function on blocks of a line between two of
// FFTN's passes. Baseline code: it is compiled in no ISA region, and a kernel has returned
// through the ordinary ABI before it runs, so no vector state or ISA attribute reaches user code.
//
// A callback stage is a stage like an edge pass: its units are the line's elements, cut into
// chunks fixed by the plan, and each chunk is one block: the line whole up to
// machine_profile::callback_block_bytes (task 5.3b), else element ranges of about that many
// bytes. Where a strategy holds several lines together (a lanes block of rows, the lanes over
// adjacent transforms, a row block) one call takes as many whole lines as fit that size, its
// lines line_stride elements apart; only a line longer than the size is cut, one line a call
// (fftn/callback.hpp's contract: a line whole up to the size, a longer one in element ranges).
// A team shares the chunks out as it does any stage's
// (work_graph.hpp), so every element is passed once, whoever runs it. The stage carries a guard as
// its kernel (driver.hpp aborts on it: an overload without exec_args); the edge driver
// (core/edge_driver.cpp) recognises it by stage::moving_factors' moving_callback bit and calls
// run_callback_stage with its edge_record's site, the slot address of its data, and the task's
// callback_env: the user pointers, the transform, the member and the member's scratch.
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
  std::size_t axis = 0;    // element_axis (an index into shape::axes)
  std::size_t logical = 0; // the logical position along `axis` of the stage's element 0
  std::size_t element_bytes = 0;
  std::size_t scratch_bytes = 0; // the callback's own (block_info::scratch's size)
  // Where a block's lines advance (task 5.3c): consecutive transforms (along_transforms), or an
  // axis of shape::axes, line k one element further along it than line k - 1 (a lanes column
  // block's adjacent columns). `lines`: the lines a stage of the walked form passes (a column
  // block's columns; 1 elsewhere, where a lanes program passes its block's rows instead).
  std::size_t line_axis = along_transforms;
  std::size_t lines = 1;
};

// What a callback stage needs of the execution and of the task it runs in.
struct callback_env {
  std::array<void*, 2> user{};  // exec_args::in_user, out_user
  std::size_t transform = 0;    // the task's transform, numbered from the batch's first
  int member = 0;               // the lane running the stage
  std::byte* scratch = nullptr; // that lane's callback area (128-byte aligned), or null
  // The logical coordinates the enclosing lines loops have reached, per axis of shape::axes
  // (task 5.3c: the edge driver adds each loop's, edge_record::origin_axis): a block's origin
  // is these plus its stage's own along its element and line axes.
  std::array<std::size_t, 3> origin{};
  // Measured planning's stand-ins (DESIGN.md 12.3, "Planning"; task 5.3b): every callback
  // stage runs, with its copies and boundaries, but calls a function that does nothing, so
  // that planning never calls the application's code.
  bool stand_in = false;
};

// The callback area, `offset` bytes into the scratch slot that starts at `slot` (a 64-byte
// boundary serially, a 128-byte one in a team; core/plan_core.cpp lays it out with room for the
// alignment): null where the plan's callbacks take no scratch (`bytes` 0).
[[nodiscard]] std::byte* callback_area(std::byte* slot, std::size_t offset,
                                       std::size_t bytes) noexcept;

// The guard kernel of a callback stage, for driver.hpp: it ends the program (contract_failure).
void callback_guard(const pass_args* args, const void* src, void* dst, std::size_t first,
                    std::size_t last);

// A callback stage over `count` elements of a line of the side's precision (real: of real
// values, else complex) at slot `where`, `stride` elements apart, starting `offset` bytes from
// the slot's cursor, whose element 0 is at logical position `logical` along the line. `lines`:
// the lines the stage runs on at most (a lanes block's rows; 1 elsewhere), line_dist elements
// apart (pass_args::dst_dist); its chunks hold at most block_bytes (set_callback_blocks). Its site
// is added to `sites` (which owns it; the stage's edge record points to it, mark_cursors).
// `output`: the output edge's callback (out_user).
[[nodiscard]] stage callback_stage(precision prec, bool real, const side_callback& call,
                                   bool output, std::size_t count, std::size_t logical, slot where,
                                   std::ptrdiff_t stride, std::ptrdiff_t offset,
                                   std::size_t block_bytes,
                                   std::vector<std::unique_ptr<callback_site>>& sites,
                                   std::size_t lines = 1, std::ptrdiff_t line_dist = 0);

// The blocks of a callback stage over `count` elements of each of up to `lines` lines of
// element_bytes each, cut at block_bytes: where a line fits, `per_call` whole lines a call
// (`ranges` 1); a longer line one a call (`per_call` 1) in `ranges` element ranges of at most
// that many bytes (at least one element each). `chunks`: the stage's, `ranges` for each group
// of per_call lines, the groups outermost.
struct callback_cut {
  std::size_t per_call = 1;
  std::size_t ranges = 1;
  std::size_t chunks = 1;
};
[[nodiscard]] callback_cut callback_cuts(std::size_t count, std::size_t lines,
                                         std::size_t element_bytes,
                                         std::size_t block_bytes) noexcept;

// Sets callback stage st's blocks (callback_cuts over its units) for up to `lines` lines a run:
// its chunks, and in its arguments, which no kernel reads, the lines a call (pass_args::rows)
// and the element ranges of each group (pass_args::block).
void set_callback_blocks(stage& st, std::size_t lines, std::size_t element_bytes,
                         std::size_t block_bytes) noexcept;

// Chunks [first, last) of callback stage st with site `site` (each chunk one block), its data at
// `data` (the stage's dst slot address; dst_cols is added here), over `lines` lines (at least
// 1, at most the stage's own) line_stride elements apart: the chunks of a group past `lines` are
// empty, and a call's first line is transform `env.transform` plus the group's first line.
void run_callback_stage(const stage& st, const callback_site& site, std::byte* data,
                        const callback_env& env, std::size_t lines, std::ptrdiff_t line_stride,
                        std::size_t first, std::size_t last) noexcept;

// Where one side's callback runs, for describe(): `where` the data's place and order ("on each
// row in the output, before the row's strategy"), `block` the blocks' shape ("blocks of up to
// 512 elements of one row"), and whether the stage is one more sweep over a line that exceeds
// the block size (its lines cut into element ranges).
struct callback_place {
  std::string where;
  std::string block;
  bool sweep = false;
};

// describe()'s line for a plan's callbacks: where each runs and the block shape.
[[nodiscard]] std::string callbacks_text(const side_callback& in, const side_callback& out,
                                         const callback_place& in_place,
                                         const callback_place& out_place);

// describe()'s words for the blocks of a callback stage over a line of `count` elements of
// element_bytes, `lines` lines a call (lanes: what), cut at block_bytes.
[[nodiscard]] callback_place callback_blocks(std::string where, std::size_t count,
                                             std::size_t lines, std::size_t element_bytes,
                                             std::size_t block_bytes, const char* what);

// The name of a callback mode.
[[nodiscard]] const char* mode_name(callback_mode m) noexcept;

} // namespace fftn::inline v0::detail
