// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Programs (DESIGN.md 8.3): a plan's executable form, a flat array of stage and loop
// records prepared at construction and interpreted by a small driver (driver.hpp).
//
// A stage is one pass over one line. Its source and destination are slots: the caller's
// input and output arrays, two scratch lines a and b, and a block holding one transform's
// half spectrum or its last column (multi-dimensional c2r out of place, DESIGN.md 9.1), or a
// grid's intermediate (strategies/grid.hpp). The driver keeps a
// cursor into the in, out and block slots; loops move it. Scratch lines do not move: every
// line uses them from their start.
//
// A program is
//
//   batch loops (one per batch dimension, outermost first), then the body: one transform.
//
// The body is, for each transformed axis in turn (a phase), loops over the lines of the
// other axes enclosing the stages of one line. Every loop's iterations are independent,
// and every stage divides into `chunks` independent ranges of units fixed by the plan:
// the work graph of phase 3 reads both off (DESIGN.md 13.2). Serial execution runs every
// chunk in order.
#pragma once

#include "passes/pass_args.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace fftn::inline v0::detail {

enum class slot : std::uint8_t { in, out, a, b, block };

// What a loop moves, as indices into loop::step: the in, out and block slots, and the
// twiddle cursor, a byte offset added to the twiddles of every stage that asks for it
// (stage::moving_twiddles): the grid's per-block tables (src/strategies/grid.hpp).
inline constexpr std::size_t moving_slots = 4; // in, out, block, twiddles

// stage::moving_factors: which of a stage's factor pointers (pass_args::in_factor,
// out_factor) are the cursors of a plan's edges' factor arrays (core/edge_driver.hpp) rather
// than fixed tables.
inline constexpr std::uint8_t moving_in_factor = 1;
inline constexpr std::uint8_t moving_out_factor = 2;
// stage::moving_factors: the stage is a block callback stage (core/callbacks.hpp), which only the
// edge driver runs, from its edge_record's site.
inline constexpr std::uint8_t moving_callback = 4;

struct stage {
  pass_fn run = nullptr; // the kernel, or for a leaf null and `leaf` set
  pass_args args;        // the kernel's src and dst come from the slots, at execution
  slot src = slot::in;
  slot dst = slot::out;
  std::size_t units = 0;    // the kernel's own units: blocks of butterflies or copies, a line,
                            // elements (src/strategies/chain.hpp); only their count matters here
  std::uint32_t chunks = 1; // independent ranges [units c / chunks, units (c+1) / chunks)
  bool moving_twiddles = false; // args.twiddles is offset by the loops' twiddle cursor
  // moving_in_factor, moving_out_factor: args.in_factor or out_factor is the cursor of that
  // edge's factor array (a plan's edges, fftn/edges.hpp), set at execution by the edge driver
  // (core/edge_driver.hpp); `run` and `leaf` are then guards and the kernels are in the plan's
  // edge_record. In the padding after moving_twiddles; the driver of other plans never reads it.
  std::uint8_t moving_factors = 0;
  // A leaf's kernel (pass_args.hpp), with units 1, the line. When the stage is a program's
  // whole body the driver runs a batch of rows in one call (driver.cpp, run_flat).
  leaf_fn leaf = nullptr;
  std::string name; // for describe()
};

enum class loop_kind : std::uint8_t {
  batch, // whole transforms
  lines, // the lines of one axis within a transform
};

struct loop {
  loop_kind kind = loop_kind::batch;
  std::size_t count = 1;
  std::array<std::ptrdiff_t, moving_slots> step{}; // bytes per iteration: in, out, block, tw
  // Bytes added once, before the first iteration: a lines loop that starts part-way along
  // an axis (the last, narrower blocks of a lanes column phase, plan_core.cpp). Batch loops
  // never have one (transform_addresses ignores it).
  std::array<std::ptrdiff_t, moving_slots> start{};
  std::uint32_t body_end = 0; // records [this + 1, body_end) are the body
  // A lines loop opening a nest whose lines are disjoint from those of the nest just before
  // it, of the same axis (the narrower last block of a lanes column phase): the work graph
  // shares both nests' lines as one phase instead of running this one after (X41).
  bool joins = false;
  // A lines loop opening a nest whose neighbouring lines write the same cache lines (the
  // columns of a plane whose rows or blocks are not whole cache lines), on a profile that
  // takes fair runs (machine_profile::fair_line_runs): a team claims a phase
  // whose shares are small in runs of a fair share (chunk_run), so that few runs border
  // another member's (X59).
  bool shares_lines = false;
};

using record = std::variant<stage, loop>;

// The first unit of chunk c of a range of `units` units cut into `chunks` chunks:
// units c / chunks in exact integers. Serial execution and a split across threads both
// cut a stage or a loop nest with it, so a chunk covers the same units whoever runs it.
[[nodiscard]] constexpr std::size_t chunk_start(std::size_t units, std::size_t chunks,
                                                std::size_t c) noexcept {
  return units / chunks * c + units % chunks * c / chunks;
}

// A stage of a flat program as the driver runs one transform of it (driver.cpp,
// run_flat_one): the stage's kernel, units, slots and a copy of its arguments, contiguous,
// so that one transform reads neither the records' variants nor their names (X28).
struct flat_stage {
  pass_fn run = nullptr;
  leaf_fn leaf = nullptr;
  std::size_t units = 0;
  slot src = slot::in;
  slot dst = slot::out;
  pass_args args;
};

// One kernel call of a walked program, unrolled for one transform (driver.cpp, mark_flat;
// experiment X41): a stage of one line, or a loop of leaves as one leaf call over its lines.
// The slots in, out and block are offset by the bytes the loops would have moved them;
// scratch never moves.
struct call {
  pass_fn run = nullptr;  // a stage's kernel over units [0, units)
  leaf_fn leaf = nullptr; // or a leaf over `units` lines, src_step and dst_step bytes apart
  std::size_t units = 0;
  std::ptrdiff_t src_off = 0; // bytes from the transform's address of the src slot
  std::ptrdiff_t dst_off = 0;
  std::ptrdiff_t src_step = 0;
  std::ptrdiff_t dst_step = 0;
  std::uint32_t args = 0; // index into program::call_args
  slot src = slot::in;
  slot dst = slot::out;
};

// The most calls per transform a program is unrolled into (mark_flat): beyond it the
// records are walked.
inline constexpr std::size_t calls_max = 256;

struct program {
  std::vector<record> records;
  std::size_t batch_loops = 0; // the leading records that are the batch loops
  std::size_t transforms = 1;  // product of the batch loops' counts
  // The lanes strategy (DESIGN.md 8.5, strategies/lanes.cpp): when `lanes` is not 0 the
  // body (the records after the batch loops) runs `lanes` consecutive transforms of the
  // innermost batch loop at once, one per vector lane, reading them through that loop's
  // steps; the transforms left over (fewer than `lanes` in a run of the innermost loop) run
  // through `lane_tail`, the same body for blocks of `lane_unit` (C) transforms, the last
  // one padded. The driver sets pass_args::rows to each block's transforms (driver.cpp). The
  // body may hold loops: the lines of the axes of a multi-axis transform whose transforms
  // are adjacent (plan_core.cpp; their steps and body_end are within records and lane_tail).
  std::size_t lanes = 0;
  std::size_t lane_unit = 0;
  std::vector<record> lane_tail;
  // The body (the records after the batch loops) is at most flat_stages_max stages, none
  // with moving twiddles, and no loop: the driver then prepares each stage's arguments once
  // per call and only moves the slots per transform (driver.cpp, run_flat). Set by
  // mark_flat (driver.hpp).
  bool flat = false;
  std::vector<flat_stage> flat_body; // the body's stages when flat, in order (mark_flat)
  // Flat, one batch loop, and the body one leaf between the caller's arrays: the whole
  // batch is one call of the leaf (driver.hpp, run_leaf_rows), made from the copies below
  // of the stage's leaf and arguments and of the batch loop's steps, so that the entry
  // reaches the leaf without walking the records (X28). Set by mark_flat.
  bool leaf_rows = false;
  bool rows_from_in = true; // the leaf reads the input (else the output, in place)
  leaf_fn rows_leaf = nullptr;
  std::ptrdiff_t rows_src_step = 0; // bytes between the lines the leaf reads
  std::ptrdiff_t rows_dst_step = 0; // and writes
  pass_args rows_args;
  // A multi-axis program (a loop over lines in its body), neither flat nor lanes, every stage
  // without moving twiddles, and at most calls_max kernel
  // calls per transform: the body unrolled into its calls, in serial order, and the
  // arguments they use, one copy per stage record (mark_flat). The driver then runs a
  // transform by calling them in turn, without walking the records (X41); empty otherwise,
  // or when the plan's build settings turn it off (build_settings::unroll_calls).
  std::vector<call> calls;
  std::vector<pass_args> call_args;
};

// The longest body run_flat handles; its arguments live on the stack.
inline constexpr std::size_t flat_stages_max = 8;

} // namespace fftn::inline v0::detail
