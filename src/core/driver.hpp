// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The program interpreter (DESIGN.md 8.3): serial, on the calling thread. It allocates
// nothing, takes no locks and cannot fail.
#pragma once

#include "core/program.hpp"

#include <cstddef>

namespace fftn::inline v0::detail {

// Where the slots are for one execution: in, out and block are moved by loops; a and b are
// scratch lines used from their start by every line.
struct slot_addresses {
  const std::byte* in = nullptr;
  std::byte* out = nullptr;
  std::byte* block = nullptr;
  std::byte* a = nullptr;
  std::byte* b = nullptr;
  std::ptrdiff_t tw = 0; // the twiddle cursor (program.hpp), in bytes
};

// Sets p.flat and the fast paths' marks from p's records (program.hpp); called once the
// program is complete.
void mark_flat(program& p) noexcept;

// Runs transforms [first, first + count) of p's batch, every chunk of every stage in order.
// in and out address element 0 of transform 0.
void run_transforms(const program& p, std::size_t first, std::size_t count,
                    const slot_addresses& base) noexcept;

// Transforms [first, first + count) of a program marked leaf_rows (program.hpp): one call of
// its leaf. in and out address element 0 of transform 0. Takes the two addresses rather
// than slot_addresses: on x86 reading a slot_addresses just written by the caller costs a
// store-forwarding stall of about a dozen cycles per call (X20).
void run_leaf_rows(const program& p, std::size_t first, std::size_t count, const void* in,
                   void* out) noexcept;

// Transforms [first, first + count) of a program marked lane_leaf_rows (program.hpp): one
// call of its lanes leaf, which reads and writes the rows themselves (no scratch). in and
// out address element 0 of transform 0; taken as two addresses for run_leaf_rows' reason.
void run_lane_leaf_rows(const program& p, std::size_t first, std::size_t count, const void* in,
                        void* out) noexcept;

// The pieces a team shares out (src/threading/work_graph.hpp), each run exactly as
// run_transforms runs it.

// Where transform t's slots are: base, with in and out moved by the batch loops.
[[nodiscard]] slot_addresses transform_addresses(const program& p, std::size_t t,
                                                 const slot_addresses& base) noexcept;

// Runs chunks [first, last) of stage st.
void run_stage_chunks(const stage& st, const slot_addresses& at, std::size_t first,
                      std::size_t last) noexcept;

// Runs lines [first, last) of the perfectly nested loops [begin, begin + nest) of p, whose
// innermost body ends at body_end; the innermost loop varies fastest, as in serial order.
void run_lines(const program& p, std::size_t begin, std::size_t nest, std::size_t body_end,
               std::size_t first, std::size_t last, const slot_addresses& at) noexcept;

} // namespace fftn::inline v0::detail
