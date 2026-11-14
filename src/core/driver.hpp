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

// Runs transforms [first, first + count) of p's batch, every chunk of every stage in order.
// in and out address element 0 of transform 0.
void run_transforms(const program& p, std::size_t first, std::size_t count,
                    const slot_addresses& base) noexcept;

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
