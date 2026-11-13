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
};

// Runs transforms [first, first + count) of p's batch, every chunk of every stage in order.
// in and out address element 0 of transform 0.
void run_transforms(const program& p, std::size_t first, std::size_t count,
                    const slot_addresses& base) noexcept;

} // namespace fftn::inline v0::detail
