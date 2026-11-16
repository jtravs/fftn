// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// A standard allocator whose blocks start on a cache-line boundary, and the vector of a
// plan's tables built on it. Allocation happens only at plan construction.
#pragma once

#include "platform/aligned_buffer.hpp"

#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace fftn::inline v0::detail {

template <class T> struct line_aligned_allocator {
  using value_type = T;

  line_aligned_allocator() noexcept = default;
  template <class U>
  // NOLINTNEXTLINE(misc-explicit-constructor): rebinding, as the allocator requirements ask
  line_aligned_allocator(const line_aligned_allocator<U>& /*other*/) noexcept {}

  [[nodiscard]] T* allocate(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    return static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t{default_alignment}));
  }
  void deallocate(T* p, std::size_t /*count*/) noexcept {
    ::operator delete(p, std::align_val_t{default_alignment});
  }

  template <class U> bool operator==(const line_aligned_allocator<U>& /*other*/) const noexcept {
    return true;
  }
};

// A table's storage: its first element on a 64-byte boundary. Twiddle tables start where
// std::vector put them before, 16 bytes into a line or so, and then half of a pass's
// 32-byte twiddle loads crossed a line; aligned, the same plans took 0.97-0.98 of the
// cycles at 1024-4096 points on the EPYC (dev/experiments/X30-x86-pass-loops.md, counter
// diagnostics; X30b measures it).
template <class T> using aligned_vector = std::vector<T, line_aligned_allocator<T>>;

// A plan's tables. Stages point into the inner buffers, which stay where they are when
// the outer vector moves.
template <class T> using plan_tables = std::vector<aligned_vector<T>>;

} // namespace fftn::inline v0::detail
