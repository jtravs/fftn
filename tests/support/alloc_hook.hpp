// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Counting the global allocation functions, so that a test can assert that a call allocated
// nothing (the execute contract, AGENTS.md section 4).
//
// alloc_hook.cpp replaces every form of the global operator new and operator delete
// (array, nothrow, aligned and sized forms included) with versions that count and forward
// to malloc and free (aligned_alloc, or _aligned_malloc on Windows, for the aligned forms).
// Link the fftn_test_alloc_hook object library into an executable to install them; the
// counters are process-wide atomics, so allocations made on any thread are seen. Under
// AddressSanitizer and ThreadSanitizer the replacements still hold, and malloc and free
// below them are the sanitizers' own, so their checks continue to work.
//
// Calls straight to malloc or to a platform's aligned allocator are not counted: an
// allocator in FFTN that bypasses operator new needs its own counter.

#pragma once

#include <cstddef>

namespace fftn::test {

// Counts of allocation and deallocation calls since construction, from all threads.
class allocation_counter {
public:
  allocation_counter();
  [[nodiscard]] std::size_t allocations() const;
  [[nodiscard]] std::size_t deallocations() const;

private:
  std::size_t new_at_start_;
  std::size_t delete_at_start_;
};

} // namespace fftn::test
