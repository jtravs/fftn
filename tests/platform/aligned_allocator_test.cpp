// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/aligned_allocator.hpp"
#include "platform/aligned_buffer.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace {

using fftn::detail::aligned_vector;
using fftn::detail::default_alignment;
using fftn::detail::plan_tables;

template <class T> void check_tables() {
  plan_tables<T> tables;
  std::vector<const T*> starts;
  for (std::size_t count = 1; count < 300; count += 7) {
    aligned_vector<T>& t = tables.emplace_back(count, T(1));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(t.data()) % default_alignment, 0U)
        << "count " << count;
    starts.push_back(t.data());
  }
  // Appending moves the outer vector, never an inner buffer (stages point into them).
  for (std::size_t i = 0; i < starts.size(); ++i) {
    EXPECT_EQ(tables[i].data(), starts[i]);
    EXPECT_EQ(tables[i].back(), T(1));
  }
  aligned_vector<T> grown;
  for (std::size_t i = 0; i < 1000; ++i) {
    grown.push_back(static_cast<T>(i));
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(grown.data()) % default_alignment, 0U);
  }
  EXPECT_EQ(grown[999], T(999));
}

TEST(AlignedAllocator, TablesStartOnACacheLine) {
  check_tables<double>();
  check_tables<float>();
}

// A count whose bytes do not fit in size_t is refused before any allocation, as
// std::allocator refuses it; the largest count that fits is only too large for memory.
template <class T> void check_too_many() {
  fftn::detail::line_aligned_allocator<T> a;
  const std::size_t most = std::numeric_limits<std::size_t>::max() / sizeof(T);
  EXPECT_THROW(static_cast<void>(a.allocate(most + 1)), std::bad_array_new_length);
  EXPECT_THROW(static_cast<void>(a.allocate(std::numeric_limits<std::size_t>::max())),
               std::bad_array_new_length);
  // Copies for other element types compare equal: any instance frees any other's memory.
  const fftn::detail::line_aligned_allocator<char> other(a);
  EXPECT_TRUE(a == other);
  T* p = a.allocate(3);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % default_alignment, 0U);
  fftn::detail::line_aligned_allocator<T>(other).deallocate(p, 3);
}

TEST(AlignedAllocator, RefusesACountWhoseBytesOverflow) {
  check_too_many<double>();
  check_too_many<float>();
}

} // namespace
