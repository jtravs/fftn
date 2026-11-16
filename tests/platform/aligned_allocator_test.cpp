// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/aligned_allocator.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
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

} // namespace
