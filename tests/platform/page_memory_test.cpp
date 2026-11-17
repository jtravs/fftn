// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Page memory for measured planning's scratch (src/platform/page_memory.hpp): zero-filled
// regions of either kind, aligned for a huge page where the kinds can differ, moved without
// copying, and reserving what reserved_for says.

#include "platform/page_memory.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

using fftn::detail::page_kind;
using fftn::detail::page_kinds_differ;
using fftn::detail::page_region;

TEST(PageMemory, RegionsAreZeroAlignedAndSized) {
  for (const page_kind kind : {page_kind::small, page_kind::huge}) {
    for (const std::size_t bytes : {std::size_t{0}, std::size_t{100}, std::size_t{3} << 20U}) {
      const page_region r(bytes, kind);
      EXPECT_EQ(r.size(), bytes);
      EXPECT_EQ(r.reserved(), page_region::reserved_for(bytes));
      EXPECT_GE(r.reserved(), bytes);
      const auto at = reinterpret_cast<std::uintptr_t>(r.data());
      EXPECT_EQ(at % (page_kinds_differ() ? (std::size_t{2} << 20U) : 64), 0U);
      for (std::size_t i = 0; i < bytes; i += 4093) {
        EXPECT_EQ(r.data()[i], std::byte{0});
      }
      if (bytes != 0) {
        r.data()[bytes - 1] = std::byte{1}; // writable to the end
      }
    }
  }
}

TEST(PageMemory, MovingKeepsTheMemory) {
  page_region a(std::size_t{1} << 16U, page_kind::huge);
  std::byte* p = a.data();
  page_region b(std::move(a));
  EXPECT_EQ(b.data(), p);
  page_region c;
  c = std::move(b);
  EXPECT_EQ(c.data(), p);
  EXPECT_EQ(c.size(), std::size_t{1} << 16U);
}

TEST(PageMemory, KindsDifferOnlyOnLinux) {
#ifndef __linux__
  EXPECT_FALSE(page_kinds_differ());
#endif
  EXPECT_EQ(page_kinds_differ(), page_kinds_differ()); // read once, stable
}

} // namespace
