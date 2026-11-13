// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "support/canary.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fftn::test {
namespace {

std::uintptr_t address(const std::byte* p) { return reinterpret_cast<std::uintptr_t>(p); }

TEST(GuardedBuffer, PlacesTheRegionAtTheRequestedOffsetAgainstTheGuard) {
  const std::size_t page = guarded_buffer::page_size();
  for (const std::size_t bytes :
       {std::size_t{0}, std::size_t{1}, std::size_t{24}, page - 8, page, 3 * page + 40}) {
    for (const std::size_t alignment : {std::size_t{8}, std::size_t{64}, std::size_t{4096}}) {
      for (const std::size_t offset : {std::size_t{0}, alignment / 2, alignment - 8}) {
        for (const guard_side side : {guard_side::above, guard_side::below}) {
          const guarded_buffer b(bytes, alignment, offset, side);
          ASSERT_NE(b.data(), nullptr);
          EXPECT_EQ(b.size(), bytes);
          EXPECT_EQ(address(b.data()) % alignment, offset);
          if (side == guard_side::above) {
            // The guard page starts within one alignment step of the region's end.
            const std::uintptr_t end = address(b.data()) + bytes;
            EXPECT_LT((page - end % page) % page, alignment);
          } else {
            EXPECT_EQ(address(b.data()) % page, offset);
          }
          EXPECT_TRUE(b.surroundings_intact());
        }
      }
    }
  }
}

TEST(GuardedBuffer, RejectsBadAlignment) {
  EXPECT_THROW(guarded_buffer(16, 48, 0), std::invalid_argument);
  EXPECT_THROW(guarded_buffer(16, 64, 64), std::invalid_argument);
  EXPECT_THROW(guarded_buffer(16, 2 * guarded_buffer::page_size(), 0), std::invalid_argument);
}

TEST(GuardedBuffer, RegionIsUsableAndSlackIsWatched) {
  const guarded_buffer b(100, 64, 8);
  for (std::size_t i = 0; i < b.size(); ++i) {
    b.data()[i] = std::byte{0x11};
  }
  EXPECT_TRUE(b.surroundings_intact());
  // The 100-byte region at offset 8 leaves slack below the guard page: a write there is
  // not a fault but is detected.
  b.data()[b.size()] = std::byte{0x11};
  EXPECT_FALSE(b.surroundings_intact());
}

TEST(GuardedBuffer, MovesOwnership) {
  guarded_buffer a(32, 64, 0);
  std::byte* p = a.data();
  guarded_buffer b(std::move(a));
  EXPECT_EQ(b.data(), p);
  guarded_buffer c(8, 8, 0);
  c = std::move(b);
  EXPECT_EQ(c.data(), p);
  c.data()[31] = std::byte{1};
}

// Writing one byte past the end, or reading one before the start, faults.
TEST(GuardedBufferDeathTest, OverrunFaults) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        const guarded_buffer b(64, 64, 0);
        volatile std::byte* p = b.data();
        p[64] = std::byte{1};
      },
      "");
}

TEST(GuardedBufferDeathTest, UnderrunFaults) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        const guarded_buffer b(64, 64, 0, guard_side::below);
        volatile const std::byte* p = b.data();
        static_cast<void>(*(p - 1));
      },
      "");
}

TEST(CanaryBytes, DetectsWritesOutsideTheDescribedBytes) {
  std::vector<std::byte> buf(64);
  canary_bytes c(buf);
  c.allow(0, 8);
  c.allow(16, 8);
  c.arm();
  EXPECT_EQ(c.violations(), 0U);
  EXPECT_EQ(c.first_violation(), buf.size());
  buf[0] = std::byte{0};
  buf[23] = std::byte{0};
  EXPECT_EQ(c.violations(), 0U);
  buf[8] = ~buf[8];
  buf[40] = ~buf[40];
  EXPECT_EQ(c.violations(), 2U);
  EXPECT_EQ(c.first_violation(), 8U);
  EXPECT_THROW(c.allow(60, 8), std::out_of_range);
}

TEST(CanaryBytes, PatternVariesWithOffset) {
  for (std::size_t i = 0; i < 255; ++i) {
    EXPECT_NE(canary_byte(i), canary_byte(i + 1));
  }
}

} // namespace
} // namespace fftn::test
