// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/aligned_buffer.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace {

using fftn::detail::aligned_buffer;

template <class T> void check_sizes_and_alignments() {
  for (std::size_t alignment = alignof(T); alignment <= 4096; alignment *= 2) {
    for (const std::size_t count : {1U, 2U, 3U, 7U, 8U, 63U, 64U, 65U, 1000U, 4097U}) {
      aligned_buffer<T> b(count, alignment);
      ASSERT_NE(b.data(), nullptr);
      EXPECT_EQ(b.size(), count);
      EXPECT_FALSE(b.empty());
      EXPECT_EQ(b.alignment(), alignment);
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b.data()) % alignment, 0U)
          << "count " << count << " alignment " << alignment;
      EXPECT_EQ(b.span().size(), count);
      // Every element is writable and keeps its value (ASan checks the bounds).
      for (std::size_t i = 0; i < count; ++i) {
        b.data()[i] = static_cast<T>(static_cast<double>(i));
      }
      const aligned_buffer<T>& view = b;
      EXPECT_EQ(view.data()[count - 1], static_cast<T>(static_cast<double>(count - 1)));
      EXPECT_EQ(view.span().data(), view.data());
    }
  }
}

TEST(AlignedBuffer, AlignsEverySizeDouble) { check_sizes_and_alignments<double>(); }
TEST(AlignedBuffer, AlignsEverySizeFloat) { check_sizes_and_alignments<float>(); }
TEST(AlignedBuffer, AlignsEverySizeComplex) { check_sizes_and_alignments<std::complex<double>>(); }

TEST(AlignedBuffer, DefaultsTo64Bytes) {
  const aligned_buffer<double> b(5);
  EXPECT_EQ(b.alignment(), 64U);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b.data()) % 64, 0U);
}

TEST(AlignedBuffer, EmptyBuffersOwnNothing) {
  const aligned_buffer<double> none;
  EXPECT_EQ(none.data(), nullptr);
  EXPECT_EQ(none.size(), 0U);
  EXPECT_TRUE(none.empty());
  const aligned_buffer<double> zero(0, 128);
  EXPECT_EQ(zero.data(), nullptr);
  EXPECT_TRUE(zero.span().empty());
}

TEST(AlignedBuffer, MovesOwnership) {
  aligned_buffer<float> a(100, 256);
  float* const p = a.data();
  aligned_buffer<float> b(std::move(a));
  EXPECT_EQ(b.data(), p);
  EXPECT_EQ(b.size(), 100U);
  EXPECT_EQ(b.alignment(), 256U);
  // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move): the contract tested
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.size(), 0U);
  // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  aligned_buffer<float> c(3);
  c = std::move(b);
  EXPECT_EQ(c.data(), p);
  EXPECT_EQ(c.size(), 100U);
  EXPECT_EQ(c.alignment(), 256U);
}

TEST(AlignedBuffer, RejectsBadAlignments) {
  for (const std::size_t bad : {0U, 3U, 4U, 24U, 100U}) {
    EXPECT_THROW(aligned_buffer<double>(1, bad), std::invalid_argument) << bad;
  }
}

TEST(AlignedBuffer, RejectsOverflowingSizes) {
  EXPECT_THROW(aligned_buffer<double>(std::numeric_limits<std::size_t>::max() / 4),
               std::bad_array_new_length);
}

} // namespace
