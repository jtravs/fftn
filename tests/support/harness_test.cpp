// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The harness and its contract checks, with hand-written calls that copy, misbehave or
// forget to write.

#include "support/canary.hpp"
#include "support/harness.hpp"
#include "support/input_snapshot.hpp"
#include "support/transform_under_test.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace fftn::test {
namespace {

using cd = std::complex<double>;

TEST(InputSnapshot, SeesAnyByteChange) {
  std::vector<double> v{1.0, -0.0, 3.0};
  const input_snapshot s(std::as_bytes(std::span(v)));
  EXPECT_TRUE(s.unchanged());
  EXPECT_EQ(s.first_change(), v.size() * sizeof(double));
  v[1] = 0.0; // same value, different bytes
  EXPECT_FALSE(s.unchanged());
  EXPECT_EQ(s.first_change(), 15U); // the sign byte of v[1] (little-endian)
}

TEST(Offsets, RowsAndPlanes) {
  EXPECT_EQ(row_offsets(3, 2, 2, 7), (std::vector<std::ptrdiff_t>{0, 2, 4, 7, 9, 11}));
  EXPECT_EQ(row_offsets(2, 2, -1, -3), (std::vector<std::ptrdiff_t>{0, -1, -3, -4}));
  EXPECT_EQ(plane_offsets(2, 2, 1, 5, 1, 0), (std::vector<std::ptrdiff_t>{0, 1, 5, 6}));
}

// A call that copies a strided row to a contiguous one, correctly or not.
struct copier {
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t out_stride = 1;
  std::size_t n = 0;
  int misbehave = 0; // 1: writes a gap; 2: changes the input; 3: skips an output

  void operator()(const cd* in, cd* out) const {
    for (std::size_t i = 0; i < n; ++i) {
      if (misbehave == 3 && i == 1) {
        continue;
      }
      out[static_cast<std::ptrdiff_t>(i) * out_stride] =
          in[static_cast<std::ptrdiff_t>(i) * in_stride];
    }
    if (misbehave == 1) {
      out[1] = cd{};
    }
    if (misbehave == 2) {
      const_cast<cd*>(in)[0] = cd{9.0, 9.0};
    }
  }
};

TEST(Execute, CopiesAndReportsNothingWhenWellBehaved) {
  const std::vector<cd> x{{1, 2}, {3, 4}, {5, 6}};
  for (const std::size_t offset : {std::size_t{0}, std::size_t{8}, std::size_t{56}}) {
    for (const guard_side side : {guard_side::above, guard_side::below}) {
      const auto in = row_offsets(3, 1, -2, 0);
      const auto out = row_offsets(3, 1, 3, 0);
      const auto r =
          execute<cd, cd>(copier{.in_stride = -2, .out_stride = 3, .n = 3, .misbehave = 0}, in, out,
                          placement::out_of_place, std::span<const cd>(x),
                          {.in_offset = offset, .out_offset = offset, .guard = side});
      EXPECT_TRUE(r.violations.empty());
      EXPECT_EQ(r.out, x);
    }
  }
}

TEST(Execute, PlacesElementZeroAtTheRequestedOffset) {
  const std::vector<cd> x(4);
  const auto offsets = row_offsets(4, 1, -1, 0);
  const auto check = [](const cd* in, cd* out) {
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(in) % 64, 24U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(out) % 64, 40U);
  };
  const auto r =
      execute<cd, cd>(check, offsets, offsets, placement::out_of_place, std::span<const cd>(x),
                      {.in_offset = 24, .out_offset = 40, .guard = guard_side::above});
  EXPECT_EQ(r.violations.size(), 0U);
  EXPECT_THROW((execute<cd, cd>(check, offsets, offsets, placement::out_of_place,
                                std::span<const cd>(x), {4, 0, guard_side::above})),
               std::invalid_argument);
}

TEST(Execute, ReportsGapWritesInputChangesAndMissingOutputs) {
  const std::vector<cd> x{{1, 2}, {3, 4}, {5, 6}};
  const auto in = row_offsets(3, 1, 1, 0);
  const auto out = row_offsets(3, 1, 2, 0);
  const auto gap = execute<cd, cd>(copier{.in_stride = 1, .out_stride = 2, .n = 3, .misbehave = 1},
                                   in, out, placement::out_of_place, std::span<const cd>(x));
  ASSERT_EQ(gap.violations.size(), 1U);
  EXPECT_NE(gap.violations[0].find("outside the output rows"), std::string::npos);
  const auto changed =
      execute<cd, cd>(copier{.in_stride = 1, .out_stride = 2, .n = 3, .misbehave = 2}, in, out,
                      placement::out_of_place, std::span<const cd>(x));
  ASSERT_EQ(changed.violations.size(), 1U);
  EXPECT_NE(changed.violations[0].find("input buffer changed"), std::string::npos);
  const auto missing =
      execute<cd, cd>(copier{.in_stride = 1, .out_stride = 2, .n = 3, .misbehave = 3}, in, out,
                      placement::out_of_place, std::span<const cd>(x));
  EXPECT_TRUE(missing.violations.empty());
  EXPECT_TRUE(std::isnan(missing.out[1].real())); // so any comparison fails
}

TEST(Execute, InPlaceSharesOneBuffer) {
  const std::vector<double> x{1, 2, 3, 4};
  // A real row of 4 in place of a complex row of 3 (FFTW's padded layout).
  const auto in = row_offsets(4, 1, 1, 0);
  const auto out = row_offsets(3, 1, 1, 0);
  const auto call = [](const double* i, cd* o) {
    EXPECT_EQ(static_cast<const void*>(i), static_cast<const void*>(o));
    const double a = i[0] + i[1];
    const double b = i[2] + i[3];
    o[0] = {a, 0};
    o[1] = {b, 0};
    o[2] = {a - b, 1};
  };
  const auto r =
      execute<double, cd>(call, in, out, placement::in_place, std::span<const double>(x));
  EXPECT_TRUE(r.violations.empty());
  EXPECT_EQ(r.out, (std::vector<cd>{{3, 0}, {7, 0}, {-4, 1}}));
  // Writing past the described rows in place is reported too.
  const auto over = [](const double* /*i*/, cd* o) { reinterpret_cast<double*>(o)[6] = 0.0; };
  const auto in2 = row_offsets(4, 1, 1, 0);
  const auto out2 = row_offsets(2, 1, 1, 0);
  const auto bad =
      execute<double, cd>(over, in2, out2, placement::in_place, std::span<const double>(x),
                          {.in_offset = 0, .out_offset = 0, .guard = guard_side::below});
  EXPECT_EQ(bad.violations.size(), 1U);
}

} // namespace
} // namespace fftn::test
