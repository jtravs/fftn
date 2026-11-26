// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fftn::easy::view from std::mdspan (C++23): the extents and strides of layout_right,
// layout_left and layout_stride arrays are taken as given, so a call through an mdspan
// computes what the call through the equivalent view computes. Compiled as C++23; skipped where the
// standard library has no std::mdspan.

#include <fftn/easy.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <vector>
#include <version>

#if defined(__cpp_lib_mdspan)
#include <array>
#include <mdspan>
#endif

namespace {

namespace easy = fftn::easy;
using cd = std::complex<double>;

TEST(easy_mdspan, views_convert_from_mdspan) {
#if !defined(__cpp_lib_mdspan)
  GTEST_SKIP() << "the standard library has no std::mdspan";
#else
  std::vector<cd> x(6 * 8);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {static_cast<double>(i % 7), static_cast<double>(i % 3)};
  }
  std::vector<cd> want(x.size());
  std::vector<cd> got(x.size());
  easy::fftn(easy::view<const cd, 2>(x.data(), {6, 8}), easy::view<cd, 2>(want.data(), {6, 8}));

  // layout_right, through the deduction guide.
  const std::mdspan<const cd, std::dextents<std::size_t, 2>> in(x.data(), 6, 8);
  const std::mdspan<cd, std::dextents<std::size_t, 2>> out(got.data(), 6, 8);
  const easy::view vi(in);
  EXPECT_EQ(vi.extents, (std::array<std::size_t, 2>{6, 8}));
  EXPECT_EQ(vi.strides, (std::array<std::ptrdiff_t, 2>{8, 1}));
  easy::fftn(vi, easy::view(out));
  EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size() * sizeof(cd)), 0);

  // layout_left over the transposed array: the same elements, the axes swapped.
  const std::mdspan<const cd, std::dextents<std::size_t, 2>, std::layout_left> t(x.data(), 8, 6);
  const easy::view<const cd, 2> vt = t;
  EXPECT_EQ(vt.strides, (std::array<std::ptrdiff_t, 2>{1, 8}));
  std::vector<cd> got_t(x.size());
  easy::fftn(vt, easy::view<cd, 2>(got_t.data(), {8, 6}, {1, 8}));
  // Another order of the axes is another plan, so equal within rounding rather than bitwise.
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_LT(std::abs(got_t[i] - want[i]), 1e-12) << i;
  }

  // layout_stride: every other row.
  using strided = std::layout_stride::mapping<std::dextents<std::size_t, 2>>;
  const std::mdspan<const cd, std::dextents<std::size_t, 2>, std::layout_stride> rows(
      x.data(), strided(std::dextents<std::size_t, 2>(3, 8), std::array<std::size_t, 2>{16, 1}));
  const easy::view<const cd, 2> vr = rows;
  EXPECT_EQ(vr.strides, (std::array<std::ptrdiff_t, 2>{16, 1}));
  std::vector<cd> tight(3 * 8);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      tight[i * 8 + j] = x[i * 16 + j];
    }
  }
  std::vector<cd> got_r(tight.size());
  std::vector<cd> want_r(tight.size());
  easy::fftn(easy::view(rows), easy::view<cd, 2>(got_r.data(), {3, 8}));
  easy::fftn(easy::view<const cd, 2>(tight.data(), {3, 8}),
             easy::view<cd, 2>(want_r.data(), {3, 8}));
  for (std::size_t i = 0; i < tight.size(); ++i) {
    EXPECT_LT(std::abs(got_r[i] - want_r[i]), 1e-12) << i;
  }

  // An empty layout_right array: extents {3, 0} give strides {0, 1}; looping over dimension 0
  // of an empty array does nothing, and is not refused for its stride of 0.
  const std::mdspan<const cd, std::dextents<std::size_t, 2>> empty(x.data(), 3, 0);
  std::vector<cd> untouched(1, cd{4});
  const std::mdspan<cd, std::dextents<std::size_t, 2>> empty_out(untouched.data(), 3, 0);
  easy::fftn(easy::view(empty), easy::view(empty_out), {0});
  EXPECT_EQ(untouched[0], cd{4});
#endif
}

} // namespace
