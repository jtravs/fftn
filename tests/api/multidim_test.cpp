// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Transforms of rank 2 and 3 beyond the c2c planes of the oracle suites: real planes and
// volumes (r2c, c2r) and complex volumes, in and out of place, against the separable
// quadruple-precision reference. The expected values are rounded to double between axes,
// which adds about one double epsilon per axis to their error; the tolerance carries it.

#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace {

using cd = std::complex<double>;
using dims_t = std::vector<std::size_t>;

std::size_t product(const dims_t& d, std::size_t from = 0) {
  std::size_t p = 1;
  for (std::size_t i = from; i < d.size(); ++i) {
    p *= d[i];
  }
  return p;
}

// The unnormalised DFT of one C-ordered array, axis by axis.
std::vector<cd> dft(int sign, const dims_t& dims, std::vector<cd> x) {
  for (std::size_t a = 0; a < dims.size(); ++a) {
    const std::size_t n = dims[a];
    const std::size_t inner = product(dims, a + 1);
    const std::size_t outer = x.size() / (n * inner);
    std::vector<cd> line(n);
    for (std::size_t o = 0; o < outer; ++o) {
      for (std::size_t i = 0; i < inner; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          line[j] = x[(o * n + j) * inner + i];
        }
        const auto y = fftn::test::reference::c2c(sign, line);
        for (std::size_t k = 0; k < n; ++k) {
          x[(o * n + k) * inner + i] = {y[k].re.hi, y[k].im.hi};
        }
      }
    }
  }
  return x;
}

template <class T> fftn::test::tolerance tolerance_for(const dims_t& dims) {
  const double eps_double = std::numeric_limits<double>::epsilon();
  return fftn::test::derived_tolerance<T>(product(dims), fftn::test::stable_fft_bound) * 2 +
         fftn::test::input_rounding<T>() +
         fftn::test::tolerance{.rms = 3 * eps_double, .max = 3 * eps_double};
}

template <class T> std::vector<fftn::test::exact_complex> expected(std::span<const cd> v) {
  std::vector<fftn::test::exact_complex> e;
  for (const cd& z : v) {
    e.push_back(fftn::test::exact_from(z));
  }
  return e;
}

template <class T>
void expect_within(std::span<const std::complex<T>> got, std::span<const cd> want,
                   const dims_t& dims, const char* what) {
  fftn::test::outcome o;
  o.within(fftn::test::compare<T>(got, expected<T>(want)), tolerance_for<T>(dims), what);
  EXPECT_TRUE(o.passed()) << o.detail();
}

constexpr std::size_t count = 2;

template <class T> fftn::small_vector<fftn::axis, 3> axes_of(const dims_t& dims) {
  fftn::small_vector<fftn::axis, 3> a;
  for (const std::size_t n : dims) {
    a.push_back({.n = n});
  }
  return a;
}

const std::vector<dims_t>& shapes() {
  static const std::vector<dims_t> v{{8, 6}, {5, 7},    {16, 16},  {32, 30},  {1, 9},    {9, 1},
                                     {6, 1}, {4, 6, 8}, {3, 5, 4}, {2, 3, 5}, {1, 4, 1}, {7, 1, 3}};
  return v;
}

template <class T> class MultiDim : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(MultiDim, precisions);

TYPED_TEST(MultiDim, ComplexVolumesAndPlanes) {
  using T = TypeParam;
  using c = std::complex<T>;
  for (const dims_t& dims : shapes()) {
    const std::size_t points = product(dims);
    fftn::test::input_source src(points);
    const auto x = src.complexes<T>(points * count);
    for (const fftn::direction dir : {fftn::direction::forward, fftn::direction::backward}) {
      std::vector<cd> want;
      for (std::size_t t = 0; t < count; ++t) {
        const std::vector<cd> one(x.begin() + static_cast<std::ptrdiff_t>(t * points),
                                  x.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
        const auto y = dft(static_cast<int>(dir), dims, one);
        want.insert(want.end(), y.begin(), y.end());
      }
      fftn::shape<c, c> s{.dir = dir, .axes = axes_of<T>(dims), .batches = {{.count = count}}};
      for (const bool in_place : {false, true}) {
        const fftn::c2c_plan<T> p(s, {.in_place = in_place});
        fftn::workspace ws(p);
        std::vector<c> y(x.size());
        if (in_place) {
          y = x;
          p.execute(y.data(), y.data(), ws);
        } else {
          p.execute(x.data(), y.data(), ws);
        }
        expect_within<T>(y, want, dims, in_place ? "c2c in place" : "c2c");
      }
    }
  }
}

TYPED_TEST(MultiDim, RealToComplex) {
  using T = TypeParam;
  using c = std::complex<T>;
  for (const dims_t& dims : shapes()) {
    const std::size_t n = dims.back();
    const std::size_t h = n / 2 + 1;
    const std::size_t points = product(dims);
    const std::size_t lines = points / n;
    fftn::test::input_source src(points + 1);
    const auto x = src.reals<T>(points * count);
    std::vector<cd> want;
    for (std::size_t t = 0; t < count; ++t) {
      const std::vector<cd> one(x.begin() + static_cast<std::ptrdiff_t>(t * points),
                                x.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
      const auto y = dft(-1, dims, one);
      for (std::size_t l = 0; l < lines; ++l) {
        want.insert(want.end(), y.begin() + static_cast<std::ptrdiff_t>(l * n),
                    y.begin() + static_cast<std::ptrdiff_t>(l * n + h));
      }
    }
    const fftn::shape<T, c> s{.axes = axes_of<T>(dims), .batches = {{.count = count}}};
    {
      const fftn::r2c_plan<T> p(s);
      fftn::workspace ws(p);
      std::vector<c> y(lines * h * count);
      p.execute(x.data(), y.data(), ws);
      expect_within<T>(y, want, dims, "r2c");
    }
    {
      // In place: each real row padded to the 2h reals of its complex row.
      const fftn::r2c_plan<T> p(s, {.in_place = true});
      fftn::workspace ws(p);
      std::vector<c> y(lines * h * count);
      auto* real = reinterpret_cast<T*>(y.data());
      for (std::size_t l = 0; l < lines * count; ++l) {
        std::memcpy(real + l * 2 * h, x.data() + l * n, n * sizeof(T));
      }
      p.execute(real, y.data(), ws);
      expect_within<T>(y, want, dims, "r2c in place");
    }
  }
}

TYPED_TEST(MultiDim, ComplexToReal) {
  using T = TypeParam;
  using c = std::complex<T>;
  for (const dims_t& dims : shapes()) {
    const std::size_t n = dims.back();
    const std::size_t h = n / 2 + 1;
    const std::size_t points = product(dims);
    const std::size_t lines = points / n;
    fftn::test::input_source src(points + 2);
    const auto x = src.reals<T>(points * count);
    // A consistent half spectrum: that of real data, so c2r returns points * x.
    std::vector<c> spectrum;
    std::vector<cd> want;
    for (std::size_t t = 0; t < count; ++t) {
      const std::vector<cd> one(x.begin() + static_cast<std::ptrdiff_t>(t * points),
                                x.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
      const auto y = dft(-1, dims, one);
      for (std::size_t l = 0; l < lines; ++l) {
        for (std::size_t k = 0; k < h; ++k) {
          spectrum.push_back(static_cast<c>(y[l * n + k]));
        }
      }
      for (const cd& v : one) {
        want.push_back(v * static_cast<double>(points));
      }
    }
    const fftn::shape<c, T> s{.axes = axes_of<T>(dims), .batches = {{.count = count}}};
    const auto check = [&](const std::vector<T>& got, const char* what) {
      const std::vector<c> as_complex(got.begin(), got.end());
      expect_within<T>(as_complex, want, dims, what);
    };
    {
      const fftn::c2r_plan<T> p(s);
      fftn::workspace ws(p);
      const std::vector<c> before = spectrum;
      std::vector<T> y(points * count);
      p.execute(spectrum.data(), y.data(), ws);
      EXPECT_EQ(std::memcmp(before.data(), spectrum.data(), before.size() * sizeof(c)), 0)
          << "c2r out of place changed its input";
      check(y, "c2r");
    }
    {
      const fftn::c2r_plan<T> p(s, {.in_place = true});
      fftn::workspace ws(p);
      std::vector<c> buffer = spectrum;
      auto* real = reinterpret_cast<T*>(buffer.data());
      p.execute(buffer.data(), real, ws);
      std::vector<T> y;
      for (std::size_t l = 0; l < lines * count; ++l) {
        y.insert(y.end(), real + l * 2 * h, real + l * 2 * h + n);
      }
      check(y, "c2r in place");
    }
  }
}

} // namespace
