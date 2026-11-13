// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Tests of the reference DFT itself: known values of its roots of unity, that hi is the
// nearest double, exactness on inputs with exact transforms, agreement of the sampled and
// full evaluations, of r2c, c2r and planes with c2c, and its cost.

#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/inputs.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fftn::test {
namespace {

void expect_same(const exact_complex& a, const exact_complex& b) {
  EXPECT_EQ(a.re.hi, b.re.hi);
  EXPECT_EQ(a.re.lo, b.re.lo);
  EXPECT_EQ(a.im.hi, b.im.hi);
  EXPECT_EQ(a.im.lo, b.im.lo);
}

TEST(ReferenceRoots, ExactPoints) {
  for (const std::size_t n : std::initializer_list<std::size_t>{1, 2, 4, 8, 12, 1000, 4096}) {
    for (const int sign : {-1, 1}) {
      const exact_complex one = reference::unit_root(sign, 0, n);
      EXPECT_EQ(one.re.hi, 1.0);
      EXPECT_EQ(one.re.lo, 0.0);
      EXPECT_EQ(one.im.hi, 0.0);
      EXPECT_EQ(one.im.lo, 0.0);
      // m is reduced modulo n first.
      expect_same(reference::unit_root(sign, n, n), one);
      if (n % 2 == 0) {
        const exact_complex half = reference::unit_root(sign, n / 2, n);
        EXPECT_EQ(half.re.hi, -1.0);
        EXPECT_EQ(half.im.hi, 0.0);
      }
      if (n % 4 == 0) {
        const exact_complex quarter = reference::unit_root(sign, n / 4, n);
        EXPECT_EQ(quarter.re.hi, 0.0);
        EXPECT_EQ(quarter.im.hi, static_cast<double>(sign));
        EXPECT_EQ(quarter.im.lo, 0.0);
      }
    }
  }
}

TEST(ReferenceRoots, KnownValuesToQuadPrecision) {
  // cos(pi/4) = sqrt(2)/2 = 0x1.6a09e667f3bcdp-1 - 0x1.bdd3413b26456p-55 (double-double).
  const exact_complex w8 = reference::unit_root(1, 1, 8);
  EXPECT_EQ(w8.re.hi, 0x1.6a09e667f3bcdp-1);
  EXPECT_NEAR(w8.re.lo, -0x1.bdd3413b26456p-55, 1e-32);
  EXPECT_EQ(w8.im.hi, w8.re.hi);
  EXPECT_EQ(w8.im.lo, w8.re.lo);
  // cos(pi/3) = 1/2 and sin(pi/6) = 1/2.
  const exact_complex w6 = reference::unit_root(1, 1, 6);
  EXPECT_EQ(w6.re.hi, 0.5);
  EXPECT_NEAR(w6.re.lo, 0.0, 1e-32);
  const exact_complex w12 = reference::unit_root(-1, 1, 12);
  EXPECT_EQ(w12.im.hi, -0.5);
  EXPECT_NEAR(w12.im.lo, 0.0, 1e-32);
}

TEST(ReferenceRoots, MatchLibmAndSymmetries) {
  for (const std::size_t n :
       std::initializer_list<std::size_t>{3, 5, 7, 10, 97, 1000, 4099, 65536}) {
    const auto& fwd = reference::unit_roots(-1, n);
    const auto& bwd = reference::unit_roots(+1, n);
    ASSERT_EQ(fwd.size(), n);
    for (std::size_t m = 0; m < n; m += 1 + n / 200) {
      const double angle = 2 * std::numbers::pi * static_cast<double>(m) / static_cast<double>(n);
      EXPECT_NEAR(bwd[m].re.hi, std::cos(angle), 1e-15) << m << "/" << n;
      EXPECT_NEAR(bwd[m].im.hi, std::sin(angle), 1e-15) << m << "/" << n;
      // The folding to the first octant makes the symmetries exact.
      EXPECT_EQ(fwd[m].re.hi, bwd[m].re.hi);
      EXPECT_EQ(fwd[m].im.hi, -bwd[m].im.hi);
      EXPECT_EQ(fwd[m].im.lo, -bwd[m].im.lo);
      expect_same(bwd[(n - m) % n], fwd[m]);
      // hi is the nearest double: the remainder is at most half an ulp.
      EXPECT_LE(std::abs(bwd[m].re.lo), ulp_of<double>(bwd[m].re.hi) / 2);
      EXPECT_LE(std::abs(bwd[m].im.lo), ulp_of<double>(bwd[m].im.hi) / 2);
      expect_same(reference::unit_root(1, m, n), bwd[m]);
    }
  }
}

TEST(ReferenceDft, ImpulsesGiveTheRootsExactly) {
  for (const std::size_t n : std::initializer_list<std::size_t>{1, 2, 3, 16, 17, 100}) {
    for (const std::size_t at : {std::size_t{0}, n / 2, n - 1}) {
      std::vector<std::complex<double>> x(n);
      x[at] = 1.0;
      const auto y = reference::c2c(-1, x);
      for (std::size_t k = 0; k < n; ++k) {
        expect_same(y[k], reference::unit_root(-1, k * at, n));
      }
    }
  }
}

TEST(ReferenceDft, ConstantGivesOneBin) {
  for (const std::size_t n : std::initializer_list<std::size_t>{1, 3, 64, 97}) {
    const std::vector<std::complex<double>> x(n, std::complex<double>(2.0, -1.0));
    const auto y = reference::c2c(1, x);
    EXPECT_EQ(y[0].re.hi, 2.0 * static_cast<double>(n));
    EXPECT_EQ(y[0].im.hi, -1.0 * static_cast<double>(n));
    for (std::size_t k = 1; k < n; ++k) {
      EXPECT_LT(magnitude(y[k]), 1e-30 * static_cast<double>(n));
    }
  }
}

TEST(ReferenceDft, SampledBinsEqualTheFullTransform) {
  input_source src(1);
  for (const std::size_t n : std::initializer_list<std::size_t>{1, 7, 64, 100}) {
    const auto x = widen(src.complexes<double>(n));
    const auto full = reference::c2c(-1, x);
    const std::vector<std::size_t> bins{n - 1, 0, n / 2, n - 1};
    const auto some = reference::c2c(-1, x, bins);
    ASSERT_EQ(some.size(), bins.size());
    for (std::size_t i = 0; i < bins.size(); ++i) {
      expect_same(some[i], full[bins[i]]);
    }
  }
}

TEST(ReferenceDft, RealTransformsAgreeWithComplex) {
  input_source src(2);
  for (const std::size_t n : std::initializer_list<std::size_t>{1, 2, 3, 8, 15, 64}) {
    const auto x = widen(src.reals<double>(n));
    const std::vector<std::complex<double>> z(x.begin(), x.end());
    const auto full = reference::c2c(-1, z);
    const auto half = reference::r2c(x);
    ASSERT_EQ(half.size(), n / 2 + 1);
    for (std::size_t k = 0; k <= n / 2; ++k) {
      expect_same(half[k], full[k]);
    }
    // c2r(r2c(x)) = n x, with garbage in the ignored imaginary parts.
    std::vector<std::complex<double>> X;
    X.reserve(half.size());
    for (const auto& v : half) {
      X.emplace_back(v.re.hi, v.im.hi);
    }
    X[0].imag(0.375);
    if (n % 2 == 0) {
      X[n / 2].imag(-0.625);
    }
    const auto back = reference::c2r(n, X);
    ASSERT_EQ(back.size(), n);
    for (std::size_t j = 0; j < n; ++j) {
      // The rounding of X to doubles leaves an error of order 1e-16 * n.
      EXPECT_NEAR(back[j].hi, static_cast<double>(n) * x[j], 1e-14 * static_cast<double>(n));
    }
    // c2r's convention: the imaginary parts of bins 0 and n/2 do not matter at all.
    auto clean = X;
    clean[0].imag(0.0);
    if (n % 2 == 0) {
      clean[n / 2].imag(0.0);
    }
    const auto back_clean = reference::c2r(n, clean);
    const std::vector<std::size_t> some{0, n - 1};
    const auto back_some = reference::c2r(n, X, some);
    for (std::size_t j = 0; j < n; ++j) {
      EXPECT_EQ(back[j].hi, back_clean[j].hi);
      EXPECT_EQ(back[j].lo, back_clean[j].lo);
    }
    EXPECT_EQ(back_some[1].hi, back[n - 1].hi);
  }
}

TEST(ReferenceDft, PlanesAreSeparable) {
  input_source src(3);
  for (const auto& [ny, nx] : {std::pair<std::size_t, std::size_t>{1, 1}, {3, 5}, {8, 6}}) {
    const auto a = widen(src.integer_complexes<double>(ny, 100));
    const auto b = widen(src.integer_complexes<double>(nx, 100));
    std::vector<std::complex<double>> x(ny * nx);
    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t xi = 0; xi < nx; ++xi) {
        x[y * nx + xi] = a[y] * b[xi];
      }
    }
    const auto fa = reference::c2c(1, a);
    const auto fb = reference::c2c(1, b);
    const auto f = reference::c2c_plane(1, ny, nx, x);
    const std::vector<std::pair<std::size_t, std::size_t>> points{{ny - 1, 0}, {0, nx - 1}};
    const auto some = reference::c2c_plane(1, ny, nx, x, points);
    for (std::size_t ky = 0; ky < ny; ++ky) {
      for (std::size_t kx = 0; kx < nx; ++kx) {
        const std::complex<double> want = std::complex<double>(fa[ky].re.hi, fa[ky].im.hi) *
                                          std::complex<double>(fb[kx].re.hi, fb[kx].im.hi);
        const exact_complex& got = f[ky * nx + kx];
        EXPECT_NEAR(got.re.hi, want.real(), 1e-12 * (1 + std::abs(want)));
        EXPECT_NEAR(got.im.hi, want.imag(), 1e-12 * (1 + std::abs(want)));
      }
    }
    expect_same(some[0], f[(ny - 1) * nx]);
    expect_same(some[1], f[nx - 1]);
  }
}

// DESIGN.md 15.1 states about 0.3 s for a 1024-point reference DFT; this prints the time on
// the machine at hand (it is not a pass/fail criterion: sanitizer builds are slower).
TEST(ReferenceDft, CostOfA1024PointTransform) {
  input_source src(4);
  const auto x = widen(src.complexes<double>(1024));
  static_cast<void>(reference::unit_roots(-1, 1024)); // twiddles are cached; time the sums
  const auto start = std::chrono::steady_clock::now();
  const auto y = reference::c2c(-1, x);
  const std::chrono::duration<double> took = std::chrono::steady_clock::now() - start;
  ASSERT_EQ(y.size(), 1024U);
  RecordProperty("seconds", std::to_string(took.count()));
  std::cout << "[ timing   ] 1024-point reference DFT: " << took.count() << " s\n";
}

TEST(ReferenceDft, RejectsBadArguments) {
  const std::vector<std::complex<double>> x(4);
  EXPECT_THROW(static_cast<void>(reference::c2c(0, x)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(reference::c2r(4, x)), std::invalid_argument);
  const std::vector<double> r(4);
  const std::vector<std::size_t> high{3};
  EXPECT_THROW(static_cast<void>(reference::r2c(r, high)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(reference::c2c_plane(1, 3, 3, x)), std::invalid_argument);
  EXPECT_TRUE(reference::c2c(-1, std::vector<std::complex<double>>{}).empty());
}

} // namespace
} // namespace fftn::test
