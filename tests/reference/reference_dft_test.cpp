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

// The full transforms come from a store that keeps only the forward transform and reads the
// backward one from it, bin k from bin (n - k) mod n: that is bit for bit what a fresh
// evaluation of either sign gives (the first call of the overload with a list of bins, whose
// store is separate), for lengths of every residue modulo 8, primes and the suites' long
// rows, whichever sign is asked for first. Repeated sampled calls (c2c, r2c, c2r) give the
// first call's bits.
TEST(ReferenceDft, BothSignsShareOneEvaluation) {
  input_source src(5);
  for (const std::size_t n :
       std::initializer_list<std::size_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 97, 100, 151, 1000}) {
    for (const int first : {-1, 1}) {
      const auto x = widen(src.complexes<double>(n));
      std::vector<std::size_t> all(n);
      for (std::size_t k = 0; k < n; ++k) {
        all[k] = k;
      }
      const auto fresh_fwd = reference::c2c(-1, x, all);
      const auto fresh_bwd = reference::c2c(1, x, all);
      const auto stored_first = reference::c2c(first, x);
      const auto stored_second = reference::c2c(-first, x);
      const auto& want_first = first < 0 ? fresh_fwd : fresh_bwd;
      const auto& want_second = first < 0 ? fresh_bwd : fresh_fwd;
      ASSERT_EQ(stored_first.size(), n);
      ASSERT_EQ(stored_second.size(), n);
      for (std::size_t k = 0; k < n; ++k) {
        SCOPED_TRACE("n = " + std::to_string(n) + ", bin " + std::to_string(k));
        expect_same(fresh_bwd[k], fresh_fwd[(n - k) % n]);
        expect_same(stored_first[k], want_first[k]);
        expect_same(stored_second[k], want_second[k]);
      }
      const std::vector<std::size_t> some{n - 1, 0, n / 2};
      const auto again = reference::c2c(1, x, all);
      const auto sampled = reference::c2c(-1, x, some);
      const auto sampled_again = reference::c2c(-1, x, some);
      for (std::size_t k = 0; k < n; ++k) {
        expect_same(again[k], fresh_bwd[k]);
      }
      for (std::size_t i = 0; i < some.size(); ++i) {
        expect_same(sampled[i], fresh_fwd[some[i]]);
        expect_same(sampled_again[i], sampled[i]);
      }
      const auto reals = widen(src.reals<double>(n));
      const std::vector<std::size_t> half{0, n / 2};
      const auto r = reference::r2c(reals, half);
      const auto r_again = reference::r2c(reals, half);
      std::vector<std::complex<double>> spectrum(n / 2 + 1);
      for (std::size_t k = 0; k <= n / 2; ++k) {
        spectrum[k] = x[k];
      }
      const auto c = reference::c2r(n, spectrum, some);
      const auto c_again = reference::c2r(n, spectrum, some);
      for (std::size_t i = 0; i < half.size(); ++i) {
        expect_same(r_again[i], r[i]);
      }
      for (std::size_t i = 0; i < some.size(); ++i) {
        EXPECT_EQ(c_again[i].hi, c[i].hi);
        EXPECT_EQ(c_again[i].lo, c[i].lo);
      }
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

// The DHT (DESIGN.md 5.7): the real DFT's bins combined, H_k = Re X_k - Im X_k, and on two
// axes the product of the 1-D transforms, its own inverse up to the number of points.
TEST(ReferenceDht, AgreesWithTheRealTransformAndIsSeparable) {
  input_source src(0xD47);
  for (const std::size_t n : {1U, 2U, 7U, 16U, 45U}) {
    const std::vector<double> x = src.reals<double>(n);
    const std::vector<std::size_t> extents{n};
    const std::vector<exact_real> h = reference::dht(extents, x);
    const std::vector<std::complex<double>> z(x.begin(), x.end());
    const std::vector<exact_complex> X = reference::c2c(-1, z);
    for (std::size_t k = 0; k < n; ++k) {
      EXPECT_NEAR(h[k].hi, X[k].re.hi - X[k].im.hi, 1e-15 * static_cast<double>(n))
          << n << " " << k;
    }
  }
  const std::size_t ny = 5;
  const std::size_t nx = 6;
  const std::vector<double> x = src.reals<double>(ny * nx);
  const std::vector<std::size_t> extents{ny, nx};
  const std::vector<exact_real> h = reference::dht(extents, x);
  std::vector<double> mid(ny * nx);
  for (std::size_t i = 0; i < mid.size(); ++i) {
    mid[i] = h[i].hi;
  }
  const std::vector<exact_real> back = reference::dht(extents, mid);
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(back[i].hi, x[i] * static_cast<double>(ny * nx), 1e-13) << i;
  }
  // One point of the plane: cas along y times cas along x.
  std::vector<double> impulse(ny * nx, 0.0);
  impulse[(1 * nx) + 2] = 1.0;
  const std::vector<exact_real> g = reference::dht(extents, impulse);
  const auto cas = [](std::size_t m, std::size_t n) {
    const double phase = 2 * std::numbers::pi * static_cast<double>(m % n) / static_cast<double>(n);
    return std::cos(phase) + std::sin(phase);
  };
  for (std::size_t ky = 0; ky < ny; ++ky) {
    for (std::size_t kx = 0; kx < nx; ++kx) {
      EXPECT_NEAR(g[(ky * nx) + kx].hi, cas(ky, ny) * cas(2 * kx, nx), 1e-15) << ky << " " << kx;
    }
  }
  EXPECT_THROW(static_cast<void>(reference::dht(extents, std::vector<double>(7))),
               std::invalid_argument);
}

// The real-to-real kinds (DESIGN.md 5.7): DCT-II and DST-III against their sums in double, the
// identities of signs and reversals the r2r suites build their references with, each kind's
// inverse up to its N, the listed outputs, and the separable product.
TEST(ReferenceR2r, DefinitionsIdentitiesAndInverses) {
  using reference::r2r_type;
  input_source src(0x2D2);
  const auto line = [](r2r_type t, const std::vector<double>& x) {
    const std::vector<std::size_t> extents{x.size()};
    const std::vector<r2r_type> kinds{t};
    std::vector<double> y;
    for (const exact_real& v : reference::r2r(kinds, extents, x)) {
      y.push_back(v.hi);
    }
    return y;
  };
  const double pi = std::numbers::pi;
  for (const std::size_t n : {1U, 2U, 3U, 8U, 15U, 16U}) {
    const double nd = static_cast<double>(n);
    const std::vector<double> x = src.reals<double>(n);
    const std::vector<double> d2 = line(r2r_type::dct2, x);
    const std::vector<double> d3 = line(r2r_type::dct3, x);
    const std::vector<double> s3 = line(r2r_type::dst3, x);
    for (std::size_t k = 0; k < n; ++k) {
      const double kd = static_cast<double>(k);
      double want_d2 = 0.0;
      double want_s3 = (k % 2 == 0 ? 1.0 : -1.0) * x[n - 1];
      for (std::size_t j = 0; j < n; ++j) {
        const double jd = static_cast<double>(j);
        want_d2 += 2.0 * x[j] * std::cos(pi * (jd + 0.5) * kd / nd);
        if (j + 1 < n) {
          want_s3 += 2.0 * x[j] * std::sin(pi * (jd + 1.0) * (kd + 0.5) / nd);
        }
      }
      EXPECT_NEAR(d2[k], want_d2, 1e-14 * nd) << n << " " << k;
      EXPECT_NEAR(s3[k], want_s3, 1e-14 * nd) << n << " " << k;
    }
    // R: x reversed; S: (-1)^j x_j.
    std::vector<double> r(n);
    std::vector<double> sx(n);
    for (std::size_t j = 0; j < n; ++j) {
      r[j] = x[n - 1 - j];
      sx[j] = (j % 2 == 0 ? 1.0 : -1.0) * x[j];
    }
    const std::vector<double> d2r = line(r2r_type::dct2, r);
    const std::vector<double> s2s = line(r2r_type::dst2, sx);
    const std::vector<double> d3s = line(r2r_type::dct3, sx);
    const std::vector<double> s3r = line(r2r_type::dst3, r);
    for (std::size_t k = 0; k < n; ++k) {
      const double sk = k % 2 == 0 ? 1.0 : -1.0;
      EXPECT_NEAR(d2r[k], sk * d2[k], 1e-15 * nd) << n << " " << k;
      EXPECT_NEAR(s2s[k], d2[n - 1 - k], 1e-15 * nd) << n << " " << k;
      EXPECT_NEAR(d3s[k], d3[n - 1 - k], 1e-15 * nd) << n << " " << k;
      EXPECT_NEAR(s3r[k], sk * d3[k], 1e-15 * nd) << n << " " << k;
    }
    // The kinds IV and I (task 6.5): DCT-IV against its sum in double, and dst4(R x)_k =
    // (-1)^k dct4(x)_k, dst4(S x)_k = dct4(x)_(n-1-k), dct4(S R x)_k = (-1)^(n-1-k)
    // dct4(x)_(n-1-k), dct1(R x)_k = (-1)^k dct1(x)_k and dst1(R x)_k = (-1)^k dst1(x)_k.
    std::vector<double> sr(n);
    for (std::size_t j = 0; j < n; ++j) {
      sr[j] = (j % 2 == 0 ? 1.0 : -1.0) * x[n - 1 - j];
    }
    const std::vector<double> d4 = line(r2r_type::dct4, x);
    const std::vector<double> s1 = line(r2r_type::dst1, x);
    const std::vector<double> s4r = line(r2r_type::dst4, r);
    const std::vector<double> s4s = line(r2r_type::dst4, sx);
    const std::vector<double> d4sr = line(r2r_type::dct4, sr);
    const std::vector<double> s1r = line(r2r_type::dst1, r);
    const std::vector<double> d1 = n >= 2 ? line(r2r_type::dct1, x) : std::vector<double>{};
    const std::vector<double> d1r = n >= 2 ? line(r2r_type::dct1, r) : std::vector<double>{};
    for (std::size_t k = 0; k < n; ++k) {
      const double kd = static_cast<double>(k);
      const double sk = k % 2 == 0 ? 1.0 : -1.0;
      const std::size_t m = n - 1 - k;
      double want_d4 = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        want_d4 += 2.0 * x[j] * std::cos(pi * (static_cast<double>(j) + 0.5) * (kd + 0.5) / nd);
      }
      EXPECT_NEAR(d4[k], want_d4, 1e-14 * nd) << n << " " << k;
      EXPECT_NEAR(s4r[k], sk * d4[k], 1e-15 * nd) << n << " " << k;
      EXPECT_NEAR(s4s[k], d4[m], 1e-15 * nd) << n << " " << k;
      EXPECT_NEAR(d4sr[k], (m % 2 == 0 ? 1.0 : -1.0) * d4[m], 1e-15 * nd) << n << " " << k;
      EXPECT_NEAR(s1r[k], sk * s1[k], 1e-15 * nd) << n << " " << k;
      if (n >= 2) {
        EXPECT_NEAR(d1r[k], sk * d1[k], 1e-15 * nd) << n << " " << k;
      }
    }
    // Each kind then its inverse: N times the input.
    const struct {
      r2r_type kind;
      r2r_type inverse;
      double factor;
    } pairs[] = {
        {r2r_type::dct2, r2r_type::dct3, 2 * nd},
        {r2r_type::dct3, r2r_type::dct2, 2 * nd},
        {r2r_type::dst2, r2r_type::dst3, 2 * nd},
        {r2r_type::dst3, r2r_type::dst2, 2 * nd},
        {r2r_type::dct4, r2r_type::dct4, 2 * nd},
        {r2r_type::dst4, r2r_type::dst4, 2 * nd},
        {r2r_type::dst1, r2r_type::dst1, 2 * (nd + 1)},
        {r2r_type::dct1, r2r_type::dct1, 2 * (nd - 1)},
        {r2r_type::dht, r2r_type::dht, nd},
    };
    for (const auto& p : pairs) {
      if (p.kind == r2r_type::dct1 && n == 1) {
        continue;
      }
      const std::vector<double> back = line(p.inverse, line(p.kind, x));
      for (std::size_t j = 0; j < n; ++j) {
        EXPECT_NEAR(back[j], p.factor * x[j], 1e-14 * p.factor)
            << n << " " << static_cast<int>(p.kind) << " " << j;
      }
      // The listed outputs are the whole transform's, bit for bit.
      const std::vector<std::size_t> ks{0, n - 1, n / 2};
      const std::vector<double> whole = line(p.kind, x);
      const std::vector<exact_real> some = reference::r2r(p.kind, x, ks);
      for (std::size_t i = 0; i < ks.size(); ++i) {
        EXPECT_EQ(some[i].hi, whole[ks[i]]) << n << " " << static_cast<int>(p.kind);
      }
    }
  }
  // Two axes: an impulse's transform is the product of the two kinds' kernels.
  const std::vector<std::size_t> extents{5, 6};
  const std::vector<r2r_type> kinds{r2r_type::dst3, r2r_type::dct2};
  std::vector<double> impulse(30, 0.0);
  impulse[(1 * 6) + 2] = 1.0;
  const std::vector<exact_real> g = reference::r2r(kinds, extents, impulse);
  for (std::size_t ky = 0; ky < 5; ++ky) {
    for (std::size_t kx = 0; kx < 6; ++kx) {
      const double s3 = 2.0 * std::sin(pi * 2.0 * (static_cast<double>(ky) + 0.5) / 5.0);
      const double c2 = 2.0 * std::cos(pi * 2.5 * static_cast<double>(kx) / 6.0);
      EXPECT_NEAR(g[(ky * 6) + kx].hi, s3 * c2, 1e-14) << ky << " " << kx;
    }
  }
  const std::vector<double> one{1.0};
  EXPECT_THROW(static_cast<void>(reference::r2r(r2r_type::dct1, one, std::vector<std::size_t>{0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(reference::r2r(r2r_type::dct2, one, std::vector<std::size_t>{1})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(reference::r2r(std::vector<r2r_type>{r2r_type::dct2}, extents,
                                                std::vector<double>(30))),
               std::invalid_argument);
}

} // namespace
} // namespace fftn::test
