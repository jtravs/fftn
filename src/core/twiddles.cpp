// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Double-double arithmetic (Dekker 1971; Hida, Li and Bailey, "Library for double-double
// and quad-double arithmetic", 2007), written from the published algorithms. The library
// is built with -ffp-contract=off, which these error-free transformations need: a fused
// multiply-add introduced by the compiler would break their exactness.

#include "core/twiddles.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

struct dd {
  double hi = 0.0;
  double lo = 0.0;
};

// a + b exactly, as s + e.
dd two_sum(double a, double b) {
  const double s = a + b;
  const double bb = s - a;
  return {.hi = s, .lo = (a - (s - bb)) + (b - bb)};
}

// a + b exactly, given |a| >= |b|.
dd quick_two_sum(double a, double b) {
  const double s = a + b;
  return {.hi = s, .lo = b - (s - a)};
}

// a * b exactly, by Dekker's splitting (no FMA, so the result does not depend on whether
// the processor or the build has one).
dd two_prod(double a, double b) {
  constexpr double splitter = 134217729.0; // 2^27 + 1
  const double ta = splitter * a;
  const double ah = ta - (ta - a);
  const double al = a - ah;
  const double tb = splitter * b;
  const double bh = tb - (tb - b);
  const double bl = b - bh;
  const double p = a * b;
  return {.hi = p, .lo = ((ah * bh - p) + ah * bl + al * bh) + al * bl};
}

dd operator+(dd x, dd y) {
  const dd s = two_sum(x.hi, y.hi);
  const dd t = two_sum(x.lo, y.lo);
  const dd u = quick_two_sum(s.hi, s.lo + t.hi);
  return quick_two_sum(u.hi, u.lo + t.lo);
}

dd operator-(dd x) { return {.hi = -x.hi, .lo = -x.lo}; }

dd operator*(dd x, dd y) {
  const dd p = two_prod(x.hi, y.hi);
  return quick_two_sum(p.hi, p.lo + (x.hi * y.lo + x.lo * y.hi));
}

// x / d for a double d.
dd operator/(dd x, double d) {
  const double q1 = x.hi / d;
  const dd p = two_prod(q1, d);
  const double r = ((x.hi - p.hi) - p.lo) + x.lo;
  return quick_two_sum(q1, r / d);
}

dd operator/(dd x, dd y) {
  const double q1 = x.hi / y.hi;
  dd r = x + -(y * dd{.hi = q1, .lo = 0.0});
  const double q2 = r.hi / y.hi;
  r = r + -(y * dd{.hi = q2, .lo = 0.0});
  const double q3 = r.hi / y.hi;
  return quick_two_sum(q1, q2) + dd{.hi = q3, .lo = 0.0};
}

// An integer below 2^63, exactly.
dd from_integer(std::uint64_t v) {
  const auto hi = static_cast<double>(v);
  // The remainder is below 2^11 in magnitude and may be negative: wrap-around subtraction
  // reinterpreted as signed gives it exactly.
  const auto rest = static_cast<std::int64_t>(v - static_cast<std::uint64_t>(hi));
  return {.hi = hi, .lo = static_cast<double>(rest)};
}

// pi / 4 to about 107 bits.
constexpr dd quarter_pi{.hi = 0x1.921fb54442d18p-1, .lo = 0x1.1a62633145c07p-55};

// Terms are added until they fall below this, far under the 2^-106 of a double-double.
constexpr double negligible = 0x1p-112;

// sin t and cos t for 0 <= t <= pi / 4, by their Taylor series.
void sin_cos(dd t, dd& sine, dd& cosine) {
  const dd t2 = t * t;
  dd term = t;
  dd sum = t;
  for (int k = 1; std::abs(term.hi) > negligible; ++k) {
    term = -(term * t2) / static_cast<double>((2 * k) * (2 * k + 1));
    sum = sum + term;
  }
  sine = sum;
  term = dd{.hi = 1.0, .lo = 0.0};
  sum = term;
  for (int k = 1; std::abs(term.hi) > negligible; ++k) {
    term = -(term * t2) / static_cast<double>((2 * k - 1) * (2 * k));
    sum = sum + term;
  }
  cosine = sum;
}

// cos and sin of 2 pi m / n, for m < n < 2^60.
void root(std::uint64_t m, std::uint64_t n, dd& c, dd& s) {
  // 2 pi m / n = (pi / 4)(o + r / n): octant o, remainder r < n, all in integers. In odd
  // octants the angle is measured back from the octant's end, (pi / 4)(n - r) / n, so the
  // series is always evaluated on [0, pi / 4] and the octant boundaries are exact.
  const std::uint64_t eighths = 8 * m;
  const std::uint64_t o = eighths / n;
  const std::uint64_t r = eighths - o * n;
  const bool odd = (o & 1U) != 0;
  const dd t = quarter_pi * (from_integer(odd ? n - r : r) / from_integer(n));
  dd st;
  dd ct;
  sin_cos(t, st, ct);
  switch (o) {
  case 0:
    c = ct;
    s = st;
    break;
  case 1:
    c = st;
    s = ct;
    break;
  case 2:
    c = -st;
    s = ct;
    break;
  case 3:
    c = -ct;
    s = st;
    break;
  case 4:
    c = -ct;
    s = -st;
    break;
  case 5:
    c = -st;
    s = -ct;
    break;
  case 6:
    c = st;
    s = -ct;
    break;
  default:
    c = ct;
    s = -st;
    break;
  }
}

template <class T> T round_once(dd v);

// hi + lo is normalised (|lo| <= half an ulp of hi), so hi is the nearest double.
template <> double round_once<double>(dd v) { return v.hi; }

// Rounding hi to float is correct unless hi lies exactly halfway between two floats, when
// lo breaks the tie (exact.hpp in the tests explains the same rule).
template <> float round_once<float>(dd v) {
  const auto f = static_cast<float>(v.hi);
  const double gap = v.hi - static_cast<double>(f);
  if (gap == 0.0 || v.lo == 0.0) {
    return f;
  }
  const float toward = std::nextafter(f, gap > 0.0 ? std::numeric_limits<float>::infinity()
                                                   : -std::numeric_limits<float>::infinity());
  const double half = std::abs(static_cast<double>(toward) - static_cast<double>(f)) / 2;
  if (std::abs(gap) == half && (v.lo > 0.0) == (gap > 0.0)) {
    return toward;
  }
  return f;
}

} // namespace

template <class T> std::vector<std::complex<T>> unit_roots(int sign, std::size_t n) {
  std::vector<std::complex<T>> w(n);
  const auto nn = static_cast<std::uint64_t>(n);
  // exp(-2 pi i (n - m) / n) is the exact conjugate of exp(-2 pi i m / n): compute the
  // first half and mirror it.
  for (std::size_t m = 0; 2 * m <= n; ++m) {
    dd c;
    dd s;
    root(static_cast<std::uint64_t>(m), nn, c, s);
    const T re = round_once<T>(c);
    const T im = round_once<T>(sign < 0 ? -s : s);
    w[m] = {re, im};
    if (m != 0 && m != n - m) {
      w[n - m] = {re, -im};
    }
  }
  return w;
}

template std::vector<std::complex<double>> unit_roots<double>(int sign, std::size_t n);
template std::vector<std::complex<float>> unit_roots<float>(int sign, std::size_t n);

} // namespace fftn::inline v0::detail
