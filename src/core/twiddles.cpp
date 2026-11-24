// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Double-double arithmetic (Dekker 1971; Hida, Li and Bailey, "Library for double-double
// and quad-double arithmetic", 2007), written from the published algorithms. The library
// is built with -ffp-contract=off, which these error-free transformations need: a fused
// multiply-add introduced by the compiler would break their exactness.

#include "core/twiddles.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
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

// The angle (pi / 4) j / n, 0 <= j <= n, on which the series are evaluated.
dd reduced_angle(std::uint64_t j, std::uint64_t n) {
  return quarter_pi * (from_integer(j) / from_integer(n));
}

// cos and sin of an angle in octant o (0 to 4), from the sine st and cosine ct of its
// reduced angle (root() below).
void place(std::uint64_t o, dd st, dd ct, dd& c, dd& s) {
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
  default: // 4, at m = n / 2 exactly
    c = -ct;
    s = -st;
    break;
  }
}

// cos and sin of 2 pi m / n, for 2 m <= n < 2^60: the callers mirror the second half of the
// circle onto the first, so the octant is at most 4.
void root(std::uint64_t m, std::uint64_t n, dd& c, dd& s) {
  assert(2 * m <= n);
  // 2 pi m / n = (pi / 4)(o + r / n): octant o, remainder r < n, all in integers. In odd
  // octants the angle is measured back from the octant's end, (pi / 4)(n - r) / n, so the
  // series is always evaluated on [0, pi / 4] and the octant boundaries are exact.
  const std::uint64_t eighths = 8 * m;
  const std::uint64_t o = eighths / n;
  const std::uint64_t r = eighths - o * n;
  const bool odd = (o & 1U) != 0;
  dd st;
  dd ct;
  sin_cos(reduced_angle(odd ? n - r : r, n), st, ct);
  place(o, st, ct, c, s);
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

// Entry m (2 m <= n) of the table from the cosine and sine of its angle, and its mirror
// n - m: exp(-2 pi i (n - m) / n) is the exact conjugate of exp(-2 pi i m / n).
template <class T>
void set_entry(std::complex<T>* w, int sign, std::size_t n, std::size_t m, dd c, dd s) {
  const T re = round_once<T>(c);
  const T im = round_once<T>(sign < 0 ? -s : s);
  w[m] = {re, im};
  if (m != 0 && m != n - m) {
    w[n - m] = {re, -im};
  }
}

// Entry m (2 m <= n, m even) copied from the table of n / 2, and its mirror.
template <class T>
void copy_entry(std::complex<T>* w, std::size_t n, std::size_t m, const std::complex<T>* half) {
  w[m] = half[m / 2];
  if (m != 0 && m != n - m) {
    w[n - m] = {w[m].real(), -w[m].imag()};
  }
}

// The table: the first half computed, the second mirrored. Given `half`, the table of n / 2
// (n even) of the same sign, the even entries are copied from it: root(2 m, n) is
// root(m, n / 2) with every operand of every operation scaled by two, which changes no
// rounding, so the two are equal bit for bit.
//
// For n divisible by 8 the first half's reduced angles lie on the grid (pi / 4)(8 i / n),
// i = 0 .. p with p = n / 8, and four entries share each one (octant symmetry): m = i,
// 2p - i, 2p + i and 4p - i, in octants m / p = 0, 1, 2 and 3 (at i = 0: m = 0, 2p, 4p; at
// i = p: m = p, 3p). The sine and cosine of each grid angle are computed once, from the
// same operands as root() would compute them for each of those m, and placed by root()'s
// octant rule: the table is bit for bit the one root() gives entry by entry, for a quarter
// of the series evaluations. The m of one i have the parity of i.
template <class T>
std::vector<std::complex<T>> build_table(int sign, std::size_t n, const std::complex<T>* half) {
  std::vector<std::complex<T>> table(n);
  std::complex<T>* w = table.data();
  const auto nn = static_cast<std::uint64_t>(n);
  if (n % 8 != 0) {
    for (std::size_t m = 0; 2 * m <= n; ++m) {
      if (half != nullptr && m % 2 == 0) {
        copy_entry(w, n, m, half);
        continue;
      }
      dd c;
      dd s;
      root(static_cast<std::uint64_t>(m), nn, c, s);
      set_entry(w, sign, n, m, c, s);
    }
    return table;
  }
  const std::size_t p = n / 8;
  for (std::size_t i = 0; i <= p; ++i) {
    std::array<std::size_t, 4> ms{i, 2 * p - i, 2 * p + i, 4 * p - i};
    std::size_t count = 4;
    if (i == 0) {
      ms = {0, 2 * p, 4 * p, 0};
      count = 3;
    } else if (i == p) {
      ms = {p, 3 * p, 0, 0};
      count = 2;
    }
    if (half != nullptr && i % 2 == 0) {
      for (std::size_t k = 0; k < count; ++k) {
        copy_entry(w, n, ms[k], half);
      }
      continue;
    }
    dd st;
    dd ct;
    sin_cos(reduced_angle(static_cast<std::uint64_t>(8 * i), nn), st, ct);
    for (std::size_t k = 0; k < count; ++k) {
      dd c;
      dd s;
      place(static_cast<std::uint64_t>(ms[k] / p), st, ct, c, s);
      set_entry(w, sign, n, ms[k], c, s);
    }
  }
  return table;
}

} // namespace

template <class T> std::vector<std::complex<T>> unit_roots(int sign, std::size_t n) {
  return build_table<T>(sign, n, nullptr);
}

template <class T>
std::vector<std::complex<T>> unit_roots(int sign, std::size_t n,
                                        std::span<const std::complex<T>> half) {
  assert(n % 2 == 0 && half.size() == n / 2);
  return build_table<T>(sign, n, half.data());
}

template <class T> std::complex<T> unit_root(int sign, std::size_t m, std::size_t n) {
  // The table stores its second half as the conjugate of the first; so does this.
  const bool mirrored = 2 * m > n;
  dd c;
  dd s;
  root(static_cast<std::uint64_t>(mirrored ? n - m : m), static_cast<std::uint64_t>(n), c, s);
  const T re = round_once<T>(c);
  const T im = round_once<T>(sign < 0 ? -s : s);
  return {re, mirrored ? -im : im};
}

template <class T>
root_cache<T>::root_cache(std::size_t n)
    : n_(n), angles_(n % 8 == 0 ? n / 8 + 1 : 0), known_(angles_.size(), 0) {}

template <class T> std::complex<T> root_cache<T>::operator()(int sign, std::size_t m) {
  if (angles_.empty()) {
    return unit_root<T>(sign, m, n_);
  }
  // unit_root's computation, with the series of the reduced angle (pi / 4) 8 i / n kept:
  // root() reduces m to octant o = m / p and j = 8 i, i = m mod p or p - m mod p.
  const bool mirrored = 2 * m > n_;
  const std::size_t mm = mirrored ? n_ - m : m;
  const std::size_t p = n_ / 8;
  const std::size_t o = mm / p;
  const std::size_t i = (o & 1U) != 0 ? p - (mm - o * p) : mm - o * p;
  std::array<double, 4>& a = angles_[i];
  if (known_[i] == 0) {
    dd st;
    dd ct;
    sin_cos(reduced_angle(static_cast<std::uint64_t>(8 * i), static_cast<std::uint64_t>(n_)), st,
            ct);
    a = {st.hi, st.lo, ct.hi, ct.lo};
    known_[i] = 1;
  }
  dd c;
  dd s;
  place(static_cast<std::uint64_t>(o), dd{.hi = a[0], .lo = a[1]}, dd{.hi = a[2], .lo = a[3]}, c,
        s);
  const T re = round_once<T>(c);
  const T im = round_once<T>(sign < 0 ? -s : s);
  return {re, mirrored ? -im : im};
}

template class root_cache<double>;
template class root_cache<float>;

template std::vector<std::complex<double>> unit_roots<double>(int sign, std::size_t n);
template std::vector<std::complex<float>> unit_roots<float>(int sign, std::size_t n);
template std::vector<std::complex<double>>
unit_roots<double>(int sign, std::size_t n, std::span<const std::complex<double>> half);
template std::vector<std::complex<float>>
unit_roots<float>(int sign, std::size_t n, std::span<const std::complex<float>> half);
template std::complex<double> unit_root<double>(int sign, std::size_t m, std::size_t n);
template std::complex<float> unit_root<float>(int sign, std::size_t m, std::size_t n);

} // namespace fftn::inline v0::detail
