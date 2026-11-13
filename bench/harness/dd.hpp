// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Double-double arithmetic (about 106 significant bits) for the harness's reference DFT.
//
// The error-free transformations are Dekker's and Knuth's, written without FMA: the build
// has -ffp-contract=off (so the compiler fuses nothing, which these algorithms rely on), and
// std::fma is a slow library call on x86-64 builds without an FMA flag. The reference only
// has to be far more accurate than a double-precision FFT (relative error ~1e-16), so
// these simple forms, accurate to a few units of 1e-32, are enough.
#pragma once

#include <cstdint>

namespace fbench {

struct dd {
  double hi = 0.0;
  double lo = 0.0;

  constexpr dd() = default;
  constexpr dd(double h, double l) : hi(h), lo(l) {}
};

struct ddc {
  dd re;
  dd im;

  constexpr ddc() = default;
  constexpr ddc(dd r, dd i) : re(r), im(i) {}
};

inline dd quick_two_sum(double a, double b) {
  const double s = a + b;
  return {s, b - (s - a)};
}

inline dd two_sum(double a, double b) {
  const double s = a + b;
  const double bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}

inline dd split(double a) {
  constexpr double splitter = 134217729.0; // 2^27 + 1
  const double c = splitter * a;
  const double hi = c - (c - a);
  return {hi, a - hi};
}

inline dd two_prod(double a, double b) {
  const double p = a * b;
  const dd as = split(a);
  const dd bs = split(b);
  const double err = ((as.hi * bs.hi - p) + as.hi * bs.lo + as.lo * bs.hi) + as.lo * bs.lo;
  return {p, err};
}

inline dd operator+(dd x, dd y) {
  dd s = two_sum(x.hi, y.hi);
  const dd t = two_sum(x.lo, y.lo);
  s.lo += t.hi;
  s = quick_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  return quick_two_sum(s.hi, s.lo);
}

inline dd operator-(dd x) { return {-x.hi, -x.lo}; }
inline dd operator-(dd x, dd y) { return x + (-y); }

inline dd operator*(dd x, dd y) {
  dd p = two_prod(x.hi, y.hi);
  p.lo += x.hi * y.lo + x.lo * y.hi;
  return quick_two_sum(p.hi, p.lo);
}

inline dd operator*(dd x, double y) {
  dd p = two_prod(x.hi, y);
  p.lo += x.lo * y;
  return quick_two_sum(p.hi, p.lo);
}

// x / y for a double y, by one correction step.
inline dd operator/(dd x, double y) {
  const double q1 = x.hi / y;
  const dd r = x - two_prod(q1, y);
  const double q2 = r.hi / y;
  return quick_two_sum(q1, q2);
}

inline ddc operator+(ddc a, ddc b) { return {a.re + b.re, a.im + b.im}; }

inline ddc operator*(ddc a, ddc b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

// Multiply-accumulate of a double-precision complex value into a double-double sum.
inline ddc mac(ddc sum, ddc w, double re, double im) {
  return {sum.re + (w.re * re - w.im * im), sum.im + (w.re * im + w.im * re)};
}

// exp(sign * 2 pi i m / n) for integers 0 <= m < n < 2^60, to about 1e-31. The angle is
// reduced exactly to an octant with integer arithmetic, so no rounding of 2 pi m / n
// enters; the octant's sine and cosine come from Taylor series in double-double.
ddc unit_root(std::int64_t m, std::int64_t n, int sign);

} // namespace fbench
