// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Values from the quadruple-precision reference, carried as an unevaluated sum hi + lo of
// two doubles (about 106 bits). That is enough to measure the error of a double result to
// its last bit, and it keeps Boost.Multiprecision out of every header but one.

#pragma once

#include <cmath>
#include <complex>
#include <limits>

namespace fftn::test {

// hi is the double nearest to the value; lo is the (rounded) remainder.
struct exact_real {
  double hi = 0.0;
  double lo = 0.0;
};

struct exact_complex {
  exact_real re;
  exact_real im;
};

inline exact_real exact_from(double v) { return {.hi = v, .lo = 0.0}; }
inline exact_complex exact_from(std::complex<double> v) {
  return {.re = exact_from(v.real()), .im = exact_from(v.imag())};
}

// got - want. When got and hi are within a factor of two the first subtraction is exact
// (Sterbenz), so the result carries a relative error of at most 2^-53: more than enough to
// compare errors with tolerances.
inline double error_of(double got, exact_real want) { return (got - want.hi) - want.lo; }

inline double magnitude(exact_complex v) { return std::hypot(v.re.hi, v.im.hi); }

// The unit in the last place of T at magnitude |v| (for normal numbers).
template <class T> double ulp_of(double v) {
  if (v == 0.0) {
    return static_cast<double>(std::numeric_limits<T>::denorm_min());
  }
  int e = 0;
  static_cast<void>(std::frexp(v, &e)); // |v| = m * 2^e with m in [0.5, 1)
  return std::ldexp(1.0, e - std::numeric_limits<T>::digits);
}

// Rounds the exact value once to T (to nearest, ties to even).
template <class T> T round_to(exact_real v);

template <> inline double round_to<double>(exact_real v) { return v.hi; }

// Rounding hi to float rounds hi + lo correctly except when hi lies exactly halfway between
// two floats: every other point of the double grid is at least one double ulp from a
// midpoint, and |lo| is at most half of one. In the halfway case lo breaks the tie.
template <> inline float round_to<float>(exact_real v) {
  const auto f = static_cast<float>(v.hi);
  const double below = std::nextafter(f, -std::numeric_limits<float>::infinity());
  const double above = std::nextafter(f, std::numeric_limits<float>::infinity());
  const double f_wide = f;
  if (v.lo > 0.0 && v.hi == (f_wide + above) / 2) {
    return static_cast<float>(above);
  }
  if (v.lo < 0.0 && v.hi == (f_wide + below) / 2) {
    return static_cast<float>(below);
  }
  return f;
}

template <class T> std::complex<T> round_to(exact_complex v) {
  return {round_to<T>(v.re), round_to<T>(v.im)};
}

} // namespace fftn::test
