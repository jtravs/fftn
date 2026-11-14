// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The exact arithmetic of each target's complex products, so that tests can compare them
// bit for bit on any data. mul, mulc and madd round differently on each target (the order
// of their FMAs, vocab_<target>.inl); a reference that ignored the order could only bound
// the error, and a bound absorbs a lost FMA. Here every rounding is spelled out: x * y is
// rounded, std::fma(x, y, z) rounds once (the tests are built with -ffp-contract=off).
#pragma once

#include "isa/target.hpp"

#include <cmath>
#include <limits>

namespace isa_test {

template <class T> struct cnum {
  T re;
  T im;
};

// a * w, or a * conj(w) when conj_w, as target t computes it.
template <class T> cnum<T> product(fftn::detail::target t, cnum<T> a, cnum<T> w, bool conj_w) {
  using fftn::detail::target;
  const T ar = a.re;
  const T ai = a.im;
  const T wr = w.re;
  const T wi = w.im;
  switch (t) {
  case target::scalar: // plain C++: two products, then one sum
    if (conj_w) {
      return {ar * wr + ai * wi, ai * wr - ar * wi};
    }
    return {ar * wr - ai * wi, ar * wi + ai * wr};
  case target::neon_fcma: // FCMLA by w's real part onto -0, then by its imaginary part
  case target::neon:      // the same order: a times (wr, wr), then (-wi, wi) fused onto it
    if (conj_w) {
      return {std::fma(wi, ai, wr * ar), std::fma(-wi, ar, wr * ai)};
    }
    return {std::fma(-wi, ai, wr * ar), std::fma(wi, ar, wr * ai)};
  case target::avx2: // FMA of a by (wr, wr) onto swap_ri(a) times (-wi, wi)
    if (conj_w) {
      return {std::fma(ar, wr, ai * wi), std::fma(ai, wr, -(ar * wi))};
    }
    return {std::fma(ar, wr, -(ai * wi)), std::fma(ai, wr, ar * wi)};
  default: // no vocabulary yet: a result no test can match
    return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
  }
}

// acc + a * w, as target t computes it.
template <class T> cnum<T> multiply_add(fftn::detail::target t, cnum<T> acc, cnum<T> a, cnum<T> w) {
  using fftn::detail::target;
  switch (t) {
  case target::scalar: {
    const cnum<T> p = product(t, a, w, false);
    return {acc.re + p.re, acc.im + p.im};
  }
  case target::neon_fcma:
  case target::neon:
    return {
        std::fma(-w.im, a.im, std::fma(w.re, a.re, acc.re)),
        std::fma(w.im, a.re, std::fma(w.re, a.im, acc.im)),
    };
  case target::avx2:
    return {
        std::fma(a.re, w.re, std::fma(a.im, -w.im, acc.re)),
        std::fma(a.im, w.re, std::fma(a.re, w.im, acc.im)),
    };
  default:
    return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
  }
}

} // namespace isa_test
