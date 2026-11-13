// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Checks against FFTW (DESIGN.md 15.1, oracle 3): a subject's outputs for random inputs
// compared with FFTW's for the same inputs, at every output. Both are computed transforms,
// so the tolerance is the sum of the two derived tolerances (tolerance.hpp). Every run also
// checks the memory contracts (harness.hpp).
//
// Only call these when the build has FFTW (fftw_subject() is not empty).

#pragma once

#include "support/checks.hpp"
#include "support/fftw_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/transform_under_test.hpp"

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace fftn::test::fftw_checks {

// The tolerance of a comparison of `s` with FFTW for transforms of `points` points.
template <class T>
tolerance tolerance_for(const subject<T>& s, const subject<T>& fftw, std::size_t points) {
  return test::tolerance_for(s, points) + test::tolerance_for(fftw, points);
}

template <class T> outcome c2c(const subject<T>& s, std::size_t n) {
  outcome o;
  const subject<T> fftw = *fftw_subject<T>();
  input_source src(0xF0C2C00ULL + n);
  const auto x = src.complexes<T>(n);
  for (const direction d : {direction::forward, direction::backward}) {
    const auto what = describe("c2c against FFTW, ", name_of(d));
    const auto y = c2c_once<T>(o, s, d, x, what);
    const auto want = c2c_once<T>(o, fftw, d, x, what);
    o.within(compare<T>(y, as_expected<T>(want)), tolerance_for(s, fftw, n), what);
  }
  return o;
}

template <class T> outcome r2c(const subject<T>& s, std::size_t n) {
  outcome o;
  const subject<T> fftw = *fftw_subject<T>();
  input_source src(0xF052C00ULL + n);
  const auto x = src.reals<T>(n);
  const auto y = r2c_once<T>(o, s, x, "r2c against FFTW");
  const auto want = r2c_once<T>(o, fftw, x, "r2c against FFTW");
  o.within(compare<T>(y, as_expected<T>(want)), tolerance_for(s, fftw, n), "r2c against FFTW");
  return o;
}

// The input has imaginary parts at bins 0 and n/2, which both must ignore.
template <class T> outcome c2r(const subject<T>& s, std::size_t n) {
  outcome o;
  const subject<T> fftw = *fftw_subject<T>();
  input_source src(0xF0C2E00ULL + n);
  const auto X = src.complexes<T>(n / 2 + 1);
  const auto y = c2r_once<T>(o, s, n, X, "c2r against FFTW");
  const auto want = c2r_once<T>(o, fftw, n, X, "c2r against FFTW");
  o.within(compare<T>(std::span<const T>(y), as_expected_real<T>(want)), tolerance_for(s, fftw, n),
           "c2r against FFTW");
  return o;
}

template <class T> outcome planes(const subject<T>& s, std::size_t ny, std::size_t nx) {
  outcome o;
  const subject<T> fftw = *fftw_subject<T>();
  input_source src(0xF02D000ULL + ny * 4099 + nx);
  const auto x = src.complexes<T>(ny * nx);
  for (const direction d : {direction::forward, direction::backward}) {
    const auto what = describe("planes against FFTW, ", name_of(d));
    const auto y = plane_once<T>(o, s, d, ny, nx, x, what);
    const auto want = plane_once<T>(o, fftw, d, ny, nx, x, what);
    o.within(compare<T>(y, as_expected<T>(want)), tolerance_for(s, fftw, ny * nx), what);
  }
  return o;
}

template <class T> std::vector<row_check<T>> row_checks() {
  return {
      {"fftw_c2c", kind::c2c, &c2c<T>},
      {"fftw_r2c", kind::r2c, &r2c<T>},
      {"fftw_c2r", kind::c2r, &c2r<T>},
  };
}

template <class T> std::vector<plane_check<T>> plane_checks() {
  return {{"fftw_planes", &planes<T>}};
}

} // namespace fftn::test::fftw_checks
