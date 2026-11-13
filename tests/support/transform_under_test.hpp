// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The transform under test: what the oracle suites are written against.
//
// A subject is a set of factories, one per transform kind. A factory takes a direction and
// a layout and returns a call object; building the call object is plan construction (it
// may allocate and may be slow), and invoking it is execution (the part the contract tests
// watch for allocation, memory outside the rows and input preservation). Any factory may be
// left empty; suites then skip that kind.
//
// Layouts follow FFTW's guru interface and DESIGN.md 5.2: strides and distances count
// elements of each side's own type (reals on the real side, complex values on the complex
// side), may be negative, and are relative to the pointer passed, which addresses element 0
// of row 0. For in-place calls the same address is passed as input and output.
//
// Task 0.5 plugs FFTN's plans in here; this task plugs in a naive DFT and FFTW.

#pragma once

#include "support/tolerance.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace fftn::test {

// The sign of the exponent, as in FFTW: forward is exp(-2 pi i jk / n).
enum class direction : std::int8_t { forward = -1, backward = +1 };

inline int sign_of(direction d) { return static_cast<int>(d); }

enum class placement : std::uint8_t { out_of_place, in_place };

// A batch of 1-D rows. For c2c both sides have n elements per row; for r2c the output has
// n/2 + 1; for c2r the input has n/2 + 1 and the output n.
struct rows {
  std::size_t n = 1;
  std::size_t count = 1;
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t in_dist = 1;
  std::ptrdiff_t out_stride = 1;
  std::ptrdiff_t out_dist = 1;
  placement place = placement::out_of_place;
};

// A batch of 2-D complex planes of ny x nx elements.
struct planes {
  std::size_t ny = 1;
  std::size_t nx = 1;
  std::size_t count = 1;
  std::ptrdiff_t in_y_stride = 1;
  std::ptrdiff_t in_x_stride = 1;
  std::ptrdiff_t in_dist = 1;
  std::ptrdiff_t out_y_stride = 1;
  std::ptrdiff_t out_x_stride = 1;
  std::ptrdiff_t out_dist = 1;
  placement place = placement::out_of_place;
};

enum class kind : std::uint8_t { c2c, r2c, c2r };

// Elements per row on each side.
inline std::size_t in_length(kind k, std::size_t n) { return k == kind::c2r ? n / 2 + 1 : n; }
inline std::size_t out_length(kind k, std::size_t n) { return k == kind::r2c ? n / 2 + 1 : n; }

// Contiguous rows, one after another. In-place real transforms use FFTW's padded layout
// (DESIGN.md 9.1): each real row occupies the n + 2 reals (n/2 + 1 complex values) of its
// complex row.
inline rows tight_rows(kind k, std::size_t n, std::size_t count,
                       placement place = placement::out_of_place) {
  const auto in_len = static_cast<std::ptrdiff_t>(in_length(k, n));
  const auto out_len = static_cast<std::ptrdiff_t>(out_length(k, n));
  const auto padded = static_cast<std::ptrdiff_t>(2 * (n / 2 + 1));
  rows r{.n = n, .count = count, .in_dist = in_len, .out_dist = out_len, .place = place};
  if (place == placement::in_place && k == kind::r2c) {
    r.in_dist = padded;
  }
  if (place == placement::in_place && k == kind::c2r) {
    r.out_dist = padded;
  }
  return r;
}

inline planes tight_planes(std::size_t ny, std::size_t nx, std::size_t count,
                           placement place = placement::out_of_place) {
  const auto x = static_cast<std::ptrdiff_t>(nx);
  const auto plane = static_cast<std::ptrdiff_t>(ny * nx);
  return {
      .ny = ny,
      .nx = nx,
      .count = count,
      .in_y_stride = x,
      .in_dist = plane,
      .out_y_stride = x,
      .out_dist = plane,
      .place = place,
  };
}

template <class T> using c2c_call = std::function<void(const std::complex<T>*, std::complex<T>*)>;
template <class T> using r2c_call = std::function<void(const T*, std::complex<T>*)>;
template <class T> using c2r_call = std::function<void(const std::complex<T>*, T*)>;

template <class T> struct subject {
  std::string name;
  std::function<c2c_call<T>(direction, const rows&)> c2c;
  std::function<r2c_call<T>(const rows&)> r2c;
  std::function<c2r_call<T>(const rows&)> c2r;
  std::function<c2c_call<T>(direction, const planes&)> c2c_planes;
  // The algorithm's error bound for a transform of n points in all (tolerance.hpp). Empty
  // means stable butterflies throughout (stable_fft_bound).
  std::function<error_bound(std::size_t)> bound;

  [[nodiscard]] error_bound bound_for(std::size_t n) const {
    return bound ? bound(n) : stable_fft_bound;
  }
};

} // namespace fftn::test
