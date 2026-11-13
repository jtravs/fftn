// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Checks against the quadruple-precision reference (DESIGN.md 15.1, oracle 2): a subject's
// outputs for random inputs compared with the direct DFT, at every output for n <= 64 and
// at a sample of outputs beyond (the reference is O(n) per output). Also the entry-by-entry
// check of twiddle tables.

#pragma once

#include "reference/reference_dft.hpp"
#include "support/checks.hpp"
#include "support/inputs.hpp"
#include "support/lengths.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace fftn::test::reference_checks {

// The outputs compared: all of them up to 64, else 32 of them: both ends, the middle and a
// fixed pseudo-random sample.
inline std::vector<std::size_t> sampled(std::size_t n) {
  std::vector<std::size_t> v;
  if (n <= 64) {
    for (std::size_t k = 0; k < n; ++k) {
      v.push_back(k);
    }
    return v;
  }
  for (const std::size_t k : {
           std::size_t{0},
           std::size_t{1},
           std::size_t{2},
           n / 2 - 1,
           n / 2,
           n / 2 + 1,
           n - 2,
           n - 1,
       }) {
    v.push_back(k);
  }
  std::uint64_t state = 0x2545F4914F6CDD1DULL ^ n;
  while (v.size() < 32) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    v.push_back(static_cast<std::size_t>((state >> 33U) % n));
  }
  std::ranges::sort(v);
  const auto dup = std::ranges::unique(v);
  v.erase(dup.begin(), dup.end());
  return v;
}

// Lengths for the reference suite. Fast subjects get fast_lengths() (lengths.hpp): every
// n <= 512 and a sample of larger lengths, compared at every output up to 64 and at the
// outputs sampled() picks beyond. O(n^2) subjects get a shorter list. The reference costs
// about 0.14 us per term, so the sampled outputs keep a length of 16384 to a few
// milliseconds. Under ThreadSanitizer both lists are shorter, for the reason given in
// lengths.hpp.
inline std::vector<std::size_t> lengths(bool fast) {
  if (under_thread_sanitizer) {
    auto v = every_length_up_to(32);
    append(v, {64, 97, 128, 1000});
    if (fast) {
      append(v, {4096, 10007});
    }
    return v;
  }
  if (fast) {
    return fast_lengths();
  }
  auto v = every_length_up_to(64);
  append(v, {72, 81, 96, 97, 100, 125, 128, 210, 243, 256});
  append(v, {289, 509, 512, 1000, 1024, 2002, 2048});
  return v;
}

// The elements of v at the given indices.
template <class E> std::vector<E> pick(const std::vector<E>& v, std::span<const std::size_t> idx) {
  std::vector<E> out;
  out.reserve(idx.size());
  for (const std::size_t i : idx) {
    out.push_back(v[i]);
  }
  return out;
}

template <class T> outcome c2c(const subject<T>& s, std::size_t n) {
  outcome o;
  const auto bins = sampled(n);
  input_source src(0xC2C0000ULL + n);
  const auto x = src.complexes<T>(n);
  for (const direction d : {direction::forward, direction::backward}) {
    const auto what = describe("c2c against the reference, ", name_of(d));
    const auto y = c2c_once<T>(o, s, d, x, what);
    o.within(compare<T>(pick(y, bins), reference::c2c(sign_of(d), widen(x), bins)),
             tolerance_for(s, n), what);
  }
  return o;
}

template <class T> outcome r2c(const subject<T>& s, std::size_t n) {
  outcome o;
  const auto bins = sampled(n / 2 + 1);
  input_source src(0x52C0000ULL + n);
  const auto x = src.reals<T>(n);
  const auto y = r2c_once<T>(o, s, x, "r2c against the reference");
  o.within(compare<T>(pick(y, bins), reference::r2c(widen(x), bins)), tolerance_for(s, n),
           "r2c against the reference");
  return o;
}

// The input has imaginary parts at bins 0 and n/2, which c2r must ignore.
template <class T> outcome c2r(const subject<T>& s, std::size_t n) {
  outcome o;
  const auto idx = sampled(n);
  input_source src(0xC2E0000ULL + n);
  const auto X = src.complexes<T>(n / 2 + 1);
  const auto y = c2r_once<T>(o, s, n, X, "c2r against the reference");
  o.within(compare<T>(std::span<const T>(pick(y, idx)), reference::c2r(n, widen(X), idx)),
           tolerance_for(s, n), "c2r against the reference");
  return o;
}

template <class T> outcome planes(const subject<T>& s, std::size_t ny, std::size_t nx) {
  outcome o;
  const auto ky_sample = sampled(ny);
  const auto kx_sample = sampled(nx);
  std::vector<std::pair<std::size_t, std::size_t>> points;
  points.reserve(ky_sample.size() * kx_sample.size());
  for (const std::size_t ky : ky_sample) {
    for (const std::size_t kx : kx_sample) {
      points.emplace_back(ky, kx);
    }
  }
  input_source src(0x2D00000ULL + ny * 4099 + nx);
  const auto x = src.complexes<T>(ny * nx);
  for (const direction d : {direction::forward, direction::backward}) {
    const auto what = describe("planes against the reference, ", name_of(d));
    const auto y = plane_once<T>(o, s, d, ny, nx, x, what);
    std::vector<std::complex<T>> got;
    got.reserve(points.size());
    for (const auto& [ky, kx] : points) {
      got.push_back(y[ky * nx + kx]);
    }
    o.within(compare<T>(got, reference::c2c_plane(sign_of(d), ny, nx, widen(x), points)),
             tolerance_for(s, ny * nx), what);
  }
  return o;
}

template <class T> std::vector<row_check<T>> row_checks() {
  return {
      {"reference_c2c", kind::c2c, &c2c<T>},
      {"reference_r2c", kind::r2c, &r2c<T>},
      {"reference_c2r", kind::c2r, &c2r<T>},
  };
}

template <class T> std::vector<plane_check<T>> plane_checks() {
  return {{"reference_planes", &planes<T>}};
}

// Every entry of a twiddle table w[m] = exp(sign 2 pi i m / n) must be correctly rounded:
// within half an ulp of the exact value in each component. A table entry one ulp off is
// invisible in transform outputs (tolerance.hpp), so tables are checked here directly.
// Task 0.5 must run this on every twiddle table FFTN builds, in both precisions and both
// directions, for every length its suites test (DESIGN.md 15.1, oracle 2); the suites'
// output tolerances do not cover a single wrong entry.
template <class T>
outcome twiddle_table(std::span<const std::complex<T>> table, int sign, std::size_t n) {
  outcome o;
  if (table.size() != n) {
    o.fail(describe("twiddle table has ", table.size(), " entries, expected ", n));
    return o;
  }
  const auto& exact = reference::unit_roots(sign, n);
  // Half an ulp, with room for the 2^-53 relative error of measuring it in double.
  constexpr double allowed = 0.5 * (1 + 1e-9);
  for (std::size_t m = 0; m < n; ++m) {
    const double ure = std::abs(error_of(table[m].real(), exact[m].re)) / ulp_of<T>(exact[m].re.hi);
    const double uim = std::abs(error_of(table[m].imag(), exact[m].im)) / ulp_of<T>(exact[m].im.hi);
    if (ure > allowed || uim > allowed) {
      o.fail(describe("twiddle ", m, " of ", n, " is ", std::max(ure, uim), " ulp from exp(", sign,
                      " 2 pi i ", m, "/", n, ")"));
    }
  }
  return o;
}

} // namespace fftn::test::reference_checks
