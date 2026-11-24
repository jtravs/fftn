// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Roots of unity for twiddle tables (DESIGN.md 8.4): each value computed on its own, with
// the angle reduced to an octant in integers, evaluated in double-double arithmetic and
// rounded once to the precision. No recurrences, so no error accumulates along a table.
// Every entry is the same function of (m, n) however the table is built (octant symmetry,
// copies from the table of n / 2, entries alone), so tables agree with each other bit for
// bit: entry m 2^k of the table of n 2^k is entry m of the table of n.
// Tests check every entry against the quadruple-precision reference
// (reference_checks::twiddle_table).
#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace fftn::inline v0::detail {

// exp(sign 2 pi i m / n) for m < n, each component correctly rounded to T (to nearest)
// except with a probability of about 2^-50 per entry. sign is -1 or +1; the +1 table is
// the exact conjugate of the -1 table. T is double or float. Precondition: n >= 1.
template <class T> [[nodiscard]] std::vector<std::complex<T>> unit_roots(int sign, std::size_t n);

// The same table for even n, its even entries copied from `half`, the table of n / 2 of the
// same sign: entry 2 m of n and entry m of n / 2 are equal bit for bit, so the result is
// that of unit_roots(sign, n), for half the work.
template <class T>
[[nodiscard]] std::vector<std::complex<T>> unit_roots(int sign, std::size_t n,
                                                      std::span<const std::complex<T>> half);

// Entry m of that table, computed alone (m < n).
template <class T> [[nodiscard]] std::complex<T> unit_root(int sign, std::size_t m, std::size_t n);

// Entries of the table of n one at a time, bit for bit unit_root's, for callers that want
// many of them but not the whole table. For n divisible by 8 the sine and cosine of each
// reduced angle are kept once computed, so the up to four entries that share one (octant
// symmetry) cost one series evaluation: 33 bytes for each of the n / 8 + 1 angles, held for
// the cache's life. Otherwise every entry is computed alone. Not safe to share between
// threads; a planning call makes its own.
template <class T> class root_cache {
public:
  explicit root_cache(std::size_t n);
  [[nodiscard]] std::complex<T> operator()(int sign, std::size_t m);

private:
  std::size_t n_;
  std::vector<std::array<double, 4>> angles_; // per angle: sine (hi, lo), cosine (hi, lo)
  std::vector<unsigned char> known_;
};

} // namespace fftn::inline v0::detail
