// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Roots of unity for twiddle tables (DESIGN.md 8.4): each value computed on its own, with
// the angle reduced to an octant in integers, evaluated in double-double arithmetic and
// rounded once to the precision. No recurrences, so no error accumulates along a table.
// Tests check every entry against the quadruple-precision reference
// (reference_checks::twiddle_table).
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace fftn::inline v0::detail {

// exp(sign 2 pi i m / n) for m < n, each component correctly rounded to T (to nearest)
// except with a probability of about 2^-50 per entry. sign is -1 or +1; the +1 table is
// the exact conjugate of the -1 table. T is double or float. Precondition: n >= 1.
template <class T> [[nodiscard]] std::vector<std::complex<T>> unit_roots(int sign, std::size_t n);

// Entry m of that table, computed alone (m < n).
template <class T> [[nodiscard]] std::complex<T> unit_root(int sign, std::size_t m, std::size_t n);

} // namespace fftn::inline v0::detail
