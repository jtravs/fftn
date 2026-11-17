// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Transform lengths: the next length FFTN computes fast, and the planner's estimate of how
/// fast a length is.
#pragma once

#include "fftn/config.hpp"
#include "fftn/isa.hpp"

#include <cstddef>
#include <type_traits>

namespace fftn::inline v0 {

/// The smallest length m >= n that FFTN transforms fast: for an application that may pad its
/// grid, for example to 1008 instead of 1001 points, or to 4096 instead of the prime 4093.
///
/// A length is fast when the planner's reference cost table prices a row of that length,
/// per point, within 1.25 times a row of the power of two at or above it. Every prime factor
/// of a fast length is at most 23, a length with a factor 11 or 17 is fast only as that
/// prime alone (experiment P4 measured such composites slower than the table believes), and
/// a length that FFTN computes in a single codelet (up to 16 points, and 17, 19 and 23) is
/// fast.
///
/// The reference table is fixed for each library version (AVX2's double table), whatever
/// the machine, so a grid chosen on one machine is the same on every other. For this
/// machine's own estimate of a length, see `relative_cost`.
///
/// With `real`, the result is even and its half is fast: real transforms of an even length
/// run a complex transform of half that length.
///
/// Preconditions: none. Returns 1 for n = 0, n itself for n <= 2, and n itself above 2^40
/// (no search). The result is at most the power of two at or above n (powers of two are
/// fast), it is non-decreasing in n, and `next_fast_length(next_fast_length(n)) ==
/// next_fast_length(n)`.
///
/// ```cpp
/// const std::size_t n = fftn::next_fast_length(1001);        // 1008 = 2^4 * 3^2 * 7
/// const std::size_t r = fftn::next_fast_length(1001, true);  // even, half fast
/// const fftn::c2c_plan<double> p(fftn::rows_c2c<double>(n, 1, fftn::direction::forward));
/// ```
[[nodiscard]] std::size_t next_fast_length(std::size_t n, bool real = false);

namespace detail {
[[nodiscard]] double relative_cost(std::size_t n, bool single, isa target);
} // namespace detail

/// The planner's estimate of how long a complex row of n points of `T` (`double` or `float`)
/// takes per point, relative to a row of the power of two at or above n, on instruction set
/// `target`: about 1 for lengths as fast per point as a power of two, 2 for lengths about
/// twice as slow per point. `isa::best` (the default) takes the instruction set a plan would
/// use on this machine; another value prices that target's cost tables, which need not run
/// here.
///
/// It is the estimate the planner's tables give for rows in cache, not a measurement. Memory
/// beyond the first-level cache and the choice between a chain and a grid for long rows
/// are not priced, and the estimate is coarser for tables that were not measured on their
/// target. For a real row of n points, ask for its half length, n / 2.
///
/// Throws `std::invalid_argument` naming `target` for a value that is not an `isa`
/// enumerator. Returns 1 for n <= 2 and for lengths computed in a single codelet.
///
/// ```cpp
/// // How much slower per point than 4096 is the prime 4093 on this machine?
/// const double prime = fftn::relative_cost<double>(4093);   // several times
/// const double padded = fftn::relative_cost<double>(4096);  // 1
/// ```
template <class T> [[nodiscard]] double relative_cost(std::size_t n, isa target = isa::best) {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                "fftn::relative_cost: T is double or float");
  return detail::relative_cost(n, std::is_same_v<T, float>, target);
}

} // namespace fftn::inline v0
