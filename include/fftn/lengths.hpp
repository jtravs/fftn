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

/// What a length is for, in `next_fast_length`.
enum class length_kind : unsigned char {
  complex, ///< a complex transform's length, or a real transform's axis that is not halved
  real,    ///< the halved axis of a real transform: an even length with a fast half
};

/// The smallest length m >= n that FFTN transforms fast: for an application that may pad its
/// grid, for example to 1008 instead of 1001 points, or to 4096 instead of the prime 4093.
///
/// A length is fast when the planner's reference cost table prices a row of that length,
/// per point, within 1.25 times a row of the power of two at or above it. Every fast length
/// is a product of 2, 3, 5, 7 and 13 or one of the primes 11, 17, 19 and 23 alone (which
/// FFTN computes in a single codelet), but not every such product is fast: 27, 39 and 45 are
/// not. Composites with a factor 11, 17, 19 or 23 are never fast (experiment P4 measured
/// them slower than the table believes).
///
/// "Fast" is a property of the reference table, not a measured guarantee on every machine.
/// The table is a frozen copy of AVX2's double table, fixed for each library version
/// whatever the machine and apart from the tables the planner tunes, so a grid chosen on
/// one machine is the same on every other. Measured (P4), most fast lengths run within 1.3
/// times the power of two above on x86 (78 % of those measured), and a few do not: 140, the
/// pick for 131, ran at 1.76 on one Xeon. Every pick measured was faster than the length it
/// replaced. For this machine's own estimate of a length, see `relative_cost`.
///
/// With `length_kind::real`, the result is even and its half is fast: real transforms of an
/// even length run a complex transform of half that length. In two or three dimensions only
/// the halved axis (the last) takes `length_kind::real`; the others take
/// `next_fast_length(n)`. (scipy's `next_fast_len(target, real=True)` differs: it may return
/// an odd length.)
///
/// Throws `std::invalid_argument` for n above 2^40, the longest length the length queries
/// accept, and naming `kind` for a value that is not a `length_kind` enumerator. Returns 1
/// for n = 0, and with `length_kind::real` 2 for n <= 2. The result is at most the
/// power of two at or above n (powers of two are fast), it is non-decreasing in n, and
/// `next_fast_length(next_fast_length(n)) == next_fast_length(n)`. A call tests only the
/// candidates in [n, 2n] and costs microseconds, under a millisecond up to 2^40.
///
/// ```cpp
/// const std::size_t n = fftn::next_fast_length(1001);        // 1008 = 2^4 * 3^2 * 7
/// const std::size_t r = fftn::next_fast_length(1001, fftn::length_kind::real); // even
/// const fftn::c2c_plan<double> p(fftn::rows_c2c<double>(n, 1, fftn::direction::forward));
/// ```
[[nodiscard]] std::size_t next_fast_length(std::size_t n, length_kind kind = length_kind::complex);

namespace detail {
[[nodiscard]] double relative_cost(std::size_t n, bool single, isa which);
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
/// enumerator, and naming the limit for n above 2^40. Returns 1 for n <= 2 and for lengths
/// the planner computes in a single codelet on that target (up to 16 points, 17, 19 and 23,
/// and 32 on NEON in double; not 20, which is always a chain of passes).
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
