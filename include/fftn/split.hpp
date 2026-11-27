// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Split-complex data: complex values kept as two arrays of reals, one of real parts and one of
/// imaginary parts, as FFTW's split interface takes them.
///
/// Provisional (task 6.10, dev/DESIGN.md 5.8): this API follows the design's block as written
/// and may still change before the first release. docs/conventions.md ("Split complex") states
/// the layout rules.
#pragma once

#include "fftn/config.hpp"

#include <type_traits>

namespace fftn::inline v0 {

/// The element tag of split-complex data in a shape and a plan, for `T` = `double` or `float`:
/// `shape<split_complex<T>, split_complex<T>>` is a complex transform (c2c) of split data,
/// `shape<T, split_complex<T>>` a real-to-complex one (r2c) whose output is split and
/// `shape<split_complex<T>, T>` a complex-to-real one (c2r) whose input is split. It names a
/// kind, not storage: no array of it is ever made, and the data of an execution arrive as a
/// `split_ptr`.
///
/// On a split side the shape's strides and distances count elements of `T` and hold for both
/// arrays alike; the two arrays may lie anywhere relative to each other (docs/conventions.md,
/// "Split complex").
///
/// Provisional (task 6.10).
template <class T> struct split_complex {};

/// The two arrays of one execution's split-complex side: `re` addresses the real part of
/// element 0, `im` its imaginary part, each laid out by the shape's strides and distances for
/// that side. `T` is `const double` or `const float` for an input, `double` or `float` for an
/// output; write `{re, im}` to pass two pointers. A `split_ptr<T>` converts to
/// `split_ptr<const T>`, so an output's arrays can be passed as an input.
///
/// Preconditions at execution: both arrays hold the planned layout and have at least `T`'s
/// alignment, and neither pointer is null; an output's two arrays overlap neither each other nor
/// any input array, unless the plan is in place, where each input array is the same as the output
/// array of its part (`in.re == out.re`, `in.im == out.im`). Overlap is not checked, since the
/// pointers arrive with each call; debug builds assert the equalities and inequalities of the
/// pointers themselves. An execution whose imaginary array is null ends the program
/// (`std::abort`, with a message on stderr naming `split_ptr::im`), as an execution without a
/// factor array an edge needs does (fftn/edges.hpp).
///
/// Provisional (task 6.10).
template <class T> struct split_ptr {
  /// The real parts.
  T* re = nullptr;
  /// The imaginary parts.
  T* im = nullptr;

  /// The same two arrays, read only. Provisional (task 6.10): a conversion function, so that
  /// `split_ptr` stays an aggregate.
  // NOLINTNEXTLINE(google-explicit-constructor): implicit, as const-qualifying a pointer is
  operator split_ptr<const T>() const noexcept
    requires(!std::is_const_v<T>)
  {
    return {re, im};
  }
};

} // namespace fftn::inline v0
