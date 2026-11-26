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
/// output; write `{re, im}` to pass two pointers.
///
/// Preconditions at execution (not checked: the pointers arrive with each call): both arrays
/// hold the planned layout and have at least `T`'s alignment; an output's two arrays overlap
/// neither each other nor any input array, unless the plan is in place, where each input array
/// is the same as the output array of its part (`in.re == out.re`, `in.im == out.im`).
///
/// Provisional (task 6.10).
template <class T> struct split_ptr {
  /// The real parts.
  T* re = nullptr;
  /// The imaginary parts.
  T* im = nullptr;
};

} // namespace fftn::inline v0
