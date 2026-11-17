// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Shapes: what a plan transforms. A shape is a kind (given by the element types), the
/// transformed axes and the batch dimensions, in the manner of FFTW's guru interface.
/// The exact meaning of every field is in docs/conventions.md.
#pragma once

#include "fftn/config.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace fftn::inline v0 {

/// The sign of the exponent of a complex transform, as in FFTW.
///
/// `forward` computes y[k] = sum_j x[j] exp(-2 pi i jk / n); `backward` uses +i. Neither
/// normalises: `backward` after `forward` multiplies by n.
enum class direction : std::int8_t {
  forward = -1,  ///< exponent -2 pi i jk / n
  backward = +1, ///< exponent +2 pi i jk / n
};

/// One transformed dimension of a shape.
///
/// Strides count elements of each side's own type: real values on the real side of a real
/// transform, complex values on the complex side. They may be negative. Zero means the
/// tight C-ordered default (docs/conventions.md). The strides of an axis with `n` = 1 are
/// never used and are ignored, whatever their value. `n` must be given: it defaults to 0,
/// which a plan refuses, so that `.axes{2}` (two value-initialised axes, not one of length
/// 2) is caught; write `.axes = {{2}}` or `.axes = {{.n = 2}}`.
struct axis {
  /// The logical length: the number of points transformed along this axis. At least 1.
  std::size_t n = 0;
  /// Distance between consecutive input elements along this axis; 0 for the default.
  std::ptrdiff_t in_stride = 0;
  /// Distance between consecutive output elements along this axis; 0 for the default.
  std::ptrdiff_t out_stride = 0;
};

/// One batch dimension of a shape: a loop over independent transforms.
///
/// Distances count elements of each side's own type, may be negative, and zero means the
/// tight C-ordered default (docs/conventions.md). The distances of a dimension with
/// `count` = 1 are never used and are ignored, whatever their value. `count` must be given:
/// it defaults to 0, which a plan refuses, so that `.batches{8}` (eight value-initialised
/// batch dimensions, not one of count 8) is caught; write `.batches = {{8}}`.
struct batch {
  /// The number of transforms along this dimension. At least 1.
  std::size_t count = 0;
  /// Distance between the first input elements of consecutive transforms; 0 for the default.
  std::ptrdiff_t in_dist = 0;
  /// Distance between the first output elements of consecutive transforms; 0 for the
  /// default.
  std::ptrdiff_t out_dist = 0;
};

/// A transform shape, typed by its input and output elements.
///
/// The element types select the kind: `shape<std::complex<T>, std::complex<T>>` is a
/// complex transform (c2c), `shape<T, std::complex<T>>` a real-to-complex one (r2c) and
/// `shape<std::complex<T>, T>` a complex-to-real one (c2r), for `T` = `double` or `float`.
/// Other combinations do not compile.
///
/// Axes are listed slowest-varying first, as the dimensions of a C array; for real kinds
/// the last axis is the halved one (n/2 + 1 complex values on the complex side). Batch
/// dimensions are listed slowest first too, and transforms are numbered in that order (the
/// last batch dimension varies fastest).
///
/// A plan validates the shape when it is constructed (`plan`), not here.
template <class In, class Out> struct shape {
  static_assert(
      (std::is_same_v<In, std::complex<double>> && std::is_same_v<Out, std::complex<double>>) ||
          (std::is_same_v<In, double> && std::is_same_v<Out, std::complex<double>>) ||
          (std::is_same_v<In, std::complex<double>> && std::is_same_v<Out, double>) ||
          (std::is_same_v<In, std::complex<float>> && std::is_same_v<Out, std::complex<float>>) ||
          (std::is_same_v<In, float> && std::is_same_v<Out, std::complex<float>>) ||
          (std::is_same_v<In, std::complex<float>> && std::is_same_v<Out, float>),
      "fftn::shape<In, Out>: the kinds are complex to complex, real to complex and complex to "
      "real, in double or float");

  /// The direction of a complex-to-complex transform. Ignored by the real kinds: r2c is
  /// always forward and c2r always backward.
  direction dir = direction::forward;
  /// The transformed axes, slowest first: one for rows, two for planes, three for volumes.
  /// A plan refuses a shape with none or with more than three.
  std::vector<axis> axes;
  /// The batch dimensions, slowest first; none means a single transform.
  std::vector<batch> batches;
};

/// A batch of `count` complex rows of length `n`, stored one after another.
///
/// Throws `std::bad_alloc` when memory runs out (the axes and batch dimensions are vectors).
///
/// Guarantees: the shape has one axis and one batch dimension with default (tight) strides
/// and distances; nothing is validated until a plan is built from it.
template <class T>
[[nodiscard]] shape<std::complex<T>, std::complex<T>> rows_c2c(std::size_t n, std::size_t count,
                                                               direction dir) {
  shape<std::complex<T>, std::complex<T>> s;
  s.dir = dir;
  s.axes.push_back(axis{.n = n});
  s.batches.push_back(batch{.count = count});
  return s;
}

/// A batch of `count` real rows of length `n`, transformed to rows of n/2 + 1 complex
/// values. Out of place the rows are tight on both sides; in place (options::in_place) each
/// real row is padded to the 2(n/2 + 1) reals of its complex row.
///
/// Exceptions and guarantees: as for rows_c2c.
template <class T>
[[nodiscard]] shape<T, std::complex<T>> rows_r2c(std::size_t n, std::size_t count) {
  shape<T, std::complex<T>> s;
  s.axes.push_back(axis{.n = n});
  s.batches.push_back(batch{.count = count});
  return s;
}

/// A batch of `count` rows of n/2 + 1 complex values, transformed to real rows of length
/// `n`. The layouts are those of rows_r2c.
///
/// Exceptions and guarantees: as for rows_c2c.
template <class T>
[[nodiscard]] shape<std::complex<T>, T> rows_c2r(std::size_t n, std::size_t count) {
  shape<std::complex<T>, T> s;
  s.dir = direction::backward;
  s.axes.push_back(axis{.n = n});
  s.batches.push_back(batch{.count = count});
  return s;
}

/// A batch of `count` complex planes of `ny` rows of `nx` values, C-ordered: x varies
/// fastest, then y, then the plane. The 2-D transform of each plane.
///
/// Exceptions and guarantees: as for rows_c2c.
template <class T>
[[nodiscard]] shape<std::complex<T>, std::complex<T>> planes_c2c(std::size_t ny, std::size_t nx,
                                                                 std::size_t count, direction dir) {
  shape<std::complex<T>, std::complex<T>> s;
  s.dir = dir;
  s.axes.push_back(axis{.n = ny});
  s.axes.push_back(axis{.n = nx});
  s.batches.push_back(batch{.count = count});
  return s;
}

/// A batch of `count` complex volumes of `nz` planes of `ny` rows of `nx` values, C-ordered:
/// x varies fastest, then y, then z, then the volume. The 3-D transform of each volume.
/// (Provisional: proposed with task 4.2, pending the maintainer's agreement.)
///
/// Exceptions and guarantees: as for rows_c2c.
template <class T>
[[nodiscard]] shape<std::complex<T>, std::complex<T>>
volumes_c2c(std::size_t nz, std::size_t ny, std::size_t nx, std::size_t count, direction dir) {
  shape<std::complex<T>, std::complex<T>> s;
  s.dir = dir;
  s.axes.push_back(axis{.n = nz});
  s.axes.push_back(axis{.n = ny});
  s.axes.push_back(axis{.n = nx});
  s.batches.push_back(batch{.count = count});
  return s;
}

} // namespace fftn::inline v0
