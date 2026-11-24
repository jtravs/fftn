// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Shapes: what a plan transforms. A shape is a kind (given by the element types), the
/// transformed axes and the batch dimensions, in the manner of FFTW's guru interface.
/// The exact meaning of every field is in docs/conventions.md.
#pragma once

#include "fftn/config.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
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
/// Throws `std::bad_alloc` when memory runs out (the axes and batch dimensions are vectors).
///
/// Guarantees: the shape has two axes and one batch dimension with default (tight) strides
/// and distances; nothing is validated until a plan is built from it.
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

/// A batch of `count` real planes of `ny` rows of `nx` values, transformed to planes of `ny`
/// rows of nx/2 + 1 complex values (the last axis halved), C-ordered as for planes_c2c. Out
/// of place both sides are tight; in place (options::in_place) each real row is padded to
/// the 2(nx/2 + 1) reals of its complex row.
///
/// Throws `std::bad_alloc` when memory runs out (the axes and batch dimensions are vectors).
///
/// Guarantees: the shape has two axes and one batch dimension with default (tight) strides
/// and distances; nothing is validated until a plan is built from it.
template <class T>
[[nodiscard]] shape<T, std::complex<T>> planes_r2c(std::size_t ny, std::size_t nx,
                                                   std::size_t count) {
  shape<T, std::complex<T>> s;
  s.axes.push_back(axis{.n = ny});
  s.axes.push_back(axis{.n = nx});
  s.batches.push_back(batch{.count = count});
  return s;
}

/// A batch of `count` planes of `ny` rows of nx/2 + 1 complex values, transformed to real
/// planes of `ny` rows of `nx` values. The layouts are those of planes_r2c.
///
/// Throws `std::bad_alloc` when memory runs out (the axes and batch dimensions are vectors).
///
/// Guarantees: the shape has two axes and one batch dimension with default (tight) strides
/// and distances; nothing is validated until a plan is built from it.
template <class T>
[[nodiscard]] shape<std::complex<T>, T> planes_c2r(std::size_t ny, std::size_t nx,
                                                   std::size_t count) {
  shape<std::complex<T>, T> s;
  s.dir = direction::backward;
  s.axes.push_back(axis{.n = ny});
  s.axes.push_back(axis{.n = nx});
  s.batches.push_back(batch{.count = count});
  return s;
}

/// A batch of `count` complex volumes of `nz` planes of `ny` rows of `nx` values, C-ordered:
/// x varies fastest, then y, then z, then the volume. The 3-D transform of each volume.
///
/// Throws `std::bad_alloc` when memory runs out (the axes and batch dimensions are vectors).
///
/// Guarantees: the shape has three axes and one batch dimension with default (tight) strides
/// and distances; nothing is validated until a plan is built from it.
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

/// A batch of `count` real volumes of `nz` planes of `ny` rows of `nx` values, transformed to
/// volumes whose rows hold nx/2 + 1 complex values (the last axis halved), C-ordered as for
/// volumes_c2c. Out of place both sides are tight; in place (options::in_place) each real row
/// is padded to the 2(nx/2 + 1) reals of its complex row.
///
/// Throws `std::bad_alloc` when memory runs out (the axes and batch dimensions are vectors).
///
/// Guarantees: the shape has three axes and one batch dimension with default (tight) strides
/// and distances; nothing is validated until a plan is built from it.
template <class T>
[[nodiscard]] shape<T, std::complex<T>> volumes_r2c(std::size_t nz, std::size_t ny, std::size_t nx,
                                                    std::size_t count) {
  shape<T, std::complex<T>> s;
  s.axes.push_back(axis{.n = nz});
  s.axes.push_back(axis{.n = ny});
  s.axes.push_back(axis{.n = nx});
  s.batches.push_back(batch{.count = count});
  return s;
}

/// A batch of `count` volumes whose rows hold nx/2 + 1 complex values, transformed to real
/// volumes of `nz` planes of `ny` rows of `nx` values. The layouts are those of volumes_r2c.
///
/// Throws `std::bad_alloc` when memory runs out (the axes and batch dimensions are vectors).
///
/// Guarantees: the shape has three axes and one batch dimension with default (tight) strides
/// and distances; nothing is validated until a plan is built from it.
template <class T>
[[nodiscard]] shape<std::complex<T>, T> volumes_c2r(std::size_t nz, std::size_t ny, std::size_t nx,
                                                    std::size_t count) {
  shape<std::complex<T>, T> s;
  s.dir = direction::backward;
  s.axes.push_back(axis{.n = nz});
  s.axes.push_back(axis{.n = ny});
  s.axes.push_back(axis{.n = nx});
  s.batches.push_back(batch{.count = count});
  return s;
}

namespace detail {

// The stride of each dimension of an array on one side: the given one, or where it is 0 the
// tight C-ordered default, in which dimension d spans the dimensions after it. `halved_dim` is
// the dimension shortened to `halved_count` elements on this side (SIZE_MAX for none).
inline std::vector<std::ptrdiff_t> array_strides(std::span<const axis> array, bool input,
                                                 std::size_t halved_dim, std::size_t halved_count) {
  std::vector<std::ptrdiff_t> stride(array.size());
  std::size_t span = 1;
  for (std::size_t i = array.size(); i > 0; --i) {
    const std::size_t d = i - 1;
    const std::ptrdiff_t given = input ? array[d].in_stride : array[d].out_stride;
    stride[d] = given != 0 ? given : static_cast<std::ptrdiff_t>(span);
    if (__builtin_mul_overflow(span, d == halved_dim ? halved_count : array[d].n, &span) ||
        span > static_cast<std::size_t>(PTRDIFF_MAX)) {
      throw std::invalid_argument("fftn::axes: the array is too large to address");
    }
  }
  return stride;
}

// The shape of the dimensions `selected` of `array`; `halved_in` and `halved_out` are the
// element counts of the last selected dimension on each side (its own n where it is not
// halved). See axes_c2c.
template <class In, class Out>
shape<In, Out> shape_along_axes(std::span<const axis> array, std::span<const std::size_t> selected,
                                direction dir, std::size_t halved_in, std::size_t halved_out) {
  for (std::size_t d = 0; d < array.size(); ++d) {
    if (array[d].n == 0) {
      throw std::invalid_argument("fftn::axes: array dimension " + std::to_string(d) +
                                  " has length 0; every dimension needs its length");
    }
  }
  std::vector<bool> transformed(array.size(), false);
  for (const std::size_t d : selected) {
    if (d >= array.size()) {
      throw std::invalid_argument("fftn::axes: axes lists dimension " + std::to_string(d) +
                                  " of an array of " + std::to_string(array.size()));
    }
    if (transformed[d]) {
      throw std::invalid_argument("fftn::axes: axes lists dimension " + std::to_string(d) +
                                  " twice");
    }
    transformed[d] = true;
  }
  const std::size_t halved_dim = selected.empty() ? SIZE_MAX : selected.back();
  const std::vector<std::ptrdiff_t> in = array_strides(array, true, halved_dim, halved_in);
  const std::vector<std::ptrdiff_t> out = array_strides(array, false, halved_dim, halved_out);
  shape<In, Out> s;
  s.dir = dir;
  for (const std::size_t d : selected) {
    s.axes.push_back(axis{.n = array[d].n, .in_stride = in[d], .out_stride = out[d]});
  }
  // The other dimensions, widest input stride first, ties in the array's order; a dimension
  // of extent 1 never steps. Neighbours that nest exactly on both sides become one.
  std::vector<batch> loops;
  for (std::size_t d = 0; d < array.size(); ++d) {
    if (!transformed[d] && array[d].n != 1) {
      loops.push_back(batch{.count = array[d].n, .in_dist = in[d], .out_dist = out[d]});
    }
  }
  const auto magnitude = [](std::ptrdiff_t v) { return v < 0 ? -v : v; };
  std::ranges::stable_sort(loops, [&](const batch& a, const batch& b) {
    return magnitude(a.in_dist) > magnitude(b.in_dist);
  });
  for (const batch& inner : loops) {
    if (!s.batches.empty()) {
      batch& outer = s.batches.back();
      // Merged only where the spans are representable; a product past PTRDIFF_MAX never
      // equals a valid distance, so such neighbours stay two batch dimensions.
      const auto c = static_cast<std::ptrdiff_t>(inner.count);
      std::ptrdiff_t in_span = 0;
      std::ptrdiff_t out_span = 0;
      if (!__builtin_mul_overflow(inner.in_dist, c, &in_span) &&
          !__builtin_mul_overflow(inner.out_dist, c, &out_span) && outer.in_dist == in_span &&
          outer.out_dist == out_span) {
        outer = batch{
            .count = outer.count * inner.count,
            .in_dist = inner.in_dist,
            .out_dist = inner.out_dist,
        };
        continue;
      }
    }
    s.batches.push_back(inner);
  }
  return s;
}

} // namespace detail

/// The complex transform along some dimensions of a strided array. Provisional: the
/// interface may change before the first release (dev/DESIGN.md 5.2).
///
/// `array` describes the whole array: one `axis` per dimension, slowest first, with its
/// length and its strides. A stride of 0 is the tight C-ordered default for that dimension of
/// the array (each dimension spans those after it), worked out here over all the dimensions,
/// selected or not. Strides may be negative, and dimensions need not nest in order.
///
/// `axes` lists the dimensions to transform, by index into `array`, one to three of them, in
/// any order: `shape::axes` takes them in that order. Every other dimension becomes a batch
/// dimension, widest input stride first (ties in the array's order); dimensions of length 1
/// are dropped, and neighbours that nest exactly on both sides are merged. Transforms are
/// numbered by those batch dimensions, last fastest (`plan::execute_transforms`), so the
/// numbering follows memory rather than the array's dimension order.
///
/// Example: dimensions {1, 2} of an array `a` of shape {nz, ny, nx}, C-ordered and tight, are
/// `nz` planes; dimensions {0, 1} are `nx` planes `nx` elements apart in memory, transformed
/// without moving the data.
///
/// Throws `std::invalid_argument` when a dimension of `array` has length 0, when `axes` names a
/// dimension outside `array` or names one twice, or when the default strides do not fit
/// `std::ptrdiff_t`; `std::bad_alloc` when memory runs out.
///
/// Guarantees: every stride and distance of the result is explicit (nonzero), the strides of a
/// selected axis of length 1 included, though a plan never uses those. Nothing else is
/// validated until a plan is built from it.
template <class T>
[[nodiscard]] shape<std::complex<T>, std::complex<T>>
axes_c2c(std::span<const axis> array, std::span<const std::size_t> axes, direction dir) {
  const std::size_t n = axes.empty() || axes.back() >= array.size() ? 0 : array[axes.back()].n;
  return detail::shape_along_axes<std::complex<T>, std::complex<T>>(array, axes, dir, n, n);
}

/// The real-to-complex transform along some dimensions of a strided array. Provisional, as
/// axes_c2c.
///
/// As axes_c2c, for an array of real values: `array` holds the real side's lengths, and the
/// last dimension listed in `axes` is the halved one, with n/2 + 1 complex values on the
/// complex side, whose default strides are worked out with that length.
///
/// `padded` chooses the real side's default strides: false, tight rows of n reals; true, FFTW's
/// padded layout, the halved dimension taken as 2(n/2 + 1) reals, the last two padding. An
/// in-place plan (options::in_place) needs `padded` true and the halved dimension the array's
/// fastest (docs/conventions.md, "In place"), and refuses the shape otherwise. `padded` true with
/// an out-of-place plan is a valid layout too: the plan then reads (or writes) the padded real
/// rows, leaving the padding alone. Pass the same value as options::in_place unless you mean that.
///
/// Exceptions and guarantees: as for axes_c2c.
template <class T>
[[nodiscard]] shape<T, std::complex<T>>
axes_r2c(std::span<const axis> array, std::span<const std::size_t> axes, bool padded = false) {
  const std::size_t n = axes.empty() || axes.back() >= array.size() ? 0 : array[axes.back()].n;
  return detail::shape_along_axes<T, std::complex<T>>(array, axes, direction::forward,
                                                      padded ? 2 * (n / 2 + 1) : n, n / 2 + 1);
}

/// The complex-to-real transform along some dimensions of a strided array, the inverse layout
/// of axes_r2c: `array` holds the real side's lengths, the last dimension listed in `axes` has
/// n/2 + 1 complex values on the input side; `padded` as for axes_r2c, the output's real side.
/// Provisional, as axes_c2c.
///
/// Exceptions and guarantees: as for axes_c2c.
template <class T>
[[nodiscard]] shape<std::complex<T>, T>
axes_c2r(std::span<const axis> array, std::span<const std::size_t> axes, bool padded = false) {
  const std::size_t n = axes.empty() || axes.back() >= array.size() ? 0 : array[axes.back()].n;
  return detail::shape_along_axes<std::complex<T>, T>(array, axes, direction::backward, n / 2 + 1,
                                                      padded ? 2 * (n / 2 + 1) : n);
}

} // namespace fftn::inline v0
