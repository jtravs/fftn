// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// A vector of fixed capacity that never allocates, used for the axes and batch
/// dimensions of a shape.
#pragma once

#include "fftn/config.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace fftn::inline v0 {

/// A sequence of at most `N` values of `T`, stored inline (no heap allocation).
///
/// It behaves like a `std::vector` that cannot grow beyond `N`: values are appended with
/// `push_back` or given as a braced list, and read by index or by iteration. Its storage
/// holds `N` default-constructed values, so `T` must be default-constructible; copying a
/// `small_vector` copies all `N`.
///
/// Guarantees: never allocates; adding a value beyond the capacity throws
/// `std::length_error` and leaves the vector unchanged.
template <class T, std::size_t N> class small_vector {
public:
  /// An empty vector.
  constexpr small_vector() noexcept = default;

  /// A vector holding `values` in order.
  ///
  /// Throws `std::length_error` when `values` has more than `N` elements.
  constexpr small_vector(std::initializer_list<T> values) {
    if (values.size() > N) {
      throw std::length_error("fftn::small_vector: more values than its capacity");
    }
    for (const T& v : values) {
      items_[size_++] = v;
    }
  }

  /// Appends `value`.
  ///
  /// Throws `std::length_error` when the vector already holds `N` values.
  constexpr void push_back(const T& value) {
    if (size_ == N) {
      throw std::length_error("fftn::small_vector: push_back beyond its capacity");
    }
    items_[size_++] = value;
  }

  /// The number of values held, at most `N`.
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }

  /// Whether the vector holds no value.
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  /// The capacity, `N`.
  [[nodiscard]] static constexpr std::size_t capacity() noexcept { return N; }

  /// The value at `index`. Precondition: `index < size()`.
  [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept { return items_[index]; }

  /// The value at `index`. Precondition: `index < size()`.
  [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
    return items_[index];
  }

  /// Pointer to the first value; the values are contiguous.
  [[nodiscard]] constexpr T* data() noexcept { return items_.data(); }

  /// Pointer to the first value; the values are contiguous.
  [[nodiscard]] constexpr const T* data() const noexcept { return items_.data(); }

  /// Iterator to the first value.
  [[nodiscard]] constexpr T* begin() noexcept { return items_.data(); }

  /// Iterator past the last value.
  [[nodiscard]] constexpr T* end() noexcept { return items_.data() + size_; }

  /// Iterator to the first value.
  [[nodiscard]] constexpr const T* begin() const noexcept { return items_.data(); }

  /// Iterator past the last value.
  [[nodiscard]] constexpr const T* end() const noexcept { return items_.data() + size_; }

private:
  std::array<T, N> items_{};
  std::size_t size_ = 0;
};

} // namespace fftn::inline v0
