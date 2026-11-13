// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// An owning, aligned, fixed-size array for scratch and tables. Allocation happens only at
// construction (plan or workspace construction), never in execute.
#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace fftn::inline v0::detail {

// 64 bytes: a cache line on x86 and the widest vector (AVX-512). Apple M-series lines are
// 128 bytes; choosing that is the planner's business, through the alignment argument.
inline constexpr std::size_t default_alignment = 64;

// The elements are left uninitialised: T must be trivially copyable and destructible, so the
// storage holds T objects as soon as it is allocated (implicit-lifetime types, C++20) and
// they need no destruction. The allocation is rounded up to a whole number of alignment
// units, so a vector load of the last aligned block never reads past it.
template <class T>
  requires std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>
class aligned_buffer {
public:
  aligned_buffer() noexcept = default;

  // Throws std::invalid_argument when `alignment` is not a power of two at least alignof(T),
  // std::bad_array_new_length when the size overflows, std::bad_alloc when memory runs out.
  explicit aligned_buffer(std::size_t count, std::size_t alignment = default_alignment)
      : storage_(nullptr, deleter{alignment}), size_(count) {
    if (alignment < alignof(T) || (alignment & (alignment - 1)) != 0) {
      throw std::invalid_argument("aligned_buffer: alignment must be a power of two at least "
                                  "the element's alignment");
    }
    if (count == 0) {
      return;
    }
    if (count > (std::numeric_limits<std::size_t>::max() - alignment) / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    const std::size_t bytes = (count * sizeof(T) + alignment - 1) & ~(alignment - 1);
    storage_.reset(static_cast<T*>(::operator new(bytes, std::align_val_t{alignment})));
  }

  aligned_buffer(aligned_buffer&& other) noexcept
      : storage_(std::move(other.storage_)), size_(std::exchange(other.size_, 0)) {}
  aligned_buffer& operator=(aligned_buffer&& other) noexcept {
    storage_ = std::move(other.storage_);
    size_ = std::exchange(other.size_, 0);
    return *this;
  }
  aligned_buffer(const aligned_buffer&) = delete;
  aligned_buffer& operator=(const aligned_buffer&) = delete;
  ~aligned_buffer() = default;

  [[nodiscard]] T* data() noexcept { return storage_.get(); }
  [[nodiscard]] const T* data() const noexcept { return storage_.get(); }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] std::size_t alignment() const noexcept { return storage_.get_deleter().alignment; }
  [[nodiscard]] std::span<T> span() noexcept { return {data(), size_}; }
  [[nodiscard]] std::span<const T> span() const noexcept { return {data(), size_}; }

private:
  struct deleter {
    std::size_t alignment = default_alignment;
    void operator()(T* p) const noexcept { ::operator delete(p, std::align_val_t{alignment}); }
  };

  std::unique_ptr<T, deleter> storage_{nullptr, deleter{}};
  std::size_t size_ = 0;
};

} // namespace fftn::inline v0::detail
