// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Contract helpers for "memory outside the described rows is untouched" (DESIGN.md 15.3).
//
// guarded_buffer places a region against an inaccessible guard page, so that a read or
// write just past it faults at once rather than corrupting a neighbour: above the end
// (overruns) or below the start (underruns, as negative strides can produce). The region
// starts at any requested offset from an alignment boundary; the few bytes between the
// region and the guard page that alignment leaves over hold canary values and are verified.
//
// canary_bytes covers the other case, memory inside a buffer that a call must not touch,
// such as the gaps between strided elements or between rows: every byte not marked as
// described is filled with a known pattern before the call and compared after it.

#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace fftn::test {

// The canary value of the byte at a given offset. It varies with the offset, so a shifted
// copy of canary memory is detected too, and as a double or float it is a small finite
// number, so a kernel that reads it produces wrong values rather than a trap.
std::byte canary_byte(std::size_t offset);

// Writes the canary values of offsets [begin, end) of a region starting at base, or checks
// that they are all still there.
void write_canary(std::byte* base, std::size_t begin, std::size_t end);
bool canary_intact(const std::byte* base, std::size_t begin, std::size_t end);

enum class guard_side : unsigned char { above, below };

class guarded_buffer {
public:
  // A region of `bytes` bytes whose first byte lies at `offset` modulo `alignment`
  // (a power of two no larger than the page size; offset < alignment), against a guard
  // page on the given side. There is a guard page on the other side too, further away.
  guarded_buffer(std::size_t bytes, std::size_t alignment, std::size_t offset,
                 guard_side side = guard_side::above);
  ~guarded_buffer();
  guarded_buffer(const guarded_buffer&) = delete;
  guarded_buffer& operator=(const guarded_buffer&) = delete;
  guarded_buffer(guarded_buffer&& other) noexcept;
  guarded_buffer& operator=(guarded_buffer&& other) noexcept;

  [[nodiscard]] std::byte* data() const { return data_; }
  [[nodiscard]] std::size_t size() const { return size_; }

  // Whether every accessible byte outside the region still holds its canary value.
  [[nodiscard]] bool surroundings_intact() const;

  static std::size_t page_size();

private:
  void release() noexcept;

  std::byte* mapping_ = nullptr; // the whole mapping, guard pages included
  std::size_t mapping_bytes_ = 0;
  std::byte* inner_ = nullptr; // the accessible pages between the guards
  std::size_t inner_bytes_ = 0;
  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

class canary_bytes {
public:
  explicit canary_bytes(std::span<std::byte> region);

  // Marks [offset, offset + bytes) as described: the call may read and write it.
  void allow(std::size_t offset, std::size_t bytes);
  // Writes the canary pattern into every byte not described.
  void arm();
  // Bytes not described whose value changed since arm().
  [[nodiscard]] std::size_t violations() const;
  // The offset of the first such byte, or the region's size if there is none.
  [[nodiscard]] std::size_t first_violation() const;

private:
  std::span<std::byte> region_;
  std::vector<std::pair<std::size_t, std::size_t>> described_; // [begin, end) offsets
  std::vector<std::pair<std::size_t, std::size_t>> gaps_;      // the rest, set by arm()
};

} // namespace fftn::test
