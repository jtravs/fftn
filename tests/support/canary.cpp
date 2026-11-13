// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "support/canary.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace fftn::test {

std::byte canary_byte(std::size_t offset) {
  // 0xA5 + 0x3B k: odd step, so 256 consecutive bytes all differ.
  return static_cast<std::byte>((0xA5U + 0x3BU * offset) & 0xFFU);
}

namespace {

// Two periods of the canary pattern, so that 256 bytes starting at any phase are
// contiguous here. Filling and checking go through memcpy and memcmp: byte loops over the
// guard pages' neighbourhood dominated the suites' run time under ThreadSanitizer.
constexpr std::size_t period = 256;

const std::array<std::byte, 2 * period>& pattern() {
  static const std::array<std::byte, 2 * period> table = [] {
    std::array<std::byte, 2 * period> t{};
    for (std::size_t i = 0; i < t.size(); ++i) {
      t[i] = canary_byte(i);
    }
    return t;
  }();
  return table;
}

} // namespace

void write_canary(std::byte* base, std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end;) {
    const std::size_t chunk = std::min(end - i, period);
    std::memcpy(base + i, pattern().data() + i % period, chunk);
    i += chunk;
  }
}

bool canary_intact(const std::byte* base, std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end;) {
    const std::size_t chunk = std::min(end - i, period);
    if (std::memcmp(base + i, pattern().data() + i % period, chunk) != 0) {
      return false;
    }
    i += chunk;
  }
  return true;
}

std::size_t guarded_buffer::page_size() {
#ifdef _WIN32
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return info.dwPageSize;
#else
  return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#endif
}

namespace {

std::byte* map_pages(std::size_t bytes) {
#ifdef _WIN32
  void* p = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
#else
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (p == MAP_FAILED) {
    throw std::bad_alloc();
  }
#endif
  return static_cast<std::byte*>(p);
}

void protect_page(std::byte* page, std::size_t bytes) {
#ifdef _WIN32
  DWORD old = 0;
  if (VirtualProtect(page, bytes, PAGE_NOACCESS, &old) == 0) {
    throw std::runtime_error("guarded_buffer: VirtualProtect failed");
  }
#else
  if (mprotect(page, bytes, PROT_NONE) != 0) {
    throw std::runtime_error("guarded_buffer: mprotect failed");
  }
#endif
}

void unmap_pages(std::byte* p, std::size_t bytes) noexcept {
#ifdef _WIN32
  static_cast<void>(bytes);
  VirtualFree(p, 0, MEM_RELEASE);
#else
  munmap(p, bytes);
#endif
}

} // namespace

guarded_buffer::guarded_buffer(std::size_t bytes, std::size_t alignment, std::size_t offset,
                               guard_side side) {
  const std::size_t page = page_size();
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment > page ||
      offset >= alignment) {
    throw std::invalid_argument("guarded_buffer: alignment must be a power of two up to the "
                                "page size, and offset below it");
  }
  const std::size_t inner_pages = (bytes + alignment + page - 1) / page;
  inner_bytes_ = inner_pages * page;
  mapping_bytes_ = inner_bytes_ + 2 * page;
  mapping_ = map_pages(mapping_bytes_);
  inner_ = mapping_ + page;
  size_ = bytes;
  try {
    protect_page(mapping_, page);
    protect_page(inner_ + inner_bytes_, page);
  } catch (...) {
    release();
    throw;
  }
  const auto inner_start = reinterpret_cast<std::uintptr_t>(inner_);
  std::size_t start = 0; // offset of the region within the inner pages
  if (side == guard_side::below) {
    start = offset; // inner_ is page aligned, hence aligned to `alignment`
  } else {
    // The highest start that is congruent to `offset` and leaves room for `bytes`.
    const std::uintptr_t last = inner_start + inner_bytes_ - bytes;
    std::uintptr_t s = ((last - offset) & ~(std::uintptr_t{alignment} - 1)) + offset;
    if (s > last) {
      s -= alignment;
    }
    start = static_cast<std::size_t>(s - inner_start);
  }
  data_ = inner_ + start;
  write_canary(inner_, 0, start);
  write_canary(inner_, start + bytes, inner_bytes_);
}

guarded_buffer::~guarded_buffer() { release(); }

guarded_buffer::guarded_buffer(guarded_buffer&& other) noexcept
    : mapping_(std::exchange(other.mapping_, nullptr)),
      mapping_bytes_(std::exchange(other.mapping_bytes_, 0)),
      inner_(std::exchange(other.inner_, nullptr)),
      inner_bytes_(std::exchange(other.inner_bytes_, 0)),
      data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

guarded_buffer& guarded_buffer::operator=(guarded_buffer&& other) noexcept {
  if (this != &other) {
    release();
    mapping_ = std::exchange(other.mapping_, nullptr);
    mapping_bytes_ = std::exchange(other.mapping_bytes_, 0);
    inner_ = std::exchange(other.inner_, nullptr);
    inner_bytes_ = std::exchange(other.inner_bytes_, 0);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

void guarded_buffer::release() noexcept {
  if (mapping_ != nullptr) {
    unmap_pages(mapping_, mapping_bytes_);
    mapping_ = nullptr;
  }
}

bool guarded_buffer::surroundings_intact() const {
  const auto start = static_cast<std::size_t>(data_ - inner_);
  return canary_intact(inner_, 0, start) && canary_intact(inner_, start + size_, inner_bytes_);
}

canary_bytes::canary_bytes(std::span<std::byte> region) : region_(region) {}

void canary_bytes::allow(std::size_t offset, std::size_t bytes) {
  if (offset > region_.size() || bytes > region_.size() - offset) {
    throw std::out_of_range("canary_bytes::allow: outside the region");
  }
  described_.emplace_back(offset, offset + bytes);
}

void canary_bytes::arm() {
  std::ranges::sort(described_);
  gaps_.clear();
  std::size_t at = 0;
  for (const auto& [begin, end] : described_) {
    if (begin > at) {
      gaps_.emplace_back(at, begin);
    }
    at = std::max(at, end);
  }
  if (at < region_.size()) {
    gaps_.emplace_back(at, region_.size());
  }
  for (const auto& [begin, end] : gaps_) {
    write_canary(region_.data(), begin, end);
  }
}

std::size_t canary_bytes::violations() const {
  std::size_t count = 0;
  for (const auto& [begin, end] : gaps_) {
    if (!canary_intact(region_.data(), begin, end)) {
      for (std::size_t i = begin; i < end; ++i) {
        if (region_[i] != canary_byte(i)) {
          ++count;
        }
      }
    }
  }
  return count;
}

std::size_t canary_bytes::first_violation() const {
  for (const auto& [begin, end] : gaps_) {
    if (!canary_intact(region_.data(), begin, end)) {
      for (std::size_t i = begin; i < end; ++i) {
        if (region_[i] != canary_byte(i)) {
          return i;
        }
      }
    }
  }
  return region_.size();
}

} // namespace fftn::test
