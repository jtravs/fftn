// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Memory of a chosen page size, for measured planning's scratch (core/measure.cpp). Where the
// kernel backs anonymous memory with transparent huge pages (Linux), the same program can run
// at different speeds on huge pages and on 4 KiB pages: power-of-two chains whose passes read
// streams a power of two apart share cache sets on huge pages (X29, X32, LESSONS 3.14), and
// P2 found measured planning choosing, on scratch that happened to be huge-page backed,
// programs that were slower on the caller's 4 KiB pages (EPYC r2c 65536, 1.18). A caller's
// memory may be either, so planning times candidates on both.
#pragma once

#include <cstddef>
#include <memory>

namespace fftn::inline v0::detail {

enum class page_kind : unsigned char {
  small, // 4 KiB pages: the kernel is asked not to use huge pages
  huge,  // transparent huge pages where the kernel offers them
};

// Whether memory of the two kinds can differ here: Linux with transparent huge pages not
// disabled (/sys/kernel/mm/transparent_hugepage/enabled not "[never]"). Read once.
[[nodiscard]] bool page_kinds_differ() noexcept;

// Frees a page_region's reservation (outside the class: a nested deleter with a default
// member initializer is not default-constructible until the enclosing class is complete).
struct page_release {
  std::size_t bytes = 0;
  void operator()(std::byte* p) const noexcept;
};

// A zero-filled region of at least `bytes` bytes, starting on a 2 MiB boundary where the
// kinds differ (so that a huge page can back it) and on a 64-byte boundary elsewhere, with
// the page-size advice of `kind` where the kernel takes advice. Move-only.
class page_region {
public:
  page_region() noexcept = default;
  page_region(std::size_t bytes, page_kind kind);
  page_region(page_region&&) noexcept = default;
  page_region& operator=(page_region&&) noexcept = default;
  page_region(const page_region&) = delete;
  page_region& operator=(const page_region&) = delete;
  ~page_region() = default;

  [[nodiscard]] std::byte* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  // The bytes this region reserves, including its alignment slack.
  [[nodiscard]] std::size_t reserved() const noexcept { return reserved_; }

  // The bytes a region of `bytes` reserves.
  [[nodiscard]] static std::size_t reserved_for(std::size_t bytes) noexcept;

private:
  std::unique_ptr<std::byte, page_release> base_;
  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t reserved_ = 0;
};

} // namespace fftn::inline v0::detail
