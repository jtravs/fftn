// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/page_memory.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>

#ifdef __linux__
#include <sys/mman.h>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#endif

namespace fftn::inline v0::detail {
namespace {

constexpr std::size_t huge_page = std::size_t{2} << 20U;
constexpr std::size_t line = 64;

std::size_t round_up(std::size_t v, std::size_t to) { return (v + to - 1) / to * to; }

} // namespace

bool page_kinds_differ() noexcept {
#ifdef __linux__
  static const bool differ = [] {
    try {
      std::ifstream in("/sys/kernel/mm/transparent_hugepage/enabled");
      const std::string text((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
      return !text.empty() && text.find("[never]") == std::string::npos;
    } catch (...) {
      return false;
    }
  }();
  return differ;
#else
  return false;
#endif
}

std::size_t page_region::reserved_for(std::size_t bytes) noexcept {
  return page_kinds_differ() ? round_up(bytes, huge_page) + huge_page : bytes + line;
}

void page_release::operator()(std::byte* p) const noexcept {
#ifdef __linux__
  if (page_kinds_differ()) {
    munmap(p, bytes);
    return;
  }
#endif
  ::operator delete(p, std::align_val_t{line});
}

page_region::page_region(std::size_t bytes, [[maybe_unused]] page_kind kind)
    : base_(nullptr, page_release{}), size_(bytes), reserved_(reserved_for(bytes)) {
#ifdef __linux__
  if (page_kinds_differ()) {
    void* p = mmap(nullptr, reserved_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
      throw std::bad_alloc();
    }
    base_ = std::unique_ptr<std::byte, page_release>(static_cast<std::byte*>(p),
                                                     page_release{reserved_});
    const auto at = reinterpret_cast<std::uintptr_t>(p);
    data_ = static_cast<std::byte*>(p) + (round_up(at, huge_page) - at);
    // Advice only: a kernel that refuses it leaves the default, which is still a valid
    // timing (the two kinds then measure the same memory).
    static_cast<void>(madvise(data_, round_up(bytes, huge_page),
                              kind == page_kind::huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE));
    std::memset(data_, 0, bytes); // anonymous memory is zero; touching it faults it in now
    return;
  }
#endif
  auto* p = static_cast<std::byte*>(::operator new(reserved_, std::align_val_t{line}));
  base_ = std::unique_ptr<std::byte, page_release>(p, page_release{reserved_});
  data_ = p;
  std::memset(data_, 0, bytes);
}

} // namespace fftn::inline v0::detail
