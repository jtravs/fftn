// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Replacements of the global allocation functions ([new.delete]) that count calls. See
// alloc_hook.hpp.

#include "support/alloc_hook.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#ifdef _WIN32
#include <malloc.h> // _aligned_malloc, _aligned_free
#endif

namespace {

// Plain atomics with constant initialisation: no constructor runs, so allocations made
// before main (by static initialisers anywhere) are counted safely.
std::atomic<std::size_t> news{0};
std::atomic<std::size_t> deletes{0};
std::atomic<std::size_t> bytes_new{0};

void* allocate(std::size_t bytes) {
  news.fetch_add(1, std::memory_order_relaxed);
  bytes_new.fetch_add(bytes, std::memory_order_relaxed);
  if (bytes == 0) {
    bytes = 1;
  }
  for (;;) {
    void* p = std::malloc(bytes);
    if (p != nullptr) {
      return p;
    }
    const std::new_handler handler = std::get_new_handler();
    if (handler == nullptr) {
      throw std::bad_alloc();
    }
    handler();
  }
}

void* allocate_aligned(std::size_t bytes, std::align_val_t align) {
  news.fetch_add(1, std::memory_order_relaxed);
  bytes_new.fetch_add(bytes, std::memory_order_relaxed);
  if (bytes == 0) {
    bytes = 1;
  }
  auto alignment = static_cast<std::size_t>(align);
  alignment = std::max(alignment, sizeof(void*));
  for (;;) {
#ifdef _WIN32
    void* p = _aligned_malloc(bytes, alignment);
#else
    // aligned_alloc wants a size that is a multiple of the alignment.
    void* p = std::aligned_alloc(alignment, (bytes + alignment - 1) / alignment * alignment);
#endif
    if (p != nullptr) {
      return p;
    }
    const std::new_handler handler = std::get_new_handler();
    if (handler == nullptr) {
      throw std::bad_alloc();
    }
    handler();
  }
}

void release(void* p) noexcept {
  if (p != nullptr) {
    deletes.fetch_add(1, std::memory_order_relaxed);
    std::free(p);
  }
}

void release_aligned(void* p) noexcept {
  if (p != nullptr) {
    deletes.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
}

template <class F> void* no_throw(F&& f) noexcept {
  try {
    return f();
  } catch (...) {
    return nullptr;
  }
}

} // namespace

namespace fftn::test {

allocation_counter::allocation_counter()
    : new_at_start_(news.load(std::memory_order_relaxed)),
      delete_at_start_(deletes.load(std::memory_order_relaxed)),
      bytes_at_start_(bytes_new.load(std::memory_order_relaxed)) {}

std::size_t allocation_counter::allocations() const {
  return news.load(std::memory_order_relaxed) - new_at_start_;
}

std::size_t allocation_counter::deallocations() const {
  return deletes.load(std::memory_order_relaxed) - delete_at_start_;
}

std::size_t allocation_counter::bytes_allocated() const {
  return bytes_new.load(std::memory_order_relaxed) - bytes_at_start_;
}

} // namespace fftn::test

// The replaceable global allocation functions, every form.

void* operator new(std::size_t bytes) { return allocate(bytes); }
void* operator new[](std::size_t bytes) { return allocate(bytes); }
void* operator new(std::size_t bytes, std::align_val_t align) {
  return allocate_aligned(bytes, align);
}
void* operator new[](std::size_t bytes, std::align_val_t align) {
  return allocate_aligned(bytes, align);
}
void* operator new(std::size_t bytes, const std::nothrow_t& /*tag*/) noexcept {
  return no_throw([bytes] { return allocate(bytes); });
}
void* operator new[](std::size_t bytes, const std::nothrow_t& /*tag*/) noexcept {
  return no_throw([bytes] { return allocate(bytes); });
}
void* operator new(std::size_t bytes, std::align_val_t align,
                   const std::nothrow_t& /*tag*/) noexcept {
  return no_throw([bytes, align] { return allocate_aligned(bytes, align); });
}
void* operator new[](std::size_t bytes, std::align_val_t align,
                     const std::nothrow_t& /*tag*/) noexcept {
  return no_throw([bytes, align] { return allocate_aligned(bytes, align); });
}

void operator delete(void* p) noexcept { release(p); }
void operator delete[](void* p) noexcept { release(p); }
void operator delete(void* p, std::size_t /*bytes*/) noexcept { release(p); }
void operator delete[](void* p, std::size_t /*bytes*/) noexcept { release(p); }
void operator delete(void* p, const std::nothrow_t& /*tag*/) noexcept { release(p); }
void operator delete[](void* p, const std::nothrow_t& /*tag*/) noexcept { release(p); }
void operator delete(void* p, std::align_val_t /*align*/) noexcept { release_aligned(p); }
void operator delete[](void* p, std::align_val_t /*align*/) noexcept { release_aligned(p); }
void operator delete(void* p, std::size_t /*bytes*/, std::align_val_t /*align*/) noexcept {
  release_aligned(p);
}
void operator delete[](void* p, std::size_t /*bytes*/, std::align_val_t /*align*/) noexcept {
  release_aligned(p);
}
void operator delete(void* p, std::align_val_t /*align*/, const std::nothrow_t& /*tag*/) noexcept {
  release_aligned(p);
}
void operator delete[](void* p, std::align_val_t /*align*/,
                       const std::nothrow_t& /*tag*/) noexcept {
  release_aligned(p);
}
