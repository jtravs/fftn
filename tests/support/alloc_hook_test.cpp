// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "support/alloc_hook.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <thread>
#include <vector>

namespace fftn::test {
namespace {

// Compilers may elide a new/delete pair whose result does not escape; storing the pointer
// through a volatile makes it escape.
void* volatile sink = nullptr;

struct alignas(128) over_aligned {
  std::array<double, 16> v;
};

TEST(AllocationCounter, CountsEveryForm) {
  const allocation_counter c;
  {
    const auto a = std::make_unique<int>(1);
    sink = a.get();
    // NOLINTNEXTLINE(modernize-avoid-c-arrays): tests operator new[]
    const auto b = std::make_unique<int[]>(10);
    sink = b.get();
    const auto d = std::make_unique<over_aligned>();
    sink = d.get();
    // NOLINTNEXTLINE(modernize-avoid-c-arrays): tests the aligned operator new[]
    const auto e = std::make_unique<over_aligned[]>(3);
    sink = e.get();
    const std::unique_ptr<int> f(new (std::nothrow) int(2));
    sink = f.get();
    void* g = ::operator new(64, std::align_val_t{256}, std::nothrow);
    sink = g;
    ::operator delete(g, std::align_val_t{256}, std::nothrow);
    void* h = ::operator new[](32, std::nothrow);
    sink = h;
    ::operator delete[](h, std::nothrow);
  }
  EXPECT_EQ(c.allocations(), 7U);
  EXPECT_EQ(c.deallocations(), 7U);
}

TEST(AllocationCounter, AlignedAllocationsAreAligned) {
  const auto p = std::make_unique<over_aligned>();
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p.get()) % alignof(over_aligned), 0U);
}

TEST(AllocationCounter, SeesNothingWhenNothingIsAllocated) {
  std::vector<double> v(1000, 1.0);
  const allocation_counter c;
  double sum = 0.0;
  for (const double x : v) {
    sum += x;
  }
  v.assign(500, 2.0); // within capacity
  EXPECT_EQ(sum, 1000.0);
  EXPECT_EQ(c.allocations(), 0U);
  EXPECT_EQ(c.deallocations(), 0U);
}

TEST(AllocationCounter, SeesOtherThreads) {
  const allocation_counter c;
  std::thread t([] {
    std::vector<int> v(100);
    sink = v.data();
  });
  t.join();
  // The vector's buffer, at least; the thread's start-up may allocate too.
  EXPECT_GE(c.allocations(), 1U);
  EXPECT_GE(c.deallocations(), 1U);
}

TEST(AllocationCounter, ZeroSizedAllocationsAreDistinct) {
  const allocation_counter c;
  void* a = ::operator new(0);
  void* b = ::operator new(0);
  EXPECT_NE(a, b);
  ::operator delete(a);
#ifdef __cpp_sized_deallocation
  ::operator delete(b, std::size_t{0});
#else
  // Clang before 19 enables sized deallocation only with -fsized-deallocation: the sized
  // form is then neither declared nor ever called by compiled code.
  ::operator delete(b);
#endif
  EXPECT_EQ(c.allocations(), 2U);
  EXPECT_EQ(c.deallocations(), 2U);
}

} // namespace
} // namespace fftn::test
