// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Shared helpers of the Kokkos adapter's tests: Views' memory as reals, reproducible fills,
// bitwise comparison, the core API's result for a shape, and refusals.
#pragma once

#include <fftn/fftn.hpp>
#include <fftn/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fftn::test::kokkos_adapter {

using host = Kokkos::DefaultHostExecutionSpace;
using fftn::direction;
using fftn::kokkos::normalization;

template <class T, class Layout = Kokkos::LayoutRight>
using complex2 = Kokkos::View<Kokkos::complex<T>**, Layout, Kokkos::HostSpace>;
template <class T, class Layout = Kokkos::LayoutRight>
using real2 = Kokkos::View<T**, Layout, Kokkos::HostSpace>;
template <class T, class Layout = Kokkos::LayoutRight>
using unmanaged_real2 =
    Kokkos::View<T**, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class Value> struct real_of {
  using type = Value;
};
template <class T> struct real_of<Kokkos::complex<T>> {
  using type = T;
};
template <class T> struct real_of<std::complex<T>> {
  using type = T;
};

// The real numbers in a View's memory, its whole span, in memory order.
template <class View> auto reals_of(const View& v) {
  using value = View::non_const_value_type;
  constexpr bool complex = !std::is_floating_point_v<value>;
  using T = real_of<value>::type;
  std::vector<T> r(v.span() * (complex ? 2 : 1));
  std::memcpy(r.data(), static_cast<const void*>(v.data()), r.size() * sizeof(T));
  return r;
}

template <class T> std::vector<std::complex<T>> complexes_of(const std::vector<T>& reals) {
  std::vector<std::complex<T>> c(reals.size() / 2);
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = {reals[2 * i], reals[2 * i + 1]};
  }
  return c;
}

template <class T> std::vector<T> reals_from(const std::vector<std::complex<T>>& c) {
  std::vector<T> r(2 * c.size());
  for (std::size_t i = 0; i < c.size(); ++i) {
    r[2 * i] = c[i].real();
    r[2 * i + 1] = c[i].imag();
  }
  return r;
}

// Fills a View's whole memory with reproducible values in [-1, 1).
template <class View> void fill(const View& v, std::uint32_t seed) {
  using value = View::non_const_value_type;
  constexpr bool complex = !std::is_floating_point_v<value>;
  using T = real_of<value>::type;
  std::vector<T> r(v.span() * (complex ? 2 : 1));
  std::uint32_t state = seed;
  for (T& x : r) {
    state = state * 1664525U + 1013904223U;
    x = static_cast<T>(static_cast<double>(state >> 8U) / 8388608.0 - 1.0);
  }
  std::memcpy(static_cast<void*>(v.data()), r.data(), r.size() * sizeof(T));
}

template <class T> void expect_same_bits(const std::vector<T>& got, const std::vector<T>& want) {
  using bits = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    ASSERT_EQ(std::bit_cast<bits>(got[i]), std::bit_cast<bits>(want[i])) << "at real " << i;
  }
}

// The core API's result for shape s on input x: out of place, or in place on a buffer that
// starts as `buffer` (whose reals the in-place real kinds read through the same address).
template <class In, class Out>
std::vector<Out> core_out_of_place(const fftn::shape<In, Out>& s, const std::vector<In>& x,
                                   std::size_t out_size) {
  const fftn::plan<In, Out> p(s);
  fftn::workspace ws(p);
  std::vector<Out> y(out_size);
  p.execute(x.data(), y.data(), ws);
  return y;
}

template <class T, class In, class Out>
std::vector<std::complex<T>> core_in_place(const fftn::shape<In, Out>& s,
                                           std::vector<std::complex<T>> buffer) {
  const fftn::plan<In, Out> p(s, {.in_place = true});
  fftn::workspace ws(p);
  // The standard allows a std::complex<T> array to be read as an array of T.
  auto* data = buffer.data();
  if constexpr (std::is_floating_point_v<In>) {
    p.execute(reinterpret_cast<const T*>(data), data, ws);
  } else if constexpr (std::is_floating_point_v<Out>) {
    p.execute(data, reinterpret_cast<T*>(data), ws);
  } else {
    p.execute(data, data, ws);
  }
  return buffer;
}

// Every refusal names its argument.
template <class F> void expect_refused(F&& f, const std::string& argument) {
  try {
    f();
    ADD_FAILURE() << "accepted; expected a refusal naming " << argument;
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    EXPECT_EQ(what.rfind("fftn::kokkos: " + argument + " ", 0), 0U) << what;
  }
}

template <class T> class KokkosAdapter : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;

} // namespace fftn::test::kokkos_adapter
