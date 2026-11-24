// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Contracts of the Kokkos adapter's plan class: scaling touches only the described output
// (padding and gaps keep their bytes), execute allocates nothing, a second concurrent
// execution of one plan is refused, out-of-place Views must not overlap, extent-1
// dimensions are ignored, and every refusal names its argument, in both precisions.

#include "kokkos/kokkos_helpers.hpp"
#include "support/alloc_hook.hpp"
#include "support/canary.hpp"

#include <fftn/fftn.hpp>
#include <fftn/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace fftn::test::kokkos_adapter;

template <class T> class KokkosAdapterContracts : public ::testing::Test {};
TYPED_TEST_SUITE(KokkosAdapterContracts, precisions);
template <class T> class KokkosAdapterValidation : public ::testing::Test {};
TYPED_TEST_SUITE(KokkosAdapterValidation, precisions);

double coefficient_of(normalization norm, direction dir, double points) {
  const bool forward = dir == direction::forward;
  if ((norm == normalization::forward && forward) ||
      (norm == normalization::backward && !forward)) {
    return 1.0 / points;
  }
  return norm == normalization::ortho ? 1.0 / std::sqrt(points) : 1.0;
}

// In-place real planes over (count, ny, nx): the real View shows FFTW's padded rows. The
// scaled run gives the unscaled run's described reals times the coefficient over all
// ny nx points, and leaves every padding real exactly as the unscaled run left it.
TYPED_TEST(KokkosAdapterContracts, ScalingInPlacePlanesTouchesOnlyDescribedReals) {
  using T = TypeParam;
  using cube = Kokkos::View<Kokkos::complex<T>***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using real_cube = Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const std::size_t count = 2;
  const std::size_t ny = 5;
  const std::size_t nx = 7;
  const std::size_t h = nx / 2 + 1;
  const cube c("c", count, ny, h);
  const real_cube padded(reinterpret_cast<T*>(c.data()), count, ny, 2 * h);
  const auto r =
      Kokkos::subview(padded, Kokkos::ALL, Kokkos::ALL, std::make_pair(std::size_t{0}, nx));
  const fftn::kokkos::axis_type<2> axes{-2, -1};
  const fftn::kokkos::view_plan<host, cube, decltype(r), 2> c2r(host{}, c, r, direction::backward,
                                                                axes);
  const fftn::kokkos::view_plan<host, decltype(r), cube, 2> r2c(host{}, r, c, direction::forward,
                                                                axes);
  const auto points = static_cast<double>(ny * nx);
  for (const normalization norm :
       {normalization::backward, normalization::forward, normalization::ortho}) {
    fill(c, 31);
    c2r.execute(c, r, normalization::none);
    const auto unscaled = reals_of(c);
    fill(c, 31);
    c2r.execute(c, r, norm);
    const auto scaled = reals_of(c);
    const auto k = static_cast<T>(coefficient_of(norm, direction::backward, points));
    for (std::size_t i = 0; i < scaled.size(); ++i) {
      const bool described = i % (2 * h) < nx;
      const T want = described ? unscaled[i] * k : unscaled[i];
      ASSERT_EQ(scaled[i], want) << "real " << i << (described ? "" : " (padding)");
    }

    // Forward in place: the whole complex output is described.
    fill(c, 37);
    r2c.execute(r, c, normalization::none);
    const auto spectrum = reals_of(c);
    fill(c, 37);
    r2c.execute(r, c, norm);
    const auto kf = static_cast<T>(coefficient_of(norm, direction::forward, points));
    auto want = spectrum;
    for (T& w : want) {
      w *= kf;
    }
    expect_same_bits(reals_of(c), want);
  }
}

// Out of place into gapped rows (LayoutStride, a row every n + 3 values) inside a canary
// buffer: scaled results are the unscaled ones times 1/N, and no byte outside the described
// elements changes, for c2c, r2c and c2r.
template <class T, class In, class Out, class Make>
void check_gapped(direction dir, std::size_t out_row, std::size_t out_n, std::size_t rows,
                  std::size_t points, const In& in, Make make_out) {
  using out_value = Out::non_const_value_type;
  const std::size_t bytes = (rows * out_row + 8) * sizeof(out_value);
  std::vector<std::byte> store(bytes);
  // The View starts 4 elements into the buffer, so its edges have canaries on both sides.
  auto* first = reinterpret_cast<out_value*>(store.data()) + 4;
  const Out out = make_out(first);
  fftn::test::canary_bytes canaries{std::span<std::byte>(store)};
  for (std::size_t r = 0; r < rows; ++r) {
    canaries.allow((4 + r * out_row) * sizeof(out_value), out_n * sizeof(out_value));
  }
  const fftn::kokkos::view_plan<host, In, Out> p(host{}, in, out, dir, -1);
  for (const normalization norm : {
           normalization::none,
           normalization::backward,
           normalization::forward,
           normalization::ortho,
       }) {
    canaries.arm();
    p.execute(in, out, normalization::none);
    std::vector<out_value> unscaled(rows * out_n);
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t j = 0; j < out_n; ++j) {
        unscaled[r * out_n + j] = first[r * out_row + j];
      }
    }
    p.execute(in, out, norm);
    ASSERT_EQ(canaries.violations(), 0U) << "first changed byte " << canaries.first_violation();
    const auto k = static_cast<T>(coefficient_of(norm, dir, static_cast<double>(points)));
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t j = 0; j < out_n; ++j) {
        const out_value got = first[r * out_row + j];
        const out_value u = unscaled[r * out_n + j];
        if constexpr (std::is_floating_point_v<out_value>) {
          ASSERT_EQ(got, u * k);
        } else {
          ASSERT_EQ(got.real(), u.real() * k);
          ASSERT_EQ(got.imag(), u.imag() * k);
        }
      }
    }
  }
}

TYPED_TEST(KokkosAdapterContracts, ScalingGappedOutputLeavesTheGaps) {
  using T = TypeParam;
  using cs = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  using rs = Kokkos::View<T**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const std::size_t rows = 3;
  const std::size_t n = 12;
  const std::size_t h = n / 2 + 1;
  const complex2<T> x("x", rows, n);
  const real2<T> xr("xr", rows, n);
  const complex2<T> xh("xh", rows, h);
  fill(x, 41);
  fill(xr, 43);
  fill(xh, 47);
  check_gapped<T, complex2<T>, cs>(direction::backward, n + 3, n, rows, n, x, [&](auto* p) {
    return cs(p, Kokkos::LayoutStride(rows, n + 3, n, 1));
  });
  check_gapped<T, real2<T>, cs>(direction::forward, h + 3, h, rows, n, xr, [&](auto* p) {
    return cs(p, Kokkos::LayoutStride(rows, h + 3, h, 1));
  });
  check_gapped<T, complex2<T>, rs>(direction::backward, n + 3, n, rows, n, xh, [&](auto* p) {
    return rs(p, Kokkos::LayoutStride(rows, n + 3, n, 1));
  });
}

// execute allocates nothing, from the first call after construction on, scaled or not, in
// place or not (the labels are built at construction).
TYPED_TEST(KokkosAdapterContracts, ExecuteAllocatesNothing) {
  using T = TypeParam;
  using cube = Kokkos::View<Kokkos::complex<T>***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using real_cube = Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const cube c("c", 2, 6, 5);
  const real_cube r("r", 2, 6, 8);
  const cube v("v", 2, 6, 8);
  const fftn::kokkos::axis_type<2> axes{-2, -1};
  const fftn::kokkos::view_plan<host, cube, real_cube, 2> c2r(host{}, c, r, direction::backward,
                                                              axes);
  const fftn::kokkos::view_plan<host, cube, cube, 2> in_place(host{}, v, v, direction::forward,
                                                              axes);
  ASSERT_GT(c2r.fftn_plan().workspace_bytes(), 0U);
  const fftn::test::allocation_counter counter;
  c2r.execute(c, r);
  c2r.execute(c, r, normalization::none);
  in_place.execute(v, v, normalization::ortho);
  fftn::kokkos::execute(in_place, v, v);
  EXPECT_EQ(counter.allocations(), 0U);
  EXPECT_EQ(counter.deallocations(), 0U);
}

// A second execution of a plan while one runs is refused, not run on the shared workspace: one
// thread executes the plan over and over while another tries until it is refused. Neither fence
// nor Tools region serves as a hook here, since a plan on Serial makes neither.
TYPED_TEST(KokkosAdapterContracts, ConcurrentExecutionIsRefused) {
  using T = TypeParam;
  const complex2<T> a("a", 8, 1024);
  const complex2<T> b("b", 8, 1024);
  const complex2<T> d("d", 8, 1024);
  fill(a, 53);
  const fftn::kokkos::view_plan p(host{}, a, b, direction::forward, -1);
  p.execute(a, b, normalization::none);
  const auto want = reals_of(b);
  std::atomic<bool> refused{false};
  std::string error;
  std::thread other([&] {
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!refused.load() && std::chrono::steady_clock::now() < give_up) {
      try {
        p.execute(a, d, normalization::none);
      } catch (const std::logic_error& e) {
        error = e.what();
        refused.store(true);
      }
    }
    refused.store(true);
  });
  while (!refused.load()) {
    try {
      p.execute(a, b, normalization::none);
    } catch (const std::logic_error&) {
      // The other thread held the plan this time.
      continue;
    }
  }
  other.join();
  EXPECT_EQ(error.rfind("fftn::kokkos: view_plan::execute was called while another", 0), 0U)
      << error;
  // Whichever executions ran gave right results, and the plan is free again afterwards.
  p.execute(a, b, normalization::none);
  expect_same_bits(reals_of(b), want);
  p.execute(a, d, normalization::none);
  expect_same_bits(reals_of(d), want);

  // The flag is released when an execution throws, too.
  const complex2<T> wrong("wrong", 2, 1024);
  expect_refused([&] { p.execute(wrong, b); }, "in.extent(0)");
  p.execute(a, b, normalization::none);
  expect_same_bits(reals_of(b), want);
}

// Out of place, execute refuses overlapping Views as construction does.
TYPED_TEST(KokkosAdapterContracts, ExecuteRefusesOverlappingViews) {
  using T = TypeParam;
  const complex2<T> big("big", 5, 8);
  const complex2<T> a("a", 3, 8);
  const complex2<T> b("b", 3, 8);
  const fftn::kokkos::view_plan p(host{}, a, b, direction::forward, -1);
  const complex2<T> rows_0_2 = Kokkos::subview(big, std::make_pair(0, 3), Kokkos::ALL);
  const complex2<T> rows_1_3 = Kokkos::subview(big, std::make_pair(1, 4), Kokkos::ALL);
  expect_refused([&] { p.execute(rows_0_2, rows_1_3); }, "out");
  const complex2<T> rows_2_4 = Kokkos::subview(big, std::make_pair(2, 5), Kokkos::ALL);
  expect_refused([&] { p.execute(rows_0_2, rows_2_4); }, "out");
}

// Dimensions of extent 1 never step: a stride of 0 there is accepted, and an in-place real
// transform does not ask them for FFTW's doubled strides.
TYPED_TEST(KokkosAdapterContracts, ExtentOneDimensionsAreIgnored) {
  using T = TypeParam;
  using cs = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  using rs = Kokkos::View<T**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const complex2<T> pool("pool", 1, 8);
  const complex2<T> pool2("pool2", 1, 8);
  fill(pool, 59);
  const cs a(pool.data(), Kokkos::LayoutStride(1, 0, 8, 1));
  const cs b(pool2.data(), Kokkos::LayoutStride(1, 0, 8, 1));
  const fftn::kokkos::view_plan p(host{}, a, b, direction::forward, -1);
  p.execute(a, b, normalization::none);
  const auto one = fftn::rows_c2c<T>(8, 1, direction::forward);
  expect_same_bits(reals_of(pool2),
                   reals_from(core_out_of_place(one, complexes_of(reals_of(pool)), 8)));

  const complex2<T> c("c", 1, 5);
  fill(c, 61);
  const auto start = complexes_of(reals_of(c));
  const rs real(reinterpret_cast<T*>(c.data()), Kokkos::LayoutStride(1, 3, 8, 1));
  const cs spectrum(c.data(), Kokkos::LayoutStride(1, 7, 5, 1));
  const fftn::kokkos::view_plan q(host{}, real, spectrum, direction::forward, -1);
  q.execute(real, spectrum, normalization::none);
  expect_same_bits(reals_of(c), reals_from(core_in_place<T>(fftn::rows_r2c<T>(8, 1), start)));
}

TYPED_TEST(KokkosAdapterValidation, AxesExtentsAndDirections) {
  using T = TypeParam;
  using c2 = complex2<T>;
  using r2 = real2<T>;
  using cplan = fftn::kokkos::view_plan<host, c2, c2>;
  const c2 a("a", 3, 8);
  const c2 b("b", 3, 8);
  expect_refused([&] { cplan(host{}, a, b, direction::forward, 2); }, "axes[0]");
  expect_refused([&] { cplan(host{}, a, b, direction::forward, -3); }, "axes[0]");
  expect_refused(
      [&] {
        fftn::kokkos::view_plan<host, c2, c2, 2>(host{}, a, b, direction::forward,
                                                 fftn::kokkos::axis_type<2>{-1, 1});
      },
      "axes");
  const c2 empty("empty", 0, 8);
  expect_refused([&] { cplan(host{}, empty, b, direction::forward, -1); }, "in.extent(0)");
  expect_refused([&] { cplan(host{}, a, empty, direction::forward, -1); }, "out.extent(0)");
  const c2 wide("wide", 3, 9);
  expect_refused([&] { cplan(host{}, a, wide, direction::forward, -1); }, "out.extent(1)");

  const r2 x("x", 3, 8);
  const c2 half("half", 3, 5);
  const c2 short_half("short_half", 3, 4);
  const c2 few("few", 2, 5);
  using r2c = fftn::kokkos::view_plan<host, r2, c2>;
  using c2r = fftn::kokkos::view_plan<host, c2, r2>;
  expect_refused([&] { r2c(host{}, x, short_half, direction::forward, -1); }, "out.extent(1)");
  expect_refused([&] { r2c(host{}, x, few, direction::forward, -1); }, "out.extent(0)");
  expect_refused([&] { c2r(host{}, short_half, x, direction::backward, -1); }, "in.extent(1)");
  expect_refused([&] { r2c(host{}, x, half, direction::backward, -1); }, "dir");
  expect_refused([&] { c2r(host{}, half, x, direction::forward, -1); }, "dir");
  expect_refused([&] { r2c(host{}, x, half, direction::forward, -1, 10); }, "n");
  expect_refused(
      [&] {
        fftn::kokkos::view_plan<host, c2, c2, 2>(host{}, a, b, direction::forward,
                                                 fftn::kokkos::axis_type<2>{0, 1},
                                                 std::array<std::size_t, 2>{3, 16});
      },
      "s[1]");
}

TYPED_TEST(KokkosAdapterValidation, LayoutsAndPlacement) {
  using T = TypeParam;
  using c2 = complex2<T>;
  using left = complex2<T, Kokkos::LayoutLeft>;
  using strided = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const c2 a("a", 4, 4);
  const left a_left(a.data(), 4, 4);
  expect_refused(
      [&] { fftn::kokkos::view_plan<host, c2, left>(host{}, a, a_left, direction::forward, -1); },
      "out.stride(0)");

  // In-place real transforms: the real View must show the padded rows.
  const c2 c("c", 3, 5);
  const unmanaged_real2<T> tight(reinterpret_cast<T*>(c.data()), 3, 8);
  expect_refused(
      [&] {
        fftn::kokkos::view_plan<host, unmanaged_real2<T>, c2>(host{}, tight, c, direction::forward,
                                                              -1);
      },
      "in.stride(0)");
  // ... with the halved axis contiguous: here it runs down the columns.
  const c2 cols("cols", 5, 3);
  const unmanaged_real2<T> cols_real(reinterpret_cast<T*>(cols.data()), 8, 3);
  expect_refused(
      [&] {
        fftn::kokkos::view_plan<host, c2, unmanaged_real2<T>>(host{}, cols, cols_real,
                                                              direction::backward, 0);
      },
      "out.stride(0)");

  // Out of place, the Views must not overlap.
  const c2 big("big", 4, 8);
  const auto rows_0_2 = Kokkos::subview(big, std::make_pair(0, 3), Kokkos::ALL);
  const auto rows_1_3 = Kokkos::subview(big, std::make_pair(1, 4), Kokkos::ALL);
  expect_refused(
      [&] { fftn::kokkos::view_plan(host{}, rows_0_2, rows_1_3, direction::forward, -1); }, "out");

  // Broadcast strides; output elements that collide (refused by the core plan, reported
  // with the Views named).
  const c2 store("store", 8, 8);
  const c2 other("other", 8, 8);
  const strided broadcast(store.data(), Kokkos::LayoutStride(3, 0, 8, 1));
  const strided fine(other.data(), Kokkos::LayoutStride(3, 8, 8, 1));
  using splan = fftn::kokkos::view_plan<host, strided, strided>;
  expect_refused([&] { splan(host{}, broadcast, fine, direction::forward, -1); }, "in.stride(0)");
  expect_refused([&] { splan(host{}, fine, broadcast, direction::forward, -1); }, "out.stride(0)");
  const strided colliding(store.data(), Kokkos::LayoutStride(3, 1, 8, 1));
  expect_refused([&] { splan(host{}, fine, colliding, direction::forward, -1); }, "in, out");
}

TYPED_TEST(KokkosAdapterValidation, ExecuteChecksItsViews) {
  using T = TypeParam;
  using c2 = complex2<T>;
  using cplan = fftn::kokkos::view_plan<host, c2, c2>;
  const c2 a("a", 3, 8);
  const c2 b("b", 3, 8);
  const c2 other_rows("other_rows", 2, 8);
  const cplan out_of_place(host{}, a, b, direction::forward, -1);
  expect_refused([&] { out_of_place.execute(other_rows, b); }, "in.extent(0)");
  expect_refused([&] { out_of_place.execute(a, other_rows); }, "out.extent(0)");
  expect_refused([&] { out_of_place.execute(a, a); }, "in, out");
  const cplan in_place(host{}, a, a, direction::forward, -1);
  expect_refused([&] { in_place.execute(a, b); }, "in, out");

  using strided = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const c2 store("store", 8, 8);
  const strided tight(store.data(), Kokkos::LayoutStride(3, 8, 8, 1));
  const strided loose(store.data(), Kokkos::LayoutStride(3, 16, 8, 1));
  const c2 store_b("store_b", 8, 8);
  const strided tight_b(store_b.data(), Kokkos::LayoutStride(3, 8, 8, 1));
  const strided loose_b(store_b.data(), Kokkos::LayoutStride(3, 9, 8, 1));
  using splan = fftn::kokkos::view_plan<host, strided, strided>;
  const splan p(host{}, tight, tight_b, direction::forward, -1);
  expect_refused([&] { p.execute(loose, tight_b); }, "in.stride(0)");
  expect_refused([&] { p.execute(tight, loose_b); }, "out.stride(0)");
}

} // namespace
