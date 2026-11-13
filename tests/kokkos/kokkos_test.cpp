// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The Kokkos adapter (fftn/kokkos.hpp) on Kokkos' Serial back end. Every transform is
// compared bit for bit with the same transform through the core API, which the oracle suites
// cover; then the Kokkos-FFT-shaped plan's normalisation, reuse and validation.

#include <fftn/fftn.hpp>
#include <fftn/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

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

template <class T> class KokkosAdapter : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(KokkosAdapter, precisions);

// Rows of n complex values: LayoutRight (rows, n) along -1 and LayoutLeft (n, rows) along 0
// have the same memory as the core's rows_c2c.
template <class T, class Layout> void check_c2c_rows(direction dir, bool in_place) {
  const std::size_t n = 12;
  const std::size_t rows = 3;
  constexpr bool right = std::is_same_v<Layout, Kokkos::LayoutRight>;
  const complex2<T, Layout> x("x", right ? rows : n, right ? n : rows);
  const complex2<T, Layout> y("y", right ? rows : n, right ? n : rows);
  fill(x, 7);
  const auto input = complexes_of(reals_of(x));
  const auto& out = in_place ? x : y;
  const fftn::kokkos::plan p(host{}, x, out, dir, right ? -1 : 0);
  p.execute(x, out, normalization::none);
  const auto s = fftn::rows_c2c<T>(n, rows, dir);
  const auto want = in_place ? core_in_place<T>(s, input) : core_out_of_place(s, input, n * rows);
  expect_same_bits(reals_of(out), reals_from(want));
}

TYPED_TEST(KokkosAdapter, C2cRowsMatchTheCoreApi) {
  using T = TypeParam;
  for (const direction dir : {direction::forward, direction::backward}) {
    for (const bool in_place : {false, true}) {
      check_c2c_rows<T, Kokkos::LayoutRight>(dir, in_place);
      check_c2c_rows<T, Kokkos::LayoutLeft>(dir, in_place);
    }
  }
}

// Real rows out of place: (rows, n) -> (rows, n/2 + 1), or the LayoutLeft transposes.
template <class T, class Layout> void check_r2c_c2r_rows() {
  const std::size_t n = 10;
  const std::size_t h = n / 2 + 1;
  const std::size_t rows = 4;
  constexpr bool right = std::is_same_v<Layout, Kokkos::LayoutRight>;
  const real2<T, Layout> x("x", right ? rows : n, right ? n : rows);
  const complex2<T, Layout> y("y", right ? rows : h, right ? h : rows);
  const real2<T, Layout> z("z", right ? rows : n, right ? n : rows);
  fill(x, 11);
  const fftn::kokkos::plan forward(host{}, x, y, direction::forward, right ? -1 : 0);
  forward.execute(x, y, normalization::none);
  const auto spectrum = core_out_of_place(fftn::rows_r2c<T>(n, rows), reals_of(x), h * rows);
  expect_same_bits(reals_of(y), reals_from(spectrum));

  const fftn::kokkos::plan backward(host{}, y, z, direction::backward, right ? -1 : 0);
  backward.execute(y, z, normalization::none);
  expect_same_bits(reals_of(z), core_out_of_place(fftn::rows_c2r<T>(n, rows), spectrum, n * rows));
}

TYPED_TEST(KokkosAdapter, RealRowsOutOfPlaceMatchTheCoreApi) {
  check_r2c_c2r_rows<TypeParam, Kokkos::LayoutRight>();
  check_r2c_c2r_rows<TypeParam, Kokkos::LayoutLeft>();
}

// Real rows in place: the real View shows FFTW's padded rows, n of the 2(n/2 + 1) reals of
// each complex row, as a subview of the complex memory read as reals.
template <class T, class Layout> void check_real_rows_in_place() {
  const std::size_t n = 9;
  const std::size_t h = n / 2 + 1;
  const std::size_t rows = 3;
  constexpr bool right = std::is_same_v<Layout, Kokkos::LayoutRight>;
  const complex2<T, Layout> c("c", right ? rows : h, right ? h : rows);
  const unmanaged_real2<T, Layout> padded(reinterpret_cast<T*>(c.data()), right ? rows : 2 * h,
                                          right ? 2 * h : rows);
  const auto first_n = std::make_pair(std::size_t{0}, n);
  const auto r = [&] {
    if constexpr (right) {
      return Kokkos::subview(padded, Kokkos::ALL, first_n);
    } else {
      return Kokkos::subview(padded, first_n, Kokkos::ALL);
    }
  }();
  fill(c, 5);
  const auto start = complexes_of(reals_of(c));

  const fftn::kokkos::plan forward(host{}, r, c, direction::forward, right ? -1 : 0);
  forward.execute(r, c, normalization::none);
  const auto spectrum = core_in_place<T>(fftn::rows_r2c<T>(n, rows), start);
  expect_same_bits(reals_of(c), reals_from(spectrum));

  const fftn::kokkos::plan backward(host{}, c, r, direction::backward, right ? -1 : 0);
  backward.execute(c, r, normalization::none);
  const auto back = core_in_place<T>(fftn::rows_c2r<T>(n, rows), spectrum);
  expect_same_bits(reals_of(c), reals_from(back));
}

TYPED_TEST(KokkosAdapter, RealRowsInPlaceMatchTheCoreApi) {
  check_real_rows_in_place<TypeParam, Kokkos::LayoutRight>();
  check_real_rows_in_place<TypeParam, Kokkos::LayoutLeft>();
}

// The application's planes: in-place 2-D c2c over (count, ny, nx), axes {-2, -1}, with an
// odd transverse size.
TYPED_TEST(KokkosAdapter, BatchedPlanesInPlaceMatchTheCoreApi) {
  using T = TypeParam;
  using view = Kokkos::View<Kokkos::complex<T>***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const std::size_t count = 3;
  const std::size_t ny = 15;
  const std::size_t nx = 8;
  const view v("v", count, ny, nx);
  for (const direction dir : {direction::forward, direction::backward}) {
    fill(v, 3);
    const auto start = complexes_of(reals_of(v));
    const fftn::kokkos::plan<host, view, view, 2> p(host{}, v, v, dir,
                                                    fftn::kokkos::axis_type<2>{-2, -1});
    p.execute(v, v, normalization::none);
    const auto want = core_in_place<T>(fftn::planes_c2c<T>(ny, nx, count, dir), start);
    expect_same_bits(reals_of(v), reals_from(want));
  }
}

// Normalisation as Kokkos-FFT: backward by 1/N by default, or as asked; each part is
// multiplied by the coefficient computed in double and rounded to T.
TYPED_TEST(KokkosAdapter, NormalisationScalesAsKokkosFft) {
  using T = TypeParam;
  const std::size_t n = 16;
  const std::size_t rows = 2;
  const complex2<T> x("x", rows, n);
  const complex2<T> y("y", rows, n);
  fill(x, 9);
  const auto input = complexes_of(reals_of(x));
  struct scaled_case {
    direction dir;
    normalization norm;
    double coefficient;
  };
  const double ortho = 1.0 / std::sqrt(16.0);
  for (const scaled_case k : {
           scaled_case{direction::backward, normalization::backward, 1.0 / 16},
           scaled_case{direction::forward, normalization::backward, 1.0},
           scaled_case{direction::forward, normalization::forward, 1.0 / 16},
           scaled_case{direction::backward, normalization::forward, 1.0},
           scaled_case{direction::forward, normalization::ortho, ortho},
           scaled_case{direction::backward, normalization::ortho, ortho},
           scaled_case{direction::backward, normalization::none, 1.0},
       }) {
    const fftn::kokkos::plan p(host{}, x, y, k.dir, -1);
    p.execute(x, y, k.norm);
    auto want = reals_from(core_out_of_place(fftn::rows_c2c<T>(n, rows, k.dir), input, n * rows));
    if (k.coefficient != 1.0) {
      for (T& w : want) {
        w *= static_cast<T>(k.coefficient);
      }
    }
    expect_same_bits(reals_of(y), want);
  }

  // Kokkos-FFT's default is backward normalisation; a c2r output is scaled as reals.
  const complex2<T> c("c", rows, n / 2 + 1);
  const real2<T> r("r", rows, n);
  fill(c, 13);
  const fftn::kokkos::plan c2r(host{}, c, r, direction::backward, -1);
  fftn::kokkos::execute(c2r, c, r);
  auto want = core_out_of_place(fftn::rows_c2r<T>(n, rows), complexes_of(reals_of(c)), n * rows);
  for (T& w : want) {
    w *= static_cast<T>(1.0 / 16);
  }
  expect_same_bits(reals_of(r), want);
}

// One plan serves any Views of its layout; const input Views are preserved; the
// Kokkos-FFT spellings in fftn::kokkos::compat are the same plan.
TYPED_TEST(KokkosAdapter, PlansServeOtherViewsAndTheCompatNames) {
  using T = TypeParam;
  namespace kfft = fftn::kokkos::compat;
  using in_view = Kokkos::View<const Kokkos::complex<T>**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using out_view = complex2<T>;
  const std::size_t n = 20;
  const std::size_t rows = 2;
  const out_view a("a", rows, n);
  const out_view b("b", rows, n);
  const out_view y("y", rows, n);
  fill(a, 1);
  const in_view ca = a;
  const kfft::Plan<host, in_view, out_view> p(host{}, ca, y, kfft::Direction::forward, -1);
  kfft::execute(p, ca, y, kfft::Normalization::none);
  const auto from_a = reals_of(y);
  const auto a_before = reals_of(a);

  fill(b, 2);
  const in_view cb = b;
  kfft::execute(p, cb, y, kfft::Normalization::none);
  expect_same_bits(reals_of(y),
                   reals_from(core_out_of_place(fftn::rows_c2c<T>(n, rows, direction::forward),
                                                complexes_of(reals_of(b)), n * rows)));
  kfft::execute(p, ca, y, kfft::Normalization::none);
  expect_same_bits(reals_of(y), from_a);
  expect_same_bits(reals_of(a), a_before);

  // Kokkos-FFT's multi-axis spelling, and the one-axis form's n given as the Views' length.
  const kfft::axis_type<1> axes{-1};
  const kfft::Plan<host, in_view, out_view> q(host{}, ca, y, kfft::Direction::forward, axes,
                                              kfft::shape_type<1>{n});
  const kfft::Plan<host, in_view, out_view> r(host{}, ca, y, kfft::Direction::forward, 1, n);
  q.execute(ca, y, kfft::Normalization::none);
  expect_same_bits(reals_of(y), from_a);
  r.execute(ca, y, kfft::Normalization::none);
  expect_same_bits(reals_of(y), from_a);
  EXPECT_NE(p.describe().find("c2c"), std::string::npos) << p.describe();
}

// View-based construction for the core API: make_plan and data_of, executed with the
// caller's own workspace, give the adapter plan's result.
TYPED_TEST(KokkosAdapter, MakePlanGivesACorePlan) {
  using T = TypeParam;
  const std::size_t n = 14;
  const std::size_t rows = 3;
  const real2<T> x("x", rows, n);
  const complex2<T> y("y", rows, n / 2 + 1);
  const complex2<T> z("z", rows, n / 2 + 1);
  fill(x, 21);
  const auto core =
      fftn::kokkos::make_plan(x, y, direction::forward, fftn::kokkos::axis_type<1>{-1});
  static_assert(std::is_same_v<decltype(core), const fftn::r2c_plan<T>>);
  fftn::workspace ws(core);
  core.execute(fftn::kokkos::data_of(x), fftn::kokkos::data_of(y), ws);
  const fftn::kokkos::plan p(host{}, x, z, direction::forward, -1);
  p.execute(x, z, normalization::none);
  expect_same_bits(reals_of(y), reals_of(z));
  EXPECT_EQ(p.fftn_plan().workspace_bytes(), core.workspace_bytes());

  // std::complex values and a rank-1 View (one transform, no batch) work the same way.
  const Kokkos::View<std::complex<T>*, Kokkos::HostSpace> u("u", n);
  const Kokkos::View<std::complex<T>*, Kokkos::HostSpace> v("v", n);
  fill(u, 23);
  const fftn::kokkos::plan q(host{}, u, v, direction::backward, 0);
  q.execute(u, v, normalization::none);
  auto one = fftn::rows_c2c<T>(n, 1, direction::backward);
  one.batches = {};
  expect_same_bits(reals_of(v), reals_from(core_out_of_place(one, complexes_of(reals_of(u)), n)));
}

// Strided axes and other ranks: a transform down the columns of a LayoutRight matrix, the
// same through LayoutStride, and a rank-4 View whose batch dimensions nest into one.
TYPED_TEST(KokkosAdapter, StridesAndRanksAreReadFromTheView) {
  using T = TypeParam;
  const std::size_t n = 6;
  const std::size_t cols = 5;
  const complex2<T> x("x", n, cols);
  const complex2<T> y("y", n, cols);
  fill(x, 17);
  const auto input = complexes_of(reals_of(x));
  fftn::shape<std::complex<T>, std::complex<T>> columns;
  columns.dir = direction::backward;
  columns.axes.push_back(fftn::axis{.n = n, .in_stride = 5, .out_stride = 5});
  columns.batches.push_back(fftn::batch{.count = cols, .in_dist = 1, .out_dist = 1});
  const auto want = reals_from(core_out_of_place(columns, input, n * cols));

  const fftn::kokkos::plan p(host{}, x, y, direction::backward, 0);
  p.execute(x, y, normalization::none);
  expect_same_bits(reals_of(y), want);

  using strided = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const strided xs(x.data(), Kokkos::LayoutStride(n, cols, cols, 1));
  const strided ys(y.data(), Kokkos::LayoutStride(n, cols, cols, 1));
  fill(y, 0);
  const fftn::kokkos::plan q(host{}, xs, ys, direction::backward, -2);
  q.execute(xs, ys, normalization::none);
  expect_same_bits(reals_of(y), want);

  using rank4 = Kokkos::View<Kokkos::complex<T>****, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const rank4 u("u", 2, 1, 3, 8);
  const rank4 w("w", 2, 1, 3, 8);
  fill(u, 19);
  const fftn::kokkos::plan r(host{}, u, w, direction::forward, 3);
  r.execute(u, w, normalization::none);
  expect_same_bits(reals_of(w),
                   reals_from(core_out_of_place(fftn::rows_c2c<T>(8, 6, direction::forward),
                                                complexes_of(reals_of(u)), 48)));
  const auto s = fftn::kokkos::shape_of(u, w, direction::forward, fftn::kokkos::axis_type<1>{-1});
  ASSERT_EQ(s.batches.size(), 1U);
  EXPECT_EQ(s.batches[0].count, 6U);
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

using c2 = complex2<double>;
using r2 = real2<double>;
using cplan = fftn::kokkos::plan<host, c2, c2>;

TEST(KokkosAdapterValidation, AxesExtentsAndDirections) {
  const c2 a("a", 3, 8);
  const c2 b("b", 3, 8);
  expect_refused([&] { cplan(host{}, a, b, direction::forward, 2); }, "axes[0]");
  expect_refused([&] { cplan(host{}, a, b, direction::forward, -3); }, "axes[0]");
  expect_refused(
      [&] {
        fftn::kokkos::plan<host, c2, c2, 2>(host{}, a, b, direction::forward,
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
  using r2c = fftn::kokkos::plan<host, r2, c2>;
  using c2r = fftn::kokkos::plan<host, c2, r2>;
  expect_refused([&] { r2c(host{}, x, short_half, direction::forward, -1); }, "out.extent(1)");
  expect_refused([&] { r2c(host{}, x, few, direction::forward, -1); }, "out.extent(0)");
  expect_refused([&] { c2r(host{}, short_half, x, direction::backward, -1); }, "in.extent(1)");
  expect_refused([&] { r2c(host{}, x, half, direction::backward, -1); }, "dir");
  expect_refused([&] { c2r(host{}, half, x, direction::forward, -1); }, "dir");
  expect_refused([&] { r2c(host{}, x, half, direction::forward, -1, 10); }, "n");
  expect_refused(
      [&] {
        fftn::kokkos::plan<host, c2, c2, 2>(host{}, a, b, direction::forward,
                                            fftn::kokkos::axis_type<2>{0, 1},
                                            fftn::kokkos::shape_type<2>{3, 16});
      },
      "s[1]");
}

TEST(KokkosAdapterValidation, LayoutsAndPlacement) {
  using left = complex2<double, Kokkos::LayoutLeft>;
  using strided = Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const c2 a("a", 4, 4);
  const left a_left(a.data(), 4, 4);
  expect_refused(
      [&] { fftn::kokkos::plan<host, c2, left>(host{}, a, a_left, direction::forward, -1); },
      "out.stride(0)");

  // In-place real transforms: the real View must show the padded rows.
  const c2 c("c", 3, 5);
  const unmanaged_real2<double> tight(reinterpret_cast<double*>(c.data()), 3, 8);
  expect_refused(
      [&] {
        fftn::kokkos::plan<host, unmanaged_real2<double>, c2>(host{}, tight, c, direction::forward,
                                                              -1);
      },
      "in.stride(0)");
  // ... with the halved axis contiguous: here it runs down the columns.
  const c2 cols("cols", 5, 3);
  const unmanaged_real2<double> cols_real(reinterpret_cast<double*>(cols.data()), 8, 3);
  expect_refused(
      [&] {
        fftn::kokkos::plan<host, c2, unmanaged_real2<double>>(host{}, cols, cols_real,
                                                              direction::backward, 0);
      },
      "out.stride(0)");

  // Out of place, the Views must not overlap.
  const c2 big("big", 4, 8);
  const auto rows_0_2 = Kokkos::subview(big, std::make_pair(0, 3), Kokkos::ALL);
  const auto rows_1_3 = Kokkos::subview(big, std::make_pair(1, 4), Kokkos::ALL);
  expect_refused([&] { fftn::kokkos::plan(host{}, rows_0_2, rows_1_3, direction::forward, -1); },
                 "out");

  // Broadcast strides; output elements that collide (refused by the core plan, reported
  // with the Views named); batch dimensions that do not nest.
  const c2 store("store", 8, 8);
  const c2 other("other", 8, 8);
  const strided broadcast(store.data(), Kokkos::LayoutStride(3, 0, 8, 1));
  const strided fine(other.data(), Kokkos::LayoutStride(3, 8, 8, 1));
  using splan = fftn::kokkos::plan<host, strided, strided>;
  expect_refused([&] { splan(host{}, broadcast, fine, direction::forward, -1); }, "in.stride(0)");
  expect_refused([&] { splan(host{}, fine, broadcast, direction::forward, -1); }, "out.stride(0)");
  const strided colliding(store.data(), Kokkos::LayoutStride(3, 1, 8, 1));
  expect_refused([&] { splan(host{}, fine, colliding, direction::forward, -1); }, "in, out");
  using strided4 =
      Kokkos::View<Kokkos::complex<double>****, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace> pool("pool", 5400);
  const Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace> pool2("pool2", 5400);
  const Kokkos::LayoutStride loose(2, 4000, 2, 1000, 2, 300, 8, 1);
  const strided4 u(pool.data(), loose);
  const strided4 v(pool2.data(), loose);
  expect_refused(
      [&] { fftn::kokkos::plan<host, strided4, strided4>(host{}, u, v, direction::forward, -1); },
      "in");
}

TEST(KokkosAdapterValidation, ExecuteChecksItsViews) {
  const c2 a("a", 3, 8);
  const c2 b("b", 3, 8);
  const c2 other_rows("other_rows", 2, 8);
  const cplan out_of_place(host{}, a, b, direction::forward, -1);
  expect_refused([&] { out_of_place.execute(other_rows, b); }, "in.extent(0)");
  expect_refused([&] { out_of_place.execute(a, other_rows); }, "out.extent(0)");
  expect_refused([&] { out_of_place.execute(a, a); }, "in, out");
  const cplan in_place(host{}, a, a, direction::forward, -1);
  expect_refused([&] { in_place.execute(a, b); }, "in, out");

  using strided = Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const c2 store("store", 8, 8);
  const strided tight(store.data(), Kokkos::LayoutStride(3, 8, 8, 1));
  const strided loose(store.data(), Kokkos::LayoutStride(3, 16, 8, 1));
  const c2 store_b("store_b", 8, 8);
  const strided tight_b(store_b.data(), Kokkos::LayoutStride(3, 8, 8, 1));
  const strided loose_b(store_b.data(), Kokkos::LayoutStride(3, 9, 8, 1));
  using splan = fftn::kokkos::plan<host, strided, strided>;
  const splan p(host{}, tight, tight_b, direction::forward, -1);
  expect_refused([&] { p.execute(loose, tight_b); }, "in.stride(0)");
  expect_refused([&] { p.execute(tight, loose_b); }, "out.stride(0)");
}

} // namespace

int main(int argc, char** argv) {
  try {
    ::testing::InitGoogleTest(&argc, argv);
    const Kokkos::ScopeGuard kokkos(argc, argv);
    return RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fftn_kokkos_tests: %s\n", e.what());
    return 1;
  }
}
