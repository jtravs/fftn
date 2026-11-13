// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The Kokkos adapter (fftn/kokkos.hpp) on Kokkos' Serial back end. Every transform is
// compared bit for bit with the same transform through the core API, which the oracle suites
// cover; then the Kokkos-FFT-shaped plan's normalisation, reuse and validation.

#include "kokkos/kokkos_helpers.hpp"

#include <fftn/fftn.hpp>
#include <fftn/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using namespace fftn::test::kokkos_adapter;

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
  const fftn::kokkos::view_plan p(host{}, x, out, dir, right ? -1 : 0);
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
  const fftn::kokkos::view_plan forward(host{}, x, y, direction::forward, right ? -1 : 0);
  forward.execute(x, y, normalization::none);
  const auto spectrum = core_out_of_place(fftn::rows_r2c<T>(n, rows), reals_of(x), h * rows);
  expect_same_bits(reals_of(y), reals_from(spectrum));

  const fftn::kokkos::view_plan backward(host{}, y, z, direction::backward, right ? -1 : 0);
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

  const fftn::kokkos::view_plan forward(host{}, r, c, direction::forward, right ? -1 : 0);
  forward.execute(r, c, normalization::none);
  const auto spectrum = core_in_place<T>(fftn::rows_r2c<T>(n, rows), start);
  expect_same_bits(reals_of(c), reals_from(spectrum));

  const fftn::kokkos::view_plan backward(host{}, c, r, direction::backward, right ? -1 : 0);
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
    const fftn::kokkos::view_plan<host, view, view, 2> p(host{}, v, v, dir,
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
    const fftn::kokkos::view_plan p(host{}, x, y, k.dir, -1);
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
  const fftn::kokkos::view_plan c2r(host{}, c, r, direction::backward, -1);
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

// compat::Direction is Kokkos-FFT's enumeration (0 and 1, underlying int), converted to
// fftn::direction by the plan: each value gives the transform of that direction, and a value
// that is neither is refused naming dir.
static_assert(std::is_same_v<std::underlying_type_t<fftn::kokkos::compat::Direction>, int>);
static_assert(static_cast<int>(fftn::kokkos::compat::Direction::forward) == 0);
static_assert(static_cast<int>(fftn::kokkos::compat::Direction::backward) == 1);

TYPED_TEST(KokkosAdapter, CompatDirectionConvertsToFftnDirection) {
  using T = TypeParam;
  namespace kfft = fftn::kokkos::compat;
  const std::size_t n = 12;
  const std::size_t rows = 2;
  const complex2<T> x("x", rows, n);
  const complex2<T> y("y", rows, n);
  fill(x, 31);
  const auto input = complexes_of(reals_of(x));
  EXPECT_EQ(fftn::kokkos::detail::direction_of(kfft::Direction::forward), direction::forward);
  EXPECT_EQ(fftn::kokkos::detail::direction_of(kfft::Direction::backward), direction::backward);
  for (const auto& [compat_dir, dir] : {
           std::pair{kfft::Direction::forward, direction::forward},
           std::pair{kfft::Direction::backward, direction::backward},
       }) {
    const kfft::Plan<host, complex2<T>, complex2<T>> p(host{}, x, y, compat_dir, -1);
    kfft::execute(p, x, y, kfft::Normalization::none);
    const auto want = core_out_of_place(fftn::rows_c2c<T>(n, rows, dir), input, n * rows);
    expect_same_bits(reals_of(y), reals_from(want));
    const kfft::Plan<host, complex2<T>, complex2<T>> q(host{}, x, y, compat_dir,
                                                       kfft::axis_type<1>{-1});
    q.execute(x, y, kfft::Normalization::none);
    expect_same_bits(reals_of(y), reals_from(want));
  }
  // Real plans check the converted direction as Kokkos-FFT does.
  const real2<T> r("r", rows, n);
  const complex2<T> h("h", rows, n / 2 + 1);
  expect_refused(
      [&] { kfft::Plan<host, real2<T>, complex2<T>>(host{}, r, h, kfft::Direction::backward, -1); },
      "dir");
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): invalid on purpose
  const auto invalid = static_cast<kfft::Direction>(2);
  expect_refused([&] { kfft::Plan<host, complex2<T>, complex2<T>>(host{}, x, y, invalid, -1); },
                 "dir");
  expect_refused(
      [&] {
        kfft::Plan<host, complex2<T>, complex2<T>>(host{}, x, y, invalid, kfft::axis_type<1>{-1});
      },
      "dir");
}

// compat::Normalization is Kokkos-FFT's enumeration (0 to 3, underlying int), converted to the
// adapter's normalization by execute: each value scales as its namesake, through the member and
// the free function, and a value that is none of them is refused naming norm.
static_assert(std::is_same_v<std::underlying_type_t<fftn::kokkos::compat::Normalization>, int>);
static_assert(static_cast<int>(fftn::kokkos::compat::Normalization::forward) == 0);
static_assert(static_cast<int>(fftn::kokkos::compat::Normalization::backward) == 1);
static_assert(static_cast<int>(fftn::kokkos::compat::Normalization::ortho) == 2);
static_assert(static_cast<int>(fftn::kokkos::compat::Normalization::none) == 3);

TYPED_TEST(KokkosAdapter, CompatNormalizationConvertsToTheAdapters) {
  using T = TypeParam;
  namespace kfft = fftn::kokkos::compat;
  const std::size_t n = 16;
  const std::size_t rows = 2;
  const complex2<T> x("x", rows, n);
  const complex2<T> y("y", rows, n);
  const complex2<T> z("z", rows, n);
  fill(x, 37);
  const kfft::Plan<host, complex2<T>, complex2<T>> p(host{}, x, y, kfft::Direction::backward, -1);
  for (const auto& [compat_norm, norm] : {
           std::pair{kfft::Normalization::forward, normalization::forward},
           std::pair{kfft::Normalization::backward, normalization::backward},
           std::pair{kfft::Normalization::ortho, normalization::ortho},
           std::pair{kfft::Normalization::none, normalization::none},
       }) {
    EXPECT_EQ(fftn::kokkos::detail::normalization_of(compat_norm), norm);
    p.execute(x, z, norm);
    const auto want = reals_of(z);
    fill(y, 0);
    p.execute(x, y, compat_norm);
    expect_same_bits(reals_of(y), want);
    fill(y, 0);
    kfft::execute(p, x, y, compat_norm);
    expect_same_bits(reals_of(y), want);
  }
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): invalid on purpose
  const auto invalid = static_cast<kfft::Normalization>(4);
  expect_refused([&] { p.execute(x, y, invalid); }, "norm");
  expect_refused([&] { kfft::execute(p, x, y, invalid); }, "norm");
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
  const fftn::kokkos::view_plan p(host{}, x, z, direction::forward, -1);
  p.execute(x, z, normalization::none);
  expect_same_bits(reals_of(y), reals_of(z));
  EXPECT_EQ(p.fftn_plan().workspace_bytes(), core.workspace_bytes());

  // std::complex values and a rank-1 View (one transform, no batch) work the same way.
  const Kokkos::View<std::complex<T>*, Kokkos::HostSpace> u("u", n);
  const Kokkos::View<std::complex<T>*, Kokkos::HostSpace> v("v", n);
  fill(u, 23);
  const fftn::kokkos::view_plan q(host{}, u, v, direction::backward, 0);
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

  const fftn::kokkos::view_plan p(host{}, x, y, direction::backward, 0);
  p.execute(x, y, normalization::none);
  expect_same_bits(reals_of(y), want);

  using strided = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const strided xs(x.data(), Kokkos::LayoutStride(n, cols, cols, 1));
  const strided ys(y.data(), Kokkos::LayoutStride(n, cols, cols, 1));
  fill(y, 0);
  const fftn::kokkos::view_plan q(host{}, xs, ys, direction::backward, -2);
  q.execute(xs, ys, normalization::none);
  expect_same_bits(reals_of(y), want);

  using rank4 = Kokkos::View<Kokkos::complex<T>****, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const rank4 u("u", 2, 1, 3, 8);
  const rank4 w("w", 2, 1, 3, 8);
  fill(u, 19);
  const fftn::kokkos::view_plan r(host{}, u, w, direction::forward, 3);
  r.execute(u, w, normalization::none);
  expect_same_bits(reals_of(w),
                   reals_from(core_out_of_place(fftn::rows_c2c<T>(8, 6, direction::forward),
                                                complexes_of(reals_of(u)), 48)));
  const auto s = fftn::kokkos::shape_of(u, w, direction::forward, fftn::kokkos::axis_type<1>{-1});
  ASSERT_EQ(s.batches.size(), 1U);
  EXPECT_EQ(s.batches[0].count, 6U);
}

// Real transforms over several axes, the halved one the last listed: 2-D planes of a batch
// and a 3-D volume, LayoutRight and LayoutLeft (the LayoutLeft Views list the same memory
// with the dimensions reversed, so the axes are listed reversed too).
template <class T, class Layout>
void check_real_multi_axis(const std::array<std::size_t, 3>& dims, std::size_t transformed) {
  constexpr bool right = std::is_same_v<Layout, Kokkos::LayoutRight>;
  using real3 = Kokkos::View<T***, Layout, Kokkos::HostSpace>;
  using complex3 = Kokkos::View<Kokkos::complex<T>***, Layout, Kokkos::HostSpace>;
  // dims are C-ordered (slowest first); the last is halved on the complex side.
  auto half = dims;
  half[2] = dims[2] / 2 + 1;
  const auto view_of = [](auto tag, const std::array<std::size_t, 3>& e, const char* label) {
    using V = decltype(tag)::type;
    return right ? V(label, e[0], e[1], e[2]) : V(label, e[2], e[1], e[0]);
  };
  const real3 x = view_of(std::type_identity<real3>{}, dims, "x");
  const complex3 y = view_of(std::type_identity<complex3>{}, half, "y");
  const real3 z = view_of(std::type_identity<real3>{}, dims, "z");
  fill(x, 29);
  fftn::shape<T, std::complex<T>> forward_shape;
  fftn::shape<std::complex<T>, T> backward_shape;
  backward_shape.dir = direction::backward;
  for (std::size_t d = 3 - transformed; d < 3; ++d) {
    forward_shape.axes.push_back(fftn::axis{.n = dims[d]});
    backward_shape.axes.push_back(fftn::axis{.n = dims[d]});
  }
  if (transformed == 2) {
    forward_shape.batches.push_back(fftn::batch{.count = dims[0]});
    backward_shape.batches.push_back(fftn::batch{.count = dims[0]});
  }
  const std::size_t points = dims[0] * dims[1] * dims[2];
  const std::size_t halves = half[0] * half[1] * half[2];
  const auto spectrum = core_out_of_place(forward_shape, reals_of(x), halves);
  const auto back = core_out_of_place(backward_shape, spectrum, points);
  if (transformed == 2) {
    const fftn::kokkos::axis_type<2> axes =
        right ? fftn::kokkos::axis_type<2>{1, 2} : fftn::kokkos::axis_type<2>{1, 0};
    const fftn::kokkos::view_plan<host, real3, complex3, 2> f(host{}, x, y, direction::forward,
                                                              axes);
    const fftn::kokkos::view_plan<host, complex3, real3, 2> b(host{}, y, z, direction::backward,
                                                              axes);
    f.execute(x, y, normalization::none);
    expect_same_bits(reals_of(y), reals_from(spectrum));
    b.execute(y, z, normalization::none);
  } else {
    const fftn::kokkos::axis_type<3> axes =
        right ? fftn::kokkos::axis_type<3>{-3, -2, -1} : fftn::kokkos::axis_type<3>{2, 1, 0};
    const fftn::kokkos::view_plan<host, real3, complex3, 3> f(host{}, x, y, direction::forward,
                                                              axes);
    const fftn::kokkos::view_plan<host, complex3, real3, 3> b(host{}, y, z, direction::backward,
                                                              axes);
    f.execute(x, y, normalization::none);
    expect_same_bits(reals_of(y), reals_from(spectrum));
    b.execute(y, z, normalization::none);
  }
  expect_same_bits(reals_of(z), back);
}

TYPED_TEST(KokkosAdapter, MultiAxisRealPlansHalveTheLastAxisListed) {
  using T = TypeParam;
  check_real_multi_axis<T, Kokkos::LayoutRight>({2, 5, 6}, 2);
  check_real_multi_axis<T, Kokkos::LayoutLeft>({2, 5, 6}, 2);
  check_real_multi_axis<T, Kokkos::LayoutRight>({3, 4, 7}, 3);
  check_real_multi_axis<T, Kokkos::LayoutLeft>({3, 4, 7}, 3);
}

// Axes listed out of memory order: {-1, -2} transforms x first in FFTN's order, and for a
// real transform halves y, the last listed, which is not the contiguous dimension.
TYPED_TEST(KokkosAdapter, AxesOutOfMemoryOrderFollowTheList) {
  using T = TypeParam;
  using complex3 = Kokkos::View<Kokkos::complex<T>***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using real3 = Kokkos::View<T***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const std::size_t count = 2;
  const std::size_t ny = 6;
  const std::size_t nx = 5;
  const auto nxs = static_cast<std::ptrdiff_t>(nx);
  const fftn::kokkos::axis_type<2> axes{-1, -2};

  const complex3 x("x", count, ny, nx);
  const complex3 y("y", count, ny, nx);
  fill(x, 67);
  fftn::shape<std::complex<T>, std::complex<T>> c;
  c.axes.push_back(fftn::axis{.n = nx, .in_stride = 1, .out_stride = 1});
  c.axes.push_back(fftn::axis{.n = ny, .in_stride = nxs, .out_stride = nxs});
  c.batches.push_back(fftn::batch{.count = count, .in_dist = 30, .out_dist = 30});
  const fftn::kokkos::view_plan<host, complex3, complex3, 2> p(host{}, x, y, direction::forward,
                                                               axes);
  p.execute(x, y, normalization::none);
  expect_same_bits(reals_of(y),
                   reals_from(core_out_of_place(c, complexes_of(reals_of(x)), count * ny * nx)));

  const std::size_t hy = ny / 2 + 1;
  const real3 xr("xr", count, ny, nx);
  const complex3 yh("yh", count, hy, nx);
  fill(xr, 71);
  fftn::shape<T, std::complex<T>> r;
  r.axes.push_back(fftn::axis{.n = nx, .in_stride = 1, .out_stride = 1});
  r.axes.push_back(fftn::axis{.n = ny, .in_stride = nxs, .out_stride = nxs});
  r.batches.push_back(fftn::batch{.count = count, .in_dist = 30, .out_dist = 20});
  const fftn::kokkos::view_plan<host, real3, complex3, 2> q(host{}, xr, yh, direction::forward,
                                                            axes);
  q.execute(xr, yh, normalization::none);
  expect_same_bits(reals_of(yh), reals_from(core_out_of_place(r, reals_of(xr), count * hy * nx)));
}

// LayoutLeft planes, in place: (nx, ny, count) along {1, 0} is the memory of the core's
// planes_c2c over (count, ny, nx).
TYPED_TEST(KokkosAdapter, LayoutLeftPlanesInPlaceMatchTheCoreApi) {
  using T = TypeParam;
  using left3 = Kokkos::View<Kokkos::complex<T>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  const left3 v("v", 8, 15, 3);
  fill(v, 73);
  const auto start = complexes_of(reals_of(v));
  const fftn::kokkos::view_plan<host, left3, left3, 2> p(host{}, v, v, direction::backward,
                                                         fftn::kokkos::axis_type<2>{1, 0});
  p.execute(v, v, normalization::none);
  const auto want = core_in_place<T>(fftn::planes_c2c<T>(15, 8, 3, direction::backward), start);
  expect_same_bits(reals_of(v), reals_from(want));
}

// A middle axis leaves two batch loops that do not merge: (3, 8, 4) along axis 1.
TYPED_TEST(KokkosAdapter, TwoBatchLoopsRemainForAMiddleAxis) {
  using T = TypeParam;
  using complex3 = Kokkos::View<Kokkos::complex<T>***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const complex3 x("x", 3, 8, 4);
  const complex3 y("y", 3, 8, 4);
  fill(x, 79);
  const auto s = fftn::kokkos::shape_of(x, y, direction::forward, fftn::kokkos::axis_type<1>{1});
  ASSERT_EQ(s.batches.size(), 2U);
  fftn::shape<std::complex<T>, std::complex<T>> want_shape;
  want_shape.axes.push_back(fftn::axis{.n = 8, .in_stride = 4, .out_stride = 4});
  want_shape.batches.push_back(fftn::batch{.count = 3, .in_dist = 32, .out_dist = 32});
  want_shape.batches.push_back(fftn::batch{.count = 4, .in_dist = 1, .out_dist = 1});
  const fftn::kokkos::view_plan p(host{}, x, y, direction::forward, 1);
  p.execute(x, y, normalization::none);
  expect_same_bits(reals_of(y),
                   reals_from(core_out_of_place(want_shape, complexes_of(reals_of(x)), 96)));
}

// Views whose other dimensions leave three batch loops that do not nest: a rank-5 View along
// axes {1, 3}, out of place and in place, and the real kinds over it; then a rank-4
// LayoutStride View along its last axis whose three batch dimensions are padded apart.
TYPED_TEST(KokkosAdapter, ThreeBatchLoopsThatDoNotNest) {
  using T = TypeParam;
  using complex5 = Kokkos::View<Kokkos::complex<T>*****, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using real5 = Kokkos::View<T*****, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const fftn::kokkos::axis_type<2> axes{1, 3};
  const complex5 x("x", 2, 4, 3, 6, 2);
  const complex5 y("y", 2, 4, 3, 6, 2);
  fill(x, 89);
  const auto s = fftn::kokkos::shape_of(x, y, direction::forward, axes);
  ASSERT_EQ(s.batches.size(), 3U);
  // (2, 4, 3, 6, 2) in C order: strides 144, 36, 12, 2, 1.
  const fftn::shape<std::complex<T>, std::complex<T>> c{
      .dir = direction::forward,
      .axes =
          {
              {.n = 4, .in_stride = 36, .out_stride = 36},
              {.n = 6, .in_stride = 2, .out_stride = 2},
          },
      .batches =
          {
              {.count = 2, .in_dist = 144, .out_dist = 144},
              {.count = 3, .in_dist = 12, .out_dist = 12},
              {.count = 2, .in_dist = 1, .out_dist = 1},
          },
  };
  const auto input = complexes_of(reals_of(x));
  const fftn::kokkos::view_plan<host, complex5, complex5, 2> p(host{}, x, y, direction::forward,
                                                               axes);
  p.execute(x, y, normalization::none);
  expect_same_bits(reals_of(y), reals_from(core_out_of_place(c, input, 288)));
  const fftn::kokkos::view_plan<host, complex5, complex5, 2> q(host{}, x, x, direction::forward,
                                                               axes);
  q.execute(x, x, normalization::none);
  expect_same_bits(reals_of(x), reals_from(core_in_place<T>(c, input)));

  // Real: (2, 4, 3, 6, 2) reals to (2, 4, 3, 4, 2) complex values, axis 3 halved.
  const real5 xr("xr", 2, 4, 3, 6, 2);
  const complex5 yh("yh", 2, 4, 3, 4, 2);
  const real5 zr("zr", 2, 4, 3, 6, 2);
  fill(xr, 97);
  const fftn::shape<T, std::complex<T>> rf{
      .axes =
          {
              {.n = 4, .in_stride = 36, .out_stride = 24},
              {.n = 6, .in_stride = 2, .out_stride = 2},
          },
      .batches =
          {
              {.count = 2, .in_dist = 144, .out_dist = 96},
              {.count = 3, .in_dist = 12, .out_dist = 8},
              {.count = 2, .in_dist = 1, .out_dist = 1},
          },
  };
  fftn::shape<std::complex<T>, T> rb;
  rb.dir = direction::backward;
  for (const fftn::axis& a : rf.axes) {
    rb.axes.push_back({.n = a.n, .in_stride = a.out_stride, .out_stride = a.in_stride});
  }
  for (const fftn::batch& b : rf.batches) {
    rb.batches.push_back({.count = b.count, .in_dist = b.out_dist, .out_dist = b.in_dist});
  }
  const fftn::kokkos::view_plan<host, real5, complex5, 2> f(host{}, xr, yh, direction::forward,
                                                            axes);
  const fftn::kokkos::view_plan<host, complex5, real5, 2> b(host{}, yh, zr, direction::backward,
                                                            axes);
  f.execute(xr, yh, normalization::none);
  const auto spectrum = core_out_of_place(rf, reals_of(xr), 192);
  expect_same_bits(reals_of(yh), reals_from(spectrum));
  b.execute(yh, zr, normalization::none);
  expect_same_bits(reals_of(zr), core_out_of_place(rb, spectrum, 288));

  using strided4 = Kokkos::View<Kokkos::complex<T>****, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const Kokkos::View<Kokkos::complex<T>*, Kokkos::HostSpace> pool("pool", 5400);
  const Kokkos::View<Kokkos::complex<T>*, Kokkos::HostSpace> pool2("pool2", 5400);
  const Kokkos::LayoutStride loose(2, 4000, 2, 1000, 2, 300, 8, 1);
  const strided4 u(pool.data(), loose);
  const strided4 v(pool2.data(), loose);
  fill(pool, 101);
  const fftn::kokkos::view_plan<host, strided4, strided4> r(host{}, u, v, direction::backward, -1);
  r.execute(u, v, normalization::none);
  const fftn::shape<std::complex<T>, std::complex<T>> loose_shape{
      .dir = direction::backward,
      .axes = {{.n = 8, .in_stride = 1, .out_stride = 1}},
      .batches =
          {
              {.count = 2, .in_dist = 4000, .out_dist = 4000},
              {.count = 2, .in_dist = 1000, .out_dist = 1000},
              {.count = 2, .in_dist = 300, .out_dist = 300},
          },
  };
  expect_same_bits(reals_of(pool2),
                   reals_from(core_out_of_place(loose_shape, complexes_of(reals_of(pool)), 5400)));
}

// Real kinds through LayoutStride: rows a few values apart on both sides.
TYPED_TEST(KokkosAdapter, RealKindsThroughLayoutStride) {
  using T = TypeParam;
  using cs = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  using rs = Kokkos::View<T**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const std::size_t rows = 3;
  const std::size_t n = 10;
  const std::size_t h = n / 2 + 1;
  const real2<T> real_store("real_store", rows, n + 3);
  const complex2<T> complex_store("complex_store", rows, h + 2);
  const real2<T> back_store("back_store", rows, n + 3);
  fill(real_store, 83);
  const rs x(real_store.data(), Kokkos::LayoutStride(rows, n + 3, n, 1));
  const cs y(complex_store.data(), Kokkos::LayoutStride(rows, h + 2, h, 1));
  const rs z(back_store.data(), Kokkos::LayoutStride(rows, n + 3, n, 1));
  fftn::shape<T, std::complex<T>> f;
  f.axes.push_back(fftn::axis{.n = n, .in_stride = 1, .out_stride = 1});
  f.batches.push_back(fftn::batch{.count = rows, .in_dist = 13, .out_dist = 8});
  fftn::shape<std::complex<T>, T> b;
  b.dir = direction::backward;
  b.axes.push_back(fftn::axis{.n = n, .in_stride = 1, .out_stride = 1});
  b.batches.push_back(fftn::batch{.count = rows, .in_dist = 8, .out_dist = 13});
  // The core's view of the same memory, whole stores: gaps stay zero on both sides.
  const auto spectrum = core_out_of_place(f, reals_of(real_store), rows * (h + 2));
  const fftn::kokkos::view_plan pf(host{}, x, y, direction::forward, -1);
  pf.execute(x, y, normalization::none);
  expect_same_bits(reals_of(complex_store), reals_from(spectrum));
  const auto back = core_out_of_place(b, spectrum, rows * (n + 3));
  const fftn::kokkos::view_plan pb(host{}, y, z, direction::backward, -1);
  pb.execute(y, z, normalization::none);
  expect_same_bits(reals_of(back_store), back);
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
