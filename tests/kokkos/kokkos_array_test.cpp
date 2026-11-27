// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fftn::kokkos::array_of (task 5.5, dev/DESIGN.md 5.5): a pair of Views as the `axis` list of
// the core's axes_* functions and of pipelines. Its entries are the Views' extents and strides;
// fed to axes_* with the axes shape_of takes, it gives shape_of's shape, and the plans and
// pipelines built from it give the same bits as those built from the equivalent shape or
// description by hand; it refuses what shape_of refuses, with shape_of's messages.

#include "kokkos/kokkos_helpers.hpp"

#include <fftn/fftn.hpp>
#include <fftn/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace fftn::test::kokkos_adapter;

template <class T> class KokkosArrayOf : public ::testing::Test {};
TYPED_TEST_SUITE(KokkosArrayOf, precisions);

// array_of's entries are the Views' extents (the real side's on the halved dimension) and
// strides, dimension by dimension.
template <class InView, class OutView>
void expect_entries(const std::array<fftn::axis, InView::rank()>& array, const InView& in,
                    const OutView& out, std::size_t halved = InView::rank()) {
  for (std::size_t d = 0; d < InView::rank(); ++d) {
    const std::size_t n =
        d == halved && std::is_floating_point_v<typename OutView::non_const_value_type>
            ? out.extent(d)
            : in.extent(d);
    EXPECT_EQ(array[d].n, n) << "dimension " << d;
    EXPECT_EQ(array[d].in_stride, static_cast<std::ptrdiff_t>(in.stride(d))) << "dimension " << d;
    EXPECT_EQ(array[d].out_stride, static_cast<std::ptrdiff_t>(out.stride(d))) << "dimension " << d;
  }
}

template <class In, class Out>
void expect_same_shape(const fftn::shape<In, Out>& got, const fftn::shape<In, Out>& want) {
  EXPECT_EQ(got.dir, want.dir);
  ASSERT_EQ(got.axes.size(), want.axes.size());
  for (std::size_t i = 0; i < got.axes.size(); ++i) {
    EXPECT_EQ(got.axes[i].n, want.axes[i].n) << "axis " << i;
    EXPECT_EQ(got.axes[i].in_stride, want.axes[i].in_stride) << "axis " << i;
    EXPECT_EQ(got.axes[i].out_stride, want.axes[i].out_stride) << "axis " << i;
  }
  ASSERT_EQ(got.batches.size(), want.batches.size());
  for (std::size_t b = 0; b < got.batches.size(); ++b) {
    EXPECT_EQ(got.batches[b].count, want.batches[b].count) << "batch " << b;
    EXPECT_EQ(got.batches[b].in_dist, want.batches[b].in_dist) << "batch " << b;
    EXPECT_EQ(got.batches[b].out_dist, want.batches[b].out_dist) << "batch " << b;
  }
}

// shape_of's axes as the non-negative indices axes_* take.
template <std::size_t Dim>
std::array<std::size_t, Dim> indices(const fftn::kokkos::axis_type<Dim>& axes, std::size_t rank) {
  std::array<std::size_t, Dim> dims{};
  for (std::size_t i = 0; i < Dim; ++i) {
    const int a = axes[i];
    dims[i] = static_cast<std::size_t>(a < 0 ? a + static_cast<int>(rank) : a);
  }
  return dims;
}

// For c2c Views x and y along `axes`: the entries, axes_c2c of them against shape_of, and a plan
// of axes_c2c's shape against make_plan, bit for bit. The data are filled and read through
// x_memory and y_memory, the Views that own the memory: a subview's span() can reach past its
// parent's allocation.
template <class T, class View, std::size_t Dim, class XMemory, class YMemory>
void check_c2c(const View& x, const View& y, const fftn::kokkos::axis_type<Dim>& axes,
               const XMemory& x_memory, const YMemory& y_memory) {
  const auto array = fftn::kokkos::array_of(x, y);
  expect_entries(array, x, y);
  const auto dims = indices(axes, View::rank());
  for (const direction dir : {direction::forward, direction::backward}) {
    const auto s = fftn::axes_c2c<T>(array, dims, dir);
    expect_same_shape(s, fftn::kokkos::shape_of(x, y, dir, axes));
    fill(x_memory, 17);
    const fftn::plan<std::complex<T>, std::complex<T>> p(s);
    fftn::workspace ws(p);
    p.execute(fftn::kokkos::data_of(x), fftn::kokkos::data_of(y), ws);
    const auto got = reals_of(y_memory);
    const auto q = fftn::kokkos::make_plan(x, y, dir, axes);
    fftn::workspace qws(q);
    q.execute(fftn::kokkos::data_of(x), fftn::kokkos::data_of(y), qws);
    expect_same_bits(got, reals_of(y_memory));
  }
}

template <class T, class View, std::size_t Dim>
void check_c2c(const View& x, const View& y, const fftn::kokkos::axis_type<Dim>& axes) {
  check_c2c<T>(x, y, axes, x, y);
}

template <class T> using c1 = Kokkos::View<Kokkos::complex<T>*, Kokkos::HostSpace>;
template <class T, class Layout>
using c3 = Kokkos::View<Kokkos::complex<T>***, Layout, Kokkos::HostSpace>;
template <class T, class Layout> using r3 = Kokkos::View<T***, Layout, Kokkos::HostSpace>;

TYPED_TEST(KokkosArrayOf, ComplexViewsOfRankOneToThreeInBothLayouts) {
  using T = TypeParam;
  const c1<T> a("a", 12);
  const c1<T> b("b", 12);
  check_c2c<T>(a, b, fftn::kokkos::axis_type<1>{0});

  const complex2<T> right_x("right_x", 5, 12);
  const complex2<T> right_y("right_y", 5, 12);
  check_c2c<T>(right_x, right_y, fftn::kokkos::axis_type<1>{-1});
  check_c2c<T>(right_x, right_y, fftn::kokkos::axis_type<1>{0});
  const complex2<T, Kokkos::LayoutLeft> left_x("left_x", 12, 5);
  const complex2<T, Kokkos::LayoutLeft> left_y("left_y", 12, 5);
  check_c2c<T>(left_x, left_y, fftn::kokkos::axis_type<1>{0});
  check_c2c<T>(left_x, left_y, fftn::kokkos::axis_type<2>{1, 0});

  const c3<T, Kokkos::LayoutRight> rx("rx", 3, 8, 6);
  const c3<T, Kokkos::LayoutRight> ry("ry", 3, 8, 6);
  check_c2c<T>(rx, ry, fftn::kokkos::axis_type<2>{-2, -1});
  check_c2c<T>(rx, ry, fftn::kokkos::axis_type<1>{1});
  check_c2c<T>(rx, ry, fftn::kokkos::axis_type<3>{0, 2, 1});
  const c3<T, Kokkos::LayoutLeft> lx("lx", 6, 8, 3);
  const c3<T, Kokkos::LayoutLeft> ly("ly", 6, 8, 3);
  check_c2c<T>(lx, ly, fftn::kokkos::axis_type<2>{1, 0});
  check_c2c<T>(lx, ly, fftn::kokkos::axis_type<1>{2});
}

// Strided subviews (LayoutStride): a block of planes, rows and columns inside a larger View, in
// both layouts, so no dimension is tight.
TYPED_TEST(KokkosArrayOf, StridedSubviews) {
  using T = TypeParam;
  const c3<T, Kokkos::LayoutRight> whole_x("whole_x", 6, 10, 9);
  const c3<T, Kokkos::LayoutRight> whole_y("whole_y", 6, 10, 9);
  const auto band = [](const auto& v) {
    return Kokkos::subview(v, std::make_pair(1, 4), std::make_pair(2, 10), std::make_pair(1, 7));
  };
  const auto x = band(whole_x);
  const auto y = band(whole_y);
  static_assert(std::is_same_v<typename decltype(x)::array_layout, Kokkos::LayoutStride>);
  check_c2c<T>(x, y, fftn::kokkos::axis_type<1>{-1}, whole_x, whole_y);
  check_c2c<T>(x, y, fftn::kokkos::axis_type<2>{0, 1}, whole_x, whole_y);

  const c3<T, Kokkos::LayoutLeft> left_x("left_x", 6, 10, 9);
  const c3<T, Kokkos::LayoutLeft> left_y("left_y", 6, 10, 9);
  check_c2c<T>(band(left_x), band(left_y), fftn::kokkos::axis_type<2>{1, 2}, left_x, left_y);
}

// For an r2c pair (x, y) and its c2r inverse (y, z) along `axes`, the last listed axis halved:
// the entries, axes_r2c and axes_c2r of them against shape_of, and plans of them against
// make_plan, bit for bit. `named` passes the halved dimension to array_of; otherwise the
// two-argument form takes the last.
template <class T, class Real, class Complex, std::size_t Dim>
void check_real(const Real& x, const Complex& y, const Real& z,
                const fftn::kokkos::axis_type<Dim>& axes, bool named) {
  const auto dims = indices(axes, Real::rank());
  const int halved = static_cast<int>(dims[Dim - 1]);
  const auto forward = named ? fftn::kokkos::array_of(x, y, halved) : fftn::kokkos::array_of(x, y);
  const auto backward = named ? fftn::kokkos::array_of(y, z, halved) : fftn::kokkos::array_of(y, z);
  expect_entries(forward, x, y, dims[Dim - 1]);
  expect_entries(backward, y, z, dims[Dim - 1]);
  const auto rs = fftn::axes_r2c<T>(forward, dims);
  const auto cs = fftn::axes_c2r<T>(backward, dims);
  expect_same_shape(rs, fftn::kokkos::shape_of(x, y, direction::forward, axes));
  expect_same_shape(cs, fftn::kokkos::shape_of(y, z, direction::backward, axes));

  fill(x, 23);
  const fftn::plan<T, std::complex<T>> r2c(rs);
  fftn::workspace ws(r2c);
  r2c.execute(fftn::kokkos::data_of(x), fftn::kokkos::data_of(y), ws);
  const auto spectrum = reals_of(y);
  const auto r2c_want = fftn::kokkos::make_plan(x, y, direction::forward, axes);
  fftn::workspace want_ws(r2c_want);
  r2c_want.execute(fftn::kokkos::data_of(x), fftn::kokkos::data_of(y), want_ws);
  expect_same_bits(spectrum, reals_of(y));

  const fftn::plan<std::complex<T>, T> c2r(cs);
  fftn::workspace back_ws(c2r);
  c2r.execute(fftn::kokkos::data_of(y), fftn::kokkos::data_of(z), back_ws);
  const auto back = reals_of(z);
  const auto c2r_want = fftn::kokkos::make_plan(y, z, direction::backward, axes);
  fftn::workspace back_want_ws(c2r_want);
  c2r_want.execute(fftn::kokkos::data_of(y), fftn::kokkos::data_of(z), back_want_ws);
  expect_same_bits(back, reals_of(z));
}

TYPED_TEST(KokkosArrayOf, RealViewsOfRankOneToThreeInBothLayouts) {
  using T = TypeParam;
  const Kokkos::View<T*, Kokkos::HostSpace> x1("x1", 10);
  const c1<T> y1("y1", 6);
  const Kokkos::View<T*, Kokkos::HostSpace> z1("z1", 10);
  check_real<T>(x1, y1, z1, fftn::kokkos::axis_type<1>{0}, false);
  check_real<T>(x1, y1, z1, fftn::kokkos::axis_type<1>{-1}, true);

  // (rows, n) halved along the last dimension; LayoutLeft (n, rows) along the first, named.
  const real2<T> x2("x2", 4, 9);
  const complex2<T> y2("y2", 4, 5);
  const real2<T> z2("z2", 4, 9);
  check_real<T>(x2, y2, z2, fftn::kokkos::axis_type<1>{-1}, false);
  check_real<T>(x2, y2, z2, fftn::kokkos::axis_type<1>{1}, true);
  const real2<T, Kokkos::LayoutLeft> lx2("lx2", 9, 4);
  const complex2<T, Kokkos::LayoutLeft> ly2("ly2", 5, 4);
  const real2<T, Kokkos::LayoutLeft> lz2("lz2", 9, 4);
  check_real<T>(lx2, ly2, lz2, fftn::kokkos::axis_type<1>{0}, true);
  check_real<T>(lx2, ly2, lz2, fftn::kokkos::axis_type<1>{-2}, true);

  // Rank 3: planes halved along the fastest dimension, and a halved middle dimension.
  const r3<T, Kokkos::LayoutRight> x3("x3", 3, 6, 8);
  const c3<T, Kokkos::LayoutRight> y3("y3", 3, 6, 5);
  const r3<T, Kokkos::LayoutRight> z3("z3", 3, 6, 8);
  check_real<T>(x3, y3, z3, fftn::kokkos::axis_type<2>{1, 2}, false);
  check_real<T>(x3, y3, z3, fftn::kokkos::axis_type<3>{0, 1, 2}, true);
  const r3<T, Kokkos::LayoutRight> mx3("mx3", 3, 8, 6);
  const c3<T, Kokkos::LayoutRight> my3("my3", 3, 5, 6);
  const r3<T, Kokkos::LayoutRight> mz3("mz3", 3, 8, 6);
  check_real<T>(mx3, my3, mz3, fftn::kokkos::axis_type<2>{2, 1}, true);
  const r3<T, Kokkos::LayoutLeft> lx3("lx3", 8, 6, 3);
  const c3<T, Kokkos::LayoutLeft> ly3("ly3", 5, 6, 3);
  const r3<T, Kokkos::LayoutLeft> lz3("lz3", 8, 6, 3);
  check_real<T>(lx3, ly3, lz3, fftn::kokkos::axis_type<2>{1, 0}, true);
}

// Real rows in place: the real View shows FFTW's padded rows over the complex memory. array_of
// does not compare the Views' addresses; the caller sets options::in_place, and the plan checks
// the padding from the axes.
TYPED_TEST(KokkosArrayOf, RealRowsInPlaceWithOptionsInPlace) {
  using T = TypeParam;
  using unmanaged_real = Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const std::size_t rows = 3;
  const std::size_t n = 10;
  const std::size_t h = n / 2 + 1;
  const complex2<T> c("c", rows, h);
  const unmanaged_real padded(reinterpret_cast<T*>(c.data()), rows, 2 * h);
  const auto r = Kokkos::subview(padded, Kokkos::ALL, std::make_pair(std::size_t{0}, n));
  fill(c, 29);
  const auto start = complexes_of(reals_of(c));

  const auto array = fftn::kokkos::array_of(r, c);
  EXPECT_EQ(array[0].in_stride, static_cast<std::ptrdiff_t>(2 * h));
  EXPECT_EQ(array[0].out_stride, static_cast<std::ptrdiff_t>(h));
  const std::array<std::size_t, 1> last{1};
  const fftn::plan<T, std::complex<T>> p(fftn::axes_r2c<T>(array, last), {.in_place = true});
  fftn::workspace ws(p);
  p.execute(fftn::kokkos::data_of(r), fftn::kokkos::data_of(c), ws);
  expect_same_bits(reals_of(c), reals_from(core_in_place<T>(fftn::rows_r2c<T>(n, rows), start)));
}

// Two real Views: the array of axes_r2r, equal extents and no halved dimension.
TYPED_TEST(KokkosArrayOf, RealPairsGiveTheArrayOfAxesR2r) {
  using T = TypeParam;
  const real2<T, Kokkos::LayoutLeft> x("x", 8, 3);
  const real2<T, Kokkos::LayoutLeft> y("y", 8, 3);
  const auto array = fftn::kokkos::array_of(x, y);
  expect_entries(array, x, y);
  const std::array<std::size_t, 1> first{0};
  const std::array<fftn::r2r_kind, 1> kinds{fftn::r2r_kind::dct2};
  const auto s = fftn::axes_r2r<T>(array, first, kinds);
  ASSERT_EQ(s.batches.size(), 1U);
  EXPECT_EQ(s.batches[0].count, 3U);
  EXPECT_EQ(s.batches[0].in_dist, 8);
  expect_refused(
      [&] {
        static_cast<void>(fftn::kokkos::array_of(x, real2<T, Kokkos::LayoutLeft>("w", 8, 4)));
      },
      "out.extent(1)");
}

// A pipeline's line and batches written from array_of's entries: every column of a LayoutRight
// (n, count) View convolved with one kernel, bit for bit as the description written by hand.
TYPED_TEST(KokkosArrayOf, PipelineLineAndBatchesFromTheEntries) {
  using T = TypeParam;
  const std::size_t n = 16;
  const std::size_t count = 5;
  const complex2<T> x("x", n, count);
  const complex2<T> y("y", n, count);
  const complex2<T> want_y("want_y", n, count);
  fill(x, 31);
  std::vector<std::complex<T>> kernel(n);
  for (std::size_t k = 0; k < n; ++k) {
    kernel[k] = {static_cast<T>(1.0 / static_cast<double>(k + 1)), static_cast<T>(0.25)};
  }

  const auto array = fftn::kokkos::array_of(x, y);
  fftn::pipeline_desc desc = fftn::complex_convolution_rows<T>(n, count);
  desc.line = array[0];
  desc.batches = {
      {.count = array[1].n, .in_dist = array[1].in_stride, .out_dist = array[1].out_stride},
  };
  const fftn::pipeline pipe(desc);
  fftn::workspace ws(pipe);
  pipe.execute(fftn::kokkos::data_of(x), fftn::kokkos::data_of(y), ws,
               {.mid_factor = kernel.data()});

  fftn::pipeline_desc by_hand = fftn::complex_convolution_rows<T>(n, count);
  by_hand.line = {
      .n = n,
      .in_stride = static_cast<std::ptrdiff_t>(count),
      .out_stride = static_cast<std::ptrdiff_t>(count),
  };
  by_hand.batches = {{.count = count, .in_dist = 1, .out_dist = 1}};
  const fftn::pipeline want(by_hand);
  fftn::workspace want_ws(want);
  want.execute(fftn::kokkos::data_of(x), fftn::kokkos::data_of(want_y), want_ws,
               {.mid_factor = kernel.data()});
  expect_same_bits(reals_of(y), reals_of(want_y));
}

// A dimension of extent 1 whose stride Kokkos reports as 0 is given stride 1, so axes_* do not
// read it as "the tight default"; strides of 0 elsewhere are refused.
TYPED_TEST(KokkosArrayOf, StridesAreExplicit) {
  using T = TypeParam;
  using strided = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const c1<T> store_x("store_x", 8);
  const c1<T> store_y("store_y", 8);
  const strided x(store_x.data(), Kokkos::LayoutStride(1, 0, 8, 1));
  const strided y(store_y.data(), Kokkos::LayoutStride(1, 0, 8, 1));
  const auto array = fftn::kokkos::array_of(x, y);
  EXPECT_EQ(array[0].n, 1U);
  EXPECT_EQ(array[0].in_stride, 1);
  EXPECT_EQ(array[0].out_stride, 1);
  EXPECT_EQ(array[1].in_stride, 1);
  const strided broadcast(store_x.data(), Kokkos::LayoutStride(2, 0, 4, 1));
  const strided fine(store_y.data(), Kokkos::LayoutStride(2, 4, 4, 1));
  expect_refused([&] { static_cast<void>(fftn::kokkos::array_of(broadcast, fine)); },
                 "in.stride(0)");
  expect_refused([&] { static_cast<void>(fftn::kokkos::array_of(fine, broadcast)); },
                 "out.stride(0)");
}

// The message with which `f` throws std::invalid_argument, or "accepted".
std::string refusal(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "accepted";
}

// Mismatched extents are refused as shape_of refuses them: the same argument, the same message.
TYPED_TEST(KokkosArrayOf, RefusesWhatShapeOfRefuses) {
  using T = TypeParam;
  const fftn::kokkos::axis_type<1> last{-1};
  const auto same = [&](const auto& in, const auto& out, direction dir,
                        const std::string& argument) {
    expect_refused([&] { static_cast<void>(fftn::kokkos::array_of(in, out)); }, argument);
    EXPECT_EQ(refusal([&] { static_cast<void>(fftn::kokkos::array_of(in, out)); }),
              refusal([&] { static_cast<void>(fftn::kokkos::shape_of(in, out, dir, last)); }));
  };

  // c2c: an extent differs on a transformed or a batch dimension; an extent is 0.
  const complex2<T> a("a", 4, 12);
  same(a, complex2<T>("wide", 4, 13), direction::forward, "out.extent(1)");
  same(a, complex2<T>("tall", 5, 12), direction::forward, "out.extent(0)");
  same(complex2<T>("empty", 0, 12), a, direction::forward, "in.extent(0)");
  same(a, complex2<T>("empty", 4, 0), direction::forward, "out.extent(1)");
  const c3<T, Kokkos::LayoutLeft> l("l", 6, 4, 3);
  same(l, c3<T, Kokkos::LayoutLeft>("l2", 6, 4, 2), direction::forward, "out.extent(2)");

  // r2c and c2r: the complex side is not n/2 + 1 along the halved dimension, or a batch
  // dimension differs.
  const real2<T> x("x", 4, 10);
  const complex2<T> half("half", 4, 6);
  same(x, complex2<T>("short", 4, 5), direction::forward, "out.extent(1)");
  same(x, complex2<T>("few", 3, 6), direction::forward, "out.extent(0)");
  same(complex2<T>("short", 4, 5), x, direction::backward, "in.extent(1)");
  same(half, real2<T>("few", 3, 10), direction::backward, "out.extent(0)");

  // The halved dimension named: out of range, or extents that fit another dimension only.
  expect_refused([&] { static_cast<void>(fftn::kokkos::array_of(x, half, 2)); }, "halved");
  expect_refused([&] { static_cast<void>(fftn::kokkos::array_of(x, half, -3)); }, "halved");
  expect_refused([&] { static_cast<void>(fftn::kokkos::array_of(x, half, 0)); }, "out.extent(0)");
  const real2<T> square("square", 10, 10);
  const complex2<T> halved_first("halved_first", 6, 10);
  EXPECT_NO_THROW(static_cast<void>(fftn::kokkos::array_of(square, halved_first, 0)));
  expect_refused([&] { static_cast<void>(fftn::kokkos::array_of(square, halved_first)); },
                 "out.extent(0)");
}

} // namespace
