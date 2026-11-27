// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The one-shot functions of fftn::kokkos::compat (step 6.7), Kokkos-FFT's fft and the rest, on
// Kokkos' Serial back end and the build's threaded one: every function bit for bit what a
// view_plan of the same Views executes with the same normalisation (or, for a real fft input,
// hfft and ihfft, the view_plan of the transform Kokkos-FFT runs on its temporary), every
// normalisation, default and explicit axes, LayoutLeft and LayoutStride, in place, an output large
// enough to share its scaling; a repeated call keeps one plan in the calling thread's convenience
// context, and host threads calling at once each keep their own; and refusals as view_plan's.

#include "kokkos/kokkos_helpers.hpp"

#include <fftn/easy.hpp>
#include <fftn/fftn.hpp>
#include <fftn/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <cstddef>
#include <latch>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using namespace fftn::test::kokkos_adapter;
namespace kfft = fftn::kokkos::compat;

template <class Space, class T> struct config {
  using space = Space;
  using real = T;
};

#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
using configs = ::testing::Types<config<Kokkos::Serial, double>, config<Kokkos::Serial, float>,
                                 config<host, double>, config<host, float>>;
#else
using configs = ::testing::Types<config<Kokkos::Serial, double>, config<Kokkos::Serial, float>>;
#endif

template <class C> class KokkosOneShot : public ::testing::Test {};
TYPED_TEST_SUITE(KokkosOneShot, configs);

template <class T, class Layout = Kokkos::LayoutRight>
using complex3 = Kokkos::View<Kokkos::complex<T>***, Layout, Kokkos::HostSpace>;
template <class T, class Layout = Kokkos::LayoutRight>
using real3 = Kokkos::View<T***, Layout, Kokkos::HostSpace>;

constexpr std::array norms{kfft::Normalization::backward, kfft::Normalization::forward,
                           kfft::Normalization::ortho, kfft::Normalization::none};

std::string name_of(kfft::Normalization n) { return std::to_string(static_cast<int>(n)); }

// fftn (or, with `real`, rfftn) over all three dimensions as call sites write it: the default
// arguments for the default normalisation, else the axes spelt out (a braced {} would be three
// zeros, in Kokkos-FFT too).
template <bool Real, class S, class In, class Out>
void all_axes(const S& space, const In& in, const Out& out, kfft::Normalization nm) {
  if constexpr (Real) {
    if (nm == kfft::Normalization::backward) {
      kfft::rfftn(space, in, out);
    } else {
      kfft::rfftn(space, in, out, kfft::axis_type<3>{-3, -2, -1}, nm);
    }
  } else if (nm == kfft::Normalization::backward) {
    kfft::fftn(space, in, out);
  } else {
    kfft::fftn(space, in, out, kfft::axis_type<3>{-3, -2, -1}, nm);
  }
}

// Calls `one_shot(in, out, norm)` and `planned(in, want, norm)` on copies of the same input and
// expects the same bits in `out` and `want`, for every normalisation.
template <class InView, class OutView, class F, class G>
void expect_as_planned(const InView& in, const OutView& out, const OutView& want, F one_shot,
                       G planned, const std::string& what) {
  for (const kfft::Normalization norm : norms) {
    fill(out, 1);
    fill(want, 2);
    one_shot(in, out, norm);
    planned(in, want, norm);
    SCOPED_TRACE(what + ", norm " + name_of(norm));
    expect_same_bits(reals_of(out), reals_of(want));
  }
}

TYPED_TEST(KokkosOneShot, OneAxisFunctionsMatchAPlan) {
  using T = TypeParam::real;
  using S = TypeParam::space;
  const S space;
  const std::size_t rows = 5;
  const std::size_t n = 24;
  const complex2<T> x("x", rows, n);
  const complex2<T> y("y", rows, n);
  const complex2<T> w("w", rows, n);
  fill(x, 3);
  expect_as_planned(
      x, y, w, [&](auto& i, auto& o, auto nm) { kfft::fft(space, i, o, nm); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::forward, -1);
        p.execute(i, o, nm);
      },
      "fft");
  expect_as_planned(
      x, y, w, [&](auto& i, auto& o, auto nm) { kfft::ifft(space, i, o, nm); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::backward, -1);
        p.execute(i, o, nm);
      },
      "ifft");
  // Along dimension 0, n given as the Views' length.
  const complex2<T> x0("x0", n, rows);
  const complex2<T> y0("y0", n, rows);
  const complex2<T> w0("w0", n, rows);
  fill(x0, 4);
  expect_as_planned(
      x0, y0, w0, [&](auto& i, auto& o, auto nm) { kfft::fft(space, i, o, nm, 0, n); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::forward, 0);
        p.execute(i, o, nm);
      },
      "fft along 0");
  // Real transforms, even and odd lengths.
  for (const std::size_t m : {std::size_t{24}, std::size_t{25}}) {
    const real2<T> r("r", rows, m);
    const complex2<T> h("h", rows, m / 2 + 1);
    const complex2<T> hw("hw", rows, m / 2 + 1);
    const real2<T> b("b", rows, m);
    const real2<T> bw("bw", rows, m);
    fill(r, 5);
    expect_as_planned(
        r, h, hw, [&](auto& i, auto& o, auto nm) { kfft::rfft(space, i, o, nm); },
        [&](auto& i, auto& o, auto nm) {
          const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::forward, -1);
          p.execute(i, o, nm);
        },
        "rfft " + std::to_string(m));
    fill(h, 6);
    expect_as_planned(
        h, b, bw, [&](auto& i, auto& o, auto nm) { kfft::irfft(space, i, o, nm); },
        [&](auto& i, auto& o, auto nm) {
          const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::backward, -1);
          p.execute(i, o, nm);
        },
        "irfft " + std::to_string(m));
  }
}

// fft of a real View runs on a complex copy, hfft on the conjugated input with the swapped
// normalisation, ihfft as rfft with the swapped normalisation and the output conjugated: what
// Kokkos-FFT computes, through the plan of each transform. hfft leaves its input unchanged.
TYPED_TEST(KokkosOneShot, RealInputHfftAndIhfftAsKokkosFft) {
  using T = TypeParam::real;
  using S = TypeParam::space;
  const S space;
  const std::size_t rows = 3;
  const std::size_t n = 20;
  const real2<T> r("r", rows, n);
  fill(r, 7);
  const complex2<T> rc("rc", rows, n);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      rc(i, j) = Kokkos::complex<T>(r(i, j), 0);
    }
  }
  const complex2<T> y("y", rows, n);
  const complex2<T> w("w", rows, n);
  expect_as_planned(
      r, y, w, [&](auto& i, auto& o, auto nm) { kfft::fft(space, i, o, nm); },
      [&](auto&, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, rc, o, kfft::Direction::forward, -1);
        p.execute(rc, o, nm);
      },
      "fft of reals");
  const real3<T> r3("r3", 2, 6, 8);
  fill(r3, 8);
  const complex3<T> r3c("r3c", 2, 6, 8);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      for (std::size_t k = 0; k < 8; ++k) {
        r3c(i, j, k) = Kokkos::complex<T>(r3(i, j, k), 0);
      }
    }
  }
  const complex3<T> y3("y3", 2, 6, 8);
  const complex3<T> w3("w3", 2, 6, 8);
  expect_as_planned(
      r3, y3, w3, [&](auto& i, auto& o, auto nm) { kfft::fft2(space, i, o, nm); },
      [&](auto&, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, r3c, o, kfft::Direction::forward,
                                        kfft::axis_type<2>{-2, -1});
        p.execute(r3c, o, nm);
      },
      "fft2 of reals");
  expect_as_planned(
      r3, y3, w3, [&](auto& i, auto& o, auto nm) { all_axes<false>(space, i, o, nm); },
      [&](auto&, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, r3c, o, kfft::Direction::forward,
                                        kfft::axis_type<3>{-3, -2, -1});
        p.execute(r3c, o, nm);
      },
      "fftn of reals");

  // hfft: n / 2 + 1 complex values in, n reals out.
  const complex2<T> h("h", rows, n / 2 + 1);
  fill(h, 9);
  const auto before = reals_of(h);
  const complex2<T> hc("hc", rows, n / 2 + 1);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < n / 2 + 1; ++j) {
      hc(i, j) = Kokkos::conj(h(i, j));
    }
  }
  const real2<T> o("o", rows, n);
  const real2<T> ow("ow", rows, n);
  const auto swapped = [](kfft::Normalization nm) {
    if (nm == kfft::Normalization::forward) {
      return kfft::Normalization::backward;
    }
    return nm == kfft::Normalization::backward ? kfft::Normalization::forward : nm;
  };
  expect_as_planned(
      h, o, ow, [&](auto& i, auto& out, auto nm) { kfft::hfft(space, i, out, nm); },
      [&](auto&, auto& out, auto nm) {
        const fftn::kokkos::view_plan p(space, hc, out, kfft::Direction::backward, -1);
        p.execute(hc, out, swapped(nm));
      },
      "hfft");
  expect_same_bits(reals_of(h), before);
  // ihfft: n reals in, n / 2 + 1 complex values out.
  const complex2<T> g("g", rows, n / 2 + 1);
  const complex2<T> gw("gw", rows, n / 2 + 1);
  expect_as_planned(
      r, g, gw, [&](auto& i, auto& out, auto nm) { kfft::ihfft(space, i, out, nm); },
      [&](auto& i, auto& out, auto nm) {
        const fftn::kokkos::view_plan p(space, i, out, kfft::Direction::forward, -1);
        p.execute(i, out, swapped(nm));
        for (std::size_t a = 0; a < rows; ++a) {
          for (std::size_t b = 0; b < n / 2 + 1; ++b) {
            out(a, b) = Kokkos::conj(out(a, b));
          }
        }
      },
      "ihfft");
}

// The 2-D and N-D functions, default and explicit axes, LayoutLeft, in place.
TYPED_TEST(KokkosOneShot, ManyAxisFunctionsMatchAPlan) {
  using T = TypeParam::real;
  using S = TypeParam::space;
  const S space;
  const complex3<T> x("x", 3, 8, 10);
  const complex3<T> y("y", 3, 8, 10);
  const complex3<T> w("w", 3, 8, 10);
  fill(x, 11);
  expect_as_planned(
      x, y, w, [&](auto& i, auto& o, auto nm) { kfft::fft2(space, i, o, nm); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::forward,
                                        kfft::axis_type<2>{-2, -1});
        p.execute(i, o, nm);
      },
      "fft2");
  expect_as_planned(
      x, y, w, [&](auto& i, auto& o, auto nm) { kfft::ifft2(space, i, o, nm, {0, 2}); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::backward,
                                        kfft::axis_type<2>{0, 2});
        p.execute(i, o, nm);
      },
      "ifft2 along 0 and 2");
  expect_as_planned(
      x, y, w, [&](auto& i, auto& o, auto nm) { all_axes<false>(space, i, o, nm); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::forward,
                                        kfft::axis_type<3>{-3, -2, -1});
        p.execute(i, o, nm);
      },
      "fftn");
  expect_as_planned(
      x, y, w,
      [&](auto& i, auto& o, auto nm) {
        kfft::ifftn(space, i, o, kfft::axis_type<2>{-1, 0}, nm, kfft::shape_type<2>{10, 0});
      },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::backward,
                                        kfft::axis_type<2>{-1, 0});
        p.execute(i, o, nm);
      },
      "ifftn along -1 and 0");
  // Real, LayoutLeft: the halved axis is the last listed.
  using left = Kokkos::LayoutLeft;
  const real3<T, left> r("r", 9, 6, 2);
  const complex3<T, left> h("h", 9, 6 / 2 + 1, 2);
  const complex3<T, left> hw("hw", 9, 6 / 2 + 1, 2);
  const real3<T, left> b("b", 9, 6, 2);
  const real3<T, left> bw("bw", 9, 6, 2);
  fill(r, 12);
  expect_as_planned(
      r, h, hw, [&](auto& i, auto& o, auto nm) { kfft::rfft2(space, i, o, nm, {0, 1}); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::forward,
                                        kfft::axis_type<2>{0, 1});
        p.execute(i, o, nm);
      },
      "rfft2 LayoutLeft");
  fill(h, 13);
  expect_as_planned(
      h, b, bw, [&](auto& i, auto& o, auto nm) { kfft::irfft2(space, i, o, nm, {0, 1}); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::backward,
                                        kfft::axis_type<2>{0, 1});
        p.execute(i, o, nm);
      },
      "irfft2 LayoutLeft");
  const real3<T> r3("r3", 4, 6, 7);
  const complex3<T> h3("h3", 4, 6, 7 / 2 + 1);
  const complex3<T> h3w("h3w", 4, 6, 7 / 2 + 1);
  const real3<T> b3("b3", 4, 6, 7);
  const real3<T> b3w("b3w", 4, 6, 7);
  fill(r3, 14);
  expect_as_planned(
      r3, h3, h3w, [&](auto& i, auto& o, auto nm) { all_axes<true>(space, i, o, nm); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::forward,
                                        kfft::axis_type<3>{-3, -2, -1});
        p.execute(i, o, nm);
      },
      "rfftn");
  fill(h3, 15);
  expect_as_planned(
      h3, b3, b3w,
      [&](auto& i, auto& o, auto nm) { kfft::irfftn(space, i, o, kfft::axis_type<2>{1, 2}, nm); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::backward,
                                        kfft::axis_type<2>{1, 2});
        p.execute(i, o, nm);
      },
      "irfftn along 1 and 2");

  // In place: one View as both arguments.
  for (const kfft::Normalization norm : norms) {
    const complex3<T> a("a", 3, 8, 10);
    const complex3<T> c("c", 3, 8, 10);
    fill(a, 16);
    fill(c, 16);
    kfft::fftn(space, a, a, kfft::axis_type<3>{-3, -2, -1}, norm);
    const fftn::kokkos::view_plan p(space, c, c, kfft::Direction::forward,
                                    kfft::axis_type<3>{-3, -2, -1});
    p.execute(c, c, norm);
    SCOPED_TRACE("fftn in place, norm " + name_of(norm));
    expect_same_bits(reals_of(a), reals_of(c));
  }
}

// An output of 2^16 elements or more shares its scaling among the instance's threads, as the
// plan's does.
TYPED_TEST(KokkosOneShot, ALargeOutputScalesAsThePlan) {
  using T = TypeParam::real;
  using S = TypeParam::space;
  const S space;
  const complex2<T> x("x", 96, 1024);
  const complex2<T> y("y", 96, 1024);
  const complex2<T> w("w", 96, 1024);
  fill(x, 17);
  expect_as_planned(
      x, y, w, [&](auto& i, auto& o, auto nm) { kfft::ifft(space, i, o, nm); },
      [&](auto& i, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, i, o, kfft::Direction::backward, -1);
        p.execute(i, o, nm);
      },
      "ifft of 96 x 1024");
}

// A repeated call finds its plan in the calling thread's convenience context; another layout,
// direction or kind adds one. The normalisation is not part of the plan.
TYPED_TEST(KokkosOneShot, RepeatedCallsKeepOnePlanInTheThreadsContext) {
  using T = TypeParam::real;
  using S = TypeParam::space;
  const S space;
  fftn::easy::context& ctx = fftn::easy::thread_context();
  ctx.clear();
  const complex2<T> x("x", 4, 32);
  const complex2<T> y("y", 4, 32);
  fill(x, 18);
  kfft::fft(space, x, y);
  EXPECT_EQ(ctx.cached_plans(), 1U);
  kfft::fft(space, x, y, kfft::Normalization::ortho);
  kfft::fft(space, x, y);
  EXPECT_EQ(ctx.cached_plans(), 1U);
  kfft::ifft(space, x, y);
  kfft::fft(space, x, y, kfft::Normalization::backward, 0);
  EXPECT_EQ(ctx.cached_plans(), 3U);
  ctx.clear();
}

// The described elements of a rank-2 View, as reals in index order (the gaps of a strided View
// left out).
template <class View> auto elements_of(const View& v) {
  using value = View::non_const_value_type;
  using T = real_of<value>::type;
  std::vector<T> r;
  for (std::size_t i = 0; i < v.extent(0); ++i) {
    for (std::size_t j = 0; j < v.extent(1); ++j) {
      if constexpr (std::is_floating_point_v<value>) {
        r.push_back(v(i, j));
      } else {
        r.push_back(v(i, j).real());
        r.push_back(v(i, j).imag());
      }
    }
  }
  return r;
}

// LayoutStride Views (rows inside wider stores): the one-shot functions as the plans of the same
// Views; a real fft input through its C-ordered temporary, as the plan of that copy.
TYPED_TEST(KokkosOneShot, LayoutStrideViewsMatchAPlan) {
  using T = TypeParam::real;
  using S = TypeParam::space;
  using cs = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  using rs = Kokkos::View<T**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const S space;
  const std::size_t rows = 4;
  const std::size_t n = 12;
  const std::size_t h = n / 2 + 1;
  const complex2<T> xs("xs", rows, n + 3);
  const complex2<T> ys("ys", rows, n + 2);
  const complex2<T> ws("ws", rows, n + 2);
  const real2<T> rs_store("rs_store", rows, n + 1);
  const complex2<T> hs("hs", rows, h + 2);
  const complex2<T> hws("hws", rows, h + 2);
  fill(xs, 31);
  fill(rs_store, 32);
  const cs x(xs.data(), Kokkos::LayoutStride(rows, n + 3, n, 1));
  const cs y(ys.data(), Kokkos::LayoutStride(rows, n + 2, n, 1));
  const cs w(ws.data(), Kokkos::LayoutStride(rows, n + 2, n, 1));
  const rs r(rs_store.data(), Kokkos::LayoutStride(rows, n + 1, n, 1));
  const cs hy(hs.data(), Kokkos::LayoutStride(rows, h + 2, h, 1));
  const cs hw(hws.data(), Kokkos::LayoutStride(rows, h + 2, h, 1));
  for (const kfft::Normalization norm : norms) {
    SCOPED_TRACE("norm " + name_of(norm));
    kfft::fft(space, x, y, norm);
    const fftn::kokkos::view_plan p(space, x, w, kfft::Direction::forward, -1);
    p.execute(x, w, norm);
    expect_same_bits(elements_of(y), elements_of(w));
    kfft::rfft(space, r, hy, norm);
    const fftn::kokkos::view_plan q(space, r, hw, kfft::Direction::forward, -1);
    q.execute(r, hw, norm);
    expect_same_bits(elements_of(hy), elements_of(hw));
    // fft of the strided real View: its temporary is C-ordered.
    const complex2<T> rc("rc", rows, n);
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        rc(i, j) = Kokkos::complex<T>(r(i, j), 0);
      }
    }
    kfft::fft(space, r, y, norm);
    const fftn::kokkos::view_plan c(space, rc, w, kfft::Direction::forward, -1);
    c.execute(rc, w, norm);
    expect_same_bits(elements_of(y), elements_of(w));
  }
}

// LayoutLeft real input to fft and complex input to hfft, along dimension 0: the temporaries keep
// LayoutLeft, so the one-shots are the plans of LayoutLeft copies.
TYPED_TEST(KokkosOneShot, LayoutLeftRealFftAndHfft) {
  using T = TypeParam::real;
  using S = TypeParam::space;
  using left = Kokkos::LayoutLeft;
  const S space;
  const std::size_t n = 18;
  const std::size_t cols = 3;
  const real2<T, left> r("r", n, cols);
  fill(r, 41);
  const complex2<T, left> rc("rc", n, cols);
  const complex2<T, left> h("h", n / 2 + 1, cols);
  fill(h, 42);
  const complex2<T, left> hc("hc", n / 2 + 1, cols);
  for (std::size_t j = 0; j < cols; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      rc(i, j) = Kokkos::complex<T>(r(i, j), 0);
    }
    for (std::size_t i = 0; i < n / 2 + 1; ++i) {
      hc(i, j) = Kokkos::conj(h(i, j));
    }
  }
  const complex2<T, left> y("y", n, cols);
  const complex2<T, left> w("w", n, cols);
  expect_as_planned(
      r, y, w, [&](auto& i, auto& o, auto nm) { kfft::fft(space, i, o, nm, 0); },
      [&](auto&, auto& o, auto nm) {
        const fftn::kokkos::view_plan p(space, rc, o, kfft::Direction::forward, 0);
        p.execute(rc, o, nm);
      },
      "fft of LayoutLeft reals");
  const real2<T, left> o("o", n, cols);
  const real2<T, left> ow("ow", n, cols);
  const auto swapped = [](kfft::Normalization nm) {
    if (nm == kfft::Normalization::forward) {
      return kfft::Normalization::backward;
    }
    return nm == kfft::Normalization::backward ? kfft::Normalization::forward : nm;
  };
  expect_as_planned(
      h, o, ow, [&](auto& i, auto& out, auto nm) { kfft::hfft(space, i, out, nm, 0); },
      [&](auto&, auto& out, auto nm) {
        const fftn::kokkos::view_plan p(space, hc, out, kfft::Direction::backward, 0);
        p.execute(hc, out, swapped(nm));
      },
      "hfft of LayoutLeft");
}

// Host threads calling one-shot functions at once, on Serial: each finds its plan in its own
// context, and every thread's results are the single thread's, bit for bit.
TEST(KokkosOneShotThreads, EachHostThreadHasItsOwnContext) {
  using T = double;
  const Kokkos::Serial space;
  const std::size_t rows = 6;
  const std::size_t n = 40;
  const int threads = 4;
  std::vector<complex2<T>> in;
  std::vector<complex2<T>> out;
  std::vector<real2<T>> real_in;
  std::vector<complex2<T>> half;
  for (int t = 0; t < threads; ++t) {
    in.emplace_back("in", rows, n);
    out.emplace_back("out", rows, n);
    real_in.emplace_back("real_in", rows, n);
    half.emplace_back("half", rows, n / 2 + 1);
    fill(in.back(), 50);
    fill(real_in.back(), 51);
  }
  const complex2<T> want("want", rows, n);
  const complex2<T> want_half("want_half", rows, n / 2 + 1);
  kfft::ifft(space, in[0], want, kfft::Normalization::ortho);
  kfft::rfft(space, real_in[0], want_half);
  std::vector<std::size_t> cached(threads, 0);
  std::latch start(threads);
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      const auto i = static_cast<std::size_t>(t);
      start.arrive_and_wait();
      for (int round = 0; round < 20; ++round) {
        kfft::ifft(space, in[i], out[i], kfft::Normalization::ortho);
        kfft::rfft(space, real_in[i], half[i]);
      }
      cached[i] = fftn::easy::thread_context().cached_plans();
    });
  }
  for (std::thread& th : pool) {
    th.join();
  }
  for (std::size_t t = 0; t < static_cast<std::size_t>(threads); ++t) {
    EXPECT_EQ(cached[t], 2U) << "thread " << t;
    expect_same_bits(reals_of(out[t]), reals_of(want));
    expect_same_bits(reals_of(half[t]), reals_of(want_half));
  }
}

// Refusals are view_plan's, naming the argument.
TYPED_TEST(KokkosOneShot, RefusalsAreThePlans) {
  using T = TypeParam::real;
  using S = TypeParam::space;
  const S space;
  const complex2<T> x("x", 4, 32);
  const complex2<T> y("y", 4, 32);
  const complex2<T> z("z", 4, 31);
  expect_refused([&] { kfft::fft(space, x, y, kfft::Normalization::backward, -1, 30); }, "n");
  expect_refused([&] { kfft::fft(space, x, y, kfft::Normalization::backward, 2); }, "axes[0]");
  expect_refused([&] { kfft::ifft(space, x, z); }, "out.extent(1)");
  expect_refused(
      [&] {
        kfft::fft2(space, x, y, kfft::Normalization::backward, {-2, -1},
                   kfft::shape_type<2>{4, 16});
      },
      "s[1]");
  expect_refused([&] { kfft::fft(space, x, y, static_cast<kfft::Normalization>(9)); }, "norm");
}

} // namespace
