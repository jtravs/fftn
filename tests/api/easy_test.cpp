// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The convenience layer (fftn/easy.hpp, step 6.6) against numpy's conventions: results
// against the quadruple-precision reference evaluated as numpy defines each function (norms,
// axes, the halved dimension of the real kinds, irfftn's order), recorded numpy results on
// small cases, bitwise equality with the core plan of the same shape, refusals that name the
// argument, the cache (a hit allocates nothing, the bound, least recently used eviction,
// clear, moves), each thread's own context, an executor, and the helpers (fftfreq,
// rfftfreq, fftshift, ifftshift).

#include "reference/reference_dft.hpp"
#include "support/alloc_hook.hpp"
#include "support/exact.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <fftn/easy.hpp>
#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <latch>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace easy = fftn::easy;
using cd = std::complex<double>;
using dims_t = std::vector<std::size_t>;
using fftn::test::exact_complex;
using fftn::test::exact_real;

constexpr std::array norms{easy::norm::backward, easy::norm::ortho, easy::norm::forward};

const char* name_of(easy::norm nm) {
  switch (nm) {
  case easy::norm::backward:
    return "backward";
  case easy::norm::ortho:
    return "ortho";
  case easy::norm::forward:
    break;
  }
  return "forward";
}

std::size_t product(const dims_t& d) {
  std::size_t p = 1;
  for (const std::size_t e : d) {
    p *= e;
  }
  return p;
}

// Deterministic values in [-1, 1), exact in float and in double.
class values {
public:
  explicit values(std::uint32_t seed) : state_(seed) {}
  template <class T> T real() {
    state_ = state_ * 1664525U + 1013904223U;
    return static_cast<T>(static_cast<double>(state_ >> 8U) / 8388608.0 - 1.0);
  }
  template <class T> std::vector<T> reals(std::size_t n) {
    std::vector<T> v(n);
    for (T& x : v) {
      x = real<T>();
    }
    return v;
  }
  template <class T> std::vector<std::complex<T>> complexes(std::size_t n) {
    std::vector<std::complex<T>> v(n);
    for (std::complex<T>& x : v) {
      const T re = real<T>();
      x = {re, real<T>()};
    }
    return v;
  }

private:
  std::uint32_t state_;
};

template <class T> std::vector<cd> widen(const std::vector<std::complex<T>>& v) {
  return {v.begin(), v.end()};
}
template <class T>
std::vector<double> widen(const std::vector<T>& v)
  requires std::is_floating_point_v<T>
{
  return {v.begin(), v.end()};
}

// The factor numpy applies: 1, 1/sqrt(N) or 1/N, as the norm and the direction say.
double factor(easy::norm nm, bool forward, std::size_t points) {
  const auto n = static_cast<double>(points);
  switch (nm) {
  case easy::norm::backward:
    return forward ? 1.0 : 1.0 / n;
  case easy::norm::ortho:
    return 1.0 / std::sqrt(n);
  case easy::norm::forward:
    break;
  }
  return forward ? 1.0 / n : 1.0;
}

// f applied to every line along dimension a of a tight C-ordered array with extents `ext`; the
// lines of the result have m elements.
template <class A, class B, class F>
std::vector<B> along(const dims_t& ext, std::size_t a, const std::vector<A>& x, std::size_t m,
                     F f) {
  std::size_t inner = 1;
  for (std::size_t d = a + 1; d < ext.size(); ++d) {
    inner *= ext[d];
  }
  const std::size_t n = ext[a];
  const std::size_t outer = x.size() / (n * inner);
  std::vector<B> y(outer * m * inner);
  std::vector<A> line(n);
  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t i = 0; i < inner; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        line[j] = x[(o * n + j) * inner + i];
      }
      const std::vector<B> r = f(line);
      for (std::size_t k = 0; k < m; ++k) {
        y[(o * m + k) * inner + i] = r[k];
      }
    }
  }
  return y;
}

std::vector<cd> ref_line(int sign, const std::vector<cd>& line) {
  const std::vector<exact_complex> e = fftn::test::reference::c2c(sign, line);
  std::vector<cd> r;
  r.reserve(e.size());
  for (const exact_complex& z : e) {
    r.emplace_back(z.re.hi, z.im.hi);
  }
  return r;
}

// numpy's fftn and ifftn (unscaled): the complex transform along each listed dimension,
// rounded to double after each.
std::vector<cd> ref_c2c(int sign, const dims_t& ext, const std::vector<std::size_t>& axes,
                        std::vector<cd> x) {
  for (const std::size_t a : axes) {
    x = along<cd, cd>(ext, a, x, ext[a],
                      [&](const std::vector<cd>& line) { return ref_line(sign, line); });
  }
  return x;
}

// numpy's rfftn (unscaled): rfft along the last listed dimension, then fft along the others.
std::vector<cd> ref_r2c(const dims_t& ext, const std::vector<std::size_t>& axes,
                        const std::vector<double>& x) {
  const std::size_t last = axes.back();
  const std::size_t m = ext[last] / 2 + 1;
  std::vector<cd> y = along<double, cd>(ext, last, x, m, [](const std::vector<double>& line) {
    const std::vector<exact_complex> e = fftn::test::reference::r2c(line);
    std::vector<cd> r;
    for (const exact_complex& z : e) {
      r.emplace_back(z.re.hi, z.im.hi);
    }
    return r;
  });
  dims_t half = ext;
  half[last] = m;
  return ref_c2c(-1, half, {axes.begin(), axes.end() - 1}, y);
}

// numpy's irfftn (unscaled), `ext` the real array's extents: ifft along the other listed
// dimensions first, then irfft along the last listed one.
std::vector<double> ref_c2r(const dims_t& ext, const std::vector<std::size_t>& axes,
                            const std::vector<cd>& x) {
  const std::size_t last = axes.back();
  const std::size_t n = ext[last];
  dims_t half = ext;
  half[last] = n / 2 + 1;
  const std::vector<cd> y = ref_c2c(+1, half, {axes.begin(), axes.end() - 1}, x);
  return along<cd, double>(half, last, y, n, [n](const std::vector<cd>& line) {
    const std::vector<exact_real> e = fftn::test::reference::c2r(n, line);
    std::vector<double> r;
    for (const exact_real& v : e) {
      r.push_back(v.hi);
    }
    return r;
  });
}

// The tolerance of a transform of `points` points along `count` dimensions against the
// references above (tolerance.hpp, as in multidim_test.cpp): the transform's own bound, a
// rounding to double per dimension of the reference and its scaling, and the scale factor
// rounded to T and multiplied in.
template <class T> fftn::test::tolerance tolerance_for(std::size_t points, std::size_t count) {
  return fftn::test::derived_tolerance<T>(points, fftn::test::fftn_bound(points)) +
         fftn::test::input_rounding<double>() * static_cast<double>(count + 1) +
         fftn::test::input_rounding<T>() * 2.0;
}

template <class T>
void expect_close(std::span<const std::complex<T>> got, const std::vector<cd>& want, double scale,
                  const fftn::test::tolerance& tol, const std::string& what) {
  std::vector<exact_complex> e;
  e.reserve(want.size());
  for (const cd& z : want) {
    e.push_back(fftn::test::exact_from(z * scale));
  }
  fftn::test::outcome o;
  o.within(fftn::test::compare<T>(got, e), tol, what);
  EXPECT_TRUE(o.passed()) << o.detail();
}

template <class T>
void expect_close(std::span<const T> got, const std::vector<double>& want, double scale,
                  const fftn::test::tolerance& tol, const std::string& what)
  requires std::is_floating_point_v<T>
{
  std::vector<exact_real> e;
  e.reserve(want.size());
  for (const double v : want) {
    e.push_back(fftn::test::exact_from(v * scale));
  }
  fftn::test::outcome o;
  o.within(fftn::test::compare<T>(got, e), tol, what);
  EXPECT_TRUE(o.passed()) << o.detail();
}

template <class V> bool same_bits(const V& a, const V& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(typename V::value_type)) == 0;
}

// The core plan's result for a tight array, with the scale the norm gives: what the layer
// must equal bit for bit.
template <class In, class Out>
std::vector<Out> core_result(const fftn::shape<In, Out>& s, double scale, const std::vector<In>& in,
                             std::size_t out_size) {
  const fftn::plan<In, Out> p(s, {.scale = scale});
  fftn::workspace ws(p);
  std::vector<Out> out(out_size);
  p.execute(in.data(), out.data(), ws);
  return out;
}

std::vector<fftn::axis> array_of(const dims_t& ext) {
  std::vector<fftn::axis> a;
  for (const std::size_t e : ext) {
    a.push_back(fftn::axis{.n = e});
  }
  return a;
}

template <class T> class easy_test : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(easy_test, precisions);

const std::vector<std::size_t> lengths_1d = {1,  2,  3,  4,  5,  6,  7,   8,   9,   12,  15,
                                             16, 17, 31, 32, 64, 97, 100, 128, 243, 256, 1000};

// fft, ifft, rfft and irfft (odd and even lengths) against the reference for each norm, the
// value forms bitwise equal to the span forms, and both bitwise equal to the core plan.
TYPED_TEST(easy_test, one_dimensional_against_the_reference) {
  using T = TypeParam;
  using ct = std::complex<T>;
  values gen(7);
  for (const std::size_t n : lengths_1d) {
    const std::vector<ct> x = gen.complexes<T>(n);
    const std::vector<T> r = gen.reals<T>(n);
    const std::vector<ct> half = gen.complexes<T>(n / 2 + 1);
    const std::vector<cd> fwd = ref_line(-1, widen(x));
    const std::vector<cd> bwd = ref_line(+1, widen(x));
    const std::vector<cd> rf = ref_r2c({n}, {0}, widen(r));
    const std::vector<double> ri = ref_c2r({n}, {0}, widen(half));
    const auto tol = tolerance_for<T>(n, 1);
    for (const easy::norm nm : norms) {
      const std::string at = "n " + std::to_string(n) + ", norm " + name_of(nm);
      std::vector<ct> y(n);
      easy::fft(x, y, nm);
      expect_close<T>(y, fwd, factor(nm, true, n), tol, "fft, " + at);
      EXPECT_TRUE(same_bits(y, easy::fft(std::span<const ct>(x), nm))) << at;
      EXPECT_TRUE(same_bits(y, core_result(fftn::rows_c2c<T>(n, 1, fftn::direction::forward),
                                           factor(nm, true, n), x, n)))
          << at;

      easy::ifft(x, y, nm);
      expect_close<T>(y, bwd, factor(nm, false, n), tol, "ifft, " + at);
      EXPECT_TRUE(same_bits(y, easy::ifft(std::span<const ct>(x), nm))) << at;
      EXPECT_TRUE(same_bits(y, core_result(fftn::rows_c2c<T>(n, 1, fftn::direction::backward),
                                           factor(nm, false, n), x, n)))
          << at;

      std::vector<ct> h(n / 2 + 1);
      easy::rfft(r, h, nm);
      expect_close<T>(h, rf, factor(nm, true, n), tol, "rfft, " + at);
      EXPECT_TRUE(same_bits(h, easy::rfft(std::span<const T>(r), nm))) << at;
      EXPECT_TRUE(
          same_bits(h, core_result(fftn::rows_r2c<T>(n, 1), factor(nm, true, n), r, n / 2 + 1)))
          << at;

      std::vector<T> back(n);
      easy::irfft(half, back, nm);
      expect_close<T>(back, ri, factor(nm, false, n), tol, "irfft, " + at);
      EXPECT_TRUE(same_bits(back, easy::irfft(std::span<const ct>(half), n, nm))) << at;
      EXPECT_TRUE(
          same_bits(back, core_result(fftn::rows_c2r<T>(n, 1), factor(nm, false, n), half, n)))
          << at;
    }
  }
}

// Results numpy gives on small cases, recorded exactly.
TYPED_TEST(easy_test, recorded_numpy_results) {
  using T = TypeParam;
  using ct = std::complex<T>;
  // np.fft.fft([1, 2, 3, 4]) and its inverse.
  const std::vector<ct> x{1, 2, 3, 4};
  const std::vector<ct> X{{10, 0}, {-2, 2}, {-2, 0}, {-2, -2}};
  EXPECT_EQ(easy::fft(std::span<const ct>(x)), X);
  EXPECT_EQ(easy::ifft(std::span<const ct>(X)), x);
  // np.fft.rfft([1, 2, 3, 4]) and np.fft.irfft of it (n = 4 by default).
  const std::vector<T> r{1, 2, 3, 4};
  const std::vector<ct> R{{10, 0}, {-2, 2}, {-2, 0}};
  EXPECT_EQ(easy::rfft(std::span<const T>(r)), R);
  EXPECT_EQ(easy::irfft(std::span<const ct>(R)), r);
  // np.fft.fft(np.ones(4), norm="ortho") and norm="forward".
  const std::vector<ct> ones(4, ct{1});
  EXPECT_EQ(easy::fft(std::span<const ct>(ones), easy::norm::ortho),
            (std::vector<ct>{{2, 0}, {0, 0}, {0, 0}, {0, 0}}));
  EXPECT_EQ(easy::fft(std::span<const ct>(ones), easy::norm::forward),
            (std::vector<ct>{{1, 0}, {0, 0}, {0, 0}, {0, 0}}));
  // np.fft.irfft([5, 0, 0], n=5): five points of 1; n=4 would read only two values.
  const std::vector<ct> dc{{5, 0}, {0, 0}, {0, 0}};
  EXPECT_EQ(easy::irfft(std::span<const ct>(dc), 5), std::vector<T>(5, T{1}));
  EXPECT_EQ(easy::irfft(std::span<const ct>(dc)), std::vector<T>(4, T{1.25}));
  // np.fft.irfft ignores the imaginary parts of bin 0 and, for even n, of bin n/2.
  const std::vector<ct> junk{{4, 3}, {0, 0}, {0, 7}};
  EXPECT_EQ(easy::irfft(std::span<const ct>(junk)), std::vector<T>(4, T{1}));
  // np.fft.fft2([[1, 2], [3, 4]]) = [[10, -2], [-4, 0]].
  const std::vector<ct> a{1, 2, 3, 4};
  std::vector<ct> a2(4);
  easy::fft2(easy::view<const ct, 2>(a.data(), {2, 2}), easy::view<ct, 2>(a2.data(), {2, 2}));
  EXPECT_EQ(a2, (std::vector<ct>{10, -2, -4, 0}));
}

// A forward transform and its inverse return the input under every norm.
TYPED_TEST(easy_test, round_trips) {
  using T = TypeParam;
  using ct = std::complex<T>;
  values gen(11);
  for (const std::size_t n : std::vector<std::size_t>{1, 2, 7, 16, 45, 101, 512}) {
    const std::vector<ct> x = gen.complexes<T>(n);
    const std::vector<T> r = gen.reals<T>(n);
    const auto tol = tolerance_for<T>(n, 1) * 2.0;
    for (const easy::norm nm : norms) {
      const std::string at = "n " + std::to_string(n) + ", norm " + name_of(nm);
      const std::vector<ct> y = easy::fft(std::span<const ct>(x), nm);
      expect_close<T>(easy::ifft(std::span<const ct>(y), nm), widen(x), 1.0, tol,
                      "ifft(fft), " + at);
      const std::vector<ct> h = easy::rfft(std::span<const T>(r), nm);
      expect_close<T>(easy::irfft(std::span<const ct>(h), n, nm), widen(r), 1.0, tol,
                      "irfft(rfft), " + at);
    }
  }
}

struct nd_case {
  dims_t ext;
  std::vector<int> axes;           // as passed (empty: the default)
  std::vector<std::size_t> chosen; // the dimensions numpy transforms, in order
};

// fftn, ifftn, rfftn and irfftn (and the 2-D forms where the default is the last two) over
// arrays of rank 1 to 4 and numpy's axes (defaults, negative, reordered, subsets), against
// the reference evaluated numpy's way, and bitwise equal to the core plan of axes_* shapes.
TYPED_TEST(easy_test, multi_dimensional_against_the_reference) {
  using T = TypeParam;
  using ct = std::complex<T>;
  const std::vector<nd_case> cases = {
      {{12}, {}, {0}},
      {{6, 10}, {}, {0, 1}},
      {{6, 10}, {0}, {0}},
      {{6, 10}, {-2}, {0}},
      {{6, 10}, {1, 0}, {1, 0}},
      {{5, 4, 6}, {}, {0, 1, 2}},
      {{5, 4, 6}, {-1, 0}, {2, 0}},
      {{5, 4, 6}, {0, 2}, {0, 2}},
      {{5, 4, 6}, {2, 1, 0}, {2, 1, 0}},
      {{5, 4, 6}, {1}, {1}},
      {{3, 4, 5, 6}, {1, 3}, {1, 3}},
      {{3, 4, 5, 6}, {-3, -2, -1}, {1, 2, 3}},
      {{2, 3, 7, 1}, {0, 2}, {0, 2}},
  };
  values gen(23);
  for (const nd_case& c : cases) {
    std::size_t points = 1;
    for (const std::size_t a : c.chosen) {
      points *= c.ext[a];
    }
    const std::size_t last = c.chosen.back();
    dims_t half = c.ext;
    half[last] = c.ext[last] / 2 + 1;
    const std::vector<ct> x = gen.complexes<T>(product(c.ext));
    const std::vector<T> r = gen.reals<T>(product(c.ext));
    const std::vector<ct> h = gen.complexes<T>(product(half));
    const auto tol = tolerance_for<T>(points, c.chosen.size());
    const std::vector<cd> fwd = ref_c2c(-1, c.ext, c.chosen, widen(x));
    const std::vector<cd> bwd = ref_c2c(+1, c.ext, c.chosen, widen(x));
    const std::vector<cd> rf = ref_r2c(c.ext, c.chosen, widen(r));
    const std::vector<double> ri = ref_c2r(c.ext, c.chosen, widen(h));
    std::string at = "extents";
    for (const std::size_t e : c.ext) {
      at += " " + std::to_string(e);
    }
    at += ", axes";
    for (const int a : c.axes) {
      at += " " + std::to_string(a);
    }
    const auto run = [&]<std::size_t R>() {
      std::array<std::size_t, R> e{};
      std::array<std::size_t, R> eh{};
      for (std::size_t d = 0; d < R; ++d) {
        e[d] = c.ext[d];
        eh[d] = half[d];
      }
      const easy::axis_list axes(c.axes);
      const std::vector<fftn::axis> array = array_of(c.ext);
      for (const easy::norm nm : norms) {
        const std::string where = at + ", norm " + name_of(nm);
        std::vector<ct> y(x.size());
        easy::fftn(easy::view<const ct, R>(x.data(), e), easy::view<ct, R>(y.data(), e), axes, nm);
        expect_close<T>(y, fwd, factor(nm, true, points), tol, "fftn, " + where);
        EXPECT_TRUE(
            same_bits(y, core_result(fftn::axes_c2c<T>(array, c.chosen, fftn::direction::forward),
                                     factor(nm, true, points), x, x.size())))
            << where;
        easy::ifftn(easy::view<const ct, R>(x.data(), e), easy::view<ct, R>(y.data(), e), axes, nm);
        expect_close<T>(y, bwd, factor(nm, false, points), tol, "ifftn, " + where);
        std::vector<ct> yh(h.size());
        easy::rfftn(easy::view<const T, R>(r.data(), e), easy::view<ct, R>(yh.data(), eh), axes,
                    nm);
        expect_close<T>(yh, rf, factor(nm, true, points), tol, "rfftn, " + where);
        EXPECT_TRUE(same_bits(yh, core_result(fftn::axes_r2c<T>(array, c.chosen),
                                              factor(nm, true, points), r, h.size())))
            << where;
        std::vector<T> yr(r.size());
        easy::irfftn(easy::view<const ct, R>(h.data(), eh), easy::view<T, R>(yr.data(), e), axes,
                     nm);
        expect_close<T>(yr, ri, factor(nm, false, points), tol, "irfftn, " + where);
        EXPECT_TRUE(same_bits(yr, core_result(fftn::axes_c2r<T>(array, c.chosen),
                                              factor(nm, false, points), h, r.size())))
            << where;
        // The 2-D forms differ from the N-D ones only in their default axes.
        if (!c.axes.empty() || R == 2) {
          std::vector<ct> y2(x.size());
          easy::fft2(easy::view<const ct, R>(x.data(), e), easy::view<ct, R>(y2.data(), e), axes,
                     nm);
          easy::fftn(easy::view<const ct, R>(x.data(), e), easy::view<ct, R>(y.data(), e), axes,
                     nm);
          EXPECT_TRUE(same_bits(y, y2)) << "fft2, " << where;
          easy::ifft2(easy::view<const ct, R>(x.data(), e), easy::view<ct, R>(y2.data(), e), axes,
                      nm);
          easy::ifftn(easy::view<const ct, R>(x.data(), e), easy::view<ct, R>(y.data(), e), axes,
                      nm);
          EXPECT_TRUE(same_bits(y, y2)) << "ifft2, " << where;
          std::vector<ct> yh2(h.size());
          easy::rfft2(easy::view<const T, R>(r.data(), e), easy::view<ct, R>(yh2.data(), eh), axes,
                      nm);
          EXPECT_TRUE(same_bits(yh, yh2)) << "rfft2, " << where;
          std::vector<T> yr2(r.size());
          easy::irfft2(easy::view<const ct, R>(h.data(), eh), easy::view<T, R>(yr2.data(), e), axes,
                       nm);
          EXPECT_TRUE(same_bits(yr, yr2)) << "irfft2, " << where;
        }
      }
    };
    switch (c.ext.size()) {
    case 1:
      run.template operator()<1>();
      break;
    case 2:
      run.template operator()<2>();
      break;
    case 3:
      run.template operator()<3>();
      break;
    default:
      run.template operator()<4>();
      break;
    }
  }
}

// fft2's default is the last two dimensions of a volume: a batch of planes.
TYPED_TEST(easy_test, fft2_defaults_to_the_last_two_dimensions) {
  using T = TypeParam;
  using ct = std::complex<T>;
  values gen(5);
  const dims_t ext{3, 4, 6};
  const std::vector<ct> x = gen.complexes<T>(product(ext));
  std::vector<ct> y(x.size());
  easy::fft2(easy::view<const ct, 3>(x.data(), {3, 4, 6}), easy::view<ct, 3>(y.data(), {3, 4, 6}));
  expect_close<T>(y, ref_c2c(-1, ext, {1, 2}, widen(x)), 1.0, tolerance_for<T>(24, 2), "fft2");
  std::vector<T> r = gen.reals<T>(product(ext));
  std::vector<ct> h(3 * 4 * 4);
  easy::rfft2(easy::view<const T, 3>(r.data(), {3, 4, 6}), easy::view<ct, 3>(h.data(), {3, 4, 4}));
  expect_close<T>(h, ref_r2c(ext, {1, 2}, widen(r)), 1.0, tolerance_for<T>(24, 2), "rfft2");
}

// Views with negative strides, gaps and a transposed output; complex transforms in place.
TYPED_TEST(easy_test, strided_views_and_in_place) {
  using T = TypeParam;
  using ct = std::complex<T>;
  values gen(31);
  const dims_t ext{6, 9};
  const std::vector<ct> x = gen.complexes<T>(54);
  const std::vector<std::size_t> both{0, 1};
  const auto tol = tolerance_for<T>(54, 2);
  const std::vector<cd> want = ref_c2c(-1, ext, both, widen(x));
  // The input with rows 12 apart, reversed along the rows: element (i, j) at
  // base + (5 - i) * 12 + j, so the view starts at the last row with stride -12.
  std::vector<ct> in(6 * 12, ct{99, 99});
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 9; ++j) {
      in[(5 - i) * 12 + j] = x[i * 9 + j];
    }
  }
  // The output transposed: element (i, j) at j * 6 + i.
  std::vector<ct> out(54);
  easy::fftn(easy::view<const ct, 2>(in.data() + 5 * 12, {6, 9}, {-12, 1}),
             easy::view<ct, 2>(out.data(), {6, 9}, {1, 6}));
  std::vector<ct> got(54);
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 9; ++j) {
      got[i * 9 + j] = out[j * 6 + i];
    }
  }
  expect_close<T>(got, want, 1.0, tol, "strided fftn");
  // In place, tight and strided.
  std::vector<ct> y = x;
  easy::fftn(easy::view<const ct, 2>(y.data(), {6, 9}), easy::view<ct, 2>(y.data(), {6, 9}));
  expect_close<T>(y, want, 1.0, tol, "fftn in place");
  std::vector<ct> v = x;
  easy::fft(v, v);
  expect_close<T>(v, ref_c2c(-1, {54}, {0}, widen(x)), 1.0, tolerance_for<T>(54, 1),
                  "fft in place");
  std::vector<ct> padded = in;
  easy::ifftn(easy::view<const ct, 2>(padded.data() + 5 * 12, {6, 9}, {-12, 1}),
              easy::view<ct, 2>(padded.data() + 5 * 12, {6, 9}, {-12, 1}), {}, easy::norm::ortho);
  std::vector<ct> got2(54);
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 9; ++j) {
      got2[i * 9 + j] = padded[(5 - i) * 12 + j];
    }
    for (std::size_t j = 9; j < 12; ++j) {
      EXPECT_EQ(padded[(5 - i) * 12 + j], (ct{99, 99})) << "a gap was written";
    }
  }
  expect_close<T>(got2, ref_c2c(+1, ext, both, widen(x)), 1.0 / std::sqrt(54.0), tol,
                  "strided ifftn in place");
  // A view of T converts to a view of const T; a strided real view.
  std::vector<T> r = gen.reals<T>(54);
  std::vector<T> rs(108, T{7});
  for (std::size_t k = 0; k < 54; ++k) {
    rs[2 * k] = r[k];
  }
  std::vector<ct> h(6 * 5);
  const easy::view<T, 2> mutable_in(rs.data(), {6, 9}, {18, 2});
  easy::rfftn(mutable_in, easy::view<ct, 2>(h.data(), {6, 5}));
  expect_close<T>(h, ref_r2c(ext, both, widen(r)), 1.0, tol, "strided rfftn");
}

// Arrays with nothing to loop over: the call does nothing.
TEST(easy, an_empty_loop_does_nothing) {
  std::vector<cd> x(1, cd{5});
  std::vector<cd> y(1, cd{6});
  easy::fftn(easy::view<const cd, 2>(x.data(), {0, 8}), easy::view<cd, 2>(y.data(), {0, 8}), {1});
  EXPECT_EQ(y[0], cd{6});
  // The strides std::layout_right gives extents {3, 0}: {0, 1}, a 0 in front of the empty
  // dimension. The array is empty, so they are not refused.
  easy::fftn(easy::view<const cd, 2>(x.data(), {3, 0}, {0, 1}),
             easy::view<cd, 2>(y.data(), {3, 0}, {0, 1}), {0});
  easy::fftshift(easy::view<const cd, 2>(x.data(), {3, 0}, {0, 1}),
                 easy::view<cd, 2>(y.data(), {3, 0}, {0, 1}));
  EXPECT_EQ(y[0], cd{6});
}

// Every refusal is std::invalid_argument naming the argument.
TEST(easy, refusals_name_the_argument) {
  const auto refuses = [](const std::function<void()>& f, const std::string& needle) {
    try {
      f();
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
          << "message \"" << e.what() << "\" lacks \"" << needle << "\"";
      return;
    }
    ADD_FAILURE() << "no std::invalid_argument; expected one naming " << needle;
  };
  std::vector<cd> c8(8);
  std::vector<cd> c7(7);
  std::vector<cd> c5(5);
  std::vector<double> r8(8);
  std::vector<double> r7(7);
  refuses([&] { easy::fft(c8, c7); }, "out.size()");
  refuses([&] { easy::fft(std::span<const cd>(), std::span<cd>()); }, "in.size() is 0");
  refuses([&] { easy::rfft(r8, c8); }, "out.size() is 8; it must be in.size()/2 + 1, 5");
  refuses([&] { easy::irfft(c8, r8); }, "in.size() is 8; it must be out.size()/2 + 1, 5");
  refuses([&] { easy::irfft(c5, std::span<double>()); }, "out.size() is 0");
  std::vector<cd> shared(5);
  refuses(
      [&] {
        // The real side viewed over the complex side's memory, as numpy cannot express.
        easy::rfft(
            std::span<const double>(reinterpret_cast<const double*>(shared.data()),
                                    8), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): one
                                        // memory seen as both types, as in-place real transforms do
            shared);
      },
      "no in-place form");
  refuses([&] { static_cast<void>(easy::irfft(std::span<const cd>(c8.data(), 1))); }, "give n");
  std::vector<cd> a(24);
  std::vector<cd> b(24);
  const easy::view<const cd, 2> in(a.data(), {4, 6});
  const easy::view<cd, 2> out(b.data(), {4, 6});
  refuses([&] { easy::fftn(in, out, {2}); }, "axes lists 2, outside arrays of 2 dimensions");
  refuses([&] { easy::fftn(in, out, {-3}); }, "axes lists -3");
  refuses([&] { easy::fftn(in, out, {1, -1}); }, "axes lists dimension 1 twice");
  refuses([&] { easy::fftn(in, out, {0, 1, 0, 1}); }, "axes lists 4 dimensions");
  refuses([&] { easy::fftn(in, out, {0, 1, 0, 1, 0, 1, 0, 1, 0}); }, "axes lists 9");
  refuses([&] { easy::fftn(in, easy::view<cd, 2>(b.data(), {6, 4})); },
          "out.extents[0] is 6; it must be in.extents[0], 4");
  refuses(
      [&] {
        easy::fft2(easy::view<const cd, 1>(a.data(), {24}), easy::view<cd, 1>(b.data(), {24}));
      },
      "transforms the last 2 by default");
  std::vector<cd> big(4 * 2 * 3 * 2);
  refuses(
      [&] {
        easy::fftn(easy::view<const cd, 4>(big.data(), {4, 2, 3, 2}),
                   easy::view<cd, 4>(b.data(), {4, 2, 3, 2}));
      },
      "pass axes");
  refuses([&] { easy::fftn(easy::view<const cd, 2>(a.data(), {4, 6}, {0, 1}), out); },
          "in.strides[0] is 0");
  refuses([&] { easy::fftn(in, easy::view<cd, 2>(a.data(), {4, 6}, {1, 4})); },
          "in place needs the same strides");
  refuses([&] { easy::fftn(in, easy::view<cd, 2>(b.data(), {4, 6}, {1, 1})); },
          "not one a plan takes");
  refuses(
      [&] {
        easy::fftn(easy::view<const cd, 2>(a.data(), {0, 6}), easy::view<cd, 2>(b.data(), {0, 6}));
      },
      "in.extents[0] is 0");
  std::vector<double> ra(24);
  std::vector<cd> hb(4 * 4);
  refuses(
      [&] {
        easy::rfftn(easy::view<const double, 2>(ra.data(), {4, 6}),
                    easy::view<cd, 2>(hb.data(), {4, 6}));
      },
      "out.extents[1] is 6; it must be in.extents[1]/2 + 1, 4");
  refuses(
      [&] {
        easy::irfftn(easy::view<const cd, 2>(hb.data(), {4, 6}),
                     easy::view<double, 2>(ra.data(), {4, 6}));
      },
      "in.extents[1] is 6");
  std::vector<double> f(4);
  refuses([&] { easy::fftfreq(5, f); }, "out has 4 elements; it needs 5");
  refuses([&] { easy::rfftfreq(0, f); }, "n is 0");
  refuses(
      [&] {
        easy::fftshift(easy::view<const cd, 2>(a.data(), {4, 6}),
                       easy::view<cd, 2>(b.data(), {6, 4}));
      },
      "out.extents");
  refuses([&] { easy::fftshift(in, out, {5}); }, "axes lists 5");
  refuses([&] { easy::fftshift(in, easy::view<cd, 2>(b.data(), {4, 6}, {0, 1})); },
          "out.strides[0] is 0");
  refuses([&] { easy::ifftshift(easy::view<const cd, 2>(a.data(), {4, 6}, {6, 0}), out); },
          "in.strides[1] is 0");
  easy::context ctx;
  refuses([&] { ctx.set_max_plans(0); }, "max_plans");
}

// A call that finds its plan allocates nothing, through a context and through the free
// functions; a call of another layout, norm, direction, kind or precision builds another plan.
TEST(easy, a_cache_hit_allocates_nothing) {
  values gen(3);
  const std::vector<cd> x = gen.complexes<double>(384);
  std::vector<cd> y(384);
  std::vector<cd> z(384);
  easy::context ctx;
  EXPECT_EQ(ctx.cached_plans(), 0U);
  EXPECT_EQ(ctx.max_plans(), easy::context::default_max_plans);
  ctx.fft(x, y);
  EXPECT_EQ(ctx.cached_plans(), 1U);
  {
    const fftn::test::allocation_counter count;
    ctx.fft(x, y);
    ctx.fft(x, z); // other data, the same layout
    EXPECT_EQ(count.allocations(), 0U);
  }
  ctx.fft(x, y, easy::norm::ortho);
  ctx.ifft(x, y);
  ctx.fft(std::span<const cd>(x.data(), 100), std::span<cd>(y.data(), 100));
  std::vector<cd> w = x;
  ctx.fft(w, w); // in place is another plan
  ctx.fftn(easy::view<const cd, 2>(x.data(), {16, 24}), easy::view<cd, 2>(y.data(), {16, 24}));
  ctx.fftn(easy::view<const cd, 2>(x.data(), {16, 24}), easy::view<cd, 2>(y.data(), {16, 24}), {0});
  const std::vector<std::complex<float>> xf(384);
  std::vector<std::complex<float>> yf(384);
  ctx.fft(xf, yf);
  EXPECT_EQ(ctx.cached_plans(), 8U);
  {
    const fftn::test::allocation_counter count;
    ctx.fft(x, y, easy::norm::ortho);
    ctx.ifft(x, y);
    ctx.fft(w, w);
    ctx.fftn(easy::view<const cd, 2>(x.data(), {16, 24}), easy::view<cd, 2>(y.data(), {16, 24}),
             {0});
    ctx.fft(xf, yf);
    EXPECT_EQ(count.allocations(), 0U);
  }
  EXPECT_EQ(ctx.cached_plans(), 8U);
  // The free functions, on this thread's context.
  easy::fft(x, y);
  {
    const fftn::test::allocation_counter count;
    easy::fft(x, y);
    easy::fft(x, z);
    EXPECT_EQ(count.allocations(), 0U);
  }
}

// At most max_plans() plans are kept, the least recently used dropped first; set_max_plans
// shrinks the cache at once; clear() drops everything.
TEST(easy, the_cache_is_bounded_and_drops_the_least_recently_used) {
  values gen(9);
  const std::vector<cd> x = gen.complexes<double>(64);
  std::vector<cd> y(64);
  const auto call = [&](easy::context& ctx, std::size_t n) {
    ctx.fft(std::span<const cd>(x.data(), n), std::span<cd>(y.data(), n));
  };
  const auto allocates = [&](easy::context& ctx, std::size_t n) {
    const fftn::test::allocation_counter count;
    call(ctx, n);
    return count.allocations() > 0;
  };
  easy::context ctx;
  ctx.set_max_plans(2);
  EXPECT_EQ(ctx.max_plans(), 2U);
  call(ctx, 8);  // A
  call(ctx, 16); // B
  call(ctx, 8);  // A again: B is now the least recently used
  call(ctx, 32); // C replaces B
  EXPECT_EQ(ctx.cached_plans(), 2U);
  EXPECT_FALSE(allocates(ctx, 8));
  EXPECT_FALSE(allocates(ctx, 32));
  EXPECT_TRUE(allocates(ctx, 16)); // B again, replacing A (32 was used after 8)
  EXPECT_TRUE(allocates(ctx, 8));
  EXPECT_FALSE(allocates(ctx, 16));
  EXPECT_EQ(ctx.cached_plans(), 2U);
  ctx.set_max_plans(1); // keeps the most recently used, 16
  EXPECT_EQ(ctx.cached_plans(), 1U);
  EXPECT_FALSE(allocates(ctx, 16));
  ctx.clear();
  EXPECT_EQ(ctx.cached_plans(), 0U);
  EXPECT_EQ(ctx.max_plans(), 1U);
  EXPECT_TRUE(allocates(ctx, 16));
  // The default bound.
  easy::context many;
  for (std::size_t n = 1; n <= 64 + 6; ++n) {
    std::vector<cd> u(n);
    std::vector<cd> v(n);
    many.fft(u, v);
  }
  EXPECT_EQ(many.cached_plans(), easy::context::default_max_plans);
}

// A context moves with its plans; the source is left empty and usable.
TEST(easy, a_context_moves_with_its_plans) {
  const std::vector<cd> x(32, cd{1});
  std::vector<cd> y(32);
  easy::context a;
  a.set_max_plans(5);
  a.fft(x, y);
  easy::context b(std::move(a));
  EXPECT_EQ(b.cached_plans(), 1U);
  EXPECT_EQ(b.max_plans(), 5U);
  EXPECT_EQ(a.cached_plans(), 0U); // NOLINT(bugprone-use-after-move): documented as empty
  a.fft(x, y);
  EXPECT_EQ(y[0], cd{32});
  easy::context c;
  c = std::move(b);
  EXPECT_EQ(c.cached_plans(), 1U);
  const fftn::test::allocation_counter count;
  c.fft(x, y);
  EXPECT_EQ(count.allocations(), 0U);
}

// Each thread has its own context: the free functions on several threads at once give the
// results they give on one thread, bit for bit, and fill only their own thread's cache.
TEST(easy, each_thread_has_its_own_context) {
  values gen(13);
  const std::vector<cd> x = gen.complexes<double>(4096);
  const std::vector<std::size_t> sizes{64, 100, 4096, 243};
  std::vector<std::vector<cd>> serial;
  for (const std::size_t n : sizes) {
    serial.push_back(easy::fft(std::span<const cd>(x.data(), n)));
  }
  const easy::context* main_context = &easy::thread_context();
  constexpr std::size_t threads = 4;
  std::vector<const easy::context*> seen(threads);
  std::vector<std::size_t> cached(threads);
  std::vector<int> agree(threads, 0);
  std::vector<std::thread> pool;
  // Every thread stays alive until all have taken their context's address: an exited thread's
  // thread_local storage may be reused by a later thread, at the same address.
  std::latch all_seen(static_cast<std::ptrdiff_t>(threads));
  for (std::size_t t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      bool ok = true;
      const std::size_t n = sizes[t];
      for (int rep = 0; rep < 20; ++rep) {
        const std::vector<cd> y = easy::fft(std::span<const cd>(x.data(), n));
        ok = ok && same_bits(y, serial[t]);
      }
      seen[t] = &easy::thread_context();
      cached[t] = easy::thread_context().cached_plans();
      agree[t] = ok ? 1 : 0;
      all_seen.arrive_and_wait();
    });
  }
  for (std::thread& th : pool) {
    th.join();
  }
  std::set<const easy::context*> distinct(seen.begin(), seen.end());
  distinct.insert(main_context);
  EXPECT_EQ(distinct.size(), threads + 1);
  for (std::size_t t = 0; t < threads; ++t) {
    EXPECT_EQ(agree[t], 1) << "thread " << t;
    EXPECT_EQ(cached[t], 1U) << "thread " << t;
  }
}

// A context on an executor gives the serial context's results, bit for bit.
TEST(easy, a_context_on_an_executor_matches_serial) {
  values gen(17);
  const std::vector<cd> x = gen.complexes<double>(64 * 256);
  std::vector<cd> serial(x.size());
  std::vector<cd> pooled(x.size());
  const fftn::thread_pool pool(4);
  easy::context on_pool(pool);
  easy::context alone;
  const easy::view<const cd, 2> in(x.data(), {64, 256});
  alone.fftn(in, easy::view<cd, 2>(serial.data(), {64, 256}), {-1});
  on_pool.fftn(in, easy::view<cd, 2>(pooled.data(), {64, 256}), {-1});
  EXPECT_TRUE(same_bits(serial, pooled));
  alone.fft2(in, easy::view<cd, 2>(serial.data(), {64, 256}));
  on_pool.fft2(in, easy::view<cd, 2>(pooled.data(), {64, 256}));
  EXPECT_TRUE(same_bits(serial, pooled));
  std::vector<double> r(64 * 256);
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i] = x[i].real();
  }
  std::vector<double> rs(r.size());
  std::vector<double> rp(r.size());
  std::vector<cd> h(64 * 129);
  alone.rfft2(easy::view<const double, 2>(r.data(), {64, 256}),
              easy::view<cd, 2>(h.data(), {64, 129}));
  alone.irfft2(easy::view<const cd, 2>(h.data(), {64, 129}),
               easy::view<double, 2>(rs.data(), {64, 256}));
  on_pool.irfft2(easy::view<const cd, 2>(h.data(), {64, 129}),
                 easy::view<double, 2>(rp.data(), {64, 256}));
  EXPECT_TRUE(same_bits(rs, rp));
}

// numpy.fft.fftfreq and rfftfreq: the integer frequency times 1 / (n d).
TEST(easy, frequencies_are_numpys) {
  std::vector<double> f8(8);
  easy::fftfreq(8, f8);
  EXPECT_EQ(f8, (std::vector<double>{0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125}));
  std::vector<double> f5(5);
  easy::fftfreq(5, f5, 0.1);
  const double u = 1.0 / (5 * 0.1);
  EXPECT_EQ(f5, (std::vector<double>{0, 1 * u, 2 * u, -2 * u, -1 * u}));
  std::vector<float> r8(5);
  easy::rfftfreq(8, r8, 2.0);
  EXPECT_EQ(r8, (std::vector<float>{0, 0.0625F, 0.125F, 0.1875F, 0.25F}));
  std::vector<double> r5(3);
  easy::rfftfreq(5, r5);
  EXPECT_EQ(r5, (std::vector<double>{0, 0.2, 0.4}));
  std::vector<double> f1(1);
  easy::fftfreq(1, f1);
  EXPECT_EQ(f1[0], 0.0);
}

// numpy.fft.fftshift and ifftshift: odd and even extents, chosen axes, in place, strided.
TEST(easy, shifts_are_numpys) {
  const auto shifted = [](std::vector<int> v, bool inverse) {
    std::vector<int> out(v.size());
    const easy::view<const int, 1> in(v.data(), {v.size()});
    if (inverse) {
      easy::ifftshift(in, easy::view<int, 1>(out.data(), {v.size()}));
    } else {
      easy::fftshift(in, easy::view<int, 1>(out.data(), {v.size()}));
    }
    return out;
  };
  EXPECT_EQ(shifted({0, 1, 2, 3, 4}, false), (std::vector<int>{3, 4, 0, 1, 2}));
  EXPECT_EQ(shifted({0, 1, 2, 3, 4}, true), (std::vector<int>{2, 3, 4, 0, 1}));
  EXPECT_EQ(shifted({0, 1, 2, 3, 4, 5}, false), (std::vector<int>{3, 4, 5, 0, 1, 2}));
  EXPECT_EQ(shifted({0, 1, 2, 3, 4, 5}, true), (std::vector<int>{3, 4, 5, 0, 1, 2}));
  EXPECT_EQ(shifted({7}, false), (std::vector<int>{7}));
  // np.fft.fftshift(np.arange(12).reshape(3, 4)) and with axes=(1,) and axes=(-2,).
  std::vector<int> a(12);
  for (int i = 0; i < 12; ++i) {
    a[static_cast<std::size_t>(i)] = i;
  }
  const easy::view<const int, 2> in(a.data(), {3, 4});
  std::vector<int> b(12);
  const easy::view<int, 2> out(b.data(), {3, 4});
  easy::fftshift(in, out);
  EXPECT_EQ(b, (std::vector<int>{10, 11, 8, 9, 2, 3, 0, 1, 6, 7, 4, 5}));
  easy::fftshift(in, out, {1});
  EXPECT_EQ(b, (std::vector<int>{2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9}));
  easy::fftshift(in, out, {-2});
  EXPECT_EQ(b, (std::vector<int>{8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7}));
  easy::ifftshift(in, out);
  EXPECT_EQ(b, (std::vector<int>{6, 7, 4, 5, 10, 11, 8, 9, 2, 3, 0, 1}));
  // In place agrees with out of place, and ifftshift undoes fftshift, over a strided volume.
  std::vector<std::complex<float>> v(5 * 6 * 7 * 2);
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = {static_cast<float>(i), -static_cast<float>(i)};
  }
  const std::vector<std::complex<float>> original = v;
  const easy::view<std::complex<float>, 3> vol(v.data() + 1, {5, 6, 7}, {84, 14, 2});
  std::vector<std::complex<float>> copy(5 * 6 * 7);
  const easy::view<std::complex<float>, 3> tight(copy.data(), {5, 6, 7});
  easy::fftshift(easy::view<const std::complex<float>, 3>(vol), tight, {0, 2});
  easy::fftshift(easy::view<const std::complex<float>, 3>(vol), vol, {0, 2});
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      for (std::size_t k = 0; k < 7; ++k) {
        EXPECT_EQ(v[1 + i * 84 + j * 14 + k * 2], copy[(i * 6 + j) * 7 + k]);
        EXPECT_EQ(v[i * 84 + j * 14 + k * 2], original[i * 84 + j * 14 + k * 2]) << "gap written";
      }
    }
  }
  easy::ifftshift(easy::view<const std::complex<float>, 3>(vol), vol, {0, 2});
  EXPECT_EQ(v, original);
}

} // namespace
