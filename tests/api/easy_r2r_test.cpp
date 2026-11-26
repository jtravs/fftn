// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The convenience layer's real-to-real functions (fftn/easy.hpp, step 6.7) against scipy.fft:
// recorded scipy results for every function, type and norm (data/scipy_r2r_cases.txt, from
// data/scipy_r2r_cases.py), the value forms bitwise equal to the span forms, every call bitwise
// equal to the core plan with the norm's scale and scipy's orthogonalising weights as real edge
// factors, round trips, in place and strided views, an executor, the cache (a hit allocates
// nothing; the weights are kept with the plan) and refusals that name the argument.

#include "support/alloc_hook.hpp"
#include "support/exact.hpp"
#include "support/outcome.hpp"
#include "support/precisions.hpp"
#include "support/tolerance.hpp"

#include <fftn/easy.hpp>
#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace easy = fftn::easy;
using fftn::r2r_kind;
using dims_t = std::vector<std::size_t>;

easy::norm norm_of(const std::string& s) {
  if (s == "ortho") {
    return easy::norm::ortho;
  }
  return s == "forward" ? easy::norm::forward : easy::norm::backward;
}

constexpr std::array norms{easy::norm::backward, easy::norm::ortho, easy::norm::forward};

std::size_t product(const dims_t& d) {
  std::size_t p = 1;
  for (const std::size_t e : d) {
    p *= e;
  }
  return p;
}

// One recorded scipy call.
struct scipy_case {
  std::string function; // dct, idct, ..., dctn, ..., dht
  int type = 0;
  std::string norm;
  dims_t extents;
  std::vector<int> axes; // as passed to scipy; empty for the default (all)
  std::vector<double> in;
  std::vector<double> out;
};

std::vector<double> numbers(std::istringstream& is) {
  std::vector<double> v;
  std::string token;
  while (is >> token) {
    v.push_back(std::strtod(token.c_str(), nullptr));
  }
  return v;
}

const std::vector<scipy_case>& recorded_cases() {
  static const std::vector<scipy_case> cases = [] {
    std::vector<scipy_case> all;
    std::ifstream file(std::string(FFTN_EASY_DATA) + "/scipy_r2r_cases.txt");
    if (!file) {
      throw std::runtime_error("cannot read " + std::string(FFTN_EASY_DATA) +
                               "/scipy_r2r_cases.txt");
    }
    std::string text;
    while (std::getline(file, text)) {
      if (text.empty() || text[0] == '#') {
        continue;
      }
      std::istringstream is(text);
      std::string tag;
      is >> tag;
      if (tag == "case") {
        scipy_case c;
        std::size_t rank = 0;
        std::size_t naxes = 0;
        is >> c.function >> c.type >> c.norm >> rank;
        c.extents.resize(rank);
        for (std::size_t& e : c.extents) {
          is >> e;
        }
        is >> naxes;
        c.axes.resize(naxes);
        for (int& a : c.axes) {
          is >> a;
        }
        all.push_back(std::move(c));
      } else if (tag == "in") {
        all.back().in = numbers(is);
      } else if (tag == "out") {
        all.back().out = numbers(is);
      }
    }
    return all;
  }();
  return cases;
}

// The kind scipy's function and type compute along each axis: the inverse functions compute the
// inverse type.
r2r_kind kind_of(const std::string& function, int type) {
  if (function == "dht") {
    return r2r_kind::dht;
  }
  const bool inverse = function[0] == 'i';
  const bool sine = function.find("dst") != std::string::npos;
  int t = type;
  if (inverse && (t == 2 || t == 3)) {
    t = 5 - t;
  }
  return static_cast<r2r_kind>((sine ? 4 : 0) + t - 1);
}

// The tolerance of a call over `phases` axes whose underlying transforms have `points` points
// in all (the largest route a kind takes, 4n for the odd DCT-IV, bounds them): the transforms'
// derived tolerance, doubled for the weights' spread of the outputs, a rounding per stage
// operation per phase (10, as r2r_test.cpp counts the DCT kinds'), the scale, the weights, and
// scipy's own rounding of the recorded result to double.
template <class T> fftn::test::tolerance tolerance_for(std::size_t points, std::size_t phases) {
  const auto p = static_cast<double>(phases);
  return fftn::test::derived_tolerance<T>(points, fftn::test::fftn_bound(points)) * 2.0 +
         fftn::test::input_rounding<T>() * (10.0 * p + 3.0) +
         fftn::test::input_rounding<double>() * (p + 1.0);
}

template <class T>
void expect_close(std::span<const T> got, const std::vector<double>& want,
                  const fftn::test::tolerance& tol, const std::string& what) {
  std::vector<fftn::test::exact_real> e;
  e.reserve(want.size());
  for (const double v : want) {
    e.push_back(fftn::test::exact_from(v));
  }
  fftn::test::outcome o;
  o.within(fftn::test::compare<T>(got, e), tol, what);
  EXPECT_TRUE(o.passed()) << o.detail();
}

template <class V> bool same_bits(const V& a, const V& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(typename V::value_type)) == 0;
}

// The transformed dimensions of a call on an array of `rank`, in the order listed.
std::vector<std::size_t> chosen(std::size_t rank, const std::vector<int>& axes) {
  std::vector<std::size_t> d;
  if (axes.empty()) {
    for (std::size_t i = 0; i < rank; ++i) {
      d.push_back(i);
    }
    return d;
  }
  for (const int a : axes) {
    d.push_back(static_cast<std::size_t>(a < 0 ? a + static_cast<int>(rank) : a));
  }
  return d;
}

// Calls the N-D function `function` on views of rank R.
template <class T, std::size_t R>
void call_n(easy::context& ctx, const std::string& function, int type, const dims_t& ext,
            const std::vector<int>& axes, easy::norm nm, const T* in, T* out,
            const std::array<std::ptrdiff_t, R>& in_strides = {},
            const std::array<std::ptrdiff_t, R>& out_strides = {}) {
  std::array<std::size_t, R> e{};
  std::copy_n(ext.begin(), R, e.begin());
  const easy::view<const T, R> vi(in, e, in_strides);
  const easy::view<T, R> vo(out, e, out_strides);
  const easy::axis_list al(axes);
  if (function == "dctn") {
    ctx.dctn(vi, vo, type, al, nm);
  } else if (function == "idctn") {
    ctx.idctn(vi, vo, type, al, nm);
  } else if (function == "dstn") {
    ctx.dstn(vi, vo, type, al, nm);
  } else {
    ctx.idstn(vi, vo, type, al, nm);
  }
}

template <class T>
void call_any(easy::context& ctx, const scipy_case& c, easy::norm nm, const std::vector<T>& in,
              std::vector<T>& out) {
  const std::span<const T> i(in);
  const std::span<T> o(out);
  if (c.function == "dct") {
    ctx.dct(i, o, c.type, nm);
  } else if (c.function == "idct") {
    ctx.idct(i, o, c.type, nm);
  } else if (c.function == "dst") {
    ctx.dst(i, o, c.type, nm);
  } else if (c.function == "idst") {
    ctx.idst(i, o, c.type, nm);
  } else if (c.function == "dht") {
    ctx.dht(i, o, nm);
  } else if (c.extents.size() == 2) {
    call_n<T, 2>(ctx, c.function, c.type, c.extents, c.axes, nm, in.data(), out.data());
  } else if (c.extents.size() == 3) {
    call_n<T, 3>(ctx, c.function, c.type, c.extents, c.axes, nm, in.data(), out.data());
  } else {
    call_n<T, 4>(ctx, c.function, c.type, c.extents, c.axes, nm, in.data(), out.data());
  }
}

// The value forms of the 1-D functions, on the calling thread's context.
template <class T>
std::vector<T> value_form(const scipy_case& c, easy::norm nm, std::span<const T> in) {
  if (c.function == "dct") {
    return easy::dct(in, c.type, nm);
  }
  if (c.function == "idct") {
    return easy::idct(in, c.type, nm);
  }
  if (c.function == "dst") {
    return easy::dst(in, c.type, nm);
  }
  if (c.function == "idst") {
    return easy::idst(in, c.type, nm);
  }
  return easy::dht(in, nm);
}

// scipy's logical size of one axis (the N its norms divide by).
double logical(r2r_kind k, std::size_t n) {
  const auto x = static_cast<double>(n);
  switch (k) {
  case r2r_kind::dct1:
    return 2 * (x - 1);
  case r2r_kind::dst1:
    return 2 * (x + 1);
  case r2r_kind::dht:
    return x;
  default:
    return 2 * x;
  }
}

// scipy's orthogonalising weights of a line of n for kind k, as the input's multipliers and the
// output's (DESIGN.md 5.4; scipy.fft.dct's notes).
std::pair<std::vector<double>, std::vector<double>> line_weights(r2r_kind k, std::size_t n) {
  std::vector<double> wi(n, 1.0);
  std::vector<double> wo(n, 1.0);
  const double r = std::sqrt(2.0);
  switch (k) {
  case r2r_kind::dct1:
    wi.front() = wi.back() = r;
    wo.front() = wo.back() = 1.0 / r;
    break;
  case r2r_kind::dct2:
    wo.front() = 1.0 / r;
    break;
  case r2r_kind::dct3:
    wi.front() = r;
    break;
  case r2r_kind::dst2:
    wo.back() = 1.0 / r;
    break;
  case r2r_kind::dst3:
    wi.back() = r;
    break;
  default:
    break;
  }
  return {wi, wo};
}

// The core plan a call must equal bit for bit, built by hand from the definitions: axes_r2r of
// the kind, options::scale the norm's factor, and the weights (under norm::ortho, on the sides
// that have them) as real factors on the edges, C-ordered over the transformed axes in the order
// listed, broadcast along the batch. The weight of a point is 2^(m/2), m the number of axes on
// which it is weighted, rounded once.
template <class T>
std::vector<T> core_result(const dims_t& ext, const std::vector<std::size_t>& dims, r2r_kind k,
                           bool inverse, easy::norm nm, const std::vector<T>& in) {
  double points = 1;
  for (const std::size_t d : dims) {
    points *= logical(k, ext[d]);
  }
  double scale = 1.0;
  if (nm == easy::norm::ortho) {
    scale = 1.0 / std::sqrt(points);
  } else if ((nm == easy::norm::forward) != inverse) {
    scale = 1.0 / points;
  }
  std::vector<fftn::axis> array;
  for (const std::size_t e : ext) {
    array.push_back(fftn::axis{.n = e});
  }
  const std::vector<r2r_kind> kinds(dims.size(), k);
  const fftn::shape<T, T> s = fftn::axes_r2r<T>(array, dims, kinds);
  fftn::plan_edges<T, T> edges;
  std::vector<T> win;
  std::vector<T> wout;
  bool any = false;
  if (nm == easy::norm::ortho) {
    for (const bool input : {true, false}) {
      std::vector<std::vector<double>> per_axis;
      bool weighted = false;
      for (const std::size_t d : dims) {
        const auto [wi, wo] = line_weights(k, ext[d]);
        per_axis.push_back(input ? wi : wo);
        weighted =
            weighted || std::ranges::any_of(per_axis.back(), [](double w) { return w != 1; });
      }
      if (!weighted) {
        continue;
      }
      any = true;
      std::vector<std::ptrdiff_t> strides(dims.size());
      std::size_t total = 1;
      for (std::size_t i = dims.size(); i > 0; --i) {
        strides[i - 1] = static_cast<std::ptrdiff_t>(total);
        total *= ext[dims[i - 1]];
      }
      std::vector<T>& w = input ? win : wout;
      w.resize(total);
      for (std::size_t e = 0; e < total; ++e) {
        int m = 0;
        std::size_t rest = e;
        for (std::size_t i = dims.size(); i > 0; --i) {
          const std::size_t n = ext[dims[i - 1]];
          m += per_axis[i - 1][rest % n] != 1.0 ? 1 : 0;
          rest /= n;
        }
        const double v = m % 2 == 0
                             ? std::ldexp(1.0, input ? m / 2 : -m / 2)
                             : std::ldexp(std::sqrt(2.0), input ? (m - 1) / 2 : -(m + 1) / 2);
        w[e] = static_cast<T>(v);
      }
      (input ? edges.in : edges.out).mul =
          fftn::factor{.kind = fftn::factor_kind::real, .axis_strides = strides};
    }
  }
  std::vector<T> out(in.size());
  const fftn::options opts{.scale = scale};
  if (any) {
    const fftn::plan<T, T> p(s, opts, edges);
    fftn::workspace ws(p);
    p.execute(in.data(), out.data(), ws,
              fftn::exec_args{.in_factor = win.empty() ? nullptr : win.data(),
                              .out_factor = wout.empty() ? nullptr : wout.data()});
  } else {
    const fftn::plan<T, T> p(s, opts);
    fftn::workspace ws(p);
    p.execute(in.data(), out.data(), ws);
  }
  return out;
}

template <class T> class easy_r2r_test : public ::testing::Test {};
TYPED_TEST_SUITE(easy_r2r_test, fftn::test::precisions, fftn::test::precision_names);

// Every recorded case: scipy's result within the derived tolerance, the value form bitwise equal
// to the span form, and the call bitwise equal to the core plan with the weights as edge factors.
TYPED_TEST(easy_r2r_test, recorded_scipy_results) {
  using T = TypeParam;
  const std::vector<scipy_case>& cases = recorded_cases();
  ASSERT_GT(cases.size(), 700U);
  std::size_t checked = 0;
  for (const scipy_case& c : cases) {
    const easy::norm nm = norm_of(c.norm);
    const std::string at = c.function + " type " + std::to_string(c.type) + " norm " + c.norm +
                           " extents " + std::to_string(c.extents.size()) + "-D of " +
                           std::to_string(product(c.extents));
    const std::vector<T> in(c.in.begin(), c.in.end());
    ASSERT_EQ(in.size(), product(c.extents)) << at;
    ASSERT_EQ(c.out.size(), in.size()) << at;
    std::vector<T> out(in.size());
    easy::context& ctx = easy::thread_context();
    call_any(ctx, c, nm, in, out);
    const std::vector<std::size_t> dims = chosen(c.extents.size(), c.axes);
    const r2r_kind k = kind_of(c.function, c.type);
    std::size_t points = 1;
    for (const std::size_t d : dims) {
      points *= k == r2r_kind::dht ? c.extents[d] : 4 * c.extents[d];
    }
    expect_close<T>(out, c.out, tolerance_for<T>(points, dims.size()), at);
    if (c.extents.size() == 1) {
      EXPECT_TRUE(same_bits(out, value_form<T>(c, nm, in))) << at;
    }
    EXPECT_TRUE(same_bits(out, core_result<T>(c.extents, dims, k, c.function[0] == 'i', nm, in)))
        << at;
    ++checked;
  }
  EXPECT_EQ(checked, cases.size());
}

// Deterministic values in [-1, 1), exact in float and in double.
template <class T> std::vector<T> values_of(std::size_t n, std::uint32_t seed) {
  std::vector<T> v(n);
  std::uint32_t state = seed;
  for (T& x : v) {
    state = state * 1664525U + 1013904223U;
    x = static_cast<T>(static_cast<double>(state >> 8U) / 8388608.0 - 1.0);
  }
  return v;
}

// Each inverse function undoes its forward one, for every type and norm, in one and two
// dimensions; the DHT with norm::forward undoes it with norm::backward, and with norm::ortho
// undoes itself.
TYPED_TEST(easy_r2r_test, round_trips) {
  using T = TypeParam;
  for (const std::size_t n : dims_t{2, 3, 8, 15, 64, 97, 100}) {
    const std::vector<T> x = values_of<T>(n, static_cast<std::uint32_t>(n));
    const auto tol = tolerance_for<T>(4 * n, 1) * 2.0;
    for (const easy::norm nm : norms) {
      for (int type = 1; type <= 4; ++type) {
        const std::string at = "n " + std::to_string(n) + " type " + std::to_string(type) +
                               " norm " + std::to_string(static_cast<int>(nm));
        const std::vector<double> want(x.begin(), x.end());
        expect_close<T>(
            easy::idct(std::span<const T>(easy::dct(std::span<const T>(x), type, nm)), type, nm),
            want, tol, "idct(dct), " + at);
        expect_close<T>(
            easy::idst(std::span<const T>(easy::dst(std::span<const T>(x), type, nm)), type, nm),
            want, tol, "idst(dst), " + at);
      }
    }
    const std::vector<double> want(x.begin(), x.end());
    const auto htol = tolerance_for<T>(n, 1) * 2.0;
    expect_close<T>(
        easy::dht(std::span<const T>(easy::dht(std::span<const T>(x))), easy::norm::forward), want,
        htol, "dht forward after backward, n " + std::to_string(n));
    expect_close<T>(
        easy::dht(std::span<const T>(easy::dht(std::span<const T>(x), easy::norm::ortho)),
                  easy::norm::ortho),
        want, htol, "dht ortho twice, n " + std::to_string(n));
  }
  // dctn and idctn of every type on a 6 x 5 array, the ortho weights on both axes.
  const std::vector<T> a = values_of<T>(30, 77);
  const std::vector<double> want(a.begin(), a.end());
  for (int type = 1; type <= 4; ++type) {
    for (const easy::norm nm : norms) {
      std::vector<T> y(30);
      std::vector<T> z(30);
      easy::dctn(easy::view<const T, 2>(a.data(), {6, 5}), easy::view<T, 2>(y.data(), {6, 5}), type,
                 {}, nm);
      easy::idctn(easy::view<const T, 2>(y.data(), {6, 5}), easy::view<T, 2>(z.data(), {6, 5}),
                  type, {}, nm);
      expect_close<T>(z, want, tolerance_for<T>(4 * 30, 2) * 2.0,
                      "idctn(dctn) type " + std::to_string(type));
      easy::dstn(easy::view<const T, 2>(a.data(), {6, 5}), easy::view<T, 2>(y.data(), {6, 5}), type,
                 {}, nm);
      easy::idstn(easy::view<const T, 2>(y.data(), {6, 5}), easy::view<T, 2>(z.data(), {6, 5}),
                  type, {}, nm);
      expect_close<T>(z, want, tolerance_for<T>(4 * 30, 2) * 2.0,
                      "idstn(dstn) type " + std::to_string(type));
    }
  }
}

// In place (the same memory and strides) and strided views (padded rows, a negative stride,
// a batch dimension between transformed ones) give the tight out-of-place result, within the
// derived tolerance; with weights, so the factors' strides follow the transformed axes only.
TYPED_TEST(easy_r2r_test, in_place_and_strided_views) {
  using T = TypeParam;
  const dims_t ext{4, 3, 6};
  const std::vector<T> x = values_of<T>(72, 5);
  for (const char* const function : {"dctn", "idctn", "dstn", "idstn"}) {
    for (int type = 1; type <= 4; ++type) {
      for (const std::vector<int>& axes : {std::vector<int>{}, std::vector<int>{0, 2}}) {
        const std::string at = std::string(function) + " type " + std::to_string(type) + " axes " +
                               std::to_string(axes.size());
        easy::context ctx;
        std::vector<T> tight(72);
        call_n<T, 3>(ctx, function, type, ext, axes, easy::norm::ortho, x.data(), tight.data());
        const std::vector<double> want(tight.begin(), tight.end());
        const auto tol = tolerance_for<T>(4 * 72, 3) * 2.0;
        // In place.
        std::vector<T> y = x;
        call_n<T, 3>(ctx, function, type, ext, axes, easy::norm::ortho, y.data(), y.data());
        expect_close<T>(y, want, tol, "in place, " + at);
        // Input rows padded to 8 and the middle dimension reversed; output in Fortran order.
        std::vector<T> padded(4 * 3 * 8);
        for (std::size_t i = 0; i < 4; ++i) {
          for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 6; ++k) {
              padded[(i * 3 + (2 - j)) * 8 + k] = x[(i * 3 + j) * 6 + k];
            }
          }
        }
        std::vector<T> fortran(72);
        call_n<T, 3>(ctx, function, type, ext, axes, easy::norm::ortho, padded.data() + 2 * 8,
                     fortran.data(), std::array<std::ptrdiff_t, 3>{24, -8, 1},
                     std::array<std::ptrdiff_t, 3>{1, 4, 12});
        std::vector<T> back(72);
        for (std::size_t i = 0; i < 4; ++i) {
          for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 6; ++k) {
              back[(i * 3 + j) * 6 + k] = fortran[i + 4 * j + 12 * k];
            }
          }
        }
        expect_close<T>(back, want, tol, "strided, " + at);
      }
    }
  }
  // The 1-D span forms in place.
  std::vector<T> y = x;
  const std::vector<T> want = easy::dct(std::span<const T>(x), 1, easy::norm::ortho);
  easy::dct(y, y, 1, easy::norm::ortho);
  expect_close<T>(y, {want.begin(), want.end()}, tolerance_for<T>(4 * 72, 1) * 2.0, "dct in place");
}

// A context on an executor gives the serial context's results, bit for bit, weights included.
TEST(easy_r2r, a_context_on_an_executor_matches_serial) {
  const std::vector<double> x = values_of<double>(64 * 256, 3);
  std::vector<double> serial(x.size());
  std::vector<double> pooled(x.size());
  const fftn::thread_pool pool(4);
  easy::context on_pool(pool);
  easy::context alone;
  const easy::view<const double, 2> in(x.data(), {64, 256});
  for (int type = 1; type <= 4; ++type) {
    alone.dctn(in, easy::view<double, 2>(serial.data(), {64, 256}), type, {-1}, easy::norm::ortho);
    on_pool.dctn(in, easy::view<double, 2>(pooled.data(), {64, 256}), type, {-1},
                 easy::norm::ortho);
    EXPECT_TRUE(same_bits(serial, pooled)) << "dctn type " << type;
    alone.idstn(in, easy::view<double, 2>(serial.data(), {64, 256}), type, {}, easy::norm::ortho);
    on_pool.idstn(in, easy::view<double, 2>(pooled.data(), {64, 256}), type, {}, easy::norm::ortho);
    EXPECT_TRUE(same_bits(serial, pooled)) << "idstn type " << type;
  }
}

// A call that finds its plan allocates nothing, its weights included; type, norm, direction and
// family each make another plan.
TEST(easy_r2r, a_cache_hit_allocates_nothing) {
  const std::vector<double> x = values_of<double>(300, 9);
  std::vector<double> y(300);
  easy::context ctx;
  ctx.dct(x, y, 1, easy::norm::ortho); // weights on both sides
  ctx.dct(x, y, 2, easy::norm::ortho);
  ctx.dct(x, y, 2);
  ctx.idct(x, y, 2, easy::norm::ortho);
  ctx.dst(x, y, 2, easy::norm::ortho);
  ctx.dht(x, y);
  ctx.dctn(easy::view<const double, 2>(x.data(), {12, 25}),
           easy::view<double, 2>(y.data(), {12, 25}), 1, {}, easy::norm::ortho);
  EXPECT_EQ(ctx.cached_plans(), 7U);
  {
    const fftn::test::allocation_counter count;
    ctx.dct(x, y, 1, easy::norm::ortho);
    ctx.dct(x, y, 2, easy::norm::ortho);
    ctx.idct(x, y, 2, easy::norm::ortho);
    ctx.dst(x, y, 2, easy::norm::ortho);
    ctx.dht(x, y);
    ctx.dctn(easy::view<const double, 2>(x.data(), {12, 25}),
             easy::view<double, 2>(y.data(), {12, 25}), 1, {}, easy::norm::ortho);
    EXPECT_EQ(count.allocations(), 0U);
  }
  EXPECT_EQ(ctx.cached_plans(), 7U);
  // dct of type 3 is the plan of idct of type 2 only up to the norm's direction: another plan.
  ctx.dct(x, y, 3, easy::norm::forward);
  ctx.idct(x, y, 2, easy::norm::forward);
  EXPECT_EQ(ctx.cached_plans(), 9U);
}

template <class F> void refuses(F f, const std::string& names) {
  try {
    f();
    ADD_FAILURE() << "no exception; expected one naming " << names;
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(names), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("fftn::easy::"), std::string::npos) << e.what();
  }
}

// Refusals name the argument, and nothing is written.
TEST(easy_r2r, refusals_name_the_argument) {
  std::vector<double> x(8, 1.0);
  std::vector<double> y(8, 7.0);
  std::vector<double> one(1, 1.0);
  std::vector<double> z(1, 7.0);
  refuses([&] { easy::dct(x, y, 0); }, "type is 0");
  refuses([&] { easy::idst(x, y, 5); }, "type is 5");
  refuses([&] { easy::dct(one, z, 1); }, "in.size() is 1");
  refuses([&] { easy::idct(one, z, 1); }, "in.size() is 1");
  refuses([&] { easy::dct(x, std::span<double>(y.data(), 7)); }, "out.size()");
  refuses([&] { easy::dht(std::span<const double>(), std::span<double>()); }, "in.size() is 0");
  refuses(
      [&] {
        easy::dctn(easy::view<const double, 2>(x.data(), {8, 1}),
                   easy::view<double, 2>(y.data(), {8, 1}), 1);
      },
      "in.extents[1] is 1");
  refuses(
      [&] {
        easy::dstn(easy::view<const double, 2>(x.data(), {2, 4}),
                   easy::view<double, 2>(y.data(), {2, 4}), 9);
      },
      "type is 9");
  refuses(
      [&] {
        easy::dctn(easy::view<const double, 2>(x.data(), {2, 4}),
                   easy::view<double, 2>(y.data(), {2, 4}), 2, {0, 0});
      },
      "axes");
  refuses(
      [&] {
        easy::idctn(easy::view<const double, 2>(x.data(), {2, 4}),
                    easy::view<double, 2>(x.data(), {2, 4}, {1, 2}));
      },
      "in place needs the same strides");
  EXPECT_EQ(y, std::vector<double>(8, 7.0));
  EXPECT_EQ(z, std::vector<double>(1, 7.0));
  // A DST-I and the other types of one point are defined: scipy's values.
  std::vector<double> w(1);
  easy::dst(one, w, 1, easy::norm::ortho);
  EXPECT_EQ(w[0], 1.0);
  easy::dct(one, w, 2, easy::norm::ortho);
  EXPECT_EQ(w[0], 1.0);
  easy::dstn(easy::view<const double, 2>(x.data(), {8, 1}), easy::view<double, 2>(y.data(), {8, 1}),
             1);
}

} // namespace
