// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The routes of real transforms (DESIGN.md 9.1, src/passes/real.inl, task 2.1), each forced
// on every instruction set the machine runs, in double and float: the pairing fused into
// the last pass (r2c) or the first (c2r) with every codelet radix in that pass and copy
// counts odd, even and spanning several registers; the pairing as a separate pass (after a
// chain, a Bluestein stage or a grid); the leaves; the rules' choice. Out of place with gaps
// between rows (canaries) and in place in FFTW's padded rows, batches, and a scale.
//
// Every bin is compared with the full-length complex transform of the same data through
// FFTN's chain (which the reference suites check), and a sample of bins with the
// quadruple-precision reference, which keeps long rows cheap. Bitwise independence of the
// team size is tests/threading's (team_cases.hpp has these routes).

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;

struct real_case {
  std::size_t n;
  build_settings settings;
  const char* route; // a phrase describe() must contain
};

template <class T> fftn::test::tolerance tolerance_of(std::size_t n) {
  return fftn::test::derived_tolerance<T>(n, fftn::test::has_large_prime_factor(n)
                                                 ? fftn::test::bluestein_bound(n)
                                                 : fftn::test::stable_fft_bound);
}

// Bins 0, 1, 2, a stride through the rest and the last three, of `count` outputs.
std::vector<std::size_t> sample(std::size_t count) {
  std::vector<std::size_t> v;
  const std::size_t step = count <= 24 ? 1 : count / 16;
  for (std::size_t k = 0; k < count; k += k < 3 ? 1 : step) {
    v.push_back(k);
  }
  for (std::size_t k = count >= 3 ? count - 3 : 0; k < count; ++k) {
    if (v.back() < k) {
      v.push_back(k);
    }
  }
  return v;
}

// The c2c transform of x (n values) through FFTN's chain on `target`, unscaled.
template <class T>
std::vector<std::complex<T>> full_length(fftn::isa target, int sign,
                                         const std::vector<std::complex<T>>& x) {
  build_settings chain;
  chain.strategy = fftn::detail::strategy_choice::chain;
  const auto dir = sign < 0 ? fftn::direction::forward : fftn::direction::backward;
  const auto core =
      make_plan_core(fftn::detail::view_of(fftn::rows_c2c<T>(x.size(), 1, dir)),
                     {.force_isa = target}, chain);
  std::vector<std::complex<T>> y(x.size());
  const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  fftn::detail::execute_core(*core, 0, 1, x.data(), y.data(), ws);
  return y;
}

template <class E, class V> std::vector<E> exact_of(const std::vector<V>& v, double scale) {
  std::vector<E> out;
  out.reserve(v.size());
  for (const V& e : v) {
    out.push_back(fftn::test::exact_from(e * scale));
  }
  return out;
}

// `count` rows of n through an r2c or c2r plan, each against the full-length transform at
// every bin and the reference at a sample; out of place, the elements between rows must be
// untouched.
template <class T>
void check(fftn::isa target, const real_case& rc, bool r2c, bool in_place, double scale,
           std::size_t count) {
  using c = std::complex<T>;
  const std::size_t n = rc.n;
  const std::size_t h1 = n / 2 + 1;
  // Out of place the rows have gaps; in place FFTW's padded rows (2 h1 reals, h1 complex).
  const std::size_t rdist = in_place ? 2 * h1 : n + 3;
  const std::size_t cdist = in_place ? h1 : h1 + 2;
  const std::string what = std::string(fftn::isa_name(target)) +
                           (sizeof(T) == 8 ? " double" : " float") + (r2c ? " r2c" : " c2r") +
                           " n = " + std::to_string(n) + (in_place ? ", in place" : "") +
                           ", scale " + std::to_string(scale) + ", " + std::to_string(count) +
                           " rows";
  SCOPED_TRACE(what);
  const fftn::options opts{.in_place = in_place, .scale = scale, .force_isa = target};
  const auto rd = static_cast<std::ptrdiff_t>(rdist);
  const auto cd = static_cast<std::ptrdiff_t>(cdist);
  fftn::test::input_source src(0x2E41ULL + n * 7 + count + (r2c ? 1 : 0));
  const auto tol = tolerance_of<T>(n);
  fftn::test::outcome o;
  std::string text;
  if (r2c) {
    const fftn::shape<T, c> shape{.axes = {{.n = n}},
                                  .batches = {{.count = count, .in_dist = rd, .out_dist = cd}}};
    const auto core = make_plan_core(fftn::detail::view_of(shape), opts, rc.settings);
    text = fftn::detail::describe(*core);
    const std::vector<T> x = src.reals<T>(rdist * count);
    const c canary(T(-7.25), T(3.5));
    std::vector<c> y(cdist * count, canary);
    std::vector<T> buffer = x; // in place: the reals, then the spectrum over them
    const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
    fftn::detail::execute_core(*core, 0, count, in_place ? buffer.data() : x.data(),
                               in_place ? static_cast<void*>(buffer.data()) : y.data(), ws);
    for (std::size_t b = 0; b < count; ++b) {
      std::vector<c> got(h1);
      std::vector<c> xc(n);
      std::vector<double> xd(n);
      for (std::size_t k = 0; k < h1; ++k) {
        got[k] = in_place ? c(buffer[b * rdist + 2 * k], buffer[b * rdist + 2 * k + 1])
                          : y[b * cdist + k];
      }
      for (std::size_t j = 0; j < n; ++j) {
        xc[j] = c(x[b * rdist + j], T(0));
        xd[j] = static_cast<double>(x[b * rdist + j]);
      }
      std::vector<c> full = full_length<T>(target, -1, xc);
      full.resize(h1);
      std::vector<std::complex<double>> fd(full.begin(), full.end());
      const std::string row = what + ", row " + std::to_string(b);
      o.within(fftn::test::compare<T>(std::span<const c>(got),
                                      exact_of<fftn::test::exact_complex>(fd, scale)),
               tol + tol, row + " against the full-length transform");
      const std::vector<std::size_t> bins = sample(h1);
      std::vector<c> picked;
      for (const std::size_t k : bins) {
        picked.push_back(got[k]);
      }
      std::vector<fftn::test::exact_complex> want = fftn::test::reference::r2c(xd, bins);
      for (fftn::test::exact_complex& w : want) {
        w = fftn::test::exact_from(std::complex<double>(w.re.hi, w.im.hi) * scale);
      }
      o.within(fftn::test::compare<T>(std::span<const c>(picked), want), tol,
               row + " against the reference");
      for (std::size_t k = h1; k < cdist && !in_place && b + 1 < count; ++k) {
        EXPECT_EQ(y[b * cdist + k], canary) << "gap element " << k << " of row " << b;
      }
    }
  } else {
    const fftn::shape<c, T> shape{.axes = {{.n = n}},
                                  .batches = {{.count = count, .in_dist = cd, .out_dist = rd}}};
    const auto core = make_plan_core(fftn::detail::view_of(shape), opts, rc.settings);
    text = fftn::detail::describe(*core);
    // The imaginary parts of bins 0 and n/2 are not zero: c2r ignores them.
    const std::vector<c> x = src.complexes<T>(cdist * count);
    const T canary(-7.25);
    std::vector<T> y(rdist * count, canary);
    std::vector<c> buffer = x;
    const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
    fftn::detail::execute_core(*core, 0, count, in_place ? buffer.data() : x.data(),
                               in_place ? static_cast<void*>(buffer.data()) : y.data(), ws);
    for (std::size_t b = 0; b < count; ++b) {
      std::vector<T> got(n);
      std::vector<c> hermitian(n);
      std::vector<std::complex<double>> xd(h1);
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t e = b * rdist + j;
        got[j] = in_place ? (j % 2 == 0 ? buffer[e / 2].real() : buffer[e / 2].imag()) : y[e];
      }
      for (std::size_t k = 0; k < h1; ++k) {
        xd[k] = std::complex<double>(x[b * cdist + k]);
        const c v = k == 0 || 2 * k == n ? c(x[b * cdist + k].real(), T(0)) : x[b * cdist + k];
        hermitian[k] = v;
        if (k != 0 && 2 * k != n) {
          hermitian[n - k] = std::conj(v);
        }
      }
      const std::vector<c> full = full_length<T>(target, +1, hermitian);
      std::vector<double> fd(n);
      for (std::size_t j = 0; j < n; ++j) {
        fd[j] = static_cast<double>(full[j].real());
      }
      const std::string row = what + ", row " + std::to_string(b);
      o.within(fftn::test::compare<T>(std::span<const T>(got),
                                      exact_of<fftn::test::exact_real>(fd, scale)),
               tol + tol, row + " against the full-length transform");
      const std::vector<std::size_t> idx = sample(n);
      std::vector<T> picked;
      for (const std::size_t j : idx) {
        picked.push_back(got[j]);
      }
      std::vector<fftn::test::exact_real> want = fftn::test::reference::c2r(n, xd, idx);
      for (fftn::test::exact_real& w : want) {
        w = fftn::test::exact_from(w.hi * scale);
      }
      o.within(fftn::test::compare<T>(std::span<const T>(picked), want), tol,
               row + " against the reference");
      for (std::size_t j = n; j < rdist && !in_place && b + 1 < count; ++j) {
        EXPECT_EQ(y[b * rdist + j], canary) << "gap element " << j << " of row " << b;
      }
    }
  }
  EXPECT_NE(text.find(rc.route), std::string::npos) << text;
  EXPECT_TRUE(o.passed()) << o.detail() << text;
}

template <class T> void check_all(fftn::isa target, const real_case& rc, bool r2c) {
  check<T>(target, rc, r2c, false, 0.5, 3);
  check<T>(target, rc, r2c, true, 1.0, 2);
}

build_settings forced(std::size_t fuse_radix, std::vector<std::size_t> radices = {}) {
  build_settings s;
  s.fuse_real_radix = fuse_radix;
  s.radices = std::move(radices);
  return s;
}

// The pairing fused into a pass of every codelet radix R: the r2c chain's last (radices
// ..., R) and the c2r chain's first (R, ...), with s (or m) = 6 (even: the copy s/2), 7
// (odd) and 64 (several registers of copies, a partial one on wide targets).
TEST(Real, FusedIntoEveryRadix) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (std::size_t r = 2; r <= 16; ++r) {
      for (const std::vector<std::size_t>& rest : {std::vector<std::size_t>{6},
                                                   std::vector<std::size_t>{7},
                                                   std::vector<std::size_t>{16, 4}}) {
        std::size_t h = r;
        for (const std::size_t f : rest) {
          h *= f;
        }
        std::vector<std::size_t> last = rest;
        last.push_back(r);
        std::vector<std::size_t> first{r};
        first.insert(first.end(), rest.rbegin(), rest.rend());
        check_all<double>(t, {2 * h, forced(16, last), "fused into its last pass"}, true);
        check_all<float>(t, {2 * h, forced(16, last), "fused into its last pass"}, true);
        check_all<double>(t, {2 * h, forced(16, first), "fused into its first pass"}, false);
        check_all<float>(t, {2 * h, forced(16, first), "fused into its first pass"}, false);
      }
    }
  }
}

// Every route the rules or the knob give, over short even lengths (leaves from h = 2), smooth
// composites, and lengths whose half has a prime factor above 16 (a Bluestein stage last:
// the r2c pairing then runs separately, the c2r one still in the first pass).
TEST(Real, EveryRoute) {
  const std::vector<std::size_t> lengths{2,  4,  6,  8,   10,  12,  14,  16,  18,  20,  22,
                                         24, 26, 28, 30,  32,  34,  36,  40,  48,  64,  96,
                                         100, 128, 222, 256, 384, 500, 514, 1000, 4096};
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths) {
      for (const bool r2c : {true, false}) {
        check_all<double>(t, {n, forced(0), r2c ? "r2c" : "c2r"}, r2c);
        check_all<float>(t, {n, forced(0), r2c ? "r2c" : "c2r"}, r2c);
        const char* separate = n == 2 ? "the pairing pass alone" : "pairing pass";
        check_all<double>(t, {n, forced(1), separate}, r2c);
        check_all<float>(t, {n, forced(1), separate}, r2c);
        if (n > 2) {
          const char* fused = n <= 32 ? "a leaf of" : "r";
          check_all<double>(t, {n, forced(16), fused}, r2c);
          check_all<float>(t, {n, forced(16), fused}, r2c);
        }
      }
    }
  }
}

// Long real rows: the half-length transform through a grid, the pairing after it (r2c, in
// place in the output) or before it (c2r, into the output, where the grid then runs in
// place); two and three phases.
TEST(Real, HalfLengthGrid) {
  struct grid_real {
    std::size_t n;
    std::vector<std::size_t> factors;
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const grid_real& g : {grid_real{512, {16, 16}}, grid_real{1024, {8, 64}},
                               grid_real{2048, {8, 8, 16}}, grid_real{1920, {24, 40}}}) {
      build_settings s;
      s.strategy = fftn::detail::strategy_choice::grid;
      s.grid_factors = g.factors;
      for (const bool r2c : {true, false}) {
        check_all<double>(t, {g.n, s, "half-length complex grid"}, r2c);
        check_all<float>(t, {g.n, s, "half-length complex grid"}, r2c);
      }
    }
  }
}

// The rules: a long row's half-length transform is a grid where a c2c row of that length
// would be one; a short row is a leaf.
TEST(Real, RulesChooseTheRoute) {
  const std::size_t big = std::size_t{1} << 20;
  EXPECT_NE(fftn::r2c_plan<double>(fftn::rows_r2c<double>(big, 1)).describe().find(
                "half-length complex grid"),
            std::string::npos);
  EXPECT_NE(fftn::c2r_plan<double>(fftn::rows_c2r<double>(big, 1)).describe().find(
                "half-length complex grid"),
            std::string::npos);
  EXPECT_NE(fftn::c2r_plan<double>(fftn::rows_c2r<double>(16, 1)).describe().find("a leaf of 8"),
            std::string::npos);
}

} // namespace
