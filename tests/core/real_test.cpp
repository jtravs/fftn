// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The routes of real transforms (DESIGN.md 9.1, src/passes/real.inl, task 2.1), each forced
// on every instruction set the machine runs, in double and float: the pairing fused into
// the last pass (r2c) or the first (c2r) with every codelet radix in that pass and copy
// counts odd, even and spanning several registers; the pairing as a separate pass (after a
// chain, a Bluestein stage or a grid); the leaves; the rules' choice. Out of place with gaps
// between rows (canaries) and in place in FFTW's padded rows, batches, and a scale; c2r's
// input preserved byte for byte out of place; planes and volumes; and a negative control
// showing that the every-bin oracle catches a planted pairing fault.
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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
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
  const auto core = make_plan_core(fftn::detail::view_of(fftn::rows_c2c<T>(x.size(), 1, dir)),
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

// An r2c row's h1 = n/2 + 1 bins against the full-length complex transform of its n reals,
// at every bin (tolerance of two transforms).
template <class T>
void r2c_every_bin(fftn::isa target, const std::vector<T>& x,
                   const std::vector<std::complex<T>>& got, double scale, const std::string& what,
                   fftn::test::outcome& o) {
  using c = std::complex<T>;
  std::vector<c> xc(x.size());
  for (std::size_t j = 0; j < x.size(); ++j) {
    xc[j] = c(x[j], T(0));
  }
  std::vector<c> full = full_length<T>(target, -1, xc);
  full.resize(got.size());
  const std::vector<std::complex<double>> fd(full.begin(), full.end());
  const auto tol = tolerance_of<T>(x.size());
  o.within(fftn::test::compare<T>(std::span<const c>(got),
                                  exact_of<fftn::test::exact_complex>(fd, scale)),
           tol + tol, what + " against the full-length transform");
}

// A c2r row's n reals against the full-length complex transform of the Hermitian spectrum
// its h1 bins describe (the imaginary parts of bins 0 and n/2 ignored), at every output.
template <class T>
void c2r_every_bin(fftn::isa target, const std::vector<std::complex<T>>& x,
                   const std::vector<T>& got, double scale, const std::string& what,
                   fftn::test::outcome& o) {
  using c = std::complex<T>;
  const std::size_t n = got.size();
  std::vector<c> hermitian(n);
  for (std::size_t k = 0; k < n / 2 + 1; ++k) {
    const bool real = k == 0 || 2 * k == n;
    hermitian[k] = real ? c(x[k].real(), T(0)) : x[k];
    if (!real) {
      hermitian[n - k] = std::conj(x[k]);
    }
  }
  const std::vector<c> full = full_length<T>(target, +1, hermitian);
  std::vector<double> fd(n);
  for (std::size_t j = 0; j < n; ++j) {
    fd[j] = static_cast<double>(full[j].real());
  }
  const auto tol = tolerance_of<T>(n);
  o.within(
      fftn::test::compare<T>(std::span<const T>(got), exact_of<fftn::test::exact_real>(fd, scale)),
      tol + tol, what + " against the full-length transform");
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
  const std::string what =
      std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
      (r2c ? " r2c" : " c2r") + " n = " + std::to_string(n) + (in_place ? ", in place" : "") +
      ", scale " + std::to_string(scale) + ", " + std::to_string(count) + " rows";
  SCOPED_TRACE(what);
  const fftn::options opts{.in_place = in_place, .scale = scale, .force_isa = target};
  const auto rd = static_cast<std::ptrdiff_t>(rdist);
  const auto cd = static_cast<std::ptrdiff_t>(cdist);
  fftn::test::input_source src(0x2E41ULL + n * 7 + count + (r2c ? 1 : 0));
  const auto tol = tolerance_of<T>(n);
  fftn::test::outcome o;
  std::string text;
  if (r2c) {
    const fftn::shape<T, c> shape{
        .axes = {{.n = n}},
        .batches = {{.count = count, .in_dist = rd, .out_dist = cd}},
    };
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
      std::vector<T> xr(x.begin() + static_cast<std::ptrdiff_t>(b * rdist),
                        x.begin() + static_cast<std::ptrdiff_t>(b * rdist + n));
      const std::vector<double> xd(xr.begin(), xr.end());
      for (std::size_t k = 0; k < h1; ++k) {
        got[k] = in_place ? c(buffer[b * rdist + 2 * k], buffer[b * rdist + 2 * k + 1])
                          : y[b * cdist + k];
      }
      const std::string row = what + ", row " + std::to_string(b);
      r2c_every_bin<T>(target, xr, got, scale, row, o);
      const std::vector<std::size_t> bins = sample(h1);
      std::vector<c> picked;
      picked.reserve(bins.size());
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
    const fftn::shape<c, T> shape{
        .axes = {{.n = n}},
        .batches = {{.count = count, .in_dist = cd, .out_dist = rd}},
    };
    const auto core = make_plan_core(fftn::detail::view_of(shape), opts, rc.settings);
    text = fftn::detail::describe(*core);
    // The imaginary parts of bins 0 and n/2 are not zero: c2r ignores them.
    std::vector<c> x = src.complexes<T>(cdist * count);
    const std::vector<c> before = x;
    const T canary(-7.25);
    std::vector<T> y(rdist * count, canary);
    std::vector<c> buffer = x;
    const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
    fftn::detail::execute_core(*core, 0, count, in_place ? buffer.data() : x.data(),
                               in_place ? static_cast<void*>(buffer.data()) : y.data(), ws);
    if (!in_place) {
      // c2r preserves its input (DESIGN.md 9.1), the gaps between rows included.
      EXPECT_EQ(std::memcmp(x.data(), before.data(), x.size() * sizeof(c)), 0)
          << "the input was written";
    }
    for (std::size_t b = 0; b < count; ++b) {
      std::vector<T> got(n);
      const std::vector<c> xrow(x.begin() + static_cast<std::ptrdiff_t>(b * cdist),
                                x.begin() + static_cast<std::ptrdiff_t>(b * cdist + h1));
      const std::vector<std::complex<double>> xd(xrow.begin(), xrow.end());
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t e = b * rdist + j;
        if (in_place) {
          got[j] = j % 2 == 0 ? buffer[e / 2].real() : buffer[e / 2].imag();
        } else {
          got[j] = y[e];
        }
      }
      const std::string row = what + ", row " + std::to_string(b);
      c2r_every_bin<T>(target, xrow, got, scale, row, o);
      const std::vector<std::size_t> idx = sample(n);
      std::vector<T> picked;
      picked.reserve(idx.size());
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
// (odd) and 64 (several registers of copies, a partial one on wide targets). The r2c rows
// force interleaved passes: where the chain has a split run (x86 by default) an r2c row
// takes the separate route instead.
TEST(Real, FusedIntoEveryRadix) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (std::size_t r = 2; r <= 16; ++r) {
      for (const std::vector<std::size_t>& rest : {
               std::vector<std::size_t>{6},
               std::vector<std::size_t>{7},
               std::vector<std::size_t>{16, 4},
           }) {
        std::size_t h = r;
        for (const std::size_t f : rest) {
          h *= f;
        }
        std::vector<std::size_t> last = rest;
        last.push_back(r);
        std::vector<std::size_t> first{r};
        first.insert(first.end(), rest.rbegin(), rest.rend());
        build_settings r2c = forced(16, last);
        r2c.layout = fftn::detail::layout_choice::interleaved;
        check_all<double>(t, {.n = 2 * h, .settings = r2c, .route = "fused into its last pass"},
                          true);
        check_all<float>(t, {.n = 2 * h, .settings = r2c, .route = "fused into its last pass"},
                         true);
        check_all<double>(
            t, {.n = 2 * h, .settings = forced(16, first), .route = "fused into its first pass"},
            false);
        check_all<float>(
            t, {.n = 2 * h, .settings = forced(16, first), .route = "fused into its first pass"},
            false);
      }
    }
  }
}

// Every route the rules or the knob give, over short even lengths (leaves from h = 2), smooth
// composites, and lengths whose half has a prime factor above 16: a codelet (17), a
// Rader stage (37, 257) or a Bluestein stage (47, 89) last, where the r2c pairing then runs
// separately and the c2r one still in the first pass.
TEST(Real, EveryRoute) {
  const std::vector<std::size_t> lengths{
      2,  4,  6,  8,  10, 12, 14,  16,  18,  20,  22,  24,  26,  28,  30,   32,   34,
      36, 40, 48, 64, 94, 96, 100, 128, 178, 222, 256, 384, 500, 514, 1000, 4096,
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : lengths) {
      for (const bool r2c : {true, false}) {
        check_all<double>(t, {.n = n, .settings = forced(0), .route = r2c ? "r2c" : "c2r"}, r2c);
        check_all<float>(t, {.n = n, .settings = forced(0), .route = r2c ? "r2c" : "c2r"}, r2c);
        const char* separate = n == 2 ? "the pairing pass alone" : "pairing pass";
        check_all<double>(t, {.n = n, .settings = forced(1), .route = separate}, r2c);
        check_all<float>(t, {.n = n, .settings = forced(1), .route = separate}, r2c);
        if (n > 2) {
          const char* fused = n <= 32 ? "a leaf of" : "r";
          check_all<double>(t, {.n = n, .settings = forced(16), .route = fused}, r2c);
          check_all<float>(t, {.n = n, .settings = forced(16), .route = fused}, r2c);
        }
      }
    }
  }
}

// Long real rows: the half-length transform through a grid, the pairing after it (r2c, in
// place in the output) or before it: out of place in two phases in phase 1's gather (c2r,
// from the input, which stays intact), otherwise as a pass into the output, where the grid
// then runs in place; two and three phases.
TEST(Real, HalfLengthGrid) {
  struct grid_real {
    std::size_t n;
    std::vector<std::size_t> factors;
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const grid_real& g : {
             grid_real{.n = 512, .factors = {16, 16}},
             grid_real{.n = 1024, .factors = {8, 64}},
             grid_real{.n = 2048, .factors = {8, 8, 16}},
             grid_real{.n = 1920, .factors = {24, 40}},
             grid_real{.n = 65536, .factors = {128, 256}},
         }) {
      // With and without the padded intermediate (X26b), which the c2r rows whose phase 1
      // gather pairs the bins must not take (they write phase 1 into the output, X29).
      for (const std::size_t period :
           {std::numeric_limits<std::size_t>::max(), std::size_t{4096}}) {
        build_settings s;
        s.strategy = fftn::detail::strategy_choice::grid;
        s.grid_factors = g.factors;
        s.profile.grid_pad_period = period;
        check_all<double>(t, {.n = g.n, .settings = s, .route = "half-length complex grid"}, true);
        check_all<float>(t, {.n = g.n, .settings = s, .route = "half-length complex grid"}, true);
        // c2r with the pairing in a pass and, out of place in two phases, in the gather.
        for (const bool paired : {false, true}) {
          s.profile.c2r_paired_gather = paired;
          check_all<double>(t, {.n = g.n, .settings = s, .route = "half-length complex grid"},
                            false);
          check_all<float>(t, {.n = g.n, .settings = s, .route = "half-length complex grid"},
                           false);
        }
      }
    }
  }
}

// Real rows whose half-length transform is a depth-first chain (experiment X72), out of place:
// r2c with the pairing pass after the chain in the output, c2r with the pairing into the
// workspace block before it; one and two top factors, a sub-row of a power of two and of a
// mixed length, scaled and not, batches with gaps (check). In place the row keeps its route.
TEST(Real, HalfLengthDepthFirst) {
  struct depth_real {
    std::size_t n;
    std::vector<std::size_t> tops;
  };
  const char* route = "half-length complex depth-first";
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const depth_real& d : {
             depth_real{.n = 4096, .tops = {4}},
             depth_real{.n = 12288, .tops = {4}},
             depth_real{.n = 10240, .tops = {8, 4}},
             depth_real{.n = 3840, .tops = {16}},
         }) {
      build_settings s;
      s.strategy = fftn::detail::strategy_choice::depth;
      s.depth_tops = d.tops;
      for (const bool r2c : {true, false}) {
        check<double>(t, {.n = d.n, .settings = s, .route = route}, r2c, false, 0.5, 3);
        check<double>(t, {.n = d.n, .settings = s, .route = route}, r2c, false, 1.0, 1);
        check<float>(t, {.n = d.n, .settings = s, .route = route}, r2c, false, 1.0, 2);
        check<double>(t, {.n = d.n, .settings = s, .route = "half-length complex"}, r2c, true, 1.0,
                      2);
      }
    }
  }
}

// The rules take the depth-first half-length chain only where the profile's
// depth_real_max_bytes allows (experiment X72): a profile with the EPYC 7763's depth-first
// fields (set here, not read from the host), with and without the limit.
TEST(Real, RulesTakeTheDepthFirstHalfLengthWhereTheProfileSays) {
  build_settings s;
  s.profile.depth_min_bytes = std::size_t{512} * 1024 / 4;
  s.profile.depth_max_bytes = std::size_t{4} * 512 * 1024;
  s.profile.depth_pow2_max_bytes = 0;
  s.profile.depth_part_bytes = std::size_t{32} * 1024 / 4;
  s.profile.depth_top = 4;
  s.profile.grid_min_bytes = std::size_t{512} * 1024 / 2;
  // A top of 4 is taken on avx2 and on the 32-register targets; one of them runs anywhere the
  // suites run (the scalar target takes no depth-first chain by the rules).
  const std::vector<fftn::isa> runnable = fftn::test::runnable_isas();
  const auto with = std::ranges::find_if(runnable, [](fftn::isa t) {
    return t == fftn::isa::avx2 || t == fftn::isa::neon || t == fftn::isa::neon_fcma;
  });
  if (with == runnable.end()) {
    GTEST_SKIP() << "no target here takes a depth-first top of 4";
  }
  const auto describe_r2c = [&](std::size_t n, bool in_place) {
    const fftn::options opts{.in_place = in_place, .force_isa = *with};
    const auto core = make_plan_core(fftn::detail::view_of(fftn::rows_r2c<double>(n, 1)), opts, s);
    return fftn::detail::describe(*core);
  };
  // Without the limit: the chain (the half length 9216 is within the rows' range).
  EXPECT_EQ(describe_r2c(18432, false).find("depth-first"), std::string::npos);
  s.profile.depth_real_max_bytes = std::size_t{4} * 512 * 1024;
  EXPECT_NE(describe_r2c(18432, false).find("half-length complex depth-first 4 x 2304"),
            std::string::npos)
      << describe_r2c(18432, false);
  // Below the rows' range, a power of two and in place: no depth-first chain.
  EXPECT_EQ(describe_r2c(6144, false).find("depth-first"), std::string::npos);
  EXPECT_EQ(describe_r2c(32768, false).find("depth-first"), std::string::npos);
  EXPECT_EQ(describe_r2c(18432, true).find("depth-first"), std::string::npos);
  // Beyond depth_real_max_bytes: none.
  s.profile.depth_real_max_bytes = std::size_t{9216} * 16 - 1;
  EXPECT_EQ(describe_r2c(18432, false).find("depth-first"), std::string::npos);
}

// The r2c pairing in the grid's last phase (X33): pairs of mirror column sets of
// hb = block / 2 columns (at least a register and a cache line), column 0 on its own; the last
// factor even and odd (the middle bin in column 0, or in column L/2 paired with itself), two and
// three phases, in and out of place, scaled and not (check_all); L of 24 splits into pairs of sets
// of 4 (double; float's sets of 4 are narrower than a line and keep a pass after the grid).
TEST(Real, PairingInTheGridsLastPhase) {
  struct grid_real {
    std::size_t n;
    std::vector<std::size_t> factors;
    std::size_t block;
    const char* route;
  };
  const char* fused = "the pairing in its last phase";
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const grid_real& g : {
             grid_real{.n = 1024, .factors = {32, 16}, .block = 16, .route = fused},
             grid_real{.n = 4096, .factors = {16, 8, 16}, .block = 16, .route = fused},
             grid_real{.n = 7680, .factors = {16, 16, 15}, .block = 16, .route = fused},
             grid_real{.n = 8192, .factors = {32, 128}, .block = 16, .route = fused},
             grid_real{
                 .n = 1920,
                 .factors = {24, 40},
                 .block = 8,
                 .route = "half-length complex grid",
             },
         }) {
      build_settings s;
      s.strategy = fftn::detail::strategy_choice::grid;
      s.grid_factors = g.factors;
      s.profile.grid_block = g.block;
      s.grid_fused = false; // a fused last phase (radices <= 8) keeps the separate pass
      s.r2c_pair_last = 1;  // every target (the rule leaves avx2 out)
      check_all<double>(t, {.n = g.n, .settings = s, .route = g.route}, true);
      check_all<float>(t, {.n = g.n, .settings = s, .route = g.route}, true);
    }
  }
}

// The negative control of the every-bin comparison: a correct result with a pairing fault
// planted where the reference sample does not look (bins k and h - k swapped, as a lane
// reversal gone wrong would leave them; one bin off by a few hundred times the tolerance)
// must fail it.
TEST(Real, EveryBinComparisonCatchesABrokenPairing) {
  using c = std::complex<double>;
  const std::size_t n = 1000;
  const std::size_t h1 = n / 2 + 1;
  const fftn::isa target = fftn::isa::scalar;
  fftn::test::input_source src(0xBAD5ULL);
  const std::vector<double> x = src.reals<double>(n);
  std::vector<c> good(h1);
  fftn::workspace ws(std::size_t{1} << 20U);
  fftn::r2c_plan<double>(fftn::rows_r2c<double>(n, 1), {.force_isa = target})
      .execute(x.data(), good.data(), ws);
  {
    fftn::test::outcome o;
    r2c_every_bin<double>(target, x, good, 1.0, "intact", o);
    EXPECT_TRUE(o.passed()) << o.detail();
  }
  const std::vector<std::size_t> sampled = sample(h1);
  const std::size_t k = 7; // not sampled for h1 = 501
  ASSERT_EQ(std::count(sampled.begin(), sampled.end(), k), 0);
  ASSERT_EQ(std::count(sampled.begin(), sampled.end(), n / 2 - k), 0);
  std::vector<c> swapped = good;
  std::swap(swapped[k], swapped[n / 2 - k]);
  fftn::test::outcome o1;
  r2c_every_bin<double>(target, x, swapped, 1.0, "bins k and h - k swapped", o1);
  EXPECT_FALSE(o1.passed());
  std::vector<c> nudged = good;
  double largest = 0;
  for (const c& v : good) {
    largest = std::max(largest, std::abs(v));
  }
  const double tol_max = tolerance_of<double>(n).max;
  nudged[k] += c(400 * tol_max * largest, 0);
  fftn::test::outcome o2;
  r2c_every_bin<double>(target, x, nudged, 1.0, "one bin off", o2);
  EXPECT_FALSE(o2.passed());

  // And c2r: one output sample off.
  const std::vector<c> spectrum = src.complexes<double>(h1);
  std::vector<double> back(n);
  fftn::c2r_plan<double>(fftn::rows_c2r<double>(n, 1), {.force_isa = target})
      .execute(spectrum.data(), back.data(), ws);
  fftn::test::outcome o3;
  c2r_every_bin<double>(target, spectrum, back, 1.0, "c2r intact", o3);
  EXPECT_TRUE(o3.passed()) << o3.detail();
  double peak = 0;
  for (const double v : back) {
    peak = std::max(peak, std::abs(v));
  }
  back[k] += 400 * tol_max * peak;
  fftn::test::outcome o4;
  c2r_every_bin<double>(target, spectrum, back, 1.0, "c2r one output off", o4);
  EXPECT_FALSE(o4.passed());
}

// Multi-dimensional real transforms on the fused and separate routes and the rules': the
// real axis last (with a Bluestein factor in {5, 68} and {2, 4100}), out of place, a batch
// of two. r2c against the multi-dimensional c2c of the reals; c2r on that spectrum against
// N times the reals (a round trip, two transforms' error), its input preserved.
template <class T>
void check_volume(fftn::isa target, const std::vector<std::size_t>& dims,
                  const build_settings& st) {
  using c = std::complex<T>;
  const std::string what = std::string(fftn::isa_name(target)) +
                           (sizeof(T) == 8 ? " double" : " float") + " dims " +
                           std::to_string(dims.size()) + " last " + std::to_string(dims.back()) +
                           " fuse " + std::to_string(st.fuse_real_radix);
  SCOPED_TRACE(what);
  const std::size_t batch = 2;
  std::size_t total = 1;
  std::vector<fftn::axis> axes;
  for (const std::size_t d : dims) {
    total *= d;
    axes.push_back({.n = d});
  }
  const std::size_t outer = total / dims.back();
  const std::size_t h1 = dims.back() / 2 + 1;
  const std::size_t half = outer * h1;
  const fftn::options opts{.force_isa = target};
  fftn::test::input_source src(0x30DULL + total);
  const std::vector<T> x = src.reals<T>(total * batch);

  const fftn::shape<T, c> r2c_shape{.axes = axes, .batches = {{.count = batch}}};
  const auto r2c = make_plan_core(fftn::detail::view_of(r2c_shape), opts, st);
  std::vector<c> spectrum(half * batch);
  const fftn::workspace ws(std::size_t{1} << 22U);
  fftn::detail::execute_core(*r2c, 0, batch, x.data(), spectrum.data(), ws);

  build_settings chain;
  chain.strategy = fftn::detail::strategy_choice::chain;
  const fftn::shape<c, c> c2c_shape{
      .dir = fftn::direction::forward,
      .axes = axes,
      .batches = {{.count = batch}},
  };
  const auto c2c = make_plan_core(fftn::detail::view_of(c2c_shape), opts, chain);
  const std::vector<c> xc(x.begin(), x.end());
  std::vector<c> full(total * batch);
  fftn::detail::execute_core(*c2c, 0, batch, xc.data(), full.data(), ws);

  const auto tol = tolerance_of<T>(total);
  fftn::test::outcome o;
  for (std::size_t b = 0; b < batch; ++b) {
    std::vector<c> got(half);
    std::vector<std::complex<double>> want(half);
    const std::vector<double> reals(total);
    for (std::size_t r = 0; r < outer; ++r) {
      for (std::size_t k = 0; k < h1; ++k) {
        got[r * h1 + k] = spectrum[b * half + r * h1 + k];
        want[r * h1 + k] = std::complex<double>(full[b * total + r * dims.back() + k]);
      }
    }
    o.within(fftn::test::compare<T>(std::span<const c>(got),
                                    exact_of<fftn::test::exact_complex>(want, 1.0)),
             tol + tol, what + " r2c, transform " + std::to_string(b));
  }

  const fftn::shape<c, T> c2r_shape{.axes = axes, .batches = {{.count = batch}}};
  const auto c2r = make_plan_core(fftn::detail::view_of(c2r_shape), opts, st);
  const std::vector<c> before = spectrum;
  std::vector<T> back(total * batch);
  fftn::detail::execute_core(*c2r, 0, batch, spectrum.data(), back.data(), ws);
  EXPECT_EQ(std::memcmp(spectrum.data(), before.data(), spectrum.size() * sizeof(c)), 0)
      << "the c2r input was written";
  for (std::size_t b = 0; b < batch; ++b) {
    std::vector<double> want(total);
    for (std::size_t j = 0; j < total; ++j) {
      want[j] = static_cast<double>(x[b * total + j]) * static_cast<double>(total);
    }
    o.within(fftn::test::compare<T>(std::span<const T>(back.data() + b * total, total),
                                    exact_of<fftn::test::exact_real>(want, 1.0)),
             tol + tol, what + " c2r round trip, transform " + std::to_string(b));
  }
  EXPECT_TRUE(o.passed()) << o.detail() << fftn::detail::describe(*r2c)
                          << fftn::detail::describe(*c2r);
}

TEST(Real, PlanesAndVolumes) {
  const std::vector<std::vector<std::size_t>> shapes{{6, 128}, {3, 5, 200}, {5, 68}, {2, 4100}};
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const auto& dims : shapes) {
      for (const std::size_t fuse : {std::size_t{0}, std::size_t{1}, std::size_t{16}}) {
        check_volume<double>(t, dims, forced(fuse));
        check_volume<float>(t, dims, forced(fuse));
      }
    }
  }
}

// The rules: a long row's half-length transform is a grid where a c2c row of that length
// would be one (with a profile fixed here: the machine's own may keep 2^19 points a chain,
// X29); a short row is a leaf.
TEST(Real, RulesChooseTheRoute) {
  const std::size_t big = std::size_t{1} << 20U;
  build_settings s;
  s.profile.grid_min_bytes = std::size_t{512} * 1024;
  s.profile.pow2_chain_bytes = 0; // the profile's power-of-two chains would keep it a chain
  for (const bool r2c : {true, false}) {
    const auto core = r2c ? make_plan_core(fftn::detail::view_of(fftn::rows_r2c<double>(big, 1)),
                                           fftn::options(), s)
                          : make_plan_core(fftn::detail::view_of(fftn::rows_c2r<double>(big, 1)),
                                           fftn::options(), s);
    EXPECT_NE(fftn::detail::describe(*core).find("half-length complex grid"), std::string::npos);
  }
  EXPECT_NE(fftn::c2r_plan<double>(fftn::rows_c2r<double>(16, 1)).describe().find("a leaf of 8"),
            std::string::npos);
}

} // namespace
