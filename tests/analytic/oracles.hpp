// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The analytic oracles (DESIGN.md 15.1, oracle 1): identities a correct transform must
// satisfy, checked against closed forms or against the subject's own other results.
//
// Expected values that are closed forms use roots of unity from the quadruple-precision
// reference, so they are exact to far below the tolerance. Where a closed-form input has to
// be rounded to T (tones, single bins), or an expected value is formed in double from
// computed ones, the tolerance gains input_rounding<T>() for it. A check that compares two
// computed transforms carries the tolerance of each (tolerance.hpp). Every run also checks
// the memory contracts (harness.hpp): input preserved, nothing outside the rows written.

#pragma once

#include "reference/reference_dft.hpp"
#include "support/checks.hpp"
#include "support/inputs.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace fftn::test::analytic {

namespace detail {

template <class T> using cvec = std::vector<std::complex<T>>;

constexpr std::array directions{direction::forward, direction::backward};

inline std::uint64_t seed(std::size_t n, std::uint64_t salt) {
  return 0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(n) + 1) + salt;
}

template <class T> cvec<T> rounded_root_row(int sign, std::size_t k0, std::size_t n) {
  const auto& w = reference::unit_roots(sign, n);
  cvec<T> x(n);
  for (std::size_t j = 0; j < n; ++j) {
    x[j] = round_to<T>(w[(k0 * j) % n]);
  }
  return x;
}

inline std::complex<double> hi(const exact_complex& v) { return {v.re.hi, v.im.hi}; }

// Expected values in double: y[k] * w^(k * s), the shift theorem's right-hand side.
template <class T>
std::vector<exact_complex> shifted(std::span<const std::complex<T>> y, int sign, std::size_t s) {
  const std::size_t n = y.size();
  const auto& w = reference::unit_roots(sign, n);
  std::vector<exact_complex> e;
  e.reserve(n);
  for (std::size_t k = 0; k < n; ++k) {
    e.push_back(exact_from(std::complex<double>(y[k]) * hi(w[(k * s) % n])));
  }
  return e;
}

} // namespace detail

// ---------------------------------------------------------------- c2c rows

template <class T> outcome c2c_impulses(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n);
  for (const direction d : detail::directions) {
    const auto& w = reference::unit_roots(sign_of(d), n);
    for (const std::size_t at : probe_positions(n)) {
      detail::cvec<T> x(n);
      x[at] = T(1);
      const auto what = describe("impulse at ", at, ", ", name_of(d));
      const auto y = c2c_once<T>(o, s, d, x, what);
      std::vector<exact_complex> want(n);
      for (std::size_t k = 0; k < n; ++k) {
        want[k] = w[(k * at) % n];
      }
      o.within(compare<T>(y, want), tol, what);
    }
  }
  return o;
}

template <class T> outcome c2c_single_bins(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) + input_rounding<T>();
  for (const direction d : detail::directions) {
    for (const std::size_t k0 : probe_positions(n)) {
      // x_j = exp(-sign 2 pi i k0 j / n) has the transform n at bin k0 and 0 elsewhere.
      const auto x = detail::rounded_root_row<T>(-sign_of(d), k0, n);
      const auto what = describe("single bin ", k0, ", ", name_of(d));
      const auto y = c2c_once<T>(o, s, d, x, what);
      std::vector<exact_complex> want(n);
      want[k0] = exact_from(std::complex<double>(static_cast<double>(n), 0.0));
      o.within(compare<T>(y, want), tol, what);
    }
  }
  return o;
}

template <class T> outcome c2c_real_tones(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) + input_rounding<T>();
  const auto& w = reference::unit_roots(+1, n);
  for (const direction d : detail::directions) {
    for (const std::size_t k0 : probe_positions(n)) {
      // cos(2 pi k0 j / n) has n/2 at bins k0 and n - k0 (n when they coincide).
      detail::cvec<T> x(n);
      for (std::size_t j = 0; j < n; ++j) {
        x[j] = round_to<T>(w[(k0 * j) % n].re);
      }
      const auto what = describe("cosine at bin ", k0, ", ", name_of(d));
      const auto y = c2c_once<T>(o, s, d, x, what);
      std::vector<exact_complex> want(n);
      want[k0].re.hi += static_cast<double>(n) / 2;
      want[(n - k0) % n].re.hi += static_cast<double>(n) / 2;
      o.within(compare<T>(y, want), tol, what);
    }
  }
  return o;
}

template <class T> outcome c2c_constant(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n);
  const std::complex<T> c(T(0.5), T(-0.25));
  for (const direction d : detail::directions) {
    const detail::cvec<T> x(n, c);
    const auto what = describe("constant, ", name_of(d));
    const auto y = c2c_once<T>(o, s, d, x, what);
    std::vector<exact_complex> want(n);
    want[0] = exact_from(std::complex<double>(c) * static_cast<double>(n));
    o.within(compare<T>(y, want), tol, what);
  }
  return o;
}

// sum |X_k|^2 = n sum |x_j|^2. With X computed as X + E, the left side changes by
// 2 Re<X, E> + |E|^2, at most (2 rms + rms^2) of it: the tolerance is twice the RMS one.
template <class T> outcome c2c_parseval(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n);
  input_source src(detail::seed(n, 1));
  const auto x = src.complexes<T>(n);
  for (const direction d : detail::directions) {
    const auto what = describe("Parseval, ", name_of(d));
    const auto y = c2c_once<T>(o, s, d, x, what);
    compensated_sum ex;
    compensated_sum ey;
    for (std::size_t j = 0; j < n; ++j) {
      ex.add(std::norm(std::complex<double>(x[j])));
      ey.add(std::norm(std::complex<double>(y[j])));
    }
    const double want = static_cast<double>(n) * ex.value();
    const double rel = std::abs(ey.value() - want) / want;
    const double limit = 2 * tol.rms + input_rounding<T>().rms;
    o.within({.rms = rel, .max = 0.0}, {.rms = limit, .max = 1.0}, what);
  }
  return o;
}

template <class T> outcome c2c_shift(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) * 2 + input_rounding<T>();
  input_source src(detail::seed(n, 2));
  const auto x = src.complexes<T>(n);
  for (const direction d : detail::directions) {
    const auto y = c2c_once<T>(o, s, d, x, describe("shift: unshifted, ", name_of(d)));
    for (const std::size_t shift : {std::size_t{1}, n / 2, n - 1}) {
      if (shift == 0 || shift >= n) {
        continue;
      }
      // x'_j = x_{j - shift} has X'_k = w^(k shift) X_k.
      detail::cvec<T> xs(n);
      for (std::size_t j = 0; j < n; ++j) {
        xs[(j + shift) % n] = x[j];
      }
      const auto what = describe("shift by ", shift, ", ", name_of(d));
      const auto ys = c2c_once<T>(o, s, d, xs, what);
      o.within(compare<T>(ys, detail::shifted<T>(y, sign_of(d), shift)), tol, what);
    }
  }
  return o;
}

// F(a x + b y) = a F(x) + b F(y). Inputs are integers up to 2^10 and a, b have three
// fractional bits, so a x + b y is exact in float and double. The three transforms' errors
// add: at most (1 + (|a| + |b|) / |(a, b)|) <= 1 + sqrt 2 times one tolerance; 3 is used.
template <class T> outcome c2c_linearity(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) * 3 + input_rounding<T>();
  input_source src(detail::seed(n, 3));
  const auto x = src.integer_complexes<T>(n, 1024);
  const auto y = src.integer_complexes<T>(n, 1024);
  const std::complex<T> a(T(0.75), T(-0.5));
  const std::complex<T> b(T(-1.25), T(0.375));
  detail::cvec<T> z(n);
  for (std::size_t j = 0; j < n; ++j) {
    z[j] = a * x[j] + b * y[j];
  }
  for (const direction d : detail::directions) {
    const auto what = describe("linearity, ", name_of(d));
    const auto fx = c2c_once<T>(o, s, d, x, what);
    const auto fy = c2c_once<T>(o, s, d, y, what);
    const auto fz = c2c_once<T>(o, s, d, z, what);
    std::vector<std::complex<double>> want(n);
    for (std::size_t k = 0; k < n; ++k) {
      want[k] = std::complex<double>(a) * std::complex<double>(fx[k]) +
                std::complex<double>(b) * std::complex<double>(fy[k]);
    }
    o.within(compare<T>(fz, as_expected(want)), tol, what);
  }
  return o;
}

template <class T> outcome c2c_round_trip(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) * 2 + input_rounding<T>();
  input_source src(detail::seed(n, 4));
  const auto x = src.complexes<T>(n);
  const auto y = c2c_once<T>(o, s, direction::forward, x, "round trip: forward");
  const auto back = c2c_once<T>(o, s, direction::backward, y, "round trip: backward");
  std::vector<std::complex<double>> got(n);
  for (std::size_t j = 0; j < n; ++j) {
    got[j] = std::complex<double>(back[j]) / static_cast<double>(n);
  }
  o.within(compare<double>(got, as_expected<T>(x)), tol, "backward(forward(x)) / n");
  return o;
}

// The transform of a real input is conjugate-symmetric: X_{n-k} = conj(X_k).
template <class T> outcome c2c_real_input_symmetry(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) * 2;
  input_source src(detail::seed(n, 5));
  const auto re = src.reals<T>(n);
  const detail::cvec<T> x(re.begin(), re.end());
  for (const direction d : detail::directions) {
    const auto what = describe("real input symmetry, ", name_of(d));
    const auto y = c2c_once<T>(o, s, d, x, what);
    detail::cvec<T> mirror(n);
    for (std::size_t k = 0; k < n; ++k) {
      mirror[k] = std::conj(y[(n - k) % n]);
    }
    o.within(compare<T>(y, as_expected<T>(mirror)), tol, what);
  }
  return o;
}

// Batches with gaps, odd distances, negative strides and unaligned element 0, against the
// same rows transformed one at a time. The harness checks that the gaps stay untouched.
template <class T> outcome c2c_batched_strided(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) * 2;
  constexpr std::size_t count = 3;
  input_source src(detail::seed(n, 6));
  const auto x = src.complexes<T>(n * count);
  const auto in = static_cast<std::ptrdiff_t>(n);
  const std::array layouts{
      rows{
          .n = n,
          .count = count,
          .in_stride = 2,
          .in_dist = 2 * in + 3,
          .out_stride = 3,
          .out_dist = 3 * in + 1,
      },
      rows{
          .n = n,
          .count = count,
          .in_stride = -1,
          .in_dist = -(in + 2),
          .out_stride = 1,
          .out_dist = in + 1,
      },
  };
  const std::array options{
      run_options{.in_offset = sizeof(T), .out_offset = 3 * sizeof(T)},
      run_options{.in_offset = 2 * sizeof(T), .out_offset = sizeof(T), .guard = guard_side::below},
  };
  for (const direction d : detail::directions) {
    std::vector<std::complex<T>> single;
    for (std::size_t r = 0; r < count; ++r) {
      const auto y = c2c_once<T>(o, s, d, std::span(x).subspan(r * n, n), "single row");
      single.insert(single.end(), y.begin(), y.end());
    }
    for (std::size_t l = 0; l < layouts.size(); ++l) {
      const auto what = describe("batched layout ", l, ", ", name_of(d));
      const auto r = run_c2c<T>(s, d, layouts[l], x, options[l]);
      note_violations(o, r, what);
      o.within(compare<T>(r.out, as_expected<T>(single)), tol, what);
    }
  }
  return o;
}

template <class T> outcome c2c_in_place(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) * 2;
  constexpr std::size_t count = 2;
  input_source src(detail::seed(n, 7));
  const auto x = src.complexes<T>(n * count);
  for (const direction d : detail::directions) {
    const auto what = describe("in place, ", name_of(d));
    const auto oop = run_c2c<T>(s, d, tight_rows(kind::c2c, n, count), x);
    const auto ip = run_c2c<T>(s, d, tight_rows(kind::c2c, n, count, placement::in_place), x);
    note_violations(o, oop, what);
    note_violations(o, ip, what);
    o.within(compare<T>(ip.out, as_expected<T>(oop.out)), tol, what);
  }
  return o;
}

// ---------------------------------------------------------------- r2c rows

template <class T> outcome r2c_impulses(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n);
  const auto& w = reference::unit_roots(-1, n);
  for (const std::size_t at : probe_positions(n)) {
    std::vector<T> x(n);
    x[at] = T(1);
    const auto what = describe("impulse at ", at);
    const auto y = r2c_once<T>(o, s, x, what);
    std::vector<exact_complex> want(n / 2 + 1);
    for (std::size_t k = 0; k <= n / 2; ++k) {
      want[k] = w[(k * at) % n];
    }
    o.within(compare<T>(y, want), tol, what);
  }
  return o;
}

template <class T> outcome r2c_tones(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) + input_rounding<T>();
  const auto& w = reference::unit_roots(+1, n);
  for (const std::size_t k0 : probe_positions(n)) {
    if (k0 > n / 2) {
      continue;
    }
    // cos(2 pi k0 j / n): n/2 at bin k0 (n at bins 0 and n/2).
    // sin(2 pi k0 j / n), 0 < k0 < n/2: -i n/2 at bin k0.
    for (const bool sine : {false, true}) {
      if (sine && (k0 == 0 || 2 * k0 == n)) {
        continue;
      }
      std::vector<T> x(n);
      for (std::size_t j = 0; j < n; ++j) {
        const exact_complex& v = w[(k0 * j) % n];
        x[j] = round_to<T>(sine ? v.im : v.re);
      }
      const auto what = describe(sine ? "sine" : "cosine", " at bin ", k0);
      const auto y = r2c_once<T>(o, s, x, what);
      std::vector<exact_complex> want(n / 2 + 1);
      const double peak =
          (k0 == 0 || 2 * k0 == n) ? static_cast<double>(n) : static_cast<double>(n) / 2;
      (sine ? want[k0].im.hi : want[k0].re.hi) = sine ? -peak : peak;
      o.within(compare<T>(y, want), tol, what);
    }
  }
  return o;
}

template <class T> outcome r2c_constant(const subject<T>& s, std::size_t n) {
  outcome o;
  const std::vector<T> x(n, T(0.75));
  const auto y = r2c_once<T>(o, s, x, "constant");
  std::vector<exact_complex> want(n / 2 + 1);
  want[0] = exact_from(std::complex<double>(0.75 * static_cast<double>(n), 0.0));
  o.within(compare<T>(y, want), tolerance_for(s, n), "constant");
  return o;
}

// n sum x_j^2 = |X_0|^2 + 2 sum_{0<k<n/2} |X_k|^2 + |X_{n/2}|^2 (the last for even n only).
template <class T> outcome r2c_parseval(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n);
  input_source src(detail::seed(n, 11));
  const auto x = src.reals<T>(n);
  const auto y = r2c_once<T>(o, s, x, "Parseval");
  compensated_sum ex;
  compensated_sum ey;
  for (const T v : x) {
    ex.add(static_cast<double>(v) * static_cast<double>(v));
  }
  for (std::size_t k = 0; k <= n / 2; ++k) {
    const bool self_paired = k == 0 || 2 * k == n;
    ey.add((self_paired ? 1.0 : 2.0) * std::norm(std::complex<double>(y[k])));
  }
  const double want = static_cast<double>(n) * ex.value();
  const double rel = std::abs(ey.value() - want) / want;
  o.within({rel, 0.0}, {2 * tol.rms + input_rounding<T>().rms, 1.0}, "Parseval");
  return o;
}

template <class T> outcome r2c_matches_c2c(const subject<T>& s, std::size_t n) {
  outcome o;
  if (!s.c2c) {
    return o;
  }
  input_source src(detail::seed(n, 12));
  const auto x = src.reals<T>(n);
  const auto y = r2c_once<T>(o, s, x, "r2c");
  const detail::cvec<T> z(x.begin(), x.end());
  auto full = c2c_once<T>(o, s, direction::forward, z, "c2c of the same input");
  full.resize(n / 2 + 1);
  o.within(compare<T>(y, as_expected<T>(full)), tolerance_for(s, n) * 2, "r2c against c2c");
  return o;
}

// Bins 0 and n/2 of a real input's transform are real.
template <class T> outcome r2c_real_bins(const subject<T>& s, std::size_t n) {
  outcome o;
  input_source src(detail::seed(n, 13));
  const auto x = src.reals<T>(n);
  const auto y = r2c_once<T>(o, s, x, "real bins");
  detail::cvec<T> want(y);
  want[0] = {y[0].real(), T(0)};
  if (n % 2 == 0) {
    want[n / 2] = {y[n / 2].real(), T(0)};
  }
  o.within(compare<T>(y, as_expected<T>(want)), tolerance_for(s, n),
           "imaginary parts of bins 0, n/2");
  return o;
}

template <class T> outcome r2c_linearity(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) * 3 + input_rounding<T>();
  input_source src(detail::seed(n, 14));
  std::vector<T> x(n);
  std::vector<T> y(n);
  for (std::size_t j = 0; j < n; ++j) {
    x[j] = static_cast<T>(src.integer(1024));
    y[j] = static_cast<T>(src.integer(1024));
  }
  const T a = T(0.75);
  const T b = T(-1.25);
  std::vector<T> z(n);
  for (std::size_t j = 0; j < n; ++j) {
    z[j] = a * x[j] + b * y[j];
  }
  const auto fx = r2c_once<T>(o, s, x, "linearity");
  const auto fy = r2c_once<T>(o, s, y, "linearity");
  const auto fz = r2c_once<T>(o, s, z, "linearity");
  std::vector<std::complex<double>> want(fz.size());
  for (std::size_t k = 0; k < fz.size(); ++k) {
    want[k] = static_cast<double>(a) * std::complex<double>(fx[k]) +
              static_cast<double>(b) * std::complex<double>(fy[k]);
  }
  o.within(compare<T>(fz, as_expected(want)), tol, "linearity");
  return o;
}

template <class T> outcome r2c_round_trip(const subject<T>& s, std::size_t n) {
  outcome o;
  if (!s.c2r) {
    return o;
  }
  input_source src(detail::seed(n, 15));
  const auto x = src.reals<T>(n);
  const auto y = r2c_once<T>(o, s, x, "round trip: r2c");
  const auto back = c2r_once<T>(o, s, n, y, "round trip: c2r");
  std::vector<double> got(n);
  for (std::size_t j = 0; j < n; ++j) {
    got[j] = static_cast<double>(back[j]) / static_cast<double>(n);
  }
  o.within(compare<double>(got, as_expected_real<T>(x)),
           tolerance_for(s, n) * 2 + input_rounding<T>(), "c2r(r2c(x)) / n");
  return o;
}

template <class T> outcome r2c_batched_strided(const subject<T>& s, std::size_t n) {
  outcome o;
  constexpr std::size_t count = 3;
  const std::size_t h = n / 2 + 1;
  input_source src(detail::seed(n, 16));
  const auto x = src.reals<T>(n * count);
  const auto in = static_cast<std::ptrdiff_t>(n);
  const auto hn = static_cast<std::ptrdiff_t>(h);
  const std::array layouts{
      rows{
          .n = n,
          .count = count,
          .in_stride = 2,
          .in_dist = 2 * in + 5,
          .out_stride = 2,
          .out_dist = 2 * hn + 3,
      },
      rows{
          .n = n,
          .count = count,
          .in_stride = -1,
          .in_dist = -(in + 1),
          .out_stride = 1,
          .out_dist = hn + 2,
      },
  };
  const std::array options{
      run_options{.in_offset = sizeof(T), .out_offset = sizeof(T)},
      run_options{
          .in_offset = 3 * sizeof(T),
          .out_offset = 2 * sizeof(T),
          .guard = guard_side::below,
      },
  };
  std::vector<std::complex<T>> single;
  for (std::size_t r = 0; r < count; ++r) {
    const auto y = r2c_once<T>(o, s, std::span(x).subspan(r * n, n), "single row");
    single.insert(single.end(), y.begin(), y.end());
  }
  for (std::size_t l = 0; l < layouts.size(); ++l) {
    const auto what = describe("batched layout ", l);
    const auto r = run_r2c<T>(s, layouts[l], x, options[l]);
    note_violations(o, r, what);
    o.within(compare<T>(r.out, as_expected<T>(single)), tolerance_for(s, n) * 2, what);
  }
  return o;
}

template <class T> outcome r2c_in_place(const subject<T>& s, std::size_t n) {
  outcome o;
  constexpr std::size_t count = 2;
  input_source src(detail::seed(n, 17));
  const auto x = src.reals<T>(n * count);
  const auto oop = run_r2c<T>(s, tight_rows(kind::r2c, n, count), x);
  const auto ip = run_r2c<T>(s, tight_rows(kind::r2c, n, count, placement::in_place), x);
  note_violations(o, oop, "out of place");
  note_violations(o, ip, "in place (padded rows)");
  o.within(compare<T>(ip.out, as_expected<T>(oop.out)), tolerance_for(s, n) * 2, "in place");
  return o;
}

// ---------------------------------------------------------------- c2r rows

namespace detail {

// A random half spectrum whose bins 0 and n/2 carry imaginary parts that c2r must ignore.
template <class T> cvec<T> half_spectrum(std::size_t n, std::uint64_t salt) {
  input_source src(seed(n, salt));
  return src.complexes<T>(n / 2 + 1);
}

template <class T> cvec<T> without_ignored_parts(cvec<T> X, std::size_t n) {
  X[0] = {X[0].real(), T(0)};
  if (n % 2 == 0) {
    X[n / 2] = {X[n / 2].real(), T(0)};
  }
  return X;
}

} // namespace detail

template <class T> outcome c2r_single_bins(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) + input_rounding<T>();
  const auto& w = reference::unit_roots(+1, n);
  const std::complex<T> a(T(0.75), T(-0.5));
  for (const std::size_t k0 : probe_positions(n)) {
    if (k0 > n / 2) {
      continue;
    }
    // X = a e_k0 gives Re(a) (bin 0), Re(a) (-1)^j (bin n/2, n even) or
    // 2 Re(a w^(j k0)) otherwise; the imaginary part of a is ignored at bins 0 and n/2.
    detail::cvec<T> X(n / 2 + 1);
    X[k0] = a;
    const auto what = describe("single bin ", k0);
    const auto y = c2r_once<T>(o, s, n, X, what);
    std::vector<double> want(n);
    for (std::size_t j = 0; j < n; ++j) {
      if (k0 == 0) {
        want[j] = 0.75;
      } else if (2 * k0 == n) {
        want[j] = j % 2 == 0 ? 0.75 : -0.75;
      } else {
        const exact_complex& v = w[(j * k0) % n];
        want[j] = 2 * (0.75 * v.re.hi + 0.5 * v.im.hi);
      }
    }
    std::vector<exact_real> e;
    e.reserve(n);
    for (const double v : want) {
      e.push_back(exact_from(v));
    }
    o.within(compare<T>(std::span<const T>(y), e), tol, what);
  }
  return o;
}

template <class T> outcome c2r_ignores_imaginary_parts(const subject<T>& s, std::size_t n) {
  outcome o;
  auto X = detail::half_spectrum<T>(n, 21);
  X[0].imag(T(0.625));
  if (n % 2 == 0) {
    X[n / 2].imag(T(-0.375));
  }
  const auto clean = detail::without_ignored_parts<T>(X, n);
  const auto y = c2r_once<T>(o, s, n, X, "with imaginary parts at bins 0, n/2");
  const auto y0 = c2r_once<T>(o, s, n, clean, "without");
  o.within(compare<T>(std::span<const T>(y), as_expected_real<T>(y0)), tolerance_for(s, n) * 2,
           "c2r ignores the imaginary parts of bins 0 and n/2");
  return o;
}

template <class T> outcome c2r_matches_c2c(const subject<T>& s, std::size_t n) {
  outcome o;
  if (!s.c2c) {
    return o;
  }
  const auto X = detail::half_spectrum<T>(n, 22);
  const auto clean = detail::without_ignored_parts<T>(X, n);
  detail::cvec<T> full(n);
  for (std::size_t k = 0; k < n; ++k) {
    full[k] = k <= n / 2 ? clean[k] : std::conj(clean[n - k]);
  }
  const auto y = c2r_once<T>(o, s, n, X, "c2r");
  const auto z = c2c_once<T>(o, s, direction::backward, full, "c2c of the Hermitian spectrum");
  std::vector<T> re(n);
  for (std::size_t j = 0; j < n; ++j) {
    re[j] = z[j].real();
  }
  o.within(compare<T>(std::span<const T>(y), as_expected_real<T>(re)), tolerance_for(s, n) * 2,
           "c2r against c2c");
  return o;
}

template <class T> outcome c2r_linearity(const subject<T>& s, std::size_t n) {
  outcome o;
  const tolerance tol = tolerance_for(s, n) * 3 + input_rounding<T>();
  input_source src(detail::seed(n, 23));
  const auto x = src.integer_complexes<T>(n / 2 + 1, 1024);
  const auto y = src.integer_complexes<T>(n / 2 + 1, 1024);
  const T a = T(0.75);
  const T b = T(-1.25);
  detail::cvec<T> z(x.size());
  for (std::size_t k = 0; k < z.size(); ++k) {
    z[k] = a * x[k] + b * y[k];
  }
  const auto fx = c2r_once<T>(o, s, n, x, "linearity");
  const auto fy = c2r_once<T>(o, s, n, y, "linearity");
  const auto fz = c2r_once<T>(o, s, n, z, "linearity");
  std::vector<exact_real> want(n);
  for (std::size_t j = 0; j < n; ++j) {
    want[j] = exact_from(static_cast<double>(a) * static_cast<double>(fx[j]) +
                         static_cast<double>(b) * static_cast<double>(fy[j]));
  }
  o.within(compare<T>(std::span<const T>(fz), want), tol, "linearity");
  return o;
}

template <class T> outcome c2r_round_trip(const subject<T>& s, std::size_t n) {
  outcome o;
  if (!s.r2c) {
    return o;
  }
  // c2r then r2c returns n X for a half spectrum whose ignored parts are zero.
  const auto X = detail::without_ignored_parts<T>(detail::half_spectrum<T>(n, 24), n);
  const auto x = c2r_once<T>(o, s, n, X, "round trip: c2r");
  const auto back = r2c_once<T>(o, s, x, "round trip: r2c");
  std::vector<std::complex<double>> got(back.size());
  for (std::size_t k = 0; k < got.size(); ++k) {
    got[k] = std::complex<double>(back[k]) / static_cast<double>(n);
  }
  o.within(compare<double>(got, as_expected<T>(X)), tolerance_for(s, n) * 2 + input_rounding<T>(),
           "r2c(c2r(X)) / n");
  return o;
}

template <class T> outcome c2r_batched_strided(const subject<T>& s, std::size_t n) {
  outcome o;
  constexpr std::size_t count = 3;
  const std::size_t h = n / 2 + 1;
  input_source src(detail::seed(n, 25));
  auto X = src.complexes<T>(h * count);
  const auto in = static_cast<std::ptrdiff_t>(n);
  const auto hn = static_cast<std::ptrdiff_t>(h);
  const std::array layouts{
      rows{
          .n = n,
          .count = count,
          .in_stride = 3,
          .in_dist = 3 * hn + 1,
          .out_stride = 2,
          .out_dist = 2 * in + 3,
      },
      rows{
          .n = n,
          .count = count,
          .in_stride = -1,
          .in_dist = -(hn + 1),
          .out_stride = 1,
          .out_dist = in + 2,
      },
  };
  const std::array options{
      run_options{.in_offset = sizeof(T), .out_offset = sizeof(T)},
      run_options{
          .in_offset = 2 * sizeof(T),
          .out_offset = 3 * sizeof(T),
          .guard = guard_side::below,
      },
  };
  std::vector<T> single;
  for (std::size_t r = 0; r < count; ++r) {
    const auto y = c2r_once<T>(o, s, n, std::span(X).subspan(r * h, h), "single row");
    single.insert(single.end(), y.begin(), y.end());
  }
  for (std::size_t l = 0; l < layouts.size(); ++l) {
    const auto what = describe("batched layout ", l);
    const auto r = run_c2r<T>(s, layouts[l], X, options[l]);
    note_violations(o, r, what);
    o.within(compare<T>(std::span<const T>(r.out), as_expected_real<T>(single)),
             tolerance_for(s, n) * 2, what);
  }
  return o;
}

template <class T> outcome c2r_in_place(const subject<T>& s, std::size_t n) {
  outcome o;
  constexpr std::size_t count = 2;
  input_source src(detail::seed(n, 26));
  const auto X = src.complexes<T>((n / 2 + 1) * count);
  const auto oop = run_c2r<T>(s, tight_rows(kind::c2r, n, count), X);
  const auto ip = run_c2r<T>(s, tight_rows(kind::c2r, n, count, placement::in_place), X);
  note_violations(o, oop, "out of place");
  note_violations(o, ip, "in place (padded rows)");
  o.within(compare<T>(std::span<const T>(ip.out), as_expected_real<T>(oop.out)),
           tolerance_for(s, n) * 2, "in place");
  return o;
}

// ---------------------------------------------------------------- planes

template <class T> outcome plane_impulses(const subject<T>& s, std::size_t ny, std::size_t nx) {
  outcome o;
  const tolerance tol = tolerance_for(s, ny * nx) + input_rounding<T>();
  for (const direction d : detail::directions) {
    const auto& wy = reference::unit_roots(sign_of(d), ny);
    const auto& wx = reference::unit_roots(sign_of(d), nx);
    for (const auto& [sy, sx] : {
             std::pair{std::size_t{0}, std::size_t{0}},
             std::pair{ny - 1, nx / 2},
             std::pair{ny / 3, nx - 1},
         }) {
      detail::cvec<T> x(ny * nx);
      x[sy * nx + sx] = T(1);
      const auto what = describe("impulse at (", sy, ", ", sx, "), ", name_of(d));
      const auto y = plane_once<T>(o, s, d, ny, nx, x, what);
      std::vector<std::complex<double>> want(ny * nx);
      for (std::size_t ky = 0; ky < ny; ++ky) {
        for (std::size_t kx = 0; kx < nx; ++kx) {
          want[ky * nx + kx] = detail::hi(wy[(ky * sy) % ny]) * detail::hi(wx[(kx * sx) % nx]);
        }
      }
      o.within(compare<T>(y, as_expected(want)), tol, what);
    }
  }
  return o;
}

// The transform of an outer product a_y b_x is the outer product of the 1-D transforms.
// Integer factors up to 2^10 keep the products exact in float.
template <class T> outcome plane_separable(const subject<T>& s, std::size_t ny, std::size_t nx) {
  outcome o;
  const tolerance tol = tolerance_for(s, ny * nx) + input_rounding<T>();
  input_source src(detail::seed(ny * 131 + nx, 31));
  const auto a = src.integer_complexes<T>(ny, 1024);
  const auto b = src.integer_complexes<T>(nx, 1024);
  detail::cvec<T> x(ny * nx);
  for (std::size_t y = 0; y < ny; ++y) {
    for (std::size_t xi = 0; xi < nx; ++xi) {
      x[y * nx + xi] = a[y] * b[xi];
    }
  }
  for (const direction d : detail::directions) {
    const auto fa = reference::c2c(sign_of(d), widen(a));
    const auto fb = reference::c2c(sign_of(d), widen(b));
    const auto what = describe("separable input, ", name_of(d));
    const auto y = plane_once<T>(o, s, d, ny, nx, x, what);
    std::vector<std::complex<double>> want(ny * nx);
    for (std::size_t ky = 0; ky < ny; ++ky) {
      for (std::size_t kx = 0; kx < nx; ++kx) {
        want[ky * nx + kx] = detail::hi(fa[ky]) * detail::hi(fb[kx]);
      }
    }
    o.within(compare<T>(y, as_expected(want)), tol, what);
  }
  return o;
}

template <class T> outcome plane_constant(const subject<T>& s, std::size_t ny, std::size_t nx) {
  outcome o;
  const std::complex<T> c(T(-0.25), T(0.5));
  for (const direction d : detail::directions) {
    const detail::cvec<T> x(ny * nx, c);
    const auto what = describe("constant, ", name_of(d));
    const auto y = plane_once<T>(o, s, d, ny, nx, x, what);
    std::vector<exact_complex> want(ny * nx);
    want[0] = exact_from(std::complex<double>(c) * static_cast<double>(ny * nx));
    o.within(compare<T>(y, want), tolerance_for(s, ny * nx), what);
  }
  return o;
}

template <class T> outcome plane_round_trip(const subject<T>& s, std::size_t ny, std::size_t nx) {
  outcome o;
  input_source src(detail::seed(ny * 131 + nx, 32));
  const auto x = src.complexes<T>(ny * nx);
  const auto y = plane_once<T>(o, s, direction::forward, ny, nx, x, "round trip: forward");
  const auto back = plane_once<T>(o, s, direction::backward, ny, nx, y, "round trip: backward");
  std::vector<std::complex<double>> got(back.size());
  for (std::size_t j = 0; j < got.size(); ++j) {
    got[j] = std::complex<double>(back[j]) / static_cast<double>(ny * nx);
  }
  o.within(compare<double>(got, as_expected<T>(x)),
           tolerance_for(s, ny * nx) * 2 + input_rounding<T>(), "backward(forward(x)) / n");
  return o;
}

template <class T>
outcome plane_batched_strided(const subject<T>& s, std::size_t ny, std::size_t nx) {
  outcome o;
  constexpr std::size_t count = 2;
  const std::size_t points = ny * nx;
  input_source src(detail::seed(ny * 131 + nx, 33));
  const auto x = src.complexes<T>(points * count);
  const auto y = static_cast<std::ptrdiff_t>(ny);
  const auto xn = static_cast<std::ptrdiff_t>(nx);
  const std::array layouts{
      // Padded rows in, every other element out.
      planes{
          .ny = ny,
          .nx = nx,
          .count = count,
          .in_y_stride = xn + 3,
          .in_x_stride = 1,
          .in_dist = y * (xn + 3) + 5,
          .out_y_stride = 2 * xn + 1,
          .out_x_stride = 2,
          .out_dist = y * (2 * xn + 1) + 1,
      },
      // Transposed planes (x is the slow axis) going backwards in memory.
      planes{
          .ny = ny,
          .nx = nx,
          .count = count,
          .in_y_stride = 1,
          .in_x_stride = y,
          .in_dist = -(y * xn + 2),
          .out_y_stride = xn,
          .out_x_stride = 1,
          .out_dist = y * xn,
      },
  };
  const std::array options{
      run_options{.in_offset = sizeof(T), .out_offset = 2 * sizeof(T)},
      run_options{.in_offset = 0, .out_offset = sizeof(T), .guard = guard_side::below},
  };
  for (const direction d : detail::directions) {
    std::vector<std::complex<T>> single;
    for (std::size_t p = 0; p < count; ++p) {
      const auto r =
          plane_once<T>(o, s, d, ny, nx, std::span(x).subspan(p * points, points), "single plane");
      single.insert(single.end(), r.begin(), r.end());
    }
    for (std::size_t l = 0; l < layouts.size(); ++l) {
      const auto what = describe("batched layout ", l, ", ", name_of(d));
      const auto r = run_c2c_planes<T>(s, d, layouts[l], x, options[l]);
      note_violations(o, r, what);
      o.within(compare<T>(r.out, as_expected<T>(single)), tolerance_for(s, points) * 2, what);
    }
  }
  return o;
}

template <class T> outcome plane_in_place(const subject<T>& s, std::size_t ny, std::size_t nx) {
  outcome o;
  constexpr std::size_t count = 2;
  input_source src(detail::seed(ny * 131 + nx, 34));
  const auto x = src.complexes<T>(ny * nx * count);
  for (const direction d : detail::directions) {
    const auto what = describe("in place, ", name_of(d));
    const auto oop = run_c2c_planes<T>(s, d, tight_planes(ny, nx, count), x);
    const auto ip = run_c2c_planes<T>(s, d, tight_planes(ny, nx, count, placement::in_place), x);
    note_violations(o, oop, what);
    note_violations(o, ip, what);
    o.within(compare<T>(ip.out, as_expected<T>(oop.out)), tolerance_for(s, ny * nx) * 2, what);
  }
  return o;
}

// ---------------------------------------------------------------- the suites

template <class T> std::vector<row_check<T>> row_checks() {
  return {
      {"c2c_impulses", kind::c2c, &c2c_impulses<T>},
      {"c2c_single_bins", kind::c2c, &c2c_single_bins<T>},
      {"c2c_real_tones", kind::c2c, &c2c_real_tones<T>},
      {"c2c_constant", kind::c2c, &c2c_constant<T>},
      {"c2c_parseval", kind::c2c, &c2c_parseval<T>},
      {"c2c_shift", kind::c2c, &c2c_shift<T>},
      {"c2c_linearity", kind::c2c, &c2c_linearity<T>},
      {"c2c_round_trip", kind::c2c, &c2c_round_trip<T>},
      {"c2c_real_input_symmetry", kind::c2c, &c2c_real_input_symmetry<T>},
      {"c2c_batched_strided", kind::c2c, &c2c_batched_strided<T>},
      {"c2c_in_place", kind::c2c, &c2c_in_place<T>},
      {"r2c_impulses", kind::r2c, &r2c_impulses<T>},
      {"r2c_tones", kind::r2c, &r2c_tones<T>},
      {"r2c_constant", kind::r2c, &r2c_constant<T>},
      {"r2c_parseval", kind::r2c, &r2c_parseval<T>},
      {"r2c_matches_c2c", kind::r2c, &r2c_matches_c2c<T>},
      {"r2c_real_bins", kind::r2c, &r2c_real_bins<T>},
      {"r2c_linearity", kind::r2c, &r2c_linearity<T>},
      {"r2c_round_trip", kind::r2c, &r2c_round_trip<T>},
      {"r2c_batched_strided", kind::r2c, &r2c_batched_strided<T>},
      {"r2c_in_place", kind::r2c, &r2c_in_place<T>},
      {"c2r_single_bins", kind::c2r, &c2r_single_bins<T>},
      {"c2r_ignores_imaginary_parts", kind::c2r, &c2r_ignores_imaginary_parts<T>},
      {"c2r_matches_c2c", kind::c2r, &c2r_matches_c2c<T>},
      {"c2r_linearity", kind::c2r, &c2r_linearity<T>},
      {"c2r_round_trip", kind::c2r, &c2r_round_trip<T>},
      {"c2r_batched_strided", kind::c2r, &c2r_batched_strided<T>},
      {"c2r_in_place", kind::c2r, &c2r_in_place<T>},
  };
}

template <class T> std::vector<plane_check<T>> plane_checks() {
  return {
      {"planes_impulses", &plane_impulses<T>},
      {"planes_separable", &plane_separable<T>},
      {"planes_constant", &plane_constant<T>},
      {"planes_round_trip", &plane_round_trip<T>},
      {"planes_batched_strided", &plane_batched_strided<T>},
      {"planes_in_place", &plane_in_place<T>},
  };
}

} // namespace fftn::test::analytic
