// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Shared pieces of the oracle checks (tests/analytic/oracles.hpp,
// tests/reference/reference_checks.hpp): the check records the suites iterate over, and
// single-transform helpers.

#pragma once

#include "support/harness.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"
#include "support/transform_under_test.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::test {

// A check over 1-D rows of length n, for one kind of transform.
template <class T> struct row_check {
  std::string_view name;
  kind k;
  outcome (*run)(const subject<T>&, std::size_t n);
};

// A check over 2-D planes of ny x nx.
template <class T> struct plane_check {
  std::string_view name;
  outcome (*run)(const subject<T>&, std::size_t ny, std::size_t nx);
};

template <class T> bool supports(const subject<T>& s, kind k) {
  switch (k) {
  case kind::c2c:
    return static_cast<bool>(s.c2c);
  case kind::r2c:
    return static_cast<bool>(s.r2c);
  case kind::c2r:
    return static_cast<bool>(s.c2r);
  }
  return false;
}

template <class T> tolerance tolerance_for(const subject<T>& s, std::size_t points) {
  return derived_tolerance<T>(points, s.bound_for(points));
}

inline std::string_view name_of(kind k) {
  switch (k) {
  case kind::c2c:
    return "c2c";
  case kind::r2c:
    return "r2c";
  case kind::c2r:
    return "c2r";
  }
  return "?";
}

inline std::string_view name_of(direction d) {
  return d == direction::forward ? "forward" : "backward";
}

// Builds a short description: describe("impulse at ", 3, ", ", "forward").
template <class... Parts> std::string describe(const Parts&... parts) {
  std::ostringstream s;
  (s << ... << parts);
  return s.str();
}

// Contract violations seen by the harness fail the check.
template <class Out>
void note_violations(outcome& o, const run_result<Out>& r, std::string_view what) {
  for (const std::string& v : r.violations) {
    o.fail(describe(what, ": ", v));
  }
}

// Distinct positions worth probing in a row of n: the ends, the middle and a third.
inline std::vector<std::size_t> probe_positions(std::size_t n) {
  std::vector<std::size_t> v;
  for (const std::size_t p : {std::size_t{0}, std::size_t{1}, n / 3, n / 2, n - 1}) {
    if (p < n && std::ranges::find(v, p) == v.end()) {
      v.push_back(p);
    }
  }
  return v;
}

// One tight, out-of-place transform, checked for contract violations as a side effect.
template <class T>
std::vector<std::complex<T>> c2c_once(outcome& o, const subject<T>& s, direction d,
                                      std::span<const std::complex<T>> x, std::string_view what) {
  const auto r = run_c2c<T>(s, d, tight_rows(kind::c2c, x.size(), 1), x);
  note_violations(o, r, what);
  return r.out;
}

template <class T>
std::vector<std::complex<T>> r2c_once(outcome& o, const subject<T>& s, std::span<const T> x,
                                      std::string_view what) {
  const auto r = run_r2c<T>(s, tight_rows(kind::r2c, x.size(), 1), x);
  note_violations(o, r, what);
  return r.out;
}

template <class T>
std::vector<T> c2r_once(outcome& o, const subject<T>& s, std::size_t n,
                        std::span<const std::complex<T>> X, std::string_view what) {
  const auto r = run_c2r<T>(s, tight_rows(kind::c2r, n, 1), X);
  note_violations(o, r, what);
  return r.out;
}

template <class T>
std::vector<std::complex<T>> plane_once(outcome& o, const subject<T>& s, direction d,
                                        std::size_t ny, std::size_t nx,
                                        std::span<const std::complex<T>> x, std::string_view what) {
  const auto r = run_c2c_planes<T>(s, d, tight_planes(ny, nx, 1), x);
  note_violations(o, r, what);
  return r.out;
}

// Expected values computed in double from other computed values.
inline std::vector<exact_complex> as_expected(std::span<const std::complex<double>> v) {
  std::vector<exact_complex> e;
  e.reserve(v.size());
  for (const auto& z : v) {
    e.push_back(exact_from(z));
  }
  return e;
}

template <class T> std::vector<exact_complex> as_expected(std::span<const std::complex<T>> v) {
  std::vector<exact_complex> e;
  e.reserve(v.size());
  for (const auto& z : v) {
    e.push_back(exact_from(std::complex<double>(z)));
  }
  return e;
}

template <class T> std::vector<exact_real> as_expected_real(std::span<const T> v) {
  std::vector<exact_real> e;
  e.reserve(v.size());
  for (const T z : v) {
    e.push_back(exact_from(static_cast<double>(z)));
  }
  return e;
}

// Neumaier's compensated sum, for energies in Parseval checks: its error does not grow
// with n, so it stays far below the tolerance even for double.
class compensated_sum {
public:
  void add(double v) {
    const double t = sum_ + v;
    if (std::abs(sum_) >= std::abs(v)) {
      carry_ += (sum_ - t) + v;
    } else {
      carry_ += (v - t) + sum_;
    }
    sum_ = t;
  }
  [[nodiscard]] double value() const { return sum_ + carry_; }

private:
  double sum_ = 0.0;
  double carry_ = 0.0;
};

} // namespace fftn::test
