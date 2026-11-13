// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The result of an oracle check, and the error metrics it is judged by.
//
// Checks report rather than assert, so that the same check can be shown to pass on a
// correct subject and to fail on a faulty one (tests/analytic/seeded_faults_test.cpp).

#pragma once

#include "support/exact.hpp"
#include "support/tolerance.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::test {

// Relative errors of a computed output against its expected value (tolerance.hpp).
struct error_stats {
  double rms = 0.0;
  double max = 0.0;
};

template <class T>
error_stats compare(std::span<const std::complex<T>> got, std::span<const exact_complex> want) {
  double err2 = 0.0;
  double ref2 = 0.0;
  double err_max = 0.0;
  double ref_max = 0.0;
  const std::size_t n = std::min(got.size(), want.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double dr = error_of(static_cast<double>(got[i].real()), want[i].re);
    const double di = error_of(static_cast<double>(got[i].imag()), want[i].im);
    const double e = std::hypot(dr, di);
    const double r = magnitude(want[i]);
    err2 += e * e;
    ref2 += r * r;
    err_max = std::max(err_max, e);
    ref_max = std::max(ref_max, r);
  }
  // A NaN anywhere must fail: NaN compares false, so it is turned into infinity.
  error_stats s{.rms = std::sqrt(err2 / ref2), .max = err_max / ref_max};
  if (std::isnan(s.rms) || got.size() != want.size()) {
    s.rms = std::numeric_limits<double>::infinity();
  }
  if (std::isnan(s.max) || got.size() != want.size()) {
    s.max = std::numeric_limits<double>::infinity();
  }
  return s;
}

template <class T> error_stats compare(std::span<const T> got, std::span<const exact_real> want) {
  std::vector<std::complex<T>> g(got.begin(), got.end());
  std::vector<exact_complex> w;
  w.reserve(want.size());
  for (const exact_real& v : want) {
    w.push_back({.re = v, .im = {}});
  }
  return compare<T>(std::span<const std::complex<T>>(g), w);
}

class outcome {
public:
  [[nodiscard]] bool passed() const { return passed_; }
  [[nodiscard]] const std::string& detail() const { return detail_; }
  // The largest error / tolerance ratio seen by within(): how close a pass came to failing.
  [[nodiscard]] double worst_ratio() const { return worst_ratio_; }

  void fail(std::string_view what) {
    passed_ = false;
    if (detail_.size() < 4000) {
      detail_.append(what).append("\n");
    }
  }

  void require(bool ok, std::string_view what) {
    if (!ok) {
      fail(what);
    }
  }

  void within(const error_stats& e, const tolerance& tol, std::string_view what) {
    const double ratio = std::max(e.rms / tol.rms, e.max / tol.max);
    worst_ratio_ = std::max(worst_ratio_, std::isnan(ratio) ? HUGE_VAL : ratio);
    // Written so that a NaN error fails: every comparison with NaN is false.
    const bool ok = e.rms <= tol.rms && e.max <= tol.max;
    if (!ok) {
      std::ostringstream s;
      s << what << ": rms " << e.rms << " (tolerance " << tol.rms << "), max " << e.max
        << " (tolerance " << tol.max << ")";
      fail(s.str());
    }
  }

  void merge(const outcome& o) {
    worst_ratio_ = std::max(worst_ratio_, o.worst_ratio_);
    if (!o.passed_) {
      passed_ = false;
      if (detail_.size() < 4000) {
        detail_ += o.detail_;
      }
    }
  }

private:
  bool passed_ = true;
  std::string detail_;
  double worst_ratio_ = 0.0;
};

} // namespace fftn::test
