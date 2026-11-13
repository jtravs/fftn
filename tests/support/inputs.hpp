// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Deterministic test inputs. The standard distributions differ between standard libraries,
// so values are formed from the raw bits of a fixed-seed engine: the same seed gives the
// same input on every platform. Values have at most 24 significant bits, so they are exact
// in float and double alike and both precisions see the same input.

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace fftn::test {

class input_source {
public:
  explicit input_source(std::uint64_t seed) : engine_(seed) {}

  // Uniform on [-1, 1) in steps of 2^-23.
  double uniform() {
    const auto bits = static_cast<std::int64_t>(engine_() >> 40U); // 24 bits
    return static_cast<double>(bits - 8388608) / 8388608.0;        // 2^23
  }

  // An integer in [-limit, limit], for inputs whose linear combinations must be exact.
  double integer(std::uint32_t limit) {
    const std::uint64_t span = 2 * std::uint64_t{limit} + 1;
    return static_cast<double>(static_cast<std::int64_t>(engine_() % span) -
                               static_cast<std::int64_t>(limit));
  }

  template <class T> std::vector<T> reals(std::size_t n) {
    std::vector<T> v(n);
    for (auto& x : v) {
      x = static_cast<T>(uniform());
    }
    return v;
  }

  template <class T> std::vector<std::complex<T>> complexes(std::size_t n) {
    std::vector<std::complex<T>> v(n);
    for (auto& x : v) {
      const double re = uniform();
      x = {static_cast<T>(re), static_cast<T>(uniform())};
    }
    return v;
  }

  template <class T>
  std::vector<std::complex<T>> integer_complexes(std::size_t n, std::uint32_t limit) {
    std::vector<std::complex<T>> v(n);
    for (auto& x : v) {
      const double re = integer(limit);
      x = {static_cast<T>(re), static_cast<T>(integer(limit))};
    }
    return v;
  }

private:
  std::mt19937_64 engine_;
};

template <class T> std::vector<std::complex<double>> widen(const std::vector<std::complex<T>>& v) {
  return {v.begin(), v.end()};
}

template <class T> std::vector<double> widen(const std::vector<T>& v) {
  return {v.begin(), v.end()};
}

} // namespace fftn::test
