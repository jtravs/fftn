// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The kernel text (src/isa/kernels.inl), compiled once per target, against a direct
// computation in double precision: every runnable target, both precisions, both
// directions. Two checks per kernel and length:
//
// - unit twiddles and integer data, where every target's arithmetic is exact, so any
//   misplaced lane, part or output shows as an exact mismatch;
// - true twiddles and random data, within a bound derived before the run: a radix-4
//   output is two levels of additions and one complex product, a radix-2 output one
//   product and one addition, so with u = eps(T) + eps(double) (the kernel's and the
//   reference's rounding) each part is within 6u and 4u times the sum of |re| + |im| of
//   the inputs it combines. A wrong twiddle or sign is off by O(1).
//
// Also checked: the input is preserved and nothing is written outside the output.

#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "isa/test_targets.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

using fftn::detail::find_kernels;
using fftn::detail::kernel_table;
using fftn::detail::kernels_of;
using fftn::detail::of;
using fftn::detail::target;

enum class precision : unsigned char { f64, f32 };
using param = std::tuple<target, precision, int>;
using cd = std::complex<double>;

constexpr double canary = -77.0;

// Interleaved complex data of T at an odd offset into its buffer, with canaries around.
template <class T> struct buffer {
  std::vector<T> data;
  explicit buffer(std::size_t complex_count) : data(2 * complex_count + 2, T(canary)) {}
  T* ptr() { return data.data() + 1; }
  [[nodiscard]] cd at(std::size_t k) const {
    return {static_cast<double>(data[2 * k + 1]), static_cast<double>(data[2 * k + 2])};
  }
  void set(std::size_t k, cd z) {
    data[2 * k + 1] = static_cast<T>(z.real());
    data[2 * k + 2] = static_cast<T>(z.imag());
  }
  [[nodiscard]] bool canaries_intact() const {
    return data.front() == T(canary) && data.back() == T(canary);
  }
};

double norm1(cd z) { return std::abs(z.real()) + std::abs(z.imag()); }

template <class T> double unit() {
  return std::numeric_limits<T>::epsilon() + std::numeric_limits<double>::epsilon();
}

// exp(sign 2 pi i k / n), rounded to T and returned in double.
template <class T> cd twiddle(int sign, std::size_t k, std::size_t n) {
  const double angle =
      sign * 2 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(n);
  return {static_cast<double>(static_cast<T>(std::cos(angle))),
          static_cast<double>(static_cast<T>(std::sin(angle)))};
}

enum class data_kind : unsigned char { exact, random };

template <class T> void fill(buffer<T>& x, std::size_t count, data_kind kind, std::mt19937& gen) {
  std::uniform_real_distribution<double> real(-1, 1);
  std::uniform_int_distribution<int> integer(-1000, 1000);
  for (std::size_t k = 0; k < count; ++k) {
    if (kind == data_kind::exact) {
      x.set(k, {static_cast<double>(integer(gen)), static_cast<double>(integer(gen))});
    } else {
      x.set(k, {real(gen), real(gen)});
    }
  }
}

// Compares y with the reference within bound[k] (zero: exactly).
template <class T>
void compare(const buffer<T>& y, const std::vector<cd>& ref, const std::vector<double>& bound) {
  int reported = 0;
  for (std::size_t k = 0; k < ref.size() && reported < 4; ++k) {
    const cd got = y.at(k);
    const double err =
        std::max(std::abs(got.real() - ref[k].real()), std::abs(got.imag() - ref[k].imag()));
    if (!(err <= bound[k])) {
      ADD_FAILURE() << "output " << k << ": got " << got << ", expected " << ref[k] << " (error "
                    << err << ", bound " << bound[k] << ")";
      ++reported;
    }
  }
  EXPECT_TRUE(y.canaries_intact()) << "wrote outside the output";
}

template <class T> void radix4(const kernels_of<T>& k, int sign, std::size_t m, data_kind kind) {
  SCOPED_TRACE("radix4_pass, m = " + std::to_string(m));
  const std::size_t n = 4 * m;
  std::mt19937 gen(static_cast<unsigned>(m));
  buffer<T> x(n);
  fill(x, n, kind, gen);
  const buffer<T> x0 = x;
  // Rows j = 1, 2, 3 of forward twiddles, then the target's storage of them.
  buffer<T> w(3 * m);
  for (std::size_t j = 1; j < 4; ++j) {
    for (std::size_t p = 0; p < m; ++p) {
      w.set((j - 1) * m + p, kind == data_kind::exact ? cd(1, 0) : twiddle<T>(-1, j * p, n));
    }
  }
  std::vector<T> storage(k.tw_reals * 3 * m);
  k.pack_twiddles(w.ptr(), 3 * m, storage.data());

  buffer<T> y(n);
  k.radix4_pass(x.ptr(), y.ptr(), storage.data(), m, sign);

  const cd quarter[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // NOLINT(modernize-avoid-c-arrays)
  std::vector<cd> ref(n);
  std::vector<double> bound(n);
  for (std::size_t p = 0; p < m; ++p) {
    for (std::size_t j = 0; j < 4; ++j) {
      cd sum = 0;
      double size = 0;
      for (std::size_t l = 0; l < 4; ++l) {
        const std::size_t power = (sign < 0 ? 4 - j * l % 4 : j * l) % 4; // i^(sign j l)
        sum += quarter[power] * x.at(p + l * m);
        size += norm1(x.at(p + l * m));
      }
      if (j > 0) {
        const cd t = w.at((j - 1) * m + p);
        sum *= sign < 0 ? t : std::conj(t);
      }
      ref[4 * p + j] = sum;
      bound[4 * p + j] = kind == data_kind::exact ? 0 : 6 * unit<T>() * size;
    }
  }
  compare(y, ref, bound);
  EXPECT_EQ(x.data, x0.data) << "input changed";
}

template <class T>
void radix2_split(const kernels_of<T>& k, int sign, std::size_t m, data_kind kind) {
  SCOPED_TRACE("radix2_split_pass, m = " + std::to_string(m));
  const std::size_t n = 2 * m;
  std::mt19937 gen(static_cast<unsigned>(m));
  buffer<T> x(n);
  fill(x, n, kind, gen);
  const buffer<T> x0 = x;
  std::vector<T> twr(m);
  std::vector<T> twi(m);
  for (std::size_t j = 0; j < m; ++j) {
    const cd t = kind == data_kind::exact ? cd(1, 0) : twiddle<T>(-1, j, n);
    twr[j] = static_cast<T>(t.real());
    twi[j] = static_cast<T>(t.imag());
  }

  buffer<T> y(n);
  k.radix2_split_pass(x.ptr(), y.ptr(), twr.data(), twi.data(), m, sign);

  std::vector<cd> ref(n);
  std::vector<double> bound(n);
  for (std::size_t j = 0; j < m; ++j) {
    const cd t{static_cast<double>(twr[j]), static_cast<double>(twi[j])};
    const cd b = x.at(j + m) * (sign < 0 ? t : std::conj(t));
    ref[j] = x.at(j) + b;
    ref[j + m] = x.at(j) - b;
    const double size = norm1(x.at(j)) + norm1(x.at(j + m));
    bound[j] = bound[j + m] = kind == data_kind::exact ? 0 : 4 * unit<T>() * size;
  }
  compare(y, ref, bound);
  EXPECT_EQ(x.data, x0.data) << "input changed";
}

class kernels : public ::testing::TestWithParam<param> {
protected:
  template <class T> void check() {
    const auto [t, prec, sign] = GetParam();
    const kernel_table* table = find_kernels(t);
    ASSERT_NE(table, nullptr);
    const kernels_of<T>& k = of<T>(*table);
    for (const data_kind kind : {data_kind::exact, data_kind::random}) {
      SCOPED_TRACE(kind == data_kind::exact ? "exact data" : "random data");
      // Multiples of every target's lane counts (C <= 4, W <= 8), small and large.
      for (const std::size_t m :
           {std::size_t{4}, std::size_t{8}, std::size_t{12}, std::size_t{100}}) {
        radix4<T>(k, sign, m, kind);
      }
      for (const std::size_t m :
           {std::size_t{8}, std::size_t{16}, std::size_t{24}, std::size_t{200}}) {
        radix2_split<T>(k, sign, m, kind);
      }
    }
  }
};

TEST_P(kernels, match_direct_computation) {
  if (std::get<1>(GetParam()) == precision::f64) {
    check<double>();
  } else {
    check<float>();
  }
}

std::string name(const ::testing::TestParamInfo<param>& info) {
  const auto [t, prec, sign] = info.param;
  return isa_test::label(t) + (prec == precision::f64 ? "_f64" : "_f32") +
         (sign < 0 ? "_forward" : "_backward");
}

INSTANTIATE_TEST_SUITE_P(isa, kernels,
                         ::testing::Combine(::testing::ValuesIn(isa_test::runnable_targets()),
                                            ::testing::Values(precision::f64, precision::f32),
                                            ::testing::Values(-1, +1)),
                         name);

} // namespace
