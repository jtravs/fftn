// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Every vocabulary operation of every runnable target, in both precisions, against a
// scalar reference written here (not the scalar target: a shared mistake would pass).
//
// Exact operations are compared bit for bit on random data. The complex products (mul,
// mulc, madd) round differently on each target, so they are checked on small integers,
// where every target's arithmetic is exact and any wrong sign, part or lane shows. The
// fused operations are compared with std::fma or with multiply-then-add, as the target
// declares, on data that includes a case where the two differ: a lost FMA fails.

#include "isa/kernel_table.hpp"
#include "isa/probe_table.hpp"
#include "isa/target.hpp"
#include "isa/test_targets.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

using fftn::detail::find_kernels;
using fftn::detail::find_probes;
using fftn::detail::kernel_table;
using fftn::detail::of;
using fftn::detail::probe_table;
using fftn::detail::target;
using fftn::detail::vop;
using fftn::detail::vop_names;

enum class precision : unsigned char { f64, f32 };
using param = std::tuple<target, precision, vop>;

// Reals per array: a multiple of 2 C^2 (32 for avx2 float) and of 2 W (16).
constexpr std::size_t n = 128;
constexpr std::size_t half = n / 2;

// Data of the operation, at an odd offset into its buffer (unaligned for every register
// width), with a canary on either side.
template <class T> struct operand {
  std::vector<T> buffer;
  explicit operand(std::size_t size) : buffer(size + 2, T(-77)) {}
  T* data() { return buffer.data() + 1; }
  [[nodiscard]] std::size_t size() const { return buffer.size() - 2; }
  T& operator[](std::size_t i) { return buffer[i + 1]; }
};

template <class T> operand<T> random_reals(std::mt19937& gen, std::size_t size) {
  std::uniform_real_distribution<T> dist(T(-1), T(1));
  operand<T> out(size);
  for (std::size_t i = 0; i < size; ++i) {
    out[i] = dist(gen);
  }
  return out;
}

// Integers small enough that products and sums of three of them are exact: |x| <= 2^20 for
// double (sums below 2^42), 2^10 for float (below 2^22).
template <class T> operand<T> random_integers(std::mt19937& gen, std::size_t size) {
  const int bound = sizeof(T) == sizeof(double) ? 1048576 : 1024;
  std::uniform_int_distribution<int> dist(-bound, bound);
  operand<T> out(size);
  for (std::size_t i = 0; i < size; ++i) {
    out[i] = static_cast<T>(dist(gen));
  }
  return out;
}

// u such that (1 + u)(1 - u) = 1 - u^2 rounds to 1 but is exact when fused.
template <class T> constexpr T tiny() {
  return sizeof(T) == sizeof(double) ? T(0x1p-30) : T(0x1p-13);
}

template <class T> T fma_or_not(bool fused, T x, T y, T z) {
  return fused ? std::fma(x, y, z) : x * y + z;
}

template <class T> struct expected_and_got {
  std::vector<T> expected;
  operand<T> got;
};

template <class T>
expected_and_got<T> run(const kernel_table& kernels, const probe_table& probes, vop op) {
  std::mt19937 gen(12345); // NOLINT(bugprone-random-generator-seed): reproducible data
  const bool exact_products = op == vop::c_mul || op == vop::c_mulc || op == vop::c_madd;
  operand<T> a = exact_products ? random_integers<T>(gen, n) : random_reals<T>(gen, n);
  operand<T> b = exact_products ? random_integers<T>(gen, n) : random_reals<T>(gen, n);
  operand<T> c = exact_products ? random_integers<T>(gen, n) : random_reals<T>(gen, n);
  T r = random_reals<T>(gen, 1)[0];
  const bool fused = kernels.fused;
  const std::size_t lanes = of<T>(kernels).complex_lanes;
  const std::size_t width = of<T>(kernels).real_lanes;

  // Fused and unfused results differ in the first element (and complex number).
  const T u = tiny<T>();
  if (!exact_products) {
    a[0] = a[1] = T(1) + u;
    b[0] = T(1) - u;
    c[0] = c[1] = op == vop::r_fmadd || op == vop::c_fmadd_r ? T(-1) : T(1);
  }
  if (op == vop::c_fmadd_r) {
    r = T(1) - u;
  }

  std::vector<T> e(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t re = i & ~std::size_t{1}; // the real part of i's complex number
    const bool odd = (i & 1U) != 0;
    const T ar = a[re];
    const T ai = a[re + 1];
    const T br = b[re];
    const T bi = b[re + 1];
    switch (op) {
    case vop::c_copy:
    case vop::r_copy:
      e[i] = a[i];
      break;
    case vop::c_add:
    case vop::r_add:
      e[i] = a[i] + b[i];
      break;
    case vop::c_sub:
    case vop::r_sub:
      e[i] = a[i] - b[i];
      break;
    case vop::c_addj: // a + i b
      e[i] = odd ? ai + br : ar - bi;
      break;
    case vop::c_subj: // a - i b
      e[i] = odd ? ai - br : ar + bi;
      break;
    case vop::c_mul:
      e[i] = odd ? ar * bi + ai * br : ar * br - ai * bi;
      break;
    case vop::c_mulc:
      e[i] = odd ? ai * br - ar * bi : ar * br + ai * bi;
      break;
    case vop::c_madd:
      e[i] = c[i] + (odd ? ar * bi + ai * br : ar * br - ai * bi);
      break;
    case vop::c_scale:
      e[i] = r * a[i];
      break;
    case vop::c_fmadd_r:
      e[i] = fma_or_not(fused, r, a[i], c[i]);
      break;
    case vop::c_neg:
    case vop::r_neg:
      e[i] = -a[i];
      break;
    case vop::c_conj:
      e[i] = odd ? -a[i] : a[i];
      break;
    case vop::c_swap_ri:
      e[i] = odd ? ar : ai;
      break;
    case vop::c_bcast: // register g holds complex number g in each of its lanes
      e[i] = a[2 * (i / (2 * lanes)) + (odd ? 1 : 0)];
      break;
    case vop::c_transpose: { // C x C blocks of complex numbers
      const std::size_t z = i / 2;
      const std::size_t block = z / (lanes * lanes) * lanes * lanes;
      const std::size_t row = (z - block) / lanes;
      const std::size_t col = (z - block) % lanes;
      e[i] = a[2 * (block + col * lanes + row) + (odd ? 1 : 0)];
      break;
    }
    case vop::r_mul:
      e[i] = a[i] * b[i];
      break;
    case vop::r_fmadd:
      e[i] = fma_or_not(fused, a[i], b[i], c[i]);
      break;
    case vop::r_fmsub:
      e[i] = fma_or_not(fused, a[i], b[i], -c[i]);
      break;
    case vop::r_fnmadd:
      e[i] = fma_or_not(fused, -a[i], b[i], c[i]);
      break;
    case vop::r_bcast:
      e[i] = a[i / width];
      break;
    case vop::r_deinterleave:
      e[i] = i < half ? a[2 * i] : a[2 * (i - half) + 1];
      break;
    case vop::r_interleave:
      e[i] = odd ? b[i / 2] : a[i / 2];
      break;
    }
  }

  // The twiddle operand of the complex products is passed through the target's own
  // twiddle storage, which tests pack_twiddles and load_tw too.
  const std::size_t tw_size = of<T>(kernels).tw_reals * half;
  operand<T> tw(tw_size);
  of<T>(kernels).pack_twiddles(b.data(), half, tw.data());
  const T* second = exact_products ? tw.data() : b.data();

  operand<T> got(n);
  if constexpr (sizeof(T) == sizeof(double)) {
    probes.f64(op, a.data(), second, c.data(), got.data(), n, &r);
  } else {
    probes.f32(op, a.data(), second, c.data(), got.data(), n, &r);
  }
  EXPECT_EQ(tw.buffer.front(), T(-77)) << "pack_twiddles wrote before its output";
  EXPECT_EQ(tw.buffer.back(), T(-77)) << "pack_twiddles wrote past its output";
  return {e, got};
}

class vocab : public ::testing::TestWithParam<param> {
protected:
  template <class T> void check() {
    const auto [t, prec, op] = GetParam();
    const kernel_table* kernels = find_kernels(t);
    const probe_table* probes = find_probes(t);
    ASSERT_NE(kernels, nullptr);
    ASSERT_NE(probes, nullptr);
    auto [expected, got] = run<T>(*kernels, *probes, op);
    EXPECT_EQ(got.buffer.front(), T(-77)) << "wrote before its output";
    EXPECT_EQ(got.buffer.back(), T(-77)) << "wrote past its output";
    int reported = 0;
    for (std::size_t i = 0; i < n && reported < 4; ++i) {
      if (!(got[i] == expected[i])) {
        ADD_FAILURE() << "element " << i << ": got " << got[i] << ", expected " << expected[i];
        ++reported;
      }
    }
  }
};

TEST_P(vocab, matches_reference) {
  if (std::get<1>(GetParam()) == precision::f64) {
    check<double>();
  } else {
    check<float>();
  }
}

// The data above must tell a fused operation from an unfused one.
TEST(vocab_data, fused_and_unfused_differ) {
  const auto ud = tiny<double>();
  EXPECT_NE(std::fma(1 + ud, 1 - ud, -1.0), (1 + ud) * (1 - ud) - 1.0);
  const auto uf = tiny<float>();
  EXPECT_NE(std::fma(1 + uf, 1 - uf, -1.0F), (1 + uf) * (1 - uf) - 1.0F);
}

std::vector<vop> all_vops() {
  std::vector<vop> out;
  out.reserve(vop_names.size());
  for (std::size_t i = 0; i < vop_names.size(); ++i) {
    out.push_back(static_cast<vop>(i));
  }
  return out;
}

std::string name(const ::testing::TestParamInfo<param>& info) {
  const auto [t, prec, op] = info.param;
  return isa_test::label(t) + (prec == precision::f64 ? "_f64_" : "_f32_") +
         std::string(vop_names.at(static_cast<std::size_t>(op)));
}

INSTANTIATE_TEST_SUITE_P(isa, vocab,
                         ::testing::Combine(::testing::ValuesIn(isa_test::runnable_targets()),
                                            ::testing::Values(precision::f64, precision::f32),
                                            ::testing::ValuesIn(all_vops())),
                         name);

} // namespace
