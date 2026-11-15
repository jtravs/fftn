// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Every vocabulary operation of every runnable target, in both precisions, against a
// scalar reference written here (not the scalar target: a shared mistake would pass).
//
// Every operation is compared bit for bit on random data. The fused operations are
// compared with std::fma or with multiply-then-add, as the target declares, on data that
// includes a case where the two differ. The complex products (mul, mulc, madd) round
// differently on each target, so their reference is that target's own order of roundings
// (reference.hpp), and the data is checked to tell it from the unfused order. A lost FMA
// therefore fails, as does any wrong sign, part or lane.

#include "isa/kernel_table.hpp"
#include "isa/probe_table.hpp"
#include "isa/reference.hpp"
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

// Reals per array: a multiple of 2 C^2 (128 for avx512 float, whose 8 x 8 transpose takes
// the whole array) and of 2 W (32).
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
  const bool products = op == vop::c_mul || op == vop::c_mulc || op == vop::c_madd ||
                        op == vop::c_cmul || op == vop::c_mul_bcast;
  operand<T> a = random_reals<T>(gen, n);
  operand<T> b = random_reals<T>(gen, n);
  operand<T> c = random_reals<T>(gen, n);
  T r = random_reals<T>(gen, 1)[0];
  const bool fused = kernels.fused;
  const std::size_t lanes = of<T>(kernels).complex_lanes;
  const std::size_t width = of<T>(kernels).real_lanes;

  // Fused and unfused results differ in the first element (and complex number).
  const T u = tiny<T>();
  if (!products) {
    a[0] = a[1] = T(1) + u;
    b[0] = T(1) - u;
    c[0] = c[1] = op == vop::r_fmadd || op == vop::c_fmadd_r ? T(-1) : T(1);
  }
  if (op == vop::c_fmadd_r) {
    r = T(1) - u;
  }

  std::vector<T> e(n);
  std::size_t differ_from_unfused = 0;
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
    case vop::c_mulc:
    case vop::c_madd:
    case vop::c_cmul:
    case vop::c_mul_bcast: {
      const isa_test::cnum<T> x{ar, ai};
      // c_mul_bcast: every lane of register g takes twiddle g, complex number g of b.
      const std::size_t g = op == vop::c_mul_bcast ? 2 * (i / (2 * lanes)) : re;
      const isa_test::cnum<T> w{b[g], b[g + 1]};
      const isa_test::cnum<T> acc{c[re], c[re + 1]};
      const bool conj_w = op == vop::c_mulc;
      const auto [pr, pi] = op == vop::c_madd ? isa_test::multiply_add(kernels.id, acc, x, w)
                                              : isa_test::product(kernels.id, x, w, conj_w);
      const auto [ur, ui] = op == vop::c_madd ? isa_test::multiply_add(target::scalar, acc, x, w)
                                              : isa_test::product(target::scalar, x, w, conj_w);
      e[i] = odd ? pi : pr;
      if (odd ? pi != ui : pr != ur) {
        ++differ_from_unfused;
      }
      break;
    }
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
    case vop::c_load_n:  // the first g mod (C + 1) complex numbers of register g, zeros after
    case vop::c_store_n: // the same, the canary after
    {
      const std::size_t g = i / (2 * lanes);
      const bool live = (i / 2) % lanes < g % (lanes + 1);
      e[i] = live ? a[i] : op == vop::c_load_n ? T(0) : T(-77);
      break;
    }
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
  // twiddle storage, which tests pack_twiddles and load_tw (or pack_bcast_twiddles and
  // bcast_tw) too. cmul takes b as it is.
  const std::size_t tw_size = of<T>(kernels).tw_reals * half;
  operand<T> tw(tw_size);
  if (op == vop::c_mul_bcast) {
    of<T>(kernels).pack_bcast_twiddles(b.data(), half, tw.data());
  } else {
    of<T>(kernels).pack_twiddles(b.data(), half, tw.data());
  }
  const T* second = products && op != vop::c_cmul ? tw.data() : b.data();
  if (products && kernels.fused) {
    EXPECT_GT(differ_from_unfused, 0U)
        << "the data cannot tell a fused product from an unfused one";
  }

  operand<T> got(n);
  if constexpr (sizeof(T) == sizeof(double)) {
    probes.f64(op, a.data(), second, c.data(), got.data(), n, &r);
  } else {
    probes.f32(op, a.data(), second, c.data(), got.data(), n, &r);
  }
  EXPECT_EQ(tw.buffer.front(), T(-77)) << "packing twiddles wrote before its output";
  EXPECT_EQ(tw.buffer.back(), T(-77)) << "packing twiddles wrote past its output";
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
    if ((op == vop::c_load_n || op == vop::c_store_n) && !probes->masked) {
      GTEST_SKIP() << "no masked partial runs on this target";
    }
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
