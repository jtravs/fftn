// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Experiment X15 (dev/experiments/X15-compiler-vectorised-scalar.md): every arm this build
// contains gives results bitwise equal to plain scalar, for c2c rows in both directions,
// r2c and c2r rows, every length of fast_lengths() (naive_lengths() through the public
// API), double and float.
//
// Plain scalar is this file's own copy of the scalar kernel text, in a region of its own
// below. It is compiled with the tests' flags: never with -fopenmp-simd or FFTN_X_OMP_SIMD
// (cmake/FftnExperiments.cmake gives consumers neither), never with the -march=native of
// arm E. Each arm's table and the reference table run the same stages, built by the
// library's chain strategy and run by its driver, so only the passes differ:
//
// - the library's scalar table: arm A as merged, arm B with FFTN_X_OMP_SIMD, arm E with
//   FFTN_X_NATIVE;
// - the scalar text in the avx2 region (arm C, FFTN_X_AVX2_SCALAR_TEXT), where the
//   processor runs AVX2;
// - whatever a public plan takes, through the public API.
//
// Arms A, B and C must match bit for bit: their loops reorder no reduction and the build
// keeps -ffp-contract=off. Arm E is held to that too unless the compiler changed the bits;
// then, as the protocol says, the outputs must lie within the transform's error bound and
// the largest difference is printed for the experiment file.

#include "core/driver.hpp"
#include "core/program.hpp"
#include "fftn/fftn.hpp"
#include "isa/arch.hpp"
#include "isa/kernel_table.hpp"
#include "isa/region.hpp"
#include "strategies/chain.hpp"
#include "support/lengths.hpp"

#include <gtest/gtest.h>

#if FFTN_ARCH_X86_64 && defined(FFTN_X_AVX2_SCALAR_TEXT) && FFTN_X_AVX2_SCALAR_TEXT
#include "isa/target.hpp"
#include "platform/cpu_features.hpp"
#endif
#if defined(FFTN_X_NATIVE) && FFTN_X_NATIVE
#include <limits>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

// The reference: plain scalar (no attribute, the scalar vocabulary, the kernel text).
#define FFTN_ISA_PUSH_x15_reference FFTN_ISA_PUSH_NONE
FFTN_ISA_BEGIN(x15_reference)

#include "isa/vocab_scalar.inl"

#include "isa/kernels.inl"

#include "passes/chain.inl"

#include "isa/table.inl"

FFTN_ISA_END

namespace {

namespace fd = fftn::detail;

enum class kind : unsigned char { c2c_forward, c2c_backward, r2c, c2r };
constexpr kind all_kinds[] = // NOLINT(modernize-avoid-c-arrays): a constant list
    {kind::c2c_forward, kind::c2c_backward, kind::r2c, kind::c2r};

const char* kind_name(kind k) {
  switch (k) {
  case kind::c2c_forward:
    return "c2c forward";
  case kind::c2c_backward:
    return "c2c backward";
  case kind::r2c:
    return "r2c";
  case kind::c2r:
    break;
  }
  return "c2r";
}

// Reals in one transform's input and output.
std::size_t reals(bool real_side, bool half_spectrum, std::size_t n) {
  if (real_side) {
    return n;
  }
  return half_spectrum ? 2 * (n / 2 + 1) : 2 * n;
}
std::size_t input_reals(kind k, std::size_t n) { return reals(k == kind::r2c, k == kind::c2r, n); }
std::size_t output_reals(kind k, std::size_t n) { return reals(k == kind::c2r, k == kind::r2c, n); }

template <class T> std::vector<T> random_input(kind k, std::size_t n, std::size_t count) {
  std::mt19937_64 rng(1000003U * n + static_cast<unsigned>(k));
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<T> v(count * input_reals(k, n));
  for (T& x : v) {
    x = static_cast<T>(u(rng));
  }
  return v;
}

// One transform through the chain strategy's stages for `table`, run by the library's
// driver. Chunks of 256 points, so long stages run in several ranges.
template <class T>
std::vector<T> run_line(const fd::kernel_table& table, kind k, std::size_t n,
                        const std::vector<T>& input) {
  std::vector<std::vector<T>> tables;
  fd::chain_builder<T> chain(fd::of<T>(table), tables, 256);
  std::vector<fd::stage> stages;
  const fd::endpoint in{.where = fd::slot::in, .stride = 1};
  const fd::endpoint out{.where = fd::slot::out, .stride = 1};
  std::size_t need = 0;
  switch (k) {
  case kind::c2c_forward:
    need = chain.c2c_line(n, -1, in, out, false, stages);
    break;
  case kind::c2c_backward:
    need = chain.c2c_line(n, +1, in, out, false, stages);
    break;
  case kind::r2c:
    need = chain.r2c_line(n, in, out, stages);
    break;
  case kind::c2r:
    need = chain.c2r_line(n, in, out, stages);
    break;
  }
  fd::program p;
  for (fd::stage& st : stages) {
    p.records.emplace_back(std::move(st));
  }
  std::vector<T> result(output_reals(k, n));
  std::vector<T> a(2 * need + 2);
  std::vector<T> b(2 * need + 2);
  const fd::slot_addresses at{
      .in = reinterpret_cast<const std::byte*>(input.data()),
      .out = reinterpret_cast<std::byte*>(result.data()),
      .a = reinterpret_cast<std::byte*>(a.data()),
      .b = reinterpret_cast<std::byte*>(b.data()),
  };
  fd::run_transforms(p, 0, 1, at);
  return result;
}

// The arm's outputs against the reference's: true when bitwise equal. Otherwise records
// a failure, except under arm E within the bound, where it prints the difference.
template <class T>
bool same_bits(const std::vector<T>& arm, const std::vector<T>& reference, std::size_t n,
               const std::string& what) {
  if (arm.size() == reference.size() &&
      std::memcmp(arm.data(), reference.data(), arm.size() * sizeof(T)) == 0) {
    return true;
  }
  double diff = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < std::min(arm.size(), reference.size()); ++i) {
    diff =
        std::max(diff, std::abs(static_cast<double>(arm[i]) - static_cast<double>(reference[i])));
    scale = std::max(scale, std::abs(static_cast<double>(reference[i])));
  }
  const double relative = scale > 0.0 ? diff / scale : diff;
#if defined(FFTN_X_NATIVE) && FFTN_X_NATIVE
  // The fbench check's bound (bench/README.md), in this precision's epsilon.
  const double bound = 10.0 * std::numeric_limits<T>::epsilon() *
                       (1.0 + std::log2(static_cast<double>(std::max<std::size_t>(n, 2))));
  std::printf("X15 arm E: %s not bitwise equal; max |diff| / max |ref| = %.3g (bound %.3g)\n",
              what.c_str(), relative, bound);
  EXPECT_LE(relative, bound) << what;
#else
  static_cast<void>(n);
  ADD_FAILURE() << what
                << ": not bitwise equal to plain scalar; max |diff| / max |ref| = " << relative;
#endif
  return false;
}

template <class T> const char* precision_name() {
  return sizeof(T) == sizeof(double) ? "double" : "float";
}

template <class T> void compare_tables(const fd::kernel_table& arm, const char* arm_name) {
  std::size_t equal = 0;
  std::size_t total = 0;
  for (const kind k : all_kinds) {
    for (const std::size_t n : fftn::test::fast_lengths()) {
      const std::vector<T> input = random_input<T>(k, n, 1);
      const std::string what = std::string(arm_name) + ", " + kind_name(k) + ", " +
                               precision_name<T>() + ", n = " + std::to_string(n);
      equal += same_bits(run_line(arm, k, n, input),
                         run_line(fd::x15_reference::table, k, n, input), n, what)
                   ? 1
                   : 0;
      ++total;
    }
  }
  std::printf("X15 %s, %s: %zu of %zu transforms bitwise equal to plain scalar\n", arm_name,
              precision_name<T>(), equal, total);
}

TEST(X15Arms, BuildSwitches) {
  std::printf("X15 library switches: %s\n", FFTN_X15_LIBRARY_SWITCHES);
  RecordProperty("x15_switches", FFTN_X15_LIBRARY_SWITCHES);
}

TEST(X15Arms, LibraryScalarTableMatchesPlainScalar) {
  compare_tables<double>(fd::scalar::table, "library scalar table");
  compare_tables<float>(fd::scalar::table, "library scalar table");
}

TEST(X15Arms, Avx2ScalarTextMatchesPlainScalar) {
#if FFTN_ARCH_X86_64 && defined(FFTN_X_AVX2_SCALAR_TEXT) && FFTN_X_AVX2_SCALAR_TEXT
  if (!fd::runs_on(fd::target::avx2, fd::detect_cpu_features())) {
    GTEST_SKIP() << "this processor cannot run the avx2 region";
  }
  compare_tables<double>(fd::avx2::scalar_text::table, "scalar text in the avx2 region");
  compare_tables<float>(fd::avx2::scalar_text::table, "scalar text in the avx2 region");
#else
  GTEST_SKIP() << "arm C (FFTN_X_AVX2_SCALAR_TEXT on x86-64) is not in this build";
#endif
}

// Batches of three rows through the public API, whatever table the plan takes: the shorter
// list, since the tables themselves are compared over every length above.
template <class T> void compare_plans() {
  constexpr std::size_t count = 3;
  for (const kind k : all_kinds) {
    for (const std::size_t n : fftn::test::naive_lengths()) {
      const std::vector<T> input = random_input<T>(k, n, count);
      std::vector<T> result(count * output_reals(k, n));
      if (k == kind::r2c) {
        const fftn::r2c_plan<T> p(fftn::rows_r2c<T>(n, count));
        fftn::workspace ws(p.workspace_bytes());
        p.execute(input.data(), reinterpret_cast<std::complex<T>*>(result.data()), ws);
      } else if (k == kind::c2r) {
        const fftn::c2r_plan<T> p(fftn::rows_c2r<T>(n, count));
        fftn::workspace ws(p.workspace_bytes());
        p.execute(reinterpret_cast<const std::complex<T>*>(input.data()), result.data(), ws);
      } else {
        const fftn::c2c_plan<T> p(fftn::rows_c2c<T>(
            n, count,
            k == kind::c2c_forward ? fftn::direction::forward : fftn::direction::backward));
        fftn::workspace ws(p.workspace_bytes());
        p.execute(reinterpret_cast<const std::complex<T>*>(input.data()),
                  reinterpret_cast<std::complex<T>*>(result.data()), ws);
      }
      std::vector<T> expected;
      for (std::size_t r = 0; r < count; ++r) {
        const auto first = static_cast<std::ptrdiff_t>(r * input_reals(k, n));
        const std::vector<T> row(input.begin() + first,
                                 input.begin() + first +
                                     static_cast<std::ptrdiff_t>(input_reals(k, n)));
        const std::vector<T> y = run_line(fd::x15_reference::table, k, n, row);
        expected.insert(expected.end(), y.begin(), y.end());
      }
      same_bits(result, expected, n,
                std::string("public plan, ") + kind_name(k) + ", " + precision_name<T>() +
                    ", n = " + std::to_string(n));
    }
  }
}

TEST(X15Arms, PlansMatchPlainScalar) {
  compare_plans<double>();
  compare_plans<float>();
}

} // namespace
