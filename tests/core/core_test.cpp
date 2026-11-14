// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The plan core's internals: twiddle tables entry by entry against the quadruple-precision
// reference, the chain's radices, and the fixed division of stages into chunks.

#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "core/twiddles.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "isa/target.hpp"
#include "planner/rules.hpp"
#include "reference/reference_checks.hpp"
#include "support/inputs.hpp"
#include "support/lengths.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstring>
#include <set>
#include <variant>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;
using fftn::detail::shape_view;

// Every length the suites test, and the lengths of the planes' axes.
std::set<std::size_t> tested_lengths() {
  std::set<std::size_t> v;
  for (const std::size_t n : fftn::test::fast_lengths()) {
    v.insert(n);
  }
  for (const auto& [ny, nx] : fftn::test::plane_shapes()) {
    v.insert(ny);
    v.insert(nx);
  }
  return v;
}

// DESIGN.md 15.2: a single table entry one ulp off is invisible in transform outputs, so
// every table is checked here, entry by entry, for every length the suites run.
template <class T> void check_tables() {
  std::size_t failures = 0;
  for (const std::size_t n : tested_lengths()) {
    for (const int sign : {-1, +1}) {
      const auto table = fftn::detail::unit_roots<T>(sign, n);
      const auto o = fftn::test::reference_checks::twiddle_table<T>(table, sign, n);
      if (!o.passed() && ++failures <= 4) {
        ADD_FAILURE() << "n = " << n << ", sign " << sign << ":\n" << o.detail();
      }
    }
  }
  EXPECT_EQ(failures, 0U);
}

TEST(Twiddles, CorrectlyRoundedInDouble) { check_tables<double>(); }
TEST(Twiddles, CorrectlyRoundedInFloat) { check_tables<float>(); }

// The octant boundaries are exact: 1, i, -1, -i and the diagonals.
TEST(Twiddles, ExactAtOctantBoundaries) {
  const auto w = fftn::detail::unit_roots<double>(-1, 8);
  const double h = 0.70710678118654752440084436210484903928;
  EXPECT_EQ(w[0], std::complex<double>(1, 0));
  EXPECT_EQ(w[1], std::complex<double>(h, -h));
  EXPECT_EQ(w[2], std::complex<double>(0, -1));
  EXPECT_EQ(w[4], std::complex<double>(-1, 0));
  EXPECT_EQ(w[6], std::complex<double>(0, 1));
  EXPECT_EQ(w[7], std::complex<double>(h, h));
}

bool is_prime(std::size_t n) {
  for (std::size_t d = 2; d <= n / d; ++d) {
    if (n % d == 0) {
      return false;
    }
  }
  return n >= 2;
}

// Planner v0's radices multiply to the length, and reproduce the published examples of the
// rules they come from (FFTL.jl, single lines): on 32 registers with merged radices
// (AArch64) and on 16 without (AVX2).
TEST(Rules, RadicesMultiplyToTheLength) {
  for (const auto t : {
           fftn::detail::target::scalar,
           fftn::detail::target::avx2,
           fftn::detail::target::neon_fcma,
       }) {
    const auto profile = fftn::detail::profile_of(t);
    for (std::size_t n = 1; n <= 4096; ++n) {
      std::size_t product = 1;
      for (const std::size_t r : fftn::detail::rule_radices(n, profile)) {
        product *= r;
        EXPECT_TRUE(r <= 16 || is_prime(r)) << n << ": radix " << r;
      }
      EXPECT_EQ(product, n) << n;
    }
    EXPECT_TRUE(fftn::detail::rule_radices(1, profile).empty());
    EXPECT_EQ(fftn::detail::rule_radices(10007, profile), (std::vector<std::size_t>{10007}));
    EXPECT_EQ(fftn::detail::rule_radices(16, profile), (std::vector<std::size_t>{16}));
    EXPECT_EQ(fftn::detail::rule_radices(32, profile), (std::vector<std::size_t>{4, 8}));
    EXPECT_EQ(fftn::detail::rule_radices(64, profile), (std::vector<std::size_t>{4, 16}));
  }
}

TEST(Rules, PublishedExamples) {
  using v = std::vector<std::size_t>;
  const fftn::detail::target_profile wide{.registers = 32, .merge = true};
  const fftn::detail::target_profile narrow{.registers = 16, .merge = false};
  const auto r = [](std::size_t n, const fftn::detail::target_profile& p) {
    return fftn::detail::rule_radices(n, p);
  };
  EXPECT_EQ(r(128, wide), (v{4, 4, 8}));
  EXPECT_EQ(r(128, narrow), (v{8, 4, 4}));
  EXPECT_EQ(r(256, wide), (v{4, 8, 8}));
  EXPECT_EQ(r(256, narrow), (v{4, 4, 4, 4}));
  EXPECT_EQ(r(1024, wide), (v{4, 4, 8, 8}));
  EXPECT_EQ(r(1024, narrow), (v{4, 4, 4, 4, 4}));
  EXPECT_EQ(r(4096, wide), (v{8, 8, 8, 8}));
  EXPECT_EQ(r(4096, narrow), (v(6, 4)));
  EXPECT_EQ(r(16384, wide), (v{4, 8, 8, 8, 8}));
  EXPECT_EQ(r(16384, narrow), (v(7, 4)));
  EXPECT_EQ(r(32768, wide), (v{8, 4, 4, 4, 4, 4, 4}));
  EXPECT_EQ(r(32768, narrow), (v{8, 4, 4, 4, 4, 4, 4}));
  EXPECT_EQ(r(48, wide), (v{4, 12}));
  EXPECT_EQ(r(48, narrow), (v{4, 4, 3}));
  EXPECT_EQ(r(120, wide), (v{10, 12}));
  EXPECT_EQ(r(120, narrow), (v{4, 2, 3, 5}));
  EXPECT_EQ(r(360, wide), (v{5, 6, 12}));
  EXPECT_EQ(r(360, narrow), (v{4, 2, 5, 9}));
  EXPECT_EQ(r(1000, wide), (v{4, 5, 5, 10}));
  EXPECT_EQ(r(1000, narrow), (v{4, 2, 5, 5, 5}));
  EXPECT_EQ(r(100000, wide), (v{4, 4, 5, 5, 5, 5, 10}));
  EXPECT_EQ(r(100000, narrow), (v{4, 4, 2, 5, 5, 5, 5, 5}));
  // Primes above the codelets go last, as the largest radices; 7 n keeps a 2 on x86.
  EXPECT_EQ(r(34, narrow), (v{2, 17}));
  EXPECT_EQ(r(56, narrow), (v{4, 2, 7}));
  EXPECT_EQ(r(56, wide), (v{8, 7}));
}

// Runs a plan core on a fresh copy of `input` and returns the output bytes.
std::vector<std::byte> run(const fftn::detail::plan_core& core, std::vector<std::byte> input,
                           std::size_t out_bytes) {
  std::vector<std::byte> output(out_bytes);
  const fftn::workspace ws(fftn::detail::workspace_bytes(core));
  fftn::detail::execute_core(core, 0, fftn::detail::transform_count(core), input.data(),
                             output.data(), ws);
  return output;
}

template <class In, class Out>
void check_chunking(const fftn::shape<In, Out>& s, std::size_t in_count, std::size_t out_count) {
  const shape_view view = fftn::detail::view_of(s);
  build_settings coarse;
  coarse.chunk_points = std::size_t{1} << 30U;
  build_settings tiny;
  tiny.chunk_points = 3;
  const fftn::options defaults;
  const auto whole = make_plan_core(view, defaults, coarse);
  const auto fine = make_plan_core(view, defaults, tiny);
  // The fine plan really is divided.
  std::size_t chunked = 0;
  for (const auto& rec : fine->prog.records) {
    if (const auto* st = std::get_if<fftn::detail::stage>(&rec); st != nullptr) {
      chunked += st->chunks > 1 ? 1 : 0;
    }
  }
  EXPECT_GT(chunked, 0U);
  fftn::test::input_source src(in_count);
  const auto values = src.reals<double>(in_count * sizeof(In) / sizeof(double) + 1);
  std::vector<std::byte> input(in_count * sizeof(In));
  std::memcpy(input.data(), values.data(), input.size());
  const auto a = run(*whole, input, out_count * sizeof(Out));
  const auto b = run(*fine, input, out_count * sizeof(Out));
  EXPECT_EQ(a, b) << "chunking changed the result";
}

// The chunks of a stage are independent ranges fixed by the plan: any division gives the
// same bits (DESIGN.md 13.2), which is what lets phase 3 share them between threads.
TEST(Chunks, DivisionDoesNotChangeBits) {
  check_chunking(fftn::rows_c2c<double>(1260, 2, fftn::direction::forward), 2520, 2520);
  check_chunking(fftn::rows_c2c<float>(2018, 1, fftn::direction::backward), 2018, 2018);
  check_chunking(fftn::rows_r2c<double>(720, 2), 1440, 722);
  check_chunking(fftn::rows_r2c<float>(45, 1), 45, 23);
  check_chunking(fftn::rows_c2r<double>(720, 1), 361, 720);
  check_chunking(fftn::rows_c2r<float>(45, 2), 46, 90);
  check_chunking(fftn::planes_c2c<double>(36, 20, 2, fftn::direction::forward), 1440, 1440);
}

// Long rows are divided by default, so the phase 3 engine has chunks to share.
TEST(Chunks, LongRowsHaveSeveralByDefault) {
  const auto core = make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<double>(16384, 1, fftn::direction::forward)), {});
  for (const auto& rec : core->prog.records) {
    if (const auto* st = std::get_if<fftn::detail::stage>(&rec); st != nullptr) {
      EXPECT_GT(st->chunks, 1U) << st->name;
    }
  }
}

} // namespace
