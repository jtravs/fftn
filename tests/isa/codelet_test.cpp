// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The generated codelets (src/gen/) compiled in every target's region, against the
// quadruple-precision reference DFT: every codelet of the metadata table, every runnable
// target, both precisions, out of place with strides and in place. Each lane of a register
// carries its own random transform, so a mixed-up lane fails too.
//
// Tolerances are derived before the run (tests/support/tolerance.hpp): a codelet is an
// R-point FFT built from stable butterflies, so stable_fft_bound at length R. A twiddle
// variant rounds one more product per value, and its reference rounds the twiddle product
// once in double, so it adds input_rounding<T>() twice. A wrong constant, sign, rotation,
// leg or lane is off by O(1). (The generator has already checked each operation list
// exactly; this checks the emitted C++ and the vocabulary under it.)

#include "gen/codelets.hpp"
#include "isa/codelet_table.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "isa/test_targets.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

namespace gen = fftn::detail::gen;
using fftn::detail::codelet_table;
using fftn::detail::find_codelets;
using fftn::detail::find_kernels;
using fftn::detail::kernel_table;
using fftn::detail::of;
using fftn::detail::target;
using fftn::test::exact_complex;
using cd = std::complex<double>;

enum class precision : unsigned char { f64, f32 };
using param = std::tuple<target, precision>;

constexpr double canary = -77.0;

std::string describe(const gen::codelet_info& c) {
  constexpr std::array<const char*, 3> kinds = {"plain", "dit", "dif"};
  return "radix " + std::to_string(c.radix) + " " + kinds.at(static_cast<std::size_t>(c.kind)) +
         (c.sign < 0 ? " forward" : " backward");
}

struct layout {
  std::size_t radix;
  std::size_t lanes;  // C
  std::size_t stride; // complex numbers between legs
  [[nodiscard]] std::size_t reals() const { return 2 * (stride * (radix - 1) + lanes) + 2; }
  // Offset (in T) of lane l of leg j, past the leading canary.
  [[nodiscard]] std::size_t at(std::size_t j, std::size_t l) const {
    return 1 + 2 * (stride * j + l);
  }
};

template <class T> const fftn::detail::codelet_runs<T>& runs_of(const codelet_table& t) {
  if constexpr (std::is_same_v<T, double>) {
    return t.f64;
  } else {
    return t.f32;
  }
}

// Runs codelet c of target table t once and checks every lane.
template <class T>
void check_codelet(const codelet_table& t, const kernel_table& k, std::size_t index,
                   bool in_place) {
  const gen::codelet_info& c = gen::codelets[index];
  const std::size_t r = c.radix;
  const std::size_t lanes = of<T>(k).complex_lanes;
  const layout in{.radix = r, .lanes = lanes, .stride = lanes + (in_place ? 0 : 1)};
  const layout out{.radix = r, .lanes = lanes, .stride = in_place ? in.stride : lanes + 3};
  std::mt19937 gen(static_cast<unsigned>(index * 2 + (in_place ? 1 : 0)));
  std::uniform_real_distribution<double> uniform(-1, 1);
  std::uniform_real_distribution<double> angle(0, 2 * std::numbers::pi);

  std::vector<T> x(in.reals(), T(canary));
  std::vector<cd> data(r * lanes);
  for (std::size_t j = 0; j < r; ++j) {
    for (std::size_t l = 0; l < lanes; ++l) {
      const auto re = static_cast<T>(uniform(gen));
      const auto im = static_cast<T>(uniform(gen));
      x[in.at(j, l)] = re;
      x[in.at(j, l) + 1] = im;
      data[j * lanes + l] = {static_cast<double>(re), static_cast<double>(im)};
    }
  }
  // Forward twiddles of arbitrary angle, rounded to T: leg-major, lane-minor, then packed
  // into the target's storage C at a time.
  std::vector<T> twiddles(2 * (r - 1) * lanes);
  for (std::size_t q = 0; q < (r - 1) * lanes; ++q) {
    const double a = angle(gen);
    twiddles[2 * q] = static_cast<T>(std::cos(a));
    twiddles[2 * q + 1] = static_cast<T>(std::sin(a));
  }
  std::vector<T> storage(std::max<std::size_t>(1, of<T>(k).tw_reals * (r - 1) * lanes));
  if (r > 1) {
    of<T>(k).pack_twiddles(twiddles.data(), (r - 1) * lanes, storage.data());
  }

  std::vector<T> y_buffer(out.reals(), T(canary));
  const std::vector<T> x0 = x;
  std::vector<T>& y_all = in_place ? x : y_buffer;
  runs_of<T>(t).run[index](x.data() + 1, in.stride, y_all.data() + 1, out.stride, storage.data());

  fftn::test::tolerance tol = fftn::test::derived_tolerance<T>(r, fftn::test::stable_fft_bound);
  if (c.kind != gen::variant::plain) {
    tol = tol + fftn::test::input_rounding<T>() * 2.0;
  }
  for (std::size_t l = 0; l < lanes; ++l) {
    SCOPED_TRACE("lane " + std::to_string(l));
    const auto tw_of = [&](std::size_t leg) {
      const cd w{static_cast<double>(twiddles[2 * ((leg - 1) * lanes + l)]),
                 static_cast<double>(twiddles[2 * ((leg - 1) * lanes + l) + 1])};
      return c.sign < 0 ? w : std::conj(w);
    };
    std::vector<cd> z(r);
    for (std::size_t j = 0; j < r; ++j) {
      z[j] = data[j * lanes + l];
      if (c.kind == gen::variant::dit && j > 0) {
        z[j] *= tw_of(j);
      }
    }
    std::vector<exact_complex> want = fftn::test::reference::c2c(c.sign, z);
    if (c.kind == gen::variant::dif) {
      for (std::size_t q = 1; q < r; ++q) {
        const cd v = cd{want[q].re.hi, want[q].im.hi} * tw_of(q);
        want[q] = fftn::test::exact_from(v);
      }
    }
    double err2 = 0;
    double ref2 = 0;
    double err_max = 0;
    double ref_max = 0;
    for (std::size_t q = 0; q < r; ++q) {
      const double er = fftn::test::error_of(static_cast<double>(y_all[out.at(q, l)]), want[q].re);
      const double ei =
          fftn::test::error_of(static_cast<double>(y_all[out.at(q, l) + 1]), want[q].im);
      const double e = std::hypot(er, ei);
      const double m = fftn::test::magnitude(want[q]);
      err2 += e * e;
      ref2 += m * m;
      err_max = std::max(err_max, e);
      ref_max = std::max(ref_max, m);
    }
    EXPECT_LE(std::sqrt(err2 / ref2), tol.rms) << "relative RMS error";
    EXPECT_LE(err_max / ref_max, tol.max) << "largest error";
  }
  EXPECT_EQ(y_all.front(), T(canary)) << "wrote before the output";
  EXPECT_EQ(y_all.back(), T(canary)) << "wrote after the output";
  if (!in_place) {
    EXPECT_EQ(x, x0) << "input changed";
    // The gaps between legs are untouched.
    for (std::size_t q = 0; q + 1 < r; ++q) {
      for (std::size_t g = out.at(q, lanes); g < out.at(q + 1, 0); ++g) {
        EXPECT_EQ(y_buffer[g], T(canary)) << "wrote between legs " << q << " and " << q + 1;
      }
    }
  }
}

class codelets : public ::testing::TestWithParam<param> {
protected:
  template <class T> void check() {
    const target t = std::get<0>(GetParam());
    const codelet_table* table = find_codelets(t);
    const kernel_table* kernels = find_kernels(t);
    ASSERT_NE(table, nullptr);
    ASSERT_NE(kernels, nullptr);
    EXPECT_EQ(table->id, t);
    for (std::size_t i = 0; i < gen::codelet_count; ++i) {
      SCOPED_TRACE(describe(gen::codelets[i]));
      check_codelet<T>(*table, *kernels, i, false);
      check_codelet<T>(*table, *kernels, i, true);
    }
  }
};

TEST_P(codelets, match_reference_dft) {
  if (std::get<1>(GetParam()) == precision::f64) {
    check<double>();
  } else {
    check<float>();
  }
}

std::string name(const ::testing::TestParamInfo<param>& info) {
  const auto [t, prec] = info.param;
  return isa_test::label(t) + (prec == precision::f64 ? "_f64" : "_f32");
}

INSTANTIATE_TEST_SUITE_P(gen, codelets,
                         ::testing::Combine(::testing::ValuesIn(isa_test::runnable_targets()),
                                            ::testing::Values(precision::f64, precision::f32)),
                         name);

// The table lists every radix in every variant and direction exactly once, in order, and
// its counts are consistent with each other.
TEST(codelet_metadata, complete_and_consistent) {
  std::set<std::tuple<int, int, int>> seen;
  for (const gen::codelet_info& c : std::span(gen::codelets)) {
    SCOPED_TRACE(describe(c));
    EXPECT_TRUE(seen.insert({c.radix, static_cast<int>(c.kind), c.sign}).second) << "listed twice";
    EXPECT_EQ(c.twiddles, c.kind == gen::variant::plain ? 0 : c.radix - 1);
    EXPECT_EQ(c.cmuls, c.twiddles);
    EXPECT_EQ(c.forced, 0) << "an output needed an extra rotation";
    EXPECT_EQ(c.adds % 2, 0);
    EXPECT_GE(c.live, c.radix) << "fewer live values than legs";
    EXPECT_GT(c.vops, 0);
  }
  EXPECT_EQ(seen.size(), gen::radix_count * 3 * 2);
  for (const std::uint8_t r : std::span(gen::radices)) {
    for (int kind = 0; kind < 3; ++kind) {
      for (const int sign : {-1, 1}) {
        EXPECT_TRUE(seen.contains({r, kind, sign})) << "radix " << int{r} << " missing";
      }
    }
  }
  EXPECT_TRUE(std::ranges::is_sorted(gen::radices));
}

} // namespace
