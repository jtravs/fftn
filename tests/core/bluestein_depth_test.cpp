// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A Bluestein stage whose convolution runs as two depth-first chains (DESIGN.md 8.5,
// src/strategies/chain.hpp bluestein_depth; experiment X50), forced on short primes so that
// the quadruple-precision reference stays cheap: one and two top factors, every instruction
// set the machine runs, double and float, both directions, the scale on the chirp pass out,
// batches with padded distances, and in place.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;
using fftn::detail::strategy_choice;

template <class T> using cvec = std::vector<std::complex<T>>;

bool depth_convolution(const std::string& text) {
  return text.find("two depth-first chains") != std::string::npos;
}

// A batch of `count` rows of the prime q, `dist` apart (the gaps are canaries), through the
// forced depth-first convolution with the given tops (empty: the default scan) where it
// fits; every row against the reference, every gap untouched. Returns whether it was built.
template <class T>
bool check_prime(fftn::isa target, std::size_t q, const std::vector<std::size_t>& tops,
                 fftn::direction dir, double scale, std::size_t count, std::size_t dist,
                 bool in_place = false, bool pow2 = false) {
  const std::string what =
      std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
      ", q = " + std::to_string(q) + ", " + std::to_string(tops.size()) + " tops given" +
      (dir == fftn::direction::forward ? ", forward" : ", backward") +
      (in_place ? ", in place" : "") + ", scale " + std::to_string(scale) + ", " +
      std::to_string(count) + " rows " + std::to_string(dist) + " apart";
  SCOPED_TRACE(what);
  build_settings settings;
  settings.strategy = strategy_choice::depth;
  settings.depth_tops = tops;
  settings.rader = 0; // Bluestein for every prime
  if (pow2) {
    // Any power-of-two part: the convolution of 4093 is then 2^13 on every target.
    settings.profile.bluestein_pow2_part_bytes = SIZE_MAX;
  }
  const fftn::shape<std::complex<T>, std::complex<T>> shape{
      .dir = dir,
      .axes = {{.n = q}},
      .batches =
          {
              {
                  .count = count,
                  .in_dist = static_cast<std::ptrdiff_t>(dist),
                  .out_dist = static_cast<std::ptrdiff_t>(dist),
              },
          },
  };
  const auto core =
      make_plan_core(fftn::detail::view_of(shape),
                     {.in_place = in_place, .scale = scale, .force_isa = target}, settings);
  const std::string text = fftn::detail::describe(*core);
  EXPECT_NE(text.find("bluestein " + std::to_string(q)), std::string::npos) << text;

  fftn::test::input_source src(0xB1D3ULL + q + count);
  const std::size_t total = dist * (count - 1) + q;
  const cvec<T> x = src.complexes<T>(total);
  const std::complex<T> canary(T(-7.25), T(3.5));
  cvec<T> y(total, canary);
  if (in_place) {
    y = x;
  }
  const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  fftn::detail::execute_core(*core, 0, count, in_place ? y.data() : x.data(), y.data(), ws);

  // A Bluestein stage's error is that of two transforms of M and the chirps around them.
  const auto tol = fftn::test::derived_tolerance<T>(4 * q, fftn::test::stable_fft_bound);
  for (std::size_t b = 0; b < count; ++b) {
    std::vector<std::complex<double>> xin(q);
    for (std::size_t j = 0; j < q; ++j) {
      xin[j] = std::complex<double>(x[b * dist + j]);
    }
    const auto exact = fftn::test::reference::c2c(static_cast<int>(dir), xin);
    std::vector<fftn::test::exact_complex> want(q);
    for (std::size_t k = 0; k < q; ++k) {
      want[k] =
          fftn::test::exact_from(std::complex<double>(exact[k].re.hi, exact[k].im.hi) * scale);
    }
    fftn::test::outcome o;
    o.within(fftn::test::compare<T>(std::span<const std::complex<T>>(y.data() + b * dist, q), want),
             tol, what + ", row " + std::to_string(b));
    EXPECT_TRUE(o.passed()) << o.detail() << text;
    if (b + 1 < count) {
      for (std::size_t j = q; j < dist; ++j) {
        const std::complex<T> expect = in_place ? x[b * dist + j] : canary;
        EXPECT_EQ(y[b * dist + j], expect) << "gap element " << j << " of row " << b;
      }
    }
  }
  return depth_convolution(text);
}

template <class T> void every_case(fftn::isa target) {
  // Primes whose convolution lengths (the profile's choice, 1000 to 9000 points) have splits
  // on every complex count; the default scan picks the tops.
  std::size_t built = 0;
  for (const std::size_t q : {std::size_t{1031}, std::size_t{2053}, std::size_t{4099}}) {
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      built += check_prime<T>(target, q, {}, dir, 1.0, 1, q) ? 1 : 0;
    }
  }
  EXPECT_GT(built, 0U) << "no prime took the depth-first convolution";
  // The scale on the chirp pass out, batches with gaps, in place.
  check_prime<T>(target, 2053, {}, fftn::direction::backward, 0.25, 2, 2053 + 3);
  check_prime<T>(target, 1031, {}, fftn::direction::forward, 1.0 / 3, 3, 1031 + 5, true);
}

TEST(BluesteinDepth, EveryTargetPrecisionAndDirection) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    every_case<double>(t);
    every_case<float>(t);
  }
}

// Tops given: one and two factors, on a convolution of 2^13, which 8 lanes divide either way.
TEST(BluesteinDepth, GivenTops) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::vector<std::size_t>& tops :
         {std::vector<std::size_t>{16}, std::vector<std::size_t>{8, 4},
          std::vector<std::size_t>{8, 8}}) {
      EXPECT_TRUE(
          check_prime<double>(t, 4093, tops, fftn::direction::forward, 1.0, 1, 4093, false, true));
      EXPECT_TRUE(
          check_prime<float>(t, 4093, tops, fftn::direction::backward, 0.5, 1, 4093, true, true));
    }
  }
}

} // namespace
