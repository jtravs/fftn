// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The depth-first chain (DESIGN.md 8.5, src/strategies/chain.hpp c2c_depth, src/passes/
// depth.inl; experiment X46) forced on short rows so that the quadruple-precision reference
// stays cheap: one and two top factors, power-of-two and smooth sub-rows, sub-row chains of
// two to four passes (the copy back, the alternation with scratch a, the detour through b),
// every instruction set the machine runs, double and float, both directions, the scale on
// the top pass, batches with padded distances and in place (through the workspace block);
// and where it cannot run (complex counts that do not divide the factors) the row stays a
// chain.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "isa/kernel_table.hpp"
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
using fftn::detail::plan_core;
using fftn::detail::strategy_choice;

template <class T> using cvec = std::vector<std::complex<T>>;

struct depth_case {
  std::size_t n;
  std::vector<std::size_t> tops;
  std::vector<std::size_t> sub; // the sub-rows' radices (empty: the rules')
};

// A batch of `count` rows of n, `dist` apart (the gaps are canaries), through the forced
// depth-first chain where it fits; every row against the reference, every gap untouched.
template <class T>
void check_depth(fftn::isa target, const depth_case& dc, fftn::direction dir, double scale,
                 std::size_t count, std::size_t dist, bool in_place = false) {
  const std::size_t n = dc.n;
  std::string tops;
  for (const std::size_t f : dc.tops) {
    tops += std::to_string(f) + " x ";
  }
  const std::string what =
      std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
      ", n = " + std::to_string(n) + " = " + tops + "..." +
      (dir == fftn::direction::forward ? " forward" : " backward") +
      (in_place ? ", in place" : "") + ", scale " + std::to_string(scale) + ", " +
      std::to_string(count) + " rows " + std::to_string(dist) + " apart";
  SCOPED_TRACE(what);
  build_settings settings;
  settings.strategy = strategy_choice::depth;
  settings.depth_tops = dc.tops;
  settings.radices = dc.sub;
  const fftn::shape<std::complex<T>, std::complex<T>> shape{
      .dir = dir,
      .axes = {{.n = n}},
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
  const std::size_t lanes =
      fftn::detail::of<T>(*fftn::detail::find_kernels(core->isa_target)).complex_lanes;
  std::size_t subs = 1;
  for (const std::size_t f : dc.tops) {
    subs *= f;
  }
  const bool fits = dc.tops[0] % lanes == 0 && (n / subs) % lanes == 0;
  EXPECT_EQ(text.find("strategy: depth-first chain") != std::string::npos, fits) << text;

  fftn::test::input_source src(0xDE97ULL + n + count);
  const std::size_t total = dist * (count - 1) + n;
  const cvec<T> x = src.complexes<T>(total);
  const std::complex<T> canary(T(-7.25), T(3.5));
  cvec<T> y(total, canary);
  if (in_place) {
    y = x;
  }
  const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  fftn::detail::execute_core(*core, 0, count, in_place ? y.data() : x.data(), y.data(), ws);

  const auto tol = fftn::test::derived_tolerance<T>(n, fftn::test::stable_fft_bound);
  for (std::size_t b = 0; b < count; ++b) {
    std::vector<std::complex<double>> xin(n);
    for (std::size_t j = 0; j < n; ++j) {
      xin[j] = std::complex<double>(x[b * dist + j]);
    }
    const auto exact = fftn::test::reference::c2c(static_cast<int>(dir), xin);
    std::vector<fftn::test::exact_complex> want(n);
    for (std::size_t k = 0; k < n; ++k) {
      want[k] =
          fftn::test::exact_from(std::complex<double>(exact[k].re.hi, exact[k].im.hi) * scale);
    }
    fftn::test::outcome o;
    o.within(fftn::test::compare<T>(std::span<const std::complex<T>>(y.data() + b * dist, n), want),
             tol, what + ", row " + std::to_string(b));
    EXPECT_TRUE(o.passed()) << o.detail() << text;
    if (b + 1 < count) {
      for (std::size_t j = n; j < dist; ++j) {
        const std::complex<T> expect = in_place ? x[b * dist + j] : canary;
        EXPECT_EQ(y[b * dist + j], expect) << "gap element " << j << " of row " << b;
      }
    }
  }
}

// Top factors that every target's complex count (1 to 8) divides, except where a case says.
const std::vector<depth_case>& cases() {
  static const std::vector<depth_case> all{
      {.n = 512, .tops = {8}, .sub = {}},
      {.n = 2048, .tops = {16}, .sub = {}},
      {.n = 960, .tops = {8}, .sub = {}},            // 3 and 5 in the sub-rows' chains
      {.n = 1024, .tops = {8}, .sub = {8, 16}},      // one pass after the first: a copy back
      {.n = 1024, .tops = {8}, .sub = {4, 4, 8}},    // two: with scratch a
      {.n = 2048, .tops = {8}, .sub = {4, 4, 4, 4}}, // three: once through b
      {.n = 1024, .tops = {8}, .sub = {4, 32}},      // a radix-32 pass in the sub-rows
      {.n = 2048, .tops = {8, 4}, .sub = {}},        // two top factors
      {.n = 1920, .tops = {8, 5}, .sub = {}},        // an odd inner top factor
      {.n = 1280, .tops = {16, 2}, .sub = {5, 8}},   // a first radix C does not divide
      {.n = 1200, .tops = {4}, .sub = {}},           // too narrow for 8 lanes
      {.n = 2000, .tops = {20}, .sub = {}},          // radix 20 at the top
  };
  return all;
}

template <class T> void every_case(fftn::isa target) {
  for (const depth_case& dc : cases()) {
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_depth<T>(target, dc, dir, 1.0, 1, dc.n);
    }
  }
  // The scale on the top pass and a batch with gaps; one and two tops.
  for (const depth_case& dc : {cases()[1], cases()[7]}) {
    check_depth<T>(target, dc, fftn::direction::backward, 0.25, 2, dc.n + 3);
    check_depth<T>(target, dc, fftn::direction::forward, 1.0 / 3, 3, dc.n + 5);
  }
  // In place through the workspace block, with a scale and a batch; one and two tops.
  check_depth<T>(target, cases()[0], fftn::direction::forward, 1.0, 1, cases()[0].n, true);
  check_depth<T>(target, cases()[7], fftn::direction::backward, 0.5, 2, cases()[7].n + 3, true);
}

TEST(Depth, EveryTargetPrecisionDirectionAndSplit) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    every_case<double>(t);
    every_case<float>(t);
  }
}

// Without the strategy forced the rules never take it (experiment X46, arm A0).
TEST(Depth, RulesDoNotTakeIt) {
  const fftn::shape<std::complex<double>, std::complex<double>> shape{
      .dir = fftn::direction::forward,
      .axes = {{.n = 16384}},
      .batches = {},
  };
  const auto core = make_plan_core(fftn::detail::view_of(shape), {}, build_settings{});
  EXPECT_EQ(fftn::detail::describe(*core).find("strategy: depth-first chain"), std::string::npos);
}

} // namespace
