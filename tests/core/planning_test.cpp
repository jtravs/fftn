// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Measured planning and wisdom (DESIGN.md 11.2 and 11.3, task 2.4): the planner's ranked
// radix sequences, the candidates' text form, the candidates each shape gets (and that every
// one computes the transform), the measure loop's choice (with a stand-in clock, so that the
// choice is known: the fastest, the rules' unless beaten by 1 %, the earlier on a tie), real
// measurements through the public API (quick and full, in place, a scratch limit), and
// wisdom: an entry rebuilds the recorded program bit for bit, a stale one is ignored, the
// text and file forms round-trip and malformed text is refused.

#include "core/candidates.hpp"
#include "core/measure.hpp"
#include "core/plan_core.hpp"
#include "core/wisdom_entry.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "isa/target.hpp"
#include "planner/rules.hpp"
#include "platform/page_memory.hpp"
#include "platform/topology.hpp"
#include "support/fftn_subject.hpp"
#include "threading/schedule.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace d = fftn::detail;
using d::build_settings;
using d::plan_choice;
using d::plan_core;
using d::strategy_choice;

// ---- Running a core --------------------------------------------------------------------

// Bytes of a core's output after one execution on a fixed input. Default (tight) layouts
// only, so each side spans [0, hi).
std::vector<unsigned char> run(const plan_core& c) {
  const auto in_bytes = static_cast<std::size_t>(c.in_hi);
  const auto out_bytes = static_cast<std::size_t>(c.out_hi);
  const std::size_t words = (std::max(in_bytes, out_bytes) + 7) / 8;
  std::vector<double> in(words);
  std::vector<double> out(words);
  if (c.prec == d::precision::f64) {
    for (std::size_t i = 0; i < in_bytes / 8; ++i) {
      in[i] = std::sin(0.37 * static_cast<double>(i) + 0.1);
    }
  } else {
    auto* f = reinterpret_cast<float*>(in.data());
    for (std::size_t i = 0; i < in_bytes / 4; ++i) {
      f[i] = static_cast<float>(std::sin(0.37 * static_cast<double>(i) + 0.1));
    }
  }
  const fftn::workspace ws(d::workspace_bytes(c, 1));
  if (c.in_place) {
    d::execute_core(c, 0, d::transform_count(c), in.data(), in.data(), ws);
    out = std::move(in);
  } else {
    d::execute_core(c, 0, d::transform_count(c), in.data(), out.data(), ws);
  }
  std::vector<unsigned char> bytes(out_bytes);
  std::memcpy(bytes.data(), out.data(), out_bytes);
  return bytes;
}

std::size_t points_of(const plan_core& c) {
  std::size_t p = 1;
  for (const std::size_t n : c.lengths) {
    p *= n;
  }
  return p;
}

// Two outputs of one transform by different programs agree within an FFT's error: the
// largest difference at most 32 log2(N) epsilon of the output's root mean square.
template <class T>
void expect_close(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b,
                  std::size_t points) {
  ASSERT_EQ(a.size(), b.size());
  const std::size_t count = a.size() / sizeof(T);
  std::vector<T> x(count);
  std::vector<T> y(count);
  std::memcpy(x.data(), a.data(), a.size());
  std::memcpy(y.data(), b.data(), b.size());
  double sum = 0;
  double worst = 0;
  for (std::size_t i = 0; i < count; ++i) {
    sum += static_cast<double>(y[i]) * static_cast<double>(y[i]);
    worst = std::max(worst, std::abs(static_cast<double>(x[i]) - static_cast<double>(y[i])));
  }
  const double rms = std::sqrt(sum / static_cast<double>(count));
  const double bound = 32 * std::log2(static_cast<double>(points) + 1) *
                       static_cast<double>(std::numeric_limits<T>::epsilon()) * rms;
  EXPECT_LE(worst, bound);
}

void expect_same_transform(const plan_core& a, const plan_core& b) {
  if (a.prec == d::precision::f64) {
    expect_close<double>(run(a), run(b), points_of(a));
  } else {
    expect_close<float>(run(a), run(b), points_of(a));
  }
}

// A profile with fixed grid sizes and none of the long-row chain rules (X29, X32), so that the
// rules make long rows grids and the candidate lists do not depend on this machine's caches.
build_settings fixed_settings() {
  build_settings s;
  // No depth-first Bluestein convolutions (X50): the host's profile may set them.
  s.profile.depth_convolution_max_bytes = 0;
  s.profile.grid_block_bytes = std::size_t{64} * 1024;
  s.profile.grid_min_bytes = std::size_t{512} * 1024;
  s.profile.grid_three_bytes = std::size_t{1} << 30U;
  s.profile.grid_two_odd_part = 0;
  s.profile.grid_three_block = 0; // X39's fields
  s.profile.grid_phase_radix = 0;
  s.profile.grid_two_block_bytes = 0;
  s.profile.pow2_grid_bytes = 0;
  s.profile.grid_two_pow2_bytes = 0;
  s.profile.chain_part_bytes = 0;
  s.profile.chain_mixed_bytes = 0;
  s.profile.pow2_chain_bytes = 0;
  s.profile.pow2_radix8_bytes = 0;
  // No depth-first chains (X46, X51, X55): the host's profile may set them.
  s.profile.depth_max_bytes = 0;
  s.profile.depth_float_max_bytes = 0;
  return s;
}

std::vector<std::string> texts(const std::vector<plan_choice>& v) {
  std::vector<std::string> t;
  t.reserve(v.size());
  for (const plan_choice& c : v) {
    t.push_back(d::choice_text(c));
  }
  return t;
}

bool has(const std::vector<std::string>& v, const std::string& s) {
  return std::ranges::find(v, s) != v.end();
}

// ---- Ranked radices -----------------------------------------------------------------------

TEST(RankedRadices, FirstIsTheRulesAndEveryOneMultipliesToN) {
  const std::vector<std::pair<d::target, std::size_t>> profiles{
      {d::target::scalar, 1}, {d::target::avx2, 2},      {d::target::avx2, 4},
      {d::target::avx512, 4}, {d::target::avx512, 8},    {d::target::neon_fcma, 1},
      {d::target::neon, 2},   {d::target::neon_fcma, 2},
  };
  for (const auto& [t, lanes] : profiles) {
    const d::target_profile p = d::profile_of(t, lanes);
    for (std::size_t n = 2; n <= 3000; ++n) {
      SCOPED_TRACE("target " + std::to_string(static_cast<int>(t)) + ", C " +
                   std::to_string(lanes) + ", n " + std::to_string(n));
      const std::vector<std::vector<std::size_t>> r = d::ranked_radices(n, p, 4);
      ASSERT_FALSE(r.empty());
      ASSERT_LE(r.size(), 4U);
      EXPECT_EQ(r.front(), d::rule_radices(n, p));
      for (std::size_t i = 0; i < r.size(); ++i) {
        std::size_t product = 1;
        for (const std::size_t f : r[i]) {
          product *= f;
        }
        EXPECT_EQ(product, n);
        for (std::size_t j = 0; j < i; ++j) {
          EXPECT_NE(r[i], r[j]);
        }
      }
    }
  }
}

TEST(RankedRadices, SmoothLengthsHaveAlternativesAndLeavesNone) {
  const d::target_profile p = d::profile_of(d::target::avx2, 2);
  EXPECT_EQ(d::ranked_radices(1024, p, 4).size(), 4U);
  EXPECT_EQ(d::ranked_radices(16, p, 4).size(), 1U);
  EXPECT_EQ(d::ranked_radices(1024, p, 1).size(), 1U);
  EXPECT_TRUE(d::ranked_radices(1024, p, 0).empty());
}

TEST(RankedRadices, LongPowersOfTwoFollowFftl) {
  EXPECT_EQ(d::long_pow2_radices(16, true), (std::vector<std::size_t>{4, 4, 8, 8, 8, 8}));
  EXPECT_EQ(d::long_pow2_radices(15, true), (std::vector<std::size_t>{8, 8, 8, 8, 8}));
  EXPECT_EQ(d::long_pow2_radices(17, false), (std::vector<std::size_t>{8, 4, 4, 4, 4, 4, 4, 4}));
}

// ---- Rows whose radices see the plan's data (X57) ---------------------------------------

TEST(FitRows, RadicesByLengthAndWhetherTheDataFit) {
  const d::target_profile p = d::profile_of(d::target::avx2, 2);
  EXPECT_EQ(d::fitting_radices(1024, p, true), (std::vector<std::size_t>{16, 64}));
  EXPECT_EQ(d::fitting_radices(1024, p, false), d::rule_radices(1024, p, false));
  EXPECT_EQ(d::fitting_radices(768, p, true), (std::vector<std::size_t>{12, 64}));
  EXPECT_EQ(d::fitting_radices(768, p, false), d::rule_radices(768, p, false));
  EXPECT_NE(d::fitting_radices(768, p, false), d::rule_radices(768, p)); // 768's override
  EXPECT_TRUE(d::fitting_radices(512, p, true).empty());
  for (const auto& [t, lanes] : std::vector<std::pair<d::target, std::size_t>>{
           {d::target::avx2, 4},
           {d::target::avx512, 4},
           {d::target::neon_fcma, 1},
       }) {
    EXPECT_TRUE(d::fitting_radices(1024, d::profile_of(t, lanes), true).empty());
  }
}

// By the profile field alone (set here, not the host's): rows take the two-pass radices only
// where the listed length's rows and output fit, a plane's rows never.
TEST(FitRows, PlansFollowTheProfileField) {
  const auto text = [](std::size_t rows, bool in_place, std::size_t fit_bytes, fftn::isa isa,
                       std::size_t n = 1024,
                       std::size_t in_place_bytes = std::numeric_limits<std::size_t>::max(),
                       std::size_t long_row_bytes = std::numeric_limits<std::size_t>::max()) {
    build_settings settings;
    settings.profile.fit_row_bytes = fit_bytes;
    settings.profile.fit_in_place_bytes = in_place_bytes;
    settings.profile.fit_long_row_bytes = long_row_bytes;
    settings.profile.fit_large_l2 = false; // X57's table, whatever the host (X69's key)
    settings.profile.depth_max_bytes = 0;
    settings.profile.vendor = d::cpu_vendor::amd; // 768's override is AMD's (DESIGN.md 13.5)
    const fftn::shape<std::complex<double>, std::complex<double>> shape{
        .dir = fftn::direction::forward,
        .axes = {{.n = n}},
        .batches =
            {
                {
                    .count = rows,
                    .in_dist = static_cast<std::ptrdiff_t>(n),
                    .out_dist = static_cast<std::ptrdiff_t>(n),
                },
            },
    };
    return d::describe(
        *d::make_plan_core(d::view_of(shape), {.in_place = in_place, .force_isa = isa}, settings));
  };
  const auto two_pass = [](const std::string& t) {
    return t.find("radices 16 x 64") != std::string::npos;
  };
  const std::size_t kib = 1024;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(t));
    if (std::string(fftn::isa_name(t)) != "avx2") {
      // No listed lengths: the field changes nothing.
      EXPECT_EQ(text(3, false, 0, t), text(3, false, std::size_t{1} << 40U, t));
      continue;
    }
    // 1024 x 3 out of place: 48 KiB of rows and 48 of output.
    EXPECT_TRUE(two_pass(text(3, false, 96 * kib, t)));
    EXPECT_FALSE(two_pass(text(3, false, 96 * kib - 1, t)));
    // In place the rows are all the data.
    EXPECT_TRUE(two_pass(text(6, true, 96 * kib, t)));
    EXPECT_FALSE(two_pass(text(7, true, 96 * kib, t)));
    EXPECT_FALSE(two_pass(text(3, false, 0, t))); // off
    // In place the smaller of the two limits; out of place fit_in_place_bytes changes nothing.
    EXPECT_TRUE(two_pass(text(4, true, 96 * kib, t, 1024, 64 * kib)));
    EXPECT_FALSE(two_pass(text(5, true, 96 * kib, t, 1024, 64 * kib)));
    EXPECT_TRUE(two_pass(text(3, false, 96 * kib, t, 1024, 16 * kib)));
    // X70: out of place, rows longer than a page (1024: 16 KiB) also within fit_long_row_bytes;
    // in place it changes nothing.
    constexpr std::size_t never = std::numeric_limits<std::size_t>::max();
    EXPECT_TRUE(two_pass(text(2, false, 96 * kib, t, 1024, never, 64 * kib)));
    EXPECT_FALSE(two_pass(text(3, false, 96 * kib, t, 1024, never, 64 * kib)));
    EXPECT_TRUE(two_pass(text(6, true, 96 * kib, t, 1024, never, 16 * kib)));
    // 768 leaves its override only where the rows do not fit.
    EXPECT_NE(text(1, false, 96 * kib, t, 768).find("radices 12 x 64"), std::string::npos);
    EXPECT_EQ(text(9, false, 96 * kib, t, 768).find("radices 12 x 64"), std::string::npos);
    // The fitting plan is the forced sequence's plan, bit for bit.
    build_settings fit;
    fit.profile.fit_row_bytes = 96 * kib;
    fit.profile.fit_large_l2 = false;
    fit.profile.depth_max_bytes = 0;
    fit.profile.vendor = d::cpu_vendor::amd;
    build_settings forced = fit;
    forced.radices = {16, 64};
    const fftn::shape<std::complex<double>, std::complex<double>> rows{
        .dir = fftn::direction::forward,
        .axes = {{.n = 1024}},
        .batches = {{.count = 3, .in_dist = 1024, .out_dist = 1024}},
    };
    EXPECT_EQ(run(*d::make_plan_core(d::view_of(rows), {.force_isa = t}, fit)),
              run(*d::make_plan_core(d::view_of(rows), {.force_isa = t}, forced)));
    // A plane's rows keep the rules whatever the field says.
    const fftn::shape<std::complex<double>, std::complex<double>> plane{
        .dir = fftn::direction::forward,
        .axes = {{.n = 1024}, {.n = 1024}},
        .batches = {},
    };
    build_settings off = fit;
    off.profile.fit_row_bytes = 0;
    EXPECT_EQ(d::describe(*d::make_plan_core(d::view_of(plane), {.force_isa = t}, fit)),
              d::describe(*d::make_plan_core(d::view_of(plane), {.force_isa = t}, off)));
  }
}

// X69: the 2 MiB-L2 key's table, `avx512` double only; beyond its limit the rules' sequence.
TEST(FitRows, LargeL2TableRadices) {
  using v = std::vector<std::size_t>;
  d::target_profile p = d::profile_of(d::target::avx512, 4, d::cpu_vendor::intel);
  EXPECT_TRUE(p.fit_overrides.empty()); // profile_of alone: no fit table on avx512
  EXPECT_EQ(d::rule_radices(1024, p), (v{8, 8, 16}));
  EXPECT_EQ(d::rule_radices(160, p), (v{4, 4, 10}));
  EXPECT_EQ(d::rule_radices(240, p), (v{4, 4, 15}));
  p.fit_overrides = d::large_l2_fit_overrides(d::target::avx512, 4);
  EXPECT_EQ(d::fitting_radices(1024, p, true), (v{32, 32}));
  EXPECT_EQ(d::fitting_radices(160, p, true), (v{8, 20}));
  EXPECT_EQ(d::fitting_radices(240, p, true), (v{12, 20}));
  for (const std::size_t n : {160U, 240U, 1024U}) {
    EXPECT_EQ(d::fitting_radices(n, p, false), d::rule_radices(n, p)); // no plain override
  }
  EXPECT_TRUE(d::fitting_radices(768, p, true).empty());
  for (const auto& [t, lanes] : std::vector<std::pair<d::target, std::size_t>>{
           {d::target::avx512, 8},
           {d::target::avx2, 2},
           {d::target::avx2, 4},
           {d::target::neon_fcma, 1},
       }) {
    EXPECT_TRUE(d::large_l2_fit_overrides(t, lanes).empty());
  }
}

// X69 by the profile fields alone (set here, not the host's): on the key, rows of 1024, 160 and
// 240 take the two-pass radices while rows and output take at most fit_row_bytes (in place the
// rows at most fit_in_place_bytes; out of place, rows longer than a page at most
// fit_long_row_bytes); without the key (Zen 4 and Zen 5 under `avx512` have fit_row_bytes from
// the large-L3 branch) the plans stay as they were. X70's limits: 1024 out of place to 64 rows,
// in place to 256, 160 to 1638 rows and 240 to 1092 in either placement.
TEST(FitRows, LargeL2PlansWithinTheLimit) {
  constexpr std::size_t kib = 1024;
  // The whole profile of a Granite Rapids VM (c4-standard-16: 48 KiB L1, 2 MiB private L2,
  // 504 MiB of L3 for 8 CPUs), as set_grid_sizes gives it, never the host's.
  d::core_class granite;
  granite.l1d = {.size = 48 * kib, .line_size = 64, .ways = 12, .shared_by = 1};
  granite.l2 = {.size = 2048 * kib, .line_size = 64, .ways = 16, .shared_by = 1};
  granite.l3 = {.size = 516096 * kib, .line_size = 64, .ways = 16, .shared_by = 8};
  build_settings key;
  key.profile = d::machine_profile{};
  d::set_grid_sizes(key.profile, granite, true);
  key.profile.vendor = d::cpu_vendor::intel;
  ASSERT_TRUE(key.profile.fit_large_l2);
  ASSERT_EQ(key.profile.fit_row_bytes, 8192 * kib);
  ASSERT_EQ(key.profile.fit_in_place_bytes, 4096 * kib);
  ASSERT_EQ(key.profile.fit_long_row_bytes, 2048 * kib);
  build_settings off = key;
  off.profile.fit_row_bytes = 0;
  // Without the key but with X57's limits from the large-L3 branch, as Zen 4 and Zen 5 have.
  build_settings zen = key;
  zen.profile.fit_large_l2 = false;
  zen.profile.fit_row_bytes = 24 * 1024 * kib;
  zen.profile.fit_in_place_bytes = 16 * 1024 * kib;
  zen.profile.fit_long_row_bytes = std::numeric_limits<std::size_t>::max();
  const auto plan = [](std::size_t n, std::size_t rows, bool in_place, const build_settings& s,
                       fftn::isa isa) {
    const fftn::shape<std::complex<double>, std::complex<double>> shape{
        .dir = fftn::direction::forward,
        .axes = {{.n = n}},
        .batches = {{.count = rows,
                     .in_dist = static_cast<std::ptrdiff_t>(n),
                     .out_dist = static_cast<std::ptrdiff_t>(n)}},
    };
    return d::make_plan_core(d::view_of(shape), {.in_place = in_place, .force_isa = isa}, s);
  };
  const auto text = [&](std::size_t n, std::size_t rows, bool in_place, const build_settings& s,
                        fftn::isa isa) { return d::describe(*plan(n, rows, in_place, s, isa)); };
  const auto has = [](const std::string& t, const char* radices) {
    return t.find(radices) != std::string::npos;
  };
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(t));
    if (std::string(fftn::isa_name(t)) != "avx512") {
      // The key's table is empty here (avx2 included: X57's avx2 table is not used on the key).
      EXPECT_EQ(text(1024, 3, false, key, t), text(1024, 3, false, off, t));
      continue;
    }
    // Out of place, rows of 1024 (16 KiB, longer than a page) with their output within the
    // long-row limit, the L2: 64 rows fit, 65 do not.
    EXPECT_TRUE(has(text(1024, 1, false, key, t), "radices 32 x 32"));
    EXPECT_TRUE(has(text(1024, 64, false, key, t), "radices 32 x 32"));
    EXPECT_TRUE(has(text(1024, 65, false, key, t), "radices 8 x 8 x 16"));
    EXPECT_EQ(text(1024, 256, false, key, t), text(1024, 256, false, off, t));
    // In place, rows within twice the L2: 256 rows.
    EXPECT_TRUE(has(text(1024, 64, true, key, t), "radices 32 x 32"));
    EXPECT_TRUE(has(text(1024, 256, true, key, t), "radices 32 x 32"));
    EXPECT_TRUE(has(text(1024, 257, true, key, t), "radices 8 x 8 x 16"));
    EXPECT_EQ(text(1024, 2048, true, key, t), text(1024, 2048, true, off, t));
    // 160 and 240 (2560 and 3840 bytes a row): rows and output within four times the L2, rows in
    // place within twice the L2, the same 1638 and 1092 rows.
    for (const bool in_place : {false, true}) {
      SCOPED_TRACE(in_place ? "in place" : "out of place");
      EXPECT_TRUE(has(text(160, 1, in_place, key, t), "radices 8 x 20"));
      EXPECT_TRUE(has(text(160, 1638, in_place, key, t), "radices 8 x 20"));
      EXPECT_TRUE(has(text(160, 1639, in_place, key, t), "radices 4 x 4 x 10"));
      EXPECT_TRUE(has(text(240, 1, in_place, key, t), "radices 12 x 20"));
      EXPECT_TRUE(has(text(240, 1092, in_place, key, t), "radices 12 x 20"));
      EXPECT_TRUE(has(text(240, 1093, in_place, key, t), "radices 4 x 4 x 15"));
      EXPECT_EQ(text(240, 65536, in_place, key, t), text(240, 65536, in_place, off, t));
    }
    // X69's limits (the fbench arm O of X70) give X69's plans: 409 rows of 160, 273 of 240, and
    // 64 rows of 1024 in either placement.
    build_settings x69 = key;
    x69.profile.fit_row_bytes = 2048 * kib;
    x69.profile.fit_in_place_bytes = 1024 * kib;
    EXPECT_TRUE(has(text(160, 409, false, x69, t), "radices 8 x 20"));
    EXPECT_TRUE(has(text(160, 410, true, x69, t), "radices 4 x 4 x 10"));
    EXPECT_TRUE(has(text(240, 273, true, x69, t), "radices 12 x 20"));
    EXPECT_TRUE(has(text(240, 274, false, x69, t), "radices 4 x 4 x 15"));
    EXPECT_TRUE(has(text(1024, 64, true, x69, t), "radices 32 x 32"));
    EXPECT_TRUE(has(text(1024, 65, true, x69, t), "radices 8 x 8 x 16"));
    EXPECT_EQ(text(1024, 64, false, x69, t), text(1024, 64, false, key, t));
    EXPECT_EQ(text(1024, 65, false, x69, t), text(1024, 65, false, key, t));
    // Off (fit_row_bytes = 0, fbench's FIT_ROW_BYTES=0) and without the key: the rules' plans.
    for (const std::size_t n : {160U, 240U, 1024U}) {
      EXPECT_EQ(text(n, 1, false, zen, t), text(n, 1, false, off, t));
      EXPECT_EQ(text(n, 16, true, zen, t), text(n, 16, true, off, t));
    }
    EXPECT_TRUE(has(text(1024, 1, false, off, t), "radices 8 x 8 x 16"));
    // The fitting plan is the forced sequence's plan, bit for bit.
    build_settings forced = off;
    forced.radices = {32, 32};
    EXPECT_EQ(run(*plan(1024, 3, false, key, t)), run(*plan(1024, 3, false, forced, t)));
    // A plane's rows and float keep their plans.
    const fftn::shape<std::complex<double>, std::complex<double>> plane{
        .dir = fftn::direction::forward,
        .axes = {{.n = 160}, {.n = 1024}},
        .batches = {},
    };
    EXPECT_EQ(d::describe(*d::make_plan_core(d::view_of(plane), {.force_isa = t}, key)),
              d::describe(*d::make_plan_core(d::view_of(plane), {.force_isa = t}, off)));
    const fftn::shape<std::complex<float>, std::complex<float>> single{
        .dir = fftn::direction::forward,
        .axes = {{.n = 1024}},
        .batches = {},
    };
    EXPECT_EQ(d::describe(*d::make_plan_core(d::view_of(single), {.force_isa = t}, key)),
              d::describe(*d::make_plan_core(d::view_of(single), {.force_isa = t}, off)));
  }
}

// ---- Choices as text --------------------------------------------------------------------

TEST(PlanChoice, TextRoundTrips) {
  const std::vector<plan_choice> choices{
      plan_choice(),
      {.strategy = strategy_choice::chain},
      {.strategy = strategy_choice::grid, .grid_phases = 3},
      {.strategy = strategy_choice::grid, .grid_factors = {64, 32, 32}},
      {.strategy = strategy_choice::lanes, .radices = {4, 4}},
      {.radices = {2, 16, 8}, .rader = 0},
      {.rader = 1},
      {.strategy = strategy_choice::depth},
      {.strategy = strategy_choice::depth, .depth_tops = {8, 4}},
      {.strategy = strategy_choice::depth, .radices = {8, 16}, .depth_tops = {16}},
  };
  for (const plan_choice& c : choices) {
    const std::string t = d::choice_text(c);
    SCOPED_TRACE(t);
    const std::optional<plan_choice> back = d::parse_choice(t);
    EXPECT_TRUE(back == c);
  }
  EXPECT_EQ(d::choice_text({}), "rules");
  EXPECT_EQ(d::choice_text({.strategy = strategy_choice::grid, .grid_factors = {64, 1024}}),
            "strategy=grid grid=64.1024");
  EXPECT_EQ(d::choice_text({.strategy = strategy_choice::depth, .depth_tops = {8, 4}}),
            "strategy=depth tops=8.4");
}

TEST(PlanChoice, MalformedTextIsRefused) {
  for (const char* t : {
           "",
           "foo",
           "strategy=",
           "strategy=fast",
           "radices=1.2",
           "radices=4..4",
           "radices=",
           "grid=64",
           "grid=2.2.2.2",
           "phases=4",
           "rader=2",
           "radices=4.4 strategy=chain",
           "rader=0 rader=1",
           "strategy=deep",
           "tops=1",
           "tops=4.4.4",
           "tops=8 strategy=depth",
           "rules extra",
           "strategy=chain  radices=4.4",
       }) {
    SCOPED_TRACE(t);
    EXPECT_FALSE(d::parse_choice(t).has_value());
  }
}

// ---- Candidates ----------------------------------------------------------------------------

std::shared_ptr<plan_core> rules_core(const d::shape_view& s, const build_settings& settings,
                                      const fftn::options& opts = {}) {
  return d::build_plan_core(s, opts, settings);
}

TEST(Candidates, LongRowsGetChainGridAndLongRadices) {
  const build_settings settings = fixed_settings();
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  const auto core = rules_core(d::view_of(shape), settings);
  const std::vector<std::string> t = texts(d::enumerate_candidates(*core, settings, 4));
  EXPECT_EQ(t.front(), "rules");
  EXPECT_TRUE(has(t, "strategy=chain"));
  EXPECT_TRUE(has(t, "strategy=grid phases=2"));
  EXPECT_TRUE(has(t, "strategy=grid phases=3")); // 1 MiB > grid_min_bytes
  EXPECT_TRUE(has(t, "strategy=chain radices=4.4.8.8.8.8"));
  EXPECT_TRUE(has(t, "strategy=chain radices=4.4.4.4.4.4.4.4"));
  EXPECT_FALSE(has(t, "strategy=lanes"));
}

TEST(Candidates, RealRowsUseTheirHalfLength) {
  const build_settings settings = fixed_settings();
  const auto shape = fftn::rows_r2c<double>(65536, 1);
  const auto core = rules_core(d::view_of(shape), settings);
  const std::vector<std::string> t = texts(d::enumerate_candidates(*core, settings, 4));
  EXPECT_TRUE(has(t, "strategy=grid phases=2"));
  EXPECT_FALSE(has(t, "strategy=grid phases=3")); // the half length is 512 KiB
  EXPECT_TRUE(has(t, "strategy=chain radices=8.8.8.8.8"));
}

TEST(Candidates, BatchesGetLanesAndShortRowsNoGrid) {
  const build_settings settings = fixed_settings();
  const auto shape = fftn::rows_c2c<float>(16, 64, fftn::direction::forward);
  const auto core = rules_core(d::view_of(shape), settings);
  const std::vector<std::string> t = texts(d::enumerate_candidates(*core, settings, 4));
  EXPECT_TRUE(has(t, "strategy=lanes"));
  EXPECT_TRUE(has(t, "strategy=chain"));
  EXPECT_FALSE(has(t, "strategy=grid phases=2"));
}

TEST(Candidates, LargePrimesGetRaderBothWays) {
  const build_settings settings = fixed_settings();
  const auto shape = fftn::rows_c2c<double>(7681, 1, fftn::direction::forward);
  const auto core = rules_core(d::view_of(shape), settings);
  const std::vector<std::string> t = texts(d::enumerate_candidates(*core, settings, 4));
  EXPECT_TRUE(has(t, "rader=0"));
  EXPECT_TRUE(has(t, "rader=1"));
}

TEST(Candidates, PlanesGetEachAxisRadices) {
  const build_settings settings = fixed_settings();
  const auto shape = fftn::planes_c2c<double>(64, 48, 1, fftn::direction::forward);
  const auto core = rules_core(d::view_of(shape), settings);
  const std::vector<plan_choice> v = d::enumerate_candidates(*core, settings, 3);
  std::size_t for64 = 0;
  std::size_t for48 = 0;
  for (const plan_choice& c : v) {
    std::size_t product = 1;
    for (const std::size_t r : c.radices) {
      product *= r;
    }
    if (product == 64) {
      ++for64;
    }
    if (product == 48) {
      ++for48;
    }
  }
  EXPECT_GE(for64, 2U);
  EXPECT_GE(for48, 2U);
}

// Every candidate of a set of shapes builds and computes the rules' transform.
template <class In, class Out>
void check_candidates(const fftn::shape<In, Out>& shape, const fftn::options& opts = {}) {
  const build_settings settings = fixed_settings();
  const auto rules = rules_core(d::view_of(shape), settings, opts);
  for (const plan_choice& c : d::enumerate_candidates(*rules, settings, 4)) {
    SCOPED_TRACE(d::choice_text(c));
    build_settings cs = settings;
    d::apply_choice(c, cs);
    const auto core = d::build_plan_core(d::view_of(shape), opts, cs);
    expect_same_transform(*core, *rules);
  }
}

TEST(Candidates, EveryCandidateComputesTheTransform) {
  check_candidates(fftn::rows_c2c<double>(65536, 1, fftn::direction::forward));
  check_candidates(fftn::rows_c2c<double>(46080, 1, fftn::direction::backward));
  check_candidates(fftn::rows_c2c<float>(16, 64, fftn::direction::forward));
  check_candidates(fftn::rows_c2c<double>(7681, 2, fftn::direction::forward));
  check_candidates(fftn::rows_c2c<double>(8192, 2, fftn::direction::forward), {.in_place = true});
  check_candidates(fftn::rows_r2c<double>(65536, 1));
  check_candidates(fftn::rows_c2r<float>(20000, 2));
  check_candidates(fftn::planes_c2c<double>(64, 48, 2, fftn::direction::forward));
}

// ---- The measure loop, with a stand-in clock ------------------------------------------

struct fake_clock {
  std::map<std::string, double> seconds;               // by choice text; 1 for the rest
  std::map<std::string, std::vector<double>> sequence; // a sample sequence, then its last value
  std::vector<std::string> asked;
  double others = 1.0; // every other choice
};

double fake_time(const std::string& choice, void* context) {
  auto& f = *static_cast<fake_clock*>(context);
  const auto seen = static_cast<std::size_t>(std::ranges::count(f.asked, choice));
  f.asked.push_back(choice);
  if (const auto q = f.sequence.find(choice); q != f.sequence.end()) {
    return q->second.at(std::min(seen, q->second.size() - 1));
  }
  const auto it = f.seconds.find(choice);
  return it == f.seconds.end() ? f.others : it->second;
}

std::shared_ptr<const plan_core> measured(const d::shape_view& s, fake_clock& clock,
                                          fftn::planning effort = fftn::planning::measure) {
  build_settings settings = fixed_settings();
  settings.measure_time = &fake_time;
  settings.measure_context = &clock;
  return d::make_plan_core(s, {.effort = effort}, settings);
}

TEST(Measure, KeepsTheFastest) {
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  fake_clock clock;
  // The rules make it a two-phase grid (fixed_settings); three phases are timed faster.
  clock.seconds["strategy=grid phases=3"] = 0.5;
  const auto core = measured(d::view_of(shape), clock);
  EXPECT_EQ(core->built, strategy_choice::grid);
  ASSERT_EQ(core->grid_factors.size(), 3U);
  EXPECT_EQ(core->program_text, "strategy=grid grid=" + std::to_string(core->grid_factors[0]) +
                                    "." + std::to_string(core->grid_factors[1]) + "." +
                                    std::to_string(core->grid_factors[2]));
  EXPECT_DOUBLE_EQ(core->rules_over_chosen, 2.0);
  EXPECT_NE(d::describe(*core).find("planning: measured"), std::string::npos);
  EXPECT_EQ(clock.asked.front(), "rules");
}

TEST(Measure, KeepsTheRulesUnlessBeatenByOnePercent) {
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  fake_clock clock;
  clock.seconds["strategy=chain"] = 0.995;
  const auto core = measured(d::view_of(shape), clock);
  const auto rules = d::build_plan_core(d::view_of(shape), fftn::options(), fixed_settings());
  EXPECT_EQ(rules->built, strategy_choice::grid);
  EXPECT_EQ(core->program_text, rules->program_text);
  EXPECT_EQ(d::program_fingerprint(*core), d::program_fingerprint(*rules));
}

TEST(Measure, TakesTheEarlierOnATie) {
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  fake_clock clock;
  clock.seconds["strategy=chain radices=4.4.8.8.8.8"] = 0.5;
  clock.seconds["strategy=chain radices=4.4.4.4.4.4.4.4"] = 0.5;
  const auto core = measured(d::view_of(shape), clock);
  EXPECT_EQ(core->program_text, "strategy=chain radices=4.4.8.8.8.8");
}

// A dropped candidate's median stays where it was while a live one's may rise; it must not
// win, since its plan has been freed (a regression: the reviewer's ASan build and parallel
// ctest crashed on it once the statistic became the median).
TEST(Measure, ADroppedCandidateNeverWins) {
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  fake_clock clock;
  clock.sequence["rules"] = {1, 1, 1, 9, 9, 9, 9, 9, 9};
  clock.seconds["strategy=chain"] = 2; // dropped after three rounds, then the lowest median
  clock.others = 20;
  const auto core = measured(d::view_of(shape), clock);
  const auto rules = d::build_plan_core(d::view_of(shape), fftn::options(), fixed_settings());
  EXPECT_EQ(d::program_fingerprint(*core), d::program_fingerprint(*rules));
  EXPECT_EQ(std::ranges::count(clock.asked, "strategy=chain"), 3);
}

TEST(Measure, QuickTimesFewerCandidates) {
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  fake_clock quick;
  fake_clock full;
  static_cast<void>(measured(d::view_of(shape), quick, fftn::planning::quick));
  static_cast<void>(measured(d::view_of(shape), full, fftn::planning::measure));
  const std::set<std::string> quick_asked(quick.asked.begin(), quick.asked.end());
  const std::set<std::string> full_asked(full.asked.begin(), full.asked.end());
  EXPECT_LE(quick_asked.size(), 4U);
  EXPECT_GT(full_asked.size(), quick_asked.size());
}

// ---- Real measurements, through the public API --------------------------------------

template <class In, class Out>
void check_measured(const fftn::shape<In, Out>& shape, fftn::options opts) {
  // Candidates are admitted while half the budget is left, so a small budget on a slow or
  // loaded machine may time nothing (CI's clang-cl runner did with 0.05 s). Two seconds
  // leave room to build a second candidate of these small shapes anywhere; the loop's own
  // round limit, not the budget, ends their timing in milliseconds.
  opts.measure_seconds = 2;
  for (const fftn::planning effort : {fftn::planning::quick, fftn::planning::measure}) {
    opts.effort = effort;
    // One plan: the one whose description is checked is the one whose account is checked.
    const auto core = d::make_plan_core(d::view_of(shape), opts);
    const std::string text = d::describe(*core);
    SCOPED_TRACE(text);
    EXPECT_NE(text.find("planning: measured (" + std::to_string(core->candidates_timed) + " of "),
              std::string::npos);
    EXPECT_GE(core->candidates_timed, 1U);
    expect_same_transform(*core, *d::build_plan_core(d::view_of(shape), {.in_place = opts.in_place},
                                                     build_settings{}));
  }
}

TEST(Measure, MeasuredPlansComputeTheTransform) {
  check_measured(fftn::rows_c2c<double>(256, 4, fftn::direction::forward), {});
  check_measured(fftn::rows_c2c<float>(4096, 1, fftn::direction::backward), {.in_place = true});
  check_measured(fftn::rows_r2c<double>(1000, 1), {});
  check_measured(fftn::rows_r2c<float>(512, 3), {.in_place = true});
  check_measured(fftn::rows_c2r<double>(2048, 2), {});
  check_measured(fftn::planes_c2c<double>(32, 24, 1, fftn::direction::forward), {});
}

// Candidates are admitted while half the budget is left: with almost none, only the rules'
// choice is built, nothing is timed, and the plan is the rules' plan.
TEST(Measure, ATinyBudgetTimesNothing) {
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  const fftn::options opts{.effort = fftn::planning::measure, .measure_seconds = 1e-9};
  const auto core = d::make_plan_core(d::view_of(shape), opts);
  // Building exhausted the budget before any timing: the plan says so.
  EXPECT_EQ(core->candidates_timed, 0U);
  EXPECT_NE(d::describe(*core).find("planning: measured (0 of 1 distinct candidates timed"),
            std::string::npos);
  EXPECT_EQ(d::program_fingerprint(*core),
            d::program_fingerprint(
                *d::build_plan_core(d::view_of(shape), fftn::options(), build_settings{})));
}

// A scratch limit below one transform's data times one transform, and only the rules'
// choice is admitted (every allocation counts against the limit); a limit of some
// transforms times that many.
TEST(Measure, ScratchLimitTimesFewerTransforms) {
  const auto shape = fftn::rows_c2c<double>(1024, 64, fftn::direction::forward);
  const fftn::c2c_plan<double> p(
      shape,
      {.effort = fftn::planning::quick, .measure_seconds = 0.05, .measure_scratch_bytes = 1});
  const std::string text = p.describe();
  EXPECT_NE(text.find("0 of 1 distinct candidates timed on 1 transform,"), std::string::npos)
      << text;
  EXPECT_NE(text.find("64 transforms"), std::string::npos) << text;
  const fftn::c2c_plan<double> q(shape, {
                                            .effort = fftn::planning::quick,
                                            .measure_seconds = 0.05,
                                            .measure_scratch_bytes = std::size_t{1} << 20U,
                                        });
  const std::string more = q.describe();
  const std::size_t at = more.find(" timed on ");
  ASSERT_NE(at, std::string::npos) << more;
  const std::size_t transforms = std::stoul(more.substr(at + 10));
  EXPECT_GT(transforms, 1U) << more;
  EXPECT_LT(transforms, 64U) << more;
}

// Large buffers are timed on 4 KiB and on huge pages where the two can differ (Linux with
// transparent huge pages), and a program is judged by the slower: P2 found a choice made on
// huge-page scratch 1.18 times slower on the caller's memory (EPYC r2c 65536).
TEST(Measure, LargeBuffersAreTimedOnBothPageKinds) {
  const auto shape = fftn::rows_r2c<double>(65536, 1);
  const auto core =
      d::make_plan_core(d::view_of(shape), {.effort = fftn::planning::quick, .measure_seconds = 2});
  const std::string text = d::describe(*core);
  const bool both = text.find(", on 4 KiB and huge pages,") != std::string::npos;
  EXPECT_EQ(both, fftn::detail::page_kinds_differ() && core->candidates_timed != 0) << text;
  const auto small =
      d::make_plan_core(d::view_of(fftn::rows_c2c<double>(256, 1, fftn::direction::forward)),
                        {.effort = fftn::planning::quick, .measure_seconds = 2});
  EXPECT_EQ(d::describe(*small).find("huge pages"), std::string::npos);
}

TEST(Measure, ReducedCountsShrinkTheOuterDimensionsFirst) {
  const std::vector<std::size_t> counts{4, 8};
  EXPECT_EQ(d::reduced_counts(counts, 3200, 0, 3200), counts); // fits
  EXPECT_EQ(d::reduced_counts(counts, 3200, 0, 800), (std::vector<std::size_t>{1, 8}));
  EXPECT_EQ(d::reduced_counts(counts, 3200, 0, 400), (std::vector<std::size_t>{1, 4}));
  EXPECT_EQ(d::reduced_counts(counts, 3200, 0, 1600), (std::vector<std::size_t>{2, 8}));
  EXPECT_EQ(d::reduced_counts(counts, 3200, 5000, 1000), (std::vector<std::size_t>{1, 1}));
  EXPECT_EQ(d::reduced_counts(std::vector<std::size_t>{}, 100, 0, 1), std::vector<std::size_t>{});
}

TEST(Measure, PlacementsChangeFromRoundToRound) {
  for (std::size_t r = 0; r < 64; ++r) {
    const d::placement p = d::placement_of(r);
    EXPECT_EQ(p.in % 64, 0U);
    EXPECT_EQ(p.out % 64, 0U);
    EXPECT_LT(p.in, 4096U);
    EXPECT_LT(p.out, 4096U);
    EXPECT_NE(p.in, p.out);
    if (r > 0) {
      EXPECT_NE(p.in, d::placement_of(r - 1).in);
    }
  }
}

// The budget's default is 0 whatever the effort, and a bad value is refused whatever the
// effort too.
TEST(Measure, BudgetDefaultsAndValidationDoNotDependOnTheEffort) {
  const auto shape = fftn::rows_c2c<double>(4096, 4, fftn::direction::forward);
  EXPECT_EQ(fftn::options{}.measure_seconds, 0.0);
  EXPECT_EQ(fftn::options{}.measure_scratch_bytes, 0U);
  // The default budget is the effort's own, room for more than one candidate. Measure's
  // (10 s) leaves that room anywhere; quick's 0.1 s did not on a coverage build in CI (run
  // 45379782870: 0 timed), where building the rules' plan took more than the half of the
  // budget in which other candidates are admitted.
  const auto core = d::make_plan_core(d::view_of(shape), {.effort = fftn::planning::measure});
  EXPECT_GE(core->candidates_timed, 2U);
  EXPECT_THROW(static_cast<void>(fftn::c2c_plan<double>(shape, {.measure_seconds = -1})),
               std::invalid_argument);
}

TEST(Measure, BudgetMustBeFiniteAndNotNegative) {
  const auto shape = fftn::rows_c2c<double>(64, 1, fftn::direction::forward);
  for (const double s :
       {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    try {
      const fftn::c2c_plan<double> p(shape,
                                     {.effort = fftn::planning::measure, .measure_seconds = s});
      ADD_FAILURE() << "accepted measure_seconds " << s;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("options.measure_seconds"), std::string::npos);
    }
  }
}

// ---- Wisdom --------------------------------------------------------------------------------

std::string entry_of(const plan_core& c) { return d::record_text(d::record_of(c)) + "\n"; }

TEST(Wisdom, AnEntryRebuildsTheSameProgramBitForBit) {
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  fake_clock clock;
  clock.seconds["strategy=chain radices=4.4.4.4.4.4.4.4"] = 0.5;
  const auto core = measured(d::view_of(shape), clock);
  ASSERT_EQ(core->program_text, "strategy=chain radices=4.4.4.4.4.4.4.4");
  const fftn::wisdom w(entry_of(*core));
  ASSERT_EQ(w.size(), 1U);
  const fftn::wisdom again(w.text());
  fake_clock unused;
  build_settings settings = fixed_settings();
  settings.measure_time = &fake_time;
  settings.measure_context = &unused;
  for (const fftn::planning effort : {fftn::planning::rules, fftn::planning::measure}) {
    const auto rebuilt =
        d::make_plan_core(d::view_of(shape), {.effort = effort, .use = &again}, settings);
    EXPECT_TRUE(unused.asked.empty()); // nothing timed
    EXPECT_EQ(d::program_fingerprint(*rebuilt), d::program_fingerprint(*core));
    EXPECT_NE(d::describe(*rebuilt).find("planning: from wisdom"), std::string::npos);
    EXPECT_EQ(run(*rebuilt), run(*core));
  }
}

TEST(Wisdom, PublicRoundTripThroughAFile) {
  const auto shape = fftn::rows_r2c<double>(3000, 2);
  const fftn::r2c_plan<double> p(shape, {.effort = fftn::planning::quick, .measure_seconds = 0.05});
  fftn::wisdom w;
  w.add(p);
  w.add(p);
  EXPECT_EQ(w.size(), 1U); // the same shape replaces its entry
  const std::filesystem::path file =
      std::filesystem::temp_directory_path() /
      ("fftn-wisdom-test-" + std::to_string(reinterpret_cast<std::uintptr_t>(&w)) + ".txt");
  w.save(file);
  const fftn::wisdom loaded = fftn::wisdom::load(file);
  std::filesystem::remove(file);
  EXPECT_EQ(loaded.text(), w.text());
  const fftn::r2c_plan<double> q(shape, {.use = &loaded});
  EXPECT_NE(q.describe().find("planning: from wisdom"), std::string::npos) << q.describe();
  const auto a = d::make_plan_core(d::view_of(shape), {.use = &loaded});
  EXPECT_EQ(d::program_fingerprint(*a),
            d::program_fingerprint(*d::make_plan_core(d::view_of(shape), {.use = &w})));
  EXPECT_THROW(static_cast<void>(fftn::wisdom::load(file)), std::runtime_error);
  // A file whose text is not wisdom: refused, naming the file and the line.
  {
    std::ofstream bad(file, std::ios::binary);
    bad << "# fftn wisdom\nnot wisdom\n";
  }
  try {
    static_cast<void>(fftn::wisdom::load(file));
    ADD_FAILURE() << "accepted a file of malformed wisdom";
  } catch (const std::invalid_argument& e) {
    EXPECT_EQ(std::string(e.what()).rfind(file.string() + ": ", 0), 0U) << e.what();
    EXPECT_NE(std::string(e.what()).find("line 2"), std::string::npos) << e.what();
  }
  std::filesystem::remove(file);
}

TEST(Wisdom, AStaleEntryIsIgnored) {
  const auto shape = fftn::rows_c2c<double>(4096, 1, fftn::direction::forward);
  const auto core = d::build_plan_core(d::view_of(shape), fftn::options(), build_settings{});
  d::wisdom_record r = d::record_of(*core);
  r.check ^= 1U;
  const fftn::wisdom w(d::record_text(r));
  const fftn::c2c_plan<double> p(shape, {.use = &w});
  EXPECT_NE(p.describe().find("does not rebuild its program here (ignored)"), std::string::npos)
      << p.describe();
}

TEST(Wisdom, OtherMachinesEntriesAreKeptButNotUsed) {
  const auto shape = fftn::rows_c2c<double>(4096, 1, fftn::direction::forward);
  const auto core = d::build_plan_core(d::view_of(shape), fftn::options(), build_settings{});
  d::wisdom_record r = d::record_of(*core);
  r.cpu = "Some Other Processor";
  r.program = "radices=2.2.2.2.2.2.2.2.2.2.2.2";
  const fftn::wisdom w(d::record_text(r));
  const fftn::c2c_plan<double> p(shape, {.use = &w});
  EXPECT_EQ(p.describe().find("planning:"), std::string::npos) << p.describe();
  EXPECT_NE(w.text().find("Some Other Processor"), std::string::npos);
}

TEST(Wisdom, MalformedTextIsRefusedNamingTheLine) {
  const auto shape = fftn::rows_c2c<double>(64, 1, fftn::direction::forward);
  const auto core = d::build_plan_core(d::view_of(shape), fftn::options(), build_settings{});
  const std::string good = d::record_text(d::record_of(*core));
  EXPECT_EQ(fftn::wisdom("# a comment\n\n" + good + "\n").size(), 1U);
  for (const std::string& bad : {
           std::string("not wisdom"),
           good + "\textra",
           good.substr(0, good.size() - 1) + "g",
           good.substr(0, good.size() - 1),
           good.substr(0, good.rfind('\t')),
       }) {
    std::string text = good;
    text += '\n';
    text += bad;
    text += '\n';
    try {
      const fftn::wisdom w(text);
      ADD_FAILURE() << "accepted " << bad;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("line 2"), std::string::npos) << e.what();
    }
  }
}

// ---- Wisdom: the reviewer's cases -------------------------------------------------------

TEST(Wisdom, CheckedProductRefusesOverflow) {
  EXPECT_EQ(d::checked_product(std::vector<std::size_t>{4, 8, 8}), 256U);
  EXPECT_EQ(d::checked_product(std::vector<std::size_t>{4611686018427387905ULL, 16, 256}), 0U);
  EXPECT_EQ(d::checked_product(std::vector<std::size_t>{}), 1U);
}

// An entry that parses but names a program this build cannot make, or one that does not
// apply to the shape, is stale: ignored, and describe() says so.
TEST(Wisdom, AnUnbuildableOrInapplicableProgramIsIgnored) {
  const auto shape = fftn::rows_c2c<double>(4096, 1, fftn::direction::forward);
  const auto core = d::build_plan_core(d::view_of(shape), fftn::options(), build_settings{});
  for (const char* program : {
           "radices=4611686018427387905.16.256",
           "grid=3.5",
           "strategy=grid grid=64.64",
           "radices=7.7",
       }) {
    SCOPED_TRACE(program);
    d::wisdom_record r = d::record_of(*core);
    r.program = program;
    const fftn::wisdom w(d::record_text(r));
    const fftn::c2c_plan<double> p(shape, {.use = &w});
    EXPECT_NE(p.describe().find("(ignored)"), std::string::npos) << p.describe();
  }
}

TEST(Wisdom, TheScaleIsPartOfTheKey) {
  const auto shape = fftn::rows_c2c<double>(512, 2, fftn::direction::forward);
  const fftn::c2c_plan<double> plain(shape);
  const fftn::c2c_plan<double> scaled(shape, {.scale = 1.0 / 512});
  fftn::wisdom w;
  w.add(plain);
  w.add(scaled);
  EXPECT_EQ(w.size(), 2U);
  const fftn::c2c_plan<double> other(shape, {.scale = 0.25, .use = &w});
  EXPECT_EQ(other.describe().find("planning:"), std::string::npos) << other.describe();
  const fftn::c2c_plan<double> again(shape, {.scale = 1.0 / 512, .use = &w});
  EXPECT_NE(again.describe().find("from wisdom"), std::string::npos) << again.describe();
}

// Lines of another build or format are kept verbatim (a future field included), exported
// again, and never used; a changed build identifier is another build.
TEST(Wisdom, OtherBuildsAndFormatsAreKeptVerbatim) {
  const auto shape = fftn::rows_c2c<double>(4096, 1, fftn::direction::forward);
  const auto core = d::build_plan_core(d::view_of(shape), fftn::options(), build_settings{});
  d::wisdom_record r = d::record_of(*core);
  const std::string build = r.build;
  ASSERT_NE(build.find('+'), std::string::npos);
  r.build = build.substr(0, build.find('+') + 1) + "0000000000000000"; // another identifier
  const std::string other_build = d::record_text(r);
  const std::string future =
      "fftn-wisdom-1\t9.9.9+ffffffffffffffff\tcpu\tisa\tkey\tnew-field=1 x\t??\textra";
  const std::string newer_format = "fftn-wisdom-2\tanything at all";
  const fftn::wisdom w(other_build + "\n" + future + "\n" + newer_format + "\n");
  EXPECT_EQ(w.size(), 3U);
  const std::string text = w.text();
  for (const std::string& line : {other_build, future, newer_format}) {
    EXPECT_NE(text.find(line + "\n"), std::string::npos) << line;
  }
  const fftn::c2c_plan<double> p(shape, {.use = &w});
  EXPECT_EQ(p.describe().find("planning:"), std::string::npos) << p.describe();
  // A malformed entry of this build is still refused.
  d::wisdom_record mine = d::record_of(*core);
  mine.program = "radices=04.4";
  EXPECT_THROW(static_cast<void>(fftn::wisdom(d::record_text(mine))), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(fftn::wisdom("not-wisdom\tx")), std::invalid_argument);
}

TEST(Wisdom, AddingWisdomReplacesEntriesOfTheSamePlace) {
  const auto a = fftn::rows_c2c<double>(256, 1, fftn::direction::forward);
  const auto b = fftn::rows_c2c<double>(512, 1, fftn::direction::forward);
  fftn::wisdom first;
  first.add(fftn::c2c_plan<double>(a));
  fftn::wisdom second;
  second.add(fftn::c2c_plan<double>(b));
  second.add(fftn::c2c_plan<double>(a));
  first.add(second);
  EXPECT_EQ(first.size(), 2U);
  EXPECT_EQ(fftn::wisdom(first.text()).text(), first.text());
}

TEST(Wisdom, SavingToAnUnwritablePathThrows) {
  const fftn::wisdom w;
  const std::filesystem::path bad =
      std::filesystem::temp_directory_path() / "fftn-no-such-directory" / "x" / "plans.wisdom";
  EXPECT_THROW(w.save(bad), std::runtime_error);
}

// A stale entry under measured planning falls back to timing.
TEST(Wisdom, AStaleEntryUnderMeasureFallsBackToTiming) {
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  build_settings settings = fixed_settings();
  const auto core = d::build_plan_core(d::view_of(shape), fftn::options(), settings);
  d::wisdom_record r = d::record_of(*core);
  r.check ^= 1U;
  const fftn::wisdom w(d::record_text(r));
  fake_clock clock;
  settings.measure_time = &fake_time;
  settings.measure_context = &clock;
  const auto measured_core = d::make_plan_core(
      d::view_of(shape), {.effort = fftn::planning::measure, .use = &w}, settings);
  EXPECT_FALSE(clock.asked.empty());
  const std::string text = d::describe(*measured_core);
  EXPECT_NE(text.find("(ignored); measured"), std::string::npos) << text;
}

// Measured planning on several threads at once, with one wisdom shared read-only: every
// plan is valid, the entries are used, and nothing races (the tests also run under TSan).
TEST(Wisdom, SharedReadOnlyByConcurrentMeasuredPlanning) {
  const std::vector<std::size_t> lengths{256, 1000, 4096, 6000};
  fftn::wisdom w;
  for (const std::size_t n : {std::size_t{256}, std::size_t{4096}}) {
    w.add(fftn::c2c_plan<double>(fftn::rows_c2c<double>(n, 2, fftn::direction::forward)));
  }
  std::vector<std::string> notes(8);
  std::vector<std::thread> threads;
  threads.reserve(notes.size());
  for (std::size_t t = 0; t < notes.size(); ++t) {
    threads.emplace_back([&, t] {
      const std::size_t n = lengths[t % lengths.size()];
      const fftn::c2c_plan<double> p(
          fftn::rows_c2c<double>(n, 2, fftn::direction::forward),
          {.effort = fftn::planning::quick, .measure_seconds = 0.02, .use = &w});
      notes[t] = p.describe();
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  for (std::size_t t = 0; t < notes.size(); ++t) {
    const std::size_t n = lengths[t % lengths.size()];
    const bool recorded = n == 256 || n == 4096;
    EXPECT_EQ(notes[t].find("planning: from wisdom") != std::string::npos, recorded) << notes[t];
    EXPECT_EQ(notes[t].find("planning: measured") != std::string::npos, !recorded) << notes[t];
  }
}

} // namespace
