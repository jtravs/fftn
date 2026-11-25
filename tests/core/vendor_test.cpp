// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Rules measured on one vendor's processors only (DESIGN.md 13.5; the maintainer, 2026-11-24):
// on `avx2` the radix overrides of X30 (512), X42 and X54, the batch leaves of X53 and X62, and the
// copies without prefetch of X26 apply only where machine_profile::vendor is amd; every other
// vendor, unknown included, keeps the plans `avx2` had before them. Every other target plans
// alike for all vendors. Each test sets the vendor itself, never the host's.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"
#include "platform/cpu_features.hpp"
#include "support/fftn_subject.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace d = fftn::detail;
using d::cpu_vendor;

constexpr std::array others{cpu_vendor::intel, cpu_vendor::unknown, cpu_vendor::apple,
                            cpu_vendor::arm_other};

bool same_span(std::span<const std::uint16_t> a, std::span<const std::uint16_t> b) {
  return std::ranges::equal(a, b);
}

// Every target but avx2 has one profile for all vendors; avx2's differ only in its overrides
// (the same pass costs) and its batch leaves (none), and keep X57's fit_overrides (a cache key).
TEST(VendorRules, ProfilesDifferOnlyOnAvx2) {
  const std::vector<std::pair<d::target, std::size_t>> profiles{
      {d::target::scalar, 1},    {d::target::scalar, 2}, {d::target::x86_v2, 1},
      {d::target::avx, 2},       {d::target::avx2, 2},   {d::target::avx2, 4},
      {d::target::avx512, 4},    {d::target::avx512, 8}, {d::target::neon_fcma, 1},
      {d::target::neon_fcma, 2}, {d::target::neon, 2},
  };
  for (const auto& [t, lanes] : profiles) {
    const d::target_profile amd = d::profile_of(t, lanes, cpu_vendor::amd);
    // The default is unknown's (the earlier plans), as profile_for's.
    EXPECT_EQ(d::profile_of(t, lanes).costs, d::profile_of(t, lanes, cpu_vendor::unknown).costs);
    for (const cpu_vendor v : others) {
      SCOPED_TRACE(std::string(d::target_name(t)) + ", C " + std::to_string(lanes) + ", " +
                   d::cpu_vendor_name(v));
      const d::target_profile p = d::profile_of(t, lanes, v);
      EXPECT_EQ(p.leaf_limit, amd.leaf_limit);
      EXPECT_EQ(p.split, amd.split);
      EXPECT_EQ(p.lanes_up_to, amd.lanes_up_to);
      EXPECT_EQ(p.rader_from, amd.rader_from);
      EXPECT_EQ(p.rader_below, amd.rader_below);
      EXPECT_EQ(p.batch_narrower, amd.batch_narrower);
      EXPECT_TRUE(same_span(p.batch_own_rows, amd.batch_own_rows));
      EXPECT_EQ(p.fit_overrides.data(), amd.fit_overrides.data());
      if (t != d::target::avx2) {
        EXPECT_EQ(p.costs, amd.costs);
        EXPECT_TRUE(same_span(p.batch_rows, amd.batch_rows));
        EXPECT_TRUE(same_span(p.batch_max_rows, amd.batch_max_rows));
        EXPECT_TRUE(same_span(p.batch_ragged_rows, amd.batch_ragged_rows));
        continue;
      }
      EXPECT_NE(p.costs, amd.costs);
      EXPECT_EQ(p.costs->unit, amd.costs->unit);
      EXPECT_EQ(p.costs->tail, amd.costs->tail);
      EXPECT_EQ(p.costs->pass, amd.costs->pass);
      EXPECT_TRUE(p.batch_rows.empty());
      EXPECT_TRUE(p.batch_max_rows.empty());
      EXPECT_TRUE(p.batch_ragged_rows.empty());
    }
  }
  // avx2 double does have batch leaves for AMD (X53, X62).
  EXPECT_FALSE(d::profile_of(d::target::avx2, 2, cpu_vendor::amd).batch_rows.empty());
}

// For other vendors avx2's radices are those before X30's 512 (by the costs), X42 (radix 25 for
// 500-25000) and X54 (768 = 4 x 16 x 12; float 512-4096 by the costs); every other length, X30's
// 256 = 8 x 32 and X38's and X2's overrides included (measured on the Xeon under avx2 too), is the
// same for all vendors.
TEST(VendorRules, Avx2RadicesBeforeTheAmdOnlyOverrides) {
  using v = std::vector<std::size_t>;
  const d::target_profile amd = d::profile_of(d::target::avx2, 2, cpu_vendor::amd);
  const d::target_profile amd_f = d::profile_of(d::target::avx2, 4, cpu_vendor::amd);
  EXPECT_EQ(d::rule_radices(256, amd), (v{8, 32}));
  EXPECT_EQ(d::rule_radices(512, amd), (v{16, 32}));
  EXPECT_EQ(d::rule_radices(768, amd), (v{12, 64}));
  EXPECT_EQ(d::rule_radices(500, amd), (v{20, 25}));
  EXPECT_EQ(d::rule_radices(512, amd_f), (v{8, 64}));
  EXPECT_EQ(d::rule_radices(3072, amd), (v{6, 16, 32})); // X72
  EXPECT_EQ(d::rule_radices(3200, amd), (v{10, 20, 16}));
  EXPECT_EQ(d::rule_radices(4608, amd), (v{6, 12, 64}));
  EXPECT_EQ(d::rule_radices(5120, amd), (v{10, 32, 16}));
  const std::set<std::size_t> by_costs{500,  512,  3072,  3125,  3200, 4608,
                                       5000, 5120, 10000, 15625, 25000};
  const std::set<std::size_t> by_costs_f{512, 640, 768, 2048, 4096};
  for (const cpu_vendor vendor : others) {
    SCOPED_TRACE(d::cpu_vendor_name(vendor));
    const d::target_profile p = d::profile_of(d::target::avx2, 2, vendor);
    const d::target_profile pf = d::profile_of(d::target::avx2, 4, vendor);
    EXPECT_EQ(d::rule_radices(256, p), (v{8, 32}));  // X30's 256, the Xeon's gain too (0.932)
    EXPECT_NE(d::rule_radices(512, p), (v{16, 32})); // X30's 512, lost on the Xeon
    EXPECT_EQ(d::rule_radices(768, p), (v{4, 16, 12}));
    EXPECT_EQ(d::rule_radices(384, p), (v{12, 32})); // X38's, measured on the Xeon too
    for (std::size_t n = 2; n <= 26000; ++n) {
      SCOPED_TRACE(n);
      if (n == 768) {
        continue;
      }
      EXPECT_EQ(d::rule_radices(n, p),
                by_costs.contains(n) ? d::rule_radices(n, amd, false) : d::rule_radices(n, amd));
      if (n <= 13000) {
        EXPECT_EQ(d::rule_radices(n, pf), by_costs_f.contains(n) ? d::rule_radices(n, amd_f, false)
                                                                 : d::rule_radices(n, amd_f));
      }
    }
    // X57's rows see their data for every vendor (its reach by cache key, accepted).
    EXPECT_EQ(d::fitting_radices(1024, p, true), (v{16, 64}));
    EXPECT_EQ(d::fitting_radices(768, p, true), (v{12, 64}));
  }
}

// ---- Plans --------------------------------------------------------------------------------

d::build_settings settings_for(cpu_vendor v) {
  d::build_settings s;
  s.profile = d::machine_profile{}; // a Zen 3 CCX's values, not the host's
  s.profile.vendor = v;
  return s;
}

template <class In, class Out>
std::shared_ptr<const d::plan_core> plan(const fftn::shape<In, Out>& s, fftn::isa t, cpu_vendor v) {
  return d::make_plan_core(d::view_of(s), {.force_isa = t}, settings_for(v));
}

// describe() without its vendor line.
std::string plan_text(const d::plan_core& c) {
  std::istringstream in(d::describe(c));
  std::string out;
  for (std::string line; std::getline(in, line);) {
    if (!line.starts_with("processor vendor: ")) {
      out += line + "\n";
    }
  }
  return out;
}

bool contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

// The pass functions of a plan's grid copies (gathers and scatters).
std::vector<d::pass_fn> copy_runs(const d::plan_core& c) {
  std::vector<d::pass_fn> out;
  for (const d::record& r : c.prog.records) {
    if (const auto* st = std::get_if<d::stage>(&r);
        st != nullptr &&
        (st->name.starts_with("grid gather") || st->name.starts_with("grid scatter") ||
         st->name.starts_with("grid twiddled scatter"))) {
      out.push_back(st->run);
    }
  }
  return out;
}

TEST(VendorRules, DescribeNamesTheVendor) {
  const auto rows = fftn::rows_c2c<double>(64, 1, fftn::direction::forward);
  EXPECT_TRUE(contains(d::describe(*plan(rows, fftn::isa::scalar, cpu_vendor::intel)),
                       "\nprocessor vendor: intel\n"));
  EXPECT_TRUE(contains(d::describe(*plan(rows, fftn::isa::scalar, cpu_vendor::unknown)),
                       "\nprocessor vendor: unknown\n"));
}

// On avx2 (where this machine runs it): AMD takes X30's 512 = 16 x 32, X62's batch leaf of 64,
// X53's of 15, and copies without prefetch (X26); Intel and unknown take the costs' 512, no batch
// leaves and prefetching copies, and 256 = 8 x 32 as AMD does. Elsewhere the plans are the same
// for every vendor.
TEST(VendorRules, PlansByVendor) {
  const auto r256 = fftn::rows_c2c<double>(256, 1, fftn::direction::forward);
  const auto r512 = fftn::rows_c2c<double>(512, 1, fftn::direction::forward);
  const auto r64x16 = fftn::rows_c2c<double>(64, 16, fftn::direction::backward);
  const auto r15x2 = fftn::rows_c2c<double>(15, 2, fftn::direction::backward);
  // 65536 points (1 MiB) are a grid by machine_profile{}'s fields (from 64 KiB, no long chains).
  const auto long_row = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  const auto real_row = fftn::rows_r2c<float>(131072, 1);
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(t));
    if (t != fftn::isa::avx2) {
      for (const cpu_vendor v : others) {
        EXPECT_EQ(plan_text(*plan(r256, t, v)), plan_text(*plan(r256, t, cpu_vendor::amd)));
        EXPECT_EQ(plan_text(*plan(r512, t, v)), plan_text(*plan(r512, t, cpu_vendor::amd)));
        EXPECT_EQ(plan_text(*plan(r64x16, t, v)), plan_text(*plan(r64x16, t, cpu_vendor::amd)));
        EXPECT_EQ(plan_text(*plan(long_row, t, v)), plan_text(*plan(long_row, t, cpu_vendor::amd)));
        EXPECT_EQ(copy_runs(*plan(long_row, t, v)), copy_runs(*plan(long_row, t, cpu_vendor::amd)));
      }
      continue;
    }
    const auto& kd = d::of<double>(*d::find_kernels(d::target::avx2)).passes;
    const auto& kf = d::of<float>(*d::find_kernels(d::target::avx2)).passes;
    const std::string amd256 = plan_text(*plan(r256, t, cpu_vendor::amd));
    EXPECT_TRUE(contains(amd256, "radices 8 x 32")) << amd256;
    const std::string amd512 = plan_text(*plan(r512, t, cpu_vendor::amd));
    EXPECT_TRUE(contains(amd512, "radices 16 x 32")) << amd512;
    EXPECT_TRUE(contains(plan_text(*plan(r64x16, t, cpu_vendor::amd)), "batch leaf radix-64"));
    EXPECT_TRUE(contains(plan_text(*plan(r15x2, t, cpu_vendor::amd)), "batch leaf radix-15"));
    const std::vector<d::pass_fn> amd_copies = copy_runs(*plan(long_row, t, cpu_vendor::amd));
    ASSERT_FALSE(amd_copies.empty());
    for (const d::pass_fn f : amd_copies) {
      EXPECT_TRUE(f == kd.block_copy[0][0] || f == kd.block_copy[0][1] ||
                  f == kd.twiddle_scatter[0][0] || f == kd.twiddle_scatter[0][1]);
    }
    for (const cpu_vendor v : others) {
      SCOPED_TRACE(d::cpu_vendor_name(v));
      EXPECT_EQ(plan_text(*plan(r256, t, v)), amd256);
      const std::string text = plan_text(*plan(r512, t, v));
      EXPECT_FALSE(contains(text, "radices 16 x 32")) << text;
      EXPECT_FALSE(contains(plan_text(*plan(r64x16, t, v)), "batch leaf"));
      EXPECT_FALSE(contains(plan_text(*plan(r15x2, t, v)), "batch leaf"));
      // The same program but for the copies' prefetch, which describe() does not show.
      const auto core = plan(long_row, t, v);
      EXPECT_EQ(plan_text(*core), plan_text(*plan(long_row, t, cpu_vendor::amd)));
      const std::vector<d::pass_fn> copies = copy_runs(*core);
      ASSERT_EQ(copies.size(), amd_copies.size());
      for (const d::pass_fn f : copies) {
        EXPECT_TRUE(f == kd.block_copy[1][0] || f == kd.block_copy[1][1] ||
                    f == kd.twiddle_scatter[1][0] || f == kd.twiddle_scatter[1][1]);
      }
      const std::vector<d::pass_fn> real_copies = copy_runs(*plan(real_row, t, v));
      ASSERT_FALSE(real_copies.empty());
      for (const d::pass_fn f : real_copies) {
        EXPECT_TRUE(f == kf.block_copy[1][0] || f == kf.block_copy[1][1] ||
                    f == kf.twiddle_scatter[1][0] || f == kf.twiddle_scatter[1][1]);
      }
    }
  }
}

// Both copy variants compute the same: a prefetch moves no data. An Intel and an AMD plan of a
// grid row, whose programs differ only in it, give the same bits.
TEST(VendorRules, PrefetchingCopiesGiveTheSameBits) {
  using c = std::complex<double>;
  const auto shape = fftn::rows_c2c<double>(65536, 1, fftn::direction::forward);
  std::vector<c> x(65536);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = c(static_cast<double>(i % 97) - 48.0, static_cast<double>(i % 89) * 0.5);
  }
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(t));
    std::vector<std::vector<c>> outs;
    for (const cpu_vendor v : {cpu_vendor::amd, cpu_vendor::intel}) {
      const auto core = plan(shape, t, v);
      const fftn::workspace ws(d::workspace_bytes(*core, 1));
      std::vector<c> y(x.size());
      d::execute_all(*core, x.data(), y.data(), ws);
      outs.push_back(std::move(y));
    }
    EXPECT_EQ(outs[0], outs[1]);
  }
}

} // namespace
