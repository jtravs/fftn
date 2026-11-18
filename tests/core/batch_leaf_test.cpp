// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The batch leaves (experiment X53, src/passes/batch_leaf.inl): C rows of a batch at once,
// one row per lane, forced (build_settings::batch_leaves) on every instruction set the
// machine runs, double and float, the target's own set and the narrower one (avx2's in an
// avx512 plan), every codelet radix, both directions, the scale on either edge. Every
// output byte (the gaps between rows included, which hold a canary) must equal what the leaf
// of the same radix writes row by row (forced radices {n}, which the batch leaves leave
// alone): a lane does the arithmetic the leaf does to its row. Batch counts from 2 to 2C + 1
// (full groups and every partial group), in place and out of place, and execute_transforms
// over sub-ranges that start and end anywhere; a sample of rows against the
// quadruple-precision reference; and where a plan cannot take them. Bitwise independence of
// the team size is tests/threading's (team_cases.hpp has batch leaf cases).

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "gen/codelets.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "planner/rules.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using fftn::detail::batch_leaf_choice;
using fftn::detail::build_settings;
using fftn::detail::make_plan_core;

std::string described(const fftn::detail::plan_core& core) { return fftn::detail::describe(core); }

fftn::detail::target target_of(fftn::isa t) {
  return static_cast<fftn::detail::target>(static_cast<unsigned char>(t));
}

// The set a plan of target t takes for radix r with `choice`, or null (plan_core.cpp).
template <class T>
const fftn::detail::batch_leaf_set<T>* set_for(fftn::isa t, std::size_t r,
                                               batch_leaf_choice choice) {
  const fftn::detail::batch_leaf_table* table = fftn::detail::find_batch_leaves(target_of(t), r);
  if (choice == batch_leaf_choice::narrower) {
    if (const auto* narrower = fftn::detail::find_narrower_batch_leaves(target_of(t), r)) {
      table = narrower;
    }
  }
  if (table == nullptr || fftn::detail::of<T>(*table).lanes == 1) {
    return nullptr;
  }
  return &fftn::detail::of<T>(*table);
}

struct rows_case {
  std::size_t n;
  std::size_t count;
  std::size_t gap; // complex values between one row's end and the next row's start
  bool in_place;
  double scale;
  bool scale_on_input;
  fftn::direction dir;
};

// Complex values past the last row in every buffer: a store beyond the batch changes them.
constexpr std::size_t guard = 5;

[[nodiscard]] std::size_t elements(const rows_case& rc) {
  return rc.count * (rc.n + rc.gap) - rc.gap;
}

// Transforms [first, first + count) of the case through a plan built with these settings;
// returns the output buffer's bytes (in place, the shared buffer), which starts as a canary
// pattern out of place, and the plan's description in `text`.
template <class T>
std::vector<std::byte> run(fftn::isa t, const rows_case& rc, const build_settings& base,
                           std::size_t first, std::size_t count, std::string& text) {
  using c = std::complex<T>;
  fftn::shape<c, c> sh;
  sh.dir = rc.dir;
  const auto dist = static_cast<std::ptrdiff_t>(rc.n + rc.gap);
  sh.axes = {{.n = rc.n}};
  sh.batches = {{.count = rc.count, .in_dist = dist, .out_dist = dist}};
  build_settings settings = base;
  settings.scale_on_input = rc.scale_on_input;
  const auto core =
      make_plan_core(fftn::detail::view_of(sh),
                     {.in_place = rc.in_place, .scale = rc.scale, .force_isa = t}, settings);
  text = described(*core);
  fftn::test::input_source src(0xBA7CULL + rc.n * 131 + rc.count);
  const auto x = src.complexes<T>(elements(rc) + guard);
  std::vector<std::byte> in(x.size() * sizeof(c));
  std::memcpy(in.data(), x.data(), in.size());
  const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  if (rc.in_place) {
    fftn::detail::execute_core(*core, first, count, in.data(), in.data(), ws);
    return in;
  }
  const std::vector<std::byte> kept = in;
  std::vector<std::byte> out(in.size(), std::byte{0xa5});
  fftn::detail::execute_core(*core, first, count, in.data(), out.data(), ws);
  EXPECT_EQ(in, kept) << "the input was not preserved";
  return out;
}

// Row `row` of an output against the reference.
template <class T>
void against_reference(const rows_case& rc, const std::vector<std::byte>& out, std::size_t row,
                       const std::string& what) {
  using c = std::complex<T>;
  fftn::test::input_source src(0xBA7CULL + rc.n * 131 + rc.count);
  const auto x = src.complexes<T>(elements(rc) + guard);
  const std::size_t base = row * (rc.n + rc.gap);
  std::vector<std::complex<double>> xin(rc.n);
  for (std::size_t j = 0; j < rc.n; ++j) {
    xin[j] = std::complex<double>(x[base + j]);
  }
  const auto exact = fftn::test::reference::c2c(static_cast<int>(rc.dir), xin);
  std::vector<fftn::test::exact_complex> want(rc.n);
  std::vector<c> got(rc.n);
  for (std::size_t k = 0; k < rc.n; ++k) {
    want[k] =
        fftn::test::exact_from(std::complex<double>(exact[k].re.hi, exact[k].im.hi) * rc.scale);
    std::memcpy(&got[k], out.data() + sizeof(c) * (base + k), sizeof(c));
  }
  fftn::test::outcome o;
  o.within(fftn::test::compare<T>(std::span<const c>(got), want),
           fftn::test::derived_tolerance<T>(rc.n, fftn::test::stable_fft_bound),
           what + ", row " + std::to_string(row));
  EXPECT_TRUE(o.passed()) << o.detail();
}

template <class T> void batch_leaves_match(fftn::isa t, batch_leaf_choice choice) {
  build_settings batch;
  batch.batch_leaves = choice;
  for (const std::uint8_t radix : fftn::detail::gen::radices) {
    const std::size_t n = radix;
    const auto* set = set_for<T>(t, n, choice);
    if (set == nullptr || set->leaf[fftn::detail::codelet_slot(n)][0][0][0] == nullptr) {
      continue;
    }
    build_settings leaf;
    leaf.radices = {n};
    const std::size_t c = set->lanes;
    for (std::size_t count = 2; count <= 2 * c + 1; ++count) {
      for (const rows_case rc : {
               rows_case{n, count, 3, false, 1.0, false, fftn::direction::forward},
               rows_case{n, count, 0, true, 0.375, false, fftn::direction::backward},
               rows_case{n, count, 1, false, 0.375, true, fftn::direction::forward},
               rows_case{n, count, 2, true, 1.0, false, fftn::direction::forward},
           }) {
        const std::string what =
            std::string(fftn::isa_name(t)) + (sizeof(T) == 8 ? " double" : " float") +
            (choice == batch_leaf_choice::own ? " own" : " narrower") +
            ", n = " + std::to_string(n) + ", " + std::to_string(count) + " rows" +
            (rc.in_place ? ", in place" : "") + (rc.scale != 1.0 ? ", scaled" : "") +
            (rc.scale_on_input ? " on input" : "") +
            (rc.dir == fftn::direction::backward ? ", backward" : "");
        SCOPED_TRACE(what);
        std::string batch_text;
        std::string leaf_text;
        const auto got = run<T>(t, rc, batch, 0, count, batch_text);
        const auto want = run<T>(t, rc, leaf, 0, count, leaf_text);
        ASSERT_NE(batch_text.find("batch leaf radix-" + std::to_string(n)), std::string::npos)
            << batch_text;
        EXPECT_NE(batch_text.find(std::to_string(c) + " rows per register"), std::string::npos)
            << batch_text;
        EXPECT_EQ(leaf_text.find("batch leaf"), std::string::npos) << leaf_text;
        EXPECT_TRUE(got == want) << batch_text << leaf_text;
        if (count > 2) {
          for (const auto& [first, part] : {
                   std::pair<std::size_t, std::size_t>{1, count - 1},
                   {count / 2, 1},
                   {count / 3, count - count / 3 - 1},
               }) {
            std::string ignored;
            EXPECT_TRUE(run<T>(t, rc, batch, first, part, ignored) ==
                        run<T>(t, rc, leaf, first, part, ignored))
                << "sub-range " << first << " + " << part;
          }
        }
        if (count == c + 1 && n <= 32) {
          against_reference<T>(rc, got, 0, what);
          against_reference<T>(rc, got, count - 1, what);
        }
      }
    }
  }
}

TEST(BatchLeaves, EqualTheLeafRowByRow) {
  int sets = 0;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const batch_leaf_choice choice : {batch_leaf_choice::own, batch_leaf_choice::narrower}) {
      if (set_for<double>(t, 8, choice) != nullptr || set_for<float>(t, 8, choice) != nullptr) {
        ++sets;
      }
      batch_leaves_match<double>(t, choice);
      batch_leaves_match<float>(t, choice);
    }
  }
  if (sets == 0) {
    GTEST_SKIP() << "no target with batch leaves runs here";
  }
}

// Every radix of a target with C > 1 has a batch leaf (radix 25 not on NEON, which has no
// passes of it), and one complex number per register has none.
TEST(BatchLeaves, TablesCoverTheCodelets) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(t));
    const bool neon = t == fftn::isa::neon || t == fftn::isa::neon_fcma;
    for (const std::uint8_t radix : fftn::detail::gen::radices) {
      const std::size_t r = radix;
      const fftn::detail::batch_leaf_table* table =
          fftn::detail::find_batch_leaves(target_of(t), r);
      if (table == nullptr) {
        EXPECT_TRUE(t == fftn::isa::scalar || t == fftn::isa::x86_v2 || t == fftn::isa::avx);
        continue;
      }
      const std::size_t slot = fftn::detail::codelet_slot(r);
      const auto check = [&](const auto& set, const char* precision) {
        const bool present = set.lanes > 1 && !(neon && r == 25);
        for (std::size_t d = 0; d < 2; ++d) {
          EXPECT_EQ(set.leaf[slot][d][0][0] != nullptr, present) << precision << " radix " << r;
          EXPECT_EQ(set.leaf[slot][d][0][1] != nullptr, present) << precision << " radix " << r;
          EXPECT_EQ(set.leaf[slot][d][1][0] != nullptr, present) << precision << " radix " << r;
          EXPECT_EQ(set.leaf[slot][d][1][1], nullptr) << precision << " radix " << r;
        }
      };
      check(table->f64, "double");
      check(table->f32, "float");
    }
    EXPECT_EQ(fftn::detail::find_batch_leaves(target_of(t), 18), nullptr);
  }
}

// Forced, the batch leaves stay off where the rows cannot take one: a single row, a strided
// row, a factor edge, forced radices, a length without a codelet, two batch dimensions,
// planes, real rows.
TEST(BatchLeaves, ForcedOnlyWhereTheRowsCanTakeThem) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    if (set_for<double>(t, 16, batch_leaf_choice::own) == nullptr) {
      continue;
    }
    SCOPED_TRACE(fftn::isa_name(t));
    build_settings batch;
    batch.batch_leaves = batch_leaf_choice::own;
    const auto has = [&](const auto& sh, const build_settings& s) {
      return described(*make_plan_core(fftn::detail::view_of(sh), {.force_isa = t}, s))
                 .find("batch leaf") != std::string::npos;
    };
    constexpr auto fwd = fftn::direction::forward;
    EXPECT_TRUE(has(fftn::rows_c2c<double>(16, 2, fwd), batch));
    EXPECT_TRUE(has(fftn::rows_c2c<double>(64, 3, fwd), batch));
    EXPECT_FALSE(has(fftn::rows_c2c<double>(16, 1, fwd), batch));
    EXPECT_FALSE(has(fftn::rows_c2c<double>(18, 4, fwd), batch));
    EXPECT_FALSE(has(fftn::planes_c2c<double>(16, 16, 2, fwd), batch));
    EXPECT_FALSE(has(fftn::rows_r2c<double>(16, 4), batch));
    build_settings never;
    never.batch_leaves = batch_leaf_choice::never;
    EXPECT_FALSE(has(fftn::rows_c2c<double>(16, 4, fwd), never));
    build_settings forced = batch;
    forced.radices = {4, 4};
    EXPECT_FALSE(has(fftn::rows_c2c<double>(16, 4, fwd), forced));
    const std::vector<std::complex<double>> factor(16, {1.0, 0.0});
    build_settings with_factor = batch;
    with_factor.out_factor = factor.data();
    EXPECT_FALSE(has(fftn::rows_c2c<double>(16, 4, fwd), with_factor));
    fftn::shape<std::complex<double>, std::complex<double>> strided;
    strided.axes = {{.n = 16, .in_stride = 2}};
    strided.batches = {{.count = 4, .in_dist = 1}};
    EXPECT_FALSE(has(strided, batch));
    fftn::shape<std::complex<double>, std::complex<double>> two;
    two.axes = {{.n = 16}};
    two.batches = {{.count = 2}, {.count = 3}};
    EXPECT_FALSE(has(two, batch));
  }
}

// The rules (target_profile::batch_rows, batch_narrower, batch_own_rows; experiment X53): a
// batch of at least the profile's rows of a length it lists takes batch leaves, the narrower
// target's where the profile says so up to batch_own_rows, the target's own from there; a
// batch one row short of it, and single rows, do not. That every row of a batch gets the
// same arithmetic is tests/api's (BatchRowsGetTheSameArithmeticAnywhere).
template <class T> void rules_choose(fftn::isa t) {
  const auto target = target_of(t);
  const std::size_t lanes = fftn::detail::of<T>(*fftn::detail::find_kernels(target)).complex_lanes;
  const fftn::detail::target_profile profile = fftn::detail::profile_of(target, lanes);
  for (std::size_t slot = 0; slot < profile.batch_rows.size(); ++slot) {
    const std::size_t rows = profile.batch_rows[slot];
    if (rows == 0) {
      continue;
    }
    const std::size_t n = fftn::detail::gen::radices[slot];
    SCOPED_TRACE(std::string(fftn::isa_name(t)) + (sizeof(T) == 8 ? " double" : " float") +
                 ", n = " + std::to_string(n) + ", from " + std::to_string(rows) + " rows");
    // Not the host's profile: a private L2 of 2 MiB leaves out the leaf of 12 (X58).
    build_settings settings;
    settings.profile.batch_leaf_12 = true;
    const auto text = [&](std::size_t count) {
      return described(*make_plan_core(
          fftn::detail::view_of(fftn::rows_c2c<T>(n, count, fftn::direction::backward)),
          {.force_isa = t}, settings));
    };
    const std::size_t own = slot < profile.batch_own_rows.size() ? profile.batch_own_rows[slot] : 0;
    const std::size_t narrower_lanes = profile.batch_narrower ? lanes / 2 : lanes;
    const std::string at_rows = text(rows);
    EXPECT_NE(at_rows.find("batch leaf radix-" + std::to_string(n) + " backward, " +
                           std::to_string(own != 0 && rows >= own ? lanes : narrower_lanes) +
                           " rows per register"),
              std::string::npos)
        << at_rows;
    if (own != 0) {
      const std::string at_own = text(own);
      EXPECT_NE(at_own.find(std::to_string(lanes) + " rows per register"), std::string::npos)
          << at_own;
    }
    if (rows > 2) {
      EXPECT_EQ(text(rows - 1).find("batch leaf"), std::string::npos);
    }
    EXPECT_EQ(text(1).find("batch leaf"), std::string::npos);
  }
}

TEST(BatchLeaves, RulesChoose) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    rules_choose<double>(t);
    rules_choose<float>(t);
  }
}

} // namespace
