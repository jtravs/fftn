// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Partial runs (DESIGN.md 8.2's tails; experiment X23): the units of grouped, lane, generic,
// Bluestein chirp and lanes-strategy passes that hold fewer than C butterflies, copies or
// rows. The x86 targets move them as masked vector accesses (load_n, store_n) where the
// build enables them (FFTN_MASKED_TAILS: GCC by default), the others through a tile.
// Chains are forced so that a grouped pass's butterfly count and a lane pass's copy count
// leave every remainder 0 .. C - 1 for every complex count (C = 1, 2, 4, 8), with first
// radices C divides and radices it does not; Bluestein and generic passes whose last block
// is partial; and the lanes strategy over adjacent rows whose count leaves every remainder.
// Double and float, both directions, out of place and in place, on every instruction set
// the machine runs. Every array ends exactly at a guard page (an access past
// its last element faults), every byte outside the described elements is a canary, and an
// out-of-place input must not change. Each row is checked against the quadruple-precision
// reference, and the targets that round alike (avx2 and avx512) must give the same bits
// with the same radices.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/harness.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;
using fftn::test::placement;

struct tail_case {
  std::size_t n;
  std::vector<std::size_t> radices; // forced; empty for the rules' choice
  std::size_t rows;
  // Rows one element apart (element i of row r at i rows + r) through the lanes strategy:
  // runs of C rows are adjacent, so the lanes' first and last passes read and write the rows
  // themselves, with a partial run where C does not divide the row count.
  bool interleaved = false;
  // Prime factors above 16 as generic passes rather than Bluestein.
  bool generic = false;
  // A stage whose name holds this text must be in the plan.
  std::string expect = {};
};

std::vector<tail_case> cases() {
  std::vector<tail_case> out;
  // Grouped first pass of radix a with m = b butterflies (remainders of b modulo 8: all),
  // then the last lane pass of radix b with s = a copies; a = 8 and 16 are whole blocks of
  // C on every target, 3, 9 and 12 are not.
  for (const std::size_t a : {3U, 8U, 9U, 12U, 16U}) {
    for (std::size_t b = 2; b <= 9; ++b) {
      out.push_back({.n = a * b, .radices = {a, b}, .rows = 2, .expect = "grouped"});
    }
  }
  // A middle lane pass with s = a copies (every remainder modulo 8) and the last with 5 a
  // (10, 15, ..., 45: every remainder modulo 8 too).
  for (std::size_t a = 2; a <= 9; ++a) {
    out.push_back({.n = a * 15, .radices = {a, 5, 3}, .rows = 1, .expect = "lane radix-5"});
  }
  // Bluestein chirp passes whose last block of C holds 1 to C - 1 data elements.
  for (const std::size_t n : {137U, 274U, 59U, 188U, 141U, 94U, 47U, 167U}) {
    out.push_back({.n = n, .radices = {}, .rows = 2, .expect = "bluestein"});
  }
  // Rader stages (experiment X27): one copy at a time (1, 3 copies) and C copies at a time.
  for (const std::size_t n : {151U, 453U, 302U, 604U, 1208U}) {
    out.push_back({.n = n, .radices = {}, .rows = 2, .expect = "rader"});
  }
  // Generic passes, first and last, with partial runs.
  out.push_back({.n = 58, .radices = {2, 29}, .rows = 2, .generic = true, .expect = "generic"});
  out.push_back({.n = 87, .radices = {29, 3}, .rows = 2, .generic = true, .expect = "generic"});
  out.push_back({.n = 93, .radices = {3, 31}, .rows = 1, .generic = true, .expect = "generic"});
  // The prime codelets (experiment X27), first and last, with partial runs.
  out.push_back({.n = 34, .radices = {2, 17}, .rows = 2, .expect = "lane radix-17"});
  out.push_back({.n = 51, .radices = {17, 3}, .rows = 2, .expect = "grouped radix-17"});
  out.push_back({.n = 57, .radices = {3, 19}, .rows = 1, .expect = "lane radix-19"});
  out.push_back({.n = 69, .radices = {23, 3}, .rows = 1, .expect = "grouped radix-23"});
  // The lanes strategy (two rows or more): a leaf, and two passes, over 2 to 9 adjacent rows.
  for (std::size_t rows = 2; rows <= 9; ++rows) {
    out.push_back(
        {.n = 8, .radices = {8}, .rows = rows, .interleaved = true, .expect = "lanes leaf"});
    out.push_back(
        {.n = 12, .radices = {3, 4}, .rows = rows, .interleaved = true, .expect = "lanes first"});
    out.push_back(
        {.n = 20, .radices = {4, 5}, .rows = rows, .interleaved = true, .expect = "lanes last"});
  }
  return out;
}

std::string stage_listing(const fftn::detail::plan_core& core) {
  std::string text;
  for (const auto& rec : core.prog.records) {
    if (const auto* st = std::get_if<fftn::detail::stage>(&rec); st != nullptr) {
      text += "\n  " + st->name;
    }
  }
  return text;
}

template <class T> auto bits_of(T x) {
  return std::bit_cast<std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>>(x);
}

template <class T>
bool bitwise_equal(const std::vector<std::complex<T>>& x, const std::vector<std::complex<T>>& y) {
  return std::ranges::equal(x, y, [](std::complex<T> a, std::complex<T> b) {
    return bits_of(a.real()) == bits_of(b.real()) && bits_of(a.imag()) == bits_of(b.imag());
  });
}

// The offset of element 0 from a 64-byte boundary that puts the end of a tight array of
// `bytes` bytes exactly at its guard page, so that reading or writing past its last element
// faults rather than touching canary bytes.
std::size_t end_at_guard(std::size_t bytes) { return (64 - bytes % 64) % 64; }

template <class T> void check_case(const tail_case& tc, fftn::direction dir) {
  using c = std::complex<T>;
  const std::size_t n = tc.n;
  const std::size_t rows = tc.rows;
  const auto stride = static_cast<std::ptrdiff_t>(tc.interleaved ? rows : 1);
  const auto dist = static_cast<std::ptrdiff_t>(tc.interleaved ? 1 : n);
  fftn::test::input_source src(0x7A11ULL + 131 * n + rows);
  const std::vector<c> x = src.complexes<T>(rows * n); // row after row, logical order

  // The reference of each row, and the tolerance.
  std::vector<std::vector<fftn::test::exact_complex>> want(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    std::vector<std::complex<double>> xr(n);
    for (std::size_t j = 0; j < n; ++j) {
      xr[j] = std::complex<double>(x[r * n + j]);
    }
    const auto exact = fftn::test::reference::c2c(static_cast<int>(dir), xr);
    want[r].resize(n);
    for (std::size_t k = 0; k < n; ++k) {
      want[r][k] = fftn::test::exact_from(std::complex<double>(exact[k].re.hi, exact[k].im.hi));
    }
  }

  build_settings settings;
  settings.radices = tc.radices;
  settings.layout = fftn::detail::layout_choice::interleaved;
  if (tc.generic) {
    settings.bluestein_above = std::numeric_limits<std::size_t>::max();
  }
  if (tc.interleaved) {
    settings.strategy = fftn::detail::strategy_choice::lanes;
  }
  auto shape = fftn::rows_c2c<T>(n, rows, dir);
  shape.axes[0].in_stride = stride;
  shape.axes[0].out_stride = stride;
  shape.batches[0].in_dist = dist;
  shape.batches[0].out_dist = dist;
  const auto view = fftn::detail::view_of(shape);
  const auto offsets = fftn::test::row_offsets(n, rows, stride, dist);
  const fftn::test::run_options guard{
      .in_offset = end_at_guard(rows * n * sizeof(c)),
      .out_offset = end_at_guard(rows * n * sizeof(c)),
      .guard = fftn::test::guard_side::above,
  };

  std::map<fftn::isa, std::vector<c>> results;
  for (const fftn::isa target : fftn::test::runnable_isas()) {
    for (const placement place : {placement::out_of_place, placement::in_place}) {
      const bool in_place = place == placement::in_place;
      const std::string what = std::string(fftn::isa_name(target)) +
                               (sizeof(T) == 8 ? " double" : " float") +
                               ", n = " + std::to_string(n) + ", rows = " + std::to_string(rows) +
                               (tc.interleaved ? " interleaved" : "") +
                               (dir == fftn::direction::forward ? ", forward" : ", backward") +
                               (in_place ? ", in place" : ", out of place");
      SCOPED_TRACE(what);
      const auto core = make_plan_core(view, {.in_place = in_place, .force_isa = target}, settings);
      const std::string listing = stage_listing(*core);
      if (!tc.expect.empty()) {
        EXPECT_NE(listing.find(tc.expect), std::string::npos)
            << "no stage '" << tc.expect << "' in" << listing;
      }
      const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
      const auto call = [&](const c* in, c* out) {
        fftn::detail::execute_core(*core, 0, fftn::detail::transform_count(*core), in, out, ws);
      };
      const auto got =
          fftn::test::execute<c, c>(call, offsets, offsets, place, std::span<const c>(x), guard);
      for (const std::string& v : got.violations) {
        ADD_FAILURE() << v << "; stages:" << listing;
      }
      const bool bluestein = listing.find("bluestein") != std::string::npos ||
                             listing.find("rader") != std::string::npos;
      const auto tol = fftn::test::derived_tolerance<T>(
          n, bluestein ? fftn::test::bluestein_bound(n) : fftn::test::stable_fft_bound);
      for (std::size_t r = 0; r < rows; ++r) {
        fftn::test::outcome o;
        o.within(fftn::test::compare<T>(std::span<const c>(got.out).subspan(r * n, n), want[r]),
                 tol, what + ", row " + std::to_string(r));
        EXPECT_TRUE(o.passed()) << o.detail() << "stages:" << listing;
      }
      if (in_place) {
        // In place must give the bits of out of place: the same passes on the same data.
        EXPECT_TRUE(bitwise_equal(got.out, results[target])) << "in place against out of place";
      } else {
        results[target] = got.out;
      }
    }
  }
  // avx2 and avx512 round alike (DESIGN.md 6.2): the same radices give the same bits. A
  // Bluestein stage's convolution length comes from each target's profile, so it is left out.
  const bool forced = !tc.radices.empty();
  if (forced && results.contains(fftn::isa::avx2) && results.contains(fftn::isa::avx512)) {
    EXPECT_TRUE(bitwise_equal(results[fftn::isa::avx2], results[fftn::isa::avx512]))
        << "avx2 and avx512 differ: n = " << n << ", rows = " << rows
        << (sizeof(T) == 8 ? " double" : " float");
  }
}

TEST(Tails, EveryPartialRunShapeAgainstTheReference) {
  for (const tail_case& tc : cases()) {
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_case<double>(tc, dir);
      check_case<float>(tc, dir);
    }
  }
}

} // namespace
