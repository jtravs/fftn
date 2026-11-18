// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The lanes strategy (DESIGN.md 8.5, src/strategies/lanes.cpp, src/passes/lanes.inl,
// experiment X7): batches of rows run one row per vector lane, forced on every instruction
// set the machine runs, double and float, both directions, c2c, r2c and c2r. Every output
// byte (the gaps between rows included, which hold a canary) must equal what the chain run
// row by row writes: the lanes give each row exactly the arithmetic it gets alone. Rows are
// contiguous with gaps, interleaved (a stride within the row, adjacent rows) or both; in
// place and out of place, with a scale; batch counts that are no multiple of the block or
// of C (the tail's padded block), forced blocks, and execute_transforms over sub-ranges
// that start and end anywhere; c2r's input preserved. A sample of rows against the
// quadruple-precision reference; and the planner's choice. Bitwise independence of the team
// size is tests/threading's (team_cases.hpp has lanes cases).

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
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
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;
using fftn::detail::strategy_choice;

std::string described(const fftn::detail::plan_core& core) { return fftn::detail::describe(core); }

bool is_lanes(const fftn::detail::plan_core& core) {
  return described(core).find("strategy: lanes") != std::string::npos;
}

template <class T> std::size_t complex_lanes(fftn::isa target) {
  return fftn::detail::of<T>(*fftn::detail::find_kernels(static_cast<fftn::detail::target>(
                                 static_cast<unsigned char>(target))))
      .complex_lanes;
}

// One layout of a batch of rows of one axis on one side, in that side's elements.
struct side_layout {
  std::ptrdiff_t stride = 1;
  std::ptrdiff_t dist = 0;
  // Elements the side spans for `count` rows of `extent` elements.
  [[nodiscard]] std::size_t span(std::size_t count, std::size_t extent) const {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(count - 1) * dist +
                                    static_cast<std::ptrdiff_t>(extent - 1) * stride) +
           1;
  }
};

struct batch_case {
  std::size_t n;
  std::size_t count;
  side_layout in;
  side_layout out;
  bool in_place = false;
  double scale = 1.0;
  fftn::direction dir = fftn::direction::forward;
};

// Runs transforms [first, first + count) of a plan of shape `sh` on the deterministic input
// and returns the output buffer's bytes (in place: the shared buffer), which starts as a
// canary pattern out of place. The input must be preserved out of place.
template <class In, class Out>
std::vector<std::byte> run(const fftn::shape<In, Out>& sh, const batch_case& bc,
                           const build_settings& settings, fftn::isa target, std::size_t in_elems,
                           std::size_t out_elems, std::size_t first, std::size_t count,
                           std::string& text) {
  const auto core =
      make_plan_core(fftn::detail::view_of(sh),
                     {.in_place = bc.in_place, .scale = bc.scale, .force_isa = target}, settings);
  text = described(*core);
  fftn::test::input_source src(0x1A4E5ULL + bc.n * 131 + bc.count);
  std::vector<std::byte> in(in_elems * sizeof(In));
  if constexpr (std::is_floating_point_v<In>) {
    const auto v = src.reals<In>(in_elems);
    std::memcpy(in.data(), v.data(), in.size());
  } else {
    const auto v = src.complexes<typename In::value_type>(in_elems);
    std::memcpy(in.data(), v.data(), in.size());
  }
  const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  if (bc.in_place) {
    std::vector<std::byte> buf(std::max(in.size(), out_elems * sizeof(Out)));
    std::memcpy(buf.data(), in.data(), in.size());
    fftn::detail::execute_core(*core, first, count, buf.data(), buf.data(), ws);
    return buf;
  }
  const std::vector<std::byte> kept = in;
  std::vector<std::byte> out(out_elems * sizeof(Out), std::byte{0xa5});
  fftn::detail::execute_core(*core, first, count, in.data(), out.data(), ws);
  EXPECT_EQ(in, kept) << "the input was not preserved";
  return out;
}

// Where two output buffers first differ, as values of Out, or nothing.
template <class Out>
std::string first_difference(const std::vector<std::byte>& a, const std::vector<std::byte>& b) {
  for (std::size_t i = 0; i + sizeof(Out) <= std::min(a.size(), b.size()); i += sizeof(Out)) {
    if (std::memcmp(a.data() + i, b.data() + i, sizeof(Out)) != 0) {
      Out x;
      Out y;
      std::memcpy(&x, a.data() + i, sizeof(Out));
      std::memcpy(&y, b.data() + i, sizeof(Out));
      std::ostringstream o;
      o.precision(17);
      o << " at element " << i / sizeof(Out) << ": " << x << " against " << y;
      return o.str();
    }
  }
  return a.size() == b.size() ? "" : " in size";
}

template <class In, class Out> fftn::shape<In, Out> shape_of(const batch_case& bc) {
  fftn::shape<In, Out> sh;
  if constexpr (std::is_same_v<In, Out>) {
    sh.dir = bc.dir;
  }
  sh.axes = {{.n = bc.n, .in_stride = bc.in.stride, .out_stride = bc.out.stride}};
  sh.batches = {{.count = bc.count, .in_dist = bc.in.dist, .out_dist = bc.out.dist}};
  return sh;
}

// The case through the lanes (forced, with `block`) and through the chain: the same bytes,
// for the whole batch and for sub-ranges. Returns the lanes plan's whole output.
template <class In, class Out>
std::vector<std::byte> lanes_equal_chain(fftn::isa target, const batch_case& bc,
                                         std::size_t in_extent, std::size_t out_extent,
                                         std::size_t block, const std::string& what,
                                         const build_settings& base = {}) {
  SCOPED_TRACE(what);
  const auto sh = shape_of<In, Out>(bc);
  const std::size_t in_elems = bc.in.span(bc.count, in_extent);
  const std::size_t out_elems = bc.out.span(bc.count, out_extent);
  build_settings lanes = base;
  lanes.strategy = strategy_choice::lanes;
  lanes.lane_block = block;
  build_settings chain = base;
  chain.strategy = strategy_choice::chain;
  // Row by row: batch leaves (X53) run some lengths as the leaf where the lanes run the chain.
  chain.batch_leaves = fftn::detail::batch_leaf_choice::never;
  std::string lanes_text;
  std::string chain_text;
  const auto got = run(sh, bc, lanes, target, in_elems, out_elems, 0, bc.count, lanes_text);
  const auto want = run(sh, bc, chain, target, in_elems, out_elems, 0, bc.count, chain_text);
  EXPECT_NE(lanes_text.find("strategy: lanes"), std::string::npos) << lanes_text;
  EXPECT_EQ(chain_text.find("strategy: lanes"), std::string::npos) << chain_text;
  // No stage copies a scratch tile onto itself (a scatter to a scratch destination once did).
  EXPECT_EQ(lanes_text.find(": a -> a"), std::string::npos) << lanes_text;
  EXPECT_EQ(lanes_text.find(": b -> b"), std::string::npos) << lanes_text;
  EXPECT_TRUE(got == want) << "lanes and chain differ" << first_difference<Out>(got, want) << "\n"
                           << lanes_text << chain_text;
  // Sub-ranges, out of place: the rows run equal the whole batch's and nothing else changes.
  if (!bc.in_place && bc.count > 2) {
    for (const auto& [first, count] : {
             std::pair<std::size_t, std::size_t>{1, bc.count - 1},
             {bc.count / 2, 1},
             {bc.count / 3, bc.count - bc.count / 3 - 1},
         }) {
      std::string ignored;
      const auto part = run(sh, bc, lanes, target, in_elems, out_elems, first, count, ignored);
      const auto part_chain =
          run(sh, bc, chain, target, in_elems, out_elems, first, count, ignored);
      EXPECT_TRUE(part == part_chain) << "sub-range " << first << " + " << count << "\n"
                                      << lanes_text;
    }
  }
  return got;
}

template <class T> std::string label(fftn::isa target, const char* kind, const batch_case& bc) {
  return std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double " : " float ") + kind +
         " n = " + std::to_string(bc.n) + ", " + std::to_string(bc.count) + " rows (in stride " +
         std::to_string(bc.in.stride) + " dist " + std::to_string(bc.in.dist) + ", out stride " +
         std::to_string(bc.out.stride) + " dist " + std::to_string(bc.out.dist) + ")" +
         (bc.in_place ? ", in place" : "") +
         (bc.dir == fftn::direction::backward ? ", backward" : "") + ", scale " +
         std::to_string(bc.scale);
}

// Row `row` of a c2c lanes output against the reference.
template <class T>
void c2c_against_reference(const batch_case& bc, const std::vector<std::byte>& out, std::size_t row,
                           const std::string& what) {
  using c = std::complex<T>;
  const std::size_t n = bc.n;
  fftn::test::input_source src(0x1A4E5ULL + bc.n * 131 + bc.count);
  const auto x = src.complexes<T>(bc.in.span(bc.count, n));
  std::vector<std::complex<double>> xin(n);
  for (std::size_t j = 0; j < n; ++j) {
    xin[j] = std::complex<double>(
        x[static_cast<std::size_t>(static_cast<std::ptrdiff_t>(row) * bc.in.dist +
                                   static_cast<std::ptrdiff_t>(j) * bc.in.stride)]);
  }
  const auto exact = fftn::test::reference::c2c(static_cast<int>(bc.dir), xin);
  std::vector<fftn::test::exact_complex> want(n);
  std::vector<c> got(n);
  for (std::size_t k = 0; k < n; ++k) {
    want[k] =
        fftn::test::exact_from(std::complex<double>(exact[k].re.hi, exact[k].im.hi) * bc.scale);
    std::memcpy(&got[k],
                out.data() + sizeof(c) * static_cast<std::size_t>(
                                             static_cast<std::ptrdiff_t>(row) * bc.out.dist +
                                             static_cast<std::ptrdiff_t>(k) * bc.out.stride),
                sizeof(c));
  }
  fftn::test::outcome o;
  o.within(fftn::test::compare<T>(std::span<const c>(got), want),
           fftn::test::derived_tolerance<T>(n, fftn::test::stable_fft_bound),
           what + ", row " + std::to_string(row));
  EXPECT_TRUE(o.passed()) << o.detail();
}

side_layout rows_with_gap(std::size_t extent, std::size_t gap) {
  return {.stride = 1, .dist = static_cast<std::ptrdiff_t>(extent + gap)};
}

template <class T> void c2c_cases(fftn::isa target) {
  const std::size_t c = complex_lanes<T>(target);
  using cx = std::complex<T>;
  for (const std::size_t n : {
           2U,  3U,  4U,  5U,  6U,  7U,  8U,  9U,   12U,  15U,  16U,
           20U, 24U, 32U, 36U, 48U, 60U, 64U, 100U, 128U, 256U,
       }) {
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      // Rows with gaps: a whole block, a block and a padded tail, fewer rows than a block.
      for (const std::size_t count : {std::size_t{8} * c, std::size_t{8} * c + 3, std::size_t{3}}) {
        const batch_case bc{
            .n = n,
            .count = count,
            .in = rows_with_gap(n, 3),
            .out = rows_with_gap(n, 5),
            .dir = dir,
        };
        const auto out =
            lanes_equal_chain<cx, cx>(target, bc, n, n, 8 * c, label<T>(target, "c2c", bc));
        if (n <= 64 && count == 3) {
          for (std::size_t row = 0; row < count; ++row) {
            c2c_against_reference<T>(bc, out, row, label<T>(target, "c2c", bc));
          }
        }
      }
    }
  }
  // In place with a scale; interleaved rows (element j of row t at count j + t, so the rows
  // are adjacent) in and out; a stride with gaps between rows; the rule's block.
  for (const std::size_t n : {8U, 12U, 64U, 100U}) {
    const std::size_t count = 5 * c + 1;
    const batch_case inplace{
        .n = n,
        .count = count,
        .in = rows_with_gap(n, 2),
        .out = rows_with_gap(n, 2),
        .in_place = true,
        .scale = 0.25,
    };
    lanes_equal_chain<cx, cx>(target, inplace, n, n, 2 * c, label<T>(target, "c2c", inplace));
    const side_layout interleaved{.stride = static_cast<std::ptrdiff_t>(count), .dist = 1};
    const batch_case columns{
        .n = n,
        .count = count,
        .in = interleaved,
        .out = interleaved,
        .scale = 0.5,
        .dir = fftn::direction::backward,
    };
    const auto out =
        lanes_equal_chain<cx, cx>(target, columns, n, n, 0, label<T>(target, "c2c", columns));
    c2c_against_reference<T>(columns, out, count - 1, label<T>(target, "c2c", columns));
    const batch_case strided{
        .n = n,
        .count = count,
        .in = {.stride = 2, .dist = static_cast<std::ptrdiff_t>(2 * n + 1)},
        .out = interleaved,
    };
    lanes_equal_chain<cx, cx>(target, strided, n, n, c, label<T>(target, "c2c", strided));
  }
}

// The unsigned integer of T's size, to compare values bit for bit.
template <class T>
using bits = std::conditional_t<sizeof(T) == sizeof(std::uint64_t), std::uint64_t, std::uint32_t>;

// Real rows whose complex side is interleaved (bin k of row t at count k + t). The chain
// runs such a row through its strided route (pairing at any stride, other arithmetic); the
// lanes read and write the tile alike for every layout, so they give each row the bits of
// the contiguous row through the chain, which is what is checked, in both directions.
template <class T> void strided_complex_side(fftn::isa target, std::size_t n, std::size_t count) {
  using cx = std::complex<T>;
  const std::size_t h1 = n / 2 + 1;
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
               " real rows of " + std::to_string(n) + " with an interleaved complex side, " +
               std::to_string(count) + " rows");
  build_settings lanes;
  lanes.strategy = strategy_choice::lanes;
  build_settings chain;
  chain.strategy = strategy_choice::chain;
  const auto ic = static_cast<std::ptrdiff_t>(count);
  fftn::test::input_source src(0x57D1ULL + n);
  const auto reals = src.reals<T>(n * count);
  const auto bins = src.complexes<T>(h1 * count);
  std::vector<cx> bins_interleaved(h1 * count);
  for (std::size_t t = 0; t < count; ++t) {
    for (std::size_t k = 0; k < h1; ++k) {
      bins_interleaved[k * count + t] = bins[t * h1 + k];
    }
  }
  const auto execute = [](const auto& sh, const build_settings& s, const void* in, void* out,
                          bool want_lanes) {
    const auto core = make_plan_core(fftn::detail::view_of(sh), {}, s);
    EXPECT_EQ(is_lanes(*core), want_lanes) << described(*core);
    const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
    fftn::detail::execute_core(*core, 0, fftn::detail::transform_count(*core), in, out, ws);
  };
  fftn::shape<T, cx> r2c_cols;
  r2c_cols.axes = {{.n = n, .out_stride = ic}};
  r2c_cols.batches = {{.count = count, .out_dist = 1}};
  std::vector<cx> got(h1 * count);
  execute(r2c_cols, lanes, reals.data(), got.data(), true);
  std::vector<cx> want(h1 * count);
  execute(fftn::rows_r2c<T>(n, count), chain, reals.data(), want.data(), false);
  for (std::size_t t = 0; t < count; ++t) {
    for (std::size_t k = 0; k < h1; ++k) {
      const cx g = got[k * count + t];
      const cx w = want[t * h1 + k];
      EXPECT_TRUE(std::bit_cast<bits<T>>(g.real()) == std::bit_cast<bits<T>>(w.real()) &&
                  std::bit_cast<bits<T>>(g.imag()) == std::bit_cast<bits<T>>(w.imag()))
          << "r2c row " << t << " bin " << k;
    }
  }
  fftn::shape<cx, T> c2r_cols;
  c2r_cols.axes = {{.n = n, .in_stride = ic}};
  c2r_cols.batches = {{.count = count, .in_dist = 1}};
  std::vector<T> got_real(n * count);
  execute(c2r_cols, lanes, bins_interleaved.data(), got_real.data(), true);
  std::vector<T> want_real(n * count);
  execute(fftn::rows_c2r<T>(n, count), chain, bins.data(), want_real.data(), false);
  EXPECT_EQ(std::memcmp(got_real.data(), want_real.data(), n * count * sizeof(T)), 0);
}

template <class T> void real_cases(fftn::isa target) {
  const std::size_t c = complex_lanes<T>(target);
  using cx = std::complex<T>;
  for (const std::size_t n :
       {4U, 6U, 8U, 10U, 16U, 20U, 24U, 32U, 48U, 64U, 100U, 128U, 200U, 256U}) {
    const std::size_t h1 = n / 2 + 1;
    for (const std::size_t count : {std::size_t{4} * c + 3, std::size_t{2}}) {
      for (const double scale : {1.0, 0.5}) {
        const batch_case r2c{
            .n = n,
            .count = count,
            .in = rows_with_gap(n, 4),
            .out = rows_with_gap(h1, 1),
            .scale = scale,
        };
        lanes_equal_chain<T, cx>(target, r2c, n, h1, 4 * c, label<T>(target, "r2c", r2c));
        const batch_case c2r{
            .n = n,
            .count = count,
            .in = rows_with_gap(h1, 3),
            .out = rows_with_gap(n, 2),
            .scale = scale,
        };
        lanes_equal_chain<cx, T>(target, c2r, h1, n, 4 * c, label<T>(target, "c2r", c2r));
      }
    }
    // In place in FFTW's padded rows; the complex side interleaved.
    const std::size_t count = 3 * c + 2;
    const auto padded = static_cast<std::ptrdiff_t>(2 * h1);
    const batch_case r2c_in{
        .n = n,
        .count = count,
        .in = {.stride = 1, .dist = padded},
        .out = {.stride = 1, .dist = padded / 2},
        .in_place = true,
    };
    lanes_equal_chain<T, cx>(target, r2c_in, n, h1, 0, label<T>(target, "r2c", r2c_in));
    const batch_case c2r_in{
        .n = n,
        .count = count,
        .in = {.stride = 1, .dist = padded / 2},
        .out = {.stride = 1, .dist = padded},
        .in_place = true,
        .scale = 2.0,
    };
    lanes_equal_chain<cx, T>(target, c2r_in, h1, n, 0, label<T>(target, "c2r", c2r_in));
    strided_complex_side<T>(target, n, count);
  }
}

TEST(Lanes, C2cEqualsTheChainBitForBit) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    c2c_cases<double>(t);
    c2c_cases<float>(t);
  }
}

TEST(Lanes, RealEqualsTheChainBitForBit) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    real_cases<double>(t);
    real_cases<float>(t);
  }
}

// Where the lanes cannot run, a forced lanes plan is a chain, and describe() says so: one
// row, a prime above the codelets (Bluestein), an input factor, a real row with a stride.
TEST(Lanes, ForcedOnlyWhereTheRowsCanBeLanes) {
  build_settings lanes;
  lanes.strategy = strategy_choice::lanes;
  const auto plan = [&](const auto& sh, const build_settings& s) {
    return make_plan_core(fftn::detail::view_of(sh), {}, s);
  };
  EXPECT_TRUE(is_lanes(*plan(fftn::rows_c2c<double>(64, 2, fftn::direction::forward), lanes)));
  EXPECT_FALSE(is_lanes(*plan(fftn::rows_c2c<double>(64, 1, fftn::direction::forward), lanes)));
  EXPECT_FALSE(is_lanes(*plan(fftn::rows_c2c<double>(97, 8, fftn::direction::forward), lanes)));
  EXPECT_FALSE(is_lanes(*plan(fftn::planes_c2c<double>(8, 8, 8, fftn::direction::forward), lanes)));
  const std::vector<std::complex<double>> factor(16, {1.0, 0.0});
  build_settings with_factor = lanes;
  with_factor.in_factor = factor.data();
  EXPECT_FALSE(
      is_lanes(*plan(fftn::rows_c2c<double>(16, 8, fftn::direction::forward), with_factor)));
  fftn::shape<double, std::complex<double>> strided_real;
  strided_real.axes = {{.n = 16, .in_stride = 2}};
  strided_real.batches = {{.count = 8, .in_dist = 1}};
  EXPECT_FALSE(is_lanes(*plan(strided_real, lanes)));
  EXPECT_FALSE(is_lanes(*plan(fftn::rows_r2c<double>(7, 8), lanes))); // odd: the full route
  build_settings chain;
  chain.strategy = strategy_choice::chain;
  EXPECT_FALSE(is_lanes(*plan(fftn::rows_c2c<double>(64, 64, fftn::direction::forward), chain)));
}

// The rules: a batch of at least C rows up to the profile's lanes_up_to (float, where X7
// found the lanes faster), a single row and longer rows not.
TEST(Lanes, PlannerChoice) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    const auto plan = [&](const auto& sh) {
      return make_plan_core(fftn::detail::view_of(sh), {.force_isa = t}, build_settings{});
    };
    const auto profile = fftn::detail::profile_of(
        static_cast<fftn::detail::target>(static_cast<unsigned char>(t)), complex_lanes<float>(t));
    SCOPED_TRACE(fftn::isa_name(t));
    if (profile.lanes_up_to >= 16) {
      EXPECT_TRUE(is_lanes(*plan(fftn::rows_c2c<float>(16, 64, fftn::direction::forward))));
      EXPECT_TRUE(is_lanes(*plan(fftn::rows_r2c<float>(16, 64))));
    }
    EXPECT_FALSE(is_lanes(*plan(fftn::rows_c2c<float>(16, 1, fftn::direction::forward))));
    EXPECT_FALSE(is_lanes(
        *plan(fftn::rows_c2c<float>(profile.lanes_up_to + 1, 64, fftn::direction::forward))));
  }
}

// A split run (DESIGN.md 6.3) forced on targets that do not choose one, so that the lanes'
// split passes (s multiplied by the block) run here: c2c and both real kinds, against the
// chain with the same forced split.
template <class In, class Out>
void split_case(fftn::isa t, const batch_case& bc, std::size_t in_extent, std::size_t out_extent,
                const char* kind, const build_settings& split) {
  lanes_equal_chain<In, Out>(t, bc, in_extent, out_extent, 0,
                             label<float>(t, kind, bc) + ", split forced", split);
  build_settings forced = split;
  forced.strategy = strategy_choice::lanes;
  const auto core =
      make_plan_core(fftn::detail::view_of(shape_of<In, Out>(bc)), {.force_isa = t}, forced);
  EXPECT_NE(described(*core).find("split lane"), std::string::npos) << described(*core);
}

TEST(Lanes, ForcedSplitRunEqualsTheChain) {
  using cx = std::complex<float>;
  build_settings split;
  split.layout = fftn::detail::layout_choice::split;
  split.radices = {16, 4, 4};
  split.fuse_real_radix = 1;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t count : {std::size_t{19}, std::size_t{64}}) {
      const batch_case c2c{
          .n = 256,
          .count = count,
          .in = rows_with_gap(256, 3),
          .out = rows_with_gap(256, 1),
      };
      split_case<cx, cx>(t, c2c, 256, 256, "c2c", split);
      // Real rows of 512 points: the half-length chain of 256 takes the forced radices.
      const batch_case r2c{
          .n = 512,
          .count = count,
          .in = rows_with_gap(512, 2),
          .out = rows_with_gap(257, 1),
      };
      split_case<float, cx>(t, r2c, 512, 257, "r2c", split);
      const batch_case c2r{
          .n = 512,
          .count = count,
          .in = rows_with_gap(257, 1),
          .out = rows_with_gap(512, 2),
      };
      split_case<cx, float>(t, c2r, 257, 512, "c2r", split);
    }
  }
}

// Runs a plan of shape sh (forced strategy s, blocks of 8) over transforms [first, first +
// count) on buffers of in_span and out_span elements whose element 0 is in_base and out_base
// elements into them (a negative distance puts later rows before it); returns the output
// buffer, which starts as a canary pattern.
template <class In, class Out>
std::vector<std::byte> run_at(const fftn::shape<In, Out>& sh, strategy_choice s, fftn::isa target,
                              std::size_t in_span, std::size_t in_base, std::size_t out_span,
                              std::size_t out_base, std::size_t first, std::size_t count,
                              std::string& text) {
  build_settings settings;
  settings.strategy = s;
  settings.lane_block = 8;
  const auto core = make_plan_core(fftn::detail::view_of(sh), {.force_isa = target}, settings);
  text = described(*core);
  fftn::test::input_source src(0xBA7C4ULL + in_span);
  std::vector<In> in;
  if constexpr (std::is_floating_point_v<In>) {
    in = src.reals<In>(in_span);
  } else {
    in = src.complexes<typename In::value_type>(in_span);
  }
  std::vector<Out> out(out_span);
  std::memset(static_cast<void*>(out.data()), 0xa5, out_span * sizeof(Out));
  const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  fftn::detail::execute_core(*core, first, count, in.data() + in_base, out.data() + out_base, ws);
  std::vector<std::byte> bytes(out_span * sizeof(Out));
  std::memcpy(bytes.data(), out.data(), bytes.size());
  return bytes;
}

template <class In, class Out>
void layouts_equal(fftn::isa target, const fftn::shape<In, Out>& sh, std::size_t in_span,
                   std::size_t in_base, std::size_t out_span, std::size_t out_base,
                   std::size_t transforms, const std::string& what) {
  SCOPED_TRACE(what);
  for (const auto& [first, count] : {
           std::pair<std::size_t, std::size_t>{0, transforms},
           {1, transforms - 2},
           {transforms / 2 - 1, 3},
       }) {
    std::string lanes_text;
    std::string chain_text;
    const auto got = run_at(sh, strategy_choice::lanes, target, in_span, in_base, out_span,
                            out_base, first, count, lanes_text);
    const auto want = run_at(sh, strategy_choice::chain, target, in_span, in_base, out_span,
                             out_base, first, count, chain_text);
    EXPECT_NE(lanes_text.find("strategy: lanes"), std::string::npos) << lanes_text;
    EXPECT_TRUE(got == want) << "transforms " << first << " + " << count
                             << first_difference<Out>(got, want) << "\n"
                             << lanes_text;
  }
}

// Negative distances (rows at descending addresses) and two batch dimensions, whose blocks
// of rows stay within the innermost one: the lanes against the chain, whole batches and
// ranges that start and end mid-block and cross the outer dimension.
TEST(Lanes, NegativeDistancesAndTwoBatchDimensions) {
  using cx = std::complex<double>;
  using fx = std::complex<float>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    const std::string isa = fftn::isa_name(t);
    {
      constexpr std::size_t n = 16;
      constexpr std::size_t count = 21;
      constexpr std::ptrdiff_t dist = -static_cast<std::ptrdiff_t>(n + 3);
      fftn::shape<cx, cx> sh;
      sh.axes = {{.n = n}};
      sh.batches = {{.count = count, .in_dist = dist, .out_dist = dist}};
      const std::size_t span = (count - 1) * (n + 3) + n;
      layouts_equal(t, sh, span, span - n, span, span - n, count, isa + " c2c, negative distance");
    }
    {
      constexpr std::size_t n = 32;
      constexpr std::size_t count = 13;
      constexpr std::size_t h1 = n / 2 + 1;
      fftn::shape<float, fx> sh;
      sh.axes = {{.n = n}};
      sh.batches = {
          {
              .count = count,
              .in_dist = -static_cast<std::ptrdiff_t>(n),
              .out_dist = -static_cast<std::ptrdiff_t>(h1),
          },
      };
      layouts_equal(t, sh, count * n, (count - 1) * n, count * h1, (count - 1) * h1, count,
                    isa + " r2c float, negative distances");
      fftn::shape<fx, float> back;
      back.axes = {{.n = n}};
      back.batches = {
          {
              .count = count,
              .in_dist = -static_cast<std::ptrdiff_t>(h1),
              .out_dist = -static_cast<std::ptrdiff_t>(n),
          },
      };
      layouts_equal(t, back, count * h1, (count - 1) * h1, count * n, (count - 1) * n, count,
                    isa + " c2r float, negative distances");
    }
    {
      // 3 groups of 11 rows of 12 points, the groups 200 elements apart (with gaps), the rows
      // of a group 13 apart.
      constexpr std::size_t n = 12;
      fftn::shape<cx, cx> sh;
      sh.axes = {{.n = n}};
      sh.batches = {
          {.count = 3, .in_dist = 200, .out_dist = 200},
          {.count = 11, .in_dist = 13, .out_dist = 13},
      };
      const std::size_t span = 2 * 200 + 10 * 13 + n;
      layouts_equal(t, sh, span, 0, span, 0, 33, isa + " c2c, two batch dimensions");
    }
  }
}

// The rules never take the lanes for real rows whose complex side is strided: the lanes
// give them the contiguous row's bits, the chain's strided route others, and avx2 and
// avx512, whose thresholds differ, must agree (DESIGN.md 6.2). With the contiguous control;
// meaningful where a profile takes the lanes at all (float on x86).
TEST(Lanes, RulesKeepStridedComplexRealRowsOnTheChain) {
  using fx = std::complex<float>;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(t));
    const auto profile = fftn::detail::profile_of(
        static_cast<fftn::detail::target>(static_cast<unsigned char>(t)), complex_lanes<float>(t));
    const auto plan = [&](const auto& sh) {
      return make_plan_core(fftn::detail::view_of(sh), {.force_isa = t}, build_settings{});
    };
    for (const std::size_t n : {std::size_t{16}, std::size_t{64}}) {
      for (const std::size_t count : {std::size_t{5}, std::size_t{64}}) {
        const auto ic = static_cast<std::ptrdiff_t>(count);
        fftn::shape<float, fx> r2c;
        r2c.axes = {{.n = n, .out_stride = ic}};
        r2c.batches = {{.count = count, .out_dist = 1}};
        fftn::shape<fx, float> c2r;
        c2r.axes = {{.n = n, .in_stride = ic}};
        c2r.batches = {{.count = count, .in_dist = 1}};
        EXPECT_FALSE(is_lanes(*plan(r2c))) << n << " x " << count;
        EXPECT_FALSE(is_lanes(*plan(c2r))) << n << " x " << count;
        const bool expected = n <= profile.lanes_up_to && count >= complex_lanes<float>(t);
        EXPECT_EQ(is_lanes(*plan(fftn::rows_r2c<float>(n, count))), expected)
            << n << " x " << count;
      }
    }
  }
}

} // namespace
