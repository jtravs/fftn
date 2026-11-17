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

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
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

// Planner v1's radices multiply to the length on every profile, and every radix is a
// codelet's or a prime for the generic pass.
TEST(Rules, RadicesMultiplyToTheLength) {
  for (const std::size_t lanes : {1U, 2U, 4U, 8U}) {
    const auto profile = fftn::detail::profile_of(fftn::detail::target::avx2, lanes);
    for (std::size_t n = 1; n <= 4096; ++n) {
      std::size_t product = 1;
      for (const std::size_t r : fftn::detail::rule_radices(n, profile)) {
        product *= r;
        EXPECT_TRUE(r <= 16 || r == 20 || r == 25 || r == 32 || r == 64 || is_prime(r))
            << n << ": radix " << r;
      }
      EXPECT_EQ(product, n) << n;
    }
    EXPECT_TRUE(fftn::detail::rule_radices(1, profile).empty());
    EXPECT_EQ(fftn::detail::rule_radices(10007, profile), (std::vector<std::size_t>{10007}));
    EXPECT_EQ(fftn::detail::rule_radices(16, profile), (std::vector<std::size_t>{16}));
    EXPECT_EQ(fftn::detail::rule_radices(4097, profile), (std::vector<std::size_t>{17, 241}));
    EXPECT_EQ(fftn::detail::rule_radices(1U << 24U, profile).size(), 7U) << lanes;
  }
}

// The leaf limit (experiment X8): 16 by default; above it, a length with a codelet is one
// leaf up to the limit, and any other length is a chain as before.
TEST(Rules, LeafLimit) {
  using fftn::detail::rule_radices;
  for (const std::size_t lanes : {1U, 2U, 4U, 8U}) {
    auto profile = fftn::detail::profile_of(fftn::detail::target::avx2, lanes);
    EXPECT_EQ(profile.leaf_limit, 16U);
    EXPECT_GT(rule_radices(32, profile).size(), 1U) << lanes;
    EXPECT_GT(rule_radices(64, profile).size(), 1U) << lanes;
    const auto chain48 = rule_radices(48, profile);
    const auto chain128 = rule_radices(128, profile);
    profile.leaf_limit = 64;
    EXPECT_EQ(rule_radices(32, profile), (std::vector<std::size_t>{32}));
    EXPECT_EQ(rule_radices(64, profile), (std::vector<std::size_t>{64}));
    EXPECT_EQ(rule_radices(48, profile), chain48);
    EXPECT_EQ(rule_radices(128, profile), chain128);
    EXPECT_EQ(rule_radices(12, profile), (std::vector<std::size_t>{12}));
    profile.leaf_limit = 32;
    EXPECT_EQ(rule_radices(32, profile), (std::vector<std::size_t>{32}));
    EXPECT_GT(rule_radices(64, profile).size(), 1U);
    profile.leaf_limit = 1; // lengths up to 16 stay leaves
    EXPECT_EQ(rule_radices(16, profile), (std::vector<std::size_t>{16}));
  }
  // X8's verdict: 32 on NEON in double (C = 1), and X14's on x86_v2 in double; 16
  // everywhere else.
  using fftn::detail::target;
  for (const target t : {
           target::scalar,
           target::x86_v2,
           target::avx,
           target::avx2,
           target::avx512,
           target::neon,
           target::neon_fcma,
       }) {
    for (const std::size_t lanes : {1U, 2U, 4U, 8U}) {
      const bool leaf32 =
          (t == target::neon || t == target::neon_fcma || t == target::x86_v2) && lanes == 1;
      const auto p = fftn::detail::profile_of(t, lanes);
      EXPECT_EQ(p.leaf_limit, leaf32 ? 32U : 16U);
      EXPECT_EQ(rule_radices(32, p).size() == 1, leaf32);
      EXPECT_GT(rule_radices(64, p).size(), 1U);
    }
  }
}

// Every target has a profile: the lanes it is given; NEON's own tables at its C (1 and 2),
// AVX-512's own tables at C = 4 (double) and 8 (float), otherwise AVX2's C = 2 table up to
// two lanes and its C = 4 table from four.
TEST(Rules, ProfilesOfEveryTarget) {
  using fftn::detail::target;
  for (const target t : {
           target::scalar,
           target::x86_v2,
           target::avx,
           target::avx2,
           target::avx512,
           target::neon,
           target::neon_fcma,
       }) {
    for (const std::size_t lanes : {1U, 2U, 4U, 8U}) {
      const auto p = fftn::detail::profile_of(t, lanes);
      EXPECT_EQ(p.lanes, lanes);
      ASSERT_NE(p.costs, nullptr);
      const bool neon_own = (t == target::neon || t == target::neon_fcma) && lanes <= 2;
      if (neon_own) {
        EXPECT_NE(p.costs, fftn::detail::profile_of(target::avx2, 2).costs);
        EXPECT_EQ(p.costs, fftn::detail::profile_of(target::neon_fcma, lanes).costs);
      } else if (t == target::avx512 && lanes >= 4) {
        EXPECT_NE(p.costs, fftn::detail::profile_of(target::avx2, 4).costs);
      } else {
        EXPECT_EQ(p.costs, fftn::detail::profile_of(target::avx2, lanes <= 2 ? 2 : 4).costs);
      }
    }
  }
  EXPECT_NE(fftn::detail::profile_of(target::avx2, 2).costs,
            fftn::detail::profile_of(target::avx2, 4).costs);
}

// The recorded radices of every 73-smooth length up to 20000 for 1, 2, 4 and 8 lanes
// (data/rule_radices.txt, generated from the specification, DESIGN.md 11.2, by an
// independent implementation).
TEST(Rules, RecordedTable) {
  std::ifstream in(FFTN_CORE_DATA "/rule_radices.txt");
  ASSERT_TRUE(in.good()) << "cannot read " << FFTN_CORE_DATA "/rule_radices.txt";
  const std::array<std::size_t, 4> lanes{1, 2, 4, 8};
  std::size_t lines = 0;
  std::size_t failures = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::size_t n = 0;
    fields >> n;
    for (const std::size_t c : lanes) {
      std::string bar;
      fields >> bar;
      ASSERT_EQ(bar, "|") << line;
      std::vector<std::size_t> want;
      while (fields.peek() == ' ') {
        fields.get();
        if (fields.peek() == '|' || fields.peek() == std::istringstream::traits_type::eof()) {
          break;
        }
        std::size_t r = 0;
        fields >> r;
        want.push_back(r);
      }
      const auto profile = fftn::detail::profile_of(fftn::detail::target::avx2, c);
      if (fftn::detail::rule_radices(n, profile) != want && ++failures <= 8) {
        ADD_FAILURE() << "n = " << n << ", " << c << " lanes: not as recorded";
      }
    }
    ++lines;
  }
  EXPECT_EQ(lines, 5208U);
  EXPECT_EQ(failures, 0U);
}

// Winners of experiment X2 on AVX2 (EPYC 7763) that the costs reproduce, and the measured
// overrides: double has two lanes, float four.
TEST(Rules, MeasuredExamples) {
  using v = std::vector<std::size_t>;
  const auto f64 = fftn::detail::profile_of(fftn::detail::target::avx2, 2);
  const auto f32 = fftn::detail::profile_of(fftn::detail::target::avx2, 4);
  const auto r = [](std::size_t n, const fftn::detail::target_profile& p) {
    return fftn::detail::rule_radices(n, p);
  };
  EXPECT_EQ(r(100, f64), (v{10, 10}));
  EXPECT_EQ(r(100, f32), (v{4, 5, 5}));
  EXPECT_EQ(r(945, f64), (v{15, 7, 9}));
  EXPECT_EQ(r(960, f64), (v{8, 10, 12}));
  EXPECT_EQ(r(1000, f64), (v{10, 10, 10}));
  EXPECT_EQ(r(1000, f32), (v{4, 5, 5, 10}));
  EXPECT_EQ(r(1024, f64), (v{4, 16, 16}));
  EXPECT_EQ(r(1024, f32), (v{8, 8, 16}));
  // Overrides.
  EXPECT_EQ(r(42, f64), (v{2, 3, 7}));
  EXPECT_EQ(r(490, f64), (v{2, 7, 5, 7}));
  EXPECT_EQ(r(700, f32), (v{4, 5, 5, 7}));
  EXPECT_EQ(r(4096, f64), (v{8, 8, 8, 8}));    // experiment X18
  EXPECT_EQ(r(16384, f64), (v{8, 16, 16, 8})); // the planner refresh
  EXPECT_EQ(r(3000, f64), (v{10, 15, 20}));    // experiment X38: a radix-20 pass
  EXPECT_EQ(r(3600, f32), (v{16, 15, 15}));
  // Primes above 16 go last (the chain makes them a codelet pass for 17, 19 and 23,
  // experiment X27, else a Bluestein stage).
  EXPECT_EQ(r(34, f64), (v{2, 17}));
  EXPECT_EQ(r(58, f64), (v{2, 29}));
}

// The planner refresh (experiment X2, Xeon Gold 6240): AVX-512 in double (C = 4) has
// avx2_float's costs, so its choice is AVX2's at C = 4 except at either table's overrides;
// in float (C = 8) its own fitted costs. And measured winners each was given.
TEST(Rules, Avx512Tables) {
  using fftn::detail::target;
  using v = std::vector<std::size_t>;
  const auto f64 = fftn::detail::profile_of(target::avx512, 4);
  const auto f32 = fftn::detail::profile_of(target::avx512, 8);
  const auto avx2 = fftn::detail::profile_of(target::avx2, 4);
  EXPECT_NE(f64.costs, avx2.costs);
  EXPECT_NE(f32.costs, avx2.costs);
  for (std::size_t n = 1; n <= 20000; ++n) {
    const auto at_n = [n](const auto& o) { return o.n == n; };
    const bool own = std::ranges::any_of(f64.costs->overrides, at_n) ||
                     std::ranges::any_of(avx2.costs->overrides, at_n);
    if (!own) {
      EXPECT_EQ(fftn::detail::rule_radices(n, f64), fftn::detail::rule_radices(n, avx2)) << n;
    }
  }
  const auto r = [](std::size_t n, const fftn::detail::target_profile& p) {
    return fftn::detail::rule_radices(n, p);
  };
  EXPECT_EQ(r(768, f64), (v{8, 12, 8}));
  EXPECT_EQ(r(1000, f64), (v{4, 10, 5, 5}));
  EXPECT_EQ(r(5000, f64), (v{20, 10, 5, 5}));   // X39 arm D: a radix-20 first pass
  EXPECT_EQ(r(25000, f64), (v{4, 25, 10, 25})); // X42 arm B3: radix-25 passes
  EXPECT_EQ(r(140, f32), (v{10, 14}));
  EXPECT_EQ(r(256, f32), (v{32, 8}));
  EXPECT_EQ(r(512, f32), (v{32, 16}));
  EXPECT_EQ(r(1024, f32), (v{16, 16, 4}));
  EXPECT_EQ(r(2048, f32), (v{16, 8, 16}));
  EXPECT_EQ(r(4096, f32), (v{32, 8, 16}));
}

// Experiment X39: a radix-20-first override (avx512 double, 5000) stays first in r2c, whose
// pairing fuses into the last pass; a c2r line, whose pairing takes the first pass, takes the
// costs' sequence instead of moving the radix 20 behind a small radix.
TEST(Rules, C2rTakesTheCostsWhereARadix20OverrideWouldMove) {
  const build_settings s;
  const fftn::options avx512{.force_isa = fftn::isa::avx512};
  std::string c2r;
  std::string r2c;
  try {
    c2r = describe(
        *make_plan_core(fftn::detail::view_of(fftn::rows_c2r<double>(10000, 1)), avx512, s));
    r2c = describe(
        *make_plan_core(fftn::detail::view_of(fftn::rows_r2c<double>(10000, 1)), avx512, s));
  } catch (const std::exception&) {
    GTEST_SKIP() << "no avx512 here";
  }
  EXPECT_NE(r2c.find("20 x 10 x 5 x 5"), std::string::npos) << r2c;
  EXPECT_EQ(c2r.find("x 20"), std::string::npos) << c2r;
}

// Runs a plan core on a fresh copy of `input` and returns the output bytes.
// describe() says where the radices came from: a plan with build_settings.radices names
// the knob (fbench's RADICES option reaches it, experiment X2), a plan without does not.
TEST(Rules, DescribeSaysWhenRadicesAreForced) {
  const auto shape = fftn::rows_c2c<double>(256, 1, fftn::direction::forward);
  const fftn::options defaults;
  build_settings forced;
  forced.radices = {2, 8, 16};
  const build_settings rules;
  const std::string with =
      describe(*make_plan_core(fftn::detail::view_of(shape), defaults, forced));
  const std::string without =
      describe(*make_plan_core(fftn::detail::view_of(shape), defaults, rules));
  EXPECT_NE(with.find("radices from build_settings.radices"), std::string::npos) << with;
  EXPECT_NE(with.find("radices 2 x 8 x 16\n"), std::string::npos) << with;
  EXPECT_EQ(without.find("build_settings"), std::string::npos) << without;
  EXPECT_NE(without.find("radices from planner v1\n"), std::string::npos) << without;
}

// describe() names a Bluestein stage, its convolution length M and M's radices, in the
// axis line and in the stages (strategies/chain.hpp); 262 = 2 x 131 is not 7-smooth.
TEST(Rules, DescribeNamesTheBluesteinStage) {
  const auto shape = fftn::rows_c2c<double>(std::size_t{4} * 263, 1, fftn::direction::forward);
  // Forced, since on targets with C >= 4 the rules make 1052 one stage (X36, X36b).
  build_settings forced;
  forced.radices = {4, 263};
  const auto core = make_plan_core(fftn::detail::view_of(shape), fftn::options(), forced);
  const std::string text = describe(*core);
  EXPECT_NE(text.find("radices 4 x bluestein 263 (M "), std::string::npos) << text;
  EXPECT_NE(text.find("bluestein in, chirp forward (Q 263, M "), std::string::npos) << text;
  EXPECT_NE(text.find("bluestein convolution: "), std::string::npos) << text;
  EXPECT_NE(text.find("bluestein out, chirp forward (Q 263, M "), std::string::npos) << text;
  EXPECT_NE(text.find("copies 4"), std::string::npos) << text;
}

// And a Rader stage (experiment X27): a prime whose q - 1 is 13-smooth, its convolution
// length L = q - 1 and L's radices, the primitive root.
TEST(Rules, DescribeNamesTheRaderStage) {
  const auto shape = fftn::rows_c2c<double>(std::size_t{4} * 257, 1, fftn::direction::forward);
  const auto core = make_plan_core(fftn::detail::view_of(shape), fftn::options(), build_settings{});
  const std::string text = describe(*core);
  EXPECT_NE(text.find("radices 4 x rader 257 (L 256 = "), std::string::npos) << text;
  EXPECT_NE(text.find("rader in, gather forward (Q 257, L 256, root 3, copies 4"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("rader convolution: "), std::string::npos) << text;
  EXPECT_NE(text.find("rader out, scatter forward (Q 257, L 256, root 3, copies 4"),
            std::string::npos)
      << text;
}

// The Rader rule: primes without a codelet in the profile's range whose q - 1 is 13-smooth,
// with their smallest primitive root; the range from experiment X27.
TEST(Rules, RaderRoot) {
  using fftn::detail::rader_root;
  using fftn::detail::target;
  const auto neon = fftn::detail::profile_of(target::neon_fcma, 1);
  EXPECT_EQ(rader_root(29, neon), 2U);
  EXPECT_EQ(rader_root(31, neon), 3U);
  EXPECT_EQ(rader_root(41, neon), 6U);
  EXPECT_EQ(rader_root(97, neon), 5U);
  EXPECT_EQ(rader_root(257, neon), 3U);
  EXPECT_EQ(rader_root(8191, neon), 17U); // 8190 = 2 x 3^2 x 5 x 7 x 13
  EXPECT_EQ(rader_root(40961, neon), 3U);
  EXPECT_EQ(rader_root(65537, neon), 0U); // NEON's limit, 2^16
  EXPECT_EQ(rader_root(17, neon), 0U);    // a codelet
  EXPECT_EQ(rader_root(47, neon), 0U);    // 46 = 2 x 23
  EXPECT_EQ(rader_root(4093, neon), 0U);  // 4092 = 4 x 3 x 11 x 31
  EXPECT_EQ(rader_root(91, neon), 0U);    // 7 x 13
  const auto avx2_double = fftn::detail::profile_of(target::avx2, 2);
  EXPECT_EQ(rader_root(29, avx2_double), 2U);
  EXPECT_EQ(rader_root(65537, avx2_double), 3U); // no limit below 2^24 on x86
  const auto avx512_float = fftn::detail::profile_of(target::avx512, 8);
  EXPECT_EQ(rader_root(97, avx512_float), 0U); // from 16 C = 128 with C = 8
  EXPECT_EQ(rader_root(257, avx512_float), 3U);
}

// The Bluestein length (experiment X36): at least 2q - 1, 7-smooth; with a limit on its
// power-of-two part, no candidate exceeds it, and the candidates then run past the next
// power of two (131071: 2q - 1 = 262141, whose only 7-smooth length up to 2^18 is 2^18).
TEST(Rules, BluesteinLength) {
  using fftn::detail::bluestein_length;
  const auto neon = fftn::detail::profile_of(fftn::detail::target::neon_fcma, 1);
  const auto power_part = [](std::size_t m) { return m & (~m + 1); };
  const auto smooth = [](std::size_t m) {
    for (const std::size_t f : {2U, 3U, 5U, 7U}) {
      while (m % f == 0) {
        m /= f;
      }
    }
    return m == 1;
  };
  EXPECT_EQ(bluestein_length(131071, neon), std::size_t{1} << 18U);
  for (const std::size_t q : {29U, 509U, 4093U, 65537U, 131071U, 999983U}) {
    for (const std::size_t limit : {std::size_t{64}, std::size_t{2048}}) {
      const std::size_t m = bluestein_length(q, neon, limit);
      EXPECT_GE(m, 2 * q - 1) << q;
      EXPECT_LT(m, 4 * q) << q;
      EXPECT_TRUE(smooth(m)) << q << " " << m;
      EXPECT_LE(power_part(m), limit) << q << " " << m;
    }
  }
  // A limit above the next power of two changes nothing.
  EXPECT_EQ(bluestein_length(509, neon, 1024), bluestein_length(509, neon));
}

std::vector<std::byte> run(const fftn::detail::plan_core& core, std::vector<std::byte> input,
                           std::size_t out_bytes) {
  std::vector<std::byte> output(out_bytes);
  const fftn::workspace ws(fftn::detail::workspace_bytes(core, 1));
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
  // The fine plan really is divided: a stage into ranges of units, or a phase of lines
  // (planes) into runs of lines.
  std::size_t chunked = 0;
  for (const auto& rec : fine->prog.records) {
    if (const auto* st = std::get_if<fftn::detail::stage>(&rec); st != nullptr) {
      chunked += st->chunks > 1 ? 1 : 0;
    }
  }
  for (const auto& ph : fine->graph.phases) {
    chunked += ph.nest != 0 && ph.chunks > 1 ? 1 : 0;
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

// Long rows are divided by default, so the phase 3 engine has chunks to share: a chain's
// stages into ranges of units, a grid's phases into blocks of columns (its stages run one
// block each). Which strategy a long row gets depends on the machine's caches, so both are
// forced here.
TEST(Chunks, LongRowsHaveSeveralByDefault) {
  fftn::detail::build_settings chain;
  chain.strategy = fftn::detail::strategy_choice::chain;
  const auto core = make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<double>(16384, 1, fftn::direction::forward)),
      fftn::options(), chain);
  for (const auto& rec : core->prog.records) {
    if (const auto* st = std::get_if<fftn::detail::stage>(&rec); st != nullptr) {
      EXPECT_GT(st->chunks, 1U) << st->name;
    }
  }
  fftn::detail::build_settings grid;
  grid.strategy = fftn::detail::strategy_choice::grid;
  const auto g = make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<double>(16384, 1, fftn::direction::forward)),
      fftn::options(), grid);
  ASSERT_FALSE(g->graph.phases.empty());
  for (const auto& ph : g->graph.phases) {
    EXPECT_GT(ph.nest, 0U);
    EXPECT_GT(ph.chunks, 1U);
  }
}

} // namespace
