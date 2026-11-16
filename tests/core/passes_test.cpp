// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The passes of DESIGN.md 8.2 forced in turn (AGENTS.md section 5): every codelet radix as
// the grouped first pass, a middle lane pass and the last lane pass, with copy counts that
// leave tails on every SIMD target; the generic pass in every position; every edge variant
// on every pass kind that has it; on every instruction set the machine runs, in double
// and float, against the quadruple-precision reference. And the acceptance test of task
// 1.2: a fused scale costs no extra pass and gives exactly the scaled result. And the
// targets whose vocabularies round alike (neon and neon_fcma, avx2 and avx512) give the
// same bits with the same passes.

#include "core/plan_core.hpp"
#include "core/program.hpp"
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
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;
using fftn::detail::plan_core;

template <class T> using cvec = std::vector<std::complex<T>>;

// The Bluestein threshold that keeps every prime factor in generic passes.
constexpr std::size_t no_bluestein = static_cast<std::size_t>(-1);

std::vector<std::string> stage_names(const plan_core& core) {
  std::vector<std::string> names;
  for (const auto& rec : core.prog.records) {
    if (const auto* st = std::get_if<fftn::detail::stage>(&rec); st != nullptr) {
      names.push_back(st->name);
    }
  }
  return names;
}

std::string joined(const std::vector<std::string>& names) {
  std::string text;
  for (const std::string& n : names) {
    text += "\n  " + n;
  }
  return text;
}

template <class In, class Out>
std::vector<Out> run(const plan_core& core, const std::vector<In>& x, std::size_t out_count) {
  std::vector<Out> y(out_count);
  const fftn::workspace ws(fftn::detail::workspace_bytes(core, 1));
  fftn::detail::execute_core(core, 0, fftn::detail::transform_count(core), x.data(), y.data(), ws);
  return y;
}

// The same in place (c2c, where both sides have the same layout).
template <class C> std::vector<C> run_in_place(const plan_core& core, const std::vector<C>& x) {
  std::vector<C> y = x;
  const fftn::workspace ws(fftn::detail::workspace_bytes(core, 1));
  fftn::detail::execute_core(core, 0, fftn::detail::transform_count(core), y.data(), y.data(), ws);
  return y;
}

// One row of n points through a plan built with these settings, against the reference:
// y = scale * out_factor . DFT(in_factor . x), the factors element by element.
template <class T>
void check_row(fftn::isa target, std::size_t n, fftn::direction dir, const build_settings& base,
               double scale, bool in_factor, bool out_factor, bool scale_on_input,
               const std::vector<std::string>& expect_in_names, std::size_t in_stride = 1,
               std::size_t out_stride = 1) {
  const std::string what = std::string(fftn::isa_name(target)) +
                           (sizeof(T) == 8 ? " double" : " float") + ", n = " + std::to_string(n) +
                           (dir == fftn::direction::forward ? " forward" : " backward");
  SCOPED_TRACE(what);
  fftn::test::input_source src(0x9A55ULL + n);
  const cvec<T> x = src.complexes<T>(n);
  const cvec<T> fin = src.complexes<T>(n);
  const cvec<T> fout = src.complexes<T>(n);
  build_settings settings = base;
  settings.in_factor = in_factor ? fin.data() : nullptr;
  settings.out_factor = out_factor ? fout.data() : nullptr;
  settings.scale_on_input = scale_on_input;
  const fftn::shape<std::complex<T>, std::complex<T>> shape{
      .dir = dir,
      .axes =
          {
              {
                  .n = n,
                  .in_stride = static_cast<std::ptrdiff_t>(in_stride),
                  .out_stride = static_cast<std::ptrdiff_t>(out_stride),
              },
          },
      .batches = {},
  };
  const auto core =
      make_plan_core(fftn::detail::view_of(shape), {.scale = scale, .force_isa = target}, settings);
  const auto names = stage_names(*core);
  const std::string listing = joined(names);
  for (const std::string& part : expect_in_names) {
    EXPECT_NE(listing.find(part), std::string::npos) << "no stage '" << part << "' in" << listing;
  }
  cvec<T> xs(n * in_stride);
  for (std::size_t j = 0; j < n; ++j) {
    xs[j * in_stride] = x[j];
  }
  const cvec<T> ys = run<std::complex<T>, std::complex<T>>(*core, xs, n * out_stride);
  cvec<T> y(n);
  for (std::size_t k = 0; k < n; ++k) {
    y[k] = ys[k * out_stride];
  }

  // The expected values, in double: the products with factors of T round at most once each
  // (exactly, for float), which input_rounding() allows for, twice.
  std::vector<std::complex<double>> xin(n);
  for (std::size_t j = 0; j < n; ++j) {
    xin[j] = std::complex<double>(x[j]) * (in_factor ? std::complex<double>(fin[j]) : 1.0);
  }
  const auto exact = fftn::test::reference::c2c(static_cast<int>(dir), xin);
  std::vector<fftn::test::exact_complex> want(n);
  for (std::size_t k = 0; k < n; ++k) {
    const std::complex<double> e(exact[k].re.hi, exact[k].im.hi);
    const std::complex<double> f = out_factor ? std::complex<double>(fout[k]) : 1.0;
    want[k] = fftn::test::exact_from(e * f * scale);
  }
  const bool bluestein =
      listing.find("bluestein") != std::string::npos || listing.find("rader") != std::string::npos;
  const auto tol = fftn::test::derived_tolerance<T>(n, bluestein ? fftn::test::bluestein_bound(n)
                                                                 : fftn::test::stable_fft_bound) +
                   fftn::test::input_rounding<T>() * 2;
  fftn::test::outcome o;
  o.within(fftn::test::compare<T>(std::span<const std::complex<T>>(y), want), tol, what);
  EXPECT_TRUE(o.passed()) << o.detail() << "stages:" << listing;
}

template <class T> void every_codelet_in_every_position(fftn::isa target) {
  for (const std::uint8_t radix : fftn::detail::gen::radices) {
    const std::size_t r = radix;
    const std::string rs = "radix-" + std::to_string(r);
    struct sequence {
      std::vector<std::size_t> radices;
      std::vector<std::string> names;
    };
    // Grouped first passes with m from 2 to 16 (tails where C does not divide m), middle
    // lane passes with s = 3 and 4 (tails where C does not divide s), last lane passes with
    // s from 2 to 16; and R as a leaf. Above 16, R x R would cost seconds of reference
    // evaluation per row (O(n^2) in quadruple precision), so R x 8 and 8 x R stand in for
    // it: whole blocks of C on every target, in the first and in the last pass.
    std::vector<sequence> sequences{
        {.radices = {r, 2}, .names = {"grouped " + rs, "lane radix-2"}},
        {.radices = {2, r}, .names = {"grouped radix-2", "lane " + rs + " forward (m 1"}},
        {.radices = {3, r, 2}, .names = {"lane " + rs + " forward (m 2, s 3)"}},
        {.radices = {4, r, 4}, .names = {"lane " + rs + " forward (m 4, s 4)"}},
        {.radices = {r}, .names = {"leaf " + rs}},
    };
    if (r <= 16) {
      sequences.push_back({.radices = {r, r}, .names = {"grouped " + rs, "lane " + rs}});
    } else {
      sequences.push_back({.radices = {r, 8}, .names = {"grouped " + rs, "lane radix-8"}});
      sequences.push_back({.radices = {8, r}, .names = {"grouped radix-8", "lane " + rs}});
    }
    for (const sequence& seq : sequences) {
      std::size_t n = 1;
      for (const std::size_t f : seq.radices) {
        n *= f;
      }
      build_settings settings;
      settings.radices = seq.radices;
      check_row<T>(target, n, fftn::direction::forward, settings, 1.0, false, false, false,
                   seq.names);
      std::vector<std::string> backward = seq.names;
      for (std::string& name : backward) {
        if (const auto at = name.find("forward"); at != std::string::npos) {
          name.replace(at, 7, "backward");
        }
      }
      check_row<T>(target, n, fftn::direction::backward, settings, 1.0, false, false, false,
                   backward);
    }
  }
}

TEST(Passes, EveryCodeletInEveryPosition) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    every_codelet_in_every_position<double>(t);
    every_codelet_in_every_position<float>(t);
  }
}

template <class T> void generic_in_every_position(fftn::isa target) {
  const std::vector<std::vector<std::size_t>> sequences{
      {29}, {29, 2}, {2, 29}, {2, 29, 3}, {3, 31, 2}, {18}, {2, 18},
  };
  for (const auto& radices : sequences) {
    std::size_t n = 1;
    for (const std::size_t f : radices) {
      n *= f;
    }
    build_settings settings;
    settings.radices = radices;
    settings.bluestein_above = no_bluestein;
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_row<T>(target, n, dir, settings, 1.0, false, false, false, {"generic radix-"});
    }
  }
}

TEST(Passes, GenericInEveryPosition) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    generic_in_every_position<double>(t);
    generic_in_every_position<float>(t);
  }
}

// Leaves above 16 (experiment X8): with the leaf limit raised, every codelet radix above 16
// is one leaf, with its edges, and an even real line of twice that length runs its
// half-length chain as one leaf; with the default limit 32 and 64 are chains, while the
// primes 17, 19 and 23 are leaves whatever the limit (experiment X27).
template <class T> void leaves_above_sixteen(fftn::isa target) {
  for (const std::uint8_t radix : fftn::detail::gen::radices) {
    const std::size_t n = radix;
    if (n <= 16) {
      continue;
    }
    const std::string leaf = "leaf radix-" + std::to_string(n);
    build_settings settings;
    settings.leaf_limit = 64;
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_row<T>(target, n, dir, settings, 1.0, false, false, false, {leaf});
      check_row<T>(target, n, dir, settings, 0.375, true, false, false,
                   {leaf, "in: factor", "out: scale"});
      check_row<T>(target, n, dir, settings, 0.375, false, true, true,
                   {leaf, "in: scale", "out: factor"});
      check_row<T>(target, n, dir, settings, 1.0, false, false, false, {leaf}, 3, 2);
    }
    // By default (X8): a leaf of 32 on NEON in double, chains elsewhere; the primes (X27) a
    // leaf everywhere.
    const bool leaf_by_default =
        n % 2 == 1 || (n == 32 && std::is_same_v<T, double> &&
                       (target == fftn::isa::neon || target == fftn::isa::neon_fcma));
    const auto plain =
        make_plan_core(fftn::detail::view_of(fftn::rows_c2c<T>(n, 1, fftn::direction::forward)),
                       {.force_isa = target}, {});
    // (A pair leaf, experiment X28b, is no leaf of n: "pair leaf radix-8 x 4".)
    EXPECT_EQ(joined(stage_names(*plain)).find(leaf + " ") != std::string::npos, leaf_by_default)
        << joined(stage_names(*plain));
    const auto real = make_plan_core(fftn::detail::view_of(fftn::rows_r2c<T>(2 * n, 1)),
                                     {.force_isa = target}, settings);
    EXPECT_NE(joined(stage_names(*real)).find(leaf), std::string::npos)
        << joined(stage_names(*real));
  }
}

TEST(Passes, LeavesAboveSixteen) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    leaves_above_sixteen<double>(t);
    leaves_above_sixteen<float>(t);
  }
}

template <class T> std::size_t complex_lanes(fftn::isa t) {
  return fftn::detail::of<T>(*fftn::detail::find_kernels(
                                 static_cast<fftn::detail::target>(static_cast<unsigned char>(t))))
      .complex_lanes;
}

// Whether the rules give n a Rader stage (experiment X27) on target t in precision T: its
// one prime factor without a codelet takes Rader under the target's profile.
template <class T> bool takes_rader(std::size_t n, fftn::isa t) {
  std::size_t big = 1;
  std::size_t count = 0;
  for (std::size_t p = 2; p <= n; ++p) {
    while (n % p == 0) {
      if (p > 16 && p != 17 && p != 19 && p != 23) {
        big = p;
        ++count;
      }
      n /= p;
    }
  }
  const auto profile = fftn::detail::profile_of(
      static_cast<fftn::detail::target>(static_cast<unsigned char>(t)), complex_lanes<T>(t));
  return count == 1 && fftn::detail::rader_root(big, profile) != 0;
}

// The Bluestein stage (strategies/chain.hpp) as the only pass, after codelet passes with copy
// counts that leave tails on every SIMD target (2, 3, 30) and that do not (4, 16), after a
// generic pass, and for a product of large primes; its convolution's first pass is grouped
// with one copy and a lane pass with several. Then every n up to 160 with a prime factor
// above 16, by the rules.
template <class T> void bluestein_in_every_position(fftn::isa target) {
  struct line {
    std::vector<std::size_t> radices;
    std::size_t above; // the Bluestein threshold
    std::vector<std::string> names;
  };
  const std::vector<std::string> both{"bluestein in", "bluestein out"};
  const std::vector<line> lines{
      {.radices = {47}, .above = 16, .names = {"copies 1", "convolution: grouped"}},
      {
          .radices = {2, 47},
          .above = 16,
          .names = {"grouped radix-2", "copies 2", "convolution: lane"},
      },
      {.radices = {3, 59}, .above = 16, .names = {"grouped radix-3", "copies 3"}},
      {.radices = {4, 47}, .above = 16, .names = {"copies 4"}},
      {.radices = {5, 6, 107}, .above = 16, .names = {"lane radix-6", "copies 30"}},
      {.radices = {16, 47}, .above = 16, .names = {"copies 16"}},
      {.radices = {47, 59}, .above = 16, .names = {"Q 2773, ", "copies 1"}},
      {.radices = {47, 59}, .above = 47, .names = {"generic radix-47", "Q 59, ", "copies 47"}},
      // A codelet's prime is never Bluestein's, whatever the threshold (experiment X27).
      {.radices = {17, 59}, .above = 16, .names = {"grouped radix-17", "Q 59, ", "copies 17"}},
      {.radices = {2, 3, 83}, .above = 16, .names = {"Q 83, ", "copies 6"}},
      // A product of primes is one Bluestein stage even where each would take Rader.
      {.radices = {29, 31}, .above = 16, .names = {"Q 899, ", "copies 1"}},
  };
  for (const line& l : lines) {
    std::size_t n = 1;
    for (const std::size_t f : l.radices) {
      n *= f;
    }
    build_settings settings;
    settings.radices = l.radices;
    settings.bluestein_above = l.above;
    std::vector<std::string> names = both;
    names.insert(names.end(), l.names.begin(), l.names.end());
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_row<T>(target, n, dir, settings, 1.0, false, false, false, names);
    }
  }
  for (std::size_t n = 17; n <= 160; ++n) {
    if (fftn::test::has_large_prime_factor(n)) {
      const std::vector<std::string> stage =
          takes_rader<T>(n, target) ? std::vector<std::string>{"rader in", "rader out"} : both;
      for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
        check_row<T>(target, n, dir, {}, 1.0, false, false, false, stage);
      }
    }
  }
}

// The Rader stage (experiment X27; strategies/chain.hpp) as the only pass and after codelet
// passes whose copy counts C divides (2, 4, 16: the vector form) and does not (1, 3, 30:
// one copy at a time), its convolution's first pass grouped with one copy and a lane pass
// with several, and as a Bluestein-threshold stage after a generic pass. The primes (151,
// 211, 193) lie in every profile's Rader range.
template <class T> void rader_in_every_position(fftn::isa target) {
  struct line {
    std::vector<std::size_t> radices;
    std::size_t above;
    std::vector<std::string> names;
  };
  const std::vector<std::string> both{"rader in", "rader out"};
  const std::vector<line> lines{
      {.radices = {151}, .above = 16, .names = {"copies 1", "convolution: grouped"}},
      {.radices = {2, 151},
       .above = 16,
       .names = {"grouped radix-2", "copies 2", "convolution: lane"}},
      {.radices = {3, 211}, .above = 16, .names = {"grouped radix-3", "root 2, copies 3"}},
      {.radices = {4, 151}, .above = 16, .names = {"copies 4"}},
      {.radices = {5, 6, 151}, .above = 16, .names = {"lane radix-6", "copies 30"}},
      {.radices = {16, 151}, .above = 16, .names = {"copies 16"}},
      {.radices = {8, 193}, .above = 16, .names = {"Q 193, L 192, root 5, copies 8"}},
      {.radices = {29, 151},
       .above = 29,
       .names = {"generic radix-29", "Q 151, L 150", "copies 29"}},
  };
  for (const line& l : lines) {
    std::size_t n = 1;
    for (const std::size_t f : l.radices) {
      n *= f;
    }
    build_settings settings;
    settings.radices = l.radices;
    settings.bluestein_above = l.above;
    std::vector<std::string> names = both;
    names.insert(names.end(), l.names.begin(), l.names.end());
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_row<T>(target, n, dir, settings, 1.0, false, false, false, names);
    }
  }
}

TEST(Passes, RaderInEveryPosition) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    rader_in_every_position<double>(t);
    rader_in_every_position<float>(t);
  }
}

TEST(Passes, BluesteinInEveryPosition) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    bluestein_in_every_position<double>(t);
    bluestein_in_every_position<float>(t);
  }
}

// Every edge operation on each edge of every pass kind that has one: the copy (n = 1), the
// leaf, the grouped first pass (input) and the last lane pass (output), the generic pass
// as the only pass (both edges), first (input) and last (output).
template <class T> void every_edge_variant(fftn::isa target) {
  struct line {
    std::size_t n;
    std::vector<std::size_t> radices;
    std::string first;
    std::string last;
    std::size_t above = 0; // the Bluestein threshold, 0 for the profile's
  };
  const std::vector<line> lines{
      {.n = 1, .radices = {}, .first = "copy", .last = "copy"},
      {.n = 12, .radices = {}, .first = "leaf radix-12", .last = "leaf radix-12"},
      {.n = 40, .radices = {5, 8}, .first = "grouped radix-5", .last = "lane radix-8"},
      {
          .n = 29,
          .radices = {},
          .first = "generic radix-29",
          .last = "generic radix-29",
          .above = no_bluestein,
      },
      {
          .n = 58,
          .radices = {2, 29},
          .first = "grouped radix-2",
          .last = "generic radix-29",
          .above = no_bluestein,
      },
      {
          .n = 58,
          .radices = {29, 2},
          .first = "generic radix-29",
          .last = "lane radix-2",
          .above = no_bluestein,
      },
      {.n = 47, .radices = {}, .first = "bluestein in,", .last = "bluestein out", .above = 16},
      {.n = 151, .radices = {}, .first = "rader in,", .last = "rader out", .above = 16},
      {.n = 302, .radices = {2, 151}, .first = "grouped radix-2", .last = "rader out", .above = 16},
      {.n = 1208,
       .radices = {8, 151},
       .first = "grouped radix-8",
       .last = "rader out",
       .above = 16},
      {
          .n = 94,
          .radices = {2, 47},
          .first = "grouped radix-2",
          .last = "bluestein out",
          .above = 16,
      },
  };
  struct variant {
    bool in_factor;
    bool out_factor;
    bool scaled;
    bool scale_on_input;
    const char* in;
    const char* out;
  };
  // Every combination but a scale on both edges, which one scale cannot be.
  const std::vector<variant> variants{
      {false, false, false, false, nullptr, nullptr},
      {false, false, true, false, nullptr, "out: scale"},
      {false, true, false, false, nullptr, "out: factor"},
      {false, false, true, true, "in: scale", nullptr},
      {false, true, true, false, "in: scale", "out: factor"},
      {true, false, false, false, "in: factor", nullptr},
      {true, false, true, false, "in: factor", "out: scale"},
      {true, true, false, false, "in: factor", "out: factor"},
  };
  for (const line& l : lines) {
    for (const variant& v : variants) {
      build_settings settings;
      settings.radices = l.radices;
      settings.bluestein_above = l.above;
      std::vector<std::string> names;
      if (v.in != nullptr) {
        names.push_back(l.first + " ");
        names.emplace_back(v.in);
      }
      if (v.out != nullptr) {
        names.push_back(l.last);
        names.emplace_back(v.out);
      }
      for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
        check_row<T>(target, l.n, dir, settings, v.scaled ? 0.375 : 1.0, v.in_factor, v.out_factor,
                     v.scale_on_input, names);
      }
    }
  }
}

TEST(Passes, EveryEdgeVariant) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    every_edge_variant<double>(t);
    every_edge_variant<float>(t);
  }
}

// The copy count from which target t's lane passes may be split, in precision T.
template <class T> std::size_t split_block_of(fftn::isa t) {
  return fftn::detail::of<T>(*fftn::detail::find_kernels(
                                 static_cast<fftn::detail::target>(static_cast<unsigned char>(t))))
      .split_block;
}

// Split passes (DESIGN.md 6.3, src/passes/split.inl), forced: every split codelet radix as
// the first pass of a split run (interleaved to BL), a middle one (BL to BL) and the last
// (BL to interleaved), in both directions, against the reference; forward with both factor
// edges (the input's on the interleaved first pass, the output's on the last split pass),
// backward with the output scale. The first radix is the target's split_block (at least
// 2), so the run starts at the second pass.
template <class T> void split_in_every_position(fftn::isa target) {
  const std::size_t f = std::max<std::size_t>(2, split_block_of<T>(target));
  for (const std::uint8_t radix : fftn::detail::gen::split_radices) {
    const std::size_t r = radix;
    const std::string rs = "split lane radix-" + std::to_string(r);
    struct sequence {
      std::vector<std::size_t> radices;
      std::string name;
    };
    const std::vector<sequence> sequences{
        {
            .radices = {f, r, 2},
            .name = rs + " forward (m 2, s " + std::to_string(f) + "; interleaved to BL",
        },
        {
            .radices = {f, 2, r},
            .name = rs + " forward (m 1, s " + std::to_string(2 * f) + "; BL to interleaved",
        },
        {
            .radices = {f, 2, r, 2},
            .name = rs + " forward (m 2, s " + std::to_string(2 * f) + "; BL)",
        },
    };
    for (const sequence& seq : sequences) {
      std::size_t n = 1;
      for (const std::size_t x : seq.radices) {
        n *= x;
      }
      build_settings settings;
      settings.radices = seq.radices;
      settings.layout = fftn::detail::layout_choice::split;
      check_row<T>(target, n, fftn::direction::forward, settings, 1.0, true, true, false,
                   {seq.name, "grouped radix-" + std::to_string(f), "out: factor"});
      std::string backward = seq.name;
      backward.replace(backward.find("forward"), 7, "backward");
      check_row<T>(target, n, fftn::direction::backward, settings, 0.375, false, false, false,
                   {backward, "out: scale"});
    }
  }
}

TEST(Passes, SplitInEveryPosition) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    split_in_every_position<double>(t);
    split_in_every_position<float>(t);
  }
}

// Where a chain has no split run, a forced split layout leaves it interleaved: two passes;
// a run of one pass (the copy count reaches split_block only before the last pass); a radix
// with no split codelet in the run; a Bluestein stage at the end. And real rows and strided
// rows, whose complex chains run on scratch, split where they fit, against the reference.
template <class T> void split_only_where_it_fits(fftn::isa target) {
  const std::size_t f = std::max<std::size_t>(2, split_block_of<T>(target));
  const auto has_split = [&](std::vector<std::size_t> radices, std::size_t above) {
    std::size_t n = 1;
    for (const std::size_t x : radices) {
      n *= x;
    }
    build_settings settings;
    settings.radices = std::move(radices);
    settings.bluestein_above = above;
    settings.layout = fftn::detail::layout_choice::split;
    const auto core =
        make_plan_core(fftn::detail::view_of(fftn::rows_c2c<T>(n, 1, fftn::direction::forward)),
                       {.force_isa = target}, settings);
    return joined(stage_names(*core)).find("split") != std::string::npos;
  };
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float"));
  EXPECT_TRUE(has_split({f, 2, 2}, 0));
  EXPECT_FALSE(has_split({f, 2}, 0)) << "two passes";
  if (f > 2) {
    EXPECT_FALSE(has_split({2, f / 2, 2}, 0)) << "a run of one pass";
  }
  EXPECT_FALSE(has_split({f, 2, 29}, no_bluestein)) << "a generic pass in the run";
  EXPECT_FALSE(has_split({f, 2, 2, 47}, 16)) << "a Bluestein stage";
  EXPECT_FALSE(has_split({f, 2, 2, 29}, 16)) << "a Rader stage";
  EXPECT_FALSE(has_split({f, 2, 17}, 0)) << "radix 17 has no split codelet";
  EXPECT_FALSE(has_split({f, 32, 2}, 0)) << "radix 32 has no split codelet";

  // r2c and c2r of 2 f x 4 x 2 points with the pairing a pass of its own (a pairing fused
  // into the chain's edge pass, task 2.1, takes the place of the split run's end): complex
  // lines of f x 4 x 2; a strided c2c row (copies in and out of scratch on wide targets);
  // in place.
  build_settings settings;
  settings.layout = fftn::detail::layout_choice::split;
  settings.fuse_real_radix = 1;
  settings.radices = {f, 4, 2};
  const std::size_t h = 8 * f;
  const std::size_t n = 2 * h;
  // Wide targets copy a strided row out of scratch, so the split run writes scratch; with
  // C = 1 nothing is copied, and the last split pass's runs of W elements need stride 1
  // where W > 1 (NEON double), so that row stays interleaved.
  const auto& kernels = fftn::detail::of<T>(*fftn::detail::find_kernels(
      static_cast<fftn::detail::target>(static_cast<unsigned char>(target))));
  std::vector<std::string> strided_names{"out: scale"};
  if (kernels.complex_lanes > 1 || kernels.real_lanes == 1) {
    strided_names.emplace_back("split lane");
  }
  check_row<T>(target, h, fftn::direction::forward, settings, 0.375, false, false, false,
               strided_names, 2, 3);
  using c = std::complex<T>;
  fftn::test::input_source src(0x51A7ULL + n);
  const std::vector<T> xr = src.reals<T>(n);
  const auto r2c = make_plan_core(fftn::detail::view_of(fftn::rows_r2c<T>(n, 1)),
                                  {.force_isa = target}, settings);
  EXPECT_NE(joined(stage_names(*r2c)).find("split lane"), std::string::npos);
  const cvec<T> spectrum = run<T, c>(*r2c, xr, n / 2 + 1);
  const auto c2r =
      make_plan_core(fftn::detail::view_of(fftn::rows_c2r<T>(n, 1)),
                     {.scale = 1.0 / static_cast<double>(n), .force_isa = target}, settings);
  EXPECT_NE(joined(stage_names(*c2r)).find("split lane"), std::string::npos);
  const std::vector<T> back = run<c, T>(*c2r, spectrum, n);
  const auto tol = fftn::test::derived_tolerance<T>(n, fftn::test::stable_fft_bound);
  double err = 0;
  double ref = 0;
  for (std::size_t j = 0; j < n; ++j) {
    err = std::max(err, std::abs(static_cast<double>(back[j]) - static_cast<double>(xr[j])));
    ref = std::max(ref, std::abs(static_cast<double>(xr[j])));
  }
  EXPECT_LE(err / ref, 4 * tol.max) << "r2c then c2r is not the identity";
  // c2r with the pairing fused into its first pass: the rest of the chain takes the same
  // split run as the separate route's complex chain, and the same bits (avx2 and avx512
  // choose the two routes differently for some lengths).
  build_settings fused = settings;
  fused.fuse_real_radix = 16;
  const auto c2r_fused =
      make_plan_core(fftn::detail::view_of(fftn::rows_c2r<T>(n, 1)),
                     {.scale = 1.0 / static_cast<double>(n), .force_isa = target}, fused);
  const std::string fused_names = joined(stage_names(*c2r_fused));
  EXPECT_NE(fused_names.find("with the c2r pairing"), std::string::npos) << fused_names;
  EXPECT_NE(fused_names.find("split lane"), std::string::npos) << fused_names;
  const std::vector<T> back_fused = run<c, T>(*c2r_fused, spectrum, n);
  EXPECT_EQ(back_fused, back) << "the fused and separate c2r routes differ";
  // In place: the same passes as out of place, placed differently.
  const cvec<T> x = src.complexes<T>(h);
  const auto shape = fftn::rows_c2c<T>(h, 1, fftn::direction::backward);
  const auto view = fftn::detail::view_of(shape);
  const auto in_place = make_plan_core(view, {.in_place = true, .force_isa = target}, settings);
  EXPECT_NE(joined(stage_names(*in_place)).find("split lane"), std::string::npos);
  const cvec<T> y_in = run_in_place(*in_place, x);
  const cvec<T> y_out = run<c, c>(*make_plan_core(view, {.force_isa = target}, settings), x, h);
  for (std::size_t k = 0; k < h; ++k) {
    EXPECT_EQ(y_in[k], y_out[k]) << "in place and out of place differ at " << k;
  }
}

TEST(Passes, SplitOnlyWhereItFits) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    split_only_where_it_fits<double>(t);
    split_only_where_it_fits<float>(t);
  }
}

// An r2c row whose pairing would fuse into the last pass (fuse_real_radix 8) and one whose
// pairing is a pass of its own (1), split forced: a fused pairing cannot end a split run, so
// where the half-length chain has one the row takes the separate route, and the two give the
// same bits (avx512 fuses at radix 8 where avx2 does not). With interleaved passes forced
// the pairing still fuses.
template <class T> void r2c_routes_agree_under_split(fftn::isa target) {
  using c = std::complex<T>;
  const std::size_t f = std::max<std::size_t>(8, split_block_of<T>(target));
  constexpr std::size_t rows = 3;
  for (const std::vector<std::size_t>& radices :
       {std::vector<std::size_t>{f, 8, 8}, std::vector<std::size_t>{f, 8, 8, 8}}) {
    std::size_t n = 2;
    for (const std::size_t x : radices) {
      n *= x;
    }
    SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
                 ", n = " + std::to_string(n));
    fftn::test::input_source src(0xF05EULL + n);
    const std::vector<T> xr = src.reals<T>(rows * n);
    const auto shape = fftn::rows_r2c<T>(n, rows);
    const auto view = fftn::detail::view_of(shape);
    const auto plan = [&](fftn::detail::layout_choice layout, std::size_t fuse) {
      build_settings settings;
      settings.layout = layout;
      settings.radices = radices;
      settings.fuse_real_radix = fuse;
      return make_plan_core(view, {.force_isa = target}, settings);
    };
    const auto fused = plan(fftn::detail::layout_choice::split, 8);
    const auto separate = plan(fftn::detail::layout_choice::split, 1);
    const std::string names = joined(stage_names(*fused));
    EXPECT_NE(names.find("split lane"), std::string::npos) << names;
    EXPECT_EQ(names, joined(stage_names(*separate)));
    const cvec<T> y_fused = run<T, c>(*fused, xr, rows * (n / 2 + 1));
    const cvec<T> y_separate = run<T, c>(*separate, xr, rows * (n / 2 + 1));
    EXPECT_EQ(y_fused, y_separate) << "the fused and separate r2c routes differ";
    const std::string interleaved =
        joined(stage_names(*plan(fftn::detail::layout_choice::interleaved, 8)));
    EXPECT_NE(interleaved.find("with the r2c pairing"), std::string::npos) << interleaved;
  }
}

TEST(Passes, R2cRoutesAgreeUnderSplit) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    r2c_routes_agree_under_split<double>(t);
    r2c_routes_agree_under_split<float>(t);
  }
}

// Internal settings that ask for impossible edges are refused.
TEST(Passes, ImpossibleEdgesAreRefused) {
  const cvec<double> f(8);
  build_settings both;
  both.in_factor = f.data();
  both.scale_on_input = true;
  EXPECT_THROW(static_cast<void>(make_plan_core(
                   fftn::detail::view_of(fftn::rows_c2c<double>(8, 1, fftn::direction::forward)),
                   {.scale = 2.0}, both)),
               std::invalid_argument);
  build_settings real;
  real.out_factor = f.data();
  EXPECT_THROW(static_cast<void>(
                   make_plan_core(fftn::detail::view_of(fftn::rows_r2c<double>(8, 1)), {}, real)),
               std::invalid_argument);
}

// The complex numbers per register of target t in precision T.
template <class T> std::size_t lanes_of(fftn::isa t) {
  return fftn::detail::of<T>(*fftn::detail::find_kernels(
                                 static_cast<fftn::detail::target>(static_cast<unsigned char>(t))))
      .complex_lanes;
}

// Rows read and written with strides: on targets with C > 1 the chain copies the row into
// scratch and out of it, and the edge operations stay on the first and last passes that
// are not copies (with C = 1 there is no copy).
template <class T> void strided_rows_with_edges(fftn::isa target) {
  const bool wide = lanes_of<T>(target) > 1;
  struct line {
    std::size_t n;
    std::size_t above; // the Bluestein threshold
    std::string first;
    std::string last;
  };
  for (const line& l : {
           line{.n = 40, .above = 0, .first = "grouped", .last = "lane"},
           line{.n = 29, .above = no_bluestein, .first = "generic", .last = "generic"},
           line{.n = 47, .above = 16, .first = "bluestein in", .last = "bluestein out"},
           line{.n = 141, .above = 16, .first = "grouped", .last = "bluestein out"},
           line{.n = 151, .above = 16, .first = "rader in", .last = "rader out"},
           line{.n = 604, .above = 16, .first = "grouped", .last = "rader out"},
           line{.n = 51, .above = 16, .first = "grouped", .last = "lane radix-17"},
       }) {
    build_settings settings;
    settings.bluestein_above = l.above;
    std::vector<std::string> names{l.first, "in: factor", l.last, "out: scale"};
    if (wide) {
      names.emplace_back("copy");
    }
    check_row<T>(target, l.n, fftn::direction::forward, settings, 0.375, true, false, false, names,
                 2, 3);
    names[1] = "in: scale";
    names[3] = "out: factor";
    check_row<T>(target, l.n, fftn::direction::backward, settings, 0.375, false, true, false, names,
                 3, 2);
  }
}

TEST(Passes, StridedRowsCopyAroundTheEdges) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    strided_rows_with_edges<double>(t);
    strided_rows_with_edges<float>(t);
  }
}

// A single generic pass between the caller's arrays uses no scratch; in place it writes
// scratch and is copied back. A Bluestein stage needs two scratch lines of its convolution
// length M and runs in place without a copy.
TEST(Passes, ScratchOnlyWhenUsed) {
  build_settings generic;
  generic.bluestein_above = no_bluestein;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    // The view borrows the shape's axes, so the shape must outlive it.
    const auto shape = fftn::rows_c2c<double>(47, 1, fftn::direction::forward);
    const auto rows = fftn::detail::view_of(shape);
    EXPECT_EQ(fftn::detail::workspace_bytes(*make_plan_core(rows, {.force_isa = t}, generic), 1),
              0U)
        << fftn::isa_name(t);
    EXPECT_GT(fftn::detail::workspace_bytes(
                  *make_plan_core(rows, {.in_place = true, .force_isa = t}, generic), 1),
              0U)
        << fftn::isa_name(t);
    const auto bluestein = make_plan_core(rows, {.in_place = true, .force_isa = t}, {});
    const std::string listing = joined(stage_names(*bluestein));
    EXPECT_NE(listing.find("bluestein out"), std::string::npos) << listing;
    EXPECT_EQ(listing.find("copy"), std::string::npos) << listing;
    const std::size_t m = fftn::detail::bluestein_length(
        47, fftn::detail::profile_of(static_cast<fftn::detail::target>(t), lanes_of<double>(t)));
    EXPECT_EQ(bluestein->scratch_line, m) << fftn::isa_name(t);
    // A Rader stage holds its convolution of L = 150 and two more rows (x_0, the sum).
    const auto shape151 = fftn::rows_c2c<double>(151, 1, fftn::direction::forward);
    const auto rader =
        make_plan_core(fftn::detail::view_of(shape151), {.in_place = true, .force_isa = t}, {});
    const std::string rader_listing = joined(stage_names(*rader));
    EXPECT_NE(rader_listing.find("rader out"), std::string::npos) << rader_listing;
    EXPECT_EQ(rader_listing.find("copy"), std::string::npos) << rader_listing;
    EXPECT_EQ(rader->scratch_line, 152U) << fftn::isa_name(t);
  }
}

// The radix test knob refuses radices no pass has: radix 1 once selected a null entry.
TEST(Passes, ForcedRadicesAreChecked) {
  // The view borrows the shape's axes, so the shape must outlive it.
  const auto shape = fftn::rows_c2c<double>(8, 1, fftn::direction::forward);
  const auto rows = fftn::detail::view_of(shape);
  build_settings one;
  one.radices = {1, 8};
  EXPECT_THROW(static_cast<void>(make_plan_core(rows, {}, one)), std::invalid_argument);
  build_settings zero;
  zero.radices = {0};
  EXPECT_THROW(static_cast<void>(make_plan_core(rows, {}, zero)), std::invalid_argument);
}

// ---- A fused scale costs no extra pass (task 1.2's acceptance).

std::size_t stage_count(const plan_core& core) { return stage_names(core).size(); }

// The plan with the scale has exactly the stages of the plan without, and its outputs are
// the unscaled outputs times the scale rounded to T, bit for bit: the scale multiplies the
// values the last pass stores, which is what a separate scaling pass would do.
template <class In, class Out, class T>
void check_scaled(fftn::isa target, const fftn::shape<In, Out>& s, std::size_t in_count,
                  std::size_t out_count, const std::string& what, bool in_place = false) {
  SCOPED_TRACE(what + " on " + fftn::isa_name(target) + (in_place ? ", in place" : ""));
  const double scale = 1.0 / 3.0;
  const auto view = fftn::detail::view_of(s);
  const auto plain = make_plan_core(view, {.in_place = in_place, .force_isa = target}, {});
  const auto scaled =
      make_plan_core(view, {.in_place = in_place, .scale = scale, .force_isa = target}, {});
  EXPECT_EQ(stage_count(*plain), stage_count(*scaled))
      << "unscaled:" << joined(stage_names(*plain)) << "\nscaled:" << joined(stage_names(*scaled));
  EXPECT_EQ(plain->bytes, scaled->bytes);
  fftn::test::input_source src(0x5CA1EULL + in_count);
  std::vector<In> x(in_count);
  if constexpr (std::is_same_v<In, T>) {
    x = src.reals<T>(in_count);
  } else {
    x = src.complexes<T>(in_count);
  }
  std::vector<Out> y0;
  std::vector<Out> y1;
  if constexpr (std::is_same_v<In, Out>) {
    y0 = in_place ? run_in_place(*plain, x) : run<In, Out>(*plain, x, out_count);
    y1 = in_place ? run_in_place(*scaled, x) : run<In, Out>(*scaled, x, out_count);
  } else {
    y0 = run<In, Out>(*plain, x, out_count);
    y1 = run<In, Out>(*scaled, x, out_count);
  }
  const auto k = static_cast<T>(scale);
  std::size_t differ = 0;
  for (std::size_t i = 0; i < out_count; ++i) {
    if (y1[i] != y0[i] * k) {
      ++differ;
    }
  }
  EXPECT_EQ(differ, 0U) << "of " << out_count << " outputs";
}

template <class T> void fused_scale(fftn::isa target) {
  using c = std::complex<T>;
  std::vector<std::size_t> lengths;
  for (std::size_t n = 1; n <= 64; ++n) {
    lengths.push_back(n);
  }
  lengths.insert(lengths.end(), {97, 128, 360, 1000, 1009, 1024, 4096});
  for (const std::size_t n : lengths) {
    const std::string at = "n = " + std::to_string(n);
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_scaled<c, c, T>(target, fftn::rows_c2c<T>(n, 2, dir), 2 * n, 2 * n, "c2c " + at);
    }
    const std::size_t h = n / 2 + 1;
    check_scaled<T, c, T>(target, fftn::rows_r2c<T>(n, 2), 2 * n, 2 * h, "r2c " + at);
    check_scaled<c, T, T>(target, fftn::rows_c2r<T>(n, 2), 2 * h, 2 * n, "c2r " + at);
  }
  check_scaled<c, c, T>(target, fftn::planes_c2c<T>(12, 40, 2, fftn::direction::forward), 960, 960,
                        "planes 12 x 40");
  // The last axis is strided: on C > 1 its chain ends in a copy out of scratch, after the
  // pass that scales.
  check_scaled<c, c, T>(target, fftn::planes_c2c<T>(40, 12, 2, fftn::direction::forward), 960, 960,
                        "planes 40 x 12");
  check_scaled<c, c, T>(target, fftn::planes_c2c<T>(17, 12, 2, fftn::direction::backward), 408, 408,
                        "planes 17 x 12");
  check_scaled<c, c, T>(target, fftn::planes_c2c<T>(29, 12, 2, fftn::direction::backward), 696, 696,
                        "planes 29 x 12");
  // In place, a prime row without a codelet is one Bluestein or Rader stage, which runs in
  // place; a prime with one is a leaf.
  for (const std::size_t n :
       {std::size_t{17}, std::size_t{151}, std::size_t{47}, std::size_t{1009}}) {
    check_scaled<c, c, T>(target, fftn::rows_c2c<T>(n, 2, fftn::direction::forward), 2 * n, 2 * n,
                          "c2c n = " + std::to_string(n), true);
  }
  fftn::shape<T, c> real_plane;
  real_plane.axes = {{.n = 6}, {.n = 10}};
  check_scaled<T, c, T>(target, real_plane, 60, 36, "r2c plane 6 x 10");
  fftn::shape<c, T> volume;
  volume.axes = {{.n = 3}, {.n = 4}, {.n = 6}};
  check_scaled<c, T, T>(target, volume, 48, 72, "c2r volume 3 x 4 x 6");
}

TEST(Passes, FusedScaleCostsNoExtraPass) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    fused_scale<double>(t);
    fused_scale<float>(t);
  }
}

// Every bit of every value equal (so -0 differs from +0).
template <class T> auto bits_of(T x) {
  return std::bit_cast<std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>>(x);
}
template <class T> bool bitwise_equal(const std::vector<T>& x, const std::vector<T>& y) {
  return std::ranges::equal(x, y, [](T a, T b) { return bits_of(a) == bits_of(b); });
}
template <class T>
bool bitwise_equal(const std::vector<std::complex<T>>& x, const std::vector<std::complex<T>>& y) {
  return std::ranges::equal(x, y, [](std::complex<T> a, std::complex<T> b) {
    return bits_of(a.real()) == bits_of(b.real()) && bits_of(a.imag()) == bits_of(b.imag());
  });
}

// The same passes on two targets whose operations round alike (DESIGN.md 6.2): every
// output bit equal, for c2c rows with a scale and the factor edges (cmul), strided (the
// copy passes) and in place, and for r2c and c2r rows, several rows each so that tails
// occur. The radices are the first target's
// rules' choice, forced on both: avx2 and avx512 key their rules on different register
// counts and lanes. With interleaved and with split lane passes forced (a pair shares
// split_block, so it chooses the same split runs).
template <class T> void same_bits(fftn::isa a, fftn::isa b, fftn::detail::layout_choice layout) {
  using c = std::complex<T>;
  const auto profile =
      fftn::detail::profile_of(static_cast<fftn::detail::target>(a), lanes_of<T>(a));
  // A Bluestein stage's convolution length and radices come from the profile, so where the
  // two profiles differ (avx2 and avx512: C differs) the large primes are generic passes.
  const auto profile_b =
      fftn::detail::profile_of(static_cast<fftn::detail::target>(b), lanes_of<T>(b));
  const std::size_t above =
      profile.lanes == profile_b.lanes && profile.costs == profile_b.costs ? 0 : no_bluestein;
  std::vector<std::size_t> lengths;
  for (std::size_t n = 1; n <= 64; ++n) {
    lengths.push_back(n);
  }
  lengths.insert(lengths.end(), {97, 100, 128, 360, 1000, 1024, 4096});
  constexpr std::size_t rows = 3;
  for (const std::size_t n : lengths) {
    SCOPED_TRACE(std::string(fftn::isa_name(a)) + " and " + fftn::isa_name(b) +
                 (sizeof(T) == 8 ? " double" : " float") + ", n = " + std::to_string(n));
    fftn::test::input_source src(0x5A3EULL + n);
    const cvec<T> x = src.complexes<T>(rows * n);
    const cvec<T> fin = src.complexes<T>(n);
    const cvec<T> fout = src.complexes<T>(n);
    // Forward with both factor edges; backward with the input factor and a scale (a plan
    // takes one scale, and not beside a factor on the same edge).
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      const bool forward = dir == fftn::direction::forward;
      build_settings settings;
      settings.layout = layout;
      settings.radices = fftn::detail::rule_radices(n, profile);
      settings.bluestein_above = above;
      settings.in_factor = fin.data();
      settings.out_factor = forward ? fout.data() : nullptr;
      const double scale = forward ? 1.0 : 0.25;
      const auto shape = fftn::rows_c2c<T>(n, rows, dir); // the view refers to it
      const auto view = fftn::detail::view_of(shape);
      const auto on_a = make_plan_core(view, {.scale = scale, .force_isa = a}, settings);
      const auto on_b = make_plan_core(view, {.scale = scale, .force_isa = b}, settings);
      EXPECT_EQ(stage_names(*on_a).size(), stage_names(*on_b).size());
      EXPECT_TRUE(bitwise_equal(run<c, c>(*on_a, x, rows * n), run<c, c>(*on_b, x, rows * n)))
          << (forward ? "c2c forward" : "c2c backward");
    }
    // Strided rows (input stride 2, output stride 3: the copy passes into and out of
    // scratch on targets with C > 1), and in place.
    {
      build_settings settings;
      settings.layout = layout;
      settings.radices = fftn::detail::rule_radices(n, profile);
      settings.bluestein_above = above;
      auto strided = fftn::rows_c2c<T>(n, rows, fftn::direction::forward);
      strided.axes[0].in_stride = 2;
      strided.axes[0].out_stride = 3;
      strided.batches[0].in_dist = static_cast<std::ptrdiff_t>(2 * n);
      strided.batches[0].out_dist = static_cast<std::ptrdiff_t>(3 * n);
      const auto view = fftn::detail::view_of(strided);
      const cvec<T> xs = src.complexes<T>(2 * rows * n);
      EXPECT_TRUE(bitwise_equal(
          run<c, c>(*make_plan_core(view, {.force_isa = a}, settings), xs, 3 * rows * n),
          run<c, c>(*make_plan_core(view, {.force_isa = b}, settings), xs, 3 * rows * n)))
          << "c2c strided";
      const auto tight = fftn::rows_c2c<T>(n, rows, fftn::direction::backward);
      const auto tight_view = fftn::detail::view_of(tight);
      EXPECT_TRUE(bitwise_equal(
          run_in_place(*make_plan_core(tight_view, {.in_place = true, .force_isa = a}, settings),
                       x),
          run_in_place(*make_plan_core(tight_view, {.in_place = true, .force_isa = b}, settings),
                       x)))
          << "c2c in place";
    }
    // Real rows: the complex line is n / 2 long for even n, n for odd n.
    build_settings real;
    real.layout = layout;
    real.radices = fftn::detail::rule_radices(n % 2 == 0 ? n / 2 : n, profile);
    real.bluestein_above = above;
    const std::size_t h = n / 2 + 1;
    const std::vector<T> xr = src.reals<T>(rows * n);
    const auto r2c_shape = fftn::rows_r2c<T>(n, rows);
    const auto r2c = fftn::detail::view_of(r2c_shape);
    EXPECT_TRUE(
        bitwise_equal(run<T, c>(*make_plan_core(r2c, {.force_isa = a}, real), xr, rows * h),
                      run<T, c>(*make_plan_core(r2c, {.force_isa = b}, real), xr, rows * h)))
        << "r2c";
    const cvec<T> xh = src.complexes<T>(rows * h);
    const auto c2r_shape = fftn::rows_c2r<T>(n, rows);
    const auto c2r = fftn::detail::view_of(c2r_shape);
    EXPECT_TRUE(
        bitwise_equal(run<c, T>(*make_plan_core(c2r, {.force_isa = a}, real), xh, rows * n),
                      run<c, T>(*make_plan_core(c2r, {.force_isa = b}, real), xh, rows * n)))
        << "c2r";
  }
  // Real rows with radix 8 last (first for c2r), where avx512 fuses the pairing and avx2,
  // fusing to radix 4, runs it as a pass of its own: with split passes forced both take the
  // separate route's split run.
  const std::vector<std::pair<std::size_t, std::vector<std::size_t>>> forced = {
      {1024, {8, 8, 8}},
      {4096, {8, 8, 4, 8}},
      {8192, {8, 8, 8, 8}},
  };
  for (const auto& [n, radices] : forced) {
    SCOPED_TRACE(std::string(fftn::isa_name(a)) + " and " + fftn::isa_name(b) +
                 (sizeof(T) == 8 ? " double" : " float") + ", n = " + std::to_string(n) +
                 ", forced radix 8");
    fftn::test::input_source src(0x8A3EULL + n);
    build_settings real;
    real.layout = layout;
    real.radices = radices;
    real.bluestein_above = above;
    const std::size_t h = n / 2 + 1;
    const std::vector<T> xr = src.reals<T>(rows * n);
    const auto r2c_shape = fftn::rows_r2c<T>(n, rows);
    const auto r2c = fftn::detail::view_of(r2c_shape);
    EXPECT_TRUE(
        bitwise_equal(run<T, c>(*make_plan_core(r2c, {.force_isa = a}, real), xr, rows * h),
                      run<T, c>(*make_plan_core(r2c, {.force_isa = b}, real), xr, rows * h)))
        << "r2c";
    const cvec<T> xh = src.complexes<T>(rows * h);
    const auto c2r_shape = fftn::rows_c2r<T>(n, rows);
    const auto c2r = fftn::detail::view_of(c2r_shape);
    EXPECT_TRUE(
        bitwise_equal(run<c, T>(*make_plan_core(c2r, {.force_isa = a}, real), xh, rows * n),
                      run<c, T>(*make_plan_core(c2r, {.force_isa = b}, real), xh, rows * n)))
        << "c2r";
  }
}

TEST(Passes, TargetsThatRoundAlikeGiveTheSameBits) {
  const std::vector<fftn::isa> runnable = fftn::test::runnable_isas();
  const auto runs = [&](fftn::isa t) { return std::ranges::find(runnable, t) != runnable.end(); };
  int pairs = 0;
  const std::vector<std::pair<fftn::isa, fftn::isa>> alike = {
      {fftn::isa::neon, fftn::isa::neon_fcma},
      {fftn::isa::avx2, fftn::isa::avx512},
  };
  for (const auto& [a, b] : alike) {
    if (runs(a) && runs(b)) {
      for (const auto layout :
           {fftn::detail::layout_choice::interleaved, fftn::detail::layout_choice::split}) {
        same_bits<double>(a, b, layout);
        same_bits<float>(a, b, layout);
      }
      ++pairs;
    }
  }
  if (pairs == 0) {
    GTEST_SKIP() << "no pair of such targets runs here";
  }
}

// The 128-bit leaves of the x86 targets (experiment X28, kernel_table.hpp) against each
// target's own leaves, which plans no longer use: every radix up to 16 and 17, 19, 23
// (X28b), both directions, every edge variant, strided lines, several of them; every
// output bit equal.
template <class T> void narrow_leaves_match(fftn::isa t) {
  const auto target = static_cast<fftn::detail::target>(static_cast<unsigned char>(t));
  const auto& own = fftn::detail::of<T>(*fftn::detail::find_kernels(target)).passes;
  const fftn::detail::narrow_leaf_table* table = fftn::detail::find_narrow_leaves(target);
  ASSERT_NE(table, nullptr);
  const auto& narrow = fftn::detail::of<T>(*table);
  constexpr std::size_t lines = 3;
  constexpr std::ptrdiff_t in_stride = 2;
  constexpr std::ptrdiff_t out_stride = 3;
  for (std::size_t slot = 0; slot < fftn::detail::narrow_leaf_slots; ++slot) {
    const std::size_t r = fftn::detail::gen::radices[slot];
    fftn::test::input_source src(0x4E41ULL + r);
    const cvec<T> x = src.complexes<T>(lines * r * in_stride);
    const cvec<T> in_factor = src.complexes<T>(r);
    const cvec<T> out_factor = src.complexes<T>(r);
    fftn::detail::pass_args args;
    args.src_stride = in_stride;
    args.dst_stride = out_stride;
    args.scale = 0.37;
    args.in_factor = in_factor.data();
    args.out_factor = out_factor.data();
    for (std::size_t d = 0; d < 2; ++d) {
      for (std::size_t ie = 0; ie < fftn::detail::edge_op_count; ++ie) {
        for (std::size_t oe = 0; oe < fftn::detail::edge_op_count; ++oe) {
          SCOPED_TRACE(std::string(fftn::isa_name(t)) + ", radix " + std::to_string(r) +
                       ", direction " + std::to_string(d) + ", edges " + std::to_string(ie) +
                       std::to_string(oe));
          const fftn::detail::leaf_fn wide = own.codelet[slot]->leaf[d][ie][oe];
          const fftn::detail::leaf_fn half = narrow.leaf[slot][d][ie][oe];
          ASSERT_EQ(wide == nullptr, half == nullptr);
          if (wide == nullptr) {
            continue;
          }
          cvec<T> y_wide(lines * r * out_stride);
          cvec<T> y_half(lines * r * out_stride);
          const auto in_dist = static_cast<std::ptrdiff_t>(r * in_stride * sizeof(std::complex<T>));
          const auto out_dist =
              static_cast<std::ptrdiff_t>(r * out_stride * sizeof(std::complex<T>));
          wide(&args, x.data(), y_wide.data(), in_dist, out_dist, lines);
          half(&args, x.data(), y_half.data(), in_dist, out_dist, lines);
          EXPECT_TRUE(bitwise_equal(y_wide, y_half));
        }
      }
    }
  }
}

// The leaves for lines of stride 1 (experiment X28b, passes/unit_leaf.inl) against the
// leaves for any stride on the same lines, the target's own and the narrow ones: every
// radix that has one, both directions, the edges none and scale, several lines in one call;
// every output bit equal. A contiguous row takes them.
template <class T> void unit_leaves_match(fftn::isa t) {
  const auto target = static_cast<fftn::detail::target>(static_cast<unsigned char>(t));
  const auto& own = fftn::detail::of<T>(*fftn::detail::find_kernels(target)).passes;
  const fftn::detail::narrow_leaf_table* narrow = fftn::detail::find_narrow_leaves(target);
  constexpr std::size_t lines = 3;
  for (const std::uint8_t radix : fftn::detail::gen::radices) {
    const std::size_t r = radix;
    const std::size_t slot = fftn::detail::codelet_slot(r);
    fftn::test::input_source src(0x7E1FULL + r);
    const cvec<T> x = src.complexes<T>(lines * r);
    fftn::detail::pass_args args;
    args.scale = 0.37;
    const auto dist = static_cast<std::ptrdiff_t>(r * sizeof(std::complex<T>));
    for (std::size_t d = 0; d < 2; ++d) {
      for (std::size_t ie = 0; ie < 2; ++ie) {
        for (std::size_t oe = 0; oe < 2; ++oe) {
          SCOPED_TRACE(std::string(fftn::isa_name(t)) + ", radix " + std::to_string(r) +
                       ", direction " + std::to_string(d) + ", edges " + std::to_string(ie) +
                       std::to_string(oe));
          const fftn::detail::leaf_fn unit = own.codelet[slot]->leaf_unit[d][ie][oe];
          const fftn::detail::leaf_fn any = own.codelet[slot]->leaf[d][ie][oe];
          ASSERT_EQ(unit == nullptr, any == nullptr || r > 32 || narrow != nullptr);
          if (unit != nullptr) {
            cvec<T> y_unit(lines * r);
            cvec<T> y_any(lines * r);
            unit(&args, x.data(), y_unit.data(), dist, dist, lines);
            any(&args, x.data(), y_any.data(), dist, dist, lines);
            EXPECT_TRUE(bitwise_equal(y_unit, y_any));
          }
          if (narrow != nullptr && slot < fftn::detail::narrow_leaf_slots) {
            const auto& half = fftn::detail::of<T>(*narrow);
            const fftn::detail::leaf_fn half_unit = half.unit[slot][d][ie][oe];
            const fftn::detail::leaf_fn half_any = half.leaf[slot][d][ie][oe];
            ASSERT_EQ(half_unit == nullptr, half_any == nullptr);
            if (half_unit != nullptr) {
              cvec<T> y_unit(lines * r);
              cvec<T> y_any(lines * r);
              half_unit(&args, x.data(), y_unit.data(), dist, dist, lines);
              half_any(&args, x.data(), y_any.data(), dist, dist, lines);
              EXPECT_TRUE(bitwise_equal(y_unit, y_any));
            }
          }
        }
      }
    }
  }
  const auto row =
      make_plan_core(fftn::detail::view_of(fftn::rows_c2c<T>(8, 1, fftn::direction::forward)),
                     {.force_isa = t}, {});
  const auto* st = std::get_if<fftn::detail::stage>(&row->prog.records.back());
  ASSERT_NE(st, nullptr);
  const auto* half = narrow != nullptr ? &fftn::detail::of<T>(*narrow) : nullptr;
  EXPECT_EQ(st->leaf, half != nullptr
                          ? half->unit[fftn::detail::codelet_slot(8)][0][0][0]
                          : own.codelet[fftn::detail::codelet_slot(8)]->leaf_unit[0][0][0]);
}

TEST(Passes, UnitLeavesMatchTheStridedLeaves) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    unit_leaves_match<double>(t);
    unit_leaves_match<float>(t);
  }
}

// The pair leaves (experiment X28b, passes/pair_leaf.inl): each entry of a target's table
// is chosen for contiguous rows of its length, gives the bits of the chain r1 x m with
// interleaved passes forced through build_settings::radices, out of place and in place,
// for a batch of rows (one call) and with the scale on either edge, and is accurate against
// the reference; a strided row, a factor edge and build_settings::pair_leaves = false take
// the chain instead.
template <class T> void pair_leaves_match(fftn::isa t) {
  const auto target = static_cast<fftn::detail::target>(static_cast<unsigned char>(t));
  const fftn::detail::pair_leaf_table* table = fftn::detail::find_pair_leaves(target);
  ASSERT_NE(table, nullptr);
  constexpr std::size_t rows = 3;
  for (const fftn::detail::pair_leaf_entry& e : fftn::detail::of<T>(*table).entry) {
    if (e.n == 0) {
      break;
    }
    const std::size_t n = e.n;
    const std::string name =
        "pair leaf radix-" + std::to_string(e.r1) + " x " + std::to_string(e.m) + " ";
    fftn::test::input_source src(0x9A1EULL + n);
    const cvec<T> x = src.complexes<T>(rows * n);
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      for (const int edges : {0, 1, 2}) {
        SCOPED_TRACE(std::string(fftn::isa_name(t)) + (sizeof(T) == 8 ? " double" : " float") +
                     ", n = " + std::to_string(n) + ", direction " +
                     std::to_string(static_cast<int>(dir)) + ", edges " + std::to_string(edges));
        const fftn::options opts{.scale = edges == 0 ? 1.0 : 0.375, .force_isa = t};
        build_settings pair;
        pair.scale_on_input = edges == 2;
        build_settings chain = pair;
        chain.radices = {e.r1, e.m};
        chain.layout = fftn::detail::layout_choice::interleaved;
        const auto rows_shape = fftn::rows_c2c<T>(n, rows, dir);
        const auto shape = fftn::detail::view_of(rows_shape);
        const auto with = make_plan_core(shape, opts, pair);
        const auto without = make_plan_core(shape, opts, chain);
        EXPECT_NE(joined(stage_names(*with)).find(name), std::string::npos)
            << joined(stage_names(*with));
        EXPECT_EQ(joined(stage_names(*without)).find("pair leaf"), std::string::npos)
            << joined(stage_names(*without));
        using c = std::complex<T>;
        const cvec<T> y = run<c, c>(*with, x, rows * n);
        EXPECT_TRUE(bitwise_equal(y, run<c, c>(*without, x, rows * n)));
        EXPECT_TRUE(bitwise_equal(y, run_in_place(*with, x)));
      }
      check_row<T>(t, n, dir, {}, 1.0, false, false, false, {name});
      check_row<T>(t, n, dir, {}, 0.375, false, false, true, {name, "in: scale"});
    }
    // Where the pair leaf does not apply, the chain runs.
    const auto row = fftn::rows_c2c<T>(n, 1, fftn::direction::forward);
    const auto plain = fftn::detail::view_of(row);
    build_settings off;
    off.pair_leaves = false;
    EXPECT_EQ(joined(stage_names(*make_plan_core(plain, {.force_isa = t}, off))).find("pair leaf"),
              std::string::npos);
    const cvec<T> f = src.complexes<T>(n);
    build_settings factor;
    factor.out_factor = f.data();
    EXPECT_EQ(
        joined(stage_names(*make_plan_core(plain, {.force_isa = t}, factor))).find("pair leaf"),
        std::string::npos);
    const fftn::shape<std::complex<T>, std::complex<T>> strided{
        .dir = fftn::direction::forward,
        .axes = {{.n = n, .in_stride = 2, .out_stride = 1}},
        .batches = {},
    };
    EXPECT_EQ(
        joined(stage_names(*make_plan_core(fftn::detail::view_of(strided), {.force_isa = t}, {})))
            .find("pair leaf"),
        std::string::npos);
  }
}

TEST(Passes, PairLeavesMatchTheForcedChain) {
  int targets = 0;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    const auto target = static_cast<fftn::detail::target>(static_cast<unsigned char>(t));
    if (fftn::detail::find_pair_leaves(target) == nullptr) {
      continue;
    }
    pair_leaves_match<double>(t);
    pair_leaves_match<float>(t);
    ++targets;
  }
  if (targets == 0) {
    GTEST_SKIP() << "no target with pair leaves runs here";
  }
}

TEST(Passes, NarrowLeavesMatchTheTargetsLeaves) {
  int targets = 0;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    const auto target = static_cast<fftn::detail::target>(static_cast<unsigned char>(t));
    if (fftn::detail::find_narrow_leaves(target) == nullptr) {
      continue;
    }
    narrow_leaves_match<double>(t);
    narrow_leaves_match<float>(t);
    ++targets;
  }
  if (targets == 0) {
    GTEST_SKIP() << "no target with narrow leaves runs here";
  }
}

} // namespace
