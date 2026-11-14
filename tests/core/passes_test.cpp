// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The passes of DESIGN.md 8.2 forced in turn (AGENTS.md section 5): every codelet radix as
// the grouped first pass, a middle lane pass and the last lane pass, with copy counts that
// leave tails on every SIMD target; the generic pass in every position; every edge variant
// on every pass kind that has it; on every instruction set the machine runs, in double
// and float, against the quadruple-precision reference. And the acceptance test of task
// 1.2: a fused scale costs no extra pass and gives exactly the scaled result.

#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "gen/codelets.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::make_plan_core;
using fftn::detail::plan_core;

template <class T> using cvec = std::vector<std::complex<T>>;

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
  const auto tol = fftn::test::derived_tolerance<T>(n, fftn::test::stable_fft_bound) +
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
    // s from 2 to 16; and R as a leaf.
    const std::vector<sequence> sequences{
        {.radices = {r, 2}, .names = {"grouped " + rs, "lane radix-2"}},
        {.radices = {2, r}, .names = {"grouped radix-2", "lane " + rs + " forward (m 1"}},
        {.radices = {3, r, 2}, .names = {"lane " + rs + " forward (m 2, s 3)"}},
        {.radices = {4, r, 4}, .names = {"lane " + rs + " forward (m 4, s 4)"}},
        {.radices = {r, r}, .names = {"grouped " + rs, "lane " + rs}},
        {.radices = {r}, .names = {"leaf " + rs}},
    };
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
      {17}, {17, 2}, {2, 17}, {2, 17, 3}, {3, 19, 2}, {18}, {2, 18},
  };
  for (const auto& radices : sequences) {
    std::size_t n = 1;
    for (const std::size_t f : radices) {
      n *= f;
    }
    build_settings settings;
    settings.radices = radices;
    for (const auto dir : {fftn::direction::forward, fftn::direction::backward}) {
      check_row<T>(target, n, dir, settings, 1.0, false, false, false, {"generic radix-1"});
    }
  }
}

TEST(Passes, GenericInEveryPosition) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    generic_in_every_position<double>(t);
    generic_in_every_position<float>(t);
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
  };
  const std::vector<line> lines{
      {.n = 1, .radices = {}, .first = "copy", .last = "copy"},
      {.n = 12, .radices = {}, .first = "leaf radix-12", .last = "leaf radix-12"},
      {.n = 40, .radices = {5, 8}, .first = "grouped radix-5", .last = "lane radix-8"},
      {.n = 17, .radices = {}, .first = "generic radix-17", .last = "generic radix-17"},
      {.n = 34, .radices = {2, 17}, .first = "grouped radix-2", .last = "generic radix-17"},
      {.n = 34, .radices = {17, 2}, .first = "generic radix-17", .last = "lane radix-2"},
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
  for (const std::size_t n : {std::size_t{40}, std::size_t{17}}) {
    const std::string first = n == 40 ? "grouped" : "generic";
    const std::string last = n == 40 ? "lane" : "generic";
    std::vector<std::string> names{first, "in: factor", last, "out: scale"};
    if (wide) {
      names.emplace_back("copy");
    }
    check_row<T>(target, n, fftn::direction::forward, {}, 0.375, true, false, false, names, 2, 3);
    names[1] = "in: scale";
    names[3] = "out: factor";
    check_row<T>(target, n, fftn::direction::backward, {}, 0.375, false, true, false, names, 3, 2);
  }
}

TEST(Passes, StridedRowsCopyAroundTheEdges) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    strided_rows_with_edges<double>(t);
    strided_rows_with_edges<float>(t);
  }
}

// A single generic pass between the caller's arrays uses no scratch; in place it writes
// scratch and is copied back.
TEST(Passes, ScratchOnlyWhenUsed) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    // The view borrows the shape's axes, so the shape must outlive it.
    const auto shape = fftn::rows_c2c<double>(17, 1, fftn::direction::forward);
    const auto rows = fftn::detail::view_of(shape);
    EXPECT_EQ(fftn::detail::workspace_bytes(*make_plan_core(rows, {.force_isa = t}, {}), 1), 0U)
        << fftn::isa_name(t);
    EXPECT_GT(fftn::detail::workspace_bytes(
                  *make_plan_core(rows, {.in_place = true, .force_isa = t}, {}), 1),
              0U)
        << fftn::isa_name(t);
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
  // In place, a prime row is one generic pass into scratch, then a copy back.
  for (const std::size_t n : {std::size_t{17}, std::size_t{1009}}) {
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

} // namespace
