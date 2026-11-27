// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Experiment X85's folded DCT-I and DST-I of odd n (core/r2r.cpp, machine_profile::r2r_folded_max),
// forced through the machine profile on every instruction set in both precisions: against the
// quadruple-precision reference within the tolerance the r2r tests apply to these kinds
// (tests/api/r2r_test.cpp), at the lengths X85 measured inside it, and the kinds twice.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/precisions.hpp"
#include "support/tolerance.hpp"
#include "threading/schedule.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace {

using fftn::r2r_kind;
using fftn::detail::build_settings;
using fftn::detail::make_plan_core;

std::string kind_name(r2r_kind k) {
  constexpr std::array<const char*, 9> names{"dct1", "dct2", "dct3", "dct4", "dst1",
                                             "dst2", "dst3", "dst4", "dht"};
  return names.at(static_cast<std::size_t>(k));
}

template <class T> class R2rRoutes : public ::testing::Test {};
TYPED_TEST_SUITE(R2rRoutes, fftn::test::precisions, fftn::test::precision_names);

// One execution of a core on `count` transforms of `reals` reals each (tight), from x (in place:
// on a copy of x), the result.
template <class T>
std::vector<T> run(const fftn::detail::plan_core& core, const std::vector<T>& x, bool in_place) {
  const fftn::workspace ws(fftn::detail::workspace_bytes(core, 1));
  std::vector<T> in = x;
  std::vector<T> out(x.size(), T(-7.25));
  const std::size_t transforms = core.prog.transforms;
  if (in_place) {
    fftn::detail::execute_core(core, 0, transforms, in.data(), in.data(), ws);
    return in;
  }
  fftn::detail::execute_core(core, 0, transforms, in.data(), out.data(), ws);
  EXPECT_EQ(in, x) << "the input was not preserved";
  return out;
}

// The lengths whose folded route X85's rule D1 measured within half the r2r tests' rms bound
// (dev/experiments/X85-r2r-speed.md): odd n up to folded_limit, some of each residue the route
// branches on, and every odd n to 65 (Rader and Bluestein half lengths among them).
constexpr std::size_t folded_limit = 257;

TYPED_TEST(R2rRoutes, FoldedDctOneAndDstOneWithinTheR2rTolerance) {
  using T = TypeParam;
  std::vector<std::size_t> lengths;
  for (std::size_t n = 3; n <= 65; n += 2) {
    lengths.push_back(n);
  }
  for (const std::size_t n : {97U, 127U, 129U, 193U, 255U, 257U, 385U, 513U, 769U, 1025U}) {
    if (n <= folded_limit) {
      lengths.push_back(n);
    }
  }
  fftn::test::outcome o;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const r2r_kind kind : {r2r_kind::dct1, r2r_kind::dst1}) {
      for (const std::size_t n : lengths) {
        if (kind == r2r_kind::dct1 && n < 33) {
          continue; // DCT-I folds from 33 points (core/r2r.cpp, folded_dct1_min)
        }
        const std::string what = kind_name(kind) + ", n = " + std::to_string(n) + ", " +
                                 fftn::isa_name(t) + (sizeof(T) == 8 ? " double" : " float");
        fftn::test::input_source src(0xF01DULL + n);
        const std::vector<double> xd = src.reals<double>(n);
        const std::array<fftn::test::reference::r2r_type, 1> type{
            static_cast<fftn::test::reference::r2r_type>(kind)};
        const std::array<std::size_t, 1> extents{n};
        const std::vector<fftn::test::exact_real> want =
            fftn::test::reference::r2r(type, extents, xd);
        build_settings folded;
        folded.profile.r2r_folded_max = static_cast<std::uint16_t>(folded_limit);
        const fftn::shape<T, T> sh = fftn::rows_r2r<T>(n, 1, kind);
        const auto core = make_plan_core(fftn::detail::view_of(sh), {.force_isa = t}, folded);
        const std::string text = fftn::detail::describe(*core);
        if (text.find("folded route") == std::string::npos) {
          o.fail(what + ": not on the folded route:\n" + text);
          continue;
        }
        const std::vector<T> x(xd.begin(), xd.end());
        const std::vector<T> y = run<T>(*core, x, false);
        // The r2r tests' tolerance for these kinds: the underlying transform of N = 2(n -+ 1)
        // points, a rounding per stage operation (10) and the input's (1).
        const std::size_t big = kind == r2r_kind::dct1 ? 2 * (n - 1) : 2 * (n + 1);
        const fftn::test::tolerance tol =
            fftn::test::derived_tolerance<T>(big, fftn::test::fftn_bound(big)) +
            fftn::test::input_rounding<T>() * 11.0;
        o.within(fftn::test::compare<T>(std::span<const T>(y), want), tol, what);
        // The kind twice: N times the input, within two transforms.
        const std::vector<T> back = run<T>(*core, y, false);
        std::vector<fftn::test::exact_real> scaled(n);
        for (std::size_t j = 0; j < n; ++j) {
          scaled[j] = fftn::test::exact_from(static_cast<double>(x[j]) * static_cast<double>(big));
        }
        o.within(fftn::test::compare<T>(std::span<const T>(back), scaled), tol * 2.0,
                 what + ", twice");
      }
    }
  }
  EXPECT_TRUE(o.passed()) << o.detail();
}

// The folded route under the layouts and edges the extension takes (strides of either sign, gaps,
// in place, a plane whose two axes fold, real factors on both sides, a scale), against the same
// plan on the extension, within twice the r2r tests' tolerance (both routes' errors).
TYPED_TEST(R2rRoutes, FoldedRouteUnderLayoutsAndEdges) {
  using T = TypeParam;
  struct layout_case {
    std::string name;
    r2r_kind kind;
    std::size_t n0; // 0: rows of n1
    std::size_t n1;
    std::size_t count;
    std::ptrdiff_t in_stride;
    std::ptrdiff_t out_stride;
    bool in_place;
    bool factors;
    double scale;
  };
  const std::vector<layout_case> cases{
      {"dct1 rows with gaps", r2r_kind::dct1, 0, 33, 3, 2, 3, false, false, 1.0},
      {"dst1 rows reversed", r2r_kind::dst1, 0, 15, 2, -1, -2, false, false, 1.0},
      {"dct1 rows in place", r2r_kind::dct1, 0, 33, 3, 1, 1, true, false, 1.0},
      {"dst1 rows with factors", r2r_kind::dst1, 0, 31, 2, 1, 1, false, true, 1.0},
      {"dct1 rows with factors and a scale", r2r_kind::dct1, 0, 65, 2, 2, 1, false, true, 0.25},
      {"dct1 planes", r2r_kind::dct1, 35, 33, 2, 1, 1, false, false, 1.0},
      {"dst1 planes in place with factors", r2r_kind::dst1, 7, 13, 2, 1, 1, true, true, 1.0},
  };
  fftn::test::outcome o;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const layout_case& lc : cases) {
      const std::string what =
          lc.name + ", " + std::string(fftn::isa_name(t)) + (sizeof(T) == 8 ? " double" : " float");
      fftn::shape<T, T> sh;
      std::vector<std::size_t> n;
      if (lc.n0 != 0) {
        n.push_back(lc.n0);
      }
      n.push_back(lc.n1);
      // The last axis at the case's strides (the in-place cases at 1), the outer axis after it.
      const std::ptrdiff_t si = lc.in_place ? 1 : lc.in_stride;
      const std::ptrdiff_t so = lc.in_place ? 1 : lc.out_stride;
      const auto mag = [](std::ptrdiff_t v) { return v < 0 ? -v : v; };
      const auto line_in = static_cast<std::ptrdiff_t>(lc.n1) * mag(si);
      const auto line_out = static_cast<std::ptrdiff_t>(lc.n1) * mag(so);
      std::ptrdiff_t span_in = line_in;
      std::ptrdiff_t span_out = line_out;
      if (lc.n0 != 0) {
        sh.axes.push_back({.n = lc.n0, .in_stride = line_in, .out_stride = line_out});
        span_in *= static_cast<std::ptrdiff_t>(lc.n0);
        span_out *= static_cast<std::ptrdiff_t>(lc.n0);
      }
      sh.axes.push_back({.n = lc.n1, .in_stride = si, .out_stride = so});
      for (std::size_t d = 0; d < sh.axes.size(); ++d) {
        sh.kinds.push_back(lc.kind);
      }
      sh.batches = {{.count = lc.count, .in_dist = span_in, .out_dist = span_out}};
      const std::size_t elements = static_cast<std::size_t>(std::max(span_in, span_out)) * lc.count;
      // Element 0 of transform 0: past the reversed strides' reach.
      const std::size_t base_in =
          si < 0 ? static_cast<std::size_t>(mag(si)) * (lc.n1 - 1) : std::size_t{0};
      const std::size_t base_out =
          so < 0 ? static_cast<std::size_t>(mag(so)) * (lc.n1 - 1) : std::size_t{0};
      fftn::plan_edges<T, T> e;
      std::size_t points = 1;
      for (const std::size_t v : n) {
        points *= v;
      }
      std::vector<T> fin(points);
      std::vector<T> fout(points);
      if (lc.factors) {
        std::vector<std::ptrdiff_t> strides;
        if (lc.n0 != 0) {
          strides.push_back(static_cast<std::ptrdiff_t>(lc.n1));
        }
        strides.push_back(1);
        e.in.mul = {.kind = fftn::factor_kind::real, .axis_strides = strides, .batch_strides = {}};
        e.out.mul = e.in.mul;
        fftn::test::input_source f(0xFAC7ULL + points);
        const std::vector<T> a = f.reals<T>(points);
        const std::vector<T> b = f.reals<T>(points);
        for (std::size_t i = 0; i < points; ++i) {
          fin[i] = T(1) + a[i] / T(4);
          fout[i] = T(1) + b[i] / T(4);
        }
      }
      const fftn::options opts{.in_place = lc.in_place, .scale = lc.scale, .force_isa = t};
      build_settings folded;
      folded.profile.r2r_folded_max = 65535;
      const auto make = [&](const build_settings& s) {
        return lc.factors ? fftn::detail::make_plan_core_with_edges(fftn::detail::view_of(sh), opts,
                                                                    fftn::detail::view_of(e), s)
                          : make_plan_core(fftn::detail::view_of(sh), opts, s);
      };
      const auto fold = make(folded);
      const auto ext = make(build_settings{});
      if (fftn::detail::describe(*fold).find("folded route") == std::string::npos) {
        o.fail(what + ": not on the folded route");
        continue;
      }
      fftn::test::input_source src(0x1A70ULL + elements);
      const std::vector<T> x = src.reals<T>(elements);
      const fftn::exec_args args{.in_factor = lc.factors ? fin.data() : nullptr,
                                 .out_factor = lc.factors ? fout.data() : nullptr};
      const auto run_one = [&](const fftn::detail::plan_core& core) {
        const fftn::workspace ws(fftn::detail::workspace_bytes(core, 1));
        std::vector<T> in = x;
        std::vector<T> out(elements, T(-7.25));
        T* dst = lc.in_place ? in.data() : out.data();
        fftn::detail::execute_core(core, 0, lc.count, in.data() + base_in,
                                   (lc.in_place ? in.data() + base_in : dst + base_out), ws, args);
        return lc.in_place ? in : out;
      };
      const std::vector<T> got = run_one(*fold);
      const std::vector<T> want = run_one(*ext);
      std::vector<fftn::test::exact_real> exact(want.size());
      for (std::size_t i = 0; i < want.size(); ++i) {
        exact[i] = fftn::test::exact_from(static_cast<double>(want[i]));
      }
      // Each route's error: the underlying transforms of N = 2(n -+ 1) points per axis (errors
      // adding in quadrature across axes), a rounding per stage operation (10) per phase, the
      // factors' and the scale's (3); twice that for the two routes.
      std::size_t big = 1;
      for (const std::size_t v : n) {
        big *= lc.kind == r2r_kind::dct1 ? 2 * (v - 1) : 2 * (v + 1);
      }
      const fftn::test::tolerance tol =
          (fftn::test::derived_tolerance<T>(big, fftn::test::fftn_bound(big)) +
           fftn::test::input_rounding<T>() * (10.0 * static_cast<double>(n.size()) + 3.0)) *
          2.0;
      o.within(fftn::test::compare<T>(std::span<const T>(got), exact), tol, what);
    }
  }
  EXPECT_TRUE(o.passed()) << o.detail();
}

// Accumulation keeps the extension (the folded DCT-I parks v_0 in the output), and so do even
// lengths, DCT-I below 33 points and lengths above the limit; the plans without the switches are
// main's.
TEST(R2rRoutes, FoldedRouteOnlyWhereItApplies) {
  build_settings folded;
  folded.profile.r2r_folded_max = 101;
  const auto text = [&](std::size_t n, r2r_kind k, const build_settings& s, bool acc) {
    const fftn::shape<double, double> sh = fftn::rows_r2r<double>(n, 2, k);
    if (!acc) {
      return fftn::detail::describe(*make_plan_core(fftn::detail::view_of(sh), {}, s));
    }
    fftn::plan_edges<double, double> e;
    e.out.accumulate = true;
    return fftn::detail::describe(*fftn::detail::make_plan_core_with_edges(
        fftn::detail::view_of(sh), {}, fftn::detail::view_of(e), s));
  };
  EXPECT_NE(text(101, r2r_kind::dct1, folded, false).find("folded route"), std::string::npos);
  EXPECT_NE(text(33, r2r_kind::dct1, folded, false).find("folded route"), std::string::npos);
  EXPECT_EQ(text(31, r2r_kind::dct1, folded, false).find("folded route"), std::string::npos);
  EXPECT_NE(text(5, r2r_kind::dst1, folded, false).find("folded route"), std::string::npos);
  EXPECT_NE(text(99, r2r_kind::dst1, folded, false).find("folded route"), std::string::npos);
  EXPECT_EQ(text(103, r2r_kind::dct1, folded, false).find("folded route"), std::string::npos);
  EXPECT_EQ(text(100, r2r_kind::dct1, folded, false).find("folded route"), std::string::npos);
  EXPECT_EQ(text(101, r2r_kind::dct1, folded, true).find("folded route"), std::string::npos);
  EXPECT_EQ(text(101, r2r_kind::dct1, build_settings{}, false).find("folded route"),
            std::string::npos);
}

} // namespace
