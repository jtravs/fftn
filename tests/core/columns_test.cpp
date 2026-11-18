// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The lanes column phase (DESIGN.md 9.3, experiments X40 and X41): the strided axes of planes
// and volumes run as the lanes strategy over blocks of adjacent columns. Each column is a
// lane computing its own chain (strategies/lanes.cpp), so the plan must write every output
// byte that the same plan with every column alone through the chain writes
// (build_settings::lanes_columns false), canaries in the gaps included: on every instruction
// set the machine runs, double and float, both directions, in place and out of place, with a
// scale; column counts that are a multiple of the block, that leave a narrower last block,
// and that are smaller than C; padded rows and planes; batches of planes; forced blocks.
// The rows and the chain's columns are checked against the reference elsewhere (core_test,
// the analytic suites); bitwise independence of the team size is tests/threading's.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using fftn::detail::build_settings;

struct plane_case {
  std::vector<std::size_t> n; // the axes, slowest first
  std::size_t count = 1;      // transforms
  std::ptrdiff_t row_pad = 0; // elements after each row of the last axis
  bool in_place = false;
  double scale = 1.0;
  fftn::direction dir = fftn::direction::forward;
};

// The shape of a case: C-ordered with `row_pad` elements after every row, planes and volumes
// tight after that, transforms one after another (with a gap of one row).
template <class T> fftn::shape<std::complex<T>, std::complex<T>> shape_of(const plane_case& pc) {
  fftn::shape<std::complex<T>, std::complex<T>> sh;
  sh.dir = pc.dir;
  const std::size_t r = pc.n.size();
  std::ptrdiff_t span = 1;
  sh.axes.resize(r);
  for (std::size_t i = r; i > 0; --i) {
    const std::size_t d = i - 1;
    sh.axes[d] = {.n = pc.n[d], .in_stride = span, .out_stride = span};
    span *= static_cast<std::ptrdiff_t>(pc.n[d]) + (d + 1 == r ? pc.row_pad : 0);
  }
  const std::ptrdiff_t dist = span + static_cast<std::ptrdiff_t>(pc.n[r - 1]) + pc.row_pad;
  sh.batches = {{.count = pc.count, .in_dist = dist, .out_dist = dist}};
  return sh;
}

template <class T>
std::vector<std::byte> run(const plane_case& pc, const build_settings& settings, fftn::isa target,
                           std::string& text) {
  using c = std::complex<T>;
  const auto sh = shape_of<T>(pc);
  const auto core = fftn::detail::make_plan_core(
      fftn::detail::view_of(sh), {.in_place = pc.in_place, .scale = pc.scale, .force_isa = target},
      settings);
  text = fftn::detail::describe(*core);
  const auto elems = static_cast<std::size_t>(sh.batches[0].in_dist) * pc.count;
  fftn::test::input_source src(0xC01ULL + elems);
  const auto v = src.complexes<T>(elems);
  std::vector<std::byte> in(elems * sizeof(c));
  std::memcpy(in.data(), v.data(), in.size());
  const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  if (pc.in_place) {
    fftn::detail::execute_core(*core, 0, pc.count, in.data(), in.data(), ws);
    return in;
  }
  const std::vector<std::byte> kept = in;
  std::vector<std::byte> out(in.size(), std::byte{0xa5});
  fftn::detail::execute_core(*core, 0, pc.count, in.data(), out.data(), ws);
  EXPECT_EQ(in, kept) << "the input was not preserved";
  return out;
}

// The radices describe() lists for each axis, in order.
std::vector<std::string> radices_of(const std::string& text) {
  std::vector<std::string> out;
  for (std::size_t at = text.find("radices "); at != std::string::npos;
       at = text.find("radices ", at + 1)) {
    out.push_back(text.substr(at, text.find('\n', at) - at));
  }
  return out;
}

// The largest difference between two buffers of complex values, relative to the largest
// magnitude in the second.
template <class T>
double relative_difference(const std::vector<std::byte>& a, const std::vector<std::byte>& b) {
  std::vector<std::complex<T>> x(a.size() / sizeof(std::complex<T>));
  std::vector<std::complex<T>> y(x.size());
  std::memcpy(x.data(), a.data(), x.size() * sizeof(std::complex<T>));
  std::memcpy(y.data(), b.data(), y.size() * sizeof(std::complex<T>));
  double most = 0;
  double largest = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    most = std::max(most, static_cast<double>(std::abs(x[i] - y[i])));
    largest = std::max(largest, static_cast<double>(std::abs(y[i])));
  }
  return largest == 0 ? most : most / largest;
}

std::string label(const plane_case& pc, std::size_t block) {
  std::string s = "n =";
  for (const std::size_t n : pc.n) {
    s += " " + std::to_string(n);
  }
  return s + ", " + std::to_string(pc.count) + " transforms, row pad " +
         std::to_string(pc.row_pad) + (pc.in_place ? ", in place" : "") +
         (pc.dir == fftn::direction::backward ? ", backward" : "") + ", scale " +
         std::to_string(pc.scale) + ", block " + std::to_string(block);
}

template <class T> void columns_equal_chain(fftn::isa target) {
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float"));
  const std::vector<plane_case> cases = {
      {.n = {8, 6}, .count = 2},
      {.n = {16, 16}, .count = 2, .in_place = true},
      {.n = {5, 7}, .count = 2, .dir = fftn::direction::backward},
      {.n = {32, 30}, .count = 1, .scale = 0.25},
      {.n = {64, 32}, .count = 3, .row_pad = 3},
      {.n = {36, 20}, .count = 2, .in_place = true, .scale = 0.5},
      {.n = {12, 3}, .count = 1, .row_pad = 1},
      {.n = {7, 2}, .count = 2, .dir = fftn::direction::backward},
      {.n = {24, 20}, .count = 3},
      {.n = {4, 6, 8}, .count = 1},
      {.n = {3, 5, 4}, .count = 2, .in_place = true},
      {.n = {6, 10, 12}, .count = 1, .row_pad = 2, .scale = 2.0},
      {.n = {256, 256}, .count = 1},
      {.n = {128, 128}, .count = 1, .in_place = true, .dir = fftn::direction::backward},
      {.n = {16, 512}, .count = 1},
  };
  for (const plane_case& pc : cases) {
    for (const std::size_t block :
         {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{16}}) {
      SCOPED_TRACE(label(pc, block));
      build_settings lanes;
      lanes.column_block = block;
      build_settings chain;
      chain.lanes_columns = false;
      std::string lanes_text;
      std::string chain_text;
      const auto got = run<T>(pc, lanes, target, lanes_text);
      // The program's calls unrolled at construction (the default, X41) and its records
      // walked give the same bytes.
      build_settings walked = lanes;
      walked.unroll_calls = false;
      std::string walked_text;
      EXPECT_TRUE(run<T>(pc, walked, target, walked_text) == got) << "unrolled and walked differ";
      const auto want = run<T>(pc, chain, target, chain_text);
      EXPECT_NE(lanes_text.find("adjacent columns"), std::string::npos) << lanes_text;
      EXPECT_EQ(chain_text.find("adjacent columns"), std::string::npos) << chain_text;
      if (radices_of(lanes_text) == radices_of(chain_text)) {
        EXPECT_TRUE(got == want) << "lanes columns and chain columns differ\n"
                                 << lanes_text << chain_text;
      } else {
        // Other radices where the row's last pass would crowd an L1 set (column_radices):
        // the same transform to within twice the tolerance of either.
        std::size_t points = 1;
        for (const std::size_t n : pc.n) {
          points *= n;
        }
        const double l = std::max(1.0, std::log2(static_cast<double>(points)));
        const double tolerance = 2 * 3 * std::numeric_limits<T>::epsilon() * l;
        EXPECT_LE(relative_difference<T>(got, want), tolerance) << lanes_text << chain_text;
      }
    }
  }
}

TEST(Columns, LanesEqualTheChainBitForBit) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    columns_equal_chain<double>(t);
    columns_equal_chain<float>(t);
  }
}

// Forced column radices (build_settings::column_radices, X44) apply to the column phase of
// columns whose length they multiply to, and only there: the same transform as the chain's
// columns within twice the tolerance of either.
template <class T> void forced_column_radices(fftn::isa target) {
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float"));
  struct forced_case {
    plane_case pc;
    std::vector<std::size_t> radices;
    bool applies;
  };
  const std::vector<forced_case> cases = {
      {{.n = {32, 32}}, {4, 8}, true},
      {{.n = {32, 32}, .in_place = true, .dir = fftn::direction::backward}, {32}, true},
      {{.n = {32, 20}, .count = 2, .scale = 0.5}, {8, 4}, true},
      {{.n = {64, 64}}, {4, 16}, true},
      {{.n = {16, 32}}, {4, 8}, false},
  };
  for (const forced_case& fc : cases) {
    SCOPED_TRACE(label(fc.pc, 0));
    build_settings forced;
    forced.column_radices = fc.radices;
    build_settings chain;
    chain.lanes_columns = false;
    std::string forced_text;
    std::string chain_text;
    const auto got = run<T>(fc.pc, forced, target, forced_text);
    const auto want = run<T>(fc.pc, chain, target, chain_text);
    std::string listed = "adjacent columns, radices ";
    for (std::size_t i = 0; i < fc.radices.size(); ++i) {
      listed += (i == 0 ? "" : " x ") + std::to_string(fc.radices[i]);
    }
    EXPECT_EQ(forced_text.find(listed + "\n") != std::string::npos, fc.applies) << forced_text;
    std::size_t points = fc.pc.n[0] * fc.pc.n[1];
    const double l = std::max(1.0, std::log2(static_cast<double>(points)));
    const double tolerance = 2 * 3 * std::numeric_limits<T>::epsilon() * l;
    EXPECT_LE(relative_difference<T>(got, want), tolerance) << forced_text << chain_text;
  }
}

TEST(Columns, ForcedColumnRadices) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    forced_column_radices<double>(t);
    forced_column_radices<float>(t);
  }
}

// A column of a leaf length whose legs fall into one L1 set (column_radices, X52 arm L) runs
// as two lane passes, the most balanced that fit (16 = 4 x 4), not as the leaf (except on
// avx512): the same
// transform as the chain's columns within twice the tolerance of either. The profile's L1 is
// fixed (32 KiB, 8 ways, 64-byte lines) so that the choice does not depend on the host.
template <class T> void crowded_leaf_columns(fftn::isa target) {
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float"));
  const std::vector<plane_case> cases = {
      {.n = {16, 4096}},
      {.n = {16, 2048}, .in_place = true, .dir = fftn::direction::backward},
      {.n = {16, 8, 512}, .count = 2},
  };
  for (const plane_case& pc : cases) {
    SCOPED_TRACE(label(pc, 0));
    build_settings lanes;
    lanes.profile.grid_block_bytes = std::size_t{32} * 1024;
    lanes.profile.l1_ways = 8;
    lanes.profile.line_bytes = 64;
    build_settings chain = lanes;
    chain.lanes_columns = false;
    std::string lanes_text;
    std::string chain_text;
    const auto got = run<T>(pc, lanes, target, lanes_text);
    const auto want = run<T>(pc, chain, target, chain_text);
    const std::size_t at = lanes_text.find("axis 0 (n = 16): lanes over");
    if (at != std::string::npos) {
      const std::string line = lanes_text.substr(at, lanes_text.find('\n', at) - at);
      // Not on avx512, where the Xeon 6240 gained too little (X52).
      EXPECT_TRUE(line.ends_with(target == fftn::isa::avx512 ? "radices 16" : "radices 4 x 4"))
          << lanes_text;
    }
    std::size_t points = 1;
    for (const std::size_t n : pc.n) {
      points *= n;
    }
    const double l = std::max(1.0, std::log2(static_cast<double>(points)));
    const double tolerance = 2 * 3 * std::numeric_limits<T>::epsilon() * l;
    EXPECT_LE(relative_difference<T>(got, want), tolerance) << lanes_text << chain_text;
  }
}

TEST(Columns, CrowdedLeafColumnsTakeTwoPasses) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    crowded_leaf_columns<double>(t);
    crowded_leaf_columns<float>(t);
  }
}

// The strided complex axes of r2c and c2r planes and volumes take the same column phase
// (X41): every output byte as with each column alone through the chain where the radices
// agree, the same transform within twice the tolerance where they do not; in place and out
// of place (the c2r input preserved), batches, a scale, blocks forced or the rule's.
template <class T> void real_columns_equal_chain(fftn::isa target, bool r2c) {
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float") +
               (r2c ? " r2c" : " c2r"));
  struct real_case {
    std::vector<std::size_t> n;
    std::size_t count = 1;
    bool in_place = false;
    double scale = 1.0;
  };
  const std::vector<real_case> cases = {
      {.n = {8, 6}, .count = 2},
      {.n = {16, 16}, .count = 2, .in_place = true},
      {.n = {5, 7}, .count = 2},
      {.n = {32, 30}, .count = 1, .scale = 0.25},
      {.n = {64, 32}, .count = 3},
      {.n = {36, 20}, .count = 2, .in_place = true, .scale = 0.5},
      {.n = {12, 4}, .count = 1},
      {.n = {4, 6, 8}, .count = 1},
      {.n = {3, 5, 4}, .count = 2, .in_place = true},
      {.n = {256, 256}, .count = 1},
      {.n = {128, 128}, .count = 1, .in_place = true},
      {.n = {16, 512}, .count = 1},
  };
  const auto run_real = [&](const real_case& rc, const build_settings& settings,
                            std::string& text) {
    fftn::shape<T, std::complex<T>> fwd;
    fftn::shape<std::complex<T>, T> bwd;
    for (const std::size_t n : rc.n) {
      fwd.axes.push_back({.n = n});
      bwd.axes.push_back({.n = n});
    }
    fwd.batches = {{.count = rc.count}};
    bwd.batches = {{.count = rc.count}};
    bwd.dir = fftn::direction::backward;
    const auto core = fftn::detail::make_plan_core(
        r2c ? fftn::detail::view_of(fwd) : fftn::detail::view_of(bwd),
        {.in_place = rc.in_place, .scale = rc.scale, .force_isa = target}, settings);
    text = fftn::detail::describe(*core);
    const auto bytes = static_cast<std::size_t>(std::max(core->in_hi, core->out_hi));
    fftn::test::input_source src(0x2EA1ULL + bytes);
    const auto v = src.reals<T>(bytes / sizeof(T));
    std::vector<std::byte> in(bytes);
    std::memcpy(in.data(), v.data(), in.size());
    const fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
    if (rc.in_place) {
      fftn::detail::execute_core(*core, 0, rc.count, in.data(), in.data(), ws);
      return in;
    }
    const std::vector<std::byte> kept = in;
    std::vector<std::byte> out(bytes, std::byte{0xa5});
    fftn::detail::execute_core(*core, 0, rc.count, in.data(), out.data(), ws);
    EXPECT_EQ(in, kept) << "the input was not preserved";
    return out;
  };
  for (const real_case& rc : cases) {
    for (const std::size_t block : {std::size_t{0}, std::size_t{3}, std::size_t{16}}) {
      std::string what = "n =";
      for (const std::size_t n : rc.n) {
        what += " " + std::to_string(n);
      }
      SCOPED_TRACE(what + ", " + std::to_string(rc.count) + " transforms" +
                   (rc.in_place ? ", in place" : "") + ", block " + std::to_string(block));
      build_settings lanes;
      lanes.column_block = block;
      build_settings chain;
      chain.lanes_columns = false;
      std::string lanes_text;
      std::string chain_text;
      const auto got = run_real(rc, lanes, lanes_text);
      build_settings walked = lanes;
      walked.unroll_calls = false;
      std::string walked_text;
      EXPECT_TRUE(run_real(rc, walked, walked_text) == got) << "unrolled and walked differ";
      const auto want = run_real(rc, chain, chain_text);
      // The rule may keep a real plane's columns on the chain (column_block: C = 1, short
      // columns, tiles beyond column_tile_bytes); a forced block always takes the lanes.
      if (block != 0) {
        EXPECT_NE(lanes_text.find("adjacent columns"), std::string::npos) << lanes_text;
      }
      EXPECT_EQ(chain_text.find("adjacent columns"), std::string::npos) << chain_text;
      if (radices_of(lanes_text) == radices_of(chain_text)) {
        EXPECT_TRUE(got == want) << "lanes columns and chain columns differ\n"
                                 << lanes_text << chain_text;
        continue;
      }
      // Other radices where the row's last pass would crowd an L1 set (column_radices):
      // the same transform to within twice the tolerance of either, compared as reals.
      std::size_t points = 1;
      for (const std::size_t n : rc.n) {
        points *= n;
      }
      const double l = std::max(1.0, std::log2(static_cast<double>(points)));
      const double tolerance = 2 * 3 * std::numeric_limits<T>::epsilon() * l;
      std::vector<T> x(got.size() / sizeof(T));
      std::vector<T> y(x.size());
      std::memcpy(x.data(), got.data(), x.size() * sizeof(T));
      std::memcpy(y.data(), want.data(), y.size() * sizeof(T));
      double most = 0;
      double largest = 0;
      for (std::size_t i = 0; i < x.size(); ++i) {
        most = std::max(most, static_cast<double>(std::abs(x[i] - y[i])));
        largest = std::max(largest, static_cast<double>(std::abs(y[i])));
      }
      EXPECT_LE(most, tolerance * largest) << lanes_text << chain_text;
    }
  }
}

TEST(Columns, RealPlanesLanesEqualTheChain) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const bool r2c : {true, false}) {
      real_columns_equal_chain<double>(t, r2c);
      real_columns_equal_chain<float>(t, r2c);
    }
  }
}

// Where the lanes cannot take a strided axis, its columns run alone through the chain: the
// last axis strided (no adjacent columns), a column length whose chain has a Bluestein
// stage or a generic pass, the last axis of length 1.
TEST(Columns, OnlyWhereTheColumnsAreAdjacent) {
  const auto text = [](const fftn::shape<std::complex<double>, std::complex<double>>& sh) {
    return fftn::detail::describe(*fftn::detail::make_plan_core(fftn::detail::view_of(sh),
                                                                fftn::options(), build_settings()));
  };
  EXPECT_NE(text(fftn::planes_c2c<double>(64, 64, 1, fftn::direction::forward))
                .find("axis 0 (n = 64): lanes over blocks of"),
            std::string::npos);
  fftn::shape<std::complex<double>, std::complex<double>> strided;
  strided.axes = {
      {.n = 16, .in_stride = 2, .out_stride = 2},
      {.n = 16, .in_stride = 32, .out_stride = 32},
  };
  EXPECT_EQ(text(strided).find("adjacent columns"), std::string::npos) << text(strided);
  EXPECT_EQ(
      text(fftn::planes_c2c<double>(97, 16, 1, fftn::direction::forward)).find("adjacent columns"),
      std::string::npos);
  EXPECT_EQ(
      text(fftn::planes_c2c<double>(29, 16, 1, fftn::direction::forward)).find("adjacent columns"),
      std::string::npos);
  EXPECT_EQ(
      text(fftn::planes_c2c<double>(16, 1, 1, fftn::direction::forward)).find("adjacent columns"),
      std::string::npos);
}

} // namespace
