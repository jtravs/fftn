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

// Wide blocks (machine_profile::column_wide_bytes, X52 arm W4b) widen the program's column
// blocks to 1 KiB where both tiles fit column_tile_bytes, and only the program's: a team runs
// the team program, which keeps the narrower blocks. Bits are the chain's either way.
template <class T> void wide_blocks_program_only(fftn::isa target) {
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float"));
  const plane_case pc{.n = {64, 64}};
  build_settings wide;
  wide.profile.column_wide_bytes = 1024;
  wide.profile.column_tile_bytes = std::size_t{1} << 20U;
  build_settings chain = wide;
  chain.lanes_columns = false;
  std::string wide_text;
  std::string chain_text;
  const auto got = run<T>(pc, wide, target, wide_text);
  const auto want = run<T>(pc, chain, target, chain_text);
  if (wide_text.find("adjacent columns") == std::string::npos) {
    return; // no lanes column phase on this target
  }
  EXPECT_NE(wide_text.find("blocks of 64 adjacent columns"), std::string::npos) << wide_text;
  const auto core = fftn::detail::make_plan_core(fftn::detail::view_of(shape_of<T>(pc)),
                                                 {.force_isa = target}, wide);
  const std::string team = fftn::detail::describe(fftn::detail::team_core(*core));
  EXPECT_EQ(team.find("blocks of 64 adjacent columns"), std::string::npos) << team;
  if (radices_of(wide_text) == radices_of(chain_text)) {
    EXPECT_TRUE(got == want) << wide_text << chain_text;
  }
}

TEST(Columns, WideBlocksInTheProgramOnly) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    wide_blocks_program_only<double>(t);
    wide_blocks_program_only<float>(t);
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

// A c2r of two or more axes out of place builds its intermediate packed in the output where
// the output's layout allows (X60; plan_core.cpp, c2r_packed) and otherwise keeps it in the
// workspace block (an odd real axis, an odd output stride, build_settings::c2r_packed 0). Both
// compute the same transform, of any input (the spectrum need not be Hermitian), to within
// twice the tolerance; the input is preserved and no output byte outside the described
// elements is written; the block shrinks to one value per row (and the workspace with it where
// the half spectrum outweighs the lanes' tiles); unrolled and walked programs
// agree. Every shape class of the real axis: the pairing alone (n = 2), the leaves, the fused
// and the separate pairing pass; planes, volumes, batches, a scale, padded outputs.
template <class T> void c2r_packed_matches_block(fftn::isa target) {
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float"));
  struct packed_case {
    std::vector<std::size_t> n;
    std::size_t count = 1;
    std::ptrdiff_t pad = 0; // reals added to every output row (the last axis' rows)
    double scale = 1.0;
    std::size_t fuse = 0; // build_settings::fuse_real_radix (1: the separate pairing pass)
    bool packs = true;
    bool shrinks = false; // the workspace too, not only the block (the lanes' tiles may grow)
  };
  const std::vector<packed_case> cases = {
      {.n = {8, 6}, .count = 2},
      {.n = {6, 2}, .count = 1},
      {.n = {12, 4}, .count = 1},
      {.n = {16, 16}, .count = 2, .pad = 2},
      {.n = {16, 16}, .count = 2, .pad = 1, .packs = false},
      {.n = {5, 7}, .count = 2, .packs = false},
      {.n = {1, 16}, .count = 3},
      {.n = {32, 30}, .count = 1, .scale = 0.25},
      {.n = {64, 32}, .count = 3},
      {.n = {48, 64}, .count = 1, .fuse = 1},
      {.n = {40, 96}, .count = 2, .pad = 4, .scale = 0.5, .fuse = 16},
      {.n = {17, 200}, .count = 1, .fuse = 1},
      {.n = {256, 256}, .count = 1, .shrinks = true},
      {.n = {16, 512}, .count = 1, .shrinks = true},
      {.n = {4, 6, 8}, .count = 1},
      {.n = {3, 5, 4}, .count = 2, .pad = 2, .scale = 0.125},
      {.n = {4, 6, 8}, .count = 1, .pad = 1, .packs = false},
      {.n = {6, 4, 34}, .count = 2},
  };
  const auto run = [&](const packed_case& pc, const build_settings& settings, std::string& text,
                       std::size_t& block, std::size_t& ws_bytes) {
    fftn::shape<std::complex<T>, T> sh;
    const std::size_t r = pc.n.size();
    std::vector<std::ptrdiff_t> stride(r, 1);
    for (std::size_t d = r - 1; d > 0; --d) {
      const auto row = static_cast<std::ptrdiff_t>(pc.n[d]) + (d + 1 == r ? pc.pad : 0);
      stride[d - 1] = stride[d] * row;
    }
    for (std::size_t d = 0; d < r; ++d) {
      sh.axes.push_back({.n = pc.n[d], .out_stride = stride[d]});
    }
    const std::ptrdiff_t dist = stride[0] * static_cast<std::ptrdiff_t>(pc.n[0]);
    sh.batches = {{.count = pc.count, .out_dist = dist}};
    const auto core = fftn::detail::make_plan_core(
        fftn::detail::view_of(sh), {.scale = pc.scale, .force_isa = target}, settings);
    text = fftn::detail::describe(*core);
    block = core->block_elements;
    ws_bytes = fftn::detail::workspace_bytes(*core, 1);
    const auto in_bytes = static_cast<std::size_t>(core->in_hi);
    const auto out_bytes = static_cast<std::size_t>(core->out_hi);
    fftn::test::input_source src(0x60ULL + in_bytes);
    const auto v = src.reals<T>(in_bytes / sizeof(T));
    std::vector<std::byte> in(in_bytes);
    std::memcpy(in.data(), v.data(), in.size());
    const std::vector<std::byte> kept = in;
    std::vector<std::byte> out(out_bytes, std::byte{0xa5});
    const fftn::workspace ws(ws_bytes);
    fftn::detail::execute_core(*core, 0, pc.count, in.data(), out.data(), ws);
    EXPECT_EQ(in, kept) << "the input was not preserved";
    // The described output elements; every other byte must be the fill.
    std::vector<bool> described(out_bytes / sizeof(T), false);
    std::vector<std::size_t> index(r, 0);
    for (std::size_t b = 0; b < pc.count; ++b) {
      for (bool more = true; more;) {
        std::ptrdiff_t at = static_cast<std::ptrdiff_t>(b) * dist;
        for (std::size_t d = 0; d < r; ++d) {
          at += static_cast<std::ptrdiff_t>(index[d]) * stride[d];
        }
        described.at(static_cast<std::size_t>(at)) = true;
        more = false;
        for (std::size_t d = r; d > 0; --d) {
          if (++index[d - 1] < pc.n[d - 1]) {
            more = true;
            break;
          }
          index[d - 1] = 0;
        }
      }
    }
    for (std::size_t i = 0; i < described.size(); ++i) {
      if (!described[i]) {
        for (std::size_t k = 0; k < sizeof(T); ++k) {
          EXPECT_EQ(out[i * sizeof(T) + k], std::byte{0xa5}) << "a gap was written at " << i;
        }
      }
    }
    return out;
  };
  for (const packed_case& pc : cases) {
    std::string what = "n =";
    std::size_t points = 1;
    std::size_t rows = 1;
    for (std::size_t d = 0; d < pc.n.size(); ++d) {
      what += " " + std::to_string(pc.n[d]);
      points *= pc.n[d];
      rows *= d + 1 < pc.n.size() ? pc.n[d] : 1;
    }
    SCOPED_TRACE(what + ", " + std::to_string(pc.count) + " transforms, pad " +
                 std::to_string(pc.pad) + ", fuse " + std::to_string(pc.fuse));
    build_settings rules;
    rules.fuse_real_radix = pc.fuse;
    rules.c2r_packed = 1; // wherever the layout allows, whatever the size
    build_settings walked = rules;
    walked.unroll_calls = false;
    build_settings kept = rules;
    kept.c2r_packed = 0;
    std::string text;
    std::string walked_text;
    std::string kept_text;
    std::size_t block = 0;
    std::size_t kept_block = 0;
    std::size_t ws_bytes = 0;
    std::size_t kept_ws = 0;
    const auto got = run(pc, rules, text, block, ws_bytes);
    EXPECT_TRUE(run(pc, walked, walked_text, block, ws_bytes) == got)
        << "unrolled and walked differ";
    const auto want = run(pc, kept, kept_text, kept_block, kept_ws);
    EXPECT_EQ(text.find("packed") != std::string::npos, pc.packs) << text;
    EXPECT_EQ(kept_text.find("packed"), std::string::npos) << kept_text;
    const std::size_t half = pc.n.back() / 2 + 1;
    EXPECT_EQ(kept_block, rows * half);
    if (pc.packs) {
      EXPECT_EQ(block, rows) << text;
      if (pc.shrinks) {
        EXPECT_LT(ws_bytes, kept_ws);
      }
    } else {
      EXPECT_TRUE(got == want) << "the same plan should give the same bytes";
      continue;
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
    EXPECT_LE(most, tolerance * largest) << text << kept_text;
  }
}

// By the rules a c2r packs its intermediate where one transform's half spectrum exceeds the
// profile's c2r_packed_bytes (X60): 64 x 64 doubles have 64 x 33 x 16 = 33792 bytes of it.
TEST(Columns, C2rPacksBeyondTheProfilesBytes) {
  fftn::shape<std::complex<double>, double> sh;
  sh.axes = {{.n = 64}, {.n = 64}};
  sh.batches = {{.count = 3}};
  const auto text = [&](std::size_t limit, int forced) {
    build_settings settings;
    settings.profile.c2r_packed_bytes = limit;
    settings.c2r_packed = forced;
    return fftn::detail::describe(
        *fftn::detail::make_plan_core(fftn::detail::view_of(sh), {}, settings));
  };
  EXPECT_NE(text(33791, -1).find("packed"), std::string::npos);
  EXPECT_EQ(text(33792, -1).find("packed"), std::string::npos);
  EXPECT_NE(text(33792, 1).find("packed"), std::string::npos);
  EXPECT_EQ(text(0, 0).find("packed"), std::string::npos);
  fftn::shape<std::complex<float>, float> single;
  single.axes = sh.axes;
  build_settings settings;
  settings.profile.c2r_packed_bytes = 16896; // the float plane's half spectrum
  EXPECT_EQ(fftn::detail::describe(
                *fftn::detail::make_plan_core(fftn::detail::view_of(single), {}, settings))
                .find("packed"),
            std::string::npos);
  settings.profile.c2r_packed_bytes = 16895;
  EXPECT_NE(fftn::detail::describe(
                *fftn::detail::make_plan_core(fftn::detail::view_of(single), {}, settings))
                .find("packed"),
            std::string::npos);
  // A limit measured on avx512 (the Xeon 6240's class) leaves every other target's plans on the
  // block; a forced setting still packs.
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    build_settings wide;
    wide.profile.c2r_packed_bytes = 0;
    wide.profile.c2r_packed_avx512_only = true;
    const auto packs = [&](int forced) {
      wide.c2r_packed = forced;
      return fftn::detail::describe(
                 *fftn::detail::make_plan_core(fftn::detail::view_of(sh), {.force_isa = t}, wide))
                 .find("packed") != std::string::npos;
    };
    EXPECT_EQ(packs(-1), t == fftn::isa::avx512) << static_cast<int>(t);
    EXPECT_TRUE(packs(1)) << static_cast<int>(t);
  }
}

TEST(Columns, C2rPackedMatchesTheBlock) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    c2r_packed_matches_block<double>(t);
    c2r_packed_matches_block<float>(t);
  }
}

// Where the lanes cannot take a strided axis, its columns run alone through the chain: the
// last axis strided (no adjacent columns), a column length whose chain has a Bluestein
// stage or a generic pass, the last axis of length 1.
// ---- Lanes over adjacent transforms (DESIGN.md 9.3, task 4.3): the axes of a transform all
// strided, its neighbours one element away in the innermost batch dimension, as when the
// dimension of an array that is not transformed varies fastest.

struct interleaved_case {
  std::vector<std::size_t> n; // the axes, slowest first
  std::size_t count = 1;      // transforms one element apart
  std::size_t outer = 1;      // such blocks after one another
  bool in_place = false;
  double scale = 1.0;
  fftn::direction dir = fftn::direction::forward;
};

template <class T>
fftn::shape<std::complex<T>, std::complex<T>> shape_of(const interleaved_case& ic) {
  fftn::shape<std::complex<T>, std::complex<T>> sh;
  sh.dir = ic.dir;
  auto span = static_cast<std::ptrdiff_t>(ic.count);
  sh.axes.resize(ic.n.size());
  for (std::size_t i = ic.n.size(); i > 0; --i) {
    sh.axes[i - 1] = {.n = ic.n[i - 1], .in_stride = span, .out_stride = span};
    span *= static_cast<std::ptrdiff_t>(ic.n[i - 1]);
  }
  if (ic.outer > 1) {
    sh.batches.push_back({.count = ic.outer, .in_dist = span, .out_dist = span});
  }
  sh.batches.push_back({.count = ic.count, .in_dist = 1, .out_dist = 1});
  return sh;
}

std::size_t elements_of(const interleaved_case& ic) {
  std::size_t e = ic.count * ic.outer;
  for (const std::size_t n : ic.n) {
    e *= n;
  }
  return e;
}

// Runs transforms [first, first + count) of the plan on `in` (out of place into `out`).
template <class T>
void run_range(const fftn::detail::plan_core& core, std::size_t first, std::size_t count,
               std::vector<std::byte>& in, std::vector<std::byte>& out, bool in_place) {
  const fftn::workspace ws(fftn::detail::workspace_bytes(core, 1));
  fftn::detail::execute_core(core, first, count, in.data(), in_place ? in.data() : out.data(), ws);
}

template <class T> void adjacent_transforms(fftn::isa target) {
  using c = std::complex<T>;
  SCOPED_TRACE(std::string(fftn::isa_name(target)) + (sizeof(T) == 8 ? " double" : " float"));
  const std::vector<interleaved_case> cases = {
      {.n = {16}, .count = 12},
      {.n = {8}, .count = 1000},
      {.n = {64}, .count = 3, .scale = 0.25},
      {.n = {256}, .count = 64, .in_place = true},
      {.n = {16, 12}, .count = 2},
      {.n = {16, 16}, .count = 7, .dir = fftn::direction::backward},
      {.n = {36, 20}, .count = 17, .in_place = true, .scale = 0.5},
      {.n = {16, 16}, .count = 7, .outer = 3},
      {.n = {30}, .count = 9, .outer = 4, .in_place = true},
      {.n = {12, 10, 8}, .count = 5},
  };
  for (const interleaved_case& ic : cases) {
    SCOPED_TRACE("n[0] = " + std::to_string(ic.n[0]) + ", axes " + std::to_string(ic.n.size()) +
                 ", " + std::to_string(ic.count) + " x " + std::to_string(ic.outer) +
                 (ic.in_place ? ", in place" : ""));
    const auto sh = shape_of<T>(ic);
    const fftn::options opts{.in_place = ic.in_place, .scale = ic.scale, .force_isa = target};
    build_settings on;
    on.profile.adjacent_min_bytes = 0;      // whatever the arrays' size against this machine's L1
    on.profile.adjacent_team_chain = false; // the route whatever the radices; a team runs it too
    on.profile.adjacent_lanes = true;
    const auto lanes = fftn::detail::make_plan_core(fftn::detail::view_of(sh), opts, on);
    const std::string lanes_text = fftn::detail::describe(*lanes);
    // Short rows of one axis in float on x86 take the lanes over rows instead (lanes_up_to).
    EXPECT_TRUE(lanes_text.find("adjacent transforms") != std::string::npos ||
                lanes_text.find("lanes (blocks of") != std::string::npos)
        << lanes_text;
    build_settings off = on;
    off.lanes_columns = false;
    const auto chain = fftn::detail::make_plan_core(fftn::detail::view_of(sh), opts, off);
    const std::string chain_text = fftn::detail::describe(*chain);
    EXPECT_EQ(chain_text.find("adjacent transforms"), std::string::npos) << chain_text;
    const std::size_t elems = elements_of(ic);
    fftn::test::input_source src(0xAD1ULL + elems);
    const auto v = src.complexes<T>(elems);
    std::vector<std::byte> in(elems * sizeof(c));
    std::memcpy(in.data(), v.data(), in.size());
    const std::size_t transforms = ic.count * ic.outer;
    const auto whole = [&](const fftn::detail::plan_core& core) {
      std::vector<std::byte> x = in;
      std::vector<std::byte> y(in.size(), std::byte{0xa5});
      run_range<T>(core, 0, transforms, x, y, ic.in_place);
      if (!ic.in_place) {
        EXPECT_EQ(x, in) << "the input was not preserved";
      }
      return ic.in_place ? x : y;
    };
    const auto got = whole(*lanes);
    const auto want = whole(*chain);
    std::size_t points = 1;
    for (const std::size_t n : ic.n) {
      points *= n;
    }
    if (radices_of(lanes_text) == radices_of(chain_text)) {
      EXPECT_TRUE(got == want) << lanes_text << chain_text;
    } else {
      const double l = std::max(1.0, std::log2(static_cast<double>(points)));
      EXPECT_LE(relative_difference<T>(got, want), 2 * 3 * std::numeric_limits<T>::epsilon() * l)
          << lanes_text << chain_text;
    }
    // Any split of the transforms into ranges gives the same bits (DESIGN.md 15.5): every
    // transform alone, then runs of 2, 3 and 5.
    for (const std::size_t run_of :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5}}) {
      std::vector<std::byte> x = in;
      std::vector<std::byte> y(in.size(), std::byte{0xa5});
      for (std::size_t first = 0; first < transforms; first += run_of) {
        run_range<T>(*lanes, first, std::min(run_of, transforms - first), x, y, ic.in_place);
      }
      EXPECT_TRUE((ic.in_place ? x : y) == got) << "runs of " << run_of;
    }
  }
}

TEST(Columns, AdjacentTransformsRunAsLanes) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    adjacent_transforms<double>(t);
    adjacent_transforms<float>(t);
  }
}

// Not where the transforms are not one element apart, an axis is contiguous, a length has no
// lane passes, or the transforms together fit adjacent_min_bytes: those stay on the chain.
TEST(Columns, AdjacentTransformsOnlyWhereTheyFit) {
  using c = std::complex<double>;
  const auto described = [](const fftn::shape<c, c>& sh, std::size_t min_bytes) {
    build_settings settings;
    settings.profile.adjacent_min_bytes = min_bytes;
    settings.profile.adjacent_team_chain = false;
    settings.profile.adjacent_lanes = true;
    return fftn::detail::describe(
        *fftn::detail::make_plan_core(fftn::detail::view_of(sh), {}, settings));
  };
  const auto strategy = [&](const fftn::shape<c, c>& sh) { return described(sh, 0); };
  fftn::shape<c, c> columns; // 32 columns of 16 points (a leaf length): 8192 bytes
  columns.axes = {{.n = 16, .in_stride = 32, .out_stride = 32}};
  columns.batches = {{.count = 32, .in_dist = 1, .out_dist = 1}};
  EXPECT_NE(described(columns, 8191).find("adjacent transforms"), std::string::npos);
  EXPECT_EQ(described(columns, 8192).find("adjacent transforms"), std::string::npos);
  {
    // With adjacent_team_chain, a plan whose lanes take the chain's radices keeps the route for
    // one thread and holds a team program that is the chain (the team tests run 16 x 16 x 7).
    build_settings teams;
    teams.profile.adjacent_min_bytes = 0;
    teams.profile.adjacent_team_chain = true;
    teams.profile.adjacent_lanes = true;
    fftn::shape<c, c> planes;
    planes.axes = {{.n = 16, .in_stride = 16 * 7, .out_stride = 16 * 7},
                   {.n = 16, .in_stride = 7, .out_stride = 7}};
    planes.batches = {{.count = 7, .in_dist = 1, .out_dist = 1}};
    for (const fftn::isa t : fftn::test::runnable_isas()) {
      const auto core =
          fftn::detail::make_plan_core(fftn::detail::view_of(planes), {.force_isa = t}, teams);
      EXPECT_NE(fftn::detail::describe(*core).find("adjacent transforms"), std::string::npos)
          << fftn::isa_name(t);
      EXPECT_NE(core->team, nullptr) << fftn::isa_name(t);
    }
  }
  {
    // Off where the profile says (machine_profile::adjacent_lanes: the EPYC 7763's class).
    build_settings off;
    off.profile.adjacent_min_bytes = 0;
    off.profile.adjacent_team_chain = false;
    off.profile.adjacent_lanes = false;
    EXPECT_EQ(fftn::detail::describe(
                  *fftn::detail::make_plan_core(fftn::detail::view_of(columns), {}, off))
                  .find("adjacent transforms"),
              std::string::npos);
  }
  fftn::shape<c, c> gapped;
  gapped.axes = {{.n = 16, .in_stride = 8, .out_stride = 8}};
  gapped.batches = {{.count = 4, .in_dist = 2, .out_dist = 2}};
  EXPECT_EQ(strategy(gapped).find("adjacent transforms"), std::string::npos);
  fftn::shape<c, c> contiguous; // rows, the batch dimension outermost
  contiguous.axes = {{.n = 16}};
  contiguous.batches = {{.count = 4}};
  EXPECT_EQ(strategy(contiguous).find("adjacent transforms"), std::string::npos);
  fftn::shape<c, c> prime; // Bluestein has no lanes
  prime.axes = {{.n = 1009, .in_stride = 4, .out_stride = 4}};
  prime.batches = {{.count = 4, .in_dist = 1, .out_dist = 1}};
  EXPECT_EQ(strategy(prime).find("adjacent transforms"), std::string::npos);
}

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
