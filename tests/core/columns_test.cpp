// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The lanes column phase (DESIGN.md 9.3, experiment X40): the strided axes of c2c planes
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
  std::vector<std::size_t> n;       // the axes, slowest first
  std::size_t count = 1;            // transforms
  std::ptrdiff_t row_pad = 0;       // elements after each row of the last axis
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
    for (const std::size_t block : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{16}}) {
      SCOPED_TRACE(label(pc, block));
      build_settings lanes;
      lanes.column_block = block;
      build_settings chain;
      chain.lanes_columns = false;
      std::string lanes_text;
      std::string chain_text;
      const auto got = run<T>(pc, lanes, target, lanes_text);
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

// Where the lanes cannot take a strided axis, its columns run alone through the chain: the
// last axis strided (no adjacent columns), a column length whose chain has a Bluestein
// stage or a generic pass, the last axis of length 1.
TEST(Columns, OnlyWhereTheColumnsAreAdjacent) {
  const auto text = [](const fftn::shape<std::complex<double>, std::complex<double>>& sh) {
    return fftn::detail::describe(
        *fftn::detail::make_plan_core(fftn::detail::view_of(sh), {}, build_settings()));
  };
  EXPECT_NE(text(fftn::planes_c2c<double>(64, 64, 1, fftn::direction::forward))
                .find("axis 0 (n = 64): lanes over blocks of"),
            std::string::npos);
  fftn::shape<std::complex<double>, std::complex<double>> strided;
  strided.axes = {{.n = 16, .in_stride = 2, .out_stride = 2}, {.n = 16, .in_stride = 32, .out_stride = 32}};
  EXPECT_EQ(text(strided).find("adjacent columns"), std::string::npos) << text(strided);
  EXPECT_EQ(text(fftn::planes_c2c<double>(97, 16, 1, fftn::direction::forward))
                .find("adjacent columns"),
            std::string::npos);
  EXPECT_EQ(text(fftn::planes_c2c<double>(29, 16, 1, fftn::direction::forward))
                .find("adjacent columns"),
            std::string::npos);
  EXPECT_EQ(text(fftn::planes_c2c<double>(16, 1, 1, fftn::direction::forward))
                .find("adjacent columns"),
            std::string::npos);
}

} // namespace
