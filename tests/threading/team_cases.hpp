// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Plans and inputs for the team tests, shared by the real-thread tests and the simulator:
// every kind, ranks 1 to 3, in place and out, strided and multi-dimensional batches, long
// rows split into many chunks. Plans are built through the private plan core so that a test
// can choose the chunking and the machine profile.

#pragma once

#include "core/plan_core.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "threading/schedule.hpp"

#include <fftn/fftn.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fftn::test {

struct team_case {
  std::string name;
  std::shared_ptr<const fftn::detail::plan_core> core;
  std::vector<std::byte> input; // in_bytes of input, then zeros up to the buffer's size
  std::size_t in_bytes = 0;
  std::size_t out_bytes = 0;
  bool in_place = false;
};

// One run's arrays: a fresh copy of the input, and the output filled with a pattern that
// no transform produces by chance.
struct case_buffers {
  explicit case_buffers(const team_case& c)
      : in(c.input), out(c.in_place ? 0 : c.out_bytes, std::byte{0xa5}), in_place(c.in_place),
        out_bytes(c.out_bytes) {}

  [[nodiscard]] const void* src() const { return in.data(); }
  [[nodiscard]] void* dst() { return in_place ? static_cast<void*>(in.data()) : out.data(); }
  [[nodiscard]] std::vector<std::byte> result() const {
    if (in_place) {
      return {in.begin(), in.begin() + static_cast<std::ptrdiff_t>(out_bytes)};
    }
    return out;
  }

  std::vector<std::byte> in;
  std::vector<std::byte> out;
  bool in_place;
  std::size_t out_bytes;
};

// A profile that splits whatever can be split and never runs inline: the engine's every
// path, on transforms small enough for tests.
inline fftn::detail::machine_profile eager_profile() {
  fftn::detail::machine_profile p;
  p.ns_per_work = 1;
  p.dispatch_ns = 0;
  p.sync_ns = 0;
  p.min_points_per_member = 1;
  p.split_min_points = 1;
  p.split_min_points_mesh = 1;
  p.remainder_split_min_points = 1;
  p.cpus = 0;
  // Whole tasks in runs of at most half a share, as chunks are (team_engine::join, X52).
  p.claim_divisor = 2;
  // Short budgets, so that waiting members block and the wake-ups are exercised.
  p.spin_ns = 2000;
  p.yield_ns = 2000;
  return p;
}

inline fftn::detail::build_settings eager_settings() {
  fftn::detail::build_settings s;
  s.chunk_points = 64;
  s.profile = eager_profile();
  return s;
}

template <class In, class Out>
team_case make_case(std::string name, const fftn::shape<In, Out>& s, std::size_t in_elements,
                    std::size_t out_elements, bool in_place,
                    const fftn::detail::build_settings& settings,
                    fftn::isa target = fftn::isa::best) {
  team_case c;
  c.name = std::move(name);
  c.core = fftn::detail::make_plan_core(fftn::detail::view_of(s),
                                        {.in_place = in_place, .force_isa = target}, settings);
  c.in_place = in_place;
  c.in_bytes = in_elements * sizeof(In);
  c.out_bytes = out_elements * sizeof(Out);
  input_source src(in_elements + 17 * out_elements);
  std::vector<std::byte> bytes(c.in_bytes);
  if constexpr (std::is_floating_point_v<In>) {
    const auto v = src.reals<In>(in_elements);
    std::memcpy(bytes.data(), v.data(), c.in_bytes);
  } else {
    const auto v = src.complexes<typename In::value_type>(in_elements);
    std::memcpy(bytes.data(), v.data(), c.in_bytes);
  }
  bytes.resize(in_place ? std::max(c.in_bytes, c.out_bytes) : c.in_bytes);
  c.input = std::move(bytes);
  return c;
}

template <class T>
void add_cases(std::vector<team_case>& out, const fftn::detail::build_settings& settings,
               const std::string& suffix, fftn::isa target = fftn::isa::best) {
  using c = std::complex<T>;
  constexpr auto fwd = fftn::direction::forward;
  constexpr auto bwd = fftn::direction::backward;
  out.push_back(make_case("c2c_2000x1" + suffix, fftn::rows_c2c<T>(2000, 1, fwd), 2000, 2000, false,
                          settings, target));
  out.push_back(make_case("c2c_360x7" + suffix, fftn::rows_c2c<T>(360, 7, bwd), 2520, 2520, false,
                          settings, target));
  out.push_back(make_case("c2c_inplace_1024x3" + suffix, fftn::rows_c2c<T>(1024, 3, fwd), 3072,
                          3072, true, settings, target));
  out.push_back(make_case("c2c_prime_97x5" + suffix, fftn::rows_c2c<T>(97, 5, fwd), 485, 485, false,
                          settings, target));
  out.push_back(make_case("r2c_1000x5" + suffix, fftn::rows_r2c<T>(1000, 5), 5000, 5 * 501, false,
                          settings, target));
  out.push_back(make_case("r2c_999x2" + suffix, fftn::rows_r2c<T>(999, 2), 1998, 2 * 500, false,
                          settings, target));
  out.push_back(make_case("c2r_1000x3" + suffix, fftn::rows_c2r<T>(1000, 3), 3 * 501, 3000, false,
                          settings, target));
  out.push_back(make_case("c2r_45x9" + suffix, fftn::rows_c2r<T>(45, 9), 9 * 23, 9 * 45, false,
                          settings, target));
  out.push_back(make_case("c2c_planes_24x20x3" + suffix, fftn::planes_c2c<T>(24, 20, 3, fwd), 1440,
                          1440, false, settings, target));
  out.push_back(make_case("c2c_planes_inplace_16x12x2" + suffix,
                          fftn::planes_c2c<T>(16, 12, 2, bwd), 384, 384, true, settings, target));
  fftn::shape<c, T> c2r_planes;
  c2r_planes.axes = {{.n = 12}, {.n = 16}};
  c2r_planes.batches = {{.count = 2}};
  out.push_back(make_case("c2r_planes_12x16x2" + suffix, c2r_planes, 2 * 12 * 9, 2 * 12 * 16, false,
                          settings, target));
  fftn::shape<T, c> r2c_planes;
  r2c_planes.axes = {{.n = 10}, {.n = 12}};
  r2c_planes.batches = {{.count = 2}};
  out.push_back(make_case("r2c_planes_10x12x2" + suffix, r2c_planes, 2 * 120, 2 * 10 * 7, false,
                          settings, target));
  fftn::shape<c, c> volume;
  volume.axes = {{.n = 8}, {.n = 6}, {.n = 10}};
  volume.batches = {{.count = 2}};
  out.push_back(
      make_case("c2c_volume_8x6x10x2" + suffix, volume, 960, 960, false, settings, target));
  out.push_back(make_case("c2c_many_64x100" + suffix, fftn::rows_c2c<T>(64, 100, fwd), 6400, 6400,
                          false, settings, target));
  const fftn::shape<T, c> batches{
      .axes = {{.n = 10}},
      .batches = {{.count = 2}, {.count = 3}, {.count = 1}, {.count = 2}},
  };
  out.push_back(
      make_case("r2c_batches_10" + suffix, batches, 120, 12 * 6, false, settings, target));
  // Interleaved rows: element j of row t at 4 j + t, read with a stride.
  const fftn::shape<c, c> strided{
      .dir = fwd,
      .axes = {{.n = 300, .in_stride = 4}},
      .batches = {{.count = 4, .in_dist = 1}},
  };
  out.push_back(
      make_case("c2c_strided_300x4" + suffix, strided, 1200, 1200, false, settings, target));
  // A plane's axes strided, its transforms one element apart: lanes over adjacent transforms
  // (task 4.3).
  const fftn::shape<c, c> interleaved{
      .dir = bwd,
      .axes = {{.n = 24, .in_stride = 20 * 11, .out_stride = 20 * 11},
               {.n = 20, .in_stride = 11, .out_stride = 11}},
      .batches = {{.count = 2, .in_dist = 24 * 20 * 11, .out_dist = 24 * 20 * 11},
                  {.count = 11, .in_dist = 1, .out_dist = 1}},
  };
  out.push_back(make_case("c2c_interleaved_24x20x11x2" + suffix, interleaved, 2 * 24 * 20 * 11,
                          2 * 24 * 20 * 11, false, settings, target));
  out.push_back(make_case("c2c_interleaved_inplace_24x20x11x2" + suffix, interleaved,
                          2 * 24 * 20 * 11, 2 * 24 * 20 * 11, true, settings, target));
  // In place: real kinds in FFTW's padded rows; a prime row, whose one pass goes through
  // scratch and a copy when source and destination are the same memory.
  out.push_back(make_case("r2c_inplace_1000x3" + suffix, fftn::rows_r2c<T>(1000, 3), 3 * 1002,
                          3 * 501, true, settings, target));
  out.push_back(make_case("c2r_inplace_999x3" + suffix, fftn::rows_c2r<T>(999, 3), 3 * 500,
                          3 * 1000, true, settings, target));
  out.push_back(make_case("c2c_inplace_prime_97x5" + suffix, fftn::rows_c2c<T>(97, 5, bwd), 485,
                          485, true, settings, target));
  // Bluestein stages (strategies/chain.hpp): after codelet passes with copies (4 x 47, its
  // chirp shared by the copies where C divides 4, per element otherwise; 3 x 47 always per
  // element where C > 1), on both axes of a plane of primes (one set of tables), and one
  // long row whose convolution passes, chirp passes and head passes are each cut into many
  // chunks (16 x 263: 16 copies of a convolution of M >= 525 points, chunks of 64 points);
  // and a real row through a half-length chain that is one Bluestein stage (s = 1).
  out.push_back(make_case("c2c_bluestein_188x3" + suffix, fftn::rows_c2c<T>(188, 3, bwd), 564, 564,
                          false, settings, target));
  out.push_back(make_case("c2c_bluestein_inplace_141x2" + suffix, fftn::rows_c2c<T>(141, 2, fwd),
                          282, 282, true, settings, target));
  out.push_back(make_case("c2c_planes_bluestein_47x47x2" + suffix,
                          fftn::planes_c2c<T>(47, 47, 2, fwd), 4418, 4418, false, settings,
                          target));
  out.push_back(make_case("c2c_bluestein_long_4208x1" + suffix, fftn::rows_c2c<T>(4208, 1, fwd),
                          4208, 4208, false, settings, target));
  out.push_back(make_case("r2c_bluestein_526x2" + suffix, fftn::rows_r2c<T>(526, 2), 1052, 2 * 264,
                          false, settings, target));
  // Rader stages (experiment X27), the same shapes: 4 x 151 (C copies per vector where C
  // divides 4), 3 x 151 in place (one copy at a time), a plane of primes, 16 x 257 (a
  // convolution of 256 points in many chunks) and a real row whose half is one Rader stage.
  out.push_back(make_case("c2c_rader_604x3" + suffix, fftn::rows_c2c<T>(604, 3, bwd), 1812, 1812,
                          false, settings, target));
  out.push_back(make_case("c2c_rader_inplace_453x2" + suffix, fftn::rows_c2c<T>(453, 2, fwd), 906,
                          906, true, settings, target));
  out.push_back(make_case("c2c_planes_rader_151x151x2" + suffix,
                          fftn::planes_c2c<T>(151, 151, 2, fwd), 45602, 45602, false, settings,
                          target));
  out.push_back(make_case("c2c_rader_long_4112x1" + suffix, fftn::rows_c2c<T>(4112, 1, fwd), 4112,
                          4112, false, settings, target));
  out.push_back(make_case("r2c_rader_514x2" + suffix, fftn::rows_r2c<T>(514, 2), 1028, 2 * 258,
                          false, settings, target));
  // A unit axis: one transform's phases mix stages and loop nests.
  out.push_back(make_case("c2c_planes_1x200x3" + suffix, fftn::planes_c2c<T>(1, 200, 3, fwd), 600,
                          600, false, settings, target));
  out.push_back(make_case("c2c_planes_150x1x2" + suffix, fftn::planes_c2c<T>(150, 1, 2, fwd), 300,
                          300, false, settings, target));
  // Grids (strategies/grid.hpp): the blocks of each phase are the lines a team shares.
  fftn::detail::build_settings grid = settings;
  grid.strategy = fftn::detail::strategy_choice::grid;
  grid.grid_factors = {16, 64};
  out.push_back(make_case("c2c_grid_1024x3" + suffix, fftn::rows_c2c<T>(1024, 3, fwd), 3072, 3072,
                          false, grid, target));
  grid.grid_factors = {8, 16, 16};
  out.push_back(make_case("c2c_grid3_inplace_2048x2" + suffix, fftn::rows_c2c<T>(2048, 2, bwd),
                          4096, 4096, true, grid, target));
  // The depth-first chain (strategies/chain.hpp, experiment X46): its first and top passes'
  // units and the loop over sub-rows are what a team shares; one and two top factors.
  fftn::detail::build_settings depth = settings;
  depth.strategy = fftn::detail::strategy_choice::depth;
  depth.depth_tops = {16};
  out.push_back(make_case("c2c_depth_2048x3" + suffix, fftn::rows_c2c<T>(2048, 3, fwd), 6144, 6144,
                          false, depth, target));
  depth.depth_tops = {8, 4};
  out.push_back(make_case("c2c_depth2_4096x2" + suffix, fftn::rows_c2c<T>(4096, 2, bwd), 8192, 8192,
                          false, depth, target));
  out.push_back(make_case("c2c_depth2_inplace_4096x2" + suffix, fftn::rows_c2c<T>(4096, 2, fwd),
                          8192, 8192, true, depth, target));
  // One top that C does not divide (X51): the first pass's partial blocks of sub-rows, the
  // units a team shares among the whole ones (5 on every x86 target, 10 on avx512's 4 lanes
  // and 8-lane float).
  depth.depth_tops = {5};
  out.push_back(make_case("c2c_depth_partial_2000x3" + suffix, fftn::rows_c2c<T>(2000, 3, fwd),
                          6000, 6000, false, depth, target));
  depth.depth_tops = {10};
  out.push_back(make_case("c2c_depth_partial_inplace_2000x2" + suffix,
                          fftn::rows_c2c<T>(2000, 2, bwd), 4000, 4000, true, depth, target));
  // A Bluestein stage on the depth-first chains (experiment X50): the fused chirp passes,
  // both chains' first and top passes and their loops over sub-rows; Bluestein forced for the
  // prime and its convolution 8192 points, which every target's complex count splits (team
  // test DepthCasesTakeTheDepthFirstConvolution checks that they did), the default tops.
  depth.depth_tops = {};
  depth.rader = 0;
  depth.bluestein_length = 8192;
  out.push_back(make_case("c2c_bluestein_depth_4093x2" + suffix, fftn::rows_c2c<T>(4093, 2, bwd),
                          8186, 8186, false, depth, target));
  out.push_back(make_case("c2c_bluestein_depth_inplace_4093x2" + suffix,
                          fftn::rows_c2c<T>(4093, 2, fwd), 8186, 8186, true, depth, target));
  // Batch leaves (experiment X53): C rows per register, the batch's rows what a team shares,
  // counts that leave a partial group of rows.
  fftn::detail::build_settings batch = settings;
  batch.batch_leaves = fftn::detail::batch_leaf_choice::own;
  out.push_back(make_case("c2c_batch_leaf_15x7" + suffix, fftn::rows_c2c<T>(15, 7, bwd), 105, 105,
                          false, batch, target));
  out.push_back(make_case("c2c_batch_leaf_inplace_64x5" + suffix, fftn::rows_c2c<T>(64, 5, fwd),
                          320, 320, true, batch, target));
  out.push_back(make_case("c2c_batch_leaf_12x9" + suffix, fftn::rows_c2c<T>(12, 9, bwd), 108, 108,
                          false, batch, target));
  out.push_back(make_case("c2c_batch_leaf_inplace_25x6" + suffix, fftn::rows_c2c<T>(25, 6, fwd),
                          150, 150, true, batch, target));
  // As the rules choose them (avx2 and avx512 double from 2 rows, NEON float's 20 and 32 too).
  out.push_back(make_case("c2c_batch_rule_25x3" + suffix, fftn::rows_c2c<T>(25, 3, bwd), 75, 75,
                          false, settings, target));
  out.push_back(make_case("c2c_batch_rule_32x5" + suffix, fftn::rows_c2c<T>(32, 5, bwd), 160, 160,
                          false, settings, target));
  // Split lane passes (DESIGN.md 6.3, strategies/chain.hpp), forced with a first radix of
  // 16, which every target's split_block divides: a run of three passes (BL in the
  // middle), of two in place, in the complex chains of real rows, and on the long axis of
  // planes. Each split pass's units are runs of W copies, cut into many chunks.
  fftn::detail::build_settings split = settings;
  split.layout = fftn::detail::layout_choice::split;
  split.radices = {16, 5, 5, 5};
  out.push_back(make_case("c2c_split_2000x1" + suffix, fftn::rows_c2c<T>(2000, 1, fwd), 2000, 2000,
                          false, split, target));
  split.radices = {16, 8, 8};
  out.push_back(make_case("c2c_split_inplace_1024x3" + suffix, fftn::rows_c2c<T>(1024, 3, bwd),
                          3072, 3072, true, split, target));
  split.fuse_real_radix = 1; // the pairing a pass of its own: the half-length chain is c2c
  out.push_back(make_case("r2c_split_2048x2" + suffix, fftn::rows_r2c<T>(2048, 2), 4096, 2 * 1025,
                          false, split, target));
  out.push_back(make_case("c2r_split_2048x2" + suffix, fftn::rows_c2r<T>(2048, 2), 2 * 1025, 4096,
                          false, split, target));
  out.push_back(make_case("c2c_planes_split_6x1024x2" + suffix,
                          fftn::planes_c2c<T>(6, 1024, 2, fwd), 12288, 12288, false, split,
                          target));
  // Real rows (DESIGN.md 9.1): the pairing fused into the chain's edge pass, whose units are
  // mirror pairs of copies or groups; the half-length transform as a grid, with the pairing
  // as a pass of its own.
  fftn::detail::build_settings fused = settings;
  fused.fuse_real_radix = 16;
  out.push_back(make_case("r2c_fused_4096x2" + suffix, fftn::rows_r2c<T>(4096, 2), 2 * 4096,
                          2 * 2049, false, fused, target));
  out.push_back(make_case("c2r_fused_inplace_4096x2" + suffix, fftn::rows_c2r<T>(4096, 2), 2 * 2049,
                          2 * 4098, true, fused, target));
  out.push_back(make_case("c2r_fused_4096x2" + suffix, fftn::rows_c2r<T>(4096, 2), 2 * 2049,
                          2 * 4096, false, fused, target));
  grid.grid_factors = {16, 64};
  out.push_back(make_case("r2c_grid_2048x2" + suffix, fftn::rows_r2c<T>(2048, 2), 2 * 2048,
                          2 * 1025, false, grid, target));
  grid.grid_factors = {8, 8, 16};
  out.push_back(make_case("c2r_grid3_inplace_2048x2" + suffix, fftn::rows_c2r<T>(2048, 2), 2 * 1025,
                          2 * 2050, true, grid, target));
  // Out of place the pairing writes the output and the grid rereads it there.
  out.push_back(make_case("c2r_grid3_2048x2" + suffix, fftn::rows_c2r<T>(2048, 2), 2 * 1025,
                          2 * 2048, false, grid, target));
  grid.grid_factors = {16, 64};
  grid.profile.c2r_paired_gather = false;
  out.push_back(make_case("c2r_grid_2048x2" + suffix, fftn::rows_c2r<T>(2048, 2), 2 * 1025,
                          2 * 2048, false, grid, target));
  // Phase 1's gather pairs the bins from the input (X29); its blocks are a team's lines.
  grid.profile.c2r_paired_gather = true;
  out.push_back(make_case("c2r_grid_paired_2048x2" + suffix, fftn::rows_c2r<T>(2048, 2), 2 * 1025,
                          2 * 2048, false, grid, target));
  // The lanes strategy (strategies/lanes.cpp): blocks of rows (here 8, and the tail's blocks
  // of C) that a team claims in runs of transforms of any length and alignment.
  fftn::detail::build_settings lanes = settings;
  lanes.strategy = fftn::detail::strategy_choice::lanes;
  lanes.lane_block = 8;
  out.push_back(make_case("c2c_lanes_24x37" + suffix, fftn::rows_c2c<T>(24, 37, fwd), 24 * 37,
                          24 * 37, false, lanes, target));
  out.push_back(make_case("c2c_lanes_inplace_64x19" + suffix, fftn::rows_c2c<T>(64, 19, bwd),
                          64 * 19, 64 * 19, true, lanes, target));
  out.push_back(make_case("r2c_lanes_32x21" + suffix, fftn::rows_r2c<T>(32, 21), 32 * 21, 17 * 21,
                          false, lanes, target));
  out.push_back(make_case("c2r_lanes_inplace_100x11" + suffix, fftn::rows_c2r<T>(100, 11), 51 * 11,
                          102 * 11, true, lanes, target));
  // Single volumes (task 4.2): whole planes of the inner two axes per member, then the outer
  // axis's blocks of pencils, out of place and in place, with odd and prime extents (a
  // narrower last block of pencils and of columns).
  fftn::shape<c, c> one_volume;
  one_volume.dir = bwd;
  one_volume.axes = {{.n = 24}, {.n = 20}, {.n = 36}};
  fftn::detail::build_settings planes = settings;
  planes.volume_planes = true; // plane at a time on every machine (X43b)
  out.push_back(
      make_case("c2c_volume_24x20x36" + suffix, one_volume, 17280, 17280, false, planes, target));
  one_volume.axes = {{.n = 17}, {.n = 15}, {.n = 37}};
  out.push_back(make_case("c2c_volume_inplace_17x15x37" + suffix, one_volume, 9435, 9435, true,
                          planes, target));
  // A team program (X43 arm B1): teams run narrower column blocks than the program (here a
  // line of columns), which must give the program's bits; a plane, a volume in place.
  fftn::detail::build_settings team = planes;
  team.profile.column_team_bytes = 32;
  out.push_back(make_case("c2c_planes_team_64x48x1" + suffix, fftn::planes_c2c<T>(64, 48, 1, fwd),
                          3072, 3072, false, team, target));
  one_volume.axes = {{.n = 12}, {.n = 40}, {.n = 36}};
  out.push_back(make_case("c2c_volume_team_inplace_12x40x36" + suffix, one_volume, 17280, 17280,
                          true, team, target));
  // A team program that runs a volume plane at a time where the program runs its axes one by
  // one (X43b arm VT): the same bits.
  fftn::detail::build_settings order = settings;
  order.volume_planes = false;
  order.profile.team_volume_planes = true;
  one_volume.axes = {{.n = 10}, {.n = 24}, {.n = 30}};
  out.push_back(make_case("c2c_volume_team_order_10x24x30" + suffix, one_volume, 7200, 7200, false,
                          order, target));
  out.push_back(make_case("c2c_volume_team_order_inplace_10x24x30" + suffix, one_volume, 7200, 7200,
                          true, order, target));
  // Three volumes with padded rows and planes, in a team plane at a time (X43b): element
  // (z, y, x) of volume v at v 7740 + z 774 + y 43 + x (rows of 30 padded to 43, planes of
  // 17 rows padded to 18).
  const fftn::shape<c, c> padded_volumes{
      .dir = fwd,
      .axes = {{.n = 10, .in_stride = 774, .out_stride = 774},
               {.n = 17, .in_stride = 43, .out_stride = 43},
               {.n = 30, .in_stride = 1, .out_stride = 1}},
      .batches = {{.count = 3, .in_dist = 7745, .out_dist = 7745}},
  };
  out.push_back(make_case("c2c_volumes_team_padded_3x10x17x30" + suffix, padded_volumes, 3 * 7745,
                          3 * 7745, false, order, target));
  out.push_back(make_case("c2c_volumes_team_padded_inplace_3x10x17x30" + suffix, padded_volumes,
                          3 * 7745, 3 * 7745, true, order, target));
  // Columns the program runs alone through the chain (their tiles exceed the budget), which
  // a team program runs as blocks of a line with the chain's radices (X43 arm B2).
  team.profile.column_tile_bytes = 1024;
  team.profile.column_near_tile_bytes = 0;
  out.push_back(make_case("c2c_planes_team_chain_256x20x1" + suffix,
                          fftn::planes_c2c<T>(256, 20, 1, bwd), 5120, 5120, false, team, target));
}

inline std::vector<team_case> standard_cases(const fftn::detail::build_settings& settings) {
  std::vector<team_case> cases;
  add_cases<double>(cases, settings, "");
  add_cases<float>(cases, settings, "_f32");
  // Float on every other instruction set the machine runs: the targets' complex counts
  // differ (C = 1 on scalar, 2 on neon_fcma, 4 on avx2), and with them the passes' units,
  // tails and copies that the team shares out.
  const fftn::isa best =
      fftn::c2c_plan<float>(fftn::rows_c2c<float>(2, 1, fftn::direction::forward))
          .instruction_set();
  for (const fftn::isa t : runnable_isas()) {
    if (t != best) {
      add_cases<float>(cases, settings, std::string("_f32_") + fftn::isa_name(t), t);
    }
  }
  return cases;
}

inline std::vector<std::byte> serial_result(const team_case& c) {
  case_buffers b(c);
  const fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, 1));
  fftn::detail::execute_core(*c.core, 0, fftn::detail::transform_count(*c.core), b.src(), b.dst(),
                             ws);
  return b.result();
}

} // namespace fftn::test
