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
  // In place: real kinds in FFTW's padded rows; a prime row, whose one pass goes through
  // scratch and a copy when source and destination are the same memory.
  out.push_back(make_case("r2c_inplace_1000x3" + suffix, fftn::rows_r2c<T>(1000, 3), 3 * 1002,
                          3 * 501, true, settings, target));
  out.push_back(make_case("c2r_inplace_999x3" + suffix, fftn::rows_c2r<T>(999, 3), 3 * 500,
                          3 * 1000, true, settings, target));
  out.push_back(make_case("c2c_inplace_prime_97x5" + suffix, fftn::rows_c2c<T>(97, 5, bwd), 485,
                          485, true, settings, target));
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
  // Real rows (DESIGN.md 9.1): the pairing fused into the chain's edge pass, whose units are
  // mirror pairs of copies or groups; the half-length transform as a grid, with the pairing
  // as a pass of its own.
  fftn::detail::build_settings fused = settings;
  fused.fuse_real_radix = 16;
  out.push_back(make_case("r2c_fused_4096x2" + suffix, fftn::rows_r2c<T>(4096, 2), 2 * 4096,
                          2 * 2049, false, fused, target));
  out.push_back(make_case("c2r_fused_inplace_4096x2" + suffix, fftn::rows_c2r<T>(4096, 2),
                          2 * 2049, 2 * 4098, true, fused, target));
  grid.grid_factors = {16, 64};
  out.push_back(make_case("r2c_grid_2048x2" + suffix, fftn::rows_r2c<T>(2048, 2), 2 * 2048,
                          2 * 1025, false, grid, target));
  grid.grid_factors = {8, 8, 16};
  out.push_back(make_case("c2r_grid3_inplace_2048x2" + suffix, fftn::rows_c2r<T>(2048, 2),
                          2 * 1025, 2 * 2050, true, grid, target));
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
