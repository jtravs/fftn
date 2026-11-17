// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Volumes (DESIGN.md 9.3, task 4.2): a c2c volume runs its two inner axes plane at a time and
// its outer axis over blocks of adjacent pencils, with scratch per member of at most a plane
// or a block of pencils, never a second volume. The transforms themselves are checked against
// the reference in tests/api (multidim_test) and their independence of the team size in
// tests/threading.

#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/fftn.hpp"
#include "support/fftn_subject.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace {

struct volume_case {
  std::size_t n0 = 0;
  std::size_t n1 = 0;
  std::size_t n2 = 0;
};

const std::vector<volume_case>& volumes() {
  static const std::vector<volume_case> v{
      {16, 16, 16},  {64, 64, 64},  {256, 256, 256}, {17, 15, 37}, {100, 100, 100},
      {8, 256, 256}, {256, 256, 8}, {31, 2, 19},     {3, 1000, 7}, {128, 64, 1024},
  };
  return v;
}

template <class T> fftn::shape<std::complex<T>, std::complex<T>> shape_of(const volume_case& vc) {
  fftn::shape<std::complex<T>, std::complex<T>> s;
  s.axes = {{.n = vc.n0}, {.n = vc.n1}, {.n = vc.n2}};
  return s;
}

std::string label(const volume_case& vc, bool in_place) {
  return std::to_string(vc.n0) + " x " + std::to_string(vc.n1) + " x " + std::to_string(vc.n2) +
         (in_place ? ", in place" : ", out of place");
}

// The scratch bound (DESIGN.md 9.3): each of a member's two scratch lines holds at most a
// plane of the inner axes or a block of pencils of the outer axis, so the workspace grows with
// a plane, not the volume. The widest block of columns or pencils is 512 bytes of them (four
// of Apple's 128-byte lines: 64 complex floats), and a block is never wider than the last axis
// rounded up to a register of complex values (at most 8), whose idle lanes the tile still
// holds.
template <class T> void scratch_within_bound(fftn::isa target) {
  constexpr std::size_t widest_block = 512 / (2 * sizeof(T));
  constexpr std::size_t register_lanes = 8;
  const auto tiny = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<T>(8, 1, fftn::direction::forward)),
      {.force_isa = target}, fftn::detail::build_settings());
  for (const volume_case& vc : volumes()) {
    for (const bool in_place : {false, true}) {
      SCOPED_TRACE(label(vc, in_place));
      const auto core = fftn::detail::make_plan_core(fftn::detail::view_of(shape_of<T>(vc)),
                                                     {.in_place = in_place, .force_isa = target},
                                                     fftn::detail::build_settings());
      const std::size_t last = (vc.n2 + register_lanes - 1) / register_lanes * register_lanes;
      const std::size_t plane = vc.n1 * last;
      const std::size_t pencils = vc.n0 * std::min(widest_block, last);
      const std::size_t bound = std::max(plane, pencils);
      EXPECT_EQ(core->block_elements, 0U) << "a volume keeps no intermediate array";
      EXPECT_LE(core->scratch_line, bound) << fftn::detail::describe(*core);
      // The bytes a member's scratch takes: two lines, each on a 64-byte boundary, in whole
      // 128-byte lines; a team adds its synchronisation state, which is the plan's no more
      // than any other plan's.
      const std::size_t lines = (2 * bound * 2 * sizeof(T) + 2 * 64 + 127) / 128 * 128;
      EXPECT_LE(core->slot_bytes, lines);
      EXPECT_LE(fftn::detail::workspace_bytes(*core, 1), lines + 64);
      for (const int team : {2, 8, 64}) {
        const auto t = static_cast<std::size_t>(team);
        EXPECT_EQ(fftn::detail::workspace_bytes(*core, team) - t * core->slot_bytes,
                  fftn::detail::workspace_bytes(*tiny, team) - t * tiny->slot_bytes);
      }
      // Never a second volume (the smallest volumes fit a few lines of scratch anyway).
      const std::size_t volume = vc.n0 * vc.n1 * vc.n2 * 2 * sizeof(T);
      if (volume >= std::size_t{1} << 20U) {
        EXPECT_LT(fftn::detail::workspace_bytes(*core, 1), volume / 8);
      }
    }
  }
}

TEST(Volumes, ScratchStaysWithinAPlaneOrABlockOfPencils) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    SCOPED_TRACE(fftn::isa_name(t));
    scratch_within_bound<double>(t);
    scratch_within_bound<float>(t);
  }
}

// The program's shape: after the batch loops, one loop over the planes of axis 0 whose body
// holds the plane's two phases, then the outer axis's phase, whose loops never step along
// axis 0 (its pencils' elements are a whole plane apart).
TEST(Volumes, InnerAxesPlaneAtATime) {
  for (const volume_case& vc : volumes()) {
    SCOPED_TRACE(label(vc, false));
    const auto core = fftn::detail::make_plan_core(fftn::detail::view_of(shape_of<double>(vc)),
                                                   fftn::options(), fftn::detail::build_settings());
    const auto& records = core->prog.records;
    const std::size_t first = core->prog.batch_loops;
    ASSERT_LT(first, records.size());
    const auto* planes = std::get_if<fftn::detail::loop>(&records[first]);
    ASSERT_NE(planes, nullptr);
    EXPECT_EQ(planes->count, vc.n0);
    const auto plane_bytes = static_cast<std::ptrdiff_t>(vc.n1 * vc.n2 * 16);
    EXPECT_EQ(planes->step[0], plane_bytes);
    EXPECT_EQ(planes->step[1], plane_bytes);
    ASSERT_LT(planes->body_end, records.size()) << "the outer axis follows the planes";
    // The plane's body holds loops of its own (its rows, its blocks of columns).
    EXPECT_TRUE(std::holds_alternative<fftn::detail::loop>(records[first + 1]));
    EXPECT_NE(fftn::detail::describe(*core).find("plane at a time"), std::string::npos);
    // The work graph: the planes are one phase of whole planes, the outer axis the next.
    ASSERT_GE(core->graph.phases.size(), 2U);
    EXPECT_EQ(core->graph.phases[0].nest, 1U);
    EXPECT_EQ(core->graph.phases[0].units, vc.n0);
  }
}

// The team program (X43 arm B1): with machine_profile::column_team_bytes a c2c plan of several
// axes keeps a second core whose column blocks are narrower and whose radices are the same,
// for teams of two or more; the workspace of a team covers it; nothing is built where the
// blocks would not differ or the field is 0. Its results equal the program's byte for byte
// (tests/threading, the team cases with a team program).
TEST(TeamProgram, NarrowerBlocksTheSameRadices) {
  fftn::detail::build_settings settings;
  settings.profile.column_team_bytes = 64;
  settings.profile.line_bytes = 64;
  const auto plane = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::planes_c2c<double>(256, 256, 1, fftn::direction::forward)),
      fftn::options(), settings);
  ASSERT_NE(plane->team, nullptr);
  EXPECT_EQ(plane->team->team, nullptr);
  const std::string text = fftn::detail::describe(*plane);
  EXPECT_NE(text.find("with a team of two or more: axis 0 (n = 256): lanes over blocks of 4 "),
            std::string::npos)
      << text;
  // The same radices: only the block in each axis line differs.
  const auto radices = [](const std::string& note) { return note.substr(note.find("radices")); };
  ASSERT_EQ(plane->axis_notes.size(), plane->team->axis_notes.size());
  for (std::size_t i = 0; i < plane->axis_notes.size(); ++i) {
    EXPECT_EQ(radices(plane->axis_notes[i]), radices(plane->team->axis_notes[i]));
  }
  for (const int team : {2, 8}) {
    EXPECT_GE(fftn::detail::workspace_bytes(*plane, team),
              static_cast<std::size_t>(team) * plane->team->slot_bytes);
  }
  EXPECT_EQ(fftn::detail::workspace_bytes(*plane, 1), plane->bytes);
  // Columns the program runs alone through the chain (tiles beyond the budget) run as blocks
  // of a line in the team program, with the chain's radices (X43 arm B2).
  fftn::detail::build_settings tight = settings;
  tight.profile.column_tile_bytes = 1024;
  tight.profile.column_near_tile_bytes = 0;
  const auto chain = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::planes_c2c<double>(2048, 36, 1, fftn::direction::forward)),
      fftn::options(), tight);
  ASSERT_NE(chain->team, nullptr);
  EXPECT_EQ(chain->axis_notes[1].find("adjacent columns"), std::string::npos);
  EXPECT_NE(chain->team->axis_notes[1].find("lanes over blocks of 4 adjacent columns"),
            std::string::npos);
  EXPECT_EQ(radices(chain->axis_notes[1]), radices(chain->team->axis_notes[1]));
  // A volume: both the plane's columns and the outer axis's pencils.
  const auto volume = fftn::detail::make_plan_core(
      fftn::detail::view_of(shape_of<double>({64, 64, 64})), fftn::options(), settings);
  ASSERT_NE(volume->team, nullptr);
  // No team program without the field, for rows, or where every block is already that narrow.
  settings.profile.column_team_bytes = 0;
  EXPECT_EQ(fftn::detail::make_plan_core(fftn::detail::view_of(fftn::planes_c2c<double>(
                                             256, 256, 1, fftn::direction::forward)),
                                         fftn::options(), settings)
                ->team,
            nullptr);
  settings.profile.column_team_bytes = 64;
  EXPECT_EQ(fftn::detail::make_plan_core(
                fftn::detail::view_of(fftn::rows_c2c<double>(256, 64, fftn::direction::forward)),
                fftn::options(), settings)
                ->team,
            nullptr);
  settings.column_block = 4;
  EXPECT_EQ(fftn::detail::make_plan_core(fftn::detail::view_of(fftn::planes_c2c<double>(
                                             256, 256, 1, fftn::direction::forward)),
                                         fftn::options(), settings)
                ->team,
            nullptr);
}

} // namespace
