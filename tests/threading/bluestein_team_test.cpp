// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A Bluestein stage whose convolution runs as two depth-first chains (DESIGN.md 8.5,
// chain_builder::bluestein_depth; experiment X50) on teams. The forward chain's top pass
// writes the convolution's spectrum into scratch line a and the backward chain's first pass
// reads it there; both are stage phases whose chunks the members of a team share, while the
// loops over sub-rows around them are line phases, each line on the scratch lines of the
// member running it (team_env.hpp). The handoff holds because every chunk of a stage phase
// uses the owner's slot, a phase starts only once the one before is done, and the owner
// stays until its task is (engine.hpp). These tests guard it: teams of 2, 3 and 4 members on
// real threads and on executors (in order on the caller, in random order on threads, a
// thread pool), one row and a batch with gaps, in place and out of place, one and two tops,
// the tops forced and by the rules, double on every instruction set the machine runs;
// bitwise against the serial plan, whose rows agree with the quadruple-precision reference.
// The tsan gate stage runs them under ThreadSanitizer.

#include "threading/executors.hpp"
#include "threading/team_cases.hpp"
#include "threading/thread_team.hpp"

#include "core/plan_core.hpp"
#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/tolerance.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace {

using fftn::detail::build_settings;
using fftn::detail::slot;
using fftn::detail::strategy_choice;
using fftn::test::case_buffers;
using fftn::test::team_case;
using cd = std::complex<double>;

// A prime whose convolution (forced to 4096 points, which every complex count splits) is
// short enough for the reference to stay cheap and long enough to cut into many chunks.
constexpr std::size_t q = 1031;
constexpr std::size_t convolution = 4096;
constexpr std::size_t most_rows = 5;

// The rows every case reads, so that the reference evaluates each once (it remembers them).
const std::vector<cd>& rows() {
  static const std::vector<cd> r =
      fftn::test::input_source(0xB1D5ULL).complexes<double>(most_rows * q);
  return r;
}

fftn::test::thread_team& threads() {
  static fftn::test::thread_team t(4);
  return t;
}

const fftn::thread_pool& pool4() {
  static const fftn::thread_pool pool(4);
  return pool;
}

struct x50_case {
  team_case run;
  std::size_t count = 1;
  std::size_t dist = q;
  fftn::direction dir = fftn::direction::forward;
};

// The eager profile (every phase split, chunks of 64 points: team_cases.hpp) with the
// depth-first fields a machine that takes the convolution sets; the host's profile is never
// read. tops: given top factors (empty: the default ones); rules: the rules choose the path
// (on targets with 32 vector registers, in double), otherwise strategy depth forces it.
build_settings settings_for(const std::vector<std::size_t>& tops, bool rules) {
  build_settings s = fftn::test::eager_settings();
  s.strategy = rules ? strategy_choice::rules : strategy_choice::depth;
  s.depth_tops = tops;
  s.rader = 0; // Bluestein for every prime
  s.bluestein_length = convolution;
  s.profile.depth_min_bytes = 0;
  s.profile.depth_part_bytes = 0;
  s.profile.depth_pow2_max_bytes = std::numeric_limits<std::size_t>::max();
  s.profile.depth_top = 16;
  s.profile.depth_top_alt = 20;
  s.profile.depth_convolution_max_bytes = std::size_t{32} << 20U;
  return s;
}

x50_case make(fftn::isa target, const std::vector<std::size_t>& tops, bool rules, std::size_t count,
              bool in_place) {
  x50_case c;
  c.count = count;
  c.dist = count == 1 ? q : q + 3; // a batch's gaps are canaries (out of place) or kept
  c.dir = in_place ? fftn::direction::backward : fftn::direction::forward;
  const fftn::shape<cd, cd> shape{
      .dir = c.dir,
      .axes = {{.n = q}},
      .batches = {{
          .count = count,
          .in_dist = static_cast<std::ptrdiff_t>(c.dist),
          .out_dist = static_cast<std::ptrdiff_t>(c.dist),
      }},
  };
  std::string name =
      std::string(fftn::isa_name(target)) + (rules ? " rules" : " forced") + ", tops";
  for (const std::size_t t : tops) {
    name += " " + std::to_string(t);
  }
  c.run.name = name + ", " + std::to_string(count) + " rows" + (in_place ? ", in place" : "");
  c.run.core = fftn::detail::make_plan_core(fftn::detail::view_of(shape),
                                            {.in_place = in_place, .force_isa = target},
                                            settings_for(tops, rules));
  const std::size_t elements = c.dist * (count - 1) + q;
  std::vector<cd> x(elements);
  for (std::size_t e = 0; e < elements; ++e) {
    const std::size_t b = e / c.dist;
    const std::size_t j = e % c.dist;
    x[e] = j < q ? rows()[b * q + j] : cd(-7.25, 3.5 + static_cast<double>(e));
  }
  c.run.in_place = in_place;
  c.run.in_bytes = elements * sizeof(cd);
  c.run.out_bytes = c.run.in_bytes;
  c.run.input.resize(c.run.in_bytes);
  std::memcpy(c.run.input.data(), x.data(), c.run.in_bytes);
  return c;
}

std::vector<x50_case> cases() {
  std::vector<x50_case> out;
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    const std::string name = fftn::isa_name(t);
    const bool wide = name == "avx512" || name == "neon" || name == "neon_fcma";
    for (const bool in_place : {false, true}) {
      for (const std::size_t count : {std::size_t{1}, most_rows}) {
        for (const std::vector<std::size_t>& tops : {
                 std::vector<std::size_t>{},
                 std::vector<std::size_t>{16},
                 std::vector<std::size_t>{8, 4},
             }) {
          out.push_back(make(t, tops, false, count, in_place));
        }
        if (wide) {
          out.push_back(make(t, {}, true, count, in_place));
        }
      }
    }
  }
  return out;
}

std::vector<std::byte> serial_of(const x50_case& c) { return fftn::test::serial_result(c.run); }

std::vector<std::byte> on_team(const x50_case& c, int size, int group, const fftn::workspace& ws) {
  case_buffers b(c.run);
  threads().run(size, [&](int rank) {
    fftn::detail::execute_team(*c.run.core, b.src(), b.dst(),
                               {.rank = rank, .size = size, .group_size = group}, ws,
                               fftn::completion::all);
  });
  return b.result();
}

std::vector<std::byte> on_executor(const x50_case& c, fftn::executor_ref ex,
                                   const fftn::workspace& ws) {
  case_buffers b(c.run);
  fftn::detail::execute_on(*c.run.core, b.src(), b.dst(), ex, ws);
  return b.result();
}

// The serial result's rows against the reference, its gaps untouched.
void check_against_reference(const x50_case& c, const std::vector<std::byte>& result) {
  std::vector<cd> y(result.size() / sizeof(cd));
  std::memcpy(y.data(), result.data(), result.size());
  // A Bluestein stage's error is that of two transforms of M and the chirps around them.
  const auto tol = fftn::test::derived_tolerance<double>(4 * q, fftn::test::stable_fft_bound);
  for (std::size_t b = 0; b < c.count; ++b) {
    const std::span<const cd> row(rows().data() + b * q, q);
    const auto exact = fftn::test::reference::c2c(static_cast<int>(c.dir), row);
    std::vector<fftn::test::exact_complex> want(q);
    for (std::size_t k = 0; k < q; ++k) {
      want[k] = fftn::test::exact_from(cd(exact[k].re.hi, exact[k].im.hi));
    }
    fftn::test::outcome o;
    o.within(fftn::test::compare<double>(std::span<const cd>(y.data() + b * c.dist, q), want), tol,
             c.run.name + ", row " + std::to_string(b));
    EXPECT_TRUE(o.passed()) << o.detail();
    if (b + 1 < c.count) {
      // In place a gap keeps the input; out of place the output's fill (case_buffers).
      for (std::size_t at = (b * c.dist + q) * sizeof(cd); at < (b + 1) * c.dist * sizeof(cd);
           ++at) {
        const std::byte expect = c.run.in_place ? c.run.input[at] : std::byte{0xa5};
        EXPECT_EQ(result[at], expect) << "gap byte " << at << " after row " << b;
      }
    }
  }
}

// The plan hands the convolution's spectrum from one stage phase to another through scratch
// line a, each phase cut into several chunks, with line phases (the sub-rows) in between.
TEST(BluesteinDepthTeam, PlansHandTheSpectrumThroughScratchA) {
  for (const x50_case& c : cases()) {
    SCOPED_TRACE(c.run.name);
    const fftn::detail::plan_core& core = *c.run.core;
    const std::string text = fftn::detail::describe(core);
    ASSERT_NE(text.find("two depth-first chains"), std::string::npos) << text;
    std::size_t writes = 0;
    std::size_t reads = 0;
    std::size_t lines_between = 0;
    for (const fftn::detail::graph_phase& ph : core.graph.phases) {
      if (ph.nest != 0) {
        lines_between += writes != 0 && reads == 0 ? 1 : 0;
        continue;
      }
      const auto& st = std::get<fftn::detail::stage>(core.prog.records[ph.record]);
      if (st.dst == slot::a && writes == 0) {
        EXPECT_GE(ph.chunks, 2U) << st.name;
        writes = ph.record;
      } else if (st.src == slot::a && writes != 0 && reads == 0) {
        EXPECT_GE(ph.chunks, 2U) << st.name;
        reads = ph.record;
      }
    }
    EXPECT_NE(writes, 0U) << text;
    EXPECT_GT(reads, writes) << text;
    EXPECT_EQ(lines_between, 0U) << "the backward first pass follows the top pass directly";
  }
}

// Every case on teams of 2, 3 and 4 (one cache group, and groups of 2), on real threads and on
// executors: bitwise the serial result, whose rows agree with the reference.
TEST(BluesteinDepthTeam, TeamsAndExecutorsEqualSerial) {
  for (const x50_case& c : cases()) {
    SCOPED_TRACE(c.run.name);
    const std::vector<std::byte> serial = serial_of(c);
    check_against_reference(c, serial);
    const fftn::detail::plan_core& core = *c.run.core;
    const fftn::workspace ws(fftn::detail::workspace_bytes(core, 4));
    fftn::test::random_order_executor random(3, c.count * 31 + 7);
    int shapes_split = 0;
    for (int size = 2; size <= 4; ++size) {
      for (const int group : {0, 2}) {
        const std::string where =
            "team of " + std::to_string(size) + ", groups of " + std::to_string(group);
        // The scheduler splits transforms among the members of a group (the eager profile).
        const fftn::detail::lane_map lanes{.lanes = size, .group_size = group};
        const fftn::detail::schedule plan =
            fftn::detail::make_schedule(core.graph, lanes, core.profile);
        std::size_t split = 0;
        for (int g = 0; g < plan.groups; ++g) {
          const fftn::detail::group_share share = fftn::detail::share_of(plan, lanes, g);
          split += share.cap >= 2 ? share.split : 0;
        }
        shapes_split += split != 0 ? 1 : 0;
        EXPECT_EQ(on_team(c, size, group, ws), serial) << where << ", on threads";
        fftn::test::sequential_executor forward{.width = size, .group = group};
        EXPECT_EQ(on_executor(c, forward, ws), serial) << where << ", executor in order";
        fftn::test::sequential_executor backward{.width = size, .group = group, .reverse = true};
        EXPECT_EQ(on_executor(c, backward, ws), serial) << where << ", executor in reverse";
        random.width = size;
        random.group = group;
        EXPECT_EQ(on_executor(c, random, ws), serial) << where << ", random order on 3 threads";
        const fftn::test::pool_view pool{.pool = &pool4(), .width = size, .group = group};
        EXPECT_EQ(on_executor(c, pool, ws), serial) << where << ", on a pool of 4";
      }
    }
    // A single row splits on every shape; a batch of 5 where a remainder of rows is left to
    // share (teams of 2 and 4 in one group: schedule.cpp, share_of).
    EXPECT_GE(shapes_split, c.count == 1 ? 6 : 2);
  }
}

} // namespace
