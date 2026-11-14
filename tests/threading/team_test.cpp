// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Team executions on real threads (DESIGN.md 13.7): bitwise equality with serial execution
// for every team size from 1 to 16 and several cache-group shapes, every kind and rank,
// with transforms split into many chunks; the two completion semantics; members that arrive
// late, one at a time or never; workspace reuse; no allocation. The tsan gate stage runs
// these under ThreadSanitizer.

#include "support/alloc_hook.hpp"
#include "threading/team_cases.hpp"
#include "threading/thread_team.hpp"

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "support/inputs.hpp"
#include "threading/team_state.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using fftn::test::case_buffers;
using fftn::test::team_case;

constexpr int max_team = 16;

fftn::test::thread_team& threads() {
  static fftn::test::thread_team t(max_team + 1);
  return t;
}

// How the members of one execution turn up.
enum class arrival : std::uint8_t { together, staggered, missing_last, one_by_one };

// Runs c on a team of `size` (cache groups of `group`) sharing ws; returns the output.
std::vector<std::byte> run_team(const team_case& c, int size, int group, fftn::workspace& ws,
                                arrival how = arrival::together) {
  case_buffers b(c);
  const auto member = [&](int rank) {
    if (how == arrival::staggered) {
      std::this_thread::sleep_for(std::chrono::microseconds(30 * rank));
    }
    fftn::detail::execute_team(*c.core, b.src(), b.dst(),
                               {.rank = rank, .size = size, .group_size = group}, ws,
                               fftn::completion::all);
  };
  if (how == arrival::one_by_one) {
    // One thread makes every member's call in turn, as a runtime without concurrent
    // forward progress may: the first does everything, the others find nothing left.
    for (int r = 0; r < size; ++r) {
      member(r);
    }
  } else {
    const int callers = how == arrival::missing_last ? size - 1 : size;
    threads().run(callers, member);
  }
  return b.result();
}

// Resets a workspace's team state: any serial execution does, even of no transforms.
void reset(const team_case& c, const fftn::workspace& ws) {
  case_buffers b(c);
  fftn::detail::execute_core(*c.core, 0, 0, b.src(), b.dst(), ws);
}

// Runs c with completion `done` on a team of `size`; returns the output, and in `early` the
// number of members that returned before it was complete.
std::vector<std::byte> run_checked(const team_case& c, int size, int group, fftn::workspace& ws,
                                   fftn::completion done, int& early) {
  case_buffers b(c);
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  std::atomic<int> before{0};
  threads().run(size, [&](int rank) {
    fftn::detail::execute_team(*c.core, b.src(), b.dst(),
                               {.rank = rank, .size = size, .group_size = group}, ws, done);
    if (done == fftn::completion::all && b.result() != serial) {
      before.fetch_add(1);
    }
  });
  early = before.load();
  return b.result();
}

class TeamBits : public ::testing::TestWithParam<std::size_t> {
public:
  static const std::vector<team_case>& cases() {
    static const std::vector<team_case> all =
        fftn::test::standard_cases(fftn::test::eager_settings());
    return all;
  }
};

// Every team size 1 to 16, cache groups of every member, 2, 3, 5 and all members.
TEST_P(TeamBits, EqualToSerialForEveryTeamSize) {
  const team_case& c = cases()[GetParam()];
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, max_team));
  for (int size = 1; size <= max_team; ++size) {
    for (const int group : {0, 1, 2, 3, 5}) {
      EXPECT_EQ(run_team(c, size, group, ws), serial)
          << c.name << ": team of " << size << ", groups of " << group;
    }
  }
}

std::string case_name(const ::testing::TestParamInfo<std::size_t>& param) {
  return TeamBits::cases()[param.param].name;
}

INSTANTIATE_TEST_SUITE_P(Cases, TeamBits,
                         ::testing::Range(std::size_t{0}, TeamBits::cases().size()), case_name);

// The same through the public API, with the default machine profile and chunking: long
// rows, leftover rows, planes, many short rows.
TEST(TeamExecute, PublicApiDefaultsEqualSerial) {
  using cd = std::complex<double>;
  const auto check = [](const auto& plan, std::size_t in_n, std::size_t out_n, auto in_tag,
                        auto out_tag) {
    using In = decltype(in_tag);
    using Out = decltype(out_tag);
    fftn::test::input_source src(in_n);
    std::vector<In> in(in_n);
    if constexpr (std::is_floating_point_v<In>) {
      in = src.reals<In>(in_n);
    } else {
      in = src.complexes<typename In::value_type>(in_n);
    }
    std::vector<Out> serial(out_n);
    fftn::workspace ws(plan, max_team);
    plan.execute(in.data(), serial.data(), ws);
    for (const int size : {2, 3, 4, 7, 8, 9, 12, 16}) {
      for (const int group : {0, 4}) {
        std::vector<Out> out(out_n);
        threads().run(size, [&](int rank) {
          plan.execute(in.data(), out.data(), {.rank = rank, .size = size, .group_size = group},
                       ws);
        });
        EXPECT_TRUE(
            std::ranges::equal(std::as_bytes(std::span(out)), std::as_bytes(std::span(serial))))
            << plan.describe() << "team of " << size << ", groups of " << group;
      }
    }
  };
  constexpr auto fwd = fftn::direction::forward;
  check(fftn::c2c_plan<double>(fftn::rows_c2c<double>(16384, 1, fwd)), 16384, 16384, cd{}, cd{});
  check(fftn::c2c_plan<double>(fftn::rows_c2c<double>(8192, 9, fwd)), std::size_t{9} * 8192,
        std::size_t{9} * 8192, cd{}, cd{});
  check(fftn::r2c_plan<double>(fftn::rows_r2c<double>(20000, 3)), 60000, std::size_t{3} * 10001,
        double{}, cd{});
  check(fftn::c2c_plan<double>(fftn::planes_c2c<double>(64, 64, 3, fwd)), std::size_t{3} * 4096,
        std::size_t{3} * 4096, cd{}, cd{});
  check(fftn::c2c_plan<float>(fftn::rows_c2c<float>(384, 300, fwd)), std::size_t{384} * 300,
        std::size_t{384} * 300, std::complex<float>{}, std::complex<float>{});
}

TEST(TeamExecute, LateMissingAndOneByOneMembers) {
  const auto& cases = TeamBits::cases();
  for (const std::size_t i : {std::size_t{0}, std::size_t{8}, std::size_t{13}}) {
    const team_case& c = cases[i];
    const std::vector<std::byte> serial = fftn::test::serial_result(c);
    fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, max_team));
    for (const int size : {2, 5, 8, 16}) {
      for (const arrival how : {arrival::staggered, arrival::missing_last, arrival::one_by_one}) {
        EXPECT_EQ(run_team(c, size, 4, ws, how), serial)
            << c.name << ": team of " << size << ", arrival " << static_cast<int>(how);
        if (how == arrival::missing_last) {
          // The missing member skipped an execution, which members may not do on a
          // workspace they go on using: a serial execution, here of no transforms, resets
          // the team state before the next team execution.
          reset(c, ws);
        }
      }
    }
  }
}

// completion::all: on whichever member returns, the whole output is already written.
TEST(TeamExecute, AllCompletesBeforeAnyMemberReturns) {
  const team_case& c = TeamBits::cases()[1];
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  constexpr int size = 6;
  fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, size));
  for (int round = 0; round < 20; ++round) {
    case_buffers b(c);
    std::vector<std::vector<std::byte>> seen(size);
    threads().run(size, [&](int rank) {
      if (rank == round % size) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
      fftn::detail::execute_team(*c.core, b.src(), b.dst(), {.rank = rank, .size = size}, ws,
                                 fftn::completion::all);
      seen[static_cast<std::size_t>(rank)] = b.result();
    });
    for (int r = 0; r < size; ++r) {
      EXPECT_EQ(seen[static_cast<std::size_t>(r)], serial) << "member " << r << ", round " << round;
    }
  }
}

// completion::own: a member returns once its own transforms are written, and they are
// exactly transforms_of(team); the ranges partition the batch in rank order.
TEST(TeamExecute, OwnWritesExactlyTheMembersTransforms) {
  constexpr std::size_t n = 48;
  constexpr std::size_t count = 23;
  const fftn::c2c_plan<double> p(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  fftn::test::input_source src(3);
  const auto in = src.complexes<double>(n * count);
  std::vector<std::complex<double>> serial(n * count);
  fftn::workspace ws(p, max_team);
  p.execute(in.data(), serial.data(), ws);
  for (int size = 1; size <= max_team; ++size) {
    std::size_t next = 0;
    for (int r = 0; r < size; ++r) {
      const fftn::transform_range mine = p.transforms_of({.rank = r, .size = size});
      EXPECT_EQ(mine.first, next);
      EXPECT_LE(mine.count, count / static_cast<std::size_t>(size) + 1);
      EXPECT_GE(mine.count, count / static_cast<std::size_t>(size));
      next = mine.first + mine.count;
    }
    EXPECT_EQ(next, count);
    std::vector<std::complex<double>> out(n * count);
    std::vector<int> right(static_cast<std::size_t>(size), 0);
    threads().run(size, [&](int rank) {
      const fftn::team t{.rank = rank, .size = size};
      p.execute(in.data(), out.data(), t, ws, fftn::completion::own);
      // Before any synchronisation with the other members: only this member's rows.
      const fftn::transform_range mine = p.transforms_of(t);
      const auto from = static_cast<std::ptrdiff_t>(mine.first * n);
      const auto to = static_cast<std::ptrdiff_t>((mine.first + mine.count) * n);
      right[static_cast<std::size_t>(rank)] =
          std::equal(out.begin() + from, out.begin() + to, serial.begin() + from) ? 1 : 0;
    });
    EXPECT_EQ(std::ranges::count(right, 1), size) << "team of " << size;
    EXPECT_EQ(out, serial);
  }
}

// One workspace serves executions in sequence: members of one execution may still be
// arriving while others run the next; serial executions and a change of team size in
// between make the next team execution set the state up afresh; memory need not be zero.
TEST(TeamExecute, WorkspaceReuse) {
  const auto& cases = TeamBits::cases();
  const team_case& a = cases[0];
  const team_case& b = cases[13];
  const std::vector<std::byte> serial_a = fftn::test::serial_result(a);
  const std::vector<std::byte> serial_b = fftn::test::serial_result(b);
  const std::size_t bytes = std::max(fftn::detail::workspace_bytes(*a.core, max_team),
                                     fftn::detail::workspace_bytes(*b.core, max_team));
  std::vector<std::byte> memory(bytes + 64, std::byte{0xff}); // not zero
  fftn::workspace ws{std::span<std::byte>(memory).subspan(7, bytes)};
  constexpr int calls = 12;
  for (const int size : {3, 8, 5}) {
    std::vector<case_buffers> buffers;
    buffers.reserve(calls);
    for (int k = 0; k < calls; ++k) {
      buffers.emplace_back(k % 2 == 0 ? a : b);
    }
    threads().run(size, [&](int rank) {
      for (int k = 0; k < calls; ++k) {
        // The last member lags behind the others by several executions at times.
        if (rank == size - 1 && k % 4 == 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(300));
        }
        const team_case& c = k % 2 == 0 ? a : b;
        auto& buf = buffers[static_cast<std::size_t>(k)];
        fftn::detail::execute_team(*c.core, buf.src(), buf.dst(), {.rank = rank, .size = size}, ws,
                                   fftn::completion::all);
      }
    });
    for (int k = 0; k < calls; ++k) {
      EXPECT_EQ(buffers[static_cast<std::size_t>(k)].result(), k % 2 == 0 ? serial_a : serial_b)
          << "team of " << size << ", execution " << k;
    }
    // A serial execution between team executions.
    case_buffers s(a);
    fftn::detail::execute_core(*a.core, 0, fftn::detail::transform_count(*a.core), s.src(), s.dst(),
                               ws);
    EXPECT_EQ(s.result(), serial_a);
  }
}

// Members that return at once (completion::own with more members than transforms) and a
// moved workspace keep working.
TEST(TeamExecute, MovedWorkspaceAndIdleMembers) {
  const team_case& c = TeamBits::cases()[0]; // one transform
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  fftn::workspace first(fftn::detail::workspace_bytes(*c.core, 4));
  EXPECT_EQ(run_team(c, 4, 0, first), serial);
  fftn::workspace ws(std::move(first));
  EXPECT_EQ(run_team(c, 4, 0, ws), serial);
  case_buffers b(c);
  threads().run(4, [&](int rank) {
    fftn::detail::execute_team(*c.core, b.src(), b.dst(), {.rank = rank, .size = 4}, ws,
                               fftn::completion::own);
  });
  EXPECT_EQ(b.result(), serial);
  fftn::workspace other;
  other = std::move(ws);
  EXPECT_EQ(run_team(c, 4, 0, other), serial);
}

TEST(TeamExecute, NeverAllocates) {
  const auto& cases = TeamBits::cases();
  for (const std::size_t i : {std::size_t{0}, std::size_t{8}, std::size_t{11}, std::size_t{13}}) {
    const team_case& c = cases[i];
    constexpr int size = 8;
    fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, size));
    case_buffers b(c);
    std::atomic<int> go{0};
    std::size_t allocations = 0;
    std::size_t frees = 0;
    threads().run(size + 1, [&](int rank) {
      if (rank == size) {
        // The counter covers exactly the executions: the members wait for it.
        const fftn::test::allocation_counter counter;
        go.store(1);
        while (go.load() != size + 1) {
          std::this_thread::yield();
        }
        allocations = counter.allocations();
        frees = counter.deallocations();
        return;
      }
      while (go.load() == 0) {
        std::this_thread::yield();
      }
      for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
        fftn::detail::execute_team(*c.core, b.src(), b.dst(),
                                   {.rank = rank, .size = size, .group_size = 3}, ws, done);
      }
      go.fetch_add(1);
    });
    EXPECT_EQ(allocations, 0U) << c.name;
    EXPECT_EQ(frees, 0U) << c.name;
  }
}

TEST(TeamExecute, WorkspaceBytesPerLanes) {
  const fftn::c2c_plan<double> p(fftn::rows_c2c<double>(1000, 3, fftn::direction::forward));
  EXPECT_EQ(p.workspace_bytes(), p.workspace_bytes(1));
  EXPECT_EQ(p.workspace_bytes(0), p.workspace_bytes(1));
  std::size_t last = p.workspace_bytes(1);
  for (int lanes = 2; lanes <= 64; ++lanes) {
    EXPECT_GT(p.workspace_bytes(lanes), last);
    last = p.workspace_bytes(lanes);
  }
  const fftn::workspace ws(p, 8);
  EXPECT_EQ(ws.size(), p.workspace_bytes(8));
  // A plan without scratch still needs the team's state.
  const fftn::c2c_plan<double> leaf(fftn::rows_c2c<double>(4, 2, fftn::direction::forward));
  EXPECT_EQ(leaf.workspace_bytes(), 0U);
  EXPECT_GT(leaf.workspace_bytes(2), 0U);
}

// completion::own with a smaller team writes its slots inside a larger team's state; the
// next execution by the larger team must set its state up afresh (the reviewer's sequence:
// all with 8, own with 2, all with 8 returned before the output was written).
TEST(TeamExecute, OwnWithASmallerTeamThenAll) {
  const team_case c = fftn::test::make_case(
      "c2c_4096x8", fftn::rows_c2c<double>(4096, 8, fftn::direction::forward),
      std::size_t{4096} * 8, std::size_t{4096} * 8, false, fftn::test::eager_settings());
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, 8));
  int early = 0;
  EXPECT_EQ(run_checked(c, 8, 1, ws, fftn::completion::all, early), serial);
  EXPECT_EQ(run_checked(c, 2, 0, ws, fftn::completion::own, early), serial);
  // Its scratch overwrote words of the 8-member state, which is no longer valid. (With
  // 40-bit tags compared for equality, scratch read as counters almost always reads as zero,
  // so the output check below alone would not show it.)
  EXPECT_EQ(fftn::detail::workspace_access::team_word(ws).load(), 0U);
  EXPECT_EQ(run_checked(c, 8, 1, ws, fftn::completion::all, early), serial);
  EXPECT_EQ(early, 0);
  // An own execution by a team of the state's size keeps it.
  EXPECT_EQ(run_checked(c, 8, 1, ws, fftn::completion::own, early), serial);
  EXPECT_NE(fftn::detail::workspace_access::team_word(ws).load(), 0U);
}

// Every mix of completions, team sizes, groupings, plans and serial executions on one
// workspace, each execution complete and right.
TEST(TeamExecute, MixedSequencesOnOneWorkspace) {
  const auto& cases = TeamBits::cases();
  const std::array<const team_case*, 4> plans{cases.data(), &cases[8], &cases[13], &cases[10]};
  std::size_t bytes = 0;
  for (const team_case* c : plans) {
    bytes = std::max(bytes, fftn::detail::workspace_bytes(*c->core, 8));
  }
  fftn::workspace ws(bytes);
  std::mt19937_64 rng(11); // NOLINT(bugprone-random-generator-seed): reproducible on purpose
  for (int step = 0; step < 60; ++step) {
    const team_case& c = *plans[rng() % 4];
    const int size = static_cast<int>(1 + rng() % 8);
    const int group = static_cast<int>(rng() % 4);
    const auto kind = rng() % 3;
    int early = 0;
    if (kind == 2) {
      case_buffers b(c);
      fftn::detail::execute_core(*c.core, 0, fftn::detail::transform_count(*c.core), b.src(),
                                 b.dst(), ws);
      EXPECT_EQ(b.result(), fftn::test::serial_result(c)) << "step " << step;
      continue;
    }
    const fftn::completion done = kind == 0 ? fftn::completion::all : fftn::completion::own;
    EXPECT_EQ(run_checked(c, size, group, ws, done, early), fftn::test::serial_result(c))
        << "step " << step << ": " << c.name << ", team of " << size << ", completion " << kind;
    EXPECT_EQ(early, 0) << "step " << step;
  }
}

// The team state ages: as if each member had made about 2^31, 2^32, 2^40 and 2^41
// executions, a split plan whose words were never written still completes before any
// member returns (with 32-bit epochs compared modulo 2^32 it did not).
TEST(TeamExecute, StateAgesWithoutLimit) {
  const auto& cases = TeamBits::cases();
  const team_case& a = cases[13]; // many rows, whole
  const team_case& b = cases[0];  // one row, split
  constexpr int size = 4;
  fftn::workspace ws(std::max(fftn::detail::workspace_bytes(*a.core, size),
                              fftn::detail::workspace_bytes(*b.core, size)));
  const auto address = reinterpret_cast<std::uintptr_t>(ws.memory().data());
  std::byte* state =
      ws.memory().data() +
      (fftn::detail::sync_line - address % fftn::detail::sync_line) % fftn::detail::sync_line;
  // Each member's count of executions jumps to `calls`. Three executions first (epochs 1 to
  // 3) leave every word plan a uses tagged 3; plan b's are never written. A jump that made
  // the next tag 3 again would be the documented limit (a word reused exactly a multiple of
  // 2^40 executions after its last write); 2^40 - 1 makes the next epoch's tag 0, which is
  // skipped, so that epoch is tagged 1.
  for (const std::uint64_t calls : {
           (std::uint64_t{1} << 31U) + 5,
           (std::uint64_t{1} << 32U) + 5,
           (std::uint64_t{1} << 40U) - 1,
           (std::uint64_t{1} << 41U) + 1000,
       }) {
    reset(a, ws);
    int early = 0;
    for (int k = 0; k < 3; ++k) {
      EXPECT_EQ(run_checked(a, size, 2, ws, fftn::completion::all, early),
                fftn::test::serial_result(a));
    }
    for (int r = 0; r < size; ++r) {
      auto* word = std::launder(reinterpret_cast<std::atomic<std::uint64_t>*>(
          state + fftn::detail::team_layout::calls(static_cast<std::size_t>(r))));
      word->store(calls);
    }
    EXPECT_EQ(run_checked(a, size, 2, ws, fftn::completion::all, early),
              fftn::test::serial_result(a))
        << calls;
    EXPECT_EQ(run_checked(b, size, 2, ws, fftn::completion::all, early),
              fftn::test::serial_result(b))
        << calls;
    EXPECT_EQ(early, 0) << calls;
  }
}

// A member that skips an execution may not go on to the next on the same workspace without
// a reset; after a serial execution of no transforms, every member returns from the next
// execution only once its output is written.
TEST(TeamExecute, ResetAfterASkippedExecution) {
  const team_case& c = TeamBits::cases()[13];
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  constexpr int size = 4;
  fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, size));
  EXPECT_EQ(run_team(c, size, 0, ws, arrival::missing_last), serial);
  reset(c, ws);
  for (int round = 0; round < 5; ++round) {
    case_buffers b(c);
    std::atomic<int> early{0};
    threads().run(size, [&](int rank) {
      if (rank != size - 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      fftn::detail::execute_team(*c.core, b.src(), b.dst(), {.rank = rank, .size = size}, ws,
                                 fftn::completion::all);
      if (b.result() != serial) {
        early.fetch_add(1);
      }
    });
    EXPECT_EQ(early.load(), 0) << "round " << round;
  }
}

// An oversubscribed team (more members than the profile's CPUs) splits nothing and does not
// spin, and still gives serial execution's bits.
TEST(TeamExecute, OversubscribedTeam) {
  fftn::detail::build_settings settings = fftn::test::eager_settings();
  settings.profile.cpus = 2;
  std::vector<team_case> cases;
  fftn::test::add_cases<double>(cases, settings, "");
  for (const std::size_t i : {std::size_t{0}, std::size_t{8}, std::size_t{11}}) {
    const team_case& c = cases[i];
    const std::vector<std::byte> serial = fftn::test::serial_result(c);
    fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, 8));
    for (const int size : {3, 8}) {
      EXPECT_EQ(run_team(c, size, 0, ws), serial) << c.name << ", team of " << size;
    }
  }
}

// A workspace size that does not fit saturates, so that allocating it fails.
TEST(TeamExecute, WorkspaceBytesSaturate) {
  const auto core = fftn::detail::make_plan_core(
      fftn::detail::view_of(fftn::rows_c2c<double>(1000, 3, fftn::direction::forward)),
      fftn::options(), fftn::detail::build_settings());
  // The core is not const: the test enlarges its slot as a huge plan's would be.
  auto& mutable_core = const_cast<fftn::detail::plan_core&>(*core);
  mutable_core.slot_bytes = std::numeric_limits<std::size_t>::max() / 3;
  EXPECT_EQ(fftn::detail::workspace_bytes(*core, 4), std::numeric_limits<std::size_t>::max());
  mutable_core.slot_bytes = std::numeric_limits<std::size_t>::max() / 4 - 64;
  EXPECT_EQ(fftn::detail::workspace_bytes(*core, 4), std::numeric_limits<std::size_t>::max());
  mutable_core.slot_bytes = 1024;
  EXPECT_LT(fftn::detail::workspace_bytes(*core, 4), 8192U);
}

} // namespace
