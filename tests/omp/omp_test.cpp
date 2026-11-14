// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The OpenMP adapters (task 3.3, DESIGN.md 13.6), on whichever runtime the compiler brings
// (libgomp with GCC, libomp with Clang): bitwise equality with serial execution for
// fftn::omp::executor at several widths and groupings and for this_team() inside regions of
// several sizes, with both completions, over the team cases; regions of constant width
// whose surplus threads have nothing to do; serial execution inside an active region and at
// width 1; the runtime's settings untouched; no allocation.

#include "support/alloc_hook.hpp"
#include "threading/team_cases.hpp"

#include "fftn/detail/core.hpp"
#include "support/inputs.hpp"

#include <fftn/fftn.hpp>
#include <fftn/omp.hpp>

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using fftn::test::case_buffers;
using fftn::test::team_case;

static_assert(fftn::executor<fftn::omp::executor>);
static_assert(fftn::executor<const fftn::omp::executor>);

// The OpenMP executor seen with another grouping, and counting its calls.
struct grouped {
  fftn::omp::executor ex;
  int group = 0;
  int calls = 0;

  [[nodiscard]] int concurrency() const { return ex.concurrency(); }
  [[nodiscard]] int group_size() const { return group; }
  void bulk(int n, fftn::task_function task, void* arg) {
    ++calls;
    ex.bulk(n, task, arg);
  }
};

class OmpBits : public ::testing::TestWithParam<std::size_t> {
public:
  static const std::vector<team_case>& cases() {
    static const std::vector<team_case> all =
        fftn::test::standard_cases(fftn::test::eager_settings());
    return all;
  }
};

TEST_P(OmpBits, ExecutorEqualsSerial) {
  const team_case& c = cases()[GetParam()];
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  const fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, 8));
  for (const int width : {1, 2, 3, 4, 8}) {
    for (const int group : {0, 2, 3}) {
      grouped ex{.ex = fftn::omp::executor(width), .group = group};
      case_buffers b(c);
      fftn::detail::execute_on(*c.core, b.src(), b.dst(), ex, ws);
      EXPECT_EQ(b.result(), serial) << c.name << ": width " << width << ", groups of " << group;
      EXPECT_EQ(ex.calls, width == 1 ? 0 : 1) << c.name << ": width " << width;
    }
  }
}

TEST_P(OmpBits, ThisTeamEqualsSerial) {
  const team_case& c = cases()[GetParam()];
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  const fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, 8));
  for (const int threads : {1, 2, 3, 5, 8}) {
    for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
      case_buffers b(c);
      std::atomic<int> wrong{0};
#pragma omp parallel num_threads(threads)
      {
        const fftn::team t = fftn::omp::this_team();
        if (t.rank != omp_get_thread_num() || t.size != omp_get_num_threads()) {
          wrong.fetch_add(1);
        }
        fftn::detail::execute_team(*c.core, b.src(), b.dst(), t, ws, done);
      }
      EXPECT_EQ(wrong.load(), 0);
      EXPECT_EQ(b.result(), serial)
          << c.name << ": " << threads << " threads, completion " << static_cast<int>(done);
    }
  }
}

std::string case_name(const ::testing::TestParamInfo<std::size_t>& param) {
  return OmpBits::cases()[param.param].name;
}

INSTANTIATE_TEST_SUITE_P(Cases, OmpBits, ::testing::Range(std::size_t{0}, OmpBits::cases().size()),
                         case_name);

TEST(OmpTeam, OutsideARegionIsATeamOfOne) {
  const fftn::team t = fftn::omp::this_team();
  EXPECT_EQ(t.rank, 0);
  EXPECT_EQ(t.size, 1);
  EXPECT_EQ(t.group_size, 0);
}

// Every region has the executor's width, and task i runs on thread i mod width, so threads
// beyond the tasks have none; a single task, width 1 and calls inside an active region run on
// the calling thread without a region.
TEST(OmpExecutor, RegionsOfConstantWidth) {
  struct seen {
    std::vector<int> threads;
    std::vector<int> thread;
    std::vector<int> level;
  };
  const auto record = [](void* arg, int i) {
    auto& s = *static_cast<seen*>(arg);
    const auto k = static_cast<std::size_t>(i);
    s.threads[k] = omp_get_num_threads();
    s.thread[k] = omp_get_thread_num();
    s.level[k] = omp_get_level();
  };
  const auto run = [&](const fftn::omp::executor& ex, int n) {
    const auto size = static_cast<std::size_t>(n);
    seen s{
        .threads = std::vector<int>(size, -1),
        .thread = std::vector<int>(size, -1),
        .level = std::vector<int>(size, -1),
    };
    ex.bulk(n, record, &s);
    return s;
  };
  const fftn::omp::executor ex(6);
  EXPECT_EQ(ex.concurrency(), 6);
  for (const int n : {2, 3, 6, 13}) {
    const seen s = run(ex, n);
    for (int i = 0; i < n; ++i) {
      const auto k = static_cast<std::size_t>(i);
      EXPECT_EQ(s.threads[k], 6) << n << " tasks";
      EXPECT_EQ(s.thread[k], i % 6) << n << " tasks";
      EXPECT_EQ(s.level[k], 1) << n << " tasks";
    }
  }
  const seen one = run(ex, 1);
  EXPECT_EQ(one.level[0], 0);
  const seen narrow = run(fftn::omp::executor(1), 3);
  EXPECT_EQ(narrow.level, std::vector<int>(3, 0));
  EXPECT_EQ(fftn::omp::executor(0).concurrency(), 1);
  EXPECT_EQ(run(ex, 0).level.size(), 0U);
  std::atomic<int> wide{0};
  std::vector<int> inner(3, -1);
#pragma omp parallel num_threads(2)
  {
    if (ex.concurrency() != 1) {
      wide.fetch_add(1);
    }
#pragma omp masked
    {
      const seen s = run(ex, 3);
      inner = {s.level[0], s.level[2], s.threads[1]};
    }
  }
  EXPECT_EQ(wide.load(), 0);
  EXPECT_EQ(inner, (std::vector<int>{1, 1, 2})); // on the calling thread, in the outer region
}

// The public overloads with the default machine profile: the executor (a temporary, the
// default width), this_team() with both completions; serial inside an active region; the
// runtime's settings unchanged; no allocation once the runtime's threads exist.
TEST(OmpExecutor, PublicApiEqualsSerial) {
  using cd = std::complex<double>;
  const int max_threads = omp_get_max_threads();
  const int dynamic = omp_get_dynamic();
  const int levels = omp_get_max_active_levels();
  const int lanes = std::max(max_threads, 8);
  for (const auto& [n, count] :
       {std::pair<std::size_t, std::size_t>{16384, 1}, {8192, 9}, {384, 300}, {64, 2}}) {
    const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
    fftn::test::input_source src(n * 3 + count);
    const std::vector<cd> in = src.complexes<double>(n * count);
    std::vector<cd> serial(n * count);
    fftn::workspace ws(plan, lanes);
    plan.execute(in.data(), serial.data(), ws);
    const auto same = [&](const std::vector<cd>& out) {
      return std::ranges::equal(std::as_bytes(std::span(out)), std::as_bytes(std::span(serial)));
    };
    std::vector<cd> out(n * count);
    plan.execute(in.data(), out.data(), fftn::omp::executor{}, ws);
    EXPECT_TRUE(same(out)) << plan.describe() << "default executor";

    std::ranges::fill(out, cd{});
    const fftn::omp::executor four(4);
    {
      const fftn::test::allocation_counter counter;
      plan.execute(in.data(), out.data(), four, ws);
      EXPECT_EQ(counter.allocations(), 0U) << plan.describe();
    }
    EXPECT_TRUE(same(out)) << plan.describe() << "4 lanes";

    for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
      std::ranges::fill(out, cd{});
#pragma omp parallel num_threads(4)
      plan.execute(in.data(), out.data(), fftn::omp::this_team(), ws, done);
      EXPECT_TRUE(same(out)) << plan.describe() << "this_team, completion "
                             << static_cast<int>(done);
    }

    // Inside an active region the executor runs each thread's execution serially.
    std::vector<std::vector<cd>> outs(3, std::vector<cd>(n * count));
    std::vector<fftn::workspace> spaces;
    spaces.reserve(3);
    for (int t = 0; t < 3; ++t) {
      spaces.emplace_back(plan, lanes);
    }
#pragma omp parallel num_threads(3)
    {
      const auto t = static_cast<std::size_t>(omp_get_thread_num());
      plan.execute(in.data(), outs[t].data(), four, spaces[t]);
    }
    for (const auto& o : outs) {
      EXPECT_TRUE(same(o)) << plan.describe() << "nested";
    }
  }
  EXPECT_EQ(omp_get_max_threads(), max_threads);
  EXPECT_EQ(omp_get_dynamic(), dynamic);
  EXPECT_EQ(omp_get_max_active_levels(), levels);
}

} // namespace
