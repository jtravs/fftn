// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The executor overload of plan::execute (task 3.3, DESIGN.md 13.6): bitwise equality with
// serial execution on every kind of executor (in order, in reverse, in random order on a few
// threads, FFTN's thread pool seen at any width and grouping, a pool described by the C
// contract) over the team cases; what FFTN asks of an executor (nothing at width 1 or for a
// job too small to share, one task per lane that has work); an executor that throws;
// nesting; no allocation. The tsan gate stage runs these under ThreadSanitizer.

#include "support/alloc_hook.hpp"
#include "threading/executors.hpp"
#include "threading/team_cases.hpp"

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "support/inputs.hpp"
#include "threading/schedule.hpp"
#include "threading/team_state.hpp"

#include <fftn/executor.h>
#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using fftn::test::case_buffers;
using fftn::test::team_case;

static_assert(fftn::executor<fftn::thread_pool>);
static_assert(fftn::executor<const fftn::thread_pool>);
static_assert(fftn::executor<fftn::executor_ref>);
static_assert(fftn::executor<fftn::test::sequential_executor>);
static_assert(!fftn::executor<const fftn::test::sequential_executor>); // bulk is not const
static_assert(!fftn::executor<fftn::team>);
static_assert(!fftn::executor<int>);
static_assert(std::is_trivially_copyable_v<fftn::executor_ref>);

const fftn::thread_pool& pool8() {
  static const fftn::thread_pool pool(8);
  return pool;
}

std::vector<std::byte> run_on(const team_case& c, fftn::executor_ref ex,
                              const fftn::workspace& ws) {
  case_buffers b(c);
  fftn::detail::execute_on(*c.core, b.src(), b.dst(), ex, ws);
  return b.result();
}

// A sequential pool in the C contract's form: what a pool written in C provides.
struct c_pool {
  int width;
  int group;
  int calls;
};
int c_concurrency(void* self) { return static_cast<c_pool*>(self)->width; }
int c_group_size(void* self) { return static_cast<c_pool*>(self)->group; }
void c_bulk(void* self, int n, fftn_task task, void* arg) {
  ++static_cast<c_pool*>(self)->calls;
  for (int i = 0; i < n; ++i) {
    task(arg, i);
  }
}

class ExecutorBits : public ::testing::TestWithParam<std::size_t> {
public:
  static const std::vector<team_case>& cases() {
    static const std::vector<team_case> all =
        fftn::test::standard_cases(fftn::test::eager_settings());
    return all;
  }
};

TEST_P(ExecutorBits, EqualToSerialOnEveryExecutor) {
  const team_case& c = cases()[GetParam()];
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  constexpr int widest = 16;
  const fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, widest));
  fftn::test::random_order_executor random(3, GetParam() * 31 + 7);
  for (const int width : {1, 2, 3, 5, 8, widest}) {
    for (const int group : {0, 2, 3}) {
      const std::string where =
          c.name + ": width " + std::to_string(width) + ", groups of " + std::to_string(group);
      fftn::test::sequential_executor forward{.width = width, .group = group};
      EXPECT_EQ(run_on(c, forward, ws), serial) << where << ", in order";
      fftn::test::sequential_executor backward{.width = width, .group = group, .reverse = true};
      EXPECT_EQ(run_on(c, backward, ws), serial) << where << ", in reverse";
      random.width = width;
      random.group = group;
      EXPECT_EQ(run_on(c, random, ws), serial) << where << ", random order on 3 threads";
      const fftn::test::pool_view pool{.pool = &pool8(), .width = width, .group = group};
      EXPECT_EQ(run_on(c, pool, ws), serial) << where << ", on a pool of 8";
      c_pool cp{.width = width, .group = group, .calls = 0};
      const fftn_executor contract{
          .self = &cp,
          .concurrency = c_concurrency,
          .bulk = c_bulk,
          .group_size = c_group_size,
      };
      EXPECT_EQ(run_on(c, fftn::executor_ref(contract), ws), serial) << where << ", C contract";
    }
  }
}

std::string case_name(const ::testing::TestParamInfo<std::size_t>& param) {
  return ExecutorBits::cases()[param.param].name;
}

INSTANTIATE_TEST_SUITE_P(Cases, ExecutorBits,
                         ::testing::Range(std::size_t{0}, ExecutorBits::cases().size()), case_name);

// The public overload with the default machine profile and chunking, on thread pools of
// several sizes, the C contract without a grouping function, and a temporary executor.
TEST(ExecutorExecute, PublicApiEqualsSerial) {
  using cd = std::complex<double>;
  const fftn::thread_pool pool3(3);
  for (const auto& [n, count] :
       {std::pair<std::size_t, std::size_t>{16384, 1}, {8192, 9}, {384, 300}, {64, 2}}) {
    const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
    fftn::test::input_source src(n + count);
    const std::vector<cd> in = src.complexes<double>(n * count);
    std::vector<cd> serial(n * count);
    fftn::workspace ws(plan, 8);
    plan.execute(in.data(), serial.data(), ws);
    const auto same = [&](const std::vector<cd>& out) {
      return std::ranges::equal(std::as_bytes(std::span(out)), std::as_bytes(std::span(serial)));
    };
    std::vector<cd> out(n * count);
    plan.execute(in.data(), out.data(), pool8(), ws);
    EXPECT_TRUE(same(out)) << plan.describe() << "pool of 8";
    std::ranges::fill(out, cd{});
    plan.execute(in.data(), out.data(), pool3, ws);
    EXPECT_TRUE(same(out)) << plan.describe() << "pool of 3";
    std::ranges::fill(out, cd{});
    c_pool cp{.width = 8, .group = 0, .calls = 0};
    const fftn_executor contract{
        .self = &cp,
        .concurrency = c_concurrency,
        .bulk = c_bulk,
        .group_size = nullptr,
    };
    plan.execute(in.data(), out.data(), fftn::executor_ref(contract), ws);
    EXPECT_TRUE(same(out)) << plan.describe() << "C contract";
    std::ranges::fill(out, cd{});
    plan.execute(in.data(), out.data(), fftn::test::sequential_executor{.width = 4, .group = 2},
                 ws);
    EXPECT_TRUE(same(out)) << plan.describe() << "temporary executor";
  }
}

// Width 1 opens nothing; a job too small to share runs inline without calling the executor;
// otherwise exactly the lanes that have work are dispatched, one task each (rule 4).
TEST(ExecutorExecute, DispatchesOnlyLanesWithWork) {
  const auto& cases = ExecutorBits::cases();
  const team_case& three = cases[2]; // 3 transforms of 1024
  const team_case& many = cases[13]; // 100 transforms of 64
  ASSERT_EQ(fftn::detail::transform_count(*three.core), 3U);
  ASSERT_EQ(fftn::detail::transform_count(*many.core), 100U);
  fftn::workspace ws(fftn::detail::workspace_bytes(*many.core, 16) +
                     fftn::detail::workspace_bytes(*three.core, 16));

  fftn::test::sequential_executor one{.width = 1};
  EXPECT_EQ(run_on(many, one, ws), fftn::test::serial_result(many));
  EXPECT_EQ(one.calls, 0);

  const auto expect_lanes = [&](const team_case& c, int width, int group, int lanes) {
    fftn::test::sequential_executor ex{.width = width, .group = group};
    EXPECT_EQ(run_on(c, ex, ws), fftn::test::serial_result(c));
    const fftn::detail::schedule s = fftn::detail::make_schedule(
        c.core->graph, {.lanes = width, .group_size = group}, c.core->profile);
    EXPECT_EQ(ex.calls, 1) << c.name << ", width " << width << ", groups of " << group;
    EXPECT_EQ(ex.last_n, s.lanes_dealt) << c.name;
    EXPECT_EQ(ex.last_n, lanes) << c.name << ", width " << width << ", groups of " << group;
  };
  expect_lanes(three, 16, 4, 4);  // three transforms: one group of 4 shares them
  expect_lanes(three, 16, 0, 16); // one group of 16: the transforms are split among all
  expect_lanes(many, 16, 4, 16);  // every group has work
  expect_lanes(many, 6, 4, 6);    // groups of 4 and 2

  // The default profile runs a short job inline.
  const fftn::c2c_plan<double> tiny(fftn::rows_c2c<double>(16, 2, fftn::direction::forward));
  fftn::test::input_source src(5);
  const auto in = src.complexes<double>(32);
  std::vector<std::complex<double>> out(32);
  std::vector<std::complex<double>> serial(32);
  fftn::workspace tiny_ws(tiny, 16);
  tiny.execute(in.data(), serial.data(), tiny_ws);
  fftn::test::sequential_executor wide{.width = 16};
  tiny.execute(in.data(), out.data(), wide, tiny_ws);
  EXPECT_EQ(wide.calls, 0);
  EXPECT_EQ(out, serial);
}

// On an executor of more threads than the profile's CPUs, the dispatched members (fewer than
// the CPUs) judge oversubscription by the executor's width; the result is the same.
TEST(ExecutorExecute, OversubscribedExecutorEqualsSerial) {
  fftn::detail::build_settings settings = fftn::test::eager_settings();
  settings.profile.cpus = 4;
  std::vector<team_case> cases;
  fftn::test::add_cases<double>(cases, settings, "");
  for (const std::size_t i : {std::size_t{0}, std::size_t{2}, std::size_t{8}}) {
    const team_case& c = cases[i];
    const fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, 16));
    const fftn::test::pool_view pool{.pool = &pool8(), .width = 16, .group = 4};
    EXPECT_EQ(run_on(c, pool, ws), fftn::test::serial_result(c)) << c.name;
  }
}

// An executor that throws part way: the exception reaches the caller, and the workspace's
// team state is dropped, so members that did not run cannot count the next execution wrong.
TEST(ExecutorExecute, ThrowingExecutorLeavesTheWorkspaceUsable) {
  const team_case& c = ExecutorBits::cases()[13];
  const std::vector<std::byte> serial = fftn::test::serial_result(c);
  const fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, 8));
  fftn::test::random_order_executor random(4, 99);
  random.width = 8;
  random.group = 2;
  EXPECT_EQ(run_on(c, random, ws), serial);
  EXPECT_NE(fftn::detail::workspace_access::team_word(ws).load(), 0U);
  const fftn::test::throwing_executor failing{.width = 8, .ran = 3};
  case_buffers b(c);
  EXPECT_THROW(fftn::detail::execute_on(*c.core, b.src(), b.dst(), failing, ws),
               std::runtime_error);
  EXPECT_EQ(fftn::detail::workspace_access::team_word(ws).load(), 0U);
  for (int k = 0; k < 5; ++k) {
    EXPECT_EQ(run_on(c, random, ws), serial) << "execution " << k << " after the failure";
  }
}

// A call from inside one of a pool's tasks sees concurrency 1 and runs serially, with the
// right result; the outer call is unaffected.
TEST(ExecutorExecute, NestedCallsRunSerially) {
  const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(2048, 4, fftn::direction::backward));
  fftn::test::input_source src(17);
  const auto in = src.complexes<double>(std::size_t{2048} * 4);
  std::vector<std::complex<double>> serial(in.size());
  fftn::workspace serial_ws(plan);
  plan.execute(in.data(), serial.data(), serial_ws);
  constexpr int outer = 6;
  struct context {
    const fftn::c2c_plan<double>* plan;
    const std::vector<std::complex<double>>* in;
    std::vector<std::vector<std::complex<double>>> out;
    std::vector<fftn::workspace> ws;
    std::vector<int> seen;
  } ctx{.plan = &plan, .in = &in, .out = {}, .ws = {}, .seen = std::vector<int>(outer, 0)};
  for (int k = 0; k < outer; ++k) {
    ctx.out.emplace_back(in.size());
    ctx.ws.emplace_back(plan, pool8().concurrency());
  }
  pool8().bulk(
      outer,
      [](void* arg, int k) {
        auto& x = *static_cast<context*>(arg);
        const auto i = static_cast<std::size_t>(k);
        x.seen[i] = pool8().concurrency();
        x.plan->execute(x.in->data(), x.out[i].data(), pool8(), x.ws[i]);
      },
      &ctx);
  for (int k = 0; k < outer; ++k) {
    EXPECT_EQ(ctx.seen[static_cast<std::size_t>(k)], 1) << "task " << k;
    EXPECT_EQ(ctx.out[static_cast<std::size_t>(k)], serial) << "task " << k;
  }
  EXPECT_EQ(pool8().concurrency(), 8);
}

TEST(ExecutorExecute, NeverAllocates) {
  const auto& cases = ExecutorBits::cases();
  for (const std::size_t i : {std::size_t{0}, std::size_t{8}, std::size_t{13}}) {
    const team_case& c = cases[i];
    const fftn::workspace ws(fftn::detail::workspace_bytes(*c.core, 8));
    case_buffers b(c);
    fftn::test::sequential_executor seq{.width = 8, .group = 3};
    const fftn::test::pool_view pool{.pool = &pool8(), .width = 8, .group = 4};
    // Once each first: nothing is allocated on first use either, but the pool's threads are
    // started by the first call of pool8().
    fftn::detail::execute_on(*c.core, b.src(), b.dst(), pool, ws);
    const fftn::test::allocation_counter counter;
    fftn::detail::execute_on(*c.core, b.src(), b.dst(), seq, ws);
    fftn::detail::execute_on(*c.core, b.src(), b.dst(), pool, ws);
    fftn::detail::execute_on(*c.core, b.src(), b.dst(), pool8(), ws);
    EXPECT_EQ(counter.allocations(), 0U) << c.name;
    EXPECT_EQ(counter.deallocations(), 0U) << c.name;
  }
}

} // namespace
