// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The threaded side of the Kokkos adapter (task 3.5), on Kokkos' Serial back end and on the
// build's threaded one (OpenMP or Threads, cmake/FftnTestDeps.cmake): the executor, team_of and
// view_plan give serial execution's bits; one parallel_for per transform; no allocation; a
// call from inside a kernel, or from a thread that may not dispatch, runs serially.

#include "kokkos/kokkos_helpers.hpp"
#include "support/alloc_hook.hpp"

#include <fftn/fftn.hpp>
#include <fftn/kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using namespace fftn::test::kokkos_adapter;
using fftn::kokkos::axis_type;
using fftn::kokkos::data_of;
using fftn::kokkos::make_plan;

template <class Space, class T> struct config {
  using space = Space;
  using real = T;
};

#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
using configs = ::testing::Types<config<Kokkos::Serial, double>, config<Kokkos::Serial, float>,
                                 config<host, double>, config<host, float>>;
#else
using configs = ::testing::Types<config<Kokkos::Serial, double>, config<Kokkos::Serial, float>>;
#endif

template <class C> class KokkosThreaded : public ::testing::Test {};
TYPED_TEST_SUITE(KokkosThreaded, configs);

template <class T, class Layout = Kokkos::LayoutRight>
using complex3 = Kokkos::View<Kokkos::complex<T>***, Layout, Kokkos::HostSpace>;
template <class T, class Layout = Kokkos::LayoutRight>
using complex4 = Kokkos::View<Kokkos::complex<T>****, Layout, Kokkos::HostSpace>;

// Sizes that FFTN shares among threads rather than running inline: 2^18 points.
constexpr std::size_t big_rows = 64;
constexpr std::size_t big_n = 4096;

// Runs p serially and through the executor on `exec`, into two outputs, and compares the bits.
template <class Space, class In, class Out, std::size_t Dim>
void check_executor(const Space& exec, const In& x, const Out& y, const Out& z, direction dir,
                    const axis_type<Dim>& axes) {
  const auto p = make_plan(x, y, dir, axes);
  fftn::workspace serial(p);
  p.execute(data_of(x), data_of(y), serial);
  const fftn::kokkos::executor<Space> ex(exec);
  fftn::workspace ws(p.workspace_bytes(ex.width()));
  p.execute(data_of(x), data_of(z), ex, ws);
  expect_same_bits(reals_of(z), reals_of(y));
}

TYPED_TEST(KokkosThreaded, ExecutorGivesSerialBitsForEveryKind) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  const space exec;
  {
    const complex2<T> x("x", big_rows, big_n);
    const complex2<T> y("y", big_rows, big_n);
    const complex2<T> z("z", big_rows, big_n);
    fill(x, 11);
    check_executor(exec, x, y, z, direction::forward, axis_type<1>{-1});
    check_executor(exec, x, y, z, direction::backward, axis_type<1>{-1});
  }
  {
    const real2<T> x("x", big_rows, big_n);
    const complex2<T> y("y", big_rows, big_n / 2 + 1);
    const complex2<T> z("z", big_rows, big_n / 2 + 1);
    const real2<T> back("back", big_rows, big_n);
    const real2<T> back2("back2", big_rows, big_n);
    fill(x, 12);
    check_executor(exec, x, y, z, direction::forward, axis_type<1>{-1});
    check_executor(exec, y, back, back2, direction::backward, axis_type<1>{-1});
  }
  {
    // LayoutLeft: the batch dimension is the slow one in memory order too, reversed.
    const complex2<T, Kokkos::LayoutLeft> x("x", big_n, big_rows);
    const complex2<T, Kokkos::LayoutLeft> y("y", big_n, big_rows);
    const complex2<T, Kokkos::LayoutLeft> z("z", big_n, big_rows);
    fill(x, 13);
    check_executor(exec, x, y, z, direction::forward, axis_type<1>{0});
  }
  {
    const complex3<T> x("x", 8, 256, 128);
    const complex3<T> y("y", 8, 256, 128);
    const complex3<T> z("z", 8, 256, 128);
    fill(x, 14);
    check_executor(exec, x, y, z, direction::forward, axis_type<2>{-2, -1});
  }
  {
    const complex4<T> x("x", 8, 32, 32, 32);
    const complex4<T> y("y", 8, 32, 32, 32);
    const complex4<T> z("z", 8, 32, 32, 32);
    fill(x, 15);
    check_executor(exec, x, y, z, direction::forward, axis_type<3>{-3, -2, -1});
  }
}

TYPED_TEST(KokkosThreaded, ExecutorGivesSerialBitsInPlaceAndThroughStrides) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  const space exec;
  // In place: the serial and the threaded run each start from the same input.
  const complex3<T> v("v", 8, 256, 128);
  const complex3<T> w("w", 8, 256, 128);
  fill(v, 21);
  Kokkos::deep_copy(w, v);
  const auto p = make_plan(v, v, direction::forward, axis_type<2>{-2, -1});
  fftn::workspace serial(p);
  p.execute(data_of(v), data_of(v), serial);
  const fftn::kokkos::executor<space> ex(exec);
  fftn::workspace ws(p.workspace_bytes(ex.width()));
  p.execute(data_of(w), data_of(w), ex, ws);
  expect_same_bits(reals_of(w), reals_of(v));

  // Rows with a gap between them, through LayoutStride.
  using strided = Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutStride, Kokkos::HostSpace>;
  const Kokkos::LayoutStride layout(big_rows, big_n + 5, big_n, 1);
  const strided x("x", layout);
  const strided y("y", layout);
  const strided z("z", layout);
  fill(x, 22);
  check_executor(exec, x, y, z, direction::backward, axis_type<1>{-1});
}

// A view_plan on the instance against one on Serial: the same transform, and the same scaling
// whether or not it is shared among threads.
template <class Space, class InView, class OutView, std::size_t Dim>
void check_view_plan(const InView& x, const OutView& y, const OutView& z, direction dir,
                     const axis_type<Dim>& axes, normalization norm) {
  const fftn::kokkos::view_plan<Kokkos::Serial, InView, OutView, Dim> serial(Kokkos::Serial{}, x, y,
                                                                             dir, axes);
  const fftn::kokkos::view_plan<Space, InView, OutView, Dim> threaded(Space{}, x, z, dir, axes);
  serial.execute(x, y, norm);
  threaded.execute(x, z, norm);
  expect_same_bits(reals_of(z), reals_of(y));
}

TYPED_TEST(KokkosThreaded, ViewPlanGivesSerialBits) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  const complex2<T> x("x", big_rows, big_n);
  const complex2<T> y("y", big_rows, big_n);
  const complex2<T> z("z", big_rows, big_n);
  fill(x, 31);
  for (const normalization norm : {normalization::none, normalization::backward,
                                   normalization::forward, normalization::ortho}) {
    check_view_plan<space>(x, y, z, direction::backward, axis_type<1>{-1}, norm);
  }
  const real2<T> r("r", big_rows, big_n);
  const complex2<T> h("h", big_rows, big_n / 2 + 1);
  const complex2<T> h2("h2", big_rows, big_n / 2 + 1);
  fill(r, 32);
  check_view_plan<space>(r, h, h2, direction::forward, axis_type<1>{-1}, normalization::ortho);
  // Few long rows: scaling cuts the long dimension, not the two rows.
  const complex2<T> a("a", 2, 1U << 17U);
  const complex2<T> b("b", 2, 1U << 17U);
  const complex2<T> c("c", 2, 1U << 17U);
  fill(a, 33);
  check_view_plan<space>(a, b, c, direction::backward, axis_type<1>{-1}, normalization::backward);
}

TYPED_TEST(KokkosThreaded, ViewPlanInPlacePlanesGiveSerialBits) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  using cube = complex3<T, Kokkos::LayoutLeft>;
  const cube v("v", 256, 128, 8);
  const cube w("w", 256, 128, 8);
  fill(v, 41);
  Kokkos::deep_copy(w, v);
  const axis_type<2> axes{0, 1};
  const fftn::kokkos::view_plan<Kokkos::Serial, cube, cube, 2> serial(Kokkos::Serial{}, v, v,
                                                                      direction::backward, axes);
  const fftn::kokkos::view_plan<space, cube, cube, 2> threaded(space{}, w, w, direction::backward,
                                                               axes);
  serial.execute(v, v, normalization::backward);
  threaded.execute(w, w, normalization::backward);
  expect_same_bits(reals_of(w), reals_of(v));
}

// Kokkos Tools: the parallel_for calls the adapter makes.
std::atomic<int> dispatches{0};
void count_dispatch(const char* name, std::uint32_t /*device*/, std::uint64_t* /*handle*/) {
  if (std::strcmp(name, "fftn::bulk") == 0) {
    dispatches.fetch_add(1, std::memory_order_relaxed);
  }
}

// Counts for one scope; a loaded tool also makes Kokkos allocate, so never around a check of
// allocations.
class dispatch_counter {
public:
  dispatch_counter() {
    dispatches.store(0);
    Kokkos::Tools::Experimental::set_begin_parallel_for_callback(count_dispatch);
  }
  dispatch_counter(const dispatch_counter&) = delete;
  dispatch_counter& operator=(const dispatch_counter&) = delete;
  dispatch_counter(dispatch_counter&&) = delete;
  dispatch_counter& operator=(dispatch_counter&&) = delete;
  ~dispatch_counter() { Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr); }
  [[nodiscard]] int count() const { return dispatches.load(); }
};

TYPED_TEST(KokkosThreaded, OneParallelForPerTransform) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  const space exec;
  const bool threaded = exec.concurrency() > 1;
  const complex2<T> x("x", big_rows, big_n);
  const complex2<T> y("y", big_rows, big_n);
  fill(x, 51);
  const fftn::kokkos::view_plan p(exec, x, y, direction::backward, -1);
  {
    const dispatch_counter counter;
    p.execute(x, y, normalization::none);
    EXPECT_EQ(counter.count(), threaded ? 1 : 0);
  }
  {
    // A second one for scaling, whose elements are enough to share.
    const dispatch_counter counter;
    p.execute(x, y, normalization::backward);
    EXPECT_EQ(counter.count(), threaded ? 2 : 0);
  }
  // A job too small to share opens no parallel_for, for the transform or for its scaling.
  const complex2<T> a("a", 2, 16);
  const complex2<T> b("b", 2, 16);
  const fftn::kokkos::view_plan small(exec, a, b, direction::backward, -1);
  const dispatch_counter counter;
  small.execute(a, b);
  EXPECT_EQ(counter.count(), 0);
}

// FFTN allocates nothing while executing. Kokkos may, per dispatch: its Threads back end builds
// a std::string for an internal fence on every parallel_for (OpenMP and Serial allocate
// nothing), so the allowance is what one dispatch of empty tasks costs, times the dispatches.
TYPED_TEST(KokkosThreaded, ExecutionAllocatesOnlyWhatKokkosDoes) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  const space exec;
  const complex2<T> x("x", big_rows, big_n);
  const complex2<T> y("y", big_rows, big_n);
  fill(x, 61);
  const fftn::kokkos::view_plan vp(exec, x, y, direction::backward, -1);
  const auto p = make_plan(x, y, direction::forward, axis_type<1>{-1});
  const fftn::kokkos::executor<space> ex(exec);
  fftn::workspace ws(p.workspace_bytes(ex.width()));
  const auto empty = [](void* /*arg*/, int /*i*/) {};
  ex.bulk(ex.width(), empty, nullptr);
  std::size_t news_per_dispatch = 0;
  std::size_t deletes_per_dispatch = 0;
  {
    const fftn::test::allocation_counter counter;
    ex.bulk(ex.width(), empty, nullptr);
    news_per_dispatch = counter.allocations();
    deletes_per_dispatch = counter.deallocations();
  }
#if !defined(KOKKOS_ENABLE_THREADS)
  EXPECT_EQ(news_per_dispatch, 0U);
#endif
  // Four dispatches when threaded: two transforms and a scaling, then one more transform.
  const std::size_t dispatches_at_most = 4;
  const fftn::test::allocation_counter counter;
  vp.execute(x, y, normalization::none);
  vp.execute(x, y); // scaled, shared among the threads
  p.execute(data_of(x), data_of(y), ex, ws);
  EXPECT_LE(counter.allocations(), dispatches_at_most * news_per_dispatch);
  EXPECT_LE(counter.deallocations(), dispatches_at_most * deletes_per_dispatch);
}

TYPED_TEST(KokkosThreaded, BulkRunsEveryTaskOnce) {
  using space = TypeParam::space;
  const fftn::kokkos::executor<space> ex{space{}};
  EXPECT_EQ(ex.concurrency(), space{}.concurrency());
  static_assert(fftn::executor<fftn::kokkos::executor<space>>);
  struct job {
    std::vector<std::atomic<int>> runs;
    std::vector<std::atomic<std::size_t>> thread;
    explicit job(std::size_t n) : runs(n), thread(n) {}
  };
  const auto task = [](void* arg, int i) {
    auto* j = static_cast<job*>(arg);
    const auto k = static_cast<std::size_t>(i);
    j->runs[k].fetch_add(1);
    j->thread[k].store(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  };
  ex.bulk(0, task, nullptr);
  for (const int n : {1, 2, 3, ex.width(), 2 * ex.width() + 1}) {
    job j(static_cast<std::size_t>(n));
    ex.bulk(n, task, &j);
    std::set<std::size_t> threads;
    for (int i = 0; i < n; ++i) {
      const auto k = static_cast<std::size_t>(i);
      EXPECT_EQ(j.runs[k].load(), 1) << "task " << i << " of " << n;
      threads.insert(j.thread[k].load());
    }
    if (n == ex.width() && n > 1) {
      EXPECT_GT(threads.size(), 1U) << "the tasks of a full-width bulk all ran on one thread";
    }
  }
}

// Inside a kernel the executor reports one lane and runs on the calling thread; each index has
// its own workspace, as such an execution requires.
TYPED_TEST(KokkosThreaded, ExecutorInsideAKernelRunsSerially) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  const space exec;
  const std::size_t count = 8;
  const std::size_t n = 256;
  const complex3<T> x("x", count, 4, n);
  const complex3<T> y("y", count, 4, n);
  const complex3<T> z("z", count, 4, n);
  fill(x, 71);
  const auto first = Kokkos::subview(x, 0, Kokkos::ALL, Kokkos::ALL);
  const auto out = Kokkos::subview(y, 0, Kokkos::ALL, Kokkos::ALL);
  const auto p = make_plan(first, out, direction::forward, axis_type<1>{-1});
  fftn::workspace serial(p);
  for (std::size_t i = 0; i < count; ++i) {
    p.execute(data_of(Kokkos::subview(x, i, Kokkos::ALL, Kokkos::ALL)),
              data_of(Kokkos::subview(y, i, Kokkos::ALL, Kokkos::ALL)), serial);
  }
  const fftn::kokkos::executor<space> ex(exec);
  std::vector<fftn::workspace> spaces;
  for (std::size_t i = 0; i < count; ++i) {
    spaces.emplace_back(p.workspace_bytes());
  }
  std::atomic<int> wider{0};
  using policy = Kokkos::RangePolicy<space, Kokkos::IndexType<int>>;
  Kokkos::parallel_for(
      "fftn test: executor in a kernel", policy(exec, 0, static_cast<int>(count)), [&](int i) {
        const auto k = static_cast<std::size_t>(i);
        if (ex.concurrency() != 1) {
          wider.fetch_add(1);
        }
        p.execute(data_of(Kokkos::subview(x, k, Kokkos::ALL, Kokkos::ALL)),
                  data_of(Kokkos::subview(z, k, Kokkos::ALL, Kokkos::ALL)), ex, spaces[k]);
      });
  exec.fence();
  EXPECT_EQ(wider.load(), 0);
  expect_same_bits(reals_of(z), reals_of(y));
}

// view_plan::execute inside a kernel neither fences (the kernel would wait for itself) nor
// dispatches; each thread uses a plan of its own.
TYPED_TEST(KokkosThreaded, ViewPlanInsideAKernelRunsSerially) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  using rows = complex2<T>;
  using plan_type = fftn::kokkos::view_plan<space, rows, rows>;
  const space exec;
  const std::size_t count = 8;
  std::vector<rows> in;
  std::vector<rows> out;
  std::vector<rows> want;
  std::vector<std::unique_ptr<plan_type>> plans;
  for (std::size_t i = 0; i < count; ++i) {
    in.emplace_back("x", 4, 256);
    out.emplace_back("y", 4, 256);
    want.emplace_back("w", 4, 256);
    fill(in[i], static_cast<std::uint32_t>(80 + i));
    plans.push_back(std::make_unique<plan_type>(exec, in[i], out[i], direction::backward, -1));
    plans[i]->execute(in[i], want[i]);
  }
  using policy = Kokkos::RangePolicy<space, Kokkos::IndexType<int>>;
  Kokkos::parallel_for("fftn test: view_plan in a kernel", policy(exec, 0, static_cast<int>(count)),
                       [&](int i) {
                         const auto k = static_cast<std::size_t>(i);
                         plans[k]->execute(in[k], out[k]);
                       });
  exec.fence();
  for (std::size_t i = 0; i < count; ++i) {
    expect_same_bits(reals_of(out[i]), reals_of(want[i]));
  }
}

// The members of a TeamPolicy kernel as an FFTN team: completion::all, then completion::own
// with Kokkos' team_barrier as the synchronisation it asks of the caller.
TYPED_TEST(KokkosThreaded, TeamOfGivesSerialBits) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  using team_policy = Kokkos::TeamPolicy<space>;
  using member = team_policy::member_type;
  const space exec;
  const complex2<T> x("x", big_rows, big_n);
  const complex2<T> y("y", big_rows, big_n);
  const complex2<T> z("z", big_rows, big_n);
  fill(x, 91);
  const auto p = make_plan(x, y, direction::forward, axis_type<1>{-1});
  fftn::workspace serial(p);
  p.execute(data_of(x), data_of(y), serial);
  const int width = exec.concurrency();
  fftn::workspace ws(p.workspace_bytes(width));
  std::atomic<int> wrong{0};
  Kokkos::parallel_for(
      "fftn test: team, completion all", team_policy(exec, 1, width), [&](const member& m) {
        const fftn::team t = fftn::kokkos::team_of(m);
        if (t.rank != m.team_rank() || t.size != m.team_size() || t.size != width) {
          wrong.fetch_add(1);
        }
        p.execute(data_of(x), data_of(z), t, ws);
      });
  exec.fence();
  EXPECT_EQ(wrong.load(), 0);
  expect_same_bits(reals_of(z), reals_of(y));

  Kokkos::deep_copy(z, Kokkos::complex<T>(0));
  Kokkos::parallel_for(
      "fftn test: team, completion own", team_policy(exec, 1, width), [&](const member& m) {
        p.execute(data_of(x), data_of(z), fftn::kokkos::team_of(m), ws, fftn::completion::own);
        m.team_barrier();
      });
  exec.fence();
  expect_same_bits(reals_of(z), reals_of(y));
}

// Several teams at once: a workspace per league rank.
TYPED_TEST(KokkosThreaded, EachTeamOfALeagueHasItsOwnWorkspace) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  using team_policy = Kokkos::TeamPolicy<space>;
  using member = team_policy::member_type;
  const space exec;
  const int team_size = std::max(exec.concurrency() / 2, 1);
  const int league = 2;
  const complex2<T> x("x", big_rows, big_n);
  const complex2<T> y("y", big_rows, big_n);
  std::array<complex2<T>, 2> z{complex2<T>("z0", big_rows, big_n),
                               complex2<T>("z1", big_rows, big_n)};
  fill(x, 92);
  const auto p = make_plan(x, y, direction::backward, axis_type<1>{-1});
  fftn::workspace serial(p);
  p.execute(data_of(x), data_of(y), serial);
  std::vector<fftn::workspace> spaces;
  for (int l = 0; l < league; ++l) {
    spaces.emplace_back(p.workspace_bytes(team_size));
  }
  Kokkos::parallel_for("fftn test: a league of teams", team_policy(exec, league, team_size),
                       [&](const member& m) {
                         const auto l = static_cast<std::size_t>(m.league_rank());
                         p.execute(data_of(x), data_of(z[l]), fftn::kokkos::team_of(m), spaces[l]);
                       });
  exec.fence();
  expect_same_bits(reals_of(z[0]), reals_of(y));
  expect_same_bits(reals_of(z[1]), reals_of(y));
}

// Host threads other than the one that started Kokkos: each with its own plan, at the same
// time. Threads lets only its first thread dispatch, so there they run serially; OpenMP takes
// their dispatches in turn.
TYPED_TEST(KokkosThreaded, OtherHostThreadsGetCorrectResults) {
  using space = TypeParam::space;
  using T = TypeParam::real;
  const space exec;
  const complex2<T> x("x", big_rows, big_n);
  const complex2<T> want("want", big_rows, big_n);
  fill(x, 95);
  const fftn::kokkos::view_plan<Kokkos::Serial, complex2<T>, complex2<T>> serial(
      Kokkos::Serial{}, x, want, direction::forward, -1);
  serial.execute(x, want, normalization::none);
  const std::size_t threads = 3;
  std::vector<complex2<T>> out;
  std::vector<std::unique_ptr<fftn::kokkos::view_plan<space, complex2<T>, complex2<T>>>> plans;
  for (std::size_t t = 0; t < threads; ++t) {
    out.emplace_back("y", big_rows, big_n);
    plans.push_back(std::make_unique<fftn::kokkos::view_plan<space, complex2<T>, complex2<T>>>(
        exec, x, out[t], direction::forward, -1));
  }
  std::vector<std::thread> team;
  team.reserve(threads);
  for (std::size_t t = 0; t < threads; ++t) {
    team.emplace_back([&, t] { plans[t]->execute(x, out[t], normalization::none); });
  }
  for (std::thread& t : team) {
    t.join();
  }
  for (std::size_t t = 0; t < threads; ++t) {
    expect_same_bits(reals_of(out[t]), reals_of(want));
  }
}

#ifdef KOKKOS_ENABLE_OPENMP
// A narrower instance than the default one is used at its own width.
TYPED_TEST(KokkosThreaded, NarrowOpenMpInstances) {
  using T = TypeParam::real;
  if (!std::is_same_v<typename TypeParam::space, Kokkos::OpenMP>) {
    GTEST_SKIP() << "OpenMP only";
  }
  const complex2<T> x("x", big_rows, big_n);
  const complex2<T> y("y", big_rows, big_n);
  const complex2<T> z("z", big_rows, big_n);
  fill(x, 96);
  const auto p = make_plan(x, y, direction::forward, axis_type<1>{-1});
  fftn::workspace serial(p);
  p.execute(data_of(x), data_of(y), serial);
  for (const int width : {1, 2, 3}) {
    if (width > Kokkos::OpenMP().concurrency()) {
      continue;
    }
    const Kokkos::OpenMP narrow(width);
    const fftn::kokkos::executor<Kokkos::OpenMP> ex(narrow);
    EXPECT_EQ(ex.concurrency(), width);
    fftn::workspace ws(p.workspace_bytes(ex.width()));
    p.execute(data_of(x), data_of(z), ex, ws);
    expect_same_bits(reals_of(z), reals_of(y));
  }
}
#endif

} // namespace
