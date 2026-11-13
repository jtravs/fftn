// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The contracts of plan, workspace and shape (DESIGN.md 15.3, AGENTS.md section 5): no
// allocation in execute, any alignment, bitwise independence of a transform's position in
// the batch (execute_rows with any first included), plan reuse across copies and threads,
// caller-supplied workspaces, and std::invalid_argument naming the argument for every
// invalid shape. Memory outside the rows (canaries) and input preservation are checked by
// the harness on every run of the oracle suites; they are exercised here too through it.

#include "support/alloc_hook.hpp"
#include "support/fftn_subject.hpp"
#include "support/harness.hpp"
#include "support/inputs.hpp"
#include "support/transform_under_test.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using fftn::test::input_source;
using fftn::test::placement;
using fftn::test::rows;
template <class T> using cvec = std::vector<std::complex<T>>;

// Byte-wise equality through std::as_bytes: no raw memory function on std::complex,
// which GCC's -Wclass-memaccess refuses.
template <class T> bool same_bits(std::span<const T> a, std::span<const T> b) {
  return a.size() == b.size() && std::ranges::equal(std::as_bytes(a), std::as_bytes(b));
}

// n random elements of type E (real or complex).
template <class E> std::vector<E> random_elements(input_source& src, std::size_t n) {
  if constexpr (std::is_floating_point_v<E>) {
    return src.reals<E>(n);
  } else {
    return src.complexes<typename E::value_type>(n);
  }
}

template <class T> class Contracts : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(Contracts, precisions);

// ---- No allocation, no throw, in execute.

template <class In, class Out>
void expect_no_allocation(const fftn::shape<In, Out>& s, std::size_t in_count,
                          std::size_t out_count, bool in_place = false) {
  const fftn::plan<In, Out> p(s, {.in_place = in_place});
  fftn::workspace ws(p);
  std::vector<std::complex<double>> storage(in_count + out_count + 2);
  auto* in = reinterpret_cast<In*>(storage.data());
  auto* out =
      in_place ? reinterpret_cast<Out*>(in) : reinterpret_cast<Out*>(storage.data() + in_count + 1);
  const fftn::test::allocation_counter counter;
  p.execute(in, out, ws);
  p.execute_rows(0, 1, in, out, ws);
  EXPECT_EQ(counter.allocations(), 0U) << p.describe();
  EXPECT_EQ(counter.deallocations(), 0U);
}

TYPED_TEST(Contracts, ExecuteNeverAllocates) {
  using T = TypeParam;
  using c = std::complex<T>;
  expect_no_allocation(fftn::rows_c2c<T>(360, 3, fftn::direction::forward), 3 * 360, 3 * 360);
  expect_no_allocation(fftn::rows_c2c<T>(97, 2, fftn::direction::backward), 2 * 97, 2 * 97, true);
  expect_no_allocation(fftn::rows_r2c<T>(1000, 2), 1000, 2 * 501);
  expect_no_allocation(fftn::rows_c2r<T>(999, 2), 2 * 500, 999);
  expect_no_allocation(fftn::planes_c2c<T>(12, 10, 2, fftn::direction::forward), 240, 240, true);
  fftn::shape<c, T> c2r_planes;
  c2r_planes.axes = {{.n = 6}, {.n = 8}};
  expect_no_allocation(c2r_planes, 6 * 5, 6 * 8); // goes through the block
  expect_no_allocation(fftn::rows_c2c<T>(1, 1, fftn::direction::forward), 1, 1, true);
}

// ---- Any alignment: the same bits at every offset of the arrays.

template <class T> void check_alignment(const fftn::test::subject<T>& s) {
  constexpr std::size_t n = 60;
  constexpr std::size_t count = 3;
  input_source src(5);
  const auto x = src.complexes<T>(n * count);
  const auto xr = src.reals<T>(n * count);
  const auto tight = [](fftn::test::kind k) { return fftn::test::tight_rows(k, n, count); };
  const auto c2c0 =
      fftn::test::run_c2c<T>(s, fftn::test::direction::forward, tight(fftn::test::kind::c2c), x);
  const auto r2c0 = fftn::test::run_r2c<T>(s, tight(fftn::test::kind::r2c), xr);
  const cvec<T> half(x.begin(), x.begin() + static_cast<std::ptrdiff_t>((n / 2 + 1) * count));
  const auto c2r0 = fftn::test::run_c2r<T>(s, tight(fftn::test::kind::c2r), half);
  for (const std::size_t in_off :
       {std::size_t{0}, sizeof(T), 2 * sizeof(T), std::size_t{24}, std::size_t{40}}) {
    for (const std::size_t out_off : {std::size_t{0}, sizeof(T), 3 * sizeof(T), std::size_t{56}}) {
      const fftn::test::run_options o{.in_offset = in_off, .out_offset = out_off};
      const auto a = fftn::test::run_c2c<T>(s, fftn::test::direction::forward,
                                            tight(fftn::test::kind::c2c), x, o);
      const auto b = fftn::test::run_r2c<T>(s, tight(fftn::test::kind::r2c), xr, o);
      const auto c = fftn::test::run_c2r<T>(s, tight(fftn::test::kind::c2r), half, o);
      EXPECT_TRUE(a.violations.empty() && b.violations.empty() && c.violations.empty());
      EXPECT_TRUE(same_bits<std::complex<T>>(a.out, c2c0.out)) << in_off << " " << out_off;
      EXPECT_TRUE(same_bits<std::complex<T>>(b.out, r2c0.out)) << in_off << " " << out_off;
      EXPECT_TRUE(same_bits<T>(c.out, c2r0.out)) << in_off << " " << out_off;
    }
  }
}

TYPED_TEST(Contracts, AnyAlignmentGivesTheSameBits) {
  check_alignment<TypeParam>(fftn::test::fftn_subject<TypeParam>());
}

// ---- A transform's result does not depend on its position in the batch.

// Runs every transform of `p` once through execute and once through execute_rows from
// several starting points, on a batch whose transforms all hold the same input; every
// transform's output must equal transform 0's bit for bit (DESIGN.md 15.5).
template <class In, class Out>
void check_positions(const fftn::plan<In, Out>& p, std::size_t transforms, std::size_t in_len,
                     std::size_t out_len, std::size_t in_dist, std::size_t out_dist) {
  fftn::workspace ws(p);
  input_source src(in_len);
  const std::vector<In> one = random_elements<In>(src, in_len);
  std::vector<In> in(in_dist * transforms);
  for (std::size_t t = 0; t < transforms; ++t) {
    std::ranges::copy(one, in.begin() + static_cast<std::ptrdiff_t>(t * in_dist));
  }
  std::vector<Out> out(out_dist * transforms);
  p.execute(in.data(), out.data(), ws);
  const std::span<const Out> first(out.data(), out_len);
  for (std::size_t t = 1; t < transforms; ++t) {
    EXPECT_TRUE(same_bits<Out>(std::span<const Out>(&out[t * out_dist], out_len), first))
        << "transform " << t << " differs from transform 0";
  }
  for (const std::size_t start : {std::size_t{1}, transforms / 2, transforms - 1}) {
    std::vector<Out> part(out.size());
    p.execute_rows(start, transforms - start, in.data(), part.data(), ws);
    for (std::size_t t = start; t < transforms; ++t) {
      EXPECT_TRUE(same_bits<Out>(std::span<const Out>(&part[t * out_dist], out_len), first))
          << "execute_rows from " << start << ", transform " << t;
    }
    // The transforms before `start` are not touched.
    const std::vector<Out> zero(start * out_dist);
    EXPECT_TRUE(same_bits<Out>(std::span<const Out>(part.data(), zero.size()), zero));
  }
}

TYPED_TEST(Contracts, ResultDoesNotDependOnPosition) {
  using T = TypeParam;
  using c = std::complex<T>;
  for (const std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{60}, std::size_t{1001}}) {
    check_positions(fftn::c2c_plan<T>(fftn::rows_c2c<T>(n, 37, fftn::direction::forward)), 37, n, n,
                    n, n);
    check_positions(fftn::r2c_plan<T>(fftn::rows_r2c<T>(n, 37)), 37, n, n / 2 + 1, n, n / 2 + 1);
    check_positions(fftn::c2r_plan<T>(fftn::rows_c2r<T>(n, 37)), 37, n / 2 + 1, n, n / 2 + 1, n);
  }
  check_positions(fftn::c2c_plan<T>(fftn::planes_c2c<T>(6, 10, 9, fftn::direction::backward)), 9,
                  60, 60, 60, 60);
  // Two batch dimensions: 3 x 4 transforms, numbered with the inner one fastest.
  fftn::shape<c, c> two;
  two.axes = {{.n = 12}};
  two.batches = {{.count = 3}, {.count = 4}};
  check_positions(fftn::c2c_plan<T>(two), 12, 12, 12, 12, 12);
}

// ---- Plans are reusable: repeated runs, copies and concurrent threads agree bit for bit.

TYPED_TEST(Contracts, PlansAreReusableAcrossCopiesAndThreads) {
  using T = TypeParam;
  constexpr std::size_t n = 210;
  constexpr std::size_t count = 8;
  const fftn::c2c_plan<T> p(fftn::rows_c2c<T>(n, count, fftn::direction::forward));
  input_source src(11);
  const auto x = src.complexes<T>(n * count);
  const auto run = [&x](const fftn::c2c_plan<T>& plan) {
    fftn::workspace ws(plan);
    cvec<T> y(x.size());
    plan.execute(x.data(), y.data(), ws);
    return y;
  };
  const cvec<T> serial = run(p);
  EXPECT_TRUE(same_bits<std::complex<T>>(run(p), serial));
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): the copy is under test
  const fftn::c2c_plan<T> copy = p;
  EXPECT_TRUE(same_bits<std::complex<T>>(run(copy), serial));
  constexpr int threads = 4;
  std::array<cvec<T>, threads> results;
  std::vector<std::thread> pool;
  pool.reserve(threads);
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      for (int i = 0; i < 10; ++i) {
        results.at(static_cast<std::size_t>(t)) = run(p);
      }
    });
  }
  for (std::thread& t : pool) {
    t.join();
  }
  for (const cvec<T>& r : results) {
    EXPECT_TRUE(same_bits<std::complex<T>>(r, serial));
  }
}

// Moving a plan copies it: the source stays a valid plan with the same results. Before the
// copy operations were declared, the implicit move left a null core and every call on the
// source dereferenced it.
TYPED_TEST(Contracts, MovedFromPlansStayValid) {
  using T = TypeParam;
  constexpr std::size_t n = 12;
  input_source src(14);
  const auto x = src.complexes<T>(n);
  const auto run = [&x](const fftn::c2c_plan<T>& plan) {
    fftn::workspace ws(plan.workspace_bytes());
    cvec<T> y(x.size());
    plan.execute(x.data(), y.data(), ws);
    return y;
  };
  // The sources are not const: a const source would be copied whatever plan declared.
  // NOLINTNEXTLINE(misc-const-correctness): a const source would not exercise a move
  fftn::c2c_plan<T> p(fftn::rows_c2c<T>(n, 1, fftn::direction::forward));
  const cvec<T> want = run(p);
  // NOLINTNEXTLINE(misc-const-correctness,performance-move-const-arg): the move is under test
  fftn::c2c_plan<T> q(std::move(p));
  EXPECT_TRUE(same_bits<std::complex<T>>(run(q), want));
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move): source under test
  EXPECT_TRUE(same_bits<std::complex<T>>(run(p), want));
  fftn::c2c_plan<T> r(fftn::rows_c2c<T>(n, 1, fftn::direction::backward));
  r = std::move(q); // NOLINT(performance-move-const-arg): the move is the subject of the test
  EXPECT_TRUE(same_bits<std::complex<T>>(run(r), want));
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move): source under test
  EXPECT_EQ(q.describe(), r.describe());
  static_assert(std::is_nothrow_move_constructible_v<fftn::c2c_plan<T>> &&
                std::is_nothrow_move_assignable_v<fftn::c2c_plan<T>>);
}

// ---- Workspaces.

TYPED_TEST(Contracts, WorkspaceOverCallerMemoryAtAnyAlignment) {
  using T = TypeParam;
  const fftn::c2c_plan<T> p(fftn::rows_c2c<T>(96, 2, fftn::direction::backward));
  input_source src(12);
  const auto x = src.complexes<T>(192);
  cvec<T> want(192);
  fftn::workspace own(p);
  EXPECT_GE(own.size(), p.workspace_bytes());
  p.execute(x.data(), want.data(), own);
  std::vector<std::byte> memory(p.workspace_bytes() + 8);
  for (std::size_t offset = 0; offset < 8; ++offset) {
    fftn::workspace borrowed(std::span<std::byte>(memory.data() + offset, p.workspace_bytes()));
    cvec<T> got(192);
    p.execute(x.data(), got.data(), borrowed);
    EXPECT_TRUE(same_bits<std::complex<T>>(got, want)) << offset;
  }
}

TEST(Workspace, MovesAndEmptyWorkspaces) {
  fftn::workspace a(100);
  std::byte* memory = a.memory().data();
  fftn::workspace b(std::move(a));
  EXPECT_EQ(b.size(), 100U);
  EXPECT_EQ(b.memory().data(), memory);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move): source under test
  EXPECT_EQ(a.size(), 0U);
  fftn::workspace c;
  c = std::move(b);
  EXPECT_EQ(c.size(), 100U);
  EXPECT_EQ(fftn::workspace(0).size(), 0U);
  // A one-point out-of-place c2c is a copy and needs no workspace at all.
  const fftn::c2c_plan<double> p(fftn::rows_c2c<double>(1, 4, fftn::direction::forward));
  EXPECT_EQ(p.workspace_bytes(), 0U);
  fftn::workspace none;
  const std::vector<std::complex<double>> x{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  std::vector<std::complex<double>> y(4);
  p.execute(x.data(), y.data(), none);
  EXPECT_EQ(x, y);
}

// ---- Execution preconditions: assertions in debug builds of the library (DESIGN.md 18.2,
// item 11). Builds with NDEBUG (Release and the sanitizer builds) compile them out; the
// gate's coverage build is a Debug build and runs these.

TEST(ExecutionDeathTest, PreconditionsAreAssertedInDebugBuilds) {
#ifdef NDEBUG
  GTEST_SKIP() << "assertions are compiled out of this build type";
#else
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using c = std::complex<double>;
  cvec<double> x(16);
  cvec<double> y(16);
  // An in-place plan given two addresses: a rank-2 in-place c2r would silently compute
  // garbage, so every in-place kind is checked.
  fftn::shape<c, double> planes;
  planes.axes = {{.n = 2}, {.n = 4}};
  const fftn::c2r_plan<double> in_place(planes, {.in_place = true});
  fftn::workspace ws(in_place);
  EXPECT_DEATH(in_place.execute(x.data(), reinterpret_cast<double*>(y.data()), ws),
               "in-place plan executed with in != out");
  const fftn::c2c_plan<double> rows(fftn::rows_c2c<double>(8, 2, fftn::direction::forward));
  fftn::workspace small(rows.workspace_bytes() - 1);
  EXPECT_DEATH(rows.execute(x.data(), y.data(), small), "workspace smaller");
  fftn::workspace enough(rows);
  EXPECT_DEATH(rows.execute_rows(1, 2, x.data(), y.data(), enough), "beyond the batch");
#endif
}

// ---- Invalid shapes are refused at construction, naming the argument.

template <class In, class Out>
void expect_refused(const fftn::shape<In, Out>& s, const std::string& names, fftn::options o = {}) {
  try {
    const fftn::plan<In, Out> p(s, o);
    ADD_FAILURE() << "accepted; expected a refusal naming " << names;
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(names), std::string::npos)
        << e.what() << " does not name " << names;
  }
}

TEST(Validation, InvalidShapesThrowNamingTheArgument) {
  using c = std::complex<double>;
  const fftn::shape<c, c> no_axes;
  expect_refused(no_axes, "shape.axes");
  const fftn::shape<c, c> four_axes{.axes = {{2}, {3}, {4}, {5}}, .batches = {{6}}};
  expect_refused(four_axes, "shape.axes");
  const fftn::shape<c, c> three_batches{.axes = {{8}}, .batches = {{2}, {3}, {4}}};
  expect_refused(three_batches, "shape.batches");
  expect_refused(fftn::rows_c2c<double>(0, 1, fftn::direction::forward), "shape.axes[0].n");
  expect_refused(fftn::planes_c2c<double>(4, 0, 1, fftn::direction::forward), "shape.axes[1].n");
  expect_refused(fftn::rows_c2c<double>(8, 0, fftn::direction::forward), "shape.batches[0].count");
  auto bad_dir = fftn::rows_c2c<double>(8, 1, fftn::direction::forward);
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): invalid on purpose
  bad_dir.dir = static_cast<fftn::direction>(3);
  expect_refused(bad_dir, "shape.dir");

  // Output rows that overlap: stride 1, distance 4, rows of 8.
  fftn::shape<c, c> overlap;
  overlap.axes = {{.n = 8}};
  overlap.batches = {{.count = 2, .in_dist = 8, .out_dist = 4}};
  expect_refused(overlap, "overlap");
  // Interleaved rows that collide: stride 2, distance 1, three rows.
  overlap.axes = {{.n = 8, .out_stride = 2}};
  overlap.batches = {{.count = 3, .out_dist = 1}};
  expect_refused(overlap, "overlap");

  // Sizes beyond the address space.
  fftn::shape<c, c> huge;
  huge.axes = {{.n = std::size_t{1} << 40U}, {.n = std::size_t{1} << 40U}};
  expect_refused(huge, "too large");
  fftn::shape<c, c> far;
  far.axes = {{.n = 4, .in_stride = static_cast<std::ptrdiff_t>(std::size_t{1} << 62U)}};
  expect_refused(far, "too large");
  expect_refused(fftn::rows_c2c<double>(std::size_t{1} << 63U, 1, fftn::direction::forward),
                 "is too large");

  // In place: layouts that disagree.
  fftn::shape<c, c> moved;
  moved.axes = {{.n = 8, .in_stride = 1, .out_stride = 2}};
  expect_refused(moved, "options.in_place", {.in_place = true});
  moved.axes = {{.n = 8}};
  moved.batches = {{.count = 2, .in_dist = 8, .out_dist = 16}};
  expect_refused(moved, "options.in_place", {.in_place = true});
  fftn::shape<double, c> real;
  real.axes = {{.n = 8, .in_stride = 2, .out_stride = 1}};
  expect_refused(real, "options.in_place", {.in_place = true});
  real.axes = {{.n = 4, .in_stride = 5, .out_stride = 5}, {.n = 8}};
  expect_refused(real, "options.in_place", {.in_place = true});
  real.axes = {{.n = 8}};
  real.batches = {{.count = 2, .in_dist = 10, .out_dist = 6}};
  expect_refused(real, "options.in_place", {.in_place = true});
}

// Strides and distances whose magnitudes or derived byte steps do not fit std::ptrdiff_t.
// Without the checks these built plans whose arithmetic overflowed (UBSan reported it in
// the plan's construction and in the copy pass).
TEST(Validation, StridesThatOverflowAreRefused) {
  using c = std::complex<double>;
  constexpr std::ptrdiff_t lowest = std::numeric_limits<std::ptrdiff_t>::min();
  fftn::shape<c, c> s;
  s.axes = {{.n = 2, .out_stride = lowest}, {.n = 8}};
  expect_refused(s, "shape.axes[0].out_stride");
  s.axes = {{.n = 8, .in_stride = lowest}};
  expect_refused(s, "shape.axes[0].in_stride");
  s.axes = {{.n = 8}};
  s.batches = {{.count = 2, .in_dist = lowest}};
  expect_refused(s, "shape.batches[0].in_dist");
  s.batches = {{.count = 3, .out_dist = lowest}};
  expect_refused(s, "shape.batches[0].out_dist");
  // A distance whose byte step overflows although its element count does not.
  s.batches = {{.count = 2, .in_dist = std::numeric_limits<std::ptrdiff_t>::max() / 4}};
  expect_refused(s, "too large");
}

// The stride of an axis of length 1 and the distance of a batch dimension of count 1 never
// step, so any value is accepted for them, PTRDIFF_MIN and values whose byte steps would
// overflow included, and the result is that of the default layout. One test per way the
// unused value used to reach arithmetic that overflowed.
constexpr std::ptrdiff_t lowest_step = std::numeric_limits<std::ptrdiff_t>::min();
constexpr std::ptrdiff_t far_step = std::numeric_limits<std::ptrdiff_t>::max() / 2 + 1; // 2^62

// An 8-point c2c of the same input through `s` and through the default layout.
void expect_default_result(const fftn::shape<std::complex<double>, std::complex<double>>& s) {
  input_source src(13);
  const auto x = src.complexes<double>(8);
  const auto run = [&x](const fftn::c2c_plan<double>& p) {
    fftn::workspace ws(p);
    cvec<double> y(x.size());
    p.execute(x.data(), y.data(), ws);
    return y;
  };
  const fftn::c2c_plan<double> plain(fftn::rows_c2c<double>(8, 1, fftn::direction::forward));
  EXPECT_TRUE(same_bits<std::complex<double>>(run(fftn::c2c_plan<double>(s)), run(plain)));
}

// The batch loop's byte step was computed from the distance even for a count of 1.
TEST(Validation, DistanceOfASingleTransformIsIgnored) {
  fftn::shape<std::complex<double>, std::complex<double>> s;
  s.axes = {{.n = 8}};
  s.batches = {{.count = 1, .in_dist = far_step, .out_dist = far_step}};
  expect_default_result(s);
  s.batches = {{.count = 1, .in_dist = lowest_step, .out_dist = -far_step}};
  expect_default_result(s);
}

// The reach and overlap checks took the magnitude of a length-1 axis's stride.
TEST(Validation, StrideOfALengthOneAxisIsIgnored) {
  fftn::shape<std::complex<double>, std::complex<double>> s;
  s.axes = {{.n = 1, .in_stride = lowest_step, .out_stride = far_step}, {.n = 8}};
  expect_default_result(s);
  s.axes = {{.n = 8}, {.n = 1, .in_stride = far_step, .out_stride = lowest_step}};
  expect_default_result(s);
}

// The copy pass of a one-point line doubled the stride to step over complex values as
// reals, here for two one-point transforms one element apart; so did the edge passes of
// a real transform.
TEST(Validation, StrideOfALengthOneAxisIsNeverDoubled) {
  using c = std::complex<double>;
  fftn::shape<c, c> s;
  s.axes = {{.n = 1, .in_stride = far_step, .out_stride = 1}};
  s.batches = {{.count = 2, .in_dist = 1, .out_dist = 1}};
  const fftn::c2c_plan<double> p(s);
  fftn::workspace ws(p);
  const cvec<double> x{{1.0, 2.0}, {3.0, 4.0}};
  cvec<double> y(2);
  p.execute(x.data(), y.data(), ws);
  EXPECT_EQ(y, x);

  fftn::shape<double, c> r;
  r.axes = {{.n = 1, .in_stride = far_step, .out_stride = far_step}};
  r.batches = {{.count = 2, .in_dist = 1, .out_dist = 1}};
  const fftn::r2c_plan<double> pr(r);
  fftn::workspace wr(pr);
  const std::vector<double> reals{3.0, -2.0};
  cvec<double> spectrum(2);
  pr.execute(reals.data(), spectrum.data(), wr);
  EXPECT_EQ(spectrum, (cvec<double>{{3.0, 0.0}, {-2.0, 0.0}}));
}

TEST(Validation, ValidEdgeCasesAreAccepted) {
  using c = std::complex<double>;
  // Interleaved rows that do not collide: stride 2, distance 1, two rows.
  fftn::shape<c, c> interleaved;
  interleaved.axes = {{.n = 8, .in_stride = 2, .out_stride = 2}};
  interleaved.batches = {{.count = 2, .in_dist = 1, .out_dist = 1}};
  EXPECT_NO_THROW(fftn::c2c_plan<double>(interleaved, {.in_place = true}));
  // A single transform needs no batch dimension.
  fftn::shape<double, c> one;
  one.axes = {{.n = 5}};
  EXPECT_NO_THROW(fftn::r2c_plan<double>(one, {.in_place = true}));
}

// ---- Describe, instruction set.

TEST(Describe, NamesTheProgram) {
  const fftn::c2c_plan<double> p(fftn::rows_c2c<double>(28, 3, fftn::direction::forward));
  const std::string d = p.describe();
  for (const char* part : {
           "c2c forward",
           "double",
           "instruction set: scalar",
           "chain",
           "radices 4 x 7",
           "radix-4 forward",
           "generic radix-7 forward",
           "loop batch x3",
           "workspace",
       }) {
    EXPECT_NE(d.find(part), std::string::npos) << "missing '" << part << "' in\n" << d;
  }
  EXPECT_EQ(p.instruction_set(), fftn::isa::scalar);

  fftn::shape<std::complex<float>, float> volume;
  volume.axes = {{.n = 3}, {.n = 4}, {.n = 6}};
  const std::string v = fftn::c2r_plan<float>(volume).describe();
  for (const char* part : {
           "c2r",
           "float",
           "block of 48 complex",
           "c2c backward",
           "half-length complex chain of 3",
           "loop lines x3",
           "out of place",
       }) {
    EXPECT_NE(v.find(part), std::string::npos) << "missing '" << part << "' in\n" << v;
  }
  const std::string r =
      fftn::r2c_plan<double>(fftn::rows_r2c<double>(9, 1), {.in_place = true}).describe();
  for (const char* part : {"r2c", "full-length complex chain (n odd)", "in place", "1 transform"}) {
    EXPECT_NE(r.find(part), std::string::npos) << "missing '" << part << "' in\n" << r;
  }
}

TEST(Isa, Names) {
  EXPECT_STREQ(fftn::isa_name(fftn::isa::scalar), "scalar");
  EXPECT_STREQ(fftn::isa_name(fftn::isa::x86_v2), "x86_v2");
  EXPECT_STREQ(fftn::isa_name(fftn::isa::avx), "avx");
  EXPECT_STREQ(fftn::isa_name(fftn::isa::avx2), "avx2");
  EXPECT_STREQ(fftn::isa_name(fftn::isa::avx512), "avx512");
  EXPECT_STREQ(fftn::isa_name(fftn::isa::neon), "neon");
  EXPECT_STREQ(fftn::isa_name(fftn::isa::neon_fcma), "neon_fcma");
  // Out of range: this read past the table of names before it was bounds-checked.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): invalid on purpose
  EXPECT_STREQ(fftn::isa_name(static_cast<fftn::isa>(200)), "unknown");
}

} // namespace
