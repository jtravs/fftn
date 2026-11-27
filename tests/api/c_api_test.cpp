// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The C API (fftn/c_api.h, step 6.8): plans built and executed from C (c_api_test.c, compiled as
// C99) give results bitwise equal to the C++ plans of the same shapes, for every kind and
// precision and in every form of execution, a C thread pool among them; every refusal fills
// fftn_error with the C++ message, and no exception escapes.

#include "c_api_test.h"

#include "fftn/c_api.h"
#include "fftn/fftn.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

namespace {

// An empty case of one axis: the tests fill in the rest.
ctest_case base_case(fftn_kind kind, fftn_precision precision) {
  ctest_case c{};
  c.kind = kind;
  c.precision = precision;
  c.direction = kind == FFTN_C2R ? FFTN_BACKWARD : FFTN_FORWARD;
  c.batch.count = 1;
  c.scale = 1.0;
  return c;
}

ctest_case lengths(ctest_case c, std::vector<std::size_t> n, std::size_t count) {
  c.rank = n.size();
  for (std::size_t d = 0; d < n.size(); ++d) {
    c.axes[d].n = n[d];
  }
  c.batch.count = count;
  return c;
}

bool complex_side(const ctest_case& c, bool input) {
  return c.kind == FFTN_C2C || (input ? c.kind == FFTN_C2R : c.kind == FFTN_R2C);
}

// The real values a side's array spans: tight C-ordered strides where a stride or distance is
// 0, the last axis halved on the complex side of a real transform, the given strides elsewhere
// (the cases with stored runs give theirs explicitly, so the full lengths bound them).
std::size_t side_reals(const ctest_case& c, bool input) {
  const bool complex = complex_side(c, input);
  const bool halved = complex && (c.kind == FFTN_R2C || c.kind == FFTN_C2R);
  std::size_t span = 1;
  std::size_t tight = 1;
  for (std::size_t i = c.rank; i > 0; --i) {
    const std::size_t d = i - 1;
    // Lengths and counts of 0, which plans refuse, count as 1.
    const std::size_t n = std::max<std::size_t>(c.axes[d].n, 1);
    const std::size_t len = halved && d + 1 == c.rank ? n / 2 + 1 : n;
    const std::ptrdiff_t given = input ? c.axes[d].in_stride : c.axes[d].out_stride;
    const auto magnitude = static_cast<std::size_t>(given < 0 ? -given : given);
    const std::size_t stride = given != 0 ? magnitude : tight;
    span += (len - 1) * stride;
    tight *= len;
  }
  const std::ptrdiff_t dist = input ? c.batch.in_dist : c.batch.out_dist;
  span += (std::max<std::size_t>(c.batch.count, 1) - 1) *
          (dist != 0 ? static_cast<std::size_t>(dist) : tight);
  return span * (complex ? 2 : 1);
}

template <class T> std::vector<T> input_values(std::size_t reals) {
  std::vector<T> v(reals);
  std::uint32_t state = 12345;
  for (T& x : v) {
    state = state * 1664525U + 1013904223U;
    x = static_cast<T>(static_cast<double>(state >> 8U) / static_cast<double>(1U << 24U) - 0.5);
  }
  return v;
}

// What the arrays hold where nothing is written.
template <class T> constexpr T untouched = static_cast<T>(-7.25);

template <class In, class Out> fftn::shape<In, Out> cpp_shape(const ctest_case& c) {
  fftn::shape<In, Out> s;
  s.dir = static_cast<fftn::direction>(c.direction);
  for (std::size_t d = 0; d < c.rank; ++d) {
    s.axes.push_back(
        {.n = c.axes[d].n, .in_stride = c.axes[d].in_stride, .out_stride = c.axes[d].out_stride});
  }
  s.batches.push_back(
      {.count = c.batch.count, .in_dist = c.batch.in_dist, .out_dist = c.batch.out_dist});
  for (std::size_t r = 0; r < c.run_count; ++r) {
    s.runs.push_back({
        .axis = c.runs[r].axis,
        .where = c.runs[r].where == FFTN_IN ? fftn::side::in : fftn::side::out,
        .logical = c.runs[r].logical,
        .stored = c.runs[r].stored,
        .count = c.runs[r].count,
    });
  }
  if constexpr (std::is_same_v<In, Out>) {
    if (c.kind == FFTN_R2R) {
      for (std::size_t d = 0; d < c.rank; ++d) {
        s.kinds.push_back(static_cast<fftn::r2r_kind>(c.r2r[d]));
      }
    }
  }
  return s;
}

// The C++ plan's result, or its refusal's message.
struct cpp_result {
  std::vector<std::byte> out;
  std::optional<std::string> refused;
  std::string description;
  std::size_t workspace_bytes = 0;
  fftn::isa instruction_set = fftn::isa::best;
  std::vector<fftn::transform_range> owned; // by the members of a team of 3
};

template <class In, class Out, class T>
cpp_result run_cpp(const ctest_case& c, const std::vector<T>& in, std::size_t out_reals) {
  cpp_result r;
  try {
    const fftn::plan<In, Out> p(cpp_shape<In, Out>(c),
                                {.in_place = c.in_place != 0, .scale = c.scale});
    fftn::workspace ws(p);
    std::vector<T> buffer = in;
    std::vector<T> out(c.in_place != 0 ? 0 : out_reals, untouched<T>);
    auto* o = c.in_place != 0 ? buffer.data() : out.data();
    p.execute(reinterpret_cast<const In*>(buffer.data()), reinterpret_cast<Out*>(o), ws);
    const std::vector<T>& result = c.in_place != 0 ? buffer : out;
    r.out.resize(result.size() * sizeof(T));
    std::memcpy(r.out.data(), result.data(), r.out.size());
    r.description = p.describe();
    r.workspace_bytes = p.workspace_bytes(3);
    r.instruction_set = p.instruction_set();
    for (int rank = 0; rank < 3; ++rank) {
      r.owned.push_back(p.transforms_of({.rank = rank, .size = 3}));
    }
  } catch (const std::invalid_argument& e) {
    r.refused = e.what();
  }
  return r;
}

template <class T>
cpp_result run_cpp_kind(const ctest_case& c, const std::vector<T>& in, std::size_t out_reals) {
  using C = std::complex<T>;
  switch (c.kind) {
  case FFTN_C2C:
    return run_cpp<C, C>(c, in, out_reals);
  case FFTN_R2C:
    return run_cpp<T, C>(c, in, out_reals);
  case FFTN_C2R:
    return run_cpp<C, T>(c, in, out_reals);
  case FFTN_R2R:
    return run_cpp<T, T>(c, in, out_reals);
  }
  return {};
}

std::string truncated(const std::string& message) {
  return message.substr(0, sizeof(fftn_error::message) - 1);
}

struct form_case {
  ctest_form form;
  int lanes;
  int group_size;
  const char* name;
};

constexpr std::array forms = {
    form_case{.form = CTEST_SERIAL, .lanes = 1, .group_size = 0, .name = "serial"},
    form_case{.form = CTEST_PIECES, .lanes = 1, .group_size = 0, .name = "pieces"},
    form_case{.form = CTEST_TEAM_ALL, .lanes = 3, .group_size = 0, .name = "team of 3"},
    form_case{.form = CTEST_TEAM_ALL, .lanes = 4, .group_size = 2, .name = "team of 4 in pairs"},
    form_case{.form = CTEST_TEAM_OWN, .lanes = 3, .group_size = 0, .name = "team of 3, own"},
    form_case{.form = CTEST_EXECUTOR, .lanes = 3, .group_size = 0, .name = "C pool of 3"},
    form_case{.form = CTEST_EXECUTOR, .lanes = 4, .group_size = 2, .name = "C pool of 4 in pairs"},
};

// Runs case `c` through the C API in every form and through the C++ API, and compares.
template <class T> void check_case(const ctest_case& c, const std::string& name) {
  SCOPED_TRACE(name);
  const std::size_t in_reals = side_reals(c, true);
  const std::size_t out_reals = side_reals(c, false);
  const std::size_t buffer_reals = c.in_place != 0 ? std::max(in_reals, out_reals) : in_reals;
  const std::vector<T> in = input_values<T>(buffer_reals);
  const cpp_result expected = run_cpp_kind<T>(c, in, out_reals);
  for (const form_case& f : forms) {
    SCOPED_TRACE(f.name);
    std::vector<T> buffer = in;
    std::vector<T> out(c.in_place != 0 ? 0 : out_reals, untouched<T>);
    void* o = c.in_place != 0 ? static_cast<void*>(buffer.data()) : static_cast<void*>(out.data());
    fftn_error error{};
    int status = -1;
    EXPECT_NO_THROW(status =
                        ctest_run(&c, f.form, f.lanes, f.group_size, buffer.data(), o, &error));
    if (expected.refused) {
      EXPECT_EQ(status, FFTN_ERROR_INVALID_ARGUMENT);
      EXPECT_EQ(error.code, FFTN_ERROR_INVALID_ARGUMENT);
      EXPECT_EQ(std::string(static_cast<const char*>(error.message)), truncated(*expected.refused));
      return; // every form fails alike
    }
    ASSERT_EQ(status, FFTN_OK) << error.message;
    const std::vector<T>& result = c.in_place != 0 ? buffer : out;
    ASSERT_EQ(result.size() * sizeof(T), expected.out.size());
    EXPECT_EQ(std::memcmp(result.data(), expected.out.data(), expected.out.size()), 0)
        << "the C plan's result differs from the C++ plan's";
  }
}

void check(const ctest_case& c, const std::string& name) {
  if (c.precision == FFTN_DOUBLE) {
    check_case<double>(c, name + ", double");
  } else {
    check_case<float>(c, name + ", float");
  }
}

constexpr std::array precisions = {FFTN_DOUBLE, FFTN_FLOAT};

TEST(CApi, ComplexPlansMatchCpp) {
  for (const fftn_precision p : precisions) {
    for (const int dir : {FFTN_FORWARD, FFTN_BACKWARD}) {
      ctest_case c = base_case(FFTN_C2C, p);
      c.direction = dir;
      const std::string d = dir == FFTN_FORWARD ? "forward" : "backward";
      check(lengths(c, {12}, 5), "c2c rows of 12, " + d);
      check(lengths(c, {17}, 4), "c2c rows of 17, " + d);
      check(lengths(c, {6, 10}, 3), "c2c planes 6 x 10, " + d);
      check(lengths(c, {4, 6, 5}, 3), "c2c volumes 4 x 6 x 5, " + d);
    }
  }
}

TEST(CApi, RealPlansMatchCpp) {
  for (const fftn_precision p : precisions) {
    for (const fftn_kind kind : {FFTN_R2C, FFTN_C2R}) {
      const ctest_case c = base_case(kind, p);
      const std::string k = kind == FFTN_R2C ? "r2c" : "c2r";
      check(lengths(c, {16}, 5), k + " rows of 16");
      check(lengths(c, {15}, 4), k + " rows of 15");
      check(lengths(c, {6, 10}, 3), k + " planes 6 x 10");
      check(lengths(c, {4, 6, 9}, 3), k + " volumes 4 x 6 x 9");
    }
  }
}

// Every real-to-real kind. A shape a kind refuses (a DCT-I of one point, say) is refused from C
// with the C++ message: the check compares either way.
TEST(CApi, RealToRealPlansMatchCpp) {
  constexpr std::array kinds = {
      FFTN_DCT1, FFTN_DCT2, FFTN_DCT3, FFTN_DCT4, FFTN_DST1,
      FFTN_DST2, FFTN_DST3, FFTN_DST4, FFTN_DHT,
  };
  for (const fftn_precision p : precisions) {
    for (const fftn_r2r_kind k : kinds) {
      ctest_case c = base_case(FFTN_R2R, p);
      c.r2r[0] = c.r2r[1] = c.r2r[2] = k;
      const std::string name = "r2r kind " + std::to_string(static_cast<int>(k));
      check(lengths(c, {16}, 4), name + ", rows of 16");
      check(lengths(c, {9}, 3), name + ", rows of 9");
      check(lengths(c, {6, 8}, 2), name + ", planes 6 x 8");
      check(lengths(c, {4, 5, 6}, 2), name + ", volumes 4 x 5 x 6");
    }
    ctest_case mixed = base_case(FFTN_R2R, p);
    mixed.r2r[0] = FFTN_DHT;
    mixed.r2r[1] = FFTN_DCT2;
    mixed.r2r[2] = FFTN_DST3;
    check(lengths(mixed, {4, 6, 8}, 2), "r2r volumes of mixed kinds");
  }
}

TEST(CApi, LayoutsAndOptionsMatchCpp) {
  for (const fftn_precision p : precisions) {
    // In place and scaled.
    ctest_case c2c = lengths(base_case(FFTN_C2C, p), {8, 12}, 3);
    c2c.in_place = 1;
    c2c.scale = 0.125;
    check(c2c, "c2c planes in place, scaled");
    ctest_case r2r = lengths(base_case(FFTN_R2R, p), {20}, 3);
    r2r.r2r[0] = FFTN_DHT;
    r2r.in_place = 1;
    check(r2r, "r2r rows in place");
    // Strides and distances: rows of 10 every 13 elements, transposed output.
    ctest_case strided = lengths(base_case(FFTN_C2C, p), {10}, 6);
    strided.axes[0].in_stride = 1;
    strided.axes[0].out_stride = 6;
    strided.batch.in_dist = 13;
    strided.batch.out_dist = 1;
    check(strided, "c2c rows strided, transposed output");
    ctest_case r2c = lengths(base_case(FFTN_R2C, p), {12}, 4);
    r2c.axes[0].in_stride = 2;
    r2c.batch.in_dist = 30;
    check(r2c, "r2c rows strided");
    // Stored runs: rows of 32 of which 6 points are stored at each end of the input.
    ctest_case runs = lengths(base_case(FFTN_C2C, p), {32}, 3);
    runs.axes[0].in_stride = 1;
    runs.axes[0].out_stride = 1;
    runs.batch.in_dist = 12;
    runs.batch.out_dist = 32;
    runs.run_count = 2;
    runs.runs[0] = {.axis = 0, .where = FFTN_IN, .logical = 0, .stored = 0, .count = 6};
    runs.runs[1] = {.axis = 0, .where = FFTN_IN, .logical = 26, .stored = 6, .count = 6};
    check(runs, "c2c rows with stored runs");
  }
}

// Shapes and options the C++ API refuses come back from C with the C++ message.
TEST(CApi, CppRefusalsComeThroughWithTheirMessages) {
  ctest_case zero = lengths(base_case(FFTN_C2C, FFTN_DOUBLE), {12}, 2);
  zero.axes[0].n = 0;
  check(zero, "an axis of length 0");
  ctest_case count = lengths(base_case(FFTN_R2C, FFTN_FLOAT), {12}, 2);
  count.batch.count = 0;
  check(count, "a batch of no transforms");
  ctest_case scale = lengths(base_case(FFTN_C2C, FFTN_DOUBLE), {12}, 2);
  scale.scale = std::numeric_limits<double>::quiet_NaN();
  check(scale, "a scale that is not a number");
  ctest_case big = lengths(base_case(FFTN_C2C, FFTN_FLOAT), {12}, 2);
  big.scale = 1e300;
  check(big, "a scale that overflows float");
  ctest_case overlap = lengths(base_case(FFTN_C2C, FFTN_DOUBLE), {12}, 2);
  overlap.axes[0].out_stride = 1;
  overlap.batch.out_dist = 3;
  check(overlap, "overlapping output");
  ctest_case in_place = lengths(base_case(FFTN_R2C, FFTN_DOUBLE), {12}, 2);
  in_place.in_place = 1;
  check(in_place, "an r2c plan in place without padding");
  ctest_case run = lengths(base_case(FFTN_C2C, FFTN_DOUBLE), {12}, 2);
  run.run_count = 1;
  run.runs[0] = {.axis = 3, .where = FFTN_OUT, .logical = 0, .stored = 0, .count = 4};
  check(run, "a stored run of an axis that does not exist");
  ctest_case dct1 = lengths(base_case(FFTN_R2R, FFTN_DOUBLE), {1}, 2);
  dct1.r2r[0] = FFTN_DCT1;
  check(dct1, "a DCT-I of one point");
}

// The plan's queries agree with the C++ plan's.
TEST(CApi, QueriesMatchCpp) {
  for (const fftn_precision p : precisions) {
    const ctest_case c = lengths(base_case(FFTN_C2C, p), {48, 30}, 7);
    const fftn_shape s = {
        .kind = c.kind,
        .precision = c.precision,
        .direction = c.direction,
        .rank = c.rank,
        .axes = c.axes,
        .batch_rank = 1,
        .batches = &c.batch,
        .r2r_kinds = nullptr,
        .run_count = 0,
        .runs = nullptr,
        .layout = FFTN_INTERLEAVED,
    };
    fftn_plan* plan = fftn_plan_create(&s, nullptr, nullptr);
    ASSERT_NE(plan, nullptr);
    const cpp_result expected =
        p == FFTN_DOUBLE ? run_cpp_kind<double>(c, input_values<double>(side_reals(c, true)),
                                                side_reals(c, false))
                         : run_cpp_kind<float>(c, input_values<float>(side_reals(c, true)),
                                               side_reals(c, false));
    std::string text(fftn_plan_describe(plan, nullptr, 0), '\0');
    EXPECT_EQ(fftn_plan_describe(plan, text.data(), text.size() + 1), text.size());
    EXPECT_EQ(text, expected.description);
    EXPECT_EQ(fftn_plan_workspace_bytes(plan, 3), expected.workspace_bytes);
    EXPECT_EQ(fftn_plan_instruction_set(plan), static_cast<int>(expected.instruction_set));
    for (int rank = 0; rank < 3; ++rank) {
      const fftn_transform_range r =
          fftn_transforms_of(plan, {.rank = rank, .size = 3, .group_size = 0});
      EXPECT_EQ(r.first, expected.owned[static_cast<std::size_t>(rank)].first);
      EXPECT_EQ(r.count, expected.owned[static_cast<std::size_t>(rank)].count);
    }
    fftn_plan_release(plan);
  }
}

TEST(CApi, DefaultOptionsAreCppDefaults) {
  const fftn_options o = fftn_options_default();
  const fftn::options d{};
  EXPECT_EQ(o.in_place != 0, d.in_place);
  EXPECT_EQ(o.scale, d.scale);
  EXPECT_EQ(o.force_isa, static_cast<int>(d.force_isa));
  EXPECT_EQ(o.effort, static_cast<int>(d.effort));
  EXPECT_EQ(o.measure_seconds, d.measure_seconds);
  EXPECT_EQ(o.measure_scratch_bytes, d.measure_scratch_bytes);
  EXPECT_EQ(o.use, nullptr);
}

TEST(CApi, RefusalsFromC) { EXPECT_EQ(ctest_refusals(), 0) << "the check of that number failed"; }

// Wisdom made in C holds the entry C++ wisdom records for the same plan.
TEST(CApi, WisdomFromC) {
  // In the test's working directory, which is the build tree's: builds do not share it.
  const std::filesystem::path path = "fftn_c_api_test.wisdom";
  std::vector<char> text(8192);
  EXPECT_EQ(ctest_wisdom(path.string().c_str(), text.data(), text.size()), 0)
      << "the check of that number failed";
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  fftn::wisdom w;
  w.add(fftn::c2c_plan<double>(fftn::rows_c2c<double>(64, 4, fftn::direction::forward)));
  EXPECT_EQ(std::string(text.data()), w.text());
}

TEST(CApi, LengthsMatchCpp) {
  for (const std::size_t n : {
           std::size_t{0},
           std::size_t{1},
           std::size_t{131},
           std::size_t{1001},
           std::size_t{4093},
           std::size_t{65537},
       }) {
    fftn_error e{};
    EXPECT_EQ(ctest_next_fast_length(n, FFTN_LENGTH_COMPLEX, &e), fftn::next_fast_length(n));
    EXPECT_EQ(ctest_next_fast_length(n, FFTN_LENGTH_REAL, &e),
              fftn::next_fast_length(n, fftn::length_kind::real));
    EXPECT_EQ(ctest_relative_cost(n, FFTN_DOUBLE, &e), fftn::relative_cost<double>(n));
    EXPECT_EQ(ctest_relative_cost(n, FFTN_FLOAT, &e), fftn::relative_cost<float>(n));
    EXPECT_EQ(e.code, FFTN_OK);
  }
  fftn_error e{};
  const std::size_t too_long = std::size_t{1} << 41U;
  EXPECT_EQ(fftn_next_fast_length(too_long, FFTN_LENGTH_COMPLEX, &e), 0U);
  try {
    (void)fftn::next_fast_length(too_long);
    ADD_FAILURE() << "next_fast_length accepted 2^41";
  } catch (const std::invalid_argument& ex) {
    EXPECT_EQ(std::string(static_cast<const char*>(e.message)), truncated(ex.what()));
  }
}

TEST(CApi, WorkspacesAndReferences) {
  fftn_error e{};
  fftn_workspace* owned = fftn_workspace_create(0, &e);
  ASSERT_NE(owned, nullptr);
  EXPECT_EQ(e.code, FFTN_OK);
  fftn_workspace_destroy(owned);
  std::vector<std::byte> memory(4096);
  fftn_workspace* borrowed = fftn_workspace_borrow(memory.data() + 1, memory.size() - 1, &e);
  ASSERT_NE(borrowed, nullptr);
  // A borrowed workspace of any alignment serves an execution.
  const fftn_axis axis{.n = 64, .in_stride = 0, .out_stride = 0};
  const fftn_batch batch{.count = 2, .in_dist = 0, .out_dist = 0};
  const fftn_shape s{
      .kind = FFTN_C2C,
      .precision = FFTN_DOUBLE,
      .direction = FFTN_FORWARD,
      .rank = 1,
      .axes = &axis,
      .batch_rank = 1,
      .batches = &batch,
      .r2r_kinds = nullptr,
      .run_count = 0,
      .runs = nullptr,
      .layout = FFTN_INTERLEAVED,
  };
  fftn_plan* plan = fftn_plan_create(&s, nullptr, &e);
  ASSERT_NE(plan, nullptr);
  ASSERT_LE(fftn_plan_workspace_bytes(plan, 1), memory.size() - 1);
  // References: the plan lives until the last is released.
  fftn_plan* again = fftn_plan_ref(plan);
  EXPECT_EQ(again, plan);
  fftn_plan_release(plan);
  const std::vector<std::complex<double>> in(128, {1.0, 0.0});
  std::vector<std::complex<double>> out(128);
  fftn_execute(again, in.data(), out.data(), borrowed);
  EXPECT_EQ(out[0], std::complex<double>(64.0, 0.0));
  EXPECT_EQ(out[64], std::complex<double>(64.0, 0.0));
  fftn_plan_release(again);
  fftn_workspace_destroy(borrowed);
}

// Out of memory: the error, not an exception. A sanitizer's allocator stops the process on an
// allocation this large instead of failing it, so the check runs only without one.
#ifdef __has_feature
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define FFTN_TEST_SANITIZED 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define FFTN_TEST_SANITIZED 1
#endif

TEST(CApi, OutOfMemoryIsAnError) {
#ifdef FFTN_TEST_SANITIZED
  GTEST_SKIP() << "the sanitizer's allocator does not fail allocations";
#else
  fftn_error error{};
  fftn_workspace* ws = nullptr;
  EXPECT_NO_THROW(ws = fftn_workspace_create(std::numeric_limits<std::size_t>::max() / 2, &error));
  EXPECT_EQ(ws, nullptr);
  EXPECT_EQ(error.code, FFTN_ERROR_OUT_OF_MEMORY);
  EXPECT_NE(error.message[0], '\0');
#endif
}

// A C++ executor behind the C struct that throws: a status, not an exception.
TEST(CApi, ThrowingExecutorIsAStatus) {
  struct throwing {
    static int concurrency(void* /*self*/) { return 4; }
    static void bulk(void* /*self*/, int /*n*/, fftn_task /*task*/, void* /*arg*/) {
      throw std::runtime_error("the pool failed");
    }
  };
  // Large enough that the plan shares it out rather than running it on the caller.
  const fftn_axis axis{.n = 4096, .in_stride = 0, .out_stride = 0};
  const fftn_batch batch{.count = 64, .in_dist = 0, .out_dist = 0};
  const fftn_shape s{
      .kind = FFTN_C2C,
      .precision = FFTN_DOUBLE,
      .direction = FFTN_FORWARD,
      .rank = 1,
      .axes = &axis,
      .batch_rank = 1,
      .batches = &batch,
      .r2r_kinds = nullptr,
      .run_count = 0,
      .runs = nullptr,
      .layout = FFTN_INTERLEAVED,
  };
  fftn_plan* plan = fftn_plan_create(&s, nullptr, nullptr);
  ASSERT_NE(plan, nullptr);
  fftn_workspace* ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, 4), nullptr);
  ASSERT_NE(ws, nullptr);
  const fftn_executor ex{
      .self = nullptr,
      .concurrency = &throwing::concurrency,
      .bulk = &throwing::bulk,
      .group_size = nullptr,
  };
  constexpr std::size_t points = std::size_t{4096} * 64;
  const std::vector<std::complex<double>> in(points);
  std::vector<std::complex<double>> out(points);
  int status = FFTN_OK;
  EXPECT_NO_THROW(status = fftn_execute_on(plan, in.data(), out.data(), &ex, ws));
  EXPECT_EQ(status, FFTN_ERROR_EXECUTOR);
  // The workspace serves the next execution.
  fftn_execute(plan, in.data(), out.data(), ws);
  fftn_workspace_destroy(ws);
  fftn_plan_release(plan);
}

} // namespace
