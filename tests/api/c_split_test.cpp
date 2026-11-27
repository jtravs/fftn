// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Split complex in the C API (fftn/c_api.h, "Split complex"; DESIGN.md 5.9, provisional): split
// plans built and executed from C (c_split_test.c, compiled as C99) through the four _split
// forms give results bitwise equal to the C++ split plans of the same shapes, for c2c in both
// directions, r2c and c2r, rows to volumes with tight and gapped layouts, out of place and in
// place, in both precisions; every array of both sides compared whole, gaps and canaries
// included. describe(), the workspace's bytes and the members' transforms agree with C++; every
// refusal fills fftn_error, the C++ messages passed through; layout mismatches at execution and
// a NULL imaginary array end the program.

#include "c_split_test.h"
#include "c_api_test.h"

#include "fftn/c_api.h"
#include "fftn/fftn.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

// One split shape of up to three axes and one batch dimension; strides and distances 0 for the
// tight defaults, given ones positive.
struct split_case {
  fftn_kind kind = FFTN_C2C;
  int direction = FFTN_FORWARD;
  std::vector<std::size_t> n{};
  std::size_t count = 1;
  std::vector<std::ptrdiff_t> in_strides{}; // empty: all 0
  std::vector<std::ptrdiff_t> out_strides{};
  std::ptrdiff_t in_dist = 0;
  std::ptrdiff_t out_dist = 0;
  bool in_place = false;
  double scale = 1.0;
  std::string name{};
};

std::ptrdiff_t stride_at(const std::vector<std::ptrdiff_t>& s, std::size_t d) {
  return s.empty() ? 0 : s[d];
}

bool complex_side(const split_case& c, bool input) {
  return c.kind == FFTN_C2C || (input ? c.kind == FFTN_C2R : c.kind == FFTN_R2C);
}

// The reals each array of a side spans, in its own elements: the last axis halved on the complex
// side of a real transform.
std::size_t side_reals(const split_case& c, bool input) {
  const bool halved = complex_side(c, input) && c.kind != FFTN_C2C;
  std::size_t span = 1;
  std::size_t tight = 1;
  for (std::size_t i = c.n.size(); i > 0; --i) {
    const std::size_t d = i - 1;
    const std::size_t len = halved && d + 1 == c.n.size() ? c.n[d] / 2 + 1 : c.n[d];
    const std::ptrdiff_t given = stride_at(input ? c.in_strides : c.out_strides, d);
    const std::size_t stride = given != 0 ? static_cast<std::size_t>(given) : tight;
    span += (len - 1) * stride;
    tight *= len;
  }
  const std::ptrdiff_t dist = input ? c.in_dist : c.out_dist;
  span += (c.count - 1) * (dist != 0 ? static_cast<std::size_t>(dist) : tight);
  return span;
}

// The C shape of a case; its arrays point into `axes` and `batch`.
fftn_shape c_shape(const split_case& c, fftn_precision precision, std::array<fftn_axis, 3>& axes,
                   fftn_batch& batch) {
  fftn_shape s{};
  s.kind = c.kind;
  s.precision = precision;
  s.direction = c.direction;
  s.rank = c.n.size();
  for (std::size_t d = 0; d < c.n.size(); ++d) {
    axes.at(d) = {.n = c.n[d],
                  .in_stride = stride_at(c.in_strides, d),
                  .out_stride = stride_at(c.out_strides, d)};
  }
  s.axes = axes.data();
  batch = {.count = c.count, .in_dist = c.in_dist, .out_dist = c.out_dist};
  s.batch_rank = 1;
  s.batches = &batch;
  s.split = 1;
  return s;
}

template <class In, class Out> fftn::shape<In, Out> cpp_shape(const split_case& c) {
  fftn::shape<In, Out> s;
  s.dir = static_cast<fftn::direction>(c.direction);
  for (std::size_t d = 0; d < c.n.size(); ++d) {
    s.axes.push_back({.n = c.n[d],
                      .in_stride = stride_at(c.in_strides, d),
                      .out_stride = stride_at(c.out_strides, d)});
  }
  s.batches.push_back({.count = c.count, .in_dist = c.in_dist, .out_dist = c.out_dist});
  return s;
}

template <class T> std::vector<T> values(std::size_t reals, std::uint32_t seed) {
  std::vector<T> v(reals);
  std::uint32_t state = seed;
  for (T& x : v) {
    state = state * 1664525U + 1013904223U;
    x = static_cast<T>(static_cast<double>(state >> 8U) / static_cast<double>(1U << 24U) - 0.5);
  }
  return v;
}

// What the arrays hold where nothing is written.
template <class T> constexpr T untouched = static_cast<T>(-7.25);

// The arrays of one execution: the input's real parts (or real array) and imaginary parts, the
// output's likewise; an empty vector for a real side's imaginary parts. In place, the output's
// arrays are the input's.
template <class T> struct arrays {
  std::vector<T> in_re, in_im, out_re, out_im;

  arrays(const split_case& c, std::size_t salt) {
    const std::size_t in_n = side_reals(c, true);
    in_re = values<T>(in_n, static_cast<std::uint32_t>(1234 + salt));
    if (complex_side(c, true)) {
      in_im = values<T>(in_n, static_cast<std::uint32_t>(98765 + salt));
    }
    if (!c.in_place) {
      const std::size_t out_n = side_reals(c, false);
      out_re.assign(out_n, untouched<T>);
      if (complex_side(c, false)) {
        out_im.assign(out_n, untouched<T>);
      }
    }
  }

  T* re_out(bool in_place) { return in_place ? in_re.data() : out_re.data(); }
  T* im_out(bool in_place) {
    std::vector<T>& v = in_place ? in_im : out_im;
    return v.empty() ? nullptr : v.data();
  }
  const T* im_in() const { return in_im.empty() ? nullptr : in_im.data(); }
  // The output's arrays, to compare.
  [[nodiscard]] std::vector<T> result_re(bool in_place) const { return in_place ? in_re : out_re; }
  [[nodiscard]] std::vector<T> result_im(bool in_place) const { return in_place ? in_im : out_im; }
};

// What the C++ split plan computes and says, or its refusal.
template <class T> struct cpp_result {
  std::optional<std::string> refused;
  std::vector<T> re, im;
  std::string description;
  std::size_t workspace_bytes = 0;
  std::vector<fftn::transform_range> owned; // by the members of a team of 3
};

template <class In, class Out, class T> cpp_result<T> run_cpp(const split_case& c) {
  cpp_result<T> r;
  try {
    const fftn::plan<In, Out> p(cpp_shape<In, Out>(c), {.in_place = c.in_place, .scale = c.scale});
    fftn::workspace ws(p);
    arrays<T> a(c, 0);
    typename fftn::plan<In, Out>::in_data in{};
    typename fftn::plan<In, Out>::out_data out{};
    if constexpr (std::is_same_v<In, T>) {
      in = a.in_re.data();
    } else {
      in = {a.in_re.data(), a.im_in()};
    }
    if constexpr (std::is_same_v<Out, T>) {
      out = a.re_out(c.in_place);
    } else {
      out = {a.re_out(c.in_place), a.im_out(c.in_place)};
    }
    p.execute(in, out, ws);
    r.re = a.result_re(c.in_place);
    r.im = a.result_im(c.in_place);
    r.description = p.describe();
    r.workspace_bytes = p.workspace_bytes(3);
    for (int rank = 0; rank < 3; ++rank) {
      r.owned.push_back(p.transforms_of({.rank = rank, .size = 3}));
    }
  } catch (const std::invalid_argument& e) {
    r.refused = e.what();
  }
  return r;
}

template <class T> cpp_result<T> run_cpp_kind(const split_case& c) {
  using S = fftn::split_complex<T>;
  switch (c.kind) {
  case FFTN_C2C:
    return run_cpp<S, S, T>(c);
  case FFTN_R2C:
    return run_cpp<T, S, T>(c);
  case FFTN_C2R:
    return run_cpp<S, T, T>(c);
  case FFTN_R2R:
    break;
  }
  return {};
}

template <class T>
constexpr fftn_precision precision_of = std::is_same_v<T, double> ? FFTN_DOUBLE : FFTN_FLOAT;

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
template <class T> void check_case(const split_case& c) {
  SCOPED_TRACE(c.name);
  const cpp_result<T> expected = run_cpp_kind<T>(c);
  ASSERT_FALSE(expected.refused.has_value()) << *expected.refused;

  std::array<fftn_axis, 3> axes{};
  fftn_batch batch{};
  const fftn_shape shape = c_shape(c, precision_of<T>, axes, batch);
  fftn_options options = fftn_options_default();
  options.in_place = c.in_place ? 1 : 0;
  options.scale = c.scale;

  // The queries, from C.
  fftn_error error{};
  fftn_plan* plan = fftn_plan_create(&shape, &options, &error);
  ASSERT_NE(plan, nullptr) << static_cast<const char*>(error.message);
  std::string text(fftn_plan_describe(plan, nullptr, 0), '\0');
  fftn_plan_describe(plan, text.data(), text.size() + 1);
  EXPECT_EQ(text, expected.description);
  EXPECT_NE(text.find("split"), std::string::npos) << text;
  EXPECT_EQ(fftn_plan_workspace_bytes(plan, 3), expected.workspace_bytes);
  for (int rank = 0; rank < 3; ++rank) {
    const fftn_transform_range own =
        fftn_transforms_of(plan, {.rank = rank, .size = 3, .group_size = 0});
    EXPECT_EQ(own.first, expected.owned.at(static_cast<std::size_t>(rank)).first);
    EXPECT_EQ(own.count, expected.owned.at(static_cast<std::size_t>(rank)).count);
  }
  fftn_plan_release(plan);

  for (const form_case& f : forms) {
    SCOPED_TRACE(f.name);
    arrays<T> a(c, 0);
    const fftn_split_const_ptr in{.re = a.in_re.data(), .im = a.im_in()};
    const fftn_split_ptr out{.re = a.re_out(c.in_place), .im = a.im_out(c.in_place)};
    fftn_error e{};
    const int status =
        csplit_run(&shape, &options, f.form, f.lanes, f.group_size, c.count, in, out, &e);
    ASSERT_EQ(status, FFTN_OK) << static_cast<const char*>(e.message);
    // Bitwise: the C plan runs the C++ plan's program on the same arrays.
    EXPECT_TRUE(a.result_re(c.in_place) == expected.re);
    EXPECT_TRUE(a.result_im(c.in_place) == expected.im);
    if (!c.in_place) {
      EXPECT_EQ(a.in_re, arrays<T>(c, 0).in_re) << "the input changed";
      EXPECT_EQ(a.in_im, arrays<T>(c, 0).in_im) << "the input changed";
    }
  }
}

std::vector<split_case> cases() {
  std::vector<split_case> v;
  for (const fftn_kind kind : {FFTN_C2C, FFTN_R2C, FFTN_C2R}) {
    const std::vector<int> directions =
        kind == FFTN_C2C ? std::vector<int>{FFTN_FORWARD, FFTN_BACKWARD}
                         : std::vector<int>{kind == FFTN_R2C ? FFTN_FORWARD : FFTN_BACKWARD};
    for (const int dir : directions) {
      const std::string k =
          std::string(kind == FFTN_C2C ? (dir == FFTN_FORWARD ? "c2c forward" : "c2c backward")
                                       : (kind == FFTN_R2C ? "r2c" : "c2r"));
      // Rows: a codelet length, a mixed one, and a prime (Bluestein) with a scale.
      v.push_back({.kind = kind, .direction = dir, .n = {16}, .count = 5, .name = k + " rows 16"});
      v.push_back({.kind = kind, .direction = dir, .n = {60}, .count = 3, .name = k + " rows 60"});
      v.push_back({.kind = kind,
                   .direction = dir,
                   .n = {97},
                   .count = 2,
                   .scale = 0.5,
                   .name = k + " rows 97 scaled"});
      // Planes and volumes, tight.
      v.push_back(
          {.kind = kind, .direction = dir, .n = {8, 12}, .count = 3, .name = k + " planes"});
      v.push_back(
          {.kind = kind, .direction = dir, .n = {4, 6, 10}, .count = 2, .name = k + " volumes"});
      // Gaps: strided rows and a plane with padded rows and transforms apart.
      v.push_back({.kind = kind,
                   .direction = dir,
                   .n = {24},
                   .count = 4,
                   .in_strides = {3},
                   .out_strides = {2},
                   .in_dist = 80,
                   .out_dist = 53,
                   .name = k + " strided rows"});
      v.push_back({.kind = kind,
                   .direction = dir,
                   .n = {6, 10},
                   .count = 2,
                   .in_strides = {13, 1},
                   .out_strides = {12, 1},
                   .in_dist = 90,
                   .out_dist = 85,
                   .name = k + " padded planes"});
      if (kind == FFTN_C2C) {
        v.push_back({.kind = kind,
                     .direction = dir,
                     .n = {32},
                     .count = 3,
                     .in_place = true,
                     .name = k + " rows in place"});
        v.push_back({.kind = kind,
                     .direction = dir,
                     .n = {6, 4, 5},
                     .count = 2,
                     .in_place = true,
                     .name = k + " volumes in place"});
      }
    }
  }
  return v;
}

TEST(CSplit, MatchesCppDouble) {
  for (const split_case& c : cases()) {
    check_case<double>(c);
  }
}

TEST(CSplit, MatchesCppFloat) {
  for (const split_case& c : cases()) {
    check_case<float>(c);
  }
}

TEST(CSplit, RefusalsFillTheError) { EXPECT_EQ(csplit_refusals(), 0); }

// The core's refusals of split shapes reach C with the C++ messages.
TEST(CSplit, RefusalsPassTheCppMessages) {
  const fftn::axis axis{.n = 16};
  const fftn_axis c_axis{.n = 16, .in_stride = 0, .out_stride = 0};
  const fftn_batch c_batch{.count = 2, .in_dist = 0, .out_dist = 0};
  fftn_shape s{};
  s.kind = FFTN_R2C;
  s.precision = FFTN_DOUBLE;
  s.rank = 1;
  s.axes = &c_axis;
  s.batch_rank = 1;
  s.batches = &c_batch;
  s.split = 1;
  fftn_options o = fftn_options_default();
  o.in_place = 1;
  fftn_error e{};
  EXPECT_EQ(fftn_plan_create(&s, &o, &e), nullptr);
  EXPECT_EQ(e.code, FFTN_ERROR_INVALID_ARGUMENT);
  fftn::shape<double, fftn::split_complex<double>> cpp;
  cpp.axes = {axis};
  cpp.batches = {{.count = 2}};
  try {
    const fftn::plan<double, fftn::split_complex<double>> p(cpp, {.in_place = true});
    ADD_FAILURE() << "C++ accepted an in-place split r2c plan";
  } catch (const std::invalid_argument& x) {
    EXPECT_EQ(std::string(static_cast<const char*>(e.message)),
              std::string(x.what()).substr(0, sizeof(e.message) - 1));
  }
}

// A split plan of the given kind, built from C.
fftn_plan* split_plan(int split) {
  static const fftn_axis axis{.n = 16, .in_stride = 0, .out_stride = 0};
  static const fftn_batch batch{.count = 2, .in_dist = 0, .out_dist = 0};
  fftn_shape s{};
  s.kind = FFTN_C2C;
  s.precision = FFTN_DOUBLE;
  s.direction = FFTN_FORWARD;
  s.rank = 1;
  s.axes = &axis;
  s.batch_rank = 1;
  s.batches = &batch;
  s.split = split;
  return fftn_plan_create(&s, nullptr, nullptr);
}

TEST(CSplitDeathTest, ASplitPlanThroughAnInterleavedFormEndsTheProgram) {
  fftn_plan* plan = split_plan(1);
  ASSERT_NE(plan, nullptr);
  EXPECT_DEATH(
      {
        std::vector<double> x(64);
        std::vector<double> y(64);
        fftn_workspace* ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, 1), nullptr);
        fftn_execute(plan, x.data(), y.data(), ws);
      },
      "without the _split suffix");
  fftn_plan_release(plan);
}

TEST(CSplitDeathTest, AnInterleavedPlanThroughASplitFormEndsTheProgram) {
  fftn_plan* plan = split_plan(0);
  ASSERT_NE(plan, nullptr);
  EXPECT_DEATH(
      {
        std::vector<double> x(64);
        std::vector<double> y(64);
        fftn_workspace* ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, 1), nullptr);
        fftn_execute_split(plan, {.re = x.data(), .im = x.data() + 32},
                           {.re = y.data(), .im = y.data() + 32}, ws);
      },
      "_split form");
  fftn_plan_release(plan);
}

TEST(CSplitDeathTest, ANullImaginaryArrayEndsTheProgram) {
  fftn_plan* plan = split_plan(1);
  ASSERT_NE(plan, nullptr);
  EXPECT_DEATH(
      {
        std::vector<double> x(64);
        std::vector<double> y(64);
        fftn_workspace* ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, 1), nullptr);
        fftn_execute_split(plan, {.re = x.data(), .im = nullptr},
                           {.re = y.data(), .im = y.data() + 32}, ws);
      },
      "split_ptr::im");
  fftn_plan_release(plan);
}

} // namespace
