// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The C API for fusion (fftn/c_api.h, step 6.9): plans with edges and row pipelines built and
// executed from C (c_fusion_test.c, compiled as C99), with callbacks written in C, give results
// bitwise equal to the C++ objects of the same descriptions, for each edge operation, callback
// mode and pipeline kind, in every form of execution; the C++ objects' callbacks call the same C
// functions. No exception crosses a C callback's boundary.

#include "c_fusion_test.h"

#include "c_api_test.h"

#include "fftn/c_api.h"
#include "fftn/fftn.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// ---- Callbacks ----

enum class cb : unsigned char { none, modify, produce, consume, record };

fftn_block_fn c_fn(cb c) {
  switch (c) {
  case cb::modify:
    return cfuse_modify;
  case cb::produce:
    return cfuse_produce;
  case cb::consume:
    return cfuse_consume;
  case cb::record:
    return cfuse_record;
  case cb::none:
    break;
  }
  return nullptr;
}

fftn_callback_mode c_mode(cb c) {
  return c == cb::produce   ? FFTN_CALLBACK_PRODUCE
         : c == cb::consume ? FFTN_CALLBACK_CONSUME
                            : FFTN_CALLBACK_MODIFY;
}

fftn::callback_mode cpp_mode(cb c) {
  return c == cb::produce   ? fftn::callback_mode::produce
         : c == cb::consume ? fftn::callback_mode::consume
                            : fftn::callback_mode::modify;
}

// The C++ objects' callback: the same C function, handed the block's place converted here, by
// the test's own code, from what the C++ object passes.
template <class T, cb C> void cpp_call(void* user, const fftn::block_info& info, T* data) noexcept {
  fftn_block_info c{};
  c.transform = info.transform;
  c.rank = info.origin.size();
  c.origin = info.origin.data();
  c.element_axis = info.element_axis;
  c.line_axis = info.line_axis;
  c.length = info.length;
  c.lines = info.lines;
  c.element_stride = info.element_stride;
  c.line_stride = info.line_stride;
  c.group = info.group;
  c.group_stride = info.group_stride;
  c.copy = info.copy;
  c.member = info.member;
  c.scratch = info.scratch.data();
  c.scratch_bytes = info.scratch.size();
  c_fn(C)(user, &c, data);
}

template <class T> fftn::block_fn<T> cpp_fn(cb c) {
  switch (c) {
  case cb::modify:
    return &cpp_call<T, cb::modify>;
  case cb::produce:
    return &cpp_call<T, cb::produce>;
  case cb::consume:
    return &cpp_call<T, cb::consume>;
  case cb::record:
    return &cpp_call<T, cb::record>;
  case cb::none:
    break;
  }
  return nullptr;
}

template <class T> constexpr cfuse_type type_of() {
  if constexpr (std::is_same_v<T, double>) {
    return CFUSE_DOUBLE;
  } else if constexpr (std::is_same_v<T, float>) {
    return CFUSE_FLOAT;
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    return CFUSE_CDOUBLE;
  } else {
    return CFUSE_CFLOAT;
  }
}

template <class T> constexpr bool is_complex = !std::is_floating_point_v<T>;

// ---- Descriptions ----

struct factor_spec {
  fftn_factor_kind kind = FFTN_FACTOR_NONE;
  std::vector<std::ptrdiff_t> axis{};
  std::vector<std::ptrdiff_t> batch{};
};

struct edge_spec {
  double scale = 1.0;
  factor_spec mul{};
  bool accumulate = false;
  cb call = cb::none;
  std::size_t scratch = 0;
};

fftn_factor c_factor(const factor_spec& f) {
  return {
      .kind = f.kind,
      .axis_stride_count = f.axis.size(),
      .axis_strides = f.axis.empty() ? nullptr : f.axis.data(),
      .batch_stride_count = f.batch.size(),
      .batch_strides = f.batch.empty() ? nullptr : f.batch.data(),
  };
}

fftn_callback c_callback(cb c, std::size_t scratch) {
  return {.fn = c_fn(c), .mode = c_mode(c), .scratch_bytes = scratch};
}

fftn_edge c_edge(const edge_spec& e) {
  fftn_edge c = fftn_edge_default();
  c.scale = e.scale;
  c.mul = c_factor(e.mul);
  c.accumulate = e.accumulate ? 1 : 0;
  c.call = c_callback(e.call, e.scratch);
  return c;
}

fftn::factor cpp_factor(const factor_spec& f) {
  return {.kind = static_cast<fftn::factor_kind>(f.kind),
          .axis_strides = f.axis,
          .batch_strides = f.batch};
}

template <class T> fftn::callback<T> cpp_callback(cb c, std::size_t scratch) {
  if (c == cb::none) {
    return {};
  }
  return {.fn = cpp_fn<T>(c), .mode = cpp_mode(c), .scratch_bytes = scratch};
}

template <class T> fftn::edge<T> cpp_edge(const edge_spec& e) {
  return {.scale = e.scale,
          .mul = cpp_factor(e.mul),
          .accumulate = e.accumulate,
          .call = cpp_callback<T>(e.call, e.scratch)};
}

// The elements an array of factors spans for coordinates of lengths `lens` (one per axis
// stride) and batch counts `counts` (one per batch stride; none broadcasts).
std::size_t factor_elements(const factor_spec& f, const std::vector<std::size_t>& lens,
                            const std::vector<std::size_t>& counts, std::size_t extra = 0) {
  std::size_t span = 1 + extra;
  for (std::size_t d = 0; d < f.axis.size(); ++d) {
    span += (lens[d] - 1) * static_cast<std::size_t>(f.axis[d]);
  }
  for (std::size_t e = 0; e < f.batch.size(); ++e) {
    span += (counts[e] - 1) * static_cast<std::size_t>(f.batch[e]);
  }
  return span;
}

template <class T> std::vector<T> values(std::size_t n, std::uint32_t seed) {
  std::vector<T> v(n);
  std::uint32_t state = seed;
  for (T& x : v) {
    state = state * 1664525U + 1013904223U;
    x = static_cast<T>(static_cast<double>(state >> 8U) / static_cast<double>(1U << 24U) - 0.5);
  }
  return v;
}

template <class T> constexpr T untouched = static_cast<T>(-7.25);

template <class T> bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
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
    form_case{.form = CTEST_TEAM_OWN, .lanes = 4, .group_size = 2, .name = "team of 4, own"},
    form_case{.form = CTEST_EXECUTOR, .lanes = 3, .group_size = 0, .name = "C pool of 3"},
};

std::string c_text(std::size_t (*describe)(const void*, char*, std::size_t), const void* object) {
  std::string text(describe(object, nullptr, 0), '\0');
  describe(object, text.data(), text.size() + 1);
  return text;
}

// ---- Plans with edges ----

struct plan_case {
  const char* name{};
  fftn_kind kind{};
  fftn_precision precision{};
  std::vector<std::size_t> n{};
  std::size_t count{};
  edge_spec in{};
  edge_spec out{};
  std::vector<fftn_r2r_kind> r2r{}; // FFTN_R2R: one kind per axis
  bool in_place = false;
  fftn_planning effort = FFTN_PLANNING_RULES;
};

int direction_of(fftn_kind k) { return k == FFTN_C2R ? FFTN_BACKWARD : FFTN_FORWARD; }

// The lengths of a side: the last axis halved on the complex side of a real transform.
std::vector<std::size_t> side_lengths(const plan_case& c, bool input) {
  std::vector<std::size_t> lens = c.n;
  if ((input && c.kind == FFTN_C2R) || (!input && c.kind == FFTN_R2C)) {
    lens.back() = lens.back() / 2 + 1;
  }
  return lens;
}

std::size_t product(const std::vector<std::size_t>& v) {
  std::size_t p = 1;
  for (const std::size_t x : v) {
    p *= x;
  }
  return p;
}

// The arrays of a case's execution: one set for C++, a fresh one per C run.
template <class In, class Out, class R> struct plan_arrays {
  std::vector<R> in;
  std::vector<R> out;
  std::vector<R> sink;
  std::vector<R> in_factor;
  std::vector<R> out_factor;
  cfuse_user in_user{};
  cfuse_user out_user{};

  explicit plan_arrays(const plan_case& c) {
    const std::vector<std::size_t> in_lens = side_lengths(c, true);
    const std::vector<std::size_t> out_lens = side_lengths(c, false);
    const std::size_t in_reals = c.count * product(in_lens) * (is_complex<In> ? 2 : 1);
    const std::size_t out_elements = c.count * product(out_lens);
    const std::size_t out_reals = out_elements * (is_complex<Out> ? 2 : 1);
    in = values<R>(in_reals, 12345U);
    out = c.out.accumulate ? values<R>(out_reals, 777U) : std::vector<R>(out_reals, untouched<R>);
    const std::vector<std::size_t> counts{c.count};
    const auto factor = [&](const factor_spec& f, const std::vector<std::size_t>& lens,
                            std::uint32_t seed) {
      if (f.kind == FFTN_FACTOR_NONE) {
        return std::vector<R>{};
      }
      return values<R>(factor_elements(f, lens, counts) * (f.kind == FFTN_FACTOR_COMPLEX ? 2 : 1),
                       seed);
    };
    in_factor = factor(c.in.mul, in_lens, 4242U);
    out_factor = factor(c.out.mul, out_lens, 999U);
    in_user.type = type_of<In>();
    out_user.type = type_of<Out>();
    if (c.out.call == cb::consume) {
      sink.assign(out_reals, untouched<R>);
      out_user.sink = sink.data();
      std::ptrdiff_t stride = 1;
      for (std::size_t d = out_lens.size(); d > 0; --d) {
        out_user.strides[d - 1] = stride;
        stride *= static_cast<std::ptrdiff_t>(out_lens[d - 1]);
      }
      out_user.dist = stride;
    }
  }
  plan_arrays(const plan_arrays&) = delete;
  plan_arrays& operator=(const plan_arrays&) = delete;
  plan_arrays(plan_arrays&&) = delete;
  plan_arrays& operator=(plan_arrays&&) = delete;
  ~plan_arrays() = default;

  [[nodiscard]] const void* in_ptr(const plan_case& c) const {
    return c.in.call == cb::produce ? nullptr : in.data();
  }
  [[nodiscard]] void* out_ptr(const plan_case& c) {
    if (c.in_place) {
      return in.data();
    }
    return c.out.call == cb::consume ? nullptr : out.data();
  }
  // The output's values: the input's array in place.
  [[nodiscard]] const std::vector<R>& result(const plan_case& c) const {
    return c.in_place ? in : out;
  }
  [[nodiscard]] fftn_exec_args c_args() {
    return {.in_factor = in_factor.empty() ? nullptr : in_factor.data(),
            .mid_factor = nullptr,
            .out_factor = out_factor.empty() ? nullptr : out_factor.data(),
            .in_user = &in_user,
            .mid_user = nullptr,
            .out_user = &out_user};
  }
};

template <class In, class Out, class R> void check_plan(const plan_case& c) {
  SCOPED_TRACE(c.name);
  // C++.
  fftn::shape<In, Out> s;
  s.dir = static_cast<fftn::direction>(direction_of(c.kind));
  for (const std::size_t n : c.n) {
    s.axes.push_back({.n = n});
  }
  s.batches.push_back({.count = c.count});
  if constexpr (std::is_same_v<In, Out>) {
    for (const fftn_r2r_kind k : c.r2r) {
      s.kinds.push_back(static_cast<fftn::r2r_kind>(k));
    }
  }

  // The C description.
  std::vector<fftn_axis> axes;
  for (const std::size_t n : c.n) {
    axes.push_back({.n = n, .in_stride = 0, .out_stride = 0});
  }
  const fftn_batch batch = {.count = c.count, .in_dist = 0, .out_dist = 0};
  const fftn_shape shape = {
      .kind = c.kind,
      .precision = c.precision,
      .direction = direction_of(c.kind),
      .rank = axes.size(),
      .axes = axes.data(),
      .batch_rank = 1,
      .batches = &batch,
      .r2r_kinds = c.r2r.empty() ? nullptr : c.r2r.data(),
      .run_count = 0,
      .runs = nullptr,
      .layout = FFTN_INTERLEAVED,
  };
  const fftn_plan_edges edges = {.in = c_edge(c.in), .out = c_edge(c.out)};
  fftn_options options = fftn_options_default();
  options.in_place = c.in_place ? 1 : 0;
  options.effort = c.effort;

  // With measured planning the C plan's program is recorded in C wisdom, from which the C++
  // object and every C execution rebuild it: measurement may choose differently each time.
  std::optional<fftn::wisdom> cpp_wisdom;
  fftn_wisdom* c_wisdom = nullptr;
  if (c.effort != FFTN_PLANNING_RULES) {
    c_wisdom = fftn_wisdom_create(nullptr);
    ASSERT_NE(c_wisdom, nullptr);
    fftn_plan* measured = fftn_plan_create_edges(&shape, &options, &edges, nullptr);
    ASSERT_NE(measured, nullptr);
    ASSERT_EQ(fftn_wisdom_add(c_wisdom, measured, nullptr), FFTN_OK);
    fftn_plan_release(measured);
    std::string text(fftn_wisdom_text(c_wisdom, nullptr, 0), '\0');
    fftn_wisdom_text(c_wisdom, text.data(), text.size() + 1);
    cpp_wisdom.emplace(text);
    options.use = c_wisdom;
  }

  // C++.
  const fftn::plan<In, Out> p(s,
                              {.in_place = c.in_place,
                               .effort = static_cast<fftn::planning>(c.effort),
                               .use = cpp_wisdom ? &*cpp_wisdom : nullptr},
                              {.in = cpp_edge<In>(c.in), .out = cpp_edge<Out>(c.out)});
  plan_arrays<In, Out, R> expected(c);
  {
    fftn::workspace ws(p);
    const fftn_exec_args a = expected.c_args();
    p.execute(static_cast<const In*>(expected.in_ptr(c)), static_cast<Out*>(expected.out_ptr(c)),
              ws,
              {.in_factor = a.in_factor,
               .mid_factor = nullptr,
               .out_factor = a.out_factor,
               .in_user = a.in_user,
               .mid_user = nullptr,
               .out_user = a.out_user});
  }

  // C, built from the same description.
  fftn_plan* plan = fftn_plan_create_edges(&shape, &options, &edges, nullptr);
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(c_text(
                [](const void* o, char* b, std::size_t n) {
                  return fftn_plan_describe(static_cast<const fftn_plan*>(o), b, n);
                },
                plan),
            p.describe());
  EXPECT_EQ(fftn_plan_workspace_bytes(plan, 3), p.workspace_bytes(3));
  EXPECT_EQ(fftn_plan_instruction_set(plan), static_cast<int>(p.instruction_set()));
  fftn_plan_release(plan);

  for (const form_case& f : forms) {
    SCOPED_TRACE(f.name);
    plan_arrays<In, Out, R> got(c);
    const fftn_exec_args args = got.c_args();
    fftn_error error{};
    const int status = cfuse_run_plan(&shape, &options, &edges, f.form, f.lanes, f.group_size,
                                      c.count, got.in_ptr(c), got.out_ptr(c), &args, &error);
    ASSERT_EQ(status, FFTN_OK) << error.message;
    EXPECT_TRUE(same_bits(got.result(c), expected.result(c)));
    EXPECT_TRUE(same_bits(got.sink, expected.sink));
  }
  fftn_wisdom_destroy(c_wisdom);
}

template <class R> void check_plan_kind(const plan_case& c) {
  using C = std::complex<R>;
  switch (c.kind) {
  case FFTN_C2C:
    check_plan<C, C, R>(c);
    break;
  case FFTN_R2C:
    check_plan<R, C, R>(c);
    break;
  case FFTN_C2R:
    check_plan<C, R, R>(c);
    break;
  case FFTN_R2R:
    check_plan<R, R, R>(c);
    break;
  }
}

void check_plan_case(const plan_case& c) {
  if (c.precision == FFTN_DOUBLE) {
    check_plan_kind<double>(c);
  } else {
    check_plan_kind<float>(c);
  }
}

const factor_spec complex_rows{.kind = FFTN_FACTOR_COMPLEX, .axis = {1}, .batch = {0}};

TEST(CFusion, EdgeVocabularyMatchesCpp) {
  check_plan_case(
      {.name = "c2c rows: a scale and a complex factor in, a real factor and "
               "accumulation out",
       .kind = FFTN_C2C,
       .precision = FFTN_DOUBLE,
       .n = {24},
       .count = 5,
       .in = {.scale = 0.5, .mul = complex_rows},
       .out = {.mul = {.kind = FFTN_FACTOR_REAL, .axis = {1}, .batch = {24}}, .accumulate = true}});
  check_plan_case({.name = "r2c planes: a real factor in, a scale out",
                   .kind = FFTN_R2C,
                   .precision = FFTN_FLOAT,
                   .n = {6, 10},
                   .count = 3,
                   .in = {.mul = {.kind = FFTN_FACTOR_REAL, .axis = {10, 1}, .batch = {}}},
                   .out = {.scale = 0.25}});
  check_plan_case({.name = "c2r rows: a complex factor in, accumulation out",
                   .kind = FFTN_C2R,
                   .precision = FFTN_DOUBLE,
                   .n = {32},
                   .count = 4,
                   .in = {.mul = {.kind = FFTN_FACTOR_COMPLEX, .axis = {1}, .batch = {17}}},
                   .out = {.accumulate = true}});
  check_plan_case({.name = "c2c rows in float: a complex factor out, a scale in",
                   .kind = FFTN_C2C,
                   .precision = FFTN_FLOAT,
                   .n = {20},
                   .count = 6,
                   .in = {.scale = 2.0},
                   .out = {.mul = complex_rows}});
}

TEST(CFusion, CallbackModesMatchCpp) {
  check_plan_case({.name = "c2c rows: modify in and out, with scratch",
                   .kind = FFTN_C2C,
                   .precision = FFTN_DOUBLE,
                   .n = {48},
                   .count = 6,
                   .in = {.call = cb::modify},
                   .out = {.call = cb::modify, .scratch = 64}});
  check_plan_case({.name = "r2c rows: produce in, consume out",
                   .kind = FFTN_R2C,
                   .precision = FFTN_FLOAT,
                   .n = {40},
                   .count = 5,
                   .in = {.call = cb::produce},
                   .out = {.call = cb::consume}});
  check_plan_case({.name = "c2r rows: modify in, a factor beside a callback out",
                   .kind = FFTN_C2R,
                   .precision = FFTN_DOUBLE,
                   .n = {30},
                   .count = 4,
                   .in = {.call = cb::modify},
                   .out = {.scale = 0.5, .call = cb::modify}});
  check_plan_case({.name = "c2c planes: modify in, consume out",
                   .kind = FFTN_C2C,
                   .precision = FFTN_DOUBLE,
                   .n = {8, 12},
                   .count = 3,
                   .in = {.call = cb::modify},
                   .out = {.call = cb::consume}});
  check_plan_case({.name = "c2r planes in float: produce in, modify out",
                   .kind = FFTN_C2R,
                   .precision = FFTN_FLOAT,
                   .n = {6, 16},
                   .count = 2,
                   .in = {.call = cb::produce},
                   .out = {.call = cb::modify}});
  check_plan_case({.name = "c2c volumes in float: modify in, a scale out",
                   .kind = FFTN_C2C,
                   .precision = FFTN_FLOAT,
                   .n = {4, 6, 8},
                   .count = 2,
                   .in = {.call = cb::modify},
                   .out = {.scale = 0.125}});
}

TEST(CFusion, InPlaceRealToRealAndMeasuredPlansMatchCpp) {
  check_plan_case({.name = "c2c rows in place: a complex factor and modify in, modify out",
                   .kind = FFTN_C2C,
                   .precision = FFTN_DOUBLE,
                   .n = {32},
                   .count = 4,
                   .in = {.mul = complex_rows, .call = cb::modify},
                   .out = {.scale = 0.5, .call = cb::modify},
                   .in_place = true});
  check_plan_case({.name = "c2c planes in float in place: modify out",
                   .kind = FFTN_C2C,
                   .precision = FFTN_FLOAT,
                   .n = {6, 10},
                   .count = 3,
                   .out = {.call = cb::modify},
                   .in_place = true});
  // Real-to-real plans take the vocabulary (callbacks are refused: cfuse_refusals).
  check_plan_case({.name = "DCT-II rows: a real factor in, a scale and accumulation out",
                   .kind = FFTN_R2R,
                   .precision = FFTN_DOUBLE,
                   .n = {24},
                   .count = 3,
                   .in = {.mul = {.kind = FFTN_FACTOR_REAL, .axis = {1}, .batch = {0}}},
                   .out = {.scale = 0.25, .accumulate = true},
                   .r2r = {FFTN_DCT2}});
  check_plan_case({.name = "DST-III by DHT planes in float in place: a real factor out",
                   .kind = FFTN_R2R,
                   .precision = FFTN_FLOAT,
                   .n = {8, 12},
                   .count = 2,
                   .out = {.mul = {.kind = FFTN_FACTOR_REAL, .axis = {12, 1}, .batch = {}}},
                   .r2r = {FFTN_DST3, FFTN_DHT},
                   .in_place = true});
  // Measured planning times stand-ins, never the C callbacks; the program is then rebuilt
  // from the C plan's wisdom.
  check_plan_case({.name = "c2c rows, quick measurement: modify in, consume out",
                   .kind = FFTN_C2C,
                   .precision = FFTN_DOUBLE,
                   .n = {64},
                   .count = 8,
                   .in = {.call = cb::modify},
                   .out = {.call = cb::consume},
                   .effort = FFTN_PLANNING_QUICK});
  check_plan_case({.name = "r2c rows in float, quick measurement: a factor and modify in",
                   .kind = FFTN_R2C,
                   .precision = FFTN_FLOAT,
                   .n = {96},
                   .count = 5,
                   .in = {.mul = {.kind = FFTN_FACTOR_REAL, .axis = {1}, .batch = {0}},
                          .call = cb::modify,
                          .scratch = 64},
                   .effort = FFTN_PLANNING_QUICK});
}

// ---- What a C callback is told: member, scratch and scratch_bytes ----

// A call's place: transform, origin, length, lines, scratch bytes, and the scratch's offset in
// the workspace's memory.
using place = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                         std::size_t, std::size_t, std::ptrdiff_t>;

// Each member's places, sorted, after checking the scratch: the bytes asked for, one address
// per member, the members' areas disjoint, no call counted bad by cfuse_record. `base` is the
// workspace's memory.
std::vector<std::vector<place>> places_of(const cfuse_log& log, std::size_t bytes,
                                          const std::byte* base) {
  std::vector<std::vector<place>> out(CFUSE_LOG_MEMBERS);
  std::vector<std::pair<const char*, const char*>> areas;
  for (std::size_t m = 0; m < CFUSE_LOG_MEMBERS; ++m) {
    EXPECT_EQ(log.bad[m], 0) << "member " << m;
    EXPECT_LE(log.count[m], static_cast<std::size_t>(CFUSE_LOG_CALLS));
    const std::size_t logged = std::min<std::size_t>(log.count[m], CFUSE_LOG_CALLS);
    const void* first = logged != 0 ? log.calls[m][0].scratch : nullptr;
    for (std::size_t k = 0; k < logged; ++k) {
      const cfuse_call& c = log.calls[m][k];
      EXPECT_EQ(c.scratch_bytes, bytes);
      EXPECT_EQ(c.scratch, first) << "member " << m << " call " << k;
      out[m].emplace_back(c.transform, c.origin[0], c.origin[1], c.origin[2], c.length, c.lines,
                          c.scratch_bytes, static_cast<const std::byte*>(c.scratch) - base);
    }
    std::sort(out[m].begin(), out[m].end());
    if (first != nullptr) {
      areas.emplace_back(static_cast<const char*>(first), static_cast<const char*>(first) + bytes);
    }
  }
  for (std::size_t a = 0; a < areas.size(); ++a) {
    for (std::size_t b = a + 1; b < areas.size(); ++b) {
      EXPECT_TRUE(areas[a].second <= areas[b].first || areas[b].second <= areas[a].first);
    }
  }
  return out;
}

struct record_case {
  const char* name{};
  fftn_kind kind{};
  fftn_precision precision{};
  std::vector<std::size_t> n{};
  std::size_t count{};
  std::size_t scratch{};
};

// Both edges of a plan record their calls (cfuse_record), from C and from the C++ object, serially
// and by a team of 3 with completion::own (whose members run their own transforms' blocks): the
// same outputs, the same places per member, the scratch as the C++ object hands it.
template <class In, class Out, class R> void check_record(const record_case& c) {
  SCOPED_TRACE(c.name);
  const plan_case pc{
      .name = c.name, .kind = c.kind, .precision = c.precision, .n = c.n, .count = c.count};
  fftn::shape<In, Out> s;
  s.dir = static_cast<fftn::direction>(direction_of(c.kind));
  for (const std::size_t n : c.n) {
    s.axes.push_back({.n = n});
  }
  s.batches.push_back({.count = c.count});
  const edge_spec e{.call = cb::record, .scratch = c.scratch};
  const fftn::plan<In, Out> p(s, {}, {.in = cpp_edge<In>(e), .out = cpp_edge<Out>(e)});

  std::vector<fftn_axis> axes;
  for (const std::size_t n : c.n) {
    axes.push_back({.n = n, .in_stride = 0, .out_stride = 0});
  }
  const fftn_batch batch = {.count = c.count, .in_dist = 0, .out_dist = 0};
  const fftn_shape shape = {.kind = c.kind,
                            .precision = c.precision,
                            .direction = direction_of(c.kind),
                            .rank = axes.size(),
                            .axes = axes.data(),
                            .batch_rank = 1,
                            .batches = &batch,
                            .r2r_kinds = nullptr,
                            .run_count = 0,
                            .runs = nullptr,
                            .layout = FFTN_INTERLEAVED};
  const fftn_plan_edges edges = {.in = c_edge(e), .out = c_edge(e)};

  for (const int lanes : {1, 3}) {
    SCOPED_TRACE(lanes == 1 ? "serial" : "team of 3, own");
    // One workspace memory, on a 4096-byte boundary, for both executions in turn: the C
    // callback's scratch must be where the C++ object's block_info puts it.
    const std::size_t bytes = p.workspace_bytes(lanes);
    std::vector<std::byte> memory(bytes + 4096);
    std::byte* const base =
        memory.data() + (4096 - reinterpret_cast<std::uintptr_t>(memory.data()) % 4096) % 4096;
    const auto run = [&](bool from_c, std::vector<R>& out, cfuse_log& in_log, cfuse_log& out_log) {
      const plan_arrays<In, Out, R> a(pc);
      out = a.out;
      in_log.type = type_of<In>();
      out_log.type = type_of<Out>();
      if (from_c) {
        const fftn_exec_args args = {.in_factor = nullptr,
                                     .mid_factor = nullptr,
                                     .out_factor = nullptr,
                                     .in_user = &in_log,
                                     .mid_user = nullptr,
                                     .out_user = &out_log};
        fftn_error error{};
        EXPECT_EQ(cfuse_run_plan_on(&shape, nullptr, &edges,
                                    lanes == 1 ? CTEST_SERIAL : CTEST_TEAM_OWN, lanes, 0, c.count,
                                    a.in.data(), out.data(), &args, base, bytes, &error),
                  FFTN_OK)
            << error.message;
        return;
      }
      fftn::workspace ws(std::span<std::byte>(base, bytes));
      const fftn::exec_args cpp_args{.in_user = &in_log, .out_user = &out_log};
      const auto* in = reinterpret_cast<const In*>(a.in.data());
      auto* o = reinterpret_cast<Out*>(out.data());
      if (lanes == 1) {
        p.execute(in, o, ws, cpp_args);
        return;
      }
      std::vector<std::thread> members;
      for (int r = 1; r < lanes; ++r) {
        members.emplace_back([&, r] {
          p.execute(in, o, {.rank = r, .size = lanes}, ws, fftn::completion::own, cpp_args);
        });
      }
      p.execute(in, o, {.rank = 0, .size = lanes}, ws, fftn::completion::own, cpp_args);
      for (std::thread& t : members) {
        t.join();
      }
    };
    // The logs are large, so on the heap; zeroed.
    const auto c_in = std::make_unique<cfuse_log>();
    const auto c_outlog = std::make_unique<cfuse_log>();
    const auto cpp_in = std::make_unique<cfuse_log>();
    const auto cpp_outlog = std::make_unique<cfuse_log>();
    std::vector<R> c_out;
    std::vector<R> cpp_out;
    run(true, c_out, *c_in, *c_outlog);
    run(false, cpp_out, *cpp_in, *cpp_outlog);
    EXPECT_TRUE(same_bits(c_out, cpp_out));
    EXPECT_EQ(places_of(*c_in, c.scratch, base), places_of(*cpp_in, c.scratch, base));
    EXPECT_EQ(places_of(*c_outlog, c.scratch, base), places_of(*cpp_outlog, c.scratch, base));
    for (std::size_t m = 0; m < CFUSE_LOG_MEMBERS; ++m) {
      if (static_cast<int>(m) >= lanes) {
        EXPECT_EQ(c_in->count[m] + c_outlog->count[m], 0U) << "member " << m;
      } else {
        EXPECT_NE(c_in->count[m], 0U) << "member " << m;
      }
    }
  }
}

TEST(CFusion, CallbacksSeeTheirMemberAndScratch) {
  check_record<std::complex<double>, std::complex<double>, double>({.name = "c2c rows",
                                                                    .kind = FFTN_C2C,
                                                                    .precision = FFTN_DOUBLE,
                                                                    .n = {48},
                                                                    .count = 6,
                                                                    .scratch = 256});
  check_record<std::complex<double>, std::complex<double>, double>({.name = "c2c planes",
                                                                    .kind = FFTN_C2C,
                                                                    .precision = FFTN_DOUBLE,
                                                                    .n = {8, 12},
                                                                    .count = 3,
                                                                    .scratch = 128});
  check_record<float, std::complex<float>, float>({.name = "r2c rows in float",
                                                   .kind = FFTN_R2C,
                                                   .precision = FFTN_FLOAT,
                                                   .n = {40},
                                                   .count = 5,
                                                   .scratch = 64});
}

// ---- Pipelines ----

struct middle_spec {
  double scale = 1.0;
  factor_spec mul{};
  cb call = cb::none;
  std::size_t scratch = 0;
};

struct pipe_case {
  const char* name{};
  fftn_pipeline_kind kind{};
  fftn_precision precision{};
  int first = FFTN_FORWARD;
  std::size_t n{};
  std::vector<std::size_t> counts{};
  edge_spec pro{};
  middle_spec mid{};
  edge_spec epi{};
  std::size_t fanout = 1;
  std::vector<fftn_stored_run> runs{};
  std::size_t group = FFTN_NO_GROUP; // a batch dimension the middle callback sees whole
};

std::size_t end_length(const pipe_case& c) {
  return c.kind == FFTN_PIPELINE_CTC ? c.n / 2 + 1 : c.n;
}
std::size_t mid_length(const pipe_case& c) {
  return c.kind == FFTN_PIPELINE_TCT ? c.n / 2 + 1 : c.n;
}

// The stored extent of a side: the end of its last run, or the whole line.
std::size_t extent(const pipe_case& c, fftn_side side) {
  std::size_t e = 0;
  for (const fftn_stored_run& r : c.runs) {
    if (r.where == side) {
      e = std::max(e, r.stored + r.count);
    }
  }
  return e != 0 ? e : end_length(c);
}

std::vector<fftn_batch> pipe_batches(const pipe_case& c) {
  std::vector<fftn_batch> b;
  for (const std::size_t count : c.counts) {
    b.push_back({.count = count, .in_dist = 0, .out_dist = 0});
  }
  if (!c.runs.empty()) { // one batch dimension, at the stored extents
    b.back().in_dist = static_cast<std::ptrdiff_t>(extent(c, FFTN_IN));
    b.back().out_dist = static_cast<std::ptrdiff_t>(extent(c, FFTN_OUT));
  }
  return b;
}

template <class In, class Mid, class Out, class R> struct pipe_arrays {
  std::vector<R> in;
  std::vector<R> out;
  std::vector<R> sink;
  std::vector<R> pro_factor;
  std::vector<R> mid_factor;
  std::vector<R> epi_factor;
  cfuse_user pro_user{};
  cfuse_user mid_user{};
  cfuse_user epi_user{};

  explicit pipe_arrays(const pipe_case& c) {
    const std::size_t lines = product(c.counts);
    const std::size_t ends = is_complex<In> ? 2 : 1;
    in = values<R>(lines * extent(c, FFTN_IN) * ends, 2468U);
    const std::size_t out_reals = c.fanout * lines * extent(c, FFTN_OUT) * ends;
    out = c.epi.accumulate ? values<R>(out_reals, 1357U) : std::vector<R>(out_reals, untouched<R>);
    const auto factor = [&](const factor_spec& f, std::size_t length, std::size_t extra,
                            std::uint32_t seed) {
      if (f.kind == FFTN_FACTOR_NONE) {
        return std::vector<R>{};
      }
      return values<R>(factor_elements(f, {length}, c.counts, extra) *
                           (f.kind == FFTN_FACTOR_COMPLEX ? 2 : 1),
                       seed);
    };
    pro_factor = factor(c.pro.mul, end_length(c), 0, 11U);
    mid_factor = factor(c.mid.mul, mid_length(c), (c.fanout - 1) * mid_length(c), 22U);
    epi_factor = factor(c.epi.mul, end_length(c), 0, 33U);
    pro_user.type = type_of<In>();
    mid_user.type = type_of<Mid>();
    if (c.group != FFTN_NO_GROUP) { // the lines between members: the counts after the group's
      mid_user.group_step = product(std::vector<std::size_t>(
          c.counts.begin() + static_cast<std::ptrdiff_t>(c.group) + 1, c.counts.end()));
    }
    epi_user.type = type_of<Out>();
    if (c.epi.call == cb::consume) {
      sink.assign(c.fanout * lines * end_length(c) * ends, untouched<R>);
      epi_user.sink = sink.data();
      epi_user.strides[0] = 1;
      epi_user.dist = static_cast<std::ptrdiff_t>(end_length(c));
      epi_user.copy_dist = static_cast<std::ptrdiff_t>(lines * end_length(c));
    }
  }
  pipe_arrays(const pipe_arrays&) = delete;
  pipe_arrays& operator=(const pipe_arrays&) = delete;
  pipe_arrays(pipe_arrays&&) = delete;
  pipe_arrays& operator=(pipe_arrays&&) = delete;
  ~pipe_arrays() = default;

  [[nodiscard]] const void* in_ptr(const pipe_case& c) const {
    return c.pro.call == cb::produce ? nullptr : in.data();
  }
  [[nodiscard]] void* out_ptr(const pipe_case& c) {
    return c.epi.call == cb::consume ? nullptr : out.data();
  }
  [[nodiscard]] fftn_exec_args c_args() {
    return {.in_factor = pro_factor.empty() ? nullptr : pro_factor.data(),
            .mid_factor = mid_factor.empty() ? nullptr : mid_factor.data(),
            .out_factor = epi_factor.empty() ? nullptr : epi_factor.data(),
            .in_user = &pro_user,
            .mid_user = &mid_user,
            .out_user = &epi_user};
  }
};

template <class In, class Mid, class Out, class R> void check_pipeline(const pipe_case& c) {
  SCOPED_TRACE(c.name);
  const std::vector<fftn_batch> batches = pipe_batches(c);
  const std::size_t lines = product(c.counts);
  const auto out_dist = static_cast<std::ptrdiff_t>(lines * extent(c, FFTN_OUT));
  const auto factor_dist = static_cast<std::ptrdiff_t>(mid_length(c));

  // C++.
  fftn::pipeline_desc<In, Mid, Out> d;
  d.first = static_cast<fftn::direction>(c.first);
  d.line = {.n = c.n};
  for (const fftn_batch& b : batches) {
    d.batches.push_back({.count = b.count, .in_dist = b.in_dist, .out_dist = b.out_dist});
  }
  for (const fftn_stored_run& r : c.runs) {
    d.runs.push_back({.axis = r.axis,
                      .where = r.where == FFTN_IN ? fftn::side::in : fftn::side::out,
                      .logical = r.logical,
                      .stored = r.stored,
                      .count = r.count});
  }
  d.prologue = cpp_edge<In>(c.pro);
  d.middle.scale = c.mid.scale;
  d.middle.mul = cpp_factor(c.mid.mul);
  d.middle.call = cpp_callback<Mid>(c.mid.call, c.mid.scratch);
  d.epilogue = cpp_edge<Out>(c.epi);
  d.fanout = c.fanout;
  d.group = c.group;
  if (c.fanout > 1) {
    d.fanout_out_dist = out_dist;
    d.fanout_factor_dist = factor_dist;
  }
  const fftn::pipeline<In, Mid, Out> p(d);
  pipe_arrays<In, Mid, Out, R> expected(c);
  {
    fftn::workspace ws(p.workspace_bytes());
    const fftn_exec_args a = expected.c_args();
    p.execute(static_cast<const In*>(expected.in_ptr(c)), static_cast<Out*>(expected.out_ptr(c)),
              ws,
              {.in_factor = a.in_factor,
               .mid_factor = a.mid_factor,
               .out_factor = a.out_factor,
               .in_user = a.in_user,
               .mid_user = a.mid_user,
               .out_user = a.out_user});
  }

  // C, from the same description.
  fftn_pipeline_desc desc = fftn_pipeline_desc_default();
  desc.kind = c.kind;
  desc.precision = c.precision;
  desc.first = c.first;
  desc.line = {.n = c.n, .in_stride = 0, .out_stride = 0};
  desc.batch_rank = batches.size();
  desc.batches = batches.data();
  desc.run_count = c.runs.size();
  desc.runs = c.runs.empty() ? nullptr : c.runs.data();
  desc.prologue = c_edge(c.pro);
  desc.middle.scale = c.mid.scale;
  desc.middle.mul = c_factor(c.mid.mul);
  desc.middle.call = c_callback(c.mid.call, c.mid.scratch);
  desc.epilogue = c_edge(c.epi);
  desc.fanout = c.fanout;
  desc.group = c.group;
  if (c.fanout > 1) {
    desc.fanout_out_dist = out_dist;
    desc.fanout_factor_dist = factor_dist;
  }

  fftn_pipeline* pipeline = fftn_pipeline_create(&desc, nullptr, nullptr);
  ASSERT_NE(pipeline, nullptr);
  EXPECT_EQ(c_text(
                [](const void* o, char* b, std::size_t n) {
                  return fftn_pipeline_describe(static_cast<const fftn_pipeline*>(o), b, n);
                },
                pipeline),
            p.describe());
  EXPECT_EQ(fftn_pipeline_workspace_bytes(pipeline, 3), p.workspace_bytes(3));
  EXPECT_EQ(fftn_pipeline_instruction_set(pipeline), static_cast<int>(p.instruction_set()));
  EXPECT_EQ(fftn_pipeline_transform_count(pipeline), p.transform_count());
  EXPECT_EQ(p.transform_count(), c.group == FFTN_NO_GROUP ? lines : lines / c.counts[c.group]);
  for (int rank = 0; rank < 3; ++rank) {
    const fftn_transform_range r =
        fftn_pipeline_transforms_of(pipeline, {.rank = rank, .size = 3, .group_size = 0});
    const fftn::transform_range e = p.transforms_of({.rank = rank, .size = 3});
    EXPECT_EQ(r.first, e.first);
    EXPECT_EQ(r.count, e.count);
  }
  fftn_pipeline_release(pipeline);

  for (const form_case& f : forms) {
    SCOPED_TRACE(f.name);
    pipe_arrays<In, Mid, Out, R> got(c);
    const fftn_exec_args args = got.c_args();
    fftn_error error{};
    const int status = cfuse_run_pipeline(&desc, nullptr, f.form, f.lanes, f.group_size,
                                          got.in_ptr(c), got.out_ptr(c), &args, &error);
    ASSERT_EQ(status, FFTN_OK) << error.message;
    EXPECT_TRUE(same_bits(got.out, expected.out));
    EXPECT_TRUE(same_bits(got.sink, expected.sink));
  }
}

template <class R> void check_pipeline_kind(const pipe_case& c) {
  using C = std::complex<R>;
  switch (c.kind) {
  case FFTN_PIPELINE_CCC:
    check_pipeline<C, C, C, R>(c);
    break;
  case FFTN_PIPELINE_TCT:
    check_pipeline<R, C, R, R>(c);
    break;
  case FFTN_PIPELINE_CTC:
    check_pipeline<C, R, C, R>(c);
    break;
  }
}

void check_pipeline_case(const pipe_case& c) {
  if (c.precision == FFTN_DOUBLE) {
    check_pipeline_kind<double>(c);
  } else {
    check_pipeline_kind<float>(c);
  }
}

const factor_spec kernel{.kind = FFTN_FACTOR_COMPLEX, .axis = {1}, .batch = {}};

TEST(CFusion, PipelineKindsMatchCpp) {
  check_pipeline_case({.name = "<C, C, C> forward first: the vocabulary and a middle callback",
                       .kind = FFTN_PIPELINE_CCC,
                       .precision = FFTN_DOUBLE,
                       .n = 48,
                       .counts = {5},
                       .pro = {.scale = 0.5},
                       .mid = {.mul = kernel, .call = cb::modify},
                       .epi = {.mul = {.kind = FFTN_FACTOR_REAL, .axis = {1}, .batch = {0}}}});
  check_pipeline_case({.name = "<C, C, C> backward first in float: a middle callback, consumed",
                       .kind = FFTN_PIPELINE_CCC,
                       .precision = FFTN_FLOAT,
                       .first = FFTN_BACKWARD,
                       .n = 32,
                       .counts = {4},
                       .mid = {.scale = 1.0 / 32, .call = cb::modify, .scratch = 32},
                       .epi = {.call = cb::consume}});
  check_pipeline_case({.name = "<T, C, T>: produced, a kernel per line, accumulated",
                       .kind = FFTN_PIPELINE_TCT,
                       .precision = FFTN_DOUBLE,
                       .n = 64,
                       .counts = {3},
                       .pro = {.call = cb::produce},
                       .mid = {.mul = {.kind = FFTN_FACTOR_COMPLEX, .axis = {1}, .batch = {33}}},
                       .epi = {.accumulate = true}});
  check_pipeline_case(
      {.name = "<C, T, C> in float: modify in, a real middle with a callback",
       .kind = FFTN_PIPELINE_CTC,
       .precision = FFTN_FLOAT,
       .n = 30,
       .counts = {4},
       .pro = {.call = cb::modify},
       .mid = {.mul = {.kind = FFTN_FACTOR_REAL, .axis = {1}, .batch = {}}, .call = cb::modify}});
  check_pipeline_case({.name = "<C, C, C> in float: two batch dimensions, a prologue factor",
                       .kind = FFTN_PIPELINE_CCC,
                       .precision = FFTN_FLOAT,
                       .n = 20,
                       .counts = {2, 3},
                       .pro = {.mul = {.kind = FFTN_FACTOR_COMPLEX, .axis = {1}, .batch = {0, 20}}},
                       .mid = {.mul = kernel},
                       .epi = {.accumulate = true}});
}

TEST(CFusion, FanOutAndRunsMatchCpp) {
  check_pipeline_case({.name = "<C, C, C> fan-out of 3, an epilogue callback per copy",
                       .kind = FFTN_PIPELINE_CCC,
                       .precision = FFTN_DOUBLE,
                       .n = 24,
                       .counts = {4},
                       .mid = {.mul = kernel},
                       .epi = {.call = cb::modify},
                       .fanout = 3});
  check_pipeline_case({.name = "<C, C, C> fan-out of 2 in float, consumed per copy",
                       .kind = FFTN_PIPELINE_CCC,
                       .precision = FFTN_FLOAT,
                       .n = 16,
                       .counts = {3},
                       .mid = {.mul = kernel, .call = cb::modify},
                       .epi = {.call = cb::consume},
                       .fanout = 2});
  check_pipeline_case(
      {.name = "<T, C, T> zero-padded: input and output runs",
       .kind = FFTN_PIPELINE_TCT,
       .precision = FFTN_DOUBLE,
       .n = 64,
       .counts = {3},
       .mid = {.mul = kernel},
       .runs = {{.axis = 0, .where = FFTN_IN, .logical = 0, .stored = 0, .count = 40},
                {.axis = 0, .where = FFTN_OUT, .logical = 8, .stored = 0, .count = 32}}});
}

TEST(CFusion, GroupsMatchCpp) {
  check_pipeline_case({.name = "<C, C, C> grouped along the inner batch dimension: a coupled "
                               "middle callback",
                       .kind = FFTN_PIPELINE_CCC,
                       .precision = FFTN_DOUBLE,
                       .n = 16,
                       .counts = {3, 2},
                       .mid = {.mul = kernel, .call = cb::modify},
                       .epi = {.call = cb::modify},
                       .group = 1});
  check_pipeline_case({.name = "<T, C, T> in float grouped along the outer batch dimension",
                       .kind = FFTN_PIPELINE_TCT,
                       .precision = FFTN_FLOAT,
                       .n = 24,
                       .counts = {2, 4},
                       .pro = {.call = cb::produce},
                       .mid = {.call = cb::modify, .scratch = 16},
                       .group = 0});
  check_pipeline_case({.name = "<C, T, C> grouped in threes, consumed",
                       .kind = FFTN_PIPELINE_CTC,
                       .precision = FFTN_DOUBLE,
                       .n = 18,
                       .counts = {3, 2},
                       .mid = {.call = cb::modify},
                       .epi = {.call = cb::consume},
                       .group = 0});
}

TEST(CFusion, RefusalsAndDefaultsFromC) {
  EXPECT_EQ(cfuse_refusals(), 0) << "the check of that number failed";
}

TEST(CFusion, DefaultsAreCppDefaults) {
  const fftn_edge e = fftn_edge_default();
  const fftn::edge<double> ce{};
  EXPECT_EQ(e.scale, ce.scale);
  EXPECT_EQ(e.accumulate != 0, ce.accumulate);
  const fftn_pipeline_desc d = fftn_pipeline_desc_default();
  const fftn::pipeline_desc<std::complex<double>, std::complex<double>, std::complex<double>> cd{};
  EXPECT_EQ(d.first, static_cast<int>(cd.first));
  EXPECT_EQ(d.middle.scale, cd.middle.scale);
  EXPECT_EQ(d.fanout, cd.fanout);
  EXPECT_EQ(d.fanout_out_dist, cd.fanout_out_dist);
  EXPECT_EQ(d.fanout_factor_dist, cd.fanout_factor_dist);
  EXPECT_EQ(d.group, cd.group);
}

// ---- The boundary ----

// A plan of complex rows whose output edge has the callback `fn`.
fftn_plan* rows_with_callback(fftn_block_fn fn) {
  static const fftn_axis axis = {.n = 16, .in_stride = 0, .out_stride = 0};
  static const fftn_batch batch = {.count = 2, .in_dist = 0, .out_dist = 0};
  const fftn_shape shape = {.kind = FFTN_C2C,
                            .precision = FFTN_DOUBLE,
                            .direction = FFTN_FORWARD,
                            .rank = 1,
                            .axes = &axis,
                            .batch_rank = 1,
                            .batches = &batch,
                            .r2r_kinds = nullptr,
                            .run_count = 0,
                            .runs = nullptr,
                            .layout = FFTN_INTERLEAVED};
  fftn_plan_edges edges = {.in = fftn_edge_default(), .out = fftn_edge_default()};
  edges.out.call.fn = fn;
  return fftn_plan_create_edges(&shape, nullptr, &edges, nullptr);
}

void run_rows(const fftn_plan* plan, bool with_args) {
  std::vector<double> x(64, 0.25);
  std::vector<double> y(64);
  fftn_workspace* ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, 1), nullptr);
  if (with_args) {
    fftn_execute_args(plan, x.data(), y.data(), ws, nullptr);
  } else {
    fftn_execute(plan, x.data(), y.data(), ws);
  }
  fftn_workspace_destroy(ws);
}

// A "C" callback written in C++ that throws: the exception ends the program at the boundary
// and never reaches the caller of the execution.
void throwing_callback(void* /*user*/, const fftn_block_info* /*info*/, void* /*data*/) {
  throw std::runtime_error("a callback threw");
}

TEST(CFusionDeathTest, AnExceptionFromACallbackEndsTheProgram) {
  fftn_plan* plan = rows_with_callback(throwing_callback);
  ASSERT_NE(plan, nullptr);
  EXPECT_DEATH(
      {
        try {
          run_rows(plan, true);
        } catch (...) {
          std::exit(0); // reaching the caller would be a failure: exit without dying
        }
      },
      "");
  fftn_plan_release(plan);
}

TEST(CFusionDeathTest, ACallbackPlanWithoutArgumentsEndsTheProgram) {
  fftn_plan* plan = rows_with_callback(cfuse_modify);
  ASSERT_NE(plan, nullptr);
  EXPECT_DEATH(run_rows(plan, false), "without exec_args");
  fftn_plan_release(plan);
}

// A callback that fails records it in the application's memory and returns; the execution runs
// to its end and the caller checks the record (c_api.h, fftn_block_fn).
void failing_callback(void* user, const fftn_block_info* /*info*/, void* /*data*/) {
  *static_cast<int*>(user) = 1;
}

TEST(CFusion, ACallbackReportsItsFailureThroughUser) {
  fftn_plan* plan = rows_with_callback(failing_callback);
  ASSERT_NE(plan, nullptr);
  std::vector<double> x(64, 0.25);
  std::vector<double> y(64);
  int failed = 0;
  fftn_exec_args args{};
  args.out_user = &failed;
  fftn_workspace* ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, 1), nullptr);
  fftn_execute_args(plan, x.data(), y.data(), ws, &args);
  EXPECT_EQ(failed, 1);
  fftn_workspace_destroy(ws);
  fftn_plan_release(plan);
}

} // namespace
