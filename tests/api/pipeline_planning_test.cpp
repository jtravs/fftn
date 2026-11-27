// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Measured planning and wisdom for pipelines (task 5.2g, DESIGN.md 12.2; experiment X67): the
// candidates are the rules' pair and the chain, depth-first and lanes pairs where the description
// can take each; the fastest is kept; wisdom records the program keyed by the description and
// rebuilds it untimed, bitwise; an entry for another description is not used.

#include "core/candidates.hpp"
#include "core/pipeline.hpp"
#include "core/plan_core.hpp"
#include "core/wisdom_entry.hpp"
#include "fftn/callback.hpp"
#include "fftn/detail/core.hpp"
#include "support/inputs.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

namespace d = fftn::detail;
using d::build_settings;
using d::strategy_choice;

// A stand-in clock (build_settings::measure_time): each choice's seconds, 1 for the others, and
// the choices asked in order.
struct fake_clock {
  std::map<std::string, double> seconds;
  std::vector<std::string> asked;
};

double fake_time(const std::string& choice, void* context) {
  auto& f = *static_cast<fake_clock*>(context);
  f.asked.push_back(choice);
  const auto it = f.seconds.find(choice);
  return it == f.seconds.end() ? 1.0 : it->second;
}

// Settings that hold whatever the host is: the depth-first pair and the lanes can be built
// (forced), the rules take the chain pair.
build_settings fixed_settings() {
  build_settings s;
  s.profile.depth_convolution_max_bytes = 0;
  return s;
}

template <class T> struct conv {
  using C = std::complex<T>;
  fftn::pipeline_desc<C, C, C> desc;

  conv(std::size_t n, std::size_t count) : desc(fftn::complex_convolution_rows<T>(n, count)) {}

  [[nodiscard]] d::pipeline_view view() const {
    return {
        .kind = d::pipeline_kind::complex_pair,
        .prec = sizeof(T) == 8 ? d::precision::f64 : d::precision::f32,
        .first = desc.first,
        .line = desc.line,
        .batches = desc.batches,
        .runs = desc.runs,
        .prologue = d::view_of(desc.prologue),
        .middle = {.scale = desc.middle.scale,
                   .kind = desc.middle.mul.kind,
                   .axis_strides = desc.middle.mul.axis_strides,
                   .batch_strides = desc.middle.mul.batch_strides},
        .epilogue = d::view_of(desc.epilogue),
        .fanout = desc.fanout,
        .fanout_out_dist = desc.fanout_out_dist,
        .fanout_factor_dist = desc.fanout_factor_dist,
    };
  }
};

std::shared_ptr<const d::plan_core> measured(const d::pipeline_view& v, fake_clock& clock,
                                             const fftn::options& opts = {
                                                 .effort = fftn::planning::quick}) {
  build_settings s = fixed_settings();
  s.measure_time = &fake_time;
  s.measure_context = &clock;
  return d::make_pipeline_core(v, opts, s);
}

bool asked(const fake_clock& c, const std::string& choice) {
  return std::ranges::find(c.asked, choice) != c.asked.end();
}

// One execution's output: every line convolved with a kernel of 1/n (a scale only, through the
// middle's factor).
template <class T>
std::vector<std::complex<T>> run(const d::plan_core& core, std::size_t n, std::size_t count) {
  using C = std::complex<T>;
  fftn::test::input_source src(0x5A2CULL + n);
  const std::vector<C> x = src.complexes<T>(n * count);
  std::vector<C> kernel(n, C(T(1) / static_cast<T>(n), T(0)));
  std::vector<C> y(n * count);
  fftn::workspace ws(d::workspace_bytes(core, 1));
  d::execute_all(core, x.data(), y.data(), ws, fftn::exec_args{.mid_factor = kernel.data()});
  return y;
}

template <class V> bool same_bits(const V& a, const V& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

// A long contiguous line: the rules' chain pair and the depth-first pair are candidates (the
// lanes pair, which one line cannot take, is not); the fastest is kept and named.
TEST(PipelinePlanning, KeepsTheFastestPair) {
  const conv<double> c(32768, 1);
  fake_clock clock;
  const auto rules = d::make_pipeline_core(c.view(), {}, fixed_settings());
  ASSERT_EQ(rules->built, strategy_choice::chain);
  EXPECT_EQ(rules->program_text, "rules");
  const auto depth_tops = [] {
    build_settings s = fixed_settings();
    s.strategy = strategy_choice::depth;
    return s;
  }();
  const auto depth = d::make_pipeline_core(c.view(), {}, depth_tops);
  ASSERT_EQ(depth->built, strategy_choice::depth);
  // The clock is asked by the candidate's choice; the plan kept records the tops it built.
  clock.seconds["strategy=depth"] = 0.5;
  const auto core = measured(c.view(), clock);
  EXPECT_TRUE(asked(clock, "rules"));
  EXPECT_TRUE(asked(clock, "strategy=depth"));
  EXPECT_FALSE(asked(clock, "strategy=lanes"));
  EXPECT_EQ(core->built, strategy_choice::depth);
  EXPECT_EQ(core->program_text, depth->program_text);
  const std::string text = d::describe(*core);
  EXPECT_NE(text.find("planning: measured"), std::string::npos) << text;
  EXPECT_NE(text.find("the depth-first pair"), std::string::npos) << text;
  EXPECT_EQ(d::program_fingerprint(*core), d::program_fingerprint(*depth));
}

// The rules' choice stays unless another is clearly faster; a batch of short lines has the lanes
// pair as a candidate, and the chain pair forced builds the rules' program, so it is not timed
// twice.
TEST(PipelinePlanning, LanesCandidateAndTheRulesKept) {
  const conv<double> c(64, 64);
  fake_clock clock;
  const auto core = measured(c.view(), clock);
  EXPECT_TRUE(asked(clock, "strategy=lanes"));
  EXPECT_FALSE(asked(clock, "strategy=chain"));
  EXPECT_EQ(core->built, strategy_choice::chain);
  EXPECT_EQ(core->program_text, "rules");

  fake_clock faster;
  faster.seconds["strategy=lanes"] = 0.5;
  const auto lanes = measured(c.view(), faster);
  EXPECT_EQ(lanes->built, strategy_choice::lanes);
  EXPECT_EQ(lanes->program_text, "strategy=lanes");
}

// Wisdom records a measured pipeline's program and rebuilds it untimed, with the same bits; an
// entry is keyed by the description, so a pipeline of another kernel layout does not use it.
TEST(PipelinePlanning, WisdomRoundTrip) {
  const std::size_t n = 32768;
  conv<double> c(n, 1);
  fake_clock clock;
  clock.seconds["strategy=depth"] = 0.25;
  const auto core = measured(c.view(), clock);
  ASSERT_EQ(core->built, strategy_choice::depth);

  // Recorded through the core's record (as wisdom::add(pipeline) does), then read back.
  const fftn::wisdom w(d::record_text(d::record_of(*core)));
  ASSERT_EQ(w.size(), 1U);
  fake_clock unused;
  build_settings s = fixed_settings();
  s.measure_time = &fake_time;
  s.measure_context = &unused;
  for (const fftn::planning effort : {fftn::planning::rules, fftn::planning::measure}) {
    const auto again = d::make_pipeline_core(c.view(), {.effort = effort, .use = &w}, s);
    EXPECT_TRUE(unused.asked.empty());
    EXPECT_EQ(again->built, strategy_choice::depth);
    EXPECT_NE(d::describe(*again).find("planning: from wisdom"), std::string::npos);
    EXPECT_TRUE(same_bits(run<double>(*again, n, 1), run<double>(*core, n, 1)));
  }
  // Another description (the middle's scale): the entry does not apply, the rules' pair is built.
  c.desc.middle.scale = 2.0;
  const auto other = d::make_pipeline_core(c.view(), {.use = &w}, fixed_settings());
  EXPECT_EQ(other->built, strategy_choice::chain);
  EXPECT_EQ(d::describe(*other).find("from wisdom"), std::string::npos);
}

// Through the public API: wisdom::add(pipeline) and options::use, a measured pipeline's output
// equal to the rebuilt one's bitwise and to the rules' within the derived tolerance.
TEST(PipelinePlanning, PublicMeasuredAndWisdom) {
  using C = std::complex<double>;
  const std::size_t n = 4096;
  const std::size_t count = 4;
  const auto desc = fftn::complex_convolution_rows<double>(n, count);
  const fftn::pipeline<C, C, C> rules(desc);
  const fftn::pipeline<C, C, C> quick(desc, {.effort = fftn::planning::quick});
  EXPECT_NE(quick.describe().find("planning: measured"), std::string::npos) << quick.describe();
  EXPECT_EQ(rules.describe().find("planning:"), std::string::npos) << rules.describe();
  fftn::wisdom w;
  w.add(quick);
  ASSERT_EQ(w.size(), 1U);
  const fftn::pipeline<C, C, C> again(desc, {.use = &w});
  EXPECT_NE(again.describe().find("planning: from wisdom"), std::string::npos);

  fftn::test::input_source src(0x77ULL);
  const std::vector<C> x = src.complexes<double>(n * count);
  const std::vector<C> kernel(n, C(1.0 / static_cast<double>(n), 0.0));
  const fftn::exec_args args{.mid_factor = kernel.data()};
  const auto out = [&](const fftn::pipeline<C, C, C>& p) {
    std::vector<C> y(n * count);
    fftn::workspace ws(p);
    p.execute(x.data(), y.data(), ws, args);
    return y;
  };
  const std::vector<C> a = out(quick);
  EXPECT_TRUE(same_bits(a, out(again)));
  const std::vector<C> b = out(rules);
  double err = 0.0;
  double ref = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    err += std::norm(a[i] - b[i]);
    ref += std::norm(x[i]);
  }
  // Each output within two transforms' bound, 20 eps (1 + log2 n) relative (fbench's delayed
  // check), of the exact result: the two outputs within twice that of each other.
  const double eps = std::numeric_limits<double>::epsilon();
  const double tol = 2 * 20 * eps * (1 + std::log2(static_cast<double>(n)));
  EXPECT_LE(std::sqrt(err / ref), tol) << "measured against the rules";
}

// The wisdom key holds every scale: two pipelines that differ only in the epilogue's or the
// middle's scale (no factor, callback or accumulation beside it) have different keys, so an entry
// of one is never used for the other.
TEST(PipelinePlanning, KeyHoldsTheScales) {
  conv<double> a(64, 4);
  a.desc.middle.mul = {};
  conv<double> b = a;
  b.desc.epilogue.scale = 0.5;
  conv<double> m = a;
  m.desc.middle.scale = 2.0;
  conv<double> pre = a;
  pre.desc.prologue.scale = 2.0;
  const std::string ka = d::pipeline_key(a.view(), {});
  EXPECT_NE(ka, d::pipeline_key(b.view(), {}));
  EXPECT_NE(ka, d::pipeline_key(m.view(), {}));
  EXPECT_NE(ka, d::pipeline_key(pre.view(), {}));
  EXPECT_EQ(ka, d::pipeline_key(conv<double>(a).view(), {}));
}

template <class E> void stand_in_never_called(void*, const fftn::block_info&, E*) noexcept {
  ADD_FAILURE() << "planning called the application's callback";
}

// Measured planning on the real clock over descriptions whose scratch factor arrays are not one
// line of n (the spans of pipeline_need): fan-out with a kernel per copy, the half spectrum of
// <T, C, T> and <C, T, C>, kernels per line and negative strides, prologue and epilogue factors,
// stored runs, a middle callback (a stand-in that does nothing) and a group. Run under ASan in CI:
// every read of the scratch arrays stays inside them.
TEST(PipelinePlanning, ScratchSpansOfEveryShape) {
  using C = std::complex<double>;
  const fftn::options quick{.effort = fftn::planning::quick, .measure_seconds = 0.02};
  const std::size_t n = 64;
  {
    auto dsc = fftn::complex_convolution_rows<double>(n, 3);
    dsc.fanout = 3;
    dsc.fanout_out_dist = static_cast<std::ptrdiff_t>(3 * n + 5);
    dsc.fanout_factor_dist = static_cast<std::ptrdiff_t>(n + 1);
    dsc.middle.mul.batch_strides = {static_cast<std::ptrdiff_t>(3 * (n + 1))};
    const fftn::pipeline<C, C, C> p(dsc, quick);
    EXPECT_NE(p.describe().find("planning: measured"), std::string::npos);
  }
  {
    auto dsc = fftn::convolution_rows<double>(n, 3);
    dsc.middle.mul.batch_strides = {static_cast<std::ptrdiff_t>(n / 2 + 1)};
    dsc.prologue.mul = {.kind = fftn::factor_kind::real, .axis_strides = {-1}};
    dsc.epilogue.mul = {.kind = fftn::factor_kind::real,
                        .axis_strides = {1},
                        .batch_strides = {static_cast<std::ptrdiff_t>(n)}};
    const fftn::pipeline<double, C, double> p(dsc, quick);
    EXPECT_NE(p.describe().find("planning: measured"), std::string::npos);
  }
  {
    fftn::pipeline_desc<C, double, C> dsc;
    dsc.line = {.n = n};
    dsc.batches = {{.count = 3}};
    dsc.middle.mul = {.kind = fftn::factor_kind::real,
                      .axis_strides = {1},
                      .batch_strides = {static_cast<std::ptrdiff_t>(n)}};
    dsc.prologue.mul = {.kind = fftn::factor_kind::complex, .axis_strides = {1}};
    const fftn::pipeline<C, double, C> p(dsc, quick);
    EXPECT_NE(p.describe().find("planning: measured"), std::string::npos);
  }
  {
    const std::vector<fftn::stored_run> runs{
        {.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = n / 2},
        {.axis = 0, .where = fftn::side::out, .logical = n / 4, .stored = 0, .count = n / 2}};
    auto dsc = fftn::complex_convolution_rows<double>(n, 3, runs);
    dsc.middle.mul.axis_strides = {-1};
    dsc.middle.call = {.fn = &stand_in_never_called<C>};
    const fftn::pipeline<C, C, C> p(dsc, quick);
    EXPECT_NE(p.describe().find("planning: measured"), std::string::npos);
  }
  {
    auto dsc = fftn::complex_convolution_rows<double>(n, 4);
    dsc.batches = {{.count = 2}, {.count = 3}};
    dsc.group = 1;
    dsc.middle.call = {.fn = &stand_in_never_called<C>};
    dsc.middle.mul.batch_strides = {0, static_cast<std::ptrdiff_t>(n)};
    const fftn::pipeline<C, C, C> p(dsc, quick);
    EXPECT_NE(p.describe().find("planning: measured"), std::string::npos);
  }
  {
    // In place without a middle factor: an execution multiplies the line by n; the samples stay
    // finite (reps_cap counts the pipeline's two transforms).
    auto dsc = fftn::complex_convolution_rows<double>(n, 3);
    dsc.middle.mul = {};
    const fftn::pipeline<C, C, C> p(dsc, {.in_place = true, .effort = fftn::planning::quick});
    EXPECT_NE(p.describe().find("planning: measured"), std::string::npos);
  }
}

} // namespace
