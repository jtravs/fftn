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
  s.profile.pipeline_depth_min_kib = 0; // X67's rules off
  s.profile.pipeline_lanes_adjacent_max = 0;
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

// X67's lanes rule (machine_profile::pipeline_lanes_adjacent_max): a plane's strided lines (one
// element apart, the line's stride their count) take the lanes pair by the rules up to the
// profile's length, the forced pair's program and bits; above it, with the field off or with a
// real kernel they keep the chain pair; a batch of rows never takes it by the rules.
TEST(PipelinePlanning, LanesRuleForAdjacentLines) {
  const std::size_t n = 64;
  const std::size_t lines = 64;
  conv<double> c(n, lines);
  c.desc.line.in_stride = c.desc.line.out_stride = static_cast<std::ptrdiff_t>(lines);
  c.desc.batches[0].in_dist = c.desc.batches[0].out_dist = 1;
  build_settings rules = fixed_settings();
  rules.profile.pipeline_lanes_adjacent_max = 512;
  const auto by_rules = d::make_pipeline_core(c.view(), {}, rules);
  EXPECT_EQ(by_rules->built, strategy_choice::lanes) << d::describe(*by_rules);
  EXPECT_NE(d::describe(*by_rules).find("by the rules for a plane's strided lines"),
            std::string::npos);
  build_settings forced = rules;
  forced.strategy = strategy_choice::lanes;
  const auto by_force = d::make_pipeline_core(c.view(), {}, forced);
  EXPECT_EQ(d::program_fingerprint(*by_rules), d::program_fingerprint(*by_force));
  EXPECT_TRUE(same_bits(run<double>(*by_rules, n, lines), run<double>(*by_force, n, lines)));

  build_settings shorter = rules;
  shorter.profile.pipeline_lanes_adjacent_max = 32;
  build_settings off = rules;
  off.profile.pipeline_lanes_adjacent_max = 0;
  for (const build_settings* s : {&shorter, &off}) {
    EXPECT_EQ(d::make_pipeline_core(c.view(), {}, *s)->built, strategy_choice::chain);
  }
  conv<double> real_kernel = c;
  real_kernel.desc.middle.mul = {.kind = fftn::factor_kind::real, .axis_strides = {1}};
  EXPECT_EQ(d::make_pipeline_core(real_kernel.view(), {}, rules)->built, strategy_choice::chain);
  const conv<double> rows(n, lines);
  EXPECT_EQ(d::make_pipeline_core(rows.view(), {}, rules)->built, strategy_choice::chain);
}

} // namespace
