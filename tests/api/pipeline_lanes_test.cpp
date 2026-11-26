// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The lanes pair of <C, C, C> pipelines (DESIGN.md 12.2; task 5.2f), forced: short rows of a batch
// as the lanes over rows (a gather where a run of C rows is no vector access), and lines of a
// strided axis whose innermost batch dimension is adjacent as the lanes over adjacent lines (the
// columns of a plane, of a volume's planes). Every line bitwise equal to the chain pair's with the
// same placement of the vocabulary (the lanes' contract): a real kernel or a complex kernel of a
// stride other than 1, each per line or one for every line, a scale alone, the backward transform
// first, in place, a pure middle callback beside the two scale products. A complex kernel of
// stride 1, which the chain pair fuses into its forward chain's last pass, within the derived
// tolerance of it. Blocks of every size down to the tail of C lines and fewer; canaries between
// the lines; the input and the kernel left as they were; the middle callback's blocks (the
// block's whole lines on the tile) and its coverage; serial, pieces of the batch, teams
// (completion::all and own) and an executor bitwise alike; on every instruction set in both
// precisions. What the pair does not take yet falls back to the chain pair, as describe() says,
// and by the rules no pipeline takes the pair. Expectations hold on every target: none assumes a
// target's radices or strategy beyond the forced pair, whose lengths are codelet powers of two.

#include "core/pipeline.hpp"
#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/tolerance.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace d = fftn::detail;
using fftn::factor_kind;
constexpr factor_kind none = factor_kind::none;
constexpr factor_kind real = factor_kind::real;
constexpr factor_kind cplx = factor_kind::complex;

template <class V> bool same_bits(const V& a, const V& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

struct lanes_case {
  const char* name = "";
  std::size_t n = 8;
  std::vector<std::size_t> counts{13}; // the batch dimensions, the innermost last
  // Rows: the line's strides (lines n |stride| + 1 apart). Adjacent: the innermost dimension's
  // lines one element apart, the line's stride its count plus one (input) or two (output).
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t out_stride = 1;
  bool adjacent = false;
  factor_kind mid = real;
  std::ptrdiff_t mid_axis = 1;
  bool per_line = false;  // a kernel per line, else one for every line (batch_strides empty)
  double mid_scale = 0.0; // 0: 1/n
  double pro_scale = 1.0;
  double options_scale = 1.0;
  bool backward = false; // the backward transform first (a spectral pair)
  bool in_place = false;
  bool callback = false;        // a pure middle callback (modify)
  std::size_t lane_block = 0;   // build_settings::lane_block (0: the rule's)
  std::size_t column_block = 0; // build_settings::column_block
  bool threads = false;
  // The chain pair's bits: false where it fuses the kernel (complex, stride 1), whose last bits
  // the tile's separate pass does not reproduce.
  bool bitwise = true;
};

// The middle callback's state: a pure, nonlinear, coordinate-dependent function of each bin, and
// the count of each (line, bin) it is passed.
struct callback_state {
  std::size_t n = 0;
  std::size_t lines = 0;
  std::vector<std::atomic<int>>* seen = nullptr; // null: no counting (teams)
  bool geometry_ok = true;
};

template <class T>
void middle_callback(void* user, const fftn::block_info& info, std::complex<T>* data) noexcept {
  auto& st = *static_cast<callback_state*>(user);
  if (info.element_axis != 0 || info.line_axis != fftn::along_transforms || info.length != st.n ||
      info.origin.size() != 1 || info.origin[0] != 0 || info.group != 1) {
    st.geometry_ok = false;
  }
  for (std::size_t k = 0; k < info.lines; ++k) {
    for (std::size_t i = 0; i < info.length; ++i) {
      std::complex<T>& v = data[static_cast<std::ptrdiff_t>(i) * info.element_stride +
                                static_cast<std::ptrdiff_t>(k) * info.line_stride];
      const T w = T(1) + T(0.001) * static_cast<T>(i);
      v = v * w + T(0.0001) * v * v * v;
      if (st.seen != nullptr) {
        const std::size_t line = info.transform + k;
        if (line < st.lines) {
          (*st.seen)[line * st.n + i].fetch_add(1, std::memory_order_relaxed);
        } else {
          st.geometry_ok = false;
        }
      }
    }
  }
}

template <class T> struct lanes_run {
  using C = std::complex<T>;
  lanes_case c;
  fftn::isa target;
  fftn::pipeline_desc<C, C, C> desc;
  std::vector<std::ptrdiff_t> in_dist;
  std::vector<std::ptrdiff_t> out_dist;
  // Element 0 of line 0 in each buffer: past the line's other elements for a negative stride.
  std::size_t in_base = 0;
  std::size_t out_base = 0;
  std::size_t lines = 1;
  std::vector<C> x;
  std::vector<C> init;
  std::vector<bool> written; // output elements that are lines' elements
  std::vector<T> kreal;
  std::vector<C> kcomplex;
  std::ptrdiff_t kzero = 0;
  std::vector<std::ptrdiff_t> kbatch;
  callback_state state;

  lanes_run(const lanes_case& lc, fftn::isa t, fftn::test::input_source& src) : c(lc), target(t) {
    const std::size_t dims = c.counts.size();
    const std::size_t inner = c.counts.back();
    std::ptrdiff_t in_stride = c.in_stride;
    std::ptrdiff_t out_stride = c.out_stride;
    in_dist.assign(dims, 0);
    out_dist.assign(dims, 0);
    const auto n = static_cast<std::ptrdiff_t>(c.n);
    if (c.adjacent) {
      in_stride = static_cast<std::ptrdiff_t>(inner) + 1;
      out_stride = c.in_place ? in_stride : static_cast<std::ptrdiff_t>(inner) + 2;
      in_dist.back() = 1;
      out_dist.back() = 1;
    } else {
      in_dist.back() = n * std::abs(in_stride) + 1;
      out_dist.back() = c.in_place ? in_dist.back() : n * std::abs(out_stride) + 1;
    }
    if (c.in_place) {
      out_stride = in_stride;
    }
    const auto base = [&](std::ptrdiff_t stride) {
      return stride < 0 ? static_cast<std::size_t>((n - 1) * -stride) : std::size_t{0};
    };
    in_base = base(in_stride);
    out_base = base(out_stride);
    // Outer dimensions: past the inner ones' reach, with a gap.
    const auto reach = [&](std::ptrdiff_t stride, const std::vector<std::ptrdiff_t>& dist,
                           std::size_t from) {
      std::ptrdiff_t r = (n - 1) * std::abs(stride);
      for (std::size_t k = from; k < dims; ++k) {
        r += static_cast<std::ptrdiff_t>(c.counts[k] - 1) * std::abs(dist[k]);
      }
      return r + 1;
    };
    for (std::size_t k = dims - 1; k-- > 0;) {
      in_dist[k] = reach(in_stride, in_dist, k + 1) + 2;
      out_dist[k] = c.in_place ? in_dist[k] : reach(out_stride, out_dist, k + 1) + 3;
    }
    for (const std::size_t count : c.counts) {
      lines *= count;
    }
    desc.first = c.backward ? fftn::direction::backward : fftn::direction::forward;
    desc.line = {.n = c.n, .in_stride = in_stride, .out_stride = out_stride};
    for (std::size_t k = 0; k < dims; ++k) {
      desc.batches.push_back(
          {.count = c.counts[k], .in_dist = in_dist[k], .out_dist = out_dist[k]});
    }
    desc.prologue.scale = c.pro_scale;
    desc.middle.scale = c.mid_scale != 0.0 ? c.mid_scale : 1.0 / static_cast<double>(c.n);
    if (c.mid != none) {
      // A kernel per line: the innermost dimension's lines n |axis| apart, the outer ones past it.
      std::ptrdiff_t step = std::abs(c.mid_axis) * n;
      kbatch.assign(dims, 0);
      std::ptrdiff_t span = (n - 1) * std::abs(c.mid_axis);
      for (std::size_t k = dims; k-- > 0;) {
        if (c.per_line) {
          kbatch[k] = step;
          span += step * static_cast<std::ptrdiff_t>(c.counts[k] - 1);
          step *= static_cast<std::ptrdiff_t>(c.counts[k]);
        }
      }
      kzero = c.mid_axis < 0 ? (n - 1) * -c.mid_axis : 0;
      const auto size = static_cast<std::size_t>(span + 1);
      if (c.mid == real) {
        kreal = src.reals<T>(size);
      } else {
        kcomplex = src.complexes<T>(size);
      }
      desc.middle.mul = {.kind = c.mid,
                         .axis_strides = {c.mid_axis},
                         .batch_strides = c.per_line ? kbatch : std::vector<std::ptrdiff_t>{}};
    }
    if (c.callback) {
      desc.middle.call = {.fn = &middle_callback<T>};
    }
    const auto size = [&](std::ptrdiff_t stride, const std::vector<std::ptrdiff_t>& dist) {
      return static_cast<std::size_t>(reach(stride, dist, 0) + 4);
    };
    x = src.complexes<T>(size(in_stride, in_dist));
    init = c.in_place ? x : src.complexes<T>(size(out_stride, out_dist));
    written.assign(init.size(), false);
    for (std::size_t line = 0; line < lines; ++line) {
      for (std::size_t j = 0; j < c.n; ++j) {
        written[out_at(j, line)] = true;
      }
    }
    state.n = c.n;
    state.lines = lines;
  }

  [[nodiscard]] std::size_t out_at(std::size_t j, std::size_t line) const {
    std::ptrdiff_t at = static_cast<std::ptrdiff_t>(out_base) +
                        static_cast<std::ptrdiff_t>(j) * desc.line.out_stride;
    std::size_t rest = line;
    for (std::size_t k = c.counts.size(); k-- > 0;) {
      at += static_cast<std::ptrdiff_t>(rest % c.counts[k]) * out_dist[k];
      rest /= c.counts[k];
    }
    return static_cast<std::size_t>(at);
  }

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
                   .batch_strides = desc.middle.mul.batch_strides,
                   .call = reinterpret_cast<d::erased_block_fn>(desc.middle.call.fn),
                   .call_mode = desc.middle.call.mode,
                   .call_scratch = desc.middle.call.scratch_bytes},
        .epilogue = d::view_of(desc.epilogue),
        .fanout = desc.fanout,
        .fanout_out_dist = desc.fanout_out_dist,
        .fanout_factor_dist = desc.fanout_factor_dist,
    };
  }

  [[nodiscard]] fftn::options opts() const {
    return {.in_place = c.in_place, .scale = c.options_scale, .force_isa = target};
  }

  [[nodiscard]] fftn::exec_args args(callback_state* user) const {
    const void* k = nullptr;
    if (!kreal.empty()) {
      k = kreal.data() + kzero;
    } else if (!kcomplex.empty()) {
      k = kcomplex.data() + kzero;
    }
    return {.mid_factor = k, .mid_user = user};
  }
};

std::string what_of(const lanes_case& c, fftn::isa t, bool dbl) {
  return std::string(c.name) + ", n = " + std::to_string(c.n) + ", " + fftn::isa_name(t) +
         (dbl ? " double" : " float");
}

// Serial execution of `core` on run's buffers: the output buffer after it.
template <class T>
std::vector<std::complex<T>> serial_of(const d::plan_core& core, lanes_run<T>& run,
                                       callback_state* user) {
  std::vector<std::complex<T>> y = run.init;
  std::vector<std::complex<T>> xs = run.x;
  fftn::workspace ws(d::workspace_bytes(core, 1));
  d::execute_all(core, (run.c.in_place ? y.data() : xs.data()) + run.in_base,
                 y.data() + run.out_base, ws, run.args(user));
  if (!run.c.in_place) {
    EXPECT_TRUE(same_bits(xs, run.x)) << "input written";
  }
  return y;
}

// The lanes pair forced against the chain pair forced, then the lanes pair's pieces of the batch,
// teams and an executor against its serial execution.
template <class T> void run_case(const lanes_case& lc, fftn::isa t) {
  using C = std::complex<T>;
  const std::string what = what_of(lc, t, sizeof(T) == 8);
  SCOPED_TRACE(what);
  fftn::test::input_source src(0x1A9E5ULL + lc.n * 37 + lc.counts.back());
  lanes_run<T> run(lc, t, src);
  d::build_settings lanes;
  lanes.strategy = d::strategy_choice::lanes;
  lanes.lane_block = lc.lane_block;
  lanes.column_block = lc.column_block;
  d::build_settings chain;
  chain.strategy = d::strategy_choice::chain;
  const auto by_lanes = d::make_pipeline_core(run.view(), run.opts(), lanes);
  const auto by_chain = d::make_pipeline_core(run.view(), run.opts(), chain);
  const std::string describe = d::describe(*by_lanes);
  ASSERT_NE(describe.find("the lanes pair (forced only"), std::string::npos) << describe;
  EXPECT_NE(describe.find(lc.adjacent ? "adjacent lines (a plane's columns)" : "form: lanes"),
            std::string::npos)
      << describe;
  ASSERT_NE(d::describe(*by_chain).find("the chain pair"), std::string::npos);
  const std::vector<C> kr_before(run.kcomplex);
  const std::vector<T> kreal_before(run.kreal);

  std::vector<std::atomic<int>> seen(run.lines * lc.n);
  run.state.seen = &seen;
  const std::vector<C> serial = serial_of<T>(*by_lanes, run, &run.state);
  if (lc.callback) {
    EXPECT_TRUE(run.state.geometry_ok) << "a middle block's geometry";
    EXPECT_TRUE(std::ranges::all_of(seen, [](const std::atomic<int>& s) { return s.load() == 1; }))
        << "each bin of each line passed exactly once";
  }
  run.state.seen = nullptr;
  callback_state chain_state = run.state;
  const std::vector<C> want = serial_of<T>(*by_chain, run, &chain_state);
  EXPECT_TRUE(same_bits(run.kcomplex, kr_before) && same_bits(run.kreal, kreal_before))
      << "kernel written";
  std::size_t touched = 0;
  for (std::size_t i = 0; i < serial.size(); ++i) {
    if (!run.written[i] && !same_bits(std::vector<C>{serial[i]}, std::vector<C>{run.init[i]})) {
      ++touched;
    }
  }
  EXPECT_EQ(touched, 0U) << "elements outside the lines written\n" << describe;
  if (lc.bitwise) {
    EXPECT_TRUE(same_bits(serial, want)) << "not the chain pair's bits\n" << describe;
  } else {
    double diff = 0;
    double norm = 0;
    for (std::size_t i = 0; i < serial.size(); ++i) {
      diff += std::norm(std::complex<double>(serial[i]) - std::complex<double>(want[i]));
      norm += std::norm(std::complex<double>(want[i]));
    }
    const auto tol = fftn::test::derived_tolerance<T>(lc.n, fftn::test::fftn_bound(lc.n)) * 4.0;
    EXPECT_LE(std::sqrt(diff / norm), tol.rms) << describe;
  }

  const auto in_of = [&](std::vector<C>& y, std::vector<C>& xs) {
    return (lc.in_place ? y.data() : xs.data()) + run.in_base;
  };
  const auto out_of = [&](std::vector<C>& y) { return y.data() + run.out_base; };
  fftn::workspace ws(d::workspace_bytes(*by_lanes, 3));
  std::vector<C> xs = run.x;
  const fftn::exec_args args = run.args(&run.state);
  std::vector<C> pieces = run.init;
  const std::size_t third = run.lines / 3;
  d::execute_core(*by_lanes, 0, third, in_of(pieces, xs), out_of(pieces), ws, args);
  d::execute_core(*by_lanes, third, run.lines - third, in_of(pieces, xs), out_of(pieces), ws, args);
  EXPECT_TRUE(same_bits(pieces, serial)) << "pieces of the batch differ\n" << describe;
  if (!lc.threads) {
    return;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    std::vector<C> teamed = run.init;
    std::vector<std::thread> members;
    for (int r = 1; r < 3; ++r) {
      members.emplace_back([&, r] {
        d::execute_team(*by_lanes, in_of(teamed, xs), out_of(teamed),
                        fftn::team{.rank = r, .size = 3}, ws, done, args);
      });
    }
    d::execute_team(*by_lanes, in_of(teamed, xs), out_of(teamed), fftn::team{.rank = 0, .size = 3},
                    ws, done, args);
    for (std::thread& m : members) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << "a team differs\n" << describe;
  }
  const fftn::thread_pool pool(3);
  std::vector<C> pooled = run.init;
  d::execute_on(*by_lanes, in_of(pooled, xs), out_of(pooled), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << "the executor differs\n" << describe;
}

// Rows of a batch: the vocabulary's placements, blocks and tails, strides and in place.
std::vector<lanes_case> row_cases(std::size_t n) {
  return {
      {.name = "rows: a real kernel for every line", .n = n, .threads = true},
      {.name = "rows: a real kernel per line, blocks of C and a tail",
       .n = n,
       .counts = {11},
       .per_line = true,
       .lane_block = 1},
      {.name = "rows: a complex kernel of stride -1 per line, two blocks and a tail",
       .n = n,
       .counts = {37},
       .mid = cplx,
       .mid_axis = -1,
       .per_line = true,
       .lane_block = 16,
       .threads = true},
      {.name = "rows: a complex kernel of stride 2",
       .n = n,
       .counts = {5},
       .mid = cplx,
       .mid_axis = 2},
      {.name = "rows: no kernel, scales",
       .n = n,
       .mid = none,
       .pro_scale = 2.0,
       .options_scale = 0.5},
      {.name = "rows: the fused kernel's placement on the chain pair",
       .n = n,
       .counts = {9},
       .mid = cplx,
       .per_line = true,
       .bitwise = false},
      {.name = "rows: strided, gathered and scattered",
       .n = n,
       .counts = {6},
       .in_stride = 2,
       .out_stride = -3,
       .per_line = true},
      {.name = "rows: two batch dimensions, a kernel per line",
       .n = n,
       .counts = {3, 5},
       .per_line = true,
       .threads = true},
      {.name = "rows: in place, the backward transform first",
       .n = n,
       .counts = {7},
       .backward = true,
       .in_place = true},
      {.name = "rows: a middle callback, the two products of scales",
       .n = n,
       .counts = {10},
       .per_line = true,
       .pro_scale = 3.0,
       .callback = true,
       .threads = true},
      {.name = "rows: a middle callback alone, backward first",
       .n = n,
       .counts = {4},
       .mid = none,
       .pro_scale = 0.25,
       .backward = true,
       .callback = true},
      {.name = "rows: fewer lines than a block", .n = n, .counts = {2}, .per_line = true},
  };
}

// The lines of strided planes: a plane's columns (the innermost dimension's lines adjacent), a
// volume's planes, forced column blocks.
std::vector<lanes_case> column_cases(std::size_t n) {
  return {
      {.name = "columns: a plane, a real kernel per column",
       .n = n,
       .counts = {12},
       .adjacent = true,
       .per_line = true,
       .threads = true},
      {.name = "columns: a volume's planes, a complex kernel of stride -1, blocks of 4",
       .n = n,
       .counts = {3, 9},
       .adjacent = true,
       .mid = cplx,
       .mid_axis = -1,
       .column_block = 4,
       .threads = true},
      {.name = "columns: in place, backward first, a callback",
       .n = n,
       .counts = {6},
       .adjacent = true,
       .backward = true,
       .in_place = true,
       .callback = true},
      {.name = "columns: the fused kernel's placement on the chain pair",
       .n = n,
       .counts = {17},
       .adjacent = true,
       .mid = cplx,
       .bitwise = false},
  };
}

template <class T> class PipelineLanes : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(PipelineLanes, precisions);

TYPED_TEST(PipelineLanes, RowsAgreeWithTheChainPair) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : {2U, 4U, 8U, 16U, 32U, 64U}) {
      for (const lanes_case& lc : row_cases(n)) {
        run_case<TypeParam>(lc, t);
      }
    }
  }
}

TYPED_TEST(PipelineLanes, ColumnsAgreeWithTheChainPair) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : {4U, 16U, 64U}) {
      for (const lanes_case& lc : column_cases(n)) {
        run_case<TypeParam>(lc, t);
      }
    }
  }
}

// What the lanes pair does not take yet is built as the chain pair, and describe() says why; by
// the rules no pipeline takes the pair (forced only until experiment X67).
TEST(PipelineLanes, FallsBackToTheChainPair) {
  using C = std::complex<double>;
  d::build_settings lanes;
  lanes.strategy = d::strategy_choice::lanes;
  const auto describe = [&](const fftn::pipeline_desc<C, C, C>& desc, const d::build_settings& s,
                            const fftn::options& opts = {}) {
    lanes_case lc;
    fftn::test::input_source src(1);
    lanes_run<double> run(lc, fftn::isa::best, src);
    run.desc = desc;
    return d::describe(*d::make_pipeline_core(run.view(), opts, s));
  };
  const fftn::pipeline_desc<C, C, C> good = fftn::complex_convolution_rows<double>(8, 16);
  EXPECT_NE(describe(good, lanes).find("the lanes pair (forced only"), std::string::npos);
  EXPECT_NE(describe(good, {}).find("the chain pair"), std::string::npos);
  const auto refused = [&](fftn::pipeline_desc<C, C, C> desc, const std::string& why,
                           const fftn::options& opts = {}) {
    const std::string text = describe(desc, lanes, opts);
    EXPECT_NE(text.find("the lanes pair (forced) does not take " + why + ": the chain pair"),
              std::string::npos)
        << why << "\n"
        << text;
  };
  auto desc = good;
  desc.prologue.mul = {.kind = real, .axis_strides = {1}, .batch_strides = {}};
  refused(desc, "a prologue or epilogue factor");
  desc = good;
  desc.epilogue.accumulate = true;
  refused(desc, "accumulation");
  desc = good;
  desc.runs = {{.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = 4}};
  desc.batches[0].in_dist = 4;
  refused(desc, "stored runs");
  desc = good;
  desc.fanout = 2;
  desc.fanout_out_dist = 8 * 16;
  refused(desc, "fan-out");
  desc = good;
  desc.epilogue.call = {.fn = &middle_callback<double>};
  refused(desc, "a prologue or epilogue callback");
  desc = good;
  desc.batches.clear();
  refused(desc, "a single line");
  desc = good;
  desc.batches[0].count = 1;
  refused(desc, "a batch of fewer than two lines in its innermost dimension");
}

} // namespace
