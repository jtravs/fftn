// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTN: one plan and one workspace per case, serial or threaded.
//
// Every case of rank 1 to 3 and every kind is supported, in double and single precision,
// with FFTW's embedding (in_last, out_last) and distances. describe reports the instruction
// set the plans run ("isa"), whether it was forced ("isa_forced"), how the arm executes and
// the options that change the plan; every case's note is its plan's per-axis lines (the
// radices it uses).
//
// Plans are built and executed through the functions that the public plan<In, Out>
// forwards every call to (include/fftn/plan.hpp: make_plan_core, execute_all, execute_on,
// execute_team), so that an arm can also set the private build settings
// (src/core/plan_core.hpp): the threading experiments T1 and T5 set the machine profile
// and chunking, experiment X2 the chain's radices. With no options the plan is exactly the
// public one.
//
// Options (bench/README.md, "FFTN options"), combinable:
//   PLANNING=rules|quick|measure   options::effort of every plan (default rules): measured
//                    planning times candidates when the plan is built, which fbench records
//                    as the case's plan_s; each case's note then starts with the chosen
//                    program, the candidates timed and the rules' time over the chosen one's
//                    (experiments P1, P2)
//   MEASURE_SECONDS=S   options::measure_seconds (default 0: the effort's budget)
//   MEASURE_SCRATCH=B   options::measure_scratch_bytes (default 0: 1 GiB)
//   ISA=<name>       force every plan to that instruction set (options::force_isa; names as
//                    fftn::isa_name writes them, "best" the default), so that arms of one
//                    build can compare targets (A=fftn+ISA=avx2 B=fftn+ISA=avx512). A target
//                    this build or processor lacks is refused when the session begins.
//   THREADS=N        execute through an executor of N lanes: fftn::thread_pool in
//                    fbench-fftn, fftn::omp::executor in fbench-fftn_omp (default: serial)
//   GROUP_SIZE=G     the lanes per cache group the executor reports (default: its own,
//                    fftn::cache_group_size(N)); 0 puts every lane in one group
//   CALL=executor|team   fbench-fftn_omp only: the executor overload (default), or every
//                    thread of a parallel region of N threads calling the team overload
//   WORKSPACE=library|page   each plan's workspace from fftn::workspace(bytes), on the heap
//                    (default), or in memory of the backend's own that starts a page, so that
//                    no other data shares the page of the team state (experiment X76)
//   CHUNK_POINTS=P   points per chunk of a stage (build_settings::chunk_points)
//   NS_PER_WORK, DISPATCH_NS, INLINE_MULTIPLE, SYNC_NS, MIN_POINTS_PER_MEMBER,
//   SPLIT_MIN_POINTS, SPLIT_MIN_POINTS_MESH, MESH_GROUP_LANES, REMAINDER_SPLIT_MIN_POINTS,
//   SPIN_NS, YIELD_NS, CLAIM_DIVISOR
//                    the machine profile's fields (src/threading/schedule.hpp); a size
//                    threshold may be `never`
//   RADICES=r1.r2... the chain's radices in pass order (build_settings::radices, an internal
//                    test knob), for c2c rows whose length they multiply to; any other case
//                    is an error, and so is a radix no pass of the target takes. Experiment
//                    X2 times radix sequences with it (dev/experiments/X2-radix-sets.md);
//                    one codelet radix equal to the length (RADICES=32 for n = 32) is a
//                    whole-transform leaf (experiment X8, dev/experiments/X8-leaves.md).
//   LEAF_LIMIT=N     the planner's leaf limit (build_settings::leaf_limit): every length
//                    up to N that has a codelet is one leaf instead of a chain (default: the
//                    profile's, 32 on NEON in double, else 16)
//   STRATEGY=rules|chain|grid|lanes|depth   the strategy of rows (build_settings::strategy):
//                    the planner's choice (default), or the chain, the grid, the lanes or the
//                    depth-first chain forced where the row can be one; each case's note says
//                    which was built (experiments X12, X7, X46)
//   DEPTH_TOPS=f1[.f2]   the depth-first chain's top factors (build_settings::depth_tops):
//                    wherever they fit with STRATEGY=depth, and on the rows the rules make
//                    depth-first chains otherwise (experiment X46)
//   LANE_BLOCK=B     the lanes strategy's block in rows (build_settings::lane_block; default
//                    the rule's), rounded up to a multiple of the complex count (X7)
//   BATCH_LEAVES=rules|never|own|narrower   the batch leaves of c2c rows (build_settings::
//                    batch_leaves): the rules' choice (default), none, the target's own, or a
//                    narrower target's that rounds alike (avx2's on avx512), where a row can
//                    take one (experiment X53)
//   COLUMNS=lanes|chain   the strided axes of planes and volumes (build_settings::
//                    lanes_columns): lanes over blocks of adjacent columns where they fit
//                    (default), or every column alone through the chain (experiment X40)
//   COLUMN_BLOCK=B   the lanes column phase's block in columns (build_settings::
//                    column_block; default the rule's), rounded up to a multiple of C (X40)
//   COLUMN_RADICES=r1.r2...   the lanes column phase's radices in pass order for columns
//                    whose length they multiply to (build_settings::column_radices; X44)
//   LAYOUT=rules|interleaved|split   the arithmetic of the chains' lane passes
//                    (build_settings::layout): the profile's choice (default), or interleaved
//                    or split forced where a chain has a split run; each case's note shows
//                    the split passes (experiment X1)
//   GRID_FACTORS=n1.n2[.n3]     the grid's factors in phase order (build_settings::
//                    grid_factors), for rows they multiply to
//   RADER=rules|0|1  Rader stages (build_settings::rader): the profile's range (default),
//                    never (Bluestein instead), or for every prime factor whose q - 1 is
//                    13-smooth (experiment X27; task 2.4's 65537 check)
//   GRID_PHASES=2|3  the grid's phase count instead of the rule's (build_settings::
//                    grid_phases), where the row is a grid; experiment P3
//   GRID_PHASE_RADIX=R          the largest radix of the grid's phase rule (build_settings::
//                    grid_phase_radix; default 16): codelet radices up to R (experiment X3)
//   GRID_MIN_BYTES, GRID_THREE_BYTES, GRID_BLOCK_BYTES, GRID_BLOCK, GRID_FUSE_BYTES,
//   GRID_PAD_PERIOD, POW2_CHAIN_BYTES, POW2_RADIX8_BYTES, GRID_TWO_ODD_PART,
//   GRID_TWO_PART_BYTES, BLUESTEIN_POW2_PART_BYTES, LONG_CONVOLUTION_BYTES,
//   BLUESTEIN_WHOLE_LANES (0 or 1), ANCHORED_RUNS (0 or 1)
//                    the planner's grid fields of the machine profile (schedule.hpp)
//   EDGE_IN=KIND.LAYOUT, EDGE_OUT=KIND.LAYOUT[.acc] or acc   plan edges (experiment X65): a
//                    real or complex factor laid out as line (one line for every transform),
//                    full (as large as the side), s2 (twice the tight strides) or row (one
//                    value per transform), every value one so that the check holds, and
//                    accumulation on the output (which is zeroed for the check); a case the
//                    edges do not apply to is unsupported
//   EDGE_ARM=library|loop   the plan with the edges (default), or the plain plan and the
//                    user's own separate loops (the input's in place before it, the output's
//                    after it, accumulation from a buffer of the transform's own)
//   PIPELINE=fused|separate   a convolution instead of the transform (task 5.5): every row
//                    of a c2c forward case (`pipeline<C, C, C>`) or of an r2c case
//                    (`pipeline<T, C, T>`, the output rows real, each at the start of its row
//                    of n/2 + 1 complex values) convolved with a kernel whose spectrum, one for
//                    every row, delays the row by one element (1/n folded in); fused is the
//                    row pipeline (fftn/pipeline.hpp, by the rules), separate the same work as a
//                    user writes it with plans and no buffer of their own: the forward plan into
//                    the output, a multiply loop there (plain C++, cloned like the edge loops),
//                    the inverse plan in place. Both arms are planned by the rules with ISA's
//                    instruction set (PLANNING and the build settings reach neither; X67, task
//                    5.2g, adds them). describe says "output":"delay", so fbench checks the
//                    delayed rows instead of the DFT, and lists c2c and r2c as partial: forward
//                    rows out of place only, other cases unsupported. Serial or on the executor
//                    (THREADS); not with CALL=team, EDGE_* or RADICES
// ISA is a public option; the others reach the library only through the private build
// settings, as a test's do: the library has no switch for them, so dev/KNOBS.toml does not
// list them.
//
// On macOS the harness raises its own thread to the user-interactive QoS class, which new
// threads do not inherit: the executor's threads are raised by one task each at begin, so
// that they too run on the performance cores.

#include "backends/common.hpp"
#include "backends/fftn_tuning.hpp"
#include "core/plan_core.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <fftn/detail/core.hpp>
#include <fftn/fftn.hpp>

#ifdef FBENCH_FFTN_OMP
#include "platform/environment.hpp"

#include <fftn/omp.hpp>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#endif

#ifdef __APPLE__
#include <pthread/qos.h>
#include <sys/qos.h>

#include <atomic> // raise_lanes
#include <chrono>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace detail = fftn::detail;
using fbench::backends::say;

#ifdef FBENCH_FFTN_OMP
using executor_type = fftn::omp::executor;
constexpr const char* executor_name = "omp::executor";
#else
using executor_type = fftn::thread_pool;
constexpr const char* executor_name = "thread_pool";
#endif

// An executor with another grouping: GROUP_SIZE.
class regrouped {
public:
  regrouped(const executor_type& e, int group) : e_(e), group_(group) {}
  [[nodiscard]] int concurrency() const { return e_.concurrency(); }
  [[nodiscard]] int group_size() const { return group_ < 0 ? e_.group_size() : group_; }
  void bulk(int n, fftn::task_function task, void* arg) const { e_.bulk(n, task, arg); }

private:
  const executor_type& e_;
  int group_;
};

// An edge as the arm gives it: a factor's kind and layout, and accumulation (output only).
// The layouts index the factor by the side's logical coordinates (fftn/edges.hpp):
//   line  one line of factors for every transform (tight strides, batch stride 0): a window
//   full  an array as large as the side (tight strides, one block per transform)
//   s2    a line read with twice the tight strides (every other element of a wider array)
//   row   one value per transform (axis strides 0, batch stride 1)
enum class factor_layout : unsigned char { none, line, full, s2, row };

struct edge_spec {
  fftn::factor_kind kind = fftn::factor_kind::none;
  factor_layout layout = factor_layout::none;
  bool accumulate = false;
  std::string text; // as given
  [[nodiscard]] bool any() const { return kind != fftn::factor_kind::none || accumulate; }
};

// Parses KIND.LAYOUT[.acc] or acc (KIND real or complex; acc on the output only).
std::optional<edge_spec> parse_edge(std::string_view text, bool output) {
  edge_spec e;
  e.text = text;
  std::size_t at = 0;
  while (at <= text.size()) {
    const std::size_t end = std::min(text.find('.', at), text.size());
    const std::string_view token = text.substr(at, end - at);
    at = end + 1;
    if (token == "real" && e.kind == fftn::factor_kind::none) {
      e.kind = fftn::factor_kind::real;
    } else if (token == "complex" && e.kind == fftn::factor_kind::none) {
      e.kind = fftn::factor_kind::complex;
    } else if (token == "line" && e.layout == factor_layout::none) {
      e.layout = factor_layout::line;
    } else if (token == "full" && e.layout == factor_layout::none) {
      e.layout = factor_layout::full;
    } else if (token == "s2" && e.layout == factor_layout::none) {
      e.layout = factor_layout::s2;
    } else if (token == "row" && e.layout == factor_layout::none) {
      e.layout = factor_layout::row;
    } else if (token == "acc" && output && !e.accumulate) {
      e.accumulate = true;
    } else {
      return std::nullopt;
    }
  }
  if ((e.kind == fftn::factor_kind::none) != (e.layout == factor_layout::none) || !e.any()) {
    return std::nullopt;
  }
  return e;
}

// edges: a plan with EDGE_IN or EDGE_OUT, executed out of line (plan_record::edge_mode says
// how), so that a plain serial plan's execute stays one test and one call (X49); pipeline: a
// PIPELINE arm (plan_record::pipe), likewise.
enum class call : unsigned char { serial, executor, team, edges, pipeline };

// PIPELINE: none, the row pipeline, or separate plans and a loop.
enum class pipe_arm : unsigned char { none, fused, separate };

struct session {
  detail::build_settings settings;
  call mode = call::serial;
  int threads = 1;
  int group_size = -1;                           // -1: the executor's own
  fftn::isa isa = fftn::isa::best;               // ISA=: the instruction set of every plan
  fftn::planning effort = fftn::planning::rules; // PLANNING=
  std::string planning = "rules";
  double measure_seconds = 0;      // MEASURE_SECONDS=
  std::size_t measure_scratch = 0; // MEASURE_SCRATCH=
  std::unique_ptr<executor_type> exec;
  std::string overrides;              // the profile and chunking options given, as JSON members
  std::string radices_text;           // RADICES as given, r1.r2... (settings.radices holds them)
  std::string column_radices_text;    // COLUMN_RADICES as given
  std::string strategy = "rules";     // STRATEGY as given
  std::string layout = "rules";       // LAYOUT as given
  std::string factors_text;           // GRID_FACTORS as given
  std::string depth_tops_text;        // DEPTH_TOPS as given
  std::string rader = "rules";        // RADER as given
  std::string batch_leaves = "rules"; // BATCH_LEAVES as given
  bool workspace_page = false;        // WORKSPACE=page
  edge_spec edge_in;                  // EDGE_IN
  edge_spec edge_out;                 // EDGE_OUT
  bool edge_loop = false;             // EDGE_ARM=loop: a plain plan and the user's own loops
  pipe_arm pipeline = pipe_arm::none; // PIPELINE
  std::string description;
};

session& state() {
  static session s;
  return s;
}

// The enumerator named `name` (fftn::isa_name), or nullopt.
std::optional<fftn::isa> isa_from_name(std::string_view name) {
  for (auto i = static_cast<unsigned>(fftn::isa::scalar);
       i <= static_cast<unsigned>(fftn::isa::best); ++i) {
    const auto t = static_cast<fftn::isa>(i);
    if (name == fftn::isa_name(t)) {
      return t;
    }
  }
  return std::nullopt;
}

// FFTW's many-interface layout: row-major arrays whose last dimension is embedded in
// `last` elements, transforms `dist` elements apart.
template <class In, class Out> fftn::shape<In, Out> shape_of(const fbench_case& c) {
  fftn::shape<In, Out> s;
  s.dir = c.sign < 0 ? fftn::direction::forward : fftn::direction::backward;
  const auto rank = static_cast<std::size_t>(c.rank);
  for (std::size_t d = 0; d < rank; ++d) {
    s.axes.push_back({.n = static_cast<std::size_t>(c.dims[d])});
  }
  // The interleaved layout (fbench_case::interleave): every element of a transform stands
  // `interleave` elements from the next.
  auto in_stride = static_cast<std::ptrdiff_t>(c.interleave);
  auto out_stride = static_cast<std::ptrdiff_t>(c.interleave);
  for (std::size_t i = rank; i > 0; --i) {
    const std::size_t d = i - 1;
    s.axes[d].in_stride = in_stride;
    s.axes[d].out_stride = out_stride;
    const bool last = d + 1 == rank;
    in_stride *= last ? c.in_last : c.dims[d];
    out_stride *= last ? c.out_last : c.dims[d];
  }
  if (c.outer > 1) {
    s.batches.push_back({
        .count = static_cast<std::size_t>(c.outer),
        .in_dist = static_cast<std::ptrdiff_t>(c.outer_in_dist),
        .out_dist = static_cast<std::ptrdiff_t>(c.outer_out_dist),
    });
  }
  s.batches.push_back({
      .count = static_cast<std::size_t>(c.batch),
      .in_dist = static_cast<std::ptrdiff_t>(c.in_dist),
      .out_dist = static_cast<std::ptrdiff_t>(c.out_dist),
  });
  return s;
}

template <class In, class Out>
std::shared_ptr<const detail::plan_core> make_core(const fbench_case& c, const fftn::options& o) {
  const fftn::shape<In, Out> s = shape_of<In, Out>(c);
  return detail::make_plan_core(detail::view_of(s), o, state().settings);
}

template <class T>
std::shared_ptr<const detail::plan_core> make_core(const fbench_case& c, const fftn::options& o) {
  using cplx = std::complex<T>;
  if (c.kind == fbench_c2c) {
    return make_core<cplx, cplx>(c, o);
  }
  if (c.kind == fbench_r2c) {
    return make_core<T, cplx>(c, o);
  }
  return make_core<cplx, T>(c, o);
}

// ---- RADICES ---------------------------------------------------------------------------

// Parses r1.r2... (each an integer of at least 2); throws std::invalid_argument.
std::vector<std::size_t> parse_radices(std::string_view text, const char* what = "RADICES") {
  std::vector<std::size_t> out;
  std::size_t at = 0;
  while (at <= text.size()) {
    const std::size_t end = std::min(text.find('.', at), text.size());
    const std::string_view token = text.substr(at, end - at);
    std::size_t r = 0;
    const auto res = std::from_chars(token.data(), token.data() + token.size(), r);
    if (token.empty() || res.ec != std::errc{} || res.ptr != token.data() + token.size() || r < 2) {
      throw std::invalid_argument(std::string(what) +
                                  " is r1.r2... with every factor at least 2, not '" +
                                  std::string(text) + "'");
    }
    out.push_back(r);
    at = end + 1;
  }
  return out;
}

// Refuses a case the forced radices cannot plan: anything but a c2c row of their product.
void check_radices_fit(const fbench_case& c) {
  const session& s = state();
  std::size_t product = 1;
  for (const std::size_t r : s.settings.radices) {
    product *= r;
  }
  if (c.rank != 1) {
    throw std::invalid_argument("RADICES=" + s.radices_text + " is for rows only");
  }
  // A real row of even length runs a half-length complex chain, which the radices force.
  const bool real = c.kind != fbench_c2c;
  if (real && c.dims[0] % 2 != 0) {
    throw std::invalid_argument("RADICES=" + s.radices_text + " is for real rows of even length");
  }
  const std::int64_t length = real ? c.dims[0] / 2 : c.dims[0];
  if (std::cmp_not_equal(product, length)) {
    throw std::invalid_argument("RADICES=" + s.radices_text + " multiplies to " +
                                std::to_string(product) + ", not the " +
                                (real ? "half length " : "length ") + std::to_string(length));
  }
}

// "4 x 8 x 8", as the plan's describe lists radices.
std::string radix_list(const std::vector<std::size_t>& radices) {
  std::string text;
  for (const std::size_t r : radices) {
    text += (text.empty() ? "" : " x ") + std::to_string(r);
  }
  return text;
}

// The plan's per-axis lines of its describe, joined by "; ".
std::string axis_lines(const std::string& describe) {
  std::string out;
  std::size_t at = 0;
  while (at < describe.size()) {
    const std::size_t end = std::min(describe.find('\n', at), describe.size());
    const std::string_view line = std::string_view(describe).substr(at, end - at);
    if (line.starts_with("axis ")) {
      out += (out.empty() ? "" : "; ") + std::string(line);
    }
    at = end + 1;
  }
  return out;
}

// ---- EDGE_IN, EDGE_OUT, EDGE_ARM (experiment X65) ----------------------------------------

// One side of a case as the edge loops see it: `rank` axes of the side's logical extents
// (n/2 + 1 on a real transform's complex side), the data's strides and distance in the side's
// elements, and the factor array's strides in its own elements.
struct side_geometry {
  bool complex_data = true;
  std::size_t rank = 1;
  std::size_t batch = 1;
  std::array<std::size_t, 3> extent{1, 1, 1};
  std::array<std::ptrdiff_t, 3> stride{0, 0, 0};
  std::ptrdiff_t dist = 0;
  fftn::factor_kind kind = fftn::factor_kind::none;
  std::array<std::ptrdiff_t, 3> factor_stride{0, 0, 0};
  std::ptrdiff_t factor_batch = 0;
  std::size_t factor_elements = 0; // the array's length in its own elements
  bool accumulate = false;
  [[nodiscard]] std::size_t lines() const {
    std::size_t l = batch;
    for (std::size_t d = 0; d + 1 < rank; ++d) {
      l *= extent[d];
    }
    return l;
  }
};

template <class In, class Out>
side_geometry side_of(const fftn::shape<In, Out>& s, const fbench_case& c, const edge_spec& e,
                      bool output) {
  side_geometry g;
  const bool real_side = c.kind == (output ? fbench_c2r : fbench_r2c);
  const bool half_side = c.kind == (output ? fbench_r2c : fbench_c2r);
  g.complex_data = !real_side;
  g.rank = s.axes.size();
  g.batch = s.batches.back().count;
  g.dist = output ? s.batches.back().out_dist : s.batches.back().in_dist;
  std::size_t tight = 1;
  for (std::size_t i = g.rank; i > 0; --i) {
    const std::size_t d = i - 1;
    g.extent[d] = s.axes[d].n;
    if (half_side && d + 1 == g.rank) {
      g.extent[d] = s.axes[d].n / 2 + 1;
    }
    g.stride[d] = output ? s.axes[d].out_stride : s.axes[d].in_stride;
    const auto t = static_cast<std::ptrdiff_t>(tight);
    g.factor_stride[d] = e.layout == factor_layout::row  ? 0
                         : e.layout == factor_layout::s2 ? 2 * t
                                                         : t;
    tight *= g.extent[d];
  }
  g.kind = e.kind;
  g.accumulate = e.accumulate;
  switch (e.layout) {
  case factor_layout::none:
    break;
  case factor_layout::line:
    g.factor_elements = tight;
    break;
  case factor_layout::s2:
    g.factor_elements = 2 * tight;
    break;
  case factor_layout::full:
    g.factor_batch = static_cast<std::ptrdiff_t>(tight);
    g.factor_elements = tight * g.batch;
    break;
  case factor_layout::row:
    g.factor_batch = 1;
    g.factor_elements = g.batch;
    break;
  }
  return g;
}

// The user's separate loop over lines [first, first + count) of a side (side_geometry::lines,
// each the last axis of one transform), as a user who cares would write it: explicit real
// arithmetic on contiguous lines, which the compiler vectorises. Without accumulation it
// multiplies the data by the factor in place; with it, data += factor * src (src the
// transform's result in a buffer of the data's layout), or data += src without a factor.
// Element j's factor index along a line of factor stride fs (FactorStride 0 or 1 when known).
template <int FactorStride>
[[gnu::always_inline]] inline std::size_t factor_at(std::size_t j, std::ptrdiff_t fs) {
  if constexpr (FactorStride == 0) {
    return 0;
  } else if constexpr (FactorStride == 1) {
    return j;
  } else {
    return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(j) * fs);
  }
}

template <class T, bool Complex, int Kind, int FactorStride, bool Acc>
[[gnu::always_inline]] inline void edge_line(T* __restrict d, const T* __restrict src,
                                             const T* __restrict f, std::ptrdiff_t fs,
                                             std::size_t n) {
  for (std::size_t j = 0; j < n; ++j) {
    if constexpr (Complex) {
      T vr = Acc ? src[2 * j] : d[2 * j];
      T vi = Acc ? src[2 * j + 1] : d[2 * j + 1];
      if constexpr (Kind == 1) {
        const T w = f[factor_at<FactorStride>(j, fs)];
        vr *= w;
        vi *= w;
      } else if constexpr (Kind == 2) {
        const T fr = f[2 * factor_at<FactorStride>(j, fs)];
        const T fi = f[2 * factor_at<FactorStride>(j, fs) + 1];
        const T r = vr * fr - vi * fi;
        vi = vr * fi + vi * fr;
        vr = r;
      }
      if constexpr (Acc) {
        d[2 * j] += vr;
        d[2 * j + 1] += vi;
      } else {
        d[2 * j] = vr;
        d[2 * j + 1] = vi;
      }
    } else {
      T v = Acc ? src[j] : d[j];
      if constexpr (Kind == 1) {
        v *= f[factor_at<FactorStride>(j, fs)];
      }
      if constexpr (Acc) {
        d[j] += v;
      } else {
        d[j] = v;
      }
    }
  }
}

template <class T, bool Complex, int Kind, int FactorStride, bool Acc>
[[gnu::always_inline]] inline void edge_lines_as(const side_geometry& g, T* data, const T* src,
                                                 const T* factor, std::size_t first,
                                                 std::size_t count) {
  const std::size_t inner = g.lines() / g.batch;
  const std::size_t n = g.extent[g.rank - 1];
  const std::ptrdiff_t width = Complex ? 2 : 1;
  const std::ptrdiff_t fw = g.kind == fftn::factor_kind::complex ? 2 : 1;
  for (std::size_t l = first; l < first + count; ++l) {
    std::size_t rest = l % inner;
    std::ptrdiff_t off = static_cast<std::ptrdiff_t>(l / inner) * g.dist;
    std::ptrdiff_t foff = static_cast<std::ptrdiff_t>(l / inner) * g.factor_batch;
    for (std::size_t i = g.rank - 1; i > 0; --i) {
      const std::size_t d = i - 1;
      const auto idx = static_cast<std::ptrdiff_t>(rest % g.extent[d]);
      rest /= g.extent[d];
      off += idx * g.stride[d];
      foff += idx * g.factor_stride[d];
    }
    edge_line<T, Complex, Kind, FactorStride, Acc>(
        data + off * width, Acc ? src + off * width : nullptr,
        Kind != 0 ? factor + foff * fw : nullptr, g.factor_stride[g.rank - 1], n);
  }
}

template <class T, bool Complex, int Kind, bool Acc>
[[gnu::always_inline]] inline void edge_lines_stride(const side_geometry& g, T* data, const T* src,
                                                     const T* factor, std::size_t first,
                                                     std::size_t count) {
  const std::ptrdiff_t fs = g.factor_stride[g.rank - 1];
  if (fs == 0) {
    edge_lines_as<T, Complex, Kind, 0, Acc>(g, data, src, factor, first, count);
  } else if (fs == 1) {
    edge_lines_as<T, Complex, Kind, 1, Acc>(g, data, src, factor, first, count);
  } else {
    edge_lines_as<T, Complex, Kind, -1, Acc>(g, data, src, factor, first, count);
  }
}

template <class T, bool Complex>
[[gnu::always_inline]] inline void edge_lines_kind(const side_geometry& g, T* data, const T* src,
                                                   const T* factor, std::size_t first,
                                                   std::size_t count) {
  const bool acc = src != nullptr;
  switch (g.kind) {
  case fftn::factor_kind::none:
    if (acc) {
      edge_lines_as<T, Complex, 0, 0, true>(g, data, src, factor, first, count);
    }
    return;
  case fftn::factor_kind::real:
    if (acc) {
      edge_lines_stride<T, Complex, 1, true>(g, data, src, factor, first, count);
    } else {
      edge_lines_stride<T, Complex, 1, false>(g, data, src, factor, first, count);
    }
    return;
  case fftn::factor_kind::complex:
    if constexpr (Complex) {
      if (acc) {
        edge_lines_stride<T, true, 2, true>(g, data, src, factor, first, count);
      } else {
        edge_lines_stride<T, true, 2, false>(g, data, src, factor, first, count);
      }
    }
    return;
  }
}

// On x86-64 Linux the loops are cloned for AVX2 and AVX-512, as a user building for their own
// machine would get them (the bench is compiled for the baseline target, SSE2); elsewhere the
// baseline is already the machine's vector set (NEON).
#if defined(__x86_64__) && defined(__linux__)
#define FBENCH_EDGE_CLONES [[gnu::target_clones("default", "avx2", "avx512f")]]
#else
#define FBENCH_EDGE_CLONES
#endif

FBENCH_EDGE_CLONES void edge_lines_f64(const side_geometry& g, double* data, const double* src,
                                       const double* factor, std::size_t first, std::size_t count) {
  if (g.complex_data) {
    edge_lines_kind<double, true>(g, data, src, factor, first, count);
  } else {
    edge_lines_kind<double, false>(g, data, src, factor, first, count);
  }
}

FBENCH_EDGE_CLONES void edge_lines_f32(const side_geometry& g, float* data, const float* src,
                                       const float* factor, std::size_t first, std::size_t count) {
  if (g.complex_data) {
    edge_lines_kind<float, true>(g, data, src, factor, first, count);
  } else {
    edge_lines_kind<float, false>(g, data, src, factor, first, count);
  }
}

// ---- PIPELINE (task 5.5) ----------------------------------------------------------------

// The separate arm's multiply loop over rows [first, first + count) of the spectrum, rows `dist`
// complex values apart: each bin times the kernel's (one for every row), the complex arithmetic
// written out so that the compiler vectorises it, as edge_line's.
template <class T>
[[gnu::always_inline]] inline void multiply_rows_as(T* __restrict spectrum, const T* __restrict k,
                                                    std::size_t bins, std::size_t dist,
                                                    std::size_t first, std::size_t count) {
  for (std::size_t r = first; r < first + count; ++r) {
    T* __restrict row = spectrum + 2 * r * dist;
    for (std::size_t j = 0; j < bins; ++j) {
      const T vr = row[2 * j];
      const T vi = row[2 * j + 1];
      row[2 * j] = vr * k[2 * j] - vi * k[2 * j + 1];
      row[2 * j + 1] = vr * k[2 * j + 1] + vi * k[2 * j];
    }
  }
}

FBENCH_EDGE_CLONES void multiply_rows_f64(double* spectrum, const double* k, std::size_t bins,
                                          std::size_t dist, std::size_t first, std::size_t count) {
  multiply_rows_as(spectrum, k, bins, dist, first, count);
}

FBENCH_EDGE_CLONES void multiply_rows_f32(float* spectrum, const float* k, std::size_t bins,
                                          std::size_t dist, std::size_t first, std::size_t count) {
  multiply_rows_as(spectrum, k, bins, dist, first, count);
}

template <class T>
void multiply_rows(std::complex<T>* spectrum, const std::complex<T>* k, std::size_t bins,
                   std::size_t dist, std::size_t first, std::size_t count) {
  if constexpr (std::is_same_v<T, float>) {
    multiply_rows_f32(reinterpret_cast<float*>(spectrum), reinterpret_cast<const float*>(k), bins,
                      dist, first, count);
  } else {
    multiply_rows_f64(reinterpret_cast<double*>(spectrum), reinterpret_cast<const double*>(k), bins,
                      dist, first, count);
  }
}

// One PIPELINE arm of a case, executed serially or on the session's executor.
class pipeline_arm {
public:
  pipeline_arm() = default;
  pipeline_arm(const pipeline_arm&) = delete;
  pipeline_arm& operator=(const pipeline_arm&) = delete;
  pipeline_arm(pipeline_arm&&) = delete;
  pipeline_arm& operator=(pipeline_arm&&) = delete;
  virtual ~pipeline_arm() = default;
  virtual void run(const void* in, void* out) const = 0;
  [[nodiscard]] virtual std::string note() const = 0;
};

// The kernel's spectrum over `bins` bins of a row of n: a delay by one element, times 1/n
// (what fbench's delayed check expects, harness/check.hpp).
template <class T> std::vector<std::complex<T>> delay_kernel(std::size_t n, std::size_t bins) {
  std::vector<std::complex<T>> k(bins);
  for (std::size_t j = 0; j < bins; ++j) {
    const double angle =
        -2.0 * std::numbers::pi * static_cast<double>(j % n) / static_cast<double>(n);
    k[j] = std::complex<T>(std::polar(1.0 / static_cast<double>(n), angle));
  }
  return k;
}

// A describe's lines joined by "; ".
std::string one_line(const std::string& text) {
  std::string out;
  for (const char ch : text) {
    if (ch == '\n') {
      out += "; ";
    } else {
      out += ch;
    }
  }
  return out;
}

// The rows of a case as a pipeline, <C, C, C> (Real false) or <T, C, T> (Real true), by the
// rules: the kernel one spectrum for every row, the rows `in_dist` and `out_dist` elements apart.
template <class T, bool Real> class fused_arm final : public pipeline_arm {
  using C = std::complex<T>;
  using In = std::conditional_t<Real, T, C>;

public:
  fused_arm(std::size_t n, std::size_t count, std::ptrdiff_t in_dist, std::ptrdiff_t out_dist,
            const fftn::options& o)
      : pipe_(desc(n, count, in_dist, out_dist), o),
        kernel_(delay_kernel<T>(n, Real ? n / 2 + 1 : n)), ws_(pipe_, state().threads) {}

  void run(const void* in, void* out) const override {
    const session& s = state();
    const fftn::exec_args args{.mid_factor = kernel_.data()};
    const auto* x = static_cast<const In*>(in);
    auto* y = static_cast<In*>(out);
    if (s.mode == call::executor) {
      pipe_.execute(x, y, regrouped(*s.exec, s.group_size), ws_, args);
    } else {
      pipe_.execute(x, y, ws_, args);
    }
  }

  [[nodiscard]] std::string note() const override { return one_line(pipe_.describe()); }

private:
  static fftn::pipeline_desc<In, C, In> desc(std::size_t n, std::size_t count,
                                             std::ptrdiff_t in_dist, std::ptrdiff_t out_dist) {
    fftn::pipeline_desc<In, C, In> d;
    if constexpr (Real) {
      d = fftn::convolution_rows<T>(n, count);
    } else {
      d = fftn::complex_convolution_rows<T>(n, count);
    }
    d.batches.at(0).in_dist = in_dist;
    d.batches.at(0).out_dist = out_dist;
    return d;
  }

  fftn::pipeline<In, C, In> pipe_;
  std::vector<C> kernel_;
  mutable fftn::workspace ws_;
};

// The same rows as a user writes them with plans, without a buffer of their own: the forward
// plan into the output (an r2c case's output rows hold n/2 + 1 complex values), the multiply loop
// in place there, the inverse plan in place (c2r in FFTW's padded layout).
template <class T, bool Real> class separate_arm final : public pipeline_arm {
  using C = std::complex<T>;
  using In = std::conditional_t<Real, T, C>;
  using forward_plan = std::conditional_t<Real, fftn::r2c_plan<T>, fftn::c2c_plan<T>>;
  using inverse_plan = std::conditional_t<Real, fftn::c2r_plan<T>, fftn::c2c_plan<T>>;

public:
  // `out_dist` in In elements, as the fused arm takes it.
  separate_arm(std::size_t n, std::size_t count, std::ptrdiff_t in_dist, std::ptrdiff_t out_dist,
               const fftn::options& o)
      : bins_(Real ? n / 2 + 1 : n),
        dist_(static_cast<std::size_t>(Real ? out_dist / 2 : out_dist)), count_(count),
        forward_(forward_shape(n, count, in_dist, dist_), o),
        inverse_(inverse_shape(n, count, dist_), in_place(o)), kernel_(delay_kernel<T>(n, bins_)),
        ws_(std::max(forward_.workspace_bytes(state().threads),
                     inverse_.workspace_bytes(state().threads))) {}

  void run(const void* in, void* out) const override {
    const session& s = state();
    const auto* x = static_cast<const In*>(in);
    auto* spectrum = static_cast<C*>(out);
    auto* y = static_cast<In*>(out);
    if (s.mode == call::executor) {
      const regrouped e(*s.exec, s.group_size);
      forward_.execute(x, spectrum, e, ws_);
      multiply_on(spectrum, *s.exec, s.threads);
      inverse_.execute(spectrum, y, e, ws_);
    } else {
      forward_.execute(x, spectrum, ws_);
      multiply_rows(spectrum, kernel_.data(), bins_, dist_, 0, count_);
      inverse_.execute(spectrum, y, ws_);
    }
  }

  [[nodiscard]] std::string note() const override {
    return "forward: " + axis_lines(forward_.describe()) +
           "; inverse in place: " + axis_lines(inverse_.describe());
  }

private:
  static fftn::options in_place(fftn::options o) {
    o.in_place = true;
    return o;
  }

  static fftn::shape<In, C> forward_shape(std::size_t n, std::size_t count, std::ptrdiff_t in_dist,
                                          std::size_t dist) {
    fftn::shape<In, C> sh;
    if constexpr (Real) {
      sh = fftn::rows_r2c<T>(n, count);
    } else {
      sh = fftn::rows_c2c<T>(n, count, fftn::direction::forward);
    }
    sh.batches.at(0).in_dist = in_dist;
    sh.batches.at(0).out_dist = static_cast<std::ptrdiff_t>(dist);
    return sh;
  }

  // In place: rows `dist` complex values apart, the real output 2 dist reals apart.
  static fftn::shape<C, In> inverse_shape(std::size_t n, std::size_t count, std::size_t dist) {
    fftn::shape<C, In> sh;
    if constexpr (Real) {
      sh = fftn::rows_c2r<T>(n, count);
    } else {
      sh = fftn::rows_c2c<T>(n, count, fftn::direction::backward);
    }
    sh.batches.at(0).in_dist = static_cast<std::ptrdiff_t>(dist);
    sh.batches.at(0).out_dist = static_cast<std::ptrdiff_t>(Real ? 2 * dist : dist);
    return sh;
  }

  // The loop on the executor's lanes, each a run of whole rows (a user with threads shares
  // their loop out too).
  struct job {
    const separate_arm* arm;
    C* spectrum;
    int tasks;
  };

  static void task(void* arg, int index) {
    const auto& j = *static_cast<const job*>(arg);
    const std::size_t rows = j.arm->count_;
    const auto t = static_cast<std::size_t>(j.tasks);
    const auto i = static_cast<std::size_t>(index);
    const std::size_t first = rows * i / t;
    multiply_rows(j.spectrum, j.arm->kernel_.data(), j.arm->bins_, j.arm->dist_, first,
                  rows * (i + 1) / t - first);
  }

  void multiply_on(C* spectrum, const executor_type& e, int threads) const {
    job j{this, spectrum,
          static_cast<int>(std::min<std::size_t>(count_, static_cast<std::size_t>(threads)))};
    if (j.tasks < 2) {
      multiply_rows(spectrum, kernel_.data(), bins_, dist_, 0, count_);
      return;
    }
    e.bulk(j.tasks, &task, &j);
  }

  std::size_t bins_;
  std::size_t dist_; // between the spectrum's rows, in complex values
  std::size_t count_;
  forward_plan forward_;
  inverse_plan inverse_;
  std::vector<C> kernel_;
  mutable fftn::workspace ws_;
};

// The arm of a case, or nullptr when PIPELINE does not apply to it.
template <class T>
std::unique_ptr<pipeline_arm> make_pipeline_arm(const fbench_case& c, const fftn::options& o) {
  const bool rows = c.rank == 1 && c.interleave == 1 && c.outer == 1 && c.in_place == 0;
  const bool cplx = c.kind == fbench_c2c && c.sign < 0;
  if (!rows || (!cplx && c.kind != fbench_r2c)) {
    return nullptr;
  }
  const auto n = static_cast<std::size_t>(c.dims[0]);
  const auto count = static_cast<std::size_t>(c.batch);
  const auto in_dist = static_cast<std::ptrdiff_t>(c.in_dist);
  // An r2c case's output rows of n/2 + 1 complex values hold the n reals.
  const auto out_dist = static_cast<std::ptrdiff_t>(cplx ? c.out_dist : 2 * c.out_dist);
  // Both arms by the rules with the session's instruction set: pipelines are planned by the rules
  // only (until X67), so the plans are too, whatever PLANNING says.
  const fftn::options like{.force_isa = o.force_isa};
  const bool fused = state().pipeline == pipe_arm::fused;
  if (cplx) {
    if (fused) {
      return std::make_unique<fused_arm<T, false>>(n, count, in_dist, out_dist, like);
    }
    return std::make_unique<separate_arm<T, false>>(n, count, in_dist, out_dist, like);
  }
  if (fused) {
    return std::make_unique<fused_arm<T, true>>(n, count, in_dist, out_dist, like);
  }
  return std::make_unique<separate_arm<T, true>>(n, count, in_dist, out_dist, like);
}

struct plan_record {
  std::unique_ptr<pipeline_arm> pipe; // PIPELINE (mode call::pipeline): the case's arm
  std::shared_ptr<const detail::plan_core> core;
  std::unique_ptr<std::byte[]> ws_memory; // WORKSPACE=page: the memory ws borrows
  fftn::workspace ws;
  // The session's mode when the plan was made (it does not change while plans exist), so that
  // a serial execute reads nothing else: reaching it through state() cost a guard check, a
  // call and three saved registers per execute, about 16 instructions that FFTW's backend
  // does not run (X49).
  call mode = call::serial;
  // A plan with edges (mode call::edges): how it executes (serial, executor or team), its
  // factor arrays (all ones, so that fbench's check of the transform holds) and, for
  // EDGE_ARM=loop, the sides the user's loops walk and the buffer an accumulating loop
  // transforms into.
  call edge_mode = call::serial;
  bool single = false;
  bool loop = false;
  side_geometry in_side, out_side;
  std::vector<double> in_factor, out_factor; // f64 storage of either precision's values
  mutable std::vector<double> temp;          // EDGE_ARM=loop with acc: the transform's result
  fftn::exec_args args{};
};

// Factor storage of `elements` values of `kind` in the plan's precision, every value one.
std::vector<double> ones(fftn::factor_kind kind, std::size_t elements, bool single) {
  const std::size_t reals = elements * (kind == fftn::factor_kind::complex ? 2 : 1);
  const std::size_t words = single ? (reals + 1) / 2 : reals;
  std::vector<double> v(std::max<std::size_t>(words, 1), 0.0);
  for (std::size_t i = 0; i < reals; i += kind == fftn::factor_kind::complex ? 2 : 1) {
    if (single) {
      reinterpret_cast<float*>(v.data())[i] = 1.0F;
    } else {
      v[i] = 1.0;
    }
  }
  return v;
}

detail::edge_view edge_view_of(const side_geometry& g, std::vector<std::ptrdiff_t>& axis,
                               std::vector<std::ptrdiff_t>& batch) {
  detail::edge_view v;
  v.kind = g.kind;
  v.accumulate = g.accumulate;
  if (g.kind != fftn::factor_kind::none) {
    axis.assign(g.factor_stride.begin(),
                g.factor_stride.begin() + static_cast<std::ptrdiff_t>(g.rank));
    batch.assign(1, g.factor_batch);
    v.axis_strides = axis;
    v.batch_strides = batch;
  }
  return v;
}

// A case's plan with the session's edges: their sides and factor arrays in `r`, and the plan
// with the edges (the library's arm) or without (the loop arm). nullptr when the edges do
// not apply to the case (a complex factor on a real side, accumulation in place, a layout
// with more than one batch dimension), which the case reports as unsupported.
template <class In, class Out>
std::shared_ptr<const detail::plan_core> make_edge_core(const fbench_case& c,
                                                        const fftn::options& o, plan_record& r) {
  const session& s = state();
  const fftn::shape<In, Out> sh = shape_of<In, Out>(c);
  if (c.outer > 1 || c.interleave > 1 || (s.edge_out.accumulate && o.in_place)) {
    return nullptr;
  }
  r.in_side = side_of(sh, c, s.edge_in, false);
  r.out_side = side_of(sh, c, s.edge_out, true);
  if ((!r.in_side.complex_data && r.in_side.kind == fftn::factor_kind::complex) ||
      (!r.out_side.complex_data && r.out_side.kind == fftn::factor_kind::complex)) {
    return nullptr;
  }
  r.single = c.precision == fbench_f32;
  r.loop = s.edge_loop;
  if (r.in_side.kind != fftn::factor_kind::none) {
    r.in_factor = ones(r.in_side.kind, r.in_side.factor_elements, r.single);
    r.args.in_factor = r.in_factor.data();
  }
  if (r.out_side.kind != fftn::factor_kind::none) {
    r.out_factor = ones(r.out_side.kind, r.out_side.factor_elements, r.single);
    r.args.out_factor = r.out_factor.data();
  }
  if (r.loop) {
    if (r.out_side.accumulate) {
      const std::size_t bytes =
          static_cast<std::size_t>(c.batch) * static_cast<std::size_t>(c.out_dist) * sizeof(Out);
      r.temp.assign(bytes / sizeof(double) + 1, 0.0);
    }
    return detail::make_plan_core(detail::view_of(sh), o, s.settings);
  }
  std::vector<std::ptrdiff_t> in_axis, in_batch, out_axis, out_batch;
  const detail::edges_view e{.in = edge_view_of(r.in_side, in_axis, in_batch),
                             .out = edge_view_of(r.out_side, out_axis, out_batch)};
  return detail::make_plan_core_with_edges(detail::view_of(sh), o, e, s.settings);
}

template <class T>
std::shared_ptr<const detail::plan_core> make_edge_core(const fbench_case& c,
                                                        const fftn::options& o, plan_record& r) {
  using cplx = std::complex<T>;
  if (c.kind == fbench_c2c) {
    return make_edge_core<cplx, cplx>(c, o, r);
  }
  if (c.kind == fbench_r2c) {
    return make_edge_core<T, cplx>(c, o, r);
  }
  return make_edge_core<cplx, T>(c, o, r);
}

// The plan's lines of its describe that say where its edges went ("edges: ...") and which of
// the edge driver's batched forms it takes ("edge forms: ...", "edge row blocks: ..."), so that a
// run's notes show the plans that take them (X65's P1).
std::string edge_lines_of(const std::string& describe) {
  std::string out;
  std::size_t at = 0;
  while (at < describe.size()) {
    const std::size_t end = std::min(describe.find('\n', at), describe.size());
    const std::string_view line = std::string_view(describe).substr(at, end - at);
    if (line.find("edges: ") != std::string_view::npos || line.starts_with("edge forms: ") ||
        line.starts_with("edge row blocks: ")) {
      out += (out.empty() ? "" : "; ") + std::string(line);
    }
    at = end + 1;
  }
  return out;
}

// ---- Executors -------------------------------------------------------------------------

#ifdef __APPLE__
// Raises every lane of the executor to the harness's QoS class: one task per lane, each
// waiting until all have started, so that no lane runs two (bounded, should a runtime give
// fewer threads than asked).
struct raise_job {
  std::atomic<int> started{0};
  int lanes = 0;
};

void raise_lane(void* arg, int /*index*/) {
  auto& job = *static_cast<raise_job*>(arg);
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  job.started.fetch_add(1);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (job.started.load() < job.lanes && std::chrono::steady_clock::now() < deadline) {
  }
}

void raise_lanes(const executor_type& e, int lanes) {
  raise_job job;
  job.lanes = lanes;
  e.bulk(lanes, &raise_lane, &job);
}
#endif

#ifdef FBENCH_FFTN_OMP
#ifndef _WIN32
// Which OpenMP runtime the process loaded: LLVM's (and Intel's) libomp exports the __kmpc_
// entry points, GCC's libgomp does not.
std::string omp_runtime() {
  const std::string file = fbench::backends::defining_file("omp_get_num_threads");
  const std::string base = file.substr(file.find_last_of('/') + 1);
  const bool kmp = dlsym(RTLD_DEFAULT, "__kmpc_fork_call") != nullptr;
  return std::string(kmp ? "libomp" : "libgomp") + (base.empty() ? "" : " (" + base + ")");
}
#else
std::string omp_runtime() { return "unknown"; }
#endif

// The environment variables that change how the runtime places and waits.
std::string omp_environment() {
  fbench::json env;
  for (const char* name : {
           "OMP_PLACES",
           "OMP_PROC_BIND",
           "OMP_WAIT_POLICY",
           "OMP_DYNAMIC",
           "OMP_THREAD_LIMIT",
           "GOMP_SPINCOUNT",
           "KMP_BLOCKTIME",
           "KMP_AFFINITY",
       }) {
    if (const std::optional<std::string> value = detail::environment_variable(name)) {
      env.str(name, *value);
    }
  }
  return env.object();
}
#endif

// The plan of a record with edges, executed as its edge_mode says, with the factor arrays
// (args) or without (the loop arm's plain plan).
void run_core(const plan_record& r, const void* in, void* out, const fftn::exec_args* args) {
  const session& s = state();
  switch (r.edge_mode) {
  case call::serial:
  case call::edges:
  case call::pipeline: // never an edge mode
    if (args != nullptr) {
      detail::execute_all(*r.core, in, out, r.ws, *args);
    } else {
      detail::execute_all(*r.core, in, out, r.ws);
    }
    return;
  case call::executor:
    if (args != nullptr) {
      detail::execute_on(*r.core, in, out, regrouped(*s.exec, s.group_size), r.ws, *args);
    } else {
      detail::execute_on(*r.core, in, out, regrouped(*s.exec, s.group_size), r.ws);
    }
    return;
  case call::team:
#ifdef FBENCH_FFTN_OMP
  {
    const int group = s.group_size;
#pragma omp parallel num_threads(s.threads)
    {
      fftn::team me = fftn::omp::this_team();
      if (group >= 0) {
        me.group_size = group;
      }
      if (args != nullptr) {
        detail::execute_team(*r.core, in, out, me, r.ws, fftn::completion::all, *args);
      } else {
        detail::execute_team(*r.core, in, out, me, r.ws, fftn::completion::all);
      }
    }
  }
#endif
    return;
  }
}

// The user's loop over one side, on the executor's lanes when the arm is threaded (a user
// with threads shares their loop out too): each task takes a run of whole lines.
struct loop_job {
  const plan_record* r;
  const side_geometry* g;
  void* data;
  const void* src;
  const void* factor;
  std::size_t lines;
  int tasks;
};

void loop_lines(const loop_job& j, std::size_t first, std::size_t count) {
  if (j.r->single) {
    edge_lines_f32(*j.g, static_cast<float*>(j.data), static_cast<const float*>(j.src),
                   static_cast<const float*>(j.factor), first, count);
  } else {
    edge_lines_f64(*j.g, static_cast<double*>(j.data), static_cast<const double*>(j.src),
                   static_cast<const double*>(j.factor), first, count);
  }
}

void loop_task(void* arg, int index) {
  const auto& j = *static_cast<const loop_job*>(arg);
  const auto t = static_cast<std::size_t>(j.tasks);
  const auto i = static_cast<std::size_t>(index);
  const std::size_t first = j.lines * i / t;
  loop_lines(j, first, j.lines * (i + 1) / t - first);
}

void run_loop(const plan_record& r, const side_geometry& g, void* data, const void* src,
              const void* factor) {
  const session& s = state();
  loop_job j{&r, &g, data, src, factor, g.lines(), 1};
  if (r.edge_mode == call::serial || j.lines < 2) {
    loop_lines(j, 0, j.lines);
    return;
  }
  j.tasks = static_cast<int>(std::min<std::size_t>(j.lines, static_cast<std::size_t>(s.threads)));
  s.exec->bulk(j.tasks, &loop_task, &j);
}

// A plan with edges: the library's (the plan with the edges and the factor arrays), or the
// user's (EDGE_ARM=loop: the input's loop in place, the plain plan, then the output's loop,
// which with accumulation adds the transform's result, made in a buffer of its own).
[[gnu::noinline]] void execute_edges(const plan_record& r, const void* in, void* out) {
  if (!r.loop) {
    run_core(r, in, out, &r.args);
    return;
  }
  if (r.in_side.kind != fftn::factor_kind::none) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): the user's loop scales its input
    run_loop(r, r.in_side, const_cast<void*>(in), nullptr, r.args.in_factor);
  }
  const bool acc = r.out_side.accumulate;
  void* dst = acc ? static_cast<void*>(r.temp.data()) : out;
  run_core(r, in, dst, nullptr);
  if (acc || r.out_side.kind != fftn::factor_kind::none) {
    run_loop(r, r.out_side, out, acc ? dst : nullptr, r.args.out_factor);
  }
}

// The executes through an executor or a team, out of line so that the serial execute needs
// no frame: with them inline GCC 14 saved three registers before testing the mode (X49).
[[gnu::noinline]] void execute_threaded(const plan_record& r, const void* in, void* out) {
  const session& s = state();
  switch (r.mode) {
  case call::edges:
    execute_edges(r, in, out);
    return;
  case call::pipeline:
    r.pipe->run(in, out);
    return;
  case call::serial:
    return;
  case call::executor:
    detail::execute_on(*r.core, in, out, regrouped(*s.exec, s.group_size), r.ws);
    return;
  case call::team:
#ifdef FBENCH_FFTN_OMP
  {
    const int group = s.group_size;
#pragma omp parallel num_threads(s.threads)
    {
      fftn::team me = fftn::omp::this_team();
      if (group >= 0) {
        me.group_size = group;
      }
      detail::execute_team(*r.core, in, out, me, r.ws, fftn::completion::all);
    }
  }
#endif
    return;
  }
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  session& s = state();
  s = session{};
  bool threaded = false;
  fbench::json overrides;
  for (const auto& [key, value] : fbench::backends::parse_options(options)) {
    std::string error;
    if (key == "PLANNING") {
      if (value != "rules" && value != "quick" && value != "measure") {
        say(message, message_size, "PLANNING is rules, quick or measure");
        return fbench_error;
      }
      s.effort = fftn::planning::rules;
      if (value == "quick") {
        s.effort = fftn::planning::quick;
      } else if (value == "measure") {
        s.effort = fftn::planning::measure;
      }
      s.planning = value;
    } else if (key == "MEASURE_SECONDS") {
      const std::optional<double> v = fbench::fftn_tuning::parse_real(value);
      if (!v) {
        say(message, message_size, "MEASURE_SECONDS is a non-negative number");
        return fbench_error;
      }
      s.measure_seconds = *v;
    } else if (key == "MEASURE_SCRATCH") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 1, false);
      if (!v) {
        say(message, message_size, "MEASURE_SCRATCH is a whole number of bytes from 1");
        return fbench_error;
      }
      s.measure_scratch = *v;
    } else if (key == "ISA") {
      const std::optional<fftn::isa> target = isa_from_name(value);
      if (!target) {
        say(message, message_size,
            "FFTN options: ISA=scalar|avx2|avx512|neon|neon_fcma|...|best (got ISA=" + value + ")");
        return fbench_error;
      }
      s.isa = *target;
    } else if (key == "THREADS") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 1, false);
      if (!v || *v > 4096) {
        say(message, message_size, "THREADS is a whole number from 1 to 4096");
        return fbench_error;
      }
      s.threads = static_cast<int>(*v);
      threaded = true;
    } else if (key == "GROUP_SIZE") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 0, false);
      if (!v || *v > 4096) {
        say(message, message_size, "GROUP_SIZE is a whole number from 0 to 4096");
        return fbench_error;
      }
      s.group_size = static_cast<int>(*v);
#ifdef FBENCH_FFTN_OMP
    } else if (key == "CALL") {
      if (value != "executor" && value != "team") {
        say(message, message_size, "CALL is executor or team");
        return fbench_error;
      }
      s.mode = value == "team" ? call::team : call::executor;
#endif
    } else if (key == "WORKSPACE") {
      if (value != "library" && value != "page") {
        say(message, message_size, "WORKSPACE is library or page");
        return fbench_error;
      }
      s.workspace_page = value == "page";
    } else if (key == "EDGE_IN" || key == "EDGE_OUT") {
      const bool output = key == "EDGE_OUT";
      const std::optional<edge_spec> e = parse_edge(value, output);
      if (!e) {
        say(message, message_size,
            key + " is KIND.LAYOUT" + (output ? "[.acc] or acc" : "") +
                ", KIND real or complex, LAYOUT line, full, s2 or row (got " + value + ")");
        return fbench_error;
      }
      (output ? s.edge_out : s.edge_in) = *e;
    } else if (key == "EDGE_ARM") {
      if (value != "library" && value != "loop") {
        say(message, message_size, "EDGE_ARM is library or loop");
        return fbench_error;
      }
      s.edge_loop = value == "loop";
    } else if (key == "PIPELINE") {
      if (value != "fused" && value != "separate") {
        say(message, message_size, "PIPELINE is fused or separate");
        return fbench_error;
      }
      s.pipeline = value == "fused" ? pipe_arm::fused : pipe_arm::separate;
    } else if (key == "RADICES") {
      try {
        s.settings.radices = parse_radices(value);
      } catch (const std::exception& e) {
        say(message, message_size, e.what());
        return fbench_error;
      }
      s.radices_text = value;
    } else if (key == "FUSE_REAL_RADIX") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 1, false);
      if (!v || *v > 64) {
        say(message, message_size, "FUSE_REAL_RADIX is a whole number from 1 to 64");
        return fbench_error;
      }
      s.settings.fuse_real_radix = *v;
      overrides.str(key, value);
    } else if (key == "LEAF_LIMIT") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 1, false);
      if (!v || *v > 4096) {
        say(message, message_size, "LEAF_LIMIT is a whole number from 1 to 4096");
        return fbench_error;
      }
      s.settings.leaf_limit = *v;
    } else if (key == "STRATEGY") {
      if (value != "rules" && value != "chain" && value != "grid" && value != "lanes" &&
          value != "depth") {
        say(message, message_size, "STRATEGY is rules, chain, grid, lanes or depth");
        return fbench_error;
      }
      s.settings.strategy = detail::strategy_choice::rules;
      if (value == "chain") {
        s.settings.strategy = detail::strategy_choice::chain;
      } else if (value == "grid") {
        s.settings.strategy = detail::strategy_choice::grid;
      } else if (value == "lanes") {
        s.settings.strategy = detail::strategy_choice::lanes;
      } else if (value == "depth") {
        s.settings.strategy = detail::strategy_choice::depth;
      }
      s.strategy = value;
    } else if (key == "DEPTH_TOPS") {
      try {
        s.settings.depth_tops = parse_radices(value, "DEPTH_TOPS");
      } catch (const std::exception& e) {
        say(message, message_size, e.what());
        return fbench_error;
      }
      if (s.settings.depth_tops.size() > 2) {
        say(message, message_size, "DEPTH_TOPS is one or two factors, f1 or f1.f2");
        return fbench_error;
      }
      s.depth_tops_text = value;
    } else if (key == "LANE_BLOCK") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 1, false);
      if (!v || *v > 4096) {
        say(message, message_size, "LANE_BLOCK is a whole number from 1 to 4096");
        return fbench_error;
      }
      s.settings.lane_block = *v;
    } else if (key == "BATCH_LEAVES") {
      if (value != "rules" && value != "never" && value != "own" && value != "narrower") {
        say(message, message_size, "BATCH_LEAVES is rules, never, own or narrower");
        return fbench_error;
      }
      s.settings.batch_leaves = detail::batch_leaf_choice::rules;
      if (value == "never") {
        s.settings.batch_leaves = detail::batch_leaf_choice::never;
      } else if (value == "own") {
        s.settings.batch_leaves = detail::batch_leaf_choice::own;
      } else if (value == "narrower") {
        s.settings.batch_leaves = detail::batch_leaf_choice::narrower;
      }
      s.batch_leaves = value;
    } else if (key == "COLUMNS") {
      if (value != "lanes" && value != "chain") {
        say(message, message_size, "COLUMNS is lanes or chain");
        return fbench_error;
      }
      s.settings.lanes_columns = value == "lanes";
    } else if (key == "ADJACENT_MIN_BYTES") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 0, false);
      if (!v) {
        say(message, message_size, "ADJACENT_MIN_BYTES is a whole number of bytes");
        return fbench_error;
      }
      s.settings.profile.adjacent_min_bytes = *v;
    } else if (key == "COLUMN_BLOCK") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 1, false);
      if (!v || *v > 4096) {
        say(message, message_size, "COLUMN_BLOCK is a whole number from 1 to 4096");
        return fbench_error;
      }
      s.settings.column_block = *v;
    } else if (key == "COLUMN_RADICES") {
      try {
        s.settings.column_radices = parse_radices(value, "COLUMN_RADICES");
      } catch (const std::exception& e) {
        say(message, message_size, e.what());
        return fbench_error;
      }
      s.column_radices_text = value;
    } else if (key == "LAYOUT") {
      if (value != "rules" && value != "interleaved" && value != "split") {
        say(message, message_size, "LAYOUT is rules, interleaved or split");
        return fbench_error;
      }
      s.settings.layout = detail::layout_choice::rules;
      if (value == "interleaved") {
        s.settings.layout = detail::layout_choice::interleaved;
      } else if (value == "split") {
        s.settings.layout = detail::layout_choice::split;
      }
      s.layout = value;
    } else if (key == "GRID_FACTORS") {
      try {
        s.settings.grid_factors = parse_radices(value, "GRID_FACTORS");
      } catch (const std::exception& e) {
        say(message, message_size, e.what());
        return fbench_error;
      }
      s.factors_text = value;
    } else if (key == "RADER") {
      if (value != "rules" && value != "0" && value != "1") {
        say(message, message_size, "RADER is rules, 0 or 1");
        return fbench_error;
      }
      s.settings.rader = -1; // rules
      if (value != "rules") {
        s.settings.rader = value == "0" ? 0 : 1;
      }
      s.rader = value;
    } else if (key == "GRID_PHASES") {
      if (value != "2" && value != "3") {
        say(message, message_size, "GRID_PHASES is 2 or 3");
        return fbench_error;
      }
      s.settings.grid_phases = value == "2" ? 2 : 3;
    } else if (key == "GRID_PHASE_RADIX") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 2, false);
      if (!v || *v > 64) {
        say(message, message_size, "GRID_PHASE_RADIX is a whole number from 2 to 64");
        return fbench_error;
      }
      s.settings.grid_phase_radix = *v;
    } else if (fbench::fftn_tuning::apply_tuning(s.settings, key, value, error)) {
      if (!error.empty()) {
        say(message, message_size, error);
        return fbench_error;
      }
      overrides.str(key, value);
    } else {
      say(message, message_size,
          "FFTN options: unknown option " + key +
              " (bench/README.md, \"FFTN options\", lists them)");
      return fbench_error;
    }
  }
  if (!threaded && (s.group_size >= 0 || s.mode != call::serial)) {
    say(message, message_size, "GROUP_SIZE and CALL need THREADS");
    return fbench_error;
  }
  if (s.edge_loop && !s.edge_in.any() && !s.edge_out.any()) {
    say(message, message_size, "EDGE_ARM needs EDGE_IN or EDGE_OUT");
    return fbench_error;
  }
  if (s.pipeline != pipe_arm::none && (s.edge_in.any() || s.edge_out.any() ||
                                       s.mode == call::team || !s.settings.radices.empty())) {
    say(message, message_size,
        "PIPELINE does not combine with EDGE_IN, EDGE_OUT, CALL=team or RADICES");
    return fbench_error;
  }
  try {
    if (threaded) {
      if (s.mode == call::serial) {
        s.mode = call::executor;
      }
      s.exec = std::make_unique<executor_type>(s.threads);
#ifdef __APPLE__
      raise_lanes(*s.exec, s.threads);
#endif
    }
    // Throws for a target this build or processor lacks.
    const fftn::c2c_plan<double> probe(fftn::rows_c2c<double>(8, 1, fftn::direction::forward),
                                       fftn::options{.force_isa = s.isa});
    fbench::json d;
    d.str("library", "fftn")
        .str("version", fftn::version())
        .str("isa", fftn::isa_name(probe.instruction_set()))
        .boolean("isa_forced", s.isa != fftn::isa::best)
        // PIPELINE takes forward c2c and r2c rows out of place only: those kinds are partial.
        .raw("supports", s.pipeline != pipe_arm::none ? "[]" : R"(["c2c","r2c","c2r"])")
        .raw("precisions", R"(["f64","f32"])")
        .str("planning", s.planning)
        .str("rader", s.rader)
        .str("strategy", s.strategy)
        .str("layout", s.layout);
    if (s.workspace_page) {
      d.str("workspace", "page");
    }
    if (s.edge_in.any()) {
      d.str("edge_in", s.edge_in.text);
    }
    if (s.edge_out.any()) {
      d.str("edge_out", s.edge_out.text);
    }
    if (s.edge_in.any() || s.edge_out.any()) {
      d.str("edge_arm", s.edge_loop ? "loop" : "library");
    }
    if (s.pipeline != pipe_arm::none) {
      d.raw("partial", R"(["c2c","r2c"])");
      // The harness then checks each row against its input delayed by one (backend.hpp).
      d.str("pipeline", s.pipeline == pipe_arm::fused ? "fused" : "separate")
          .str("output", "delay");
    }
    if (s.measure_seconds != 0) {
      d.str("measure_seconds", std::to_string(s.measure_seconds));
    }
    if (s.measure_scratch != 0) {
      d.integer("measure_scratch", static_cast<std::int64_t>(s.measure_scratch));
    }
    if (!s.factors_text.empty()) {
      d.str("grid_factors", s.factors_text);
    }
    if (!s.settings.radices.empty()) {
      d.str("radices", s.radices_text);
    }
    if (s.settings.grid_phases != 0) {
      d.integer("grid_phases", static_cast<std::int64_t>(s.settings.grid_phases));
    }
    if (s.settings.grid_phase_radix != 0) {
      d.integer("grid_phase_radix", static_cast<std::int64_t>(s.settings.grid_phase_radix));
    }
    if (s.settings.leaf_limit != 0) {
      d.integer("leaf_limit", static_cast<std::int64_t>(s.settings.leaf_limit));
    }
    if (!s.settings.depth_tops.empty()) {
      d.str("depth_tops", s.depth_tops_text);
    }
    if (s.settings.lane_block != 0) {
      d.integer("lane_block", static_cast<std::int64_t>(s.settings.lane_block));
    }
    if (s.batch_leaves != "rules") {
      d.str("batch_leaves", s.batch_leaves);
    }
    if (!s.settings.lanes_columns) {
      d.str("columns", "chain");
    }
    if (s.settings.column_block != 0) {
      d.integer("column_block", static_cast<std::int64_t>(s.settings.column_block));
    }
    if (!s.settings.column_radices.empty()) {
      d.str("column_radices", s.column_radices_text);
    }
#ifdef FBENCH_FFTN_OMP
    // Always, so that this binary's identity differs from fbench-fftn's even without options
    // (fbench refuses arms that are meant to differ but describe themselves identically).
    d.str("omp_runtime", omp_runtime()).raw("omp_env", omp_environment());
#endif
    if (threaded) {
      const regrouped e(*s.exec, s.group_size);
      d.integer("threads", s.threads)
          .str("executor", s.mode == call::team ? "omp::this_team" : executor_name)
          .integer("group_size", e.group_size());
#ifdef __APPLE__
      d.str("worker_qos", "user-interactive");
#endif
    } else {
      d.str("executor", "serial");
    }
    s.overrides = overrides.inner();
    if (!s.overrides.empty()) {
      d.raw("tuning", overrides.object());
    }
    s.description = d.inner();
  } catch (const std::exception& e) {
    say(message, message_size, e.what());
    return fbench_error;
  }
  return fbench_ok;
}

void fbench_end() { state() = session{}; }

const char* fbench_describe() { return state().description.c_str(); }

std::int32_t fbench_create(const fbench_case* c, void* /*in*/, void* out, void** plan,
                           char* message, std::size_t message_size) {
  *plan = nullptr;
  try {
    const session& s = state();
    const bool forced = !s.settings.radices.empty();
    if (forced) {
      check_radices_fit(*c);
    }
    auto record = std::make_unique<plan_record>();
    record->mode = s.mode;
    const fftn::options opts{
        .in_place = c->in_place != 0,
        .force_isa = s.isa,
        .effort = s.effort,
        .measure_seconds = s.measure_seconds,
        .measure_scratch_bytes = s.measure_scratch,
    };
    if (s.pipeline != pipe_arm::none) {
      record->mode = call::pipeline;
      record->pipe = c->precision == fbench_f32 ? make_pipeline_arm<float>(*c, opts)
                                                : make_pipeline_arm<double>(*c, opts);
      if (!record->pipe) {
        say(message, message_size,
            "PIPELINE takes rows of c2c forward and r2c cases, out of place, only");
        return fbench_unsupported;
      }
      say(message, message_size, record->pipe->note());
      *plan = record.release();
      return fbench_ok;
    }
    const bool edges = s.edge_in.any() || s.edge_out.any();
    if (edges) {
      record->edge_mode = s.mode;
      record->mode = call::edges;
      record->core = c->precision == fbench_f32 ? make_edge_core<float>(*c, opts, *record)
                                                : make_edge_core<double>(*c, opts, *record);
      if (!record->core) {
        say(message, message_size,
            "the edges do not apply (a complex factor on a real side, accumulation in place, or "
            "more than one batch dimension)");
        return fbench_unsupported;
      }
      if (record->out_side.accumulate && out != nullptr) {
        // Accumulation adds to the output: fbench's check of the transform holds on zeros.
        const std::size_t element =
            (record->single ? sizeof(float) : sizeof(double)) * (c->kind == fbench_c2r ? 1 : 2);
        std::memset(out, 0,
                    static_cast<std::size_t>(c->batch) * static_cast<std::size_t>(c->out_dist) *
                        element);
      }
    } else {
      record->core =
          c->precision == fbench_f32 ? make_core<float>(*c, opts) : make_core<double>(*c, opts);
    }
    const std::string described = detail::describe(*record->core);
    // The axis line ends after the radices, or goes on after a comma (a split run's ", split
    // from pass k", a real row's pairing).
    const std::string used = "radices " + radix_list(s.settings.radices);
    if (forced && described.find(used + "\n") == std::string::npos &&
        described.find(used + ", ") == std::string::npos) {
      throw std::logic_error("the plan does not use RADICES=" + s.radices_text + ":\n" + described);
    }
    const std::size_t ws_bytes = detail::workspace_bytes(*record->core, s.threads);
    if (s.workspace_page && ws_bytes != 0) {
      // X76: the team state, at the start of the workspace, then shares its page with no
      // allocation before it (on the heap: the plan's own twiddles, read by every member).
      constexpr std::size_t page = 4096;
      record->ws_memory = std::make_unique<std::byte[]>(ws_bytes + page);
      const auto at = reinterpret_cast<std::uintptr_t>(record->ws_memory.get());
      std::byte* start = record->ws_memory.get() + (page - at % page) % page;
      record->ws = fftn::workspace(std::span<std::byte>(start, ws_bytes));
    } else {
      record->ws = fftn::workspace(ws_bytes);
    }
    std::string note = axis_lines(described);
    if (edges) {
      const std::string edge_text = edge_lines_of(described);
      note += (edge_text.empty() ? "" : "; ") + edge_text;
    }
    if (s.effort != fftn::planning::rules) {
      // The chosen program first, then the account: P1 reads rules/chosen, P2 the count.
      const detail::plan_core& k = *record->core;
      note = "chosen: " + k.program_text + "; timed " + std::to_string(k.candidates_timed) +
             "; rank " + std::to_string(k.chosen_rank) + "; rules/chosen " +
             std::to_string(k.rules_over_chosen) + "; " + note;
    }
    say(message, message_size, note);
    *plan = record.release();
    return fbench_ok;
  } catch (const std::exception& e) {
    say(message, message_size, e.what());
    return fbench_error;
  }
}

// NOLINTNEXTLINE(misc-const-correctness): the harness fixes this C signature
void fbench_execute(const void* plan, void* in, void* out) {
  const auto& r = *static_cast<const plan_record*>(plan);
  if (r.mode != call::serial) {
    execute_threaded(r, in, out);
    return;
  }
  // What plan::execute calls: one call into the library (X28), not transform_count and
  // execute_core, whose extra call fbench timed until X37; and nothing before it (X49).
  detail::execute_all(*r.core, in, out, r.ws);
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
}
}
