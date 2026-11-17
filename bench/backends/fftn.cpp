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
//   DEPTH_TOPS=f1[.f2]   the depth-first chain's top factors (build_settings::depth_tops;
//                    default the rule's; experiment X46)
//   LANE_BLOCK=B     the lanes strategy's block in rows (build_settings::lane_block; default
//                    the rule's), rounded up to a multiple of the complex count (X7)
//   COLUMNS=lanes|chain   the strided axes of planes and volumes (build_settings::
//                    lanes_columns): lanes over blocks of adjacent columns where they fit
//                    (default), or every column alone through the chain (experiment X40)
//   COLUMN_BLOCK=B   the lanes column phase's block in columns (build_settings::
//                    column_block; default the rule's), rounded up to a multiple of C (X40)
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
//   GRID_TWO_PART_BYTES, BLUESTEIN_POW2_PART_BYTES, LONG_CONVOLUTION_BYTES
//                    the planner's grid fields of the machine profile (schedule.hpp)
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
#include <charconv>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
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

enum class call : unsigned char { serial, executor, team };

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
  std::string overrides;          // the profile and chunking options given, as JSON members
  std::string radices_text;       // RADICES as given, r1.r2... (settings.radices holds them)
  std::string strategy = "rules"; // STRATEGY as given
  std::string layout = "rules";   // LAYOUT as given
  std::string factors_text;       // GRID_FACTORS as given
  std::string depth_tops_text;    // DEPTH_TOPS as given
  std::string rader = "rules";    // RADER as given
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

struct plan_record {
  std::shared_ptr<const detail::plan_core> core;
  fftn::workspace ws;
};

// FFTW's many-interface layout: row-major arrays whose last dimension is embedded in
// `last` elements, transforms `dist` elements apart.
template <class In, class Out> fftn::shape<In, Out> shape_of(const fbench_case& c) {
  fftn::shape<In, Out> s;
  s.dir = c.sign < 0 ? fftn::direction::forward : fftn::direction::backward;
  const auto rank = static_cast<std::size_t>(c.rank);
  for (std::size_t d = 0; d < rank; ++d) {
    s.axes.push_back({.n = static_cast<std::size_t>(c.dims[d])});
  }
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t out_stride = 1;
  for (std::size_t i = rank; i > 0; --i) {
    const std::size_t d = i - 1;
    s.axes[d].in_stride = in_stride;
    s.axes[d].out_stride = out_stride;
    const bool last = d + 1 == rank;
    in_stride *= last ? c.in_last : c.dims[d];
    out_stride *= last ? c.out_last : c.dims[d];
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
  if (c.kind != fbench_c2c || c.rank != 1) {
    throw std::invalid_argument("RADICES=" + s.radices_text + " is for c2c rows only");
  }
  if (std::cmp_not_equal(product, c.dims[0])) {
    throw std::invalid_argument("RADICES=" + s.radices_text + " multiplies to " +
                                std::to_string(product) + ", not the length " +
                                std::to_string(c.dims[0]));
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
    } else if (key == "RADICES") {
      try {
        s.settings.radices = parse_radices(value);
      } catch (const std::exception& e) {
        say(message, message_size, e.what());
        return fbench_error;
      }
      s.radices_text = value;
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
    } else if (key == "COLUMNS") {
      if (value != "lanes" && value != "chain") {
        say(message, message_size, "COLUMNS is lanes or chain");
        return fbench_error;
      }
      s.settings.lanes_columns = value == "lanes";
    } else if (key == "COLUMN_BLOCK") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(value, 1, false);
      if (!v || *v > 4096) {
        say(message, message_size, "COLUMN_BLOCK is a whole number from 1 to 4096");
        return fbench_error;
      }
      s.settings.column_block = *v;
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
        .raw("supports", R"(["c2c","r2c","c2r"])")
        .raw("precisions", R"(["f64","f32"])")
        .str("planning", s.planning)
        .str("rader", s.rader)
        .str("strategy", s.strategy)
        .str("layout", s.layout);
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
    if (!s.settings.lanes_columns) {
      d.str("columns", "chain");
    }
    if (s.settings.column_block != 0) {
      d.integer("column_block", static_cast<std::int64_t>(s.settings.column_block));
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

std::int32_t fbench_create(const fbench_case* c, void* /*in*/, void* /*out*/, void** plan,
                           char* message, std::size_t message_size) {
  *plan = nullptr;
  try {
    const session& s = state();
    const bool forced = !s.settings.radices.empty();
    if (forced) {
      check_radices_fit(*c);
    }
    auto record = std::make_unique<plan_record>();
    const fftn::options opts{
        .in_place = c->in_place != 0,
        .force_isa = s.isa,
        .effort = s.effort,
        .measure_seconds = s.measure_seconds,
        .measure_scratch_bytes = s.measure_scratch,
    };
    record->core =
        c->precision == fbench_f32 ? make_core<float>(*c, opts) : make_core<double>(*c, opts);
    const std::string described = detail::describe(*record->core);
    // The axis line ends after the radices, or after a split run's ", split from pass k".
    const std::string used = "radices " + radix_list(s.settings.radices);
    if (forced && described.find(used + "\n") == std::string::npos &&
        described.find(used + ", split from pass ") == std::string::npos) {
      throw std::logic_error("the plan does not use RADICES=" + s.radices_text + ":\n" + described);
    }
    record->ws = fftn::workspace(detail::workspace_bytes(*record->core, s.threads));
    std::string note = axis_lines(described);
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
  const session& s = state();
  switch (s.mode) {
  case call::serial:
    // What plan::execute calls: one call into the library (X28), not transform_count and
    // execute_core, whose extra call fbench timed until X37.
    detail::execute_all(*r.core, in, out, r.ws);
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

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
}
}
