// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTN: one plan and one workspace per case, serial or threaded.
//
// Every case of rank 1 to 3 and every kind is supported, in double precision, with FFTW's
// embedding (in_last, out_last) and distances. describe reports the instruction set FFTN's
// plans choose on this machine and how the arm executes.
//
// Plans are built and executed through the functions that the public plan<In, Out>
// forwards every call to (include/fftn/plan.hpp: make_plan_core, execute_core, execute_on,
// execute_team), so that an arm can also set the private build settings
// (src/core/plan_core.hpp) for the threading experiments T1 and T5. With none of those
// options the plan is exactly the public one.
//
// Options (bench/README.md, "FFTN options"):
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
// These options reach the library only through the private build settings, as a test's
// do: the library has no switch for them, so dev/KNOBS.toml does not list them.
//
// On macOS the harness raises its own thread to the user-interactive QoS class, which new
// threads do not inherit: the executor's threads are raised by one task each at begin, so
// that they too run on the performance cores.

#include "backends/common.hpp"
#include "core/plan_core.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"
#include "threading/schedule.hpp"

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
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace {

namespace detail = fftn::detail;
using fbench::backends::say;
using cplx = std::complex<double>;

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
  int group_size = -1; // -1: the executor's own
  std::unique_ptr<executor_type> exec;
  std::string overrides; // the profile and chunking options given, as JSON members
  std::string description;
};

session& state() {
  static session s;
  return s;
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

// ---- Options ---------------------------------------------------------------------------

// A whole number at least `least`, or `never` (the largest value) when allowed.
std::optional<std::size_t> parse_size(std::string_view text, std::size_t least, bool never) {
  if (never && text == "never") {
    return std::numeric_limits<std::size_t>::max();
  }
  if (text.empty() || text.find_first_not_of("0123456789") != std::string_view::npos ||
      text.size() > 18) {
    return std::nullopt;
  }
  const std::size_t v = std::stoull(std::string(text));
  return v >= least ? std::optional(v) : std::nullopt;
}

std::optional<double> parse_real(std::string_view text) {
  const std::string s(text);
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || end != s.c_str() + s.size() || !(v >= 0) ||
      v > std::numeric_limits<double>::max()) {
    return std::nullopt;
  }
  return v;
}

struct real_field {
  std::string_view key;
  double detail::machine_profile::* member;
};
struct size_field {
  std::string_view key;
  std::size_t detail::machine_profile::* member;
};
struct int_field {
  std::string_view key;
  int detail::machine_profile::* member;
};

constexpr std::array real_fields{
    real_field{.key = "NS_PER_WORK", .member = &detail::machine_profile::ns_per_work},
    real_field{.key = "DISPATCH_NS", .member = &detail::machine_profile::dispatch_ns},
    real_field{.key = "INLINE_MULTIPLE", .member = &detail::machine_profile::inline_multiple},
    real_field{.key = "SYNC_NS", .member = &detail::machine_profile::sync_ns},
    real_field{.key = "SPIN_NS", .member = &detail::machine_profile::spin_ns},
    real_field{.key = "YIELD_NS", .member = &detail::machine_profile::yield_ns},
};
constexpr std::array size_fields{
    size_field{
        .key = "MIN_POINTS_PER_MEMBER",
        .member = &detail::machine_profile::min_points_per_member,
    },
    size_field{.key = "SPLIT_MIN_POINTS", .member = &detail::machine_profile::split_min_points},
    size_field{
        .key = "SPLIT_MIN_POINTS_MESH",
        .member = &detail::machine_profile::split_min_points_mesh,
    },
    size_field{
        .key = "REMAINDER_SPLIT_MIN_POINTS",
        .member = &detail::machine_profile::remainder_split_min_points,
    },
};
constexpr std::array int_fields{
    int_field{.key = "MESH_GROUP_LANES", .member = &detail::machine_profile::mesh_group_lanes},
    int_field{.key = "CLAIM_DIVISOR", .member = &detail::machine_profile::claim_divisor},
};

// Applies one profile or chunking option; false when the key is not one.
bool apply_tuning(session& s, std::string_view key, std::string_view value, std::string& error) {
  const std::string bad = std::string(key) + " has a bad value: " + std::string(value);
  for (const real_field& f : real_fields) {
    if (key == f.key) {
      const std::optional<double> v = parse_real(value);
      if (!v) {
        error = bad + " (a non-negative number)";
      } else {
        s.settings.profile.*f.member = *v;
      }
      return true;
    }
  }
  for (const size_field& f : size_fields) {
    if (key == f.key) {
      const std::optional<std::size_t> v = parse_size(value, 1, true);
      if (!v) {
        error = bad + " (a whole number from 1, or never)";
      } else {
        s.settings.profile.*f.member = *v;
      }
      return true;
    }
  }
  for (const int_field& f : int_fields) {
    if (key == f.key) {
      const std::optional<std::size_t> v = parse_size(value, 1, false);
      if (!v || *v > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = bad + " (a whole number from 1)";
      } else {
        s.settings.profile.*f.member = static_cast<int>(*v);
      }
      return true;
    }
  }
  if (key == "CHUNK_POINTS") {
    const std::optional<std::size_t> v = parse_size(value, 1, false);
    if (!v) {
      error = bad + " (a whole number from 1)";
    } else {
      s.settings.chunk_points = *v;
    }
    return true;
  }
  return false;
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
    if (key == "THREADS") {
      const std::optional<std::size_t> v = parse_size(value, 1, false);
      if (!v || *v > 4096) {
        say(message, message_size, "THREADS is a whole number from 1 to 4096");
        return fbench_error;
      }
      s.threads = static_cast<int>(*v);
      threaded = true;
    } else if (key == "GROUP_SIZE") {
      const std::optional<std::size_t> v = parse_size(value, 0, false);
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
    } else if (apply_tuning(s, key, value, error)) {
      if (!error.empty()) {
        say(message, message_size, error);
        return fbench_error;
      }
      overrides.str(key, value);
    } else {
      say(message, message_size, "unknown FFTN option " + key);
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
    const fftn::c2c_plan<double> probe(fftn::rows_c2c<double>(8, 1, fftn::direction::forward));
    fbench::json d;
    d.str("library", "fftn")
        .str("version", fftn::version())
        .str("isa", fftn::isa_name(probe.instruction_set()))
        .str("precision", "double")
        .raw("supports", R"(["c2c","r2c","c2r"])")
        .str("strategy", "chain");
    if (threaded) {
      const regrouped e(*s.exec, s.group_size);
      d.integer("threads", s.threads)
          .str("executor", s.mode == call::team ? "omp::this_team" : executor_name)
          .integer("group_size", e.group_size());
#ifdef FBENCH_FFTN_OMP
      d.str("omp_runtime", omp_runtime()).raw("omp_env", omp_environment());
#endif
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
    auto record = std::make_unique<plan_record>();
    const fftn::options opts{.in_place = c->in_place != 0};
    if (c->kind == fbench_c2c) {
      record->core = make_core<cplx, cplx>(*c, opts);
    } else if (c->kind == fbench_r2c) {
      record->core = make_core<double, cplx>(*c, opts);
    } else {
      record->core = make_core<cplx, double>(*c, opts);
    }
    record->ws = fftn::workspace(detail::workspace_bytes(*record->core, state().threads));
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
    detail::execute_core(*r.core, 0, detail::transform_count(*r.core), in, out, r.ws);
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
