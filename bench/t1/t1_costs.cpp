// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The harness of experiment T1 (dev/experiments/T1-dispatch-wait-costs.md): the fixed costs
// of team executions on this machine, one JSON line per probe, team size and variant, each
// metric as its median and spread over repetitions inside this process. The driver
// (bench/t1/t1.py) runs it in process rounds and reports medians over rounds.
//
//   t1_costs [run] [--threads 2,4,8] [--probes work,bulk,team,phase,claim,wake] [--reps N]
//            [--pin] [--quick]
//   t1_costs schedule --cases FILE [--threads 8,16] [--group G] [--cpus N] [KEY=VALUE...]
//
// `schedule` prints what the scheduler decides for fbench case lines (the catalogue's form)
// at each team size, under the tuning options given (the fbench FFTN backend's names).
//
// Built as t1_costs and, where the compiler has OpenMP, as t1_costs_omp (one runtime per
// binary): the second adds fftn::omp::executor to `bulk` and `team`, and the threads of a
// parallel region to `team` and `phase`. Harness threads are pinned one per allowed CPU
// with --pin (Linux); FFTN's pool never is, as users get it; OpenMP threads follow
// OMP_PLACES and OMP_PROC_BIND. On macOS every thread is raised to the user-interactive
// QoS class, which keeps it on the performance cores.
//
// The probes use the library's private engine and waiting code (src/threading/), like the
// tests: what they time is what team executions run.

#include "backends/fftn_tuning.hpp"
#include "core/plan_core.hpp"
#include "fbench_build_info.hpp"
#include "harness/json.hpp"
#include "threading/engine.hpp"
#include "threading/schedule.hpp"
#include "threading/team_state.hpp"
#include "threading/wait.hpp"
#include "threading/work_graph.hpp"

#include <fftn/detail/core.hpp>
#include <fftn/fftn.hpp>

#ifdef T1_WITH_OPENMP
#include "platform/environment.hpp"

#include <fftn/omp.hpp>
#include <omp.h>
#endif

#ifdef __APPLE__
#include <pthread/qos.h>
#include <sys/qos.h>
#endif
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace detail = fftn::detail;
using cplx = std::complex<double>;
using steady = std::chrono::steady_clock;

std::int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(steady::now().time_since_epoch())
      .count();
}

void spin_for(std::int64_t ns) {
  const std::int64_t until = now_ns() + ns;
  while (now_ns() < until) {
    detail::cpu_relax();
  }
}

// Waits `ns`: by spinning for short gaps (precise), by sleeping for long ones (a waiting
// thread must not take a CPU that the threads being woken need).
void gap(std::int64_t ns) {
  if (ns >= 1'000'000) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
  } else if (ns > 0) {
    spin_for(ns);
  }
}

void raise_qos() {
#ifdef __APPLE__
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

// ---- Output ------------------------------------------------------------------------------

std::string summary(std::vector<double> v) {
  std::ranges::sort(v);
  const auto at = [&v](double q) {
    return v[static_cast<std::size_t>(std::lround(q * static_cast<double>(v.size() - 1)))];
  };
  return fbench::json()
      .num("median", at(0.5))
      .num("p10", at(0.1))
      .num("p90", at(0.9))
      .num("min", v.front())
      .num("max", v.back())
      .object();
}

using metric_map = std::map<std::string, std::vector<double>>;

struct probe_key {
  std::string probe;
  std::string executor; // what ran the members
  int threads = 0;
  int group = 0;
  std::string variant;
};

void emit(const probe_key& k, const metric_map& metrics) {
  fbench::json m;
  std::size_t reps = 0;
  for (const auto& [name, values] : metrics) {
    if (!values.empty()) {
      m.raw(name, summary(values));
      reps = std::max(reps, values.size());
    }
  }
  std::cout << fbench::json()
                   .str("type", "probe")
                   .str("probe", k.probe)
                   .str("executor", k.executor)
                   .integer("threads", k.threads)
                   .integer("group", k.group)
                   .str("variant", k.variant)
                   .integer("reps", static_cast<std::int64_t>(reps))
                   .raw("metrics", m.object())
                   .object()
            << '\n'
            << std::flush;
}

// ---- Settings ----------------------------------------------------------------------------

struct settings {
  std::vector<int> threads;
  std::set<std::string> probes{"work", "bulk", "team", "phase", "claim", "wake"};
  int reps = 200;
  bool pin = false;
  bool quick = false;
  // schedule
  std::string cases;
  int group = -1;
  int cpus = -1;
  detail::build_settings build;
};

std::vector<int> parse_list(std::string_view text) {
  std::vector<int> out;
  std::stringstream in{std::string(text)};
  for (std::string item; std::getline(in, item, ',');) {
    const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(item, 1, false);
    if (!v || *v > 4096) {
      throw std::invalid_argument("bad team size: " + item);
    }
    out.push_back(static_cast<int>(*v));
  }
  return out;
}

std::set<std::string> parse_names(std::string_view text) {
  std::set<std::string> out;
  std::stringstream in{std::string(text)};
  for (std::string item; std::getline(in, item, ',');) {
    out.insert(item);
  }
  return out;
}

// ---- Placement ---------------------------------------------------------------------------

// The CPUs this process may run on, in order (Linux); empty where not known.
std::vector<int> allowed_cpus() {
  std::vector<int> out;
#ifdef __linux__
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int c = 0; c < CPU_SETSIZE; ++c) {
      if (CPU_ISSET(c, &set)) {
        out.push_back(c);
      }
    }
  }
#endif
  return out;
}

void pin_self(int cpu) {
#ifdef __linux__
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }
#else
  (void)cpu;
#endif
}

// The calling thread's CPU mask, restored when the scope ends: the crew pins its caller.
class affinity_scope {
public:
  affinity_scope() {
#ifdef __linux__
    CPU_ZERO(&saved_);
    ok_ = pthread_getaffinity_np(pthread_self(), sizeof(saved_), &saved_) == 0;
#endif
  }
  affinity_scope(const affinity_scope&) = delete;
  affinity_scope& operator=(const affinity_scope&) = delete;
  affinity_scope(affinity_scope&&) = delete;
  affinity_scope& operator=(affinity_scope&&) = delete;
  ~affinity_scope() {
#ifdef __linux__
    if (ok_) {
      pthread_setaffinity_np(pthread_self(), sizeof(saved_), &saved_);
    }
#endif
  }

private:
#ifdef __linux__
  cpu_set_t saved_{};
  bool ok_ = false;
#endif
};

// A sense-reversing spin barrier for the harness's own threads.
class spin_barrier {
public:
  explicit spin_barrier(int members) : members_(members) {}
  void arrive_and_wait() {
    const std::uint64_t phase = phase_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == members_) {
      arrived_.store(0, std::memory_order_relaxed);
      phase_.store(phase + 1, std::memory_order_release);
      return;
    }
    while (phase_.load(std::memory_order_acquire) == phase) {
      detail::cpu_relax();
    }
  }

private:
  alignas(detail::sync_line) std::atomic<int> arrived_{0};
  int members_;
  alignas(detail::sync_line) std::atomic<std::uint64_t> phase_{0};
};

// k harness threads (the caller is member 0) that run one function together, then wait
// (spinning, then yielding) for the next. Members sit on `cpus` when given.
class crew {
public:
  crew(int members, std::vector<int> cpus) : members_(members), cpus_(std::move(cpus)) {
    pin_self(cpu_of(0));
    for (int r = 1; r < members_; ++r) {
      threads_.emplace_back([this, r] { work(r); });
    }
  }
  crew(const crew&) = delete;
  crew& operator=(const crew&) = delete;
  crew(crew&&) = delete;
  crew& operator=(crew&&) = delete;
  ~crew() {
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_acq_rel);
    for (std::thread& t : threads_) {
      t.join();
    }
  }

  // Runs fn(rank) on every member; returns when all have returned.
  void run(const std::function<void(int)>& fn) {
    fn_ = &fn;
    done_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_acq_rel);
    fn(0);
    while (done_.load(std::memory_order_acquire) != members_ - 1) {
      detail::cpu_relax();
    }
  }

private:
  [[nodiscard]] int cpu_of(int rank) const {
    return static_cast<std::size_t>(rank) < cpus_.size() ? cpus_[static_cast<std::size_t>(rank)]
                                                         : -1;
  }

  void work(int rank) {
    raise_qos();
    pin_self(cpu_of(rank));
    std::uint64_t seen = 0;
    for (;;) {
      std::uint64_t g = generation_.load(std::memory_order_acquire);
      for (unsigned round = 0; g == seen; ++round) {
        if (round < 4096) {
          detail::cpu_relax();
        } else {
          std::this_thread::yield();
        }
        g = generation_.load(std::memory_order_acquire);
      }
      seen = g;
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
      (*fn_)(rank);
      done_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  alignas(detail::sync_line) std::atomic<int> done_{0};
  int members_;
  const std::function<void(int)>* fn_ = nullptr;
  std::vector<int> cpus_;
  std::vector<std::thread> threads_;
  affinity_scope caller_; // restored at destruction (members are built before the body pins)
  std::atomic<bool> stop_{false};
  alignas(detail::sync_line) std::atomic<std::uint64_t> generation_{0};
};

// The CPUs of a packed team of k (the first k allowed), or none when not pinning.
std::vector<int> packed(const settings& s, int k) {
  std::vector<int> cpus;
  if (s.pin) {
    const std::vector<int> all = allowed_cpus();
    for (std::size_t i = 0; i < all.size() && std::cmp_less(i, k); ++i) {
      cpus.push_back(all[i]);
    }
  }
  return cpus;
}

// ---- Executors ---------------------------------------------------------------------------

// Raises every lane of an executor to the harness's QoS class (macOS): one task per lane,
// each waiting until all have started, so that no lane runs two.
struct lane_job {
  std::atomic<int> started{0};
  int lanes = 0;
};

void raise_lane(void* arg, int /*index*/) {
  auto& job = *static_cast<lane_job*>(arg);
  raise_qos();
  job.started.fetch_add(1);
  const std::int64_t deadline = now_ns() + 1'000'000'000;
  while (job.started.load() < job.lanes && now_ns() < deadline) {
    detail::cpu_relax();
  }
}

template <class E> void raise_lanes(const E& e, int lanes) {
#ifdef __APPLE__
  lane_job job;
  job.lanes = lanes;
  e.bulk(lanes, &raise_lane, &job);
#else
  (void)e;
  (void)lanes;
#endif
}

// ---- Probe: work -------------------------------------------------------------------------

// Serial time per execution of a plan, over `batches` timed batches of at least `batch_ns`.
template <class Run> std::vector<double> time_runs(Run run, int batches, std::int64_t batch_ns) {
  constexpr int max_count = 16'777'216;
  int count = 1;
  for (;;) {
    const std::int64_t t0 = now_ns();
    for (int i = 0; i < count; ++i) {
      run();
    }
    if (now_ns() - t0 >= batch_ns || count >= max_count) {
      break;
    }
    count *= 2;
  }
  std::vector<double> out;
  for (int b = 0; b < batches; ++b) {
    const std::int64_t t0 = now_ns();
    for (int i = 0; i < count; ++i) {
      run();
    }
    out.push_back(static_cast<double>(now_ns() - t0) / count);
  }
  return out;
}

struct row_plan {
  std::shared_ptr<const detail::plan_core> core;
  fftn::workspace ws;
  std::vector<cplx> in;
  std::vector<cplx> out;
};

row_plan make_rows(std::size_t n, std::size_t count, const detail::build_settings& b, int lanes) {
  const fftn::shape<cplx, cplx> s = fftn::rows_c2c<double>(n, count, fftn::direction::forward);
  row_plan p;
  p.core = detail::make_plan_core(detail::view_of(s), fftn::options(), b);
  p.ws = fftn::workspace(detail::workspace_bytes(*p.core, lanes));
  p.in.resize(n * count);
  for (std::size_t i = 0; i < p.in.size(); ++i) {
    p.in[i] = cplx(static_cast<double>(i % 7) - 3.0, static_cast<double>(i % 5) - 2.0);
  }
  p.out.resize(n * count);
  return p;
}

void probe_work(const settings& s) {
  const std::size_t total = s.quick ? std::size_t{1} << 14U : std::size_t{1} << 20U;
  const std::int64_t batch_ns = s.quick ? 200'000 : 5'000'000;
  for (std::size_t n = 256; n <= (s.quick ? 4096U : 262144U); n *= 4) {
    const std::size_t count = std::max<std::size_t>(total / n, 1);
    row_plan p = make_rows(n, count, detail::build_settings(), 1);
    const std::vector<double> t = time_runs(
        [&p] {
          detail::execute_core(*p.core, 0, detail::transform_count(*p.core), p.in.data(),
                               p.out.data(), p.ws);
        },
        s.quick ? 3 : 9, batch_ns);
    std::vector<double> per_work;
    per_work.reserve(t.size());
    for (const double v : t) {
      per_work.push_back(v / (static_cast<double>(count) * p.core->graph.work));
    }
    emit(
        {
            .probe = "work",
            .executor = "serial",
            .threads = 1,
            .variant = "c2c-" + std::to_string(n) + "-b" + std::to_string(count),
        },
        {{"exec_ns", t}, {"ns_per_work", per_work}});
  }
}

// ---- Probe: bulk -------------------------------------------------------------------------

struct bulk_job {
  alignas(detail::sync_line) std::atomic<int> arrived{0};
  int lanes = 0;
  std::int64_t deadline = 0;
  std::vector<std::int64_t> start;
  std::vector<std::int64_t> end;
  std::atomic<bool> incomplete{false};
};

// One task per lane: records when it starts, waits until every lane has started (bounded),
// records when it ends. No lane can run two tasks, so `start` shows when each lane joined.
void bulk_task(void* arg, int index) {
  auto& job = *static_cast<bulk_job*>(arg);
  const auto i = static_cast<std::size_t>(index);
  job.start[i] = now_ns();
  job.arrived.fetch_add(1, std::memory_order_acq_rel);
  while (job.arrived.load(std::memory_order_acquire) < job.lanes) {
    detail::cpu_relax();
    if (now_ns() > job.deadline) {
      job.incomplete.store(true);
      break;
    }
  }
  job.end[i] = now_ns();
}

struct gap_variant {
  const char* name;
  std::int64_t ns;
};

std::vector<gap_variant> gaps(const settings& s) {
  if (s.quick) {
    return {{.name = "hot", .ns = 0}, {.name = "parked", .ns = 2'000'000}};
  }
  // After 40 us the pool's threads are in their yield stage, after 2 ms blocked (spin 20 us,
  // yield 50 us: src/threading/thread_pool.cpp).
  return {
      {.name = "hot", .ns = 0},
      {.name = "yield", .ns = 40'000},
      {.name = "parked", .ns = 2'000'000},
  };
}

int reps_for(const settings& s, std::int64_t gap_ns) {
  // Long gaps: fewer repetitions, so that a run stays within minutes.
  return gap_ns >= 1'000'000 ? std::max(s.reps / 4, 5) : s.reps;
}

template <class E> void probe_bulk(const settings& s, const E& e, const char* name, int lanes) {
  const int group = e.group_size();
  for (const gap_variant& g : gaps(s)) {
    metric_map m;
    bool incomplete = false;
    const int reps = reps_for(s, g.ns);
    for (int r = -3; r < reps; ++r) {
      bulk_job job;
      job.lanes = lanes;
      job.start.assign(static_cast<std::size_t>(lanes), 0);
      job.end.assign(static_cast<std::size_t>(lanes), 0);
      gap(g.ns);
      const std::int64_t t0 = now_ns();
      job.deadline = t0 + 1'000'000'000;
      e.bulk(lanes, &bulk_task, &job);
      const std::int64_t t1 = now_ns();
      incomplete = incomplete || job.incomplete.load();
      if (r < 0) {
        continue; // warm-up
      }
      std::vector<std::int64_t> starts = job.start;
      std::ranges::sort(starts);
      const std::int64_t last_end = *std::ranges::max_element(job.end);
      m["start_last"].push_back(static_cast<double>(starts.back() - t0));
      m["start_median"].push_back(static_cast<double>(starts[starts.size() / 2] - t0));
      m["return"].push_back(static_cast<double>(t1 - last_end));
      m["round_trip"].push_back(static_cast<double>(t1 - t0));
    }
    emit(
        {
            .probe = "bulk",
            .executor = name,
            .threads = lanes,
            .group = group,
            .variant = std::string(g.name) + (incomplete ? "-incomplete" : ""),
        },
        m);
  }
}

// ---- Probe: team -------------------------------------------------------------------------

// One 64-point row per lane, scheduled with dispatch_ns 0 so that rule 1 does not run it
// inline: the fixed cost of a team execution beyond its work.
detail::build_settings dispatched() {
  detail::build_settings b;
  b.profile.dispatch_ns = 0;
  return b;
}

double serial_one(const settings& s) {
  row_plan p = make_rows(64, 1, detail::build_settings(), 1);
  const std::vector<double> t =
      time_runs([&p] { detail::execute_core(*p.core, 0, 1, p.in.data(), p.out.data(), p.ws); }, 5,
                s.quick ? 100'000 : 2'000'000);
  std::vector<double> sorted = t;
  std::ranges::sort(sorted);
  return sorted[sorted.size() / 2];
}

template <class E>
void probe_team_executor(const settings& s, const E& e, const char* name, int lanes) {
  row_plan p = make_rows(64, static_cast<std::size_t>(lanes), dispatched(), lanes);
  const double one = serial_one(s);
  for (const gap_variant& g : gaps(s)) {
    metric_map m;
    const int reps = reps_for(s, g.ns);
    for (int r = -3; r < reps; ++r) {
      gap(g.ns);
      const std::int64_t t0 = now_ns();
      detail::execute_on(*p.core, p.in.data(), p.out.data(), e, p.ws);
      const auto t = static_cast<double>(now_ns() - t0);
      if (r >= 0) {
        m["exec_ns"].push_back(t);
        m["overhead_ns"].push_back(t - one);
      }
    }
    m["serial_row_ns"].push_back(one);
    emit(
        {
            .probe = "team",
            .executor = name,
            .threads = lanes,
            .group = e.group_size(),
            .variant = g.name,
        },
        m);
  }
}

// The team overload with every member already running (harness threads): the engine's
// fixed cost without any dispatch.
void probe_team_crew(const settings& s, int lanes) {
  row_plan p = make_rows(64, static_cast<std::size_t>(lanes), dispatched(), lanes);
  const double one = serial_one(s);
  const int group = fftn::cache_group_size(lanes);
  std::vector<double> times(static_cast<std::size_t>(s.reps), 0);
  spin_barrier barrier(lanes);
  crew c(lanes, packed(s, lanes));
  c.run([&](int rank) {
    const fftn::team me{.rank = rank, .size = lanes, .group_size = group};
    for (int r = -3; r < s.reps; ++r) {
      barrier.arrive_and_wait();
      const std::int64_t t0 = now_ns();
      detail::execute_team(*p.core, p.in.data(), p.out.data(), me, p.ws, fftn::completion::all);
      if (rank == 0 && r >= 0) {
        times[static_cast<std::size_t>(r)] = static_cast<double>(now_ns() - t0);
      }
    }
  });
  metric_map m{{"exec_ns", times}, {"serial_row_ns", {one}}};
  for (const double t : times) {
    m["overhead_ns"].push_back(t - one);
  }
  emit(
      {
          .probe = "team",
          .executor = s.pin ? "crew-pinned" : "crew",
          .threads = lanes,
          .group = group,
          .variant = "running",
      },
      m);
}

#ifdef T1_WITH_OPENMP
void probe_team_region(const settings& s, int lanes) {
  row_plan p = make_rows(64, static_cast<std::size_t>(lanes), dispatched(), lanes);
  const double one = serial_one(s);
  std::vector<double> times(static_cast<std::size_t>(s.reps), 0);
  int group = 0;
#pragma omp parallel num_threads(lanes)
  {
    raise_qos();
    const fftn::team me = fftn::omp::this_team();
#pragma omp single
    group = me.group_size;
    for (int r = -3; r < s.reps; ++r) {
#pragma omp barrier
      const std::int64_t t0 = now_ns();
      detail::execute_team(*p.core, p.in.data(), p.out.data(), me, p.ws, fftn::completion::all);
      if (me.rank == 0 && r >= 0) {
        times[static_cast<std::size_t>(r)] = static_cast<double>(now_ns() - t0);
      }
    }
  }
  metric_map m{{"exec_ns", times}, {"serial_row_ns", {one}}};
  for (const double t : times) {
    m["overhead_ns"].push_back(t - one);
  }
  emit(
      {
          .probe = "team",
          .executor = "omp-region",
          .threads = lanes,
          .group = group,
          .variant = "running",
      },
      m);
}
#endif

// ---- Probe: phase ------------------------------------------------------------------------

// The claiming engine with empty (or fixed-length) chunks: what a phase boundary costs.
class null_env {
public:
  null_env(detail::wait_budget budget, std::int64_t chunk_ns)
      : budget_(budget), chunk_ns_(chunk_ns) {}
  void point() const noexcept {}
  template <class Ready>
  void wait(detail::sync_word w, const detail::sync_word* blocked, Ready ready) const noexcept {
    detail::wait_for(w, blocked, ready, budget_);
  }
  static void wake(detail::sync_word w, const detail::sync_word* blocked) noexcept {
    detail::wake(w, blocked);
  }
  void run_whole(std::size_t /*first*/, std::size_t count, int /*lane*/) const noexcept {
    spin_for(chunk_ns_ * static_cast<std::int64_t>(count));
  }
  void run_chunk(std::size_t /*task*/, std::size_t /*phase*/, std::uint32_t /*chunk*/,
                 int /*owner*/, int /*lane*/) const noexcept {
    spin_for(chunk_ns_);
  }

private:
  detail::wait_budget budget_;
  std::int64_t chunk_ns_;
};

// One task of `phases` phases of k chunks.
detail::work_graph synthetic(int phases, int k) {
  detail::work_graph g;
  const auto chunks = static_cast<std::uint32_t>(k);
  for (int p = 0; p < phases; ++p) {
    g.phases.push_back(
        {.chunks = chunks, .first = static_cast<std::uint64_t>(p) * chunks, .work = 1});
  }
  g.tasks = 1;
  g.chunks = static_cast<std::uint64_t>(phases) * chunks;
  g.widest = chunks;
  g.work = 1e12;
  g.points = std::size_t{1} << 30U;
  return g;
}

// A profile under which the synthetic task is split among all k members; waiting as the
// default profile waits.
detail::machine_profile split_all() {
  detail::machine_profile p = detail::default_machine_profile();
  p.dispatch_ns = 0;
  p.sync_ns = 0;
  p.min_points_per_member = 1;
  p.split_min_points = 1;
  p.split_min_points_mesh = 1;
  p.remainder_split_min_points = 1;
  p.cpus = 0;
  return p;
}

detail::wait_budget default_budget() {
  const detail::machine_profile p = detail::default_machine_profile();
  return {
      .spin = std::chrono::nanoseconds(static_cast<std::int64_t>(p.spin_ns)),
      .yield = std::chrono::nanoseconds(static_cast<std::int64_t>(p.yield_ns)),
  };
}

// Aligned team state for k lanes, in a vector (no owning raw pointer).
struct team_state {
  explicit team_state(int k)
      : bytes(detail::team_layout{static_cast<std::size_t>(k)}.bytes() + detail::sync_line) {}
  std::byte* aligned() {
    const auto address = reinterpret_cast<std::uintptr_t>(bytes.data());
    return bytes.data() + (detail::sync_line - address % detail::sync_line) % detail::sync_line;
  }
  std::vector<std::byte> bytes;
  std::atomic<std::uint64_t> flag{0};
};

constexpr int long_phases = 64;

struct phase_setup {
  detail::work_graph one;
  detail::work_graph many;
  detail::schedule plan_one;
  detail::schedule plan_many;
  detail::lane_map lanes;
};

phase_setup make_phase_setup(int k) {
  phase_setup ps;
  ps.one = synthetic(1, k);
  ps.many = synthetic(long_phases, k);
  ps.lanes = {.lanes = k, .group_size = 0};
  const detail::machine_profile p = split_all();
  ps.plan_one = detail::make_schedule(ps.one, ps.lanes, p);
  ps.plan_many = detail::make_schedule(ps.many, ps.lanes, p);
  const detail::group_share share = detail::share_of(ps.plan_many, ps.lanes, 0);
  if (share.split != 1 || share.cap != k) {
    throw std::logic_error("phase probe: the synthetic task is not split among every member");
  }
  return ps;
}

// One member's alternating executions of the 1-phase and 64-phase tasks; member 0 records
// each execution's time. `sync` is called before each execution (a barrier).
template <class Sync>
void phase_member(int rank, const phase_setup& ps, team_state& st, const null_env& env, int reps,
                  Sync sync, std::vector<double>& t1, std::vector<double>& t64) {
  const detail::team_job one{
      .graph = &ps.one,
      .plan = ps.plan_one,
      .lanes = ps.lanes,
      .state = st.aligned(),
      .flag = detail::sync_word(&st.flag),
      .rank = rank,
  };
  detail::team_job many = one;
  many.graph = &ps.many;
  many.plan = ps.plan_many;
  for (int r = -3; r < reps; ++r) {
    for (int which = 0; which < 2; ++which) {
      sync();
      const std::int64_t t0 = now_ns();
      null_env e = env;
      detail::team_engine<null_env>(which == 0 ? one : many, e).run();
      if (rank == 0 && r >= 0) {
        (which == 0 ? t1 : t64)[static_cast<std::size_t>(r)] = static_cast<double>(now_ns() - t0);
      }
    }
  }
}

metric_map phase_metrics(const std::vector<double>& t1, const std::vector<double>& t64,
                         std::int64_t chunk_ns) {
  metric_map m{{"exec_1_phase_ns", t1}, {"exec_64_phases_ns", t64}};
  for (std::size_t i = 0; i < t1.size(); ++i) {
    m["per_phase_ns"].push_back((t64[i] - t1[i]) / (long_phases - 1) -
                                static_cast<double>(chunk_ns));
  }
  return m;
}

void probe_phase_crew(const settings& s, int k) {
  const phase_setup ps = make_phase_setup(k);
  for (const std::int64_t chunk_ns : {std::int64_t{0}, std::int64_t{1000}}) {
    team_state st(k);
    const null_env env(default_budget(), chunk_ns);
    std::vector<double> t1(static_cast<std::size_t>(s.reps), 0);
    std::vector<double> t64(static_cast<std::size_t>(s.reps), 0);
    spin_barrier barrier(k);
    crew c(k, packed(s, k));
    c.run([&](int rank) {
      phase_member(rank, ps, st, env, s.reps, [&barrier] { barrier.arrive_and_wait(); }, t1, t64);
    });
    emit(
        {
            .probe = "phase",
            .executor = s.pin ? "crew-pinned" : "crew",
            .threads = k,
            .variant = "chunk-" + std::to_string(chunk_ns) + "ns",
        },
        phase_metrics(t1, t64, chunk_ns));
  }
}

#ifdef T1_WITH_OPENMP
void probe_phase_region(const settings& s, int k) {
  const phase_setup ps = make_phase_setup(k);
  for (const std::int64_t chunk_ns : {std::int64_t{0}, std::int64_t{1000}}) {
    team_state st(k);
    const null_env env(default_budget(), chunk_ns);
    std::vector<double> t1(static_cast<std::size_t>(s.reps), 0);
    std::vector<double> t64(static_cast<std::size_t>(s.reps), 0);
    int got = 0;
#pragma omp parallel num_threads(k)
    {
      raise_qos();
#pragma omp single
      got = omp_get_num_threads();
      if (got == k) {
        phase_member(
            omp_get_thread_num(), ps, st, env, s.reps,
            [] {
#pragma omp barrier
            },
            t1, t64);
      }
    }
    if (got == k) {
      emit(
          {
              .probe = "phase",
              .executor = "omp-region",
              .threads = k,
              .variant = "chunk-" + std::to_string(chunk_ns) + "ns",
          },
          phase_metrics(t1, t64, chunk_ns));
    }
  }
}
#endif

// ---- Probe: claim ------------------------------------------------------------------------

void probe_claim(const settings& s, int k) {
  const int per_member = s.quick ? 200 : 5000;
  alignas(detail::sync_line) std::atomic<std::uint64_t> word{0};
  const detail::sync_word w(&word);
  const std::uint64_t tag = 1;
  const auto limit = static_cast<std::uint32_t>(per_member * k);
  std::vector<double> wall(static_cast<std::size_t>(s.reps), 0);
  std::vector<double> own(static_cast<std::size_t>(s.reps), 0);
  spin_barrier barrier(k);
  crew c(k, packed(s, k));
  c.run([&](int rank) {
    for (int r = -1; r < s.reps; ++r) {
      if (rank == 0) {
        word.store(detail::tagged(tag, 0));
      }
      barrier.arrive_and_wait();
      const std::int64_t t0 = now_ns();
      int mine = 0;
      // The engine's claim: read, then compare-exchange the tagged count up by one.
      for (;;) {
        std::uint64_t seen = w.load();
        const std::uint32_t at = detail::count_in(seen, tag);
        if (at >= limit) {
          break;
        }
        if (w.compare_exchange(seen, detail::tagged(tag, at + 1))) {
          ++mine;
        }
      }
      const std::int64_t t1 = now_ns();
      barrier.arrive_and_wait();
      if (rank == 0 && r >= 0) {
        wall[static_cast<std::size_t>(r)] =
            static_cast<double>(now_ns() - t0) / static_cast<double>(limit);
        own[static_cast<std::size_t>(r)] =
            static_cast<double>(t1 - t0) / static_cast<double>(std::max(mine, 1));
      }
    }
  });
  emit(
      {
          .probe = "claim",
          .executor = s.pin ? "crew-pinned" : "crew",
          .threads = k,
          .variant = "one-word",
      },
      {{"ns_per_claim", wall}, {"member0_ns_per_claim", own}});
}

// ---- Probe: wake -------------------------------------------------------------------------

enum class wait_mode : unsigned char { spin, yield, block, policy };

struct wake_case {
  wait_mode mode;
  std::string name;
  std::int64_t gap_ns;
};

std::vector<wake_case> wake_cases(const settings& s) {
  std::vector<wake_case> out{
      {.mode = wait_mode::spin, .name = "spin", .gap_ns = 20'000},
      {.mode = wait_mode::yield, .name = "yield", .gap_ns = 20'000},
      {.mode = wait_mode::block, .name = "atomic-wait", .gap_ns = 1'000'000},
  };
  // The engine's policy after the waiter has waited this long: inside the spin budget,
  // inside the yield budget, and blocked (20 us and 50 us by default).
  const std::vector<std::int64_t> policy_gaps =
      s.quick
          ? std::vector<std::int64_t>{1'000, 200'000}
          : std::vector<std::int64_t>{1'000, 10'000, 30'000, 60'000, 100'000, 300'000, 1'000'000};
  for (const std::int64_t g : policy_gaps) {
    out.push_back({
        .mode = wait_mode::policy,
        .name = "policy-after-" + std::to_string(g / 1000) + "us",
        .gap_ns = g,
    });
  }
  return out;
}

void probe_wake_pair(const settings& s, const std::vector<int>& cpus, const std::string& pair) {
  for (const wake_case& wc : wake_cases(s)) {
    const int reps = reps_for(s, wc.gap_ns);
    alignas(detail::sync_line) std::atomic<std::uint64_t> word{0};
    alignas(detail::sync_line) std::atomic<std::uint64_t> blocked_word{0};
    alignas(detail::sync_line) std::atomic<std::uint64_t> ready{0};
    const detail::sync_word w(&word);
    const detail::sync_word blocked(&blocked_word);
    const detail::wait_budget budget = default_budget();
    std::vector<std::int64_t> sent(static_cast<std::size_t>(reps + 2), 0);
    std::vector<std::int64_t> woke(static_cast<std::size_t>(reps + 2), 0);
    std::vector<double> notify_cost;
    spin_barrier barrier(2);
    crew c(2, cpus);
    c.run([&](int rank) {
      for (int r = -2; r < reps; ++r) {
        // r runs from -2 (warm-up): unsigned wrap-around makes at start at 0.
        const auto at = static_cast<std::size_t>(r) + 2;
        const std::uint64_t seq = at + 1;
        barrier.arrive_and_wait();
        if (rank == 1) {
          ready.store(seq, std::memory_order_release);
          const auto arrived = [seq](std::uint64_t v) { return v == seq; };
          switch (wc.mode) {
          case wait_mode::spin:
            while (word.load(std::memory_order_acquire) != seq) {
              detail::cpu_relax();
            }
            break;
          case wait_mode::yield:
            while (word.load(std::memory_order_acquire) != seq) {
              std::this_thread::yield();
            }
            break;
          case wait_mode::block:
            for (std::uint64_t v = word.load(); v != seq; v = word.load()) {
              word.wait(v);
            }
            break;
          case wait_mode::policy:
            detail::wait_for(w, &blocked, arrived, budget);
            break;
          }
          woke[at] = now_ns();
        } else {
          while (ready.load(std::memory_order_acquire) != seq) {
            detail::cpu_relax();
          }
          gap(wc.gap_ns);
          const std::int64_t t0 = now_ns();
          word.store(seq);
          if (wc.mode == wait_mode::block) {
            word.notify_all();
          } else if (wc.mode == wait_mode::policy) {
            detail::wake(w, &blocked);
          }
          const std::int64_t t1 = now_ns();
          sent[at] = t0;
          if (r >= 0 && (wc.mode == wait_mode::block || wc.mode == wait_mode::policy)) {
            notify_cost.push_back(static_cast<double>(t1 - t0));
          }
        }
      }
    });
    metric_map m;
    for (int r = 0; r < reps; ++r) {
      const auto at = static_cast<std::size_t>(r) + 2;
      m["latency_ns"].push_back(static_cast<double>(woke[at] - sent[at]));
    }
    m["notify_ns"] = notify_cost;
    emit(
        {
            .probe = "wake",
            .executor = cpus.empty() ? "crew" : "crew-pinned",
            .threads = 2,
            .variant = wc.name + "-" + pair,
        },
        m);
  }
}

void probe_wake(const settings& s) {
  if (!s.pin) {
    probe_wake_pair(s, {}, "os");
    return;
  }
  const std::vector<int> all = allowed_cpus();
  if (all.size() >= 2) {
    probe_wake_pair(s, {all[0], all[1]}, "near");
  }
  // A CPU of the next cache group, as a packed team of every allowed CPU would be grouped.
  const auto group = static_cast<std::size_t>(fftn::cache_group_size(static_cast<int>(all.size())));
  if (group > 0 && group < all.size()) {
    probe_wake_pair(s, {all[0], all[group]}, "far");
  }
}

// ---- describe ----------------------------------------------------------------------------

std::string profile_json(const detail::machine_profile& p) {
  return fbench::json()
      .num("ns_per_work", p.ns_per_work)
      .num("dispatch_ns", p.dispatch_ns)
      .num("inline_multiple", p.inline_multiple)
      .num("sync_ns", p.sync_ns)
      .integer("min_points_per_member", static_cast<std::int64_t>(p.min_points_per_member))
      .integer("split_min_points", static_cast<std::int64_t>(p.split_min_points))
      .integer("split_min_points_mesh", static_cast<std::int64_t>(p.split_min_points_mesh))
      .integer("mesh_group_lanes", p.mesh_group_lanes)
      .integer("remainder_split_min_points",
               static_cast<std::int64_t>(p.remainder_split_min_points))
      .num("spin_ns", p.spin_ns)
      .num("yield_ns", p.yield_ns)
      .integer("claim_divisor", p.claim_divisor)
      .integer("cpus", p.cpus)
      .object();
}

void describe(const settings& s) {
  const fftn::c2c_plan<double> probe(fftn::rows_c2c<double>(8, 1, fftn::direction::forward));
  const auto cpus = static_cast<int>(std::thread::hardware_concurrency());
  fbench::json d;
  d.str("type", "describe")
      .str("tool", "t1_costs")
      .str("fftn", fftn::version())
      .str("isa", fftn::isa_name(probe.instruction_set()))
      .str("compiler", fbench::build::compiler)
      .str("build_type", fbench::build::build_type)
      .str("flags", fbench::build::flags)
      .str("sanitize", fbench::build::sanitize)
      .str("commit", fbench::build::commit)
      .integer("cpus", cpus)
      .integer("allowed_cpus", static_cast<std::int64_t>(allowed_cpus().size()))
      .integer("cache_group", fftn::cache_group_size(cpus))
      .boolean("pinned", s.pin)
      .boolean("quick", s.quick)
      .raw("profile", profile_json(detail::default_machine_profile()));
#ifdef T1_WITH_OPENMP
  fbench::json env;
  for (const char* name : {
           "OMP_PLACES",
           "OMP_PROC_BIND",
           "OMP_WAIT_POLICY",
           "GOMP_SPINCOUNT",
           "KMP_BLOCKTIME",
       }) {
    if (const std::optional<std::string> value = detail::environment_variable(name)) {
      env.str(name, *value);
    }
  }
  d.str("openmp", std::to_string(_OPENMP)).raw("omp_env", env.object());
#else
  d.str("openmp", "none");
#endif
  std::cout << d.object() << '\n' << std::flush;
}

// ---- run ---------------------------------------------------------------------------------

std::vector<int> default_threads(bool quick) {
  if (quick) {
    return {2, 3};
  }
  const auto cpus = static_cast<int>(std::thread::hardware_concurrency());
  std::vector<int> out;
  for (int k = 2; k <= std::min(cpus, 64); k *= 2) {
    out.push_back(k);
  }
  return out;
}

int run(settings s) {
  if (s.threads.empty()) {
    s.threads = default_threads(s.quick);
  }
  raise_qos();
  describe(s);
  const auto has = [&s](const char* name) { return s.probes.contains(name); };
  if (has("work")) {
    probe_work(s);
  }
  for (const int k : s.threads) {
    if (has("bulk") || has("team")) {
      const fftn::thread_pool pool(k);
      raise_lanes(pool, k);
      if (has("bulk")) {
        probe_bulk(s, pool, "thread_pool", k);
      }
      if (has("team")) {
        probe_team_executor(s, pool, "thread_pool", k);
      }
    }
#ifdef T1_WITH_OPENMP
    if (has("bulk") || has("team")) {
      const fftn::omp::executor omp(k);
      raise_lanes(omp, k);
      if (has("bulk")) {
        probe_bulk(s, omp, "omp::executor", k);
      }
      if (has("team")) {
        probe_team_executor(s, omp, "omp::executor", k);
      }
    }
    if (has("team")) {
      probe_team_region(s, k);
    }
    if (has("phase")) {
      probe_phase_region(s, k);
    }
#endif
    if (has("team")) {
      probe_team_crew(s, k);
    }
    if (has("phase")) {
      probe_phase_crew(s, k);
    }
    if (has("claim")) {
      probe_claim(s, k);
    }
  }
  if (has("claim")) {
    probe_claim(s, 1);
  }
  if (has("wake")) {
    probe_wake(s);
  }
  std::cout << fbench::json().str("type", "end").object() << '\n';
  return 0;
}

// ---- schedule ----------------------------------------------------------------------------

// An fbench case line's shape, laid out as fbench lays it out (FFTW's conventions).
std::shared_ptr<const detail::plan_core> plan_of(const std::string& line,
                                                 const detail::build_settings& b, std::string& id) {
  std::map<std::string, std::string> f;
  std::stringstream in(line);
  for (std::string token; in >> token;) {
    const std::size_t eq = token.find('=');
    if (eq != std::string::npos) {
      f[token.substr(0, eq)] = token.substr(eq + 1);
    }
  }
  id = f["id"];
  std::vector<std::size_t> dims;
  std::stringstream ds(f["dims"]);
  for (std::string d; std::getline(ds, d, 'x');) {
    dims.push_back(static_cast<std::size_t>(std::stoull(d)));
  }
  const std::string kind = f["kind"];
  const bool in_place = f["place"] == "in";
  const std::size_t n = dims.back();
  const std::size_t half = n / 2 + 1;
  const std::size_t padded = in_place ? 2 * half : n;
  std::size_t in_last = n;
  std::size_t out_last = n;
  detail::transform_kind tk = detail::transform_kind::c2c;
  if (kind == "r2c") {
    in_last = padded;
    out_last = half;
    tk = detail::transform_kind::r2c;
  } else if (kind == "c2r") {
    in_last = half;
    out_last = padded;
    tk = detail::transform_kind::c2r;
  }
  std::vector<fftn::axis> axes(dims.size());
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t out_stride = 1;
  for (std::size_t i = dims.size(); i > 0; --i) {
    const std::size_t d = i - 1;
    axes[d] = {.n = dims[d], .in_stride = in_stride, .out_stride = out_stride};
    in_stride *= static_cast<std::ptrdiff_t>(d + 1 == dims.size() ? in_last : dims[d]);
    out_stride *= static_cast<std::ptrdiff_t>(d + 1 == dims.size() ? out_last : dims[d]);
  }
  const fftn::batch b0{
      .count = static_cast<std::size_t>(std::stoull(f["batch"])),
      .in_dist = in_stride,
      .out_dist = out_stride,
  };
  const std::vector<fftn::batch> batches{b0};
  const detail::shape_view v{
      .kind = tk,
      .prec = detail::precision::f64,
      .dir = f["sign"] == "1" ? fftn::direction::backward : fftn::direction::forward,
      .axes = axes,
      .batches = batches,
  };
  return detail::make_plan_core(v, {.in_place = in_place}, b);
}

int schedule(settings s) {
  if (s.threads.empty()) {
    s.threads = default_threads(false);
  }
  if (s.cpus >= 0) {
    s.build.profile.cpus = s.cpus;
  }
  std::ifstream file(s.cases);
  if (!file) {
    throw std::invalid_argument("cannot read " + s.cases);
  }
  std::cout << fbench::json()
                   .str("type", "describe")
                   .str("tool", "t1_costs schedule")
                   .raw("profile", profile_json(s.build.profile))
                   .integer("chunk_points", static_cast<std::int64_t>(s.build.chunk_points))
                   .object()
            << '\n';
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::string id;
    const std::shared_ptr<const detail::plan_core> core = plan_of(line, s.build, id);
    const detail::work_graph& g = core->graph;
    for (const int k : s.threads) {
      const int group = s.group >= 0 ? s.group : fftn::cache_group_size(k);
      // The executor overload's view: the scheduler asked with the executor's width.
      const detail::lane_map lanes{.lanes = k, .group_size = group};
      const detail::schedule plan = detail::make_schedule(g, lanes, core->profile);
      std::size_t whole = 0;
      std::size_t split = 0;
      int cap = 0;
      for (int h = 0; h < plan.groups; ++h) {
        const detail::group_share share = detail::share_of(plan, lanes, h);
        whole += share.whole;
        split += share.split;
        cap = std::max(cap, share.cap);
      }
      std::cout << fbench::json()
                       .str("type", "schedule")
                       .str("id", id)
                       .integer("threads", k)
                       .integer("group", group)
                       .integer("tasks", static_cast<std::int64_t>(g.tasks))
                       .integer("points", static_cast<std::int64_t>(g.points))
                       .integer("phases", static_cast<std::int64_t>(g.phases.size()))
                       .integer("chunks", static_cast<std::int64_t>(g.chunks))
                       .integer("widest", g.widest)
                       .num("task_ns", plan.task_ns)
                       .boolean("inline", plan.inline_all)
                       .boolean("oversubscribed", plan.oversubscribed)
                       .integer("groups", plan.groups)
                       .integer("lanes_dealt", plan.lanes_dealt)
                       .integer("whole", static_cast<std::int64_t>(whole))
                       .integer("split", static_cast<std::int64_t>(split))
                       .integer("cap", cap)
                       .object()
                << '\n';
    }
  }
  std::cout << fbench::json().str("type", "end").object() << '\n';
  return 0;
}

int main_of(std::span<char* const> args) {
  settings s;
  std::size_t i = 1;
  std::string command = "run";
  if (args.size() > 1 && !std::string_view(args[1]).starts_with("--")) {
    command = args[1];
    i = 2;
  }
  for (; i < args.size(); ++i) {
    const std::string_view a = args[i];
    const auto next = [&]() -> std::string_view {
      if (i + 1 >= args.size()) {
        throw std::invalid_argument(std::string(a) + " needs a value");
      }
      return args[++i];
    };
    if (a == "--threads") {
      s.threads = parse_list(next());
    } else if (a == "--probes") {
      s.probes = parse_names(next());
    } else if (a == "--reps") {
      s.reps = parse_list(next()).at(0);
    } else if (a == "--pin") {
      s.pin = true;
    } else if (a == "--quick") {
      s.quick = true;
      s.reps = std::min(s.reps, 12);
    } else if (a == "--cases") {
      s.cases = next();
    } else if (a == "--group") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(next(), 0, false);
      if (!v || *v > 4096) {
        throw std::invalid_argument("bad --group");
      }
      s.group = static_cast<int>(*v);
    } else if (a == "--cpus") {
      const std::optional<std::size_t> v = fbench::fftn_tuning::parse_size(next(), 0, false);
      if (!v || *v > (std::size_t{1} << 20U)) {
        throw std::invalid_argument("bad --cpus");
      }
      s.cpus = static_cast<int>(*v);
    } else if (const std::size_t eq = a.find('='); eq != std::string_view::npos) {
      std::string error;
      if (!fbench::fftn_tuning::apply_tuning(s.build, a.substr(0, eq), a.substr(eq + 1), error)) {
        throw std::invalid_argument("unknown tuning option " + std::string(a));
      }
      if (!error.empty()) {
        throw std::invalid_argument(error);
      }
    } else {
      throw std::invalid_argument("unknown argument " + std::string(a));
    }
  }
  if (command == "run") {
    return run(s);
  }
  if (command == "schedule") {
    return schedule(s);
  }
  throw std::invalid_argument("unknown command " + command);
}

} // namespace

int main(int argc, char** argv) {
  try {
    return main_of(std::span<char* const>(argv, static_cast<std::size_t>(argc)));
  } catch (const std::exception& e) {
    std::cerr << "t1_costs: " << e.what() << '\n';
    return 2;
  }
}
