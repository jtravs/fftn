// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The probe of experiment X76 (dev/experiments/X76-bimodal-864.md): one process of a threaded
// batch of rows, timed as fbench times it, with what might pick its speed recorded beside the
// time. Linux only (sched_getcpu, perf_event_open, /proc).
//
//   x76_probe [KEY=VALUE...]
//
//   n=864 batch=384 sign=1 place=out   the case (c2c, double; sign 1 is backward)
//   threads=16                         lanes of fftn::omp::executor
//   offset_in=0 offset_out=0           bytes into a page where the data start (fbench's offsets)
//   huge=0|1                           1: buffers 2 MiB aligned with madvise(MADV_HUGEPAGE)
//   prelude=BYTES                      first allocate, touch and free a block of BYTES, as an
//                                      earlier case of an fbench process does (0: none)
//   before=KIND:N:BATCH                first plan and execute rows of another case (c2c, r2c or
//                                      c2r) through the same executor for before_seconds=0.05,
//                                      then free them, as an earlier case of an fbench process
//   idle=SECONDS                       after the checking execute, keep the team idle this long
//                                      (fbench checks the result on its main thread meanwhile)
//   seconds=1 batch_time=0.002 warmup=0.02   timing: fbench's batches, for `seconds` in all
//   sample_every=10                    batches between samples of each lane's CPU
//   pmu=EVENTS                         per-lane counters, as fbench's --pmu (empty: none)
//   any fbench FFTN tuning option (bench/backends/fftn_tuning.hpp), e.g. ANCHORED_RUNS=1
//
// One JSON line: the plan's description; the buffers' addresses modulo 4 KiB and 2 MiB and
// the huge-page kilobytes of their mappings; "fbench_ns", the median of the first 7 batches
// after the warm-up (what one fbench round records); every batch's time per call; per lane
// its thread, the CPUs it was seen on, its migrations and context switches, and its counts per
// call; and the busy fraction, over the timing, of the process's CPUs' SMT siblings and of the
// rest of their socket.

#include "backends/fftn_tuning.hpp"
#include "core/plan_core.hpp"
#include "harness/json.hpp"

#include <fftn/detail/core.hpp>
#include <fftn/fftn.hpp>
#include <fftn/omp.hpp>

#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace {

using steady = std::chrono::steady_clock;
namespace detail = fftn::detail;
using cplx = std::complex<double>;

struct settings {
  std::size_t n = 864;
  std::size_t batch = 384;
  int sign = 1;
  bool in_place = false;
  int threads = 16;
  std::size_t offset_in = 0;
  std::size_t offset_out = 0;
  bool huge = false;
  std::size_t prelude = 0;
  std::string before;
  double before_seconds = 0.05;
  double idle = 0;
  double seconds = 1.0;
  double batch_time = 2e-3;
  double warmup = 0.02;
  int sample_every = 10;
  std::string pmu;
  detail::build_settings build;
  std::string tuning; // the tuning options given, as JSON members
};

template <class T> T number(std::string_view text, std::string_view what) {
  T v{};
  const auto res = std::from_chars(text.data(), text.data() + text.size(), v);
  if (res.ec != std::errc{} || res.ptr != text.data() + text.size()) {
    throw std::invalid_argument("bad " + std::string(what) + ": " + std::string(text));
  }
  return v;
}

double real_number(std::string_view text, std::string_view what) {
  const std::string s(text);
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || end != s.c_str() + s.size() || !(v > 0)) {
    throw std::invalid_argument("bad " + std::string(what) + ": " + s);
  }
  return v;
}

settings parse(int argc, char** argv) {
  settings s;
  fbench::json tuning;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const std::size_t eq = arg.find('=');
    if (eq == std::string_view::npos) {
      throw std::invalid_argument("arguments are KEY=VALUE, not " + std::string(arg));
    }
    const std::string_view key = arg.substr(0, eq);
    const std::string_view value = arg.substr(eq + 1);
    std::string error;
    if (key == "n") {
      s.n = number<std::size_t>(value, key);
    } else if (key == "batch") {
      s.batch = number<std::size_t>(value, key);
    } else if (key == "sign") {
      s.sign = number<int>(value, key);
    } else if (key == "place") {
      if (value != "in" && value != "out") {
        throw std::invalid_argument("place is in or out");
      }
      s.in_place = value == "in";
    } else if (key == "threads") {
      s.threads = number<int>(value, key);
    } else if (key == "offset_in") {
      s.offset_in = number<std::size_t>(value, key);
    } else if (key == "offset_out") {
      s.offset_out = number<std::size_t>(value, key);
    } else if (key == "before") {
      s.before = value;
    } else if (key == "before_seconds") {
      s.before_seconds = real_number(value, key);
    } else if (key == "idle") {
      s.idle = real_number(value, key);
    } else if (key == "prelude") {
      s.prelude = number<std::size_t>(value, key);
    } else if (key == "huge") {
      s.huge = number<int>(value, key) != 0;
    } else if (key == "seconds") {
      s.seconds = real_number(value, key);
    } else if (key == "batch_time") {
      s.batch_time = real_number(value, key);
    } else if (key == "warmup") {
      s.warmup = real_number(value, key);
    } else if (key == "sample_every") {
      s.sample_every = std::max(number<int>(value, key), 1);
    } else if (key == "pmu") {
      s.pmu = value;
    } else if (fbench::fftn_tuning::apply_tuning(s.build, key, value, error)) {
      if (!error.empty()) {
        throw std::invalid_argument(error);
      }
      tuning.str(key, value);
    } else {
      throw std::invalid_argument("unknown option " + std::string(key));
    }
  }
  if (s.n < 1 || s.batch < 1 || s.threads < 1 || s.offset_in % 64 != 0 || s.offset_out % 64 != 0 ||
      s.offset_in >= 4096 || s.offset_out >= 4096 || (s.sign != 1 && s.sign != -1)) {
    throw std::invalid_argument("n, batch and threads from 1; sign 1 or -1; offsets multiples of "
                                "64 below 4096");
  }
  s.tuning = tuning.object();
  return s;
}

// ---- Buffers ---------------------------------------------------------------------------

struct free_deleter {
  void operator()(void* p) const noexcept { std::free(p); } // NOLINT(cppcoreguidelines-no-malloc)
};

class buffer {
public:
  buffer(std::size_t bytes, std::size_t offset, bool huge) {
    const std::size_t align = huge ? std::size_t{2} << 20U : 4096;
    const std::size_t total = (bytes + offset + align) / align * align;
    base_.reset(std::aligned_alloc(align, total)); // NOLINT(cppcoreguidelines-no-malloc)
    if (base_ == nullptr) {
      throw std::bad_alloc();
    }
    if (huge && madvise(base_.get(), total, MADV_HUGEPAGE) != 0) {
      throw std::system_error(errno, std::system_category(), "madvise");
    }
    data_ = static_cast<std::byte*>(base_.get()) + offset;
  }
  [[nodiscard]] std::byte* data() const { return data_; }

private:
  std::unique_ptr<void, free_deleter> base_;
  std::byte* data_ = nullptr;
};

// The AnonHugePages kilobytes of the mapping that holds p (/proc/self/smaps), or -1.
std::int64_t huge_kb(const void* p) {
  const auto at = reinterpret_cast<std::uintptr_t>(p);
  std::ifstream smaps("/proc/self/smaps");
  bool inside = false;
  for (std::string line; std::getline(smaps, line);) {
    const std::size_t dash = line.find('-');
    const std::size_t space = line.find(' ');
    if (dash != std::string::npos && space != std::string::npos && dash < space &&
        line.find_first_not_of("0123456789abcdef") == dash) {
      const std::uintptr_t lo = std::stoull(line.substr(0, dash), nullptr, 16);
      const std::uintptr_t hi = std::stoull(line.substr(dash + 1, space - dash - 1), nullptr, 16);
      inside = lo <= at && at < hi;
    } else if (inside && line.starts_with("AnonHugePages:")) {
      return std::stoll(line.substr(14));
    }
  }
  return -1;
}

// ---- Lanes -----------------------------------------------------------------------------

std::int64_t nr_migrations() {
  std::ifstream sched("/proc/thread-self/sched");
  for (std::string line; std::getline(sched, line);) {
    if (line.starts_with("se.nr_migrations")) {
      return std::stoll(line.substr(line.find(':') + 1));
    }
  }
  return -1;
}

// One lane's counter group, opened on the lane's own thread.
struct lane_counters {
  std::vector<int> fds;
  std::vector<std::string> names;
  std::string error;

  void open(const std::string& spec) {
    std::size_t at = 0;
    while (at <= spec.size()) {
      const std::size_t end = std::min(spec.find(',', at), spec.size());
      const std::string token = spec.substr(at, end - at);
      at = end + 1;
      if (token.empty()) {
        continue;
      }
      perf_event_attr attr{};
      attr.size = sizeof(attr);
      const std::size_t eq = token.find('=');
      const std::string name = token.substr(0, eq);
      if (eq == std::string::npos && name == "cycles") {
        attr.type = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_CPU_CYCLES;
      } else if (eq == std::string::npos && name == "instructions") {
        attr.type = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_INSTRUCTIONS;
      } else if (eq == std::string::npos && name == "ref-cycles") {
        attr.type = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_REF_CPU_CYCLES;
      } else if (eq != std::string::npos && token.compare(eq + 1, 2, "0x") == 0) {
        attr.type = PERF_TYPE_RAW;
        attr.config = std::stoull(token.substr(eq + 3), nullptr, 16);
      } else {
        error = "pmu takes cycles, instructions, ref-cycles or NAME=0xCONFIG, not " + token;
        return;
      }
      const bool leader = fds.empty();
      attr.disabled = leader ? 1 : 0;
      attr.exclude_kernel = 1;
      attr.exclude_hv = 1;
      attr.read_format = static_cast<std::uint64_t>(PERF_FORMAT_GROUP) |
                         static_cast<std::uint64_t>(PERF_FORMAT_TOTAL_TIME_ENABLED) |
                         static_cast<std::uint64_t>(PERF_FORMAT_TOTAL_TIME_RUNNING);
      const long fd = syscall(SYS_perf_event_open, &attr, 0, -1, leader ? -1 : fds.front(), 0);
      if (fd < 0) {
        error = "perf_event_open failed for " + token + ": " + std::strerror(errno);
        return;
      }
      fds.push_back(static_cast<int>(fd));
      names.push_back(name);
    }
  }
  void control(unsigned long request) const {
    if (!fds.empty()) {
      ioctl(fds.front(), request, PERF_IOC_FLAG_GROUP);
    }
  }
  // The group's enabled and running times and counts so far; empty without counters.
  [[nodiscard]] std::vector<std::uint64_t> values() const {
    if (fds.empty()) {
      return {};
    }
    std::vector<std::uint64_t> buf(3 + fds.size());
    const auto bytes = static_cast<ssize_t>(buf.size() * sizeof(std::uint64_t));
    if (read(fds.front(), buf.data(), buf.size() * sizeof(std::uint64_t)) != bytes) {
      return {};
    }
    buf.erase(buf.begin()); // the number of events
    return buf;
  }
  // Counts per call from summed deltas of values() over `calls` executes.
  [[nodiscard]] std::string per_call(const std::vector<double>& sum, double calls) const {
    fbench::json out;
    if (fds.empty() || sum.size() != names.size() + 2 || calls <= 0) {
      return out.str("error", error.empty() ? "no counts" : error).object();
    }
    out.num("running", sum[0] > 0 ? sum[1] / sum[0] : 0.0);
    for (std::size_t i = 0; i < names.size(); ++i) {
      out.num(names[i], sum[2 + i] / calls);
    }
    return out.object();
  }
  lane_counters() = default;
  lane_counters(const lane_counters&) = delete;
  lane_counters& operator=(const lane_counters&) = delete;
  lane_counters(lane_counters&&) = default;
  lane_counters& operator=(lane_counters&&) = default;
  ~lane_counters() {
    for (std::size_t i = fds.size(); i > 0; --i) {
      close(fds[i - 1]);
    }
  }
};

struct lane_state {
  long tid = 0;
  std::set<int> cpus;
  std::set<long> tids; // every thread seen running this lane (one when the runtime keeps it)
  std::int64_t migrations_start = 0;
  std::int64_t migrations_end = 0;
  long nvcsw_start = 0;
  long nivcsw_start = 0;
  long nvcsw_end = 0;
  long nivcsw_end = 0;
  long minflt_start = 0;
  long minflt_end = 0;
  lane_counters pmu;
  std::vector<std::uint64_t> last;          // pmu.values() after the previous batch
  std::vector<std::vector<double>> batches; // per batch, the deltas of pmu.values()
};

struct lanes_job {
  std::vector<lane_state>* lanes = nullptr;
  int what = 0; // 0 sample, 1 start, 2 end, 3 open counters
  const std::string* pmu = nullptr;
  std::vector<int>* sample = nullptr; // this sample's CPU per lane
};

void lane_task(void* arg, int index) {
  auto& job = *static_cast<lanes_job*>(arg);
  lane_state& l = job.lanes->at(static_cast<std::size_t>(index));
  const int cpu = sched_getcpu();
  const long tid = syscall(SYS_gettid);
  l.cpus.insert(cpu);
  l.tids.insert(tid);
  l.tid = tid;
  if (job.sample != nullptr) {
    job.sample->at(static_cast<std::size_t>(index)) = cpu;
  }
  rusage ru{};
  if (job.what == 1 || job.what == 2) {
    getrusage(RUSAGE_THREAD, &ru);
  }
  if (job.what == 1) {
    l.migrations_start = nr_migrations();
    l.nvcsw_start = ru.ru_nvcsw;
    l.nivcsw_start = ru.ru_nivcsw;
    l.minflt_start = ru.ru_minflt;
  } else if (job.what == 2) {
    l.migrations_end = nr_migrations();
    l.nvcsw_end = ru.ru_nvcsw;
    l.nivcsw_end = ru.ru_nivcsw;
    l.minflt_end = ru.ru_minflt;
  } else if (job.what == 3) {
    l.pmu.open(*job.pmu);
  }
}

// ---- The machine around the process -------------------------------------------------------

std::vector<int> allowed_cpus() {
  cpu_set_t set;
  CPU_ZERO(&set);
  sched_getaffinity(0, sizeof(set), &set);
  std::vector<int> out;
  for (std::size_t c = 0; c < static_cast<std::size_t>(CPU_SETSIZE); ++c) {
    if (CPU_ISSET(c, &set)) {
      out.push_back(static_cast<int>(c));
    }
  }
  return out;
}

std::string read_line(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  std::getline(f, line);
  return line;
}

// The CPUs of a list such as "0-3,36".
std::vector<int> cpu_list(const std::string& text) {
  std::vector<int> out;
  std::stringstream ss(text);
  for (std::string part; std::getline(ss, part, ',');) {
    const std::size_t dash = part.find('-');
    const int lo = std::stoi(part.substr(0, dash));
    const int hi = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
    for (int c = lo; c <= hi; ++c) {
      out.push_back(c);
    }
  }
  return out;
}

// Busy and total jiffies per CPU (/proc/stat).
std::map<int, std::pair<std::uint64_t, std::uint64_t>> cpu_times() {
  std::map<int, std::pair<std::uint64_t, std::uint64_t>> out;
  std::ifstream stat("/proc/stat");
  for (std::string line; std::getline(stat, line);) {
    if (line.size() < 4 || !line.starts_with("cpu") || line[3] < '0' || line[3] > '9') {
      continue;
    }
    std::stringstream ss(line.substr(3));
    int cpu = 0;
    ss >> cpu;
    std::vector<std::uint64_t> v;
    for (std::uint64_t x = 0; ss >> x;) {
      v.push_back(x);
    }
    std::uint64_t total = 0;
    for (const std::uint64_t x : v) {
      total += x;
    }
    const std::uint64_t idle = v.size() > 4 ? v[3] + v[4] : 0;
    out[cpu] = {total - idle, total};
  }
  return out;
}

double busy_of(const std::vector<int>& cpus,
               const std::map<int, std::pair<std::uint64_t, std::uint64_t>>& a,
               const std::map<int, std::pair<std::uint64_t, std::uint64_t>>& b) {
  double busy = 0;
  for (const int c : cpus) {
    const auto ia = a.find(c);
    const auto ib = b.find(c);
    if (ia == a.end() || ib == b.end()) {
      continue;
    }
    const auto total = static_cast<double>(ib->second.second - ia->second.second);
    if (total > 0) {
      busy += static_cast<double>(ib->second.first - ia->second.first) / total;
    }
  }
  return busy; // in CPUs
}

std::string hex(const void* p) {
  std::ostringstream ss;
  ss << std::hex << reinterpret_cast<std::uintptr_t>(p);
  return ss.str();
}

std::string int_list(const std::vector<int>& v) {
  std::string out(1, '[');
  for (const int x : v) {
    out += (out.size() > 1 ? "," : "") + std::to_string(x);
  }
  return out + ']';
}

// ---- The run -----------------------------------------------------------------------------

// Plans rows of KIND:N:BATCH (out of place), executes them through `exec` for `seconds`, and
// frees everything, as fbench's process does with a case before the next.
template <class In, class Out>
void run_before(std::size_t n, std::size_t batch, double seconds, const fftn::omp::executor& exec,
                int threads) {
  constexpr bool real_in = std::is_same_v<In, double>;
  constexpr bool real_out = std::is_same_v<Out, double>;
  const std::size_t half = n / 2 + 1;
  const std::size_t in_n = real_out ? half : n;
  const std::size_t out_n = real_in ? half : n;
  fftn::shape<In, Out> shape;
  shape.dir = real_out ? fftn::direction::backward : fftn::direction::forward;
  shape.axes.push_back({.n = n, .in_stride = 1, .out_stride = 1});
  shape.batches.push_back({.count = batch,
                           .in_dist = static_cast<std::ptrdiff_t>(in_n),
                           .out_dist = static_cast<std::ptrdiff_t>(out_n)});
  const std::shared_ptr<const detail::plan_core> core =
      detail::make_plan_core(detail::view_of(shape), fftn::options{}, detail::build_settings{});
  const fftn::workspace ws(detail::workspace_bytes(*core, threads));
  std::vector<In> x(in_n * batch);
  std::vector<Out> y(out_n * batch);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = In(static_cast<double>(i % 7) - 3.0);
  }
  const steady::time_point start = steady::now();
  while (std::chrono::duration<double>(steady::now() - start).count() < seconds) {
    detail::execute_on(*core, x.data(), y.data(), exec, ws);
  }
}

void run_before(const std::string& spec, double seconds, const fftn::omp::executor& exec,
                int threads) {
  const std::size_t a = spec.find(':');
  const std::size_t b = spec.find(':', a + 1);
  if (a == std::string::npos || b == std::string::npos) {
    throw std::invalid_argument("before is KIND:N:BATCH, not " + spec);
  }
  const std::string kind = spec.substr(0, a);
  const auto n = number<std::size_t>(std::string_view(spec).substr(a + 1, b - a - 1), "before");
  const auto batch = number<std::size_t>(std::string_view(spec).substr(b + 1), "before");
  if (kind == "c2c") {
    run_before<cplx, cplx>(n, batch, seconds, exec, threads);
  } else if (kind == "r2c") {
    run_before<double, cplx>(n, batch, seconds, exec, threads);
  } else if (kind == "c2r") {
    run_before<cplx, double>(n, batch, seconds, exec, threads);
  } else {
    throw std::invalid_argument("before's kind is c2c, r2c or c2r, not " + kind);
  }
}

int run(const settings& s) {
  const fftn::omp::executor exec(s.threads);
  if (!s.before.empty()) {
    run_before(s.before, s.before_seconds, exec, s.threads);
  }
  const auto rows = static_cast<std::ptrdiff_t>(s.n);
  fftn::shape<cplx, cplx> shape;
  shape.dir = s.sign < 0 ? fftn::direction::forward : fftn::direction::backward;
  shape.axes.push_back({.n = s.n, .in_stride = 1, .out_stride = 1});
  shape.batches.push_back({.count = s.batch, .in_dist = rows, .out_dist = rows});
  const fftn::options opts{.in_place = s.in_place};
  const std::shared_ptr<const detail::plan_core> core =
      detail::make_plan_core(detail::view_of(shape), opts, s.build);
  const fftn::workspace ws(detail::workspace_bytes(*core, s.threads));

  if (s.prelude != 0) {
    // glibc serves a large block by mmap and, when such a block is freed, raises its mmap
    // threshold to that size (up to 32 MiB): later blocks below it come from the heap.
    const std::unique_ptr<void, free_deleter> block(std::malloc(s.prelude)); // NOLINT
    if (block == nullptr) {
      throw std::bad_alloc();
    }
    std::memset(block.get(), 1, s.prelude);
  }
  const std::size_t bytes = s.n * s.batch * sizeof(cplx);
  const buffer in(bytes, s.offset_in, s.huge);
  const std::unique_ptr<buffer> out =
      s.in_place ? nullptr : std::make_unique<buffer>(bytes, s.offset_out, s.huge);
  std::byte* const out_data = s.in_place ? in.data() : out->data();
  {
    // The input as fbench's: written by this thread; the output first touched by the team.
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<cplx> x(s.n * s.batch);
    for (cplx& v : x) {
      v = {u(rng), u(rng)};
    }
    std::memcpy(in.data(), x.data(), bytes);
  }

  std::vector<lane_state> lanes(static_cast<std::size_t>(s.threads));
  lanes_job job{.lanes = &lanes, .what = 0, .pmu = &s.pmu, .sample = nullptr};
  if (!s.pmu.empty()) {
    job.what = 3;
    exec.bulk(s.threads, &lane_task, &job);
  }

  const auto execute = [&] { detail::execute_on(*core, in.data(), out_data, exec, ws); };
  execute(); // fbench's checking execute
  if (s.idle > 0) {
    const steady::time_point idle_start = steady::now();
    while (std::chrono::duration<double>(steady::now() - idle_start).count() < s.idle) {
    }
  }
  const auto time_calls = [&](std::int64_t calls) {
    const steady::time_point t0 = steady::now();
    for (std::int64_t i = 0; i < calls; ++i) {
      execute();
    }
    return std::chrono::duration<double>(steady::now() - t0).count();
  };
  const steady::time_point warm_start = steady::now();
  std::int64_t calls = 1;
  double elapsed = time_calls(calls);
  while (elapsed < 0.5 * s.batch_time) {
    calls *= 2;
    elapsed = time_calls(calls);
  }
  calls = std::max<std::int64_t>(
      1, static_cast<std::int64_t>(std::ceil(static_cast<double>(calls) * s.batch_time / elapsed)));
  while (std::chrono::duration<double>(steady::now() - warm_start).count() < s.warmup) {
    time_calls(calls);
  }

  // The job's CPU set as the job script gives it: with OMP_PROC_BIND the runtime has already
  // narrowed this thread's own mask to one CPU.
  const char* const given = std::getenv("X76_CPUS");
  const std::vector<int> mine = given != nullptr ? cpu_list(given) : allowed_cpus();
  std::vector<int> siblings;
  std::vector<int> socket_rest;
  {
    std::set<int> own(mine.begin(), mine.end());
    std::set<int> sib;
    std::set<int> rest;
    for (const int c : mine) {
      const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
      for (const int t : cpu_list(read_line(base + "thread_siblings_list"))) {
        if (own.count(t) == 0) {
          sib.insert(t);
        }
      }
      std::string package = read_line(base + "package_cpus_list");
      if (package.empty()) {
        package = read_line(base + "core_siblings_list");
      }
      for (const int t : cpu_list(package)) {
        rest.insert(t);
      }
    }
    for (const int c : own) {
      rest.erase(c);
    }
    for (const int c : sib) {
      rest.erase(c);
    }
    siblings.assign(sib.begin(), sib.end());
    socket_rest.assign(rest.begin(), rest.end());
  }

  job.what = 1;
  exec.bulk(s.threads, &lane_task, &job);
  const std::string load_start = read_line("/proc/loadavg");
  const auto times_start = cpu_times();
  std::vector<double> ns;
  std::vector<int> stacked; // per sample: lanes beyond the first on a shared CPU
  std::vector<int> sample(static_cast<std::size_t>(s.threads), -1);
  const steady::time_point timing_start = steady::now();
  std::int64_t counted = 0;
  for (int b = 0; std::chrono::duration<double>(steady::now() - timing_start).count() < s.seconds;
       ++b) {
    for (const lane_state& l : lanes) {
      l.pmu.control(PERF_EVENT_IOC_ENABLE);
    }
    const double t = time_calls(calls);
    for (const lane_state& l : lanes) {
      l.pmu.control(PERF_EVENT_IOC_DISABLE);
    }
    counted += calls;
    for (lane_state& l : lanes) {
      std::vector<std::uint64_t> now = l.pmu.values();
      std::vector<double> delta(now.size());
      for (std::size_t e = 0; e < now.size(); ++e) {
        delta[e] = static_cast<double>(now[e] - (e < l.last.size() ? l.last[e] : 0));
      }
      l.batches.push_back(std::move(delta));
      l.last = std::move(now);
    }
    ns.push_back(std::round(t / static_cast<double>(calls) * 1e10) / 10); // 0.1 ns
    if (b % s.sample_every == 0) {
      job.what = 0;
      job.sample = &sample;
      exec.bulk(s.threads, &lane_task, &job);
      job.sample = nullptr;
      std::map<int, int> per_cpu;
      for (const int c : sample) {
        ++per_cpu[c];
      }
      int extra = 0;
      for (const auto& [c, k] : per_cpu) {
        extra += k - 1;
      }
      stacked.push_back(extra);
    }
  }
  const auto times_end = cpu_times();
  const std::string load_end = read_line("/proc/loadavg");
  job.what = 2;
  exec.bulk(s.threads, &lane_task, &job);

  std::vector<double> first(ns.begin(), ns.begin() + std::min<std::ptrdiff_t>(7, std::ssize(ns)));
  std::ranges::sort(first);
  const double fbench_ns = first.empty() ? 0 : first[first.size() / 2];
  std::vector<double> sorted_ns = ns;
  std::ranges::sort(sorted_ns);
  const double median_ns = sorted_ns.empty() ? 0 : sorted_ns[sorted_ns.size() / 2];

  std::string lanes_text(1, '[');
  for (const lane_state& l : lanes) {
    fbench::json j;
    j.integer("tid", l.tid)
        .integer("tids", std::ssize(l.tids))
        .raw("cpus", int_list(std::vector<int>(l.cpus.begin(), l.cpus.end())))
        .integer("migrations", l.migrations_end - l.migrations_start)
        .integer("nvcsw", l.nvcsw_end - l.nvcsw_start)
        .integer("nivcsw", l.nivcsw_end - l.nivcsw_start)
        .integer("minflt", l.minflt_end - l.minflt_start);
    if (!s.pmu.empty()) {
      // Over the typical batches (within 10 % of the median batch) and over all, so that a
      // stall of a few batches does not hide what the usual execution counts.
      std::vector<double> all;
      std::vector<double> usual;
      std::size_t usual_batches = 0;
      for (std::size_t b = 0; b < l.batches.size(); ++b) {
        const std::vector<double>& d = l.batches[b];
        all.resize(d.size());
        usual.resize(d.size());
        const bool typical = std::abs(ns[b] - median_ns) <= 0.1 * median_ns;
        usual_batches += typical ? 1 : 0;
        for (std::size_t e = 0; e < d.size(); ++e) {
          all[e] += d[e];
          usual[e] += typical ? d[e] : 0.0;
        }
      }
      j.raw("pmu",
            l.pmu.per_call(usual, static_cast<double>(usual_batches) * static_cast<double>(calls)))
          .raw("pmu_all", l.pmu.per_call(all, static_cast<double>(counted)));
    }
    lanes_text += (lanes_text.size() > 1 ? "," : "") + j.object();
  }
  lanes_text += ']';

  const auto mod = [](const void* p, std::uintptr_t m) {
    return static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(p) % m);
  };
  fbench::json o;
  o.str("type", "x76")
      .str("plan", detail::describe(*core))
      .raw("tuning", s.tuning)
      .integer("threads", s.threads)
      .boolean("huge", s.huge)
      .str("in_at", hex(in.data()))
      .str("out_at", hex(out_data))
      .integer("in_mod_4k", mod(in.data(), 4096))
      .integer("out_mod_4k", mod(out_data, 4096))
      .integer("in_mod_2m", mod(in.data(), std::uintptr_t{2} << 20U))
      .integer("out_mod_2m", mod(out_data, std::uintptr_t{2} << 20U))
      .integer("in_huge_kb", huge_kb(in.data()))
      .integer("out_huge_kb", huge_kb(out_data))
      .raw("allowed", int_list(mine))
      .num("siblings_busy", busy_of(siblings, times_start, times_end))
      .integer("siblings", std::ssize(siblings))
      .num("socket_rest_busy", busy_of(socket_rest, times_start, times_end))
      .integer("socket_rest", std::ssize(socket_rest))
      .num("own_busy", busy_of(mine, times_start, times_end))
      .str("load_start", load_start)
      .str("load_end", load_end)
      .integer("calls", calls)
      .num("fbench_ns", fbench_ns)
      .num("median_ns", median_ns)
      .numbers("batches_ns", ns)
      .raw("stacked", int_list(stacked))
      .raw("lanes", lanes_text);
  std::cout << o.object() << '\n';
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(parse(argc, argv));
  } catch (const std::exception& e) {
    std::cerr << "x76_probe: " << e.what() << '\n';
    return 2;
  }
}
