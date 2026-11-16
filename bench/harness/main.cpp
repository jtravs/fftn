// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The fbench harness: one process of one arm (DESIGN.md 16). Linked with exactly one backend.
//
//   fbench-<lib> describe                     the library and build, one JSON line
//   fbench-<lib> selftest                     checks the reference DFT, the check and describe
//   fbench-<lib> run --cases FILE [options]   plans, checks and times every case
//
// `run --pmu EVENTS` (Linux) also counts hardware events in user mode over the executes of
// each case, after its timed batches, and reports them per call (experiment X25). EVENTS is
// a comma-separated list of `cycles`, `instructions` or `NAME=0xCONFIG` (a raw event in the
// kernel's encoding for the CPU's core PMU, e.g. Zen 3 FP pipe 0 `fp0=0x0100`). The events
// form one group, so they must fit the PMU's counters at once; a group the kernel did not
// keep scheduled the whole time reports its running fraction below 1.
//
// Every result is one JSON line on standard output. The driver (bench/fbench.py) owns the
// statistics; this program measures one process and says what it measured.

#include "backend.hpp"
#include "check.hpp"
#include "fbench_build_info.hpp"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <pthread/qos.h>
#include <sys/qos.h>
#endif

namespace {

using steady = std::chrono::steady_clock;

double seconds_since(steady::time_point start) {
  return std::chrono::duration<double>(steady::now() - start).count();
}

// ---- Buffers ---------------------------------------------------------------------------

struct aligned_free {
  void operator()(void* p) const noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};

// A buffer whose data starts `offset` bytes into a page: the driver randomises the offset
// between rounds so that an aliasing accident between buffers cannot decide a result.
class buffer {
public:
  buffer(std::size_t bytes, std::size_t offset) {
    constexpr std::size_t page = 4096;
    const std::size_t total = (bytes + offset + page) / page * page;
#ifdef _WIN32
    base_.reset(_aligned_malloc(total, page));
#else
    base_.reset(std::aligned_alloc(page, total));
#endif
    if (base_ == nullptr) {
      throw std::bad_alloc();
    }
    data_ = static_cast<std::byte*>(base_.get()) + offset;
  }

  [[nodiscard]] void* data() const { return data_; }

private:
  std::unique_ptr<void, aligned_free> base_;
  std::byte* data_ = nullptr;
};

// ---- Command line ----------------------------------------------------------------------

struct settings {
  std::string cases;
  std::string backend_options;
  bool cold = false;
  std::int64_t pool_bytes = std::int64_t{256} * 1024 * 1024;
  int batches = 7;
  double batch_time = 2e-3;
  double warmup = 0.02;
  std::size_t offset_in = 0;
  std::size_t offset_out = 0;
  std::int64_t check_budget = std::int64_t{4} * 1024 * 1024;
  bool check_only = false;
  double delay = 0.0;
  double perturb = 0.0;
  std::uint64_t seed = 1;
  std::string pmu;
};

template <class T> T parse_number(std::string_view text, std::string_view what) {
  T value{};
  const auto res = std::from_chars(text.data(), text.data() + text.size(), value);
  if (res.ec != std::errc{} || res.ptr != text.data() + text.size()) {
    throw std::invalid_argument("bad " + std::string(what) + ": " + std::string(text));
  }
  return value;
}

settings parse_settings(std::span<char* const> args) {
  settings s;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view key = args[i];
    const auto next = [&]() -> std::string_view {
      if (i + 1 >= args.size()) {
        throw std::invalid_argument("missing value after " + std::string(key));
      }
      return args[++i];
    };
    if (key == "--cases") {
      s.cases = next();
    } else if (key == "--options") {
      s.backend_options = next();
    } else if (key == "--mode") {
      const std::string_view mode = next();
      if (mode != "hot" && mode != "cold") {
        throw std::invalid_argument("--mode is hot or cold");
      }
      s.cold = mode == "cold";
    } else if (key == "--pool-bytes") {
      s.pool_bytes = parse_number<std::int64_t>(next(), key);
    } else if (key == "--batches") {
      s.batches = parse_number<int>(next(), key);
    } else if (key == "--batch-time") {
      s.batch_time = parse_number<double>(next(), key);
    } else if (key == "--warmup") {
      s.warmup = parse_number<double>(next(), key);
    } else if (key == "--offset-in") {
      s.offset_in = parse_number<std::size_t>(next(), key);
    } else if (key == "--offset-out") {
      s.offset_out = parse_number<std::size_t>(next(), key);
    } else if (key == "--check-budget") {
      s.check_budget = parse_number<std::int64_t>(next(), key);
    } else if (key == "--check-only") {
      s.check_only = true;
    } else if (key == "--delay") {
      s.delay = parse_number<double>(next(), key);
    } else if (key == "--perturb") {
      s.perturb = parse_number<double>(next(), key);
    } else if (key == "--seed") {
      s.seed = parse_number<std::uint64_t>(next(), key);
    } else if (key == "--pmu") {
      s.pmu = next();
    } else {
      throw std::invalid_argument("unknown option " + std::string(key));
    }
  }
  if (s.cases.empty()) {
    throw std::invalid_argument("--cases is required");
  }
  if (s.batches < 1 || s.batch_time <= 0.0 || s.delay < 0.0 || s.offset_in % 64 != 0 ||
      s.offset_out % 64 != 0 || s.offset_in >= 4096 || s.offset_out >= 4096) {
    throw std::invalid_argument("batches >= 1, batch time > 0, delay >= 0, and offsets are "
                                "multiples of 64 below 4096");
  }
  return s;
}

// ---- Cases -----------------------------------------------------------------------------

struct case_spec {
  std::string id;
  fbench::layout layout;
};

// One line of `key=value` tokens: id, kind (c2c, r2c, c2r), dims (e.g. 64x48), batch,
// place (in, out), for c2c sign (-1, +1), and prec (f64, the default, or f32).
case_spec parse_case(const std::string& line) {
  case_spec c;
  std::int32_t kind = fbench_c2c;
  int rank = 0;
  std::array<std::int64_t, 3> dims{1, 1, 1};
  std::int64_t batch = 1;
  int sign = -1;
  bool in_place = false;
  std::int32_t precision = fbench_f64;
  std::size_t at = 0;
  while (at < line.size()) {
    const std::size_t end = std::min(line.find(' ', at), line.size());
    const std::string_view token = std::string_view(line).substr(at, end - at);
    at = end + 1;
    if (token.empty()) {
      continue;
    }
    const std::size_t eq = token.find('=');
    if (eq == std::string_view::npos) {
      throw std::invalid_argument("token without '=': " + std::string(token));
    }
    const std::string_view key = token.substr(0, eq);
    const std::string_view val = token.substr(eq + 1);
    if (key == "id") {
      c.id = val;
    } else if (key == "kind") {
      if (val == "c2c") {
        kind = fbench_c2c;
      } else if (val == "r2c") {
        kind = fbench_r2c;
      } else if (val == "c2r") {
        kind = fbench_c2r;
      } else {
        throw std::invalid_argument("unknown kind " + std::string(val));
      }
    } else if (key == "dims") {
      std::size_t from = 0;
      while (from <= val.size()) {
        const std::size_t x = std::min(val.find('x', from), val.size());
        if (rank == 3) {
          throw std::invalid_argument("at most three dimensions");
        }
        dims.at(static_cast<std::size_t>(rank++)) =
            parse_number<std::int64_t>(val.substr(from, x - from), "dims");
        from = x + 1;
      }
    } else if (key == "batch") {
      batch = parse_number<std::int64_t>(val, key);
    } else if (key == "place") {
      if (val != "in" && val != "out") {
        throw std::invalid_argument("place is in or out");
      }
      in_place = val == "in";
    } else if (key == "sign") {
      sign = parse_number<int>(val, key);
    } else if (key == "prec") {
      if (val != "f64" && val != "f32") {
        throw std::invalid_argument("prec is f64 or f32");
      }
      precision = val == "f32" ? fbench_f32 : fbench_f64;
    } else {
      throw std::invalid_argument("unknown case key " + std::string(key));
    }
  }
  if (c.id.empty() || rank == 0) {
    throw std::invalid_argument("a case needs id and dims");
  }
  c.layout = fbench::make_layout(kind, rank, dims, batch, sign, in_place, precision);
  return c;
}

std::uint64_t hash(std::string_view s) {
  std::uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a
  for (const char c : s) {
    h = (h ^ static_cast<unsigned char>(c)) * 0x100000001b3ULL;
  }
  return h;
}

// ---- Hardware counters -----------------------------------------------------------------

// One group of user-mode counters on the calling thread (perf_event_open), enabled only
// around the execute loops. Linux only; elsewhere opening fails with a message.
class counters {
public:
  explicit counters(std::string_view spec) {
#ifdef __linux__
    std::size_t at = 0;
    while (at <= spec.size()) {
      const std::size_t end = std::min(spec.find(',', at), spec.size());
      const std::string_view token = spec.substr(at, end - at);
      at = end + 1;
      if (token.empty()) {
        continue;
      }
      perf_event_attr attr{};
      attr.size = sizeof(attr);
      const std::size_t eq = token.find('=');
      const std::string_view name = token.substr(0, eq);
      if (eq == std::string_view::npos && name == "cycles") {
        attr.type = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_CPU_CYCLES;
      } else if (eq == std::string_view::npos && name == "instructions") {
        attr.type = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_INSTRUCTIONS;
      } else if (eq != std::string_view::npos && token.substr(eq + 1).starts_with("0x")) {
        attr.type = PERF_TYPE_RAW;
        const std::string_view hex = token.substr(eq + 3);
        const auto res = std::from_chars(hex.data(), hex.data() + hex.size(), attr.config, 16);
        if (res.ec != std::errc{} || res.ptr != hex.data() + hex.size()) {
          throw std::invalid_argument("bad --pmu event: " + std::string(token));
        }
      } else {
        throw std::invalid_argument("--pmu takes cycles, instructions or NAME=0xCONFIG, not " +
                                    std::string(token));
      }
      const bool leader = fds_.empty();
      attr.disabled = leader ? 1 : 0;
      attr.exclude_kernel = 1;
      attr.exclude_hv = 1;
      attr.read_format =
          PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
      const long fd = syscall(SYS_perf_event_open, &attr, 0, -1, leader ? -1 : fds_.front(), 0);
      if (fd < 0) {
        const int error = errno;
        close_all();
        throw std::runtime_error("perf_event_open failed for " + std::string(token) + ": " +
                                 std::system_category().message(error));
      }
      fds_.push_back(static_cast<int>(fd));
      names_.emplace_back(name);
    }
    if (fds_.empty()) {
      throw std::invalid_argument("--pmu needs at least one event");
    }
#else
    throw std::invalid_argument("--pmu needs Linux (perf_event_open); not counting " +
                                std::string(spec));
#endif
  }
  counters(const counters&) = delete;
  counters& operator=(const counters&) = delete;
  counters(counters&&) = delete;
  counters& operator=(counters&&) = delete;
  ~counters() { close_all(); }

  void reset() const { control(reset_op); }
  void enable() const { control(enable_op); }
  void disable() const { control(disable_op); }

  // The counts per call over `calls` executes, and the fraction of the enabled time the
  // group was scheduled (1 when the counts are exact).
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static): reads fds_ on Linux
  [[nodiscard]] std::string per_call(std::int64_t calls) const {
    fbench::json out; // NOLINT(misc-const-correctness): filled on Linux
#ifdef __linux__
    std::vector<std::uint64_t> buf(3 + fds_.size());
    const auto bytes = static_cast<std::ptrdiff_t>(buf.size() * sizeof(std::uint64_t));
    if (read(fds_.front(), buf.data(), buf.size() * sizeof(std::uint64_t)) != bytes) {
      return out.str("error", "short read of the counter group").object();
    }
    const double enabled = static_cast<double>(buf[1]);
    out.integer("calls", calls)
        .num("running", enabled > 0 ? static_cast<double>(buf[2]) / enabled : 0.0);
    for (std::size_t i = 0; i < names_.size(); ++i) {
      out.num(names_[i], static_cast<double>(buf[3 + i]) / static_cast<double>(calls));
    }
#else
    static_cast<void>(calls);
#endif
    return out.object();
  }

private:
  enum op : std::uint8_t { reset_op, enable_op, disable_op };
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static): uses fds_ on Linux
  void control(op o) const {
#ifdef __linux__
    unsigned long request = PERF_EVENT_IOC_DISABLE;
    switch (o) {
    case reset_op:
      request = PERF_EVENT_IOC_RESET;
      break;
    case enable_op:
      request = PERF_EVENT_IOC_ENABLE;
      break;
    case disable_op:
      break;
    }
    ioctl(fds_.front(), request, PERF_IOC_FLAG_GROUP);
#else
    static_cast<void>(o);
#endif
  }
  void close_all() {
#ifdef __linux__
    for (auto it = fds_.rbegin(); it != fds_.rend(); ++it) {
      close(*it);
    }
#endif
    fds_.clear();
  }
  std::vector<int> fds_;
  std::vector<std::string> names_;
};

// ---- One case --------------------------------------------------------------------------

class plan_handle {
public:
  plan_handle() = default;
  plan_handle(const plan_handle&) = delete;
  plan_handle& operator=(const plan_handle&) = delete;
  plan_handle(plan_handle&&) = delete;
  plan_handle& operator=(plan_handle&&) = delete;
  ~plan_handle() {
    if (plan_ != nullptr) {
      fbench_destroy(plan_);
    }
  }
  void** slot() { return &plan_; }
  [[nodiscard]] void* get() const { return plan_; }

private:
  void* plan_ = nullptr;
};

// The buffer sets of one case: one in hot mode; in cold mode enough to exceed the pool
// size (larger than the last-level cache), used in rotation.
struct buffer_sets {
  std::vector<buffer> in;
  std::vector<buffer> out;
  bool in_place = false;

  [[nodiscard]] std::size_t size() const { return in.size(); }
  [[nodiscard]] void* input(std::size_t s) const { return in[s].data(); }
  [[nodiscard]] void* output(std::size_t s) const {
    return in_place ? in[s].data() : out[s].data();
  }
};

class timer {
public:
  // pristine: the input in the case's precision, l.buffer_bytes_in bytes.
  timer(const void* plan, const fbench::layout& l, const buffer_sets& sets,
        const std::vector<std::byte>& pristine)
      : plan_(plan), sets_(sets), pristine_(pristine), bytes_(l.buffer_bytes_in) {
    // In-place transforms and c2r (whose input FFTW's default plans destroy) change their
    // own input. Unnormalised transforms grow by about sqrt(n) per call, so the input is
    // restored (outside the timer) before overflow could occur: at most `limit` calls per
    // buffer set between restores.
    restore_ = l.in_place || l.kind == fbench_c2r;
    const double digits = std::log10(std::max(2.0, static_cast<double>(l.n_total)));
    const auto limit = std::max<std::int64_t>(1, static_cast<std::int64_t>(400.0 / digits));
    chunk_ = limit * static_cast<std::int64_t>(sets.size());
  }

  void restore() const {
    for (std::size_t s = 0; s < sets_.size(); ++s) {
      std::memcpy(sets_.input(s), pristine_.data(), bytes_);
    }
  }

  // Runs `calls` executes and returns the time they took, plus `delay` times that time
  // spent spinning (the seeded slowdown of the +DELAY option). With `pmu`, its counters run
  // during the executes only (not the restores).
  double run(std::int64_t calls, double delay, const counters* pmu = nullptr) {
    double total = 0.0;
    std::int64_t done = 0;
    while (done < calls) {
      const std::int64_t n = restore_ ? std::min(calls - done, chunk_) : calls - done;
      if (restore_) {
        restore();
      }
      if (pmu != nullptr) {
        pmu->enable();
      }
      const steady::time_point t0 = steady::now();
      for (std::int64_t i = 0; i < n; ++i) {
        fbench_execute(plan_, sets_.input(cursor_), sets_.output(cursor_));
        cursor_ = cursor_ + 1 == sets_.size() ? 0 : cursor_ + 1;
      }
      steady::time_point t1 = steady::now();
      if (pmu != nullptr) {
        pmu->disable();
      }
      if (delay > 0.0) {
        const steady::time_point until =
            t1 + std::chrono::duration_cast<steady::duration>((t1 - t0) * delay);
        while (t1 < until) {
          t1 = steady::now();
        }
      }
      total += std::chrono::duration<double>(t1 - t0).count();
      done += n;
    }
    return total;
  }

  [[nodiscard]] std::int64_t chunk() const { return restore_ ? chunk_ : 0; }

private:
  const void* plan_;
  const buffer_sets& sets_;
  const std::vector<std::byte>& pristine_;
  std::size_t bytes_;
  bool restore_ = false;
  std::int64_t chunk_ = 1;
  std::size_t cursor_ = 0;
};

// Whether the backend's describe lists "f32" among its "precisions".
bool takes_f32() {
  const std::string text = fbench_describe();
  const std::string key = R"("precisions":[)";
  const std::size_t at = text.find(key);
  if (at == std::string::npos) {
    return false;
  }
  const std::size_t end = text.find(']', at);
  return text.substr(at, end - at).find(R"("f32")") != std::string::npos;
}

std::string run_case(const case_spec& c, const settings& s, const counters* pmu) {
  const fbench::layout& l = c.layout;
  const std::uint64_t seed = hash(c.id) ^ s.seed;
  fbench::json out;
  out.str("type", "case").str("id", c.id);
  if (l.precision == fbench_f32 && !takes_f32()) {
    return out.str("status", "unsupported")
        .str("message", "the backend lists no f32 among its precisions")
        .object();
  }

  buffer_sets sets;
  sets.in_place = l.in_place;
  const std::size_t set_bytes = l.buffer_bytes_in + l.buffer_bytes_out;
  std::size_t count = 1;
  if (s.cold && !s.check_only) {
    const auto pool = static_cast<std::size_t>(s.pool_bytes);
    count = std::clamp<std::size_t>((pool + set_bytes - 1) / set_bytes, 2, 4096);
  }
  sets.in.reserve(count);
  sets.out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    sets.in.emplace_back(l.buffer_bytes_in, s.offset_in);
    if (!l.in_place) {
      sets.out.emplace_back(l.buffer_bytes_out, s.offset_out);
    }
  }

  const fbench_case desc = l.to_case();
  plan_handle plan;
  std::array<char, 8192> message{};
  const steady::time_point plan_start = steady::now();
  const std::int32_t status = fbench_create(&desc, sets.input(0), sets.output(0), plan.slot(),
                                            message.data(), message.size());
  const double plan_s = seconds_since(plan_start);
  if (status != fbench_ok) {
    return out.str("status", status == fbench_unsupported ? "unsupported" : "error")
        .str("message", message.data())
        .object();
  }

  const std::string note = message.data();

  // Correctness: one execute on the deterministic input, against the reference. The
  // reference sees exactly the values the library sees: in single precision, the input
  // rounded to float (rounding keeps the c2r input's Hermitian symmetry).
  const bool single = l.precision == fbench_f32;
  std::vector<double> reference_in(l.reals_in());
  fbench::fill_input(l, reference_in.data(), seed);
  std::vector<std::byte> pristine(l.buffer_bytes_in);
  if (single) {
    std::vector<float> rounded(reference_in.size());
    for (std::size_t i = 0; i < rounded.size(); ++i) {
      rounded[i] = static_cast<float>(reference_in[i]);
      reference_in[i] = static_cast<double>(rounded[i]);
    }
    std::memcpy(pristine.data(), rounded.data(), l.buffer_bytes_in);
  } else {
    std::memcpy(pristine.data(), reference_in.data(), l.buffer_bytes_in);
  }
  std::memcpy(sets.input(0), pristine.data(), l.buffer_bytes_in);
  fbench_execute(plan.get(), sets.input(0), sets.output(0));
  std::vector<double> widened;
  void* result = sets.output(0);
  if (single) {
    widened.resize(l.reals_out());
    const auto* y = static_cast<const float*>(sets.output(0));
    for (std::size_t i = 0; i < widened.size(); ++i) {
      widened[i] = static_cast<double>(y[i]);
    }
    result = widened.data();
  }
  if (s.perturb != 0.0) {
    fbench::perturb_output(l, result, s.perturb, seed);
  }
  const fbench::check_result chk =
      fbench::check_output(l, reference_in.data(), result, s.check_budget, seed);
  out.str("status", "ok").num("plan_s", plan_s);
  if (!note.empty()) {
    out.str("note", note);
  }
  out.raw("check", fbench::json()
                       .boolean("pass", chk.pass)
                       .integer("rows", chk.rows)
                       .integer("samples", chk.samples)
                       .num("err_rms", chk.err_rms)
                       .num("err_max", chk.err_max)
                       .num("parseval", chk.parseval)
                       .num("tol", chk.tol)
                       .object());
  if (s.check_only) {
    return out.object();
  }

  // Timing: calibrate the calls per batch (which also warms up), then timed batches.
  timer t(plan.get(), l, sets, pristine);
  t.restore();
  const steady::time_point warm_start = steady::now();
  std::int64_t calls = 1;
  double elapsed = t.run(calls, 0.0);
  while (elapsed < 0.5 * s.batch_time) {
    calls *= 2;
    elapsed = t.run(calls, 0.0);
  }
  calls = std::max<std::int64_t>(
      1, static_cast<std::int64_t>(std::ceil(static_cast<double>(calls) * s.batch_time / elapsed)));
  while (seconds_since(warm_start) < s.warmup) {
    t.run(calls, 0.0);
  }
  std::vector<double> ns;
  ns.reserve(static_cast<std::size_t>(s.batches));
  for (int b = 0; b < s.batches; ++b) {
    ns.push_back(t.run(calls, s.delay) / static_cast<double>(calls) * 1e9);
  }
  std::vector<double> sorted = ns;
  std::ranges::sort(sorted);
  const std::size_t mid = sorted.size() / 2;
  const double median =
      sorted.size() % 2 == 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
  if (pmu != nullptr) {
    // As many executes as the timed batches, counted after them (warm).
    const std::int64_t counted = calls * s.batches;
    pmu->reset();
    t.run(counted, 0.0, pmu);
    out.raw("pmu", pmu->per_call(counted));
  }
  return out.integer("calls", calls)
      .integer("chunk", t.chunk())
      .integer("sets", static_cast<std::int64_t>(sets.size()))
      .numbers("batches_ns", ns)
      .num("median_ns", median)
      .num("min_ns", sorted.front())
      .object();
}

// ---- Commands --------------------------------------------------------------------------

std::string error_line(std::string_view message) {
  return fbench::json().str("type", "error").str("message", message).object();
}

std::string describe_line() {
#ifdef __APPLE__
  const char* qos = "user-interactive";
#else
  const char* qos = "default";
#endif
  return fbench::json()
      .str("type", "describe")
      .members(fbench_describe())
      .raw("harness", fbench::json()
                          .str("compiler", fbench::build::compiler)
                          .str("build_type", fbench::build::build_type)
                          .str("flags", fbench::build::flags)
                          .str("sanitize", fbench::build::sanitize)
                          .str("commit", fbench::build::commit)
                          .str("qos", qos)
                          .object())
      .object();
}

// The path of this executable, or empty where it is not known.
std::filesystem::path own_executable() {
#ifdef __APPLE__
  std::array<char, 4096> path{};
  auto size = static_cast<std::uint32_t>(path.size());
  if (_NSGetExecutablePath(path.data(), &size) != 0) {
    return {};
  }
  return path.data();
#elif defined(__linux__)
  return "/proc/self/exe";
#else
  return {};
#endif
}

// A backend that names the shared library it loaded must name the file that really
// defines the library, never this executable: on Linux, symbol addresses seen from the
// executable point into the executable itself (PLT entries, copy relocations), which once
// made every FFTW arm look as if it had resolved the wrong library.
std::string check_describe() {
  std::array<char, 512> message{};
  if (fbench_begin("", message.data(), message.size()) != fbench_ok) {
    return std::string("fbench_begin failed: ") + message.data();
  }
  const std::string text = fbench_describe();
  fbench_end();
  const std::string key = R"("loaded_from":")";
  const std::size_t at = text.find(key);
  if (at == std::string::npos) {
    return {};
  }
  const std::size_t from = at + key.size();
  const std::string loaded = text.substr(from, text.find('"', from) - from);
  const std::filesystem::path self = own_executable();
  if (loaded.empty() || self.empty()) {
    return {};
  }
  std::error_code ec;
  const std::filesystem::path library = std::filesystem::canonical(loaded, ec);
  if (ec) {
    return "describe names a library that does not exist: " + loaded;
  }
  const std::filesystem::path exe = std::filesystem::canonical(self, ec);
  if (!ec && library == exe) {
    return "describe says the library was loaded from this executable: " + loaded;
  }
  return {};
}

int command_run(std::span<char* const> args) {
  const settings s = parse_settings(args);
  std::ifstream file(s.cases);
  if (!file) {
    throw std::invalid_argument("cannot read " + s.cases);
  }
  std::vector<std::string> lines;
  for (std::string line; std::getline(file, line);) {
    if (!line.empty() && line.front() != '#') {
      lines.push_back(line);
    }
  }
  std::array<char, 512> message{};
  if (fbench_begin(s.backend_options.c_str(), message.data(), message.size()) != fbench_ok) {
    std::cout << error_line(message.data()) << '\n';
    return 1;
  }
  std::cout << describe_line() << '\n' << std::flush;
  const std::unique_ptr<counters> pmu = s.pmu.empty() ? nullptr : std::make_unique<counters>(s.pmu);
  for (const std::string& line : lines) {
    std::string result;
    try {
      result = run_case(parse_case(line), s, pmu.get());
    } catch (const std::exception& e) {
      result = fbench::json()
                   .str("type", "case")
                   .str("line", line)
                   .str("status", "error")
                   .str("message", e.what())
                   .object();
    }
    std::cout << result << '\n' << std::flush;
  }
  fbench_end();
  std::cout << fbench::json().str("type", "end").object() << '\n';
  return 0;
}

} // namespace

int main(int argc, char** argv) {
#ifdef __APPLE__
  // Ask for the performance cores (DESIGN.md 16.3); macOS has no affinity control.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  const std::span<char* const> args(argv, static_cast<std::size_t>(argc));
  const std::string_view command = args.size() > 1 ? args[1] : "";
  try {
    if (command == "describe") {
      std::array<char, 512> message{};
      if (fbench_begin("", message.data(), message.size()) != fbench_ok) {
        std::cout << error_line(message.data()) << '\n';
        return 1;
      }
      std::cout << describe_line() << '\n';
      fbench_end();
      return 0;
    }
    if (command == "selftest") {
      std::string failure = fbench::selftest();
      if (failure.empty()) {
        failure = check_describe();
      }
      std::cout << (failure.empty() ? "selftest: ok" : "selftest: " + failure) << '\n';
      return failure.empty() ? 0 : 1;
    }
    if (command == "run") {
      return command_run(args.subspan(2));
    }
  } catch (const std::exception& e) {
    std::cerr << "fbench harness: " << e.what() << '\n';
    return 2;
  }
  std::cerr << "usage: " << (args.empty() ? "fbench-<lib>" : args[0])
            << " describe | selftest | run --cases FILE [options]\n";
  return 2;
}
