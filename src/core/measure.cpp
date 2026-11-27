// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Measured planning (DESIGN.md 11.2, task 2.4) and plans from wisdom (11.3): build the
// candidates (core/candidates.hpp), time each whole program on scratch data at the
// transform's size, keep the fastest. Candidates are timed in turn, round after round, so
// that a drift in the machine's speed touches them all alike, with the input and output at
// other offsets in every round (placement_of) and, where the kernel offers transparent huge
// pages, on 4 KiB pages and on huge pages in alternate rounds (page_memory.hpp); the
// statistic is each candidate's median sample on the kind of memory it runs slower on. The budgets
// are first guesses that experiment P2 (dev/experiments/P2-*.md) sets. Pipelines (task 5.2g)
// go through the same timing with their own candidates, the pairs (plan_pipeline_with_effort).

#include "core/measure.hpp"

#include "core/candidates.hpp"
#include "core/drive.hpp"
#include "core/edge_driver.hpp"
#include "core/edges.hpp"
#include "core/pipeline.hpp"
#include "core/plan_core.hpp"
#include "core/wisdom_entry.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/workspace.hpp"
#include "platform/aligned_buffer.hpp"
#include "platform/page_memory.hpp"
#include "platform/timer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <ios>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

std::vector<std::size_t> reduced_counts(std::span<const std::size_t> counts, std::size_t scaled,
                                        std::size_t fixed, std::size_t limit) {
  std::vector<std::size_t> out(counts.begin(), counts.end());
  std::size_t transforms = 1;
  for (const std::size_t c : counts) {
    transforms *= c; // the shape's reach checks have bounded the product
  }
  if (scaled + fixed <= limit || transforms <= 1) {
    return out;
  }
  const std::size_t per = std::max<std::size_t>(scaled / transforms, 1);
  const std::size_t room = limit > fixed ? limit - fixed : 0;
  const std::size_t want = std::max<std::size_t>(room / per, 1);
  std::size_t product = transforms;
  for (std::size_t& c : out) {
    if (product <= want) {
      break;
    }
    const std::size_t inner = product / c;
    c = std::max<std::size_t>(1, std::min(c, want / std::max<std::size_t>(inner, 1)));
    product = inner * c;
  }
  return out;
}

placement placement_of(std::size_t round) noexcept {
  // Two odd steps through the 64 lines of a page; in - out = 16 round - 37 (mod 64) is never
  // 0, so the two differ in every round, by a distance that changes too.
  return {.in = (round * 29 % 64) * 64, .out = ((round * 13 + 37) % 64) * 64};
}

namespace {

// One effort's budget (P2 sets them).
struct budget {
  double seconds;            // the target for the whole planning, building and timing
  std::size_t candidates;    // the most programs timed, the rules' choice included
  std::size_t radix_choices; // radix sequences per line length (enumerate_candidates)
  double sample_seconds;     // a sample's executions take about this long
  std::size_t min_rounds;    // samples per candidate before any is dropped
  std::size_t max_rounds;
};

constexpr budget quick_budget{
    .seconds = 0.1,
    .candidates = 4,
    .radix_choices = 2,
    .sample_seconds = 2e-4,
    .min_rounds = 2,
    .max_rounds = 3,
};
constexpr budget measure_budget{
    .seconds = 10,
    .candidates = 16,
    .radix_choices = 4,
    .sample_seconds = 1e-3,
    .min_rounds = 3,
    .max_rounds = 9,
};

// The rules' choice stays unless another candidate is at least this much faster: timings of
// close programs swap from run to run, and the rules' choice is the reproducible one.
constexpr double keep_rules_margin = 0.01;
// After min_rounds, a candidate this much slower than the fastest gets no more samples.
constexpr double drop_margin = 0.25;
// The default scratch limit (options::measure_scratch_bytes).
constexpr std::size_t default_scratch_bytes = std::size_t{1} << 30U;
// The room each data buffer keeps for placement_of's offsets.
constexpr std::size_t placement_slack = 4096;

double seconds_since(planning_clock::time_point start) {
  return std::chrono::duration<double>(planning_clock::now() - start).count();
}

std::size_t span_bytes(std::ptrdiff_t lo, std::ptrdiff_t hi) {
  return static_cast<std::size_t>(hi - lo);
}

// The data the timings allocate for a plan, for one kind of page memory: input and output
// (one buffer in place, then with a pristine copy of the input), with room for the offsets.
std::size_t data_bytes(const plan_core& c) {
  const std::size_t in = span_bytes(c.in_lo, c.in_hi);
  const std::size_t out = span_bytes(c.out_lo, c.out_hi);
  return c.in_place ? 2 * (std::max(in, out) + placement_slack) : in + out + 2 * placement_slack;
}

std::size_t table_bytes(const plan_core& c) {
  std::size_t bytes = 0;
  for (const auto& t : c.tables_f64) {
    bytes += t.size() * sizeof(double);
  }
  for (const auto& t : c.tables_f32) {
    bytes += t.size() * sizeof(float);
  }
  return bytes;
}

// Below this many bytes in the largest buffer, the page size of the scratch cannot matter
// enough to time both kinds (P2: the case that misled was 512 KiB per buffer).
constexpr std::size_t two_kinds_from = std::size_t{128} * 1024;

// How many kinds of page memory the timings use for a plan with a workspace of `ws` bytes:
// both (huge and 4 KiB) where they can differ and the buffers are large, else one.
std::size_t page_kinds_for(const plan_core& c, std::size_t ws) {
  const std::size_t largest =
      std::max({span_bytes(c.in_lo, c.in_hi), span_bytes(c.out_lo, c.out_hi), ws});
  return page_kinds_differ() && largest >= two_kinds_from ? 2 : 1;
}

// The bytes the timings reserve for a plan: per kind of page memory, its input and output
// (one buffer in place) with room for the offsets and a workspace, then a pristine copy of
// the input in place.
std::size_t scratch_bytes(const plan_core& c, std::size_t ws, std::size_t kinds) {
  const std::size_t in = span_bytes(c.in_place ? std::min(c.in_lo, c.out_lo) : c.in_lo,
                                    c.in_place ? std::max(c.in_hi, c.out_hi) : c.in_hi);
  std::size_t per = page_region::reserved_for(in + placement_slack) + page_region::reserved_for(ws);
  if (!c.in_place) {
    per += page_region::reserved_for(span_bytes(c.out_lo, c.out_hi) + placement_slack);
  }
  return kinds * per + (c.in_place ? in : 0);
}

// Scratch data for timing, on one or two kinds of page memory (page_memory.hpp): per kind an
// input and an output buffer (one in place) filled with the same small values, and a
// workspace. Round r uses kind r % kinds, at placement_of(r)'s offsets.
class scratch_data {
public:
  scratch_data(const plan_core& c, std::size_t ws_bytes, std::size_t kinds)
      : in_place_(c.in_place) {
    const std::ptrdiff_t lo = c.in_place ? std::min(c.in_lo, c.out_lo) : c.in_lo;
    const std::ptrdiff_t hi = c.in_place ? std::max(c.in_hi, c.out_hi) : c.in_hi;
    in_bytes_ = span_bytes(lo, hi);
    in_zero_ = -lo;
    out_zero_ = -c.out_lo;
    for (std::size_t k = 0; k < kinds; ++k) {
      const page_kind kind = k == 0 ? page_kind::small : page_kind::huge;
      memory& m = kinds_.emplace_back();
      m.in = page_region(in_bytes_ + placement_slack, kind);
      if (!c.in_place) {
        m.out = page_region(span_bytes(c.out_lo, c.out_hi) + placement_slack, kind);
      }
      m.ws_memory = page_region(ws_bytes, kind);
      m.ws = workspace(std::span<std::byte>(m.ws_memory.data(), ws_bytes));
    }
    fill(c.prec);
    for (std::size_t k = 1; k < kinds; ++k) {
      std::memcpy(kinds_[k].in.data(), kinds_[0].in.data(), in_bytes_);
    }
    if (c.in_place) {
      pristine_ = aligned_buffer<std::byte>(in_bytes_);
      std::memcpy(pristine_.data(), kinds_[0].in.data(), in_bytes_);
    }
  }

  [[nodiscard]] std::size_t kinds() const { return kinds_.size(); }
  [[nodiscard]] std::size_t kind() const { return current_; }
  [[nodiscard]] const void* in() const {
    const memory& m = kinds_[current_];
    return m.in.data() + m.at.in + in_zero_;
  }
  [[nodiscard]] void* out() {
    const memory& m = kinds_[current_];
    return in_place_ ? m.in.data() + m.at.in + in_zero_ : m.out.data() + m.at.out + out_zero_;
  }
  [[nodiscard]] const workspace& ws() const { return kinds_[current_].ws; }
  // Selects round r's kind of memory and moves its input (and output) to r's offsets.
  void place(std::size_t round) {
    current_ = round % kinds_.size();
    memory& m = kinds_[current_];
    const placement next = placement_of(round);
    if (!in_place_ && next.in != m.at.in) {
      std::memmove(m.in.data() + next.in, m.in.data() + m.at.in, in_bytes_);
    }
    m.at = next;
  }
  // In place, the input is the output: restore it, so that repeated transforms do not grow
  // the values without bound (reps_cap bounds the growth within one sample).
  void restore() {
    if (in_place_) {
      const memory& m = kinds_[current_];
      std::memcpy(m.in.data() + m.at.in, pristine_.data(), in_bytes_);
    }
  }

private:
  struct memory {
    page_region in;
    page_region out;
    page_region ws_memory;
    workspace ws;
    placement at{};
  };

  void fill(precision p) {
    // A fixed linear congruential sequence in [-0.5, 0.5): reproducible, never denormal.
    std::byte* to = kinds_[0].in.data();
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;
    const auto next = [&] {
      x = x * 6364136223846793005ULL + 1442695040888963407ULL;
      return static_cast<double>(x >> 11U) * 0x1.0p-53 - 0.5;
    };
    if (p == precision::f64) {
      for (std::size_t i = 0; i + sizeof(double) <= in_bytes_; i += sizeof(double)) {
        const double v = next();
        std::memcpy(to + i, &v, sizeof v);
      }
      return;
    }
    for (std::size_t i = 0; i + sizeof(float) <= in_bytes_; i += sizeof(float)) {
      const auto v = static_cast<float>(next());
      std::memcpy(to + i, &v, sizeof v);
    }
  }

  bool in_place_;
  std::size_t in_bytes_ = 0;
  std::vector<memory> kinds_;
  std::size_t current_ = 0;
  aligned_buffer<std::byte> pristine_;
  std::ptrdiff_t in_zero_ = 0; // element 0's offset in its buffer, before the placement
  std::ptrdiff_t out_zero_ = 0;
};

// The most executions in one sample of an in-place plan before an unnormalised transform of
// `points` points could overflow from values below 1: each multiplies the largest magnitude
// by at most `points`, and an execution makes `transforms` such transforms one after another (a
// pipeline two: its forward and inverse transforms, with the middle's scratch factor 1/n where
// there is one, and none, a growth of n per transform, where there is not). Out of place the
// input is never written, so no bound.
std::size_t reps_cap(const plan_core& c, std::size_t transforms) {
  if (!c.in_place) {
    return std::numeric_limits<std::size_t>::max();
  }
  std::size_t points = 1;
  for (const std::size_t n : c.lengths) {
    points *= n;
  }
  const double growth =
      static_cast<double>(transforms) * std::max(1.0, std::log2(static_cast<double>(points)));
  const double headroom = c.prec == precision::f64 ? 900.0 : 100.0; // bits of exponent, less margin
  return std::max<std::size_t>(1, static_cast<std::size_t>(headroom / growth));
}

double median(std::vector<double> v) {
  std::ranges::sort(v);
  const std::size_t h = v.size() / 2;
  return v.size() % 2 != 0 ? v[h] : (v[h - 1] + v[h]) / 2;
}

struct candidate {
  plan_choice choice;
  std::shared_ptr<plan_core> core;
  std::size_t tables = 0; // bytes of the core's tables, while it is kept
  std::size_t reps = 1;
  std::array<std::vector<double>, 2> samples{}; // seconds per execution, by kind of memory
  double estimate = std::numeric_limits<double>::infinity();

  [[nodiscard]] bool timed() const { return !samples[0].empty() || !samples[1].empty(); }
  // The worse kind's median: a program is judged by the memory it runs slower on, since a
  // caller's arrays may be either.
  void update() {
    estimate = 0;
    for (const std::vector<double>& v : samples) {
      if (!v.empty()) {
        estimate = std::max(estimate, median(v));
      }
    }
  }
  bool live = true;
};

// One factor array the timings need: its kind and the span [lo, hi] of elements its strides
// reach from the address passed (none: no array), and the value of its real parts.
struct factor_need {
  factor_kind kind = factor_kind::none;
  std::ptrdiff_t lo = 0;
  std::ptrdiff_t hi = 0;
  double value = 0.5;
};

// Scratch factor arrays for timing plans with edges (fftn/edges.hpp) and pipelines: each array
// over its span on the timing shape, filled with `value` (real parts) and half of it (imaginary
// parts), so that accumulation over many repetitions grows the output by a bounded amount per
// execution.
class factor_scratch {
public:
  factor_scratch(precision p, const factor_need& in, const factor_need& out,
                 const factor_need& mid) {
    args_.in_factor = fill(p, in, in_);
    args_.out_factor = fill(p, out, out_);
    args_.mid_factor = fill(p, mid, mid_);
  }
  factor_scratch(const shape_view& s, const edges_view& e)
      : factor_scratch(s.prec, need(s, e.in, false), need(s, e.out, true), {}) {}
  [[nodiscard]] const exec_args& args() const { return args_; }

private:
  static factor_need need(const shape_view& s, const edge_view& e, bool output) {
    const auto [lo, hi] = factor_span(s, e, output);
    return {.kind = e.kind, .lo = lo, .hi = hi};
  }

  static const void* fill(precision p, const factor_need& f, aligned_buffer<std::byte>& buffer) {
    if (f.kind == factor_kind::none) {
      return nullptr;
    }
    const std::size_t real = p == precision::f64 ? sizeof(double) : sizeof(float);
    const std::size_t width = f.kind == factor_kind::complex ? 2 : 1;
    const auto elements = static_cast<std::size_t>(f.hi - f.lo + 1);
    buffer = aligned_buffer<std::byte>(elements * width * real);
    for (std::size_t i = 0; i < elements * width; ++i) {
      const double v = i % width == 0 ? f.value : f.value / 2;
      if (p == precision::f64) {
        std::memcpy(buffer.data() + i * real, &v, sizeof v);
      } else {
        const auto x = static_cast<float>(v);
        std::memcpy(buffer.data() + i * real, &x, sizeof x);
      }
    }
    return buffer.data() + static_cast<std::size_t>(-f.lo) * width * real;
  }

  aligned_buffer<std::byte> in_;
  aligned_buffer<std::byte> out_;
  aligned_buffer<std::byte> mid_;
  exec_args args_;
};

// Times `reps` executions; seconds per execution. `f`: the scratch factor arrays of a plan with
// edges, or null. A plan's callbacks run as stand-ins that do nothing (callback_env::stand_in;
// DESIGN.md 12.3, "Planning"): planning never calls the application's code.
double time_once(const plan_core& c, scratch_data& d, std::size_t reps, const factor_scratch* f) {
  d.restore();
  const auto start = planning_clock::now();
  if (f != nullptr && c.needs_args) {
    edge_state state = edge_state_of(f->args(), c.factor_args);
    state.stand_in = true;
    for (std::size_t i = 0; i < reps; ++i) {
      execute_core_cursors(c, 0, c.prog.transforms, d.in(), d.out(), d.ws(), state);
    }
    return seconds_since(start) / static_cast<double>(reps);
  }
  if (f != nullptr) {
    for (std::size_t i = 0; i < reps; ++i) {
      execute_core(c, 0, c.prog.transforms, d.in(), d.out(), d.ws(), f->args());
    }
    return seconds_since(start) / static_cast<double>(reps);
  }
  for (std::size_t i = 0; i < reps; ++i) {
    execute_core(c, 0, c.prog.transforms, d.in(), d.out(), d.ws());
  }
  return seconds_since(start) / static_cast<double>(reps);
}

// Builds a candidate's plan for a shape and settings: build_plan_core, or build_with_edges for
// a plan with edges (without name_forced_strategy, which only the plan kept needs).
using plan_builder =
    std::function<std::shared_ptr<plan_core>(const shape_view&, const build_settings&)>;

// What measure() times, apart from the budget: the candidates' choices in order (the rules'
// first), a candidate built on the timing shape (`build_timing`, which may throw for a choice
// the shape cannot take) and on the whole one (`build_full`, used when the timing shape is
// reduced), the scratch factor arrays (`factors`; empty: none) and what the plan kept needs
// once built (`finish`; empty: nothing).
struct measure_job {
  std::vector<plan_choice> choices;
  bool same_shape = true;
  std::function<std::shared_ptr<plan_core>(const build_settings&)> build_timing;
  std::function<std::shared_ptr<plan_core>(const build_settings&)> build_full;
  std::function<std::unique_ptr<factor_scratch>()> factors;
  std::function<void(plan_core&, const build_settings&)> finish;
  std::size_t transforms = 1; // unnormalised transforms per execution (reps_cap)
};

// The batch counts of the timing shape for the rules' plan (reduced_counts): its data, workspace
// and tables within the scratch limit.
std::vector<std::size_t> timing_counts(const plan_core& rules, std::size_t scratch_limit) {
  const std::size_t rules_tables = table_bytes(rules);
  const std::size_t rules_ws = workspace_bytes(rules, 1);
  const std::size_t rules_kinds = page_kinds_for(rules, rules_ws);
  return reduced_counts(rules.counts, rules_kinds * data_bytes(rules),
                        rules_kinds * rules_ws + rules_tables, scratch_limit);
}

std::string fixed(double v, int digits) {
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(digits);
  o << v;
  return o.str();
}

std::string transforms_text(std::size_t n) {
  return std::to_string(n) + (n == 1 ? " transform" : " transforms");
}

struct measure_limits {
  double seconds;
  std::size_t scratch;
};

measure_limits limits_of(const options& opts, const budget& b) {
  return {
      .seconds = opts.measure_seconds > 0 ? opts.measure_seconds : b.seconds,
      .scratch =
          opts.measure_scratch_bytes != 0 ? opts.measure_scratch_bytes : default_scratch_bytes,
  };
}

std::shared_ptr<const plan_core> measure(const measure_job& job, const build_settings& settings,
                                         std::shared_ptr<plan_core> rules, const budget& b,
                                         const measure_limits& limits,
                                         const std::string& wisdom_note,
                                         planning_clock::time_point start) {
  const std::vector<plan_choice>& choices = job.choices;
  const bool same_shape = job.same_shape;
  const bool forced = settings.strategy != strategy_choice::rules || !settings.radices.empty() ||
                      !settings.grid_factors.empty() || settings.grid_phases != 0 ||
                      settings.rader >= 0;
  const bool real_clock = settings.measure_time == nullptr;

  // Candidates are built in the model's order and admitted while half the time budget is
  // left and their tables, the workspace and the data fit the scratch limit (the rules'
  // choice always). Building all of them before timing spent the whole budget at 2^22
  // points, where one plan takes 0.5-1.7 s to build (P1's first pass on the Xeon).
  std::vector<candidate> cands;
  std::vector<std::uint64_t> seen;
  std::size_t tables = 0;
  std::size_t ws_bytes = 0;
  for (const plan_choice& choice : choices) {
    if (cands.size() == b.candidates ||
        (!cands.empty() && seconds_since(start) > limits.seconds / 2)) {
      break;
    }
    build_settings cs = settings;
    apply_choice(choice, cs);
    std::shared_ptr<plan_core> core;
    try {
      core = same_shape && !forced && cands.empty() ? rules : job.build_timing(cs);
    } catch (const std::exception&) {
      continue; // a candidate the shape cannot take; the rules' plan always builds
    }
    const std::uint64_t f = program_fingerprint(*core);
    if (std::ranges::find(seen, f) != seen.end()) {
      continue;
    }
    seen.push_back(f);
    const std::size_t t = table_bytes(*core);
    const std::size_t w = std::max(ws_bytes, workspace_bytes(*core, 1));
    if (!cands.empty() &&
        scratch_bytes(*core, w, page_kinds_for(*core, w)) + tables + t > limits.scratch) {
      break;
    }
    ws_bytes = w;
    tables += t;
    cands.push_back({.choice = choice, .core = std::move(core), .tables = t});
  }
  if (rules != cands.front().core) {
    rules.reset(); // not a candidate (another shape or a test's forced settings): free it
  }

  std::size_t kinds_timed = 0; // kinds of page memory the timings used
  if (cands.size() == 1) {
    // Nothing to compare: no scratch data, no timing.
    cands.front().estimate = 1;
  } else {
    // The real clock times executions on scratch data; a test's stand-in clock
    // (build_settings::measure_time) gives each sample instead, through the same rounds.
    std::unique_ptr<scratch_data> scratch;
    std::unique_ptr<factor_scratch> factors;
    if (real_clock) {
      const plan_core& first = *cands.front().core;
      scratch = std::make_unique<scratch_data>(first, ws_bytes, page_kinds_for(first, ws_bytes));
      kinds_timed = scratch->kinds();
      if (job.factors) {
        factors = job.factors();
      }
    }
    const auto sample = [&](const candidate& c) {
      return real_clock ? time_once(*c.core, *scratch, c.reps, factors.get())
                        : settings.measure_time(choice_text(c.choice), settings.measure_context);
    };
    // A warm-up execution sets each candidate's repetitions per sample; a candidate reached
    // after the budget is spent is not timed (the rules' choice always is).
    for (candidate& c : cands) {
      if (&c != &cands.front() && seconds_since(start) > limits.seconds) {
        c.live = false;
        continue;
      }
      if (real_clock) {
        const double once = time_once(*c.core, *scratch, 1, factors.get());
        const double want = b.sample_seconds / std::max(once, 1e-9);
        c.reps = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::ceil(want)), 1,
            std::min<std::size_t>(reps_cap(*c.core, job.transforms), 1U << 20U));
      }
    }
    for (std::size_t round = 0; round < b.max_rounds; ++round) {
      if (real_clock) {
        scratch->place(round);
      }
      bool any = false;
      // Each round starts one candidate later, so that no candidate always takes the first
      // place after a round's checks (P2's first pass on the Mac: the rules' choice, always
      // timed first, read 1.2-1.3 times its time in single rounds).
      for (std::size_t k = 0; k < cands.size(); ++k) {
        candidate& c = cands[(k + round) % cands.size()];
        if (!c.live || (c.timed() && seconds_since(start) > limits.seconds)) {
          continue;
        }
        c.samples.at(real_clock ? scratch->kind() : 0).push_back(sample(c));
        c.update();
        any = true;
      }
      if (round + 1 >= b.min_rounds) {
        double fastest = std::numeric_limits<double>::infinity();
        for (const candidate& c : cands) {
          if (c.live) {
            fastest = std::min(fastest, c.estimate);
          }
        }
        for (std::size_t i = 0; i < cands.size(); ++i) {
          candidate& c = cands[i];
          c.live = c.live && c.estimate <= fastest * (1 + drop_margin);
          if (!c.live && i != 0 && c.core) {
            c.core.reset(); // a dropped candidate no longer competes: free its tables now
          }
        }
      }
      if (!any || seconds_since(start) > limits.seconds) {
        break;
      }
    }
  }

  // The fastest still held, the earlier on a tie; the rules' choice unless clearly beaten.
  // Only live candidates compete: a dropped one's median is frozen while the others' may
  // still rise, and its plan has been freed.
  std::size_t win = 0;
  for (std::size_t i = 1; i < cands.size(); ++i) {
    if (cands[i].live && cands[i].core && cands[i].timed() &&
        cands[i].estimate < cands[win].estimate) {
      win = i;
    }
  }
  if (cands.front().estimate <= cands[win].estimate * (1 + keep_rules_margin)) {
    win = 0;
  }
  std::size_t timed = 0;
  for (const candidate& c : cands) {
    if (c.timed()) {
      ++timed;
    }
  }
  const candidate& w = cands[win];
  std::shared_ptr<plan_core> final_core = w.core;
  const std::size_t timing_transforms =
      cands.front().core ? cands.front().core->prog.transforms : final_core->prog.transforms;
  build_settings cs = settings;
  apply_choice(w.choice, cs);
  if (!same_shape) {
    final_core = job.build_full(cs);
  }
  // For a measured plan, the kept candidate's settings stand for the rules.
  if (job.finish) {
    job.finish(*final_core, cs);
  }
  final_core->program_text = choice_text(pinned_choice(w.choice, *final_core));
  final_core->planning_seconds = seconds_since(start);
  final_core->candidates_timed = timed;
  final_core->rules_over_chosen = cands.front().estimate / w.estimate;
  final_core->chosen_rank = win;
  final_core->planning_note =
      wisdom_note + "measured (" + std::to_string(timed) + " of " + std::to_string(cands.size()) +
      " distinct candidates timed" +
      (same_shape ? "" : " on " + transforms_text(timing_transforms)) +
      (kinds_timed == 2 ? ", on 4 KiB and huge pages" : "") + ", " +
      fixed(final_core->planning_seconds, 3) + " s): " + final_core->program_text +
      (win == 0 ? ""
                : "; the rules' choice took " + fixed(final_core->rules_over_chosen, 3) +
                      " times as long");
  return final_core;
}

} // namespace

std::shared_ptr<const plan_core> plan_with_effort(const shape_view& s, const options& opts,
                                                  const build_settings& settings,
                                                  const edges_view* edges) {
  options base = opts;
  base.use = nullptr;
  base.effort = planning::rules;
  const auto start = planning_clock::now(); // the budget covers the rules' plan too
  const plan_builder build = [&](const shape_view& shape, const build_settings& cs) {
    return edges != nullptr ? build_with_edges(shape, base, *edges, cs, false)
                            : build_plan_core(shape, base, cs);
  };

  std::string wisdom_note;
  if (opts.use != nullptr) {
    // The entry is found from the shape, the options, the instruction set, the processor and
    // the build, without building the rules' plan; its program is then built and checked.
    const std::string place =
        wisdom_place(isa_name(static_cast<isa>(static_cast<unsigned char>(plan_target_of(base)))),
                     edges != nullptr ? shape_key_with_edges(s, base, *edges) : shape_key(s, base));
    if (const std::optional<wisdom_record> e = find_entry(*opts.use, place)) {
      try {
        if (const std::optional<plan_choice> choice = parse_choice(e->program)) {
          build_settings cs = settings;
          apply_choice(*choice, cs);
          std::shared_ptr<plan_core> core = build(s, cs);
          // The same program, and the entry names it as this build would.
          if (program_fingerprint(*core) == e->check && core->program_text == e->program) {
            if (edges != nullptr) {
              name_forced_strategy(*core, s, base, *edges, cs);
            }
            core->program_text = e->program;
            core->planning_note = "from wisdom: " + e->program;
            return core;
          }
        }
      } catch (const std::exception&) { // NOLINT(bugprone-empty-catch): noted just below
        // A program this build cannot make (a forged or foreign entry): stale, ignored. A
        // shape that is not valid is refused below, by the rules' plan.
      }
      wisdom_note = "a wisdom entry for this shape does not rebuild its program here (ignored); ";
    }
  }
  std::shared_ptr<plan_core> rules = build(s, settings);
  if (opts.effort == planning::rules) {
    if (edges != nullptr) {
      name_forced_strategy(*rules, s, base, *edges, settings);
    }
    rules->planning_note = wisdom_note.empty() ? "" : wisdom_note + "the rules' choice";
    return rules;
  }
  const budget& b = opts.effort == planning::quick ? quick_budget : measure_budget;
  const measure_limits limits = limits_of(opts, b);
  // The candidates are timed on fewer transforms when the data, the workspace and the rules'
  // tables would not fit the scratch limit.
  const std::vector<std::size_t> counts = timing_counts(*rules, limits.scratch);
  std::vector<batch> reduced(s.batches.begin(), s.batches.end());
  for (std::size_t i = 0; i < reduced.size(); ++i) {
    reduced[i].count = counts[i];
  }
  shape_view timing = s;
  timing.batches = reduced;
  measure_job job{
      .choices = enumerate_candidates(*rules, settings, b.radix_choices),
      .same_shape = counts == rules->counts,
      .build_timing = [&](const build_settings& cs) { return build(timing, cs); },
      .build_full = [&](const build_settings& cs) { return build(s, cs); },
      .factors = {},
      .finish = {},
      .transforms = 1,
  };
  if (edges != nullptr) {
    job.factors = [&] { return std::make_unique<factor_scratch>(timing, *edges); };
    job.finish = [&](plan_core& c, const build_settings& cs) {
      name_forced_strategy(c, s, base, *edges, cs);
    };
  }
  return measure(job, settings, std::move(rules), b, limits, wisdom_note, start);
}

namespace {

// The factor array of one of a pipeline's edges on its timing description: the span its strides
// reach over the side's `length` points along the line, the batch dimensions and, for the
// middle (`middle`), the fan-out copies. The middle's values are scaled by 1/n, so that an
// execution in place (two unnormalised transforms) grows the line no more than a plan's one
// transform does (reps_cap).
factor_need pipeline_need(const pipeline_view& d, const edge_view& e, std::size_t length,
                          bool middle) {
  if (e.kind == factor_kind::none) {
    return {};
  }
  std::ptrdiff_t lo = 0;
  std::ptrdiff_t hi = 0;
  const auto reach = [&](std::ptrdiff_t stride, std::size_t extent) {
    if (extent > 1) {
      const std::ptrdiff_t last = stride * static_cast<std::ptrdiff_t>(extent - 1);
      lo += std::min<std::ptrdiff_t>(last, 0);
      hi += std::max<std::ptrdiff_t>(last, 0);
    }
  };
  reach(e.axis_strides.empty() ? 0 : e.axis_strides[0], length);
  for (std::size_t b = 0; b < d.batches.size() && b < e.batch_strides.size(); ++b) {
    reach(e.batch_strides[b], d.batches[b].count);
  }
  if (middle) {
    reach(d.fanout_factor_dist, d.fanout);
  }
  return {.kind = e.kind,
          .lo = lo,
          .hi = hi,
          .value = middle ? 0.5 / static_cast<double>(std::max<std::size_t>(d.line.n, 1)) : 0.5};
}

// The candidates of a pipeline (task 5.2g, X67): the rules' pair, then the chain, depth-first
// and lanes pairs, each forced (a pair the description cannot take is not a candidate).
std::vector<plan_choice> pipeline_choices() {
  return {plan_choice{}, plan_choice{.strategy = strategy_choice::chain},
          plan_choice{.strategy = strategy_choice::depth},
          plan_choice{.strategy = strategy_choice::lanes}};
}

} // namespace

std::shared_ptr<const plan_core> plan_pipeline_with_effort(const pipeline_view& d,
                                                           const options& opts,
                                                           const build_settings& settings,
                                                           const pipeline_builder& build) {
  options base = opts;
  base.use = nullptr;
  base.effort = planning::rules;
  const auto start = planning_clock::now();
  std::string wisdom_note;
  if (opts.use != nullptr) {
    // As a plan's entry (plan_with_effort): found without building the rules' pipeline, then
    // its program built and checked.
    const std::string place =
        wisdom_place(isa_name(static_cast<isa>(static_cast<unsigned char>(plan_target_of(base)))),
                     pipeline_key(d, base));
    if (const std::optional<wisdom_record> e = find_entry(*opts.use, place)) {
      try {
        if (const std::optional<plan_choice> choice = parse_choice(e->program)) {
          build_settings cs = settings;
          apply_choice(*choice, cs);
          std::shared_ptr<plan_core> core = build(d, base, cs);
          if (program_fingerprint(*core) == e->check && core->program_text == e->program) {
            core->planning_note = "from wisdom: " + e->program;
            return core;
          }
        }
      } catch (const std::exception&) { // NOLINT(bugprone-empty-catch): noted just below
        // A program this build cannot make: stale, ignored, as for plans.
      }
      wisdom_note =
          "a wisdom entry for this pipeline does not rebuild its program here (ignored); ";
    }
  }
  std::shared_ptr<plan_core> rules = build(d, base, settings);
  if (opts.effort == planning::rules) {
    rules->planning_note = wisdom_note.empty() ? "" : wisdom_note + "the rules' choice";
    return rules;
  }
  const budget& b = opts.effort == planning::quick ? quick_budget : measure_budget;
  const measure_limits limits = limits_of(opts, b);
  const std::vector<std::size_t> counts = timing_counts(*rules, limits.scratch);
  std::vector<batch> reduced(d.batches.begin(), d.batches.end());
  for (std::size_t i = 0; i < reduced.size(); ++i) {
    reduced[i].count = counts[i];
  }
  pipeline_view timing = d;
  timing.batches = reduced;
  // A forced pair the description cannot take builds another pair: not a candidate of its own.
  const auto build_pair = [&](const pipeline_view& desc, const build_settings& cs) {
    std::shared_ptr<plan_core> core = build(desc, base, cs);
    if (cs.strategy != strategy_choice::rules && core->built != cs.strategy) {
      throw std::invalid_argument("fftn: internal: the pair is not built for this pipeline");
    }
    return core;
  };
  // The sides' lengths along the line: n, or the half spectrum's n/2 + 1 bins.
  const std::size_t n = d.line.n;
  const std::size_t half = n / 2 + 1;
  const bool r2c_first = d.kind == pipeline_kind::real_convolution;
  const bool c2r_first = d.kind == pipeline_kind::real_middle;
  const measure_job job{
      .choices = pipeline_choices(),
      .same_shape = counts == rules->counts,
      .build_timing = [&](const build_settings& cs) { return build_pair(timing, cs); },
      .build_full = [&](const build_settings& cs) { return build(d, base, cs); },
      .factors =
          [&] {
            return std::make_unique<factor_scratch>(
                d.prec, pipeline_need(timing, d.prologue, c2r_first ? half : n, false),
                pipeline_need(timing, d.epilogue, c2r_first ? half : n, false),
                pipeline_need(timing, d.middle, r2c_first ? half : n, true));
          },
      .finish = {},
      .transforms = 2,
  };
  return measure(job, settings, std::move(rules), b, limits, wisdom_note, start);
}

} // namespace fftn::inline v0::detail
