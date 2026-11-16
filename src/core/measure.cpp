// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Measured planning (DESIGN.md 11.2, task 2.4) and plans from wisdom (11.3): build the
// candidates (core/candidates.hpp), time each whole program on scratch data at the
// transform's size, keep the fastest. Candidates are timed in turn, round after round, so
// that a drift in the machine's speed touches them all alike; the statistic is each
// candidate's fastest sample, the one least disturbed by other work on a shared machine.
// The budgets below are first guesses that experiment P2 (dev/experiments/P2-*.md) sets.

#include "core/candidates.hpp"
#include "core/plan_core.hpp"
#include "core/wisdom_entry.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/workspace.hpp"
#include "platform/aligned_buffer.hpp"
#include "platform/timer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

// One effort's budget (P2 sets them).
struct budget {
  double seconds;            // the whole planning, building and timing
  std::size_t candidates;    // the most programs timed, the rules' choice included
  std::size_t radix_choices; // radix sequences per line length (enumerate_candidates)
  double sample_seconds;     // a sample's executions take about this long
  std::size_t min_rounds;    // samples per candidate before any is dropped
  std::size_t max_rounds;
};

constexpr budget quick_budget{.seconds = 0.1,
                              .candidates = 4,
                              .radix_choices = 2,
                              .sample_seconds = 2e-4,
                              .min_rounds = 2,
                              .max_rounds = 3};
constexpr budget measure_budget{.seconds = 10,
                                .candidates = 16,
                                .radix_choices = 4,
                                .sample_seconds = 1e-3,
                                .min_rounds = 3,
                                .max_rounds = 9};

// The rules' choice stays unless another candidate is at least this much faster: timings of
// close programs swap from run to run, and the rules' choice is the reproducible one.
constexpr double keep_rules_margin = 0.01;
// After min_rounds, a candidate this much slower than the fastest gets no more samples.
constexpr double drop_margin = 0.25;
// The default scratch limit (options::measure_scratch_bytes).
constexpr std::size_t default_scratch_bytes = std::size_t{1} << 30U;

[[noreturn]] void reject(const std::string& argument, const std::string& why) {
  throw std::invalid_argument("fftn::plan: " + argument + " " + why);
}

double seconds_since(planning_clock::time_point start) {
  return std::chrono::duration<double>(planning_clock::now() - start).count();
}

std::size_t data_bytes(const plan_core& c) {
  const auto in = static_cast<std::size_t>(c.in_hi - c.in_lo);
  const auto out = static_cast<std::size_t>(c.out_hi - c.out_lo);
  return c.in_place ? std::max(in, out) : in + out;
}

// The batch counts of a shape whose data and workspace fit `limit` bytes, at least one
// transform: the outer dimensions shrink first.
std::vector<batch> reduced_batches(const shape_view& s, const plan_core& full, std::size_t limit) {
  std::vector<batch> b(s.batches.begin(), s.batches.end());
  const std::size_t transforms = std::max<std::size_t>(full.prog.transforms, 1);
  const std::size_t need = data_bytes(full) + full.bytes;
  if (need <= limit || transforms == 1) {
    return b;
  }
  const std::size_t per = std::max<std::size_t>(need / transforms, 1);
  const std::size_t want = std::max<std::size_t>(limit / per, 1);
  std::size_t product = transforms;
  for (batch& d : b) {
    if (product <= want) {
      break;
    }
    const std::size_t inner = product / d.count;
    d.count = std::max<std::size_t>(1, std::min(d.count, want / std::max<std::size_t>(inner, 1)));
    product = inner * d.count;
  }
  return b;
}

// Scratch data for timing: one buffer per side (one in place), filled with small values.
class scratch_data {
public:
  scratch_data(const plan_core& c, std::size_t workspace_bytes) : ws_(workspace_bytes) {
    const std::ptrdiff_t lo = c.in_place ? std::min(c.in_lo, c.out_lo) : c.in_lo;
    const std::ptrdiff_t hi = c.in_place ? std::max(c.in_hi, c.out_hi) : c.in_hi;
    in_ = aligned_buffer<double>(words(hi - lo));
    in_at_ = -lo;
    if (!c.in_place) {
      out_ = aligned_buffer<double>(words(c.out_hi - c.out_lo));
      out_at_ = -c.out_lo;
    }
    fill(c.prec);
    if (c.in_place) {
      pristine_ = aligned_buffer<double>(in_.size());
      std::memcpy(pristine_.data(), in_.data(), in_.size() * sizeof(double));
    }
  }

  [[nodiscard]] const void* in() const { return as_bytes(in_) + in_at_; }
  [[nodiscard]] void* out() { return out_.size() != 0 ? as_bytes(out_) + out_at_ : in_bytes(); }
  [[nodiscard]] const workspace& ws() const { return ws_; }
  // In place, the input is the output: restore it, so that repeated transforms do not grow
  // the values without bound (reps_cap bounds the growth within one sample).
  void restore() {
    if (pristine_.size() != 0) {
      std::memcpy(in_.data(), pristine_.data(), in_.size() * sizeof(double));
    }
  }

private:
  static std::size_t words(std::ptrdiff_t bytes) {
    return (static_cast<std::size_t>(bytes) + sizeof(double) - 1) / sizeof(double);
  }
  static std::byte* as_bytes(aligned_buffer<double>& b) {
    return reinterpret_cast<std::byte*>(b.data());
  }
  static const std::byte* as_bytes(const aligned_buffer<double>& b) {
    return reinterpret_cast<const std::byte*>(b.data());
  }
  std::byte* in_bytes() { return as_bytes(in_) + in_at_; }
  void fill(precision p) {
    // A fixed linear congruential sequence in [-0.5, 0.5): reproducible, never denormal.
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;
    const auto next = [&] {
      x = x * 6364136223846793005ULL + 1442695040888963407ULL;
      return static_cast<double>(x >> 11U) * 0x1.0p-53 - 0.5;
    };
    if (p == precision::f64) {
      for (std::size_t i = 0; i < in_.size(); ++i) {
        in_.data()[i] = next();
      }
      return;
    }
    auto* f = reinterpret_cast<float*>(in_.data());
    for (std::size_t i = 0; i < 2 * in_.size(); ++i) {
      f[i] = static_cast<float>(next());
    }
  }

  aligned_buffer<double> in_;
  aligned_buffer<double> out_;
  aligned_buffer<double> pristine_;
  std::ptrdiff_t in_at_ = 0;
  std::ptrdiff_t out_at_ = 0;
  workspace ws_;
};

// The most executions in one sample of an in-place plan before an unnormalised transform of
// `points` points could overflow from values below 1: each multiplies the largest magnitude
// by at most `points`. Out of place the input is never written, so no bound.
std::size_t reps_cap(const plan_core& c) {
  if (!c.in_place) {
    return std::numeric_limits<std::size_t>::max();
  }
  std::size_t points = 1;
  for (const std::size_t n : c.lengths) {
    points *= n;
  }
  const double growth = std::max(1.0, std::log2(static_cast<double>(points)));
  const double headroom = c.prec == precision::f64 ? 900.0 : 100.0; // bits of exponent, less margin
  return std::max<std::size_t>(1, static_cast<std::size_t>(headroom / growth));
}

struct candidate {
  plan_choice choice;
  std::shared_ptr<plan_core> core;
  std::size_t reps = 1;
  double best = std::numeric_limits<double>::infinity(); // seconds per execution
  std::size_t samples = 0;
  bool live = true;
};

// Times `reps` executions; seconds per execution.
double time_once(const plan_core& c, scratch_data& d, std::size_t reps) {
  d.restore();
  const auto start = planning_clock::now();
  for (std::size_t i = 0; i < reps; ++i) {
    execute_core(c, 0, c.prog.transforms, d.in(), d.out(), d.ws());
  }
  return seconds_since(start) / static_cast<double>(reps);
}

std::string fixed(double v, int digits) {
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(digits);
  o << v;
  return o.str();
}

std::shared_ptr<const plan_core> measure(const shape_view& s, const options& base,
                                         const build_settings& settings,
                                         std::shared_ptr<plan_core> rules, const budget& b,
                                         double seconds, std::size_t scratch_limit,
                                         const std::string& wisdom_note) {
  const auto start = planning_clock::now();
  const std::vector<plan_choice> choices = enumerate_candidates(*rules, settings, b.radix_choices);

  // The candidates are timed on fewer transforms when the data would not fit the limit.
  const std::vector<batch> reduced = reduced_batches(s, *rules, scratch_limit);
  bool same_shape = true;
  for (std::size_t i = 0; i < reduced.size(); ++i) {
    same_shape = same_shape && reduced[i].count == s.batches[i].count;
  }
  shape_view timing = s;
  timing.batches = reduced;

  std::vector<candidate> cands;
  std::vector<std::uint64_t> seen;
  for (const plan_choice& choice : choices) {
    if (cands.size() == b.candidates || (!cands.empty() && seconds_since(start) > seconds)) {
      break;
    }
    build_settings cs = settings;
    apply_choice(choice, cs);
    std::shared_ptr<plan_core> core;
    try {
      core = same_shape && cands.empty() ? rules : build_plan_core(timing, base, cs);
    } catch (const std::exception&) {
      continue; // a candidate the shape cannot take; the rules' plan always builds
    }
    const std::uint64_t f = program_fingerprint(*core);
    if (std::ranges::find(seen, f) != seen.end()) {
      continue;
    }
    seen.push_back(f);
    cands.push_back({.choice = choice, .core = std::move(core)});
  }

  if (settings.measure_time == nullptr) {
    std::size_t ws_bytes = 0;
    for (const candidate& c : cands) {
      ws_bytes = std::max(ws_bytes, workspace_bytes(*c.core, 1));
    }
    scratch_data data(*cands.front().core, ws_bytes);
    // Round 0: a warm-up execution sets each candidate's repetitions per sample.
    for (candidate& c : cands) {
      if (&c != &cands.front() && seconds_since(start) > seconds) {
        c.live = false;
        continue;
      }
      const double once = time_once(*c.core, data, 1);
      const double want = b.sample_seconds / std::max(once, 1e-9);
      c.reps = std::clamp<std::size_t>(static_cast<std::size_t>(std::ceil(want)), 1,
                                       std::min<std::size_t>(reps_cap(*c.core), 1U << 20U));
    }
    for (std::size_t round = 0; round < b.max_rounds; ++round) {
      bool any = false;
      for (candidate& c : cands) {
        if (!c.live || (c.samples > 0 && seconds_since(start) > seconds)) {
          continue;
        }
        c.best = std::min(c.best, time_once(*c.core, data, c.reps));
        ++c.samples;
        any = true;
      }
      if (round + 1 >= b.min_rounds) {
        double fastest = std::numeric_limits<double>::infinity();
        for (const candidate& c : cands) {
          fastest = std::min(fastest, c.best);
        }
        for (candidate& c : cands) {
          c.live = c.live && c.best <= fastest * (1 + drop_margin);
        }
      }
      if (!any || seconds_since(start) > seconds) {
        break;
      }
    }
  } else {
    for (candidate& c : cands) {
      c.best = settings.measure_time(choice_text(c.choice), settings.measure_context);
      c.samples = 1;
    }
  }

  // The fastest, the earlier on a tie; the rules' choice unless clearly beaten.
  std::size_t win = 0;
  for (std::size_t i = 1; i < cands.size(); ++i) {
    if (cands[i].samples != 0 && cands[i].best < cands[win].best) {
      win = i;
    }
  }
  if (cands.front().best <= cands[win].best * (1 + keep_rules_margin)) {
    win = 0;
  }
  std::size_t timed = 0;
  for (const candidate& c : cands) {
    timed += c.samples != 0 ? 1 : 0;
  }
  const candidate& w = cands[win];
  std::shared_ptr<plan_core> final_core = w.core;
  if (!same_shape) {
    build_settings cs = settings;
    apply_choice(w.choice, cs);
    final_core = build_plan_core(s, base, cs);
  }
  final_core->program_text = choice_text(pinned_choice(w.choice, *final_core));
  final_core->planning_seconds = seconds_since(start);
  final_core->candidates_timed = timed;
  final_core->rules_over_chosen = cands.front().best / w.best;
  final_core->chosen_rank = win;
  final_core->planning_note =
      wisdom_note + "measured (" + std::to_string(timed) + " of " + std::to_string(cands.size()) +
      " distinct candidates timed" +
      (same_shape ? ""
                  : " on " + std::to_string(cands.front().core->prog.transforms) + " transforms") +
      ", " + fixed(final_core->planning_seconds, 3) + " s): " + final_core->program_text +
      (win == 0 ? ""
                : "; the rules' choice took " + fixed(final_core->rules_over_chosen, 3) +
                      " times as long");
  return final_core;
}

} // namespace

std::shared_ptr<const plan_core> plan_with_effort(const shape_view& s, const options& opts,
                                                  const build_settings& settings) {
  if (!std::isfinite(opts.measure_seconds) || opts.measure_seconds < 0) {
    reject("options.measure_seconds", "must be finite and not negative");
  }
  options base = opts;
  base.use = nullptr;
  base.effort = planning::rules;
  std::shared_ptr<plan_core> rules = build_plan_core(s, base, settings);

  std::string wisdom_note;
  if (opts.use != nullptr) {
    if (const std::optional<wisdom_record> e = find_entry(*opts.use, *rules)) {
      if (const std::optional<plan_choice> choice = parse_choice(e->program)) {
        build_settings cs = settings;
        apply_choice(*choice, cs);
        std::shared_ptr<plan_core> core = build_plan_core(s, base, cs);
        if (program_fingerprint(*core) == e->check) {
          core->program_text = e->program;
          core->planning_note = "from wisdom: " + e->program;
          return core;
        }
      }
      wisdom_note = "a wisdom entry for this shape does not rebuild its program here (ignored); ";
    }
  }
  if (opts.effort == planning::rules) {
    rules->planning_note = wisdom_note.empty() ? "" : wisdom_note + "the rules' choice";
    return rules;
  }
  const budget& b = opts.effort == planning::quick ? quick_budget : measure_budget;
  const double seconds = opts.measure_seconds > 0 ? opts.measure_seconds : b.seconds;
  const std::size_t limit =
      opts.measure_scratch_bytes != 0 ? opts.measure_scratch_bytes : default_scratch_bytes;
  return measure(s, base, settings, std::move(rules), b, seconds, limit, wisdom_note);
}

} // namespace fftn::inline v0::detail
