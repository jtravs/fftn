// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/candidates.hpp"

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"
#include "threading/schedule.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

std::string dotted(const std::vector<std::size_t>& v) {
  std::string text;
  for (const std::size_t x : v) {
    text += (text.empty() ? "" : ".") + std::to_string(x);
  }
  return text;
}

std::optional<std::vector<std::size_t>> parse_dotted(std::string_view text) {
  std::vector<std::size_t> out;
  std::size_t at = 0;
  while (at <= text.size()) {
    const std::size_t end = std::min(text.find('.', at), text.size());
    const std::string_view token = text.substr(at, end - at);
    std::size_t v = 0;
    const auto res = std::from_chars(token.data(), token.data() + token.size(), v);
    if (token.empty() || res.ec != std::errc{} || res.ptr != token.data() + token.size() || v < 2) {
      return std::nullopt;
    }
    out.push_back(v);
    at = end + 1;
  }
  return out;
}

const char* strategy_name(strategy_choice s) {
  switch (s) {
  case strategy_choice::chain:
    return "chain";
  case strategy_choice::grid:
    return "grid";
  case strategy_choice::lanes:
    return "lanes";
  case strategy_choice::rules:
    break;
  }
  return "rules";
}

// Whether n has a prime factor that no codelet covers (a Bluestein or Rader stage).
bool has_big_prime(std::size_t n) {
  std::size_t rest = n;
  for (std::size_t q = 2; q <= rest / q; ++q) {
    while (rest % q == 0) {
      if (!has_codelet(q)) {
        return true;
      }
      rest /= q;
    }
  }
  return rest > 1 && !has_codelet(rest);
}

std::size_t complex_lanes(target t, precision p) {
  const kernel_table& k = *find_kernels(t);
  return p == precision::f64 ? of<double>(k).complex_lanes : of<float>(k).complex_lanes;
}

void add(std::vector<plan_choice>& out, plan_choice c) {
  if (std::ranges::find(out, c) == out.end()) {
    out.push_back(std::move(c));
  }
}

} // namespace

std::string choice_text(const plan_choice& c) {
  std::string text;
  const auto field = [&](const std::string& f) { text += (text.empty() ? "" : " ") + f; };
  if (c.strategy != strategy_choice::rules) {
    field(std::string("strategy=") + strategy_name(c.strategy));
  }
  if (!c.radices.empty()) {
    field("radices=" + dotted(c.radices));
  }
  if (!c.grid_factors.empty()) {
    field("grid=" + dotted(c.grid_factors));
  }
  if (c.grid_phases != 0) {
    field("phases=" + std::to_string(c.grid_phases));
  }
  if (c.rader >= 0) {
    field("rader=" + std::to_string(c.rader));
  }
  return text.empty() ? "rules" : text;
}

std::optional<plan_choice> parse_choice(std::string_view text) {
  plan_choice c;
  if (text == "rules") {
    return c;
  }
  std::size_t at = 0;
  int last = -1; // the fields come in choice_text's order, each once
  while (at < text.size()) {
    const std::size_t end = std::min(text.find(' ', at), text.size());
    const std::string_view token = text.substr(at, end - at);
    at = end + 1;
    const std::size_t eq = token.find('=');
    if (eq == std::string_view::npos) {
      return std::nullopt;
    }
    const std::string_view key = token.substr(0, eq);
    const std::string_view value = token.substr(eq + 1);
    int field = -1;
    if (key == "strategy") {
      field = 0;
      if (value == "chain") {
        c.strategy = strategy_choice::chain;
      } else if (value == "grid") {
        c.strategy = strategy_choice::grid;
      } else if (value == "lanes") {
        c.strategy = strategy_choice::lanes;
      } else {
        return std::nullopt;
      }
    } else if (key == "radices" || key == "grid") {
      field = key == "radices" ? 1 : 2;
      std::optional<std::vector<std::size_t>> v = parse_dotted(value);
      if (!v || (field == 2 && (v->size() < 2 || v->size() > 3))) {
        return std::nullopt;
      }
      (field == 1 ? c.radices : c.grid_factors) = *v;
    } else if (key == "phases") {
      field = 3;
      if (value != "2" && value != "3") {
        return std::nullopt;
      }
      c.grid_phases = value == "2" ? 2 : 3;
    } else if (key == "rader") {
      field = 4;
      if (value != "0" && value != "1") {
        return std::nullopt;
      }
      c.rader = value == "0" ? 0 : 1;
    } else {
      return std::nullopt;
    }
    if (field <= last) {
      return std::nullopt;
    }
    last = field;
  }
  if (last < 0) {
    return std::nullopt;
  }
  return c;
}

void apply_choice(const plan_choice& c, build_settings& s) {
  s.strategy = c.strategy;
  s.radices = c.radices;
  s.grid_factors = c.grid_factors;
  s.grid_phases = c.grid_phases;
  s.rader = c.rader;
}

plan_choice pinned_choice(const plan_choice& c, const plan_core& built) {
  plan_choice p = c;
  if (built.built == strategy_choice::grid && !built.grid_factors.empty()) {
    p.strategy = strategy_choice::grid;
    p.grid_factors = built.grid_factors;
    p.grid_phases = 0;
  } else {
    // No grid was built: the grid's fields did nothing, and neither did lanes not built.
    p.grid_factors.clear();
    p.grid_phases = 0;
    if (p.strategy == strategy_choice::grid ||
        (p.strategy == strategy_choice::lanes && built.built != strategy_choice::lanes)) {
      p.strategy = strategy_choice::rules;
    }
  }
  // Radices apply only to a complex line of their product (chain_builder::forces).
  if (!p.radices.empty()) {
    const std::size_t product = checked_product(p.radices);
    bool applies = false;
    const std::size_t r = built.lengths.size();
    for (std::size_t d = 0; d < r; ++d) {
      const std::size_t n = built.lengths[d];
      const bool halved = built.kind != transform_kind::c2c && d + 1 == r && n % 2 == 0;
      applies = applies || product == (halved ? n / 2 : n) ||
                (built.kind != transform_kind::c2c && d + 1 == r && product == n);
    }
    if (!applies) {
      p.radices.clear();
    }
  }
  if (p.rader >= 0 && !std::ranges::any_of(built.lengths, has_big_prime)) {
    p.rader = -1;
  }
  return p;
}

std::vector<plan_choice> enumerate_candidates(const plan_core& rules_plan,
                                              const build_settings& settings,
                                              std::size_t radix_choices) {
  std::vector<plan_choice> out{plan_choice{}};
  const machine_profile& m = settings.profile;
  target_profile profile =
      profile_of(rules_plan.isa_target, complex_lanes(rules_plan.isa_target, rules_plan.prec));
  if (settings.leaf_limit != 0) {
    profile.leaf_limit = settings.leaf_limit;
  }
  const std::size_t complex_bytes = rules_plan.prec == precision::f64 ? 16 : 8;
  const std::size_t r = rules_plan.lengths.size();
  const bool real = rules_plan.kind != transform_kind::c2c;

  // The complex line lengths of the plan, each once: the real axis's half-length transform
  // when its length is even.
  std::vector<std::size_t> lines;
  for (std::size_t d = 0; d < r; ++d) {
    std::size_t n = rules_plan.lengths[d];
    if (real && d + 1 == r) {
      if (n % 2 != 0) {
        continue;
      }
      n /= 2;
    }
    if (n >= 2 && std::ranges::find(lines, n) == lines.end()) {
      lines.push_back(n);
    }
  }

  std::vector<plan_choice> radix_alternatives;
  for (const std::size_t n : lines) {
    const bool beyond = n * complex_bytes > m.grid_min_bytes;
    const std::size_t k = beyond ? std::min<std::size_t>(radix_choices, 2) : radix_choices;
    for (std::vector<std::size_t>& rad : ranked_radices(n, profile, k)) {
      if (rad.size() >= 2) {
        add(radix_alternatives, plan_choice{.radices = std::move(rad)});
      }
    }
  }

  if (r == 1) {
    const std::size_t n = rules_plan.lengths[0];
    // The complex line: n, or half an even real n (an odd real n has none).
    std::size_t line = n;
    if (real) {
      line = n % 2 == 0 ? n / 2 : 0;
    }
    if (line >= 2 && line * complex_bytes > m.grid_block_bytes) {
      add(out, {.strategy = strategy_choice::chain});
      add(out, {.strategy = strategy_choice::grid, .grid_phases = 2});
      if (line * complex_bytes > m.grid_min_bytes) {
        add(out, {.strategy = strategy_choice::grid, .grid_phases = 3});
      }
      if (std::has_single_bit(line) && line >= 4) {
        const auto k = static_cast<std::size_t>(std::countr_zero(line));
        add(out, {.strategy = strategy_choice::chain, .radices = long_pow2_radices(k, true)});
        add(out, {.strategy = strategy_choice::chain, .radices = long_pow2_radices(k, false)});
      }
    }
    if (!rules_plan.counts.empty() && rules_plan.counts.back() >= 2) {
      add(out, {.strategy = strategy_choice::lanes});
      add(out, {.strategy = strategy_choice::chain});
    }
  }
  for (plan_choice& c : radix_alternatives) {
    add(out, std::move(c));
  }
  if (std::ranges::any_of(rules_plan.lengths, has_big_prime)) {
    add(out, {.rader = 0});
    add(out, {.rader = 1});
  }
  return out;
}

std::uint64_t program_fingerprint(const plan_core& c) {
  const std::string text = describe(c);
  std::uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a, 64 bits
  std::size_t at = 0;
  while (at < text.size()) {
    const std::size_t end = std::min(text.find('\n', at), text.size());
    const std::string_view line = std::string_view(text).substr(at, end - at);
    at = end + 1;
    if (line.starts_with("planning:") || line.starts_with("strategy:")) {
      continue;
    }
    for (const char ch : line) {
      h ^= static_cast<unsigned char>(ch);
      h *= 0x100000001b3ULL;
    }
    h ^= static_cast<unsigned char>('\n');
    h *= 0x100000001b3ULL;
  }
  return h;
}

} // namespace fftn::inline v0::detail
