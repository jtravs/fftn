// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The convenience layer's calls and cache (fftn/easy.hpp, DESIGN.md 5.4). It is built on the
// public API alone: every call finds or builds an fftn::plan and executes it with
// plan::execute, so it computes exactly what that plan computes. Nothing here includes a
// private header.

#include "fftn/easy.hpp"

#include "fftn/executor.hpp"
#include "fftn/options.hpp"
#include "fftn/plan.hpp"
#include "fftn/shape.hpp"
#include "fftn/workspace.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace fftn::inline v0::easy {

namespace {

using any_plan = std::variant<c2c_plan<double>, r2c_plan<double>, c2r_plan<double>, c2c_plan<float>,
                              r2c_plan<float>, c2r_plan<float>>;

// A cached plan and the description of the calls it serves.
struct entry {
  std::vector<std::uint64_t> key;
  std::uint64_t hash = 0;
  std::uint64_t used = 0; // the context's tick at the last call that used it
  any_plan plan;
};

// A call, validated.
struct prepared {
  std::array<std::size_t, 3> axes{}; // the transformed dimensions, in the order listed
  std::size_t count = 0;             // how many
  bool in_place = false;
  bool empty = false; // a looped dimension has extent 0: nothing to compute
  double scale = 1.0;
};

[[noreturn]] void refuse(const detail::easy_call& c, const std::string& what) {
  throw std::invalid_argument(std::string("fftn::easy::") + c.function + ": " + what);
}

std::string extent_name(const detail::easy_call& c, const char* side, std::size_t d) {
  return c.spans ? std::string(side) + ".size()"
                 : std::string(side) + ".extents[" + std::to_string(d) + "]";
}

bool any_given(std::span<const std::ptrdiff_t> strides) {
  return std::ranges::any_of(strides, [](std::ptrdiff_t s) { return s != 0; });
}

// Strides given with a 0 on a dimension that steps would make every element of it one.
void check_strides(const detail::easy_call& c, const char* side,
                   std::span<const std::size_t> extents, std::span<const std::ptrdiff_t> strides) {
  if (!any_given(strides)) {
    return;
  }
  for (std::size_t d = 0; d < strides.size(); ++d) {
    if (strides[d] == 0 && extents[d] > 1) {
      refuse(c, std::string(side) + ".strides[" + std::to_string(d) +
                    "] is 0 among nonzero strides; give every stride, or all 0 for tight C order");
    }
  }
}

// Stride d of a side: as given, or that of tight C order when all are 0. A given 0 (on an
// extent of 1, which never steps) becomes 1, so the plan sees no default to work out.
std::ptrdiff_t stride_at(std::span<const std::size_t> extents,
                         std::span<const std::ptrdiff_t> strides, bool given, std::size_t d) {
  if (given) {
    return strides[d] != 0 ? strides[d] : 1;
  }
  std::ptrdiff_t span = 1;
  for (std::size_t i = d + 1; i < extents.size(); ++i) {
    span *= static_cast<std::ptrdiff_t>(extents[i]);
  }
  return span;
}

prepared prepare(const detail::easy_call& c) {
  prepared p;
  const std::size_t rank = c.in_extents.size();
  // The dimensions to transform: numpy's axes, negative ones from the end.
  if (c.axes.size() == 0) {
    const std::size_t k = c.default_axes == 0 ? rank : c.default_axes;
    if (k > rank) {
      refuse(c, "the arrays have " + std::to_string(rank) + " dimension" + (rank == 1 ? "" : "s") +
                    "; " + c.function + " transforms the last " + std::to_string(k) +
                    " by default, so pass axes");
    }
    if (k > 3) {
      refuse(c, "axes is empty and the arrays have " + std::to_string(rank) +
                    " dimensions, more than the three a transform takes; pass axes");
    }
    for (std::size_t i = 0; i < k; ++i) {
      p.axes[i] = rank - k + i;
    }
    p.count = k;
  } else {
    if (c.axes.size() > 3) {
      refuse(c, "axes lists " + std::to_string(c.axes.size()) +
                    " dimensions; a transform takes one to three");
    }
    const auto r = static_cast<long long>(rank);
    for (const int a : c.axes.indices()) {
      if (a < -r || a >= r) {
        refuse(c, "axes lists " + std::to_string(a) + ", outside arrays of " +
                      std::to_string(rank) + " dimensions");
      }
      const auto d = static_cast<std::size_t>(a < 0 ? a + r : a);
      if (std::find(p.axes.begin(), p.axes.begin() + static_cast<std::ptrdiff_t>(p.count), d) !=
          p.axes.begin() + static_cast<std::ptrdiff_t>(p.count)) {
        refuse(c, "axes lists dimension " + std::to_string(d) + " twice");
      }
      p.axes[p.count++] = d;
    }
  }
  const std::size_t halved = p.axes[p.count - 1];
  const auto transformed = [&](std::size_t d) {
    return std::find(p.axes.begin(), p.axes.begin() + static_cast<std::ptrdiff_t>(p.count), d) !=
           p.axes.begin() + static_cast<std::ptrdiff_t>(p.count);
  };
  // Extents: equal on both sides but along the halved dimension of a real transform.
  double points = 1;
  for (std::size_t d = 0; d < rank; ++d) {
    const std::size_t ie = c.in_extents[d];
    const std::size_t oe = c.out_extents[d];
    const bool real_out = c.kind == detail::easy_kind::c2r;
    const std::size_t n = real_out ? oe : ie;
    const char* real_side = real_out ? "out" : "in";
    const char* other_side = real_out ? "in" : "out";
    if (transformed(d) && n == 0) {
      refuse(c, extent_name(c, real_side, d) + " is 0; a transformed dimension needs a point");
    }
    const bool halve = c.kind != detail::easy_kind::c2c && d == halved;
    const std::size_t want = halve ? n / 2 + 1 : n;
    const std::size_t have = real_out ? ie : oe;
    if (have != want) {
      refuse(c, extent_name(c, other_side, d) + " is " + std::to_string(have) + "; it must be " +
                    (halve ? extent_name(c, real_side, d) + "/2 + 1, "
                           : extent_name(c, real_side, d) + ", ") +
                    std::to_string(want));
    }
    if (n == 0) {
      p.empty = true;
    }
    if (transformed(d)) {
      points *= static_cast<double>(n);
    }
  }
  // An empty array is checked no further: std::layout_right gives the dimensions in front of an
  // extent of 0 a stride of 0, which the check below would refuse.
  if (p.empty) {
    return p;
  }
  check_strides(c, "in", c.in_extents, c.in_strides);
  check_strides(c, "out", c.out_extents, c.out_strides);
  if (c.in == c.out) {
    if (c.kind != detail::easy_kind::c2c) {
      refuse(c, "out is in's memory; the real transforms have no in-place form");
    }
    const bool gi = any_given(c.in_strides);
    const bool go = any_given(c.out_strides);
    for (std::size_t d = 0; d < rank; ++d) {
      if (c.in_extents[d] > 1 && stride_at(c.in_extents, c.in_strides, gi, d) !=
                                     stride_at(c.out_extents, c.out_strides, go, d)) {
        refuse(c, "out is in's memory with other strides; in place needs the same strides");
      }
    }
    p.in_place = true;
  }
  const bool forward = c.kind == detail::easy_kind::r2c ||
                       (c.kind == detail::easy_kind::c2c && c.dir == direction::forward);
  switch (c.scaling) {
  case norm::backward:
    p.scale = forward ? 1.0 : 1.0 / points;
    break;
  case norm::ortho:
    p.scale = 1.0 / std::sqrt(points);
    break;
  case norm::forward:
    p.scale = forward ? 1.0 / points : 1.0;
    break;
  }
  return p;
}

// The words that identify a call's plan: kind, precision, direction, norm, placement, the
// axes, and both sides' extents and strides as given. Equal words give equal shapes and
// options, so the plan of one call serves the other. Visited rather than stored, so that a
// lookup allocates nothing.
template <class F> void visit_key(const detail::easy_call& c, const prepared& p, F&& f) {
  f(static_cast<std::uint64_t>(c.kind) | (static_cast<std::uint64_t>(c.single) << 8U) |
    (static_cast<std::uint64_t>(c.dir == direction::forward) << 9U) |
    (static_cast<std::uint64_t>(p.in_place) << 10U) |
    (static_cast<std::uint64_t>(c.scaling) << 12U));
  f(static_cast<std::uint64_t>(c.in_extents.size()));
  f(static_cast<std::uint64_t>(p.count));
  for (std::size_t i = 0; i < p.count; ++i) {
    f(static_cast<std::uint64_t>(p.axes[i]));
  }
  for (std::size_t d = 0; d < c.in_extents.size(); ++d) {
    f(static_cast<std::uint64_t>(c.in_extents[d]));
    f(static_cast<std::uint64_t>(c.out_extents[d]));
    f(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(c.in_strides[d])));
    f(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(c.out_strides[d])));
  }
}

// FNV-1a over 64-bit words: only a filter before the full comparison.
constexpr std::uint64_t hash_start = 0xcbf29ce484222325ULL;
constexpr std::uint64_t mix(std::uint64_t h, std::uint64_t w) { return (h ^ w) * 0x100000001b3ULL; }

template <class In, class Out> any_plan build(const detail::easy_call& c, const prepared& p) {
  const std::size_t rank = c.in_extents.size();
  const bool gi = any_given(c.in_strides);
  const bool go = any_given(c.out_strides);
  const bool real_out = c.kind == detail::easy_kind::c2r;
  std::vector<axis> array(rank);
  for (std::size_t d = 0; d < rank; ++d) {
    array[d] = axis{
        .n = real_out ? c.out_extents[d] : c.in_extents[d],
        .in_stride = stride_at(c.in_extents, c.in_strides, gi, d),
        .out_stride = stride_at(c.out_extents, c.out_strides, go, d),
    };
  }
  const std::vector<std::size_t> selected(p.axes.begin(),
                                          p.axes.begin() + static_cast<std::ptrdiff_t>(p.count));
  const options opts{.in_place = p.in_place, .scale = p.scale};
  try {
    if constexpr (std::is_same_v<In, Out>) {
      using T = Out::value_type;
      return any_plan(std::in_place_type<plan<In, Out>>, axes_c2c<T>(array, selected, c.dir), opts);
    } else if constexpr (std::is_same_v<Out, std::complex<In>>) {
      return any_plan(std::in_place_type<plan<In, Out>>, axes_r2c<In>(array, selected), opts);
    } else {
      return any_plan(std::in_place_type<plan<In, Out>>, axes_c2r<Out>(array, selected), opts);
    }
  } catch (const std::invalid_argument& e) {
    // The core names its own arguments (shape.axes[1], ...): say whose they are.
    refuse(c, std::string("the arrays' layout is not one a plan takes: ") + e.what());
  }
}

} // namespace

struct context::state {
  std::vector<entry> entries;
  std::size_t max_plans = default_max_plans;
  std::uint64_t tick = 0;
  workspace ws;
};

context::context() noexcept = default;

context::context(executor_ref ex) noexcept : executor_(ex) {}

context::context(context&& other) noexcept
    : state_(std::move(other.state_)), executor_(other.executor_) {}

context& context::operator=(context&& other) noexcept {
  state_ = std::move(other.state_);
  executor_ = other.executor_;
  return *this;
}

context::~context() = default;

context::state& context::own() {
  if (!state_) {
    state_ = std::make_unique<state>();
  }
  return *state_;
}

void context::set_max_plans(std::size_t n) {
  if (n == 0) {
    throw std::invalid_argument(
        "fftn::easy::context::set_max_plans: max_plans is 0; a context keeps at least one plan");
  }
  state& s = own();
  s.max_plans = n;
  if (s.entries.size() > n) {
    std::ranges::sort(s.entries, [](const entry& a, const entry& b) { return a.used > b.used; });
    s.entries.erase(s.entries.begin() + static_cast<std::ptrdiff_t>(n), s.entries.end());
  }
}

std::size_t context::max_plans() const noexcept {
  return state_ ? state_->max_plans : default_max_plans;
}

std::size_t context::cached_plans() const noexcept { return state_ ? state_->entries.size() : 0; }

void context::clear() noexcept {
  if (state_) {
    std::vector<entry>().swap(state_->entries);
    state_->ws = workspace();
  }
}

std::size_t context::irfft_length(std::size_t in_size, std::size_t n) {
  if (n != 0) {
    return n;
  }
  if (in_size < 2) {
    throw std::invalid_argument("fftn::easy::irfft: in has " + std::to_string(in_size) +
                                " element" + (in_size == 1 ? "" : "s") +
                                ", so the default n, 2(in.size() - 1), is not a length; give n");
  }
  return 2 * (in_size - 1);
}

void context::run_1d(const char* function, detail::easy_kind kind, bool single, direction dir,
                     norm nm, const void* in, std::size_t in_size, void* out,
                     std::size_t out_size) {
  const std::array<std::size_t, 1> in_extents{in_size};
  const std::array<std::size_t, 1> out_extents{out_size};
  const std::array<std::ptrdiff_t, 1> unit{1};
  run({
      .function = function,
      .kind = kind,
      .single = single,
      .dir = dir,
      .scaling = nm,
      .in_extents = in_extents,
      .in_strides = unit,
      .out_extents = out_extents,
      .out_strides = unit,
      .axes = {},
      .default_axes = 1,
      .spans = true,
      .in = in,
      .out = out,
  });
}

namespace {

template <class In, class Out>
void execute_with(const any_plan& held, const void* in, void* out, workspace& ws,
                  const std::optional<executor_ref>& ex) {
  const plan<In, Out>& p = std::get<plan<In, Out>>(held);
  const auto* i = static_cast<const In*>(in);
  auto* o = static_cast<Out*>(out);
  if (ex) {
    p.execute(i, o, *ex, ws);
  } else {
    p.execute(i, o, ws);
  }
}

// Calls f.template operator()<In, Out>() for the call's kind and precision.
template <class F> decltype(auto) dispatch(const detail::easy_call& c, F&& f) {
  using cd = std::complex<double>;
  using cf = std::complex<float>;
  switch (c.kind) {
  case detail::easy_kind::c2c:
    return c.single ? f.template operator()<cf, cf>() : f.template operator()<cd, cd>();
  case detail::easy_kind::r2c:
    return c.single ? f.template operator()<float, cf>() : f.template operator()<double, cd>();
  case detail::easy_kind::c2r:
    break;
  }
  return c.single ? f.template operator()<cf, float>() : f.template operator()<cd, double>();
}

} // namespace

void context::run(const detail::easy_call& c) {
  const prepared p = prepare(c);
  if (p.empty) {
    return;
  }
  state& s = own();
  std::uint64_t hash = hash_start;
  std::size_t words = 0;
  visit_key(c, p, [&](std::uint64_t w) {
    hash = mix(hash, w);
    ++words;
  });
  entry* found = nullptr;
  for (entry& e : s.entries) {
    if (e.hash != hash || e.key.size() != words) {
      continue;
    }
    std::size_t i = 0;
    bool same = true;
    visit_key(c, p, [&](std::uint64_t w) { same = same && e.key[i++] == w; });
    if (same) {
      found = &e;
      break;
    }
  }
  const int lanes = executor_ ? executor_->concurrency() : 1;
  if (found == nullptr) {
    // A miss: build the plan and its workspace before changing anything, so that an
    // exception leaves the context as it was.
    any_plan built =
        dispatch(c, [&]<class In, class Out>() -> any_plan { return build<In, Out>(c, p); });
    const std::size_t need =
        std::visit([&](const auto& q) { return q.workspace_bytes(lanes); }, built);
    std::optional<workspace> grown;
    if (need > s.ws.size()) {
      grown.emplace(need);
    }
    entry fresh{.key = {}, .hash = hash, .used = 0, .plan = std::move(built)};
    fresh.key.reserve(words);
    visit_key(c, p, [&](std::uint64_t w) { fresh.key.push_back(w); });
    if (s.entries.size() < s.max_plans) {
      s.entries.push_back(std::move(fresh));
      found = &s.entries.back();
    } else {
      found = &*std::ranges::min_element(s.entries, {}, &entry::used);
      *found = std::move(fresh);
    }
    if (grown) {
      s.ws = std::move(*grown);
    }
  } else {
    const std::size_t need =
        std::visit([&](const auto& q) { return q.workspace_bytes(lanes); }, found->plan);
    if (need > s.ws.size()) {
      s.ws = workspace(need);
    }
  }
  found->used = ++s.tick;
  dispatch(c, [&]<class In, class Out> {
    execute_with<In, Out>(found->plan, c.in, c.out, s.ws, executor_);
  });
}

context& thread_context() noexcept {
  thread_local context own;
  return own;
}

} // namespace fftn::inline v0::easy
