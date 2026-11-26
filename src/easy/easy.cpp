// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The convenience layer's calls and cache (fftn/easy.hpp, DESIGN.md 5.4). It is built on the
// public API alone: every call finds or builds an fftn::plan and executes it with
// plan::execute, so it computes exactly what that plan computes. Nothing here includes a
// private header.

#include "fftn/easy.hpp"

#include "fftn/edges.hpp"
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

using any_plan =
    std::variant<c2c_plan<double>, r2c_plan<double>, c2r_plan<double>, r2r_plan<double>,
                 c2c_plan<float>, r2c_plan<float>, c2r_plan<float>, r2r_plan<float>>;

// The arrays of a real-to-real plan's orthogonalising weights (scipy's orthogonalize), the real
// factors of its input and output edges, in the plan's precision; empty on a side without
// weights. Kept with the plan, whose executions read them.
struct weights {
  std::vector<double> in_double, out_double;
  std::vector<float> in_float, out_float;

  [[nodiscard]] bool any() const noexcept {
    return !in_double.empty() || !out_double.empty() || !in_float.empty() || !out_float.empty();
  }
  [[nodiscard]] exec_args args() const noexcept {
    const auto address = [](const auto& d, const auto& f) -> const void* {
      return !d.empty() ? static_cast<const void*>(d.data())
                        : (!f.empty() ? static_cast<const void*>(f.data()) : nullptr);
    };
    return {.in_factor = address(in_double, in_float),
            .out_factor = address(out_double, out_float)};
  }
};

// A cached plan and the description of the calls it serves.
struct entry {
  std::vector<std::uint64_t> key;
  std::uint64_t hash = 0;
  std::uint64_t used = 0; // the context's tick at the last call that used it
  any_plan plan;
  weights factors;
};

// A call, validated.
struct prepared {
  std::array<std::size_t, 3> axes{}; // the transformed dimensions, in the order listed
  std::size_t count = 0;             // how many
  bool in_place = false;
  bool empty = false; // a looped dimension has extent 0: nothing to compute
  double scale = 1.0;
  r2r_kind r2r = r2r_kind::dht; // a real-to-real call: every transformed axis's kind
  // A real-to-real call with norm::ortho over several axes whose kind has weights: they are
  // applied by passes over the weighted hyperplanes, not by edge factors (one array of every
  // transformed point would be as large as the data).
  bool passes = false;
  // Those passes weight the input: the input is copied into the output first and the plan,
  // in place on the output's layout, runs there (in_place is then set too).
  bool copy_in = false;
};

// The kind of a real-to-real call: scipy's type of the DCT or DST, or the DHT. The inverse
// functions compute the inverse type, as scipy's do: types 1 and 4 are their own, 2 and 3 each
// other's.
r2r_kind kind_of(detail::easy_r2r family, int type, bool inverse) {
  if (family == detail::easy_r2r::hartley) {
    return r2r_kind::dht;
  }
  int t = type;
  if (inverse && (t == 2 || t == 3)) {
    t = 5 - t;
  }
  const int first = family == detail::easy_r2r::cosine ? static_cast<int>(r2r_kind::dct1)
                                                       : static_cast<int>(r2r_kind::dst1);
  return static_cast<r2r_kind>(first + t - 1);
}

// scipy's "logical" size of a real-to-real transform of n points: the N of a transform followed
// by its inverse multiplying by N (DESIGN.md 5.7's table).
double logical_size(r2r_kind k, std::size_t n) {
  const auto x = static_cast<double>(n);
  switch (k) {
  case r2r_kind::dct1:
    return 2 * (x - 1);
  case r2r_kind::dst1:
    return 2 * (x + 1);
  case r2r_kind::dht:
    return x;
  default:
    return 2 * x;
  }
}

// Which points of a line of n carry scipy's orthogonalising weight under norm="ortho": the
// input points multiplied by sqrt 2 and the output points divided by it, for kind k.
bool weighted_input(r2r_kind k, std::size_t n, std::size_t j) {
  switch (k) {
  case r2r_kind::dct1:
    return j == 0 || j == n - 1;
  case r2r_kind::dct3:
    return j == 0;
  case r2r_kind::dst3:
    return j == n - 1;
  default:
    return false;
  }
}
bool weighted_output(r2r_kind k, std::size_t n, std::size_t j) {
  switch (k) {
  case r2r_kind::dct1:
    return j == 0 || j == n - 1;
  case r2r_kind::dct2:
    return j == 0;
  case r2r_kind::dst2:
    return j == n - 1;
  default:
    return false;
  }
}
bool has_input_weights(r2r_kind k) {
  return k == r2r_kind::dct1 || k == r2r_kind::dct3 || k == r2r_kind::dst3;
}
bool has_output_weights(r2r_kind k) {
  return k == r2r_kind::dct1 || k == r2r_kind::dct2 || k == r2r_kind::dst2;
}

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
  const bool real_kind = c.kind == detail::easy_kind::r2c || c.kind == detail::easy_kind::c2r;
  if (c.kind == detail::easy_kind::r2r) {
    if (c.family != detail::easy_r2r::hartley && (c.type < 1 || c.type > 4)) {
      refuse(c, "type is " + std::to_string(c.type) + "; scipy's types are 1, 2, 3 and 4");
    }
    p.r2r = kind_of(c.family, c.type, c.dir == direction::backward);
  }
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
    if (transformed(d) && n == 1 && c.kind == detail::easy_kind::r2r && p.r2r == r2r_kind::dct1) {
      refuse(c,
             extent_name(c, real_side, d) + " is 1; the DCT of type 1 needs at least two points");
    }
    const bool halve = real_kind && d == halved;
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
      points *= c.kind == detail::easy_kind::r2r ? logical_size(p.r2r, n) : static_cast<double>(n);
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
    if (real_kind) {
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
  if (c.kind == detail::easy_kind::r2r && c.scaling == norm::ortho && p.count > 1 &&
      (has_input_weights(p.r2r) || has_output_weights(p.r2r))) {
    p.passes = true;
    p.copy_in = has_input_weights(p.r2r) && !p.in_place;
    p.in_place = p.in_place || p.copy_in;
  }
  const bool forward = c.kind == detail::easy_kind::r2c ||
                       (c.kind != detail::easy_kind::c2r && c.dir == direction::forward);
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
    (static_cast<std::uint64_t>(c.scaling) << 12U) | (static_cast<std::uint64_t>(c.family) << 16U) |
    (static_cast<std::uint64_t>(c.kind == detail::easy_kind::r2r ? p.r2r : r2r_kind{}) << 20U));
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

// The array of one side's weights along the one transformed axis of a real-to-real call with
// norm::ortho (scipy's orthogonalize), its n points, broadcast along the batch: the real factor of
// that side's edge. Empty when the side has no weights, or when the call transforms several axes
// (prepared::passes).
template <class T>
std::vector<T> side_weights(const detail::easy_call& c, const prepared& p, bool input,
                            std::vector<std::ptrdiff_t>& strides) {
  strides.assign(p.count, 0);
  if (c.scaling != norm::ortho || p.count != 1 ||
      !(input ? has_input_weights(p.r2r) : has_output_weights(p.r2r))) {
    return {};
  }
  strides[0] = 1;
  const std::size_t n = c.in_extents[p.axes[0]];
  const double root2 = std::sqrt(2.0);
  std::vector<T> w(n, T{1});
  for (std::size_t j = 0; j < n; ++j) {
    if (input ? weighted_input(p.r2r, n, j) : weighted_output(p.r2r, n, j)) {
      w[j] = static_cast<T>(input ? root2 : root2 / 2);
    }
  }
  return w;
}

// The offset of flat C-order position e of an array of `extents` with `strides` as a view gives
// them (`given`; else tight C order), skipping dimension `skip` (whose index is 0 there; SIZE_MAX
// skips none). Divisions rather than an odometer, so that any rank needs no storage.
std::ptrdiff_t offset_of(std::size_t e, std::span<const std::size_t> extents,
                         std::span<const std::ptrdiff_t> strides, bool given, std::size_t skip) {
  std::ptrdiff_t at = 0;
  std::ptrdiff_t span = 1;
  for (std::size_t d = extents.size(); d > 0; --d) {
    const std::size_t n = extents[d - 1];
    const std::ptrdiff_t stride = given ? strides[d - 1] : span;
    span *= static_cast<std::ptrdiff_t>(n);
    if (d - 1 == skip) {
      continue;
    }
    at += static_cast<std::ptrdiff_t>(e % n) * stride;
    e /= n;
  }
  return at;
}

// scipy's orthogonalising weights of a real-to-real call over several axes (prepared::passes),
// on `data` (the output's layout): along each transformed axis, the hyperplanes of the weighted
// positions multiplied by sqrt 2 (the input's) or 1/sqrt 2 (the output's). The weight of a
// point is the product over its axes, so the passes, one per axis, give it; each touches N/n
// points.
template <class T>
void weight_hyperplanes(const detail::easy_call& c, const prepared& p, T* data, bool input) {
  const std::span<const std::size_t> ext = c.out_extents;
  const bool given = any_given(c.out_strides);
  std::size_t total = 1;
  for (const std::size_t e : ext) {
    total *= e;
  }
  const double root2 = std::sqrt(2.0);
  const auto w = static_cast<T>(input ? root2 : root2 / 2);
  for (std::size_t i = 0; i < p.count; ++i) {
    const std::size_t a = p.axes[i];
    const std::size_t n = ext[a];
    for (std::size_t k = 0; k < n; ++k) {
      if (!(input ? weighted_input(p.r2r, n, k) : weighted_output(p.r2r, n, k))) {
        continue;
      }
      T* plane = data + static_cast<std::ptrdiff_t>(k) * stride_at(ext, c.out_strides, given, a);
      for (std::size_t e = 0; e < total / n; ++e) {
        plane[offset_of(e, ext, c.out_strides, given, a)] *= w;
      }
    }
  }
}

template <class In, class Out>
any_plan build(const detail::easy_call& c, const prepared& p, weights& factors) {
  const std::size_t rank = c.in_extents.size();
  const bool gi = any_given(c.in_strides);
  const bool go = any_given(c.out_strides);
  const bool real_out = c.kind == detail::easy_kind::c2r;
  std::vector<axis> array(rank);
  for (std::size_t d = 0; d < rank; ++d) {
    const std::ptrdiff_t os = stride_at(c.out_extents, c.out_strides, go, d);
    array[d] = axis{
        .n = real_out ? c.out_extents[d] : c.in_extents[d],
        // A weighted input is copied into the output, where the plan runs in place.
        .in_stride = p.copy_in ? os : stride_at(c.in_extents, c.in_strides, gi, d),
        .out_stride = os,
    };
  }
  const std::vector<std::size_t> selected(p.axes.begin(),
                                          p.axes.begin() + static_cast<std::ptrdiff_t>(p.count));
  const options opts{.in_place = p.in_place, .scale = p.scale};
  try {
    if constexpr (std::is_floating_point_v<In> && std::is_same_v<In, Out>) {
      // A real-to-real call: scipy's weights, where norm::ortho gives some, as real factors on
      // the edges; without weights, the plain plan.
      std::vector<std::ptrdiff_t> in_strides;
      std::vector<std::ptrdiff_t> out_strides;
      std::vector<In> win = side_weights<In>(c, p, true, in_strides);
      std::vector<In> wout = side_weights<In>(c, p, false, out_strides);
      const std::vector<r2r_kind> kinds(p.count, p.r2r);
      const shape<In, Out> s = axes_r2r<In>(array, selected, kinds);
      plan_edges<In, Out> edges;
      if (!win.empty()) {
        edges.in.mul = factor{.kind = factor_kind::real, .axis_strides = in_strides};
      }
      if (!wout.empty()) {
        edges.out.mul = factor{.kind = factor_kind::real, .axis_strides = out_strides};
      }
      any_plan made = win.empty() && wout.empty()
                          ? any_plan(std::in_place_type<plan<In, Out>>, s, opts)
                          : any_plan(std::in_place_type<plan<In, Out>>, s, opts, edges);
      if constexpr (std::is_same_v<In, double>) {
        factors.in_double = std::move(win);
        factors.out_double = std::move(wout);
      } else {
        factors.in_float = std::move(win);
        factors.out_float = std::move(wout);
      }
      return made;
    } else if constexpr (std::is_same_v<In, Out>) {
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
                     norm nm, const void* in, std::size_t in_size, void* out, std::size_t out_size,
                     detail::easy_r2r family, int type) {
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
      .family = family,
      .type = type,
  });
}

namespace {

template <class In, class Out>
void execute_with(const entry& held, const void* in, void* out, workspace& ws,
                  const std::optional<executor_ref>& ex) {
  const plan<In, Out>& p = std::get<plan<In, Out>>(held.plan);
  const auto* i = static_cast<const In*>(in);
  auto* o = static_cast<Out*>(out);
  // Only a plan with weights takes the overloads with exec_args; every other plan executes as
  // it did before the real-to-real calls existed.
  if (held.factors.any()) {
    const exec_args args = held.factors.args();
    if (ex) {
      p.execute(i, o, *ex, ws, args);
    } else {
      p.execute(i, o, ws, args);
    }
  } else if (ex) {
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
  case detail::easy_kind::r2r:
    return c.single ? f.template operator()<float, float>()
                    : f.template operator()<double, double>();
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
    weights factors;
    any_plan built = dispatch(
        c, [&]<class In, class Out>() -> any_plan { return build<In, Out>(c, p, factors); });
    const std::size_t need =
        std::visit([&](const auto& q) { return q.workspace_bytes(lanes); }, built);
    std::optional<workspace> grown;
    if (need > s.ws.size()) {
      grown.emplace(need);
    }
    entry fresh{.key = {},
                .hash = hash,
                .used = 0,
                .plan = std::move(built),
                .factors = std::move(factors)};
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
  if (!p.passes) {
    dispatch(c, [&]<class In, class Out> {
      execute_with<In, Out>(*found, c.in, c.out, s.ws, executor_);
    });
    return;
  }
  // Weights over several axes, by passes on the calling thread (prepared::passes).
  const auto weighted = [&]<class T>() {
    auto* out = static_cast<T*>(c.out);
    const auto* in = static_cast<const T*>(c.in);
    if (has_input_weights(p.r2r)) {
      if (p.copy_in) {
        std::size_t total = 1;
        for (const std::size_t e : c.out_extents) {
          total *= e;
        }
        const bool gi = any_given(c.in_strides);
        const bool go = any_given(c.out_strides);
        for (std::size_t e = 0; e < total; ++e) {
          out[offset_of(e, c.out_extents, c.out_strides, go, SIZE_MAX)] =
              in[offset_of(e, c.in_extents, c.in_strides, gi, SIZE_MAX)];
        }
      }
      weight_hyperplanes<T>(c, p, out, true);
      in = out;
    }
    execute_with<T, T>(*found, in, out, s.ws, executor_);
    if (has_output_weights(p.r2r)) {
      weight_hyperplanes<T>(c, p, out, false);
    }
  };
  if (c.single) {
    weighted.template operator()<float>();
  } else {
    weighted.template operator()<double>();
  }
}

context& thread_context() noexcept {
  thread_local context own;
  return own;
}

} // namespace fftn::inline v0::easy
