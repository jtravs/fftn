// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Groups on row pipelines (fftn/pipeline.hpp, `pipeline_desc::group`; DESIGN.md 12.3; task 5.3f):
// a middle callback that couples the G lines of one batch dimension, a rotation mixing two
// components (and, for three, a cyclic mixing) by an angle that depends on the bin and the group,
// against the first transform, the coupling and the second transform evaluated with the
// quadruple-precision reference, within the derived tolerance. The callback checks its block
// (all G members of the group, group_stride apart, member 0's line number as its transform) and
// counts every bin of every group, exactly once; an epilogue callback checks its line's number.
// The group's dimension outermost, innermost and between two others; a kernel per line (stepping
// along the group's dimension too) fused or in a pass, a prologue factor, the two products of
// scales; <C, C, C> either direction first, <T, C, T> and <C, T, C>; strided lines and in place;
// serial, pieces of the batch (groups), teams (completion::all and own, which deal groups) and an
// executor bitwise alike, on every instruction set in both precisions. Refusals name desc.group.

#include "core/pipeline.hpp"
#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "reference/reference_dft.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/tolerance.hpp"
#include "threading/schedule.hpp"
#include "threading/work_graph.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using fftn::factor_kind;

template <class V> bool same_bits(const V& a, const V& b) {
  return a.size() == b.size() &&
         std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
}

enum class kind : unsigned char { ccc, tct, ctc };

struct group_case {
  const char* name = "";
  kind k = kind::ccc;
  std::size_t n = 16;
  std::vector<std::size_t> counts{2, 5}; // the batch dimensions, slowest first
  std::size_t group = 0;
  bool backward = false;
  factor_kind mid = factor_kind::none; // the kernel: complex of stride 1 (fused) or real
  bool per_line = false;
  bool prologue_factor = false;
  // Side factors per line (stepping along the group's dimension inside the members loops): the
  // prologue's and the epilogue's, real on a real side.
  bool side_factors_per_line = false;
  double pro_scale = 1.0;
  double options_scale = 1.0;
  std::ptrdiff_t stride = 1;
  bool in_place = false;
  bool epilogue_callback = false;
  bool threads = false;
};

// The angle of the coupling at bin j of the group whose member 0 is line `line0`.
double angle(std::size_t j, std::size_t line0) {
  return 0.3 + 0.05 * static_cast<double>(j) + 0.11 * static_cast<double>(line0);
}

// The coupling of G values at one bin (in double): a rotation of two, or the cyclic mixing
// v_k' = c v_k + s v_{k+1} of three.
template <class V> struct real_of {
  using type = V;
};
template <class R> struct real_of<std::complex<R>> {
  using type = R;
};

template <class V> void couple(std::vector<V>& v, double a) {
  using R = typename real_of<V>::type;
  const auto c = static_cast<R>(std::cos(a));
  const auto s = static_cast<R>(std::sin(a));
  if (v.size() == 2) {
    const V x = v[0];
    v[0] = c * x - s * v[1];
    v[1] = s * x + c * v[1];
  } else {
    const std::vector<V> x = v;
    for (std::size_t k = 0; k < v.size(); ++k) {
      v[k] = c * x[k] + s * x[(k + 1) % v.size()];
    }
  }
}

struct callback_state {
  std::size_t members = 0;
  std::size_t bins = 0;
  std::size_t lines = 0;
  std::size_t inner = 1; // lines between consecutive members
  std::vector<std::atomic<int>>* seen = nullptr;
  std::atomic<bool> ok{true};
};

template <class E>
void middle_callback(void* user, const fftn::block_info& info, E* data) noexcept {
  auto& st = *static_cast<callback_state*>(user);
  if (info.group != st.members || info.group_stride <= 0 || info.lines != 1 ||
      info.length != st.bins || info.origin.size() != 1 || info.origin[0] != 0 ||
      info.element_axis != 0) {
    st.ok = false;
    return;
  }
  // member 0's line: its group index within the inner dimensions below one member step.
  if (info.transform / st.inner % st.members != 0) {
    st.ok = false;
  }
  std::vector<E> v(st.members);
  for (std::size_t j = 0; j < info.length; ++j) {
    const std::ptrdiff_t at = static_cast<std::ptrdiff_t>(j) * info.element_stride;
    for (std::size_t k = 0; k < st.members; ++k) {
      v[k] = data[at + static_cast<std::ptrdiff_t>(k) * info.group_stride];
    }
    couple(v, angle(j, info.transform));
    for (std::size_t k = 0; k < st.members; ++k) {
      data[at + static_cast<std::ptrdiff_t>(k) * info.group_stride] = v[k];
    }
    if (st.seen != nullptr && info.transform < st.lines) {
      (*st.seen)[info.transform * st.bins + j].fetch_add(1, std::memory_order_relaxed);
    }
  }
}

// The epilogue callback's check: every element's line is the block's transform, which it reads
// from the user's map of line numbers to output addresses (none: nothing checked).
struct epilogue_state {
  std::vector<const void*> line_starts;
  std::size_t element_bytes = 0;
  std::atomic<bool> ok{true};
};

template <class E>
void epilogue_callback(void* user, const fftn::block_info& info, E* data) noexcept {
  if (user == nullptr) {
    return;
  }
  auto& st = *static_cast<epilogue_state*>(user);
  if (info.transform >= st.line_starts.size() || info.group != 1 ||
      static_cast<const void*>(data - static_cast<std::ptrdiff_t>(info.origin[0]) *
                                          info.element_stride) != st.line_starts[info.transform]) {
    st.ok = false;
  }
}

template <class T, kind K> struct types;
template <class T> struct types<T, kind::ccc> {
  using in = std::complex<T>;
  using mid = std::complex<T>;
  using out = std::complex<T>;
};
template <class T> struct types<T, kind::tct> {
  using in = T;
  using mid = std::complex<T>;
  using out = T;
};
template <class T> struct types<T, kind::ctc> {
  using in = std::complex<T>;
  using mid = T;
  using out = std::complex<T>;
};

using dvec = std::vector<std::complex<double>>;

// The exact transforms, rounded to double: the first (forward, or backward for a spectral pair;
// r2c's n/2 + 1 bins; c2r's n reals as complex values) and the second.
dvec exact(int sign, const dvec& x) {
  const auto y = fftn::test::reference::c2c(sign, x);
  dvec out(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) {
    out[i] = {y[i].re.hi, y[i].im.hi};
  }
  return out;
}
// A half spectrum of n/2 + 1 bins as c2r reads it: bin 0's and (even n) bin n/2's imaginary
// parts ignored, the rest Hermitian.
dvec hermitian(const dvec& h, std::size_t n) {
  dvec full(n);
  for (std::size_t j = 0; j <= n / 2; ++j) {
    full[j] = h[j];
  }
  full[0] = {h[0].real(), 0.0};
  if (n % 2 == 0) {
    full[n / 2] = {h[n / 2].real(), 0.0};
  }
  for (std::size_t j = n / 2 + 1; j < n; ++j) {
    full[j] = std::conj(full[n - j]);
  }
  return full;
}

template <class T, kind K> struct group_run {
  using In = typename types<T, K>::in;
  using Mid = typename types<T, K>::mid;
  using Out = typename types<T, K>::out;
  using desc_t = fftn::pipeline_desc<In, Mid, Out>;
  group_case c;
  fftn::isa target;
  std::size_t side = 0; // elements of a line on each side
  std::size_t bins = 0; // of the middle
  std::size_t lines = 1;
  std::size_t members = 0;
  std::size_t inner = 1;
  desc_t desc;
  std::vector<std::ptrdiff_t> dist;
  std::vector<In> x;
  std::vector<Out> init;
  std::vector<T> kreal;
  std::vector<std::complex<T>> kcomplex;
  std::vector<std::ptrdiff_t> kbatch;
  std::vector<std::complex<T>> pfactor;
  std::vector<T> rpfactor;
  std::vector<std::complex<T>> efactor;
  std::vector<T> refactor;

  group_run(const group_case& gc, fftn::isa t, fftn::test::input_source& src) : c(gc), target(t) {
    side = K == kind::ctc ? c.n / 2 + 1 : c.n;
    bins = K == kind::tct ? c.n / 2 + 1 : c.n;
    members = c.counts.at(c.group);
    for (std::size_t b = c.group + 1; b < c.counts.size(); ++b) {
      inner *= c.counts[b];
    }
    const auto s = static_cast<std::ptrdiff_t>(side);
    dist.assign(c.counts.size(), 0);
    std::ptrdiff_t step = s * c.stride + 1;
    for (std::size_t b = c.counts.size(); b-- > 0;) {
      dist[b] = step;
      step *= static_cast<std::ptrdiff_t>(c.counts[b]);
      lines *= c.counts[b];
    }
    desc.first = c.backward ? fftn::direction::backward : fftn::direction::forward;
    desc.line = {.n = c.n, .in_stride = c.stride, .out_stride = c.stride};
    for (std::size_t b = 0; b < c.counts.size(); ++b) {
      desc.batches.push_back({.count = c.counts[b], .in_dist = dist[b], .out_dist = dist[b]});
    }
    desc.group = c.group;
    desc.prologue.scale = c.pro_scale;
    desc.middle.scale = 1.0 / static_cast<double>(c.n);
    desc.middle.call = {.fn = &middle_callback<Mid>};
    if (c.epilogue_callback) {
      desc.epilogue.call = {.fn = &epilogue_callback<Out>};
    }
    if (c.mid != factor_kind::none) {
      // A kernel per line: line l's at l bins.
      kbatch.assign(c.counts.size(), 0);
      std::ptrdiff_t kstep = static_cast<std::ptrdiff_t>(bins);
      for (std::size_t b = c.counts.size(); b-- > 0;) {
        kbatch[b] = c.per_line ? kstep : 0;
        kstep *= static_cast<std::ptrdiff_t>(c.counts[b]);
      }
      const std::size_t size = c.per_line ? bins * lines : bins;
      if (c.mid == factor_kind::real) {
        kreal = src.reals<T>(size);
      } else {
        kcomplex = src.complexes<T>(size);
      }
      desc.middle.mul = {.kind = c.mid, .axis_strides = {1}, .batch_strides = kbatch};
    }
    if (c.prologue_factor) {
      pfactor = src.complexes<T>(side);
      desc.prologue.mul = {.kind = factor_kind::complex, .axis_strides = {1}, .batch_strides = {}};
    }
    if (c.side_factors_per_line) {
      // Line l's factors at l * side.
      std::vector<std::ptrdiff_t> fb(c.counts.size(), 0);
      std::ptrdiff_t fstep = s;
      for (std::size_t b = c.counts.size(); b-- > 0;) {
        fb[b] = fstep;
        fstep *= static_cast<std::ptrdiff_t>(c.counts[b]);
      }
      const auto fk = [](bool real_side) {
        return real_side ? factor_kind::real : factor_kind::complex;
      };
      if constexpr (std::is_floating_point_v<In>) {
        rpfactor = src.reals<T>(side * lines);
      } else {
        pfactor = src.complexes<T>(side * lines);
      }
      if constexpr (std::is_floating_point_v<Out>) {
        refactor = src.reals<T>(side * lines);
      } else {
        efactor = src.complexes<T>(side * lines);
      }
      desc.prologue.mul = {
          .kind = fk(std::is_floating_point_v<In>), .axis_strides = {1}, .batch_strides = fb};
      desc.epilogue.mul = {
          .kind = fk(std::is_floating_point_v<Out>), .axis_strides = {1}, .batch_strides = fb};
    }
    const std::size_t size = static_cast<std::size_t>(step) + 3;
    if constexpr (std::is_floating_point_v<In>) {
      x = src.reals<T>(size);
    } else {
      x = src.complexes<T>(size);
    }
    if constexpr (std::is_floating_point_v<Out>) {
      init = src.reals<T>(size);
    } else {
      init = src.complexes<T>(size);
    }
    if (c.in_place) {
      if constexpr (std::is_same_v<In, Out>) {
        init = x;
      }
    }
  }

  [[nodiscard]] std::size_t at(std::size_t j, std::size_t line) const {
    std::ptrdiff_t a = static_cast<std::ptrdiff_t>(j) * c.stride;
    std::size_t rest = line;
    for (std::size_t b = c.counts.size(); b-- > 0;) {
      a += static_cast<std::ptrdiff_t>(rest % c.counts[b]) * dist[b];
      rest /= c.counts[b];
    }
    return static_cast<std::size_t>(a);
  }

  [[nodiscard]] fftn::exec_args args(callback_state* mid, epilogue_state* epi) const {
    const void* k = nullptr;
    if (!kreal.empty()) {
      k = kreal.data();
    } else if (!kcomplex.empty()) {
      k = kcomplex.data();
    }
    const void* in_f = !rpfactor.empty()  ? static_cast<const void*>(rpfactor.data())
                       : !pfactor.empty() ? static_cast<const void*>(pfactor.data())
                                          : nullptr;
    const void* out_f = !refactor.empty()  ? static_cast<const void*>(refactor.data())
                        : !efactor.empty() ? static_cast<const void*>(efactor.data())
                                           : nullptr;
    return {
        .in_factor = in_f, .mid_factor = k, .out_factor = out_f, .mid_user = mid, .out_user = epi};
  }

  // The side factors at element j of line `line` (1 without one).
  [[nodiscard]] std::complex<double> side_factor(bool input, std::size_t j,
                                                 std::size_t line) const {
    const std::size_t i = c.side_factors_per_line ? line * side + j : j;
    const std::vector<T>& r = input ? rpfactor : refactor;
    const std::vector<std::complex<T>>& z = input ? pfactor : efactor;
    if (!r.empty()) {
      return {static_cast<double>(r[i]), 0.0};
    }
    if (!z.empty()) {
      return std::complex<double>(z[i]);
    }
    return {1.0, 0.0};
  }

  [[nodiscard]] std::complex<double> kernel(std::size_t j, std::size_t line) const {
    const std::size_t i = (c.per_line ? line * bins : 0) + j;
    if (!kreal.empty()) {
      return {static_cast<double>(kreal[i]), 0.0};
    }
    if (!kcomplex.empty()) {
      return std::complex<double>(kcomplex[i]);
    }
    return {1.0, 0.0};
  }

  // The reference's output lines, one per line, in double.
  [[nodiscard]] std::vector<dvec> reference() const {
    const int s1 = c.backward ? +1 : -1;
    std::vector<dvec> spectra(lines);
    for (std::size_t line = 0; line < lines; ++line) {
      dvec v(side);
      for (std::size_t j = 0; j < side; ++j) {
        std::complex<double> e{};
        if constexpr (std::is_floating_point_v<In>) {
          e = {static_cast<double>(x[at(j, line)]), 0.0};
        } else {
          e = std::complex<double>(x[at(j, line)]);
        }
        e *= side_factor(true, j, line);
        v[j] = e * c.pro_scale;
      }
      if constexpr (K == kind::ccc) {
        spectra[line] = exact(s1, v);
      } else if constexpr (K == kind::tct) {
        dvec full = exact(-1, v);
        full.resize(bins);
        spectra[line] = full;
      } else {
        dvec r = exact(+1, hermitian(v, c.n));
        for (auto& e : r) {
          e = {e.real(), 0.0};
        }
        spectra[line] = r;
      }
    }
    // The coupling per group, then the kernel and the scale.
    for (std::size_t line0 = 0; line0 < lines; ++line0) {
      if (line0 / inner % members != 0) {
        continue;
      }
      std::vector<std::complex<double>> v(members);
      for (std::size_t j = 0; j < bins; ++j) {
        for (std::size_t k = 0; k < members; ++k) {
          v[k] = spectra[line0 + k * inner][j];
        }
        couple(v, angle(j, line0));
        for (std::size_t k = 0; k < members; ++k) {
          spectra[line0 + k * inner][j] = v[k];
        }
      }
    }
    std::vector<dvec> out(lines);
    for (std::size_t line = 0; line < lines; ++line) {
      dvec z(bins);
      for (std::size_t j = 0; j < bins; ++j) {
        z[j] = spectra[line][j] * kernel(j, line) / static_cast<double>(c.n);
      }
      if constexpr (K == kind::ccc) {
        out[line] = exact(-s1, z);
      } else if constexpr (K == kind::tct) {
        out[line] = exact(+1, hermitian(z, c.n));
      } else {
        dvec full = exact(-1, z);
        full.resize(side);
        out[line] = full;
      }
      for (std::size_t j = 0; j < side; ++j) {
        out[line][j] *= side_factor(false, j, line) * c.options_scale;
      }
    }
    return out;
  }
};

template <class T, kind K> void run_case(const group_case& gc, fftn::isa t) {
  using Run = group_run<T, K>;
  using In = typename Run::In;
  using Mid = typename Run::Mid;
  using Out = typename Run::Out;
  const std::string what = std::string(gc.name) + ", n = " + std::to_string(gc.n) + ", " +
                           fftn::isa_name(t) + (sizeof(T) == 8 ? " double" : " float");
  SCOPED_TRACE(what);
  fftn::test::input_source src(0x6E0B5ULL + gc.n * 13 + gc.group);
  Run run(gc, t, src);
  const fftn::pipeline<In, Mid, Out> p(
      run.desc, {.in_place = gc.in_place, .scale = gc.options_scale, .force_isa = t});
  const std::string describe = p.describe();
  EXPECT_NE(describe.find("group: batch dimension " + std::to_string(gc.group)), std::string::npos)
      << describe;
  const std::size_t groups = run.lines / run.members;
  EXPECT_EQ(p.transforms_of(fftn::team{.rank = 0, .size = 1}).count, groups);
  EXPECT_EQ(p.transform_count(), groups);

  callback_state st;
  st.members = run.members;
  st.bins = run.bins;
  st.lines = run.lines;
  st.inner = run.inner;
  std::vector<std::atomic<int>> seen(run.lines * run.bins);
  st.seen = &seen;
  epilogue_state es;
  const auto fresh = [&] { return run.init; };
  const auto in_of = [&](std::vector<Out>& y, std::vector<In>& xs) -> const In* {
    if constexpr (std::is_same_v<In, Out>) {
      return gc.in_place ? y.data() : xs.data();
    } else {
      return xs.data();
    }
  };
  fftn::workspace ws(p, 3);
  std::vector<In> xs = run.x;
  std::vector<Out> serial = fresh();
  es.line_starts.clear();
  for (std::size_t line = 0; line < run.lines; ++line) {
    es.line_starts.push_back(serial.data() + run.at(0, line));
  }
  p.execute(in_of(serial, xs), serial.data(), ws,
            run.args(&st, gc.epilogue_callback ? &es : nullptr));
  EXPECT_TRUE(st.ok) << "a middle block's geometry";
  EXPECT_TRUE(es.ok) << "an epilogue block's line";
  for (std::size_t line = 0; line < run.lines; ++line) {
    for (std::size_t j = 0; j < run.bins; ++j) {
      const int want = line / run.inner % run.members == 0 ? 1 : 0;
      if (seen[line * run.bins + j].load() != want) {
        ADD_FAILURE() << "group of line " << line << ", bin " << j << " passed "
                      << seen[line * run.bins + j].load() << " times";
        line = run.lines;
        break;
      }
    }
  }
  if (!gc.in_place) {
    EXPECT_TRUE(same_bits(xs, run.x)) << "input written";
  }
  const std::vector<dvec> want = run.reference();
  double diff = 0;
  double norm = 0;
  std::vector<bool> written(serial.size(), false);
  for (std::size_t line = 0; line < run.lines; ++line) {
    for (std::size_t j = 0; j < run.side; ++j) {
      const std::size_t a = run.at(j, line);
      written[a] = true;
      std::complex<double> got{};
      if constexpr (std::is_floating_point_v<Out>) {
        got = {static_cast<double>(serial[a]), 0.0};
      } else {
        got = std::complex<double>(serial[a]);
      }
      diff += std::norm(got - want[line][j]);
      norm += std::norm(want[line][j]);
    }
  }
  const auto tol = fftn::test::derived_tolerance<T>(gc.n, fftn::test::fftn_bound(gc.n)) * 8.0;
  EXPECT_LE(std::sqrt(diff / norm), tol.rms) << describe;
  if (!gc.in_place) {
    std::size_t touched = 0;
    for (std::size_t i = 0; i < serial.size(); ++i) {
      if (!written[i] && !same_bits(std::vector<Out>{serial[i]}, std::vector<Out>{run.init[i]})) {
        ++touched;
      }
    }
    EXPECT_EQ(touched, 0U) << "elements outside the lines written";
  }

  st.seen = nullptr;
  const fftn::exec_args args = run.args(&st, nullptr);
  std::vector<Out> pieces = fresh();
  const std::size_t half = groups / 2;
  p.execute_transforms(0, half, in_of(pieces, xs), pieces.data(), ws, args);
  p.execute_transforms(half, groups - half, in_of(pieces, xs), pieces.data(), ws, args);
  EXPECT_TRUE(same_bits(pieces, serial)) << "pieces of the batch differ\n" << describe;
  if (!gc.threads) {
    return;
  }
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    std::vector<Out> teamed = fresh();
    std::vector<std::thread> threads;
    for (int r = 1; r < 3; ++r) {
      threads.emplace_back([&, r] {
        p.execute(in_of(teamed, xs), teamed.data(), fftn::team{.rank = r, .size = 3}, ws, done,
                  args);
      });
    }
    p.execute(in_of(teamed, xs), teamed.data(), fftn::team{.rank = 0, .size = 3}, ws, done, args);
    for (std::thread& m : threads) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << "a team differs\n" << describe;
  }
  const fftn::thread_pool pool(3);
  std::vector<Out> pooled = fresh();
  p.execute(in_of(pooled, xs), pooled.data(), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << "the executor differs\n" << describe;
  EXPECT_TRUE(st.ok);
}

namespace d = fftn::detail;

// The view the public class builds of a <C, C, C> description, for the construction knobs.
template <class T>
d::pipeline_view
ccc_view(const fftn::pipeline_desc<std::complex<T>, std::complex<T>, std::complex<T>>& desc) {
  return {
      .kind = d::pipeline_kind::complex_pair,
      .prec = sizeof(T) == 8 ? d::precision::f64 : d::precision::f32,
      .first = desc.first,
      .line = desc.line,
      .batches = desc.batches,
      .runs = desc.runs,
      .prologue = d::view_of(desc.prologue),
      .middle = {.scale = desc.middle.scale,
                 .kind = desc.middle.mul.kind,
                 .axis_strides = desc.middle.mul.axis_strides,
                 .batch_strides = desc.middle.mul.batch_strides,
                 .call = reinterpret_cast<d::erased_block_fn>(desc.middle.call.fn),
                 .call_mode = desc.middle.call.mode,
                 .call_scratch = desc.middle.call.scratch_bytes},
      .epilogue = d::view_of(desc.epilogue),
      .fanout = desc.fanout,
      .fanout_out_dist = desc.fanout_out_dist,
      .fanout_factor_dist = desc.fanout_factor_dist,
      .group = desc.group,
  };
}

// A group's members loops shared by a team: the schedule forced to split one task (every
// threshold at its least, a line per chunk, as pipeline_fanout_test.cpp's split copies), each
// member's line on a team member's own scratch with the owner's block, bitwise as serial
// execution, by teams (completion::all and own) and an executor.
template <class T> void run_split(const group_case& gc, bool one_task) {
  using Run = group_run<T, kind::ccc>;
  using C = std::complex<T>;
  SCOPED_TRACE(gc.name);
  d::build_settings s;
  s.chunk_points = 1;
  s.profile.inline_multiple = 0;
  s.profile.split_min_points = 1;
  s.profile.split_min_points_mesh = 1;
  s.profile.split_min_points_lines = 1;
  s.profile.remainder_split_min_points = 1;
  s.profile.min_points_per_member = 1;
  s.profile.sync_ns = 0;
  fftn::test::input_source src(0x6E0C0ULL + gc.n);
  Run run(gc, fftn::isa::best, src);
  const d::pipeline_view view = ccc_view<T>(run.desc);
  const auto core = d::make_pipeline_core(view, {.force_isa = run.target}, s);
  const std::string describe = d::describe(*core);
  const auto members = std::ranges::find_if(core->graph.phases,
                                            [](const d::graph_phase& ph) { return ph.nest != 0; });
  ASSERT_NE(members, core->graph.phases.end()) << describe;
  EXPECT_EQ(members->chunks, run.members) << describe;
  const d::lane_map lanes{.lanes = 3};
  const d::schedule plan = d::make_schedule(core->graph, lanes, core->profile);
  // One task (one group) for three members: the team must split it to share it at all.
  EXPECT_TRUE(!one_task || d::share_of(plan, lanes, 0).split != 0U)
      << "the team does not split the group\n"
      << describe;
  callback_state st;
  st.members = run.members;
  st.bins = run.bins;
  st.lines = run.lines;
  st.inner = run.inner;
  const fftn::exec_args args = run.args(&st, nullptr);
  fftn::workspace ws(d::workspace_bytes(*core, 3));
  std::vector<C> xs = run.x;
  std::vector<C> serial = run.init;
  d::execute_all(*core, xs.data(), serial.data(), ws, args);
  for (const fftn::completion done : {fftn::completion::all, fftn::completion::own}) {
    std::vector<C> teamed = run.init;
    std::vector<std::thread> threads;
    for (int r = 1; r < 3; ++r) {
      threads.emplace_back([&, r] {
        d::execute_team(*core, xs.data(), teamed.data(), fftn::team{.rank = r, .size = 3}, ws, done,
                        args);
      });
    }
    d::execute_team(*core, xs.data(), teamed.data(), fftn::team{.rank = 0, .size = 3}, ws, done,
                    args);
    for (std::thread& m : threads) {
      m.join();
    }
    EXPECT_TRUE(same_bits(teamed, serial)) << "a team differs\n" << describe;
  }
  const fftn::thread_pool pool(3);
  std::vector<C> pooled = run.init;
  d::execute_on(*core, xs.data(), pooled.data(), pool, ws, args);
  EXPECT_TRUE(same_bits(pooled, serial)) << "the executor differs\n" << describe;
  EXPECT_TRUE(st.ok);
}

std::vector<group_case> complex_cases(std::size_t n) {
  return {
      {.name = "polarisation outermost", .n = n, .threads = true},
      {.name = "polarisation innermost, a fused kernel per line",
       .n = n,
       .counts = {5, 2},
       .group = 1,
       .mid = factor_kind::complex,
       .per_line = true,
       .threads = true},
      {.name = "three members between two dimensions, a real kernel per line, scaled",
       .n = n,
       .counts = {2, 3, 2},
       .group = 1,
       .mid = factor_kind::real,
       .per_line = true,
       .pro_scale = 2.0,
       .threads = true},
      {.name = "backward first, a prologue factor, an epilogue callback",
       .n = n,
       .backward = true,
       .prologue_factor = true,
       .epilogue_callback = true},
      {.name = "strided lines, a shared kernel",
       .n = n,
       .counts = {3, 2},
       .group = 1,
       .mid = factor_kind::complex,
       .stride = 2},
      {.name = "in place", .n = n, .counts = {2, 4}, .in_place = true},
      {.name = "only the group: one field, two polarisations",
       .n = n,
       .counts = {2},
       .threads = true},
      {.name = "side factors per line, options.scale",
       .n = n,
       .counts = {3, 2},
       .group = 1,
       .mid = factor_kind::real,
       .side_factors_per_line = true,
       .options_scale = 0.5,
       .threads = true},
  };
}

std::vector<group_case> real_cases(std::size_t n) {
  return {
      {.name = "polarisation outermost", .n = n, .threads = true},
      {.name = "innermost, a real kernel per line, an epilogue callback",
       .n = n,
       .counts = {4, 2},
       .group = 1,
       .mid = factor_kind::real,
       .per_line = true,
       .epilogue_callback = true,
       .threads = true},
      {.name = "only the group, side factors per line, options.scale",
       .n = n,
       .counts = {2},
       .side_factors_per_line = true,
       .options_scale = 2.0},
  };
}

template <class T> class PipelineGroups : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(PipelineGroups, precisions);

TYPED_TEST(PipelineGroups, ComplexPairs) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : {1U, 5U, 16U, 60U, 128U}) {
      for (const group_case& gc : complex_cases(n)) {
        run_case<TypeParam, kind::ccc>(gc, t);
      }
    }
  }
}

TYPED_TEST(PipelineGroups, RealPairs) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    for (const std::size_t n : {4U, 9U, 32U, 100U}) {
      for (const group_case& gc : real_cases(n)) {
        run_case<TypeParam, kind::tct>(gc, t);
        run_case<TypeParam, kind::ctc>(gc, t);
      }
    }
  }
}

TYPED_TEST(PipelineGroups, MembersSplitAmongATeam) {
  for (const std::size_t n : {16U, 64U}) {
    run_split<TypeParam>({.name = "one group of six split",
                          .n = n,
                          .counts = {6},
                          .mid = factor_kind::complex,
                          .per_line = true},
                         true);
    run_split<TypeParam>({.name = "two groups of three, side factors",
                          .n = n,
                          .counts = {3, 2},
                          .group = 0,
                          .side_factors_per_line = true},
                         false);
  }
}

void noop(void*, const fftn::block_info&, std::complex<double>*) noexcept {}

template <class D> void expect_refused(const D& desc, const std::string& why) {
  try {
    const fftn::pipeline<std::complex<double>, std::complex<double>, std::complex<double>> p(desc);
    ADD_FAILURE() << "not refused: " << why;
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("desc.group"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find(why), std::string::npos) << e.what();
  }
}

TEST(PipelineGroups, RefusalsNameTheField) {
  auto desc = fftn::complex_convolution_rows<double>(8, 4);
  desc.middle.call = {.fn = &noop};
  desc.group = 1;
  expect_refused(desc, "not a dimension of desc.batches");
  desc.group = fftn::no_group - 1;
  expect_refused(desc, "not a dimension of desc.batches");
  desc.group = 0;
  desc.middle.call = {};
  expect_refused(desc, "without desc.middle.call");
  desc.middle.call = {.fn = &noop};
  desc.fanout = 2;
  desc.fanout_out_dist = 32;
  expect_refused(desc, "desc.fanout above 1");
  desc.fanout = 1;
  const fftn::pipeline<std::complex<double>, std::complex<double>, std::complex<double>> ok(desc);
  EXPECT_EQ(ok.transforms_of(fftn::team{.rank = 0, .size = 1}).count, 1U);
  EXPECT_NE(ok.describe().find("in groups of 4"), std::string::npos) << ok.describe();
  // The forced lanes and depth-first pairs give way to the chain pair, as describe() says.
  for (const d::strategy_choice forced : {d::strategy_choice::lanes, d::strategy_choice::depth}) {
    d::build_settings s;
    s.strategy = forced;
    s.depth_tops = {2};
    const std::string text = d::describe(*d::make_pipeline_core(ccc_view<double>(desc), {}, s));
    EXPECT_NE(text.find(std::string(forced == d::strategy_choice::lanes ? "the lanes pair"
                                                                        : "the depth-first pair") +
                        " (forced) does not take a group: the chain pair instead"),
              std::string::npos)
        << text;
  }
}

} // namespace
