// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Split-complex data (fftn/split.hpp, DESIGN.md 5.8; task 6.10): the c2c, r2c and c2r kinds of
// rows, planes, volumes and strided axes on split arrays, against the separable
// quadruple-precision reference (rounded to double after each axis) within the derived
// tolerance; bitwise equal to the interleaved plan of the same program applied to the input
// converted to the interleaved form; the two arrays of a side at any relative position (apart,
// interleaved in one buffer, the imaginary parts first), strides and distances of both signs
// with gaps, in place; canaries around and between both arrays and in their gaps, the input left
// as it was; serial, pieces of the batch, teams and an executor bitwise alike; describe()'s two
// passes; and the refusals.

#include "core/plan_core.hpp"
#include "fftn/detail/core.hpp"
#include "reference/reference_dft.hpp"
#include "support/exact.hpp"
#include "support/fftn_subject.hpp"
#include "support/inputs.hpp"
#include "support/outcome.hpp"
#include "support/precisions.hpp"
#include "support/tolerance.hpp"

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cd = std::complex<double>;
using dims_t = std::vector<std::size_t>;
using strides_t = std::vector<std::ptrdiff_t>;
using fftn::detail::transform_kind;

std::size_t product(const dims_t& d) {
  std::size_t p = 1;
  for (const std::size_t n : d) {
    p *= n;
  }
  return p;
}

// The unnormalised DFT of one C-ordered array, axis by axis, rounded to double after each axis
// (as tests/api/multidim_test.cpp evaluates it).
std::vector<cd> dft(int sign, const dims_t& dims, std::vector<cd> x) {
  std::size_t inner = x.size();
  for (std::size_t a = 0; a < dims.size(); ++a) {
    const std::size_t n = dims[a];
    inner /= n;
    const std::size_t outer = x.size() / (n * inner);
    std::vector<cd> line(n);
    for (std::size_t o = 0; o < outer; ++o) {
      for (std::size_t i = 0; i < inner; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          line[j] = x[(o * n + j) * inner + i];
        }
        const auto y = fftn::test::reference::c2c(sign, line);
        for (std::size_t k = 0; k < n; ++k) {
          x[(o * n + k) * inner + i] = {y[k].re.hi, y[k].im.hi};
        }
      }
    }
  }
  return x;
}

// The tolerance against dft(): the transform's derived tolerance over all its points, an input
// rounding to double per axis for the expected values, and for c2r the spectrum's rounding to T.
template <class T> fftn::test::tolerance tolerance_for(const dims_t& dims, bool c2r) {
  fftn::test::tolerance t =
      fftn::test::derived_tolerance<T>(product(dims), fftn::test::fftn_bound(product(dims))) +
      fftn::test::input_rounding<double>() * static_cast<double>(dims.size());
  if (c2r) {
    t = t + fftn::test::input_rounding<T>();
  }
  return t;
}

// One side's layout, in its own elements (reals on a split or real side): per axis in C order a
// stride, and the transforms `dist` apart.
struct layout {
  dims_t ext;
  strides_t stride;
  std::ptrdiff_t dist = 0;
  std::size_t count = 1;

  // The offset of every element from element 0 of transform 0, transform after transform, each
  // in C order.
  [[nodiscard]] std::vector<std::ptrdiff_t> offsets() const {
    std::vector<std::ptrdiff_t> o;
    const std::size_t points = product(ext);
    for (std::size_t b = 0; b < count; ++b) {
      for (std::size_t i = 0; i < points; ++i) {
        std::size_t rest = i;
        std::ptrdiff_t at = static_cast<std::ptrdiff_t>(b) * dist;
        for (std::size_t d = ext.size(); d > 0; --d) {
          at += static_cast<std::ptrdiff_t>(rest % ext[d - 1]) * stride[d - 1];
          rest /= ext[d - 1];
        }
        o.push_back(at);
      }
    }
    return o;
  }
};

// Which strides of a layout are negative: the last axis's, the first's (of planes and volumes)
// and the batch distance's.
enum flip : unsigned { flip_last = 1, flip_first = 2, flip_batch = 4 };

// C order with `gap` elements after each row and each transform, every stride and the distance
// times `spread` (2: the side's two arrays interleaved in one buffer), negated as `flips` says.
layout make_layout(const dims_t& ext, std::size_t count, std::ptrdiff_t gap, std::ptrdiff_t spread,
                   unsigned flips) {
  layout l{.ext = ext, .stride = strides_t(ext.size()), .dist = 0, .count = count};
  std::ptrdiff_t span = 1;
  for (std::size_t i = ext.size(); i > 0; --i) {
    const std::size_t d = i - 1;
    l.stride[d] = span;
    span *= static_cast<std::ptrdiff_t>(ext[d]);
    if (d + 1 == ext.size()) {
      span += gap;
    }
  }
  l.dist = span + gap;
  for (std::ptrdiff_t& s : l.stride) {
    s *= spread;
  }
  l.dist *= spread;
  if ((flips & flip_last) != 0) {
    l.stride.back() = -l.stride.back();
  }
  if ((flips & flip_first) != 0 && ext.size() > 1) {
    l.stride.front() = -l.stride.front();
  }
  if ((flips & flip_batch) != 0) {
    l.dist = -l.dist;
  }
  return l;
}

// Where a split side's two arrays lie: in two regions of one buffer, apart; interleaved, the
// imaginary parts one real after the real parts; or the imaginary parts one real before.
enum class placement : unsigned char { apart, interleaved, swapped };

constexpr std::ptrdiff_t margin = 16; // reals of canary before, between and after the arrays

// The memory of one side: a buffer of reals holding its arrays (one for a real side), filled with
// a canary everywhere the layout does not describe.
template <class T> struct side_memory {
  std::vector<T> buffer;
  std::ptrdiff_t re = 0; // element 0 of the real parts (or of the real side)
  std::ptrdiff_t im = -1;
  std::vector<std::ptrdiff_t> offsets;

  static T canary() { return static_cast<T>(-1234.5625); }

  side_memory(const layout& l, bool split, placement where) : offsets(l.offsets()) {
    const auto [lo, hi] = std::ranges::minmax(offsets);
    const std::ptrdiff_t span = hi - lo + 1;
    re = margin - lo;
    if (split && where == placement::apart) {
      im = re + span + margin;
    } else if (split && where == placement::interleaved) {
      im = re + 1;
    } else if (split) {
      im = re;
      re = im + 1;
    }
    buffer.assign(static_cast<std::size_t>(2 * span + 4 * margin), canary());
  }

  T* real_parts() { return buffer.data() + re; }
  T* imaginary_parts() { return buffer.data() + im; }

  void put(const std::vector<std::complex<T>>& v) {
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      buffer[static_cast<std::size_t>(re + offsets[i])] = v[i].real();
      if (im >= 0) {
        buffer[static_cast<std::size_t>(im + offsets[i])] = v[i].imag();
      }
    }
  }
  [[nodiscard]] std::vector<std::complex<T>> get() const {
    std::vector<std::complex<T>> v;
    for (const std::ptrdiff_t o : offsets) {
      v.emplace_back(buffer[static_cast<std::size_t>(re + o)],
                     im >= 0 ? buffer[static_cast<std::size_t>(im + o)] : T{});
    }
    return v;
  }
  // Whether every real the layout does not describe still holds the canary.
  [[nodiscard]] bool canaries_intact() const {
    std::vector<bool> described(buffer.size(), false);
    for (const std::ptrdiff_t o : offsets) {
      described[static_cast<std::size_t>(re + o)] = true;
      if (im >= 0) {
        described[static_cast<std::size_t>(im + o)] = true;
      }
    }
    const T c = canary();
    for (std::size_t i = 0; i < buffer.size(); ++i) {
      if (!described[i] && std::memcmp(&buffer[i], &c, sizeof(T)) != 0) {
        return false;
      }
    }
    return true;
  }
};

template <class T> bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// A case: a kind, its logical lengths (C order, the last the halved one of a real kind), the
// batch, and the layouts of its sides.
struct split_case {
  const char* name = "";
  transform_kind kind = transform_kind::c2c;
  dims_t dims;
  std::size_t count = 1;
  fftn::direction dir = fftn::direction::forward;
  std::ptrdiff_t gap = 0;
  unsigned flips = 0;
  placement where = placement::apart;
  bool in_place = false; // c2c only
  bool reference = true; // against the quadruple-precision reference (small cases)
};

dims_t side_dims(const split_case& c, bool output) {
  dims_t d = c.dims;
  if (output ? c.kind == transform_kind::r2c : c.kind == transform_kind::c2r) {
    d.back() = d.back() / 2 + 1;
  }
  return d;
}

bool split_side(const split_case& c, bool output) {
  return output ? c.kind != transform_kind::c2r : c.kind != transform_kind::r2c;
}

// The layout of a side: split sides spread by 2 when their arrays share one buffer interleaved.
layout side_layout(const split_case& c, bool output) {
  const bool spread = split_side(c, output) && c.where != placement::apart;
  return make_layout(side_dims(c, output), c.count, c.gap, spread ? 2 : 1, c.flips);
}

template <class In, class Out> fftn::shape<In, Out> shape_of(const split_case& c) {
  const layout in = side_layout(c, false);
  const layout out = side_layout(c, true);
  fftn::shape<In, Out> s;
  s.dir = c.dir;
  for (std::size_t d = 0; d < c.dims.size(); ++d) {
    s.axes.push_back({.n = c.dims[d], .in_stride = in.stride[d], .out_stride = out.stride[d]});
  }
  s.batches.push_back({.count = c.count, .in_dist = in.dist, .out_dist = out.dist});
  return s;
}

// The logical input of a case, transform after transform in C order (complex values; real ones
// with zero imaginary parts), and the expected output from the reference (empty unless asked).
template <class T> struct case_data {
  std::vector<std::complex<T>> input;
  std::vector<cd> want;
};

template <class T> case_data<T> data_of(const split_case& c) {
  const std::size_t points = product(c.dims);
  fftn::test::input_source src(points * 7 + c.count);
  case_data<T> d;
  const auto split_rows = [&](const std::vector<cd>& full, std::size_t t, std::vector<cd>& to) {
    const std::size_t n = c.dims.back();
    for (std::size_t l = 0; l < points / n; ++l) {
      for (std::size_t k = 0; k <= n / 2; ++k) {
        to.push_back(full[t * points + l * n + k]);
      }
    }
  };
  if (c.kind == transform_kind::c2c) {
    d.input = src.complexes<T>(points * c.count);
    if (c.reference) {
      for (std::size_t t = 0; t < c.count; ++t) {
        const std::vector<cd> one(d.input.begin() + static_cast<std::ptrdiff_t>(t * points),
                                  d.input.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
        const auto y = dft(static_cast<int>(c.dir), c.dims, one);
        d.want.insert(d.want.end(), y.begin(), y.end());
      }
    }
    return d;
  }
  const auto x = src.reals<T>(points * c.count);
  if (c.kind == transform_kind::r2c) {
    d.input.assign(x.begin(), x.end());
    if (c.reference) {
      std::vector<cd> full;
      for (std::size_t t = 0; t < c.count; ++t) {
        const std::vector<cd> one(x.begin() + static_cast<std::ptrdiff_t>(t * points),
                                  x.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
        const auto y = dft(-1, c.dims, one);
        full.insert(full.end(), y.begin(), y.end());
      }
      for (std::size_t t = 0; t < c.count; ++t) {
        split_rows(full, t, d.want);
      }
    }
    return d;
  }
  if (!c.reference) {
    // c2r without the reference: any half spectrum, for the bitwise comparisons.
    const std::size_t n = c.dims.back();
    d.input = src.complexes<T>(points / n * (n / 2 + 1) * c.count);
    return d;
  }
  // c2r: the half spectrum of real data rounded to T, so that the result is points times it.
  std::vector<cd> full;
  for (std::size_t t = 0; t < c.count; ++t) {
    const std::vector<cd> one(x.begin() + static_cast<std::ptrdiff_t>(t * points),
                              x.begin() + static_cast<std::ptrdiff_t>((t + 1) * points));
    const auto y = dft(-1, c.dims, one);
    full.insert(full.end(), y.begin(), y.end());
  }
  std::vector<cd> half;
  for (std::size_t t = 0; t < c.count; ++t) {
    split_rows(full, t, half);
  }
  for (const cd& z : half) {
    d.input.emplace_back(static_cast<T>(z.real()), static_cast<T>(z.imag()));
  }
  if (c.reference) {
    for (const T v : x) {
      d.want.emplace_back(static_cast<double>(v) * static_cast<double>(points), 0.0);
    }
  }
  return d;
}

// Checks one result; the worst error/tolerance ratio of the test is printed at its end.
class checker {
public:
  checker() = default;
  checker(const checker&) = delete;
  checker& operator=(const checker&) = delete;
  checker(checker&&) = delete;
  checker& operator=(checker&&) = delete;
  ~checker() {
    std::cout << "[ worst    ] error/tolerance " << o_.worst_ratio() << " (" << worst_at_ << ")\n";
  }

  template <class T>
  void within(std::span<const std::complex<T>> got, std::span<const cd> want,
              const fftn::test::tolerance& tol, const std::string& what) {
    std::vector<fftn::test::exact_complex> e;
    for (const cd& z : want) {
      e.push_back(fftn::test::exact_from(z));
    }
    fftn::test::outcome o;
    o.within(fftn::test::compare<T>(got, e), tol, what);
    EXPECT_TRUE(o.passed()) << o.detail();
    if (o.worst_ratio() > o_.worst_ratio()) {
      worst_at_ = what;
    }
    o_.merge(o);
  }

private:
  fftn::test::outcome o_;
  std::string worst_at_ = "-";
};

// The arrays of one execution as the split plan takes them.
template <class T> struct split_io {
  side_memory<T> in;
  side_memory<T> out;
  bool in_place = false;

  split_io(const split_case& c, const std::vector<std::complex<T>>& x)
      : in(side_layout(c, false), split_side(c, false), c.where),
        out(side_layout(c, true), split_side(c, true), c.where), in_place(c.in_place) {
    in.put(x);
  }
  side_memory<T>& result() { return in_place ? in : out; }
};

// The core's arguments for a side: the slot's array and the cursor's (fftn/detail/core.hpp).
template <class T> const void* slot_of(side_memory<T>& m) { return m.real_parts(); }
template <class T> const void* cursor_of(side_memory<T>& m) {
  return m.im >= 0 ? m.imaginary_parts() : nullptr;
}

// Runs a case's split core serially (or, `pieces`, in three calls of execute_transforms), on io.
template <class T>
void run_serial(const fftn::detail::plan_core& core, split_io<T>& io, const fftn::workspace& ws,
                bool pieces = false) {
  side_memory<T>& to = io.in_place ? io.in : io.out;
  const fftn::exec_args args{.in_factor = cursor_of(io.in), .out_factor = cursor_of(to)};
  void* out = to.real_parts();
  if (!pieces) {
    fftn::detail::execute_all(core, slot_of(io.in), out, ws, args);
    return;
  }
  const std::size_t t = fftn::detail::transform_count(core);
  const std::size_t a = t / 3;
  const std::size_t b = (2 * t) / 3;
  fftn::detail::execute_core(core, b, t - b, slot_of(io.in), out, ws, args);
  fftn::detail::execute_core(core, 0, a, slot_of(io.in), out, ws, args);
  fftn::detail::execute_core(core, a, b - a, slot_of(io.in), out, ws, args);
}

template <class T> fftn::shape<std::complex<T>, std::complex<T>> c2c_of(const split_case& c) {
  return shape_of<std::complex<T>, std::complex<T>>(c);
}

// The interleaved plan of the same logical shape, built with `settings`, applied to the case's
// input converted to the interleaved form: the result in logical order. Its complex sides are
// tight (in place for c2c: the split plan's transform runs in place in the workspace block), its
// real side (r2c's input, c2r's output) the case's own layout, which the split plan's transform
// reads or writes itself.
template <class T>
std::vector<std::complex<T>>
interleaved_result(const split_case& c, const std::vector<std::complex<T>>& x, fftn::options opts,
                   const fftn::detail::build_settings& settings) {
  const bool real_in = c.kind == transform_kind::r2c;
  const bool real_out = c.kind == transform_kind::c2r;
  const layout in =
      real_in ? side_layout(c, false) : make_layout(side_dims(c, false), c.count, 0, 1, 0);
  const layout out =
      real_out ? side_layout(c, true) : make_layout(side_dims(c, true), c.count, 0, 1, 0);
  std::vector<fftn::axis> axes;
  for (std::size_t d = 0; d < c.dims.size(); ++d) {
    axes.push_back({.n = c.dims[d], .in_stride = in.stride[d], .out_stride = out.stride[d]});
  }
  const std::vector<fftn::batch> batches{
      {.count = c.count, .in_dist = in.dist, .out_dist = out.dist}};
  fftn::detail::shape_view v;
  v.kind = c.kind;
  v.prec = std::is_same_v<T, double> ? fftn::detail::precision::f64 : fftn::detail::precision::f32;
  v.dir = c.kind == transform_kind::c2c ? c.dir : fftn::direction::forward;
  v.axes = axes;
  v.batches = batches;
  opts.in_place = c.kind == transform_kind::c2c;
  const auto core = fftn::detail::make_plan_core(v, opts, settings);
  fftn::workspace ws(fftn::detail::workspace_bytes(*core, 1));
  if (c.kind == transform_kind::c2c) {
    std::vector<std::complex<T>> y = x;
    fftn::detail::execute_all(*core, y.data(), y.data(), ws);
    return y;
  }
  if (real_in) {
    side_memory<T> r(in, false, placement::apart);
    r.put(x);
    std::vector<std::complex<T>> y(out.offsets().size());
    fftn::detail::execute_all(*core, r.real_parts(), y.data(), ws);
    return y;
  }
  side_memory<T> r(out, false, placement::apart);
  fftn::detail::execute_all(*core, x.data(), r.real_parts(), ws);
  return r.get();
}

template <class T> class Split : public ::testing::Test {};
TYPED_TEST_SUITE(Split, fftn::test::precisions, fftn::test::precision_names);

// Every kind and rank, layouts of both signs, with gaps, each placement of the two arrays, in
// place; against the reference, the interleaved plan bitwise, canaries, the input preserved,
// and the executions bitwise alike.
std::vector<split_case> cases() {
  using enum transform_kind;
  constexpr auto fw = fftn::direction::forward;
  constexpr auto bw = fftn::direction::backward;
  return {
      {.name = "c2c rows", .kind = c2c, .dims = {64}, .count = 5},
      {.name = "c2c rows backward, apart, gaps, reversed",
       .kind = c2c,
       .dims = {30},
       .count = 4,
       .dir = bw,
       .gap = 3,
       .flips = flip_last | flip_batch},
      {.name = "c2c rows interleaved",
       .kind = c2c,
       .dims = {17},
       .count = 3,
       .where = placement::interleaved},
      {.name = "c2c rows, imaginary parts first, in place",
       .kind = c2c,
       .dims = {48},
       .count = 6,
       .where = placement::swapped,
       .in_place = true},
      {.name = "c2c rows in place, apart, reversed",
       .kind = c2c,
       .dims = {100},
       .count = 2,
       .gap = 1,
       .flips = flip_last | flip_batch,
       .in_place = true},
      {.name = "c2c one row of a large prime", .kind = c2c, .dims = {1009}, .count = 1},
      {.name = "c2c planes", .kind = c2c, .dims = {16, 12}, .count = 3, .dir = fw},
      {.name = "c2c planes backward, reversed, gaps",
       .kind = c2c,
       .dims = {9, 20},
       .count = 2,
       .dir = bw,
       .gap = 2,
       .flips = flip_first | flip_last | flip_batch},
      {.name = "c2c planes in place, interleaved",
       .kind = c2c,
       .dims = {8, 10},
       .count = 3,
       .where = placement::interleaved,
       .in_place = true},
      {.name = "c2c volumes", .kind = c2c, .dims = {6, 8, 10}, .count = 2},
      {.name = "c2c volumes, imaginary parts first, reversed",
       .kind = c2c,
       .dims = {5, 4, 12},
       .count = 2,
       .gap = 1,
       .flips = flip_first | flip_batch,
       .where = placement::swapped},
      {.name = "r2c rows", .kind = r2c, .dims = {64}, .count = 4},
      {.name = "r2c rows odd, reversed, gaps",
       .kind = r2c,
       .dims = {45},
       .count = 3,
       .gap = 2,
       .flips = flip_last | flip_batch},
      {.name = "r2c planes interleaved",
       .kind = r2c,
       .dims = {12, 16},
       .count = 2,
       .where = placement::interleaved},
      {.name = "r2c volumes reversed",
       .kind = r2c,
       .dims = {4, 6, 10},
       .count = 2,
       .gap = 1,
       .flips = flip_first | flip_batch,
       .where = placement::swapped},
      {.name = "c2r rows", .kind = c2r, .dims = {64}, .count = 4},
      {.name = "c2r rows odd, reversed, gaps",
       .kind = c2r,
       .dims = {27},
       .count = 3,
       .gap = 1,
       .flips = flip_last | flip_batch,
       .where = placement::interleaved},
      {.name = "c2r planes", .kind = c2r, .dims = {10, 12}, .count = 2},
      {.name = "c2r volumes, imaginary parts first, reversed",
       .kind = c2r,
       .dims = {6, 4, 8},
       .count = 2,
       .gap = 2,
       .flips = flip_first | flip_last,
       .where = placement::swapped},
  };
}

TYPED_TEST(Split, EveryKindAndRank) {
  using T = TypeParam;
  checker check;
  fftn::detail::build_settings same_program;
  // Rows on the chain without batch leaves, and the c2r intermediate in the block, so that the
  // interleaved plan runs the program the split plan runs in the block (a split plan's rows take
  // the chain whatever the rules say, DESIGN.md 5.8).
  same_program.strategy = fftn::detail::strategy_choice::chain;
  same_program.batch_leaves = fftn::detail::batch_leaf_choice::never;
  same_program.c2r_packed = 0;
  const fftn::thread_pool pool(3);
  for (const split_case& c : cases()) {
    SCOPED_TRACE(c.name);
    const case_data<T> data = data_of<T>(c);
    fftn::detail::shape_view v;
    std::vector<fftn::axis> axes;
    std::vector<fftn::batch> batches;
    std::shared_ptr<const fftn::detail::plan_core> core;
    std::string describe;
    // The public plan (by the rules) against the reference.
    {
      split_io<T> io(c, data.input);
      const std::vector<T> before = io.in.buffer;
      const auto run = [&](const auto& p) {
        using P = std::remove_cvref_t<decltype(p)>;
        using S = fftn::split_complex<T>;
        fftn::workspace ws(p);
        if constexpr (std::is_same_v<P, fftn::plan<T, S>>) {
          p.execute(io.in.real_parts(), {io.out.real_parts(), io.out.imaginary_parts()}, ws);
        } else if constexpr (std::is_same_v<P, fftn::plan<S, T>>) {
          p.execute({io.in.real_parts(), io.in.imaginary_parts()}, io.out.real_parts(), ws);
        } else if (c.in_place) {
          p.execute({io.in.real_parts(), io.in.imaginary_parts()},
                    {io.in.real_parts(), io.in.imaginary_parts()}, ws);
        } else {
          p.execute({io.in.real_parts(), io.in.imaginary_parts()},
                    {io.out.real_parts(), io.out.imaginary_parts()}, ws);
        }
        describe = p.describe();
      };
      using S = fftn::split_complex<T>;
      const fftn::options opts{.in_place = c.in_place};
      if (c.kind == transform_kind::c2c) {
        run(fftn::plan<S, S>(shape_of<S, S>(c), opts));
      } else if (c.kind == transform_kind::r2c) {
        run(fftn::plan<T, S>(shape_of<T, S>(c), opts));
      } else {
        run(fftn::plan<S, T>(shape_of<S, T>(c), opts));
      }
      EXPECT_TRUE(io.in.canaries_intact()) << "the input's buffer outside its arrays";
      EXPECT_TRUE(io.out.canaries_intact()) << "the output's buffer outside its arrays";
      if (!c.in_place) {
        EXPECT_TRUE(same_bits(io.in.buffer, before)) << "the input changed";
      }
      EXPECT_NE(describe.find("on split-complex data"), std::string::npos) << describe;
      // A pass for each split side: two for c2c, one for a real kind.
      EXPECT_NE(describe.find(c.kind == transform_kind::c2c ? "they cost 2 separate passes"
                                                            : "they cost 1 separate pass"),
                std::string::npos)
          << describe;
      if (c.reference) {
        const std::vector<std::complex<T>> got = io.result().get();
        check.within<T>(got, data.want, tolerance_for<T>(c.dims, c.kind == transform_kind::c2r),
                        c.name);
      }
    }
    // The same program on interleaved data, bitwise: the split core and the interleaved plan
    // built with the same settings, on every instruction set the machine runs.
    {
      const auto s = c2c_of<T>(c); // the same axes and batches as the split shape's
      axes = s.axes;
      batches = s.batches;
    }
    for (const fftn::isa target : fftn::test::runnable_isas()) {
      SCOPED_TRACE(fftn::isa_name(target));
      const fftn::options opts{.in_place = c.in_place, .force_isa = target};
      v.kind = c.kind;
      v.prec =
          std::is_same_v<T, double> ? fftn::detail::precision::f64 : fftn::detail::precision::f32;
      v.dir = c.kind == transform_kind::c2c ? c.dir : fftn::direction::forward;
      v.axes = axes;
      v.batches = batches;
      v.split = true;
      core = fftn::detail::make_plan_core(v, opts, same_program);
      split_io<T> serial(c, data.input);
      fftn::workspace ws(fftn::detail::workspace_bytes(*core, 4));
      run_serial(*core, serial, ws);
      const std::vector<std::complex<T>> got = serial.result().get();
      const std::vector<std::complex<T>> want =
          interleaved_result<T>(c, data.input, opts, same_program);
      EXPECT_TRUE(same_bits(got, want)) << "not the interleaved plan's bits\n"
                                        << fftn::detail::describe(*core);
      // Pieces of the batch, a team of three and an executor: the serial bits.
      {
        split_io<T> pieces(c, data.input);
        run_serial(*core, pieces, ws, true);
        EXPECT_TRUE(same_bits(pieces.result().buffer, serial.result().buffer)) << "pieces differ";
      }
      {
        split_io<T> teamed(c, data.input);
        side_memory<T>& to = c.in_place ? teamed.in : teamed.out;
        const fftn::exec_args args{.in_factor = cursor_of(teamed.in), .out_factor = cursor_of(to)};
        std::vector<std::thread> members;
        for (int r = 1; r < 3; ++r) {
          members.emplace_back([&, r] {
            fftn::detail::execute_team(*core, slot_of(teamed.in), to.real_parts(),
                                       fftn::team{.rank = r, .size = 3}, ws, fftn::completion::all,
                                       args);
          });
        }
        fftn::detail::execute_team(*core, slot_of(teamed.in), to.real_parts(),
                                   fftn::team{.rank = 0, .size = 3}, ws, fftn::completion::all,
                                   args);
        for (std::thread& m : members) {
          m.join();
        }
        EXPECT_TRUE(same_bits(teamed.result().buffer, serial.result().buffer)) << "a team differs";
      }
      {
        split_io<T> pooled(c, data.input);
        side_memory<T>& to = c.in_place ? pooled.in : pooled.out;
        const fftn::exec_args args{.in_factor = cursor_of(pooled.in), .out_factor = cursor_of(to)};
        fftn::detail::execute_on(*core, slot_of(pooled.in), to.real_parts(), pool, ws, args);
        EXPECT_TRUE(same_bits(pooled.result().buffer, serial.result().buffer))
            << "the executor differs";
      }
    }
  }
}

// Large transforms that a team splits among its members (a long row's chain, a plane's and a
// volume's phases by lines), through the public plan: team, executor and pieces bitwise with
// the serial execution, which equals the interleaved plan's.
TYPED_TEST(Split, TeamsShareLargeTransforms) {
  using T = TypeParam;
  using S = fftn::split_complex<T>;
  using enum transform_kind;
  const std::vector<split_case> large{
      {.name = "c2c row of 65536", .kind = c2c, .dims = {65536}, .count = 2, .reference = false},
      {.name = "c2c planes of 256 x 256",
       .kind = c2c,
       .dims = {256, 256},
       .count = 2,
       .flips = flip_first,
       .reference = false},
      {.name = "c2c volumes of 32^3 in place",
       .kind = c2c,
       .dims = {32, 32, 32},
       .count = 2,
       .where = placement::interleaved,
       .in_place = true,
       .reference = false},
      {.name = "r2c volumes of 32 x 32 x 64",
       .kind = r2c,
       .dims = {32, 32, 64},
       .count = 2,
       .reference = false},
      {.name = "c2r planes of 128 x 512",
       .kind = c2r,
       .dims = {128, 512},
       .count = 2,
       .where = placement::swapped,
       .reference = false},
  };
  const fftn::thread_pool pool(4);
  for (const split_case& c : large) {
    SCOPED_TRACE(c.name);
    const case_data<T> data = data_of<T>(c);
    const auto run = [&](const auto& p) {
      split_io<T> serial(c, data.input);
      split_io<T> teamed(c, data.input);
      split_io<T> pooled(c, data.input);
      split_io<T> pieces(c, data.input);
      fftn::workspace ws(p, 4);
      const auto in = [&](split_io<T>& io) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(p)>, fftn::plan<T, S>>) {
          return static_cast<const T*>(io.in.real_parts());
        } else {
          return fftn::split_ptr<const T>{io.in.real_parts(), io.in.imaginary_parts()};
        }
      };
      const auto out = [&](split_io<T>& io) {
        side_memory<T>& m = io.result();
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(p)>, fftn::plan<S, T>>) {
          return m.real_parts();
        } else {
          return fftn::split_ptr<T>{m.real_parts(), m.imaginary_parts()};
        }
      };
      p.execute(in(serial), out(serial), ws);
      std::vector<std::thread> members;
      for (int r = 1; r < 4; ++r) {
        members.emplace_back(
            [&, r] { p.execute(in(teamed), out(teamed), fftn::team{.rank = r, .size = 4}, ws); });
      }
      p.execute(in(teamed), out(teamed), fftn::team{.rank = 0, .size = 4}, ws);
      for (std::thread& m : members) {
        m.join();
      }
      p.execute(in(pooled), out(pooled), pool, ws);
      const std::size_t t = 2;
      p.execute_transforms(1, t - 1, in(pieces), out(pieces), ws);
      p.execute_transforms(0, 1, in(pieces), out(pieces), ws);
      EXPECT_TRUE(same_bits(teamed.result().buffer, serial.result().buffer)) << "a team differs";
      EXPECT_TRUE(same_bits(pooled.result().buffer, serial.result().buffer))
          << "the executor differs";
      EXPECT_TRUE(same_bits(pieces.result().buffer, serial.result().buffer)) << "pieces differ";
      EXPECT_TRUE(serial.out.canaries_intact());
      EXPECT_TRUE(serial.in.canaries_intact());
    };
    const fftn::options opts{.in_place = c.in_place};
    if (c.kind == c2c) {
      run(fftn::plan<S, S>(shape_of<S, S>(c), opts));
    } else if (c.kind == r2c) {
      run(fftn::plan<T, S>(shape_of<T, S>(c), opts));
    } else {
      run(fftn::plan<S, T>(shape_of<S, T>(c), opts));
    }
  }
}

// Strided axes of an array (axes_c2c's shapes, their strides counting reals on the split side):
// the transform along some dimensions of a 3-D array, the others a batch.
TYPED_TEST(Split, AxesOfAnArray) {
  using T = TypeParam;
  using S = fftn::split_complex<T>;
  checker check;
  const std::vector<fftn::axis> array{{.n = 6}, {.n = 5}, {.n = 8}};
  const std::vector<std::vector<std::size_t>> selections{{0}, {1}, {0, 2}, {2, 0}, {0, 1}};
  const std::size_t points = 6 * 5 * 8;
  fftn::test::input_source src(99);
  const auto x = src.complexes<T>(points);
  for (const auto& selected : selections) {
    const auto interleaved = fftn::axes_c2c<T>(array, selected, fftn::direction::forward);
    fftn::shape<S, S> s;
    s.dir = interleaved.dir;
    s.axes = interleaved.axes;
    s.batches = interleaved.batches;
    const fftn::plan<S, S> p(s);
    fftn::workspace ws(p);
    std::vector<T> re(points);
    std::vector<T> im(points);
    std::vector<T> ore(points);
    std::vector<T> oim(points);
    for (std::size_t i = 0; i < points; ++i) {
      re[i] = x[i].real();
      im[i] = x[i].imag();
    }
    p.execute({re.data(), im.data()}, {ore.data(), oim.data()}, ws);
    std::vector<std::complex<T>> got(points);
    for (std::size_t i = 0; i < points; ++i) {
      got[i] = {ore[i], oim[i]};
    }
    // The reference: the DFT along the selected dimensions of the C-ordered array.
    std::vector<cd> want(x.begin(), x.end());
    const dims_t dims{6, 5, 8};
    for (const std::size_t d : selected) {
      const std::size_t n = dims[d];
      std::size_t inner = 1;
      for (std::size_t e = d + 1; e < 3; ++e) {
        inner *= dims[e];
      }
      const std::size_t outer = points / (n * inner);
      std::vector<cd> line(n);
      for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < inner; ++i) {
          for (std::size_t j = 0; j < n; ++j) {
            line[j] = want[(o * n + j) * inner + i];
          }
          const auto yl = fftn::test::reference::c2c(-1, line);
          for (std::size_t k = 0; k < n; ++k) {
            want[(o * n + k) * inner + i] = {yl[k].re.hi, yl[k].im.hi};
          }
        }
      }
    }
    dims_t sel;
    for (const std::size_t d : selected) {
      sel.push_back(dims[d]);
    }
    check.within<T>(got, want, tolerance_for<T>(sel, false), "axes of an array");
  }
}

// describe() names the conversion's two passes, and the strategy of the plan on interleaved
// data where the split plan takes another; measured planning gives way to the rules.
TEST(SplitPlan, Describe) {
  using S = fftn::split_complex<double>;
  fftn::shape<S, S> s;
  s.dir = fftn::direction::forward;
  s.axes = {{.n = 32}};
  s.batches = {{.count = 64}};
  const std::string text = fftn::plan<S, S>(s).describe();
  EXPECT_NE(text.find("fftn plan: c2c forward on split-complex data"), std::string::npos) << text;
  EXPECT_NE(text.find("split complex: the input read from its real and imaginary arrays into "
                      "the interleaved form"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("edge in: the real and imaginary arrays into the interleaved form"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("edge out: the result into the real and imaginary arrays"), std::string::npos)
      << text;
  // Forced lanes: the interleaved plan takes them, the split plan the chain (and says why).
  fftn::detail::build_settings lanes;
  lanes.strategy = fftn::detail::strategy_choice::lanes;
  const auto split = fftn::detail::make_plan_core(fftn::detail::view_of(s), {}, lanes);
  fftn::detail::shape_view plain = fftn::detail::view_of(s);
  plain.split = false;
  const auto interleaved = fftn::detail::make_plan_core(plain, {}, lanes);
  const std::string forced = fftn::detail::describe(*split);
  if (interleaved->strategy_name != split->strategy_name) {
    EXPECT_NE(forced.find("not the strategy the rules would choose: " + interleaved->strategy_name +
                          " without the split arrays"),
              std::string::npos)
        << forced;
    EXPECT_NE(forced.find("the conversion's passes leave each row in the workspace block"),
              std::string::npos)
        << forced;
  } else {
    EXPECT_EQ(forced.find("not the strategy"), std::string::npos) << forced;
  }
  const std::string measured =
      fftn::plan<S, S>(s, {.effort = fftn::planning::measure, .measure_seconds = 0.001}).describe();
  EXPECT_NE(measured.find("planning: by the rules: measured planning and wisdom take no "
                          "split-complex plans yet"),
            std::string::npos)
      << measured;
}

// The refusals, naming the field.
TEST(SplitPlan, Refusals) {
  using S = fftn::split_complex<double>;
  const auto message = [](const auto& build) {
    try {
      build();
    } catch (const std::invalid_argument& e) {
      return std::string(e.what());
    }
    return std::string("no exception");
  };
  fftn::shape<double, S> r2c;
  r2c.axes = {{.n = 16}};
  r2c.batches = {{.count = 2}};
  EXPECT_NE(message([&] {
              const fftn::plan<double, S> p(r2c, {.in_place = true});
            }).find("options.in_place"),
            std::string::npos);
  fftn::shape<S, double> c2r;
  c2r.dir = fftn::direction::backward;
  c2r.axes = {{.n = 16}};
  c2r.batches = {{.count = 2}};
  EXPECT_NE(message([&] {
              const fftn::plan<S, double> p(c2r, {.in_place = true});
            }).find("options.in_place"),
            std::string::npos);
  fftn::shape<S, S> runs;
  runs.axes = {{.n = 16}};
  runs.runs = {{.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = 4}};
  EXPECT_NE(message([&] { const fftn::plan<S, S> p(runs); }).find("shape.runs"), std::string::npos);
  // Edges are not offered (plan's constructor is constrained); the core refuses them too.
  fftn::shape<S, S> c2c;
  c2c.axes = {{.n = 16}};
  EXPECT_NE(message([&] {
              static_cast<void>(fftn::detail::make_plan_core_with_edges(
                  fftn::detail::view_of(c2c), {}, {}, fftn::detail::build_settings{}));
            }).find("edges"),
            std::string::npos);
  // The layout checks as for interleaved data: an overlapping output.
  fftn::shape<S, S> overlap;
  overlap.axes = {{.n = 16, .in_stride = 1, .out_stride = 1}};
  overlap.batches = {{.count = 2, .in_dist = 16, .out_dist = 8}};
  EXPECT_NE(message([&] { const fftn::plan<S, S> p(overlap); }).find("overlap"), std::string::npos);
  static_assert(!std::is_constructible_v<fftn::plan<S, S>, const fftn::shape<S, S>&,
                                         const fftn::options&, const fftn::plan_edges<S, S>&>);
}

} // namespace
