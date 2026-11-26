// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Owner-computes (DESIGN.md 12.4; task 5.5): with `completion::own`, member r of a team runs
// exactly the transforms `transforms_of(r)` deals it, whole, for plans with edges and callbacks
// and for row pipelines of every kind built, as `plan::transforms_of` and
// `pipeline::transforms_of` say. Two checks per object and team size:
//
// - The members arrive one after another on one thread (nothing in `own` waits for another
//   member): after each member's call, exactly the output elements of the transforms dealt to
//   it and to the members before it hold the serial result, bit for bit, and every other element
//   still holds its prior value. Accumulation, in place, fan-out and stored runs included.
// - The members run at once, each on its own thread: the output is bitwise the serial result,
//   and a block callback records the member that ran each transform (`block_info::member`; the
//   pipeline's middle callback per line, its epilogue's per line and copy), which must be the
//   owner `transforms_of` names. The deal's ranges are contiguous, in rank order, cover the
//   batch once and differ in length by at most one.
//
// Lengths, counts and team sizes are chosen so that some members own nothing and long rows
// that `completion::all` splits run whole. Expectations hold on every target: none assumes a
// target's strategy.

#include <fftn/fftn.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <class V> bool same_bits(const V& a, std::size_t i, const V& b, std::size_t j) {
  return std::ranges::equal(std::as_bytes(std::span(&a[i], 1)), std::as_bytes(std::span(&b[j], 1)));
}

// Deterministic values in [-1, 1).
class source {
public:
  explicit source(std::uint64_t seed) : state_(seed) {}
  double next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state_ >> 11U) * 0x1.0p-52 - 1.0;
  }
  template <class T> void fill(std::vector<T>& v) {
    for (T& e : v) {
      if constexpr (std::is_floating_point_v<T>) {
        e = static_cast<T>(next());
      } else {
        using R = typename T::value_type;
        const auto re = static_cast<R>(next());
        e = T(re, static_cast<R>(next()));
      }
    }
  }

private:
  std::uint64_t state_;
};

// Who ran what: one slot per transform (or per transform and copy), the member that ran it, and
// whether any slot was run by two members.
struct member_log {
  explicit member_log(std::size_t slots) : who(slots) { reset(); }
  void reset() {
    for (std::atomic<int>& w : who) {
      w.store(-1);
    }
    conflict.store(false);
    calls.store(0);
  }
  void record(std::size_t slot, int member) noexcept {
    if (slot >= who.size()) {
      conflict.store(true);
      return;
    }
    int seen = -1;
    if (!who[slot].compare_exchange_strong(seen, member) && seen != member) {
      conflict.store(true);
    }
    calls.fetch_add(1);
  }
  std::vector<std::atomic<int>> who;
  std::atomic<bool> conflict{false};
  std::atomic<std::size_t> calls{0};
};

// A `modify` callback that leaves its block alone and records the member running it: for every
// line of the block its transform (consecutive transforms when the lines are along transforms,
// else the block's one transform), times `copies` slots with the copy (an epilogue's fan-out).
struct recorder {
  member_log* log = nullptr;
  std::size_t transforms = 0;
  template <class T> void operator()(const fftn::block_info& info, T* /*data*/) const noexcept {
    const std::size_t lines = info.line_axis == fftn::along_transforms ? info.lines : 1;
    for (std::size_t k = 0; k < lines; ++k) {
      log->record((info.copy * transforms) + info.transform + k, info.member);
    }
  }
};

// The deal's own properties: contiguous ranges in rank order covering the batch once, differing
// by at most one transform.
void expect_deal_shape(const std::function<fftn::transform_range(const fftn::team&)>& of,
                       std::size_t transforms, int size, const std::string& what) {
  std::size_t next = 0;
  std::size_t shortest = transforms;
  std::size_t longest = 0;
  for (int r = 0; r < size; ++r) {
    const fftn::transform_range mine = of(fftn::team{.rank = r, .size = size});
    if (mine.count != 0) {
      EXPECT_EQ(mine.first, next) << what << ": rank " << r << " of " << size;
    }
    next += mine.count;
    if (std::cmp_less(r, transforms)) {
      shortest = std::min(shortest, mine.count);
    }
    longest = std::max(longest, mine.count);
  }
  EXPECT_EQ(next, transforms) << what << ": the deal of " << size << " members";
  EXPECT_LE(longest - std::min(shortest, longest), 1U) << what << ": the deal of " << size;
}

// One object (a plan or a pipeline) under the deal: `owner_of(i)` is the transform output element
// i belongs to (or `none` for elements outside every transform's output); `logs` the callbacks'
// logs with the transform of each slot (slot % transforms).
constexpr std::size_t none = static_cast<std::size_t>(-1);

template <class Obj, class In, class Out> struct deal_case {
  std::string name;
  const Obj* obj = nullptr;
  std::size_t transforms = 0;
  const std::vector<In>* x = nullptr;
  std::vector<Out> init; // the output's prior values (the input itself, in place)
  bool in_place = false;
  fftn::exec_args args{};
  std::function<std::size_t(std::size_t)> owner_of;
  std::vector<member_log*> logs;
};

template <class Obj, class In, class Out> void expect_deal(deal_case<Obj, In, Out>& c) {
  const Obj& obj = *c.obj;
  // Element 0 of the input: the output buffer itself in place.
  const auto in_of = [&](std::vector<Out>& y) -> const In* {
    if constexpr (std::is_same_v<In, Out>) {
      if (c.in_place) {
        return y.data();
      }
    }
    return c.x->data();
  };
  const auto reset_logs = [&] {
    for (member_log* log : c.logs) {
      log->reset();
    }
  };
  std::vector<Out> serial = c.init;
  {
    fftn::workspace ws(obj);
    obj.execute(in_of(serial), serial.data(), ws, c.args);
  }
  for (member_log* log : c.logs) {
    EXPECT_GT(log->calls.load(), 0U) << c.name << ": a callback never ran";
  }

  for (const int size : {2, 3, 5, static_cast<int>(c.transforms) + 2}) {
    const std::string what = c.name + ", a team of " + std::to_string(size);
    expect_deal_shape([&](const fftn::team& t) { return obj.transforms_of(t); }, c.transforms, size,
                      what);
    fftn::workspace ws(obj, size);

    // The members one after another: each writes exactly its own transforms.
    std::vector<Out> y = c.init;
    std::vector<bool> done(c.transforms, false);
    reset_logs();
    for (int r = 0; r < size; ++r) {
      const fftn::team me{.rank = r, .size = size};
      obj.execute(in_of(y), y.data(), me, ws, fftn::completion::own, c.args);
      const fftn::transform_range mine = obj.transforms_of(me);
      for (std::size_t t = mine.first; t < mine.first + mine.count; ++t) {
        done[t] = true;
      }
      std::size_t wrong = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        const std::size_t t = c.owner_of(i);
        const bool written = t != none && done[t];
        wrong += same_bits(y, i, written ? serial : c.init, i) ? 0U : 1U;
      }
      EXPECT_EQ(wrong, 0U) << what << ": after rank " << r << " (transforms " << mine.first << " + "
                           << mine.count << "), elements not as dealt";
    }
    const auto expect_logs = [&](const char* how) {
      for (member_log* log : c.logs) {
        EXPECT_FALSE(log->conflict.load()) << what << ", " << how << ": a slot run twice";
        for (std::size_t slot = 0; slot < log->who.size(); ++slot) {
          const std::size_t t = slot % c.transforms;
          int owner = -1;
          for (int r = 0; r < size; ++r) {
            const fftn::transform_range m = obj.transforms_of(fftn::team{.rank = r, .size = size});
            if (t >= m.first && t < m.first + m.count) {
              owner = r;
            }
          }
          EXPECT_EQ(log->who[slot].load(), owner)
              << what << ", " << how << ": transform " << t << " (slot " << slot << ")";
        }
      }
    };
    expect_logs("one after another");

    // The members at once, each on its own thread.
    std::vector<Out> z = c.init;
    reset_logs();
    std::vector<std::thread> members;
    members.reserve(static_cast<std::size_t>(size));
    for (int r = 1; r < size; ++r) {
      members.emplace_back([&, r] {
        obj.execute(in_of(z), z.data(), fftn::team{.rank = r, .size = size}, ws,
                    fftn::completion::own, c.args);
      });
    }
    obj.execute(in_of(z), z.data(), fftn::team{.rank = 0, .size = size}, ws, fftn::completion::own,
                c.args);
    for (std::thread& m : members) {
      m.join();
    }
    std::size_t differ = 0;
    for (std::size_t i = 0; i < z.size(); ++i) {
      differ += same_bits(z, i, serial, i) ? 0U : 1U;
    }
    EXPECT_EQ(differ, 0U) << what << ": the team differs from serial execution";
    expect_logs("at once");
  }
}

// Output elements of `count` transforms of `per` elements each, one after another.
std::function<std::size_t(std::size_t)> tight(std::size_t per, std::size_t count) {
  return [=](std::size_t i) { return i / per < count ? i / per : none; };
}

template <class T> class OwnerComputes : public ::testing::Test {};
using precisions = ::testing::Types<double, float>;
TYPED_TEST_SUITE(OwnerComputes, precisions);

// Plans with edges and callbacks: c2c rows (short rows by the rules' lanes or batch leaves, long
// rows a team would split), real rows, planes, volumes, in place, accumulation, one transform.
TYPED_TEST(OwnerComputes, PlansWithEdgesAndCallbacks) {
  using T = TypeParam;
  using C = std::complex<T>;
  using fftn::factor_kind;
  source src(0x55A1);

  struct rows_spec {
    const char* name;
    std::size_t n;
    std::size_t count;
    bool in_place;
    bool accumulate;
  };
  for (const rows_spec& rs : {
           rows_spec{"c2c rows of 64, a window and accumulation", 64, 10, false, true},
           rows_spec{"c2c rows of 16, many", 16, 37, false, false},
           rows_spec{"c2c rows of 16384, long", 16384, 3, false, false},
           rows_spec{"c2c rows of 32, in place", 32, 9, true, false},
           rows_spec{"c2c, one row", 128, 1, false, false},
       }) {
    member_log log(rs.count);
    recorder rec{.log = &log, .transforms = rs.count};
    fftn::plan_edges<C, C> edges{
        .in = {.mul = {.kind = factor_kind::real, .axis_strides = {1}, .batch_strides = {0}}},
        .out = {.scale = 0.5,
                .accumulate = rs.accumulate,
                .call = {.fn = fftn::callback_of<recorder, C>()}},
    };
    const fftn::c2c_plan<T> plan(fftn::rows_c2c<T>(rs.n, rs.count, fftn::direction::forward),
                                 {.in_place = rs.in_place}, edges);
    std::vector<T> window(rs.n);
    src.fill(window);
    std::vector<C> x(rs.n * rs.count);
    src.fill(x);
    deal_case<fftn::c2c_plan<T>, C, C> dc{
        .name = rs.name,
        .obj = &plan,
        .transforms = rs.count,
        .x = &x,
        .init = x,
        .in_place = rs.in_place,
        .args = {.in_factor = window.data(), .out_user = &rec},
        .owner_of = tight(rs.n, rs.count),
        .logs = {&log},
    };
    if (!rs.in_place) {
      src.fill(dc.init);
    }
    expect_deal(dc);
  }

  {
    // Real rows: a real input factor per row, the callback on the bins.
    const std::size_t n = 30;
    const std::size_t count = 9;
    member_log log(count);
    recorder rec{.log = &log, .transforms = count};
    const fftn::r2c_plan<T> plan(
        fftn::rows_r2c<T>(n, count), {},
        {.in = {.mul = {.kind = factor_kind::real, .axis_strides = {1}, .batch_strides = {n}}},
         .out = {.call = {.fn = fftn::callback_of<recorder, C>()}}});
    std::vector<T> f(n * count);
    src.fill(f);
    std::vector<T> x(n * count);
    src.fill(x);
    deal_case<fftn::r2c_plan<T>, T, C> dc{
        .name = "r2c rows of 30",
        .obj = &plan,
        .transforms = count,
        .x = &x,
        .init = std::vector<C>((n / 2 + 1) * count),
        .args = {.in_factor = f.data(), .out_user = &rec},
        .owner_of = tight(n / 2 + 1, count),
        .logs = {&log},
    };
    src.fill(dc.init);
    expect_deal(dc);
  }

  {
    // c2r planes: a scale on the input, the callback on the real output.
    const std::size_t ny = 6;
    const std::size_t nx = 8;
    const std::size_t count = 5;
    member_log log(count);
    recorder rec{.log = &log, .transforms = count};
    const fftn::c2r_plan<T> plan(
        fftn::planes_c2r<T>(ny, nx, count), {},
        {.in = {.scale = 0.25}, .out = {.call = {.fn = fftn::callback_of<recorder, T>()}}});
    std::vector<C> x(ny * (nx / 2 + 1) * count);
    src.fill(x);
    deal_case<fftn::c2r_plan<T>, C, T> dc{
        .name = "c2r planes of 6 x 8",
        .obj = &plan,
        .transforms = count,
        .x = &x,
        .init = std::vector<T>(ny * nx * count),
        .args = {.out_user = &rec},
        .owner_of = tight(ny * nx, count),
        .logs = {&log},
    };
    src.fill(dc.init);
    expect_deal(dc);
  }

  {
    // c2c volumes: the callback on the input, a complex factor on the output.
    const std::size_t count = 3;
    const std::size_t points = 4 * 6 * 8;
    member_log log(count);
    recorder rec{.log = &log, .transforms = count};
    const fftn::c2c_plan<T> plan(
        fftn::volumes_c2c<T>(4, 6, 8, count, fftn::direction::backward), {},
        {.in = {.call = {.fn = fftn::callback_of<recorder, C>()}},
         .out = {.mul = {.kind = factor_kind::complex, .axis_strides = {48, 8, 1}}}});
    std::vector<C> f(points);
    src.fill(f);
    std::vector<C> x(points * count);
    src.fill(x);
    deal_case<fftn::c2c_plan<T>, C, C> dc{
        .name = "c2c volumes of 4 x 6 x 8",
        .obj = &plan,
        .transforms = count,
        .x = &x,
        .init = std::vector<C>(points * count),
        .args = {.out_factor = f.data(), .in_user = &rec},
        .owner_of = tight(points, count),
        .logs = {&log},
    };
    src.fill(dc.init);
    expect_deal(dc);
  }
}

// Row pipelines of every kind built: <C, C, C> forward first with a kernel, a middle callback and
// an epilogue callback; spectral pairs inverse first; <T, C, T> with the valid half's runs;
// <C, T, C>; fan-out (the epilogue per copy); in place; a long line.
TYPED_TEST(OwnerComputes, Pipelines) {
  using T = TypeParam;
  using C = std::complex<T>;
  using fftn::factor_kind;
  source src(0x99E5);

  struct ccc_spec {
    const char* name;
    std::size_t n;
    std::size_t count;
    fftn::direction first;
    std::size_t fanout;
    bool in_place;
  };
  for (const ccc_spec& cs : {
           ccc_spec{"<C, C, C> rows of 64", 64, 10, fftn::direction::forward, 1, false},
           ccc_spec{"<C, C, C> inverse first", 48, 7, fftn::direction::backward, 1, false},
           ccc_spec{"<C, C, C> fan-out of 3", 32, 5, fftn::direction::forward, 3, false},
           ccc_spec{"<C, C, C> in place", 40, 6, fftn::direction::forward, 1, true},
           ccc_spec{"<C, C, C> long lines", 4096, 2, fftn::direction::forward, 1, false},
       }) {
    member_log mid_log(cs.count);
    member_log epi_log(cs.count * cs.fanout);
    recorder mid{.log = &mid_log, .transforms = cs.count};
    recorder epi{.log = &epi_log, .transforms = cs.count};
    fftn::pipeline_desc desc = fftn::spectral_pair_rows<T>(cs.n, cs.count, cs.first);
    desc.middle.mul = {.kind = factor_kind::complex, .axis_strides = {1}};
    desc.middle.scale = 1.0 / static_cast<double>(cs.n);
    desc.middle.call = {.fn = fftn::callback_of<recorder, C>()};
    desc.epilogue.call = {.fn = fftn::callback_of<recorder, C>()};
    desc.fanout = cs.fanout;
    if (cs.fanout > 1) {
      desc.fanout_factor_dist = static_cast<std::ptrdiff_t>(cs.n);
      desc.fanout_out_dist = static_cast<std::ptrdiff_t>(cs.n * cs.count);
    }
    const fftn::pipeline pipe(desc, {.in_place = cs.in_place});
    std::vector<C> kernel(cs.n * cs.fanout);
    src.fill(kernel);
    std::vector<C> x(cs.n * cs.count);
    src.fill(x);
    const std::size_t n = cs.n;
    const std::size_t count = cs.count;
    deal_case<fftn::pipeline<C, C, C>, C, C> dc{
        .name = cs.name,
        .obj = &pipe,
        .transforms = cs.count,
        .x = &x,
        .init = cs.in_place ? x : std::vector<C>(cs.n * cs.count * cs.fanout),
        .in_place = cs.in_place,
        .args = {.mid_factor = kernel.data(), .mid_user = &mid, .out_user = &epi},
        .owner_of = [=](std::size_t i) { return (i % (n * count)) / n; },
        .logs = {&mid_log, &epi_log},
    };
    if (!cs.in_place) {
      src.fill(dc.init);
    }
    expect_deal(dc);
  }

  {
    // <T, C, T>: the valid half of a zero-padded convolution (runs on both sides), the middle's
    // callback per line.
    const std::size_t m = 20;
    const std::size_t n = 2 * m;
    const std::size_t count = 6;
    member_log log(count);
    recorder rec{.log = &log, .transforms = count};
    const std::vector<fftn::stored_run> runs{
        {.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = m},
        {.axis = 0, .where = fftn::side::out, .logical = m / 2, .stored = 0, .count = m},
    };
    fftn::pipeline_desc desc = fftn::convolution_rows<T>(n, count, runs);
    desc.middle.scale = 1.0 / static_cast<double>(n);
    desc.middle.call = {.fn = fftn::callback_of<recorder, C>()};
    const fftn::pipeline pipe(desc);
    std::vector<C> kernel(n / 2 + 1);
    src.fill(kernel);
    std::vector<T> x(m * count);
    src.fill(x);
    deal_case<fftn::pipeline<T, C, T>, T, T> dc{
        .name = "<T, C, T> valid halves",
        .obj = &pipe,
        .transforms = count,
        .x = &x,
        .init = std::vector<T>(m * count),
        .args = {.mid_factor = kernel.data(), .mid_user = &rec},
        .owner_of = tight(m, count),
        .logs = {&log},
    };
    src.fill(dc.init);
    expect_deal(dc);
  }

  {
    // <C, T, C>: half spectra in and out, a real middle factor per line, the epilogue's callback.
    const std::size_t n = 32;
    const std::size_t h = n / 2 + 1;
    const std::size_t count = 5;
    member_log log(count);
    recorder rec{.log = &log, .transforms = count};
    fftn::pipeline_desc<C, T, C> desc;
    desc.line = {.n = n};
    desc.batches = {{.count = count}};
    desc.middle.mul = {.kind = factor_kind::real,
                       .axis_strides = {1},
                       .batch_strides = {static_cast<std::ptrdiff_t>(n)}};
    desc.epilogue.call = {.fn = fftn::callback_of<recorder, C>()};
    const fftn::pipeline pipe(desc);
    std::vector<T> f(n * count);
    src.fill(f);
    std::vector<C> x(h * count);
    src.fill(x);
    deal_case<fftn::pipeline<C, T, C>, C, C> dc{
        .name = "<C, T, C>",
        .obj = &pipe,
        .transforms = count,
        .x = &x,
        .init = std::vector<C>(h * count),
        .args = {.mid_factor = f.data(), .out_user = &rec},
        .owner_of = tight(h, count),
        .logs = {&log},
    };
    src.fill(dc.init);
    expect_deal(dc);
  }
}

} // namespace
