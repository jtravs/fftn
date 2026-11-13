// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Plan construction: validation, the default layouts, and the program of a transform of
// any rank up to three (DESIGN.md 9.3): each transform is finished before the next, one
// axis at a time, one line at a time, every line through the chain strategy. The layout
// rules are stated for users in docs/conventions.md; this file is their implementation.

#include "core/plan_core.hpp"

#include "core/driver.hpp"
#include "core/program.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/workspace.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "platform/cpu_features.hpp"
#include "strategies/chain.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

[[noreturn]] void reject(const std::string& argument, const std::string& why) {
  throw std::invalid_argument("fftn::plan: " + argument + " " + why);
}

std::string axis_name(std::size_t d) { return "shape.axes[" + std::to_string(d) + "]"; }
std::string batch_name(std::size_t b) { return "shape.batches[" + std::to_string(b) + "]"; }

std::ptrdiff_t checked_mul(std::ptrdiff_t a, std::ptrdiff_t b, const std::string& what) {
  std::ptrdiff_t r = 0;
  if (__builtin_mul_overflow(a, b, &r)) {
    reject(what, "makes the array too large to address");
  }
  return r;
}

std::ptrdiff_t checked_add(std::ptrdiff_t a, std::ptrdiff_t b, const std::string& what) {
  std::ptrdiff_t r = 0;
  if (__builtin_add_overflow(a, b, &r)) {
    reject(what, "makes the array too large to address");
  }
  return r;
}

std::ptrdiff_t to_signed(std::size_t v, const std::string& what) {
  if (v > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    reject(what, "is too large");
  }
  return static_cast<std::ptrdiff_t>(v);
}

// Only for strides and distances, which resolve() keeps above PTRDIFF_MIN.
std::ptrdiff_t magnitude(std::ptrdiff_t v) { return v < 0 ? -v : v; }

// A given stride or distance: its magnitude must be representable, so PTRDIFF_MIN is refused.
// Everything derived from it at construction or in the kernels (reaches, byte steps, the
// doubled strides of complex values read as reals) is then bounded by check_reach.
std::ptrdiff_t given_step(std::ptrdiff_t v, const std::string& what) {
  if (v == std::numeric_limits<std::ptrdiff_t>::min()) {
    reject(what, "is too large in magnitude");
  }
  return v;
}

// One side of a transform with every default resolved, in elements of the side's type.
struct side {
  std::vector<std::size_t> extent;    // per axis (the halved one included)
  std::vector<std::ptrdiff_t> stride; // per axis
  std::vector<std::ptrdiff_t> dist;   // per batch dimension
  std::ptrdiff_t element_bytes = 0;
  bool complex = true;
};

// The C-ordered defaults: the last axis contiguous (padded to `last_physical`), each axis
// spanning the ones after it, the innermost batch dimension one whole array apart and each
// outer one spanning the inner ones.
//
// An axis of extent 1 and a batch dimension of count 1 never step, so whatever stride or
// distance was given for them is replaced by 0: check_reach does not see them (they add
// (1 - 1) x stride to the reach), so nothing computed from them could be proven in range.
side resolve(const shape_view& s, const std::vector<std::size_t>& extent, std::size_t last_physical,
             bool input, bool complex, std::ptrdiff_t element_bytes) {
  const std::size_t r = extent.size();
  side out;
  out.extent = extent;
  out.complex = complex;
  out.element_bytes = element_bytes;
  out.stride.assign(r, 0);
  std::ptrdiff_t span = 1;
  for (std::size_t i = r; i > 0; --i) {
    const std::size_t d = i - 1;
    const std::ptrdiff_t given = input ? s.axes[d].in_stride : s.axes[d].out_stride;
    if (extent[d] > 1) {
      out.stride[d] = given != 0
                          ? given_step(given, axis_name(d) + (input ? ".in_stride" : ".out_stride"))
                          : span;
    }
    span = checked_mul(span, to_signed(d + 1 == r ? last_physical : extent[d], axis_name(d) + ".n"),
                       axis_name(d) + ".n");
  }
  out.dist.assign(s.batches.size(), 0);
  for (std::size_t i = s.batches.size(); i > 0; --i) {
    const std::size_t b = i - 1;
    const std::ptrdiff_t given = input ? s.batches[b].in_dist : s.batches[b].out_dist;
    if (s.batches[b].count > 1) {
      out.dist[b] =
          given != 0 ? given_step(given, batch_name(b) + (input ? ".in_dist" : ".out_dist")) : span;
    }
    span = checked_mul(span, to_signed(s.batches[b].count, batch_name(b) + ".count"),
                       batch_name(b) + ".count");
  }
  return out;
}

// The farthest element from element 0, in bytes, must be addressable; so must every step.
void check_reach(const shape_view& s, const side& v, bool input) {
  const std::string what = input ? "shape (input layout)" : "shape (output layout)";
  std::ptrdiff_t reach = 0;
  for (std::size_t d = 0; d < v.extent.size(); ++d) {
    reach = checked_add(
        reach, checked_mul(to_signed(v.extent[d] - 1, what), magnitude(v.stride[d]), what), what);
  }
  for (std::size_t b = 0; b < v.dist.size(); ++b) {
    reach = checked_add(
        reach, checked_mul(to_signed(s.batches[b].count - 1, what), magnitude(v.dist[b]), what),
        what);
  }
  static_cast<void>(checked_mul(checked_add(reach, 1, what), v.element_bytes, what));
}

// No two output elements may share an address. Ordered by size of stride, each dimension's
// stride must exceed the reach of all smaller ones: a sufficient condition that admits
// every nested layout (rows, columns, planes, padding, reversed strides).
void check_no_overlap(const shape_view& s, const side& v) {
  std::vector<std::pair<std::ptrdiff_t, std::size_t>> dims; // |stride|, extent
  for (std::size_t d = 0; d < v.extent.size(); ++d) {
    if (v.extent[d] > 1) {
      dims.emplace_back(magnitude(v.stride[d]), v.extent[d]);
    }
  }
  for (std::size_t b = 0; b < v.dist.size(); ++b) {
    if (s.batches[b].count > 1) {
      dims.emplace_back(magnitude(v.dist[b]), s.batches[b].count);
    }
  }
  std::ranges::sort(dims);
  std::ptrdiff_t reach = 0;
  for (const auto& [stride, extent] : dims) {
    if (stride <= reach) {
      reject("shape", "describes output elements that overlap (out_stride and out_dist)");
    }
    reach += static_cast<std::ptrdiff_t>(extent - 1) * stride;
  }
}

// In place, every transform must read and write the same memory: identical layouts for
// c2c; for the real kinds FFTW's padded layout, the last axis contiguous on both sides and
// every other stride and distance on the real side twice the complex side's.
void check_in_place(const shape_view& s, const side& in, const side& out) {
  const std::size_t r = in.extent.size();
  if (s.kind == transform_kind::c2c) {
    for (std::size_t d = 0; d < r; ++d) {
      if (in.extent[d] > 1 && in.stride[d] != out.stride[d]) {
        reject("options.in_place", "needs " + axis_name(d) + ".in_stride == out_stride");
      }
    }
    for (std::size_t b = 0; b < in.dist.size(); ++b) {
      if (s.batches[b].count > 1 && in.dist[b] != out.dist[b]) {
        reject("options.in_place", "needs " + batch_name(b) + ".in_dist == out_dist");
      }
    }
    return;
  }
  const side& real = in.complex ? out : in;
  const side& cplx = in.complex ? in : out;
  if (s.axes[r - 1].n > 1 && (real.stride[r - 1] != 1 || cplx.stride[r - 1] != 1)) {
    reject("options.in_place", "needs the last axis contiguous on both sides (" + axis_name(r - 1) +
                                   " strides 1) for a real transform");
  }
  for (std::size_t d = 0; d + 1 < r; ++d) {
    if (s.axes[d].n > 1 && real.stride[d] != 2 * cplx.stride[d]) {
      reject("options.in_place",
             "needs " + axis_name(d) + "'s real-side stride twice its complex-side stride");
    }
  }
  for (std::size_t b = 0; b < real.dist.size(); ++b) {
    if (s.batches[b].count > 1 && real.dist[b] != 2 * cplx.dist[b]) {
      reject("options.in_place",
             "needs " + batch_name(b) + "'s real-side distance twice its complex-side distance");
    }
  }
}

template <class T> std::vector<std::vector<T>>& tables_of(plan_core& c) {
  if constexpr (std::is_same_v<T, double>) {
    return c.tables_f64;
  } else {
    return c.tables_f32;
  }
}

// The fastest target whose table has the chain passes for T: scalar in phase 0.
template <class T> target choose_chain_target() {
  const target best = best_target(detect_cpu_features());
  const kernel_table* table = find_kernels(best);
  if (table != nullptr && of<T>(*table).chain.copy != nullptr) {
    return best;
  }
  return target::scalar;
}

// A phase's view of one slot: where it is and its strides per axis, in elements.
struct view {
  slot where = slot::in;
  std::vector<std::ptrdiff_t> stride;
  std::ptrdiff_t element_bytes = 0;
};

// Appends the loops over the lines of every axis but `axis` (outermost first), the line's
// stages inside them, and closes the loops.
void add_phase(program& p, std::size_t axis, const std::vector<std::size_t>& extent,
               const view& src, const view& dst, std::vector<stage>&& line) {
  std::vector<std::size_t> open;
  for (std::size_t e = 0; e < extent.size(); ++e) {
    if (e == axis || extent[e] == 1) {
      continue;
    }
    loop l{.kind = loop_kind::lines, .count = extent[e]};
    for (const view* v : {&src, &dst}) {
      const std::ptrdiff_t bytes = v->stride[e] * v->element_bytes;
      switch (v->where) {
      case slot::in:
        l.step[0] = bytes;
        break;
      case slot::out:
        l.step[1] = bytes;
        break;
      case slot::block:
        l.step[2] = bytes;
        break;
      case slot::a:
      case slot::b:
        break;
      }
    }
    open.push_back(p.records.size());
    p.records.emplace_back(l);
  }
  for (stage& st : line) {
    p.records.emplace_back(std::move(st));
  }
  for (const std::size_t i : open) {
    std::get<loop>(p.records[i]).body_end = static_cast<std::uint32_t>(p.records.size());
  }
}

std::string radix_list(std::size_t n) {
  const std::vector<std::size_t> radices = chain_radices(n);
  if (radices.empty()) {
    return "none (a copy)";
  }
  std::string text;
  for (const std::size_t r : radices) {
    text += (text.empty() ? "" : " x ") + std::to_string(r);
  }
  return text;
}

std::string real_note(std::size_t n, const char* kind) {
  if (n % 2 == 0) {
    return std::string(kind) + " through a half-length complex chain of " + std::to_string(n / 2) +
           ", radices " + radix_list(n / 2) +
           (kind[0] == 'r' ? ", then a post-pass" : ", after a pre-pass");
  }
  return std::string(kind) + " through a full-length complex chain (n odd), radices " +
         radix_list(n);
}

template <class T>
void build(plan_core& c, const shape_view& s, const side& in, const side& out,
           const build_settings& settings) {
  c.isa_target = choose_chain_target<T>();
  const kernels_of<T>& kernels = of<T>(*find_kernels(c.isa_target));
  chain_builder<T> chain(kernels, tables_of<T>(c), settings.chunk_points);
  program& p = c.prog;
  const std::size_t r = s.axes.size();
  const auto complex_bytes = static_cast<std::ptrdiff_t>(2 * sizeof(T));

  p.transforms = 1;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    p.records.emplace_back(loop{
        .kind = loop_kind::batch,
        .count = s.batches[b].count,
        .step = {in.dist[b] * in.element_bytes, out.dist[b] * out.element_bytes, 0},
    });
    p.transforms *= s.batches[b].count;
  }
  p.batch_loops = s.batches.size();

  const view in_view{.where = slot::in, .stride = in.stride, .element_bytes = in.element_bytes};
  const view out_view{.where = slot::out, .stride = out.stride, .element_bytes = out.element_bytes};
  std::size_t scratch = 0;
  const auto lines = [&](std::size_t axis, const std::vector<std::size_t>& extent, const view& src,
                         const view& dst, std::size_t line_scratch, std::vector<stage>&& stages) {
    scratch = std::max(scratch, line_scratch);
    add_phase(p, axis, extent, src, dst, std::move(stages));
  };
  const auto at = [](const view& v, std::size_t axis) {
    return endpoint{.where = v.where, .stride = v.stride[axis]};
  };
  const int sign = static_cast<int>(s.dir);

  if (s.kind == transform_kind::c2c) {
    for (std::size_t i = r; i > 0; --i) {
      const std::size_t d = i - 1;
      const view& src = d + 1 == r ? in_view : out_view;
      const bool aliased = src.where == slot::out || c.in_place;
      std::vector<stage> st;
      const std::size_t need =
          chain.c2c_line(c.lengths[d], sign, at(src, d), at(out_view, d), aliased, st);
      lines(d, in.extent, src, out_view, need, std::move(st));
      c.axis_notes.push_back("axis " + std::to_string(d) + " (n = " + std::to_string(c.lengths[d]) +
                             "): radices " + radix_list(c.lengths[d]));
    }
  } else if (s.kind == transform_kind::r2c) {
    std::vector<stage> st;
    const std::size_t need =
        chain.r2c_line(c.lengths[r - 1], at(in_view, r - 1), at(out_view, r - 1), st);
    lines(r - 1, in.extent, in_view, out_view, need, std::move(st));
    c.axis_notes.push_back("axis " + std::to_string(r - 1) +
                           " (n = " + std::to_string(c.lengths[r - 1]) +
                           "): " + real_note(c.lengths[r - 1], "r2c"));
    for (std::size_t i = r - 1; i > 0; --i) {
      const std::size_t d = i - 1;
      std::vector<stage> cs;
      const std::size_t n = c.lengths[d];
      const std::size_t cneed = chain.c2c_line(n, -1, at(out_view, d), at(out_view, d), true, cs);
      lines(d, out.extent, out_view, out_view, cneed, std::move(cs));
      c.axis_notes.push_back("axis " + std::to_string(d) + " (n = " + std::to_string(n) +
                             "): c2c forward, radices " + radix_list(n));
    }
  } else {
    // c2r: the complex axes first, then the real one, which must come last because the
    // real output cannot hold a complex intermediate. Out of place the input is preserved,
    // so the intermediate goes to the block: one transform's half spectrum (DESIGN.md 9.1).
    view block_view{
        .where = slot::block,
        .stride = std::vector<std::ptrdiff_t>(r, 0),
        .element_bytes = complex_bytes,
    };
    if (r > 1 && !c.in_place) {
      // The block is tight and complex. The checks of the two sides do not bound its
      // bytes (the input may overlap itself; the real output has half the bytes per
      // element), so its size is checked here.
      std::ptrdiff_t span = 1;
      for (std::size_t i = r; i > 0; --i) {
        block_view.stride[i - 1] = span;
        span = checked_mul(span, to_signed(in.extent[i - 1], "shape"), "shape");
      }
      static_cast<void>(checked_mul(span, complex_bytes, "shape"));
      c.block_elements = static_cast<std::size_t>(span);
    }
    // In place the complex phases work on the output slot, which is the input's memory,
    // through the input's (complex) strides.
    const view work =
        c.in_place ? view{.where = slot::out, .stride = in.stride, .element_bytes = complex_bytes}
                   : block_view;
    for (std::size_t i = r - 1; i > 0; --i) {
      const std::size_t d = i - 1;
      const view& src = d + 2 == r && !c.in_place ? in_view : work;
      std::vector<stage> cs;
      const std::size_t n = c.lengths[d];
      const std::size_t cneed =
          chain.c2c_line(n, +1, at(src, d), at(work, d), src.where == work.where, cs);
      lines(d, in.extent, src, work, cneed, std::move(cs));
      c.axis_notes.push_back("axis " + std::to_string(d) + " (n = " + std::to_string(n) +
                             "): c2c backward, radices " + radix_list(n));
    }
    const view& src = r > 1 && !c.in_place ? block_view : in_view;
    std::vector<stage> st;
    const std::size_t need =
        chain.c2r_line(c.lengths[r - 1], at(src, r - 1), at(out_view, r - 1), st);
    lines(r - 1, out.extent, src, out_view, need, std::move(st));
    c.axis_notes.push_back("axis " + std::to_string(r - 1) +
                           " (n = " + std::to_string(c.lengths[r - 1]) +
                           "): " + real_note(c.lengths[r - 1], "c2r"));
  }
  for (std::size_t b = 0; b < p.batch_loops; ++b) {
    std::get<loop>(p.records[b]).body_end = static_cast<std::uint32_t>(p.records.size());
  }

  // Workspace layout.
  constexpr std::size_t line_alignment = 64;
  const auto round_up = [](std::size_t v) {
    return (v + line_alignment - 1) / line_alignment * line_alignment;
  };
  c.scratch_line = scratch;
  const std::size_t line_bytes = round_up(scratch * 2 * sizeof(T));
  c.offset_b = line_bytes;
  c.offset_block = 2 * line_bytes;
  const std::size_t used = c.offset_block + c.block_elements * 2 * sizeof(T);
  c.bytes = used == 0 ? 0 : used + line_alignment;
}

const char* kind_name(transform_kind k) {
  switch (k) {
  case transform_kind::c2c:
    break;
  case transform_kind::r2c:
    return "r2c";
  case transform_kind::c2r:
    return "c2r";
  }
  return "c2c";
}

const char* slot_name(slot s) {
  switch (s) {
  case slot::in:
    return "in";
  case slot::out:
    return "out";
  case slot::a:
    return "a";
  case slot::b:
    return "b";
  case slot::block:
    break;
  }
  return "block";
}

} // namespace

std::shared_ptr<const plan_core> make_plan_core(const shape_view& s, const options& opts,
                                                const build_settings& settings) {
  const std::size_t r = s.axes.size();
  if (r == 0) {
    reject("shape.axes", "must hold at least one axis");
  }
  for (std::size_t d = 0; d < r; ++d) {
    if (s.axes[d].n == 0) {
      reject(axis_name(d) + ".n", "must be at least 1");
    }
  }
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    if (s.batches[b].count == 0) {
      reject(batch_name(b) + ".count", "must be at least 1");
    }
  }
  if (s.kind == transform_kind::c2c && s.dir != direction::forward &&
      s.dir != direction::backward) {
    reject("shape.dir", "must be direction::forward or direction::backward");
  }

  auto c = std::make_shared<plan_core>();
  c->kind = s.kind;
  c->prec = s.prec;
  c->dir = s.dir;
  if (s.kind != transform_kind::c2c) {
    c->dir = s.kind == transform_kind::r2c ? direction::forward : direction::backward;
  }
  c->in_place = opts.in_place;
  for (const axis& a : s.axes) {
    c->lengths.push_back(a.n);
  }
  for (const batch& b : s.batches) {
    c->counts.push_back(b.count);
  }

  const std::size_t real_bytes = s.prec == precision::f64 ? sizeof(double) : sizeof(float);
  const std::size_t half = s.axes[r - 1].n / 2 + 1;
  std::vector<std::size_t> full = c->lengths;
  std::vector<std::size_t> halved = c->lengths;
  halved[r - 1] = half;
  // The padded real row of an in-place real transform: the 2(n/2 + 1) reals of its complex
  // row.
  const std::size_t padded = opts.in_place ? 2 * half : s.axes[r - 1].n;
  const auto rb = static_cast<std::ptrdiff_t>(real_bytes);
  const auto cb = static_cast<std::ptrdiff_t>(2 * real_bytes);
  side in;
  side out;
  switch (s.kind) {
  case transform_kind::c2c:
    in = resolve(s, full, full[r - 1], true, true, cb);
    out = resolve(s, full, full[r - 1], false, true, cb);
    break;
  case transform_kind::r2c:
    in = resolve(s, full, padded, true, false, rb);
    out = resolve(s, halved, half, false, true, cb);
    break;
  case transform_kind::c2r:
    in = resolve(s, halved, half, true, true, cb);
    out = resolve(s, full, padded, false, false, rb);
    break;
  }
  check_reach(s, in, true);
  check_reach(s, out, false);
  check_no_overlap(s, out);
  if (opts.in_place) {
    check_in_place(s, in, out);
  }

  if (s.prec == precision::f64) {
    build<double>(*c, s, in, out, settings);
  } else {
    build<float>(*c, s, in, out, settings);
  }
  return c;
}

std::shared_ptr<const plan_core> make_plan_core(const shape_view& shape, const options& opts) {
  return make_plan_core(shape, opts, build_settings{});
}

void execute_core(const plan_core& core, std::size_t first, std::size_t count, const void* in,
                  void* out, const workspace& ws) noexcept {
  assert(ws.size() >= core.bytes && "fftn: workspace smaller than plan::workspace_bytes()");
  assert(first <= core.prog.transforms && count <= core.prog.transforms - first &&
         "fftn: execute_rows beyond the batch");
  slot_addresses at{.in = static_cast<const std::byte*>(in), .out = static_cast<std::byte*>(out)};
  if (core.bytes != 0) {
    // The workspace may have any alignment; its size includes the slack for this.
    std::byte* raw = ws.memory().data();
    const auto address = reinterpret_cast<std::uintptr_t>(raw);
    const std::size_t skip = (64 - address % 64) % 64;
    std::byte* base = raw + skip;
    at.a = base;
    at.b = base + core.offset_b;
    at.block = core.block_elements != 0 ? base + core.offset_block : nullptr;
  }
  run_transforms(core.prog, first, count, at);
}

std::size_t transform_count(const plan_core& core) noexcept { return core.prog.transforms; }

std::size_t workspace_bytes(const plan_core& core) noexcept { return core.bytes; }

isa instruction_set(const plan_core& core) noexcept {
  return static_cast<isa>(static_cast<unsigned char>(core.isa_target));
}

std::string describe(const plan_core& core) {
  std::ostringstream o;
  o << "fftn plan: " << kind_name(core.kind);
  if (core.kind == transform_kind::c2c) {
    o << (core.dir == direction::forward ? " forward" : " backward");
  }
  o << ", " << (core.prec == precision::f64 ? "double" : "float") << ", n = ";
  for (std::size_t d = 0; d < core.lengths.size(); ++d) {
    o << (d == 0 ? "" : " x ") << core.lengths[d];
  }
  o << ", " << core.prog.transforms << (core.prog.transforms == 1 ? " transform" : " transforms")
    << (core.in_place ? ", in place" : ", out of place") << "\n";
  o << "instruction set: " << isa_name(instruction_set(core)) << "\n";
  o << "strategy: chain (Stockham autosort, breadth-first), one line at a time; each "
       "transform finished before the next\n";
  for (const std::string& note : core.axis_notes) {
    o << note << "\n";
  }
  o << "workspace: " << core.bytes << " bytes (scratch lines a, b of " << core.scratch_line
    << " complex each; block of " << core.block_elements << " complex)\n";
  o << "program:\n";
  std::vector<std::size_t> ends;
  for (std::size_t i = 0; i < core.prog.records.size(); ++i) {
    while (!ends.empty() && ends.back() == i) {
      ends.pop_back();
    }
    const std::string indent(2 * (ends.size() + 1), ' ');
    const record& rec = core.prog.records[i];
    if (const loop* l = std::get_if<loop>(&rec)) {
      o << indent << "loop " << (l->kind == loop_kind::batch ? "batch" : "lines") << " x"
        << l->count << " (steps in bytes: in " << l->step[0] << ", out " << l->step[1] << ", block "
        << l->step[2] << ")\n";
      ends.push_back(l->body_end);
      continue;
    }
    const auto& st = std::get<stage>(rec);
    o << indent << st.name << ": " << slot_name(st.src) << " -> " << slot_name(st.dst) << ", "
      << st.units << " units in " << st.chunks << (st.chunks == 1 ? " chunk" : " chunks") << "\n";
  }
  return o.str();
}

} // namespace fftn::inline v0::detail
