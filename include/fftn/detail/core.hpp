// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The type-erased plan core behind plan<In, Out> (DESIGN.md 14.2): the typed wrapper in
// plan.hpp converts its shape to a shape_view and forwards everything here. Not part of the
// public API; the definitions are in src/core/.
#pragma once

#include "fftn/callback.hpp"
#include "fftn/config.hpp"
#include "fftn/edges.hpp"
#include "fftn/executor.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/split.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <type_traits>

namespace fftn::inline v0::detail {

// r2r: a real-to-real shape (shape<T, T>), whose shape_view::kinds give each axis's transform
// (DESIGN.md 5.7, src/core/r2r.cpp).
enum class transform_kind : unsigned char { c2c, r2c, c2r, r2r };
enum class precision : unsigned char { f64, f32 };

// The kind and precision of a plan<In, Out>, from its element types.
template <class In, class Out> struct kind_of;
template <class T> struct kind_of<std::complex<T>, std::complex<T>> {
  static constexpr transform_kind value = transform_kind::c2c;
};
template <class T> struct kind_of<T, std::complex<T>> {
  static constexpr transform_kind value = transform_kind::r2c;
};
template <class T> struct kind_of<std::complex<T>, T> {
  static constexpr transform_kind value = transform_kind::c2r;
};
template <> struct kind_of<double, double> {
  static constexpr transform_kind value = transform_kind::r2r;
};
template <> struct kind_of<float, float> {
  static constexpr transform_kind value = transform_kind::r2r;
};

// The split-complex kinds (fftn/split.hpp, DESIGN.md 5.8; task 6.10): the same kinds, the
// complex sides split (shape_view::split).
template <class T> struct kind_of<split_complex<T>, split_complex<T>> {
  static constexpr transform_kind value = transform_kind::c2c;
};
template <class T> struct kind_of<T, split_complex<T>> {
  static constexpr transform_kind value = transform_kind::r2c;
};
template <class T> struct kind_of<split_complex<T>, T> {
  static constexpr transform_kind value = transform_kind::c2r;
};

template <class E> inline constexpr bool is_split = false;
template <class T> inline constexpr bool is_split<split_complex<T>> = true;

// Whether plan<In, Out> or shape<In, Out> is of split-complex data.
template <class In, class Out> inline constexpr bool split_kind = is_split<In> || is_split<Out>;

template <class In, class Out>
inline constexpr precision precision_of =
    std::is_same_v<In, double> || std::is_same_v<In, std::complex<double>> ||
            std::is_same_v<In, split_complex<double>>
        ? precision::f64
        : precision::f32;

// What one execution passes for a side of element type E (DESIGN.md 5.8, "One class
// template"): a pointer to E, or for split-complex data the two arrays.
template <class E> struct data_of {
  using in = const E*;
  using out = E*;
};
template <class T> struct data_of<split_complex<T>> {
  using in = split_ptr<const T>;
  using out = split_ptr<T>;
};
template <class E> using in_data = typename data_of<E>::in;
template <class E> using out_data = typename data_of<E>::out;

// The array a split execution passes through the slots (the real parts, or a real side's own
// array) and the arrays it passes as the conversion passes' cursors (core/edges.hpp: the
// imaginary parts, read and written through exec_args' factor pointers; null on a real side).
template <class T> T* slot_array(T* p) noexcept { return p; }
template <class T> T* slot_array(split_ptr<T> p) noexcept { return p.re; }
template <class T> const void* cursor_array(T* /*p*/) noexcept { return nullptr; }
template <class T> const void* cursor_array(split_ptr<T> p) noexcept { return p.im; }
template <class In, class Out> exec_args split_args(In in, Out out) noexcept {
  return {.in_factor = cursor_array(in), .out_factor = cursor_array(out)};
}

// A shape with its type information as values. The axes and batch dimensions are borrowed
// from the shape's vectors, which must outlive the view (plan construction uses it within one
// full-expression): only plain structs cross into the library, never a std::vector, and
// nothing is copied.
struct shape_view {
  transform_kind kind = transform_kind::c2c;
  precision prec = precision::f64;
  direction dir = direction::forward;
  std::span<const axis> axes;
  std::span<const batch> batches;
  std::span<const stored_run> runs{};
  std::span<const r2r_kind> kinds{}; // shape::kinds: one per axis for r2r, empty otherwise
  // The complex sides are split-complex arrays (fftn/split.hpp, task 6.10): their strides and
  // distances count reals, and the plan converts them through the workspace block.
  bool split = false;
};

template <class In, class Out> shape_view view_of(const shape<In, Out>& s) noexcept {
  return {
      .kind = kind_of<In, Out>::value,
      .prec = precision_of<In, Out>,
      .dir = s.dir,
      .axes = s.axes,
      .batches = s.batches,
      .runs = s.runs,
      .kinds = s.kinds,
      .split = split_kind<In, Out>,
  };
}

// A block callback (fftn/callback.hpp) with its data type erased: the library converts it back
// to the block_fn of the edge's element type before calling it (core/callbacks.cpp), so that it
// is never called through another type.
using erased_block_fn = void (*)(void* user, const block_info& info, void* data) noexcept;

// One edge of plan_edges<In, Out> (fftn/edges.hpp) as plain values; the strides are borrowed
// from the edge's vectors, as shape_view borrows the axes.
struct edge_view {
  double scale = 1.0;
  factor_kind kind = factor_kind::none;
  std::span<const std::ptrdiff_t> axis_strides;
  std::span<const std::ptrdiff_t> batch_strides;
  bool accumulate = false;
  erased_block_fn call = nullptr;
  callback_mode call_mode = callback_mode::modify;
  std::size_t call_scratch = 0;
};

struct edges_view {
  edge_view in;
  edge_view out;
};

template <class T> edge_view view_of(const edge<T>& e) noexcept {
  return {
      .scale = e.scale,
      .kind = e.mul.kind,
      .axis_strides = e.mul.axis_strides,
      .batch_strides = e.mul.batch_strides,
      .accumulate = e.accumulate,
      .call = reinterpret_cast<erased_block_fn>(e.call.fn), // converted back before any call
      .call_mode = e.call.mode,
      .call_scratch = e.call.scratch_bytes,
  };
}

template <class In, class Out> edges_view view_of(const plan_edges<In, Out>& e) noexcept {
  return {.in = view_of(e.in), .out = view_of(e.out)};
}

// The kinds of pipeline<In, Mid, Out> (fftn/pipeline.hpp, DESIGN.md 12.2), from its element
// types: two complex transforms, r2c then c2r (a real convolution), or c2r then r2c.
enum class pipeline_kind : unsigned char { complex_pair, real_convolution, real_middle };

template <class In, class Mid, class Out> struct pipeline_kind_of;
template <class T> struct pipeline_kind_of<std::complex<T>, std::complex<T>, std::complex<T>> {
  static constexpr pipeline_kind value = pipeline_kind::complex_pair;
};
template <class T> struct pipeline_kind_of<T, std::complex<T>, T> {
  static constexpr pipeline_kind value = pipeline_kind::real_convolution;
};
template <class T> struct pipeline_kind_of<std::complex<T>, T, std::complex<T>> {
  static constexpr pipeline_kind value = pipeline_kind::real_middle;
};

// A pipeline description (pipeline_desc) as plain values, borrowing its vectors as shape_view
// does; the middle is an edge_view without accumulation.
struct pipeline_view {
  pipeline_kind kind = pipeline_kind::complex_pair;
  precision prec = precision::f64;
  direction first = direction::forward;
  axis line{};
  std::span<const batch> batches;
  std::span<const stored_run> runs;
  edge_view prologue;
  edge_view middle;
  edge_view epilogue;
  std::size_t fanout = 1;
  std::ptrdiff_t fanout_out_dist = 0;
  std::ptrdiff_t fanout_factor_dist = 0;
  std::size_t group = static_cast<std::size_t>(-1); // fftn::no_group
};

class plan_core;

// Validates the shape and options and builds the plan; throws std::invalid_argument naming
// the offending argument.
[[nodiscard]] std::shared_ptr<const plan_core> make_plan_core(const shape_view& shape,
                                                              const options& opts);
// The same with edges (src/core/edges.cpp validates them).
[[nodiscard]] std::shared_ptr<const plan_core>
make_plan_core_with_edges(const shape_view& shape, const options& opts, const edges_view& edges);

// Validates a pipeline description and builds its program (src/core/pipeline.cpp); throws
// std::invalid_argument naming the offending field (desc.line.n, desc.middle.mul.kind, ...). The
// pipeline runs on the plan entry points below, with an exec_args.
[[nodiscard]] std::shared_ptr<const plan_core> make_pipeline_core(const pipeline_view& desc,
                                                                  const options& opts);

// Runs transforms [first, first + count) of the batch; in and out address element 0 of
// transform 0. Preconditions as for plan<In, Out>::execute_transforms.
void execute_core(const plan_core& core, std::size_t first, std::size_t count, const void* in,
                  void* out, const workspace& ws) noexcept;

// Runs every transform of the batch: plan<In, Out>::execute, one call into the library
// (execute_core with the batch's count read inside, X28).
void execute_all(const plan_core& core, const void* in, void* out, const workspace& ws) noexcept;

// The team overload of plan<In, Out>::execute (src/threading/team_execute.cpp).
void execute_team(const plan_core& core, const void* in, void* out, const team& t,
                  const workspace& ws, completion done) noexcept;

// The executor overload of plan<In, Out>::execute (src/threading/executor_execute.cpp).
void execute_on(const plan_core& core, const void* in, void* out, const executor_ref& ex,
                const workspace& ws);

// The twins of the four above that take the data of one execution (fftn/edges.hpp): a plan
// whose edges have factors reads their arrays from `args`; any other plan ignores it.
void execute_core(const plan_core& core, std::size_t first, std::size_t count, const void* in,
                  void* out, const workspace& ws, const exec_args& args) noexcept;
void execute_all(const plan_core& core, const void* in, void* out, const workspace& ws,
                 const exec_args& args) noexcept;
void execute_team(const plan_core& core, const void* in, void* out, const team& t,
                  const workspace& ws, completion done, const exec_args& args) noexcept;
void execute_on(const plan_core& core, const void* in, void* out, const executor_ref& ex,
                const workspace& ws, const exec_args& args);

[[nodiscard]] transform_range transforms_of(const plan_core& core, const team& t) noexcept;
[[nodiscard]] std::size_t transform_count(const plan_core& core) noexcept;
[[nodiscard]] std::size_t workspace_bytes(const plan_core& core, int lanes) noexcept;
[[nodiscard]] std::string describe(const plan_core& core);
[[nodiscard]] isa instruction_set(const plan_core& core) noexcept;

} // namespace fftn::inline v0::detail
