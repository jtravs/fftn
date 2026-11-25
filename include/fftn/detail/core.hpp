// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The type-erased plan core behind plan<In, Out> (DESIGN.md 14.2): the typed wrapper in
// plan.hpp converts its shape to a shape_view and forwards everything here. Not part of the
// public API; the definitions are in src/core/.
#pragma once

#include "fftn/config.hpp"
#include "fftn/edges.hpp"
#include "fftn/executor.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <type_traits>

namespace fftn::inline v0::detail {

enum class transform_kind : unsigned char { c2c, r2c, c2r };
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

template <class In, class Out>
inline constexpr precision precision_of =
    std::is_same_v<In, double> || std::is_same_v<In, std::complex<double>> ? precision::f64
                                                                           : precision::f32;

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
};

template <class In, class Out> shape_view view_of(const shape<In, Out>& s) noexcept {
  return {
      .kind = kind_of<In, Out>::value,
      .prec = precision_of<In, Out>,
      .dir = s.dir,
      .axes = s.axes,
      .batches = s.batches,
  };
}

// One edge of plan_edges<In, Out> (fftn/edges.hpp) as plain values; the strides are borrowed
// from the edge's vectors, as shape_view borrows the axes.
struct edge_view {
  double scale = 1.0;
  factor_kind kind = factor_kind::none;
  std::span<const std::ptrdiff_t> axis_strides;
  std::span<const std::ptrdiff_t> batch_strides;
  bool accumulate = false;
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
  };
}

template <class In, class Out> edges_view view_of(const plan_edges<In, Out>& e) noexcept {
  return {.in = view_of(e.in), .out = view_of(e.out)};
}

class plan_core;

// Validates the shape and options and builds the plan; throws std::invalid_argument naming
// the offending argument.
[[nodiscard]] std::shared_ptr<const plan_core> make_plan_core(const shape_view& shape,
                                                              const options& opts);
// The same with edges (src/core/edges.cpp validates them).
[[nodiscard]] std::shared_ptr<const plan_core>
make_plan_core_with_edges(const shape_view& shape, const options& opts, const edges_view& edges);

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
