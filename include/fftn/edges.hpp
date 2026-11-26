// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Plan edges: linear pointwise work fused into a plan's first and last passes (scales,
/// real and complex factors, accumulation), and the data of one execution (`exec_args`).
///
/// Provisional (task 5.1, 2026-11-25): this API follows dev/DESIGN.md section 12 as the
/// maintainer agreed it on 2026-11-24, and may still change before the first release.
/// docs/conventions.md ("Edges") states what an edge computes.
#pragma once

#include "fftn/callback.hpp" // IWYU pragma: export
#include "fftn/config.hpp"

#include <cstddef>
#include <vector>

namespace fftn::inline v0 {

/// What an edge's factor multiplies by: nothing, an array of real values (a window, say) or
/// an array of complex values, in the plan's precision.
enum class factor_kind : unsigned char {
  none,    ///< no factor
  real,    ///< real values (`double` or `float`, the plan's precision)
  complex, ///< complex values (`std::complex` of the plan's precision)
};

/// A factor of an edge: every element of the edge's side is multiplied by the element of an
/// array at the same logical coordinates. The array is not part of the plan: each execution
/// passes its address (`exec_args`), so one plan serves many arrays.
///
/// The array is indexed through its own strides, counted in its own elements (real values
/// for `factor_kind::real`, complex values for `factor_kind::complex`), which may be
/// negative: the element for transformed coordinates (j_0, ..., j_{r-1}) of transform
/// (b_0, ..., b_{m-1}) of the batch is at
/// `sum_d axis_strides[d] j_d + sum_e batch_strides[e] b_e` from the address passed. A stride
/// of 0 broadcasts the array along that dimension: `batch_strides` all 0, or empty, applies
/// one line of factors to every transform. The strides of an axis of length 1 or a batch dimension
/// of count 1 are ignored. The coordinates are those of the edge's side: input positions for the
/// input edge, output bins for the output edge (for c2c, the same range).
///
/// Preconditions, checked at plan construction (`std::invalid_argument` naming the field):
/// with a factor, `axis_strides` has one stride per axis of the shape and `batch_strides`
/// one per batch dimension or none, and the array's span in bytes fits `std::ptrdiff_t`; a complex
/// factor is not allowed on a real side (the input of r2c, the output of c2r, both sides of
/// r2r).
struct factor {
  /// The kind of the array; `factor_kind::none` (the default) means no factor, and the
  /// strides are then ignored.
  factor_kind kind = factor_kind::none;
  /// The array's stride along each transformed axis, in the order of `shape::axes`.
  std::vector<std::ptrdiff_t> axis_strides{};
  /// The array's stride along each batch dimension, in the order of `shape::batches`; empty
  /// broadcasts along every batch dimension, as strides of 0 do.
  std::vector<std::ptrdiff_t> batch_strides{};
};

/// The vocabulary of one edge of a plan (dev/DESIGN.md 12.1). `T` is the element type of
/// the edge's side: `In` for the input edge, `Out` for the output edge.
///
/// The input edge multiplies every input element by `mul`'s factor and by `scale` before the
/// transform; the output edge multiplies every output element by `mul`'s factor and by
/// `scale` after it, then stores it, or adds it to the output's value when `accumulate` is
/// set. A default edge does nothing.
///
/// With a callback (`call`, fftn/callback.hpp) the operations apply in this order: on the input
/// edge, the input is read (or `produce`d by the callback), a `modify` callback runs, then the
/// factor and scale; on the output edge, the factor and scale, then a `modify` callback, then
/// the store, the accumulation or the callback's `consume`. The scales of both edges and
/// `options::scale` are then applied after the input's callback and before the output's.
///
/// Guarantees: the result equals the transform of the multiplied input, multiplied and
/// added as above, within the transform's error bound plus one rounding per multiplication
/// and addition (dev/DESIGN.md 15.5); it is not necessarily bitwise equal to the same work
/// done by separate loops. The scales of both edges and `options::scale` are applied as one
/// product, which differs from separate multiplications only in rounding. A plan's results stay
/// bitwise identical between serial, team and executor executions.
template <class T> struct edge {
  /// A real factor; finite, and for `float` plans 0 or of a magnitude from the smallest
  /// positive `float` to the largest. The scales of the two edges and `options::scale` may be
  /// applied as one product, which must meet the same conditions.
  double scale = 1.0;
  /// The factor array's kind and strides; its address arrives with each execution.
  factor mul{};
  /// Output edges only: `out += result` instead of `out = result`. Construction refuses it
  /// on the input edge and on an in-place plan (`options::in_place`), whose output's prior
  /// values are its input.
  bool accumulate = false;
  /// A block callback (fftn/callback.hpp), none by default; its argument arrives with each
  /// execution (`exec_args::in_user`, `out_user`). Provisional (tasks 5.3a, 5.3b): plans of one
  /// axis only (c2c, r2c, c2r), on the strategy the rules or measured planning choose, except
  /// that a consumed output keeps a row off the strategies that write the output as working
  /// memory (the grid, the depth-first chains, the lanes over adjacent transforms, and a real
  /// row's lanes whose input is in the workspace block too), which then run as the chain
  /// (`plan::describe()` says so); measured planning times candidates with stand-ins that
  /// do nothing, never calling the callback. A plan with a callback must be executed through an
  /// overload that takes `exec_args`: the others end the program (`std::abort`, with a message
  /// on stderr). A pipeline's prologue and epilogue refuse one until task 5.3d.
  callback<T> call{};
};

/// The two edges of a plan (`plan::plan`): `in` for the input, `out` for the output.
///
/// Implemented so far (task 5.1): scales, factors and accumulation on every plan, in the
/// strategy the rules or measured planning choose with a few exceptions, where
/// `plan::describe()` names the strategy the plan without the edges takes, the one taken
/// instead and why. A factor is fused into a c2c row's chain where it can be;
/// elsewhere each factor and accumulation is a pass of its own over the data (describe()
/// counts them), and the
/// workspace may hold one transform (or row) more. A wisdom entry of a plan with edges serves
/// plans with the same edges only. A real-to-real plan (`shape<T, T>`, task 6.3) applies the
/// input's real factor as it reads its first axis's lines and the output's real factor, the
/// scales and accumulation as it writes its last axis's, with no separate pass.
template <class In, class Out> struct plan_edges {
  /// The input edge.
  edge<In> in{};
  /// The output edge.
  edge<Out> out{};
};

/// The data of one execution (dev/DESIGN.md 12): the factor arrays of the edges and the
/// pointers their block callbacks receive. Every pointer may be null where the plan has nothing
/// that reads it; a callback's pointer is passed to it as given, null included.
///
/// Preconditions at execution: a plan whose input or output edge has a factor needs that
/// edge's array (a pipeline likewise its prologue's, middle's and epilogue's, fftn/pipeline.hpp),
/// addressing coordinate 0 (with negative strides not the lowest address), valid for every element
/// the factor's strides reach, with any alignment of at least its element type's, and not written
/// during the execution. The plan only reads it. An execution without an array the plan needs
/// (null, or through an overload of `execute` that takes no `exec_args`) ends the program
/// (`std::abort`, with a message on stderr).
struct exec_args {
  /// The input edge's factor array (`edge::mul` of `plan_edges::in`).
  const void* in_factor = nullptr;
  /// A pipeline's middle factor array (`pipeline_desc::middle`, fftn/pipeline.hpp); plans
  /// ignore it.
  const void* mid_factor = nullptr;
  /// The output edge's factor array (`edge::mul` of `plan_edges::out`).
  const void* out_factor = nullptr;
  /// The input edge's callback argument (`edge::call` of `plan_edges::in`), its `user`.
  void* in_user = nullptr;
  /// A pipeline middle's callback argument (`pipeline_desc::middle.call`, task 5.3d; until then,
  /// and for plans, ignored).
  void* mid_user = nullptr;
  /// The output edge's callback argument (`edge::call` of `plan_edges::out`), its `user`.
  void* out_user = nullptr;
};

} // namespace fftn::inline v0
