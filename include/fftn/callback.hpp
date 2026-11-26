// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Block callbacks: a function compiled by the application, which a plan calls on blocks of an
/// edge's data while they are in cache (dev/DESIGN.md 12.3). An edge takes one through
/// `edge::call` (fftn/edges.hpp); its argument arrives with each execution (`exec_args`).
///
/// Provisional (tasks 5.3a to 5.3d, 2026-11-26): this API follows dev/DESIGN.md 12.3 with the
/// proposals of 18.2 ("Callbacks"), which the maintainer has yet to settle, and may change before
/// the first release. A callback is taken by c2c, r2c and c2r plans of every rank and by row
/// pipelines (fftn/pipeline.hpp: the prologue, the middle and the epilogue); construction refuses
/// one on real-to-real plans and on planes and volumes with stored runs, naming the field.
///
/// The contract, for every callback:
///
/// - **Coverage.** Each element of the edge's side (each stored element, with stored runs) is
///   passed exactly once per execution (a pipeline's middle: each bin once per line; its
///   epilogue: each element once per copy of a fan-out), in blocks whose shape the plan chooses: a
///   line whole up to a size set per machine (half the L1 data cache by default), a longer one in
///   element ranges, and several short lines in one call where the plan holds them together
///   (consecutive transforms, or on a plane or volume adjacent lines along another axis). Blocks of
///   one execution may run concurrently on different members of a team or executor, and in any
///   order; block shapes may differ between execution forms (serial, team, executor). A
///   pipeline's middle callback is passed each line's whole spectrum in one call.
/// - **Purity and determinism.** When the callback is pure per element (its result depends only
///   on the element's value and coordinates; a pipeline's middle callback: pure per line), the
///   plan's output is bitwise identical for serial execution, pieces of the batch, any team, any
///   executor and `completion::own`. Shared state
///   the callback writes must be indexed by coordinates or by `block_info::member`.
/// - **What a callback must not do.** Throw (`noexcept` is part of `block_fn`); use the
///   workspace it runs under, except its `block_info::scratch`; write outside its block, except
///   to the application's own memory; assume any order of blocks. It may allocate, lock or do
///   I/O: that is the application's choice, not the plan's.
/// - **Nested execution.** A callback may execute another plan through a serial overload
///   (`execute(in, out, ws[, args])`, `execute_transforms`), with a workspace of its own: for
///   example `fftn::workspace{info.scratch}` when `callback::scratch_bytes` is at least that
///   plan's `workspace_bytes()`, or the application's memory indexed by `member`, never the
///   outer workspace. Team and executor overloads inside a callback are not supported (and not
///   detected).
/// - **Environment.** The callback runs outside every instruction-set region of the library, on
///   the calling thread's floating-point environment, which FFTN never changes.
#pragma once

#include "fftn/config.hpp"

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace fftn::inline v0 {

/// `block_info::line_axis` when the lines of a block are consecutive transforms: line k is
/// transform `transform + k` of the batch (consecutive in the innermost batch dimension).
inline constexpr std::size_t along_transforms = static_cast<std::size_t>(-1);

/// Where a block lies, handed to a callback with each call.
///
/// Element i of line k of the block is at `data[i * element_stride + k * line_stride]`. Its
/// coordinates are `origin` advanced by i along `element_axis` and by k along `line_axis`
/// (or, for `along_transforms`, the transform `transform + k`). Coordinates along a transformed
/// axis are logical positions: on an axis with stored runs (`shape::runs`), positions in the
/// whole logical line, not in memory. Batch coordinates follow from the transform number in the
/// documented batch order (the innermost batch dimension fastest; docs/conventions.md).
struct block_info {
  /// The batch number of line 0's transform, counted from transform 0 of the whole batch
  /// (`execute_transforms(first, ...)` numbers its transforms from `first`).
  std::size_t transform = 0;
  /// The logical index of element 0 of line 0 along each transformed axis, in the order of
  /// `shape::axes`; valid during the call only.
  std::span<const std::size_t> origin{};
  /// The transformed axis (an index into `shape::axes`) along which elements advance.
  std::size_t element_axis = 0;
  /// The axis along which lines advance, or `along_transforms`.
  std::size_t line_axis = along_transforms;
  /// The elements per line.
  std::size_t length = 0;
  /// The lines of the block, at least 1.
  std::size_t lines = 1;
  /// The distance between consecutive elements of a line, in elements of the data's type.
  std::ptrdiff_t element_stride = 1;
  /// The distance between consecutive lines, in elements of the data's type (0 for one line).
  std::ptrdiff_t line_stride = 0;
  /// The members of a group handed together (a pipeline's `group`, not built yet): 1 otherwise.
  std::size_t group = 1;
  /// The distance between consecutive members of a group, in elements (0 without a group).
  std::ptrdiff_t group_stride = 0;
  /// Which inverse of a pipeline's fan-out the block belongs to (an epilogue's block; task
  /// 5.3d): 0 otherwise, the middle's included.
  std::size_t copy = 0;
  /// The member running the call: 0 in serial execution, the rank in a team, the task's index
  /// under an executor; always below the lanes the workspace was sized for.
  int member = 0;
  /// This member's callback scratch: exactly `callback::scratch_bytes` bytes, starting on a
  /// 128-byte boundary (empty when that is 0). Its contents are not kept between calls.
  std::span<std::byte> scratch{};
};

/// A block callback on data of type `T` (the edge's element type: `std::complex<R>` on a complex
/// side, `R` on a real side; a pipeline middle's `Mid`). `user` is the edge's pointer in
/// `exec_args` (`in_user`, `mid_user`, `out_user`), passed as given, null included. `data`
/// addresses element 0 of line 0.
template <class T> using block_fn = void (*)(void* user, const block_info& info, T* data) noexcept;

/// What a callback does with its block.
enum class callback_mode : unsigned char {
  /// The block holds the edge's values, which the callback may change in place: the input before
  /// the transform (an out-of-place plan's input itself is never changed: the callback sees a
  /// copy), the output after it.
  modify,
  /// Input edges only: the callback fills the block, and the plan reads nothing from the input,
  /// which may then be null. Its prior contents are unspecified.
  produce,
  /// Output edges only: the callback takes the block, and the plan stores nothing to the output,
  /// which may then be null. Not with `edge::accumulate`, nor on an in-place plan.
  consume,
};

/// The callback of an edge (`edge::call`).
///
/// A null `fn` is no callback; construction then refuses `mode` and `scratch_bytes` other than
/// their defaults (`std::invalid_argument` naming the field). An input edge takes `modify` or
/// `produce`, an output edge `modify` or `consume`.
template <class T> struct callback {
  /// The function; null for none.
  block_fn<T> fn = nullptr;
  /// What it does with its block.
  callback_mode mode = callback_mode::modify;
  /// The bytes of scratch each member's calls receive (`block_info::scratch`), carved from the
  /// workspace: `workspace_bytes()` counts them for each lane.
  std::size_t scratch_bytes = 0;
};

namespace detail {
template <class F, class T>
void call_callable(void* user, const block_info& info, T* data) noexcept {
  (*static_cast<F*>(user))(info, data);
}
} // namespace detail

/// A `block_fn<T>` that calls the callable object of type `F` its `user` pointer addresses, as
/// `(*static_cast<F*>(user))(info, data)`: a lambda or function object as a callback, its
/// address passed in `exec_args`.
///
/// `F` must be nothrow-invocable with `(const block_info&, T*)` (so a lambda is declared
/// `noexcept`); anything else does not compile. `F` may be const-qualified.
template <class F, class T>
  requires std::is_nothrow_invocable_v<F&, const block_info&, T*>
constexpr block_fn<T> callback_of() noexcept {
  return &detail::call_callable<F, T>;
}

} // namespace fftn::inline v0
