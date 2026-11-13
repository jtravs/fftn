// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Plans: a shape prepared for execution.
#pragma once

#include "fftn/config.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/workspace.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <string>

namespace fftn::inline v0 {

/// A transform of a fixed shape, ready to execute on any arrays of that layout.
///
/// Everything is decided when the plan is constructed: the instruction set, the
/// factorisation, the passes and the twiddle tables. A plan holds no pointers to the
/// caller's arrays and no scratch, so it can be executed any number of times, on any data
/// of the planned layout, at any alignment, and by several threads at once (each with its
/// own workspace).
///
/// Guarantees:
/// - Construction validates the shape and options, and throws `std::invalid_argument`
///   naming the offending argument (for example `shape.axes[1].n`) when they are not
///   valid. It does not touch the caller's arrays.
/// - Execution never allocates, locks, throws or does I/O.
/// - Results are bitwise reproducible for a given plan: every transform of the batch gets
///   the same arithmetic, whatever its position, and repeated executions agree bit for
///   bit. Plans built with different options, or on different instruction sets, may differ
///   in the last bits (docs/conventions.md, "Determinism").
/// - Plans are cheap to copy; copies share their immutable tables. There is no empty plan:
///   moving a plan copies it, so the source of a move stays a valid plan too.
template <class In, class Out> class plan {
public:
  /// Builds the plan of `s` with options `opts`.
  ///
  /// Throws `std::invalid_argument` when the shape or the options are not valid (the
  /// message names the argument; the rules are in docs/conventions.md): an axis or batch
  /// count of zero, no axis or more than three axes, more than two batch dimensions,
  /// overlapping output elements, sizes that overflow, or in-place layouts that do not
  /// agree. Throws `std::bad_alloc` when memory runs out.
  explicit plan(const shape<In, Out>& s, const options& opts = {})
      : core_(detail::make_plan_core(detail::view_of(s), opts)) {}

  /// Makes a plan that shares `other`'s immutable tables: it transforms exactly as `other`
  /// does. Also used for moves (the class declares no move operations), so the source of
  /// a move is left unchanged and valid.
  plan(const plan& other) noexcept = default;

  /// Makes this plan share `other`'s immutable tables, as the copy constructor does; also
  /// used for move assignment, leaving the source unchanged.
  plan& operator=(const plan& other) noexcept = default;

  /// Releases this plan's share of the tables; the last plan to go frees them.
  ~plan() = default;

  /// Transforms every transform of the batch, serially on the calling thread.
  ///
  /// `in` and `out` address element 0 of the first transform's input and output (strides
  /// and distances may be negative, so this need not be the lowest address). For an
  /// in-place plan they must be the same address; otherwise the arrays must not overlap.
  /// Any alignment of at least the element type's is accepted.
  ///
  /// Preconditions: the arrays hold the planned layout; `ws.size()` is at least
  /// `workspace_bytes()`, and `ws` is used by no other execution meanwhile.
  ///
  /// Guarantees: the input is preserved unless the plan is in place; nothing outside the
  /// described output elements is written. Never allocates, locks, throws or does I/O.
  void execute(const In* in, Out* out, workspace& ws) const noexcept {
    detail::execute_core(*core_, 0, detail::transform_count(*core_), in, out, ws);
  }

  /// Transforms `count` transforms of the batch, starting with transform `first`, serially
  /// on the calling thread: for callers that share a batch out themselves.
  ///
  /// Transforms are numbered in batch order (the last batch dimension varies fastest).
  /// `in` and `out` are the same addresses as for `execute`: element 0 of transform 0, not
  /// of transform `first`.
  ///
  /// Preconditions: as for `execute`, and `first + count` is at most the number of
  /// transforms in the batch.
  ///
  /// Guarantees: as for `execute`; each transform's result is bitwise identical to what
  /// `execute` gives it.
  void execute_rows(std::size_t first, std::size_t count, const In* in, Out* out,
                    workspace& ws) const noexcept {
    detail::execute_core(*core_, first, count, in, out, ws);
  }

  /// The bytes of workspace an execution of this plan needs; 0 when it needs none.
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return detail::workspace_bytes(*core_);
  }

  /// The plan in human-readable form: the transform, the instruction set, the strategy,
  /// the radices and passes of every axis, the loops over the batch, and the workspace.
  /// For people, not for parsing: the format may change between versions.
  ///
  /// Throws `std::bad_alloc` when memory runs out.
  [[nodiscard]] std::string describe() const { return detail::describe(*core_); }

  /// The instruction set the plan's kernels use.
  [[nodiscard]] isa instruction_set() const noexcept { return detail::instruction_set(*core_); }

private:
  std::shared_ptr<const detail::plan_core> core_;
};

/// A complex-to-complex plan in precision `T` (`double` or `float`).
template <class T> using c2c_plan = plan<std::complex<T>, std::complex<T>>;

/// A real-to-complex plan in precision `T` (`double` or `float`).
template <class T> using r2c_plan = plan<T, std::complex<T>>;

/// A complex-to-real plan in precision `T` (`double` or `float`).
template <class T> using c2r_plan = plan<std::complex<T>, T>;

} // namespace fftn::inline v0
