// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Plans: a shape prepared for execution.
#pragma once

#include "fftn/config.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/executor.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/team.hpp"
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
/// own workspace), by a team of the caller's threads sharing one workspace (`fftn::team`),
/// or on an executor (`fftn::executor`).
///
/// Guarantees:
/// - Construction validates the shape and options, and throws `std::invalid_argument`
///   naming the offending argument (for example `shape.axes[1].n`) when they are not
///   valid. It does not touch the caller's arrays.
/// - Execution never allocates, locks, throws or does I/O (on an executor, only the
///   executor's own calls may).
/// - Results are bitwise reproducible for a given plan: every transform of the batch gets
///   the same arithmetic, whatever its position, repeated executions agree bit for bit,
///   and so do serial and team executions, whatever the team's size and timing. Plans built with
///   different options, or on different instruction sets, may differ in the last bits
///   (docs/conventions.md, "Determinism").
/// - Plans are cheap to copy; copies share their immutable tables. There is no empty plan:
///   moving a plan copies it, so the source of a move stays a valid plan too.
template <class In, class Out> class plan {
public:
  /// Builds the plan of `s` with options `opts`.
  ///
  /// Throws `std::invalid_argument` when the shape or the options are not valid (the
  /// message names the argument; the rules are in docs/conventions.md): an axis or batch
  /// count of zero, no axis or more than three axes, overlapping output elements, sizes
  /// that overflow, in-place layouts that do not agree, or a negative or non-finite
  /// `options::measure_seconds`. Throws `std::bad_alloc` when memory runs out, including
  /// the scratch memory of measured planning (`options::effort`).
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
    detail::execute_all(*core_, in, out, ws);
  }

  /// Transforms `count` transforms of the batch, starting with transform `first`, serially
  /// on the calling thread: for callers that share a batch out themselves.
  ///
  /// A transform is one row, plane or volume, as the shape has one, two or three axes.
  /// Transforms are numbered in batch order (the last batch dimension varies fastest).
  /// `in` and `out` are the same addresses as for `execute`: element 0 of transform 0, not
  /// of transform `first`.
  ///
  /// Preconditions: as for `execute`, and `first + count` is at most the number of
  /// transforms in the batch.
  ///
  /// Guarantees: as for `execute`; each transform's result is bitwise identical to what
  /// `execute` gives it.
  void execute_transforms(std::size_t first, std::size_t count, const In* in, Out* out,
                          workspace& ws) const noexcept {
    detail::execute_core(*core_, first, count, in, out, ws);
  }

  /// Transforms every transform of the batch, shared among the members of a team: every
  /// member calls this with its own `t` and the same arguments otherwise.
  ///
  /// With `completion::all` the members claim work dynamically: whole transforms first,
  /// then the pieces of transforms that the plan splits among the members of one cache
  /// group (docs/threading.md). The call returns on each member once the whole output is
  /// written. It does not wait for members that have not arrived: the members present do
  /// everything, and members that arrive late find nothing left. With `completion::own`
  /// each member transforms exactly `transforms_of(t)`, whole, and returns when they are
  /// written.
  ///
  /// Preconditions: as for `execute`, plus those of `fftn::team`; every member passes the
  /// same `in`, `out` and `done`, and the same workspace object (not another workspace over
  /// the same memory); `ws.size()` is at least `workspace_bytes(t.size)`. The team
  /// executions a workspace serves form one sequence, which every member follows in the
  /// same order: a member may arrive late, even after the others have moved on to later
  /// executions, but must not skip one, because a member counts its own calls to know which
  /// execution it is in (a member that skipped one would return from the next before its
  /// output is written). A member that is absent from an execution must therefore make no
  /// more team calls on the workspace until it has been reset: any serial execution on it
  /// resets it, `execute_transforms(0, 0, in, out, ws)` among them. One rank's successive
  /// calls are made by one thread, or ordered by happens-before (a mutex, a barrier, a
  /// thread join). A team execution lasts until its last member has returned (or, for
  /// `completion::own`, until the caller has synchronised the members); only then may the
  /// workspace serve a serial execution or a team of another size.
  ///
  /// Guarantees: as for `execute`, and the output is bitwise identical to serial
  /// execution, whatever the team's size, grouping or timing. Never allocates, locks,
  /// throws or does I/O. A member that finds nothing to do waits for the others by
  /// spinning briefly, then yielding, then blocking in `std::atomic::wait`; it never
  /// waits for a member that has not arrived.
  void execute(const In* in, Out* out, const team& t, workspace& ws,
               completion done = completion::all) const noexcept {
    detail::execute_team(*core_, in, out, t, ws, done);
  }

  /// Transforms every transform of the batch on executor `ex`: anything that can run n
  /// tasks and return when they are done (`fftn::executor`), such as `fftn::thread_pool`,
  /// `fftn::omp::executor` or an adapter over another pool. The call returns when the whole
  /// output is written.
  ///
  /// FFTN decides how many of the executor's `concurrency()` lanes the plan can use, and
  /// asks the executor to run one task per lane that has work, as a team (the team overload
  /// below, with `ex.group_size()` members per cache group). A job too small to share runs
  /// on the calling thread without calling the executor, and so does every job when
  /// `ex.concurrency()` is 1 (as executors report from inside their own tasks). The tasks
  /// may run in any order and any number at a time: none waits for one that has not
  /// started.
  ///
  /// Preconditions: as for `execute`; `ex` meets the requirements of `fftn::executor`
  /// (among them the ordering of `bulk` and its tasks, and that a throwing `bulk` throws
  /// only once every task it started has returned); `ws.size()` is at least
  /// `workspace_bytes(ex.concurrency())`, and `ws` serves no other execution meanwhile.
  ///
  /// Guarantees: as for `execute`, and the output is bitwise identical to serial execution,
  /// whatever the executor, its concurrency and the order its tasks run in. FFTN itself
  /// never allocates, locks, throws or does I/O here; the executor may. If `ex.concurrency()`
  /// or `ex.bulk` throws, the exception propagates, the output is unspecified and the
  /// workspace is left ready for its next execution.
  void execute(const In* in, Out* out, executor_ref ex, workspace& ws) const {
    detail::execute_on(*core_, in, out, ex, ws);
  }

  /// The transforms a member of team `t` owns: those it transforms with `completion::own`.
  /// The members' ranges are contiguous, in rank order, and together cover the batch once;
  /// they differ in length by at most one transform, and members beyond the batch's size
  /// own none. Callers use them to run their own work on the same transforms as the same
  /// thread (owner-computes).
  ///
  /// Preconditions: those of `fftn::team`.
  [[nodiscard]] transform_range transforms_of(const team& t) const noexcept {
    return detail::transforms_of(*core_, t);
  }

  /// The bytes of workspace an execution of this plan needs when run by teams of up to
  /// `lanes` members; 0 when it needs none. With `lanes` 1 (or less) this is the workspace
  /// of a serial execution; larger teams also need a scratch area per member and the
  /// team's synchronisation state.
  [[nodiscard]] std::size_t workspace_bytes(int lanes = 1) const noexcept {
    return detail::workspace_bytes(*core_, lanes);
  }

  /// The plan in human-readable form: the transform, the instruction set, the processor
  /// vendor the plan was made for (some rules apply to one vendor's processors only), the
  /// strategy, the radices and passes of every axis, the loops over the batch, and the
  /// workspace.
  /// For people, not for parsing: the format may change between versions.
  ///
  /// Throws `std::bad_alloc` when memory runs out.
  [[nodiscard]] std::string describe() const { return detail::describe(*core_); }

  /// The instruction set the plan's kernels use: the one `options::force_isa` named, or the
  /// one the plan chose. Never `isa::best`.
  [[nodiscard]] isa instruction_set() const noexcept { return detail::instruction_set(*core_); }

private:
  std::shared_ptr<const detail::plan_core> core_;
  friend class wisdom; // wisdom::add reads the program
};

/// A complex-to-complex plan in precision `T` (`double` or `float`).
template <class T> using c2c_plan = plan<std::complex<T>, std::complex<T>>;

/// A real-to-complex plan in precision `T` (`double` or `float`).
template <class T> using r2c_plan = plan<T, std::complex<T>>;

/// A complex-to-real plan in precision `T` (`double` or `float`).
template <class T> using c2r_plan = plan<std::complex<T>, T>;

} // namespace fftn::inline v0
