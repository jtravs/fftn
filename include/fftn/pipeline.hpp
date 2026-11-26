// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Row pipelines: a transform, a pointwise middle and the inverse transform over one axis, in
/// one sweep, the spectrum never reaching memory (dev/DESIGN.md 12.2).
///
/// Provisional (tasks 5.2b and 5.2c, 2026-11-26): this API follows dev/DESIGN.md section 12.2's
/// API block with the proposals of its section 18.2 ("Pipelines"), which the maintainer has yet to
/// settle; it may change before the first release. Built so far: `pipeline<C, C, C>` with the
/// forward transform first and `pipeline<T, C, T>` (r2c, then c2r), with their prologue, middle
/// and epilogue vocabulary (scales, real and complex factors, accumulation), one inverse per
/// forward, and every line stored whole.
/// docs/conventions.md ("Pipelines") states what a pipeline computes.
#pragma once

#include "fftn/config.hpp"
#include "fftn/detail/core.hpp"
#include "fftn/edges.hpp"
#include "fftn/executor.hpp"
#include "fftn/isa.hpp"
#include "fftn/options.hpp"
#include "fftn/shape.hpp"
#include "fftn/team.hpp"
#include "fftn/workspace.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace fftn::inline v0 {

/// What a pipeline computes, before it is built (`pipeline`): one transformed axis, the line,
/// under any number of batch dimensions, each line taken through
///
///     prologue -> transform -> middle -> inverse transform -> epilogue
///
/// and stored once. The element types select the kind: `pipeline_desc<C, C, C>` (C =
/// `std::complex<T>`) is two complex transforms, `pipeline_desc<T, C, T>` an r2c then a c2r
/// transform (a real convolution) and `pipeline_desc<C, T, C>` a c2r then an r2c transform, for
/// `T` = `double` or `float`. Other combinations do not compile.
///
/// The prologue and the epilogue are edges (fftn/edges.hpp) on the input and the output, indexed
/// by the line's logical position and the batch: their factors' `axis_strides` hold one stride,
/// along the line, and their `batch_strides` one per batch dimension or none. On the real input
/// and output of a `<T, C, T>` pipeline they take real factors only. The middle multiplies the
/// spectrum between the transforms by its `scale` and its factor, indexed by the bin along the
/// line and the batch in the same way: the kernel of a convolution, whose array each execution
/// passes in `exec_args::mid_factor`. The spectrum has n bins for `<C, C, C>` and the n/2 + 1
/// bins of the r2c transform for `<T, C, T>` (bins 0 to n/2), whose c2r transform ignores the
/// imaginary parts of bin 0 and, for even n, of bin n/2, as a c2r plan does.
/// Nothing is normalised: a round trip multiplies by n unless a scale or the kernel says
/// otherwise. The scales of the prologue, the middle, the epilogue and `options::scale` are
/// applied as one product.
///
/// Provisional (tasks 5.2b to 5.2d). A pipeline refuses, naming the field, what is not built yet:
/// the inverse transform first (task 5.3), `fanout` above 1 (task 5.2e), the kind `<C, T, C>`
/// (task 5.3), and stored runs on an in-place pipeline or beside a prologue or epilogue factor.
template <class In, class Mid, class Out> struct pipeline_desc {
  static_assert(
      (std::is_same_v<In, std::complex<double>> && std::is_same_v<Mid, std::complex<double>> &&
       std::is_same_v<Out, std::complex<double>>) ||
          (std::is_same_v<In, double> && std::is_same_v<Mid, std::complex<double>> &&
           std::is_same_v<Out, double>) ||
          (std::is_same_v<In, std::complex<double>> && std::is_same_v<Mid, double> &&
           std::is_same_v<Out, std::complex<double>>) ||
          (std::is_same_v<In, std::complex<float>> && std::is_same_v<Mid, std::complex<float>> &&
           std::is_same_v<Out, std::complex<float>>) ||
          (std::is_same_v<In, float> && std::is_same_v<Mid, std::complex<float>> &&
           std::is_same_v<Out, float>) ||
          (std::is_same_v<In, std::complex<float>> && std::is_same_v<Mid, float> &&
           std::is_same_v<Out, std::complex<float>>),
      "fftn::pipeline_desc<In, Mid, Out>: the kinds are <C, C, C>, <T, C, T> and <C, T, C> with "
      "C = std::complex<T>, in double or float");

  /// The direction of the first transform of a `<C, C, C>` pipeline; the second has the other.
  /// `direction::forward` makes a convolution; `direction::backward` (a spectral pair) is
  /// refused until task 5.3. The real kinds ignore it.
  direction first = direction::forward;
  /// The line: its logical length `n` (at least 1), the stride of its input elements
  /// (`in_stride`, in `In` elements) and of its output elements (`out_stride`, in `Out`
  /// elements); 0 for the tight default, 1.
  axis line{};
  /// The batch dimensions, slowest first, with the distances between the first input elements
  /// of consecutive lines (`in_dist`, in `In` elements) and between the first output elements
  /// (`out_dist`, in `Out` elements); 0 for the tight default. None means a single line. Lines
  /// are numbered in batch order (the last batch dimension varies fastest).
  std::vector<batch> batches{};
  /// Stored runs of the line (`stored_run`, axis 0; fftn/shape.hpp), on the input (`side::in`)
  /// or the output (`side::out`), validated as a shape's runs: input elements outside the input's
  /// runs are zero and not read, output elements outside the output's runs are not written, and
  /// each side's default distances and reach are by its stored extent (the end of its last run).
  /// The logical positions are those of the line's n elements on both sides, for `<T, C, T>` too.
  /// Refused on an in-place pipeline and beside a prologue or epilogue factor (task 5.2d).
  std::vector<stored_run> runs{};
  /// The input edge: a scale and a factor applied to each input element before the first
  /// transform (`accumulate` is refused here; a block callback, `call`, until task 5.3d).
  edge<In> prologue{};
  /// The middle: what multiplies each bin of the spectrum between the two transforms.
  struct {
    /// A real factor; finite, and for `float` pipelines 0 or of a magnitude from the smallest
    /// positive `float` to the largest. Applied as one product with the other scales.
    double scale = 1.0;
    /// The middle's factor indexed by bin: `Mid` values with `factor_kind::complex`, or real `T`
    /// values with `factor_kind::real`. `axis_strides` holds one stride (the stride between
    /// consecutive bins, in the array's elements), `batch_strides` one per batch dimension (0
    /// for a kernel shared along it) or none (one kernel for every line). Its array arrives with
    /// each execution (`exec_args::mid_factor`).
    factor mul{};
  } middle{}; ///< The middle: what multiplies each bin of the spectrum between the transforms.
  /// The output edge: a scale and a factor applied to each output element after the second
  /// transform, then stored, or added to the output's value with `accumulate` (a block
  /// callback, `call`, is refused until task 5.3d).
  edge<Out> epilogue{};
  /// How many inverse transforms run on each middle spectrum, each with its own middle factor:
  /// 1 (more is refused until task 5.2e).
  std::size_t fanout = 1;
  /// With `fanout` above 1, the distance between copies' outputs, in `Out` elements (task 5.2e).
  std::ptrdiff_t fanout_out_dist = 0;
  /// With `fanout` above 1, the distance between copies' middle factors, in the array's elements
  /// (task 5.2e).
  std::ptrdiff_t fanout_factor_dist = 0;
};

namespace detail {
// The helpers' batch dimension: `count` rows one after another, tight (distances 0, the
// defaults) or, with runs, their stored extents on each side (the end of the side's last run, or
// the whole line where the side has none).
[[nodiscard]] inline batch rows_batch(std::size_t n, std::size_t count,
                                      std::span<const stored_run> runs) {
  std::size_t in_extent = 0;
  std::size_t out_extent = 0;
  for (const stored_run& r : runs) {
    std::size_t& e = r.where == side::in ? in_extent : out_extent;
    e = std::max(e, r.stored + r.count);
  }
  const auto dist = [n](std::size_t extent) {
    return static_cast<std::ptrdiff_t>(extent != 0 ? extent : n);
  };
  return batch{
      .count = count,
      .in_dist = runs.empty() ? 0 : dist(in_extent),
      .out_dist = runs.empty() ? 0 : dist(out_extent),
  };
}
} // namespace detail

/// A batch of `count` complex rows of length `n`, stored one after another, convolved: the
/// forward transform, a complex middle factor (one kernel spectrum of `n` bins, stride 1,
/// shared by every row: `middle.mul.batch_strides` empty), and the backward transform. Not
/// normalised: fold 1/n into the kernel or `middle.scale`.
///
/// `runs`: stored runs of the rows (axis 0); the rows' distances are then their stored extents
/// on each side, the end of their last run (the zero-padded convolution's valid half, say).
///
/// Throws `std::bad_alloc` when memory runs out. Guarantees: nothing is validated until a
/// pipeline is built from the description. Provisional (task 5.2b).
template <class T>
[[nodiscard]] pipeline_desc<std::complex<T>, std::complex<T>, std::complex<T>>
complex_convolution_rows(std::size_t n, std::size_t count, std::span<const stored_run> runs = {}) {
  pipeline_desc<std::complex<T>, std::complex<T>, std::complex<T>> d;
  d.first = direction::forward;
  d.line = axis{.n = n};
  d.runs.assign(runs.begin(), runs.end());
  d.batches.push_back(detail::rows_batch(n, count, runs));
  d.middle.mul = factor{.kind = factor_kind::complex, .axis_strides = {1}, .batch_strides = {}};
  return d;
}

/// A batch of `count` real rows of length `n`, stored one after another, convolved: the r2c
/// transform, a complex middle factor (one kernel spectrum of the n/2 + 1 bins, stride 1, shared
/// by every row: `middle.mul.batch_strides` empty), and the c2r transform. Each row is replaced by
/// its circular convolution with the kernel whose r2c transform the spectrum is, times n: fold
/// 1/n into the kernel or `middle.scale`. A linear (zero-padded) convolution of a row of m values
/// with a kernel of L taps is the circular one of a length n of at least m + L - 1, the row and
/// the kernel padded with zeros; runs keep those zeros out of memory (an input run of the m values
/// from logical 0, and an output run of the outputs wanted).
///
/// `runs`: as for `complex_convolution_rows`. Exceptions, guarantees and status: as for
/// `complex_convolution_rows` (task 5.2c).
template <class T>
[[nodiscard]] pipeline_desc<T, std::complex<T>, T>
convolution_rows(std::size_t n, std::size_t count, std::span<const stored_run> runs = {}) {
  pipeline_desc<T, std::complex<T>, T> d;
  d.line = axis{.n = n};
  d.runs.assign(runs.begin(), runs.end());
  d.batches.push_back(detail::rows_batch(n, count, runs));
  d.middle.mul = factor{.kind = factor_kind::complex, .axis_strides = {1}, .batch_strides = {}};
  return d;
}

/// A batch of `count` complex rows of length `n`, stored one after another, as a spectral pair:
/// the transform of direction `first`, the middle (none: set it), and the transform of the
/// other direction. With `first` = `direction::forward` this is a convolution without its
/// kernel; `complex_convolution_rows` names that case. Pipelines refuse
/// `first` = `direction::backward` until task 5.3.
///
/// `runs`: as for `complex_convolution_rows`. Exceptions, guarantees and status: as for
/// `complex_convolution_rows`.
template <class T>
[[nodiscard]] pipeline_desc<std::complex<T>, std::complex<T>, std::complex<T>>
spectral_pair_rows(std::size_t n, std::size_t count, direction first,
                   std::span<const stored_run> runs = {}) {
  pipeline_desc<std::complex<T>, std::complex<T>, std::complex<T>> d =
      complex_convolution_rows<T>(n, count, runs);
  d.first = first;
  d.middle.mul = factor{};
  return d;
}

/// A row pipeline (`pipeline_desc`) prepared for execution, as a plan is a prepared transform:
/// immutable, cheap to copy, executed on any arrays of the described layout by any number of
/// threads at once (each with its own workspace), by a team of the caller's threads sharing one
/// workspace, or on an executor.
///
/// Each line is read once, transformed into the workspace, given the middle, transformed back
/// and stored once: the spectrum never reaches memory. The execute calls have the plan's forms
/// (fftn/plan.hpp), each with the data of the execution (`exec_args`): the prologue's factor
/// array in `in_factor`, the middle's in `mid_factor` and the epilogue's in `out_factor`. There
/// are no overloads without `exec_args`, since a convolution always reads its kernel.
///
/// Guarantees:
/// - Construction validates the description and the options, and throws
///   `std::invalid_argument` naming the offending field (for example `desc.middle.mul.kind`).
///   It does not touch the caller's arrays.
/// - Execution never allocates, locks, throws or does I/O (on an executor, only the executor's
///   own calls may).
/// - Each line's result equals the transform of the input times the prologue, times the middle,
///   transformed back and times the epilogue, within two transforms' error bound plus one
///   rounding per multiplication and addition (dev/DESIGN.md 15.5); it is not necessarily
///   bitwise equal to the same work done by separate plans and loops. Results are bitwise
///   identical between serial, `execute_transforms` pieces, team and executor executions.
///
/// Options: `scale` (a shortcut for `epilogue.scale`, refused when both are other than 1),
/// `force_isa` and `in_place` (allowed without accumulation: each line lives in the workspace
/// between its read and its store). `effort` and `use` are ignored: pipelines are planned by
/// the rules until task 5.2g, and `describe()` says so.
///
/// Provisional (tasks 5.2b to 5.2d): see `pipeline_desc`.
template <class In, class Mid, class Out> class pipeline {
public:
  /// Builds the pipeline `desc` with options `opts`.
  ///
  /// Throws `std::invalid_argument` naming the field: a length or count of 0, overlapping
  /// output elements, sizes that overflow, in-place layouts that do not agree, an edge or
  /// middle that is not valid (as `plan`'s edges are judged, with names such as
  /// `desc.epilogue.mul.batch_strides`; a complex factor on the real input or output of a
  /// `<T, C, T>` pipeline names `desc.prologue.mul.kind` or `desc.epilogue.mul.kind`),
  /// `accumulate` on the prologue or on an in-place pipeline, both `options::scale` and
  /// `desc.epilogue.scale` other than 1, a product of scales not representable in the
  /// precision, a negative or non-finite `options::measure_seconds`, or what is not built yet
  /// (`pipeline_desc`). Throws `std::bad_alloc` when memory runs out.
  explicit pipeline(const pipeline_desc<In, Mid, Out>& desc, const options& opts = {})
      : core_(detail::make_pipeline_core(view_of(desc), opts)) {}

  /// Makes a pipeline that shares `other`'s immutable tables; also used for moves, which
  /// leave the source unchanged and valid.
  pipeline(const pipeline& other) noexcept = default;

  /// Makes this pipeline share `other`'s immutable tables; also used for move assignment.
  pipeline& operator=(const pipeline& other) noexcept = default;

  /// Releases this pipeline's share of the tables.
  ~pipeline() = default;

  /// Runs every line of the batch, serially on the calling thread.
  ///
  /// `in` and `out` address element 0 of the first line's input and output; in place they are
  /// the same address, otherwise the arrays must not overlap. `args` carries the factor arrays
  /// the description has (`exec_args`): an execution without one it needs ends the program.
  ///
  /// Preconditions: the arrays hold the described layout; `ws.size()` is at least
  /// `workspace_bytes()`, and `ws` is used by no other execution meanwhile.
  ///
  /// Guarantees: the input and the factor arrays are only read unless the pipeline is in
  /// place; nothing outside the described output elements is written. Never allocates, locks,
  /// throws or does I/O.
  void execute(const In* in, Out* out, workspace& ws, const exec_args& args) const noexcept {
    detail::execute_all(*core_, in, out, ws, args);
  }

  /// Runs `count` lines of the batch, starting with line `first`, serially on the calling
  /// thread: for callers that share a batch out themselves. `in`, `out` and the factor
  /// arrays address line 0, as for `execute`.
  ///
  /// Preconditions: as for `execute`, and `first + count` is at most the number of lines.
  /// Guarantees: as for `execute`; each line's result is bitwise identical to what `execute`
  /// gives it.
  void execute_transforms(std::size_t first, std::size_t count, const In* in, Out* out,
                          workspace& ws, const exec_args& args) const noexcept {
    detail::execute_core(*core_, first, count, in, out, ws, args);
  }

  /// Runs every line of the batch, shared among the members of a team, with
  /// `completion::all`: as the team `plan::execute` (fftn/plan.hpp), whose preconditions and
  /// guarantees hold here; every member passes the same arguments but `t`.
  /// `ws.size()` is at least `workspace_bytes(t.size)`.
  void execute(const In* in, Out* out, const team& t, workspace& ws,
               const exec_args& args) const noexcept {
    detail::execute_team(*core_, in, out, t, ws, completion::all, args);
  }

  /// The team `execute` with an explicit `completion`: with `completion::own` each member runs
  /// exactly the lines `transforms_of(t)`, whole. Preconditions and guarantees: those of the
  /// team `plan::execute`.
  void execute(const In* in, Out* out, const team& t, workspace& ws, completion done,
               const exec_args& args) const noexcept {
    detail::execute_team(*core_, in, out, t, ws, done, args);
  }

  /// Runs every line of the batch on executor `ex`, returning when the whole output is written:
  /// as the executor `plan::execute`, whose preconditions, exceptions and guarantees hold here.
  /// `ws.size()` is at least `workspace_bytes(ex.concurrency())`.
  void execute(const In* in, Out* out, executor_ref ex, workspace& ws,
               const exec_args& args) const {
    detail::execute_on(*core_, in, out, ex, ws, args);
  }

  /// The lines a member of team `t` owns: those it runs with `completion::own`, contiguous
  /// ranges in rank order covering the batch once, as `plan::transforms_of` deals transforms.
  ///
  /// Preconditions: those of `fftn::team`.
  [[nodiscard]] transform_range transforms_of(const team& t) const noexcept {
    return detail::transforms_of(*core_, t);
  }

  /// The bytes of workspace an execution needs when run by teams of up to `lanes` members (1:
  /// serially); the workspace holds the scratch-resident line.
  [[nodiscard]] std::size_t workspace_bytes(int lanes = 1) const noexcept {
    return detail::workspace_bytes(*core_, lanes);
  }

  /// The pipeline in human-readable form: its kind, length and batch, the instruction set,
  /// how it is built (the pair of transforms, the program form a line runs in, where each
  /// operation goes), the workspace and the program. For people, not for parsing.
  ///
  /// Throws `std::bad_alloc` when memory runs out.
  [[nodiscard]] std::string describe() const { return detail::describe(*core_); }

  /// The instruction set the pipeline's kernels use: the one `options::force_isa` named, or
  /// the one chosen. Never `isa::best`.
  [[nodiscard]] isa instruction_set() const noexcept { return detail::instruction_set(*core_); }

private:
  static detail::pipeline_view view_of(const pipeline_desc<In, Mid, Out>& d) noexcept {
    return {
        .kind = detail::pipeline_kind_of<In, Mid, Out>::value,
        .prec = detail::precision_of<In, Out>,
        .first = d.first,
        .line = d.line,
        .batches = d.batches,
        .runs = d.runs,
        .prologue = detail::view_of(d.prologue),
        .middle =
            {
                .scale = d.middle.scale,
                .kind = d.middle.mul.kind,
                .axis_strides = d.middle.mul.axis_strides,
                .batch_strides = d.middle.mul.batch_strides,
            },
        .epilogue = detail::view_of(d.epilogue),
        .fanout = d.fanout,
        .fanout_out_dist = d.fanout_out_dist,
        .fanout_factor_dist = d.fanout_factor_dist,
    };
  }

  std::shared_ptr<const detail::plan_core> core_;
};

} // namespace fftn::inline v0
