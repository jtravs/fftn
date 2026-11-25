// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Plan options: choices the caller makes at plan construction besides the shape.
#pragma once

#include "fftn/config.hpp"
#include "fftn/isa.hpp"

#include <cstddef>

namespace fftn::inline v0 {

class wisdom; // fftn/wisdom.hpp

/// How plan construction chooses the program that computes the transform (the planner's
/// effort). Whatever the effort, every program computes the same transform within the same
/// error bounds; only the speed, and the last bits of the results, differ between programs.
///
/// ```cpp
/// // Time the candidates once, then keep the plan for many executions.
/// const fftn::c2c_plan<double> p(fftn::rows_c2c<double>(100000, 1, fftn::direction::forward),
///                                {.effort = fftn::planning::measure});
/// ```
enum class planning : unsigned char {
  /// The planner's rules and cost tables choose: deterministic (the same shape, options,
  /// library version and processor model give the same program) and fast, microseconds to
  /// a few milliseconds. The default.
  rules,
  /// A short measurement: the rules' choice and the model's next few candidates, each
  /// timed briefly at the transform's size; the fastest is kept. Milliseconds for small
  /// transforms; its budget, a tenth of a second by default, is a target that the largest
  /// transforms exceed by the time it takes to build their plans (`measure_seconds`).
  quick,
  /// The full measurement: the model's top candidates (the rules' choice, the strategies
  /// the rules decide between by size thresholds, the cheapest other radix sequences), each
  /// built and timed as a whole program at the transform's size on scratch data, warm and
  /// repeated; the fastest is kept. Up to seconds per plan for large transforms, of the
  /// order of FFTW's FFTW_MEASURE. The rules' choice stays unless another program is at
  /// least 1 % faster.
  measure,
};

/// Options for building a plan. Every member has a default, so `options{}` is the usual
/// choice.
struct options {
  /// The input and output are the same array: every execute call passes the same address
  /// as input and output, and the input is overwritten. The two layouts must then agree
  /// (docs/conventions.md, "In place"); plan construction checks that they do.
  bool in_place = false;

  /// A real factor applied to every output value: the plan computes `scale` times the
  /// transform. Use 1/N (N the number of points in one transform) for a normalised
  /// transform, or 1/sqrt(N) for a unitary one. The multiplication is fused into the pass
  /// that writes the output, so it costs no extra pass over the data.
  ///
  /// Precondition: finite, and for `float` plans 0 or of a magnitude from the smallest
  /// positive `float` to the largest; plan construction refuses other values (NaN,
  /// infinities, values that would overflow or vanish in `float`), naming `options.scale`.
  ///
  /// Guarantees: each output equals the unscaled plan's output multiplied by `scale`
  /// (converted once to the plan's precision), bit for bit; 1 (the default) changes
  /// nothing.
  ///
  /// It is a shortcut for the output edge's scale (`edge::scale`, fftn/edges.hpp): plan
  /// construction refuses a plan that gives both other than 1, naming `edges.out.scale`.
  double scale = 1.0;

  /// The instruction set the plan must use, for tests and benchmarks: `isa::best` (the
  /// default) lets the plan choose the fastest one the processor and the build support.
  /// Forcing one affects only this plan.
  ///
  /// Plan construction throws `std::invalid_argument` naming `options.force_isa` when the
  /// build does not contain that instruction set or the processor cannot run it.
  isa force_isa = isa::best;

  /// How the program is chosen (`planning`). With `planning::quick` or `planning::measure`,
  /// construction executes candidate programs on scratch memory it allocates and frees
  /// itself (never the caller's arrays), serially on the calling thread, and the chosen
  /// program can differ from run to run when candidates are close in speed: record it with
  /// `wisdom::add` to reproduce it.
  ///
  /// Guarantees: whatever the effort, the plan is valid and computes the same transform;
  /// `plan::describe()` says how the program was chosen.
  planning effort = planning::rules;

  /// The time, in seconds, that measured planning aims to spend on this plan, building the
  /// rules' plan and the candidates and timing them; 0 (the default) takes the effort's own
  /// budget (`planning::quick` 0.1 s, `planning::measure` 10 s). It is a target, not a
  /// bound: candidates are admitted in the model's order while half of it is left, each one
  /// admitted is executed at least twice, and the plan chosen is built once more when the
  /// timings used fewer transforms, so a large transform exceeds it by about the time of
  /// one plan's construction and a few executions per candidate (12.5 s for a 10 s budget
  /// at 2^22 points in experiment P2). When only the rules' choice is admitted, nothing is
  /// timed. Ignored by `planning::rules`.
  ///
  /// Precondition: finite and not negative, whatever the effort; plan construction refuses
  /// other values, naming `options.measure_seconds`.
  double measure_seconds = 0;

  /// The most memory, in bytes, that measured planning may allocate besides the plan it
  /// returns: the input and output of its timings (with a copy of the input in place), the
  /// workspace and the candidates' tables; 0 (the default) is 1 GiB. When the shape's data
  /// need more, the candidates are timed on fewer transforms of the batch (outer batch
  /// dimensions first), down to one, and no candidate is admitted whose tables would pass
  /// the limit. The rules' plan and a single transform's data are always allocated, however
  /// large. Ignored by `planning::rules`.
  std::size_t measure_scratch_bytes = 0;

  /// Wisdom to plan from (`fftn::wisdom`), or none. When it holds an entry for this shape
  /// and these options, recorded by this library version on this processor model and
  /// instruction set, the plan rebuilds that entry's program without timing anything,
  /// whatever the effort: the same program as the plan the entry was recorded from, with
  /// bitwise identical results. An entry that would not rebuild the same program (on a
  /// machine whose caches differ, say) is ignored, and the effort decides; `describe()`
  /// says so. The plan reads the wisdom only during construction and keeps no reference.
  ///
  /// ```cpp
  /// fftn::wisdom w; // or fftn::wisdom::load("plans.wisdom")
  /// const auto shape = fftn::rows_c2c<double>(4096, 64, fftn::direction::forward);
  /// const fftn::c2c_plan<double> measured(shape, {.effort = fftn::planning::measure});
  /// w.add(measured);
  /// const fftn::c2c_plan<double> again(shape, {.use = &w}); // the same program, no timing
  /// ```
  const wisdom* use = nullptr;
};

} // namespace fftn::inline v0
