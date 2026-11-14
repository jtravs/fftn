// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Plan options: choices the caller makes at plan construction besides the shape.
#pragma once

#include "fftn/config.hpp"
#include "fftn/isa.hpp"

namespace fftn::inline v0 {

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
  /// Precondition: finite; plan construction refuses NaN and infinities.
  ///
  /// Guarantees: each output equals the unscaled plan's output multiplied by `scale`
  /// (converted once to the plan's precision), bit for bit; 1 (the default) changes
  /// nothing.
  double scale = 1.0;

  /// The instruction set the plan must use, for tests and benchmarks: `isa::best` (the
  /// default) lets the plan choose the fastest one the processor and the build support.
  /// Forcing one affects only this plan.
  ///
  /// Plan construction throws `std::invalid_argument` naming `options.force_isa` when the
  /// build does not contain that instruction set or the processor cannot run it.
  isa force_isa = isa::best;
};

} // namespace fftn::inline v0
