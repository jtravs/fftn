// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Plan options: choices the caller makes at plan construction besides the shape.
#pragma once

#include "fftn/config.hpp"

namespace fftn::inline v0 {

/// Options for building a plan. Every member has a default, so `options{}` is the usual
/// choice.
struct options {
  /// The input and output are the same array: every execute call passes the same address
  /// as input and output, and the input is overwritten. The two layouts must then agree
  /// (docs/conventions.md, "In place"); plan construction checks that they do.
  bool in_place = false;
};

} // namespace fftn::inline v0
