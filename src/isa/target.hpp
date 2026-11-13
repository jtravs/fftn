// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The instruction-set targets FFTN generates kernels for (DESIGN.md 6.1), and the choice of
// one from a processor's features. Chosen once per plan, at construction; never per call.
#pragma once

#include "platform/cpu_features.hpp"

namespace fftn::inline v0::detail {

enum class target : unsigned char {
  scalar,
  x86_v2,
  avx,
  avx2,
  avx512,
  neon,
  neon_fcma,
};

// The fastest target this build contains that the features allow (task 0.3).
[[nodiscard]] target best_target(const cpu_features& features) noexcept;

} // namespace fftn::inline v0::detail
