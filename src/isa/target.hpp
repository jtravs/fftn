// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The instruction-set targets FFTN generates kernels for (DESIGN.md 6.1), and the choice of
// one from a processor's features. Chosen once per plan, at construction; never per call.
// Every function here is pure: the features are an argument, never detected here, so tests
// can hand in any processor.
#pragma once

#include "platform/cpu_features.hpp"

#include <optional>
#include <string_view>

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

// Whether this build contains code for t: scalar everywhere, avx2 on x86-64, neon and
// neon_fcma on AArch64. x86_v2, avx and avx512 arrive with the rest of phase 1.3.
[[nodiscard]] bool compiled_in(target t) noexcept;

// Whether a processor with these features can run t's code: every feature t's attribute
// string enables that its code may use is present (region.hpp); for avx512 that is
// AVX-512 F, DQ and VL with AVX2 and FMA (the operating system saving the ZMM state is part
// of each AVX-512 flag, cpu_features.hpp), for neon_fcma FCMA and pointer authentication.
// neon needs Advanced SIMD alone, so an AArch64 processor without either of those runs
// neon.
[[nodiscard]] bool runs_on(target t, const cpu_features& features) noexcept;

// The fastest target this build contains that the features allow.
[[nodiscard]] target best_target(const cpu_features& features) noexcept;

// The target a plan uses: best_target(features), or `forced` when given. Forcing a lower
// target is for tests and benchmarks and affects only the plan that asks for it. Throws
// std::invalid_argument naming `forced` when that target is not compiled in or the
// processor cannot run it.
[[nodiscard]] target choose_target(const cpu_features& features, std::optional<target> forced);

// The target's name as written in the code ("avx2", "neon_fcma", ...), and back. The name
// of a value outside the enumeration is "unknown".
[[nodiscard]] std::string_view target_name(target t) noexcept;
[[nodiscard]] std::optional<target> target_from_name(std::string_view name) noexcept;

} // namespace fftn::inline v0::detail
