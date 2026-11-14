// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Instruction sets: the kernel families a plan can run on.
#pragma once

#include "fftn/config.hpp"

namespace fftn::inline v0 {

/// An instruction-set target that FFTN has kernels for. A plan chooses one when it is
/// constructed (the fastest that both the processor and the build support for the plan's
/// transform, unless `options::force_isa` names one) and reports it through
/// `plan::instruction_set()`.
enum class isa : unsigned char {
  scalar,    ///< portable code with no vector instructions; runs on every processor
  x86_v2,    ///< x86-64, SSE2 to SSE4.2 (the x86-64-v2 level)
  avx,       ///< x86-64, 256-bit AVX without AVX2 or FMA
  avx2,      ///< x86-64, AVX2 with FMA (the x86-64-v3 level)
  avx512,    ///< x86-64, AVX-512 F, DQ and VL (the x86-64-v4 level)
  neon,      ///< AArch64, Advanced SIMD
  neon_fcma, ///< AArch64, Advanced SIMD with the Armv8.3 complex instructions (FCMA)
  best,      ///< not a target: in `options::force_isa`, let the plan choose (the default)
};

/// The instruction set's name as written in this header: "scalar", "avx2", "neon_fcma"
/// and so on ("best" for `isa::best`); "unknown" for a value that is not one of the
/// enumerators.
///
/// Guarantees: the pointer refers to a static string and is never null.
[[nodiscard]] const char* isa_name(isa target) noexcept;

} // namespace fftn::inline v0
