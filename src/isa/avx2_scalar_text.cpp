// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Experiment X15, arm C (dev/experiments/X15-compiler-vectorised-scalar.md): the scalar
// kernel text, chain passes included, compiled inside the avx2 region (its attribute and
// its namespace) with the scalar vocabulary, so the compiler may use AVX2 and FMA
// registers on it but has no hand-written complex shuffles. Built only with the switch
// FFTN_X_AVX2_SCALAR_TEXT (cmake/FftnExperiments.cmake), and then taken by plans on AVX2
// processors for the chain passes (src/core/plan_core.cpp). Never a default; deleted with
// the experiment's verdict.
//
// The nested namespace keeps this unit's vocabulary and table apart from the avx2
// target's own (avx2.cpp). The table's id is target::scalar, from the scalar vocabulary: its
// storage forms (one complex per register, two reals per twiddle) are the scalar ones.

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "isa/kernel_table.hpp" // IWYU pragma: keep (table.inl defines a kernel_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx2)
namespace scalar_text {

#include "isa/vocab_scalar.inl"

#include "isa/kernels.inl"

#include "passes/chain.inl"

#include "isa/table.inl"

} // namespace scalar_text
FFTN_ISA_END

#endif
