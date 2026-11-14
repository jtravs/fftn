// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Target avx512 (AVX-512 F, DQ and VL with AVX2 and FMA; x86-64 only, guarded here). No ISA
// flag is ever set for this file: the instruction set comes from the region's attribute
// (DESIGN.md 6.4), so the library can be built at any x86-64 baseline and still use
// AVX-512 where the processor has it.

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "gen/codelets.hpp"     // IWYU pragma: keep (the passes read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (table.inl defines a kernel_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx512)

#include "isa/vocab_avx512.inl"

#include "isa/kernels.inl"

#include "gen/complex_codelets.inl"

#include "gen/split_codelets.inl"

#include "passes/passes.inl"

#include "isa/table.inl"

FFTN_ISA_END

#endif
