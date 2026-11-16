// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Target avx512: row unit 3 of the codelet table (isa/rows.inl says which rows). Like
// avx512.cpp, the target's main unit, it sets no ISA flag (DESIGN.md 6.4).

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#define FFTN_ISA_ROWS 3

#include "gen/codelets.hpp"     // IWYU pragma: keep (the passes read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (the vocabulary names its target)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx512)

#include "isa/vocab_avx512.inl"

#include "isa/kernel_text.inl"

#include "isa/rows.inl"

FFTN_ISA_END

#endif
