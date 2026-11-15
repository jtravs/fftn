// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Target avx2: row unit 4 of the codelet table (isa/rows.inl says which rows). Like
// avx2.cpp, the target's main unit, it sets no ISA flag (DESIGN.md 6.4).

#define FFTN_ISA_ROWS 4

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "gen/codelets.hpp"     // IWYU pragma: keep (the passes read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (the vocabulary names its target)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx2)

#include "isa/vocab_avx2.inl"

#include "isa/kernel_text.inl"

#include "isa/rows.inl"

FFTN_ISA_END

#endif
