// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Target neon: row unit 2 of the codelet table (isa/rows.inl says which rows). Like
// neon.cpp, the target's main unit, it sets no ISA flag (DESIGN.md 6.4).

#include "isa/arch.hpp"

#if FFTN_ARCH_AARCH64

#define FFTN_ISA_ROWS 2

#include "gen/codelets.hpp"     // IWYU pragma: keep (the passes read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (the vocabulary names its target)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(neon)

#include "isa/vocab_neon.inl"

#include "isa/kernel_text.inl"

#include "isa/rows.inl"

FFTN_ISA_END

#endif
