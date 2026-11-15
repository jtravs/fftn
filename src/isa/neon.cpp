// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Target neon (Advanced SIMD alone; AArch64 only, guarded here). No ISA flag is ever set
// for this file, and its region carries no attribute: Advanced SIMD is part of the AArch64
// base architecture, so the baseline of every AArch64 build has it (region.hpp).
//
// The target's main unit: its function table and the passes outside the codelet rows.
// The rows are compiled in rows1_neon.cpp to rows5_neon.cpp (isa/rows.inl), so that no
// unit needs more memory to compile than a CI runner has (DESIGN.md 6.4).

#include "isa/arch.hpp"

#if FFTN_ARCH_AARCH64

#include "gen/codelets.hpp"     // IWYU pragma: keep (the passes read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (table.inl defines a kernel_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(neon)

#include "isa/vocab_neon.inl"

#include "isa/kernel_text.inl"

#include "isa/table.inl"

FFTN_ISA_END

#endif
