// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The batch leaves of the avx512 target for radices 32 and 64 (experiment X53;
// passes/batch_leaf.inl, isa/batch.inl, kernel_table.hpp): C rows of a batch at once, one row per
// lane. Like every kernel unit, it sets no ISA flag (DESIGN.md 6.4).

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#define FFTN_BATCH_PART 2

#include "gen/codelets.hpp"     // IWYU pragma: keep (the codelets read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (defines a batch_leaf_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx512)
namespace batch {

#include "isa/vocab_avx512.inl"

#include "isa/kernels.inl"

#include "gen/complex_codelets.inl"

#include "passes/access.inl"

#include "passes/batch_leaf.inl"

#include "isa/batch.inl"

} // namespace batch
FFTN_ISA_END

#endif
