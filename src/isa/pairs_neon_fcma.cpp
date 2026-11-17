// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The pair leaves of the neon_fcma target (experiment X37; passes/pair_leaf.inl,
// kernel_table.hpp): short rows of n = r1 m points that are not a codelet, as their two
// chain passes held in registers. One complex double per register (C = 1), so a row takes
// n of the 32 registers; the factors are those of the chain the rules choose (rules.cpp),
// so a row keeps its results. Like every kernel unit, it sets no ISA flag (DESIGN.md 6.4).

#include "isa/arch.hpp"

#if FFTN_ARCH_AARCH64

#include "gen/codelets.hpp"     // IWYU pragma: keep (the codelets read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (defines a pair_leaf_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(neon_fcma)
namespace pairs {

#include "isa/vocab_neon_fcma.inl"

#include "isa/kernels.inl"

#include "gen/complex_codelets.inl"

#include "passes/access.inl"

#include "passes/pair_leaf.inl"

#include "isa/pairs_neon.inl"

} // namespace pairs
FFTN_ISA_END

#endif
