// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The pair leaves of the neon target (experiment X37): as pairs_neon_fcma.cpp, with the
// vocabulary of NEON without FCMA. Like every kernel unit, it sets no ISA flag (DESIGN.md
// 6.4).

#include "isa/arch.hpp"

#if FFTN_ARCH_AARCH64

#include "gen/codelets.hpp"     // IWYU pragma: keep (the codelets read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (defines a pair_leaf_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(neon)
namespace pairs {

#include "isa/vocab_neon.inl"

#include "isa/kernels.inl"

#include "gen/complex_codelets.inl"

#include "passes/access.inl"

#include "passes/pair_leaf.inl"

#include "isa/pairs_neon.inl"

} // namespace pairs
FFTN_ISA_END

#endif
