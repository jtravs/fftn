// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The pair leaves of the avx2 target (experiment X28b; passes/pair_leaf.inl,
// kernel_table.hpp): short rows of n = r1 m points, C dividing m, as their two chain passes
// held in registers. Lengths whose n / C registers come near AVX2's 16. Like every kernel
// unit, it sets no ISA flag (DESIGN.md 6.4).

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "gen/codelets.hpp"     // IWYU pragma: keep (the codelets read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (defines a pair_leaf_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx2)
namespace pairs {

#include "isa/vocab_avx2.inl"

#include "isa/kernels.inl"

#include "gen/complex_codelets.inl"

#include "passes/access.inl"

#include "passes/pair_leaf.inl"

// C = 2 in double: m = 2 (one unit) up to 32 points, then m = 4. 18, 22 and 26 (X37) are
// also what avx512 plans take, whose C = 4 divides none of them (find_more_pair_leaves).
#define FFTN_PAIRS_F64                                                                             \
  FFTN_PAIR(double, 9, 2), FFTN_PAIR(double, 10, 2), FFTN_PAIR(double, 11, 2),                     \
      FFTN_PAIR(double, 12, 2), FFTN_PAIR(double, 13, 2), FFTN_PAIR(double, 15, 2),                \
      FFTN_PAIR(double, 16, 2), FFTN_PAIR(double, 9, 4), FFTN_PAIR(double, 10, 4),                 \
      FFTN_PAIR(double, 12, 4)
// C = 4 in float: m = 4, one unit.
#define FFTN_PAIRS_F32                                                                             \
  FFTN_PAIR(float, 5, 4), FFTN_PAIR(float, 6, 4), FFTN_PAIR(float, 8, 4), FFTN_PAIR(float, 9, 4),  \
      FFTN_PAIR(float, 10, 4), FFTN_PAIR(float, 12, 4), FFTN_PAIR(float, 15, 4),                   \
      FFTN_PAIR(float, 16, 4)

#include "isa/pairs.inl"

} // namespace pairs
FFTN_ISA_END

#endif
