// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The pair leaves of the avx512 target (experiment X28b; passes/pair_leaf.inl,
// kernel_table.hpp): short rows of n = r1 m points, C dividing m, as their two chain passes
// held in registers. Lengths whose n / C registers fit AVX-512's 32 with room for the
// codelets' temporaries. Like every kernel unit, it sets no ISA flag (DESIGN.md 6.4).

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "gen/codelets.hpp"     // IWYU pragma: keep (the codelets read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (defines a pair_leaf_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx512)
namespace pairs {

#include "isa/vocab_avx512.inl"

#include "isa/kernels.inl"

#include "gen/complex_codelets.inl"

#include "passes/access.inl"

#include "passes/pair_leaf.inl"

// C = 4 in double: m = 4 (one unit) up to 64 points, then m = 8. 16 = 4 x 4 (X37) replaces
// the 128-bit leaf of 16 for rows of stride 1. Not 8 = 2 x 4 or 12 = 3 x 4: with r1 < C the
// second pass stores partial runs, and an in-place batch of 3 rows whose next execution
// reads them back ran 2.07 and 1.50 times as long (no store forwarding from a masked
// store); those rows take avx2's 4 x 2 and 6 x 2 (find_more_pair_leaves).
#define FFTN_PAIRS_F64                                                                             \
  FFTN_PAIR(double, 4, 4), FFTN_PAIR(double, 5, 4), FFTN_PAIR(double, 6, 4),                       \
      FFTN_PAIR(double, 8, 4), FFTN_PAIR(double, 9, 4), FFTN_PAIR(double, 10, 4),                  \
      FFTN_PAIR(double, 12, 4), FFTN_PAIR(double, 15, 4), FFTN_PAIR(double, 16, 4),                \
      FFTN_PAIR(double, 9, 8), FFTN_PAIR(double, 10, 8), FFTN_PAIR(double, 12, 8),                 \
      FFTN_PAIR(double, 15, 8)
// C = 8 in float: m = 8, one unit.
#define FFTN_PAIRS_F32                                                                             \
  FFTN_PAIR(float, 3, 8), FFTN_PAIR(float, 4, 8), FFTN_PAIR(float, 5, 8), FFTN_PAIR(float, 6, 8),  \
      FFTN_PAIR(float, 8, 8), FFTN_PAIR(float, 9, 8), FFTN_PAIR(float, 10, 8),                     \
      FFTN_PAIR(float, 12, 8), FFTN_PAIR(float, 15, 8), FFTN_PAIR(float, 16, 8)

#include "isa/pairs.inl"

} // namespace pairs
FFTN_ISA_END

#endif
