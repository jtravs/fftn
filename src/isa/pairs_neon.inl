// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The pair leaves of both NEON targets (experiment X37; passes/pair_leaf.inl), included
// inside the region of pairs_neon.cpp and pairs_neon_fcma.cpp after the kernel text: the
// rows of 18 to 30 points that are not a codelet, as r1 x m in the order of the chain the
// rules choose (rules.cpp; the test Passes.PairLeavesMatchTheForcedChain holds the bits),
// with one complex double per register (C = 1): n of the 32 registers. FFTW runs 20 points
// with a codelet of 20 (1.78 times faster than the chain 4 x 5 in the phase 1 exit check).
// None in float, where NEON's chains are not behind. No include guard.

#define FFTN_PAIRS_F64                                                                             \
  FFTN_PAIR(double, 3, 6), FFTN_PAIR(double, 4, 5), FFTN_PAIR(double, 2, 11),                      \
      FFTN_PAIR(double, 4, 6), FFTN_PAIR(double, 2, 13), FFTN_PAIR(double, 4, 7),                  \
      FFTN_PAIR(double, 6, 5)
#define FFTN_PAIRS_F32

#include "isa/pairs.inl"
