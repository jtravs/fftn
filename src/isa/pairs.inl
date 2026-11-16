// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The table of a target's pair leaves (kernel_table.hpp, pair_leaf_table; experiment X28b),
// included last inside the region of the target's pairs unit (pairs_<target>.cpp), after
// the kernel text and after the unit defines FFTN_PAIRS_F64 and FFTN_PAIRS_F32: its entries
// as FFTN_PAIR(T, r1, m), ascending in r1 m. No include guard.

namespace {

inline constexpr edge_op pair_nop = edge_op::none;
inline constexpr edge_op pair_scl = edge_op::scale;

} // namespace

// NOLINTBEGIN(bugprone-macro-parentheses): template argument lists, not expressions.
// clang-format off: one row per input edge
#define FFTN_PAIR_EDGES(T, R1, M, S)                                                               \
  {{&pair_leaf<T, R1, M, S, pair_nop, pair_nop>, &pair_leaf<T, R1, M, S, pair_nop, pair_scl>},     \
   {&pair_leaf<T, R1, M, S, pair_scl, pair_nop>, nullptr}}
#define FFTN_PAIR(T, R1, M)                                                                        \
  {.n = (R1) * (M),                                                                                \
   .r1 = R1,                                                                                       \
   .m = M,                                                                                         \
   .leaf = {FFTN_PAIR_EDGES(T, R1, M, -1), FFTN_PAIR_EDGES(T, R1, M, +1)}}
// clang-format on
// NOLINTEND(bugprone-macro-parentheses)

constinit const pair_leaf_table leaves{
    .f64 = {.entry = {FFTN_PAIRS_F64}, .lanes = C<double>, .pack_twiddles = &pack_twiddles<double>},
    .f32 = {.entry = {FFTN_PAIRS_F32}, .lanes = C<float>, .pack_twiddles = &pack_twiddles<float>},
};

#undef FFTN_PAIR
#undef FFTN_PAIR_EDGES
#undef FFTN_PAIRS_F64
#undef FFTN_PAIRS_F32
