// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The table of a target's batch leaves (kernel_table.hpp, batch_leaf_table; experiment
// X53), included last inside the region of the target's batch units (batch_<target>.cpp,
// batch_large_<target>.cpp), after the kernel text and passes/batch_leaf.inl, once the unit
// has defined FFTN_BATCH_PART: 1 for the radices below 32 (the table `leaves`), 2 for 32 and
// 64 (`large_leaves`), which cost as much to compile as all the others together (isa/rows.inl
// gives the numbers for the codelet rows). A precision with one complex number per register
// has no batch leaves: its leaf already fills the register. No include guard.

namespace {

inline constexpr edge_op batch_nop = edge_op::none;
inline constexpr edge_op batch_scl = edge_op::scale;

// Radix 25 has no passes on NEON (passes/entries.inl, radix_on_target), so no leaf either.
// Lengths 16, 32 and 64 as the chains 4 x 4, 4 x 8 and 8 x 8 through a tile (batch_pair_leaf,
// X53 arm B5); the others one codelet.
template <std::size_t R>
inline constexpr std::size_t batch_r1 = R == 16 || R == 32 ? 4
                                        : R == 64          ? 8
                                                           : 0;

template <class T, std::size_t R, int S, edge_op I, edge_op O> constexpr leaf_fn batch_entry() {
  if constexpr (C<T> == 1 || (R == 25 && (id == target::neon || id == target::neon_fcma))) {
    return nullptr;
  } else if constexpr (batch_r1<R> != 0) {
    return &batch_pair_leaf<T, batch_r1<R>, R / batch_r1<R>, S, I, O>;
  } else {
    return &batch_leaf<T, R, S, I, O>;
  }
}

} // namespace

static_assert(gen::radix_count == 22 && gen::radices[19] == 25 && gen::radices[20] == 32 &&
                  gen::radices[21] == 64,
              "the table below lists gen::radices in codelet_slot order, 32 and 64 last");

// NOLINTBEGIN(bugprone-macro-parentheses): template argument lists, not expressions.
// clang-format off: one row per input edge, one radix set per line
#define FFTN_BATCH_EDGES(T, R, S)                                                                  \
  {{batch_entry<T, R, S, batch_nop, batch_nop>(), batch_entry<T, R, S, batch_nop, batch_scl>()},   \
   {batch_entry<T, R, S, batch_scl, batch_nop>(), nullptr}}
#define FFTN_BATCH_RADIX(T, R) {FFTN_BATCH_EDGES(T, R, -1), FFTN_BATCH_EDGES(T, R, +1)}
#if FFTN_BATCH_PART == 1
#define FFTN_BATCH_SET(T)                                                                          \
  {FFTN_BATCH_RADIX(T, 2),  FFTN_BATCH_RADIX(T, 3),  FFTN_BATCH_RADIX(T, 4),                       \
   FFTN_BATCH_RADIX(T, 5),  FFTN_BATCH_RADIX(T, 6),  FFTN_BATCH_RADIX(T, 7),                       \
   FFTN_BATCH_RADIX(T, 8),  FFTN_BATCH_RADIX(T, 9),  FFTN_BATCH_RADIX(T, 10),                      \
   FFTN_BATCH_RADIX(T, 11), FFTN_BATCH_RADIX(T, 12), FFTN_BATCH_RADIX(T, 13),                      \
   FFTN_BATCH_RADIX(T, 14), FFTN_BATCH_RADIX(T, 15), FFTN_BATCH_RADIX(T, 16),                      \
   FFTN_BATCH_RADIX(T, 17), FFTN_BATCH_RADIX(T, 19), FFTN_BATCH_RADIX(T, 20),                      \
   FFTN_BATCH_RADIX(T, 23), FFTN_BATCH_RADIX(T, 25), {}, {}}
#define FFTN_BATCH_TABLE leaves
#elif FFTN_BATCH_PART == 2
#define FFTN_BATCH_SET(T)                                                                          \
  {{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},                 \
   FFTN_BATCH_RADIX(T, 32), FFTN_BATCH_RADIX(T, 64)}
#define FFTN_BATCH_TABLE large_leaves
#else
#error "batch.inl: FFTN_BATCH_PART names no part"
#endif
// clang-format on
// NOLINTEND(bugprone-macro-parentheses)

#define FFTN_BATCH_R1                                                                              \
  {0,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, batch_r1<16>, 0, 0, 0, 0, 0, batch_r1<32>,  \
   batch_r1<64>}
constinit const batch_leaf_table FFTN_BATCH_TABLE{
    .f64 = {.leaf = FFTN_BATCH_SET(double), .r1 = FFTN_BATCH_R1, .lanes = C<double>},
    .f32 = {.leaf = FFTN_BATCH_SET(float), .r1 = FFTN_BATCH_R1, .lanes = C<float>},
};
#undef FFTN_BATCH_R1

#undef FFTN_BATCH_TABLE
#undef FFTN_BATCH_SET
#undef FFTN_BATCH_RADIX
#undef FFTN_BATCH_EDGES
#undef FFTN_BATCH_PART
