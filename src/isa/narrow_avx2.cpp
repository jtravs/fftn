// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The 128-bit leaves of the x86 targets (experiment X28; kernel_table.hpp,
// narrow_leaf_table): the leaves of radices 2 to 16 compiled inside the avx2 region with
// avx2's operations at half width (vocab_avx2_narrow.inl). A leaf computes one line with
// every input broadcast into every lane, so on 256- and 512-bit registers the extra lanes
// only repeat the work; FFTW runs single rows of 8 and 16 points with 128-bit codelets
// for the same reason, and on Cascade Lake a 512-bit leaf also lowers the clock (X25).
// Like every kernel unit, it sets no ISA flag (DESIGN.md 6.4).

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "gen/codelets.hpp"     // IWYU pragma: keep (the passes read the codelet table)
#include "isa/kernel_table.hpp" // IWYU pragma: keep (defines a narrow_leaf_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx2)
namespace narrow {

#include "isa/vocab_avx2_narrow.inl"

#include "isa/kernels.inl"

#include "gen/complex_codelets.inl"

#include "passes/access.inl"

#include "passes/stockham.inl"

#include "passes/unit_leaf.inl"

namespace {

inline constexpr edge_op nop = edge_op::none;
inline constexpr edge_op scl = edge_op::scale;
inline constexpr edge_op fac = edge_op::factor;

} // namespace

static_assert(gen::radices[0] == 2 && gen::radices[dense_codelet_radix - 2] == dense_codelet_radix,
              "the table below lists radices 2 to 16 in codelet_slot order");

// As entries.inl's leaf rows: [input edge][output edge], null where both would scale.
// NOLINTBEGIN(bugprone-macro-parentheses): template argument lists, not expressions.
// clang-format off: one row per input edge, one radix set per line
#define FFTN_NARROW_EDGES(T, R, S)                                                                 \
  {{&leaf_lines<T, R, S, nop, nop>, &leaf_lines<T, R, S, nop, scl>,                                \
    &leaf_lines<T, R, S, nop, fac>},                                                               \
   {&leaf_lines<T, R, S, scl, nop>, nullptr, &leaf_lines<T, R, S, scl, fac>},                      \
   {&leaf_lines<T, R, S, fac, nop>, &leaf_lines<T, R, S, fac, scl>,                                \
    &leaf_lines<T, R, S, fac, fac>}}
#define FFTN_NARROW_RADIX(T, R) {FFTN_NARROW_EDGES(T, R, -1), FFTN_NARROW_EDGES(T, R, +1)}
#define FFTN_NARROW_SET(T)                                                                         \
  {FFTN_NARROW_RADIX(T, 2),  FFTN_NARROW_RADIX(T, 3),  FFTN_NARROW_RADIX(T, 4),                    \
   FFTN_NARROW_RADIX(T, 5),  FFTN_NARROW_RADIX(T, 6),  FFTN_NARROW_RADIX(T, 7),                    \
   FFTN_NARROW_RADIX(T, 8),  FFTN_NARROW_RADIX(T, 9),  FFTN_NARROW_RADIX(T, 10),                   \
   FFTN_NARROW_RADIX(T, 11), FFTN_NARROW_RADIX(T, 12), FFTN_NARROW_RADIX(T, 13),                   \
   FFTN_NARROW_RADIX(T, 14), FFTN_NARROW_RADIX(T, 15), FFTN_NARROW_RADIX(T, 16)}
// The unit-stride leaves (unit_leaf.inl): [input edge][output edge], none or scale.
#define FFTN_NARROW_UNIT_EDGES(T, R, S)                                                            \
  {{&unit_leaf<T, R, S, nop, nop>, &unit_leaf<T, R, S, nop, scl>},                                 \
   {&unit_leaf<T, R, S, scl, nop>, nullptr}}
#define FFTN_NARROW_UNIT_RADIX(T, R)                                                               \
  {FFTN_NARROW_UNIT_EDGES(T, R, -1), FFTN_NARROW_UNIT_EDGES(T, R, +1)}
#define FFTN_NARROW_UNIT_SET(T)                                                                    \
  {FFTN_NARROW_UNIT_RADIX(T, 2),  FFTN_NARROW_UNIT_RADIX(T, 3),  FFTN_NARROW_UNIT_RADIX(T, 4),     \
   FFTN_NARROW_UNIT_RADIX(T, 5),  FFTN_NARROW_UNIT_RADIX(T, 6),  FFTN_NARROW_UNIT_RADIX(T, 7),     \
   FFTN_NARROW_UNIT_RADIX(T, 8),  FFTN_NARROW_UNIT_RADIX(T, 9),  FFTN_NARROW_UNIT_RADIX(T, 10),    \
   FFTN_NARROW_UNIT_RADIX(T, 11), FFTN_NARROW_UNIT_RADIX(T, 12), FFTN_NARROW_UNIT_RADIX(T, 13),    \
   FFTN_NARROW_UNIT_RADIX(T, 14), FFTN_NARROW_UNIT_RADIX(T, 15), FFTN_NARROW_UNIT_RADIX(T, 16)}
// clang-format on
// NOLINTEND(bugprone-macro-parentheses)

constinit const narrow_leaf_table leaves{
    .f64 = {.leaf = FFTN_NARROW_SET(double), .unit = FFTN_NARROW_UNIT_SET(double)},
    .f32 = {.leaf = FFTN_NARROW_SET(float), .unit = FFTN_NARROW_UNIT_SET(float)},
};

#undef FFTN_NARROW_UNIT_SET
#undef FFTN_NARROW_UNIT_RADIX
#undef FFTN_NARROW_UNIT_EDGES
#undef FFTN_NARROW_SET
#undef FFTN_NARROW_RADIX
#undef FFTN_NARROW_EDGES

} // namespace narrow
FFTN_ISA_END

#endif
