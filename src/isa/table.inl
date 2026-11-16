// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The target's function table (kernel_table.hpp), included last inside the region of the
// target's main unit, after the kernel text. The entry points are instantiated here, inside
// the region, so they get its attribute on every compiler (no reliance on how a template
// defined in a region is treated when it is first instantiated after the region closes).
// The codelet rows are instantiated by the target's row units instead (isa/rows.inl); the
// table points at them.

static_assert(gen::radix_count == 20, "the declarations below list the rows of gen::radices");
// NOLINTBEGIN(bugprone-macro-parentheses): a template argument list, not an expression.
#define FFTN_ISA_ROW(I)                                                                            \
  extern template struct codelet_row<double, I>;                                                   \
  extern template struct codelet_row<float, I>;
// NOLINTEND(bugprone-macro-parentheses)
FFTN_ISA_ROW(0)
FFTN_ISA_ROW(1)
FFTN_ISA_ROW(2)
FFTN_ISA_ROW(3)
FFTN_ISA_ROW(4)
FFTN_ISA_ROW(5)
FFTN_ISA_ROW(6)
FFTN_ISA_ROW(7)
FFTN_ISA_ROW(8)
FFTN_ISA_ROW(9)
FFTN_ISA_ROW(10)
FFTN_ISA_ROW(11)
FFTN_ISA_ROW(12)
FFTN_ISA_ROW(13)
FFTN_ISA_ROW(14)
FFTN_ISA_ROW(15)
FFTN_ISA_ROW(16)
FFTN_ISA_ROW(17)
FFTN_ISA_ROW(18)
FFTN_ISA_ROW(19)
#undef FFTN_ISA_ROW

constinit const kernel_table table{
    .id = id,
    .fused = fused,
    .f64 =
        {
            .pack_twiddles = &pack_twiddles<double>,
            .pack_bcast_twiddles = &pack_bcast_twiddles<double>,
            .complex_lanes = C<double>,
            .real_lanes = W<double>,
            .tw_reals = tw_reals<double>,
            .split_block = split_block<double>,
            .passes = pass_table<double>,
        },
    .f32 =
        {
            .pack_twiddles = &pack_twiddles<float>,
            .pack_bcast_twiddles = &pack_bcast_twiddles<float>,
            .complex_lanes = C<float>,
            .real_lanes = W<float>,
            .tw_reals = tw_reals<float>,
            .split_block = split_block<float>,
            .passes = pass_table<float>,
        },
};
