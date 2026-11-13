// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The target's function table (kernel_table.hpp), included last inside each region, and the
// only name a region exports. The entry points are instantiated here, inside the region, so
// they get its attribute on every compiler (no reliance on how a template defined in a
// region is treated when it is first instantiated after the region closes).

constinit const kernel_table table{
    .id = id,
    .fused = fused,
    .f64 =
        {
            .radix4_pass = &radix4_pass<double>,
            .radix2_split_pass = &radix2_split_pass<double>,
            .pack_twiddles = &pack_twiddles<double>,
            .complex_lanes = C<double>,
            .real_lanes = W<double>,
            .tw_reals = tw_reals<double>,
            .chain = chain_entries<double>,
        },
    .f32 =
        {
            .radix4_pass = &radix4_pass<float>,
            .radix2_split_pass = &radix2_split_pass<float>,
            .pack_twiddles = &pack_twiddles<float>,
            .complex_lanes = C<float>,
            .real_lanes = W<float>,
            .tw_reals = tw_reals<float>,
            .chain = chain_entries<float>,
        },
};
