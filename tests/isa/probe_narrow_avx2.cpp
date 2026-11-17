// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary probes of the avx2 target at half width (vocab_avx2_narrow.inl, experiment X28;
// probe_table.hpp), in the nested namespace narrow as isa/narrow_avx2.cpp compiles it. The
// narrow leaves have no kernel table of their own, so the vocabulary's geometry and twiddle
// packing, which the probes' reference reads from a kernel table, are given here in one.

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "isa/kernel_table.hpp"
#include "isa/probe_table.hpp"
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx2)
namespace narrow {

#include "isa/vocab_avx2_narrow.inl"

#include "isa/kernels.inl"

#include "isa/probe.inl"

constinit const probe_table probes{
    .id = id,
    .masked = partial_runs<double>,
    .f64 = &probe<double>,
    .f32 = &probe<float>,
};

// No passes: only the probes' reference reads this table.
constinit const kernel_table geometry{
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
            .passes = {},
        },
    .f32 =
        {
            .pack_twiddles = &pack_twiddles<float>,
            .pack_bcast_twiddles = &pack_bcast_twiddles<float>,
            .complex_lanes = C<float>,
            .real_lanes = W<float>,
            .tw_reals = tw_reals<float>,
            .split_block = split_block<float>,
            .passes = {},
        },
};

} // namespace narrow
FFTN_ISA_END

#endif
