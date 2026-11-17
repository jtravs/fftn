// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary probes of the avx2 target (probe_table.hpp).

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "isa/probe_table.hpp"
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx2)

#include "isa/vocab_avx2.inl"

#include "isa/probe.inl"

constinit const probe_table probes{
    .id = id,
    .masked = partial_runs<double>,
    .f64 = &probe<double>,
    .f32 = &probe<float>,
};

FFTN_ISA_END

#endif
