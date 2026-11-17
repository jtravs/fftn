// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary probes of the neon target (probe_table.hpp).

#include "isa/arch.hpp"

#if FFTN_ARCH_AARCH64

#include "isa/probe_table.hpp"
#include "isa/region.hpp"

FFTN_ISA_BEGIN(neon)

#include "isa/vocab_neon.inl"

#include "isa/probe.inl"

constinit const probe_table probes{
    .id = id,
    .masked = partial_runs<double>,
    .f64 = &probe<double>,
    .f32 = &probe<float>,
};

FFTN_ISA_END

#endif
