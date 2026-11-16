// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary probes of the scalar target (probe_table.hpp).

#include "isa/probe_table.hpp"
#include "isa/region.hpp"

FFTN_ISA_BEGIN(scalar)

#include "isa/vocab_scalar.inl"

#include "isa/probe.inl"

constinit const probe_table probes{
    .id = id,
    .masked = masked,
    .f64 = &probe<double>,
    .f32 = &probe<float>,
};

FFTN_ISA_END
