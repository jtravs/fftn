// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The generated codelets compiled for the neon target, with entry points for the tests
// (codelet_table.hpp).

#include "isa/arch.hpp"

#if FFTN_ARCH_AARCH64

#include "isa/codelet_table.hpp"
#include "isa/region.hpp"

FFTN_ISA_BEGIN(neon)

#include "isa/vocab_neon.inl"

#include "gen/complex_codelets.inl"

#include "isa/codelet_runner.inl"

constinit const codelet_table codelet_entries{
    .id = id,
    .f64 = make_runs<double>(),
    .f32 = make_runs<float>(),
};

FFTN_ISA_END

#endif
