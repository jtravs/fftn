// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The generated codelets compiled for the avx512 target, with entry points for the tests
// (codelet_table.hpp).

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "isa/codelet_table.hpp"
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx512)

#include "isa/vocab_avx512.inl"

#include "gen/complex_codelets.inl"

#include "isa/codelet_runner.inl"

constinit const codelet_table codelet_entries{
    .id = id,
    .f64 = make_runs<double>(),
    .f32 = make_runs<float>(),
};

FFTN_ISA_END

#endif
