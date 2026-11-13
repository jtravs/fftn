// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The generated codelets compiled for the avx2 target, with entry points for the tests
// (codelet_table.hpp).

#include "isa/arch.hpp"

#if FFTN_ARCH_X86_64

#include "isa/codelet_table.hpp"
#include "isa/region.hpp"

FFTN_ISA_BEGIN(avx2)

#include "isa/vocab_avx2.inl"

#include "gen/complex_codelets.inl"

#include "isa/codelet_runner.inl"

FFTN_ISA_END

#endif
