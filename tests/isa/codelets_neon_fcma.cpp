// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The generated codelets compiled for the neon_fcma target, with entry points for the tests
// (codelet_table.hpp).

#include "isa/arch.hpp"

#if FFTN_ARCH_AARCH64

#include "isa/codelet_table.hpp"
#include "isa/region.hpp"

FFTN_ISA_BEGIN(neon_fcma)

#include "isa/vocab_neon_fcma.inl"

#include "gen/complex_codelets.inl"

#include "isa/codelet_runner.inl"

FFTN_ISA_END

#endif
