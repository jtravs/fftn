// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The generated codelets compiled for the scalar target, with entry points for the tests
// (codelet_table.hpp).

#include "isa/codelet_table.hpp"
#include "isa/region.hpp"

FFTN_ISA_BEGIN(scalar)

#include "isa/vocab_scalar.inl"

#include "gen/complex_codelets.inl"

#include "isa/codelet_runner.inl"

FFTN_ISA_END
