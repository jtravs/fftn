// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Target scalar: portable kernels, the fallback on every processor and the reference for
// the generator tests (DESIGN.md 6.1). No ISA flag is ever set for this file, and its
// region carries no attribute: it is compiled for the baseline of the build.

#include "isa/kernel_table.hpp" // IWYU pragma: keep (table.inl defines a kernel_table)
#include "isa/region.hpp"

FFTN_ISA_BEGIN(scalar)

#include "isa/vocab_scalar.inl"

#include "isa/kernels.inl"

#include "isa/table.inl"

FFTN_ISA_END
