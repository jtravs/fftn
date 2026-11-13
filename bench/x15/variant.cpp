// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Experiment X15: one copy of the scalar kernel text for the pass-level harness
// (passes.cpp), in a region of its own. bench/CMakeLists.txt compiles this file several
// times, each as its own object library:
//
//   X15_REGION   the region's namespace (x15_scalar, x15_scalar_simd, ...)
//   X15_AVX2     1: the region carries the avx2 attribute (x86-64 only; arm C), 0: none
//   FFTN_X_OMP_SIMD and -fopenmp-simd for the copies with omp simd (arm B)
//
// FFTN_ISA_BEGIN pastes FFTN_ISA_PUSH_ to its argument unexpanded, so the attribute is
// chosen by defining FFTN_ISA_PUSH_X15_REGION; the namespace takes the expanded name.

#include "isa/arch.hpp"

#if !X15_AVX2 || FFTN_ARCH_X86_64

#include "isa/kernel_table.hpp" // IWYU pragma: keep (table.inl defines a kernel_table)
#include "isa/region.hpp"

#if X15_AVX2
#define FFTN_ISA_PUSH_X15_REGION FFTN_ISA_PUSH_avx2
#else
#define FFTN_ISA_PUSH_X15_REGION FFTN_ISA_PUSH_NONE
#endif

FFTN_ISA_BEGIN(X15_REGION)

#include "isa/vocab_scalar.inl"

#include "isa/kernels.inl"

#include "passes/chain_absent.inl"

// External, as the library's tables are by kernel_table.hpp: passes.cpp declares it too.
extern const kernel_table table;

#include "isa/table.inl"

FFTN_ISA_END

#endif
