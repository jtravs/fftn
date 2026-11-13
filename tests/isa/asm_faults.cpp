// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Seeded faults for the disassembly check's own tests (tests/isa/CMakeLists.txt): each
// function below breaks one region rule on purpose, and tools/asm_check.py must report it.
// Never linked into anything.

#include "isa/arch.hpp"
#include "isa/region.hpp"

#if FFTN_ARCH_AARCH64
#include <arm_acle.h>
#endif

// NOLINTBEGIN(misc-use-internal-linkage): external, so the compiler keeps the faults.

#if FFTN_ARCH_X86_64

FFTN_ISA_BEGIN(avx2)

// NOLINTBEGIN(portability-simd-intrinsics): seeded faults, compiled but never run.

// A vector helper that is not force-inlined: the region calls it.
[[gnu::noinline]] inline __m256d fault_helper(__m256d a) { return _mm256_add_pd(a, a); }
void fault_call(double* p) { _mm256_storeu_pd(p, fault_helper(_mm256_loadu_pd(p))); }

// AVX-512 code in the avx2 namespace.
__attribute__((target("avx512f"))) void fault_evex(double* p) {
  _mm512_storeu_pd(p, _mm512_add_pd(_mm512_loadu_pd(p), _mm512_loadu_pd(p + 8)));
}

FFTN_ISA_END

namespace fftn::inline v0::detail {
// AVX2 code outside every target namespace, where baseline code lives.
__attribute__((target("avx2"))) void fault_baseline(double* p) {
  _mm256_storeu_pd(p, _mm256_add_pd(_mm256_loadu_pd(p), _mm256_loadu_pd(p + 4)));
}
} // namespace fftn::inline v0::detail

// NOLINTEND(portability-simd-intrinsics)

#elif FFTN_ARCH_AARCH64

FFTN_ISA_BEGIN(neon_fcma)

// A vector helper that is not force-inlined: the region calls it.
[[gnu::noinline]] inline float64x2_t fault_helper(float64x2_t a) { return vcaddq_rot90_f64(a, a); }
void fault_call(double* p) { vst1q_f64(p, fault_helper(vld1q_f64(p))); }

// FJCVTZS, which "arch=armv8.3-a" allows beside FCMA. (Pointer authentication, such as the
// RETAA that branch protection may emit here, is allowed in neon_fcma: DESIGN.md 6.4.)
int fault_jscvt(double x) { return __jcvt(x); }

FFTN_ISA_END

namespace fftn::inline v0::detail {
// FCMA code outside every target namespace, where baseline code lives.
__attribute__((target("arch=armv8.3-a"))) void fault_baseline(double* p) {
  vst1q_f64(p, vcaddq_rot90_f64(vld1q_f64(p), vld1q_f64(p + 2)));
}
// Pointer authentication outside the hint space, outside neon_fcma: PACIA X0, X1, written
// as its encoding so that no assembler needs Armv8.3 enabled to accept it.
void fault_pauth() {
  asm volatile(".inst 0xdac10020" ::: "x0"); // NOLINT(portability-no-assembler): the seeded fault
}
} // namespace fftn::inline v0::detail

#endif

// NOLINTEND(misc-use-internal-linkage)
