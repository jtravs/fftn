// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// ISA regions (DESIGN.md 6.4). A target's code is written between
//
//   FFTN_ISA_BEGIN(avx2)
//   #include "isa/vocab_avx2.inl"     // the target's vocabulary
//   #include "isa/kernels.inl"        // kernel text, written once for every target
//   #include "isa/table.inl"          // the target's function table
//   FFTN_ISA_END
//
// which applies the target's attribute to every function declared in the region and opens
// the namespace fftn::v0::detail::<target>, so each target's functions have their own
// symbols. The rules that keep this sound are in AGENTS.md section 4 and DESIGN.md 6.4;
// the reasons, in brief:
//
// - Every standard, system and intrinsic header a region needs is included here, before any
//   region opens. A header first included inside a region compiles its inline functions
//   for that target, and the linker may keep that copy for baseline callers: an
//   illegal-instruction crash, reproduced in the header-only study. Inside a region only
//   .inl kernel and vocabulary text is ever included.
// - Vector types never cross a function boundary that is not force-inlined. Experiment B1:
//   GCC 13, 15 and 16 accept a 256-bit vector passed by value from an AVX2 region into a
//   baseline template with only a -Wpsabi warning, and the program then segfaults at -O0
//   and -O2 (Clang rejects the same code). The library is therefore compiled with
//   -Werror=psabi on GCC (src/isa/CMakeLists.txt), and entry points take only pointers and
//   integers.
// - Vector types are never template arguments: GCC warns -Wignored-attributes for each use
//   (the proof of concept had 56 such warnings), which -Werror turns into a failed build.
//   Per-target vector types are reached through vec<T> specialisations instead
//   (vocabulary.inl), and kernels use C arrays of vectors, not std::array.
// - No lambdas and no standard-library templates inside a region: they do not inherit the
//   target (B1: a lambda in a region is not inlined at -O0 under GCC).
// - Region text has internal linkage (vocabulary.inl), so no unit's copy of a function can
//   stand in for another's at link time.
#pragma once

#include "isa/arch.hpp"

#include <cstddef>

#if FFTN_ARCH_X86_64
#include <immintrin.h>
#elif FFTN_ARCH_AARCH64
#include <arm_neon.h>
#endif

// Force-inlined at every optimisation level. Every vocabulary operation and every kernel
// helper carries it: without it, -O0 calls each operation out of line (10-50 times slower),
// and a vector would cross a call boundary.
#define FFTN_FORCE_INLINE inline __attribute__((always_inline))

#define FFTN_PRAGMA(text) _Pragma(#text)
#ifdef __clang__
// apply_to = function: free functions, templates, members and lambdas declared in the
// region all get the attribute (the header-only study, experiment E04).
#define FFTN_ISA_PUSH(target_string)                                                               \
  FFTN_PRAGMA(clang attribute push(__attribute__((target(target_string))), apply_to = function))
#define FFTN_ISA_PUSH_NONE FFTN_PRAGMA(clang attribute push)
#define FFTN_ISA_POP FFTN_PRAGMA(clang attribute pop)
#elif defined(__GNUC__)
#define FFTN_ISA_PUSH(target_string)                                                               \
  FFTN_PRAGMA(GCC push_options) FFTN_PRAGMA(GCC target(target_string))
#define FFTN_ISA_PUSH_NONE FFTN_PRAGMA(GCC push_options)
#define FFTN_ISA_POP FFTN_PRAGMA(GCC pop_options)
#else
#error "FFTN needs GCC or Clang (including clang-cl): ISA regions use their target pragmas."
#endif

// The targets and their attribute strings. Adding a target (phase 1.3: x86_v2, avx) is one
// line here, one vocabulary header, one unit, and its entries in target.cpp and
// kernel_table.hpp.
//
// - scalar: no attribute; the baseline of whatever the library is built for.
// - avx2: AVX2 and FMA, the features runs_on() checks (target.cpp). The string implies
//   AVX and SSE4.2 and below, and in Clang also POPCNT and CRC32, which runs_on() does not
//   check but every AVX2 processor has; it enables no AVX-512 feature, so no EVEX encoding
//   and no zmm register can appear (the disassembly test checks this).
// - avx512: AVX-512 F, DQ and VL with AVX2 and FMA, the features runs_on() checks. Clang's
//   avx512f also implies F16C (and the avx2 part POPCNT and CRC32), which runs_on() does
//   not check but every AVX-512 processor has. BW and CD, which every processor with F,
//   DQ and VL also has, are left out so that no instruction of theirs can appear. AVX2
//   and FMA are named although Clang implies them from avx512f, so that the string does
//   not depend on a compiler's implications. GCC 14 and newer and Clang 18 and newer imply
//   EVEX512 (the 512-bit forms) from avx512f when no AVX10 option is given (GCC 14
//   compiles these units on the Xeon); "evex512" is not named because Clang 16 and 17 do
//   not know it.
// - neon: no attribute. Advanced SIMD is part of the AArch64 base architecture, so the
//   region is a namespace only and its code has exactly what the build's baseline has:
//   no FCMA and no pointer authentication on an Armv8.0 baseline (the disassembly test
//   checks this with no allowance), FCMA where the compiler targets Apple silicon.
// - neon_fcma: Clang's arm_neon.h declares the FCMLA/FCADD intrinsics with
//   target("v8.3a"), so "+fcma" is rejected; "arch=armv8.3-a" is the spelling GCC and
//   Clang both accept. It also allows other Armv8.3 instructions: the disassembly test
//   checks that no LRCPC or JSCVT instruction appears. Pointer authentication may appear
//   (branch protection can return with RETAA here), so neon_fcma is chosen only where the
//   processor has it too (target.cpp).
#define FFTN_ISA_PUSH_scalar FFTN_ISA_PUSH_NONE
#define FFTN_ISA_PUSH_avx2 FFTN_ISA_PUSH("avx2,fma")
#define FFTN_ISA_PUSH_avx512 FFTN_ISA_PUSH("avx2,fma,avx512f,avx512dq,avx512vl")
#define FFTN_ISA_PUSH_neon FFTN_ISA_PUSH_NONE
#define FFTN_ISA_PUSH_neon_fcma FFTN_ISA_PUSH("arch=armv8.3-a")

// The pragmas come last in each expansion: GCC (16, AArch64) drops the tokens that follow a
// _Pragma("GCC target ...") or a _Pragma("GCC pop_options") that changes the target within
// the same macro expansion, so "pragma, then namespace" loses the namespace. The
// attribute applies to functions declared after the pragma, so its position relative to
// the brace changes nothing else.
// NOLINTBEGIN(bugprone-macro-parentheses): a namespace name cannot be parenthesised.
#define FFTN_ISA_BEGIN(target_name)                                                                \
  namespace fftn::inline v0::detail::target_name {                                                 \
  FFTN_ISA_PUSH_##target_name
#define FFTN_ISA_END                                                                               \
  }                                                                                                \
  FFTN_ISA_POP
// NOLINTEND(bugprone-macro-parentheses)
