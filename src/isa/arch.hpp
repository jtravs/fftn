// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The architecture being compiled for, and so which targets this build contains. Each
// target's unit is compiled on every architecture with no ISA flag and guards its body with
// these macros (DESIGN.md 6.4); the spellings cover GCC, Clang and clang on the MSVC ABI.
#pragma once

// NOLINTBEGIN(modernize-macro-to-enum): these are tested in #if.
#if defined(__x86_64__) || defined(_M_X64)
#define FFTN_ARCH_X86_64 1
#else
#define FFTN_ARCH_X86_64 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define FFTN_ARCH_AARCH64 1
#else
#define FFTN_ARCH_AARCH64 0
#endif
// NOLINTEND(modernize-macro-to-enum)
