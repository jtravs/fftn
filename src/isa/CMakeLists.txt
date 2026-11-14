# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# One translation unit per instruction set (DESIGN.md 6.4). Each is compiled on every
# architecture, with no ISA flag; its body is guarded by the architecture it targets, so
# universal and cross builds need no CMake logic. Kernel units are always optimised.

set(fftn_isa_units scalar.cpp avx2.cpp avx512.cpp neon.cpp neon_fcma.cpp)
target_sources(fftn PRIVATE ${fftn_isa_units} target.cpp)
fftn_always_optimise(${fftn_isa_units})

# GCC's AddressSanitizer moves stack frames into a "fake stack" for use-after-return
# detection, aligned to 32 bytes; AVX-512 code spills zmm registers there with aligned 64-byte
# stores and faults (Xeon Gold 6240, GCC 14, check-sanitize; Clang's runtime aligns them).
# Use-after-return instrumentation is off for the avx512 units in GCC sanitizer builds only;
# every other check stays on.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND FFTN_SANITIZE MATCHES "address")
  set_property(SOURCE avx512.cpp APPEND PROPERTY COMPILE_OPTIONS --param=asan-use-after-return=0)
endif()

# Experiment B1: GCC accepts a 256-bit vector passed by value from an AVX2 region into a
# baseline function with only a -Wpsabi warning, and the program then segfaults. Clang refuses
# such code outright. The warning is an error in every GCC build, not only in the gate.
target_compile_options(fftn PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror=psabi>)
