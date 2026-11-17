# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Each instruction set is a main unit (<target>.cpp: the function table and the passes
# outside the codelet rows) and row units (rows<k>_<target>.cpp: the codelet rows that
# isa/rows.inl gives unit k), DESIGN.md 6.4. Every unit is compiled on every architecture,
# with no ISA flag; its body is guarded by the architecture it targets, so universal and
# cross builds need no CMake logic. Kernel units are always optimised.

set(fftn_isa_targets scalar x86_v2 avx avx2 avx512 neon neon_fcma)
set(fftn_isa_row_units 8) # FFTN_ISA_ROWS 1 to 8 in isa/rows.inl
set(fftn_isa_units "")
foreach(target IN LISTS fftn_isa_targets)
  list(APPEND fftn_isa_units ${target}.cpp)
  foreach(k RANGE 1 ${fftn_isa_row_units})
    list(APPEND fftn_isa_units rows${k}_${target}.cpp)
  endforeach()
endforeach()
# The 128-bit leaves of the x86 targets (experiment X28), in the avx2 region.
list(APPEND fftn_isa_units narrow_avx2.cpp)
# The pair leaves of the x86 targets (experiment X28b) and of NEON (X37).
list(APPEND fftn_isa_units pairs_avx2.cpp pairs_avx512.cpp pairs_neon.cpp pairs_neon_fcma.cpp)
target_sources(fftn PRIVATE ${fftn_isa_units} target.cpp)
fftn_always_optimise(${fftn_isa_units})
# For the disassembly test's second compilation (tests/isa).
list(TRANSFORM fftn_isa_units PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set_property(GLOBAL PROPERTY FFTN_ISA_UNITS ${fftn_isa_units})

# Experiment B1: GCC accepts a 256-bit vector passed by value from an AVX2 region into a
# baseline function with only a -Wpsabi warning, and the program then segfaults. Clang refuses
# such code outright. The warning is an error in every GCC build, not only in the gate.
target_compile_options(fftn PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror=psabi>)
