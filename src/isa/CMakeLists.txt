# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Each instruction set is a main unit (<target>.cpp: the function table and the passes
# outside the codelet rows) and row units (rows<k>_<target>.cpp: the codelet rows that
# isa/rows.inl gives unit k), DESIGN.md 6.4. Every unit is compiled on every architecture,
# with no ISA flag; its body is guarded by the architecture it targets, so universal and
# cross builds need no CMake logic. Kernel units are always optimised.

set(fftn_isa_targets scalar avx2 avx512 neon neon_fcma)
if(FFTN_OLD_X86_TARGETS) # CMakeLists.txt; their units guard themselves to x86-64 too
  list(APPEND fftn_isa_targets x86_v2 avx)
endif()
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
# The avx512 units' branches are padded so that none crosses or ends on a 32-byte boundary.
# Skylake-derived cores (the Xeon 6240) fetch such a branch from the slow decoders once the
# JCC erratum's microcode update is installed, and which kernel lands where is luck: the merge
# of X40 moved code and single c2c rows of 160, 180 and 240 points took 1.07-1.10 of their
# time, 360 and 384 1.04-1.06 (bisect, Xeon 6240: 20261117-184544-8056). With the padding
# they take 0.89-0.94 of the unpadded time and the other small rows stay level
# (20261117-204818-1541; dev/LESSONS.md 2.11). No effect where the units never run.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(fftn_jcc_units "")
  foreach(unit IN LISTS fftn_isa_units)
    if(unit MATCHES "avx512")
      list(APPEND fftn_jcc_units ${CMAKE_CURRENT_SOURCE_DIR}/${unit})
    endif()
  endforeach()
  # TARGET_DIRECTORY: the target is created in src/, so a property set here alone is not seen.
  set_source_files_properties(${fftn_jcc_units} TARGET_DIRECTORY fftn
    PROPERTIES COMPILE_OPTIONS -Wa,-mbranches-within-32B-boundaries)
endif()
# Clang on AArch64 compiles the NEON units without load and store pairs (LDP, STP). On the
# Apple M5 Max a load fed by a recent store forwards in about 1 ns when both are single
# 128-bit accesses and takes 3.4-3.8 ns when either is a pair, so pairs cost wherever data
# comes back through memory soon: in-place executions of tiny transforms (each reads what
# the last one stored; 2 x 3 in place took 1.46 of FFTW's time) and codelets that spill
# (the 32-point leaf). Experiment X49; dev/LESSONS.md.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  include(CheckCXXSourceCompiles)
  # A string in COMPILE_FLAGS, not a list in COMPILE_OPTIONS, which would merge the repeated
  # -Xclang options.
  set(fftn_unpaired_flags
      "-Xclang -target-feature -Xclang +disable-ldp -Xclang -target-feature -Xclang +disable-stp")
  set(CMAKE_REQUIRED_FLAGS "${fftn_unpaired_flags} -Werror")
  check_cxx_source_compiles("int main() { return 0; }" FFTN_HAVE_UNPAIRED_NEON)
  unset(CMAKE_REQUIRED_FLAGS)
  if(FFTN_HAVE_UNPAIRED_NEON)
    set(fftn_neon_units "")
    foreach(unit IN LISTS fftn_isa_units)
      if(unit MATCHES "neon")
        list(APPEND fftn_neon_units ${CMAKE_CURRENT_SOURCE_DIR}/${unit})
      endif()
    endforeach()
    set_source_files_properties(${fftn_neon_units} TARGET_DIRECTORY fftn
      PROPERTIES COMPILE_FLAGS "${fftn_unpaired_flags}")
  endif()
endif()
# For the disassembly test's second compilation (tests/isa).
list(TRANSFORM fftn_isa_units PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set_property(GLOBAL PROPERTY FFTN_ISA_UNITS ${fftn_isa_units})

# Experiment B1: GCC accepts a 256-bit vector passed by value from an AVX2 region into a
# baseline function with only a -Wpsabi warning, and the program then segfaults. Clang refuses
# such code outright. The warning is an error in every GCC build, not only in the gate.
target_compile_options(fftn PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror=psabi>)
