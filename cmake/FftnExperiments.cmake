# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Experimental switches (AGENTS.md section 6; each is registered in dev/KNOBS.toml). Every
# switch is OFF by default and no preset sets one. They exist on an experiment branch
# for the length of the experiment and are deleted with its verdict.
#
# Experiment X15 (dev/experiments/X15-compiler-vectorised-scalar.md), what a compiler gets
# from the scalar pass loops:
#
#   FFTN_X_OMP_SIMD           arm B: the library is built with -fopenmp-simd (the simd
#                             pragmas only; no OpenMP runtime, nothing linked) and the
#                             innermost unit loops of the scalar kernel text carry
#                             `omp simd` (FFTN_SIMD_LOOP, src/isa/region.hpp).
#   FFTN_X_AVX2_SCALAR_TEXT   arm C (x86-64 only): the scalar kernel text, chain passes
#                             included, compiled a second time inside the avx2 region
#                             (src/isa/avx2_scalar_text.cpp); plans on AVX2 processors take
#                             it for the chain passes.
#   FFTN_X_NATIVE             arm E: the whole library compiled for the build machine
#                             (-march=native, -mcpu=native on AArch64). A diagnostic
#                             ceiling that breaks the region rules (AGENTS.md section 4)
#                             and must never be distributed or adopted.
#
# fftn_x15_consumer(<target>) gives a test or benchmark the definitions it needs to know
# which arms the library contains: FFTN_X_AVX2_SCALAR_TEXT and FFTN_X_NATIVE as the library
# has them, and FFTN_X15_LIBRARY_SWITCHES, a string naming every switch that is on. Never
# FFTN_X_OMP_SIMD itself: it would turn the pragma on in the consumer's own region text.

option(FFTN_X_OMP_SIMD "Experiment X15 arm B: omp simd on the scalar pass loops" OFF)
option(FFTN_X_AVX2_SCALAR_TEXT "Experiment X15 arm C: scalar kernel text in the avx2 region" OFF)
option(FFTN_X_NATIVE "Experiment X15 arm E: whole library for this machine (non-distributable)"
       OFF)

set(_fftn_x15_switches "")
set(_fftn_x15_consumer_definitions "")
if(FFTN_X_OMP_SIMD)
  list(APPEND _fftn_x15_switches omp_simd)
endif()
if(FFTN_X_AVX2_SCALAR_TEXT)
  list(APPEND _fftn_x15_switches avx2_scalar_text)
  list(APPEND _fftn_x15_consumer_definitions FFTN_X_AVX2_SCALAR_TEXT=1)
endif()
if(FFTN_X_NATIVE)
  if(FFTN_WARNINGS_AS_ERRORS)
    # Every gate preset sets FFTN_WARNINGS_AS_ERRORS: the gate never builds this arm.
    message(FATAL_ERROR "FFTN_X_NATIVE (experiment X15 arm E) is a throwaway, non-distributable "
                        "build and cannot be combined with FFTN_WARNINGS_AS_ERRORS (the gate's "
                        "presets). Configure it by hand, without a preset.")
  endif()
  list(APPEND _fftn_x15_switches native)
  list(APPEND _fftn_x15_consumer_definitions FFTN_X_NATIVE=1)
  foreach(_fftn_line IN ITEMS
      "################################################################################"
      "FFTN_X_NATIVE is ON: experiment X15 arm E. The whole library is compiled for THIS"
      "machine (-march=native / -mcpu=native). It breaks the ISA region rules, may crash"
      "on any other processor and must never be installed, distributed or adopted."
      "################################################################################")
    message(WARNING "${_fftn_line}")
  endforeach()
endif()
if(NOT _fftn_x15_switches)
  set(_fftn_x15_switches none)
endif()
list(JOIN _fftn_x15_switches "," FFTN_X15_SWITCHES)

# omp simd for one target's region text: the definition FFTN_SIMD_LOOP reads, and the flag.
function(fftn_x15_omp_simd target)
  target_compile_definitions(${target} PRIVATE FFTN_X_OMP_SIMD=1)
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(${target} PRIVATE /clang:-fopenmp-simd)
  else()
    target_compile_options(${target} PRIVATE -fopenmp-simd)
  endif()
endfunction()

# The switches' effect on the library itself; called from src/CMakeLists.txt.
function(fftn_x15_library target)
  target_compile_definitions(${target} PRIVATE ${_fftn_x15_consumer_definitions})
  if(FFTN_X_OMP_SIMD)
    fftn_x15_omp_simd(${target})
  endif()
  if(FFTN_X_NATIVE)
    # By what the compiler targets, not the host: GCC and Clang spell it -mcpu on AArch64.
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("#ifndef __aarch64__\n#error\n#endif\nint main() {}"
                              FFTN_X15_TARGET_IS_AARCH64)
    if(FFTN_X15_TARGET_IS_AARCH64)
      target_compile_options(${target} PRIVATE -mcpu=native)
    else()
      target_compile_options(${target} PRIVATE -march=native)
    endif()
  endif()
endfunction()

function(fftn_x15_consumer target)
  target_compile_definitions(${target} PRIVATE ${_fftn_x15_consumer_definitions}
                             "FFTN_X15_LIBRARY_SWITCHES=\"${FFTN_X15_SWITCHES}\"")
endfunction()
