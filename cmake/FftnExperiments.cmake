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
#
# fftn_x15_consumer(<target>) gives a test or benchmark FFTN_X15_LIBRARY_SWITCHES, a string
# naming every switch the library was built with. Never FFTN_X_OMP_SIMD itself: it would
# turn the pragma on in the consumer's own region text.

option(FFTN_X_OMP_SIMD "Experiment X15 arm B: omp simd on the scalar pass loops" OFF)

set(_fftn_x15_switches "")
set(_fftn_x15_consumer_definitions "")
if(FFTN_X_OMP_SIMD)
  list(APPEND _fftn_x15_switches omp_simd)
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
endfunction()

function(fftn_x15_consumer target)
  target_compile_definitions(${target} PRIVATE ${_fftn_x15_consumer_definitions}
                             "FFTN_X15_LIBRARY_SWITCHES=\"${FFTN_X15_SWITCHES}\"")
endfunction()
