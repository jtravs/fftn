# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25)
project(fftn VERSION 0.0.0 LANGUAGES CXX)

option(FFTN_BUILD_TESTS "Build the tests and the documented examples" ${PROJECT_IS_TOP_LEVEL})
option(FFTN_BUILD_BENCH "Build fbench and its backends" ${PROJECT_IS_TOP_LEVEL})
option(FFTN_WARNINGS_AS_ERRORS "Treat compiler and clang-tidy warnings as errors" OFF)
option(FFTN_COVERAGE "Instrument FFTN's own targets for line coverage" OFF)
set(FFTN_SANITIZE "" CACHE STRING "Sanitizers for every target, e.g. address,undefined or thread")
set(FFTN_CLANG_TIDY "" CACHE STRING "clang-tidy command (a CMake list) run on FFTN's own targets")

# Read by scripts/check.sh to report the compiler with every gate run (AGENTS.md section 3).
set(FFTN_CXX_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" CACHE INTERNAL "")

# Intel's icpx compiles everything, third-party code included, with a fast floating-point
# model by default, and an optimised main() it compiles sets flush-to-zero and
# denormals-are-zero for the whole process. The tests' main() comes from GoogleTest, so the
# IEEE model is set for every target here, not only FFTN's own (experiment X22).
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  add_compile_options(-fno-fast-math)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(FftnTargets)

# Sanitizers apply to everything, third-party code included: a partly instrumented process
# gives false reports (TSan in particular).
if(FFTN_SANITIZE)
  add_compile_options(-fsanitize=${FFTN_SANITIZE} -fno-sanitize-recover=all
                      -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${FFTN_SANITIZE})
  # GCC folds identical functions under the sanitizers (radix 2's forward and backward
  # passes, scalar's two twiddle packers), leaving jumps or, under TSan, calling wrappers
  # that asm_check (tools/asm_check.py) cannot tell from missed inlining. Release builds do
  # not fold them; turning the folding off here keeps the checked code comparable.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fno-ipa-icf)
  endif()
  # Clang links its TSan runtime statically on Linux, and that runtime defines the global
  # operator new and delete as strong symbols (compiler-rt's tsan_new_delete.cpp; ASan's
  # are weak), so the tests' counting replacements (tests/support/alloc_hook.cpp) clash
  # with them at link time. The shared runtime lets them interpose, as GCC's libtsan, a
  # shared library, always has. Its directory is not on the loader's path: hence the rpath.
  if(FFTN_SANITIZE MATCHES "thread" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
     AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-runtime-dir
                    OUTPUT_VARIABLE _fftn_runtime_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                    COMMAND_ERROR_IS_FATAL ANY)
    add_link_options(-shared-libsan "LINKER:-rpath,${_fftn_runtime_dir}")
  endif()
endif()

add_subdirectory(src)

if(FFTN_BUILD_TESTS)
  enable_testing()
  include(FftnDeps)
  include(FftnTesting)
  add_subdirectory(tests)
  add_subdirectory(examples)
  fftn_write_coverage_manifest()
endif()

if(FFTN_BUILD_BENCH)
  add_subdirectory(bench)
endif()
