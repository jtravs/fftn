# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25)
project(fftn VERSION 0.0.0 LANGUAGES CXX)

option(FFTN_BUILD_TESTS "Build the tests and the documented examples" ${PROJECT_IS_TOP_LEVEL})
option(FFTN_BUILD_BENCH "Build fbench and its backends" ${PROJECT_IS_TOP_LEVEL})
# Install rules for the library, its headers, the CMake package and fftn.pc
# (cmake/FftnPackage.cmake, docs/building.md). Off when FFTN is a subproject, as with
# FetchContent: the including project decides what it installs.
option(FFTN_INSTALL "Install the library, its headers and its CMake and pkg-config files"
       ${PROJECT_IS_TOP_LEVEL})
# The C API as a shared library as well (src/c_api/CMakeLists.txt, docs/c_api.md): libfftn_c,
# which exports the fftn_ functions of fftn/c_api.h and nothing else. Provisional (step 6.8).
# The default may come from the environment variable of the same name (CI's GCC job turns it on
# that way, .github/workflows/ci.yml, so that its tests run on every pull request); an empty
# value counts as unset.
set(_fftn_c_shared_default OFF)
if(NOT "$ENV{FFTN_BUILD_C_SHARED}" STREQUAL "")
  set(_fftn_c_shared_default "$ENV{FFTN_BUILD_C_SHARED}")
endif()
option(FFTN_BUILD_C_SHARED "Also build fftn_c, a shared library of the C API (fftn/c_api.h)"
       ${_fftn_c_shared_default})
option(FFTN_WARNINGS_AS_ERRORS "Treat compiler and clang-tidy warnings as errors" OFF)
option(FFTN_COVERAGE "Instrument FFTN's own targets for line coverage" OFF)
set(FFTN_SANITIZE "" CACHE STRING "Sanitizers for every target, e.g. address,undefined or thread")
set(FFTN_CLANG_TIDY "" CACHE STRING "clang-tidy command (a CMake list) run on FFTN's own targets")
# Partial runs as masked vector accesses in the avx2 and avx512 kernels (experiment X23).
# They take 0.60-0.90 of the tile's time with GCC on 7-smooth rows, but Clang 23 lost up to 1.42 on odd double
# lengths for a reason not yet found, so AUTO turns them on for GCC only; icpx is Clang.
# Either way the bits are the same.
set(FFTN_MASKED_TAILS AUTO CACHE STRING "Masked partial runs on x86: AUTO (on with GCC), ON or OFF")
set_property(CACHE FFTN_MASKED_TAILS PROPERTY STRINGS AUTO ON OFF)
if(FFTN_MASKED_TAILS STREQUAL "AUTO")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(FFTN_MASKED_TAILS_ON 1)
  else()
    set(FFTN_MASKED_TAILS_ON 0)
  endif()
elseif(FFTN_MASKED_TAILS STREQUAL "ON")
  set(FFTN_MASKED_TAILS_ON 1)
elseif(FFTN_MASKED_TAILS STREQUAL "OFF")
  set(FFTN_MASKED_TAILS_ON 0)
else()
  message(FATAL_ERROR "FFTN_MASKED_TAILS must be AUTO, ON or OFF, not '${FFTN_MASKED_TAILS}'")
endif()
message(STATUS "fftn: masked partial runs on x86 ${FFTN_MASKED_TAILS_ON} (FFTN_MASKED_TAILS=${FFTN_MASKED_TAILS})")

# The old x86 targets, x86_v2 (SSE4.2) and avx (AVX without AVX2 or FMA): 52 % of an x86
# library's text (experiment X14) for processors from before 2013. ON keeps dispatch
# complete, as a distributed library needs; OFF leaves both out, and such processors run
# the scalar target. The default may come from the environment variable of the same name
# (CI builds the OFF configuration that way, .github/workflows/ci.yml); an empty value counts
# as unset. DESIGN.md 14.3; the choice is pending the maintainer (DESIGN.md 18.2).
set(_fftn_old_x86_default ON)
if(NOT "$ENV{FFTN_OLD_X86_TARGETS}" STREQUAL "")
  set(_fftn_old_x86_default "$ENV{FFTN_OLD_X86_TARGETS}")
endif()
option(FFTN_OLD_X86_TARGETS "Build the x86_v2 and avx targets (old x86-64 processors)"
       ${_fftn_old_x86_default})
if(FFTN_OLD_X86_TARGETS)
  set(FFTN_OLD_X86_TARGETS_ON 1)
else()
  set(FFTN_OLD_X86_TARGETS_ON 0)
endif()
message(STATUS "fftn: old x86 targets (x86_v2, avx) ${FFTN_OLD_X86_TARGETS_ON}")

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
  # GCC's ASan miscompiles AVX-512 code that uses the use-after-return fake stack: a 64-byte
  # local (an accessor's tile, aligned by GCC to a cache line) is stored with an aligned
  # vmovapd at an offset that assumes the frame base is 32 bytes off a 64-byte boundary, as
  # it is on the real stack, but the fake frame comes 64-byte aligned, so the store faults
  # (Xeon Gold 6240, GCC 14.4 and 15.2; 16.2 is correct; dev/LESSONS.md 6.4). The runtime
  # turns the fake stack on by default; the instrumentation is left out with those versions so
  # that tests pass however they are started.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16
     AND FFTN_SANITIZE MATCHES "address")
    add_compile_options(--param=asan-use-after-return=0)
  endif()
  # Sanitizer reports need line tables only (file and line of every frame), and full debug
  # information of the kernel units' huge functions multiplied the cost of an instrumented
  # build. src/isa/neon_fcma.cpp under ASan + UBSan at -O2 (Apple M5 Max): GCC 16 took
  # 1483 s and 23.4 GB with -g (variable tracking overflowed and was redone for every
  # radix-64 pass) against 809 s and 5.6 GB with -g1; Clang 23 671 s and 9.1 GB with -g
  # against 331 s and 4.7 GB with -gline-tables-only. With -g, CI's sanitizer jobs ran out
  # of memory (GCC on 16 GB runners) or time (macOS). These come after the build type's -g.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-gline-tables-only)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-g1)
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
  # The C API's tests and example are C (cmake/FftnCCompiler.cmake picks the compiler).
  include(FftnCCompiler)
  enable_language(C)
  include(FftnDeps)
  include(FftnTesting)
  add_subdirectory(tests)
  add_subdirectory(examples)
  fftn_write_coverage_manifest()
endif()

if(FFTN_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# Last: it reads the directories that the test and benchmark dependencies added.
include(FftnPackage)
