# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# fftn_configure_target(<target>): the language level, warnings, floating-point flags,
# coverage instrumentation and clang-tidy that every FFTN target gets (AGENTS.md section
# 4). Third-party targets never get these: their warnings are not ours.

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(_fftn_warnings /W4 -Wconversion -Wsign-conversion -Wshadow)
  set(_fftn_werror /WX)
  # The compiler adds no FMAs; every FMA is explicit in the vocabulary (DESIGN.md 15.5).
  set(_fftn_fp /clang:-ffp-contract=off)
else()
  set(_fftn_warnings -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow)
  set(_fftn_werror -Werror)
  # -fno-fast-math is the default of GCC and Clang, but not of Intel's icpx, whose default
  # floating-point model reassociates, drops signed zeros, uses approximate reciprocals and
  # flushes denormals without defining __FAST_MATH__ (experiment X22). It comes first: icpx
  # warns when -ffp-contract=off overrides its -fp-model=precise, but not this pair.
  set(_fftn_fp -fno-fast-math -ffp-contract=off)
endif()

# clang-tidy from PyPI has no default SDK on macOS, so it would not find the standard
# headers that Homebrew's or Apple's compiler finds by itself.
set(_fftn_tidy_extra "")
if(APPLE AND FFTN_CLANG_TIDY)
  execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE _fftn_sdk
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(_fftn_tidy_extra "--extra-arg=-isysroot${_fftn_sdk}")
endif()

function(fftn_configure_target target)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_compile_options(${target} PRIVATE ${_fftn_warnings} ${_fftn_fp})
  # Read by the x86 vocabularies (CMakeLists.txt): the library's units and the tests' probes
  # must agree.
  target_compile_definitions(${target} PRIVATE FFTN_MASKED_TAILS=${FFTN_MASKED_TAILS_ON})
  # Whether the x86_v2 and avx targets are built (CMakeLists.txt): read by isa/arch.hpp.
  target_compile_definitions(${target} PRIVATE FFTN_OLD_X86_TARGETS=${FFTN_OLD_X86_TARGETS_ON})
  if(FFTN_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE ${_fftn_werror})
  endif()
  if(FFTN_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(${target} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
      target_link_options(${target} PRIVATE -fprofile-instr-generate)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(${target} PRIVATE --coverage)
      target_link_options(${target} PRIVATE --coverage)
    else()
      message(FATAL_ERROR "FFTN_COVERAGE needs Clang or GCC")
    endif()
  endif()
  if(FFTN_CLANG_TIDY)
    # Only FFTN's own headers are checked; generated codelets are covered by the
    # generator's verification instead (.clang-tidy).
    set(tidy ${FFTN_CLANG_TIDY}
      "--header-filter=^${PROJECT_SOURCE_DIR}/(include|src|tests|bench|examples)/"
      "--exclude-header-filter=^${PROJECT_SOURCE_DIR}/src/gen/" ${_fftn_tidy_extra})
    if(FFTN_WARNINGS_AS_ERRORS)
      list(APPEND tidy "--warnings-as-errors=*")
    endif()
    set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${tidy}")
  endif()
endfunction()

# Kernel translation units are optimised whatever the build type: unoptimised kernels run
# 30-60 times slower (DESIGN.md 14.1), which would make debug builds of applications useless.
function(fftn_always_optimise)
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(opt /O2)
  else()
    set(opt -O2)
  endif()
  set_property(SOURCE ${ARGN} APPEND PROPERTY COMPILE_OPTIONS
    "$<$<NOT:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:${opt}>")
endfunction()

# fftn_kernel_sanitizer_checks(<target> <units>...): in a build with both ASan and UBSan, the
# kernel units (those that compile the kernel text, isa/kernel_text.inl) leave out UBSan's
# pointer-overflow and object-size checks, which ASan already makes there: ASan checks every
# load and store against its object's exact bounds, which subsumes object-size's check of
# accesses against __builtin_object_size, and an address that wrapped or left its object is
# reported where it is used. The kernels' every access and index computation multiplied these
# two checks: Clang 23, one unit alone on an EPYC 7763, ASan + UBSan took 131 s
# (rows8_avx2) and 281 s (rows6_avx512), ASan alone 42 and 85 s, ASan + UBSan without the two
# 42 and 113 s; every other check together cost little. A whole cold build of the kernel units
# took 64 of the CI job's 78 minutes (PR 114). Every other UBSan check stays, and the other
# units keep all of them. <target> is the target the units belong to (a source property set
# in another directory is not seen there).
function(fftn_kernel_sanitizer_checks target)
  if(FFTN_SANITIZE MATCHES "address" AND FFTN_SANITIZE MATCHES "undefined")
    set_property(SOURCE ${ARGN} TARGET_DIRECTORY ${target} APPEND PROPERTY COMPILE_OPTIONS
      -fno-sanitize=pointer-overflow,object-size)
  endif()
endfunction()
