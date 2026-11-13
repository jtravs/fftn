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
  set(_fftn_fp -ffp-contract=off)
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
