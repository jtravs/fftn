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

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(FftnTargets)

# Sanitizers apply to everything, third-party code included: a partly instrumented process
# gives false reports (TSan in particular).
if(FFTN_SANITIZE)
  add_compile_options(-fsanitize=${FFTN_SANITIZE} -fno-sanitize-recover=all
                      -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${FFTN_SANITIZE})
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
