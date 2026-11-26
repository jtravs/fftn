# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# A separate project that uses FFTN as its users do (docs/building.md), built by CI
# (.github/workflows/package.yml) against an install tree and through FetchContent. It
# compiles the documented examples next to this directory; each exits with 0 when its results
# are right, run as a test.
#
#   FFTN_PACKAGE_FROM  find_package (default): an installed FFTN on CMAKE_PREFIX_PATH;
#                      fetch: FetchContent (FETCHCONTENT_SOURCE_DIR_FFTN for a local copy)
#   FFTN_PACKAGE_OMP   also the OpenMP adapter's example, through fftn::omp

cmake_minimum_required(VERSION 3.25)
project(fftn_package_test LANGUAGES CXX)

set(FFTN_PACKAGE_FROM find_package CACHE STRING "How FFTN arrives: find_package or fetch")
option(FFTN_PACKAGE_OMP "Build the OpenMP adapter's example through fftn::omp" OFF)

if(FFTN_PACKAGE_FROM STREQUAL "fetch")
  include(FetchContent)
  FetchContent_Declare(fftn GIT_REPOSITORY https://github.com/jtravs/fftn.git GIT_TAG main)
  FetchContent_MakeAvailable(fftn)
elseif(FFTN_PACKAGE_FROM STREQUAL "find_package")
  set(components)
  if(FFTN_PACKAGE_OMP)
    list(APPEND components omp)
  endif()
  find_package(fftn 0.0 REQUIRED COMPONENTS ${components})
  message(STATUS "fftn ${fftn_VERSION} from ${fftn_DIR}")
else()
  message(FATAL_ERROR "FFTN_PACKAGE_FROM must be find_package or fetch")
endif()

enable_testing()
add_executable(rows_c2c ../rows_c2c.cpp)
target_link_libraries(rows_c2c PRIVATE fftn::fftn)
add_test(NAME rows_c2c COMMAND rows_c2c)
if(FFTN_PACKAGE_OMP)
  add_executable(omp_rows ../omp_rows.cpp)
  target_link_libraries(omp_rows PRIVATE fftn::omp)
  add_test(NAME omp_rows COMMAND omp_rows)
endif()
