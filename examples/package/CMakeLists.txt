# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# A separate project that uses FFTN as its users do (docs/building.md), built by CI
# (.github/workflows/package.yml) against an install tree and through FetchContent. It
# compiles the documented examples next to this directory, the C API's in C; each exits with 0
# when its results are right, run as a test.
#
#   FFTN_PACKAGE_FROM  find_package (default): an installed FFTN on CMAKE_PREFIX_PATH;
#                      fetch: FetchContent (FETCHCONTENT_SOURCE_DIR_FFTN for a local copy)
#   FFTN_PACKAGE_OMP   also the OpenMP adapter's example, through fftn::omp

cmake_minimum_required(VERSION 3.25)
project(fftn_package_test LANGUAGES C CXX)

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
# A C program on the C API (fftn/c_api.h), linked with the C++ static library.
add_executable(c_api ../c_api.c)
set_target_properties(c_api PROPERTIES C_STANDARD 99 C_EXTENSIONS OFF)
target_link_libraries(c_api PRIVATE fftn::fftn)
add_test(NAME c_api COMMAND c_api)
# The same program on the C API's shared library, when the installed FFTN has it
# (FFTN_BUILD_C_SHARED); on Windows its DLL is copied beside the program.
if(TARGET fftn::fftn_c)
  add_executable(c_api_shared ../c_api.c)
  set_target_properties(c_api_shared PROPERTIES C_STANDARD 99 C_EXTENSIONS OFF)
  target_link_libraries(c_api_shared PRIVATE fftn::fftn_c)
  if(WIN32)
    add_custom_command(TARGET c_api_shared POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:c_api_shared>
              $<TARGET_FILE_DIR:c_api_shared>
      COMMAND_EXPAND_LISTS)
  endif()
  add_test(NAME c_api_shared COMMAND c_api_shared)
endif()
if(FFTN_PACKAGE_OMP)
  add_executable(omp_rows ../omp_rows.cpp)
  target_link_libraries(omp_rows PRIVATE fftn::omp)
  add_test(NAME omp_rows COMMAND omp_rows)
endif()
