# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# fftn_add_test(<name> SOURCES <files...> [LIBRARIES <targets...>] [OWN_MAIN])
#   A GoogleTest executable with FFTN's flags, access to the private sources (src/) and
#   the shared test support (tests/), registered with CTest test by test and recorded for
#   the coverage gate. OWN_MAIN: the sources define main() (for a runtime that must be
#   initialised around the tests, such as Kokkos) instead of GoogleTest's.
#
# fftn_write_coverage_manifest()
#   Writes <build>/fftn-coverage.json: the coverage tool and the instrumented executables,
#   read by tools/coverage_gate.py. Called once, after every test is defined.

include(GoogleTest)

function(fftn_add_test name)
  cmake_parse_arguments(ARG "OWN_MAIN" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${ARG_SOURCES})
  fftn_configure_target(${name})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests)
  if(ARG_OWN_MAIN)
    set(gtest GTest::gtest)
  else()
    set(gtest GTest::gtest_main)
  endif()
  target_link_libraries(${name} PRIVATE fftn::fftn ${gtest} ${ARG_LIBRARIES})
  gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST)
  set_property(GLOBAL APPEND PROPERTY FFTN_TEST_EXECUTABLES "$<TARGET_FILE:${name}>")
endfunction()

function(fftn_write_coverage_manifest)
  if(NOT FFTN_COVERAGE)
    return()
  endif()
  get_filename_component(bindir "${CMAKE_CXX_COMPILER}" DIRECTORY)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    execute_process(COMMAND xcrun -f llvm-cov OUTPUT_VARIABLE cov OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND xcrun -f llvm-profdata OUTPUT_VARIABLE prof
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(kind llvm)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # The tools must match the compiler's profile format: take them from its directory.
    find_program(cov NAMES llvm-cov HINTS "${bindir}" NO_DEFAULT_PATH)
    find_program(prof NAMES llvm-profdata HINTS "${bindir}" NO_DEFAULT_PATH)
    set(kind llvm)
  else()
    # g++-14 pairs with gcov-14, g++ with gcov, in the same directory.
    get_filename_component(cxx "${CMAKE_CXX_COMPILER}" NAME)
    string(REGEX REPLACE "^.*g\\+\\+" "gcov" gcov_name "${cxx}")
    find_program(cov NAMES ${gcov_name} gcov HINTS "${bindir}" NO_DEFAULT_PATH)
    set(prof "")
    set(kind gcc)
  endif()
  if(NOT cov)
    message(FATAL_ERROR "coverage: no coverage tool found next to ${CMAKE_CXX_COMPILER}")
  endif()
  get_property(exes GLOBAL PROPERTY FFTN_TEST_EXECUTABLES)
  list(JOIN exes "\",\n    \"" exes_json)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/fftn-coverage.json" CONTENT "{
  \"kind\": \"${kind}\",
  \"cov\": \"${cov}\",
  \"profdata\": \"${prof}\",
  \"source_root\": \"${PROJECT_SOURCE_DIR}\",
  \"build_root\": \"${CMAKE_BINARY_DIR}\",
  \"executables\": [
    \"${exes_json}\"
  ]
}
")
endfunction()
