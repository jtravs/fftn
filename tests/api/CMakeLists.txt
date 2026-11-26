# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build contracts of the public headers (DESIGN.md 15.3): each compiles alone, they link
# from several translation units, and they refuse fast-math. Then the API's own contracts.

fftn_add_test(fftn_api_tests SOURCES version_test.cpp)

# The plan, workspace and shape contracts (task 0.5), and transforms of rank 2 and 3 against
# the reference. The allocation hook replaces operator new in this executable only.
fftn_add_test(fftn_contract_tests
  SOURCES contract_test.cpp multidim_test.cpp
  LIBRARIES fftn_test_support fftn_test_alloc_hook)

# Plan edges (task 5.1a) against the quadruple-precision reference, which is heavy, so the
# test is optimised in every build type.
fftn_add_test(fftn_edges_tests SOURCES edges_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(edges_test.cpp)
fftn_add_test(fftn_edges_shapes_tests SOURCES edges_shapes_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(edges_shapes_test.cpp)
# Block callbacks (task 5.3a) against the reference and the user's own loops.
fftn_add_test(fftn_callbacks_tests SOURCES callbacks_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(callbacks_test.cpp)
# Block callbacks in every row strategy and on real rows (task 5.3b).
fftn_add_test(fftn_callbacks_strategies_tests SOURCES callbacks_strategies_test.cpp
              LIBRARIES fftn_test_support)
fftn_always_optimise(callbacks_strategies_test.cpp)
# Block callbacks on planes and volumes (task 5.3c).
fftn_add_test(fftn_callbacks_axes_tests SOURCES callbacks_axes_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(callbacks_axes_test.cpp)

# Row pipelines (tasks 5.2b, 5.2c) against the quadruple-precision reference, optimised likewise.
fftn_add_test(fftn_pipeline_tests SOURCES pipeline_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(pipeline_test.cpp)
# Stored runs and the depth-first pair on row pipelines (task 5.2d), likewise.
fftn_add_test(fftn_pipeline_runs_tests SOURCES pipeline_runs_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(pipeline_runs_test.cpp)

# Real-to-real plans (task 6.3) against the quadruple-precision reference, optimised likewise.
fftn_add_test(fftn_r2r_tests SOURCES r2r_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(r2r_test.cpp)

# The C API (step 6.8): its C half compiled as C99 with the strict warnings (-Wpedantic among
# them, from fftn_configure_target), checked against the C++ plans of the same shapes; threads
# for the team members and the C thread pool.
find_package(Threads REQUIRED)
fftn_add_test(fftn_c_api_tests SOURCES c_api_test.cpp c_api_test.c LIBRARIES Threads::Threads)
set_target_properties(fftn_c_api_tests PROPERTIES C_STANDARD 99 C_EXTENSIONS OFF)
if(NOT CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(fftn_c_api_tests PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wstrict-prototypes -Wmissing-prototypes>)
endif()

# Every C symbol of the library starts with fftn_, and the shared library of the C API exports
# nothing else (cmake/FftnCheckSymbols.cmake). Where CMake finds no nm, and with clang-cl, whose
# objects are COFF with their own decoration, the check is left out.
if(CMAKE_NM AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(fftn_symbols ${CMAKE_COMMAND} -DNM=${CMAKE_NM})
  if(APPLE)
    list(APPEND fftn_symbols -DUNDERSCORE=ON)
  endif()
  add_test(NAME api.c_symbols
           COMMAND ${fftn_symbols} -DLIBRARY=$<TARGET_FILE:fftn> -DMODE=static
                   -P ${PROJECT_SOURCE_DIR}/cmake/FftnCheckSymbols.cmake)
  if(TARGET fftn_c)
    add_test(NAME api.c_symbols_shared
             COMMAND ${fftn_symbols} -DLIBRARY=$<TARGET_FILE:fftn_c> -DMODE=shared
                     -P ${PROJECT_SOURCE_DIR}/cmake/FftnCheckSymbols.cmake)
  endif()
endif()

# One translation unit per public header, including only that header, all linked into one
# executable (which also proves the headers define nothing twice across units).
file(GLOB public_headers CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/fftn/*.hpp)
# kokkos.hpp and omp.hpp are the optional adapters: they need Kokkos and OpenMP, so each gets
# an executable of its own below, built only when the test build has its runtime.
list(FILTER public_headers EXCLUDE REGEX "/(kokkos|omp)\\.hpp$")
set(alone_units "")
foreach(header IN LISTS public_headers)
  get_filename_component(stem "${header}" NAME_WE)
  set(unit "${CMAKE_CURRENT_BINARY_DIR}/alone_${stem}.cpp")
  # The unit exists only to include the header, which include-cleaner would call unused.
  # A NOLINT rather than SKIP_LINTING: that property needs CMake 3.27 and FFTN supports 3.25.
  file(CONFIGURE OUTPUT "${unit}" CONTENT
    "#include \"fftn/${stem}.hpp\" // NOLINT(misc-include-cleaner): this unit only includes it\n")
  list(APPEND alone_units "${unit}")
endforeach()
add_executable(fftn_headers_alone headers_alone_main.cpp ${alone_units})
fftn_configure_target(fftn_headers_alone)
target_link_libraries(fftn_headers_alone PRIVATE fftn::fftn)
add_test(NAME api.headers_alone COMMAND fftn_headers_alone)

# kokkos.hpp alone, in two units linked together (it defines nothing twice), with Kokkos.
if(TARGET Kokkos::kokkos)
  set(kokkos_units "")
  foreach(n IN ITEMS 1 2)
    set(unit "${CMAKE_CURRENT_BINARY_DIR}/alone_kokkos_${n}.cpp")
    file(CONFIGURE OUTPUT "${unit}" CONTENT
      "#include \"fftn/kokkos.hpp\" // NOLINT(misc-include-cleaner): this unit only includes it\n")
    list(APPEND kokkos_units "${unit}")
  endforeach()
  add_executable(fftn_kokkos_header_alone headers_alone_main.cpp ${kokkos_units})
  fftn_configure_target(fftn_kokkos_header_alone)
  target_link_libraries(fftn_kokkos_header_alone PRIVATE fftn::fftn)
  fftn_link_kokkos(fftn_kokkos_header_alone)
  add_test(NAME api.kokkos_header_alone COMMAND fftn_kokkos_header_alone)
endif()

# omp.hpp alone, in two units linked together, with OpenMP.
if(FFTN_TEST_WITH_OPENMP AND TARGET OpenMP::OpenMP_CXX)
  set(omp_units "")
  foreach(n IN ITEMS 1 2)
    set(unit "${CMAKE_CURRENT_BINARY_DIR}/alone_omp_${n}.cpp")
    file(CONFIGURE OUTPUT "${unit}" CONTENT
      "#include \"fftn/omp.hpp\" // NOLINT(misc-include-cleaner): this unit only includes it\n")
    list(APPEND omp_units "${unit}")
  endforeach()
  add_executable(fftn_omp_header_alone headers_alone_main.cpp ${omp_units})
  fftn_configure_target(fftn_omp_header_alone)
  target_link_libraries(fftn_omp_header_alone PRIVATE fftn::fftn)
  fftn_link_openmp(fftn_omp_header_alone)
  add_test(NAME api.omp_header_alone COMMAND fftn_omp_header_alone)
endif()

# The C contract of executors (fftn/executor.h) is C: it must compile as C99, with every
# warning, by the C front end of the compiler that builds FFTN (a GNU-style driver; clang-cl
# would need its own spelling of the options).
if(NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  add_test(NAME api.executor_h_is_c
           COMMAND ${CMAKE_CXX_COMPILER} -x c -std=c99 -pedantic-errors -Wall -Wextra -Werror
                   -fsyntax-only ${PROJECT_SOURCE_DIR}/include/fftn/executor.h)
endif()

# A translation unit compiled with fast-math must be refused with FFTN's own message.
if(NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  add_library(fftn_fast_math_refused OBJECT EXCLUDE_FROM_ALL fast_math_refused.cpp)
  target_link_libraries(fftn_fast_math_refused PRIVATE fftn::fftn)
  target_compile_options(fftn_fast_math_refused PRIVATE -ffast-math)
  add_test(NAME api.fast_math_refused
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target fftn_fast_math_refused)
  set_tests_properties(api.fast_math_refused PROPERTIES
    PASS_REGULAR_EXPRESSION "FFTN must not be compiled with fast-math")
endif()

# The convenience layer (step 6.6) against the reference evaluated numpy's way, with the
# allocation hook for its cache. Optimised in every build type, as the reference is heavy.
fftn_add_test(fftn_easy_tests SOURCES easy_test.cpp
  LIBRARIES fftn_test_support fftn_test_alloc_hook)
fftn_always_optimise(easy_test.cpp)
# Its real-to-real functions (step 6.7) against scipy's results recorded in data/ (written by
# data/scipy_r2r_cases.py, run by hand outside the build).
fftn_add_test(fftn_easy_r2r_tests SOURCES easy_r2r_test.cpp
  LIBRARIES fftn_test_support fftn_test_alloc_hook)
fftn_always_optimise(easy_r2r_test.cpp)
target_compile_definitions(fftn_easy_r2r_tests PRIVATE
  FFTN_EASY_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
# Its conversion from std::mdspan, in a C++23 unit where the compiler has the standard; the test
# skips itself where the standard library lacks std::mdspan.
if(cxx_std_23 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  fftn_add_test(fftn_easy_mdspan_tests SOURCES easy_mdspan_test.cpp)
  set_target_properties(fftn_easy_mdspan_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
endif()
