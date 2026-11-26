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

# Row pipelines (tasks 5.2b, 5.2c) against the quadruple-precision reference, optimised likewise.
fftn_add_test(fftn_pipeline_tests SOURCES pipeline_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(pipeline_test.cpp)
# Stored runs and the depth-first pair on row pipelines (task 5.2d), likewise.
fftn_add_test(fftn_pipeline_runs_tests SOURCES pipeline_runs_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(pipeline_runs_test.cpp)

# Real-to-real plans (task 6.3) against the quadruple-precision reference, optimised likewise.
fftn_add_test(fftn_r2r_tests SOURCES r2r_test.cpp LIBRARIES fftn_test_support)
fftn_always_optimise(r2r_test.cpp)

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
