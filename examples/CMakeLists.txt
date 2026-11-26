# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Compiled, tested examples quoted by docs/ (AGENTS.md section 4). Each is a program that
# exits with 0 when its results are right, run as a test.

# team_threads and executor_rows start std::threads, which need the platform's thread library
# on some systems.
find_package(Threads REQUIRED)
foreach(example IN ITEMS axes callbacks easy edges executor_rows lengths pipeline planning
                        planes_volumes r2r rows_c2c team_threads)
  add_executable(fftn_example_${example} ${example}.cpp)
  fftn_configure_target(fftn_example_${example})
  target_link_libraries(fftn_example_${example} PRIVATE fftn::fftn Threads::Threads)
  add_test(NAME example.${example} COMMAND fftn_example_${example})
endforeach()

# The C API's example: a C99 program that links the C++ static library (CMake links it with
# the C++ compiler, which brings the C++ standard library).
add_executable(fftn_example_c_api c_api.c)
fftn_configure_target(fftn_example_c_api)
set_target_properties(fftn_example_c_api PROPERTIES C_STANDARD 99 C_EXTENSIONS OFF)
target_link_libraries(fftn_example_c_api PRIVATE fftn::fftn)
add_test(NAME example.c_api COMMAND fftn_example_c_api)
# The same program on the C API's shared library, when it is built (FFTN_BUILD_C_SHARED).
if(TARGET fftn_c)
  add_executable(fftn_example_c_api_shared c_api.c)
  fftn_configure_target(fftn_example_c_api_shared)
  set_target_properties(fftn_example_c_api_shared PROPERTIES C_STANDARD 99 C_EXTENSIONS OFF)
  target_link_libraries(fftn_example_c_api_shared PRIVATE fftn::fftn_c)
  # Windows finds a DLL beside the program, not in the build directory it was made in.
  if(WIN32)
    add_custom_command(TARGET fftn_example_c_api_shared POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_RUNTIME_DLLS:fftn_example_c_api_shared>
              $<TARGET_FILE_DIR:fftn_example_c_api_shared>
      COMMAND_EXPAND_LISTS)
  endif()
  add_test(NAME example.c_api_shared COMMAND fftn_example_c_api_shared)
endif()

# The Kokkos adapter's example needs Kokkos, which the test dependencies provide unless
# FFTN_TEST_WITH_KOKKOS is off (cmake/FftnTestDeps.cmake).
if(TARGET Kokkos::kokkos)
  add_executable(fftn_example_kokkos_rows_planes kokkos_rows_planes.cpp)
  fftn_configure_target(fftn_example_kokkos_rows_planes)
  target_link_libraries(fftn_example_kokkos_rows_planes PRIVATE fftn::fftn)
  fftn_link_kokkos(fftn_example_kokkos_rows_planes)
  add_test(NAME example.kokkos_rows_planes COMMAND fftn_example_kokkos_rows_planes)
endif()

# The OpenMP adapter's example needs the compiler's OpenMP runtime, which the test build finds
# unless FFTN_TEST_WITH_OPENMP is off (cmake/FftnTestDeps.cmake). Its imported target is
# local to the tests' directory, so it is looked up again here (the result is cached).
if(FFTN_TEST_WITH_OPENMP)
  find_package(OpenMP COMPONENTS CXX QUIET)
endif()
if(FFTN_TEST_WITH_OPENMP AND TARGET OpenMP::OpenMP_CXX)
  add_executable(fftn_example_omp_rows omp_rows.cpp)
  fftn_configure_target(fftn_example_omp_rows)
  target_link_libraries(fftn_example_omp_rows PRIVATE fftn::fftn)
  fftn_link_openmp(fftn_example_omp_rows)
  add_test(NAME example.omp_rows COMMAND fftn_example_omp_rows)
  # Owner-computes with a pipeline, the application's kernel and a plan with edges (task 5.5).
  add_executable(fftn_example_owner_computes owner_computes.cpp)
  fftn_configure_target(fftn_example_owner_computes)
  target_link_libraries(fftn_example_owner_computes PRIVATE fftn::fftn)
  fftn_link_openmp(fftn_example_owner_computes)
  add_test(NAME example.owner_computes COMMAND fftn_example_owner_computes)
endif()
