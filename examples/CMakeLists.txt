# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Compiled, tested examples quoted by docs/ (AGENTS.md section 4). Each is a program that
# exits with 0 when its results are right, run as a test.

foreach(example IN ITEMS rows_c2c)
  add_executable(fftn_example_${example} ${example}.cpp)
  fftn_configure_target(fftn_example_${example})
  target_link_libraries(fftn_example_${example} PRIVATE fftn::fftn)
  add_test(NAME example.${example} COMMAND fftn_example_${example})
endforeach()

# The Kokkos adapter's example needs Kokkos, which the test dependencies provide unless
# FFTN_TEST_WITH_KOKKOS is off (cmake/FftnTestDeps.cmake).
if(TARGET Kokkos::kokkos)
  add_executable(fftn_example_kokkos_rows_planes kokkos_rows_planes.cpp)
  fftn_configure_target(fftn_example_kokkos_rows_planes)
  target_link_libraries(fftn_example_kokkos_rows_planes PRIVATE fftn::fftn)
  fftn_link_kokkos(fftn_example_kokkos_rows_planes)
  add_test(NAME example.kokkos_rows_planes COMMAND fftn_example_kokkos_rows_planes)
endif()
