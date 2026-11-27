# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The optional Kokkos adapter, fftn/kokkos.hpp, on Kokkos' Serial back end and the build's
# threaded one (tasks 0.5b and 3.5; FFTN_KOKKOS_BACKEND in cmake/FftnTestDeps.cmake): results
# bitwise equal to the core API's and to serial execution, the Kokkos-FFT-shaped plan, the
# executor and team_of, their contracts and validation. The allocation hook replaces operator
# new in this executable only. Skipped when the build has no Kokkos (FFTN_TEST_WITH_KOKKOS off).

if(TARGET Kokkos::kokkos)
  fftn_add_test(fftn_kokkos_tests
    SOURCES kokkos_test.cpp kokkos_contract_test.cpp kokkos_executor_test.cpp kokkos_oneshot_test.cpp
    LIBRARIES fftn_test_support fftn_test_alloc_hook OWN_MAIN)
  fftn_link_kokkos(fftn_kokkos_tests)
  # The whole suite again on a pool of one thread: the threaded back end at width 1.
  add_test(NAME kokkos.one_thread COMMAND fftn_kokkos_tests)
  set_tests_properties(kokkos.one_thread PROPERTIES ENVIRONMENT FFTN_KOKKOS_TEST_THREADS=1)
endif()
