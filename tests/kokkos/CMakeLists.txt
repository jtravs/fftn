# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The optional Kokkos adapter, fftn/kokkos.hpp, on Kokkos' Serial back end (task 0.5b):
# results bitwise equal to the core API's, the Kokkos-FFT-shaped plan, its contracts and its
# validation. The allocation hook replaces operator new in this executable only.
# Skipped when the build has no Kokkos (FFTN_TEST_WITH_KOKKOS off).

if(TARGET Kokkos::kokkos)
  fftn_add_test(fftn_kokkos_tests SOURCES kokkos_test.cpp kokkos_contract_test.cpp
    LIBRARIES fftn_test_support fftn_test_alloc_hook OWN_MAIN)
  fftn_link_kokkos(fftn_kokkos_tests)
endif()
