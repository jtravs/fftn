# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0

# Test-only oracles (Boost.Multiprecision, FFTW) for the suites below.
include(FftnTestDeps)

add_subdirectory(api)
add_subdirectory(platform)
add_subdirectory(isa)
add_subdirectory(support)
add_subdirectory(reference)
add_subdirectory(analytic)
add_subdirectory(fftw)
add_subdirectory(core)
add_subdirectory(kokkos)
