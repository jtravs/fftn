# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# ISA regions, vocabularies and dispatch (DESIGN.md 6, 15.3).

# The vocabulary probes: one unit per target, like the library's kernel units, compiled
# into an object library so the disassembly check can read them.
add_library(fftn_isa_probes OBJECT probe_scalar.cpp probe_avx2.cpp probe_neon_fcma.cpp
                                   probe_tables.cpp)
fftn_configure_target(fftn_isa_probes)
# Only fftn's headers: the test executable links the library itself.
target_include_directories(fftn_isa_probes PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                   ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests)
target_compile_options(fftn_isa_probes PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror=psabi>)

fftn_add_test(fftn_isa_tests
  SOURCES target_test.cpp vocab_test.cpp kernel_test.cpp link_test.cpp test_targets.cpp
  LIBRARIES fftn_isa_probes)
