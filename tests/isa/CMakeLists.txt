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

# Disassembly test: the library's units and the probes, per target (tools/asm_check.py).
# Python runs in the project's uv environment, or any Python 3.14 if uv is missing.
find_program(FFTN_UV uv)
if(FFTN_UV)
  set(fftn_python ${FFTN_UV} run --frozen --quiet python)
else()
  find_package(Python3 3.14 REQUIRED COMPONENTS Interpreter)
  set(fftn_python ${Python3_EXECUTABLE})
endif()
if(NOT CMAKE_OBJDUMP)
  message(FATAL_ERROR "isa tests: no objdump found for the disassembly test")
endif()
set(fftn_asm_check ${fftn_python} ${PROJECT_SOURCE_DIR}/tools/asm_check.py
                   --objdump ${CMAKE_OBJDUMP})
# What the build's own baseline already has may appear in baseline code. On Apple platforms
# the compiler targets Apple silicon (apple-m1 or newer), which has FCMA, and keeps doing so
# under -march=armv8-a (only -mcpu=generic drops it): scalar code there may use FCADD
# although __ARM_FEATURE_COMPLEX is not defined. Every Apple arm64 processor has FCMA, and
# pointer authentication, which instrumented baseline code there may use (ThreadSanitizer's
# function-entry hook strips its return address with XPACI).
include(CheckCXXSourceCompiles)
set(fftn_baseline_has "")
set(CMAKE_REQUIRED_QUIET ON)
foreach(feature IN ITEMS "avx:__AVX__" "avx512:__AVX512F__" "fcma:__ARM_FEATURE_COMPLEX"
                        "pauth:__ARM_FEATURE_PAUTH")
  string(REPLACE ":" ";" feature "${feature}")
  list(GET feature 0 name)
  list(GET feature 1 macro)
  check_cxx_source_compiles("#ifndef ${macro}\n#error\n#endif\nint main() {}"
                            FFTN_BASELINE_HAS_${name})
  if(FFTN_BASELINE_HAS_${name})
    list(APPEND fftn_baseline_has --baseline-has ${name})
  endif()
endforeach()
unset(CMAKE_REQUIRED_QUIET)
if(APPLE)
  foreach(name IN ITEMS fcma pauth)
    if(NOT FFTN_BASELINE_HAS_${name})
      list(APPEND fftn_baseline_has --baseline-has ${name})
    endif()
  endforeach()
endif()
# With LTO the objects hold compiler IR, not code: the linked tests are checked instead.
if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  set(fftn_asm_inputs $<TARGET_FILE:fftn_isa_tests>)
else()
  set(fftn_asm_inputs $<TARGET_FILE:fftn> "$<TARGET_OBJECTS:fftn_isa_probes>")
endif()
add_test(NAME isa.asm_check
         COMMAND ${fftn_asm_check} ${fftn_baseline_has} ${fftn_asm_inputs}
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# The check's own tests: a unit that breaks the region rules on purpose (asm_faults.cpp)
# must be reported, one kind of fault per test.
add_library(fftn_isa_asm_faults OBJECT asm_faults.cpp)
fftn_configure_target(fftn_isa_asm_faults)
# Machine code even in LTO builds: the check reads this object itself.
set_target_properties(fftn_isa_asm_faults PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
target_include_directories(fftn_isa_asm_faults PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(fftn_isa_asm_faults PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror=psabi>)
function(fftn_asm_fault_test kind pattern)
  add_test(NAME isa.asm_check_finds_${kind}
           COMMAND ${fftn_asm_check} "$<TARGET_OBJECTS:fftn_isa_asm_faults>"
           WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(isa.asm_check_finds_${kind} PROPERTIES PASS_REGULAR_EXPRESSION "${pattern}")
endfunction()
fftn_asm_fault_test(calls "calls .* must inline")
fftn_asm_fault_test(baseline_isa "baseline: (VEX or EVEX code|FCMA instruction) in")
fftn_asm_fault_test(target_isa "EVEX feature in|Armv8.3 instruction outside FCMA in")
check_cxx_source_compiles("#if !defined(__aarch64__) && !defined(_M_ARM64)\n#error\n#endif\nint main() {}"
                          FFTN_TARGET_IS_AARCH64)
if(FFTN_TARGET_IS_AARCH64)
  fftn_asm_fault_test(pauth "pointer authentication outside neon_fcma in")
endif()
