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

# One translation unit per public header, including only that header, all linked into one
# executable (which also proves the headers define nothing twice across units).
file(GLOB public_headers CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/fftn/*.hpp)
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
