# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build contracts of the public headers (DESIGN.md 15.3): each compiles alone, they link
# from several translation units, and they refuse fast-math.

fftn_add_test(fftn_api_tests SOURCES version_test.cpp)

# One translation unit per public header, including only that header, all linked into one
# executable (which also proves the headers define nothing twice across units).
file(GLOB public_headers CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/fftn/*.hpp)
set(alone_units "")
foreach(header IN LISTS public_headers)
  get_filename_component(stem "${header}" NAME_WE)
  set(unit "${CMAKE_CURRENT_BINARY_DIR}/alone_${stem}.cpp")
  file(CONFIGURE OUTPUT "${unit}" CONTENT "#include \"fftn/${stem}.hpp\"\n")
  list(APPEND alone_units "${unit}")
endforeach()
# The generated units are one #include each: nothing for clang-tidy to check but the header,
# which it checks through the other units.
set_source_files_properties(${alone_units} PROPERTIES SKIP_LINTING ON)
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
