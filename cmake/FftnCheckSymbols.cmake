# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The C API's symbol rule (step 6.8, DESIGN.md 5.9): every C symbol FFTN defines starts with
# fftn_, so that it cannot collide with another library's. Run as a test (tests/api):
#
#   cmake -DNM=<nm> -DLIBRARY=<file> -DMODE=static|shared [-DUNDERSCORE=ON] -P FftnCheckSymbols.cmake
#
# static: the global symbols defined in the static library may be C++ (mangled, _Z...),
#   the implementation's own (__ prefixed, or compiler-generated names with a dot), or fftn_.
# shared: the symbols the shared library fftn_c exports must all be fftn_.
# UNDERSCORE: the object format prefixes C names with an underscore (Mach-O).
if(MODE STREQUAL "shared" AND NOT UNDERSCORE)
  set(table -D) # ELF: the dynamic symbol table, what the library exports
else()
  set(table)
endif()
execute_process(COMMAND ${NM} -g ${table} --defined-only ${LIBRARY}
                OUTPUT_VARIABLE listing ERROR_VARIABLE errors RESULT_VARIABLE status)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "${NM} failed on ${LIBRARY}: ${errors}")
endif()
string(REPLACE "\n" ";" lines "${listing}")
set(bad)
set(c_api 0)
foreach(line IN LISTS lines)
  # "address type name", or "type name"; archive member headers end with a colon.
  if(NOT line MATCHES "^[0-9a-fA-F ]*[A-Za-z] ([^ ]+)$")
    continue()
  endif()
  set(name "${CMAKE_MATCH_1}")
  # One underscore only (REGEX REPLACE would take every leading one).
  if(UNDERSCORE AND name MATCHES "^_")
    string(SUBSTRING "${name}" 1 -1 name)
  endif()
  if(name MATCHES "^fftn_")
    math(EXPR c_api "${c_api} + 1")
  elseif(MODE STREQUAL "static" AND (name MATCHES "^_Z" OR name MATCHES "^__" OR name MATCHES "\\."))
    # C++, or the implementation's
  else()
    list(APPEND bad "${name}")
  endif()
endforeach()
if(c_api EQUAL 0)
  message(FATAL_ERROR "no fftn_ symbols found in ${LIBRARY}; the listing was:\n${listing}")
endif()
if(bad)
  list(REMOVE_DUPLICATES bad)
  list(JOIN bad "\n  " bad_lines)
  message(FATAL_ERROR "${LIBRARY} defines symbols outside the rules (${MODE}):\n  ${bad_lines}")
endif()
message(STATUS "${LIBRARY}: ${c_api} fftn_ symbols, nothing else of C's")
