# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The C API (fftn/c_api.h, step 6.8; DESIGN.md 5.9), compiled into the static library.

target_sources(fftn PRIVATE c_api.cpp)

# The optional shared library of the C API (FFTN_BUILD_C_SHARED; DESIGN.md 18.2, Q20), for
# callers that can only load a shared library (Python's ctypes, Julia's ccall, some Fortran
# builds). It holds the static library inside and exports the fftn_ functions and nothing else
# (the linker's export list, and FFTN_C_API on the declarations), so its C++ stays private and
# two copies of FFTN in one process do not interpose (DESIGN.md 14.2). The C++ API stays static.
if(FFTN_BUILD_C_SHARED)
  set_target_properties(fftn PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(fftn_c SHARED c_api.cpp)
  add_library(fftn::fftn_c ALIAS fftn_c)
  fftn_configure_target(fftn_c)
  target_include_directories(fftn_c
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
    PRIVATE ${PROJECT_SOURCE_DIR}/src)
  # FFTN_C_EXPORTS exports the functions here; FFTN_C_SHARED imports them on Windows.
  target_compile_definitions(fftn_c PRIVATE FFTN_C_EXPORTS INTERFACE FFTN_C_SHARED)
  target_link_libraries(fftn_c PRIVATE fftn)
  set_target_properties(fftn_c PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  if(APPLE)
    target_link_options(fftn_c PRIVATE
      "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/fftn_c.exports")
  elseif(NOT WIN32)
    target_link_options(fftn_c PRIVATE
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/fftn_c.map")
  endif()
endif()
