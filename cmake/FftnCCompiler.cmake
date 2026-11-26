# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# The C compiler of the C API's tests and example (fftn/c_api.h; tests/api, examples/c_api.c),
# included before C is enabled. When neither CMAKE_C_COMPILER nor CC names one, it is the C++
# compiler's sibling (clang++-23 gives clang-23, g++-16 gcc-16, c++ cc; clang-cl is both), so
# that the C and C++ objects of one executable come from one toolchain: the sanitizer and
# coverage runtimes of two compilers do not mix, and CI names only CXX.
if(NOT CMAKE_C_COMPILER AND "$ENV{CC}" STREQUAL "")
  get_filename_component(_fftn_cxx_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
  get_filename_component(_fftn_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
  set(_fftn_c_name "")
  if(_fftn_cxx_name MATCHES "^(.*)clang\\+\\+(.*)$")
    set(_fftn_c_name "${CMAKE_MATCH_1}clang${CMAKE_MATCH_2}")
  elseif(_fftn_cxx_name MATCHES "^(.*)g\\+\\+(.*)$")
    set(_fftn_c_name "${CMAKE_MATCH_1}gcc${CMAKE_MATCH_2}")
  elseif(_fftn_cxx_name MATCHES "^(.*)c\\+\\+(.*)$")
    set(_fftn_c_name "${CMAKE_MATCH_1}cc${CMAKE_MATCH_2}")
  elseif(_fftn_cxx_name MATCHES "^icpx(.*)$")
    set(_fftn_c_name "icx${CMAKE_MATCH_1}")
  elseif(_fftn_cxx_name MATCHES "^clang-cl")
    set(_fftn_c_name "${_fftn_cxx_name}")
  endif()
  if(_fftn_c_name AND EXISTS "${_fftn_cxx_dir}/${_fftn_c_name}")
    set(CMAKE_C_COMPILER "${_fftn_cxx_dir}/${_fftn_c_name}")
  endif()
endif()
