# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Test-only oracles beyond GoogleTest (DESIGN.md 14.3 and 15.1), fetched pinned by SHA-256
# and treated as system code. The library itself depends on none of them.
#
#   fftn_test_boost_mp  Boost.Multiprecision, header-only and standalone, for the
#                       quadruple-precision reference DFT. Its standalone mode still needs
#                       Boost.Config (dependency-free, header-only), so both are fetched;
#                       nothing else of Boost is used (sin, cos and pi come from
#                       Multiprecision's own default operations).
#   fftn_test_fftw      FFTW 3.3.11 in double and float, an independent implementation
#                       (FFTN_TEST_WITH_FFTW, default ON). It is built from source with its
#                       own CMake build, plain C without SIMD (an oracle needs no speed),
#                       or taken from FFTN_FFTW_ROOT, an existing install that has both
#                       libfftw3 and libfftw3f. Consumers always link this target; the
#                       definition FFTN_TEST_HAVE_FFTW (0 or 1) tells them whether FFTW is
#                       there, and tests that need it skip themselves when it is not.
#
# Licence: FFTW is GPL-2.0-or-later. Only test executables link it; they are built for our
# own use and never distributed (AGENTS.md section 10). Boost is under the Boost Software
# License.
#
# Offline builds: FETCHCONTENT_SOURCE_DIR_BOOST_CONFIG, FETCHCONTENT_SOURCE_DIR_BOOST_MP and
# FETCHCONTENT_SOURCE_DIR_FFTN_FFTW point at unpacked copies of the archives below.

include_guard(GLOBAL)
include(FetchContent)

# SOURCE_SUBDIR names a directory the archives do not have, so FetchContent only unpacks
# them: their own CMake projects (Boost's superproject machinery, FFTW's install rules)
# never join ours.
FetchContent_Declare(boost_config
  URL https://github.com/boostorg/config/archive/refs/tags/boost-1.92.0.tar.gz
  URL_HASH SHA256=b4171037f13373203ba79cbc141d612982052283e696a315185ab5bea46102a0
  SOURCE_SUBDIR fftn-unpack-only
  SYSTEM)
FetchContent_Declare(boost_mp
  URL https://github.com/boostorg/multiprecision/archive/refs/tags/boost-1.92.0.tar.gz
  URL_HASH SHA256=9da997843edd802f9d5e25d53872a7676330f31a52a5e47a418d1c94ba3b01d0
  SOURCE_SUBDIR fftn-unpack-only
  SYSTEM)
FetchContent_MakeAvailable(boost_config boost_mp)

add_library(fftn_test_boost_mp INTERFACE)
target_include_directories(fftn_test_boost_mp SYSTEM INTERFACE
  ${boost_mp_SOURCE_DIR}/include ${boost_config_SOURCE_DIR}/include)
target_compile_definitions(fftn_test_boost_mp INTERFACE BOOST_MP_STANDALONE)

option(FFTN_TEST_WITH_FFTW "Use FFTW 3.3.11 as a test oracle (GPL; test binaries only)" ON)
set(FFTN_FFTW_ROOT "" CACHE PATH
  "Existing FFTW install with libfftw3 and libfftw3f (empty: build FFTW from source)")
set(FFTN_TEST_C_COMPILER "" CACHE FILEPATH
  "C compiler for the FFTW build (empty: the C driver next to the C++ compiler)")

add_library(fftn_test_fftw INTERFACE)
if(NOT FFTN_TEST_WITH_FFTW)
  target_compile_definitions(fftn_test_fftw INTERFACE FFTN_TEST_HAVE_FFTW=0)
  return()
endif()
target_compile_definitions(fftn_test_fftw INTERFACE FFTN_TEST_HAVE_FFTW=1)
if(UNIX AND NOT APPLE)
  set(_fftn_fftw_libm m)
else()
  set(_fftn_fftw_libm "")
endif()

if(FFTN_FFTW_ROOT)
  find_path(_fftn_fftw_include fftw3.h PATHS ${FFTN_FFTW_ROOT}/include NO_DEFAULT_PATH)
  find_library(_fftn_fftw_double fftw3 PATHS ${FFTN_FFTW_ROOT}/lib ${FFTN_FFTW_ROOT}/lib64
               NO_DEFAULT_PATH)
  find_library(_fftn_fftw_float fftw3f PATHS ${FFTN_FFTW_ROOT}/lib ${FFTN_FFTW_ROOT}/lib64
               NO_DEFAULT_PATH)
  if(NOT _fftn_fftw_include OR NOT _fftn_fftw_double OR NOT _fftn_fftw_float)
    message(FATAL_ERROR "FFTN_FFTW_ROOT=${FFTN_FFTW_ROOT} must hold include/fftw3.h and both "
                        "libfftw3 and libfftw3f")
  endif()
  target_include_directories(fftn_test_fftw SYSTEM INTERFACE ${_fftn_fftw_include})
  target_link_libraries(fftn_test_fftw INTERFACE
    ${_fftn_fftw_double} ${_fftn_fftw_float} ${_fftn_fftw_libm})
  return()
endif()

FetchContent_Declare(fftn_fftw
  URL https://www.fftw.org/fftw-3.3.11.tar.gz
  URL_HASH SHA256=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
  SOURCE_SUBDIR fftn-unpack-only)
FetchContent_MakeAvailable(fftn_fftw)

# FFTW is C. The project enables only C++, so the C driver is taken from beside the C++
# compiler: mixing compilers would mix sanitizer runtimes.
if(FFTN_TEST_C_COMPILER)
  set(_fftn_cc ${FFTN_TEST_C_COMPILER})
else()
  get_filename_component(_fftn_cxx_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
  get_filename_component(_fftn_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
  string(REGEX REPLACE "clang\\+\\+" "clang" _fftn_cc_name "${_fftn_cxx_name}")
  string(REGEX REPLACE "g\\+\\+" "gcc" _fftn_cc_name "${_fftn_cc_name}")
  string(REGEX REPLACE "^c\\+\\+" "cc" _fftn_cc_name "${_fftn_cc_name}")
  set(_fftn_cc "${_fftn_cxx_dir}/${_fftn_cc_name}")
  if(NOT EXISTS "${_fftn_cc}")
    message(FATAL_ERROR "No C compiler found next to ${CMAKE_CXX_COMPILER} for the FFTW "
                        "oracle: set FFTN_TEST_C_COMPILER, FFTN_FFTW_ROOT or turn "
                        "FFTN_TEST_WITH_FFTW off")
  endif()
endif()

# FFTW is optimised whatever the build type (a Debug FFTW would only slow the tests down)
# and gets the process's address or thread sanitizer, so that no part of a test process is
# uninstrumented. It does not get UBSan: FFTW 3.3.11's planners do arithmetic on null
# pointers (dft/dftw-direct.c, dft/direct.c, rdft/ct-hc2c-direct.c), which UBSan reports
# and which is not ours to fix.
set(_fftn_fftw_cflags "")
set(_fftn_fftw_sanitizers ${FFTN_SANITIZE})
string(REPLACE "," ";" _fftn_fftw_sanitizers "${_fftn_fftw_sanitizers}")
list(REMOVE_ITEM _fftn_fftw_sanitizers undefined)
if(_fftn_fftw_sanitizers)
  list(JOIN _fftn_fftw_sanitizers "," _fftn_fftw_sanitizers)
  set(_fftn_fftw_cflags "-fsanitize=${_fftn_fftw_sanitizers} -fno-omit-frame-pointer")
endif()
set(_fftn_fftw_args
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=${_fftn_cc}
  "-DCMAKE_C_FLAGS=${_fftn_fftw_cflags}"
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  # FFTW 3.3.11 asks for CMake 3.0; CMake 4 has dropped compatibility below 3.5.
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_TESTS=OFF
  -DDISABLE_FORTRAN=ON)
if(CMAKE_OSX_SYSROOT)
  list(APPEND _fftn_fftw_args -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT})
endif()

include(ExternalProject)
target_include_directories(fftn_test_fftw SYSTEM INTERFACE ${fftn_fftw_SOURCE_DIR}/api)
foreach(_fftn_prec IN ITEMS double float)
  if(_fftn_prec STREQUAL "float")
    set(_fftn_lib fftw3f)
    set(_fftn_float ON)
  else()
    set(_fftn_lib fftw3)
    set(_fftn_float OFF)
  endif()
  # The archive is taken from FFTW's build tree (headers come from its source tree): an
  # install step would write it after the step that Ninja knows produces it, and a test
  # could then link the previous archive.
  set(_fftn_build ${CMAKE_BINARY_DIR}/_deps/fftw-${_fftn_prec}-build)
  set(_fftn_archive
    ${_fftn_build}/${CMAKE_STATIC_LIBRARY_PREFIX}${_fftn_lib}${CMAKE_STATIC_LIBRARY_SUFFIX})
  ExternalProject_Add(fftn_fftw_${_fftn_prec}
    SOURCE_DIR ${fftn_fftw_SOURCE_DIR}
    BINARY_DIR ${_fftn_build}
    CMAKE_ARGS ${_fftn_fftw_args} -DENABLE_FLOAT=${_fftn_float}
    BUILD_BYPRODUCTS ${_fftn_archive}
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL ON)
  add_library(fftn_test_${_fftn_lib} STATIC IMPORTED GLOBAL)
  set_target_properties(fftn_test_${_fftn_lib} PROPERTIES IMPORTED_LOCATION ${_fftn_archive})
  add_dependencies(fftn_test_${_fftn_lib} fftn_fftw_${_fftn_prec})
  target_link_libraries(fftn_test_fftw INTERFACE fftn_test_${_fftn_lib})
endforeach()
target_link_libraries(fftn_test_fftw INTERFACE ${_fftn_fftw_libm})
