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
#   OpenMP::OpenMP_CXX  the compiler's OpenMP runtime, when found, for the tests and the
#                       example of the optional header fftn/omp.hpp (FFTN_TEST_WITH_OPENMP,
#                       default ON except under ThreadSanitizer). Never linked by the library.
#   Kokkos::kokkos      Kokkos core 5.2.1 with the Serial back end and one threaded back end
#                       (FFTN_KOKKOS_BACKEND), for the tests and the example of the optional
#                       header fftn/kokkos.hpp
#                       (FFTN_TEST_WITH_KOKKOS, default ON; it builds in seconds). The
#                       library never links it: the header is compiled in its user's
#                       translation units. Consumers test `if(TARGET Kokkos::kokkos)`.
#
# Licence: FFTW is GPL-2.0-or-later. Only test executables link it; they are built for our
# own use and never distributed (AGENTS.md section 10). Boost is under the Boost Software
# License; Kokkos under Apache-2.0 WITH LLVM-exception.
#
# Offline builds: FETCHCONTENT_SOURCE_DIR_BOOST_CONFIG, FETCHCONTENT_SOURCE_DIR_BOOST_MP,
# FETCHCONTENT_SOURCE_DIR_FFTN_KOKKOS and FETCHCONTENT_SOURCE_DIR_FFTN_FFTW point at unpacked
# copies of the archives below.

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

# Kokkos is a CMake subproject (its targets are linked, so its build joins ours) and SYSTEM, so
# its headers never meet FFTN's warning flags. The tarball and its SHA-256 are the official
# release assets of github.com/kokkos/kokkos (kokkos-5.2.1.tar.gz and
# kokkos-5.2.1-SHA-256.txt); the hash was also computed from the downloaded file.
# Off by default in GCC ThreadSanitizer builds: GCC warns that TSan does not support
# atomic_thread_fence (-Wtsan) for Kokkos' desul atomics after they are inlined into our test
# units, where SYSTEM cannot silence it, and the gate builds with -Werror. Clang's TSan builds
# run these tests, on the Threads back end.
# Off by default on Windows on Arm too: Kokkos 5.2.1 adds -mcx16 for every Clang build on
# Windows (cmake/kokkos_arch.cmake), an x86 option that clang-cl refuses for Arm64 targets.
set(_fftn_kokkos_default ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND FFTN_SANITIZE MATCHES "thread")
  set(_fftn_kokkos_default OFF)
endif()
if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
  set(_fftn_kokkos_default OFF)
endif()
option(FFTN_TEST_WITH_KOKKOS "Test fftn/kokkos.hpp against Kokkos 5.2.1"
       ${_fftn_kokkos_default})
# Kokkos builds with one threaded host back end at a time, so each configuration tests Serial
# and the one named here: Threads by default (no OpenMP runtime enters the test binaries),
# OpenMP in the CI job that sets the environment variable, Serial alone on Windows, where the
# Threads back end has not been tried.
set(_fftn_kokkos_backend Threads)
if(WIN32)
  set(_fftn_kokkos_backend Serial)
endif()
if(NOT "$ENV{FFTN_KOKKOS_BACKEND}" STREQUAL "")
  set(_fftn_kokkos_backend "$ENV{FFTN_KOKKOS_BACKEND}")
endif()
set(FFTN_KOKKOS_BACKEND ${_fftn_kokkos_backend} CACHE STRING
    "Kokkos' threaded back end for the tests of fftn/kokkos.hpp: Threads, OpenMP or Serial (none)")
set_property(CACHE FFTN_KOKKOS_BACKEND PROPERTY STRINGS Threads OpenMP Serial)
if(FFTN_TEST_WITH_KOKKOS)
  if(NOT FFTN_KOKKOS_BACKEND MATCHES "^(Threads|OpenMP|Serial)$")
    message(FATAL_ERROR "FFTN_KOKKOS_BACKEND must be Threads, OpenMP or Serial")
  endif()
  FetchContent_Declare(fftn_kokkos
    URL https://github.com/kokkos/kokkos/releases/download/5.2.1/kokkos-5.2.1.tar.gz
    URL_HASH SHA256=3f754c99aa6130b1dd6520d904db7b2fd44ed618cd91e0dfd921956f23f6812d
    SYSTEM)
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  set(_fftn_kokkos_openmp OFF)
  set(_fftn_kokkos_threads OFF)
  if(FFTN_KOKKOS_BACKEND STREQUAL "OpenMP")
    set(_fftn_kokkos_openmp ON)
  elseif(FFTN_KOKKOS_BACKEND STREQUAL "Threads")
    set(_fftn_kokkos_threads ON)
  endif()
  set(Kokkos_ENABLE_OPENMP ${_fftn_kokkos_openmp} CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_THREADS ${_fftn_kokkos_threads} CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(fftn_kokkos)
endif()

# fftn_link_kokkos(<target>): links Kokkos into a test or example of fftn/kokkos.hpp. The
# clang static analyzer reports a leak in every allocating View constructor: Kokkos keeps the
# allocation record's address in a tagged pointer (impl/Kokkos_SharedAlloc.hpp) that the
# analyzer cannot follow, and the report sits in Kokkos' header where no NOLINT can reach. Only
# these targets turn that one check off.
function(fftn_link_kokkos target)
  target_link_libraries(${target} PRIVATE Kokkos::kokkos)
  get_target_property(tidy ${target} CXX_CLANG_TIDY)
  if(tidy)
    list(APPEND tidy "--checks=-clang-analyzer-cplusplus.NewDeleteLeaks")
    set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${tidy}")
  endif()
endfunction()

# OpenMP, for the tests and the example of the optional header fftn/omp.hpp: the compiler's own
# runtime (libgomp with GCC, libomp with Clang), found by CMake's FindOpenMP. Only the
# executables of those tests and that example link it, one runtime each; the library never
# does (the header is compiled in its user's translation units). On by default wherever a
# runtime is found, except under ThreadSanitizer: the runtimes are not instrumented (libomp
# only with its Archer tool, which the toolchains here do not ship), so TSan reports races
# between work in a region and the code after it, which the region's join orders (checked
# with Homebrew LLVM 23's libomp). The pool and custom executors carry the TSan coverage of the
# executor path. Consumers test `if(FFTN_TEST_WITH_OPENMP AND TARGET OpenMP::OpenMP_CXX)`.
set(_fftn_openmp_default ON)
if(FFTN_SANITIZE MATCHES "thread")
  set(_fftn_openmp_default OFF)
endif()
option(FFTN_TEST_WITH_OPENMP "Test fftn/omp.hpp against the compiler's OpenMP runtime"
       ${_fftn_openmp_default})
if(FFTN_TEST_WITH_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(NOT OpenMP_CXX_FOUND)
    message(STATUS "fftn: no OpenMP runtime found; the tests of fftn/omp.hpp are not built")
  endif()
endif()

# fftn_link_openmp(<target>): links the OpenMP runtime into a test or example of fftn/omp.hpp.
# The compiler finds omp.h in its own resource directory, which clang-tidy from PyPI does not
# have: the header alone is copied into a directory of its own and handed to clang-tidy, so
# that none of the compiler's other built-in headers shadow clang-tidy's.
if(FFTN_TEST_WITH_OPENMP AND OpenMP_CXX_FOUND AND FFTN_CLANG_TIDY)
  set(_fftn_omp_dirs ${OpenMP_CXX_INCLUDE_DIRS})
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
                    OUTPUT_VARIABLE _fftn_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND _fftn_omp_dirs ${_fftn_resource_dir}/include)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=include
                    OUTPUT_VARIABLE _fftn_gcc_include OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND _fftn_omp_dirs ${_fftn_gcc_include})
  endif()
  find_file(_fftn_omp_h omp.h PATHS ${_fftn_omp_dirs} NO_DEFAULT_PATH)
  if(_fftn_omp_h)
    # Cached, so that the examples' directory sees it too.
    set(FFTN_TIDY_OMP_INCLUDE ${CMAKE_BINARY_DIR}/fftn-tidy-omp CACHE INTERNAL "")
    configure_file(${_fftn_omp_h} ${FFTN_TIDY_OMP_INCLUDE}/omp.h COPYONLY)
  endif()
endif()
function(fftn_link_openmp target)
  target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  get_target_property(tidy ${target} CXX_CLANG_TIDY)
  if(tidy AND FFTN_TIDY_OMP_INCLUDE)
    list(APPEND tidy "--extra-arg=-isystem${FFTN_TIDY_OMP_INCLUDE}")
    set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${tidy}")
  endif()
endfunction()

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
  string(REGEX REPLACE "icpx" "icx" _fftn_cc_name "${_fftn_cc_name}")
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
# icx's default floating-point model is a fast one (CMakeLists.txt, experiment X22).
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(_fftn_fftw_cflags "-fno-fast-math")
endif()
set(_fftn_fftw_sanitizers ${FFTN_SANITIZE})
string(REPLACE "," ";" _fftn_fftw_sanitizers "${_fftn_fftw_sanitizers}")
list(REMOVE_ITEM _fftn_fftw_sanitizers undefined)
if(_fftn_fftw_sanitizers)
  list(JOIN _fftn_fftw_sanitizers "," _fftn_fftw_sanitizers)
  string(APPEND _fftn_fftw_cflags " -fsanitize=${_fftn_fftw_sanitizers} -fno-omit-frame-pointer")
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
