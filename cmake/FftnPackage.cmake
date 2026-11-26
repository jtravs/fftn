# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# FFTN as a package (DESIGN.md 14.3, docs/building.md): the link interface that every way of
# using the library shares, the header-only adapters as targets, and, with FFTN_INSTALL, the
# install rules: headers, libfftn, the exported fftn::fftn with fftnConfig.cmake and its version
# file, and fftn.pc for pkg-config. Test and benchmark dependencies are never installed.

# The thread pool and the team engine use std::thread, which needs the platform's thread
# library on some systems (glibc before 2.34). A static library's link dependencies are its
# users', so this one is exported, found again by fftnConfig.cmake and named in fftn.pc.
find_package(Threads REQUIRED)
target_link_libraries(fftn PRIVATE Threads::Threads)

# The adapters (fftn/omp.hpp, fftn/kokkos.hpp) are compiled in their user's translation units
# against the user's runtime (DESIGN.md 14.2), so their targets only add that runtime to
# fftn::fftn. In a build tree (add_subdirectory, FetchContent) each exists when its runtime is
# found here: OpenMP from the compiler, Kokkos when the including project made Kokkos::kokkos
# before adding FFTN. An installed package makes them as components (cmake/fftnConfig.cmake.in).
find_package(OpenMP COMPONENTS CXX QUIET)
if(TARGET OpenMP::OpenMP_CXX)
  add_library(fftn_omp INTERFACE)
  target_link_libraries(fftn_omp INTERFACE fftn OpenMP::OpenMP_CXX)
  add_library(fftn::omp ALIAS fftn_omp)
endif()
if(TARGET Kokkos::kokkos)
  add_library(fftn_kokkos INTERFACE)
  target_link_libraries(fftn_kokkos INTERFACE fftn Kokkos::kokkos)
  add_library(fftn::kokkos ALIAS fftn_kokkos)
endif()

# Third-party projects that joined the build as subdirectories (Kokkos and GoogleTest for the
# tests) bring install rules of their own, which `cmake --install` would run. Every directory
# outside FFTN's own (src, tests, examples, bench) is excluded from the default build, which
# also skips its install rules; their targets that FFTN's tests link are still built.
function(_fftn_exclude_third_party dir)
  get_property(subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
  foreach(sub IN LISTS subdirs)
    set(ours FALSE)
    foreach(top IN ITEMS src tests examples bench)
      cmake_path(APPEND PROJECT_SOURCE_DIR ${top} OUTPUT_VARIABLE top_dir)
      cmake_path(IS_PREFIX top_dir "${sub}" NORMALIZE inside)
      if(inside)
        set(ours TRUE)
      endif()
    endforeach()
    if(ours)
      _fftn_exclude_third_party(${sub})
    else()
      set_property(DIRECTORY ${sub} PROPERTY EXCLUDE_FROM_ALL TRUE)
    endif()
  endforeach()
endfunction()
_fftn_exclude_third_party(${PROJECT_SOURCE_DIR})

if(NOT FFTN_INSTALL)
  return()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(_fftn_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/fftn)

install(TARGETS fftn EXPORT fftn_targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
# The C API's optional shared library (FFTN_BUILD_C_SHARED), exported as fftn::fftn_c.
if(TARGET fftn_c)
  install(TARGETS fftn_c EXPORT fftn_targets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/fftn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
install(EXPORT fftn_targets NAMESPACE fftn:: FILE fftnTargets.cmake
        DESTINATION ${_fftn_cmake_dir})
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/fftnConfig.cmake.in
  ${PROJECT_BINARY_DIR}/fftnConfig.cmake INSTALL_DESTINATION ${_fftn_cmake_dir})
# Until 1.0 a minor version may change the API (AGENTS.md section 1), so only the same
# major and minor version satisfies a request.
write_basic_package_version_file(${PROJECT_BINARY_DIR}/fftnConfigVersion.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES ${PROJECT_BINARY_DIR}/fftnConfig.cmake ${PROJECT_BINARY_DIR}/fftnConfigVersion.cmake
        DESTINATION ${_fftn_cmake_dir})

# fftn.pc finds the prefix from its own location, so the tree can be installed with
# `cmake --install --prefix` or moved.
set(_fftn_pc_dir ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  set(FFTN_PC_PREFIX "${CMAKE_INSTALL_PREFIX}")
else()
  file(RELATIVE_PATH _fftn_up /fftn-prefix/${_fftn_pc_dir} /fftn-prefix)
  string(REGEX REPLACE "/$" "" _fftn_up "${_fftn_up}")
  set(FFTN_PC_PREFIX "\${pcfiledir}/${_fftn_up}")
endif()
foreach(kind IN ITEMS INCLUDEDIR LIBDIR)
  if(IS_ABSOLUTE "${CMAKE_INSTALL_${kind}}")
    set(FFTN_PC_${kind} "${CMAKE_INSTALL_${kind}}")
  else()
    set(FFTN_PC_${kind} "\${prefix}/${CMAKE_INSTALL_${kind}}")
  endif()
endforeach()
set(FFTN_PC_THREADS "${CMAKE_THREAD_LIBS_INIT}")
configure_file(${PROJECT_SOURCE_DIR}/cmake/fftn.pc.in ${PROJECT_BINARY_DIR}/fftn.pc @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/fftn.pc DESTINATION ${_fftn_pc_dir})

install(FILES ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
