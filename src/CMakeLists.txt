# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0

add_library(fftn STATIC version.cpp)
add_library(fftn::fftn ALIAS fftn)
target_include_directories(fftn
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(fftn PUBLIC cxx_std_20)
fftn_configure_target(fftn)
# Symbols are hidden: two copies of FFTN in one process must not interpose (DESIGN.md 14.2).
set_target_properties(fftn PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

add_subdirectory(platform)
add_subdirectory(isa)
add_subdirectory(core)
add_subdirectory(strategies)
add_subdirectory(planner)
add_subdirectory(threading)
add_subdirectory(easy)
add_subdirectory(c_api)

# The build identifier of wisdom records (cmake/FftnBuildId.cmake): a hash of the library's
# sources and of the configuration that changes its code, regenerated when any of them
# changes, since the version number does not change between development commits.
file(GLOB_RECURSE fftn_identity_sources CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.hpp ${PROJECT_SOURCE_DIR}/src/*.inl
  ${PROJECT_SOURCE_DIR}/include/fftn/*.hpp ${PROJECT_SOURCE_DIR}/include/fftn/*.h)
list(SORT fftn_identity_sources)
set(fftn_identity_list ${CMAKE_CURRENT_BINARY_DIR}/build_id_files.txt)
string(JOIN "\n" fftn_identity_lines
  "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} $<CONFIG> masked-tails ${FFTN_MASKED_TAILS} old-x86 ${FFTN_OLD_X86_TARGETS_ON}"
  ${fftn_identity_sources})
file(GENERATE OUTPUT ${fftn_identity_list} CONTENT "${fftn_identity_lines}\n")
set(fftn_build_id_cpp ${CMAKE_CURRENT_BINARY_DIR}/build_id.cpp)
add_custom_command(OUTPUT ${fftn_build_id_cpp}
  COMMAND ${CMAKE_COMMAND} -DOUTPUT=${fftn_build_id_cpp} -DLIST=${fftn_identity_list}
          -DROOT=${PROJECT_SOURCE_DIR} -P ${PROJECT_SOURCE_DIR}/cmake/FftnBuildId.cmake
  DEPENDS ${fftn_identity_sources} ${fftn_identity_list}
          ${PROJECT_SOURCE_DIR}/cmake/FftnBuildId.cmake
  COMMENT "Hashing FFTN's sources for the wisdom build identifier"
  VERBATIM)
target_sources(fftn PRIVATE ${fftn_build_id_cpp})
