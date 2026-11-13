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
