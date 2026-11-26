/* SPDX-FileCopyrightText: 2026 John C. Travers and contributors
 * SPDX-License-Identifier: Apache-2.0 */

/**
 * @file
 * The version of the FFTN headers, as macros usable in `#if` from C and C++: included by
 * fftn/config.hpp (so by every C++ header) and by fftn/c_api.h. fftn::version() and
 * fftn_version() give the version of the compiled library that is linked.
 */
#ifndef FFTN_VERSION_H
#define FFTN_VERSION_H

/* NOLINTBEGIN(modernize-macro-to-enum): the version must be usable in #if. */
/** Major version: changes when the public API changes incompatibly. */
#define FFTN_VERSION_MAJOR 0
/** Minor version: changes when features are added compatibly. */
#define FFTN_VERSION_MINOR 0
/** Patch version: changes for fixes only. */
#define FFTN_VERSION_PATCH 0
/* NOLINTEND(modernize-macro-to-enum) */

#endif /* FFTN_VERSION_H */
