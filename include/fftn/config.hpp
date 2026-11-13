// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// The build contract every FFTN header enforces, and the library's version.
///
/// Every public header includes this one. It refuses translation units that are not C++20
/// or newer, or that are compiled with fast-math: FFTN's numerical guarantees (accuracy
/// bounds, bitwise determinism across thread counts) assume IEEE arithmetic in the code
/// that calls it as well as inside it.
#pragma once

#if __cplusplus < 202002L && (!defined(_MSVC_LANG) || _MSVC_LANG < 202002L)
#error "FFTN needs C++20 or newer."
#endif

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error                                                                                             \
    "FFTN must not be compiled with fast-math or finite-math-only: its results assume IEEE arithmetic."
#endif

// NOLINTBEGIN(modernize-macro-to-enum): the version must be usable in #if.
/// Major version: changes when the public API changes incompatibly.
#define FFTN_VERSION_MAJOR 0
/// Minor version: changes when features are added compatibly.
#define FFTN_VERSION_MINOR 0
/// Patch version: changes for fixes only.
#define FFTN_VERSION_PATCH 0
// NOLINTEND(modernize-macro-to-enum)
