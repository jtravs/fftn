// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// The main FFTN header: includes the whole core API.
#pragma once

#include "fftn/config.hpp"      // IWYU pragma: export
#include "fftn/executor.hpp"    // IWYU pragma: export
#include "fftn/isa.hpp"         // IWYU pragma: export
#include "fftn/lengths.hpp"     // IWYU pragma: export
#include "fftn/options.hpp"     // IWYU pragma: export
#include "fftn/plan.hpp"        // IWYU pragma: export
#include "fftn/shape.hpp"       // IWYU pragma: export
#include "fftn/team.hpp"        // IWYU pragma: export
#include "fftn/thread_pool.hpp" // IWYU pragma: export
#include "fftn/wisdom.hpp"      // IWYU pragma: export
#include "fftn/workspace.hpp"   // IWYU pragma: export

/// @namespace fftn
/// FFTN: fast Fourier transforms for CPUs.
///
/// Every public name is declared in `fftn`. Each lives in a versioned inline namespace
/// (`fftn::v0`), so two copies of FFTN of different versions linked into one process (two
/// shared libraries that each embed it, for example) cannot collide. Callers never name
/// the inner namespace.

/// @namespace fftn::v0
/// Version namespace of this release (see the description of `fftn`).

namespace fftn::inline v0 {

/// The library's version as "major.minor.patch".
///
/// Guarantees: the value is the version of the compiled library that is linked, which
/// can differ from the FFTN_VERSION_* macros of the headers if the build mixes versions.
/// The pointer refers to a static string and is never null.
[[nodiscard]] const char* version() noexcept;

} // namespace fftn::inline v0
