// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Reading environment variables portably. The Microsoft C runtime marks std::getenv
// deprecated (it returns a pointer into an environment that another thread may change);
// its replacement, _dupenv_s, copies the value instead. This is the one place that
// chooses between them, so that neither the warning nor its suppression spreads.
#pragma once

#include <optional>
#include <string>

namespace fftn::inline v0::detail {

// The value of the environment variable `name`, or nothing when it is not set. FFTN never
// sets environment variables; it reads its own once, before any plan exists.
[[nodiscard]] std::optional<std::string> environment_variable(const char* name);

} // namespace fftn::inline v0::detail
