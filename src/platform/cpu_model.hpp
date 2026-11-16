// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The processor's model name, for wisdom records (DESIGN.md 11.3): an entry measured on one
// model is used only on that model. One line of text, never empty ("unknown" when the
// platform does not say).
#pragma once

#include <string>
#include <string_view>

namespace fftn::inline v0::detail {

// The model of the running processor: Linux /proc/cpuinfo's "model name" (x86) or its
// implementer, variant and part (AArch64, which has no model name), macOS's
// machdep.cpu.brand_string, Windows' ProcessorNameString.
[[nodiscard]] std::string cpu_model();

// The model named by the text of a Linux /proc/cpuinfo (the first processor's).
[[nodiscard]] std::string cpu_model_from_cpuinfo(std::string_view text);

// The text as one line: tabs, line breaks and runs of spaces become one space, the ends are
// trimmed, and empty text becomes "unknown".
[[nodiscard]] std::string one_line(std::string_view text);

} // namespace fftn::inline v0::detail
