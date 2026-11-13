// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Sets of logical CPU numbers in the Linux "cpulist" text form ("0-3,8,10-11"), used by
// sysfs, /proc and cgroups, and by the topology dump.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::inline v0::detail {

// Parses a cpulist into ascending, distinct CPU numbers. An empty or blank list is the empty
// set. Returns nullopt for anything malformed (the caller then treats the item as unknown).
[[nodiscard]] std::optional<std::vector<unsigned>> parse_cpu_list(std::string_view text);

// Formats ascending, distinct CPU numbers as a cpulist; the empty set is "".
[[nodiscard]] std::string format_cpu_list(const std::vector<unsigned>& cpus);

} // namespace fftn::inline v0::detail
