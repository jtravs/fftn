// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Stable text forms of CPU features and topologies, for fftn_platform_dump and the tests.
// One "key value" pair per line, in a fixed order, so that two machines' dumps diff well.
#pragma once

#include "platform/cpu_features.hpp"
#include "platform/topology.hpp"

#include <string>

namespace fftn::test {

[[nodiscard]] std::string features_text(const fftn::detail::cpu_features& features);
[[nodiscard]] std::string topology_text(const fftn::detail::topology& t);

} // namespace fftn::test
