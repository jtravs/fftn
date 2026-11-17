// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Targets the suites leave out although the machine runs them: the names in
// FFTN_TEST_SKIP_ISAS, separated by commas ("x86_v2,avx"). For emulated runs only
// (.github/scripts/emulate.sh): under Intel SDE the legacy-SSE code of x86_v2 ran so slowly
// on the Haswell and Skylake-SP models that the core tests passed the job's time limit, and
// the Nehalem and Sandy Bridge jobs already run those targets as their best ones.
#pragma once

#include "platform/environment.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace fftn::test {

inline bool target_skipped(std::string_view name) {
  const std::optional<std::string> list = fftn::detail::environment_variable("FFTN_TEST_SKIP_ISAS");
  if (!list) {
    return false;
  }
  std::string_view rest = *list;
  while (!rest.empty()) {
    const std::size_t comma = rest.find(',');
    if (rest.substr(0, comma) == name) {
      return true;
    }
    if (comma == std::string_view::npos) {
      break;
    }
    rest.remove_prefix(comma + 1);
  }
  return false;
}

} // namespace fftn::test
