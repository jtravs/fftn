// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "isa/test_targets.hpp"

#include "isa/target.hpp"
#include "platform/cpu_features.hpp"
#include "support/skipped_targets.hpp"

#include <string>
#include <vector>

namespace isa_test {

using fftn::detail::compiled_in;
using fftn::detail::runs_on;

cpu_features machine_features() { return fftn::detail::detect_cpu_features(); }

std::vector<target> runnable_targets() {
  const cpu_features f = machine_features();
  std::vector<target> out;
  for (const target t : all_targets) {
    if (compiled_in(t) && runs_on(t, f) && !fftn::test::target_skipped(label(t))) {
      out.push_back(t);
    }
  }
  return out;
}

std::string label(target t) { return std::string(fftn::detail::target_name(t)); }

} // namespace isa_test
