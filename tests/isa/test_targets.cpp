// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "isa/test_targets.hpp"

#include "isa/arch.hpp"
#include "isa/target.hpp"

#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace isa_test {
namespace {

using fftn::detail::compiled_in;
using fftn::detail::runs_on;
using fftn::detail::target_from_name;

bool assumed(target t) {
  const char* env =
      std::getenv("FFTN_TEST_ASSUME"); // NOLINT(concurrency-mt-unsafe): read once, before threads
  if (env == nullptr) {
    return false;
  }
  std::string_view rest(env);
  while (!rest.empty()) {
    const std::size_t comma = rest.find(',');
    if (target_from_name(rest.substr(0, comma)) == t) {
      return true;
    }
    rest = comma == std::string_view::npos ? std::string_view() : rest.substr(comma + 1);
  }
  return false;
}

} // namespace

cpu_features machine_features() {
  cpu_features f;
#if FFTN_ARCH_AARCH64
  f.neon = true;
#ifdef __APPLE__
  f.fcma = true;
#endif
#elif FFTN_ARCH_X86_64
  f.sse2 = true;
#endif
  return f;
}

std::vector<target> runnable_targets() {
  const cpu_features f = machine_features();
  std::vector<target> out;
  for (const target t : all_targets) {
    if (compiled_in(t) && (runs_on(t, f) || assumed(t))) {
      out.push_back(t);
    }
  }
  return out;
}

std::string label(target t) { return std::string(fftn::detail::target_name(t)); }

} // namespace isa_test
