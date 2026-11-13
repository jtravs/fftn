// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// fftn_platform_dump: prints what FFTN detects on this machine (CPU features, the planning
// clock's resolution, the topology) as stable text, to record per machine.
//
//   fftn_platform_dump            the system topology (honours FFTN_TOPOLOGY)
//   fftn_platform_dump <path>     a recorded topology (a sysfs root or a sysctl file)

#include "platform/cpu_features.hpp"
#include "platform/platform_text.hpp"
#include "platform/timer.hpp"
#include "platform/topology.hpp"

#include <cstdio>
#include <exception>
#include <span>
#include <string>

int main(int argc, char** argv) {
  try {
    const std::span<char*> args(argv, static_cast<std::size_t>(argc));
    if (args.size() > 2) {
      std::fputs("usage: fftn_platform_dump [sysfs root or sysctl file]\n", stderr);
      return 2;
    }
    std::string text = fftn::test::features_text(fftn::detail::detect_cpu_features());
    text += "clock_resolution_ns " +
            std::to_string(fftn::detail::measured_clock_resolution().count()) + '\n';
    text += fftn::test::topology_text(args.size() == 2 ? fftn::detail::load_topology(args[1])
                                                       : fftn::detail::system_topology());
    std::fputs(text.c_str(), stdout);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fftn_platform_dump: %s\n", e.what());
    return 1;
  }
}
