# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Platform layer (task 0.2): CPU features, cache geometry and topology, aligned memory,
# timers. Every reader is compiled on every OS; only the calls that fetch live data are
# guarded by the OS they need, so the parsers are tested everywhere against fixtures.

target_sources(fftn PRIVATE
  cpu_features.cpp
  cpu_list.cpp
  file_source.cpp
  sysctl_values.cpp
  timer.cpp
  topology.cpp
  topology_linux.cpp
  topology_macos.cpp
  topology_windows.cpp)
