// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The per-OS readers behind detect_topology(). The Linux reader reads through a file_source
// (the live root, or a sysfs snapshot) and the macOS reader a sysctl snapshot, so tests run
// both, on any machine, against recorded and synthetic fixtures (tests/platform/data/). Each reader
// collects per-CPU records; assemble_topology() derives the groups and indices the same way for all
// of them.
#pragma once

#include "platform/cpu_features.hpp"
#include "platform/file_source.hpp"
#include "platform/sysctl_values.hpp"
#include "platform/topology.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::inline v0::detail {

struct cpu_record {
  unsigned id = 0;
  std::vector<unsigned> smt_siblings; // including id; empty: id alone
  unsigned package = 0;               // the OS's package number
  unsigned node = 0;                  // the OS's NUMA node number
  unsigned core_class = 0;            // index into topology_parts::classes
  std::vector<unsigned> l2_sharing;   // CPUs sharing its L2; empty: no L2 known
  std::vector<unsigned> l3_sharing;
};

struct topology_parts {
  std::string source;
  bool os_cpu_ids = true;
  std::vector<cpu_record> cpus;
  // Names, capacities and cache geometry; the counts are filled in by assemble_topology.
  std::vector<core_class> classes;
  std::vector<unsigned> allowed_cpus;
  double cpu_quota = 0;
  std::size_t page_size = 0;
  std::optional<cpu_vendor> vendor;
};

// The vendor a recording states (topology::vendor): `name` is cpu_vendor_name's name of it,
// surrounding white space ignored; nullopt for no name. Throws std::runtime_error naming
// `where` for any other text.
[[nodiscard]] std::optional<cpu_vendor> recorded_vendor(const std::optional<std::string>& name,
                                                        std::string_view where);

// Builds the topology from per-CPU records. Sibling and sharing lists are restricted to the
// recorded CPUs (offline CPUs drop out) and always contain the CPU itself.
[[nodiscard]] topology assemble_topology(topology_parts parts);

// Linux: reads sys/devices/system/{cpu,node}, sys/devices/cpu_{core,atom},
// proc/self/{status,cgroup} and sys/fs/cgroup, relative to the source's root, and a
// recording's fftn/cpu_vendor (recorded_vendor; a live root has none). The page size is
// passed in because sysfs does not hold it.
[[nodiscard]] topology topology_from_sysfs(const file_source& files, std::size_t page_size);

// macOS: reads hw.nperflevels, hw.perflevelN.*, hw.cachelinesize, hw.pagesize and, on Macs
// without performance levels (Intel), hw.{physical,logical}cpu, hw.l{1d,2,3}cachesize,
// hw.cacheconfig and hw.packages, and a recording's fftn.cpu_vendor (recorded_vendor; the
// kernel has no such key).
[[nodiscard]] topology topology_from_sysctl(const sysctl_values& values);

// Reads the keys topology_from_sysctl needs from the running macOS kernel; empty elsewhere.
[[nodiscard]] sysctl_values read_topology_sysctl();

// Windows: GetLogicalProcessorInformationEx. Returns an empty topology elsewhere.
[[nodiscard]] topology topology_from_windows();

// N logical CPUs, each its own core, nothing else known.
[[nodiscard]] topology fallback_topology(std::size_t logical_cpus);

} // namespace fftn::inline v0::detail
