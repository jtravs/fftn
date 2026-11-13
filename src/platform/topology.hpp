// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The machine as the planner and the threading engine see it: logical CPUs, cores and their
// SMT siblings, the groups that share an L2 or an L3, NUMA nodes, packages, core classes
// with their cache geometry, the allowed-CPU mask, any cgroup CPU quota and the page size
// (DESIGN.md 10 and 13.5). Plain data: everything that uses it takes it as an argument, so
// tests and callers substitute their own (the programmatic override). Internal for now; the
// public shape is decided at the API review (task 0.9).
#pragma once

#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fftn::inline v0::detail {

inline constexpr unsigned no_index = std::numeric_limits<unsigned>::max();

// One cache level as one core class sees it. Zero means unknown (or, for size, absent):
// macOS does not expose associativity, so it is never invented.
struct cache_geometry {
  std::size_t size = 0;      // bytes
  std::size_t line_size = 0; // bytes
  unsigned ways = 0;         // associativity
  unsigned shared_by = 0;    // logical CPUs sharing one instance

  bool operator==(const cache_geometry&) const = default;
};

// CPUs of one kind. Classes are ordered fastest first.
struct core_class {
  std::string name;      // the OS's name (macOS perflevel name, "core"/"atom"), or ""
  unsigned capacity = 0; // Linux cpu_capacity (1024 = the fastest core); 0: not reported
  std::size_t cores = 0;
  std::size_t logical_cpus = 0;
  cache_geometry l1d;
  cache_geometry l2;
  cache_geometry l3;

  bool operator==(const core_class&) const = default;
};

// A set of logical CPUs: a core, an L2 or L3 group, a package or a NUMA node.
struct cpu_group {
  unsigned id = 0;            // the OS's number for packages and NUMA nodes; else the position
  std::vector<unsigned> cpus; // ascending logical CPU numbers

  bool operator==(const cpu_group&) const = default;
};

struct logical_cpu {
  unsigned id = 0;         // the logical CPU number (see topology::os_cpu_ids)
  unsigned core = 0;       // index into topology::cores
  unsigned package = 0;    // index into topology::packages
  unsigned node = 0;       // index into topology::numa_nodes
  unsigned core_class = 0; // index into topology::classes
  unsigned l2_group = no_index;
  unsigned l3_group = no_index;

  bool operator==(const logical_cpu&) const = default;
};

struct topology {
  std::string source; // "linux-sysfs", "macos-sysctl", "windows" or "fallback"
  // Whether logical CPU numbers are the operating system's (Linux, Windows). macOS does not
  // say which CPU is which, so there they are numbered in class order, fastest first.
  bool os_cpu_ids = true;
  std::vector<logical_cpu> cpus; // the online CPUs, ascending
  std::vector<cpu_group> cores;  // SMT siblings, ordered by first CPU
  std::vector<cpu_group> l2_groups;
  std::vector<cpu_group> l3_groups;
  std::vector<cpu_group> packages;
  std::vector<cpu_group> numa_nodes;
  std::vector<core_class> classes;
  // The process's allowed CPUs as the OS reports them (Linux: /proc/self/status, which an
  // OpenMP runtime may already have narrowed; hence recorded beside, not instead of, the
  // online CPUs). Empty: unknown or unrestricted.
  std::vector<unsigned> allowed_cpus;
  double cpu_quota = 0; // cgroup CPU bandwidth limit in CPUs (quota / period); 0: none
  std::size_t page_size = 0;

  bool operator==(const topology&) const = default;
};

// Detects the running machine's topology afresh. Never throws for missing or malformed OS
// data: what cannot be read is reported as unknown, and with no CPU information at all the
// result is a "fallback" topology of std::thread::hardware_concurrency() CPUs.
[[nodiscard]] topology detect_topology();

// Reads a recorded topology: a directory is a Linux sysfs root (holding sys/ and proc/), a
// file is `sysctl` text from macOS. Throws std::runtime_error if the path does not exist.
[[nodiscard]] topology load_topology(const std::filesystem::path& path);

// The process's topology: load_topology($FFTN_TOPOLOGY) when that variable is set and not
// empty (for tests), else detect_topology(). Computed on first use and immutable afterwards
// (a function-local constant; FFTN keeps no mutable global state).
[[nodiscard]] const topology& system_topology();

// The size of a memory page of the running process.
[[nodiscard]] std::size_t system_page_size() noexcept;

} // namespace fftn::inline v0::detail
