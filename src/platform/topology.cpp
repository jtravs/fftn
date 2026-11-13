// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/topology.hpp"

#include "platform/environment.hpp"
#include "platform/file_source.hpp"
#include "platform/sysctl_values.hpp"
#include "platform/topology_sources.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fftn::inline v0::detail {

namespace {

// Keeps the members of `set` that are recorded CPUs, and adds `self` if it is missing.
std::vector<unsigned> normalise_set(const std::vector<unsigned>& set,
                                    const std::vector<unsigned>& known, unsigned self) {
  std::vector<unsigned> out;
  std::ranges::set_intersection(set, known, std::back_inserter(out));
  if (!std::ranges::binary_search(out, self)) {
    out.insert(std::ranges::upper_bound(out, self), self);
  }
  return out;
}

// Distinct sets, ordered by first member, and each CPU's index among them.
class group_builder {
public:
  void add(unsigned cpu, std::vector<unsigned> set) {
    members_.emplace_back(cpu, set);
    index_.emplace(std::move(set), 0);
  }

  std::vector<cpu_group> finish(std::map<unsigned, unsigned>& index_of_cpu) {
    std::vector<cpu_group> groups;
    for (auto& [set, index] : index_) {
      index = static_cast<unsigned>(groups.size());
      groups.push_back({.id = index, .cpus = set});
    }
    for (const auto& [cpu, set] : members_) {
      index_of_cpu[cpu] = index_.at(set);
    }
    return groups;
  }

private:
  std::vector<std::pair<unsigned, std::vector<unsigned>>> members_;
  std::map<std::vector<unsigned>, unsigned> index_;
};

// Groups by an OS number (package, NUMA node); returns the groups and each CPU's index.
std::vector<cpu_group> group_by_number(const std::vector<std::pair<unsigned, unsigned>>& cpu_number,
                                       std::map<unsigned, unsigned>& index_of_cpu) {
  std::map<unsigned, std::vector<unsigned>> by_number;
  for (const auto& [cpu, number] : cpu_number) {
    by_number[number].push_back(cpu);
  }
  std::vector<cpu_group> groups;
  std::map<unsigned, unsigned> index_of_number;
  for (auto& [number, cpus] : by_number) {
    index_of_number[number] = static_cast<unsigned>(groups.size());
    groups.push_back({.id = number, .cpus = std::move(cpus)});
  }
  for (const auto& [cpu, number] : cpu_number) {
    index_of_cpu[cpu] = index_of_number[number];
  }
  return groups;
}

} // namespace

topology assemble_topology(topology_parts parts) {
  std::ranges::sort(parts.cpus, {}, &cpu_record::id);
  const auto same_id = std::ranges::unique(parts.cpus, {}, &cpu_record::id);
  parts.cpus.erase(same_id.begin(), same_id.end());
  std::vector<unsigned> known;
  known.reserve(parts.cpus.size());
  for (const cpu_record& cpu : parts.cpus) {
    known.push_back(cpu.id);
  }

  topology t;
  t.source = std::move(parts.source);
  t.os_cpu_ids = parts.os_cpu_ids;
  t.allowed_cpus = std::move(parts.allowed_cpus);
  t.cpu_quota = parts.cpu_quota;
  t.page_size = parts.page_size;
  t.classes = std::move(parts.classes);

  group_builder cores;
  group_builder l2;
  group_builder l3;
  std::vector<std::pair<unsigned, unsigned>> package_of;
  std::vector<std::pair<unsigned, unsigned>> node_of;
  for (const cpu_record& cpu : parts.cpus) {
    cores.add(cpu.id, normalise_set(cpu.smt_siblings, known, cpu.id));
    if (!cpu.l2_sharing.empty()) {
      l2.add(cpu.id, normalise_set(cpu.l2_sharing, known, cpu.id));
    }
    if (!cpu.l3_sharing.empty()) {
      l3.add(cpu.id, normalise_set(cpu.l3_sharing, known, cpu.id));
    }
    package_of.emplace_back(cpu.id, cpu.package);
    node_of.emplace_back(cpu.id, cpu.node);
  }
  std::map<unsigned, unsigned> core_index;
  std::map<unsigned, unsigned> l2_index;
  std::map<unsigned, unsigned> l3_index;
  std::map<unsigned, unsigned> package_index;
  std::map<unsigned, unsigned> node_index;
  t.cores = cores.finish(core_index);
  t.l2_groups = l2.finish(l2_index);
  t.l3_groups = l3.finish(l3_index);
  t.packages = group_by_number(package_of, package_index);
  t.numa_nodes = group_by_number(node_of, node_index);

  for (const cpu_record& cpu : parts.cpus) {
    logical_cpu out;
    out.id = cpu.id;
    out.core = core_index.at(cpu.id);
    out.package = package_index.at(cpu.id);
    out.node = node_index.at(cpu.id);
    out.core_class = cpu.core_class;
    if (const auto it = l2_index.find(cpu.id); it != l2_index.end()) {
      out.l2_group = it->second;
    }
    if (const auto it = l3_index.find(cpu.id); it != l3_index.end()) {
      out.l3_group = it->second;
    }
    t.cpus.push_back(out);
  }

  // Counts per class. A core belongs to the class of its first CPU.
  std::size_t class_count = t.classes.size();
  for (const logical_cpu& cpu : t.cpus) {
    class_count = std::max<std::size_t>(class_count, std::size_t{cpu.core_class} + 1);
  }
  t.classes.resize(class_count);
  for (core_class& c : t.classes) {
    c.cores = 0;
    c.logical_cpus = 0;
  }
  for (const logical_cpu& cpu : t.cpus) {
    core_class& c = t.classes[cpu.core_class];
    ++c.logical_cpus;
    if (t.cores[cpu.core].cpus.front() == cpu.id) {
      ++c.cores;
    }
  }
  return t;
}

topology fallback_topology(std::size_t logical_cpus) {
  topology_parts parts;
  parts.source = "fallback";
  for (std::size_t i = 0; i < std::max<std::size_t>(logical_cpus, 1); ++i) {
    cpu_record cpu;
    cpu.id = static_cast<unsigned>(i);
    parts.cpus.push_back(cpu);
  }
  parts.page_size = system_page_size();
  return assemble_topology(std::move(parts));
}

std::size_t system_page_size() noexcept {
#ifdef _WIN32
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  return info.dwPageSize;
#else
  const long size = sysconf(_SC_PAGESIZE);
  return size > 0 ? static_cast<std::size_t>(size) : 4096;
#endif
}

topology detect_topology() {
#ifdef __linux__
  topology t = topology_from_sysfs(directory_files("/"), system_page_size());
#elif defined(__APPLE__)
  topology t = topology_from_sysctl(read_topology_sysctl());
#elif defined(_WIN32)
  topology t = topology_from_windows();
#else
  topology t;
#endif
  if (t.cpus.empty()) {
    // Without /sys (some sandboxes) or on an unknown OS, nothing but the CPU count is known.
    return fallback_topology(std::thread::hardware_concurrency());
  }
  return t;
}

topology load_topology(const std::filesystem::path& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) {
    return topology_from_sysfs(directory_files(path), system_page_size());
  }
  const std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("fftn: cannot read the topology file " + path.string());
  }
  std::ostringstream text;
  text << file.rdbuf();
  const std::string content = std::move(text).str();
  if (is_sysfs_snapshot(content)) {
    return topology_from_sysfs(snapshot_files(content), system_page_size());
  }
  return topology_from_sysctl(parse_sysctl_text(content));
}

const topology& system_topology() {
  // The environment is read once, at first use, before any plan exists.
  static const topology t = [] {
    const auto override_path = environment_variable("FFTN_TOPOLOGY");
    if (override_path.has_value() && !override_path->empty()) {
      return load_topology(*override_path);
    }
    return detect_topology();
  }();
  return t;
}

} // namespace fftn::inline v0::detail
