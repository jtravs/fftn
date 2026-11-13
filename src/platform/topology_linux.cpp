// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The Linux topology reader. It is ordinary file parsing, compiled on every OS so that its
// tests run everywhere against fixture trees. Sources: the kernel's sysfs ABI documents
// (Documentation/ABI/stable/sysfs-devices-system-cpu, .../testing/sysfs-devices-system-cpu,
// .../stable/sysfs-devices-node), proc(5) and the cgroup v1 and v2 documentation.
//
// The CPUs are the online ones, not the calling thread's affinity: an OpenMP runtime may
// already have bound the thread that plans (DESIGN.md 13.5). The process's allowed CPUs are
// recorded separately.

#include "platform/cpu_list.hpp"
#include "platform/parse_text.hpp"
#include "platform/topology.hpp"
#include "platform/topology_sources.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

namespace {

namespace fs = std::filesystem;

// The first line of a file, trimmed; nullopt if the file cannot be read.
std::optional<std::string> read_line(const fs::path& path) {
  std::ifstream file(path);
  std::string line;
  if (!file || !std::getline(file, line)) {
    return std::nullopt;
  }
  return std::string(trim(line));
}

std::optional<std::uint64_t> read_unsigned(const fs::path& path) {
  const auto line = read_line(path);
  return line ? parse_unsigned(*line) : std::nullopt;
}

std::optional<std::vector<unsigned>> read_cpu_list(const fs::path& path) {
  const auto line = read_line(path);
  return line ? parse_cpu_list(*line) : std::nullopt;
}

// Cache sizes are printed as "32K"; the suffixes M and G are accepted too.
std::optional<std::size_t> parse_cache_size(std::string_view text) {
  std::uint64_t scale = 1;
  if (!text.empty()) {
    switch (text.back()) {
    case 'K':
      scale = std::uint64_t{1} << 10U;
      break;
    case 'M':
      scale = std::uint64_t{1} << 20U;
      break;
    case 'G':
      scale = std::uint64_t{1} << 30U;
      break;
    default:
      break;
    }
  }
  if (scale != 1) {
    text.remove_suffix(1);
  }
  const auto value = parse_unsigned(text);
  if (!value || *value > std::numeric_limits<std::size_t>::max() / scale) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(*value * scale);
}

unsigned to_unsigned(std::uint64_t value) {
  return static_cast<unsigned>(
      std::min<std::uint64_t>(value, std::numeric_limits<unsigned>::max()));
}

// The number after `prefix` in a directory name ("cpu12", "node3"), if that is all it is.
std::optional<unsigned> numbered(const std::string& name, std::string_view prefix) {
  if (!name.starts_with(prefix)) {
    return std::nullopt;
  }
  const auto value = parse_unsigned(std::string_view(name).substr(prefix.size()));
  return value ? std::optional<unsigned>(to_unsigned(*value)) : std::nullopt;
}

// The numbers of the directories <dir>/<prefix><N>, ascending.
std::vector<unsigned> numbered_entries(const fs::path& dir, std::string_view prefix) {
  std::vector<unsigned> numbers;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    if (const auto n = numbered(it->path().filename().string(), prefix)) {
      numbers.push_back(*n);
    }
  }
  std::ranges::sort(numbers);
  return numbers;
}

struct cpu_caches {
  cache_geometry l1d;
  cache_geometry l2;
  cache_geometry l3;
  std::vector<unsigned> l2_sharing;
  std::vector<unsigned> l3_sharing;
};

std::size_t count_online(const std::vector<unsigned>& cpus, const std::vector<unsigned>& online) {
  return static_cast<std::size_t>(std::ranges::count_if(
      cpus, [&](unsigned c) { return std::ranges::binary_search(online, c); }));
}

cpu_caches read_caches(const fs::path& cpu_dir, const std::vector<unsigned>& online) {
  cpu_caches caches;
  for (const unsigned index : numbered_entries(cpu_dir / "cache", "index")) {
    const fs::path dir = cpu_dir / "cache" / ("index" + std::to_string(index));
    const auto level = read_unsigned(dir / "level");
    const auto type = read_line(dir / "type");
    if (!level || !type || *type == "Instruction") {
      continue;
    }
    cache_geometry g;
    const auto size_text = read_line(dir / "size");
    g.size = size_text ? parse_cache_size(*size_text).value_or(0) : 0;
    g.line_size = static_cast<std::size_t>(read_unsigned(dir / "coherency_line_size").value_or(0));
    g.ways = to_unsigned(read_unsigned(dir / "ways_of_associativity").value_or(0));
    const auto sharing = read_cpu_list(dir / "shared_cpu_list").value_or(std::vector<unsigned>{});
    g.shared_by = to_unsigned(count_online(sharing, online));
    if (*level == 1) {
      caches.l1d = g;
    } else if (*level == 2) {
      caches.l2 = g;
      caches.l2_sharing = sharing;
    } else if (*level == 3) {
      caches.l3 = g;
      caches.l3_sharing = sharing;
    }
  }
  return caches;
}

// Core classes, fastest first, and each online CPU's class. Arm systems report a relative
// capacity per CPU; hybrid Intel systems list their P- and E-cores as two PMUs.
std::vector<core_class> read_classes(const fs::path& root, const fs::path& cpu_root,
                                     const std::vector<unsigned>& online,
                                     std::map<unsigned, unsigned>& class_of) {
  std::map<unsigned, std::uint64_t> capacity;
  for (const unsigned cpu : online) {
    if (const auto c = read_unsigned(cpu_root / ("cpu" + std::to_string(cpu)) / "cpu_capacity")) {
      capacity[cpu] = *c;
    }
  }
  std::vector<core_class> classes;
  if (!capacity.empty()) {
    std::vector<std::uint64_t> distinct;
    distinct.reserve(online.size());
    for (const unsigned cpu : online) {
      distinct.push_back(capacity.contains(cpu) ? capacity[cpu] : 0);
    }
    std::ranges::sort(distinct, std::ranges::greater{});
    const auto repeated = std::ranges::unique(distinct);
    distinct.erase(repeated.begin(), repeated.end());
    for (const std::uint64_t c : distinct) {
      core_class k;
      k.capacity = to_unsigned(c);
      classes.push_back(k);
    }
    for (const unsigned cpu : online) {
      const std::uint64_t c = capacity.contains(cpu) ? capacity[cpu] : 0;
      class_of[cpu] = static_cast<unsigned>(std::ranges::find(distinct, c) - distinct.begin());
    }
    return classes;
  }
  const auto big = read_cpu_list(root / "sys/devices/cpu_core/cpus");
  const auto little = read_cpu_list(root / "sys/devices/cpu_atom/cpus");
  if (big && little) {
    classes.resize(2);
    classes[0].name = "core";
    classes[1].name = "atom";
    for (const unsigned cpu : online) {
      class_of[cpu] = std::ranges::binary_search(*little, cpu) ? 1 : 0;
    }
    return classes;
  }
  classes.emplace_back();
  for (const unsigned cpu : online) {
    class_of[cpu] = 0;
  }
  return classes;
}

std::map<unsigned, unsigned> read_nodes(const fs::path& root) {
  std::map<unsigned, unsigned> node_of;
  const fs::path node_root = root / "sys/devices/system/node";
  for (const unsigned node : numbered_entries(node_root, "node")) {
    const auto cpus = read_cpu_list(node_root / ("node" + std::to_string(node)) / "cpulist");
    for (const unsigned cpu : cpus.value_or(std::vector<unsigned>{})) {
      node_of.emplace(cpu, node);
    }
  }
  return node_of;
}

std::vector<unsigned> read_allowed(const fs::path& root) {
  std::ifstream status(root / "proc/self/status");
  constexpr std::string_view key = "Cpus_allowed_list:";
  for (std::string line; std::getline(status, line);) {
    if (line.starts_with(key)) {
      return parse_cpu_list(std::string_view(line).substr(key.size()))
          .value_or(std::vector<unsigned>{});
    }
  }
  return {};
}

// The CPU bandwidth limit of a cgroup and its ancestors (the tightest one applies), in CPUs.
// Each `limit` call returns the limit set at one directory, if any.
template <class Limit>
double tightest_limit(const fs::path& mount, std::string_view cgroup_path, Limit limit) {
  double tightest = 0;
  fs::path path = fs::path(cgroup_path).relative_path();
  while (true) {
    if (const auto cpus = limit(mount / path); cpus && (tightest == 0 || *cpus < tightest)) {
      tightest = *cpus;
    }
    if (path.empty()) {
      return tightest;
    }
    path = path.has_parent_path() ? path.parent_path() : fs::path();
  }
}

std::optional<double> ratio(std::optional<std::uint64_t> quota,
                            std::optional<std::uint64_t> period) {
  if (!quota || !period || *quota == 0 || *period == 0) {
    return std::nullopt;
  }
  return static_cast<double>(*quota) / static_cast<double>(*period);
}

// cgroup v2 cpu.max holds "max <period>" or "<quota> <period>".
std::optional<double> v2_limit(const fs::path& dir) {
  const auto line = read_line(dir / "cpu.max");
  if (!line) {
    return std::nullopt;
  }
  const auto space = line->find(' ');
  if (space == std::string::npos) {
    return std::nullopt;
  }
  const std::string_view text(*line);
  return ratio(parse_unsigned(text.substr(0, space)), parse_unsigned(text.substr(space + 1)));
}

// cgroup v1 holds the quota (-1: none, which does not parse as unsigned) and the period
// in two files.
std::optional<double> v1_limit(const fs::path& dir) {
  return ratio(read_unsigned(dir / "cpu.cfs_quota_us"), read_unsigned(dir / "cpu.cfs_period_us"));
}

// /proc/self/cgroup lines are "<hierarchy>:<controllers>:<path>"; v2 is "0::<path>".
double read_cpu_quota(const fs::path& root) {
  std::ifstream cgroups(root / "proc/self/cgroup");
  const fs::path mount = root / "sys/fs/cgroup";
  for (std::string line; std::getline(cgroups, line);) {
    const auto first = line.find(':');
    const auto second = line.find(':', first == std::string::npos ? 0 : first + 1);
    if (first == std::string::npos || second == std::string::npos) {
      continue;
    }
    const std::string_view text(line);
    const std::string_view controllers = text.substr(first + 1, second - first - 1);
    const std::string_view path = text.substr(second + 1);
    if (text.substr(0, first) == "0" && controllers.empty()) {
      if (const double q = tightest_limit(mount, path, v2_limit); q > 0) {
        return q;
      }
      continue;
    }
    // The cpu controller's mount point is named after the controllers it carries.
    const std::string list = "," + std::string(controllers) + ",";
    if (list.find(",cpu,") != std::string::npos) {
      for (const std::string_view dir : {"cpu,cpuacct", "cpuacct,cpu", "cpu"}) {
        if (const double q = tightest_limit(mount / dir, path, v1_limit); q > 0) {
          return q;
        }
      }
    }
  }
  return 0;
}

} // namespace

topology topology_from_sysfs(const std::filesystem::path& root, std::size_t page_size) {
  const fs::path cpu_root = root / "sys/devices/system/cpu";
  std::vector<unsigned> online =
      read_cpu_list(cpu_root / "online").value_or(std::vector<unsigned>{});
  if (online.empty()) {
    online = numbered_entries(cpu_root, "cpu");
  }

  topology_parts parts;
  parts.source = "linux-sysfs";
  parts.page_size = page_size;
  parts.allowed_cpus = read_allowed(root);
  parts.cpu_quota = read_cpu_quota(root);
  std::map<unsigned, unsigned> class_of;
  parts.classes = read_classes(root, cpu_root, online, class_of);
  const std::map<unsigned, unsigned> node_of = read_nodes(root);

  std::vector<bool> class_seen(parts.classes.size(), false);
  for (const unsigned id : online) {
    const fs::path dir = cpu_root / ("cpu" + std::to_string(id));
    cpu_record cpu;
    cpu.id = id;
    // core_cpus_list is the current name (Linux 5.4); thread_siblings_list the older one.
    auto siblings = read_cpu_list(dir / "topology/core_cpus_list");
    if (!siblings) {
      siblings = read_cpu_list(dir / "topology/thread_siblings_list");
    }
    cpu.smt_siblings = siblings.value_or(std::vector<unsigned>{});
    // Some Arm firmware reports -1 (no package information), which does not parse; it
    // means one package.
    cpu.package = to_unsigned(read_unsigned(dir / "topology/physical_package_id").value_or(0));
    const auto node = node_of.find(id);
    cpu.node = node == node_of.end() ? 0 : node->second;
    cpu.core_class = class_of.at(id);
    cpu_caches caches = read_caches(dir, online);
    cpu.l2_sharing = std::move(caches.l2_sharing);
    cpu.l3_sharing = std::move(caches.l3_sharing);
    // A class's geometry is that of its first CPU.
    if (!class_seen[cpu.core_class]) {
      class_seen[cpu.core_class] = true;
      core_class& c = parts.classes[cpu.core_class];
      c.l1d = caches.l1d;
      c.l2 = caches.l2;
      c.l3 = caches.l3;
    }
    parts.cpus.push_back(std::move(cpu));
  }
  return assemble_topology(std::move(parts));
}

} // namespace fftn::inline v0::detail
