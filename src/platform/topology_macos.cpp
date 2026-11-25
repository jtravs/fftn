// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The macOS topology reader, from a sysctl snapshot (compiled everywhere, tested against
// recorded snapshots). macOS reports counts and sizes per performance level (perflevel 0 is
// the fastest), not which logical CPU is which, so CPUs are numbered in class order and the
// sharing groups are consecutive runs of them; the topology says so (os_cpu_ids = false).
// There is no associativity, no affinity mask and no NUMA: those stay unknown or trivial.

#include "platform/parse_text.hpp"
#include "platform/sysctl_values.hpp"
#include "platform/topology.hpp"
#include "platform/topology_sources.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

namespace {

unsigned to_unsigned(std::uint64_t value) {
  return static_cast<unsigned>(
      std::min<std::uint64_t>(value, std::numeric_limits<unsigned>::max()));
}

unsigned integer(const sysctl_values& values, const std::string& name) {
  return to_unsigned(sysctl_integer(values, name).value_or(0));
}

std::size_t size(const sysctl_values& values, const std::string& name) {
  return static_cast<std::size_t>(sysctl_integer(values, name).value_or(0));
}

// The consecutive run of `per_group` CPUs, counted from `first`, that holds `cpu`.
std::vector<unsigned> run_of(unsigned first, unsigned cpu, unsigned per_group, unsigned end) {
  std::vector<unsigned> run;
  if (per_group == 0) {
    return run;
  }
  const unsigned start = first + ((cpu - first) / per_group) * per_group;
  for (unsigned c = start; c < std::min(start + per_group, end); ++c) {
    run.push_back(c);
  }
  return run;
}

// One class of `logical` CPUs on `physical` cores, numbered from `first`.
struct class_shape {
  unsigned logical = 0;
  unsigned physical = 0;
  unsigned per_l2 = 0;
  unsigned per_l3 = 0;
  unsigned packages = 1;
};

void add_cpus(topology_parts& parts, unsigned first, unsigned core_class, const class_shape& s) {
  const unsigned threads = s.physical > 0 ? std::max(1U, s.logical / s.physical) : 1;
  const unsigned end = first + s.logical;
  const unsigned per_package = std::max(1U, s.logical / std::max(1U, s.packages));
  for (unsigned cpu = first; cpu < end; ++cpu) {
    cpu_record r;
    r.id = cpu;
    r.smt_siblings = run_of(first, cpu, threads, end);
    r.package = (cpu - first) / per_package;
    r.core_class = core_class;
    r.l2_sharing = run_of(first, cpu, s.per_l2, end);
    r.l3_sharing = run_of(first, cpu, s.per_l3, end);
    parts.cpus.push_back(std::move(r));
  }
}

cache_geometry geometry(std::size_t bytes, std::size_t line, unsigned shared_by) {
  if (bytes == 0) {
    return {};
  }
  return {.size = bytes, .line_size = line, .ways = 0, .shared_by = shared_by};
}

// hw.cacheconfig: logical CPUs sharing each level, index 0 being memory ("18 1 6 0").
std::vector<unsigned> cache_config(const sysctl_values& values) {
  std::vector<unsigned> sharing;
  const auto it = values.find("hw.cacheconfig");
  if (it == values.end()) {
    return sharing;
  }
  std::string_view text = it->second;
  while (!text.empty()) {
    const auto space = text.find(' ');
    const std::string_view item = text.substr(0, space);
    text = space == std::string_view::npos ? std::string_view{} : text.substr(space + 1);
    sharing.push_back(to_unsigned(parse_unsigned(item).value_or(0)));
  }
  return sharing;
}

} // namespace

topology topology_from_sysctl(const sysctl_values& values) {
  topology_parts parts;
  parts.source = "macos-sysctl";
  parts.os_cpu_ids = false;
  parts.page_size = size(values, "hw.pagesize");
  if (const auto vendor = values.find("fftn.cpu_vendor"); vendor != values.end()) {
    parts.vendor = recorded_vendor(vendor->second, "fftn.cpu_vendor");
  }
  const std::size_t line = size(values, "hw.cachelinesize");

  const unsigned levels = integer(values, "hw.nperflevels");
  unsigned next = 0;
  for (unsigned p = 0; p < levels; ++p) {
    const std::string key = "hw.perflevel" + std::to_string(p) + ".";
    class_shape s;
    s.logical = integer(values, key + "logicalcpu");
    s.physical = integer(values, key + "physicalcpu");
    s.per_l2 = integer(values, key + "cpusperl2");
    s.per_l3 = integer(values, key + "cpusperl3");
    core_class c;
    if (const auto name = values.find(key + "name"); name != values.end()) {
      c.name = name->second;
    }
    const unsigned threads = s.physical > 0 ? std::max(1U, s.logical / s.physical) : 1;
    c.l1d = geometry(size(values, key + "l1dcachesize"), line, threads);
    c.l2 = geometry(size(values, key + "l2cachesize"), line, s.per_l2);
    c.l3 = geometry(size(values, key + "l3cachesize"), line, s.per_l3);
    if (c.l2.size == 0) {
      s.per_l2 = 0;
    }
    if (c.l3.size == 0) {
      s.per_l3 = 0;
    }
    add_cpus(parts, next, p, s);
    next += s.logical;
    parts.classes.push_back(std::move(c));
  }

  if (levels == 0) {
    // Intel Macs: one class; hw.cacheconfig gives the sharing of each level.
    class_shape s;
    s.logical = integer(values, "hw.logicalcpu");
    s.physical = integer(values, "hw.physicalcpu");
    s.packages = std::max(1U, integer(values, "hw.packages"));
    const std::vector<unsigned> sharing = cache_config(values);
    const auto shared = [&](std::size_t level) {
      return level < sharing.size() ? sharing[level] : 0U;
    };
    core_class c;
    c.l1d = geometry(size(values, "hw.l1dcachesize"), line, shared(1));
    c.l2 = geometry(size(values, "hw.l2cachesize"), line, shared(2));
    c.l3 = geometry(size(values, "hw.l3cachesize"), line, shared(3));
    s.per_l2 = c.l2.size > 0 ? shared(2) : 0;
    s.per_l3 = c.l3.size > 0 ? shared(3) : 0;
    add_cpus(parts, 0, 0, s);
    parts.classes.push_back(std::move(c));
  }
  return assemble_topology(std::move(parts));
}

sysctl_values read_topology_sysctl() {
  static constexpr std::array<sysctl_key, 10> base_keys{
      {
          {.name = "hw.nperflevels"},
          {.name = "hw.pagesize"},
          {.name = "hw.cachelinesize"},
          {.name = "hw.logicalcpu"},
          {.name = "hw.physicalcpu"},
          {.name = "hw.packages"},
          {.name = "hw.l1dcachesize"},
          {.name = "hw.l2cachesize"},
          {.name = "hw.l3cachesize"},
          {.name = "hw.cacheconfig", .kind = sysctl_kind::integers},
      },
  };
  sysctl_values values = read_sysctl(base_keys);

  static constexpr std::array<const char*, 8> fields{
      "logicalcpu",   "physicalcpu", "cpusperl2",   "cpusperl3",
      "l1dcachesize", "l2cachesize", "l3cachesize", "name",
  };
  const unsigned levels = integer(values, "hw.nperflevels");
  std::vector<std::string> names;
  names.reserve(std::size_t{levels} * fields.size());
  for (unsigned p = 0; p < levels; ++p) {
    for (const char* field : fields) {
      names.push_back("hw.perflevel" + std::to_string(p) + "." + field);
    }
  }
  std::vector<sysctl_key> keys;
  keys.reserve(names.size());
  for (const std::string& name : names) {
    keys.push_back(
        {.name = name, .kind = name.ends_with(".name") ? sysctl_kind::text : sysctl_kind::integer});
  }
  values.merge(read_sysctl(keys));
  return values;
}

} // namespace fftn::inline v0::detail
