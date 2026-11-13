// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/platform_text.hpp"

#include "platform/cpu_features.hpp"
#include "platform/cpu_list.hpp"
#include "platform/topology.hpp"

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace fftn::test {

namespace {

using fftn::detail::cache_geometry;
using fftn::detail::cpu_group;
using fftn::detail::format_cpu_list;

// "name count: list", or "name 0" for the empty set.
void set(std::ostringstream& out, const char* name, const std::vector<unsigned>& cpus) {
  out << name << ' ' << cpus.size();
  if (!cpus.empty()) {
    out << ": " << format_cpu_list(cpus);
  }
  out << '\n';
}

void groups(std::ostringstream& out, const char* name, const std::vector<cpu_group>& list) {
  out << name << ' ' << list.size() << '\n';
  for (const cpu_group& g : list) {
    out << "  " << g.id << ": " << format_cpu_list(g.cpus) << '\n';
  }
}

void cache(std::ostringstream& out, const char* name, const cache_geometry& g) {
  out << "  " << name << " size=" << g.size << " line=" << g.line_size << " ways=" << g.ways
      << " shared_by=" << g.shared_by << '\n';
}

} // namespace

std::string features_text(const fftn::detail::cpu_features& f) {
  std::ostringstream out;
  const auto flag = [&](const char* name, bool value) {
    out << name << ' ' << (value ? 1 : 0) << '\n';
  };
  flag("sse2", f.sse2);
  flag("sse3", f.sse3);
  flag("ssse3", f.ssse3);
  flag("sse41", f.sse41);
  flag("sse42", f.sse42);
  flag("popcnt", f.popcnt);
  flag("avx", f.avx);
  flag("avx2", f.avx2);
  flag("fma", f.fma);
  flag("avx512f", f.avx512f);
  flag("avx512dq", f.avx512dq);
  flag("avx512bw", f.avx512bw);
  flag("avx512vl", f.avx512vl);
  flag("neon", f.neon);
  flag("fcma", f.fcma);
  out << "sve_bits " << f.sve_bits << '\n';
  return out.str();
}

std::string topology_text(const fftn::detail::topology& t) {
  std::ostringstream out;
  out << "source " << t.source << '\n';
  out << "os_cpu_ids " << (t.os_cpu_ids ? 1 : 0) << '\n';
  out << "page_size " << t.page_size << '\n';
  std::vector<unsigned> ids;
  ids.reserve(t.cpus.size());
  for (const auto& cpu : t.cpus) {
    ids.push_back(cpu.id);
  }
  set(out, "logical_cpus", ids);
  set(out, "allowed_cpus", t.allowed_cpus);
  out << "cpu_quota " << t.cpu_quota << '\n';
  out << "classes " << t.classes.size() << '\n';
  for (std::size_t i = 0; i < t.classes.size(); ++i) {
    const auto& c = t.classes[i];
    std::vector<unsigned> members;
    for (const auto& cpu : t.cpus) {
      if (cpu.core_class == i) {
        members.push_back(cpu.id);
      }
    }
    out << "  " << i << ": name=\"" << c.name << "\" capacity=" << c.capacity
        << " cores=" << c.cores << " logical_cpus=" << c.logical_cpus
        << " cpus=" << format_cpu_list(members) << '\n';
    cache(out, "  l1d", c.l1d);
    cache(out, "  l2", c.l2);
    cache(out, "  l3", c.l3);
  }
  groups(out, "packages", t.packages);
  groups(out, "numa_nodes", t.numa_nodes);
  groups(out, "l3_groups", t.l3_groups);
  groups(out, "l2_groups", t.l2_groups);
  groups(out, "cores", t.cores);
  return out.str();
}

} // namespace fftn::test
