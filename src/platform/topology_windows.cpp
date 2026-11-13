// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The Windows topology reader, from GetLogicalProcessorInformationEx(RelationAll) (Win32
// documentation of SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX). Logical CPU numbers are
// 64 * processor group + bit. Not yet run on Windows: it is compiled by the Windows CI job
// (task 0.8), and until then it is the least tested reader.

#include "platform/topology.hpp"
#include "platform/topology_sources.hpp"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

namespace {

std::vector<unsigned> cpus_of(const GROUP_AFFINITY& mask) {
  std::vector<unsigned> cpus;
  for (unsigned b = 0; b < 64; ++b) {
    if (((static_cast<std::uint64_t>(mask.Mask) >> b) & 1U) != 0) {
      cpus.push_back(64U * mask.Group + b);
    }
  }
  return cpus;
}

struct cache_record {
  cache_geometry geometry;
  std::vector<unsigned> sharing;
};

} // namespace

topology topology_from_windows() {
  DWORD length = 0;
  if (GetLogicalProcessorInformationEx(RelationAll, nullptr, &length) != FALSE ||
      GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return {};
  }
  // Stored as 64-bit words so that every record is suitably aligned.
  std::vector<std::uint64_t> buffer(length / sizeof(std::uint64_t) + 1);
  auto* const first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
  if (GetLogicalProcessorInformationEx(RelationAll, first, &length) == FALSE) {
    return {};
  }

  std::map<unsigned, cpu_record> cpus;
  std::map<unsigned, unsigned> efficiency;
  std::map<unsigned, std::map<unsigned, cache_record>> caches; // cpu -> level -> cache
  unsigned package = 0;
  const auto* const bytes = reinterpret_cast<const unsigned char*>(buffer.data());
  for (DWORD offset = 0; offset < length;) {
    const auto* const info =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(bytes + offset);
    offset += info->Size;
    switch (info->Relationship) {
    case RelationProcessorCore: {
      std::vector<unsigned> siblings;
      for (WORD g = 0; g < info->Processor.GroupCount; ++g) {
        const auto part = cpus_of(info->Processor.GroupMask[g]);
        siblings.insert(siblings.end(), part.begin(), part.end());
      }
      for (const unsigned cpu : siblings) {
        cpus[cpu].id = cpu;
        cpus[cpu].smt_siblings = siblings;
        efficiency[cpu] = info->Processor.EfficiencyClass;
      }
      break;
    }
    case RelationProcessorPackage:
      for (WORD g = 0; g < info->Processor.GroupCount; ++g) {
        for (const unsigned cpu : cpus_of(info->Processor.GroupMask[g])) {
          cpus[cpu].package = package;
        }
      }
      ++package;
      break;
    case RelationCache: {
      const CACHE_RELATIONSHIP& c = info->Cache;
      if (c.Type == CacheInstruction || c.Level < 1 || c.Level > 3) {
        break;
      }
      // Only the first group's mask: a cache never spans processor groups in practice.
      cache_record r;
      r.sharing = cpus_of(c.GroupMask);
      r.geometry.size = c.CacheSize;
      r.geometry.line_size = c.LineSize;
      // CACHE_FULLY_ASSOCIATIVE (0xFF) is reported as unknown ways.
      r.geometry.ways = c.Associativity == CACHE_FULLY_ASSOCIATIVE ? 0U : c.Associativity;
      r.geometry.shared_by = static_cast<unsigned>(r.sharing.size());
      for (const unsigned cpu : r.sharing) {
        caches[cpu][c.Level] = r;
      }
      break;
    }
    case RelationNumaNode:
      for (const unsigned cpu : cpus_of(info->NumaNode.GroupMask)) {
        cpus[cpu].node = info->NumaNode.NodeNumber;
      }
      break;
    default:
      break;
    }
  }

  // Higher EfficiencyClass values are faster cores; classes are ordered fastest first.
  std::vector<unsigned> distinct;
  for (const auto& [cpu, e] : efficiency) {
    distinct.push_back(e);
  }
  std::ranges::sort(distinct, std::ranges::greater{});
  const auto repeated = std::ranges::unique(distinct);
  distinct.erase(repeated.begin(), repeated.end());

  topology_parts parts;
  parts.source = "windows";
  parts.page_size = system_page_size();
  parts.classes.resize(std::max<std::size_t>(distinct.size(), 1));
  std::vector<bool> seen(parts.classes.size(), false);
  for (auto& [id, record] : cpus) {
    const auto e = efficiency.find(id);
    record.core_class =
        e == efficiency.end()
            ? 0U
            : static_cast<unsigned>(std::ranges::find(distinct, e->second) - distinct.begin());
    auto& levels = caches[id];
    record.l2_sharing = levels[2].sharing;
    record.l3_sharing = levels[3].sharing;
    if (!seen[record.core_class]) {
      seen[record.core_class] = true;
      core_class& c = parts.classes[record.core_class];
      c.l1d = levels[1].geometry;
      c.l2 = levels[2].geometry;
      c.l3 = levels[3].geometry;
    }
    parts.cpus.push_back(record);
  }
  return assemble_topology(std::move(parts));
}

} // namespace fftn::inline v0::detail

#else

namespace fftn::inline v0::detail {

topology topology_from_windows() { return {}; }

} // namespace fftn::inline v0::detail

#endif
