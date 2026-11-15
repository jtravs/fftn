// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "threading/schedule.hpp"

#include "core/program.hpp"
#include "fftn/team.hpp"
#include "platform/topology.hpp"
#include "threading/cache_groups.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>

namespace fftn::inline v0::detail {

machine_profile profile_for(const topology& t) noexcept {
  machine_profile p;
  if (!t.classes.empty()) {
    set_grid_sizes(p, t.classes.front());
  }
  // The defaults are a Zen 3 CCX's. Two other kinds of cache group were measured (T1, T5):
  // an Apple processor (macOS's reader, which numbers CPUs itself, topology.hpp, and an L2
  // shared by a cluster of cores rather than each core's own: an Intel Mac has per-core L2s
  // and takes the defaults), whose performance cluster runs a short row so fast that
  // splitting pays only from longer rows; a last-level group of more than mesh_group_lanes
  // cores is a mesh (the Xeon's 18 cores). On neither did T1's measured costs pass the
  // adoption test, so both keep the provisional synchronisation cost and claim runs.
  const bool cluster_l2 = !t.classes.empty() && t.classes.front().l2.shared_by >
                                                    std::max(t.classes.front().l1d.shared_by, 1U);
  const bool apple = !t.os_cpu_ids && cluster_l2;
  std::size_t cpus_per_core = t.cores.empty() ? 1 : t.cores.front().cpus.size();
  for (const cpu_group& core : t.cores) {
    cpus_per_core = std::min(cpus_per_core, core.cpus.size());
  }
  const auto group_cpus = static_cast<std::size_t>(cpus_per_cache_group(t));
  const bool mesh = group_cpus > static_cast<std::size_t>(p.mesh_group_lanes) *
                                     std::max<std::size_t>(cpus_per_core, 1);
  if (apple) {
    p.split_min_points = 8192;
  }
  if (apple || mesh) {
    p.sync_ns = 700;
    p.claim_divisor = 2;
  }
  return p;
}

machine_profile default_machine_profile() noexcept {
  machine_profile p;
  try {
    p = profile_for(system_topology());
  } catch (...) {
    // system_topology() throws on an unreadable FFTN_TOPOLOGY recording; this function
    // is noexcept, so the defaults stand (as cache_group_size does).
    p = machine_profile{};
  }
  const unsigned cpus = std::thread::hardware_concurrency();
  p.cpus = static_cast<int>(std::min<unsigned>(cpus, std::numeric_limits<int>::max()));
  return p;
}

// Experiment X12b. Where each core has its own L2 (x86: shared only by a core's SMT
// threads, like its L1), a chain streams from it well: the grid paid only beyond half of
// L2, and three phases only beyond L2 (Xeon Gold 6240, EPYC 7763). An L2 shared by a
// cluster (Apple) is far: there the chain lost twofold once the row left L1, and three
// phases paid from an eighth of L2 (X12).
void set_grid_sizes(machine_profile& p, const core_class& c) noexcept {
  const bool private_l2 = c.l2.shared_by != 0 && c.l2.shared_by <= std::max(c.l1d.shared_by, 1U);
  if (c.l1d.size != 0) {
    p.grid_min_bytes = c.l1d.size;
    p.grid_block_bytes = c.l1d.size;
  }
  if (c.l2.size != 0) {
    if (private_l2) {
      p.grid_min_bytes = std::max(p.grid_min_bytes, c.l2.size / 2);
    }
    p.grid_three_bytes = private_l2 ? c.l2.size : c.l2.size / 8;
    p.grid_fuse_bytes = c.l2.size / 4;
  }
}

bool runs_inline(const work_graph& g, const machine_profile& profile) noexcept {
  return static_cast<double>(g.tasks) * (g.work * profile.ns_per_work) <
         profile.inline_multiple * profile.dispatch_ns;
}

schedule make_schedule(const work_graph& g, const lane_map& lanes,
                       const machine_profile& profile) noexcept {
  schedule s;
  s.profile = profile;
  s.tasks = g.tasks;
  s.task_ns = g.work * profile.ns_per_work;
  s.phases = g.phases.size();
  s.widest = g.widest;
  s.points = g.points;
  s.splittable = g.chunks > 1 && g.chunks < max_counted;
  s.oversubscribed = profile.cpus > 0 && std::max(lanes.lanes, lanes.threads) > profile.cpus;
  s.grain = g.tasks < max_counted ? 1 : (g.tasks + max_counted - 1) / max_counted;
  // Rule 1, and teams too large for the counters.
  if (lanes.lanes <= 1 || std::cmp_greater_equal(lanes.lanes, max_counted) ||
      runs_inline(g, profile)) {
    s.inline_all = true;
    s.lanes_dealt = lanes.size_of(0);
    s.dispatch = 1;
    return s;
  }
  // Rules 4 and 5: as many groups as have a group's worth of tasks, at least one; a batch
  // smaller than a group stays in one.
  const auto per_group = static_cast<std::size_t>(lanes.per_group());
  const auto groups = static_cast<std::size_t>(lanes.groups());
  s.groups = static_cast<int>(std::min(groups, (g.tasks + per_group - 1) / per_group));
  s.lanes_dealt = std::min(lanes.lanes, s.groups * lanes.per_group());
  s.dispatch = s.lanes_dealt;
  if (std::cmp_less(g.tasks, s.lanes_dealt)) {
    bool splits = false;
    for (int h = 0; h < s.groups && !splits; ++h) {
      splits = share_of(s, lanes, h).split != 0;
    }
    if (!splits) {
      s.dispatch = static_cast<int>(g.tasks);
    }
  }
  return s;
}

group_share share_of(const schedule& s, const lane_map& lanes, int group) noexcept {
  group_share share;
  if (group >= s.groups) {
    return share;
  }
  // Contiguous ranges in proportion to the groups' members, in exact integers.
  const auto dealt = static_cast<std::size_t>(s.lanes_dealt);
  const auto from = static_cast<std::size_t>(lanes.base(group));
  const auto to =
      static_cast<std::size_t>(std::min(lanes.base(group) + lanes.size_of(group), s.lanes_dealt));
  share.first = chunk_start(s.tasks, dealt, from);
  share.count = chunk_start(s.tasks, dealt, to) - share.first;
  share.whole = share.count;
  if (s.inline_all || s.oversubscribed || !s.splittable || share.count == 0) {
    return share;
  }
  // Rules 2 and 3.
  const auto members = static_cast<std::size_t>(lanes.size_of(group));
  const machine_profile& p = s.profile;
  const bool mesh = lanes.size_of(group) > p.mesh_group_lanes;
  if (s.points < (mesh ? p.split_min_points_mesh : p.split_min_points)) {
    return share;
  }
  std::size_t split = 0;
  if (share.count >= members) {
    split = share.count % members;
    if (split == 0 || s.points < p.remainder_split_min_points) {
      return share;
    }
  } else {
    split = share.count;
  }
  const std::size_t by_points = s.points / std::max<std::size_t>(p.min_points_per_member, 1);
  const std::size_t cap = std::min({members / split, by_points, std::size_t{s.widest}});
  if (cap < 2) {
    return share;
  }
  const double team_ns =
      s.task_ns / static_cast<double>(cap) + static_cast<double>(s.phases) * p.sync_ns;
  if (team_ns >= s.task_ns) {
    return share;
  }
  share.whole = share.count - split;
  share.split = split;
  share.cap = static_cast<int>(cap);
  return share;
}

std::size_t whole_units(const schedule& s, const group_share& share) noexcept {
  return (share.whole + s.grain - 1) / s.grain;
}

std::uint32_t done_target(const schedule& s, const lane_map& lanes) noexcept {
  if (s.inline_all) {
    return 1;
  }
  std::size_t units = 0;
  for (int g = 0; g < s.groups; ++g) {
    const group_share share = share_of(s, lanes, g);
    units += whole_units(s, share) + share.split;
  }
  return static_cast<std::uint32_t>(units);
}

transform_range owned_tasks(std::size_t tasks, int lanes, int lane) noexcept {
  const auto l = static_cast<std::size_t>(lanes);
  const auto r = static_cast<std::size_t>(lane);
  if (tasks < l) {
    return r < tasks ? transform_range{.first = r, .count = 1}
                     : transform_range{.first = tasks, .count = 0};
  }
  const std::size_t first = chunk_start(tasks, l, r);
  return {.first = first, .count = chunk_start(tasks, l, r + 1) - first};
}

} // namespace fftn::inline v0::detail
