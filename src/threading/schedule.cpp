// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "threading/schedule.hpp"

#include "core/program.hpp"
#include "threading/work_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>

namespace fftn::inline v0::detail {

machine_profile default_machine_profile() noexcept {
  machine_profile p;
  const unsigned cpus = std::thread::hardware_concurrency();
  p.cpus = static_cast<int>(std::min<unsigned>(cpus, std::numeric_limits<int>::max()));
  return p;
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
  s.oversubscribed = profile.cpus > 0 && lanes.lanes > profile.cpus;
  // Rule 1, and batches too large for the counters.
  const double job_ns = static_cast<double>(g.tasks) * s.task_ns;
  if (lanes.lanes <= 1 || g.tasks >= max_counted ||
      job_ns < profile.inline_multiple * profile.dispatch_ns) {
    s.inline_all = true;
    s.lanes_dealt = lanes.size_of(0);
    return s;
  }
  // Rules 4 and 5: as many groups as have a group's worth of tasks, at least one; a batch
  // smaller than a group stays in one.
  const auto per_group = static_cast<std::size_t>(lanes.per_group());
  const auto groups = static_cast<std::size_t>(lanes.groups());
  s.groups = static_cast<int>(std::min(groups, (g.tasks + per_group - 1) / per_group));
  s.lanes_dealt = std::min(lanes.lanes, s.groups * lanes.per_group());
  return s;
}

group_share share_of(const schedule& s, const lane_map& lanes, int group) noexcept {
  group_share share;
  if (group >= s.groups) {
    return share;
  }
  // Contiguous ranges in proportion to the groups' members. tasks < 2^31 and lanes < 2^31,
  // so the products fit.
  const auto tasks = static_cast<std::uint64_t>(s.tasks);
  const auto dealt = static_cast<std::uint64_t>(s.lanes_dealt);
  const auto from = static_cast<std::uint64_t>(lanes.base(group));
  const auto to =
      static_cast<std::uint64_t>(std::min(lanes.base(group) + lanes.size_of(group), s.lanes_dealt));
  share.first = static_cast<std::size_t>(tasks * from / dealt);
  share.count = static_cast<std::size_t>(tasks * to / dealt) - share.first;
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

task_range owned_tasks(std::size_t tasks, int lanes, int lane) noexcept {
  const auto l = static_cast<std::size_t>(lanes);
  const auto r = static_cast<std::size_t>(lane);
  const std::size_t first = chunk_start(tasks, l, r);
  return {.first = first, .count = chunk_start(tasks, l, r + 1) - first};
}

} // namespace fftn::inline v0::detail
