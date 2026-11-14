// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Teams: several of the caller's threads executing one plan together.
#pragma once

#include "fftn/config.hpp"

#include <cstddef>

namespace fftn::inline v0 {

/// One member of a team of the caller's threads that execute a plan together, each by
/// calling the team overload of `plan::execute` with its own `team` value and one shared
/// workspace (docs/threading.md).
///
/// A team is described by plain numbers, so any threading system can form one: the
/// threads of an OpenMP parallel region (`omp_get_thread_num()`, `omp_get_num_threads()`),
/// a Kokkos team, or `std::thread`s the caller started. FFTN starts no threads.
///
/// Members are grouped by the cache they share: ranks `[g * group_size, (g + 1) *
/// group_size)` form cache group `g` (the last group may be smaller). FFTN splits a
/// transform only among the members of one group, and deals whole transforms to groups.
/// The grouping affects speed only, never results.
///
/// Preconditions, for every execution: `0 <= rank < size`; every member of one execution
/// passes the same `size` and `group_size`, and no two pass the same `rank`.
struct team {
  /// This member's number, from 0 to `size - 1`.
  int rank = 0;

  /// The number of members; 1 means the calling thread alone.
  int size = 1;

  /// Members per cache group; 0 (or `size` or more) puts every member in one group.
  int group_size = 0;
};

/// When a member's call of the team overload of `plan::execute` returns.
enum class completion : unsigned char {
  /// When the whole output is written, on every member. Work is shared dynamically: a
  /// member takes whatever is left, so no member waits for one that has not arrived, and
  /// members that arrive late find nothing left to do. Every member still makes every
  /// call, late or not (the team overload of `plan::execute` says why).
  all,

  /// When this member's own transforms (`plan::transforms_of`) are written. Transforms are
  /// dealt statically and never split or taken by another member, so every member must
  /// take part. The caller synchronises its threads before reading transforms that other
  /// members own, and before the workspace serves another execution.
  own,
};

/// A range of the transforms of a plan's batch, in batch order: `count` transforms from
/// transform `first`, the arguments `plan::execute_transforms` takes.
struct transform_range {
  /// The first transform of the range.
  std::size_t first = 0;

  /// The number of transforms in the range; 0 for an empty range.
  std::size_t count = 0;
};

} // namespace fftn::inline v0
