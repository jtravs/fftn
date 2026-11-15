// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The tuning options of FFTN's benchmark tools (bench/README.md, "FFTN options"): the machine
// profile's fields (src/threading/schedule.hpp) and the plan's chunking, by the names the
// fbench backend and the T1 harness accept. They set the private build settings
// (src/core/plan_core.hpp), as a test does; the library has no switch for them.
#pragma once

#include "core/plan_core.hpp"
#include "threading/schedule.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace fbench::fftn_tuning {

// A whole number at least `least`, or `never` (the largest value) when allowed.
inline std::optional<std::size_t> parse_size(std::string_view text, std::size_t least, bool never) {
  if (never && text == "never") {
    return std::numeric_limits<std::size_t>::max();
  }
  if (text.empty() || text.find_first_not_of("0123456789") != std::string_view::npos ||
      text.size() > 18) {
    return std::nullopt;
  }
  const std::size_t v = std::stoull(std::string(text));
  return v >= least ? std::optional(v) : std::nullopt;
}

inline std::optional<double> parse_real(std::string_view text) {
  const std::string s(text);
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || end != s.c_str() + s.size() || !(v >= 0) ||
      v > std::numeric_limits<double>::max()) {
    return std::nullopt;
  }
  return v;
}

struct real_field {
  std::string_view key;
  double fftn::detail::machine_profile::* member;
};
struct size_field {
  std::string_view key;
  std::size_t fftn::detail::machine_profile::* member;
};
struct int_field {
  std::string_view key;
  int fftn::detail::machine_profile::* member;
};

constexpr std::array real_fields{
    real_field{.key = "NS_PER_WORK", .member = &fftn::detail::machine_profile::ns_per_work},
    real_field{.key = "DISPATCH_NS", .member = &fftn::detail::machine_profile::dispatch_ns},
    real_field{.key = "INLINE_MULTIPLE", .member = &fftn::detail::machine_profile::inline_multiple},
    real_field{.key = "SYNC_NS", .member = &fftn::detail::machine_profile::sync_ns},
    real_field{.key = "SPIN_NS", .member = &fftn::detail::machine_profile::spin_ns},
    real_field{.key = "YIELD_NS", .member = &fftn::detail::machine_profile::yield_ns},
};
constexpr std::array size_fields{
    size_field{
        .key = "MIN_POINTS_PER_MEMBER",
        .member = &fftn::detail::machine_profile::min_points_per_member,
    },
    size_field{
        .key = "SPLIT_MIN_POINTS",
        .member = &fftn::detail::machine_profile::split_min_points,
    },
    size_field{
        .key = "SPLIT_MIN_POINTS_MESH",
        .member = &fftn::detail::machine_profile::split_min_points_mesh,
    },
    size_field{
        .key = "REMAINDER_SPLIT_MIN_POINTS",
        .member = &fftn::detail::machine_profile::remainder_split_min_points,
    },
    size_field{.key = "GRID_MIN_BYTES", .member = &fftn::detail::machine_profile::grid_min_bytes},
    size_field{
        .key = "GRID_THREE_BYTES",
        .member = &fftn::detail::machine_profile::grid_three_bytes,
    },
    size_field{
        .key = "GRID_FUSE_BYTES",
        .member = &fftn::detail::machine_profile::grid_fuse_bytes,
    },
    size_field{
        .key = "GRID_BLOCK_BYTES",
        .member = &fftn::detail::machine_profile::grid_block_bytes,
    },
    size_field{.key = "GRID_BLOCK", .member = &fftn::detail::machine_profile::grid_block},
    size_field{
        .key = "CHAIN_PART_BYTES",
        .member = &fftn::detail::machine_profile::chain_part_bytes,
    },
    size_field{
        .key = "CHAIN_MIXED_BYTES",
        .member = &fftn::detail::machine_profile::chain_mixed_bytes,
    },
};
constexpr std::array int_fields{
    int_field{
        .key = "MESH_GROUP_LANES",
        .member = &fftn::detail::machine_profile::mesh_group_lanes,
    },
    int_field{.key = "CLAIM_DIVISOR", .member = &fftn::detail::machine_profile::claim_divisor},
};

// Applies one profile or chunking option; false when the key is not one.
inline bool apply_tuning(fftn::detail::build_settings& s, std::string_view key,
                         std::string_view value, std::string& error) {
  const std::string bad = std::string(key) + " has a bad value: " + std::string(value);
  for (const real_field& f : real_fields) {
    if (key == f.key) {
      const std::optional<double> v = parse_real(value);
      if (!v) {
        error = bad + " (a non-negative number)";
      } else {
        s.profile.*f.member = *v;
      }
      return true;
    }
  }
  for (const size_field& f : size_fields) {
    if (key == f.key) {
      const std::optional<std::size_t> v = parse_size(value, 1, true);
      if (!v) {
        error = bad + " (a whole number from 1, or never)";
      } else {
        s.profile.*f.member = *v;
      }
      return true;
    }
  }
  for (const int_field& f : int_fields) {
    if (key == f.key) {
      const std::optional<std::size_t> v = parse_size(value, 1, false);
      if (!v || *v > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error = bad + " (a whole number from 1)";
      } else {
        s.profile.*f.member = static_cast<int>(*v);
      }
      return true;
    }
  }
  if (key == "CHUNK_POINTS") {
    const std::optional<std::size_t> v = parse_size(value, 1, false);
    if (!v) {
      error = bad + " (a whole number from 1)";
    } else {
      s.chunk_points = *v;
    }
    return true;
  }
  return false;
}

} // namespace fbench::fftn_tuning
