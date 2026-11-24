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
        .key = "GRID_TWO_ODD_PART",
        .member = &fftn::detail::machine_profile::grid_two_odd_part,
    },
    size_field{
        .key = "GRID_TWO_PART_BYTES",
        .member = &fftn::detail::machine_profile::grid_two_part_bytes,
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
        .key = "GRID_TWO_BLOCK_BYTES",
        .member = &fftn::detail::machine_profile::grid_two_block_bytes,
    },
    size_field{
        .key = "GRID_TWO_POW2_BYTES",
        .member = &fftn::detail::machine_profile::grid_two_pow2_bytes,
    },
    size_field{
        .key = "DEPTH_MIN_BYTES",
        .member = &fftn::detail::machine_profile::depth_min_bytes,
    },
    size_field{
        .key = "DEPTH_MAX_BYTES",
        .member = &fftn::detail::machine_profile::depth_max_bytes,
    },
    size_field{
        .key = "POW2_GRID_BYTES",
        .member = &fftn::detail::machine_profile::pow2_grid_bytes,
    },
    size_field{
        .key = "GRID_PROFILE_PHASE_RADIX",
        .member = &fftn::detail::machine_profile::grid_phase_radix,
    },
    size_field{
        .key = "GRID_THREE_BLOCK",
        .member = &fftn::detail::machine_profile::grid_three_block,
    },
    size_field{
        .key = "CHAIN_PART_BYTES",
        .member = &fftn::detail::machine_profile::chain_part_bytes,
    },
    size_field{
        .key = "CHAIN_MIXED_BYTES",
        .member = &fftn::detail::machine_profile::chain_mixed_bytes,
    },
    size_field{
        .key = "CHAIN_PART_LIMIT_BYTES",
        .member = &fftn::detail::machine_profile::chain_part_limit_bytes,
    },
    size_field{
        .key = "GRID_PAD_PERIOD",
        .member = &fftn::detail::machine_profile::grid_pad_period,
    },
    size_field{
        .key = "POW2_CHAIN_BYTES",
        .member = &fftn::detail::machine_profile::pow2_chain_bytes,
    },
    size_field{
        .key = "POW2_RADIX8_BYTES",
        .member = &fftn::detail::machine_profile::pow2_radix8_bytes,
    },
    size_field{
        .key = "BLUESTEIN_POW2_PART_BYTES",
        .member = &fftn::detail::machine_profile::bluestein_pow2_part_bytes,
    },
    size_field{
        .key = "LONG_CONVOLUTION_BYTES",
        .member = &fftn::detail::machine_profile::long_convolution_bytes,
    },
    // FFTN_X_SCREEN1124: campaign-only options of the cloud reach screen (branch
    // exp/cloud-screen-1124, never merged; dev/KNOBS.toml): X50's convolution limit and X63's
    // packed c2r threshold set per arm (C2R_PACKED_BYTES=never: the workspace block).
    size_field{
        .key = "DEPTH_CONVOLUTION_MAX_BYTES",
        .member = &fftn::detail::machine_profile::depth_convolution_max_bytes,
    },
    size_field{
        .key = "C2R_PACKED_BYTES",
        .member = &fftn::detail::machine_profile::c2r_packed_bytes,
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
  // 0 is a value here: the field off, the plans as without it (X57's control arm).
  if (key == "FIT_ROW_BYTES") {
    const std::optional<std::size_t> v = parse_size(value, 0, false);
    if (!v) {
      error = bad + " (a whole number)";
    } else {
      s.profile.fit_row_bytes = *v;
    }
    return true;
  }
  // X62's switch for a Bluestein stage over C copies (machine_profile::bluestein_whole_lanes).
  if (key == "BLUESTEIN_WHOLE_LANES") {
    if (value == "0" || value == "1") {
      s.profile.bluestein_whole_lanes = value == "1";
    } else {
      error = bad + " (0 or 1)";
    }
    return true;
  }
  // FFTN_X_SCREEN1124: X58's batch leaf of 12 (machine_profile::batch_leaf_12) per arm.
  if (key == "BATCH_LEAF_12") {
    if (value == "0" || value == "1") {
      s.profile.batch_leaf_12 = value == "1";
    } else {
      error = bad + " (0 or 1)";
    }
    return true;
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
