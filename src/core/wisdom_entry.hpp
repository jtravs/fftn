// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Wisdom records (DESIGN.md 11.3): one line of text per plan, tab-separated fields
//   fftn-wisdom-1, library version, processor model, instruction set, shape key, program,
//   check
// where the program is a candidate's choice_text (core/candidates.hpp) and the check the
// program's fingerprint (16 hexadecimal digits). A record applies only to the same version,
// processor model, instruction set and shape key, and only if its program rebuilds to the
// same fingerprint: so an entry either reproduces the recorded plan exactly or is ignored.
#pragma once

#include "fftn/detail/core.hpp"
#include "fftn/options.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace fftn::inline v0 {
class wisdom;
} // namespace fftn::inline v0

namespace fftn::inline v0::detail {

inline constexpr std::string_view wisdom_format = "fftn-wisdom-1";

struct wisdom_record {
  std::string version;
  std::string cpu;
  std::string isa;
  std::string key;
  std::string program;
  std::uint64_t check = 0;
};

// A line's record, or nullopt if the line is not one (wrong field count or format tag, a
// program that is not a choice, a check that is not 16 hexadecimal digits).
[[nodiscard]] std::optional<wisdom_record> parse_record(std::string_view line);
[[nodiscard]] std::string record_text(const wisdom_record& r);

// The record of a plan, on this machine and library version.
[[nodiscard]] wisdom_record record_of(const plan_core& c);

// The entry of w for the plan's version, processor model, instruction set and shape key.
[[nodiscard]] std::optional<wisdom_record> find_entry(const wisdom& w, const plan_core& c);

// The shape and the options that change the program, as one line of text: the kind,
// precision, direction (c2c), in place or not, each axis's length and strides and each
// batch dimension's count and distances as given (0 for a default).
[[nodiscard]] std::string shape_key(const shape_view& s, const options& opts);

} // namespace fftn::inline v0::detail
