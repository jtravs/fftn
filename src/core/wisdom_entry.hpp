// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Wisdom records (DESIGN.md 11.3): one line of text per plan, tab-separated fields
//   fftn-wisdom-1, build, processor model, instruction set, shape key, program, check
// where the build is the library version and its build identifier ("0.0.0+1a2b..."), the
// program a candidate's choice_text (core/candidates.hpp) and the check the program's
// fingerprint (16 hexadecimal digits). A record applies only to the same build, processor
// model, instruction set and shape key, and only if its program rebuilds to the same
// fingerprint: so an entry either reproduces the recorded plan exactly or is ignored.
// Lines of another format or build are kept as they are, never parsed further.
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
  std::string build;
  std::string cpu;
  std::string isa;
  std::string key;
  std::string program;
  std::uint64_t check = 0;
};

// This library's build field: its version, "+" and its build identifier.
[[nodiscard]] std::string wisdom_build();
// This processor's model, as records write it.
[[nodiscard]] const std::string& wisdom_cpu();

// The fields that say where a record applies (format, build, processor, instruction set,
// shape key), joined by tabs: wisdom's index, and what find_entry looks up.
[[nodiscard]] std::string wisdom_place(std::string_view isa, std::string_view key);

// A line of this format and build as a record, or nullopt if it is not one (wrong field
// count, a program that is not a choice in its canonical text, a check that is not 16
// lowercase hexadecimal digits).
[[nodiscard]] std::optional<wisdom_record> parse_record(std::string_view line);
[[nodiscard]] std::string record_text(const wisdom_record& r);

// The record of a plan, on this machine and build.
[[nodiscard]] wisdom_record record_of(const plan_core& c);

// The entry of w at `place` (wisdom_place), parsed.
[[nodiscard]] std::optional<wisdom_record> find_entry(const wisdom& w, const std::string& place);

// The shape and the options that change the program or its results, as one line of text:
// the kind, precision, direction (c2c), in place or not, the scale (exactly, in hexadecimal),
// each axis's length and strides and each batch dimension's count and distances as given
// (0 for a default). The instruction set is a field of its own.
[[nodiscard]] std::string shape_key(const shape_view& s, const options& opts);

} // namespace fftn::inline v0::detail
