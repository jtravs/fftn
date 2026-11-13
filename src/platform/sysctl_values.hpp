// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// A snapshot of macOS sysctl values, keyed by name ("hw.perflevel0.l2cachesize"), each held
// as the text `sysctl` prints. Feature and topology detection on macOS read only such a
// snapshot, so the same decoders run on the live system and on recorded fixtures.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace fftn::inline v0::detail {

using sysctl_values = std::map<std::string, std::string, std::less<>>;

// Parses `sysctl` output: one "name: value" per line. Lines without ": " are ignored.
[[nodiscard]] sysctl_values parse_sysctl_text(std::string_view text);

// The value of an integer key; nullopt when it is absent or not a non-negative integer.
[[nodiscard]] std::optional<std::uint64_t> sysctl_integer(const sysctl_values& values,
                                                          std::string_view name);

// How a key's value is stored by the kernel, which sysctlbyname does not say.
enum class sysctl_kind : unsigned char {
  integer,  // a 32- or 64-bit integer
  text,     // a NUL-terminated string
  integers, // an array of 64-bit integers, printed space-separated
};

struct sysctl_key {
  std::string_view name;
  sysctl_kind kind = sysctl_kind::integer;
};

// Reads the named keys from the running macOS kernel; keys the kernel lacks are left out.
// Returns an empty snapshot on other operating systems.
[[nodiscard]] sysctl_values read_sysctl(std::span<const sysctl_key> keys);

} // namespace fftn::inline v0::detail
