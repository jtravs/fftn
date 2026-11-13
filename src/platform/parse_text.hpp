// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The two text primitives the platform readers share: trimming and strict decimal parsing.
#pragma once

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>
#include <system_error>

namespace fftn::inline v0::detail {

[[nodiscard]] inline std::string_view trim(std::string_view text) noexcept {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  text.remove_suffix(text.size() - last - 1);
  text.remove_prefix(first);
  return text;
}

// The whole of `text` as a non-negative decimal integer; nullopt for anything else (signs,
// spaces, suffixes, overflow, the empty string).
[[nodiscard]] inline std::optional<std::uint64_t> parse_unsigned(std::string_view text) noexcept {
  std::uint64_t value = 0;
  const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || ec != std::errc{} || ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

} // namespace fftn::inline v0::detail
