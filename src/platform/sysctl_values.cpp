// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/sysctl_values.hpp"

#include "platform/parse_text.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#ifdef __APPLE__
#include <cstddef>
#include <cstring>
#include <sys/sysctl.h>
#include <vector>
#endif

namespace fftn::inline v0::detail {

sysctl_values parse_sysctl_text(std::string_view text) {
  sysctl_values values;
  while (!text.empty()) {
    const auto newline = text.find('\n');
    std::string_view line = text.substr(0, newline);
    text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    const auto separator = line.find(": ");
    if (separator == std::string_view::npos || separator == 0) {
      continue;
    }
    values.insert_or_assign(std::string(line.substr(0, separator)),
                            std::string(line.substr(separator + 2)));
  }
  return values;
}

std::optional<std::uint64_t> sysctl_integer(const sysctl_values& values, std::string_view name) {
  const auto it = values.find(name);
  if (it == values.end()) {
    return std::nullopt;
  }
  return parse_unsigned(it->second);
}

#ifdef __APPLE__

namespace {

std::optional<std::string> read_one(const sysctl_key& key) {
  const std::string name(key.name);
  std::size_t size = 0;
  if (sysctlbyname(name.c_str(), nullptr, &size, nullptr, 0) != 0 || size == 0) {
    return std::nullopt;
  }
  std::vector<unsigned char> bytes(size);
  if (sysctlbyname(name.c_str(), bytes.data(), &size, nullptr, 0) != 0) {
    return std::nullopt;
  }
  bytes.resize(size);
  switch (key.kind) {
  case sysctl_kind::text: {
    std::string text(bytes.begin(), bytes.end());
    text.resize(std::strlen(text.c_str()));
    return text;
  }
  case sysctl_kind::integer:
    // Signed values (hw.optional.arm.caps) are not read through this path; the keys FFTN
    // reads are sizes, counts and flags.
    if (size == sizeof(std::uint32_t)) {
      std::uint32_t value = 0;
      std::memcpy(&value, bytes.data(), sizeof value);
      return std::to_string(value);
    }
    if (size == sizeof(std::uint64_t)) {
      std::uint64_t value = 0;
      std::memcpy(&value, bytes.data(), sizeof value);
      return std::to_string(value);
    }
    return std::nullopt;
  case sysctl_kind::integers: {
    std::string text;
    for (std::size_t offset = 0; offset + sizeof(std::uint64_t) <= size;
         offset += sizeof(std::uint64_t)) {
      std::uint64_t value = 0;
      std::memcpy(&value, bytes.data() + offset, sizeof value);
      if (!text.empty()) {
        text += ' ';
      }
      text += std::to_string(value);
    }
    return text;
  }
  }
  return std::nullopt;
}

} // namespace

sysctl_values read_sysctl(std::span<const sysctl_key> keys) {
  sysctl_values values;
  for (const sysctl_key& key : keys) {
    if (auto value = read_one(key)) {
      values.insert_or_assign(std::string(key.name), *std::move(value));
    }
  }
  return values;
}

#else

sysctl_values read_sysctl(std::span<const sysctl_key> /*keys*/) { return {}; }

#endif

} // namespace fftn::inline v0::detail
