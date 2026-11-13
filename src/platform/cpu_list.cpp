// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/cpu_list.hpp"

#include "platform/parse_text.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fftn::inline v0::detail {

namespace {

// Larger CPU numbers than this are taken as corruption, not as a machine: a range such as
// "0-4000000000" must not allocate gigabytes. Linux's own limit (NR_CPUS) is 8192.
constexpr unsigned max_cpu_number = 1U << 20U;

std::optional<unsigned> parse_number(std::string_view text) {
  const auto value = parse_unsigned(text);
  if (!value || *value > max_cpu_number) {
    return std::nullopt;
  }
  return static_cast<unsigned>(*value);
}

} // namespace

std::optional<std::vector<unsigned>> parse_cpu_list(std::string_view text) {
  text = trim(text);
  std::vector<unsigned> cpus;
  while (!text.empty()) {
    const auto comma = text.find(',');
    const std::string_view item = text.substr(0, comma);
    text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    const auto dash = item.find('-');
    const auto first = parse_number(item.substr(0, dash));
    const auto last = dash == std::string_view::npos ? first : parse_number(item.substr(dash + 1));
    if (!first || !last || *last < *first) {
      return std::nullopt;
    }
    for (unsigned cpu = *first; cpu <= *last; ++cpu) {
      cpus.push_back(cpu);
    }
  }
  std::ranges::sort(cpus);
  const auto duplicates = std::ranges::unique(cpus);
  cpus.erase(duplicates.begin(), duplicates.end());
  return cpus;
}

std::string format_cpu_list(const std::vector<unsigned>& cpus) {
  std::string text;
  std::size_t i = 0;
  while (i < cpus.size()) {
    std::size_t j = i;
    while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) {
      ++j;
    }
    if (!text.empty()) {
      text += ',';
    }
    text += std::to_string(cpus[i]);
    if (j > i) {
      text += '-';
      text += std::to_string(cpus[j]);
    }
    i = j + 1;
  }
  return text;
}

} // namespace fftn::inline v0::detail
