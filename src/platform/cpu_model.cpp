// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/cpu_model.hpp"

#include "platform/sysctl_values.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace fftn::inline v0::detail {

std::string one_line(std::string_view text) {
  std::string out;
  bool space = false;
  for (const char ch : text) {
    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
      space = !out.empty();
      continue;
    }
    if (space) {
      out += ' ';
      space = false;
    }
    out += ch;
  }
  return out.empty() ? "unknown" : out;
}

std::string cpu_model_from_cpuinfo(std::string_view text) {
  // "key<tabs>: value" lines; the first processor's block ends at the first blank line.
  std::string model;
  std::string implementer;
  std::string variant;
  std::string part;
  std::size_t at = 0;
  while (at < text.size()) {
    const std::size_t end = std::min(text.find('\n', at), text.size());
    const std::string_view line = text.substr(at, end - at);
    at = end + 1;
    if (line.find_first_not_of(" \t\r") == std::string_view::npos) {
      if (!model.empty() || !part.empty()) {
        break;
      }
      continue;
    }
    const std::size_t colon = line.find(':');
    if (colon == std::string_view::npos) {
      continue;
    }
    const std::string key = one_line(line.substr(0, colon));
    const std::string value = one_line(line.substr(colon + 1));
    if (key == "model name" && model.empty()) {
      model = value;
    } else if (key == "CPU implementer" && implementer.empty()) {
      implementer = value;
    } else if (key == "CPU variant" && variant.empty()) {
      variant = value;
    } else if (key == "CPU part" && part.empty()) {
      part = value;
    }
  }
  if (!model.empty()) {
    return model;
  }
  if (!part.empty()) {
    return "arm implementer " + implementer + " variant " + variant + " part " + part;
  }
  return "unknown";
}

std::string cpu_model() {
#if defined(__APPLE__)
  constexpr std::array keys{
      sysctl_key{.name = "machdep.cpu.brand_string", .kind = sysctl_kind::text}};
  const sysctl_values values = read_sysctl(keys);
  const auto it = values.find("machdep.cpu.brand_string");
  return it == values.end() ? std::string("unknown") : one_line(it->second);
#elif defined(_WIN32)
  std::array<char, 256> name{};
  DWORD size = static_cast<DWORD>(name.size());
  if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                   "ProcessorNameString", RRF_RT_REG_SZ, nullptr, name.data(),
                   &size) != ERROR_SUCCESS) {
    return "unknown";
  }
  return one_line(std::string_view(name.data()));
#else
  std::ifstream in("/proc/cpuinfo");
  if (!in) {
    return "unknown";
  }
  std::ostringstream text;
  text << in.rdbuf();
  return cpu_model_from_cpuinfo(text.str());
#endif
}

} // namespace fftn::inline v0::detail
