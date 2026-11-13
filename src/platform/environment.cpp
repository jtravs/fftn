// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/environment.hpp"

#include <cstdlib>
#include <optional>
#include <string>

#ifdef _WIN32
#include <cstddef>
#include <memory>
#endif

namespace fftn::inline v0::detail {

std::optional<std::string> environment_variable(const char* name) {
#ifdef _WIN32
  char* value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) {
    return std::nullopt;
  }
  // _dupenv_s allocates the copy with malloc.
  const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return std::string(value);
#else
  // NOLINTNEXTLINE(concurrency-mt-unsafe): FFTN never sets variables; the value is copied at once
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::string(value);
#endif
}

} // namespace fftn::inline v0::detail
