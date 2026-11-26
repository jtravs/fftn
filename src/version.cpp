// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "fftn/config.hpp"
#include "fftn/fftn.hpp"

#include <array>
#include <cstddef>

namespace fftn::inline v0 {
namespace {

// "major.minor.patch" built from the version macros at compile time, so the version is
// stated once (include/fftn/version.h).
constexpr std::array<char, 32> version_text = [] {
  std::array<char, 32> text{};
  std::size_t at = 0;
  const auto put = [&](int value) {
    std::array<char, 10> digits{};
    std::size_t count = 0;
    do {
      digits.at(count++) = static_cast<char>('0' + value % 10);
      value /= 10;
    } while (value > 0);
    for (; count > 0; --count) {
      text.at(at++) = digits.at(count - 1);
    }
  };
  put(FFTN_VERSION_MAJOR);
  text.at(at++) = '.';
  put(FFTN_VERSION_MINOR);
  text.at(at++) = '.';
  put(FFTN_VERSION_PATCH);
  return text;
}();

} // namespace

const char* version() noexcept { return version_text.data(); }

} // namespace fftn::inline v0
