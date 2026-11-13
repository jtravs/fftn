// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Input preservation (DESIGN.md 5.1): a byte-for-byte copy of a buffer taken before a call
// and compared after it. Bytes, not values, so that a NaN rewritten in place or a negative
// zero turned positive is caught too.

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace fftn::test {

class input_snapshot {
public:
  explicit input_snapshot(std::span<const std::byte> region)
      : region_(region), copy_(region.begin(), region.end()) {}

  [[nodiscard]] bool unchanged() const { return std::ranges::equal(region_, copy_); }

  // The offset of the first changed byte, or the region's size if there is none.
  [[nodiscard]] std::size_t first_change() const {
    const auto [a, b] = std::ranges::mismatch(region_, copy_);
    return static_cast<std::size_t>(a - region_.begin());
  }

private:
  std::span<const std::byte> region_;
  std::vector<std::byte> copy_;
};

} // namespace fftn::test
