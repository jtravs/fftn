// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The two precisions every forced path runs in (AGENTS.md section 5, "Both precisions"), as
// GoogleTest types: a typed suite over them shows each test once per precision in the
// listing, as Suite/Double.Name and Suite/Float.Name (task 6.11).

#pragma once

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

namespace fftn::test {

using precisions = ::testing::Types<double, float>;

struct precision_names {
  template <class T> static std::string GetName(int /*index*/) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    return std::is_same_v<T, double> ? "Double" : "Float";
  }
};

} // namespace fftn::test
