// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "fftn/config.hpp"
#include "fftn/fftn.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(Version, MatchesHeaderMacros) {
  const std::string expected = std::to_string(FFTN_VERSION_MAJOR) + "." +
                               std::to_string(FFTN_VERSION_MINOR) + "." +
                               std::to_string(FFTN_VERSION_PATCH);
  ASSERT_NE(fftn::version(), nullptr);
  EXPECT_EQ(std::string(fftn::version()), expected);
}
