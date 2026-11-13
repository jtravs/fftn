// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/environment.hpp"

#include <gtest/gtest.h>

namespace {

using fftn::detail::environment_variable;

// PATH is set wherever the tests run (CTest, a shell, CI, on every OS), so the test needs
// no call that changes the environment. The FFTN_TOPOLOGY runs of fftn_platform_dump
// cover a variable set to a known value, and set but empty.
TEST(Environment, ReadsASetVariable) {
  EXPECT_FALSE(environment_variable("PATH").value_or("").empty());
}

TEST(Environment, UnsetIsNothing) {
  EXPECT_FALSE(environment_variable("FFTN_TEST_ENVIRONMENT_NEVER_SET").has_value());
}

} // namespace
