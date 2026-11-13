// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/sysctl_values.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>

namespace {

using fftn::detail::parse_sysctl_text;
using fftn::detail::read_sysctl;
using fftn::detail::sysctl_integer;
using fftn::detail::sysctl_key;
using fftn::detail::sysctl_kind;

TEST(SysctlValues, ParsesSysctlOutput) {
  const auto values = parse_sysctl_text("hw.ncpu: 18\r\nhw.perflevel0.name: Super\n"
                                        "no separator here\n: no name\nhw.cacheconfig: 18 1 6 0");
  ASSERT_EQ(values.size(), 3U);
  EXPECT_EQ(values.at("hw.ncpu"), "18");
  EXPECT_EQ(values.at("hw.perflevel0.name"), "Super");
  EXPECT_EQ(values.at("hw.cacheconfig"), "18 1 6 0");
}

TEST(SysctlValues, IntegersAreStrict) {
  const auto values = parse_sysctl_text("a: 16384\nb: -1\nc: 12K\nd: \ne: Super\n");
  EXPECT_EQ(sysctl_integer(values, "a"), std::optional<std::uint64_t>(16384));
  for (const char* name : {"b", "c", "d", "e", "missing"}) {
    EXPECT_EQ(sysctl_integer(values, name), std::nullopt) << name;
  }
}

TEST(SysctlValues, ReadsTheRunningKernel) {
  static constexpr std::array<sysctl_key, 4> keys{
      {
          {.name = "hw.pagesize", .kind = sysctl_kind::integer},
          {.name = "hw.memsize", .kind = sysctl_kind::integer}, // 64-bit
          {.name = "hw.cacheconfig", .kind = sysctl_kind::integers},
          {.name = "fftn.no.such.key", .kind = sysctl_kind::text},
      },
  };
  const auto values = read_sysctl(keys);
#ifdef __APPLE__
  EXPECT_GE(sysctl_integer(values, "hw.pagesize").value_or(0), 4096U);
  EXPECT_GT(sysctl_integer(values, "hw.memsize").value_or(0), 1U << 30U);
  EXPECT_FALSE(values.at("hw.cacheconfig").empty());
  EXPECT_FALSE(values.contains("fftn.no.such.key"));
  static constexpr sysctl_key as_text{.name = "kern.ostype", .kind = sysctl_kind::text};
  EXPECT_EQ(read_sysctl({&as_text, 1}).at("kern.ostype"), "Darwin");
  // A string read as an integer has neither integer size and is left out.
  static constexpr sysctl_key as_integer{.name = "kern.ostype", .kind = sysctl_kind::integer};
  EXPECT_TRUE(read_sysctl({&as_integer, 1}).empty());
#else
  EXPECT_TRUE(values.empty());
#endif
}

} // namespace
