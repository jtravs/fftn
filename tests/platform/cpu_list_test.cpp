// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/cpu_list.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

namespace {

using fftn::detail::format_cpu_list;
using fftn::detail::parse_cpu_list;
using cpus = std::vector<unsigned>;

TEST(CpuList, ParsesRangesAndSingles) {
  EXPECT_EQ(parse_cpu_list("0-3,8,10-11"), (cpus{0, 1, 2, 3, 8, 10, 11}));
  EXPECT_EQ(parse_cpu_list("5"), (cpus{5}));
  EXPECT_EQ(parse_cpu_list(" 0-1\n"), (cpus{0, 1}));
}

TEST(CpuList, SortsAndDeduplicates) { EXPECT_EQ(parse_cpu_list("4,0-2,1"), (cpus{0, 1, 2, 4})); }

TEST(CpuList, EmptyIsTheEmptySet) {
  EXPECT_EQ(parse_cpu_list(""), cpus{});
  EXPECT_EQ(parse_cpu_list("\n"), cpus{});
}

TEST(CpuList, RejectsMalformedLists) {
  for (const char* bad : {"x", "1-", "-1", "3-1", "1,,2", "1-2-3", "0-99999999", "1 2"}) {
    EXPECT_EQ(parse_cpu_list(bad), std::nullopt) << bad;
  }
}

TEST(CpuList, FormatsRuns) {
  EXPECT_EQ(format_cpu_list({}), "");
  EXPECT_EQ(format_cpu_list({3}), "3");
  EXPECT_EQ(format_cpu_list({0, 1, 2, 3, 8, 10, 11}), "0-3,8,10-11");
}

TEST(CpuList, RoundTrips) {
  for (const char* text : {"0", "0-71", "0-17,36-53", "1,3,5,7", "0-1,4-5,8"}) {
    EXPECT_EQ(format_cpu_list(parse_cpu_list(text).value_or(cpus{})), std::string(text));
  }
}

} // namespace
