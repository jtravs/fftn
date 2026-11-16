// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The processor model wisdom records (src/platform/cpu_model.hpp): /proc/cpuinfo of an x86
// and an AArch64 Linux machine, and the live reading on this machine.

#include "platform/cpu_model.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

using fftn::detail::cpu_model;
using fftn::detail::cpu_model_from_cpuinfo;
using fftn::detail::one_line;

TEST(CpuModel, X86CpuinfoGivesTheFirstModelName) {
  const std::string text = "processor\t: 0\n"
                           "vendor_id\t: GenuineIntel\n"
                           "model name\t: Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz\n"
                           "\n"
                           "processor\t: 1\n"
                           "model name\t: Something else\n";
  EXPECT_EQ(cpu_model_from_cpuinfo(text), "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz");
}

TEST(CpuModel, AArch64CpuinfoGivesImplementerVariantAndPart) {
  const std::string text = "processor\t: 0\n"
                           "BogoMIPS\t: 2000.00\n"
                           "CPU implementer\t: 0x41\n"
                           "CPU architecture: 8\n"
                           "CPU variant\t: 0x1\n"
                           "CPU part\t: 0xd40\n"
                           "\n"
                           "processor\t: 1\n"
                           "CPU part\t: 0xd4f\n";
  EXPECT_EQ(cpu_model_from_cpuinfo(text), "arm implementer 0x41 variant 0x1 part 0xd40");
}

TEST(CpuModel, UnknownWhenNothingNamesIt) {
  EXPECT_EQ(cpu_model_from_cpuinfo(""), "unknown");
  EXPECT_EQ(cpu_model_from_cpuinfo("processor\t: 0\nflags\t: fpu\n"), "unknown");
}

TEST(CpuModel, OneLineCollapsesWhitespace) {
  EXPECT_EQ(one_line("  Apple\tM5   Max \n"), "Apple M5 Max");
  EXPECT_EQ(one_line(" \t\n"), "unknown");
}

TEST(CpuModel, ThisMachineHasOneLine) {
  const std::string model = cpu_model();
  ASSERT_FALSE(model.empty());
  EXPECT_EQ(model.find_first_of("\t\n\r"), std::string::npos);
  EXPECT_EQ(one_line(model), model);
}

} // namespace
