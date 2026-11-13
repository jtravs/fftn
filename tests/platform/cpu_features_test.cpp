// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/cpu_feature_sources.hpp"
#include "platform/cpu_features.hpp"
#include "platform/sysctl_values.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using fftn::detail::cpu_features;
using fftn::detail::decode_linux_hwcap;
using fftn::detail::decode_macos_sysctl;
using fftn::detail::decode_x86_cpuid;
using fftn::detail::parse_sysctl_text;
using fftn::detail::x86_cpuid_words;

constexpr std::uint32_t bit(unsigned n) { return std::uint32_t{1} << n; }

// CPUID words of an AVX-512 server part (Skylake-SP class) with every state enabled.
x86_cpuid_words skylake_sp() {
  x86_cpuid_words w;
  w.max_leaf = 0x16;
  w.leaf1_edx = bit(26);
  w.leaf1_ecx = bit(0) | bit(9) | bit(12) | bit(19) | bit(20) | bit(23) | bit(27) | bit(28);
  w.leaf7_ebx = bit(5) | bit(16) | bit(17) | bit(30) | bit(31);
  w.xcr0 = 0xE7;
  return w;
}

TEST(CpuFeaturesX86, DecodesAFullAvx512Machine) {
  const cpu_features f = decode_x86_cpuid(skylake_sp());
  EXPECT_TRUE(f.sse2 && f.sse3 && f.ssse3 && f.sse41 && f.sse42 && f.popcnt);
  EXPECT_TRUE(f.avx && f.fma && f.avx2);
  EXPECT_TRUE(f.avx512f && f.avx512dq && f.avx512bw && f.avx512vl);
  EXPECT_FALSE(f.neon || f.fcma || f.pauth);
  EXPECT_EQ(f.sve_bits, 0U);
}

TEST(CpuFeaturesX86, NoLeavesNoFeatures) {
  x86_cpuid_words w = skylake_sp();
  w.max_leaf = 0;
  EXPECT_EQ(decode_x86_cpuid(w), cpu_features{});
}

TEST(CpuFeaturesX86, Leaf7IsIgnoredAboveMaxLeaf) {
  x86_cpuid_words w = skylake_sp();
  w.max_leaf = 6;
  const cpu_features f = decode_x86_cpuid(w);
  EXPECT_TRUE(f.avx && f.fma);
  EXPECT_FALSE(f.avx2 || f.avx512f);
}

TEST(CpuFeaturesX86, AvxNeedsTheOsToSaveYmm) {
  // Without OSXSAVE, XCR0 is not consulted at all.
  x86_cpuid_words w = skylake_sp();
  w.leaf1_ecx &= ~bit(27);
  cpu_features f = decode_x86_cpuid(w);
  EXPECT_TRUE(f.sse42 && f.popcnt);
  EXPECT_FALSE(f.avx || f.fma || f.avx2 || f.avx512f);
  // OSXSAVE but the YMM state off (XMM only).
  w = skylake_sp();
  w.xcr0 = 0x3;
  f = decode_x86_cpuid(w);
  EXPECT_FALSE(f.avx || f.fma || f.avx2 || f.avx512f || f.avx512vl);
}

TEST(CpuFeaturesX86, Avx512NeedsTheOsToSaveZmm) {
  // A kernel that saves YMM but not the opmask and ZMM state.
  x86_cpuid_words w = skylake_sp();
  w.xcr0 = 0x7;
  const cpu_features f = decode_x86_cpuid(w);
  EXPECT_TRUE(f.avx && f.avx2 && f.fma);
  EXPECT_FALSE(f.avx512f || f.avx512dq || f.avx512bw || f.avx512vl);
  // Missing any one of the three AVX-512 state components disables it.
  for (const std::uint64_t missing : {0x20U, 0x40U, 0x80U}) {
    w.xcr0 = 0xE7U & ~missing;
    EXPECT_FALSE(decode_x86_cpuid(w).avx512f) << missing;
  }
}

TEST(CpuFeaturesX86, Zen3HasAvx2ButNotAvx512) {
  x86_cpuid_words w = skylake_sp();
  w.leaf7_ebx = bit(5);
  w.xcr0 = 0x207; // x87, SSE, AVX and MPK, as Zen 3 reports
  const cpu_features f = decode_x86_cpuid(w);
  EXPECT_TRUE(f.avx2 && f.fma);
  EXPECT_FALSE(f.avx512f);
}

TEST(CpuFeaturesX86, SandyBridgeHasAvxOnly) {
  x86_cpuid_words w = skylake_sp();
  w.leaf1_ecx &= ~bit(12);
  w.leaf7_ebx = 0;
  w.xcr0 = 0x7;
  const cpu_features f = decode_x86_cpuid(w);
  EXPECT_TRUE(f.avx);
  EXPECT_FALSE(f.fma || f.avx2);
}

TEST(CpuFeaturesLinuxArm, DecodesHwcap) {
  constexpr std::uint64_t asimd = 1U << 1U;
  constexpr std::uint64_t fcma = 1U << 14U;
  constexpr std::uint64_t sve = 1U << 22U;
  constexpr std::uint64_t paca = 1U << 30U;
  cpu_features f = decode_linux_hwcap(asimd | fcma | sve | paca, 32);
  EXPECT_TRUE(f.neon && f.fcma && f.pauth);
  EXPECT_EQ(f.sve_bits, 256U);      // Graviton 3
  f = decode_linux_hwcap(asimd, 0); // Neoverse N1
  EXPECT_TRUE(f.neon);
  EXPECT_FALSE(f.fcma || f.pauth);
  EXPECT_EQ(f.sve_bits, 0U);
  f = decode_linux_hwcap(asimd | fcma | sve, 64); // A64FX: FCMA without PAuth
  EXPECT_TRUE(f.neon && f.fcma);
  EXPECT_FALSE(f.pauth);
  EXPECT_EQ(f.sve_bits, 512U);
  // A vector length is ignored without the SVE hwcap, and FCMA without ASIMD.
  f = decode_linux_hwcap(fcma, 16);
  EXPECT_FALSE(f.neon || f.fcma);
  EXPECT_EQ(f.sve_bits, 0U);
  EXPECT_FALSE(f.avx || f.sse2);
}

TEST(CpuFeaturesMacos, DecodesSysctl) {
  EXPECT_EQ(decode_macos_sysctl(parse_sysctl_text("")), cpu_features{});
  cpu_features f = decode_macos_sysctl(parse_sysctl_text(
      "hw.optional.arm.AdvSIMD: 1\nhw.optional.arm.FEAT_FCMA: 1\nhw.optional.arm.FEAT_PAuth: 1\n"));
  EXPECT_TRUE(f.neon && f.fcma && f.pauth);
  // macOS 11 names; it has none for pointer authentication.
  f = decode_macos_sysctl(
      parse_sysctl_text("hw.optional.neon: 1\nhw.optional.armv8_3_compnum: 1\n"));
  EXPECT_TRUE(f.neon && f.fcma);
  EXPECT_FALSE(f.pauth);
  f = decode_macos_sysctl(
      parse_sysctl_text("hw.optional.AdvSIMD: 1\nhw.optional.arm.FEAT_FCMA: 0\n"));
  EXPECT_TRUE(f.neon);
  EXPECT_FALSE(f.fcma);
}

TEST(CpuFeaturesMacos, DecodesTheRecordedM5Max) {
  const auto values = parse_sysctl_text(R"(hw.optional.arm.AdvSIMD: 1
hw.optional.arm.FEAT_FCMA: 1
hw.optional.arm.FEAT_PAuth: 1
hw.optional.arm.FEAT_SME: 1
hw.optional.arm.caps: -4612108402691149825
)");
  const cpu_features f = decode_macos_sysctl(values);
  EXPECT_TRUE(f.neon && f.fcma && f.pauth);
  EXPECT_EQ(f.sve_bits, 0U);
}

// The running machine's features obey the architectural implications; on the machines
// FFTN is developed on, the expected target features are present.
TEST(CpuFeaturesLive, SelfConsistent) {
  const cpu_features f = fftn::detail::detect_cpu_features();
  EXPECT_EQ(f, fftn::detail::detect_cpu_features()); // detected once, stable
  if (f.avx512vl || f.avx512dq || f.avx512bw) {
    EXPECT_TRUE(f.avx512f);
  }
  if (f.avx512f) {
    EXPECT_TRUE(f.avx2);
  }
  if (f.avx2 || f.fma) {
    EXPECT_TRUE(f.avx);
  }
  if (f.avx) {
    EXPECT_TRUE(f.sse42);
  }
  if (f.sse42) {
    EXPECT_TRUE(f.sse41 && f.ssse3 && f.sse3 && f.sse2);
  }
  if (f.fcma || f.pauth || f.sve_bits != 0) {
    EXPECT_TRUE(f.neon);
  }
  EXPECT_EQ(f.sve_bits % 128, 0U);
#if defined(__x86_64__) || defined(_M_X64)
  EXPECT_TRUE(f.sse2); // part of x86-64
  EXPECT_FALSE(f.neon);
#elif defined(__aarch64__) || defined(_M_ARM64)
  EXPECT_TRUE(f.neon); // part of AArch64
  EXPECT_FALSE(f.sse2);
#ifdef __APPLE__
  EXPECT_TRUE(f.fcma);  // every Apple M-series core has FEAT_FCMA
  EXPECT_TRUE(f.pauth); // and FEAT_PAuth (macOS 12 and later name it)
#endif
#endif
}

} // namespace
