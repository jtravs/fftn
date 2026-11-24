// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/cpu_feature_sources.hpp"
#include "platform/cpu_features.hpp"
#include "platform/sysctl_values.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>

namespace {

using fftn::detail::cpu_features;
using fftn::detail::cpu_vendor;
using fftn::detail::decode_linux_hwcap;
using fftn::detail::decode_macos_sysctl;
using fftn::detail::decode_windows_arm64;
using fftn::detail::decode_x86_cpuid;
using fftn::detail::decode_x86_vendor;
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

// CPUID leaf 0's vendor string, in EBX, EDX, ECX (DESIGN.md 13.5): Intel's and AMD's; any other
// (Hygon's Zen-based parts, a hypervisor's, none) is unknown. The features do not depend on it.
x86_cpuid_words with_vendor(x86_cpuid_words w, std::uint32_t ebx, std::uint32_t edx,
                            std::uint32_t ecx) {
  w.vendor_ebx = ebx;
  w.vendor_edx = edx;
  w.vendor_ecx = ecx;
  return w;
}

TEST(CpuFeaturesX86, DecodesTheVendor) {
  // "Genu" "ineI" "ntel" and "Auth" "enti" "cAMD", little-endian, as CPUID returns them.
  const x86_cpuid_words intel = with_vendor(skylake_sp(), 0x756E6547, 0x49656E69, 0x6C65746E);
  const x86_cpuid_words amd = with_vendor(skylake_sp(), 0x68747541, 0x69746E65, 0x444D4163);
  const x86_cpuid_words hygon = with_vendor(skylake_sp(), 0x6F677948, 0x6E65476E, 0x656E6975);
  EXPECT_EQ(decode_x86_vendor(intel), cpu_vendor::intel);
  EXPECT_EQ(decode_x86_vendor(amd), cpu_vendor::amd);
  EXPECT_EQ(decode_x86_vendor(hygon), cpu_vendor::unknown);
  EXPECT_EQ(decode_x86_vendor(skylake_sp()), cpu_vendor::unknown);
  // EDX and ECX swapped ("GenuntelineI") is not Intel's string.
  EXPECT_EQ(decode_x86_vendor(with_vendor(skylake_sp(), 0x756E6547, 0x6C65746E, 0x49656E69)),
            cpu_vendor::unknown);
  cpu_features f = decode_x86_cpuid(amd);
  EXPECT_EQ(f.vendor, cpu_vendor::amd);
  f.vendor = cpu_vendor::unknown;
  EXPECT_EQ(f, decode_x86_cpuid(skylake_sp()));
  // The vendor without any feature leaf.
  x86_cpuid_words bare = intel;
  bare.max_leaf = 0;
  EXPECT_EQ(decode_x86_cpuid(bare), cpu_features{.vendor = cpu_vendor::intel});
}

TEST(CpuFeatures, VendorNames) {
  EXPECT_STREQ(fftn::detail::cpu_vendor_name(cpu_vendor::unknown), "unknown");
  EXPECT_STREQ(fftn::detail::cpu_vendor_name(cpu_vendor::intel), "intel");
  EXPECT_STREQ(fftn::detail::cpu_vendor_name(cpu_vendor::amd), "amd");
  EXPECT_STREQ(fftn::detail::cpu_vendor_name(cpu_vendor::apple), "apple");
  EXPECT_STREQ(fftn::detail::cpu_vendor_name(cpu_vendor::arm_other), "arm_other");
  EXPECT_STREQ(fftn::detail::cpu_vendor_name(static_cast<cpu_vendor>(200)), "unknown");
}

TEST(CpuFeaturesLinuxArm, DecodesHwcap) {
  constexpr std::uint64_t asimd = 1U << 1U;
  constexpr std::uint64_t fcma = 1U << 14U;
  constexpr std::uint64_t sve = 1U << 22U;
  constexpr std::uint64_t paca = 1U << 30U;
  cpu_features f = decode_linux_hwcap(asimd | fcma | sve | paca, 32);
  EXPECT_TRUE(f.neon && f.fcma && f.pauth);
  EXPECT_EQ(f.sve_bits, 256U); // Graviton 3
  EXPECT_EQ(f.vendor, cpu_vendor::arm_other);
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
  EXPECT_EQ(decode_macos_sysctl(parse_sysctl_text("")), cpu_features{.vendor = cpu_vendor::apple});
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
  EXPECT_EQ(f.vendor, cpu_vendor::apple);
}

TEST(CpuFeaturesWindowsArm, DecodesTheRecordedCobalt100) {
  // CP 4031 of the windows-11-arm runner (Azure Cobalt 100, Neoverse N2; experiment B2):
  // FCMA 1, APA 5 (QARMA5 with combined FPAC), API 0. Windows recorded no CP 4032.
  const cpu_features f = decode_windows_arm64(true, 0x0010111101211052U, std::nullopt);
  EXPECT_TRUE(f.neon && f.fcma && f.pauth);
  EXPECT_EQ(f.sve_bits, 0U);
  EXPECT_EQ(f.vendor, cpu_vendor::arm_other);
}

TEST(CpuFeaturesWindowsArm, DecodesEachField) {
  constexpr std::uint64_t fcma = std::uint64_t{1} << 16U;
  constexpr std::uint64_t apa = std::uint64_t{1} << 4U;
  constexpr std::uint64_t api = std::uint64_t{1} << 8U;
  constexpr std::uint64_t apa3 = std::uint64_t{1} << 12U; // in ID_AA64ISAR2_EL1
  // Missing registry values: only NEON, from IsProcessorFeaturePresent.
  cpu_features f = decode_windows_arm64(true, std::nullopt, std::nullopt);
  EXPECT_TRUE(f.neon);
  EXPECT_FALSE(f.fcma || f.pauth);
  f = decode_windows_arm64(true, fcma, std::nullopt); // FCMA alone (as A64FX)
  EXPECT_TRUE(f.fcma);
  EXPECT_FALSE(f.pauth);
  EXPECT_TRUE(decode_windows_arm64(true, api, std::nullopt).pauth);
  EXPECT_TRUE(decode_windows_arm64(true, apa, 0U).pauth);
  f = decode_windows_arm64(true, fcma, apa3); // QARMA3 is in the second register
  EXPECT_TRUE(f.fcma && f.pauth);
  // Neighbouring fields do not count: JSCVT (15:12), LRCPC (23:20), GPA (27:24), GPA3.
  f = decode_windows_arm64(true, 0x0FF0'F000U, std::uint64_t{0xF} << 8U);
  EXPECT_FALSE(f.fcma || f.pauth);
  // FCMA without NEON is not reported.
  EXPECT_FALSE(decode_windows_arm64(false, fcma, std::nullopt).fcma);
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
  EXPECT_NE(f.vendor, cpu_vendor::apple);
  EXPECT_NE(f.vendor, cpu_vendor::arm_other);
  EXPECT_TRUE(f.sse2); // part of x86-64
  EXPECT_FALSE(f.neon);
#elif defined(__aarch64__) || defined(_M_ARM64)
  EXPECT_TRUE(f.neon); // part of AArch64
  EXPECT_FALSE(f.sse2);
#ifdef __APPLE__
  EXPECT_EQ(f.vendor, cpu_vendor::apple);
  EXPECT_TRUE(f.fcma);  // every Apple M-series core has FEAT_FCMA
  EXPECT_TRUE(f.pauth); // and FEAT_PAuth (macOS 12 and later name it)
#endif
#endif
}

// The live vendor against the operating system's own reading (DESIGN.md 13.5): on Linux x86-64,
// /proc/cpuinfo's vendor_id (GenuineIntel: intel, AuthenticAMD: amd, any other: unknown). A
// mistake in CPUID's register order would make every machine unknown, silently changing plans.
TEST(CpuFeaturesLive, VendorMatchesTheOperatingSystem) {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__linux__)
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string vendor_id;
  for (std::string line; vendor_id.empty() && std::getline(cpuinfo, line);) {
    if (line.starts_with("vendor_id")) {
      const std::size_t colon = line.find(':');
      const std::size_t first =
          line.find_first_not_of(" \t", colon == std::string::npos ? line.size() : colon + 1);
      vendor_id = first == std::string::npos ? std::string("?") : line.substr(first);
      vendor_id.erase(vendor_id.find_last_not_of(" \t\r") + 1);
    }
  }
  if (vendor_id.empty()) {
    GTEST_SKIP() << "/proc/cpuinfo is unreadable or names no vendor_id";
  }
  const cpu_vendor expected = vendor_id == "GenuineIntel"   ? cpu_vendor::intel
                              : vendor_id == "AuthenticAMD" ? cpu_vendor::amd
                                                            : cpu_vendor::unknown;
  EXPECT_EQ(fftn::detail::detect_cpu_features().vendor, expected) << vendor_id;
#else
  GTEST_SKIP() << "Linux x86-64 only";
#endif
}

} // namespace
