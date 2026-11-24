// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Bit positions are from the Intel SDM (CPUID leaves 1 and 7, XCR0), the Linux arm64 ELF
// hwcaps documentation (Documentation/arch/arm64/elf_hwcaps.rst) and the Arm Architecture
// Reference Manual (the ID_AA64ISAR1_EL1 and ID_AA64ISAR2_EL1 field descriptions).

#include "platform/cpu_features.hpp"

#include "platform/cpu_feature_sources.hpp"
#include "platform/sysctl_values.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

#if defined(__x86_64__) || defined(_M_X64)
#define FFTN_PLATFORM_X86_64 1
#include <array>
#ifdef _MSC_VER
#include <cstddef>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define FFTN_PLATFORM_AARCH64 1
#ifdef __linux__
#include <elf.h> // AT_HWCAP (sys/auxv.h includes it, but provides only getauxval)
#include <sys/auxv.h>
#include <sys/prctl.h>
#elif defined(__APPLE__)
#include <array>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#endif

namespace fftn::inline v0::detail {

namespace {

constexpr bool bit(std::uint64_t word, unsigned position) noexcept {
  return ((word >> position) & 1U) != 0;
}

bool sysctl_flag(const sysctl_values& values, std::string_view name) noexcept {
  const auto value = sysctl_integer(values, name);
  return value.has_value() && *value != 0;
}

// The vendor string's three words as CPUID returns them (EBX, EDX, ECX), little-endian.
constexpr std::uint32_t le32(const char* s) noexcept {
  return static_cast<std::uint32_t>(static_cast<unsigned char>(s[0])) |
         static_cast<std::uint32_t>(static_cast<unsigned char>(s[1])) << 8U |
         static_cast<std::uint32_t>(static_cast<unsigned char>(s[2])) << 16U |
         static_cast<std::uint32_t>(static_cast<unsigned char>(s[3])) << 24U;
}

constexpr bool vendor_is(const x86_cpuid_words& w, const char* twelve) noexcept {
  return w.vendor_ebx == le32(twelve) && w.vendor_edx == le32(twelve + 4) &&
         w.vendor_ecx == le32(twelve + 8);
}

} // namespace

const char* cpu_vendor_name(cpu_vendor v) noexcept {
  switch (v) {
  case cpu_vendor::unknown:
    break;
  case cpu_vendor::intel:
    return "intel";
  case cpu_vendor::amd:
    return "amd";
  case cpu_vendor::apple:
    return "apple";
  case cpu_vendor::arm_other:
    return "arm_other";
  }
  return "unknown";
}

cpu_vendor decode_x86_vendor(const x86_cpuid_words& words) noexcept {
  if (vendor_is(words, "GenuineIntel")) {
    return cpu_vendor::intel;
  }
  if (vendor_is(words, "AuthenticAMD")) {
    return cpu_vendor::amd;
  }
  return cpu_vendor::unknown;
}

cpu_features decode_x86_cpuid(const x86_cpuid_words& words) noexcept {
  cpu_features f;
  f.vendor = decode_x86_vendor(words);
  if (words.max_leaf < 1) {
    return f;
  }
  f.sse2 = bit(words.leaf1_edx, 26);
  f.sse3 = bit(words.leaf1_ecx, 0);
  f.ssse3 = bit(words.leaf1_ecx, 9);
  f.sse41 = bit(words.leaf1_ecx, 19);
  f.sse42 = bit(words.leaf1_ecx, 20);
  f.popcnt = bit(words.leaf1_ecx, 23);

  // AVX, FMA and AVX2 use the YMM state: the OS must save XMM (XCR0 bit 1) and the upper
  // YMM halves (bit 2). AVX-512 also needs the opmask and both ZMM halves (bits 5-7).
  const bool osxsave = bit(words.leaf1_ecx, 27);
  constexpr std::uint64_t ymm_state = 0x6;
  constexpr std::uint64_t zmm_state = 0xE6;
  const bool ymm = osxsave && (words.xcr0 & ymm_state) == ymm_state;
  const bool zmm = osxsave && (words.xcr0 & zmm_state) == zmm_state;
  f.avx = ymm && bit(words.leaf1_ecx, 28);
  f.fma = f.avx && bit(words.leaf1_ecx, 12);
  const std::uint32_t leaf7 = words.max_leaf >= 7 ? words.leaf7_ebx : 0;
  f.avx2 = f.avx && bit(leaf7, 5);
  f.avx512f = f.avx && zmm && bit(leaf7, 16);
  f.avx512dq = f.avx512f && bit(leaf7, 17);
  f.avx512bw = f.avx512f && bit(leaf7, 30);
  f.avx512vl = f.avx512f && bit(leaf7, 31);
  return f;
}

cpu_features decode_linux_hwcap(std::uint64_t hwcap, unsigned sve_vector_bytes) noexcept {
  // HWCAP_ASIMD, HWCAP_FCMA, HWCAP_SVE and HWCAP_PACA (address authentication); spelled
  // out because older C libraries lack the newer names.
  constexpr unsigned asimd = 1;
  constexpr unsigned fcma = 14;
  constexpr unsigned sve = 22;
  constexpr unsigned paca = 30;
  cpu_features f;
  f.vendor = cpu_vendor::arm_other;
  f.neon = bit(hwcap, asimd);
  f.fcma = f.neon && bit(hwcap, fcma);
  f.pauth = bit(hwcap, paca);
  f.sve_bits = bit(hwcap, sve) ? 8 * sve_vector_bytes : 0;
  return f;
}

cpu_features decode_macos_sysctl(const sysctl_values& values) noexcept {
  cpu_features f;
  f.vendor = cpu_vendor::apple; // AArch64 macOS runs only on Apple's processors
  // macOS 12 and later name the features hw.optional.arm.FEAT_*; macOS 11 had only the
  // older names, which current releases still provide.
  f.neon = sysctl_flag(values, "hw.optional.arm.AdvSIMD") ||
           sysctl_flag(values, "hw.optional.AdvSIMD") || sysctl_flag(values, "hw.optional.neon");
  f.fcma = f.neon && (sysctl_flag(values, "hw.optional.arm.FEAT_FCMA") ||
                      sysctl_flag(values, "hw.optional.armv8_3_compnum"));
  // macOS 11 has no name for pointer authentication, so it is reported absent there.
  f.pauth = sysctl_flag(values, "hw.optional.arm.FEAT_PAuth");
  // Apple silicon has SVE only in SME's streaming mode, which ordinary code never enters.
  return f;
}

cpu_features decode_windows_arm64(bool neon, std::optional<std::uint64_t> isar1,
                                  std::optional<std::uint64_t> isar2) noexcept {
  // ID register fields are four bits wide; zero means "not implemented". FCMA is
  // ISAR1[19:16]; address authentication is APA (QARMA5, ISAR1[7:4]), API (an
  // implementation-defined algorithm, ISAR1[11:8]) or APA3 (QARMA3, ISAR2[15:12]), any of
  // which provides the instructions neon_fcma code may contain.
  const auto field = [](std::optional<std::uint64_t> reg, unsigned lsb) {
    return reg.has_value() ? (*reg >> lsb) & 0xFU : 0U;
  };
  cpu_features f;
  f.vendor = cpu_vendor::arm_other;
  f.neon = neon;
  f.fcma = neon && field(isar1, 16) != 0;
  f.pauth = field(isar1, 4) != 0 || field(isar1, 8) != 0 || field(isar2, 12) != 0;
  return f;
}

namespace {

#ifdef FFTN_PLATFORM_X86_64

std::array<std::uint32_t, 4> cpuid(std::uint32_t leaf, std::uint32_t subleaf) noexcept {
  std::array<std::uint32_t, 4> r{};
#ifdef _MSC_VER
  std::array<int, 4> regs{};
  __cpuidex(regs.data(), static_cast<int>(leaf), static_cast<int>(subleaf));
  for (std::size_t i = 0; i < 4; ++i) {
    r[i] = static_cast<std::uint32_t>(regs[i]);
  }
#else
  __cpuid_count(leaf, subleaf, r[0], r[1], r[2], r[3]);
#endif
  return r;
}

// XGETBV through inline assembly: the _xgetbv intrinsic needs the xsave target feature on
// the translation unit, and FFTN sets no ISA flags (AGENTS.md section 4). The instruction
// is safe to execute here because the caller checked CPUID.1:ECX.OSXSAVE first.
std::uint64_t xgetbv0() noexcept {
  std::uint32_t eax = 0;
  std::uint32_t edx = 0;
  // NOLINTNEXTLINE(portability-no-assembler): no ISA flag may enable the intrinsic (above)
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0U));
  return (static_cast<std::uint64_t>(edx) << 32U) | eax;
}

cpu_features compute_features() noexcept {
  x86_cpuid_words words;
  const auto leaf0 = cpuid(0, 0);
  words.max_leaf = leaf0[0];
  words.vendor_ebx = leaf0[1];
  words.vendor_ecx = leaf0[2];
  words.vendor_edx = leaf0[3];
  if (words.max_leaf >= 1) {
    const auto leaf1 = cpuid(1, 0);
    words.leaf1_ecx = leaf1[2];
    words.leaf1_edx = leaf1[3];
  }
  if (words.max_leaf >= 7) {
    words.leaf7_ebx = cpuid(7, 0)[1];
  }
  if (bit(words.leaf1_ecx, 27)) {
    words.xcr0 = xgetbv0();
  }
#ifdef __APPLE__
  // macOS enables the AVX-512 state lazily, on a thread's first AVX-512 instruction, so
  // XCR0 does not show it yet; the kernel reports its support through sysctl instead.
  static constexpr std::array<sysctl_key, 1> avx512_key{{{.name = "hw.optional.avx512f"}}};
  if (sysctl_flag(read_sysctl(avx512_key), "hw.optional.avx512f")) {
    words.xcr0 |= 0xE0U;
  }
#endif
  return decode_x86_cpuid(words);
}

#elif defined(FFTN_PLATFORM_AARCH64) && defined(__APPLE__)

cpu_features compute_features() noexcept {
  static constexpr std::array<sysctl_key, 6> keys{
      {
          {.name = "hw.optional.arm.AdvSIMD"},
          {.name = "hw.optional.AdvSIMD"},
          {.name = "hw.optional.neon"},
          {.name = "hw.optional.arm.FEAT_FCMA"},
          {.name = "hw.optional.armv8_3_compnum"},
          {.name = "hw.optional.arm.FEAT_PAuth"},
      },
  };
  cpu_features f = decode_macos_sysctl(read_sysctl(keys));
  f.neon = true; // Advanced SIMD is part of the AArch64 base architecture.
  return f;
}

#elif defined(FFTN_PLATFORM_AARCH64) && defined(__linux__)

cpu_features compute_features() noexcept {
  // PR_SVE_GET_VL (Linux 4.15); the value's low 16 bits are the vector length in bytes.
  constexpr int pr_sve_get_vl = 51;
  constexpr unsigned sve_vl_mask = 0xFFFF;
  constexpr unsigned hwcap_sve = 1U << 22U;
  const std::uint64_t hwcap = getauxval(AT_HWCAP);
  unsigned sve_bytes = 0;
  if ((hwcap & hwcap_sve) != 0) {
    const int vl = prctl(pr_sve_get_vl, 0, 0, 0, 0);
    sve_bytes = vl > 0 ? static_cast<unsigned>(vl) & sve_vl_mask : 0;
  }
  return decode_linux_hwcap(hwcap, sve_bytes);
}

#elif defined(FFTN_PLATFORM_AARCH64) && defined(_WIN32)

// Processor 0's values stand for the machine, as on Linux, where the kernel reports only
// what every core has: the cores of one DynamIQ cluster implement the same architecture
// version, so they agree on FCMA and pointer authentication.
std::optional<std::uint64_t> registry_id_register(const wchar_t* name) noexcept {
  std::uint64_t value = 0;
  DWORD size = sizeof(value);
  if (RegGetValueW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", name,
                   RRF_RT_REG_QWORD, nullptr, &value, &size) != ERROR_SUCCESS) {
    return std::nullopt;
  }
  return value;
}

cpu_features compute_features() noexcept {
  return decode_windows_arm64(IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != 0,
                              registry_id_register(L"CP 4031"), registry_id_register(L"CP 4032"));
}

#elif defined(FFTN_PLATFORM_AARCH64)

cpu_features compute_features() noexcept {
  // Another AArch64 operating system: only the base architecture is assumed.
  cpu_features f;
  f.vendor = cpu_vendor::arm_other;
  f.neon = true;
  return f;
}

#else

cpu_features compute_features() noexcept { return {}; }

#endif

} // namespace

cpu_features detect_cpu_features() noexcept {
  static const cpu_features features = compute_features();
  return features;
}

} // namespace fftn::inline v0::detail
