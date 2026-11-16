// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The decoders behind detect_cpu_features(). Each source of raw facts (CPUID, Linux
// hwcaps, macOS sysctl, the Windows registry) has a pure decoder, so every decoding rule is tested
// on any machine from synthetic inputs; only the few lines that fetch the raw facts are specific to
// an architecture and operating system.
#pragma once

#include "platform/cpu_features.hpp"
#include "platform/sysctl_values.hpp"

#include <cstdint>
#include <optional>

namespace fftn::inline v0::detail {

// The CPUID and XGETBV words the x86 decoder reads. A leaf above max_leaf reads as zero.
struct x86_cpuid_words {
  std::uint32_t max_leaf = 0;  // CPUID.0:EAX
  std::uint32_t leaf1_ecx = 0; // CPUID.1:ECX
  std::uint32_t leaf1_edx = 0; // CPUID.1:EDX
  std::uint32_t leaf7_ebx = 0; // CPUID.(7,0):EBX
  std::uint64_t xcr0 = 0;      // XGETBV(0); meaningful only when CPUID.1:ECX.OSXSAVE is set
};

[[nodiscard]] cpu_features decode_x86_cpuid(const x86_cpuid_words& words) noexcept;

// AArch64 Linux: getauxval(AT_HWCAP) and the SVE vector length in bytes (prctl
// PR_SVE_GET_VL; 0 without SVE).
[[nodiscard]] cpu_features decode_linux_hwcap(std::uint64_t hwcap,
                                              unsigned sve_vector_bytes) noexcept;

// AArch64 macOS: the hw.optional.* keys.
[[nodiscard]] cpu_features decode_macos_sysctl(const sysctl_values& values) noexcept;

// AArch64 Windows: whether IsProcessorFeaturePresent reports NEON, and the copies of
// ID_AA64ISAR1_EL1 and ID_AA64ISAR2_EL1 that Windows keeps in the registry (values "CP 4031"
// and "CP 4032" of HARDWARE\DESCRIPTION\System\CentralProcessor\0; nothing when a value is
// absent). Windows has no processor-feature flag for FCMA or pointer authentication, and
// user code cannot read the ID registers itself: MRS faults (experiment B2).
[[nodiscard]] cpu_features decode_windows_arm64(bool neon, std::optional<std::uint64_t> isar1,
                                                std::optional<std::uint64_t> isar2) noexcept;

} // namespace fftn::inline v0::detail
