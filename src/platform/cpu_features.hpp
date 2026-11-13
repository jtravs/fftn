// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The instruction-set features of the running processor that FFTN's targets need. Filled
// once per process by detect_cpu_features(); read by target selection (task 0.3). A feature
// is set only when both the processor and the operating system support it (for AVX and
// AVX-512 that means XGETBV reports the register state as enabled, not only CPUID:
// DESIGN.md 14.2). The decoders behind it are in cpu_feature_sources.hpp; this header
// includes nothing, since target selection and the ISA units include it.
#pragma once

namespace fftn::inline v0::detail {

struct cpu_features {
  // x86-64. The x86_v2 target needs sse2 through sse42 and popcnt; avx2 needs avx, avx2
  // and fma; avx512 needs avx512f, avx512dq and avx512vl.
  bool sse2 = false;
  bool sse3 = false;
  bool ssse3 = false;
  bool sse41 = false;
  bool sse42 = false;
  bool popcnt = false;
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool avx512f = false;
  bool avx512dq = false;
  bool avx512bw = false;
  bool avx512vl = false;
  // AArch64. neon is Advanced SIMD (always present on AArch64); fcma is FCMLA/FCADD
  // (Armv8.3 complex arithmetic). sve_bits is the SVE vector length, 0 without SVE.
  bool neon = false;
  bool fcma = false;
  unsigned sve_bits = 0;

  bool operator==(const cpu_features&) const = default;
};

// Detects the features of the processor the calling thread runs on. Never fails: a
// feature that cannot be determined is reported absent. Detection runs once per process;
// later calls return the same value (a function-local constant, so no mutable state).
[[nodiscard]] cpu_features detect_cpu_features() noexcept;

} // namespace fftn::inline v0::detail
