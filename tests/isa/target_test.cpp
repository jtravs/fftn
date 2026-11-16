// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Target selection as a pure function of hand-made feature sets and of the decoded features
// of recorded processors: which targets a processor runs, which one is best, forcing one,
// and names.

#include "isa/arch.hpp"
#include "isa/target.hpp"
#include "isa/test_targets.hpp"
#include "platform/cpu_feature_sources.hpp"
#include "platform/cpu_features.hpp"
#include "platform/sysctl_values.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using fftn::detail::best_target;
using fftn::detail::choose_target;
using fftn::detail::compiled_in;
using fftn::detail::cpu_features;
using fftn::detail::runs_on;
using fftn::detail::target;
using fftn::detail::target_from_name;
using fftn::detail::target_name;

cpu_features x86_v2() {
  cpu_features f;
  f.sse2 = f.sse3 = f.ssse3 = f.sse41 = f.sse42 = f.popcnt = true;
  return f;
}
// AVX without AVX2 or FMA.
cpu_features sandy_bridge() {
  cpu_features f = x86_v2();
  f.avx = true;
  return f;
}
cpu_features haswell() {
  cpu_features f = sandy_bridge();
  f.avx2 = f.fma = true;
  return f;
}
cpu_features skylake_sp() {
  cpu_features f = haswell();
  f.avx512f = f.avx512dq = f.avx512bw = f.avx512vl = true;
  return f;
}
cpu_features neoverse_n1() {
  cpu_features f;
  f.neon = true;
  return f;
}
cpu_features apple_m() {
  cpu_features f = neoverse_n1();
  f.fcma = f.pauth = true;
  return f;
}
// FCMA without pointer authentication (an Armv8.2 design with SVE).
cpu_features a64fx() {
  cpu_features f = neoverse_n1();
  f.fcma = true;
  f.sve_bits = 512;
  return f;
}

TEST(target, builds_scalar_and_the_simd_targets_of_its_architecture) {
  EXPECT_TRUE(compiled_in(target::scalar));
  for (const target t : {target::x86_v2, target::avx, target::avx2, target::avx512}) {
    EXPECT_EQ(compiled_in(t), FFTN_ARCH_X86_64 != 0) << target_name(t);
  }
  EXPECT_EQ(compiled_in(target::neon), FFTN_ARCH_AARCH64 != 0);
  EXPECT_EQ(compiled_in(target::neon_fcma), FFTN_ARCH_AARCH64 != 0);
}

TEST(target, runs_on_needs_every_feature_the_attribute_enables) {
  const cpu_features none;
  EXPECT_TRUE(runs_on(target::scalar, none));
  for (const target t : isa_test::all_targets) {
    if (t == target::scalar) {
      continue;
    }
    EXPECT_FALSE(runs_on(t, none)) << target_name(t);
  }
  EXPECT_TRUE(runs_on(target::x86_v2, x86_v2()));
  EXPECT_FALSE(runs_on(target::avx, x86_v2()));
  EXPECT_TRUE(runs_on(target::avx, sandy_bridge()));
  EXPECT_FALSE(runs_on(target::avx2, sandy_bridge()));
  EXPECT_TRUE(runs_on(target::avx, haswell()));
  EXPECT_TRUE(runs_on(target::avx2, haswell()));
  EXPECT_FALSE(runs_on(target::avx512, haswell()));
  EXPECT_TRUE(runs_on(target::avx512, skylake_sp()));
  EXPECT_TRUE(runs_on(target::neon, neoverse_n1()));
  EXPECT_FALSE(runs_on(target::neon_fcma, neoverse_n1()));
  EXPECT_TRUE(runs_on(target::neon_fcma, apple_m()));
  EXPECT_FALSE(runs_on(target::avx2, apple_m()));
  EXPECT_FALSE(runs_on(target::neon, haswell()));

  // Removing any one feature the avx2 attribute string enables disqualifies avx2.
  for (bool cpu_features::* const feature : {
           &cpu_features::sse2,
           &cpu_features::sse3,
           &cpu_features::ssse3,
           &cpu_features::sse41,
           &cpu_features::sse42,
           &cpu_features::avx,
           &cpu_features::avx2,
           &cpu_features::fma,
       }) {
    cpu_features f = haswell();
    f.*feature = false;
    EXPECT_FALSE(runs_on(target::avx2, f));
  }
  for (bool cpu_features::* const feature :
       {&cpu_features::avx512f, &cpu_features::avx512dq, &cpu_features::avx512vl}) {
    cpu_features f = skylake_sp();
    f.*feature = false;
    EXPECT_FALSE(runs_on(target::avx512, f));
  }
  // Likewise for x86_v2's string (SSE4.2 and POPCNT, with what SSE4.2 implies) and avx's,
  // which adds AVX.
  for (bool cpu_features::* const feature : {
           &cpu_features::sse2,
           &cpu_features::sse3,
           &cpu_features::ssse3,
           &cpu_features::sse41,
           &cpu_features::sse42,
           &cpu_features::popcnt,
       }) {
    cpu_features f = x86_v2();
    f.*feature = false;
    EXPECT_FALSE(runs_on(target::x86_v2, f));
    cpu_features g = sandy_bridge();
    g.*feature = false;
    EXPECT_FALSE(runs_on(target::avx, g));
  }
  cpu_features without_neon = apple_m();
  without_neon.neon = false;
  EXPECT_FALSE(runs_on(target::neon_fcma, without_neon));
  // neon_fcma code may contain pointer-authentication instructions (branch protection).
  EXPECT_FALSE(runs_on(target::neon_fcma, a64fx()));
  EXPECT_TRUE(runs_on(target::neon, a64fx()));
}

TEST(target, best_is_the_fastest_compiled_target_the_processor_runs) {
  const bool x86 = FFTN_ARCH_X86_64 != 0;
  const bool arm = FFTN_ARCH_AARCH64 != 0;
  EXPECT_EQ(best_target(cpu_features{}), target::scalar);
  EXPECT_EQ(best_target(x86_v2()), x86 ? target::x86_v2 : target::scalar);
  EXPECT_EQ(best_target(sandy_bridge()), x86 ? target::avx : target::scalar);
  EXPECT_EQ(best_target(haswell()), x86 ? target::avx2 : target::scalar);
  EXPECT_EQ(best_target(skylake_sp()), x86 ? target::avx512 : target::scalar);
  EXPECT_EQ(best_target(neoverse_n1()), arm ? target::neon : target::scalar);
  EXPECT_EQ(best_target(apple_m()), arm ? target::neon_fcma : target::scalar);
  EXPECT_EQ(best_target(a64fx()), arm ? target::neon : target::scalar);
  // Pointer authentication without FCMA (Armv8.3 makes both mandatory; a core reporting only
  // one gets neon either way).
  cpu_features pauth_only = neoverse_n1();
  pauth_only.pauth = true;
  EXPECT_EQ(best_target(pauth_only), arm ? target::neon : target::scalar);
}

constexpr std::uint32_t bit(unsigned n) { return std::uint32_t{1} << n; }

// CPUID words as the recorded x86 machines report them (tests/platform/data has their
// topology): leaf 1 ECX with SSE3, SSSE3, FMA, SSE4.1, SSE4.2, POPCNT, OSXSAVE and AVX, leaf
// 1 EDX with SSE2, and the operating system's XCR0.
fftn::detail::x86_cpuid_words recorded_x86(std::uint32_t leaf7_ebx, std::uint64_t xcr0) {
  fftn::detail::x86_cpuid_words w;
  w.max_leaf = 0x16;
  w.leaf1_edx = bit(26);
  w.leaf1_ecx = bit(0) | bit(9) | bit(12) | bit(19) | bit(20) | bit(23) | bit(27) | bit(28);
  w.leaf7_ebx = leaf7_ebx;
  w.xcr0 = xcr0;
  return w;
}

TEST(target, recorded_machines_get_their_target) {
  using fftn::detail::decode_linux_hwcap;
  using fftn::detail::decode_macos_sysctl;
  using fftn::detail::decode_windows_arm64;
  using fftn::detail::decode_x86_cpuid;
  const bool x86 = FFTN_ARCH_X86_64 != 0;
  const bool arm = FFTN_ARCH_AARCH64 != 0;

  // Xeon Gold 6240 (Cascade Lake): AVX2, AVX-512 F, DQ, CD, BW and VL; XCR0 with the opmask,
  // ZMM and PKRU state.
  const cpu_features xeon = decode_x86_cpuid(
      recorded_x86(bit(5) | bit(16) | bit(17) | bit(28) | bit(30) | bit(31), 0x2E7));
  EXPECT_EQ(best_target(xeon), x86 ? target::avx512 : target::scalar);
  // The same processor under an operating system that does not save the ZMM state.
  const cpu_features xeon_without_zmm =
      decode_x86_cpuid(recorded_x86(bit(5) | bit(16) | bit(17) | bit(28) | bit(30) | bit(31), 0x7));
  EXPECT_EQ(best_target(xeon_without_zmm), x86 ? target::avx2 : target::scalar);
  // EPYC 7763 (Zen 3): AVX2 and FMA, no AVX-512.
  const cpu_features epyc = decode_x86_cpuid(recorded_x86(bit(5), 0x207));
  EXPECT_EQ(best_target(epyc), x86 ? target::avx2 : target::scalar);

  // Linux AArch64: Neoverse N1 (Graviton 2) reports ASIMD alone; A64FX FCMA and SVE without
  // pointer authentication; Graviton 3 (Neoverse V1) all three.
  constexpr std::uint64_t asimd = 1U << 1U;
  constexpr std::uint64_t fcma = 1U << 14U;
  constexpr std::uint64_t sve = 1U << 22U;
  constexpr std::uint64_t paca = 1U << 30U;
  EXPECT_EQ(best_target(decode_linux_hwcap(asimd, 0)), arm ? target::neon : target::scalar);
  EXPECT_EQ(best_target(decode_linux_hwcap(asimd | fcma | sve, 64)),
            arm ? target::neon : target::scalar);
  EXPECT_EQ(best_target(decode_linux_hwcap(asimd | fcma | sve | paca, 32)),
            arm ? target::neon_fcma : target::scalar);
  // macOS: the recorded M5 Max; a system too old to name pointer authentication.
  EXPECT_EQ(best_target(decode_macos_sysctl(fftn::detail::parse_sysctl_text(
                "hw.optional.arm.AdvSIMD: 1\nhw.optional.arm.FEAT_FCMA: 1\n"
                "hw.optional.arm.FEAT_PAuth: 1\n"))),
            arm ? target::neon_fcma : target::scalar);
  EXPECT_EQ(best_target(decode_macos_sysctl(fftn::detail::parse_sysctl_text(
                "hw.optional.neon: 1\nhw.optional.armv8_3_compnum: 1\n"))),
            arm ? target::neon : target::scalar);
  // Windows on Arm: the recorded Cobalt 100, and a system whose registry lacks the copies of
  // the ID registers (only NEON is known: neon, not scalar).
  EXPECT_EQ(best_target(decode_windows_arm64(true, 0x0010111101211052U, std::nullopt)),
            arm ? target::neon_fcma : target::scalar);
  EXPECT_EQ(best_target(decode_windows_arm64(true, std::nullopt, std::nullopt)),
            arm ? target::neon : target::scalar);
}

TEST(target, choose_without_forcing_is_best) {
  EXPECT_EQ(choose_target(haswell(), std::nullopt), best_target(haswell()));
  EXPECT_EQ(choose_target(apple_m(), std::nullopt), best_target(apple_m()));
}

TEST(target, forcing_a_lower_target_is_honoured) {
  EXPECT_EQ(choose_target(haswell(), target::scalar), target::scalar);
  EXPECT_EQ(choose_target(apple_m(), target::scalar), target::scalar);
  if (FFTN_ARCH_X86_64 != 0) {
    EXPECT_EQ(choose_target(skylake_sp(), target::avx2), target::avx2);
    EXPECT_EQ(choose_target(skylake_sp(), target::avx512), target::avx512);
  }
  if (FFTN_ARCH_AARCH64 != 0) {
    EXPECT_EQ(choose_target(apple_m(), target::neon), target::neon);
    EXPECT_EQ(choose_target(apple_m(), target::neon_fcma), target::neon_fcma);
  }
}

std::string message_of_forcing(const cpu_features& f, target t) {
  try {
    static_cast<void>(choose_target(f, t));
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

TEST(target, forcing_an_absent_target_is_refused_by_name) {
  // The other architecture's SIMD target is never compiled in.
  const target foreign = FFTN_ARCH_X86_64 != 0 ? target::neon : target::avx512;
  const std::string absent = message_of_forcing(skylake_sp(), foreign);
  EXPECT_NE(absent.find("forced target"), std::string::npos) << absent;
  EXPECT_NE(absent.find("'" + std::string(target_name(foreign)) + "' is not compiled"),
            std::string::npos)
      << absent;

  // A compiled target the processor cannot run: avx512 on Haswell, neon_fcma on A64FX.
  const target simd = FFTN_ARCH_X86_64 != 0 ? target::avx512 : target::neon_fcma;
  const cpu_features lacking = FFTN_ARCH_X86_64 != 0 ? haswell() : a64fx();
  if (compiled_in(simd)) {
    const std::string unable = message_of_forcing(lacking, simd);
    EXPECT_NE(unable.find("forced target"), std::string::npos) << unable;
    EXPECT_NE(unable.find("cannot run '" + std::string(target_name(simd)) + "'"), std::string::npos)
        << unable;
  }
}

TEST(target, names_round_trip) {
  for (const target t : isa_test::all_targets) {
    EXPECT_EQ(target_from_name(target_name(t)), t) << target_name(t);
  }
  EXPECT_EQ(target_name(target::neon_fcma), "neon_fcma");
  EXPECT_EQ(target_from_name("sse2"), std::nullopt);
  EXPECT_EQ(target_from_name(""), std::nullopt);
}

} // namespace
