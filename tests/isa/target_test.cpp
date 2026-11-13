// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Target selection as a pure function of hand-made feature sets: which targets a processor
// runs, which one is best, forcing one, and names.

#include "isa/arch.hpp"
#include "isa/target.hpp"
#include "isa/test_targets.hpp"
#include "platform/cpu_features.hpp"

#include <gtest/gtest.h>

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
cpu_features haswell() {
  cpu_features f = x86_v2();
  f.avx = f.avx2 = f.fma = true;
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
  f.fcma = true;
  return f;
}

TEST(target, phase0_builds_scalar_and_the_simd_target_of_its_architecture) {
  EXPECT_TRUE(compiled_in(target::scalar));
  EXPECT_EQ(compiled_in(target::avx2), FFTN_ARCH_X86_64 != 0);
  EXPECT_EQ(compiled_in(target::neon_fcma), FFTN_ARCH_AARCH64 != 0);
  for (const target t : {target::x86_v2, target::avx, target::avx512, target::neon}) {
    EXPECT_FALSE(compiled_in(t)) << target_name(t);
  }
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
  cpu_features fcma_without_neon;
  fcma_without_neon.fcma = true;
  EXPECT_FALSE(runs_on(target::neon_fcma, fcma_without_neon));
}

TEST(target, best_is_the_fastest_compiled_target_the_processor_runs) {
  const target x86_simd = FFTN_ARCH_X86_64 != 0 ? target::avx2 : target::scalar;
  const target arm_simd = FFTN_ARCH_AARCH64 != 0 ? target::neon_fcma : target::scalar;
  EXPECT_EQ(best_target(cpu_features{}), target::scalar);
  EXPECT_EQ(best_target(x86_v2()), target::scalar); // x86_v2 is not compiled yet
  EXPECT_EQ(best_target(haswell()), x86_simd);
  EXPECT_EQ(best_target(skylake_sp()), x86_simd); // avx512 is not compiled yet
  EXPECT_EQ(best_target(neoverse_n1()), target::scalar);
  EXPECT_EQ(best_target(apple_m()), arm_simd);
}

TEST(target, choose_without_forcing_is_best) {
  EXPECT_EQ(choose_target(haswell(), std::nullopt), best_target(haswell()));
  EXPECT_EQ(choose_target(apple_m(), std::nullopt), best_target(apple_m()));
}

TEST(target, forcing_a_lower_target_is_honoured) {
  EXPECT_EQ(choose_target(haswell(), target::scalar), target::scalar);
  EXPECT_EQ(choose_target(apple_m(), target::scalar), target::scalar);
  const target simd = FFTN_ARCH_X86_64 != 0 ? target::avx2 : target::neon_fcma;
  const cpu_features has = FFTN_ARCH_X86_64 != 0 ? haswell() : apple_m();
  if (compiled_in(simd)) {
    EXPECT_EQ(choose_target(has, simd), simd);
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
  const std::string absent = message_of_forcing(skylake_sp(), target::avx512);
  EXPECT_NE(absent.find("forced target"), std::string::npos) << absent;
  EXPECT_NE(absent.find("'avx512' is not compiled"), std::string::npos) << absent;

  const target simd = FFTN_ARCH_X86_64 != 0 ? target::avx2 : target::neon_fcma;
  if (compiled_in(simd)) {
    const cpu_features none;
    const std::string unable = message_of_forcing(none, simd);
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
