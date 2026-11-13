// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Target selection. Baseline code: it only reads features and hands out table addresses.

#include "isa/target.hpp"

#include "isa/arch.hpp"
#include "isa/kernel_table.hpp"
#include "platform/cpu_features.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fftn::inline v0::detail {
namespace {

// Indexed by the enumerator's value, in declaration order.
constexpr std::array all_targets = {
    target::scalar, target::x86_v2, target::avx,       target::avx2,
    target::avx512, target::neon,   target::neon_fcma,
};
constexpr std::array<std::string_view, all_targets.size()> names = {
    "scalar", "x86_v2", "avx", "avx2", "avx512", "neon", "neon_fcma",
};

// Fastest first. best_target takes the first that is compiled in and runs; scalar, which
// runs anywhere, is the fallback.
constexpr std::array preference = {
    target::avx512, target::avx2, target::avx, target::x86_v2, target::neon_fcma, target::neon,
};

// The table of each target this build contains; null for the others.
#if FFTN_ARCH_X86_64
constexpr const kernel_table* avx2_kernels = &avx2::table;
#else
constexpr const kernel_table* avx2_kernels = nullptr;
#endif
#if FFTN_ARCH_AARCH64
constexpr const kernel_table* neon_fcma_kernels = &neon_fcma::table;
#else
constexpr const kernel_table* neon_fcma_kernels = nullptr;
#endif

bool has_x86_v2(const cpu_features& f) noexcept {
  return f.sse2 && f.sse3 && f.ssse3 && f.sse41 && f.sse42 && f.popcnt;
}
// What "avx2,fma" enables: AVX2 and FMA, with AVX and SSE4.2 and below implied.
bool has_avx2(const cpu_features& f) noexcept {
  return f.sse2 && f.sse3 && f.ssse3 && f.sse41 && f.sse42 && f.avx && f.avx2 && f.fma;
}

} // namespace

bool compiled_in(target t) noexcept {
  switch (t) {
  case target::scalar:
    return true;
  case target::avx2:
    return FFTN_ARCH_X86_64 != 0;
  case target::neon_fcma:
    return FFTN_ARCH_AARCH64 != 0;
  case target::x86_v2: // phase 1.3
  case target::avx:
  case target::avx512:
  case target::neon:
    break;
  }
  return false;
}

bool runs_on(target t, const cpu_features& f) noexcept {
  switch (t) {
  case target::scalar:
    break;
  case target::x86_v2:
    return has_x86_v2(f);
  case target::avx:
    return has_x86_v2(f) && f.avx;
  case target::avx2:
    return has_avx2(f);
  case target::avx512:
    return has_avx2(f) && f.avx512f && f.avx512dq && f.avx512vl;
  case target::neon:
    return f.neon;
  case target::neon_fcma:
    // Pointer authentication too: branch protection may sign this target's returns with
    // instructions outside the hint space (DESIGN.md 6.4). Both are mandatory from Armv8.3.
    return f.neon && f.fcma && f.pauth;
  }
  return true; // scalar runs anywhere
}

target best_target(const cpu_features& features) noexcept {
  for (const target t : preference) {
    if (compiled_in(t) && runs_on(t, features)) {
      return t;
    }
  }
  return target::scalar;
}

target choose_target(const cpu_features& features, std::optional<target> forced) {
  if (!forced) {
    return best_target(features);
  }
  const std::string name(target_name(*forced));
  if (!compiled_in(*forced)) {
    throw std::invalid_argument("forced target: '" + name + "' is not compiled into this build");
  }
  if (!runs_on(*forced, features)) {
    throw std::invalid_argument("forced target: this processor cannot run '" + name + "'");
  }
  return *forced;
}

std::string_view target_name(target t) noexcept { return names[static_cast<std::size_t>(t)]; }

std::optional<target> target_from_name(std::string_view name) noexcept {
  for (const target t : all_targets) {
    if (target_name(t) == name) {
      return t;
    }
  }
  return std::nullopt;
}

const kernel_table* find_kernels(target t) noexcept {
  switch (t) {
  case target::scalar:
    return &scalar::table;
  case target::avx2:
    return avx2_kernels;
  case target::neon_fcma:
    return neon_fcma_kernels;
  case target::x86_v2: // phase 1.3
  case target::avx:
  case target::avx512:
  case target::neon:
    break;
  }
  return nullptr;
}

} // namespace fftn::inline v0::detail
