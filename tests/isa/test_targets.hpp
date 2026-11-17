// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The targets the ISA tests force in turn: those compiled in that the machine running the
// tests can execute.
#pragma once

#include "isa/target.hpp"
#include "platform/cpu_features.hpp"

#include <array>
#include <string>
#include <vector>

namespace isa_test {

using fftn::detail::cpu_features;
using fftn::detail::target;

inline constexpr std::array all_targets = {
    target::scalar, target::x86_v2, target::avx,       target::avx2,
    target::avx512, target::neon,   target::neon_fcma,
};

// The features of the machine running the tests (detect_cpu_features()).
[[nodiscard]] cpu_features machine_features();

// Targets that are compiled in and run on machine_features(), less those FFTN_TEST_SKIP_ISAS
// names (support/skipped_targets.hpp). Emulators that report what
// they execute through CPUID are covered: Rosetta 2 with ROSETTA_ADVERTISE_AVX=1 in the
// environment reports AVX, AVX2 and FMA (CPUID and XGETBV), so the avx2 suites run on an
// Apple silicon Mac (checked with macOS 27 on an M5 Max); without it, SSE4.2 at most.
[[nodiscard]] std::vector<target> runnable_targets();

// "scalar", "avx2", ... for test names.
[[nodiscard]] std::string label(target t);

} // namespace isa_test
