// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Entry points for the generated codelets (src/gen/), one per entry of gen::codelets, so
// that tests can run every codelet of every target before passes use them. The runner text
// (codelet_runner.inl) is written once and included in each target's region, like the
// library's kernels; each region exports a table of pointer-and-integer entry points.
#pragma once

#include "gen/codelets.hpp"
#include "isa/arch.hpp"
#include "isa/target.hpp"

#include <cstddef>

namespace fftn::inline v0::detail {

// Runs gen::codelets[i] (run) or gen::pruned_codelets[i] (pruned) on C transforms at once (C complex numbers per register, one
// transform per lane). Leg j of the input is the C complex numbers at x + 2 xs j, leg k of
// the output those at y + 2 ys k (xs, ys >= C, in complex numbers; x == y with xs == ys is
// in place). The twiddles of leg k >= 1 are C twiddles in the target's storage form
// (pack_twiddles of kernel_table) at w + (k - 1) C tw_reals; plain codelets ignore w. A
// pruned codelet reads (dif_head) or writes (plain_head) legs 0 .. R/2 - 1 only.
template <class T>
using codelet_fn = void (*)(const T* x, std::size_t xs, T* y, std::size_t ys, const T* w);

template <class T> struct codelet_runs {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): filled inside ISA regions, which use no std templates
  codelet_fn<T> run[gen::codelet_count];
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): as above
  codelet_fn<T> pruned[gen::pruned_codelet_count];
};

struct codelet_table {
  target id;
  codelet_runs<double> f64;
  codelet_runs<float> f32;
};

// The codelet entry points of target t, or null when t is not compiled in.
[[nodiscard]] const codelet_table* find_codelets(target t) noexcept;

namespace scalar {
extern const codelet_table codelet_entries;
}
#if FFTN_ARCH_X86_64
namespace x86_v2 {
extern const codelet_table codelet_entries;
}
namespace avx {
extern const codelet_table codelet_entries;
}
namespace avx2 {
extern const codelet_table codelet_entries;
}
namespace avx512 {
extern const codelet_table codelet_entries;
}
#endif
#if FFTN_ARCH_AARCH64
namespace neon {
extern const codelet_table codelet_entries;
}
namespace neon_fcma {
extern const codelet_table codelet_entries;
}
#endif

} // namespace fftn::inline v0::detail
