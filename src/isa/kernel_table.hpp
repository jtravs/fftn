// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The per-target function tables. Each target's unit fills one (table.inl) with the
// addresses of its entry points, which take only pointers and integers. Baseline code only
// stores and calls these pointers; it never calls a vocabulary function. A plan looks its
// target's table up once, at construction (DESIGN.md 6.1), and keeps the pointers.
#pragma once

#include "isa/arch.hpp"
#include "isa/target.hpp"
#include "passes/pass_args.hpp"

#include <cstddef>
#include <type_traits>

namespace fftn::inline v0::detail {

// The entry points of one target for one precision T. Kernel semantics are in kernels.inl.
template <class T> struct kernels_of {
  // Radix-4 decimation-in-frequency pass, n = 4m (kernels.inl). m % complex_lanes == 0.
  void (*radix4_pass)(const T* x, T* y, const T* tw, std::size_t m, int sign);
  // Radix-2 decimation-in-time pass in split arithmetic, n = 2m. m % real_lanes == 0.
  void (*radix2_split_pass)(const T* x, T* y, const T* twr, const T* twi, std::size_t m, int sign);
  // Converts count interleaved twiddles into this target's twiddle storage, which holds
  // tw_reals reals per twiddle, C twiddles per register (load_tw). count % complex_lanes
  // == 0.
  void (*pack_twiddles)(const T* w, std::size_t count, T* out);
  // The same into broadcast storage, tw_reals reals per twiddle, each read into every lane
  // (bcast_tw). Any count.
  void (*pack_bcast_twiddles)(const T* w, std::size_t count, T* out);
  std::size_t complex_lanes; // C: complex numbers per complex register
  std::size_t real_lanes;    // W: reals per real register
  std::size_t tw_reals;      // reals stored per twiddle: 2, or 4 where pre-arranged (x86)
  // The passes (passes/pass_args.hpp).
  pass_kernels<T> passes;
};

struct kernel_table {
  target id;
  bool fused; // fmadd and its relatives round once (FMA); otherwise multiply, then add
  kernels_of<double> f64;
  kernels_of<float> f32;
};

// The entry points of a table for precision T.
template <class T> [[nodiscard]] constexpr const kernels_of<T>& of(const kernel_table& table) {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
  if constexpr (std::is_same_v<T, double>) {
    return table.f64;
  } else {
    return table.f32;
  }
}

// The table of target t, or null when this build does not contain t (compiled_in(t) is
// false).
[[nodiscard]] const kernel_table* find_kernels(target t) noexcept;

// One table per target compiled into this build, defined by the target's unit.
namespace scalar {
extern const kernel_table table;
}
#if FFTN_ARCH_X86_64
namespace avx2 {
extern const kernel_table table;
}
#endif
#if FFTN_ARCH_AARCH64
namespace neon_fcma {
extern const kernel_table table;
}
#endif

} // namespace fftn::inline v0::detail
