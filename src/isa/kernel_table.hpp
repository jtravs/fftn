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

// The entry points of one target for one precision T: twiddle packing (kernels.inl) and the
// passes (src/passes/).
template <class T> struct kernels_of {
  // Converts count interleaved twiddles into this target's twiddle storage, which holds
  // tw_reals reals per twiddle, C twiddles per register (load_tw). count % complex_lanes
  // == 0.
  void (*pack_twiddles)(const T* w, std::size_t count, T* out);
  // The same into broadcast storage, tw_reals reals per twiddle, each read into every lane
  // (bcast_tw). Any count.
  void (*pack_bcast_twiddles)(const T* w, std::size_t count, T* out);
  std::size_t complex_lanes; // C: complex numbers per complex register
  std::size_t real_lanes;    // W: reals per real register
  std::size_t tw_reals;      // reals stored per twiddle: 2, or 4 where pre-arranged (x86, neon)
  std::size_t split_block;   // copies from which lane passes may be split (vocabulary.inl)
  // The passes (passes/pass_args.hpp).
  pass_kernels<T> passes;
};

struct kernel_table {
  target id;
  bool fused; // fmadd and its relatives round once (FMA); otherwise multiply, then add
  kernels_of<double> f64;
  kernels_of<float> f32;
};

// The 128-bit leaves of the x86 targets (experiment X28, isa/narrow_avx2.cpp): the leaves
// of radices 2 to dense_codelet_radix and of the primes 17, 19 and 23 (X28b), the first
// narrow_leaf_slots rows of gen::radices (20's row empty: radix 20 is no leaf, X38), compiled with
// avx2's operations on 128-bit registers (vocab_avx2_narrow.inl), indexed by codelet_slot,
// direction and edges as radix_passes::leaf, null where that is. They round as avx2's and avx512's
// leaves do, so either target's plans may use them.
inline constexpr std::size_t narrow_leaf_slots = codelet_slot(23) + 1;
template <class T> struct narrow_leaf_set {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): filled in an ISA region, no std templates
  leaf_fn leaf[narrow_leaf_slots][2][edge_op_count][edge_op_count];
  // The same for lines of stride 1 (radix_passes::leaf_unit, experiment X28b).
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): as above
  leaf_fn unit[narrow_leaf_slots][2][2][2];
};
struct narrow_leaf_table {
  narrow_leaf_set<double> f64;
  narrow_leaf_set<float> f32;
};

template <class T>
[[nodiscard]] constexpr const narrow_leaf_set<T>& of(const narrow_leaf_table& table) {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
  if constexpr (std::is_same_v<T, double>) {
    return table.f64;
  } else {
    return table.f32;
  }
}

// The narrow leaves that plans of target t may use, or null (every target but avx2 and
// avx512).
[[nodiscard]] const narrow_leaf_table* find_narrow_leaves(target t) noexcept;

// The pair leaves of a target (experiment X28b, passes/pair_leaf.inl): the chain r1 x m of
// a row of n = r1 m points held in registers, as one leaf, for rows of stride 1 on both
// sides; per direction and edges, which are none or scale ([input][output], null where both
// would scale). Entries with n = 0 end the list; lengths ascend.
struct pair_leaf_entry {
  std::size_t n;
  std::size_t r1;
  std::size_t m;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): filled inside an ISA region
  leaf_fn leaf[2][2][2];
};
inline constexpr std::size_t pair_leaf_capacity = 16;
template <class T> struct pair_leaf_set {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): filled inside an ISA region
  pair_leaf_entry entry[pair_leaf_capacity];
  // The storage of the leaves' twiddle tables (kernels.inl): C of the leaves' vocabulary
  // and its packing, which differ from the plan's own target where a plan takes another
  // target's pair leaves (find_more_pair_leaves).
  std::size_t lanes;
  void (*pack_twiddles)(const T* w, std::size_t count, T* out);
};
struct pair_leaf_table {
  pair_leaf_set<double> f64;
  pair_leaf_set<float> f32;
};

template <class T>
[[nodiscard]] constexpr const pair_leaf_set<T>& of(const pair_leaf_table& table) {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
  if constexpr (std::is_same_v<T, double>) {
    return table.f64;
  } else {
    return table.f32;
  }
}

// The pair leaves of target t, or null (the scalar target).
[[nodiscard]] const pair_leaf_table* find_pair_leaves(target t) noexcept;

// The pair leaves that plans of target t take for lengths its own table lacks, or null:
// avx2's (C = 2) for avx512, whose C = 4 divides none of 18, 22, 26, 30 (experiment X37).
[[nodiscard]] const pair_leaf_table* find_more_pair_leaves(target t) noexcept;

// The batch leaves of a target (experiment X53, passes/batch_leaf.inl): the leaf of each
// codelet radix over `lanes` rows of a batch at once, one row per lane, for rows of stride 1
// on both sides; per codelet_slot, direction and edges, which are none or scale
// ([input][output], null where both would scale, and for the whole set where lanes is 1).
// A target's leaves are two units (isa/batch.inl): radices 32 and 64 in the second, so a set
// is null outside its unit's slots.
template <class T> struct batch_leaf_set {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): filled inside an ISA region
  leaf_fn leaf[gen::radix_count][2][2][2];
  std::size_t lanes; // rows per register: C of the leaves' vocabulary
};
struct batch_leaf_table {
  batch_leaf_set<double> f64;
  batch_leaf_set<float> f32;
};

template <class T>
[[nodiscard]] constexpr const batch_leaf_set<T>& of(const batch_leaf_table& table) {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
  if constexpr (std::is_same_v<T, double>) {
    return table.f64;
  } else {
    return table.f32;
  }
}

// The batch leaves of target t's own vocabulary for a codelet radix r (the table of the
// unit holding r), or null (the scalar target, and r without a codelet).
[[nodiscard]] const batch_leaf_table* find_batch_leaves(target t, std::size_t r) noexcept;

// The batch leaves of a narrower target that rounds as t does, for codelet radix r, or null:
// avx2's (C = 2 in double) for avx512, whose C = 4 idles half its lanes on a batch of two.
[[nodiscard]] const batch_leaf_table* find_narrower_batch_leaves(target t, std::size_t r) noexcept;

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
namespace x86_v2 {
extern const kernel_table table;
}
namespace avx {
extern const kernel_table table;
}
namespace avx2 {
extern const kernel_table table;
}
namespace avx512 {
extern const kernel_table table;
}
namespace avx2::narrow {
extern const narrow_leaf_table leaves;
}
namespace avx2::pairs {
extern const pair_leaf_table leaves;
}
namespace avx512::pairs {
extern const pair_leaf_table leaves;
}
namespace avx2::batch {
extern const batch_leaf_table leaves;
extern const batch_leaf_table large_leaves;
} // namespace avx2::batch
namespace avx512::batch {
extern const batch_leaf_table leaves;
extern const batch_leaf_table large_leaves;
} // namespace avx512::batch
#endif
#if FFTN_ARCH_AARCH64
namespace neon {
extern const kernel_table table;
}
namespace neon_fcma {
extern const kernel_table table;
}
namespace neon::pairs {
extern const pair_leaf_table leaves;
}
namespace neon_fcma::pairs {
extern const pair_leaf_table leaves;
}
namespace neon::batch {
extern const batch_leaf_table leaves;
extern const batch_leaf_table large_leaves;
} // namespace neon::batch
namespace neon_fcma::batch {
extern const batch_leaf_table leaves;
extern const batch_leaf_table large_leaves;
} // namespace neon_fcma::batch
#endif

} // namespace fftn::inline v0::detail
