// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary probes: per-target entry points that apply one vocabulary operation to arrays,
// so tests can check every operation of every target against a scalar reference. The
// probe text (probe.inl) is written once and included in each target's region, like the
// library's kernels; each region exports a table of pointer-and-integer entry points.
#pragma once

#include "isa/arch.hpp"
#include "isa/target.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace fftn::inline v0::detail {

// Operations, by family. Complex operations work on interleaved arrays of complex numbers,
// real operations on arrays of reals (vocabulary.inl gives the meanings).
enum class vop : unsigned char {
  c_copy, // load, then store
  c_add,
  c_sub,
  c_addj,
  c_subj,
  c_mul, // the twiddle operand comes from twiddle storage, through load_tw
  c_mulc,
  c_madd,
  c_scale,
  c_fmadd_r,
  c_neg,
  c_conj,
  c_swap_ri,
  c_bcast,
  c_transpose,
  c_cmul,      // a * b, both complex registers
  c_mul_bcast, // register g of a times twiddle g from broadcast storage (bcast_tw)
  r_copy,
  r_add,
  r_sub,
  r_mul,
  r_fmadd,
  r_fmsub,
  r_fnmadd,
  r_neg,
  r_bcast,
  r_deinterleave,
  r_interleave,
};

inline constexpr std::array<std::string_view, 28> vop_names = {
    "c_copy",      "c_add",   "c_sub",       "c_addj", "c_subj",  "c_mul",          "c_mulc",
    "c_madd",      "c_scale", "c_fmadd_r",   "c_neg",  "c_conj",  "c_swap_ri",      "c_bcast",
    "c_transpose", "c_cmul",  "c_mul_bcast", "r_copy", "r_add",   "r_sub",          "r_mul",
    "r_fmadd",     "r_fmsub", "r_fnmadd",    "r_neg",  "r_bcast", "r_deinterleave", "r_interleave",
};

// Applies op to n reals of input (n/2 complex numbers for complex operations), writing
// out. Operands: a, b, c and the real *r, as each operation needs them; for c_mul, c_mulc
// and c_madd, b is twiddle storage (pack_twiddles of the same target); for c_mul_bcast it is
// broadcast storage (pack_bcast_twiddles), one twiddle per register of a. For c_bcast and
// r_bcast, output register g is the broadcast of input number g. For r_deinterleave, out
// receives the n/2 real parts and then the n/2 imaginary parts; r_interleave takes them as
// a and b. n is a multiple of 2 C^2 and of 2 W.
template <class T>
using probe_fn = void (*)(vop op, const T* a, const T* b, const T* c, T* out, std::size_t n,
                          const T* r);

struct probe_table {
  target id;
  probe_fn<double> f64;
  probe_fn<float> f32;
};

// The probes of target t, or null when t is not compiled in.
[[nodiscard]] const probe_table* find_probes(target t) noexcept;

namespace scalar {
extern const probe_table probes;
}
#if FFTN_ARCH_X86_64
namespace avx2 {
extern const probe_table probes;
}
namespace avx512 {
extern const probe_table probes;
}
#endif
#if FFTN_ARCH_AARCH64
namespace neon {
extern const probe_table probes;
}
namespace neon_fcma {
extern const probe_table probes;
}
#endif

} // namespace fftn::inline v0::detail
