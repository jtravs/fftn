// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary of the x86_v2 target: x86-64 with SSE2 to SSE4.2 and no AVX (the x86-64-v2
// level: Nehalem, Westmere, Atom-class cores, older AMD). 128-bit registers: one complex
// double or two complex floats per complex register, the shape of plain neon (DESIGN.md
// 6.1). No FMA: fmadd and its relatives multiply, then add. Included inside
// FFTN_ISA_BEGIN(x86_v2); the contract is in vocabulary.inl.
//
// Every operation rounds exactly as avx's does (vocab_avx.inl), so the two targets give the
// same bits with the same passes (DESIGN.md 6.2). Both round as the scalar target too: each
// product is rounded, then summed, and the sums pair the terms as scalar's do.

#include "isa/vocabulary.inl"

// NOLINTBEGIN(portability-simd-intrinsics): this file is where FFTN's x86 intrinsics live.
namespace {

inline constexpr target id = target::x86_v2;
inline constexpr bool fused = false;
// No masked loads before AVX (MASKMOVDQU only stores, and bypasses the caches): partial
// runs go through a tile (access.inl).
inline constexpr bool masked = false;

// Twiddles pre-arranged as on the other x86 targets (DESIGN.md 6.2): r holds (wr, wr) and
// i holds (-wi, wi) for each lane's twiddle, so a*w = r*a + i*swap_ri(a) with one shuffle
// and no sign fix-up. Special members trivial, defined inside the region.
struct tw_f64 {
  __m128d r;
  __m128d i;
};
struct tw_f32 {
  __m128 r;
  __m128 i;
};

template <> struct vec<double> {
  using cv = __m128d;
  using rv = __m128d;
  using tw = tw_f64;
  static constexpr std::size_t C = 1;
  static constexpr std::size_t W = 2;
  static constexpr std::size_t split_block = 2 * W; // avx's W: the pair rounds alike
  static constexpr std::size_t tw_reals = 4;
};

template <> struct vec<float> {
  using cv = __m128;
  using rv = __m128;
  using tw = tw_f32;
  static constexpr std::size_t C = 2;
  static constexpr std::size_t W = 4;
  static constexpr std::size_t split_block = 2 * W; // avx's W: the pair rounds alike
  static constexpr std::size_t tw_reals = 4;
};

// Complex-register operations.

FFTN_FORCE_INLINE __m128d load(const double* p) { return _mm_loadu_pd(p); }
FFTN_FORCE_INLINE __m128 load(const float* p) { return _mm_loadu_ps(p); }
FFTN_FORCE_INLINE void store(double* p, __m128d a) { _mm_storeu_pd(p, a); }
FFTN_FORCE_INLINE void store(float* p, __m128 a) { _mm_storeu_ps(p, a); }
// One complex double fills the register.
FFTN_FORCE_INLINE __m128d bcast(const double* p) { return _mm_loadu_pd(p); }
// A 64-bit load through an __m128i pointer, which the intrinsics' headers declare as
// may-alias and unaligned (a complex float is only 4-byte aligned), then duplicated.
FFTN_FORCE_INLINE __m128 bcast(const float* p) {
  const __m128 x = _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
  return _mm_movelh_ps(x, x);
}
FFTN_FORCE_INLINE tw_f64 load_tw(const double* p) {
  return {.r = _mm_loadu_pd(p), .i = _mm_loadu_pd(p + 2)};
}
FFTN_FORCE_INLINE tw_f32 load_tw(const float* p) {
  return {.r = _mm_loadu_ps(p), .i = _mm_loadu_ps(p + 4)};
}
// Storage of C twiddles: W reals (wr, wr) per lane, then W reals (-wi, wi) per lane.
template <class T> FFTN_FORCE_INLINE void pack_tw(T* out, const T* p) {
  for (std::size_t k = 0; k < C<T>; ++k) {
    out[2 * k] = p[2 * k];
    out[2 * k + 1] = p[2 * k];
    out[W<T> + 2 * k] = -p[2 * k + 1];
    out[W<T> + 2 * k + 1] = p[2 * k + 1];
  }
}
// Broadcast storage of one twiddle: (wr, wr, -wi, wi), one lane of the pre-arranged form.
// For double that is the whole twiddle; for float each half is a 64-bit load duplicated.
FFTN_FORCE_INLINE tw_f64 bcast_tw(const double* p) { return load_tw(p); }
FFTN_FORCE_INLINE tw_f32 bcast_tw(const float* p) { return {.r = bcast(p), .i = bcast(p + 2)}; }
template <class T> FFTN_FORCE_INLINE void pack_tw1(T* out, const T* p) {
  out[0] = p[0];
  out[1] = p[0];
  out[2] = -p[1];
  out[3] = p[1];
}

FFTN_FORCE_INLINE __m128d add(__m128d a, __m128d b) { return _mm_add_pd(a, b); }
FFTN_FORCE_INLINE __m128 add(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
FFTN_FORCE_INLINE __m128d sub(__m128d a, __m128d b) { return _mm_sub_pd(a, b); }
FFTN_FORCE_INLINE __m128 sub(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
FFTN_FORCE_INLINE __m128d swap_ri(__m128d a) { return _mm_shuffle_pd(a, a, 0x1); }
FFTN_FORCE_INLINE __m128 swap_ri(__m128 a) { return _mm_shuffle_ps(a, a, 0xB1); }
// Flips the sign bit of the imaginary parts (-0 in odd lanes).
FFTN_FORCE_INLINE __m128d conj(__m128d a) { return _mm_xor_pd(a, _mm_setr_pd(0.0, -0.0)); }
FFTN_FORCE_INLINE __m128 conj(__m128 a) {
  return _mm_xor_ps(a, _mm_setr_ps(0.0F, -0.0F, 0.0F, -0.0F));
}
// a + i*b = (ar - bi, ai + br): SSE3's ADDSUB subtracts in even lanes and adds in odd ones.
FFTN_FORCE_INLINE __m128d addj(__m128d a, __m128d b) { return _mm_addsub_pd(a, swap_ri(b)); }
FFTN_FORCE_INLINE __m128 addj(__m128 a, __m128 b) { return _mm_addsub_ps(a, swap_ri(b)); }
// a - i*b = (ar + bi, ai - br): swap_ri(b) with its odd lanes negated (exact), then one
// addition. No SUBADD exists, and without FMA there is no multiply by (1, -1) to fold in.
FFTN_FORCE_INLINE __m128d subj(__m128d a, __m128d b) { return _mm_add_pd(a, conj(swap_ri(b))); }
FFTN_FORCE_INLINE __m128 subj(__m128 a, __m128 b) { return _mm_add_ps(a, conj(swap_ri(b))); }
// a*w = (ar*wr - ai*wi, ai*wr + ar*wi) = r*a + i*swap_ri(a): both products rounded, then
// summed, as the scalar target sums them.
FFTN_FORCE_INLINE __m128d mul(__m128d a, tw_f64 w) {
  return _mm_add_pd(_mm_mul_pd(a, w.r), _mm_mul_pd(swap_ri(a), w.i));
}
FFTN_FORCE_INLINE __m128 mul(__m128 a, tw_f32 w) {
  return _mm_add_ps(_mm_mul_ps(a, w.r), _mm_mul_ps(swap_ri(a), w.i));
}
// a*conj(w) = (ar*wr + ai*wi, ai*wr - ar*wi) = r*a - i*swap_ri(a).
FFTN_FORCE_INLINE __m128d mulc(__m128d a, tw_f64 w) {
  return _mm_sub_pd(_mm_mul_pd(a, w.r), _mm_mul_pd(swap_ri(a), w.i));
}
FFTN_FORCE_INLINE __m128 mulc(__m128 a, tw_f32 w) {
  return _mm_sub_ps(_mm_mul_ps(a, w.r), _mm_mul_ps(swap_ri(a), w.i));
}
// a*b for two complex registers: b rearranged into mul's twiddle form, (br, br) and
// (-bi, bi), in registers (the negation flips a sign bit, as pack_tw's does).
FFTN_FORCE_INLINE __m128d cmul(__m128d a, __m128d b) {
  const __m128d i = _mm_xor_pd(_mm_unpackhi_pd(b, b), _mm_setr_pd(-0.0, 0.0));
  return mul(a, tw_f64{.r = _mm_movedup_pd(b), .i = i});
}
FFTN_FORCE_INLINE __m128 cmul(__m128 a, __m128 b) {
  const __m128 i = _mm_xor_ps(_mm_movehdup_ps(b), _mm_setr_ps(-0.0F, 0.0F, -0.0F, 0.0F));
  return mul(a, tw_f32{.r = _mm_moveldup_ps(b), .i = i});
}
// acc + a*w: the product rounded as mul's, then added (the scalar target's order).
FFTN_FORCE_INLINE __m128d madd(__m128d acc, __m128d a, tw_f64 w) { return add(acc, mul(a, w)); }
FFTN_FORCE_INLINE __m128 madd(__m128 acc, __m128 a, tw_f32 w) { return add(acc, mul(a, w)); }
FFTN_FORCE_INLINE __m128d scale(__m128d a, double r) { return _mm_mul_pd(a, _mm_set1_pd(r)); }
FFTN_FORCE_INLINE __m128 scale(__m128 a, float r) { return _mm_mul_ps(a, _mm_set1_ps(r)); }
FFTN_FORCE_INLINE __m128d fmadd_r(__m128d acc, __m128d a, double r) {
  return _mm_add_pd(acc, _mm_mul_pd(a, _mm_set1_pd(r)));
}
FFTN_FORCE_INLINE __m128 fmadd_r(__m128 acc, __m128 a, float r) {
  return _mm_add_ps(acc, _mm_mul_ps(a, _mm_set1_ps(r)));
}
FFTN_FORCE_INLINE __m128d neg(__m128d a) { return _mm_xor_pd(a, _mm_set1_pd(-0.0)); }
FFTN_FORCE_INLINE __m128 neg(__m128 a) { return _mm_xor_ps(a, _mm_set1_ps(-0.0F)); }
// One complex double per register: a 1 x 1 block is its own transpose.
FFTN_FORCE_INLINE void transpose(__m128d* /*v*/) {}
// Two complex floats per register: a 2 x 2 block, each complex number moved as one 64-bit
// lane.
FFTN_FORCE_INLINE void transpose(__m128* v) {
  const __m128 a = v[0];
  const __m128 b = v[1];
  v[0] = _mm_movelh_ps(a, b); // a0 b0
  v[1] = _mm_movehl_ps(b, a); // a1 b1
}

// Real-register operations.
namespace real {

FFTN_FORCE_INLINE __m128d load(const double* p) { return _mm_loadu_pd(p); }
FFTN_FORCE_INLINE __m128 load(const float* p) { return _mm_loadu_ps(p); }
FFTN_FORCE_INLINE void store(double* p, __m128d a) { _mm_storeu_pd(p, a); }
FFTN_FORCE_INLINE void store(float* p, __m128 a) { _mm_storeu_ps(p, a); }
FFTN_FORCE_INLINE __m128d bcast(double x) { return _mm_set1_pd(x); }
FFTN_FORCE_INLINE __m128 bcast(float x) { return _mm_set1_ps(x); }
FFTN_FORCE_INLINE __m128d add(__m128d a, __m128d b) { return _mm_add_pd(a, b); }
FFTN_FORCE_INLINE __m128 add(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
FFTN_FORCE_INLINE __m128d sub(__m128d a, __m128d b) { return _mm_sub_pd(a, b); }
FFTN_FORCE_INLINE __m128 sub(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
FFTN_FORCE_INLINE __m128d mul(__m128d a, __m128d b) { return _mm_mul_pd(a, b); }
FFTN_FORCE_INLINE __m128 mul(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
// Not fused: the product rounded, then the sum.
FFTN_FORCE_INLINE __m128d fmadd(__m128d a, __m128d b, __m128d c) {
  return _mm_add_pd(_mm_mul_pd(a, b), c);
}
FFTN_FORCE_INLINE __m128 fmadd(__m128 a, __m128 b, __m128 c) {
  return _mm_add_ps(_mm_mul_ps(a, b), c);
}
FFTN_FORCE_INLINE __m128d fmsub(__m128d a, __m128d b, __m128d c) {
  return _mm_sub_pd(_mm_mul_pd(a, b), c);
}
FFTN_FORCE_INLINE __m128 fmsub(__m128 a, __m128 b, __m128 c) {
  return _mm_sub_ps(_mm_mul_ps(a, b), c);
}
FFTN_FORCE_INLINE __m128d fnmadd(__m128d a, __m128d b, __m128d c) {
  return _mm_sub_pd(c, _mm_mul_pd(a, b));
}
FFTN_FORCE_INLINE __m128 fnmadd(__m128 a, __m128 b, __m128 c) {
  return _mm_sub_ps(c, _mm_mul_ps(a, b));
}
FFTN_FORCE_INLINE __m128d neg(__m128d a) { return _mm_xor_pd(a, _mm_set1_pd(-0.0)); }
FFTN_FORCE_INLINE __m128 neg(__m128 a) { return _mm_xor_ps(a, _mm_set1_ps(-0.0F)); }
FFTN_FORCE_INLINE void deinterleave(const double* p, __m128d& re, __m128d& im) {
  const __m128d a = _mm_loadu_pd(p);     // r0 i0
  const __m128d b = _mm_loadu_pd(p + 2); // r1 i1
  re = _mm_unpacklo_pd(a, b);
  im = _mm_unpackhi_pd(a, b);
}
FFTN_FORCE_INLINE void deinterleave(const float* p, __m128& re, __m128& im) {
  const __m128 a = _mm_loadu_ps(p);     // r0 i0 r1 i1
  const __m128 b = _mm_loadu_ps(p + 4); // r2 i2 r3 i3
  re = _mm_shuffle_ps(a, b, 0x88);
  im = _mm_shuffle_ps(a, b, 0xDD);
}
FFTN_FORCE_INLINE void interleave(double* p, __m128d re, __m128d im) {
  _mm_storeu_pd(p, _mm_unpacklo_pd(re, im));
  _mm_storeu_pd(p + 2, _mm_unpackhi_pd(re, im));
}
FFTN_FORCE_INLINE void interleave(float* p, __m128 re, __m128 im) {
  _mm_storeu_ps(p, _mm_unpacklo_ps(re, im));
  _mm_storeu_ps(p + 4, _mm_unpackhi_ps(re, im));
}

} // namespace real

} // namespace
// NOLINTEND(portability-simd-intrinsics)
