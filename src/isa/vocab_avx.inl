// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary of the avx target: x86-64 AVX without AVX2 or FMA (Sandy Bridge, Ivy Bridge,
// the AMD Bulldozer family). 256-bit registers: two complex doubles or four complex floats
// per complex register, as avx2 (vocab_avx2.inl), whose shape this follows with three
// differences: no FMA (fmadd and its relatives multiply, then add), no cross-lane permute
// of 64-bit units (VPERMPD is AVX2), no integer operations on 256-bit registers. Included
// inside FFTN_ISA_BEGIN(avx); the contract is in vocabulary.inl.
//
// Every operation rounds exactly as x86_v2's does (vocab_x86_v2.inl), so the two targets
// give the same bits with the same passes (DESIGN.md 6.2).

#include "isa/vocabulary.inl"

#ifndef FFTN_MASKED_TAILS
#error "FFTN_MASKED_TAILS (0 or 1) is defined by the build (CMakeLists.txt)"
#endif

// NOLINTBEGIN(portability-simd-intrinsics): this file is where FFTN's x86 intrinsics live.
namespace {

inline constexpr target id = target::avx;
inline constexpr bool fused = false;
// Masked partial runs where the build enables them, as on avx2 (experiment X23): VMASKMOV
// is AVX.
inline constexpr bool masked = FFTN_MASKED_TAILS != 0;

// Twiddles pre-arranged for mul as on avx2: r holds (wr, wr) and i holds (-wi, wi) for each
// lane's twiddle. Special members trivial, defined inside the region.
struct tw_f64 {
  __m256d r;
  __m256d i;
};
struct tw_f32 {
  __m256 r;
  __m256 i;
};

template <> struct vec<double> {
  using cv = __m256d;
  using rv = __m256d;
  using tw = tw_f64;
  static constexpr std::size_t C = 2;
  static constexpr std::size_t W = 4;
  static constexpr std::size_t split_block = W;
  static constexpr std::size_t tw_reals = 4;
};

template <> struct vec<float> {
  using cv = __m256;
  using rv = __m256;
  using tw = tw_f32;
  static constexpr std::size_t C = 4;
  static constexpr std::size_t W = 8;
  static constexpr std::size_t split_block = W;
  static constexpr std::size_t tw_reals = 4;
};

// Complex-register operations.

FFTN_FORCE_INLINE __m256d load(const double* p) { return _mm256_loadu_pd(p); }
FFTN_FORCE_INLINE __m256 load(const float* p) { return _mm256_loadu_ps(p); }
FFTN_FORCE_INLINE void store(double* p, __m256d a) { _mm256_storeu_pd(p, a); }
FFTN_FORCE_INLINE void store(float* p, __m256 a) { _mm256_storeu_ps(p, a); }
// Partial runs (experiment X23), as on avx2. The mask has the sign bits of the 2n reals of
// n complex numbers set; it is built by a floating-point comparison, since AVX has no
// 256-bit integer comparison. Masked-off elements are not accessed and raise no fault
// (Intel SDM, VMASKMOV). Stores: plain stores of 128 and 64 bits chosen by n.
FFTN_FORCE_INLINE __m256i mask_n_f64(std::size_t n) {
  return _mm256_castpd_si256(_mm256_cmp_pd(_mm256_set1_pd(static_cast<double>(2 * n)),
                                           _mm256_setr_pd(0.0, 1.0, 2.0, 3.0), _CMP_GT_OQ));
}
FFTN_FORCE_INLINE __m256i mask_n_f32(std::size_t n) {
  return _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_set1_ps(static_cast<float>(2 * n)),
                    _mm256_setr_ps(0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F), _CMP_GT_OQ));
}
FFTN_FORCE_INLINE __m256d load_n(const double* p, std::size_t n) {
  return _mm256_maskload_pd(p, mask_n_f64(n));
}
FFTN_FORCE_INLINE __m256 load_n(const float* p, std::size_t n) {
  return _mm256_maskload_ps(p, mask_n_f32(n));
}
FFTN_FORCE_INLINE void store_n(double* p, std::size_t n, __m256d a) {
  if (n == 2) {
    _mm256_storeu_pd(p, a);
  } else if (n == 1) {
    _mm_storeu_pd(p, _mm256_castpd256_pd128(a));
  }
}
// The 64-bit stores go through __m128i pointers, which the intrinsics' headers declare as
// may-alias and unaligned (a complex float is only 4-byte aligned).
FFTN_FORCE_INLINE void store_n(float* p, std::size_t n, __m256 a) {
  const __m128 lo = _mm256_castps256_ps128(a);
  if (n >= 2) {
    _mm_storeu_ps(p, lo);
    const __m128 hi = _mm256_extractf128_ps(a, 1);
    if (n == 4) {
      _mm_storeu_ps(p + 4, hi);
    } else if (n == 3) {
      _mm_storel_epi64(reinterpret_cast<__m128i*>(p + 4), _mm_castps_si128(hi));
    }
  } else if (n == 1) {
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm_castps_si128(lo));
  }
}
FFTN_FORCE_INLINE __m256d bcast(const double* p) {
  const __m128d x = _mm_loadu_pd(p);
  return _mm256_set_m128d(x, x);
}
// Built from scalars rather than an 8-byte load through a cast pointer: std::complex<float>
// is only 4-byte aligned.
FFTN_FORCE_INLINE __m256 bcast(const float* p) {
  const __m128 x = _mm_setr_ps(p[0], p[1], p[0], p[1]);
  return _mm256_set_m128(x, x);
}
FFTN_FORCE_INLINE tw_f64 load_tw(const double* p) {
  return {.r = _mm256_loadu_pd(p), .i = _mm256_loadu_pd(p + 4)};
}
FFTN_FORCE_INLINE tw_f32 load_tw(const float* p) {
  return {.r = _mm256_loadu_ps(p), .i = _mm256_loadu_ps(p + 8)};
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
// Broadcast storage of one twiddle: (wr, wr, -wi, wi), repeated into every lane by two
// 128-bit broadcasts (double) or 64-bit ones (float, built from scalars as bcast is).
FFTN_FORCE_INLINE tw_f64 bcast_tw(const double* p) {
  const __m128d r = _mm_loadu_pd(p);
  const __m128d i = _mm_loadu_pd(p + 2);
  return {.r = _mm256_set_m128d(r, r), .i = _mm256_set_m128d(i, i)};
}
FFTN_FORCE_INLINE tw_f32 bcast_tw(const float* p) {
  const __m128 r = _mm_setr_ps(p[0], p[1], p[0], p[1]);
  const __m128 i = _mm_setr_ps(p[2], p[3], p[2], p[3]);
  return {.r = _mm256_set_m128(r, r), .i = _mm256_set_m128(i, i)};
}
template <class T> FFTN_FORCE_INLINE void pack_tw1(T* out, const T* p) {
  out[0] = p[0];
  out[1] = p[0];
  out[2] = -p[1];
  out[3] = p[1];
}

FFTN_FORCE_INLINE __m256d add(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
FFTN_FORCE_INLINE __m256 add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
FFTN_FORCE_INLINE __m256d sub(__m256d a, __m256d b) { return _mm256_sub_pd(a, b); }
FFTN_FORCE_INLINE __m256 sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
FFTN_FORCE_INLINE __m256d swap_ri(__m256d a) { return _mm256_permute_pd(a, 0x5); }
FFTN_FORCE_INLINE __m256 swap_ri(__m256 a) { return _mm256_permute_ps(a, 0xB1); }
FFTN_FORCE_INLINE __m256d conj(__m256d a) {
  return _mm256_xor_pd(a, _mm256_setr_pd(0.0, -0.0, 0.0, -0.0));
}
FFTN_FORCE_INLINE __m256 conj(__m256 a) {
  return _mm256_xor_ps(a, _mm256_setr_ps(0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F));
}
// a + i*b = (ar - bi, ai + br): ADDSUB subtracts in even lanes and adds in odd ones.
FFTN_FORCE_INLINE __m256d addj(__m256d a, __m256d b) { return _mm256_addsub_pd(a, swap_ri(b)); }
FFTN_FORCE_INLINE __m256 addj(__m256 a, __m256 b) { return _mm256_addsub_ps(a, swap_ri(b)); }
// a - i*b = (ar + bi, ai - br): swap_ri(b) with its odd lanes negated (exact), then one
// addition (avx2 folds the negation into an FMA by (1, -1), which AVX lacks).
FFTN_FORCE_INLINE __m256d subj(__m256d a, __m256d b) { return _mm256_add_pd(a, conj(swap_ri(b))); }
FFTN_FORCE_INLINE __m256 subj(__m256 a, __m256 b) { return _mm256_add_ps(a, conj(swap_ri(b))); }
// a*w = r*a + i*swap_ri(a): both products rounded, then summed (as x86_v2 and scalar).
FFTN_FORCE_INLINE __m256d mul(__m256d a, tw_f64 w) {
  return _mm256_add_pd(_mm256_mul_pd(a, w.r), _mm256_mul_pd(swap_ri(a), w.i));
}
FFTN_FORCE_INLINE __m256 mul(__m256 a, tw_f32 w) {
  return _mm256_add_ps(_mm256_mul_ps(a, w.r), _mm256_mul_ps(swap_ri(a), w.i));
}
// a*conj(w) = r*a - i*swap_ri(a).
FFTN_FORCE_INLINE __m256d mulc(__m256d a, tw_f64 w) {
  return _mm256_sub_pd(_mm256_mul_pd(a, w.r), _mm256_mul_pd(swap_ri(a), w.i));
}
FFTN_FORCE_INLINE __m256 mulc(__m256 a, tw_f32 w) {
  return _mm256_sub_ps(_mm256_mul_ps(a, w.r), _mm256_mul_ps(swap_ri(a), w.i));
}
// a*b for two complex registers: b rearranged into mul's twiddle form, (br, br) and
// (-bi, bi), in registers (the negation flips a sign bit, as pack_tw's does).
FFTN_FORCE_INLINE __m256d cmul(__m256d a, __m256d b) {
  const __m256d i = _mm256_xor_pd(_mm256_permute_pd(b, 0xF), _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0));
  return mul(a, tw_f64{.r = _mm256_movedup_pd(b), .i = i});
}
FFTN_FORCE_INLINE __m256 cmul(__m256 a, __m256 b) {
  const __m256 i = _mm256_xor_ps(
      _mm256_movehdup_ps(b), _mm256_setr_ps(-0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F));
  return mul(a, tw_f32{.r = _mm256_moveldup_ps(b), .i = i});
}
// acc + a*w: the product rounded as mul's, then added.
FFTN_FORCE_INLINE __m256d madd(__m256d acc, __m256d a, tw_f64 w) { return add(acc, mul(a, w)); }
FFTN_FORCE_INLINE __m256 madd(__m256 acc, __m256 a, tw_f32 w) { return add(acc, mul(a, w)); }
FFTN_FORCE_INLINE __m256d scale(__m256d a, double r) { return _mm256_mul_pd(a, _mm256_set1_pd(r)); }
FFTN_FORCE_INLINE __m256 scale(__m256 a, float r) { return _mm256_mul_ps(a, _mm256_set1_ps(r)); }
FFTN_FORCE_INLINE __m256d fmadd_r(__m256d acc, __m256d a, double r) {
  return _mm256_add_pd(acc, _mm256_mul_pd(a, _mm256_set1_pd(r)));
}
FFTN_FORCE_INLINE __m256 fmadd_r(__m256 acc, __m256 a, float r) {
  return _mm256_add_ps(acc, _mm256_mul_ps(a, _mm256_set1_ps(r)));
}
FFTN_FORCE_INLINE __m256d neg(__m256d a) { return _mm256_xor_pd(a, _mm256_set1_pd(-0.0)); }
FFTN_FORCE_INLINE __m256 neg(__m256 a) { return _mm256_xor_ps(a, _mm256_set1_ps(-0.0F)); }
// Two complex doubles per register: a 2 x 2 block of 128-bit halves.
FFTN_FORCE_INLINE void transpose(__m256d* v) {
  const __m256d a = v[0];
  const __m256d b = v[1];
  v[0] = _mm256_permute2f128_pd(a, b, 0x20);
  v[1] = _mm256_permute2f128_pd(a, b, 0x31);
}
// Four complex floats per register: a 4 x 4 block, each complex number moved as one
// 64-bit lane (unpack within halves, then exchange halves).
FFTN_FORCE_INLINE void transpose(__m256* v) {
  const __m256d a = _mm256_castps_pd(v[0]);
  const __m256d b = _mm256_castps_pd(v[1]);
  const __m256d c = _mm256_castps_pd(v[2]);
  const __m256d d = _mm256_castps_pd(v[3]);
  const __m256d ab0 = _mm256_unpacklo_pd(a, b); // a0 b0 a2 b2
  const __m256d ab1 = _mm256_unpackhi_pd(a, b); // a1 b1 a3 b3
  const __m256d cd0 = _mm256_unpacklo_pd(c, d);
  const __m256d cd1 = _mm256_unpackhi_pd(c, d);
  v[0] = _mm256_castpd_ps(_mm256_permute2f128_pd(ab0, cd0, 0x20)); // a0 b0 c0 d0
  v[1] = _mm256_castpd_ps(_mm256_permute2f128_pd(ab1, cd1, 0x20)); // a1 b1 c1 d1
  v[2] = _mm256_castpd_ps(_mm256_permute2f128_pd(ab0, cd0, 0x31)); // a2 b2 c2 d2
  v[3] = _mm256_castpd_ps(_mm256_permute2f128_pd(ab1, cd1, 0x31)); // a3 b3 c3 d3
}

// Real-register operations.
namespace real {

FFTN_FORCE_INLINE __m256d load(const double* p) { return _mm256_loadu_pd(p); }
FFTN_FORCE_INLINE __m256 load(const float* p) { return _mm256_loadu_ps(p); }
FFTN_FORCE_INLINE void store(double* p, __m256d a) { _mm256_storeu_pd(p, a); }
FFTN_FORCE_INLINE void store(float* p, __m256 a) { _mm256_storeu_ps(p, a); }
FFTN_FORCE_INLINE __m256d bcast(double x) { return _mm256_set1_pd(x); }
FFTN_FORCE_INLINE __m256 bcast(float x) { return _mm256_set1_ps(x); }
FFTN_FORCE_INLINE __m256d add(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
FFTN_FORCE_INLINE __m256 add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
FFTN_FORCE_INLINE __m256d sub(__m256d a, __m256d b) { return _mm256_sub_pd(a, b); }
FFTN_FORCE_INLINE __m256 sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
FFTN_FORCE_INLINE __m256d mul(__m256d a, __m256d b) { return _mm256_mul_pd(a, b); }
FFTN_FORCE_INLINE __m256 mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
// Not fused: the product rounded, then the sum.
FFTN_FORCE_INLINE __m256d fmadd(__m256d a, __m256d b, __m256d c) {
  return _mm256_add_pd(_mm256_mul_pd(a, b), c);
}
FFTN_FORCE_INLINE __m256 fmadd(__m256 a, __m256 b, __m256 c) {
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}
FFTN_FORCE_INLINE __m256d fmsub(__m256d a, __m256d b, __m256d c) {
  return _mm256_sub_pd(_mm256_mul_pd(a, b), c);
}
FFTN_FORCE_INLINE __m256 fmsub(__m256 a, __m256 b, __m256 c) {
  return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
}
FFTN_FORCE_INLINE __m256d fnmadd(__m256d a, __m256d b, __m256d c) {
  return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
}
FFTN_FORCE_INLINE __m256 fnmadd(__m256 a, __m256 b, __m256 c) {
  return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
}
FFTN_FORCE_INLINE __m256d neg(__m256d a) { return _mm256_xor_pd(a, _mm256_set1_pd(-0.0)); }
FFTN_FORCE_INLINE __m256 neg(__m256 a) { return _mm256_xor_ps(a, _mm256_set1_ps(-0.0F)); }
// Without AVX2's VPERMPD the halves are exchanged first: the two 128-bit halves holding
// complex numbers 0, 1 and 2, 3 (double) are paired, then unpacked within each half, which
// leaves re and im in natural order, lined up with data loaded by real::load.
FFTN_FORCE_INLINE void deinterleave(const double* p, __m256d& re, __m256d& im) {
  const __m256d a = _mm256_loadu_pd(p);                 // r0 i0 r1 i1
  const __m256d b = _mm256_loadu_pd(p + 4);             // r2 i2 r3 i3
  const __m256d lo = _mm256_permute2f128_pd(a, b, 0x20); // r0 i0 r2 i2
  const __m256d hi = _mm256_permute2f128_pd(a, b, 0x31); // r1 i1 r3 i3
  re = _mm256_unpacklo_pd(lo, hi);                       // r0 r1 r2 r3
  im = _mm256_unpackhi_pd(lo, hi);
}
FFTN_FORCE_INLINE void deinterleave(const float* p, __m256& re, __m256& im) {
  const __m256 a = _mm256_loadu_ps(p);                 // r0 i0 r1 i1 | r2 i2 r3 i3
  const __m256 b = _mm256_loadu_ps(p + 8);             // r4 i4 r5 i5 | r6 i6 r7 i7
  const __m256 lo = _mm256_permute2f128_ps(a, b, 0x20); // r0 i0 r1 i1 | r4 i4 r5 i5
  const __m256 hi = _mm256_permute2f128_ps(a, b, 0x31); // r2 i2 r3 i3 | r6 i6 r7 i7
  re = _mm256_shuffle_ps(lo, hi, 0x88);                 // r0 r1 r2 r3 | r4 r5 r6 r7
  im = _mm256_shuffle_ps(lo, hi, 0xDD);
}
FFTN_FORCE_INLINE void interleave(double* p, __m256d re, __m256d im) {
  const __m256d lo = _mm256_unpacklo_pd(re, im); // r0 i0 r2 i2
  const __m256d hi = _mm256_unpackhi_pd(re, im); // r1 i1 r3 i3
  _mm256_storeu_pd(p, _mm256_permute2f128_pd(lo, hi, 0x20));
  _mm256_storeu_pd(p + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
}
FFTN_FORCE_INLINE void interleave(float* p, __m256 re, __m256 im) {
  const __m256 lo = _mm256_unpacklo_ps(re, im); // r0 i0 r1 i1 | r4 i4 r5 i5
  const __m256 hi = _mm256_unpackhi_ps(re, im); // r2 i2 r3 i3 | r6 i6 r7 i7
  _mm256_storeu_ps(p, _mm256_permute2f128_ps(lo, hi, 0x20));
  _mm256_storeu_ps(p + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
}

} // namespace real

} // namespace
// NOLINTEND(portability-simd-intrinsics)
