// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary of the avx512 target: x86-64 AVX-512 F, DQ and VL with AVX2 and FMA
// (x86-64-v4: Intel server parts since Skylake-SP, Zen 4 and newer). 512-bit registers for
// both precisions: four complex doubles or eight complex floats per complex register.
// Included inside FFTN_ISA_BEGIN(avx512); the contract is in vocabulary.inl.
//
// Every operation rounds exactly as avx2's does (the same FMA order in mul, mulc and madd;
// addj and subj round one addition), so the two targets give the same bits. DQ is used for
// the floating-point XORs (sign flips) and the 128-bit broadcast of a complex double.

#include "isa/vocabulary.inl"

#ifndef FFTN_MASKED_TAILS
#error "FFTN_MASKED_TAILS (0 or 1) is defined by the build (CMakeLists.txt)"
#endif

// NOLINTBEGIN(portability-simd-intrinsics): this file is where FFTN's AVX-512 intrinsics live.
namespace {

inline constexpr target id = target::avx512;
inline constexpr bool fused = true;
// Masked partial runs (load_n, store_n below) where the build enables them: on with GCC,
// off with Clang and icpx by default (FFTN_MASKED_TAILS in CMakeLists.txt; experiment X23).
inline constexpr bool masked = FFTN_MASKED_TAILS != 0;

// Twiddles as mul consumes them, as on avx2 (DESIGN.md 6.2): r holds (wr, wr) and i holds
// (wi, wi) for each lane's twiddle. Defined inside the region; special members trivial.
struct tw_f64 {
  __m512d r;
  __m512d i;
};
struct tw_f32 {
  __m512 r;
  __m512 i;
};

template <> struct vec<double> {
  using cv = __m512d;
  using rv = __m512d;
  using tw = tw_f64;
  static constexpr std::size_t C = 4;
  static constexpr std::size_t W = 8;
  static constexpr std::size_t split_block = W;
  static constexpr std::size_t tw_reals = 2;
  static constexpr std::size_t tw_tail = 1;
};

template <> struct vec<float> {
  using cv = __m512;
  using rv = __m512;
  using tw = tw_f32;
  static constexpr std::size_t C = 8;
  static constexpr std::size_t W = 16;
  static constexpr std::size_t split_block = W;
  static constexpr std::size_t tw_reals = 2;
  static constexpr std::size_t tw_tail = 1;
};

// Complex-register operations.

FFTN_FORCE_INLINE __m512d load(const double* p) { return _mm512_loadu_pd(p); }
FFTN_FORCE_INLINE __m512 load(const float* p) { return _mm512_loadu_ps(p); }
FFTN_FORCE_INLINE void store(double* p, __m512d a) { _mm512_storeu_pd(p, a); }
FFTN_FORCE_INLINE void store(float* p, __m512 a) { _mm512_storeu_ps(p, a); }
// Partial runs (experiment X23): a mask of the 2n reals of n complex numbers. Masked-off
// elements of a masked load or store are not accessed, and AVX-512 suppresses their faults.
FFTN_FORCE_INLINE __m512d load_n(const double* p, std::size_t n) {
  return _mm512_maskz_loadu_pd(static_cast<__mmask8>((1U << (2 * n)) - 1U), p);
}
FFTN_FORCE_INLINE __m512 load_n(const float* p, std::size_t n) {
  return _mm512_maskz_loadu_ps(static_cast<__mmask16>((1U << (2 * n)) - 1U), p);
}
FFTN_FORCE_INLINE void store_n(double* p, std::size_t n, __m512d a) {
  _mm512_mask_storeu_pd(p, static_cast<__mmask8>((1U << (2 * n)) - 1U), a);
}
FFTN_FORCE_INLINE void store_n(float* p, std::size_t n, __m512 a) {
  _mm512_mask_storeu_ps(p, static_cast<__mmask16>((1U << (2 * n)) - 1U), a);
}
// The complex number at p in every lane: its 16 bytes in every 128-bit lane (double).
FFTN_FORCE_INLINE __m512d bcast(const double* p) { return _mm512_broadcast_f64x2(_mm_loadu_pd(p)); }
// Its 8 bytes in every 64-bit lane (float), built from scalars rather than loaded through a
// cast pointer: std::complex<float> is only 4-byte aligned (as avx2's bcast).
FFTN_FORCE_INLINE __m512 bcast(const float* p) {
  return _mm512_broadcast_f32x4(_mm_setr_ps(p[0], p[1], p[0], p[1]));
}
// Twiddle storage is interleaved, (wr, wi) per twiddle, read by duplicating loads (load
// ports only), i from one real further on: load_tw reads tw_tail = 1 real past the C
// twiddles, as on avx2 (experiment X30).
FFTN_FORCE_INLINE tw_f64 load_tw(const double* p) {
  return {.r = _mm512_movedup_pd(_mm512_loadu_pd(p)),
          .i = _mm512_movedup_pd(_mm512_loadu_pd(p + 1))};
}
FFTN_FORCE_INLINE tw_f32 load_tw(const float* p) {
  return {.r = _mm512_moveldup_ps(_mm512_loadu_ps(p)),
          .i = _mm512_moveldup_ps(_mm512_loadu_ps(p + 1))};
}
template <class T> FFTN_FORCE_INLINE void pack_tw(T* out, const T* p) {
  for (std::size_t k = 0; k < 2 * C<T>; ++k) {
    out[k] = p[k];
  }
}
// Broadcast storage of one twiddle: (wr, wi), each part broadcast into every lane (a
// broadcast the compiler may fold into the multiply that uses it).
FFTN_FORCE_INLINE tw_f64 bcast_tw(const double* p) {
  return {.r = _mm512_set1_pd(p[0]), .i = _mm512_set1_pd(p[1])};
}
FFTN_FORCE_INLINE tw_f32 bcast_tw(const float* p) {
  return {.r = _mm512_set1_ps(p[0]), .i = _mm512_set1_ps(p[1])};
}
template <class T> FFTN_FORCE_INLINE void pack_tw1(T* out, const T* p) {
  out[0] = p[0];
  out[1] = p[1];
}

// Sign masks: -0 in the imaginary parts (odd lanes).
FFTN_FORCE_INLINE __m512d odd_sign_f64() {
  return _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
}
FFTN_FORCE_INLINE __m512 odd_sign_f32() {
  return _mm512_setr_ps(0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F,
                        -0.0F, 0.0F, -0.0F, 0.0F, -0.0F);
}

FFTN_FORCE_INLINE __m512d add(__m512d a, __m512d b) { return _mm512_add_pd(a, b); }
FFTN_FORCE_INLINE __m512 add(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }
FFTN_FORCE_INLINE __m512d sub(__m512d a, __m512d b) { return _mm512_sub_pd(a, b); }
FFTN_FORCE_INLINE __m512 sub(__m512 a, __m512 b) { return _mm512_sub_ps(a, b); }
FFTN_FORCE_INLINE __m512d swap_ri(__m512d a) { return _mm512_permute_pd(a, 0x55); }
FFTN_FORCE_INLINE __m512 swap_ri(__m512 a) { return _mm512_permute_ps(a, 0xB1); }
// a + i*b = (ar - bi, ai + br). AVX-512 has no ADDSUB; FMADDSUB by ones subtracts in even
// lanes and adds in odd ones, and with an exact product it rounds once, as an addition.
FFTN_FORCE_INLINE __m512d addj(__m512d a, __m512d b) {
  return _mm512_fmaddsub_pd(a, _mm512_set1_pd(1.0), swap_ri(b));
}
FFTN_FORCE_INLINE __m512 addj(__m512 a, __m512 b) {
  return _mm512_fmaddsub_ps(a, _mm512_set1_ps(1.0F), swap_ri(b));
}
// a - i*b = (ar + bi, ai - br): FMSUBADD adds in even lanes and subtracts in odd ones.
FFTN_FORCE_INLINE __m512d subj(__m512d a, __m512d b) {
  return _mm512_fmsubadd_pd(a, _mm512_set1_pd(1.0), swap_ri(b));
}
FFTN_FORCE_INLINE __m512 subj(__m512 a, __m512 b) {
  return _mm512_fmsubadd_ps(a, _mm512_set1_ps(1.0F), swap_ri(b));
}
// a*w = (ar*wr - ai*wi, ai*wr + ar*wi): FMADDSUB, rounding as avx2's mul (X30).
FFTN_FORCE_INLINE __m512d mul(__m512d a, tw_f64 w) {
  return _mm512_fmaddsub_pd(a, w.r, _mm512_mul_pd(swap_ri(a), w.i));
}
FFTN_FORCE_INLINE __m512 mul(__m512 a, tw_f32 w) {
  return _mm512_fmaddsub_ps(a, w.r, _mm512_mul_ps(swap_ri(a), w.i));
}
// a*conj(w) = (ar*wr + ai*wi, ai*wr - ar*wi): FMSUBADD.
FFTN_FORCE_INLINE __m512d mulc(__m512d a, tw_f64 w) {
  return _mm512_fmsubadd_pd(a, w.r, _mm512_mul_pd(swap_ri(a), w.i));
}
FFTN_FORCE_INLINE __m512 mulc(__m512 a, tw_f32 w) {
  return _mm512_fmsubadd_ps(a, w.r, _mm512_mul_ps(swap_ri(a), w.i));
}
// a*b for two complex registers: b rearranged into mul's twiddle form, (br, br) and
// (bi, bi), in registers.
FFTN_FORCE_INLINE __m512d cmul(__m512d a, __m512d b) {
  return mul(a, tw_f64{.r = _mm512_movedup_pd(b), .i = _mm512_permute_pd(b, 0xFF)});
}
FFTN_FORCE_INLINE __m512 cmul(__m512 a, __m512 b) {
  return mul(a, tw_f32{.r = _mm512_moveldup_ps(b), .i = _mm512_movehdup_ps(b)});
}
// acc + a*w, as avx2's madd.
FFTN_FORCE_INLINE __m512d madd(__m512d acc, __m512d a, tw_f64 w) {
  return _mm512_fmaddsub_pd(a, w.r, _mm512_fmaddsub_pd(swap_ri(a), w.i, acc));
}
FFTN_FORCE_INLINE __m512 madd(__m512 acc, __m512 a, tw_f32 w) {
  return _mm512_fmaddsub_ps(a, w.r, _mm512_fmaddsub_ps(swap_ri(a), w.i, acc));
}
FFTN_FORCE_INLINE __m512d scale(__m512d a, double r) { return _mm512_mul_pd(a, _mm512_set1_pd(r)); }
FFTN_FORCE_INLINE __m512 scale(__m512 a, float r) { return _mm512_mul_ps(a, _mm512_set1_ps(r)); }
FFTN_FORCE_INLINE __m512d fmadd_r(__m512d acc, __m512d a, double r) {
  return _mm512_fmadd_pd(a, _mm512_set1_pd(r), acc);
}
FFTN_FORCE_INLINE __m512 fmadd_r(__m512 acc, __m512 a, float r) {
  return _mm512_fmadd_ps(a, _mm512_set1_ps(r), acc);
}
FFTN_FORCE_INLINE __m512d neg(__m512d a) { return _mm512_xor_pd(a, _mm512_set1_pd(-0.0)); }
FFTN_FORCE_INLINE __m512 neg(__m512 a) { return _mm512_xor_ps(a, _mm512_set1_ps(-0.0F)); }
FFTN_FORCE_INLINE __m512d conj(__m512d a) { return _mm512_xor_pd(a, odd_sign_f64()); }
FFTN_FORCE_INLINE __m512 conj(__m512 a) { return _mm512_xor_ps(a, odd_sign_f32()); }

// The 4 x 4 transpose of 128-bit units across v[0..3]: unit j of v[i] goes to unit i of
// v[j]. Two rounds of VSHUFF64X2: pairs of units, then single units.
FFTN_FORCE_INLINE void transpose_units(__m512d* v) {
  const __m512d t0 = _mm512_shuffle_f64x2(v[0], v[1], 0x44); // v0.0 v0.1 v1.0 v1.1
  const __m512d t1 = _mm512_shuffle_f64x2(v[0], v[1], 0xEE); // v0.2 v0.3 v1.2 v1.3
  const __m512d t2 = _mm512_shuffle_f64x2(v[2], v[3], 0x44); // v2.0 v2.1 v3.0 v3.1
  const __m512d t3 = _mm512_shuffle_f64x2(v[2], v[3], 0xEE); // v2.2 v2.3 v3.2 v3.3
  v[0] = _mm512_shuffle_f64x2(t0, t2, 0x88);                 // v0.0 v1.0 v2.0 v3.0
  v[1] = _mm512_shuffle_f64x2(t0, t2, 0xDD);                 // v0.1 v1.1 v2.1 v3.1
  v[2] = _mm512_shuffle_f64x2(t1, t3, 0x88);                 // v0.2 v1.2 v2.2 v3.2
  v[3] = _mm512_shuffle_f64x2(t1, t3, 0xDD);                 // v0.3 v1.3 v2.3 v3.3
}
// Four complex doubles per register: a 4 x 4 block of 128-bit units.
FFTN_FORCE_INLINE void transpose(__m512d* v) { transpose_units(v); }
// Eight complex floats per register: an 8 x 8 block, each complex number moved as one
// 64-bit element. Unpacking rows 2k and 2k + 1 pairs their elements within 128-bit units
// (unit u of the low half holds elements 2u of both rows, of the high half elements
// 2u + 1); the 4 x 4 transposes of those units then gather column j's pairs in order.
FFTN_FORCE_INLINE void transpose(__m512* v) {
  __m512d even[4]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  __m512d odd[4];  // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  for (std::size_t k = 0; k < 4; ++k) {
    const __m512d a = _mm512_castps_pd(v[2 * k]);
    const __m512d b = _mm512_castps_pd(v[2 * k + 1]);
    even[k] = _mm512_unpacklo_pd(a, b); // a0 b0 | a2 b2 | a4 b4 | a6 b6
    odd[k] = _mm512_unpackhi_pd(a, b);  // a1 b1 | a3 b3 | a5 b5 | a7 b7
  }
  transpose_units(even); // even[u]: column 2u of rows 0-7
  transpose_units(odd);  // odd[u]: column 2u + 1
  for (std::size_t u = 0; u < 4; ++u) {
    v[2 * u] = _mm512_castpd_ps(even[u]);
    v[2 * u + 1] = _mm512_castpd_ps(odd[u]);
  }
}

// Real-register operations.
namespace real {

FFTN_FORCE_INLINE __m512d load(const double* p) { return _mm512_loadu_pd(p); }
FFTN_FORCE_INLINE __m512 load(const float* p) { return _mm512_loadu_ps(p); }
FFTN_FORCE_INLINE void store(double* p, __m512d a) { _mm512_storeu_pd(p, a); }
FFTN_FORCE_INLINE void store(float* p, __m512 a) { _mm512_storeu_ps(p, a); }
FFTN_FORCE_INLINE __m512d bcast(double x) { return _mm512_set1_pd(x); }
FFTN_FORCE_INLINE __m512 bcast(float x) { return _mm512_set1_ps(x); }
FFTN_FORCE_INLINE __m512d add(__m512d a, __m512d b) { return _mm512_add_pd(a, b); }
FFTN_FORCE_INLINE __m512 add(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }
FFTN_FORCE_INLINE __m512d sub(__m512d a, __m512d b) { return _mm512_sub_pd(a, b); }
FFTN_FORCE_INLINE __m512 sub(__m512 a, __m512 b) { return _mm512_sub_ps(a, b); }
FFTN_FORCE_INLINE __m512d mul(__m512d a, __m512d b) { return _mm512_mul_pd(a, b); }
FFTN_FORCE_INLINE __m512 mul(__m512 a, __m512 b) { return _mm512_mul_ps(a, b); }
FFTN_FORCE_INLINE __m512d fmadd(__m512d a, __m512d b, __m512d c) {
  return _mm512_fmadd_pd(a, b, c);
}
FFTN_FORCE_INLINE __m512 fmadd(__m512 a, __m512 b, __m512 c) { return _mm512_fmadd_ps(a, b, c); }
FFTN_FORCE_INLINE __m512d fmsub(__m512d a, __m512d b, __m512d c) {
  return _mm512_fmsub_pd(a, b, c);
}
FFTN_FORCE_INLINE __m512 fmsub(__m512 a, __m512 b, __m512 c) { return _mm512_fmsub_ps(a, b, c); }
FFTN_FORCE_INLINE __m512d fnmadd(__m512d a, __m512d b, __m512d c) {
  return _mm512_fnmadd_pd(a, b, c);
}
FFTN_FORCE_INLINE __m512 fnmadd(__m512 a, __m512 b, __m512 c) { return _mm512_fnmadd_ps(a, b, c); }
FFTN_FORCE_INLINE __m512d neg(__m512d a) { return _mm512_xor_pd(a, _mm512_set1_pd(-0.0)); }
FFTN_FORCE_INLINE __m512 neg(__m512 a) { return _mm512_xor_ps(a, _mm512_set1_ps(-0.0F)); }
// Two-source permutes (VPERMT2PD/PS) pick the even reals, then the odd ones, of 2W reals
// in natural order; interleaving is the inverse pair.
FFTN_FORCE_INLINE void deinterleave(const double* p, __m512d& re, __m512d& im) {
  const __m512d a = _mm512_loadu_pd(p);
  const __m512d b = _mm512_loadu_pd(p + 8);
  re = _mm512_permutex2var_pd(a, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), b);
  im = _mm512_permutex2var_pd(a, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), b);
}
FFTN_FORCE_INLINE void deinterleave(const float* p, __m512& re, __m512& im) {
  const __m512 a = _mm512_loadu_ps(p);
  const __m512 b = _mm512_loadu_ps(p + 16);
  re = _mm512_permutex2var_ps(
      a, _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), b);
  im = _mm512_permutex2var_ps(
      a, _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31), b);
}
FFTN_FORCE_INLINE void interleave(double* p, __m512d re, __m512d im) {
  _mm512_storeu_pd(p, _mm512_permutex2var_pd(re, _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), im));
  _mm512_storeu_pd(p + 8,
                   _mm512_permutex2var_pd(re, _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15), im));
}
FFTN_FORCE_INLINE void interleave(float* p, __m512 re, __m512 im) {
  _mm512_storeu_ps(
      p, _mm512_permutex2var_ps(
             re, _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23), im));
  _mm512_storeu_ps(
      p + 16,
      _mm512_permutex2var_ps(
          re, _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31), im));
}

} // namespace real

} // namespace
// NOLINTEND(portability-simd-intrinsics)
