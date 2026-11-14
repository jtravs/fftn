// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary of the neon_fcma target: AArch64 Advanced SIMD with the Armv8.3 complex
// instructions FCMLA and FCADD (Apple M-series, Neoverse V1/N2 and newer). 128-bit
// registers: one complex double or two complex floats per complex register. Included
// inside FFTN_ISA_BEGIN(neon_fcma); the contract is in vocabulary.inl.

#include "isa/vocabulary.inl"

#include "isa/vocab_neon_common.inl"

namespace {

inline constexpr target id = target::neon_fcma;
inline constexpr bool fused = true;

template <> struct vec<double> {
  using cv = float64x2_t;
  using rv = float64x2_t;
  using tw = float64x2_t;
  static constexpr std::size_t C = 1;
  static constexpr std::size_t W = 2;
  static constexpr std::size_t tw_reals = 2;
};

template <> struct vec<float> {
  using cv = float32x4_t;
  using rv = float32x4_t;
  using tw = float32x4_t;
  static constexpr std::size_t C = 2;
  static constexpr std::size_t W = 4;
  static constexpr std::size_t tw_reals = 2;
};

// Complex-register operations (memory, add, sub, scale, fmadd_r, neg, conj, swap_ri,
// transpose and the real family are vocab_neon_common.inl's).
//
// FCMLA multiplies by the real (rotation 0) or the imaginary part (rotation 90 or 270) of
// its first multiplicand and accumulates. With the twiddle as that operand:
//   rotation 0:   (+wr*ar, +wr*ai)      rotation 90:  (-wi*ai, +wi*ar)
//   rotation 270: (+wi*ai, -wi*ar)
// so 0 then 90 accumulate a*w, and 0 then 270 accumulate a*conj(w). mul starts from -0, the
// identity of addition, so that a product of -0 keeps its sign as in a plain multiply.

FFTN_FORCE_INLINE float64x2_t load_tw(const double* p) { return vld1q_f64(p); }
FFTN_FORCE_INLINE float32x4_t load_tw(const float* p) { return vld1q_f32(p); }
template <class T> FFTN_FORCE_INLINE void pack_tw(T* out, const T* p) {
  for (std::size_t k = 0; k < 2 * C<T>; ++k) {
    out[k] = p[k];
  }
}
// Broadcast storage is the twiddle itself: a 16-byte load for double, a 64-bit load
// duplicated for float.
FFTN_FORCE_INLINE float64x2_t bcast_tw(const double* p) { return vld1q_f64(p); }
FFTN_FORCE_INLINE float32x4_t bcast_tw(const float* p) { return bcast(p); }
template <class T> FFTN_FORCE_INLINE void pack_tw1(T* out, const T* p) {
  out[0] = p[0];
  out[1] = p[1];
}

FFTN_FORCE_INLINE float64x2_t addj(float64x2_t a, float64x2_t b) { return vcaddq_rot90_f64(a, b); }
FFTN_FORCE_INLINE float32x4_t addj(float32x4_t a, float32x4_t b) { return vcaddq_rot90_f32(a, b); }
FFTN_FORCE_INLINE float64x2_t subj(float64x2_t a, float64x2_t b) { return vcaddq_rot270_f64(a, b); }
FFTN_FORCE_INLINE float32x4_t subj(float32x4_t a, float32x4_t b) { return vcaddq_rot270_f32(a, b); }
FFTN_FORCE_INLINE float64x2_t madd(float64x2_t acc, float64x2_t a, float64x2_t w) {
  return vcmlaq_rot90_f64(vcmlaq_f64(acc, w, a), w, a);
}
FFTN_FORCE_INLINE float32x4_t madd(float32x4_t acc, float32x4_t a, float32x4_t w) {
  return vcmlaq_rot90_f32(vcmlaq_f32(acc, w, a), w, a);
}
FFTN_FORCE_INLINE float64x2_t mul(float64x2_t a, float64x2_t w) {
  return madd(vdupq_n_f64(-0.0), a, w);
}
FFTN_FORCE_INLINE float32x4_t mul(float32x4_t a, float32x4_t w) {
  return madd(vdupq_n_f32(-0.0F), a, w);
}
FFTN_FORCE_INLINE float64x2_t mulc(float64x2_t a, float64x2_t w) {
  return vcmlaq_rot270_f64(vcmlaq_f64(vdupq_n_f64(-0.0), w, a), w, a);
}
FFTN_FORCE_INLINE float32x4_t mulc(float32x4_t a, float32x4_t w) {
  return vcmlaq_rot270_f32(vcmlaq_f32(vdupq_n_f32(-0.0F), w, a), w, a);
}
// Twiddle storage is plain complex numbers here, so a register is already a twiddle.
FFTN_FORCE_INLINE float64x2_t cmul(float64x2_t a, float64x2_t b) { return mul(a, b); }
FFTN_FORCE_INLINE float32x4_t cmul(float32x4_t a, float32x4_t b) { return mul(a, b); }

} // namespace
