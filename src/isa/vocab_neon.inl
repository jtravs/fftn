// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary of the neon target: AArch64 Advanced SIMD alone, the AArch64 baseline
// (Neoverse N1, Graviton 2, Ampere Altra, Cortex-A72; A64FX, which has FCMA but no pointer
// authentication). No complex instructions: products are shuffles and FMAs. 128-bit
// registers: one complex double or two complex floats per complex register. Included
// inside FFTN_ISA_BEGIN(neon); the contract is in vocabulary.inl.
//
// Every operation rounds exactly as neon_fcma's does, so the two targets give the same
// bits: the products below take the FCMLA order (the product with wr rounded, then the wi
// term fused onto it), and addj and subj, like FCADD, round one addition.

#include "isa/vocabulary.inl"

#include "isa/vocab_neon_common.inl"

namespace {

inline constexpr target id = target::neon;
inline constexpr bool fused = true;
inline constexpr bool masked = false;

// Twiddles pre-arranged as on x86 (DESIGN.md 6.2): r holds (wr, wr) and i holds (-wi, wi)
// for each lane's twiddle, so a product is one shuffle and two FMA-unit instructions with
// no sign fix-up; stored plainly as (wr, wi), it would need a sign flip of a shuffled
// value in every product. Special members trivial, defined inside the region.
struct tw_f64 {
  float64x2_t r;
  float64x2_t i;
};
struct tw_f32 {
  float32x4_t r;
  float32x4_t i;
};

template <> struct vec<double> {
  using cv = float64x2_t;
  using rv = float64x2_t;
  using tw = tw_f64;
  static constexpr std::size_t C = 1;
  static constexpr std::size_t W = 2;
  static constexpr std::size_t split_block = W;
  static constexpr std::size_t tw_reals = 4;
};

template <> struct vec<float> {
  using cv = float32x4_t;
  using rv = float32x4_t;
  using tw = tw_f32;
  static constexpr std::size_t C = 2;
  static constexpr std::size_t W = 4;
  static constexpr std::size_t split_block = W;
  static constexpr std::size_t tw_reals = 4;
};

// Complex-register operations (memory, add, sub, scale, fmadd_r, neg, conj, swap_ri,
// transpose and the real family are vocab_neon_common.inl's).

FFTN_FORCE_INLINE tw_f64 load_tw(const double* p) {
  return {.r = vld1q_f64(p), .i = vld1q_f64(p + 2)};
}
FFTN_FORCE_INLINE tw_f32 load_tw(const float* p) {
  return {.r = vld1q_f32(p), .i = vld1q_f32(p + 4)};
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

// (-1, 1) and (1, -1) in each complex number, for addj and subj.
FFTN_FORCE_INLINE float64x2_t signs_f64(double re) {
  return vtrn1q_f64(vdupq_n_f64(re), vdupq_n_f64(-re));
}
FFTN_FORCE_INLINE float32x4_t signs_f32(float re) {
  return vtrn1q_f32(vdupq_n_f32(re), vdupq_n_f32(-re));
}
// a + i*b = (ar - bi, ai + br): swap_ri(b) times (-1, 1), fused onto a. The product is
// exact, so this rounds once, exactly as an addition (and as FCADD).
FFTN_FORCE_INLINE float64x2_t addj(float64x2_t a, float64x2_t b) {
  return vfmaq_f64(a, swap_ri(b), signs_f64(-1.0));
}
FFTN_FORCE_INLINE float32x4_t addj(float32x4_t a, float32x4_t b) {
  return vfmaq_f32(a, swap_ri(b), signs_f32(-1.0F));
}
// a - i*b = (ar + bi, ai - br).
FFTN_FORCE_INLINE float64x2_t subj(float64x2_t a, float64x2_t b) {
  return vfmaq_f64(a, swap_ri(b), signs_f64(1.0));
}
FFTN_FORCE_INLINE float32x4_t subj(float32x4_t a, float32x4_t b) {
  return vfmaq_f32(a, swap_ri(b), signs_f32(1.0F));
}
// a*w = r*a + i*swap_ri(a): (ar*wr - ai*wi, ai*wr + ar*wi), the r product rounded first.
FFTN_FORCE_INLINE float64x2_t mul(float64x2_t a, tw_f64 w) {
  return vfmaq_f64(vmulq_f64(a, w.r), swap_ri(a), w.i);
}
FFTN_FORCE_INLINE float32x4_t mul(float32x4_t a, tw_f32 w) {
  return vfmaq_f32(vmulq_f32(a, w.r), swap_ri(a), w.i);
}
// a*conj(w) = r*a - i*swap_ri(a): (ar*wr + ai*wi, ai*wr - ar*wi).
FFTN_FORCE_INLINE float64x2_t mulc(float64x2_t a, tw_f64 w) {
  return vfmsq_f64(vmulq_f64(a, w.r), swap_ri(a), w.i);
}
FFTN_FORCE_INLINE float32x4_t mulc(float32x4_t a, tw_f32 w) {
  return vfmsq_f32(vmulq_f32(a, w.r), swap_ri(a), w.i);
}
FFTN_FORCE_INLINE float64x2_t madd(float64x2_t acc, float64x2_t a, tw_f64 w) {
  return vfmaq_f64(vfmaq_f64(acc, a, w.r), swap_ri(a), w.i);
}
FFTN_FORCE_INLINE float32x4_t madd(float32x4_t acc, float32x4_t a, tw_f32 w) {
  return vfmaq_f32(vfmaq_f32(acc, a, w.r), swap_ri(a), w.i);
}
// a*b for two complex registers: b rearranged into mul's twiddle form, (br, br) and
// (-bi, bi), in registers (the negation flips a sign bit, as pack_tw's does).
FFTN_FORCE_INLINE float64x2_t cmul(float64x2_t a, float64x2_t b) {
  const uint64x2_t even = vreinterpretq_u64_f64(vtrn1q_f64(vdupq_n_f64(-0.0), vdupq_n_f64(0.0)));
  const float64x2_t i =
      vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vdupq_laneq_f64(b, 1)), even));
  return mul(a, tw_f64{.r = vdupq_laneq_f64(b, 0), .i = i});
}
FFTN_FORCE_INLINE float32x4_t cmul(float32x4_t a, float32x4_t b) {
  const uint32x4_t even = vreinterpretq_u32_f32(vtrn1q_f32(vdupq_n_f32(-0.0F), vdupq_n_f32(0.0F)));
  const float32x4_t i =
      vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vtrn2q_f32(b, b)), even));
  return mul(a, tw_f32{.r = vtrn1q_f32(b, b), .i = i});
}

} // namespace
