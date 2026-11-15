// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The vocabulary operations both AArch64 targets (neon and neon_fcma) define the same way:
// memory, the exact complex operations that need no complex instruction, and the whole
// real-register family. Plain Advanced SIMD, part of the AArch64 base architecture.
// Included by vocab_neon.inl and vocab_neon_fcma.inl, inside their regions, after
// vocabulary.inl; each target adds its twiddle form and its complex products. No include
// guard.

namespace {

FFTN_FORCE_INLINE float64x2_t load(const double* p) { return vld1q_f64(p); }
FFTN_FORCE_INLINE float32x4_t load(const float* p) { return vld1q_f32(p); }
FFTN_FORCE_INLINE void store(double* p, float64x2_t a) { vst1q_f64(p, a); }
FFTN_FORCE_INLINE void store(float* p, float32x4_t a) { vst1q_f32(p, a); }
FFTN_FORCE_INLINE float64x2_t bcast(const double* p) { return vld1q_f64(p); }
FFTN_FORCE_INLINE float32x4_t bcast(const float* p) {
  const float32x2_t x = vld1_f32(p);
  return vcombine_f32(x, x);
}

FFTN_FORCE_INLINE float64x2_t add(float64x2_t a, float64x2_t b) { return vaddq_f64(a, b); }
FFTN_FORCE_INLINE float32x4_t add(float32x4_t a, float32x4_t b) { return vaddq_f32(a, b); }
FFTN_FORCE_INLINE float64x2_t sub(float64x2_t a, float64x2_t b) { return vsubq_f64(a, b); }
FFTN_FORCE_INLINE float32x4_t sub(float32x4_t a, float32x4_t b) { return vsubq_f32(a, b); }
FFTN_FORCE_INLINE float64x2_t scale(float64x2_t a, double r) { return vmulq_n_f64(a, r); }
FFTN_FORCE_INLINE float32x4_t scale(float32x4_t a, float r) { return vmulq_n_f32(a, r); }
FFTN_FORCE_INLINE float64x2_t fmadd_r(float64x2_t acc, float64x2_t a, double r) {
  return vfmaq_n_f64(acc, a, r);
}
FFTN_FORCE_INLINE float32x4_t fmadd_r(float32x4_t acc, float32x4_t a, float r) {
  return vfmaq_n_f32(acc, a, r);
}
FFTN_FORCE_INLINE float64x2_t neg(float64x2_t a) { return vnegq_f64(a); }
FFTN_FORCE_INLINE float32x4_t neg(float32x4_t a) { return vnegq_f32(a); }
// Flips the sign bit of the imaginary parts: one EOR with a constant (-0 in odd lanes).
FFTN_FORCE_INLINE float64x2_t conj(float64x2_t a) {
  const uint64x2_t odd = vreinterpretq_u64_f64(vtrn1q_f64(vdupq_n_f64(0.0), vdupq_n_f64(-0.0)));
  return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), odd));
}
FFTN_FORCE_INLINE float32x4_t conj(float32x4_t a) {
  const uint32x4_t odd = vreinterpretq_u32_f32(vtrn1q_f32(vdupq_n_f32(0.0F), vdupq_n_f32(-0.0F)));
  return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), odd));
}
FFTN_FORCE_INLINE float64x2_t swap_ri(float64x2_t a) { return vextq_f64(a, a, 1); }
FFTN_FORCE_INLINE float32x4_t swap_ri(float32x4_t a) { return vrev64q_f32(a); }
// One complex double per register: a 1 x 1 block is its own transpose.
FFTN_FORCE_INLINE void transpose(float64x2_t* /*v*/) {}
// Two complex floats per register: a 2 x 2 block, each complex number moved as one 64-bit
// lane.
FFTN_FORCE_INLINE void transpose(float32x4_t* v) {
  const float64x2_t a = vreinterpretq_f64_f32(v[0]);
  const float64x2_t b = vreinterpretq_f64_f32(v[1]);
  v[0] = vreinterpretq_f32_f64(vzip1q_f64(a, b));
  v[1] = vreinterpretq_f32_f64(vzip2q_f64(a, b));
}

// Real-register operations.
namespace real {

FFTN_FORCE_INLINE float64x2_t load(const double* p) { return vld1q_f64(p); }
FFTN_FORCE_INLINE float32x4_t load(const float* p) { return vld1q_f32(p); }
FFTN_FORCE_INLINE void store(double* p, float64x2_t a) { vst1q_f64(p, a); }
FFTN_FORCE_INLINE void store(float* p, float32x4_t a) { vst1q_f32(p, a); }
FFTN_FORCE_INLINE float64x2_t bcast(double x) { return vdupq_n_f64(x); }
FFTN_FORCE_INLINE float32x4_t bcast(float x) { return vdupq_n_f32(x); }
FFTN_FORCE_INLINE float64x2_t add(float64x2_t a, float64x2_t b) { return vaddq_f64(a, b); }
FFTN_FORCE_INLINE float32x4_t add(float32x4_t a, float32x4_t b) { return vaddq_f32(a, b); }
FFTN_FORCE_INLINE float64x2_t sub(float64x2_t a, float64x2_t b) { return vsubq_f64(a, b); }
FFTN_FORCE_INLINE float32x4_t sub(float32x4_t a, float32x4_t b) { return vsubq_f32(a, b); }
FFTN_FORCE_INLINE float64x2_t mul(float64x2_t a, float64x2_t b) { return vmulq_f64(a, b); }
FFTN_FORCE_INLINE float32x4_t mul(float32x4_t a, float32x4_t b) { return vmulq_f32(a, b); }
FFTN_FORCE_INLINE float64x2_t fmadd(float64x2_t a, float64x2_t b, float64x2_t c) {
  return vfmaq_f64(c, a, b);
}
FFTN_FORCE_INLINE float32x4_t fmadd(float32x4_t a, float32x4_t b, float32x4_t c) {
  return vfmaq_f32(c, a, b);
}
// No vector FNMSUB: a*b - c is -c + a*b, fused; negation is exact.
FFTN_FORCE_INLINE float64x2_t fmsub(float64x2_t a, float64x2_t b, float64x2_t c) {
  return vfmaq_f64(vnegq_f64(c), a, b);
}
FFTN_FORCE_INLINE float32x4_t fmsub(float32x4_t a, float32x4_t b, float32x4_t c) {
  return vfmaq_f32(vnegq_f32(c), a, b);
}
FFTN_FORCE_INLINE float64x2_t fnmadd(float64x2_t a, float64x2_t b, float64x2_t c) {
  return vfmsq_f64(c, a, b);
}
FFTN_FORCE_INLINE float32x4_t fnmadd(float32x4_t a, float32x4_t b, float32x4_t c) {
  return vfmsq_f32(c, a, b);
}
FFTN_FORCE_INLINE float64x2_t neg(float64x2_t a) { return vnegq_f64(a); }
FFTN_FORCE_INLINE float32x4_t neg(float32x4_t a) { return vnegq_f32(a); }
FFTN_FORCE_INLINE void deinterleave(const double* p, float64x2_t& re, float64x2_t& im) {
  const float64x2x2_t v = vld2q_f64(p);
  re = v.val[0];
  im = v.val[1];
}
FFTN_FORCE_INLINE void deinterleave(const float* p, float32x4_t& re, float32x4_t& im) {
  const float32x4x2_t v = vld2q_f32(p);
  re = v.val[0];
  im = v.val[1];
}
FFTN_FORCE_INLINE void interleave(double* p, float64x2_t re, float64x2_t im) {
  float64x2x2_t v;
  v.val[0] = re;
  v.val[1] = im;
  vst2q_f64(p, v);
}
FFTN_FORCE_INLINE void interleave(float* p, float32x4_t re, float32x4_t im) {
  float32x4x2_t v;
  v.val[0] = re;
  v.val[1] = im;
  vst2q_f32(p, v);
}

} // namespace real

} // namespace
