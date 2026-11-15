// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vocabulary of the scalar target: one complex number per complex register, one real per
// real register, nothing fused. Portable C++ with no intrinsics; the reference the other
// targets and the generator are tested against. Included inside FFTN_ISA_BEGIN(scalar);
// the contract is in vocabulary.inl.

#include "isa/vocabulary.inl"

namespace {

inline constexpr target id = target::scalar;
inline constexpr bool fused = false;

template <class T> struct cplx {
  T re;
  T im;
};

template <> struct vec<double> {
  using cv = cplx<double>;
  using rv = double;
  using tw = cplx<double>;
  static constexpr std::size_t C = 1;
  static constexpr std::size_t W = 1;
  static constexpr std::size_t split_block = W;
  static constexpr std::size_t tw_reals = 2;
};

template <> struct vec<float> {
  using cv = cplx<float>;
  using rv = float;
  using tw = cplx<float>;
  static constexpr std::size_t C = 1;
  static constexpr std::size_t W = 1;
  static constexpr std::size_t split_block = W;
  static constexpr std::size_t tw_reals = 2;
};

// Complex-register operations.

template <class T> FFTN_FORCE_INLINE cplx<T> load(const T* p) { return {p[0], p[1]}; }
template <class T> FFTN_FORCE_INLINE void store(T* p, cplx<T> a) {
  p[0] = a.re;
  p[1] = a.im;
}
template <class T> FFTN_FORCE_INLINE cplx<T> bcast(const T* p) { return {p[0], p[1]}; }
template <class T> FFTN_FORCE_INLINE cplx<T> load_tw(const T* p) { return {p[0], p[1]}; }
template <class T> FFTN_FORCE_INLINE void pack_tw(T* out, const T* p) {
  out[0] = p[0];
  out[1] = p[1];
}
template <class T> FFTN_FORCE_INLINE cplx<T> bcast_tw(const T* p) { return {p[0], p[1]}; }
template <class T> FFTN_FORCE_INLINE void pack_tw1(T* out, const T* p) {
  out[0] = p[0];
  out[1] = p[1];
}

template <class T> FFTN_FORCE_INLINE cplx<T> add(cplx<T> a, cplx<T> b) {
  return {a.re + b.re, a.im + b.im};
}
template <class T> FFTN_FORCE_INLINE cplx<T> sub(cplx<T> a, cplx<T> b) {
  return {a.re - b.re, a.im - b.im};
}
template <class T> FFTN_FORCE_INLINE cplx<T> addj(cplx<T> a, cplx<T> b) {
  return {a.re - b.im, a.im + b.re};
}
template <class T> FFTN_FORCE_INLINE cplx<T> subj(cplx<T> a, cplx<T> b) {
  return {a.re + b.im, a.im - b.re};
}
template <class T> FFTN_FORCE_INLINE cplx<T> mul(cplx<T> a, cplx<T> w) {
  return {a.re * w.re - a.im * w.im, a.re * w.im + a.im * w.re};
}
template <class T> FFTN_FORCE_INLINE cplx<T> mulc(cplx<T> a, cplx<T> w) {
  return {a.re * w.re + a.im * w.im, a.im * w.re - a.re * w.im};
}
template <class T> FFTN_FORCE_INLINE cplx<T> cmul(cplx<T> a, cplx<T> b) { return mul(a, b); }
template <class T> FFTN_FORCE_INLINE cplx<T> madd(cplx<T> acc, cplx<T> a, cplx<T> w) {
  return add(acc, mul(a, w));
}
template <class T> FFTN_FORCE_INLINE cplx<T> scale(cplx<T> a, T r) { return {r * a.re, r * a.im}; }
template <class T> FFTN_FORCE_INLINE cplx<T> fmadd_r(cplx<T> acc, cplx<T> a, T r) {
  return {acc.re + r * a.re, acc.im + r * a.im};
}
template <class T> FFTN_FORCE_INLINE cplx<T> neg(cplx<T> a) { return {-a.re, -a.im}; }
template <class T> FFTN_FORCE_INLINE cplx<T> conj(cplx<T> a) { return {a.re, -a.im}; }
template <class T> FFTN_FORCE_INLINE cplx<T> swap_ri(cplx<T> a) { return {a.im, a.re}; }
// One complex number per register: a 1 x 1 block is its own transpose.
template <class T> FFTN_FORCE_INLINE void transpose(cplx<T>* /*v*/) {}

// Real-register operations.
namespace real {

template <class T> FFTN_FORCE_INLINE T load(const T* p) { return *p; }
template <class T> FFTN_FORCE_INLINE void store(T* p, T a) { *p = a; }
template <class T> FFTN_FORCE_INLINE T bcast(T x) { return x; }
template <class T> FFTN_FORCE_INLINE T add(T a, T b) { return a + b; }
template <class T> FFTN_FORCE_INLINE T sub(T a, T b) { return a - b; }
template <class T> FFTN_FORCE_INLINE T mul(T a, T b) { return a * b; }
template <class T> FFTN_FORCE_INLINE T fmadd(T a, T b, T c) { return a * b + c; }
template <class T> FFTN_FORCE_INLINE T fmsub(T a, T b, T c) { return a * b - c; }
template <class T> FFTN_FORCE_INLINE T fnmadd(T a, T b, T c) { return c - a * b; }
template <class T> FFTN_FORCE_INLINE T neg(T a) { return -a; }
template <class T> FFTN_FORCE_INLINE void deinterleave(const T* p, T& re, T& im) {
  re = p[0];
  im = p[1];
}
template <class T> FFTN_FORCE_INLINE void interleave(T* p, T re, T im) {
  p[0] = re;
  p[1] = im;
}

} // namespace real

} // namespace
