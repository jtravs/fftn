// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The edge passes of real transforms (DESIGN.md 9.1): packing reals into complex scratch,
// and the post-pass and pre-pass that pair bins k and h - k around a half-length complex
// transform. Kernel text in the manner of kernels.inl, included after access.inl. No
// include guard.
//
// These still run separately from the complex chain, one element at a time (bcast on
// loading, lane 0 stored, any stride on every target); fusing them into the first or last
// pass is phase 2. The output edges take the plan's scale (Scaled): every real they store
// is multiplied by it, exactly as if the transform were followed by a scaling pass. Pure
// data movement (packing reals, taking parts) is written as plain loads and stores of T.

namespace {

template <class T, bool Scaled> FFTN_FORCE_INLINE T scaled(T v, T s) {
  if constexpr (Scaled) {
    return v * s;
  } else {
    return v;
  }
}

template <class T, bool Scaled> FFTN_FORCE_INLINE cv<T> scaled(cv<T> v, T s) {
  if constexpr (Scaled) {
    return scale(v, s);
  } else {
    return v;
  }
}

// ---- Real to complex, n = 2h even: z_j = x_2j + i x_2j+1, Z = DFT_h(z), then the post-pass.

// z_j = (x[2j], x[2j+1]) for unit j < h. src is the real line, dst complex scratch.
template <class T> void r2c_pack_pairs(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  for (std::size_t j = first; j < last; ++j) {
    z[at(j, zs)] = x[at(2 * j, xs)];
    z[at(j, zs) + 1] = x[at(2 * j + 1, xs)];
  }
}

// X_k for k = 0..h from Z = DFT_h(z), unit u for the pair k = u, h - u (unit 0: bins 0
// and h). With E = (Z_k + conj Z_h-k) / 2, O = (Z_k - conj Z_h-k) / 2i and w = exp(-2 pi i
// / n):
//
//   X_k = E + w^k O,   X_h-k = conj(E - w^k O).
//
// twiddles: w^k for k = 0 .. h/2, broadcast storage. Reads src, n, twiddles, scale. Units
// h/2 + 1.
template <class T, bool Scaled>
void r2c_post(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t h = a->n / 2;
  const T* table = static_cast<const T*>(a->twiddles);
  const auto s = static_cast<T>(a->scale);
  for (std::size_t k = first; k < last; ++k) {
    if (k == 0) {
      // Bins 0 and h are real: Re Z_0 +- Im Z_0.
      const T re = z[0];
      const T im = z[1];
      y[0] = scaled<T, Scaled>(re + im, s);
      y[1] = scaled<T, Scaled>(T(0), s);
      y[at(h, ys)] = scaled<T, Scaled>(re - im, s);
      y[at(h, ys) + 1] = scaled<T, Scaled>(T(0), s);
      continue;
    }
    const cv<T> zk = bcast(z + at(k, zs));
    const cv<T> zc = conj(bcast(z + at(h - k, zs)));
    const cv<T> e = scale(add(zk, zc), T(0.5));
    // w^k O = -i (w^k (Z_k - conj Z_h-k) / 2); multiplying by -i is conj(swap_ri(.)).
    const cv<T> wo =
        conj(swap_ri(scale(mul(sub(zk, zc), bcast_tw(table + tw_reals<T> * k)), T(0.5))));
    store_first(y + at(k, ys), scaled<T, Scaled>(add(e, wo), s));
    if (k != h - k) {
      store_first(y + at(h - k, ys), scaled<T, Scaled>(conj(sub(e, wo)), s));
    }
  }
}

// ---- Real to complex, n odd: the full complex transform of (x_j, 0).

// z_j = (x[j], 0), unit j < n.
template <class T> void r2c_pack(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  for (std::size_t j = first; j < last; ++j) {
    z[at(j, zs)] = x[at(j, xs)];
    z[at(j, zs) + 1] = T(0);
  }
}

// X_k = Z_k for unit k <= n/2; the imaginary part of X_0 is exactly 0 (a sum of reals).
// Reads src, scale.
template <class T, bool Scaled>
void r2c_store(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const auto s = static_cast<T>(a->scale);
  for (std::size_t k = first; k < last; ++k) {
    y[at(k, ys)] = scaled<T, Scaled>(z[at(k, zs)], s);
    y[at(k, ys) + 1] = scaled<T, Scaled>(k == 0 ? T(0) : z[at(k, zs) + 1], s);
  }
}

// ---- Complex to real, n = 2h even: the pre-pass, Z = IDFT_h, then x_2j + i x_2j+1 = z_j.

// Z_k for k < h from X_0 .. X_h, unit u for the pair k = u, h - u (unit 0: Z_0 alone).
// With E = X_k + conj X_h-k and O = (X_k - conj X_h-k) exp(+2 pi i k / n):
//
//   Z_k = E + i O,   Z_h-k = conj(E - i O).
//
// The imaginary parts of X_0 and X_h are ignored. twiddles: the forward w^k = exp(-2 pi i
// k / n) for k = 0 .. h/2, broadcast storage, used conjugated. Reads src, n, twiddles.
// Units h/2 + 1.
template <class T> void c2r_pre(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  const std::size_t h = a->n / 2;
  const T* table = static_cast<const T*>(a->twiddles);
  for (std::size_t k = first; k < last; ++k) {
    if (k == 0) {
      const T x0 = x[0];
      const T xh = x[at(h, xs)];
      z[0] = x0 + xh;
      z[1] = x0 - xh;
      continue;
    }
    const cv<T> xk = bcast(x + at(k, xs));
    const cv<T> xc = conj(bcast(x + at(h - k, xs)));
    const cv<T> e = add(xk, xc);
    const cv<T> o = mulc(sub(xk, xc), bcast_tw(table + tw_reals<T> * k));
    store_first(z + at(k, zs), addj(e, o));
    if (k != h - k) {
      store_first(z + at(h - k, zs), conj(subj(e, o)));
    }
  }
}

// x[2j] = Re z_j, x[2j+1] = Im z_j, unit j < h. src complex scratch, dst the real line.
// Reads src, scale.
template <class T, bool Scaled>
void c2r_unpack_pairs(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* x = static_cast<T*>(a->dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t xs = a->dst_stride;
  const auto s = static_cast<T>(a->scale);
  for (std::size_t j = first; j < last; ++j) {
    x[at(2 * j, xs)] = scaled<T, Scaled>(z[at(j, zs)], s);
    x[at(2 * j + 1, xs)] = scaled<T, Scaled>(z[at(j, zs) + 1], s);
  }
}

// ---- Complex to real, n odd: the full Hermitian spectrum, then its complex transform.

// Unit 0: Z_0 = Re X_0 (its imaginary part is ignored). Unit k in 1 .. (n-1)/2: Z_k = X_k
// and Z_n-k = conj X_k. Reads src, n. Units n/2 + 1.
template <class T> void c2r_expand(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  const std::size_t n = a->n;
  for (std::size_t k = first; k < last; ++k) {
    if (k == 0) {
      z[0] = x[0];
      z[1] = T(0);
      continue;
    }
    const T re = x[at(k, xs)];
    const T im = x[at(k, xs) + 1];
    z[at(k, zs)] = re;
    z[at(k, zs) + 1] = im;
    z[at(n - k, zs)] = re;
    z[at(n - k, zs) + 1] = -im;
  }
}

// x[j] = Re z_j, unit j < n. Reads src, scale.
template <class T, bool Scaled>
void c2r_store(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* x = static_cast<T*>(a->dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t xs = a->dst_stride;
  const auto s = static_cast<T>(a->scale);
  for (std::size_t j = first; j < last; ++j) {
    x[at(j, xs)] = scaled<T, Scaled>(z[at(j, zs)], s);
  }
}

} // namespace
