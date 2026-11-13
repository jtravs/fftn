// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The passes of the chain strategy (DESIGN.md 8.5): radix-2, 3, 4 and 5 Stockham passes,
// the generic pass for any other prime, a copy, and the edge passes of real transforms
// (DESIGN.md 9.1). Kernel text in the manner of kernels.inl: included inside a region after
// its vocabulary, written against the vocabulary, entry points take pointers and integers
// (passes/pass_args.hpp), everything else is force-inlined. No include guard.
//
// These are the walking skeleton's kernels: correct for every length, simple, and slow.
// They assume one complex number per register (C<T> == 1), so only the scalar region
// includes them; generated codelets and lane-aware pass loops replace them in phase 1.
// Arithmetic goes through the vocabulary so that the operation order is the one the
// SIMD versions will have. Pure data movement (packing reals, taking parts) is written as
// plain loads and stores of T.
//
// Signs follow FFTW: Sign = -1 is the forward transform, exp(-2 pi i jk / n). Twiddle
// tables always hold forward twiddles; backward passes multiply by their conjugates (mulc),
// which is exact, so a backward table is the exact conjugate of the forward one.

namespace {

static_assert(C<double> == 1 && C<float> == 1,
              "the walking skeleton's passes handle one complex number per register");

// The butterfly constants, each rounded once to T from 36 significant digits.
template <class T> struct butterfly_constants;
template <> struct butterfly_constants<double> {
  static constexpr double sin60 = 0.866025403784438646763723170752936183;
  static constexpr double cos72 = 0.309016994374947424102293417182819059;
  static constexpr double cos144 = -0.809016994374947424102293417182819059;
  static constexpr double sin72 = 0.951056516295153572116439333379382143;
  static constexpr double sin144 = 0.587785252292473129168705954639072769;
};
template <> struct butterfly_constants<float> {
  static constexpr float sin60 = 0.866025403784438646763723170752936183F;
  static constexpr float cos72 = 0.309016994374947424102293417182819059F;
  static constexpr float cos144 = -0.809016994374947424102293417182819059F;
  static constexpr float sin72 = 0.951056516295153572116439333379382143F;
  static constexpr float sin144 = 0.587785252292473129168705954639072769F;
};

FFTN_FORCE_INLINE std::ptrdiff_t at(std::size_t index, std::ptrdiff_t stride) {
  return static_cast<std::ptrdiff_t>(index) * stride;
}

// a + i b (Sign > 0) or a - i b (Sign < 0): the rotation a radix-R butterfly applies.
template <class T, int Sign> FFTN_FORCE_INLINE cv<T> add_rot(cv<T> a, cv<T> b) {
  if constexpr (Sign < 0) {
    return subj(a, b);
  } else {
    return addj(a, b);
  }
}

// a * w (forward) or a * conj(w) (backward), w a forward twiddle.
template <class T, int Sign> FFTN_FORCE_INLINE cv<T> twiddle(cv<T> a, tw<T> w) {
  if constexpr (Sign < 0) {
    return mul(a, w);
  } else {
    return mulc(a, w);
  }
}

// The DFT of R values in registers, in place: v[k] <- sum_j v[j] exp(Sign 2 pi i jk / R).
template <class T, std::size_t R, int Sign> FFTN_FORCE_INLINE void small_dft(cv<T>* v) {
  using k = butterfly_constants<T>;
  if constexpr (R == 2) {
    const cv<T> a = v[0];
    v[0] = add(a, v[1]);
    v[1] = sub(a, v[1]);
  } else if constexpr (R == 3) {
    // v0 - (v1 + v2) / 2 +- i (sqrt 3 / 2)(v1 - v2)
    const cv<T> t = add(v[1], v[2]);
    const cv<T> u = scale(sub(v[1], v[2]), k::sin60);
    const cv<T> mid = fmadd_r(v[0], t, T(-0.5));
    v[0] = add(v[0], t);
    v[1] = add_rot<T, Sign>(mid, u);
    v[2] = add_rot<T, -Sign>(mid, u);
  } else if constexpr (R == 4) {
    const cv<T> t0 = add(v[0], v[2]);
    const cv<T> t1 = sub(v[0], v[2]);
    const cv<T> t2 = add(v[1], v[3]);
    const cv<T> t3 = sub(v[1], v[3]);
    v[0] = add(t0, t2);
    v[2] = sub(t0, t2);
    v[1] = add_rot<T, Sign>(t1, t3);
    v[3] = add_rot<T, -Sign>(t1, t3);
  } else {
    static_assert(R == 5);
    // With t1 = v1 + v4, t2 = v2 + v3, u1 = v1 - v4, u2 = v2 - v3:
    //   v1, v4 = v0 + cos72 t1 + cos144 t2 +- i (sin72 u1 + sin144 u2)
    //   v2, v3 = v0 + cos144 t1 + cos72 t2 +- i (sin144 u1 - sin72 u2)
    const cv<T> t1 = add(v[1], v[4]);
    const cv<T> t2 = add(v[2], v[3]);
    const cv<T> u1 = sub(v[1], v[4]);
    const cv<T> u2 = sub(v[2], v[3]);
    const cv<T> m1 = fmadd_r(fmadd_r(v[0], t1, k::cos72), t2, k::cos144);
    const cv<T> m2 = fmadd_r(fmadd_r(v[0], t1, k::cos144), t2, k::cos72);
    const cv<T> w1 = fmadd_r(scale(u1, k::sin72), u2, k::sin144);
    const cv<T> w2 = fmadd_r(scale(u1, k::sin144), u2, -k::sin72);
    v[0] = add(v[0], add(t1, t2));
    v[1] = add_rot<T, Sign>(m1, w1);
    v[4] = add_rot<T, -Sign>(m1, w1);
    v[2] = add_rot<T, Sign>(m2, w2);
    v[3] = add_rot<T, -Sign>(m2, w2);
  }
}

// One radix-R pass of a Stockham autosort chain in decimation-in-frequency form (DESIGN.md
// 8.1). For a line of n = R m s complex values, butterfly group p < m and copy q < s:
//
//   y[q + s (R p + k)] = w^(p k) sum_j x[q + s (p + m j)] exp(Sign 2 pi i jk / R),
//
// w = exp(Sign 2 pi i / (R m)). Unit b is the butterfly (p, q) = (b / s, b % s), so the
// copies q, which share twiddles, are innermost. twiddles: R - 1 per group p, the forward
// w^(p k) for k = 1 .. R-1. Reads src, radix, m, s, twiddles.
template <class T, std::size_t R, int Sign>
void stockham_pass(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t m = a->m;
  const std::size_t s = a->s;
  const T* table = static_cast<const T*>(a->twiddles);
  std::size_t b = first;
  while (b < last) {
    const std::size_t p = b / s;
    const std::size_t q0 = b % s;
    const std::size_t q1 = last - b < s - q0 ? q0 + (last - b) : s;
    const T* wp = table + tw_reals<T> * (R - 1) * p;
    FFTN_SIMD_LOOP
    for (std::size_t q = q0; q < q1; ++q) {
      cv<T> v[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      for (std::size_t j = 0; j < R; ++j) {
        v[j] = load(x + at(q + s * (p + m * j), xs));
      }
      small_dft<T, R, Sign>(v);
      store(y + at(q + s * R * p, ys), v[0]);
      for (std::size_t k = 1; k < R; ++k) {
        const cv<T> z = twiddle<T, Sign>(v[k], load_tw(wp + tw_reals<T> * (k - 1)));
        store(y + at(q + s * (R * p + k), ys), z);
      }
    }
    b += q1 - q0;
  }
}

// The same pass for any radix R = a->radix by direct evaluation, O(R^2) per butterfly: the
// route for prime factors above 5 (DESIGN.md 9.2). Each output sums its R products in a
// pairwise tree (a binary counter of partial sums), so its rounding error grows like
// log R rather than R: a sequential sum fails the tolerances from R of a few hundred.
// roots: the R forward roots exp(-2 pi i t / R), t < R, in twiddle storage. Reads src,
// radix, m, s, twiddles, roots. The source must not overlap the destination.
template <class T, int Sign>
void generic_pass(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t r = a->radix;
  const std::size_t m = a->m;
  const std::size_t s = a->s;
  const T* table = static_cast<const T*>(a->twiddles);
  const T* roots = static_cast<const T*>(a->roots);
  constexpr std::size_t levels = 64; // partial sums of 2^level terms; r < 2^64
  for (std::size_t b = first; b < last; ++b) {
    const std::size_t p = b / s;
    const std::size_t q = b % s;
    const T* wp = table + tw_reals<T> * (r - 1) * p;
    for (std::size_t k = 0; k < r; ++k) {
      cv<T> partial[levels]; // NOLINT(modernize-avoid-c-arrays): no standard containers here
      std::size_t t = 0;     // j k mod r
      for (std::size_t j = 0; j < r; ++j) {
        cv<T> term = load(x + at(q + s * (p + m * j), xs));
        if (t != 0) {
          term = twiddle<T, Sign>(term, load_tw(roots + tw_reals<T> * t));
        }
        std::size_t level = 0;
        for (std::size_t c = j; (c & 1U) != 0; c >>= 1U) {
          term = add(partial[level], term);
          ++level;
        }
        partial[level] = term;
        t += k;
        if (t >= r) {
          t -= r;
        }
      }
      // The partial sums left over are those of the set bits of r, smallest first.
      std::size_t level = 0;
      while (((r >> level) & 1U) == 0) {
        ++level;
      }
      cv<T> sum = partial[level];
      for (++level; level < levels && (r >> level) != 0; ++level) {
        if (((r >> level) & 1U) != 0) {
          sum = add(partial[level], sum);
        }
      }
      if (k != 0) {
        sum = twiddle<T, Sign>(sum, load_tw(wp + tw_reals<T> * (k - 1)));
      }
      store(y + at(q + s * (r * p + k), ys), sum);
    }
  }
}

// y[i] = x[i], unit i. Reads src.
template <class T> void copy_pass(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  FFTN_SIMD_LOOP
  for (std::size_t i = first; i < last; ++i) {
    store(y + at(i, ys), load(x + at(i, xs)));
  }
}

// ---- Real to complex, n = 2h even: z_j = x_2j + i x_2j+1, Z = DFT_h(z), then the post-pass.

// z_j = (x[2j], x[2j+1]) for unit j < h. src is the real line, dst complex scratch.
template <class T> void r2c_pack_pairs(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  FFTN_SIMD_LOOP
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
// twiddles: w^k for k = 0 .. h/2. Reads src, n, twiddles. Units h/2 + 1.
template <class T> void r2c_post(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t h = a->n / 2;
  const T* table = static_cast<const T*>(a->twiddles);
  FFTN_SIMD_LOOP
  for (std::size_t k = first; k < last; ++k) {
    if (k == 0) {
      // Bins 0 and h are real: Re Z_0 +- Im Z_0.
      const T re = z[0];
      const T im = z[1];
      y[0] = re + im;
      y[1] = T(0);
      y[at(h, ys)] = re - im;
      y[at(h, ys) + 1] = T(0);
      continue;
    }
    const cv<T> zk = load(z + at(k, zs));
    const cv<T> zc = conj(load(z + at(h - k, zs)));
    const cv<T> e = scale(add(zk, zc), T(0.5));
    // w^k O = -i (w^k (Z_k - conj Z_h-k) / 2); multiplying by -i is conj(swap_ri(.)).
    const cv<T> wo =
        conj(swap_ri(scale(mul(sub(zk, zc), load_tw(table + tw_reals<T> * k)), T(0.5))));
    store(y + at(k, ys), add(e, wo));
    if (k != h - k) {
      store(y + at(h - k, ys), conj(sub(e, wo)));
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
  FFTN_SIMD_LOOP
  for (std::size_t j = first; j < last; ++j) {
    z[at(j, zs)] = x[at(j, xs)];
    z[at(j, zs) + 1] = T(0);
  }
}

// X_k = Z_k for unit k <= n/2; the imaginary part of X_0 is exactly 0 (a sum of reals).
template <class T> void r2c_store(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  FFTN_SIMD_LOOP
  for (std::size_t k = first; k < last; ++k) {
    store(y + at(k, ys), load(z + at(k, zs)));
  }
  if (first == 0 && last > 0) {
    y[1] = T(0);
  }
}

// ---- Complex to real, n = 2h even: the pre-pass, Z = IDFT_h, then x_2j + i x_2j+1 = z_j.

// Z_k for k < h from X_0 .. X_h, unit u for the pair k = u, h - u (unit 0: Z_0 alone).
// With E = X_k + conj X_h-k and O = (X_k - conj X_h-k) exp(+2 pi i k / n):
//
//   Z_k = E + i O,   Z_h-k = conj(E - i O).
//
// The imaginary parts of X_0 and X_h are ignored. twiddles: the forward w^k = exp(-2 pi i
// k / n) for k = 0 .. h/2, used conjugated. Reads src, n, twiddles. Units h/2 + 1.
template <class T> void c2r_pre(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  const std::size_t h = a->n / 2;
  const T* table = static_cast<const T*>(a->twiddles);
  FFTN_SIMD_LOOP
  for (std::size_t k = first; k < last; ++k) {
    if (k == 0) {
      const T x0 = x[0];
      const T xh = x[at(h, xs)];
      z[0] = x0 + xh;
      z[1] = x0 - xh;
      continue;
    }
    const cv<T> xk = load(x + at(k, xs));
    const cv<T> xc = conj(load(x + at(h - k, xs)));
    const cv<T> e = add(xk, xc);
    const cv<T> o = mulc(sub(xk, xc), load_tw(table + tw_reals<T> * k));
    store(z + at(k, zs), addj(e, o));
    if (k != h - k) {
      store(z + at(h - k, zs), conj(subj(e, o)));
    }
  }
}

// x[2j] = Re z_j, x[2j+1] = Im z_j, unit j < h. src complex scratch, dst the real line.
template <class T> void c2r_unpack_pairs(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* x = static_cast<T*>(a->dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t xs = a->dst_stride;
  FFTN_SIMD_LOOP
  for (std::size_t j = first; j < last; ++j) {
    x[at(2 * j, xs)] = z[at(j, zs)];
    x[at(2 * j + 1, xs)] = z[at(j, zs) + 1];
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
  FFTN_SIMD_LOOP
  for (std::size_t k = first; k < last; ++k) {
    if (k == 0) {
      z[0] = x[0];
      z[1] = T(0);
      continue;
    }
    const cv<T> xk = load(x + at(k, xs));
    store(z + at(k, zs), xk);
    store(z + at(n - k, zs), conj(xk));
  }
}

// x[j] = Re z_j, unit j < n.
template <class T> void c2r_store(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* x = static_cast<T*>(a->dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t xs = a->dst_stride;
  FFTN_SIMD_LOOP
  for (std::size_t j = first; j < last; ++j) {
    x[at(j, xs)] = z[at(j, zs)];
  }
}

template <class T>
inline constexpr chain_kernels<T> chain_entries{
    .radix2 = {.forward = &stockham_pass<T, 2, -1>, .backward = &stockham_pass<T, 2, +1>},
    .radix3 = {.forward = &stockham_pass<T, 3, -1>, .backward = &stockham_pass<T, 3, +1>},
    .radix4 = {.forward = &stockham_pass<T, 4, -1>, .backward = &stockham_pass<T, 4, +1>},
    .radix5 = {.forward = &stockham_pass<T, 5, -1>, .backward = &stockham_pass<T, 5, +1>},
    .generic = {.forward = &generic_pass<T, -1>, .backward = &generic_pass<T, +1>},
    .copy = &copy_pass<T>,
    .r2c_pack_pairs = &r2c_pack_pairs<T>,
    .r2c_post = &r2c_post<T>,
    .r2c_pack = &r2c_pack<T>,
    .r2c_store = &r2c_store<T>,
    .c2r_pre = &c2r_pre<T>,
    .c2r_unpack_pairs = &c2r_unpack_pairs<T>,
    .c2r_expand = &c2r_expand<T>,
    .c2r_store = &c2r_store<T>,
};

} // namespace
