// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Kernel text, written once and included inside every target's region after its
// vocabulary (DESIGN.md 6.4). No include guard. It uses only the vocabulary
// (vocabulary.inl) and plain scalar C++: no standard-library templates, no lambdas. Entry
// points take only pointers and integers; helpers that handle vectors are force-inlined.
// The kernels are the phase 0 vehicle for the mechanism; the walking skeleton and the
// generated codelets replace them.
//
// Signs follow FFTW: sign = -1 is the forward transform, exp(-2 pi i jk / n); sign = +1 the
// backward one. Twiddle tables always hold forward twiddles; backward passes multiply by
// their conjugates.

namespace {

// Radix-4 decimation-in-frequency pass (the first Stockham stage) of a length n = 4m:
//
//   y[4p + j] = w^(jp) * sum_l exp(sign 2 pi i jl / 4) x[p + l m],  w = exp(sign 2 pi i / n)
//
// for p < m and j < 4. x and y are interleaved complex arrays of n numbers. twiddles holds the
// forward twiddles exp(-2 pi i jp / n) as three rows j = 1, 2, 3 of m twiddles each, in
// this target's storage (pack_twiddles). Vectorised along p: the four outputs of C
// consecutive p are transposed in registers into contiguous stores. Needs m % C == 0 and
// 4 % C == 0.
template <class T, int Sign>
FFTN_FORCE_INLINE void radix4_pass_impl(const T* x, T* y, const T* twiddles, std::size_t m) {
  for (std::size_t p = 0; p < m; p += C<T>) {
    const cv<T> a = load(x + 2 * p);
    const cv<T> b = load(x + 2 * (p + m));
    const cv<T> c = load(x + 2 * (p + 2 * m));
    const cv<T> d = load(x + 2 * (p + 3 * m));
    const cv<T> apc = add(a, c);
    const cv<T> amc = sub(a, c);
    const cv<T> bpd = add(b, d);
    const cv<T> bmd = sub(b, d);
    cv<T> v[4]; // NOLINT(modernize-avoid-c-arrays): std::array is not allowed in a region
    v[0] = add(apc, bpd);
    v[2] = sub(apc, bpd);
    if constexpr (Sign < 0) {
      v[1] = subj(amc, bmd);
      v[3] = addj(amc, bmd);
    } else {
      v[1] = addj(amc, bmd);
      v[3] = subj(amc, bmd);
    }
    for (std::size_t j = 1; j < 4; ++j) {
      const tw<T> w = load_tw(twiddles + tw_reals<T> * ((j - 1) * m + p));
      if constexpr (Sign < 0) {
        v[j] = mul(v[j], w);
      } else {
        v[j] = mulc(v[j], w);
      }
    }
    // Block b holds outputs j = bC .. bC + C - 1 for the C values of p; after the transpose
    // its register r holds them for p + r, which are contiguous in y.
    for (std::size_t blk = 0; blk < 4 / C<T>; ++blk) {
      transpose(v + blk * C<T>);
      for (std::size_t r = 0; r < C<T>; ++r) {
        store(y + 2 * (4 * (p + r) + blk * C<T>), v[blk * C<T> + r]);
      }
    }
  }
}

template <class T> void radix4_pass(const T* x, T* y, const T* twiddles, std::size_t m, int sign) {
  if (sign < 0) {
    radix4_pass_impl<T, -1>(x, y, twiddles, m);
  } else {
    radix4_pass_impl<T, +1>(x, y, twiddles, m);
  }
}

// Radix-2 decimation-in-time butterflies of a length n = 2m, in split arithmetic:
//
//   y[k] = x[k] + w^k x[k + m],  y[k + m] = x[k] - w^k x[k + m],  w = exp(sign 2 pi i / n)
//
// for k < m. x and y are interleaved; the forward twiddles exp(-2 pi i k / n) are split,
// real parts in twr and imaginary parts in twi (m each). Needs m % W == 0.
template <class T, int Sign>
FFTN_FORCE_INLINE void radix2_split_impl(const T* x, T* y, const T* twr, const T* twi,
                                         std::size_t m) {
  for (std::size_t k = 0; k < m; k += W<T>) {
    rv<T> ar;
    rv<T> ai;
    rv<T> br;
    rv<T> bi;
    real::deinterleave(x + 2 * k, ar, ai);
    real::deinterleave(x + 2 * (k + m), br, bi);
    const rv<T> wr = real::load(twr + k);
    const rv<T> wi = real::load(twi + k);
    rv<T> tr;
    rv<T> ti;
    if constexpr (Sign < 0) { // b * w
      tr = real::fmsub(br, wr, real::mul(bi, wi));
      ti = real::fmadd(br, wi, real::mul(bi, wr));
    } else { // b * conj(w)
      tr = real::fmadd(br, wr, real::mul(bi, wi));
      ti = real::fnmadd(br, wi, real::mul(bi, wr));
    }
    real::interleave(y + 2 * k, real::add(ar, tr), real::add(ai, ti));
    real::interleave(y + 2 * (k + m), real::sub(ar, tr), real::sub(ai, ti));
  }
}

template <class T>
void radix2_split_pass(const T* x, T* y, const T* twr, const T* twi, std::size_t m, int sign) {
  if (sign < 0) {
    radix2_split_impl<T, -1>(x, y, twr, twi, m);
  } else {
    radix2_split_impl<T, +1>(x, y, twr, twi, m);
  }
}

// Writes count interleaved twiddles w as this target's twiddle storage (tw_reals<T> reals
// each). Run once, when a plan's tables are built. Needs count % C == 0.
template <class T> void pack_twiddles(const T* w, std::size_t count, T* out) {
  for (std::size_t k = 0; k < count; k += C<T>) {
    pack_tw(out + tw_reals<T> * k, w + 2 * k);
  }
}

} // namespace
