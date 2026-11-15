// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The two edge passes of a Rader stage (DESIGN.md 8.5, experiment X27; C. M. Rader, Proc.
// IEEE 56, 1968): for a prime q with primitive root g, the transform's bins other than 0
// are a cyclic convolution of length L = q - 1,
//
//   X_(g^-k) = x_0 + sum_m x_(g^m) w^(g^(m - k)),   k, m < L,  w = exp(sign 2 pi i / q),
//
// of a_m = x_(g^m) with the fixed b_l = w^(g^-l), and X_0 = x_0 + sum_(j >= 1) x_j. The pass
// in gathers a (and keeps x_0 and the sum), the convolution's chain multiplies by the
// kernel spectrum dft(b) / L as a Bluestein stage's does, and the pass out scatters
// x_0 + c_k to bin g^-k, which for m = -k mod L is bin g^m. Like a Bluestein stage it is the
// last pass of a chain (m = 1): element t + s j of the line is copy t, sequence index j.
// Kernel text in the manner of kernels.inl, included after bluestein.inl. No include guard.
//
// The index table (pass_args::roots) holds g^m mod q for m < L as values of T (exact: the
// rule admits q < 2^24), read by both passes. Units. Shared (C divides s): a unit is C
// adjacent copies through all L + 1 sequence indices, one vector per index. Otherwise a
// unit is one copy, in plain scalar arithmetic, which is what a single prime row (s = 1)
// takes: its gather and scatter go one element at a time anyway. The scratch line holds
// the convolution's s L elements, then x_0 and the sum (two more rows of s).

namespace {

// The sum of L values with the rounding error of a pairwise sum: values are summed
// sequentially in blocks of 16, and the block sums merged as a binary counter merges
// carries, so a sum sees about 16 + log2(L / 16) roundings rather than L (a sequential sum
// fails the tolerances at a few hundred points: the generic pass's decision, DESIGN.md
// section 19). V is cv<T> (shared units) or T (one part of a scalar sum).
template <class V> struct rader_sum {
  V level[24]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  V block;
  std::size_t in_block;
  std::size_t blocks;
};

template <class V> FFTN_FORCE_INLINE void rader_sum_start(rader_sum<V>& r, V zero) {
  r.block = zero;
  r.in_block = 0;
  r.blocks = 0;
}

FFTN_FORCE_INLINE double rader_plus(double a, double b) { return a + b; }
FFTN_FORCE_INLINE float rader_plus(float a, float b) { return a + b; }
FFTN_FORCE_INLINE cv<double> rader_plus(cv<double> a, cv<double> b) { return add(a, b); }
FFTN_FORCE_INLINE cv<float> rader_plus(cv<float> a, cv<float> b) { return add(a, b); }

template <class V> FFTN_FORCE_INLINE void rader_sum_add(rader_sum<V>& r, V v, V zero) {
  r.block = rader_plus(r.block, v);
  if (++r.in_block < 16) {
    return;
  }
  V carry = r.block;
  std::size_t b = r.blocks;
  std::size_t lvl = 0;
  while ((b & 1U) != 0) {
    carry = rader_plus(r.level[lvl], carry);
    b >>= 1U;
    ++lvl;
  }
  r.level[lvl] = carry;
  ++r.blocks;
  r.block = zero;
  r.in_block = 0;
}

template <class V> FFTN_FORCE_INLINE V rader_sum_total(const rader_sum<V>& r) {
  V total = r.block;
  std::size_t b = r.blocks;
  for (std::size_t lvl = 0; b != 0; ++lvl, b >>= 1U) {
    if ((b & 1U) != 0) {
      total = rader_plus(r.level[lvl], total);
    }
  }
  return total;
}

// An edge operation on one complex value in scalar arithmetic (the scalar units): times the
// scale, or times element i of the factor array.
template <class T, edge_op Op>
FFTN_FORCE_INLINE void rader_edge(const pass_args* a, const void* factor, std::size_t i, T& re,
                                  T& im) {
  if constexpr (Op == edge_op::scale) {
    const auto sc = static_cast<T>(a->scale);
    re *= sc;
    im *= sc;
  } else if constexpr (Op == edge_op::factor) {
    const T* f = static_cast<const T*>(factor) + at(i, 2);
    const T r = re * f[0] - im * f[1];
    im = re * f[1] + im * f[0];
    re = r;
  }
}

// z[t + s m] = in(x[t + s g^m]) for m < L, z[t + s L] = in(x[t]), z[t + s (L + 1)] = the sum
// of in(x[t + s j]) over 1 <= j < q, for each copy t of the unit. Arguments as a Bluestein
// stage's edge passes (radix 1): n = s q, m = L, s, roots = the index table. The input
// edge's operation applies to what is loaded. src: any stride when C == 1, else 1; dst:
// scratch, stride 1, not src. Reads src, dst, src_stride, m, s, roots, scale, in_factor.
template <class T, edge_op InOp, bool Shared>
void rader_in(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::size_t s = a->s;
  const std::size_t len = a->m;
  const T* index = static_cast<const T*>(a->roots);
  if constexpr (Shared) {
    const edge<T, InOp> op = make_edge<T, InOp>(a, a->in_factor, 0, 1);
    rader_sum<cv<T>> sum;
    for (std::size_t u = first; u < last; ++u) {
      const std::size_t t = u * C<T>;
      rader_sum_start(sum, zeros<T>());
      for (std::size_t m = 0; m < len; ++m) {
        const std::size_t i = t + s * static_cast<std::size_t>(index[m]);
        const cv<T> v =
            on_run<T, InOp, true>(op, load_run<T, true>(x + at(i, xs), xs, C<T>), i, C<T>);
        store(z + at(t + s * m, 2), v);
        rader_sum_add(sum, v, zeros<T>());
      }
      store(z + at(t + s * len, 2),
            on_run<T, InOp, true>(op, load_run<T, true>(x + at(t, xs), xs, C<T>), t, C<T>));
      store(z + at(t + s * (len + 1), 2), rader_sum_total(sum));
    }
  } else {
    rader_sum<T> sre;
    rader_sum<T> sim;
    for (std::size_t t = first; t < last; ++t) {
      rader_sum_start(sre, T(0));
      rader_sum_start(sim, T(0));
      for (std::size_t m = 0; m < len; ++m) {
        const std::size_t i = t + s * static_cast<std::size_t>(index[m]);
        T re = x[at(i, xs)];
        T im = x[at(i, xs) + 1];
        rader_edge<T, InOp>(a, a->in_factor, i, re, im);
        z[at(t + s * m, 2)] = re;
        z[at(t + s * m, 2) + 1] = im;
        rader_sum_add(sre, re, T(0));
        rader_sum_add(sim, im, T(0));
      }
      T re = x[at(t, xs)];
      T im = x[at(t, xs) + 1];
      rader_edge<T, InOp>(a, a->in_factor, t, re, im);
      z[at(t + s * len, 2)] = re;
      z[at(t + s * len, 2) + 1] = im;
      z[at(t + s * (len + 1), 2)] = rader_sum_total(sre);
      z[at(t + s * (len + 1), 2) + 1] = rader_sum_total(sim);
    }
  }
}

// y[t] = out(x_0 + sum), y[t + s g^m] = out(x_0 + c_((L - m) mod L)) for m < L, where c is
// the convolution in the first s L elements of src and x_0 and the sum are the two rows
// after it (rader_in). Units, arguments: as rader_in's. The output edge's operation applies
// to what is stored. src: scratch, stride 1; dst: any stride when C == 1, else 1; dst may
// be the stage's source line (rader_in has read it), never src. Reads src, dst, dst_stride,
// m, s, roots, scale, out_factor.
template <class T, edge_op OutOp, bool Shared>
void rader_out(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t s = a->s;
  const std::size_t len = a->m;
  const T* index = static_cast<const T*>(a->roots);
  if constexpr (Shared) {
    const edge<T, OutOp> op = make_edge<T, OutOp>(a, a->out_factor, 0, 1);
    for (std::size_t u = first; u < last; ++u) {
      const std::size_t t = u * C<T>;
      const cv<T> x0 = load(z + at(t + s * len, 2));
      const cv<T> total = add(x0, load(z + at(t + s * (len + 1), 2)));
      store_run<T, true>(y + at(t, ys), ys, C<T>, on_run<T, OutOp, true>(op, total, t, C<T>));
      for (std::size_t m = 0; m < len; ++m) {
        const std::size_t k = m == 0 ? 0 : len - m;
        const std::size_t i = t + s * static_cast<std::size_t>(index[m]);
        const cv<T> v = add(x0, load(z + at(t + s * k, 2)));
        store_run<T, true>(y + at(i, ys), ys, C<T>, on_run<T, OutOp, true>(op, v, i, C<T>));
      }
    }
  } else {
    for (std::size_t t = first; t < last; ++t) {
      const T x0r = z[at(t + s * len, 2)];
      const T x0i = z[at(t + s * len, 2) + 1];
      T re = x0r + z[at(t + s * (len + 1), 2)];
      T im = x0i + z[at(t + s * (len + 1), 2) + 1];
      rader_edge<T, OutOp>(a, a->out_factor, t, re, im);
      y[at(t, ys)] = re;
      y[at(t, ys) + 1] = im;
      for (std::size_t m = 0; m < len; ++m) {
        const std::size_t k = m == 0 ? 0 : len - m;
        const std::size_t i = t + s * static_cast<std::size_t>(index[m]);
        re = x0r + z[at(t + s * k, 2)];
        im = x0i + z[at(t + s * k, 2) + 1];
        rader_edge<T, OutOp>(a, a->out_factor, i, re, im);
        y[at(i, ys)] = re;
        y[at(i, ys) + 1] = im;
      }
    }
  }
}

} // namespace
