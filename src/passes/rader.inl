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
// Units. Shared (C divides s): a unit is C adjacent copies through all L + 1 sequence
// indices, one vector per index. Otherwise a unit is one copy, one element at a time
// (broadcast on loading, lane 0 stored), which is what a single prime row (s = 1) takes; the
// passes are O(q) against the convolution's O(q log q). The scratch line holds the
// convolution's s L elements, then x_0 and the sum (two more rows of s).

namespace {

// g^m mod q to g^(m + 1) mod q. g is the smallest primitive root, small for every prime
// the rule admits (at most 11 below 1000), so the loop is short and needs no division.
FFTN_FORCE_INLINE std::size_t rader_next(std::size_t idx, std::size_t g, std::size_t q) {
  idx *= g;
  while (idx >= q) {
    idx -= q;
  }
  return idx;
}

// The sum of the gathered values, with the rounding error of a pairwise sum: values are
// summed sequentially in blocks of 16, and the block sums are merged as a binary counter
// merges carries, so a sum of L values sees about 16 + log2(L / 16) roundings, not L. Up
// to 2^48 values.
template <class T> struct rader_sum {
  cv<T> level[44]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  cv<T> block;
  std::size_t in_block;
  std::size_t blocks;
};

template <class T> FFTN_FORCE_INLINE void rader_sum_start(rader_sum<T>& r) {
  r.block = zeros<T>();
  r.in_block = 0;
  r.blocks = 0;
}

template <class T> FFTN_FORCE_INLINE void rader_sum_add(rader_sum<T>& r, cv<T> v) {
  r.block = add(r.block, v);
  if (++r.in_block < 16) {
    return;
  }
  // Merge the full block like adding 1 to a binary counter: each carry is a pairwise sum.
  cv<T> carry = r.block;
  std::size_t b = r.blocks;
  std::size_t lvl = 0;
  while ((b & 1U) != 0) {
    carry = add(r.level[lvl], carry);
    b >>= 1U;
    ++lvl;
  }
  r.level[lvl] = carry;
  ++r.blocks;
  r.block = zeros<T>();
  r.in_block = 0;
}

template <class T> FFTN_FORCE_INLINE cv<T> rader_sum_total(const rader_sum<T>& r) {
  cv<T> total = r.block;
  std::size_t b = r.blocks;
  for (std::size_t lvl = 0; b != 0; ++lvl, b >>= 1U) {
    if ((b & 1U) != 0) {
      total = add(r.level[lvl], total);
    }
  }
  return total;
}

// z[t + s m] = in(x[t + s g^m]) for m < L, z[t + s L] = in(x[t]), z[t + s (L + 1)] = the sum
// of in(x[t + s j]) over 1 <= j < q, for each copy t of the unit. Arguments as a Bluestein
// stage's edge passes (radix 1): n = s q, m = L, s, root = g. The input edge's operation
// applies to what is loaded. src: any stride when C == 1, else 1; dst: scratch, stride 1,
// not src. Reads src, dst, src_stride, n, m, s, root, scale, in_factor.
template <class T, edge_op InOp, bool Shared>
void rader_in(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::size_t s = a->s;
  const std::size_t len = a->m;
  const std::size_t q = len + 1;
  const std::size_t g = a->root;
  const edge<T, InOp> op = make_edge<T, InOp>(a, a->in_factor, 0, 1);
  rader_sum<T> sum;
  for (std::size_t u = first; u < last; ++u) {
    rader_sum_start(sum);
    if constexpr (Shared) {
      const std::size_t t = u * C<T>;
      std::size_t idx = 1;
      for (std::size_t m = 0; m < len; ++m) {
        const std::size_t i = t + s * idx;
        const cv<T> v =
            on_run<T, InOp, true>(op, load_run<T, true>(x + at(i, xs), xs, C<T>), i, C<T>);
        store(z + at(t + s * m, 2), v);
        rader_sum_add(sum, v);
        idx = rader_next(idx, g, q);
      }
      store(z + at(t + s * len, 2),
            on_run<T, InOp, true>(op, load_run<T, true>(x + at(t, xs), xs, C<T>), t, C<T>));
      store(z + at(t + s * (len + 1), 2), rader_sum_total(sum));
    } else {
      const std::size_t t = u;
      std::size_t idx = 1;
      for (std::size_t m = 0; m < len; ++m) {
        const std::size_t i = t + s * idx;
        const cv<T> v = on_one<T, InOp>(op, bcast(x + at(i, xs)), i);
        store_first(z + at(t + s * m, 2), v);
        rader_sum_add(sum, v);
        idx = rader_next(idx, g, q);
      }
      store_first(z + at(t + s * len, 2), on_one<T, InOp>(op, bcast(x + at(t, xs)), t));
      store_first(z + at(t + s * (len + 1), 2), rader_sum_total(sum));
    }
  }
}

// y[t] = out(x_0 + sum), y[t + s g^m] = out(x_0 + c_((L - m) mod L)) for m < L, where c is
// the convolution in the first s L elements of src and x_0 and the sum are the two rows
// after it (rader_in). Units, arguments: as rader_in's. The output edge's operation applies
// to what is stored. src: scratch, stride 1; dst: any stride when C == 1, else 1; dst may
// be the stage's source line (rader_in has read it), never src. Reads src, dst, dst_stride,
// n, m, s, root, scale, out_factor.
template <class T, edge_op OutOp, bool Shared>
void rader_out(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t s = a->s;
  const std::size_t len = a->m;
  const std::size_t q = len + 1;
  const std::size_t g = a->root;
  const edge<T, OutOp> op = make_edge<T, OutOp>(a, a->out_factor, 0, 1);
  for (std::size_t u = first; u < last; ++u) {
    if constexpr (Shared) {
      const std::size_t t = u * C<T>;
      const cv<T> x0 = load(z + at(t + s * len, 2));
      const cv<T> total = add(x0, load(z + at(t + s * (len + 1), 2)));
      store_run<T, true>(y + at(t, ys), ys, C<T>, on_run<T, OutOp, true>(op, total, t, C<T>));
      std::size_t idx = 1;
      for (std::size_t m = 0; m < len; ++m) {
        const std::size_t k = m == 0 ? 0 : len - m;
        const std::size_t i = t + s * idx;
        const cv<T> v = add(x0, load(z + at(t + s * k, 2)));
        store_run<T, true>(y + at(i, ys), ys, C<T>, on_run<T, OutOp, true>(op, v, i, C<T>));
        idx = rader_next(idx, g, q);
      }
    } else {
      const std::size_t t = u;
      const cv<T> x0 = bcast(z + at(t + s * len, 2));
      const cv<T> total = add(x0, bcast(z + at(t + s * (len + 1), 2)));
      store_first(y + at(t, ys), on_one<T, OutOp>(op, total, t));
      std::size_t idx = 1;
      for (std::size_t m = 0; m < len; ++m) {
        const std::size_t k = m == 0 ? 0 : len - m;
        const std::size_t i = t + s * idx;
        const cv<T> v = add(x0, bcast(z + at(t + s * k, 2)));
        store_first(y + at(i, ys), on_one<T, OutOp>(op, v, i));
        idx = rader_next(idx, g, q);
      }
    }
  }
}

} // namespace
