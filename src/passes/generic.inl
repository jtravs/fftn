// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The passes that are not codelets: the generic pass for any radix (the primes above the
// codelets, DESIGN.md 9.2) and the copy. Kernel text in the manner of kernels.inl, included
// after access.inl. No include guard.

namespace {

// One run of the generic pass: the r outputs of butterfly group p for the copies of one
// run, each output summed directly from the r inputs.
template <class T, int Sign, class In, class Out>
FFTN_FORCE_INLINE void generic_run(const In& in, const Out& out, std::size_t r, const T* roots,
                                   const T* twiddles) {
  constexpr std::size_t levels = 64; // partial sums of 2^level terms; r < 2^64
  for (std::size_t k = 0; k < r; ++k) {
    cv<T> partial[levels]; // NOLINT(modernize-avoid-c-arrays): no standard containers here
    std::size_t t = 0;     // j k mod r
    for (std::size_t j = 0; j < r; ++j) {
      cv<T> term = in.get(j);
      if (t != 0) {
        term = twiddle<T, Sign>(term, bcast_tw(roots + tw_reals<T> * t));
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
    if (k != 0 && twiddles != nullptr) {
      sum = twiddle<T, Sign>(sum, bcast_tw(twiddles + tw_reals<T> * (k - 1)));
    }
    out.put(k, sum);
  }
}

// The Stockham pass of stockham.inl for any radix r = a->radix by direct evaluation, O(r^2)
// per butterfly, with lanes over copies as lane_pass (the same units and tails). Each
// output sums its r products in a pairwise tree (a binary counter of partial sums), so its
// rounding error grows like log r rather than r: a sequential sum fails the tolerances from
// r of a few hundred. roots: the r forward roots exp(-2 pi i t / r), t < r, in broadcast
// storage; twiddles: as lane_pass's, or null when m = 1 (all ones). Both edges'
// operations apply, since the pass may be a chain's first, last or only one. src and dst:
// any stride when C == 1, else 1; the source must not overlap the destination. Reads src,
// dst, src_stride, dst_stride, radix, m, s, twiddles, roots, scale, in_factor, out_factor.
template <class T, int Sign, edge_op InOp, edge_op OutOp>
void generic_pass(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t r = a->radix;
  const std::size_t m = a->m;
  const std::size_t s = a->s;
  const std::size_t blocks = (s + C<T> - 1) / C<T>;
  const T* table = static_cast<const T*>(a->twiddles);
  const T* roots = static_cast<const T*>(a->roots);
  const std::ptrdiff_t in_leg = at(s * m, xs);
  const std::ptrdiff_t out_leg = at(s, ys);
  std::size_t p = first / blocks;
  std::size_t b = first % blocks;
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t q0 = b * C<T>;
    const std::size_t valid = s - q0 < C<T> ? s - q0 : C<T>;
    const std::size_t ei = q0 + s * p;
    const std::size_t eo = q0 + s * r * p;
    const edge<T, InOp> iop = make_edge<T, InOp>(a, a->in_factor, ei, s * m);
    const edge<T, OutOp> oop = make_edge<T, OutOp>(a, a->out_factor, eo, s);
    const T* wp = table != nullptr ? table + tw_reals<T> * (r - 1) * p : nullptr;
    if (valid == C<T>) {
      const run_in<T, InOp, true> in{
          .p = x + at(ei, xs),
          .leg = in_leg,
          .step = xs,
          .valid = valid,
          .op = iop,
      };
      const run_out<T, OutOp, true> out{
          .p = y + at(eo, ys),
          .leg = out_leg,
          .step = ys,
          .valid = valid,
          .op = oop,
      };
      generic_run<T, Sign>(in, out, r, roots, wp);
    } else {
      const run_in<T, InOp, false> in{
          .p = x + at(ei, xs),
          .leg = in_leg,
          .step = xs,
          .valid = valid,
          .op = iop,
      };
      const run_out<T, OutOp, false> out{
          .p = y + at(eo, ys),
          .leg = out_leg,
          .step = ys,
          .valid = valid,
          .op = oop,
      };
      generic_run<T, Sign>(in, out, r, roots, wp);
    }
    if (++b == blocks) {
      b = 0;
      ++p;
    }
  }
}

// y[i] = x[i] with both edges' operations, unit i: one element at a time, any strides, and
// x may be y. The line of n = 1, and the moves to and from scratch where a SIMD pass cannot
// read or write the caller's stride (src/strategies/chain.cpp). Reads src, dst, src_stride,
// dst_stride, scale, in_factor, out_factor.
template <class T, edge_op InOp, edge_op OutOp>
void copy_pass(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  if constexpr (InOp == edge_op::none && OutOp == edge_op::none) {
    for (std::size_t i = first; i < last; ++i) {
      const T re = x[at(i, xs)];
      const T im = x[at(i, xs) + 1];
      y[at(i, ys)] = re;
      y[at(i, ys) + 1] = im;
    }
  } else {
    const edge<T, InOp> iop = make_edge<T, InOp>(a, a->in_factor, 0, 1);
    const edge<T, OutOp> oop = make_edge<T, OutOp>(a, a->out_factor, 0, 1);
    for (std::size_t i = first; i < last; ++i) {
      const cv<T> v = on_one<T, InOp>(iop, bcast(x + at(i, xs)), i);
      store_first(y + at(i, ys), on_one<T, OutOp>(oop, v, i));
    }
  }
}

} // namespace
