// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The two edge passes of a Bluestein stage (DESIGN.md 8.5, src/strategies/chain.cpp): the
// chirp multiplication into the zero-padded convolution buffer, and the chirp
// multiplication out of it. A Bluestein stage is always the last pass of a chain (m = 1),
// so element i of its input and of its output is the same index of the line: both passes
// are elementwise over i, in blocks of C consecutive elements, whatever the copy count.
// Kernel text in the manner of kernels.inl, included after access.inl. No include guard.

namespace {

// A register of zeros.
template <class T> FFTN_FORCE_INLINE cv<T> zeros() {
  T tile[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  for (std::size_t l = 0; l < 2 * C<T>; ++l) {
    tile[l] = T(0);
  }
  return load(tile);
}

// z[i] = in(x[i]) w[i] for i < n = s Q, z[i] = 0 for n <= i < s M: the input of the
// convolution, s copies of length M interleaved (copy innermost). Unit u is the elements
// u C .. u C + C - 1 of z. twiddles: the chirp, per-lane storage (pack_twiddles) of n
// entries padded with ones to a multiple of C. An edge pass's arguments (radix 1, which the
// work graph reads as such): n, m = M and s. The input edge's operation applies to what is
// loaded. src: any stride when C == 1, else 1; dst: scratch, stride 1, not src. Reads src,
// dst, src_stride, n, m, s, twiddles, scale, in_factor.
template <class T, edge_op InOp>
void bluestein_in(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::size_t data = a->n;
  const std::size_t total = a->m * a->s;
  const T* chirp = static_cast<const T*>(a->twiddles);
  const edge<T, InOp> op = make_edge<T, InOp>(a, a->in_factor, 0, 1);
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t i0 = u * C<T>;
    T* zp = z + at(i0, 2);
    if (i0 + C<T> <= data) {
      const cv<T> v = on_run<T, InOp, true>(op, load_run<T, true>(x + at(i0, xs), xs, C<T>), i0, C<T>);
      store(zp, mul(v, load_tw(chirp + tw_reals<T> * i0)));
    } else if (i0 >= data) {
      store_run<T, false>(zp, 2, total - i0 < C<T> ? total - i0 : C<T>, zeros<T>());
    } else {
      // The block that holds the last data element: the lanes beyond it load zero.
      const std::size_t valid = data - i0;
      const cv<T> v =
          on_run<T, InOp, false>(op, load_run<T, false>(x + at(i0, xs), xs, valid), i0, valid);
      store_run<T, false>(zp, 2, total - i0 < C<T> ? total - i0 : C<T>,
                          mul(v, load_tw(chirp + tw_reals<T> * i0)));
    }
  }
}

// y[i] = out(z[i] w[i]) for i < n = s Q: the transform from the convolution's first Q
// outputs of each copy. Unit u is the elements u C .. u C + C - 1 of y. twiddles, n: as
// bluestein_in's. The output edge's operation applies to what is stored. src: scratch,
// stride 1; dst: any stride when C == 1, else 1; src may be dst (each element is read
// before it is written, at the same index). Reads src, dst, dst_stride, n, twiddles, scale,
// out_factor.
template <class T, edge_op OutOp>
void bluestein_out(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t data = a->n;
  const T* chirp = static_cast<const T*>(a->twiddles);
  const edge<T, OutOp> op = make_edge<T, OutOp>(a, a->out_factor, 0, 1);
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t i0 = u * C<T>;
    if (i0 + C<T> <= data) {
      const cv<T> v = mul(load(z + at(i0, 2)), load_tw(chirp + tw_reals<T> * i0));
      store_run<T, true>(y + at(i0, ys), ys, C<T>, on_run<T, OutOp, true>(op, v, i0, C<T>));
    } else {
      const std::size_t valid = data - i0;
      const cv<T> v =
          mul(load_run<T, false>(z + at(i0, 2), 2, valid), load_tw(chirp + tw_reals<T> * i0));
      store_run<T, false>(y + at(i0, ys), ys, valid, on_run<T, OutOp, false>(op, v, i0, valid));
    }
  }
}

} // namespace
