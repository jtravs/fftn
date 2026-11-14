// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The two edge passes of a Bluestein stage (DESIGN.md 8.5, src/strategies/chain.cpp): the
// chirp multiplication into the zero-padded convolution buffer, and the chirp
// multiplication out of it. A Bluestein stage is always the last pass of a chain (m = 1),
// so element i of its input and of its output is the same index of the line: both passes
// are elementwise over i, in blocks of C consecutive elements, whatever the copy count.
// The kernel spectrum between them is applied by the convolution's lane passes (lane_pass's
// factor_sharing in stockham.inl). Kernel text in the manner of kernels.inl, included
// after access.inl. No include guard.

namespace {

// A register of zeros.
template <class T> FFTN_FORCE_INLINE cv<T> zeros() {
  T tile[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  for (std::size_t l = 0; l < 2 * C<T>; ++l) {
    tile[l] = T(0);
  }
  return load(tile);
}

// The chirp, as both passes read it: element i = t + s j of the line (copy t < s, sequence
// index j < Q) takes w_j. Shared: C divides s, so the C elements of a unit share one j; the
// table holds Q entries in broadcast storage (pack_bcast_twiddles) and the units run row
// by row (the s / C units of one j), the entry loaded once per row. Otherwise the table
// holds one entry per element (s Q, per-lane storage, padded with ones to a multiple of C).
// Tracking j unit by unit instead was slower than the per-element table at 4096 x 17 on the
// Mac (C = 1, task 2.3b); row by row it is faster.

// z[i] = in(x[i]) w[i] for i < n = s Q, z[i] = 0 for n <= i < s M: the input of the
// convolution, s copies of length M interleaved (copy innermost). Unit u is the elements
// u C .. u C + C - 1 of z. twiddles: the chirp (above). An edge pass's arguments (radix 1,
// which the work graph reads as such): n, m = M and s. The input edge's operation applies
// to what is loaded. src: any stride when C == 1, else 1; dst: scratch, stride 1, not src.
// Reads src, dst, src_stride, n, m, s, twiddles, scale, in_factor.
template <class T, edge_op InOp, bool Shared>
void bluestein_in(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::size_t data = a->n;
  const std::size_t total = a->m * a->s;
  const T* chirp = static_cast<const T*>(a->twiddles);
  const edge<T, InOp> op = make_edge<T, InOp>(a, a->in_factor, 0, 1);
  if constexpr (Shared) {
    // Every unit is whole: C divides s, so it divides n and s M.
    const std::size_t per_row = a->s / C<T>;
    std::size_t u = first;
    for (std::size_t j = first < per_row ? 0 : first / per_row; u < last; ++j) {
      const std::size_t row_end = (j + 1) * per_row;
      const std::size_t end = row_end < last ? row_end : last;
      if (j * a->s < data) {
        const tw<T> w = bcast_tw(chirp + tw_reals<T> * j);
        for (; u < end; ++u) {
          const std::size_t i0 = u * C<T>;
          const cv<T> v =
              on_run<T, InOp, true>(op, load_run<T, true>(x + at(i0, xs), xs, C<T>), i0, C<T>);
          store(z + at(i0, 2), mul(v, w));
        }
      } else {
        for (; u < end; ++u) {
          store(z + at(u * C<T>, 2), zeros<T>());
        }
      }
    }
  } else {
    for (std::size_t u = first; u < last; ++u) {
      const std::size_t i0 = u * C<T>;
      T* zp = z + at(i0, 2);
      if (i0 + C<T> <= data) {
        const cv<T> v =
            on_run<T, InOp, true>(op, load_run<T, true>(x + at(i0, xs), xs, C<T>), i0, C<T>);
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
}

// y[i] = out(z[i] w[i]) for i < n = s Q: the transform from the convolution's first Q
// outputs of each copy. Unit u is the elements u C .. u C + C - 1 of y. twiddles, n, s: as
// bluestein_in's. The output edge's operation applies to what is stored. src: scratch,
// stride 1; dst: any stride when C == 1, else 1; src may be dst (each element is read
// before it is written, at the same index). Reads src, dst, dst_stride, n, s, twiddles,
// scale, out_factor.
template <class T, edge_op OutOp, bool Shared>
void bluestein_out(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t data = a->n;
  const T* chirp = static_cast<const T*>(a->twiddles);
  const edge<T, OutOp> op = make_edge<T, OutOp>(a, a->out_factor, 0, 1);
  if constexpr (Shared) {
    const std::size_t per_row = a->s / C<T>;
    std::size_t u = first;
    for (std::size_t j = first < per_row ? 0 : first / per_row; u < last; ++j) {
      const std::size_t row_end = (j + 1) * per_row;
      const std::size_t end = row_end < last ? row_end : last;
      const tw<T> w = bcast_tw(chirp + tw_reals<T> * j);
      for (; u < end; ++u) {
        const std::size_t i0 = u * C<T>;
        const cv<T> v = mul(load(z + at(i0, 2)), w);
        store_run<T, true>(y + at(i0, ys), ys, C<T>, on_run<T, OutOp, true>(op, v, i0, C<T>));
      }
    }
  } else {
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
}

} // namespace
