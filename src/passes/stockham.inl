// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The pass kinds of DESIGN.md 8.2 on the generated codelets (src/gen/complex_codelets.inl):
// grouped, lane and leaf passes. Kernel text in the manner of kernels.inl (included inside
// a region after its vocabulary, the codelets and access.inl; no standard-library
// templates; entry points take pointers and integers). No include guard.
//
// A chain (DESIGN.md 8.5) is a Stockham autosort sequence of radix-R passes in
// decimation-in-frequency form. For a line of n = R m s complex values, butterfly group
// p < m and copy q < s, a pass computes
//
//   y[q + s (R p + k)] = w^(p k) sum_j x[q + s (p + m j)] exp(Sign 2 pi i j k / R),
//
// w = exp(Sign 2 pi i / (R m)): the DIF codelet with twiddles w^(p k), or the plain one
// when m = 1 (the last pass). Its first pass has s = 1 and is grouped; the others are lane
// passes. Signs follow FFTW: Sign = -1 is forward. Tables hold forward twiddles; backward
// passes multiply by their conjugates (the codelets' mulc), which is exact.

namespace {

// The R outputs of a grouped pass's C butterflies p0 .. p0 + C - 1, v[k] holding leg k of
// every lane, to y[R (p0 + l) + k] (tight). C x C blocks of legs are transposed in
// registers, so that each store writes C legs of one butterfly.
template <class T, std::size_t R> FFTN_FORCE_INLINE void store_transposed(T* y, cv<T>* v) {
  for (std::size_t b = 0; b < R / C<T>; ++b) {
    transpose(v + b * C<T>);
    for (std::size_t l = 0; l < C<T>; ++l) {
      store(y + 2 * (R * l + b * C<T>), v[b * C<T> + l]);
    }
  }
}

// The same through a tile, element by element, for the first `valid` butterflies: for R not
// a multiple of C, and for the last unit of a pass.
template <class T, std::size_t R>
FFTN_FORCE_INLINE void store_tiled(T* y, const cv<T>* v, std::size_t valid) {
  T tile[R][2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  for (std::size_t k = 0; k < R; ++k) {
    store(tile[k], v[k]);
  }
  for (std::size_t l = 0; l < valid; ++l) {
    for (std::size_t k = 0; k < R; ++k) {
      y[2 * (R * l + k)] = tile[k][2 * l];
      y[2 * (R * l + k) + 1] = tile[k][2 * l + 1];
    }
  }
}

// The first pass of a chain, s = 1 and n = R m, with lanes over butterflies (DESIGN.md
// 8.2): unit u is the butterflies p = u C .. u C + C - 1, fewer in the last unit when C
// does not divide m. Lane l loads x[p + l + m j], which are adjacent. twiddles: per-lane
// storage (pack_twiddles), R - 1 rows of mpad = m rounded up to a multiple of C entries,
// row k - 1 holding w^(p k) and padded with ones. The input edge's operation applies to
// what is loaded. src: any stride when C == 1, else 1; dst: scratch, stride 1. Reads src,
// src_stride, m, twiddles, scale, in_factor.
template <class T, std::size_t R, int Sign, edge_op InOp>
void grouped_pass(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::size_t m = a->m;
  const std::size_t mpad = (m + C<T> - 1) / C<T> * C<T>;
  const T* table = static_cast<const T*>(a->twiddles);
  const std::ptrdiff_t row = at(mpad, static_cast<std::ptrdiff_t>(tw_reals<T>));
  const std::ptrdiff_t leg = at(m, xs);
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t p0 = u * C<T>;
    const std::size_t valid = m - p0 < C<T> ? m - p0 : C<T>;
    const T* xp = x + at(p0, xs);
    const edge<T, InOp> op = make_edge<T, InOp>(a, a->in_factor, p0, m);
    const lane_twiddles<T> twid{.p = table + tw_reals<T> * p0, .row = row};
    cv<T> v[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    const to_registers<T> out{v};
    T* yp = y + 2 * R * p0;
    if (valid == C<T>) {
      const run_in<T, InOp, true> in{.p = xp, .leg = leg, .step = xs, .valid = valid, .op = op};
      cdft<R>::template dif<T, Sign>(in, out, twid);
      if constexpr (R % C<T> == 0) {
        store_transposed<T, R>(yp, v);
      } else {
        store_tiled<T, R>(yp, v, valid);
      }
    } else {
      const run_in<T, InOp, false> in{.p = xp, .leg = leg, .step = xs, .valid = valid, .op = op};
      cdft<R>::template dif<T, Sign>(in, out, twid);
      store_tiled<T, R>(yp, v, valid);
    }
  }
}

// The butterflies of one unit of a lane pass, from accessors in and out.
template <class T, std::size_t R, int Sign, bool Twiddled, class In, class Out>
FFTN_FORCE_INLINE void lane_unit(const In& in, const Out& out, const T* table, std::size_t p) {
  if constexpr (Twiddled) {
    const shared_twiddles<T> twid{table + tw_reals<T> * (R - 1) * p};
    cdft<R>::template dif<T, Sign>(in, out, twid);
  } else {
    cdft<R>::template plain<T, Sign>(in, out);
  }
}

// The input accessor of a lane pass's unit: plain, or (factor_sharing::input) each leg j
// times entry p + m j of in_factor, the same for every copy (run_in_shared).
template <class T, bool Full, factor_sharing Share>
FFTN_FORCE_INLINE auto lane_input(const pass_args* a, const T* xp, std::ptrdiff_t leg,
                                  std::size_t valid, std::size_t p) {
  if constexpr (Share == factor_sharing::input) {
    return run_in_shared<T, Full>{
        .p = xp,
        .leg = leg,
        .step = 2,
        .valid = valid,
        .f = static_cast<const T*>(a->in_factor) + at(p, 2),
        .f_leg = at(a->m, 2),
    };
  } else {
    return run_in<T, edge_op::none, Full>{
        .p = xp,
        .leg = leg,
        .step = 2,
        .valid = valid,
        .op = {},
    };
  }
}

// The output accessor of a lane pass's unit: with the output edge, or
// (factor_sharing::output, the last pass, m = 1) each leg k times entry
// g + k s / factor_copies of out_factor, broadcast, g = q0 / factor_copies: output
// element q + s k divided by factor_copies, which C divides, so the unit's lanes share it
// (run_out_shared).
template <class T, bool Full, edge_op OutOp, factor_sharing Share>
FFTN_FORCE_INLINE auto lane_output(const pass_args* a, T* yp, std::ptrdiff_t leg,
                                   std::ptrdiff_t step, std::size_t valid, std::size_t e0,
                                   std::ptrdiff_t f_leg,
                                   std::size_t g) {
  if constexpr (Share == factor_sharing::output) {
    return run_out_shared<T, Full>{
        .p = yp,
        .leg = leg,
        .step = step,
        .valid = valid,
        .f = static_cast<const T*>(a->out_factor) + at(g, 2),
        .f_leg = f_leg,
    };
  } else {
    return run_out<T, OutOp, Full>{
        .p = yp,
        .leg = leg,
        .step = step,
        .valid = valid,
        .op = make_edge<T, OutOp>(a, a->out_factor, e0, a->s),
    };
  }
}

// A later pass of a chain, with lanes over copies q (DESIGN.md 8.2): a middle pass
// (Twiddled, m > 1, the DIF codelet) or the last (m = 1, the plain codelet). Unit u is
// group p = u / blocks and copies q = (u % blocks) C .. + C - 1, blocks = ceil(s / C);
// the last block of a group has fewer when C does not divide s. twiddles: broadcast
// storage (pack_bcast_twiddles), R - 1 per group p, w^(p k) for k = 1 .. R - 1, the same
// for every lane. The output edge's operation applies to what is stored. Share: a factor
// table of one entry per element of a sequence whose copies are interleaved in the line,
// read once for all of them (a Bluestein convolution's kernel spectrum, bluestein.inl):
// input leg j of group p times in_factor's entry p + m j (input, the first backward pass,
// whose lanes are the copies), or output element e times out_factor's entry
// e / factor_copies (output, the forward chain's last pass; C divides factor_copies).
// src: scratch, stride 1; dst: any stride when C == 1, else 1. Reads src, dst_stride, m,
// s, twiddles, scale, out_factor, and in_factor or factor_copies when sharing.
template <class T, std::size_t R, int Sign, bool Twiddled, edge_op OutOp,
          factor_sharing Share = factor_sharing::none>
void lane_pass(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t m = a->m;
  const std::size_t s = a->s;
  const std::size_t blocks = (s + C<T> - 1) / C<T>;
  const T* table = static_cast<const T*>(a->twiddles);
  const std::ptrdiff_t in_leg = at(s * m, 2);
  const std::ptrdiff_t out_leg = at(s, ys);
  // No division when the range starts in the first group (a whole stage): it is slow on
  // some x86 cores (dev/experiments/X18-x86-kernels.md).
  std::size_t p = first < blocks ? 0 : first / blocks;
  std::size_t b = first < blocks ? first : first % blocks;
  // factor_sharing::output (m = 1, so unit u is copies u C ..): the entry g = u C /
  // factor_copies, advanced every per_entry units rather than divided per unit (a
  // division per unit of a radix-12 last pass is a few per cent of it), and the entries
  // between legs.
  std::size_t g = 0;
  std::size_t g_left = 0;
  std::size_t per_entry = 1;
  std::ptrdiff_t f_leg = 0;
  if constexpr (Share == factor_sharing::output) {
    per_entry = a->factor_copies / C<T>;
    g = first < per_entry ? 0 : first / per_entry;
    g_left = per_entry - (first < per_entry ? first : first % per_entry);
    f_leg = at(s / a->factor_copies, 2);
  }
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t q0 = b * C<T>;
    const std::size_t valid = s - q0 < C<T> ? s - q0 : C<T>;
    const T* xp = x + at(q0 + s * p, 2);
    const std::size_t e0 = q0 + s * R * p;
    T* yp = y + at(e0, ys);
    if (valid == C<T>) {
      lane_unit<T, R, Sign, Twiddled>(lane_input<T, true, Share>(a, xp, in_leg, valid, p),
                                      lane_output<T, true, OutOp, Share>(a, yp, out_leg, ys,
                                                                         valid, e0, f_leg, g),
                                      table, p);
    } else {
      lane_unit<T, R, Sign, Twiddled>(lane_input<T, false, Share>(a, xp, in_leg, valid, p),
                                      lane_output<T, false, OutOp, Share>(a, yp, out_leg, ys,
                                                                          valid, e0, f_leg, g),
                                      table, p);
    }
    if (++b == blocks) {
      b = 0;
      ++p;
    }
    if constexpr (Share == factor_sharing::output) {
      if (--g_left == 0) {
        ++g;
        g_left = per_entry;
      }
    }
  }
}

// A whole transform of n = R points in registers (DESIGN.md 8.2), one unit (the line):
// y[k] = sum_j x[j] exp(Sign 2 pi i j k / R), with both edges' operations. Values are
// broadcast on loading and lane 0 is stored, so every lane computes the same transform
// and any strides work; every load precedes every store, so src may be dst (in place).
// Unit u is the line at src + u src_dist, dst + u dst_dist bytes (one line in a chain; a
// batch of rows when the driver runs them in one call, X20); every line gets the same
// arithmetic. Reads src, dst, src_stride, dst_stride, src_dist, dst_dist, scale, in_factor,
// out_factor.
template <class T, std::size_t R, int Sign, edge_op InOp, edge_op OutOp>
void leaf_pass(const pass_args* a, std::size_t first, std::size_t last) {
  one_in<T, InOp> in{
      .p = nullptr,
      .leg = 2 * a->src_stride,
      .op = make_edge<T, InOp>(a, a->in_factor, 0, 1),
  };
  one_out<T, OutOp> out{
      .p = nullptr,
      .leg = 2 * a->dst_stride,
      .op = make_edge<T, OutOp>(a, a->out_factor, 0, 1),
  };
  // Batch distances are whole elements, so whole reals.
  constexpr auto real_bytes = static_cast<std::ptrdiff_t>(sizeof(T));
  const std::ptrdiff_t src_dist = a->src_dist / real_bytes;
  const std::ptrdiff_t dst_dist = a->dst_dist / real_bytes;
  for (std::size_t u = first; u < last; ++u) {
    const auto k = static_cast<std::ptrdiff_t>(u);
    in.p = static_cast<const T*>(a->src) + k * src_dist;
    out.p = static_cast<T*>(a->dst) + k * dst_dist;
    cdft<R>::template plain<T, Sign>(in, out);
  }
}

} // namespace
