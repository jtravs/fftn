// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The two passes of the depth-first chain (strategies/chain.hpp, c2c_depth; experiment
// X46) that the Stockham passes of stockham.inl do not cover. Kernel text in the manner of
// stockham.inl, included after it. No include guard.
//
// The depth-first chain computes a line of n = R1 M by decimation in time at the top:
// X[k + M t] = sum_r exp(Sign 2 pi i r t / R1) w_n^(r k) Y_r[k], where Y_r is the length-M
// transform of the sub-row x[R1 j + r] (and the same once more inside each Y_r with two top
// passes). The sub-rows are R1 interleaved sequences, the layout a Stockham lane pass reads
// with s = R1 copies: the first pass below runs the first pass of every sub-row's chain with
// lanes over C adjacent sub-rows and writes each sub-row into its own block of M elements,
// so that the rest of its chain (the ordinary lane passes) runs on one block at a time, in
// cache; the top pass then combines the blocks in place.

namespace {

// Leg j of a run of a Bluestein convolution's input with the chirp pass fused in (experiment
// X50 arm C): the C elements from i + leg j of the line, x[e] w[e] for e < data and zero
// beyond (a run that straddles data loads only its valid lanes). w: the chirp in per-element
// storage, at element i.
template <class T> struct run_in_chirp {
  const T* p;
  const T* w;
  std::size_t i;
  std::size_t leg;
  std::size_t data;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const {
    const std::size_t e = i + j * leg;
    if (e >= data) {
      return zeros<T>();
    }
    const T* xp = p + at(j * leg, 2);
    const tw<T> wv = load_tw(w + tw_reals<T> * (j * leg));
    if (e + C<T> <= data) {
      return mul(load(xp), wv);
    }
    return mul(load_run<T, false>(xp, 2, data - e), wv);
  }
};

// Leg k of a run of a Bluestein stage's output with its chirp pass fused in (arm C): the C
// elements from i + leg k of the convolution, stored times w[e] with the output edge's
// operation for e < data; nothing from data on (the stage keeps its first Q outputs).
template <class T, edge_op Op> struct run_out_chirp {
  T* p;
  const T* w;
  std::size_t i;
  std::size_t leg;
  std::size_t data;
  edge<T, Op> op;
  FFTN_FORCE_INLINE void put(std::size_t k, cv<T> v) const {
    const std::size_t e = i + k * leg;
    if (e >= data) {
      return;
    }
    T* yp = p + at(k * leg, 2);
    const cv<T> z = mul(v, load_tw(w + tw_reals<T> * (k * leg)));
    if (e + C<T> <= data) {
      store(yp, on_run<T, Op, true>(op, z, k, C<T>));
    } else {
      store_run<T, false>(yp, 2, data - e, on_run<T, Op, false>(op, z, k, data - e));
    }
  }
};

// Leg j of a run of a c2r row's half-length chain with the pairing fused in (experiment X78):
// the C elements from i + leg j of the line, Z_e = A + conj(v_e) B with A = X_e + conj X_(h-e),
// B = X_e - conj X_(h-e) and v_e = -i w_n^e (real.inl's c2r pairing, which for e above h/2 is the
// same expression, since v_(h-e) = conj v_e), from the h + 1 bins X of the input: X_e upwards,
// X_(h-e) downwards. Z_0 takes only the real parts of X_0 and X_h, as the separate pass does.
// w: v in per-element storage, at element i.
template <class T> struct run_in_c2r {
  const T* x;
  const T* w;
  std::size_t i;
  std::size_t leg;
  std::size_t h;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const {
    const std::size_t e = i + j * leg;
    const cv<T> xk = load(x + at(e, 2));
    const cv<T> xc = conj(load_down<T, true>(x + at(h - e, 2), C<T>));
    const cv<T> z = add(add(xk, xc), mulc(sub(xk, xc), load_tw(w + tw_reals<T> * (j * leg))));
    if (j == 0 && i == 0) {
      T tile[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      store(tile, z);
      tile[0] = x[0] + x[at(h, 2)];
      tile[1] = x[0] - x[at(h, 2)];
      return load(tile);
    }
    return z;
  }
};

// What the first pass below reads: the line itself, a Bluestein convolution's input times the
// chirp (experiment X50 arm C), or a c2r row's bins paired on loading (experiment X78).
enum class first_input { plain, chirp, c2r };

// The partial blocks among the first pass's units first .. last - 1 (below): with one top
// factor that C does not divide (cols = s), the last block of each group p, unit p blocks +
// blocks - 1, copies q0 .. s - 1 (experiment X51: 25000 = 10 x 2500 on 4 lanes). The pass
// runs them after its loop over whole blocks on the targets with masked partial runs, for
// the reason given at grouped_tail (stockham.inl); units are independent, so the order
// changes no bit.
template <class T, std::size_t R, int Sign>
FFTN_FORCE_INLINE void depth_first_tails(const pass_args* a, const T* x, T* y, std::size_t first,
                                         std::size_t last, std::size_t blocks) {
  const std::size_t m = a->m;
  const std::size_t s = a->s;
  const auto lane = static_cast<std::size_t>(a->dst_qstride);
  const T* table = static_cast<const T*>(a->twiddles);
  const std::ptrdiff_t in_leg = at(s * m, 2);
  const std::size_t q0 = (blocks - 1) * C<T>;
  const std::size_t valid = s - q0;
  // From the group of unit `first`, whose last block is at or after it.
  for (std::size_t p = first < blocks ? 0 : first / blocks; p * blocks + blocks - 1 < last; ++p) {
    cv<T> v[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    const to_registers<T> out{v};
    lane_unit<T, R, Sign, true>(
        run_in<T, edge_op::none, false>{
            .p = x + at(q0 + s * p, 2),
            .leg = in_leg,
            .step = 2,
            .valid = valid,
            .op = {},
        },
        out, table, p);
    T* yp = y + at(q0 * lane + R * p, 2);
    if constexpr (R % C<T> == 0) {
      store_transposed<T, R>(yp, v, valid, lane);
    } else if constexpr (masked) {
      store_padded<T, R>(yp, v, valid, lane);
    } else {
      store_tiled<T, R>(yp, v, valid, lane);
    }
  }
}

// The first pass of the chains of the s sub-rows of a line of n = s R m (DESIGN.md 8.2's
// lane pass with s copies, m groups, twiddled by w_(R m)^(p k)), every copy's outputs
// written to its own block of M = R m elements: copy q's y[o(q) + R p + k] instead of the
// lane pass's y[q + s (R p + k)], where o(q) = (q % cols) dst_qstride + (q / cols) M. With
// cols = s (one top pass) the blocks are in copy order, o(q) = q M; with two top passes of
// F1 = cols and F2 = s / cols, copy q = d1 + F1 d2 goes to block d1 F2 + d2, so that each
// of the F1 sub-transforms of the first top pass has its F2 sub-rows together
// (dst_qstride = F2 M). Unit u is group p = u / blocks and copies q = (u % blocks) C .. + C
// - 1, blocks = ceil(s / C); with two tops C divides cols, so a unit's copies are adjacent
// within one group of cols; with one top the last block of a group has fewer copies when C
// does not divide s (X51; depth_first_tails). Its R outputs are transposed in registers so
// that each store writes C legs of one copy. twiddles: broadcast storage, as the lane pass's
// (chain_builder::shared_twiddles). src and dst: stride 1, distinct. Reads m, s, dst_cols,
// dst_qstride, twiddles (n holds the line's length, for the work graph's cost,
// work_graph.cpp; M = R m). In chirp (a Bluestein stage's chirp pass fused in, experiment X50
// arm C): element i of the line is src[i] w[i] for i < Q and zero from Q on, w the chirp in
// per-element storage (in_factor, chain_builder::bluestein_tables_of) and Q = src_cols; src
// then holds only Q elements. C divides s then (chain_builder::bluestein_depth_fits), so
// every block is whole and a run's chirp entries are one aligned block of the table. In c2r (a
// c2r row's pairing fused in, experiment X78): the line of h = n (the pass's n) elements is Z
// of run_in_c2r, from the h + 1 bins of src, with v in per-element storage (in_factor,
// chain_builder::depth_c2r_twiddles); C divides s then too (chain_builder::c2r_depth_fits).
template <class T, std::size_t R, int Sign, first_input In = first_input::plain>
void depth_first_pass(const pass_args* a, const void* src, void* dst, std::size_t first,
                      std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::size_t m = a->m;
  const std::size_t s = a->s;
  const std::size_t sub = R * m;
  const std::size_t cols = a->dst_cols;
  const auto lane = static_cast<std::size_t>(a->dst_qstride);
  const std::size_t whole = s / C<T>;
  const std::size_t blocks = (s + C<T> - 1) / C<T>;
  const T* table = static_cast<const T*>(a->twiddles);
  [[maybe_unused]] const std::ptrdiff_t in_leg = at(s * m, 2);
  std::size_t p = first < blocks ? 0 : first / blocks;
  std::size_t b = first < blocks ? first : first % blocks;
  // q0 = b C as (q0 % cols, q0 / cols), stepped rather than divided per unit.
  std::size_t d1 = b * C<T> % cols;
  std::size_t d2 = b * C<T> / cols;
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t q0 = b * C<T>;
    if (b < whole) {
      cv<T> v[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      const to_registers<T> out{v};
      if constexpr (In == first_input::chirp) {
        const std::size_t i0 = q0 + s * p;
        lane_unit<T, R, Sign, true>(
            run_in_chirp<T>{
                .p = x + at(i0, 2),
                .w = static_cast<const T*>(a->in_factor) + tw_reals<T> * i0,
                .i = i0,
                .leg = s * m,
                .data = a->src_cols,
            },
            out, table, p);
      } else if constexpr (In == first_input::c2r) {
        const std::size_t i0 = q0 + s * p;
        lane_unit<T, R, Sign, true>(
            run_in_c2r<T>{
                .x = x,
                .w = static_cast<const T*>(a->in_factor) + tw_reals<T> * i0,
                .i = i0,
                .leg = s * m,
                .h = a->n,
            },
            out, table, p);
      } else {
        lane_unit<T, R, Sign, true>(
            run_in<T, edge_op::none, true>{
                .p = x + at(q0 + s * p, 2),
                .leg = in_leg,
                .step = 2,
                .valid = C<T>,
                .op = {},
            },
            out, table, p);
      }
      T* yp = y + at(d1 * lane + d2 * sub + R * p, 2);
      if constexpr (R % C<T> == 0) {
        store_transposed<T, R>(yp, v, C<T>, lane);
      } else if constexpr (masked) {
        store_padded<T, R>(yp, v, C<T>, lane);
      } else {
        store_tiled<T, R>(yp, v, C<T>, lane);
      }
    } else if constexpr (!masked && C<T> > 1) {
      // C = 1 (NEON double) has no partial block: no tail code in its loop (X23).
      depth_first_tails<T, R, Sign>(a, x, y, u, u + 1, blocks);
    }
    d1 += C<T>;
    if (d1 == cols) {
      d1 = 0;
      ++d2;
    }
    if (++b == blocks) {
      b = 0;
      d1 = 0;
      d2 = 0;
      ++p;
    }
  }
  if constexpr (masked && C<T> > 1) {
    if (whole != blocks) {
      depth_first_tails<T, R, Sign>(a, x, y, first, last, blocks);
    }
  }
}

// Leg j of a run times a twiddle every lane shares (j >= 1; leg 0 untouched): the value at
// hi + tw_reals (j - 1), broadcast. With the codelet's per-lane twiddles after it, a leg is
// multiplied by w_n^(r k_hi B) and then by w_n^(r k_lo) (the top pass below).
template <class T, int Sign> struct run_in_twiddled {
  const T* p;
  std::ptrdiff_t leg;
  const T* hi;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const {
    const cv<T> v = load(p + at(j, leg));
    if (j == 0) {
      return v;
    }
    return twiddle<T, Sign>(v, bcast_tw(hi + tw_reals<T> * (j - 1)));
  }
};

// The top pass of a depth-first chain of n = R M: for k < M, legs r < R at k + M r times
// w_n^(r k), the R-point DFT (the DIT codelet), outputs t at k + M t. Lanes over C adjacent
// k (C divides M), unit u is k = u C .. u C + C - 1. The twiddles are derived (experiment
// X46 arm B): for k = k_hi B + k_lo, B = s (a multiple of C dividing M), the per-lane table
// `twiddles` of w_n^(r k_lo) (rows r - 1 of B entries) and the broadcast table `roots` of
// w_n^(r k_hi B) (R - 1 per k_hi), applied one after the other: about (R - 1)(B + M / B)
// entries instead of the (R - 1) M of a stored table, which held as many bytes as the row on
// x86 and kept the chain missing L2 (X46: 0.3-0.7 misses per point at 16384-25000 on the
// Xeon 6240, 0.00-0.01 derived), for one more complex multiplication per leg. Every output
// depends on every input, so src may be dst (it is, in the chain). The output edge's
// operation (none, the scale, or a factor per element of the line: a Bluestein convolution's
// kernel spectrum, experiment X50) applies to what is stored. Reads m (M), s, twiddles,
// roots, scale, out_factor. Chirp (a Bluestein stage's chirp pass out fused in, experiment
// X50 arm C): only the outputs below Q = dst_cols are stored, each times the chirp
// (out_factor, per-element storage) with the output edge's operation (none or the scale);
// dst then holds Q elements and is not src.
template <class T, std::size_t R, int Sign, edge_op OutOp, bool Chirp = false>
void depth_top_pass(const pass_args* a, const void* src, void* dst, std::size_t first,
                    std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::size_t m = a->m;
  const std::size_t block = a->s;
  const T* table = static_cast<const T*>(a->twiddles);
  const std::ptrdiff_t row = at(block, static_cast<std::ptrdiff_t>(tw_reals<T>));
  const std::ptrdiff_t leg = at(m, 2);
  std::size_t k_lo = first * C<T>;
  const T* hi = static_cast<const T*>(a->roots) + tw_reals<T> * (R - 1) * (k_lo / block);
  k_lo %= block;
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t k0 = u * C<T>;
    if constexpr (Chirp) {
      cdft<R>::template dit<T, Sign>(
          run_in_twiddled<T, Sign>{.p = x + at(k0, 2), .leg = leg, .hi = hi},
          run_out_chirp<T, OutOp>{
              .p = y + at(k0, 2),
              .w = static_cast<const T*>(a->out_factor) + tw_reals<T> * k0,
              .i = k0,
              .leg = m,
              .data = a->dst_cols,
              .op = make_edge<T, OutOp>(a, nullptr, 0, 0),
          },
          lane_twiddles<T>{.p = table + tw_reals<T> * k_lo, .row = row});
    } else {
      cdft<R>::template dit<T, Sign>(
          run_in_twiddled<T, Sign>{.p = x + at(k0, 2), .leg = leg, .hi = hi},
          run_out<T, OutOp, true>{
              .p = y + at(k0, 2),
              .leg = leg,
              .step = 2,
              .valid = C<T>,
              .op = make_edge<T, OutOp>(a, a->out_factor, k0, m),
          },
          lane_twiddles<T>{.p = table + tw_reals<T> * k_lo, .row = row});
    }
    k_lo += C<T>;
    if (k_lo == block) {
      k_lo = 0;
      hi += tw_reals<T> * (R - 1);
    }
  }
}

} // namespace
