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

// The first pass of the chains of the s sub-rows of a line of n = s R m (DESIGN.md 8.2's
// lane pass with s copies, m groups, twiddled by w_(R m)^(p k)), every copy's outputs
// written to its own block of M = R m elements: copy q's y[o(q) + R p + k] instead of the
// lane pass's y[q + s (R p + k)], where o(q) = (q % cols) dst_qstride + (q / cols) M. With
// cols = s (one top pass) the blocks are in copy order, o(q) = q M; with two top passes of
// F1 = cols and F2 = s / cols, copy q = d1 + F1 d2 goes to block d1 F2 + d2, so that each
// of the F1 sub-transforms of the first top pass has its F2 sub-rows together
// (dst_qstride = F2 M). Unit u is group p = u / blocks and copies q = (u % blocks) C .. + C
// - 1, blocks = s / C (C divides cols, so a unit's copies are adjacent within one group of
// cols); its R outputs are transposed in registers so that each store writes C legs of one
// copy. twiddles: broadcast storage, as the lane pass's (chain_builder::shared_twiddles).
// src and dst: stride 1, distinct. Reads m, s, n (M), dst_cols, dst_qstride, twiddles.
template <class T, std::size_t R, int Sign>
void depth_first_pass(const pass_args* a, const void* src, void* dst, std::size_t first,
                      std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::size_t m = a->m;
  const std::size_t s = a->s;
  const std::size_t sub = a->n;
  const std::size_t cols = a->dst_cols;
  const auto lane = static_cast<std::size_t>(a->dst_qstride);
  const std::size_t blocks = s / C<T>;
  const T* table = static_cast<const T*>(a->twiddles);
  const std::ptrdiff_t in_leg = at(s * m, 2);
  std::size_t p = first < blocks ? 0 : first / blocks;
  std::size_t b = first < blocks ? first : first % blocks;
  // q0 = b C as (q0 % cols, q0 / cols), stepped rather than divided per unit.
  std::size_t d1 = b * C<T> % cols;
  std::size_t d2 = b * C<T> / cols;
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t q0 = b * C<T>;
    cv<T> v[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    const to_registers<T> out{v};
    lane_unit<T, R, Sign, true>(
        run_in<T, edge_op::none, true>{
            .p = x + at(q0 + s * p, 2), .leg = in_leg, .step = 2, .valid = C<T>, .op = {}},
        out, table, p);
    T* yp = y + at(d1 * lane + d2 * sub + R * p, 2);
    if constexpr (R % C<T> == 0) {
      store_transposed<T, R>(yp, v, C<T>, lane);
    } else if constexpr (masked) {
      store_padded<T, R>(yp, v, C<T>, lane);
    } else {
      store_tiled<T, R>(yp, v, C<T>, lane);
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
}

// The top pass of a depth-first chain of n = R M: for k < M, legs r < R at k + M r times
// w_n^(r k), the R-point DFT (the DIT codelet), outputs t at k + M t. Lanes over C adjacent
// k (C divides M), unit u is k = u C .. u C + C - 1. twiddles: the grouped first pass's table
// of (n, R) (chain_builder::grouped_twiddles), whose row r - 1 holds w_n^(k r) for k < M, in
// per-lane storage. Every output depends on every input, so src may be dst (it is, in the
// chain). The output edge's operation (none or the scale) applies to what is stored. Reads
// m (M), twiddles, scale.
template <class T, std::size_t R, int Sign, edge_op OutOp>
void depth_top_pass(const pass_args* a, const void* src, void* dst, std::size_t first,
                    std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::size_t m = a->m;
  const T* table = static_cast<const T*>(a->twiddles);
  const std::ptrdiff_t row = at(m, static_cast<std::ptrdiff_t>(tw_reals<T>));
  const std::ptrdiff_t leg = at(m, 2);
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t k0 = u * C<T>;
    cdft<R>::template dit<T, Sign>(
        run_in<T, edge_op::none, true>{
            .p = x + at(k0, 2), .leg = leg, .step = 2, .valid = C<T>, .op = {}},
        run_out<T, OutOp, true>{.p = y + at(k0, 2),
                                .leg = leg,
                                .step = 2,
                                .valid = C<T>,
                                .op = make_edge<T, OutOp>(a, nullptr, 0, 0)},
        lane_twiddles<T>{.p = table + tw_reals<T> * k0, .row = row});
  }
}

} // namespace
