// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Pair leaves (experiment X28b, dev/experiments/X28b-short-leaves.md): the chain R1 x M of
// a short row, its two passes held in registers, as one leaf_fn. Kernel text in the manner
// of stockham.inl (included inside a region after its vocabulary, the codelets and
// access.inl; no standard-library templates; entry points take pointers and integers). No
// include guard.
//
// For n = R1 M with C dividing M, the chain's first pass (grouped, stockham.inl) is M / C
// whole units: unit u loads legs j of the butterflies p = u C .. u C + C - 1 and leaves
// v_u[k] holding y[R1 p + k] in lane p - u C. Its second and last pass (plain, lanes over
// the R1 copies q) reads leg j = u C + l of copies q0 .. q0 + C - 1 as y[q + R1 j]: lane
// q - q0 of that register is lane l of v_u[q], so the C x C transpose of the block
// v_u[q0 .. q0 + C - 1] (with legs past R1 as zeros) is exactly the second pass's input.
// The whole transform then needs about n / C registers, no store and load between the
// passes, and no partial unit in memory: the last block of copies is computed in full and
// stored with a partial run. Each lane does the arithmetic the chain R1 x M with
// interleaved passes does to that element, so the bits are that chain's (a test forces it
// through build_settings::radices).

namespace {

// The second pass's input for block Bk of copies: leg j from the first pass's transposed
// registers, v[u RP + k] for unit u and leg k (RP legs per unit, R1 rounded up to C).
template <class T, std::size_t RP, std::size_t Bk> struct pair_input {
  const cv<T>* v;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const {
    return v[(j / C<T>)*RP + Bk * C<T> + j % C<T>];
  }
};

// Unit U0 of the first pass and the ones after it; every index into v is a constant.
template <class T, std::size_t R1, std::size_t M, int Sign, edge_op InOp, std::size_t RP,
          std::size_t U0>
FFTN_FORCE_INLINE void pair_first(const pass_args* a, const T* x, const T* table, cv<T>* v) {
  constexpr std::size_t p0 = U0 * C<T>;
  const run_in<T, InOp, true> in{
      .p = x + 2 * p0,
      .leg = static_cast<std::ptrdiff_t>(2 * M),
      .step = 2,
      .valid = C<T>,
      .op = make_edge<T, InOp>(a, a->in_factor, p0, M),
  };
  const lane_twiddles<T> twid{
      .p = table + tw_reals<T> * p0,
      .row = static_cast<std::ptrdiff_t>(M * tw_reals<T>),
  };
  cv<T>* w = v + U0 * RP;
  const to_registers<T> out{w};
  cdft<R1>::template dif<T, Sign>(in, out, twid);
  for (std::size_t k = R1; k < RP; ++k) {
    w[k] = zeros<T>();
  }
  for (std::size_t b = 0; b < RP / C<T>; ++b) {
    transpose(w + b * C<T>);
  }
  if constexpr (p0 + C<T> < M) {
    pair_first<T, R1, M, Sign, InOp, RP, U0 + 1>(a, x, table, v);
  }
}

// Block Bk of copies of the second pass and the ones after it: outputs k of copies
// q0 .. q0 + C - 1 to y[q + R1 k], the last block partial when C does not divide R1.
template <class T, std::size_t R1, std::size_t M, int Sign, edge_op OutOp, std::size_t RP,
          std::size_t Bk>
FFTN_FORCE_INLINE void pair_second(const pass_args* a, T* y, const cv<T>* v) {
  constexpr std::size_t q0 = Bk * C<T>;
  constexpr bool full = q0 + C<T> <= R1;
  const pair_input<T, RP, Bk> in{v};
  const run_out<T, OutOp, full> out{
      .p = y + 2 * q0,
      .leg = static_cast<std::ptrdiff_t>(2 * R1),
      .step = 2,
      .valid = full ? C<T> : R1 - q0,
      .op = make_edge<T, OutOp>(a, a->out_factor, q0, R1),
  };
  cdft<M>::template plain<T, Sign>(in, out);
  if constexpr (q0 + C<T> < R1) {
    pair_second<T, R1, M, Sign, OutOp, RP, Bk + 1>(a, y, v);
  }
}

// The transform of n = R1 M points of `count` lines, each of stride 1 on both sides (line
// u at src + u src_dist, dst + u dst_dist bytes), with both edges' operations. twiddles:
// the first pass's table as the grouped pass reads it (per-lane storage, R1 - 1 rows of M
// entries, row k - 1 holding w_n^(p k)). Every load of a line precedes its first store, so
// src may be dst (in place). Reads twiddles, scale, in_factor, out_factor.
template <class T, std::size_t R1, std::size_t M, int Sign, edge_op InOp, edge_op OutOp>
void pair_leaf(const pass_args* a, const void* src, void* dst, std::ptrdiff_t src_dist,
               std::ptrdiff_t dst_dist, std::size_t count) {
  static_assert(M % C<T> == 0, "a pair leaf's first pass has whole units only");
  constexpr std::size_t RP = (R1 + C<T> - 1) / C<T> * C<T>;
  const T* table = static_cast<const T*>(a->twiddles);
  constexpr auto real_bytes = static_cast<std::ptrdiff_t>(sizeof(T));
  const std::ptrdiff_t src_step = src_dist / real_bytes;
  const std::ptrdiff_t dst_step = dst_dist / real_bytes;
  for (std::size_t u = 0; u < count; ++u) {
    const auto k = static_cast<std::ptrdiff_t>(u);
    cv<T> v[M / C<T> * RP]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    pair_first<T, R1, M, Sign, InOp, RP, 0>(a, static_cast<const T*>(src) + k * src_step, table,
                                            v);
    pair_second<T, R1, M, Sign, OutOp, RP, 0>(a, static_cast<T*>(dst) + k * dst_step, v);
  }
}

} // namespace
