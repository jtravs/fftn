// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Batch leaves (experiment X53, dev/experiments/X53-two-row-leaves.md): the leaf of a
// codelet radix R over C rows of a batch at once, row u + l in lane l of every register.
// Kernel text in the manner of stockham.inl (included inside a region after its vocabulary,
// the codelets and access.inl; no standard-library templates; entry points take pointers
// and integers). No include guard.
//
// A leaf (stockham.inl's leaf_lines, unit_leaf.inl) broadcasts each element into every
// lane, so on C > 1 the other lanes repeat the work; the narrow leaves of the x86 targets
// halve that waste, and the lanes strategy fills the lanes with rows but moves them through
// a tile in memory, which costs more than a leaf for short batches (X49: 1.6-10 times the
// chain on 15 x 2 and 64 x 2). Here the codelet's input j is element j of C rows, gathered
// when the codelet asks for it (C broadcasts, transposed: lane l of register 0 is row l's
// element; on avx2 in double the compilers make it one load and one insert of the other
// half from memory), and its output k goes back lane by lane (the output in every register,
// transposed: lane 0 of register l is lane l; one store of the low half and one of the high
// half from the register). The loads and stores stay where the codelet puts them, as in the
// leaf, so a codelet needs no more registers than the leaf's. X53's first arm gathered all R
// inputs by C x C block transposes before the codelet and scattered the outputs after it:
// all R live at once, and GCC 16 spilled the radix-15 leaf's registers about twice as often.
//
// Each lane does the arithmetic the leaf does to its row (the same codelet, with the edge's
// scale on loading or storing), so the bits are the leaf's, the narrow leaf's on x86 (which
// round alike), whatever the group a row falls in. A last row alone runs as the leaf runs it
// (broadcast into every lane, lane 0 stored); a partial group of more (C > 2) holds zeros in
// its idle lanes and neither reads nor writes their rows.

namespace {

// Element j of rows row[0 .. C - 1] (the first `valid`; Full: all C), row l in lane l, the
// other lanes zero, times the input edge.
template <class T, bool Full, edge_op InOp> struct rows_in {
  const T* const* row;
  std::size_t valid;
  edge<T, InOp> e;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const {
    cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    for (std::size_t l = 0; l < C<T>; ++l) {
      v[l] = Full || l < valid ? bcast(row[l] + at(j, 2)) : zeros<T>();
    }
    transpose(v);
    return on_one<T, InOp>(e, v[0], j);
  }
};

// Lane l of output k times the output edge to element k of row row[l], for the first `valid`
// rows (Full: all C).
template <class T, bool Full, edge_op OutOp> struct rows_out {
  T* const* row;
  std::size_t valid;
  edge<T, OutOp> e;
  FFTN_FORCE_INLINE void put(std::size_t k, cv<T> y) const {
    const cv<T> out = on_one<T, OutOp>(e, y, k);
    cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    for (std::size_t l = 0; l < C<T>; ++l) {
      v[l] = out;
    }
    transpose(v); // v[l]: lane l of out, in every lane
    for (std::size_t l = 0; l < C<T>; ++l) {
      if (Full || l < valid) {
        store_first(row[l] + at(k, 2), v[l]);
      }
    }
  }
};

// Rows u .. u + valid - 1 (Full: valid = C) through the codelet. The codelet reads every
// input before it writes an output, so src may be dst (in place).
template <class T, std::size_t R, int Sign, edge_op InOp, edge_op OutOp, bool Full>
FFTN_FORCE_INLINE void batch_group(const void* src, void* dst, std::ptrdiff_t src_dist,
                                   std::ptrdiff_t dst_dist, std::size_t u, std::size_t valid,
                                   const edge<T, InOp>& ein, const edge<T, OutOp>& eout) {
  const T* in[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  T* out[C<T>];      // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  for (std::size_t l = 0; l < C<T>; ++l) {
    // Idle rows form no address: their pointers stay at the group's first row.
    const std::size_t r = Full || l < valid ? u + l : u;
    in[l] = line_at<T>(src, r, src_dist);
    out[l] = line_at<T>(dst, r, dst_dist);
  }
  const rows_in<T, Full, InOp> x{.row = in, .valid = valid, .e = ein};
  const rows_out<T, Full, OutOp> y{.row = out, .valid = valid, .e = eout};
  cdft<R>::template plain<T, Sign>(x, y);
}

// The transform of R points of `count` lines of stride 1 on both sides (line u at src +
// u src_dist, dst + u dst_dist bytes), C lines at a time, with the scale on either edge.
// Reads scale.
template <class T, std::size_t R, int Sign, edge_op InOp, edge_op OutOp>
void batch_leaf(const pass_args* a, const void* src, void* dst, std::ptrdiff_t src_dist,
                std::ptrdiff_t dst_dist, std::size_t count) {
  static_assert(InOp != edge_op::factor && OutOp != edge_op::factor,
                "the factor edges keep leaf_lines");
  const edge<T, InOp> ein = make_edge<T, InOp>(a, nullptr, 0, 1);
  const edge<T, OutOp> eout = make_edge<T, OutOp>(a, nullptr, 0, 1);
  std::size_t u = 0;
  for (; u + C<T> <= count; u += C<T>) {
    batch_group<T, R, Sign, InOp, OutOp, true>(src, dst, src_dist, dst_dist, u, C<T>, ein, eout);
  }
  if (count - u == 1) {
    // One row left: the leaf's own form, the row in every lane (one load and one store per
    // element), which costs a row what the leaf does; a partial group gathers and scatters
    // as a whole group does and cost about as much as two leaves (X53 arm B3).
    one_in<T, InOp> in{.p = line_at<T>(src, u, src_dist), .leg = 2, .op = ein};
    one_out<T, OutOp> out{.p = line_at<T>(dst, u, dst_dist), .leg = 2, .op = eout};
    cdft<R>::template plain<T, Sign>(in, out);
  } else if constexpr (C<T> > 2) {
    if (u < count) {
      batch_group<T, R, Sign, InOp, OutOp, false>(src, dst, src_dist, dst_dist, u, count - u, ein,
                                                  eout);
    }
  }
}

} // namespace
