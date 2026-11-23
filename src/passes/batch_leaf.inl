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
// chain on 15 x 2 and 64 x 2). Here element j of C rows is gathered into one register by
// C x C transposes in registers: C consecutive elements of each row loaded as one vector,
// transposed, and the block's registers are the codelet's inputs j .. j + C - 1. The outputs
// go back the same way. Where C does not divide R the last block overlaps the one before
// it (it starts at R - C), so every access is a whole vector: the overlap loads elements
// twice and stores the same values twice. Rows shorter than C take one element of each row
// at a time (C broadcasts, transposed: lane l of register 0 is row l's element), and store
// lane l of an output from register l of the transposed broadcasts.
//
// Each lane does the arithmetic the leaf does to its row (the same codelet, with the edge's
// scale on loading or storing), so the bits are the leaf's of R, the narrow leaf's on x86
// (which round alike), whatever the group a row falls in: every row of a plan gets the same
// arithmetic, as DESIGN 15.5 requires (not a plan of one row's, where that row is not the
// leaf of R). A last row alone runs as the leaf runs it
// (broadcast into every lane, lane 0 stored): a partial group gathers and scatters as a whole
// one does and cost about as much (X53 arm B3); a partial group of more rows (C > 2) holds
// zeros in its idle lanes and neither reads nor writes their rows.
//
// X53's arm B2 gathered each input when the codelet asked for it (C broadcasts, transposed)
// and stored each output lane by lane, so that no more values were live than in the leaf: it
// was slower than the block transposes on the Xeon 6240 (GCC 14: 1.05-1.25 of them where
// they win) and the M5 Max (Clang 23, float: 1.1-1.2; runs in the experiment file).

namespace {

// The codelet's inputs from registers already loaded.
template <class T> struct from_registers {
  const cv<T>* v;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const { return v[j]; }
};

// The first element of block b: b C, or R - C for the block that overlaps when C does not
// divide R.
template <class T, std::size_t R> constexpr std::size_t batch_block_start(std::size_t b) {
  return (b + 1) * C<T> <= R ? b * C<T> : R - C<T>;
}

// The blocks of R elements: whole vectors where R >= C.
template <class T, std::size_t R>
inline constexpr std::size_t batch_blocks = R >= C<T> ? (R + C<T> - 1) / C<T> : 0;

// Elements 0 .. R - 1 of rows row[0 .. C - 1] (the first `valid`; Full: all C) into x, row l
// in lane l, the other lanes zero, each times the input edge.
template <class T, std::size_t R, bool Full, edge_op InOp>
FFTN_FORCE_INLINE void batch_gather(const T* const* row, std::size_t valid, const edge<T, InOp>& e,
                                    cv<T>* x) {
  for (std::size_t b = 0; b < batch_blocks<T, R>; ++b) {
    const std::size_t j0 = batch_block_start<T, R>(b);
    cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    for (std::size_t l = 0; l < C<T>; ++l) {
      v[l] = Full || l < valid ? load(row[l] + at(j0, 2)) : zeros<T>();
    }
    transpose(v);
    for (std::size_t l = 0; l < C<T>; ++l) {
      x[j0 + l] = on_one<T, InOp>(e, v[l], j0 + l);
    }
  }
  if constexpr (R < C<T>) {
    for (std::size_t j = 0; j < R; ++j) {
      cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      for (std::size_t l = 0; l < C<T>; ++l) {
        v[l] = Full || l < valid ? bcast(row[l] + at(j, 2)) : zeros<T>();
      }
      transpose(v);
      x[j] = on_one<T, InOp>(e, v[0], j);
    }
  }
}

// The inverse for the outputs y: lane l of y[k] times the output edge to element k of row
// row[l], for the first `valid` rows (Full: all C).
template <class T, std::size_t R, bool Full, edge_op OutOp>
FFTN_FORCE_INLINE void batch_scatter(T* const* row, std::size_t valid, const edge<T, OutOp>& e,
                                     const cv<T>* y) {
  for (std::size_t b = 0; b < batch_blocks<T, R>; ++b) {
    const std::size_t k0 = batch_block_start<T, R>(b);
    cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    for (std::size_t l = 0; l < C<T>; ++l) {
      v[l] = on_one<T, OutOp>(e, y[k0 + l], k0 + l);
    }
    transpose(v);
    for (std::size_t l = 0; l < C<T>; ++l) {
      if (Full || l < valid) {
        store(row[l] + at(k0, 2), v[l]);
      }
    }
  }
  if constexpr (R < C<T>) {
    for (std::size_t k = 0; k < R; ++k) {
      cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      const cv<T> out = on_one<T, OutOp>(e, y[k], k);
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
  }
}

// The rows in[], out[] (the first `valid`; Full: all C): gathered, the codelet, scattered.
// Every load precedes every store, so a row's source may be its destination (in place).
template <class T, std::size_t R, int Sign, edge_op InOp, edge_op OutOp, bool Full>
FFTN_FORCE_INLINE void batch_run(const T* const* in, T* const* out, std::size_t valid,
                                 const edge<T, InOp>& ein, const edge<T, OutOp>& eout) {
  cv<T> x[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  cv<T> y[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  batch_gather<T, R, Full, InOp>(in, valid, ein, x);
  cdft<R>::template plain<T, Sign>(from_registers<T>{x}, to_registers<T>{y});
  batch_scatter<T, R, Full, OutOp>(out, valid, eout, y);
}

// Experiment X62: on avx2 in double (two rows per register), the leaves of radix 32 and 64
// take each input from its two rows when the codelet asks for it (load_halves: a 128-bit load
// and an insert from memory, FFTW's form) and store each output's halves as the codelet
// produces it. The block transposes above hold all R inputs and all R outputs at once, which
// past 16 registers become a copy through the stack on each side; here the codelet's own
// order decides its spills (X54). Same arithmetic per lane, so the same bits.
template <class T, std::size_t R>
inline constexpr bool batch_by_element = R >= 32 && C<T> == 2 && id == target::avx2;

template <class T, edge_op InOp> struct pair_rows_in {
  const T* r0;
  const T* r1;
  edge<T, InOp> e;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const {
    return on_one<T, InOp>(e, load_halves(r0 + at(j, 2), r1 + at(j, 2)), j);
  }
};

template <class T, edge_op OutOp> struct pair_rows_out {
  T* r0;
  T* r1;
  edge<T, OutOp> e;
  FFTN_FORCE_INLINE void put(std::size_t k, cv<T> y) const {
    store_halves(r0 + at(k, 2), r1 + at(k, 2), on_one<T, OutOp>(e, y, k));
  }
};

// Rows u .. u + valid - 1 (Full: valid = C).
template <class T, std::size_t R, int Sign, edge_op InOp, edge_op OutOp, bool Full>
FFTN_FORCE_INLINE void batch_group(const void* src, void* dst, std::ptrdiff_t src_dist,
                                   std::ptrdiff_t dst_dist, std::size_t u, std::size_t valid,
                                   const edge<T, InOp>& ein, const edge<T, OutOp>& eout) {
  if constexpr (Full) {
    const T* in[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    T* out[C<T>];      // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    for (std::size_t l = 0; l < C<T>; ++l) {
      in[l] = line_at<T>(src, u + l, src_dist);
      out[l] = line_at<T>(dst, u + l, dst_dist);
    }
    batch_run<T, R, Sign, InOp, OutOp, true>(in, out, valid, ein, eout);
  } else {
    // Initialised only here: GCC 14 reports a partial group's arrays maybe uninitialized, and
    // initialising a whole group's as well changed the code of most of its leaves.
    const T* in[C<T>] = {}; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    T* out[C<T>] = {};      // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    for (std::size_t l = 0; l < C<T>; ++l) {
      // Idle rows form no address: their pointers stay at the group's first row.
      const std::size_t r = l < valid ? u + l : u;
      in[l] = line_at<T>(src, r, src_dist);
      out[l] = line_at<T>(dst, r, dst_dist);
    }
    batch_run<T, R, Sign, InOp, OutOp, false>(in, out, valid, ein, eout);
  }
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
  if constexpr (batch_by_element<T, R>) {
    // Rows u and u + 1 through one copy of the codelet, and a last row alone in both lanes
    // (its halves loaded and stored twice) through the same copy. The codelet reads every input
    // before it writes an output (each output depends on every input), so src may be dst.
    // X62 arm K3: a second inlined copy for the last row, in the leaf's form (arm K) or with
    // these accessors (arm K2), took 570-750 cycles for that row against 260-310 where it
    // stayed out of the pairs' way: the two copies evicted each other from the op cache
    // (190-300 misses a call on the EPYC 7763 at 5, 7 and 17 rows backward, none at 4).
    for (std::size_t u = 0; u < count; u += 2) {
      const std::size_t v = u + 1 < count ? u + 1 : u;
      const pair_rows_in<T, InOp> x{
          .r0 = line_at<T>(src, u, src_dist), .r1 = line_at<T>(src, v, src_dist), .e = ein};
      const pair_rows_out<T, OutOp> y{
          .r0 = line_at<T>(dst, u, dst_dist), .r1 = line_at<T>(dst, v, dst_dist), .e = eout};
      cdft<R>::template plain<T, Sign>(x, y);
    }
    return;
  }
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
