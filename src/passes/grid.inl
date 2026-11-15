// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The grid strategy's own passes (DESIGN.md 8.5, src/strategies/grid.hpp): moving a block
// of columns between the caller's memory and a tight scratch block, plain or with the
// twiddles between phases, and the twiddled transpose of the first phase. Everything else in a grid
// is the chain's lane passes on the scratch block. Kernel text in the manner of stockham.inl
// (included inside a region after access.inl; no standard-library templates). No include guard.
//
// A block is `s` adjacent columns of `units` rows: element (q, r) of the block is at
// q + stride r of each side (src_stride, dst_stride, in complex elements). s is a multiple
// of C, so every access is a whole register; the grid is planned only where it is.

namespace {

// How many rows ahead a block copy asks for the rows it will read or write next: the rows
// of a long grid lie a page or more apart, a stride hardware prefetchers do not follow
// (experiment X12).
inline constexpr std::size_t rows_ahead = 8;

// Reals per 64-byte cache line.
template <class T> inline constexpr std::size_t line_reals = 64 / sizeof(T);

// The copy loops below move rows of V registers (V = 0: `vectors`, known only at run time),
// and with Ahead ask for the row rows_ahead further on, one prefetch per cache line of it.
// Rows of a few registers are the common case: with blocks of 4 columns on AVX-512 every
// row is one 64-byte line, and a loop that multiplied each row's address, ran two one-trip
// prefetch loops and a one-trip inner loop spent about 33 instructions on it, 40-52 % of a
// long row's instructions (experiments X25, X26). Here the pointers advance by the strides,
// the rows that prefetch run as their own loop, and a V of 1, 2, 4 or 8 is a compile-time
// count, so that a row costs its loads, stores and prefetches.

// Asks for the `lines` cache lines from p, to read (Write = 0) or to write (1).
template <class T, int Write> FFTN_FORCE_INLINE void prefetch_lines(const T* p, std::size_t lines) {
  for (std::size_t i = 0; i < lines; ++i) {
    __builtin_prefetch(p + i * line_reals<T>, Write);
  }
}

template <class T> FFTN_FORCE_INLINE std::size_t lines_of(std::size_t vectors) {
  return (vectors * 2 * C<T> + line_reals<T> - 1) / line_reals<T>;
}

// y[q + ys r] = x[q + xs r] for `rows` rows of V registers (strides in reals), scaled on
// storing for OutOp == scale, prefetching the rows ahead of x (AheadX) and of y (AheadY);
// leaves x and y past the last row.
template <class T, edge_op OutOp, std::size_t V, bool AheadX, bool AheadY>
FFTN_FORCE_INLINE void copy_rows(const T*& x, T*& y, std::ptrdiff_t xs, std::ptrdiff_t ys,
                                 std::size_t rows, std::size_t vectors, T factor) {
  const std::size_t nv = V != 0 ? V : vectors;
  const std::size_t lines = lines_of<T>(nv);
  const std::ptrdiff_t xa = at(rows_ahead, xs);
  const std::ptrdiff_t ya = at(rows_ahead, ys);
  for (std::size_t r = 0; r < rows; ++r) {
    if constexpr (AheadX) {
      prefetch_lines<T, 0>(x + xa, lines);
    }
    if constexpr (AheadY) {
      prefetch_lines<T, 1>(y + ya, lines);
    }
    for (std::size_t q = 0; q < nv; ++q) {
      cv<T> v = load(x + 2 * C<T> * q); // NOLINT(misc-const-correctness): set under if constexpr
      if constexpr (OutOp == edge_op::scale) {
        v = scale(v, factor);
      }
      store(y + 2 * C<T> * q, v);
    }
    x += xs;
    y += ys;
  }
}

// Only the side whose rows are apart is prefetched: the other is the tight scratch block,
// already in L1, and a prefetch of it only takes a fill buffer from the rows that need one
// (X26: prefetching both sides with the faster loop raised the Xeon's fill-buffer-full
// cycles fourfold and slowed the EPYC's 2^18-2^20 rows by up to 1.8 times).
template <class T, edge_op OutOp, std::size_t V>
FFTN_FORCE_INLINE void copy_block(const T* x, T* y, std::ptrdiff_t xs, std::ptrdiff_t ys,
                                  std::size_t rows, std::size_t vectors, T factor) {
  const std::size_t ahead = rows > rows_ahead ? rows - rows_ahead : 0;
  const auto tight = static_cast<std::ptrdiff_t>(2 * C<T> * (V != 0 ? V : vectors));
  if (xs != tight && ys != tight) {
    copy_rows<T, OutOp, V, true, true>(x, y, xs, ys, ahead, vectors, factor);
  } else if (xs != tight) {
    copy_rows<T, OutOp, V, true, false>(x, y, xs, ys, ahead, vectors, factor);
  } else if (ys != tight) {
    copy_rows<T, OutOp, V, false, true>(x, y, xs, ys, ahead, vectors, factor);
  } else {
    copy_rows<T, OutOp, V, false, false>(x, y, xs, ys, ahead, vectors, factor);
  }
  copy_rows<T, OutOp, V, false, false>(x, y, xs, ys, rows - ahead, vectors, factor);
}

// Copies a block row by row: y[q + dst_stride r] = x[q + src_stride r], scaled on storing
// for OutOp == scale (the output edge, DESIGN.md 12.1). A unit is a row. The gather into
// scratch and the scatter out of it: reading or writing whole rows at once is what keeps a
// column block's rows, which lie a power of two apart for most long lengths, from evicting
// each other while a pass works on them (experiment X12). Reads src, dst, src_stride,
// dst_stride, s, scale.
template <class T, edge_op OutOp>
void block_copy(const pass_args* a, std::size_t first, std::size_t last) {
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const T* x = static_cast<const T*>(a->src) + at(first, xs);
  T* y = static_cast<T*>(a->dst) + at(first, ys);
  const std::size_t vectors = a->s / C<T>;
  const T factor = static_cast<T>(a->scale);
  const std::size_t rows = last - first;
  switch (vectors) {
  case 1:
    copy_block<T, OutOp, 1>(x, y, xs, ys, rows, vectors, factor);
    break;
  case 2:
    copy_block<T, OutOp, 2>(x, y, xs, ys, rows, vectors, factor);
    break;
  case 4:
    copy_block<T, OutOp, 4>(x, y, xs, ys, rows, vectors, factor);
    break;
  case 8:
    copy_block<T, OutOp, 8>(x, y, xs, ys, rows, vectors, factor);
    break;
  default:
    copy_block<T, OutOp, 0>(x, y, xs, ys, rows, vectors, factor);
    break;
  }
}

// Phase 1 of a grid: z, a tight block of s columns and m rows (z[q + s k]), to the block's
// s rows of m in the next array, y[k + m q], element (q, k) times the twiddle w^((c0 + q) k)
// of its column c0 + q (the conjugate for Sign = +1), formed as w^(q k) w^(c0 k): roots
// holds the first factor for every (q, k) of a block, twiddles the second for the block's
// rows (m values; the stage advances them per block, stage::moving_twiddles). Both are
// plain interleaved values (not twiddle storage: they multiply as factor arrays do, cmul);
// roots holds C x C per unit in unit order, row i of a unit's its C columns at row k0 + i.
// The product rounds once more than a stored twiddle would; storing w^((c0 + q) k) for every
// element instead is a table of n values, which X12 measured costing more than the
// multiplication once it no longer fits the cache. A unit is a C x C tile, tiles k fastest
// within a column tile, transposed in registers so that every store is C adjacent elements
// of one row of y. m and s are multiples of C. Reads src, dst, m, s, roots, twiddles.
template <class T, int Sign>
void twiddle_transpose(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const T* local = static_cast<const T*>(a->roots);
  const T* rows = static_cast<const T*>(a->twiddles);
  const std::size_t m = a->m;
  const std::size_t cols = a->s;
  const std::size_t ktiles = m / C<T>;
  std::size_t q0 = first / ktiles * C<T>;
  std::size_t k0 = first % ktiles * C<T>;
  for (std::size_t u = first; u < last; ++u) {
    cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    for (std::size_t i = 0; i < C<T>; ++i) {
      const cv<T> f = cmul(load(local + 2 * C<T> * (C<T> * u + i)), bcast(rows + 2 * (k0 + i)));
      const cv<T> x = load(z + 2 * (q0 + cols * (k0 + i)));
      if constexpr (Sign < 0) {
        v[i] = cmul(x, f);
      } else {
        v[i] = cmul(x, conj(f));
      }
    }
    if constexpr (C<T> > 1) {
      transpose(v);
    }
    for (std::size_t l = 0; l < C<T>; ++l) {
      store(y + 2 * (k0 + m * (q0 + l)), v[l]);
    }
    k0 += C<T>;
    if (k0 == m) {
      k0 = 0;
      q0 += C<T>;
    }
  }
}

// As copy_rows, row r times f = w[r] (one plain interleaved value per row; its conjugate
// for Sign = +1); only the destination prefetches.
template <class T, int Sign, std::size_t V, bool Ahead>
FFTN_FORCE_INLINE void twiddle_rows(const T*& x, T*& y, const T*& w, std::ptrdiff_t xs,
                                    std::ptrdiff_t ys, std::size_t rows, std::size_t vectors) {
  const std::size_t nv = V != 0 ? V : vectors;
  const std::size_t lines = lines_of<T>(nv);
  const std::ptrdiff_t ya = at(rows_ahead, ys);
  for (std::size_t r = 0; r < rows; ++r) {
    if constexpr (Ahead) {
      prefetch_lines<T, 1>(y + ya, lines);
    }
    cv<T> f = bcast(w); // NOLINT(misc-const-correctness): set under if constexpr
    if constexpr (Sign > 0) {
      f = conj(f);
    }
    for (std::size_t q = 0; q < nv; ++q) {
      store(y + 2 * C<T> * q, cmul(load(x + 2 * C<T> * q), f));
    }
    x += xs;
    y += ys;
    w += 2;
  }
}

template <class T, int Sign, std::size_t V>
FFTN_FORCE_INLINE void twiddle_block(const T* x, T* y, const T* w, std::ptrdiff_t xs,
                                     std::ptrdiff_t ys, std::size_t rows, std::size_t vectors) {
  const std::size_t ahead = rows > rows_ahead ? rows - rows_ahead : 0;
  twiddle_rows<T, Sign, V, true>(x, y, w, xs, ys, ahead, vectors);
  twiddle_rows<T, Sign, V, false>(x, y, w, xs, ys, rows - ahead, vectors);
}

// A later phase of a grid: the block's rows from tight scratch to the next array, as
// block_copy, row r times twiddles[r] (one plain interleaved value per row, the conjugate
// for Sign = +1); the stage advances the twiddles per column c (stage::moving_twiddles).
// Reads src, dst, src_stride, dst_stride, s, twiddles.
template <class T, int Sign>
void twiddle_scatter(const pass_args* a, std::size_t first, std::size_t last) {
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const T* x = static_cast<const T*>(a->src) + at(first, xs);
  T* y = static_cast<T*>(a->dst) + at(first, ys);
  const T* w = static_cast<const T*>(a->twiddles) + 2 * first;
  const std::size_t vectors = a->s / C<T>;
  const std::size_t rows = last - first;
  switch (vectors) {
  case 1:
    twiddle_block<T, Sign, 1>(x, y, w, xs, ys, rows, vectors);
    break;
  case 2:
    twiddle_block<T, Sign, 2>(x, y, w, xs, ys, rows, vectors);
    break;
  case 4:
    twiddle_block<T, Sign, 4>(x, y, w, xs, ys, rows, vectors);
    break;
  case 8:
    twiddle_block<T, Sign, 8>(x, y, w, xs, ys, rows, vectors);
    break;
  default:
    twiddle_block<T, Sign, 0>(x, y, w, xs, ys, rows, vectors);
    break;
  }
}

// Output leg k times one twiddle per row (the conjugate for Sign = +1): a later phase's
// twiddles, applied where its last pass stores.
template <class T, int Sign> struct row_twiddled_out {
  T* p;
  std::ptrdiff_t leg;
  const T* w;
  std::ptrdiff_t wleg;
  FFTN_FORCE_INLINE void put(std::size_t k, cv<T> v) const {
    cv<T> f = bcast(w + at(k, wleg)); // NOLINT(misc-const-correctness): set under if constexpr
    if constexpr (Sign > 0) {
      f = conj(f);
    }
    store(p + at(k, leg), cmul(v, f));
  }
};

// A grid phase's first or last pass, reading or writing the caller's (or the work array's)
// rows directly instead of through a gather or scatter: the lane pass of stockham.inl
// (lanes over copies; the DIF codelet with shared twiddles when Twiddled, the plain one
// when m = 1) with copy q of s at (q % cols) + qstride (q / cols) + stride idx on each side,
// idx = p + m j on loading and R p + k on storing (src_cols, src_qstride, src_stride and
// their dst counterparts; cols = 0 means s, the tight scratch). A block's columns are the
// q % cols, adjacent; q / cols counts the copies a phase's chain has made so far. s and
// cols are multiples of C, so every access is a whole register. InOp scales what is loaded
// (the input edge of phase 1). Out: 0 a plain store, 1 scaled (the output edge), 2 times
// row twiddles (twiddles, one plain value per destination row qq + (stride / qstride) k,
// advanced per column by the stage, stage::moving_twiddles). Reads src, dst, m, s, the
// strides and columns, twiddles, scale.
template <class T, std::size_t R, int Sign, bool Twiddled, edge_op InOp, int Out>
void phase_pass(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::size_t m = a->m;
  const std::size_t s = a->s;
  const std::size_t blocks = s / C<T>;
  const std::size_t xcols = a->src_cols == 0 ? s : a->src_cols;
  const std::size_t ycols = a->dst_cols == 0 ? s : a->dst_cols;
  const std::ptrdiff_t xq = a->src_qstride;
  const std::ptrdiff_t yq = a->dst_qstride;
  const std::ptrdiff_t xg = a->src_stride;
  const std::ptrdiff_t yg = a->dst_stride;
  const T* table = static_cast<const T*>(a->twiddles);
  const std::ptrdiff_t in_leg = 2 * xg * static_cast<std::ptrdiff_t>(m);
  const std::ptrdiff_t out_leg = 2 * yg;
  std::size_t p = first / blocks;
  std::size_t qc = first % blocks * C<T>;
  std::size_t xlo = qc % xcols;
  std::size_t xhi = qc / xcols;
  std::size_t ylo = qc % ycols;
  std::size_t yhi = qc / ycols;
  for (std::size_t u = first; u < last; ++u) {
    const T* xp =
        x + 2 * (static_cast<std::ptrdiff_t>(xlo) + xq * static_cast<std::ptrdiff_t>(xhi) +
                 xg * static_cast<std::ptrdiff_t>(p));
    T* yp = y + 2 * (static_cast<std::ptrdiff_t>(ylo) + yq * static_cast<std::ptrdiff_t>(yhi) +
                     yg * static_cast<std::ptrdiff_t>(R * p));
    const run_in<T, InOp, true> in{
        .p = xp,
        .leg = in_leg,
        .step = 2,
        .valid = C<T>,
        .op = make_edge<T, InOp>(a, nullptr, 0, 0),
    };
    if constexpr (Out == 2) {
      const row_twiddled_out<T, Sign> out{
          .p = yp,
          .leg = out_leg,
          .w = table + 2 * yhi,
          .wleg = 2 * (yg / yq),
      };
      cdft<R>::template plain<T, Sign>(in, out);
    } else {
      constexpr edge_op out_op = Out == 1 ? edge_op::scale : edge_op::none;
      const run_out<T, out_op, true> out{
          .p = yp,
          .leg = out_leg,
          .step = 2,
          .valid = C<T>,
          .op = make_edge<T, out_op>(a, nullptr, 0, 0),
      };
      if constexpr (Twiddled) {
        const shared_twiddles<T> twid{table + tw_reals<T> * (R - 1) * p};
        cdft<R>::template dif<T, Sign>(in, out, twid);
      } else {
        cdft<R>::template plain<T, Sign>(in, out);
      }
    }
    qc += C<T>;
    if (qc == s) {
      qc = 0;
      ++p;
      xlo = 0;
      xhi = 0;
      ylo = 0;
      yhi = 0;
    } else {
      xlo += C<T>;
      if (xlo == xcols) {
        xlo = 0;
        ++xhi;
      }
      ylo += C<T>;
      if (ylo == ycols) {
        ylo = 0;
        ++yhi;
      }
    }
  }
}

} // namespace
