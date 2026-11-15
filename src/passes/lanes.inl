// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The edge passes of the lanes strategy (DESIGN.md 8.5, strategies/chain.hpp, experiment
// X7): a block of B rows of a batch (B a multiple of C) is gathered into a tile, the chain of
// the row's length runs on the tile as lane passes whose copies are the rows, and the tile is
// scattered back. In the tile element i of row r is at i B + r, so B rows of n are exactly
// a line of B n with B interleaved copies (the layout of a Bluestein convolution's copies,
// stockham.inl): every pass of the chain then has its copy count multiplied by B, and each
// lane computes its own row's arithmetic, the same codelets and twiddles as the row alone.
// Kernel text in the manner of kernels.inl, included after access.inl, stockham.inl and
// real.inl. No include guard.
//
// The driver (core/driver.cpp) runs the passes of one block at a time and sets
// pass_args::rows to the rows the block holds: B, or fewer in the last block of a batch,
// whose missing rows the gather fills with zeros (no denormals or NaN in the idle lanes) and
// the scatter never writes. Rows are any distance apart and their elements any stride
// apart; complex values only (a real row is read and written as complex pairs).

namespace {

// C rows' element at p, the rows dist reals apart: the first `valid` of them, the other
// lanes zero.
template <class T>
FFTN_FORCE_INLINE cv<T> load_rows(const T* p, std::ptrdiff_t dist, std::size_t valid) {
  if constexpr (C<T> == 1) {
    if (valid == 1) {
      return load(p);
    }
  }
  return load_part(p, dist, valid);
}

template <class T>
FFTN_FORCE_INLINE void store_rows(T* p, std::ptrdiff_t dist, std::size_t valid, cv<T> v) {
  if constexpr (C<T> == 1) {
    store(p, v);
  } else {
    store_part(p, dist, valid, v);
  }
}

// The tile of the block's rows: element i of row r (r < rows) to i B + r, zeros for the rows
// from `rows` to B. Rows of contiguous elements go through C x C transposes in registers
// (C elements of each of C rows loaded, transposed, stored as C elements of C rows of the
// tile); adjacent rows (a distance of one complex value) are one load per element; any
// other layout goes element by element. One unit. Reads src, src_stride (reals between a
// row's elements), src_dist (reals between rows), n (elements per row), s (B), rows; dst:
// scratch.
template <class T> void lanes_gather(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t xs = a->src_stride;
  const std::ptrdiff_t xd = a->src_dist;
  const std::size_t n = a->n;
  const std::size_t b = a->s;
  const std::size_t rows = a->rows;
  const std::ptrdiff_t tile_leg = at(b, 2);
  for (std::size_t u = first; u < last; ++u) { // the one unit, when the range holds it
    for (std::size_t r0 = 0; r0 < b; r0 += C<T>) {
      const std::size_t valid = rows <= r0 ? 0 : (rows - r0 < C<T> ? rows - r0 : C<T>);
      const T* xr = x + at(r0, xd);
      T* yr = y + at(r0, 2);
      if (valid == C<T> && xs == 2) {
        std::size_t i = 0;
        for (; i + C<T> <= n; i += C<T>) {
          cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
          for (std::size_t l = 0; l < C<T>; ++l) {
            v[l] = load(xr + at(l, xd) + at(i, 2));
          }
          transpose(v);
          for (std::size_t l = 0; l < C<T>; ++l) {
            store(yr + at(i + l, tile_leg), v[l]);
          }
        }
        for (; i < n; ++i) {
          store(yr + at(i, tile_leg), load_rows(xr + at(i, 2), xd, C<T>));
        }
      } else if (valid == C<T> && xd == 2) {
        for (std::size_t i = 0; i < n; ++i) {
          store(yr + at(i, tile_leg), load(xr + at(i, xs)));
        }
      } else {
        for (std::size_t i = 0; i < n; ++i) {
          store(yr + at(i, tile_leg), load_rows(xr + at(i, xs), xd, valid));
        }
      }
    }
  }
}

// The inverse of lanes_gather for the first `rows` rows of the tile: element i B + r to
// element i of row r, dst_stride reals between a row's elements and dst_dist between rows.
// One unit. Reads src (scratch), dst, dst_stride, dst_dist, n, s (B), rows.
template <class T> void lanes_scatter(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::ptrdiff_t ys = a->dst_stride;
  const std::ptrdiff_t yd = a->dst_dist;
  const std::size_t n = a->n;
  const std::size_t b = a->s;
  const std::size_t rows = a->rows;
  const std::ptrdiff_t tile_leg = at(b, 2);
  for (std::size_t u = first; u < last; ++u) { // the one unit, when the range holds it
    for (std::size_t r0 = 0; r0 < rows; r0 += C<T>) {
      const std::size_t valid = rows - r0 < C<T> ? rows - r0 : C<T>;
      const T* xr = x + at(r0, 2);
      T* yr = y + at(r0, yd);
      if (valid == C<T> && ys == 2) {
        std::size_t i = 0;
        for (; i + C<T> <= n; i += C<T>) {
          cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
          for (std::size_t l = 0; l < C<T>; ++l) {
            v[l] = load(xr + at(i + l, tile_leg));
          }
          transpose(v);
          for (std::size_t l = 0; l < C<T>; ++l) {
            store(yr + at(l, yd) + at(i, 2), v[l]);
          }
        }
        for (; i < n; ++i) {
          store_rows(yr + at(i, 2), yd, C<T>, load(xr + at(i, tile_leg)));
        }
      } else if (valid == C<T> && yd == 2) {
        for (std::size_t i = 0; i < n; ++i) {
          store(yr + at(i, ys), load(xr + at(i, tile_leg)));
        }
      } else {
        for (std::size_t i = 0; i < n; ++i) {
          store_rows(yr + at(i, ys), yd, valid, load(xr + at(i, tile_leg)));
        }
      }
    }
  }
}

// The r2c pairing of every row of a tile (n = 2h even): X_0 .. X_h of each row (tile rows of
// h + 1) from its Z = DFT_h (tile rows of h), lanes over rows. The arithmetic of every other
// route's pairing (post_pair, post_middle, post_ends; real.inl), with the twiddle of bin k
// broadcast. twiddles: w^(k + n/4) for k = 0 .. h/2, broadcast storage (the leaves'
// table). One unit. Reads src, dst (scratch, not src), n, s (B), twiddles, scale.
template <class T, bool Scaled>
void lanes_r2c_pair(const pass_args* a, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(a->src);
  T* y = static_cast<T*>(a->dst);
  const std::size_t h = a->n / 2;
  const std::size_t b = a->s;
  const T* table = static_cast<const T*>(a->twiddles);
  const auto sc = static_cast<T>(a->scale);
  const T half = half_of<T, Scaled>(a);
  const std::ptrdiff_t leg = at(b, 2);
  for (std::size_t u = first; u < last; ++u) { // the one unit, when the range holds it
    for (std::size_t r0 = 0; r0 < b; r0 += C<T>) {
      const T* zr = z + at(r0, 2);
      T* yr = y + at(r0, 2);
      T z0[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      store(z0, load(zr));
      for (std::size_t l = 0; l < C<T>; ++l) {
        post_ends<T, Scaled>(z0[2 * l], z0[2 * l + 1], yr + at(l, 2), yr + at(h, leg) + at(l, 2),
                             sc);
      }
      for (std::size_t k = 1; k < h - k; ++k) {
        cv<T> lo;
        cv<T> hi;
        post_pair<T>(load(zr + at(k, leg)), conj(load(zr + at(h - k, leg))),
                     bcast_tw(table + tw_reals<T> * k), half, lo, hi);
        store(yr + at(k, leg), lo);
        store(yr + at(h - k, leg), hi);
      }
      if (h % 2 == 0) {
        store(yr + at(h / 2, leg), scaled<T, Scaled>(post_middle<T>(load(zr + at(h / 2, leg))), sc));
      }
    }
  }
}

// The c2r pairing of every row of a tile: Z_0 .. Z_h-1 of each row (tile rows of h) from
// X_0 .. X_h (tile rows of h + 1), as pre_pair, pre_middle and the leaves' Z_0. twiddles:
// as lanes_r2c_pair's. One unit. Reads src, dst (scratch, not src), n, s (B), twiddles.
template <class T> void lanes_c2r_pair(const pass_args* a, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(a->src);
  T* z = static_cast<T*>(a->dst);
  const std::size_t h = a->n / 2;
  const std::size_t b = a->s;
  const T* table = static_cast<const T*>(a->twiddles);
  const std::ptrdiff_t leg = at(b, 2);
  for (std::size_t u = first; u < last; ++u) { // the one unit, when the range holds it
    for (std::size_t r0 = 0; r0 < b; r0 += C<T>) {
      const T* xr = x + at(r0, 2);
      T* zr = z + at(r0, 2);
      T x0[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      T xh[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      store(x0, load(xr));
      store(xh, load(xr + at(h, leg)));
      for (std::size_t l = 0; l < C<T>; ++l) {
        const T re = x0[2 * l];
        const T im = xh[2 * l];
        x0[2 * l] = re + im;
        x0[2 * l + 1] = re - im;
      }
      store(zr, load(x0));
      for (std::size_t j = 1; j < h - j; ++j) {
        cv<T> lo;
        cv<T> hi;
        pre_pair<T>(load(xr + at(j, leg)), conj(load(xr + at(h - j, leg))),
                    bcast_tw(table + tw_reals<T> * j), lo, hi);
        store(zr + at(j, leg), lo);
        store(zr + at(h - j, leg), hi);
      }
      if (h % 2 == 0) {
        store(zr + at(h / 2, leg), pre_middle<T>(load(xr + at(h / 2, leg))));
      }
    }
  }
}

} // namespace
