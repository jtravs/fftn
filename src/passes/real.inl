// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The passes of real transforms (DESIGN.md 9.1). Kernel text in the manner of kernels.inl,
// included after access.inl and stockham.inl. No include guard.
//
// Even n = 2h runs through a complex transform of h: the reals x[2j], x[2j+1] of a line
// of stride 1 are read in place as the complex z_j, and bins k and h - k are paired
// around the transform. With w = exp(-2 pi i / n) and v = -i w^k (w^(k + n/4) when 4
// divides n; the tables hold (Im w^k, -Re w^k), an exact rearrangement of a root-table
// entry, for every even n, so the pairing needs no factor i of its own):
//
// - r2c: Z = DFT_h(z); with A = Z_k + conj Z_h-k and B = Z_k - conj Z_h-k,
//   X_k = (A + v B) / 2 and X_h-k = conj(A - v B) / 2; X_0, X_h = Re Z_0 +- Im Z_0.
// - c2r: with A = X_k + conj X_h-k and B = X_k - conj X_h-k, Z_k = A + conj(v) B and
//   Z_h-k = conj(A - conj(v) B) (Z_0 from the real parts of X_0 and X_h, whose imaginary
//   parts are ignored); then z = IDFT_h(Z) holds the reals in pairs.
//
// That is an addition, a subtraction and a complex multiplication per pair, plus the
// halving (which takes the plan's scale with it: x (s/2) is exactly (x / 2) s).
//
// The pairing is fused into the pass that touches the spectrum where that is a codelet
// pass: the last pass of the r2c chain (r2c_last) and the first of the c2r chain
// (c2r_first) take the butterflies of copy or group q and of its mirror together, because
// their legs hold paired bins; the leaves of h <= 16 do it in registers. Elsewhere (a
// Bluestein or generic edge pass, the grid) it is a separate pass in place on the h + 1
// values of the complex side (r2c_post, c2r_pre). Lanes run over consecutive bins, copies
// or groups; the mirror's C lanes are C consecutive elements in reverse order, read (and
// for r2c written) through a tile.
//
// A line with a stride other than 1 on either side (out of place only) packs the reals into
// scratch and pairs the bins in a separate pass at any stride (r2c_pack_pairs,
// r2c_post_strided, c2r_pre_strided, c2r_unpack_pairs). Odd n runs the full complex
// transform of (x_j, 0), or of the expanded Hermitian spectrum (r2c_pack, r2c_store,
// c2r_expand, c2r_store: one element at a time, any stride).
//
// The output edges take the plan's scale (Scaled): every value they store is multiplied
// by it, exactly as if the transform were followed by a scaling pass.

namespace {

template <class T, bool Scaled> FFTN_FORCE_INLINE T scaled(T v, T s) {
  if constexpr (Scaled) {
    return v * s;
  } else {
    return v;
  }
}

template <class T, bool Scaled> FFTN_FORCE_INLINE cv<T> scaled(cv<T> v, T s) {
  if constexpr (Scaled) {
    return scale(v, s);
  } else {
    return v;
  }
}

// Lane l holds the complex number at p - 2 l (the first `valid` lanes; the others zero):
// C consecutive elements read downwards.
template <class T, bool Full> FFTN_FORCE_INLINE cv<T> load_down(const T* p, std::size_t valid) {
  if constexpr (C<T> == 1) {
    return load(p);
  } else {
    T tile[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    for (std::size_t l = 0; l < C<T>; ++l) {
      const bool live = Full || l < valid;
      tile[2 * l] = live ? p[-at(l, 2)] : T(0);
      tile[2 * l + 1] = live ? p[1 - at(l, 2)] : T(0);
    }
    return load(tile);
  }
}

// Lane l of v to p - 2 l, for the first `valid` lanes.
template <class T, bool Full> FFTN_FORCE_INLINE void store_down(T* p, std::size_t valid, cv<T> v) {
  if constexpr (C<T> == 1) {
    store(p, v);
  } else {
    T tile[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    store(tile, v);
    for (std::size_t l = 0; l < (Full ? C<T> : valid); ++l) {
      p[-at(l, 2)] = tile[2 * l];
      p[1 - at(l, 2)] = tile[2 * l + 1];
    }
  }
}

// The complex number (re, im) in every lane.
template <class T> FFTN_FORCE_INLINE cv<T> splat(T re, T im) {
  T tile[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  for (std::size_t l = 0; l < C<T>; ++l) {
    tile[2 * l] = re;
    tile[2 * l + 1] = im;
  }
  return load(tile);
}

// The r2c pairing of C bins: zk = Z_k, zc = conj Z_h-k, v = w^(k + n/4); half is 1/2
// times the scale. lo = X_k, hi = X_h-k.
template <class T>
FFTN_FORCE_INLINE void post_pair(cv<T> zk, cv<T> zc, tw<T> v, T half, cv<T>& lo, cv<T>& hi) {
  const cv<T> a = add(zk, zc);
  const cv<T> p = mul(sub(zk, zc), v);
  lo = scale(add(a, p), half);
  hi = scale(conj(sub(a, p)), half);
}

// The c2r pairing: xk = X_k, xc = conj X_h-k, v = w^(k + n/4). lo = Z_k, hi = Z_h-k.
template <class T>
FFTN_FORCE_INLINE void pre_pair(cv<T> xk, cv<T> xc, tw<T> v, cv<T>& lo, cv<T>& hi) {
  const cv<T> a = add(xk, xc);
  const cv<T> p = mulc(sub(xk, xc), v);
  lo = add(a, p);
  hi = conj(sub(a, p));
}

// The paired bin h/2 = conj Z_h/2 (r2c; z holds Z_h/2) or 2 conj X_h/2 (c2r).
template <class T> FFTN_FORCE_INLINE cv<T> post_middle(cv<T> z) { return conj(z); }
template <class T> FFTN_FORCE_INLINE cv<T> pre_middle(cv<T> x) {
  const cv<T> c = conj(x);
  return add(c, c);
}

// X_0 and X_h from Z_0 = (re, im): Re Z_0 +- Im Z_0, real.
template <class T, bool Scaled> FFTN_FORCE_INLINE void post_ends(T re, T im, T* y0, T* yh, T s) {
  y0[0] = scaled<T, Scaled>(re + im, s);
  y0[1] = scaled<T, Scaled>(T(0), s);
  yh[0] = scaled<T, Scaled>(re - im, s);
  yh[1] = scaled<T, Scaled>(T(0), s);
}

// 1/2 times the scale of an output edge (exact: halving does not round).
template <class T, bool Scaled> FFTN_FORCE_INLINE T half_of(const pass_args* a) {
  return Scaled ? static_cast<T>(0.5 * a->scale) : T(0.5);
}

// Codelet inputs from registers.
template <class T> struct from_registers {
  const cv<T>* v;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const { return v[j]; }
};

// ---- Separate passes, in place on the complex side (n = 2h even).

// The bins k0 .. k0 + valid - 1 and their partners h - k of a separate pairing pass, from
// x to y (r2c: Z to X, with half; c2r: X to Z, with the scale when Scaled).
template <class T, bool R2c, bool Full, bool Scaled>
FFTN_FORCE_INLINE void pairing_unit(const T* x, T* y, std::size_t h, std::size_t k0,
                                    std::size_t valid, const T* table, T factor) {
  const tw<T> v = load_tw(table + tw_reals<T> * (k0 - 1));
  const cv<T> xk = load_run<T, Full>(x + at(k0, 2), 2, valid);
  const cv<T> xc = conj(load_down<T, Full>(x + at(h - k0, 2), valid));
  cv<T> lo;
  cv<T> hi;
  if constexpr (R2c) {
    post_pair<T>(xk, xc, v, factor, lo, hi);
  } else {
    pre_pair<T>(xk, xc, v, lo, hi);
    lo = scaled<T, Scaled>(lo, factor);
    hi = scaled<T, Scaled>(hi, factor);
  }
  store_run<T, Full>(y + at(k0, 2), 2, valid, lo);
  store_down<T, Full>(y + at(h - k0, 2), valid, hi);
}

// X_0 .. X_h from Z = DFT_h(z) in the first h places of the same line. Unit 0 is bins 0, h
// and (h even) h/2; unit u >= 1 is bins k = 1 + (u - 1) C .. + C - 1 up to K = (h - 1) /
// 2, each with its partner h - k. twiddles: w^(k + n/4) for k = 1 .. K, padded with ones to
// a multiple of C, per-lane storage. src may be dst (a unit reads and writes the same
// bins). Units 1 + ceil(K / C). Reads src, dst, n, twiddles, scale.
template <class T, bool Scaled>
void r2c_post(const pass_args* a, const void* src, void* dst, std::size_t first, std::size_t last) {
  const T* z = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::size_t h = a->n / 2;
  const std::size_t pairs = (h - 1) / 2;
  const T* table = static_cast<const T*>(a->twiddles);
  const auto s = static_cast<T>(a->scale);
  const T half = half_of<T, Scaled>(a);
  std::size_t u = first;
  if (u == 0 && u < last) {
    // Bins 0 and h are read before they are written; h/2 is neither.
    post_ends<T, Scaled>(z[0], z[1], y, y + at(h, 2), s);
    if (h % 2 == 0) {
      store_first(y + at(h / 2, 2), scaled<T, Scaled>(post_middle<T>(bcast(z + at(h / 2, 2))), s));
    }
    ++u;
  }
  // The full units, then the last if it is partial: no test of the lane count in the loop.
  for (const std::size_t full = 1 + pairs / C<T>; u < last && u < full; ++u) {
    pairing_unit<T, true, true, Scaled>(z, y, h, 1 + (u - 1) * C<T>, C<T>, table, half);
  }
  for (; u < last; ++u) {
    const std::size_t k0 = 1 + (u - 1) * C<T>;
    pairing_unit<T, true, false, Scaled>(z, y, h, k0, pairs - k0 + 1, table, half);
  }
}

// Z_0 .. Z_h-1 from X_0 .. X_h in the same line, the mirror of r2c_post: unit 0 is Z_0 and
// (h even) Z_h/2; unit u >= 1 the bins k = 1 + (u - 1) C .. up to K with their partners.
// twiddles: as r2c_post's. src may be dst. The scale (Scaled) is for h = 1, where no
// transform follows. Reads src, dst, n, twiddles, scale.
template <class T, bool Scaled>
void c2r_pre(const pass_args* a, const void* src, void* dst, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* z = static_cast<T*>(dst);
  const std::size_t h = a->n / 2;
  const std::size_t pairs = (h - 1) / 2;
  const T* table = static_cast<const T*>(a->twiddles);
  const auto s = static_cast<T>(a->scale);
  std::size_t u = first;
  if (u == 0 && u < last) {
    const T x0 = x[0];
    const T xh = x[at(h, 2)];
    if (h % 2 == 0) {
      store_first(z + at(h / 2, 2), scaled<T, Scaled>(pre_middle<T>(bcast(x + at(h / 2, 2))), s));
    }
    z[0] = scaled<T, Scaled>(x0 + xh, s);
    z[1] = scaled<T, Scaled>(x0 - xh, s);
    ++u;
  }
  for (const std::size_t full = 1 + pairs / C<T>; u < last && u < full; ++u) {
    pairing_unit<T, false, true, Scaled>(x, z, h, 1 + (u - 1) * C<T>, C<T>, table, s);
  }
  for (; u < last; ++u) {
    const std::size_t k0 = 1 + (u - 1) * C<T>;
    pairing_unit<T, false, false, Scaled>(x, z, h, k0, pairs - k0 + 1, table, s);
  }
}

// ---- Fused passes (n = 2h even).

// Leg j of a run read downwards: lane l at p + leg j - 2 l, the first `valid` lanes.
template <class T, bool Full> struct run_down {
  const T* p;
  std::ptrdiff_t leg;
  std::size_t valid;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const {
    return load_down<T, Full>(p + at(j, leg), valid);
  }
};

// One unit of r2c_last's copies q0 .. q0 + valid - 1 (lanes up) and s - q0 .. (lanes down):
// the plain codelet on both, then leg k of the first with leg R - 1 - k of the second.
// tq: the unit's column of the per-lane pairing table, rows `row` reals apart.
template <class T, std::size_t R, bool Full>
FFTN_FORCE_INLINE void r2c_copies(const T* x, T* y, std::size_t h, std::size_t s, std::size_t q0,
                                  std::size_t valid, const T* tq, std::ptrdiff_t row, T half) {
  cv<T> va[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  cv<T> vb[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  const std::ptrdiff_t leg = at(s, 2);
  const run_in<T, edge_op::none, Full> ina{
      .p = x + at(q0, 2),
      .leg = leg,
      .step = 2,
      .valid = valid,
      .op = {},
  };
  cdft<R>::template plain<T, -1>(ina, to_registers<T>{va});
  const run_down<T, Full> inb{.p = x + at(s - q0, 2), .leg = leg, .valid = valid};
  cdft<R>::template plain<T, -1>(inb, to_registers<T>{vb});
  T* y_lo = y + at(q0, 2);
  T* y_hi = y + at(h - q0, 2);
  for (std::size_t k = 0; k < R; ++k) {
    cv<T> lo;
    cv<T> hi;
    post_pair<T>(va[k], conj(vb[R - 1 - k]), load_tw(tq + at(k, row)), half, lo, hi);
    store_run<T, Full>(y_lo + at(s * k, 2), 2, valid, lo);
    store_down<T, Full>(y_hi - at(s * k, 2), valid, hi);
  }
}

// The last pass of an r2c chain of h = R s with the pairing fused (m = 1, the plain
// codelet, lanes over copies). Bin q + s k (copy q, leg k) pairs with bin (s - q) + s (R -
// 1 - k), so copies q and s - q run together. Unit 0 is copy 0 and (s even) copy s/2, one
// element in every lane: copy 0's legs pair k with R - k (k = 0 gives X_0 and X_h, k = R/2
// the bin h/2), copy s/2's pair k with R - 1 - k. Unit u >= 1 is copies q = 1 + (u - 1) C
// .. up to qm = (s - 1) / 2 with copies s - q, read downwards. twiddles: per-lane storage,
// R rows of qpad = qm rounded up to C entries, row k entry i holding w^(1 + i + s k + n/4)
// (padding ones); roots: broadcast storage, w^(s k + n/4) then w^(s/2 + s k + n/4) for k <
// R. src: scratch, stride 1; dst: the h + 1 bins, stride 1. Units 1 + ceil(qm / C). Reads
// src, dst, s, twiddles, roots, scale.
template <class T, std::size_t R, bool Scaled>
void r2c_last(const pass_args* a, const void* src, void* dst, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::size_t s = a->s;
  const std::size_t h = R * s;
  const std::size_t qm = (s - 1) / 2;
  const std::size_t qpad = (qm + C<T> - 1) / C<T> * C<T>;
  const T* table = static_cast<const T*>(a->twiddles);
  const T* ends = static_cast<const T*>(a->roots);
  const std::ptrdiff_t row = at(qpad, static_cast<std::ptrdiff_t>(tw_reals<T>));
  const std::ptrdiff_t leg = at(s, 2);
  const auto sc = static_cast<T>(a->scale);
  const T half = half_of<T, Scaled>(a);
  std::size_t u = first;
  if (u == 0 && u < last) {
    cv<T> va[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    cv<T> vb[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    {
      const one_in<T, edge_op::none> in0{.p = x, .leg = leg, .op = {}};
      cdft<R>::template plain<T, -1>(in0, to_registers<T>{va});
      T z0[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      store(z0, va[0]);
      post_ends<T, Scaled>(z0[0], z0[1], y, y + at(h, 2), sc);
      for (std::size_t k = 1; k < R - k; ++k) {
        cv<T> lo;
        cv<T> hi;
        post_pair<T>(va[k], conj(va[R - k]), bcast_tw(ends + tw_reals<T> * k), half, lo, hi);
        store_first(y + at(s * k, 2), lo);
        store_first(y + at(s * (R - k), 2), hi);
      }
      if constexpr (R % 2 == 0) {
        store_first(y + at(h / 2, 2), scaled<T, Scaled>(post_middle<T>(va[R / 2]), sc));
      }
      if (s % 2 == 0) {
        const one_in<T, edge_op::none> inh{.p = x + at(s / 2, 2), .leg = leg, .op = {}};
        cdft<R>::template plain<T, -1>(inh, to_registers<T>{vb});
        for (std::size_t k = 0; k < R - 1 - k; ++k) {
          cv<T> lo;
          cv<T> hi;
          post_pair<T>(vb[k], conj(vb[R - 1 - k]), bcast_tw(ends + tw_reals<T> * (R + k)), half, lo,
                       hi);
          store_first(y + at(s / 2 + s * k, 2), lo);
          store_first(y + at(s / 2 + s * (R - 1 - k), 2), hi);
        }
        if constexpr (R % 2 == 1) {
          store_first(y + at(h / 2, 2), scaled<T, Scaled>(post_middle<T>(vb[(R - 1) / 2]), sc));
        }
      }
    }
    ++u;
  }
  for (const std::size_t full = 1 + qm / C<T>; u < last && u < full; ++u) {
    const std::size_t q0 = 1 + (u - 1) * C<T>;
    r2c_copies<T, R, true>(x, y, h, s, q0, C<T>, table + tw_reals<T> * (q0 - 1), row, half);
  }
  for (; u < last; ++u) {
    const std::size_t q0 = 1 + (u - 1) * C<T>;
    r2c_copies<T, R, false>(x, y, h, s, q0, qm - q0 + 1, table + tw_reals<T> * (q0 - 1), row, half);
  }
}

// The outputs of a grouped pass's C butterflies whose lane l is butterfly p_top - l (the
// mirror groups of c2r_first, in reverse order), v[k] holding leg k: y + 2 R p_top is
// butterfly p_top's first output. As store_transposed and store_tiled, with lane l
// written R elements below lane l - 1.
template <class T, std::size_t R, bool Full>
FFTN_FORCE_INLINE void store_groups_down(T* y_top, cv<T>* v, std::size_t valid) {
  if constexpr (Full && R % C<T> == 0) {
    for (std::size_t b = 0; b < R / C<T>; ++b) {
      transpose(v + b * C<T>);
      for (std::size_t l = 0; l < C<T>; ++l) {
        store(y_top - at(R * l, 2) + at(b * C<T>, 2), v[b * C<T> + l]);
      }
    }
  } else {
    T tile[R][2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    for (std::size_t k = 0; k < R; ++k) {
      store(tile[k], v[k]);
    }
    for (std::size_t l = 0; l < valid; ++l) {
      for (std::size_t k = 0; k < R; ++k) {
        y_top[at(k, 2) - at(R * l, 2)] = tile[k][2 * l];
        y_top[at(k, 2) - at(R * l, 2) + 1] = tile[k][2 * l + 1];
      }
    }
  }
}

// One unit of c2r_first's groups p0 .. p0 + valid - 1 (lanes up) and m - p0 .. m - p0 -
// valid + 1 (lanes down): leg j of group p and leg R - 1 - j of group m - p hold the paired
// bins e = p + m j and h - e.
template <class T, std::size_t R, bool Full>
FFTN_FORCE_INLINE void c2r_groups(const T* x, T* y, std::size_t h, std::size_t m, std::size_t p0,
                                  std::size_t valid, const T* tw_up, const T* tw_down, const T* pre,
                                  std::ptrdiff_t row) {
  cv<T> va[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  cv<T> vb[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  for (std::size_t j = 0; j < R; ++j) {
    const std::size_t e = p0 + m * j;
    pre_pair<T>(load_run<T, Full>(x + at(e, 2), 2, valid),
                conj(load_down<T, Full>(x + at(h - e, 2), valid)), load_tw(pre + at(j, row)), va[j],
                vb[R - 1 - j]);
  }
  cdft<R>::template dif<T, +1>(from_registers<T>{va}, to_registers<T>{va},
                               lane_twiddles<T>{.p = tw_up, .row = row});
  cdft<R>::template dif<T, +1>(from_registers<T>{vb}, to_registers<T>{vb},
                               lane_twiddles<T>{.p = tw_down, .row = row});
  if constexpr (Full && R % C<T> == 0) {
    store_transposed<T, R>(y + at(R * p0, 2), va, C<T>);
  } else {
    store_tiled<T, R>(y + at(R * p0, 2), va, valid);
  }
  store_groups_down<T, R, Full>(y + at(R * (m - p0), 2), vb, valid);
}

// The first pass of a c2r chain of h = R m (s = 1, lanes over butterfly groups, the
// backward DIF codelet) with the pairing fused into its loads: leg j of group p and leg
// R - 1 - j of group m - p hold paired bins, so the two groups run together and each pair is
// formed once. Unit 0 is group 0 (legs j and R - j; leg 0 from X_0 and X_h, leg R/2 the bin
// h/2; its twiddles are ones, so the plain codelet) and (m even) group m/2 (legs j and
// R - 1 - j), one element in every lane; unit u >= 1 is groups p = 1 + (u - 1) C .. up to
// pm = (m - 1) / 2 with groups m - p. twiddles: per-lane storage of three tables of rows of
// ppad = pm rounded up to C entries: R - 1 rows k of w_h^((1 + i) k) (the groups p), R - 1
// rows of w_h^((m - 1 - i) k) (the groups m - p), R rows j of w^(1 + i + m j + n/4) (the
// pairing); roots: broadcast storage, w^(m j + n/4) and w^(m/2 + m j + n/4) for j < R, then
// w_h^((m/2) k) for k = 1 .. R - 1. src: the h + 1 bins, stride 1; dst: scratch. Units
// 1 + ceil(pm / C). Reads src, dst, m, twiddles, roots.
template <class T, std::size_t R>
void c2r_first(const pass_args* a, const void* src, void* dst, std::size_t first,
               std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::size_t m = a->m;
  const std::size_t h = R * m;
  const std::size_t pm = (m - 1) / 2;
  const std::size_t ppad = (pm + C<T> - 1) / C<T> * C<T>;
  const T* table = static_cast<const T*>(a->twiddles);
  const T* ends = static_cast<const T*>(a->roots);
  const std::ptrdiff_t row = at(ppad, static_cast<std::ptrdiff_t>(tw_reals<T>));
  const T* tw_up = table;
  const T* tw_down = table + at(R - 1, row);
  const T* pre = table + at(2 * (R - 1), row);
  std::size_t u = first;
  if (u == 0 && u < last) {
    {
      cv<T> v[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
      v[0] = splat<T>(x[0] + x[at(h, 2)], x[0] - x[at(h, 2)]);
      for (std::size_t j = 1; j < R - j; ++j) {
        pre_pair<T>(bcast(x + at(m * j, 2)), conj(bcast(x + at(m * (R - j), 2))),
                    bcast_tw(ends + tw_reals<T> * j), v[j], v[R - j]);
      }
      if constexpr (R % 2 == 0) {
        v[R / 2] = pre_middle<T>(bcast(x + at(h / 2, 2)));
      }
      cdft<R>::template plain<T, +1>(from_registers<T>{v}, to_registers<T>{v});
      for (std::size_t k = 0; k < R; ++k) {
        store_first(y + at(k, 2), v[k]);
      }
      if (m % 2 == 0) {
        for (std::size_t j = 0; j < R - 1 - j; ++j) {
          pre_pair<T>(bcast(x + at(m / 2 + m * j, 2)),
                      conj(bcast(x + at(m / 2 + m * (R - 1 - j), 2))),
                      bcast_tw(ends + tw_reals<T> * (R + j)), v[j], v[R - 1 - j]);
        }
        if constexpr (R % 2 == 1) {
          v[(R - 1) / 2] = pre_middle<T>(bcast(x + at(h / 2, 2)));
        }
        cdft<R>::template dif<T, +1>(from_registers<T>{v}, to_registers<T>{v},
                                     shared_twiddles<T>{ends + tw_reals<T> * 2 * R});
        for (std::size_t k = 0; k < R; ++k) {
          store_first(y + at(R * (m / 2) + k, 2), v[k]);
        }
      }
    }
    ++u;
  }
  for (const std::size_t full = 1 + pm / C<T>; u < last && u < full; ++u) {
    const std::size_t p0 = 1 + (u - 1) * C<T>;
    const std::ptrdiff_t off = at(p0 - 1, static_cast<std::ptrdiff_t>(tw_reals<T>));
    c2r_groups<T, R, true>(x, y, h, m, p0, C<T>, tw_up + off, tw_down + off, pre + off, row);
  }
  for (; u < last; ++u) {
    const std::size_t p0 = 1 + (u - 1) * C<T>;
    const std::ptrdiff_t off = at(p0 - 1, static_cast<std::ptrdiff_t>(tw_reals<T>));
    c2r_groups<T, R, false>(x, y, h, m, p0, pm - p0 + 1, tw_up + off, tw_down + off, pre + off,
                            row);
  }
}

// An r2c of n = 2R, R <= 16 or a prime codelet radix, in registers: the codelet on the R pairs,
// then the pairing. One unit. twiddles: w^(k + n/4) for k = 0 .. R/2, broadcast storage. Every load
// precedes every store, so src may be dst. Reads src, dst, twiddles, scale.
template <class T, std::size_t R, bool Scaled>
void r2c_leaf(const pass_args* a, const void* src, void* dst, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const T* table = static_cast<const T*>(a->twiddles);
  const auto sc = static_cast<T>(a->scale);
  const T half = half_of<T, Scaled>(a);
  for (std::size_t u = first; u < last; ++u) { // the one unit, when the range holds it
    cv<T> v[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    const one_in<T, edge_op::none> in{.p = x, .leg = 2, .op = {}};
    cdft<R>::template plain<T, -1>(in, to_registers<T>{v});
    T z0[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    store(z0, v[0]);
    post_ends<T, Scaled>(z0[0], z0[1], y, y + at(R, 2), sc);
    for (std::size_t k = 1; k < R - k; ++k) {
      cv<T> lo;
      cv<T> hi;
      post_pair<T>(v[k], conj(v[R - k]), bcast_tw(table + tw_reals<T> * k), half, lo, hi);
      store_first(y + at(k, 2), lo);
      store_first(y + at(R - k, 2), hi);
    }
    if constexpr (R % 2 == 0) {
      store_first(y + at(R / 2, 2), scaled<T, Scaled>(post_middle<T>(v[R / 2]), sc));
    }
  }
}

// A c2r of n = 2R, R <= 16 or a prime codelet radix, in registers: the pairing, then the backward
// codelet, storing the reals in pairs with the output edge's operation. One unit. twiddles: w^(j +
// n/4) for j = 0 .. R/2, broadcast storage. src may be dst. Reads src, dst, twiddles, scale.
template <class T, std::size_t R, edge_op OutOp>
void c2r_leaf(const pass_args* a, const void* src, void* dst, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(src);
  const T* table = static_cast<const T*>(a->twiddles);
  const one_out<T, OutOp> out{
      .p = static_cast<T*>(dst),
      .leg = 2,
      .op = make_edge<T, OutOp>(a, nullptr, 0, 1),
  };
  for (std::size_t u = first; u < last; ++u) { // the one unit, when the range holds it
    cv<T> v[R]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    v[0] = splat<T>(x[0] + x[at(R, 2)], x[0] - x[at(R, 2)]);
    for (std::size_t j = 1; j < R - j; ++j) {
      pre_pair<T>(bcast(x + at(j, 2)), conj(bcast(x + at(R - j, 2))),
                  bcast_tw(table + tw_reals<T> * j), v[j], v[R - j]);
    }
    if constexpr (R % 2 == 0) {
      v[R / 2] = pre_middle<T>(bcast(x + at(R / 2, 2)));
    }
    cdft<R>::template plain<T, +1>(from_registers<T>{v}, out);
  }
}

// ---- Strided lines (n = 2h even, a stride other than 1 on either side): the reals are
// packed into scratch, the chain of h runs there and the pairing is a separate pass, one
// element at a time (bcast on loading, lane 0 stored), at any stride.

// z_j = (x[2j], x[2j+1]) for unit j < h. src is the real line, dst complex scratch.
template <class T>
void r2c_pack_pairs(const pass_args* a, const void* src, void* dst, std::size_t first,
                    std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* z = static_cast<T*>(dst);
  const std::ptrdiff_t xs = a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  for (std::size_t j = first; j < last; ++j) {
    z[at(j, zs)] = x[at(2 * j, xs)];
    z[at(j, zs) + 1] = x[at(2 * j + 1, xs)];
  }
}

// X_k for k = 0..h from Z = DFT_h(z), unit u for the pair k = u, h - u (unit 0: bins 0
// and h). With E = (Z_k + conj Z_h-k) / 2, O = (Z_k - conj Z_h-k) / 2i and w = exp(-2 pi i
// / n):
//
//   X_k = E + w^k O,   X_h-k = conj(E - w^k O).
//
// twiddles: w^k for k = 0 .. h/2, broadcast storage. Reads src, n, twiddles, scale. Units
// h/2 + 1.
template <class T, bool Scaled>
void r2c_post_strided(const pass_args* a, const void* src, void* dst, std::size_t first,
                      std::size_t last) {
  const T* z = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const std::size_t h = a->n / 2;
  const T* table = static_cast<const T*>(a->twiddles);
  const auto s = static_cast<T>(a->scale);
  for (std::size_t k = first; k < last; ++k) {
    if (k == 0) {
      // Bins 0 and h are real: Re Z_0 +- Im Z_0.
      const T re = z[0];
      const T im = z[1];
      y[0] = scaled<T, Scaled>(re + im, s);
      y[1] = scaled<T, Scaled>(T(0), s);
      y[at(h, ys)] = scaled<T, Scaled>(re - im, s);
      y[at(h, ys) + 1] = scaled<T, Scaled>(T(0), s);
      continue;
    }
    const cv<T> zk = bcast(z + at(k, zs));
    const cv<T> zc = conj(bcast(z + at(h - k, zs)));
    const cv<T> e = scale(add(zk, zc), T(0.5));
    // w^k O = -i (w^k (Z_k - conj Z_h-k) / 2); multiplying by -i is conj(swap_ri(.)).
    const cv<T> wo =
        conj(swap_ri(scale(mul(sub(zk, zc), bcast_tw(table + tw_reals<T> * k)), T(0.5))));
    store_first(y + at(k, ys), scaled<T, Scaled>(add(e, wo), s));
    if (k != h - k) {
      store_first(y + at(h - k, ys), scaled<T, Scaled>(conj(sub(e, wo)), s));
    }
  }
}

// Z_k for k < h from X_0 .. X_h, unit u for the pair k = u, h - u (unit 0: Z_0 alone).
// With E = X_k + conj X_h-k and O = (X_k - conj X_h-k) exp(+2 pi i k / n):
//
//   Z_k = E + i O,   Z_h-k = conj(E - i O).
//
// The imaginary parts of X_0 and X_h are ignored. twiddles: the forward w^k = exp(-2 pi i
// k / n) for k = 0 .. h/2, broadcast storage, used conjugated. Reads src, n, twiddles.
// Units h/2 + 1.
template <class T>
void c2r_pre_strided(const pass_args* a, const void* src, void* dst, std::size_t first,
                     std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* z = static_cast<T*>(dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  const std::size_t h = a->n / 2;
  const T* table = static_cast<const T*>(a->twiddles);
  for (std::size_t k = first; k < last; ++k) {
    if (k == 0) {
      const T x0 = x[0];
      const T xh = x[at(h, xs)];
      z[0] = x0 + xh;
      z[1] = x0 - xh;
      continue;
    }
    const cv<T> xk = bcast(x + at(k, xs));
    const cv<T> xc = conj(bcast(x + at(h - k, xs)));
    const cv<T> e = add(xk, xc);
    const cv<T> o = mulc(sub(xk, xc), bcast_tw(table + tw_reals<T> * k));
    store_first(z + at(k, zs), addj(e, o));
    if (k != h - k) {
      store_first(z + at(h - k, zs), conj(subj(e, o)));
    }
  }
}

// x[2j] = Re z_j, x[2j+1] = Im z_j, unit j < h. src complex scratch, dst the real line.
// Reads src, scale.
template <class T, bool Scaled>
void c2r_unpack_pairs(const pass_args* a, const void* src, void* dst, std::size_t first,
                      std::size_t last) {
  const T* z = static_cast<const T*>(src);
  T* x = static_cast<T*>(dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t xs = a->dst_stride;
  const auto s = static_cast<T>(a->scale);
  for (std::size_t j = first; j < last; ++j) {
    x[at(2 * j, xs)] = scaled<T, Scaled>(z[at(j, zs)], s);
    x[at(2 * j + 1, xs)] = scaled<T, Scaled>(z[at(j, zs) + 1], s);
  }
}

// ---- Odd n: the full complex transform of (x_j, 0).

// z_j = (x[j], 0), unit j < n.
template <class T>
void r2c_pack(const pass_args* a, const void* src, void* dst, std::size_t first, std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* z = static_cast<T*>(dst);
  const std::ptrdiff_t xs = a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  for (std::size_t j = first; j < last; ++j) {
    z[at(j, zs)] = x[at(j, xs)];
    z[at(j, zs) + 1] = T(0);
  }
}

// X_k = Z_k for unit k <= n/2; the imaginary part of X_0 is exactly 0 (a sum of reals).
// Reads src, scale.
template <class T, bool Scaled>
void r2c_store(const pass_args* a, const void* src, void* dst, std::size_t first,
               std::size_t last) {
  const T* z = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t ys = 2 * a->dst_stride;
  const auto s = static_cast<T>(a->scale);
  for (std::size_t k = first; k < last; ++k) {
    y[at(k, ys)] = scaled<T, Scaled>(z[at(k, zs)], s);
    y[at(k, ys) + 1] = scaled<T, Scaled>(k == 0 ? T(0) : z[at(k, zs) + 1], s);
  }
}

// ---- Odd n, complex to real: the full Hermitian spectrum, then its complex transform.

// Unit 0: Z_0 = Re X_0 (its imaginary part is ignored). Unit k in 1 .. (n-1)/2: Z_k = X_k
// and Z_n-k = conj X_k. Reads src, n. Units n/2 + 1.
template <class T>
void c2r_expand(const pass_args* a, const void* src, void* dst, std::size_t first,
                std::size_t last) {
  const T* x = static_cast<const T*>(src);
  T* z = static_cast<T*>(dst);
  const std::ptrdiff_t xs = 2 * a->src_stride;
  const std::ptrdiff_t zs = 2 * a->dst_stride;
  const std::size_t n = a->n;
  for (std::size_t k = first; k < last; ++k) {
    if (k == 0) {
      z[0] = x[0];
      z[1] = T(0);
      continue;
    }
    const T re = x[at(k, xs)];
    const T im = x[at(k, xs) + 1];
    z[at(k, zs)] = re;
    z[at(k, zs) + 1] = im;
    z[at(n - k, zs)] = re;
    z[at(n - k, zs) + 1] = -im;
  }
}

// x[j] = Re z_j, unit j < n. Reads src, scale.
template <class T, bool Scaled>
void c2r_store(const pass_args* a, const void* src, void* dst, std::size_t first,
               std::size_t last) {
  const T* z = static_cast<const T*>(src);
  T* x = static_cast<T*>(dst);
  const std::ptrdiff_t zs = 2 * a->src_stride;
  const std::ptrdiff_t xs = a->dst_stride;
  const auto s = static_cast<T>(a->scale);
  for (std::size_t j = first; j < last; ++j) {
    x[at(j, xs)] = scaled<T, Scaled>(z[at(j, zs)], s);
  }
}

} // namespace
