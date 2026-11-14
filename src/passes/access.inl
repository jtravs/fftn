// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Accessors: how a pass hands values to a codelet and takes them back (gen/README.md,
// "Calling convention"), with the edge operations of DESIGN.md 12.1 applied on the way.
// Pass kinds, edge variants and tails differ only in their accessors; the codelet text
// never changes. Kernel text in the manner of kernels.inl: included inside a region after
// its vocabulary, no standard-library templates, everything force-inlined, no include
// guard. The accessors hold pointers and integers, never vectors.
//
// Two shapes of access:
//
// - A run: C consecutive elements of a line, one per lane (lane and grouped passes). Full
//   runs are one vector load or store, so their elements must be adjacent in memory (stride
//   1, or C == 1); partial runs (the tails, where fewer than C elements remain) go through
//   a small tile, element by element, at any stride. The lanes a partial run leaves empty
//   hold zero and are never stored. Every lane performs the same operations either way,
//   so an element's result does not depend on whether its run was full.
// - One element in every lane (leaves, copies, the edges of real transforms): loaded by
//   bcast, stored from lane 0, at any stride.

namespace {

FFTN_FORCE_INLINE std::ptrdiff_t at(std::size_t index, std::ptrdiff_t stride) {
  return static_cast<std::ptrdiff_t>(index) * stride;
}

// The first `valid` of C elements at p, step reals apart, as a register whose other lanes
// hold zero.
template <class T>
FFTN_FORCE_INLINE cv<T> load_part(const T* p, std::ptrdiff_t step, std::size_t valid) {
  T tile[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  for (std::size_t l = 0; l < C<T>; ++l) {
    tile[2 * l] = T(0);
    tile[2 * l + 1] = T(0);
    if (l < valid) {
      tile[2 * l] = p[at(l, step)];
      tile[2 * l + 1] = p[at(l, step) + 1];
    }
  }
  return load(tile);
}

// The first `valid` lanes of v to p, step reals apart.
template <class T>
FFTN_FORCE_INLINE void store_part(T* p, std::ptrdiff_t step, std::size_t valid, cv<T> v) {
  T tile[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
  store(tile, v);
  for (std::size_t l = 0; l < valid; ++l) {
    p[at(l, step)] = tile[2 * l];
    p[at(l, step) + 1] = tile[2 * l + 1];
  }
}

template <class T, bool Full>
FFTN_FORCE_INLINE cv<T> load_run(const T* p, std::ptrdiff_t step, std::size_t valid) {
  if constexpr (Full) {
    return load(p);
  } else {
    return load_part(p, step, valid);
  }
}

template <class T, bool Full>
FFTN_FORCE_INLINE void store_run(T* p, std::ptrdiff_t step, std::size_t valid, cv<T> v) {
  if constexpr (Full) {
    store(p, v);
  } else {
    store_part(p, step, valid, v);
  }
}

// Lane 0 of v to p.
template <class T> FFTN_FORCE_INLINE void store_first(T* p, cv<T> v) {
  if constexpr (C<T> == 1) {
    store(p, v);
  } else {
    T tile[2 * C<T>]; // NOLINT(modernize-avoid-c-arrays): no standard containers in a region
    store(tile, v);
    p[0] = tile[0];
    p[1] = tile[1];
  }
}

// An edge operation's data for one access: the scale, or the factor values of leg 0's
// first element and the reals between legs (the factor array is tight, so lanes are
// adjacent in it).
template <class T, edge_op Op> struct edge {
  T s = T(1);
  const T* f = nullptr;
  std::ptrdiff_t f_leg = 0;
};

// The edge for an access whose leg j starts at element e0 + leg j of the line.
template <class T, edge_op Op>
FFTN_FORCE_INLINE edge<T, Op> make_edge(const pass_args* a, const void* factor, std::size_t e0,
                                        std::size_t leg) {
  edge<T, Op> e;
  if constexpr (Op == edge_op::scale) {
    e.s = static_cast<T>(a->scale);
  } else if constexpr (Op == edge_op::factor) {
    e.f = static_cast<const T*>(factor) + at(e0, 2);
    e.f_leg = at(leg, 2);
  }
  return e;
}

// Applies the operation to leg j of a run (v holds its first `valid` lanes when not Full).
template <class T, edge_op Op, bool Full>
FFTN_FORCE_INLINE cv<T> on_run(const edge<T, Op>& e, cv<T> v, std::size_t j, std::size_t valid) {
  if constexpr (Op == edge_op::scale) {
    return scale(v, e.s);
  } else if constexpr (Op == edge_op::factor) {
    return cmul(v, load_run<T, Full>(e.f + at(j, e.f_leg), 2, valid));
  } else {
    return v;
  }
}

// Applies the operation to element j held in every lane.
template <class T, edge_op Op>
FFTN_FORCE_INLINE cv<T> on_one(const edge<T, Op>& e, cv<T> v, std::size_t j) {
  if constexpr (Op == edge_op::scale) {
    return scale(v, e.s);
  } else if constexpr (Op == edge_op::factor) {
    return cmul(v, bcast(e.f + at(j, e.f_leg)));
  } else {
    return v;
  }
}

// Leg j of a run: the elements at p + leg j, step reals apart.
template <class T, edge_op Op, bool Full> struct run_in {
  const T* p;
  std::ptrdiff_t leg;
  std::ptrdiff_t step;
  std::size_t valid;
  edge<T, Op> op;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const {
    return on_run<T, Op, Full>(op, load_run<T, Full>(p + at(j, leg), step, valid), j, valid);
  }
};

template <class T, edge_op Op, bool Full> struct run_out {
  T* p;
  std::ptrdiff_t leg;
  std::ptrdiff_t step;
  std::size_t valid;
  edge<T, Op> op;
  FFTN_FORCE_INLINE void put(std::size_t k, cv<T> v) const {
    store_run<T, Full>(p + at(k, leg), step, valid, on_run<T, Op, Full>(op, v, k, valid));
  }
};

// Element j of a line at p, leg reals apart, in every lane.
template <class T, edge_op Op> struct one_in {
  const T* p;
  std::ptrdiff_t leg;
  edge<T, Op> op;
  [[nodiscard]] FFTN_FORCE_INLINE cv<T> get(std::size_t j) const {
    return on_one<T, Op>(op, bcast(p + at(j, leg)), j);
  }
};

template <class T, edge_op Op> struct one_out {
  T* p;
  std::ptrdiff_t leg;
  edge<T, Op> op;
  FFTN_FORCE_INLINE void put(std::size_t k, cv<T> v) const {
    store_first(p + at(k, leg), on_one<T, Op>(op, v, k));
  }
};

// Codelet outputs kept in registers, for passes that rearrange them before storing.
template <class T> struct to_registers {
  cv<T>* v;
  FFTN_FORCE_INLINE void put(std::size_t k, cv<T> x) const { v[k] = x; }
};

// Twiddles of a grouped pass: C per-lane twiddles per leg, rows of one table (load_tw).
template <class T> struct lane_twiddles {
  const T* p;
  std::ptrdiff_t row;
  [[nodiscard]] FFTN_FORCE_INLINE tw<T> get(std::size_t k) const {
    return load_tw(p + at(k - 1, row));
  }
};

// Twiddles of a lane pass: one per leg, shared by every lane (bcast_tw, DESIGN.md 8.4).
template <class T> struct shared_twiddles {
  const T* p;
  [[nodiscard]] FFTN_FORCE_INLINE tw<T> get(std::size_t k) const {
    return bcast_tw(p + tw_reals<T> * (k - 1));
  }
};

// a * w (forward) or a * conj(w) (backward), w a forward twiddle.
template <class T, int Sign> FFTN_FORCE_INLINE cv<T> twiddle(cv<T> a, tw<T> w) {
  if constexpr (Sign < 0) {
    return mul(a, w);
  } else {
    return mulc(a, w);
  }
}

} // namespace
