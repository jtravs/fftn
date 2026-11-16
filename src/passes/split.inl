// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Split lane passes (DESIGN.md 6.3, 8.2): the lane passes of a chain on the split codelets
// (src/gen/split_codelets.inl), which work on separate real and imaginary registers of W
// reals and so need no shuffle in their arithmetic. Kernel text in the manner of
// stockham.inl (included inside a region after its vocabulary, the codelets and
// access.inl; no include guard).
//
// Between split passes a line is in the BL layout: blocks of W reals followed by their W
// imaginary parts, element e's real part at 2 W (e / W) + e % W and its imaginary part W
// reals later. A run of W elements that starts at a multiple of W is one block: real parts
// at 2 e, imaginary parts at 2 e + W. The passes need W | s, so every run is one block and
// no pass has a tail. A chain's split passes are its last ones (strategies/chain.hpp): the
// first of them reads the interleaved line (IL) and deinterleaves it on loading, the last
// writes the interleaved line, interleaving on storing, with the output edge. So the
// layout never shows outside a chain, and the conversion costs one shuffle per register at
// each end (DESIGN.md 6.3).
//
// Direction: split codelets are forward only. A backward pass exchanges the real and
// imaginary parts on loading and on storing (gen/fftn_gen/split.py), with the same forward
// twiddles, so the two directions round alike (mirror images).

namespace {

static_assert(split_block<double> % W<double> == 0 && split_block<float> % W<float> == 0,
              "a split run's copies are whole blocks of W");

// The parts of leg j of a BL run at p + leg j, exchanged when Swap.
template <class T, bool Swap> struct bl_in {
  const T* p;
  std::ptrdiff_t leg;
  [[nodiscard]] FFTN_FORCE_INLINE rv<T> re(std::size_t j) const {
    return real::load(p + at(j, leg) + (Swap ? static_cast<std::ptrdiff_t>(W<T>) : 0));
  }
  [[nodiscard]] FFTN_FORCE_INLINE rv<T> im(std::size_t j) const {
    return real::load(p + at(j, leg) + (Swap ? 0 : static_cast<std::ptrdiff_t>(W<T>)));
  }
};

// The parts of leg j of an interleaved run of W elements at p + leg j (stride 1, or any
// stride on a target with W = 1), exchanged when Swap. re and im deinterleave the same two
// loads; the compiler keeps one copy of the loads.
template <class T, bool Swap> struct il_in {
  const T* p;
  std::ptrdiff_t leg;
  [[nodiscard]] FFTN_FORCE_INLINE rv<T> re(std::size_t j) const {
    rv<T> r;
    rv<T> i;
    real::deinterleave(p + at(j, leg), r, i);
    if constexpr (Swap) {
      return i;
    } else {
      return r;
    }
  }
  [[nodiscard]] FFTN_FORCE_INLINE rv<T> im(std::size_t j) const {
    rv<T> r;
    rv<T> i;
    real::deinterleave(p + at(j, leg), r, i);
    if constexpr (Swap) {
      return r;
    } else {
      return i;
    }
  }
};

template <class T, bool Swap> struct bl_out {
  T* p;
  std::ptrdiff_t leg;
  FFTN_FORCE_INLINE void put(std::size_t k, rv<T> a, rv<T> b) const {
    T* q = p + at(k, leg);
    if constexpr (Swap) {
      real::store(q, b);
      real::store(q + W<T>, a);
    } else {
      real::store(q, a);
      real::store(q + W<T>, b);
    }
  }
};

// Leg k to an interleaved run at p + leg k, with the output edge's operation: the scale, or
// times the factor array's values for the run's elements (tight, interleaved, deinterleaved
// here), rounded in the order of the x86 and plain NEON cmul (vocab_avx2.inl): the real
// part rounds im fi and fuses re fr onto it, the imaginary part rounds re fi and fuses
// im fr onto it.
template <class T, bool Swap, edge_op Op> struct il_out {
  T* p;
  std::ptrdiff_t leg;
  edge<T, Op> op;
  FFTN_FORCE_INLINE void put(std::size_t k, rv<T> a, rv<T> b) const {
    rv<T> re = a; // NOLINT(misc-const-correctness): set under if constexpr
    rv<T> im = b; // NOLINT(misc-const-correctness): set under if constexpr
    if constexpr (Swap) {
      re = b;
      im = a;
    }
    if constexpr (Op == edge_op::scale) {
      const rv<T> s = real::bcast(op.s);
      re = real::mul(re, s);
      im = real::mul(im, s);
    } else if constexpr (Op == edge_op::factor) {
      rv<T> fr;
      rv<T> fi;
      real::deinterleave(op.f + at(k, op.f_leg), fr, fi);
      const rv<T> r = real::fmsub(re, fr, real::mul(im, fi));
      im = real::fmadd(im, fr, real::mul(re, fi));
      re = r;
    }
    real::interleave(p + at(k, leg), re, im);
  }
};

// The twiddles of a split lane pass's group: R - 1 forward twiddles, tight and interleaved
// (wr, wi), each part broadcast to every lane.
template <class T> struct split_twiddles {
  const T* p;
  [[nodiscard]] FFTN_FORCE_INLINE rv<T> re(std::size_t k) const {
    return real::bcast(p[2 * (k - 1)]);
  }
  [[nodiscard]] FFTN_FORCE_INLINE rv<T> im(std::size_t k) const {
    return real::bcast(p[2 * (k - 1) + 1]);
  }
};

template <class T, std::size_t R, bool Twiddled, class In, class Out>
FFTN_FORCE_INLINE void split_unit(const In& in, const Out& out, const T* table, std::size_t p) {
  if constexpr (Twiddled) {
    const split_twiddles<T> twid{table + 2 * (R - 1) * p};
    sdft<R>::template dif<T>(in, out, twid);
  } else {
    sdft<R>::template plain<T>(in, out);
  }
}

// A split lane pass of a chain (the geometry of stockham.inl's lane_pass): Twiddled, m > 1
// (DIF codelet), writing BL; or the last pass, m = 1 (plain codelet), writing the
// interleaved line with the output edge. InIL: the source is the interleaved line (the
// first split pass) rather than BL. Unit u is group p = u / blocks and copies
// q = (u % blocks) W .. + W - 1, blocks = s / W (W divides s). twiddles: R - 1 per group,
// tight interleaved forward roots (chain_builder::split_twiddles). src: scratch, stride 1;
// dst: scratch, or for the last pass any stride when W == 1, else 1. Reads src, dst_stride,
// m, s, twiddles, scale, out_factor.
template <class T, std::size_t R, int Sign, bool Twiddled, bool InIL, edge_op OutOp>
void split_pass(const pass_args* a, const void* src, void* dst, std::size_t first,
                std::size_t last) {
  constexpr bool swap = Sign > 0;
  const T* x = static_cast<const T*>(src);
  T* y = static_cast<T*>(dst);
  const std::ptrdiff_t ys = Twiddled ? 2 : 2 * a->dst_stride;
  const std::size_t m = a->m;
  const std::size_t s = a->s;
  const std::size_t blocks = s / W<T>;
  const T* table = static_cast<const T*>(a->twiddles);
  const std::ptrdiff_t in_leg = at(s * m, 2);
  const std::ptrdiff_t out_leg = at(s, ys);
  // No division when the range starts in the first group (stockham.inl, lane_pass).
  std::size_t p = first < blocks ? 0 : first / blocks;
  std::size_t b = first < blocks ? first : first % blocks;
  for (std::size_t u = first; u < last; ++u) {
    const std::size_t q0 = b * W<T>;
    const T* xp = x + at(q0 + s * p, 2);
    const std::size_t e0 = q0 + s * R * p;
    T* yp = y + at(e0, ys);
    if constexpr (Twiddled) {
      const bl_out<T, swap> out{.p = yp, .leg = out_leg};
      if constexpr (InIL) {
        split_unit<T, R, true>(il_in<T, swap>{.p = xp, .leg = in_leg}, out, table, p);
      } else {
        split_unit<T, R, true>(bl_in<T, swap>{.p = xp, .leg = in_leg}, out, table, p);
      }
    } else {
      static_assert(!InIL, "the last split pass reads BL: a split run has two passes or more");
      const il_out<T, swap, OutOp> out{
          .p = yp,
          .leg = out_leg,
          .op = make_edge<T, OutOp>(a, a->out_factor, e0, s),
      };
      split_unit<T, R, false>(bl_in<T, swap>{.p = xp, .leg = in_leg}, out, table, p);
    }
    if (++b == blocks) {
      b = 0;
      ++p;
    }
  }
}

} // namespace
