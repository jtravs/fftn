// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The vector vocabulary (DESIGN.md 6.2): the names every target defines, under the same
// spelling, in its own namespace. Kernel text uses only these names and plain scalar C++,
// so one text compiles in every region. No include guard: each vocab_<target>.inl includes
// this first, inside its region.
//
// Types and constants, for T = double or float:
//
//   cv<T>        complex register: C<T> complex numbers, interleaved (re, im, re, im, ...)
//   rv<T>        real register: W<T> reals (W = 2C on SIMD targets; 1 on scalar)
//   tw<T>        C<T> twiddles in the form mul, mulc and madd consume
//   tw_reals<T>  reals stored per twiddle: 2 (re, im), or 4 on x86 and plain NEON, where each
//                twiddle w is pre-arranged as (wr, wr) and (-wi, wi) so mul needs one
//                shuffle and no sign fix-up
//   fused        whether fmadd and its relatives round once (FMA). Where false they multiply,
//                then add: two roundings, and the operation order the generator emits for
//                such targets (DESIGN.md 6.1). The scalar target is never fused, so it gives
//                the same bits on every processor.
//   masked       whether the target defines load_n and store_n (below): a partial run of
//                contiguous complex numbers as one masked vector access (x86; experiment
//                X23). Elsewhere passes move partial runs through a tile (access.inl).
//   id           the target (target.hpp)
//   split_block  the copy count, a multiple of W, from which a chain's lane passes may run
//                in split arithmetic (strategies/chain.hpp): the largest W among the targets
//                that round alike (DESIGN.md 6.2), so that both targets of a pair choose the
//                same passes and give the same bits (avx2 takes avx512's W)
//
// Complex-register operations, a, b, acc of type cv<T>, w of type tw<T>, r of type T,
// p a pointer to T:
//
//   add(a, b), sub(a, b)    a + b, a - b
//   addj(a, b), subj(a, b)  a + i*b, a - i*b
//   mul(a, w), mulc(a, w)   a * w, a * conj(w)
//   madd(acc, a, w)         acc + a * w
//   scale(a, r)             r * a
//   fmadd_r(acc, a, r)      acc + r * a   (fused where `fused`)
//   neg(a), conj(a)         -a, conj(a)
//   swap_ri(a)              (im, re) of each complex number
//   load(p), store(p, a)    C<T> complex numbers at p (2C reals, any alignment)
//   bcast(p)                the complex number at p in every lane
//   load_n(p, n)            where `masked`, for n <= C<T>: the first n complex numbers at p,
//                           the other lanes zero. Touches no memory beyond them: the
//                           masked-off elements are not accessed and cannot fault, so a run
//                           may end at the end of the caller's array
//   store_n(p, n, a)        where `masked`, for n <= C<T>: the first n lanes of a to p,
//                           nothing else written
//   load_tw(p)              C<T> twiddles from storage written by pack_tw
//   pack_tw(out, p)         writes the C<T> interleaved twiddles at p as storage at out
//                           (C * tw_reals reals); scalar code, run when a table is built
//   bcast_tw(p)             one twiddle, from storage written by pack_tw1, in every lane:
//                           the twiddle of a lane pass, whose lanes share it (DESIGN.md 8.4)
//   pack_tw1(out, p)        writes the one interleaved twiddle at p as broadcast storage at
//                           out (tw_reals reals); scalar code, run when a table is built
//   cmul(a, b)              a * b for two complex registers (a factor array's values, not
//                           twiddle storage); rounds exactly as mul(a, w) with w = b
//   transpose(v)            v points to C<T> registers holding a C x C block of complex
//                           numbers; transposes it in place (lane j of v[i] <-> lane i of
//                           v[j]). No R x C form: passes store radices C does not divide
//                           through a tile (DESIGN.md 6.2).
//
// Exact operations (add, sub, addj, subj, scale, neg, conj, swap_ri, the memory operations
// and transpose) give the same bits on every target. mul, mulc, madd and cmul differ
// between targets in their last bit, by the order of their roundings.
//
// Real-register operations, in namespace real (a, b, c of type rv<T>, x of type T):
//
//   add, sub, mul           elementwise
//   fmadd(a, b, c)          a * b + c
//   fmsub(a, b, c)          a * b - c
//   fnmadd(a, b, c)         c - a * b      (the x86 names; all fused where `fused`)
//   neg(a)
//   load(p), store(p, a)    W<T> reals at p
//   bcast(x)                x in every lane
//   deinterleave(p, re, im) W<T> complex numbers at p (2W reals, interleaved) into the
//                           real parts re and the imaginary parts im, in order
//   interleave(p, re, im)   the inverse: W<T> complex numbers written to p
//
// The real family lives in its own namespace because cv<T> and rv<T> are one machine type
// on SIMD targets, and mul and bcast mean different things in the two families.
//
// Everything in a region has internal linkage: each .inl wraps its text in an unnamed
// namespace, which is the same one for all of them within a target's namespace. Each unit
// therefore keeps its own copy of every function, and the linker can never substitute one
// unit's copy, compiled for one target, for another's (the hazard of the header-only
// study's experiment E06). Only a unit's function table is external.

namespace {

template <class T> struct vec;

template <class T> using cv = vec<T>::cv;
template <class T> using rv = vec<T>::rv;
template <class T> using tw = vec<T>::tw;
template <class T> inline constexpr std::size_t C = vec<T>::C;
template <class T> inline constexpr std::size_t W = vec<T>::W;
template <class T> inline constexpr std::size_t tw_reals = vec<T>::tw_reals;
template <class T> inline constexpr std::size_t split_block = vec<T>::split_block;

} // namespace
