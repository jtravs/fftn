// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Probe text: written once, included inside each target's region after its vocabulary,
// under the same rules as the library's kernel text (src/isa/kernels.inl). No include
// guard. Semantics of each operation are in probe_table.hpp.

namespace {

// Whether the vocabulary has load_n and store_n: the x86 ones do, whether or not their
// passes use them (`masked`), so the probes test them in every build.
template <class T>
constexpr bool partial_runs = requires(const T* p, T* q, cv<T> v) {
  load_n(p, std::size_t{1});
  store_n(q, std::size_t{1}, v);
};

template <class T>
void probe(vop op, const T* a, const T* b, const T* c, T* out, std::size_t n, const T* r) {
  const std::size_t step = 2 * C<T>; // reals per complex register
  switch (op) {
  case vop::c_copy:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, load(a + k));
    }
    break;
  case vop::c_add:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, add(load(a + k), load(b + k)));
    }
    break;
  case vop::c_sub:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, sub(load(a + k), load(b + k)));
    }
    break;
  case vop::c_addj:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, addj(load(a + k), load(b + k)));
    }
    break;
  case vop::c_subj:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, subj(load(a + k), load(b + k)));
    }
    break;
  case vop::c_mul:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, mul(load(a + k), load_tw(b + tw_reals<T> * (k / 2))));
    }
    break;
  case vop::c_mulc:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, mulc(load(a + k), load_tw(b + tw_reals<T> * (k / 2))));
    }
    break;
  case vop::c_madd:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, madd(load(c + k), load(a + k), load_tw(b + tw_reals<T> * (k / 2))));
    }
    break;
  case vop::c_scale:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, scale(load(a + k), *r));
    }
    break;
  case vop::c_fmadd_r:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, fmadd_r(load(c + k), load(a + k), *r));
    }
    break;
  case vop::c_neg:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, neg(load(a + k)));
    }
    break;
  case vop::c_conj:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, conj(load(a + k)));
    }
    break;
  case vop::c_swap_ri:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, swap_ri(load(a + k)));
    }
    break;
  case vop::c_bcast:
    for (std::size_t g = 0; g * step < n; ++g) {
      store(out + g * step, bcast(a + 2 * g));
    }
    break;
  case vop::c_transpose:
    for (std::size_t k = 0; k < n; k += step * C<T>) {
      cv<T> v[C<T>]; // NOLINT(modernize-avoid-c-arrays): std::array is not allowed in a region
      for (std::size_t i = 0; i < C<T>; ++i) {
        v[i] = load(a + k + i * step);
      }
      transpose(v);
      for (std::size_t i = 0; i < C<T>; ++i) {
        store(out + k + i * step, v[i]);
      }
    }
    break;
  case vop::c_cmul:
    for (std::size_t k = 0; k < n; k += step) {
      store(out + k, cmul(load(a + k), load(b + k)));
    }
    break;
  case vop::c_mul_bcast:
    for (std::size_t g = 0; g * step < n; ++g) {
      store(out + g * step, mul(load(a + g * step), bcast_tw(b + tw_reals<T> * g)));
    }
    break;
  case vop::c_load_n:
    if constexpr (partial_runs<T>) {
      for (std::size_t k = 0; k < n; k += step) {
        store(out + k, load_n(a + k, (k / step) % (C<T> + 1)));
      }
    }
    break;
  case vop::c_store_n:
    if constexpr (partial_runs<T>) {
      for (std::size_t k = 0; k < n; k += step) {
        store_n(out + k, (k / step) % (C<T> + 1), load(a + k));
      }
    }
    break;
  case vop::r_copy:
    for (std::size_t k = 0; k < n; k += W<T>) {
      real::store(out + k, real::load(a + k));
    }
    break;
  case vop::r_add:
    for (std::size_t k = 0; k < n; k += W<T>) {
      real::store(out + k, real::add(real::load(a + k), real::load(b + k)));
    }
    break;
  case vop::r_sub:
    for (std::size_t k = 0; k < n; k += W<T>) {
      real::store(out + k, real::sub(real::load(a + k), real::load(b + k)));
    }
    break;
  case vop::r_mul:
    for (std::size_t k = 0; k < n; k += W<T>) {
      real::store(out + k, real::mul(real::load(a + k), real::load(b + k)));
    }
    break;
  case vop::r_fmadd:
    for (std::size_t k = 0; k < n; k += W<T>) {
      real::store(out + k, real::fmadd(real::load(a + k), real::load(b + k), real::load(c + k)));
    }
    break;
  case vop::r_fmsub:
    for (std::size_t k = 0; k < n; k += W<T>) {
      real::store(out + k, real::fmsub(real::load(a + k), real::load(b + k), real::load(c + k)));
    }
    break;
  case vop::r_fnmadd:
    for (std::size_t k = 0; k < n; k += W<T>) {
      real::store(out + k, real::fnmadd(real::load(a + k), real::load(b + k), real::load(c + k)));
    }
    break;
  case vop::r_neg:
    for (std::size_t k = 0; k < n; k += W<T>) {
      real::store(out + k, real::neg(real::load(a + k)));
    }
    break;
  case vop::r_bcast:
    for (std::size_t g = 0; g * W<T> < n; ++g) {
      real::store(out + g * W<T>, real::bcast(a[g]));
    }
    break;
  case vop::r_deinterleave:
    for (std::size_t k = 0; k < n / 2; k += W<T>) {
      rv<T> re;
      rv<T> im;
      real::deinterleave(a + 2 * k, re, im);
      real::store(out + k, re);
      real::store(out + n / 2 + k, im);
    }
    break;
  case vop::r_interleave:
    for (std::size_t k = 0; k < n / 2; k += W<T>) {
      real::interleave(out + 2 * k, real::load(a + k), real::load(b + k));
    }
    break;
  }
}

} // namespace
