// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "strategies/chain.hpp"

#include "core/program.hpp"
#include "core/twiddles.hpp"
#include "isa/kernel_table.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

std::vector<std::size_t> chain_radices(std::size_t n) {
  std::vector<std::size_t> radices;
  while (n % 4 == 0) {
    radices.push_back(4);
    n /= 4;
  }
  if (n % 2 == 0) {
    radices.push_back(2);
    n /= 2;
  }
  for (const std::size_t p : {std::size_t{3}, std::size_t{5}}) {
    while (n % p == 0) {
      radices.push_back(p);
      n /= p;
    }
  }
  for (std::size_t p = 7; p <= n / p; p += 2) {
    while (n % p == 0) {
      radices.push_back(p);
      n /= p;
    }
  }
  if (n > 1) {
    radices.push_back(n);
  }
  return radices;
}

template <class T>
chain_builder<T>::chain_builder(const kernels_of<T>& kernels, std::vector<std::vector<T>>& tables,
                                std::size_t chunk_points)
    : kernels_(kernels), tables_(tables), chunk_points_(std::max<std::size_t>(chunk_points, 1)) {}

template <class T> const std::vector<std::complex<T>>& chain_builder<T>::roots(std::size_t n) {
  auto it = roots_.find(n);
  if (it == roots_.end()) {
    it = roots_.emplace(n, unit_roots<T>(-1, n)).first;
  }
  return it->second;
}

// Converts interleaved twiddles to the target's storage and keeps them.
template <class T> const T* chain_builder<T>::pack(const std::vector<std::complex<T>>& w) {
  std::vector<T> plain(2 * w.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    plain[2 * i] = w[i].real();
    plain[2 * i + 1] = w[i].imag();
  }
  std::vector<T>& stored = tables_.emplace_back(w.size() * kernels_.tw_reals);
  kernels_.pack_twiddles(plain.data(), w.size(), stored.data());
  return stored.data();
}

// w_n^(p k s) for p < m, k = 1 .. radix-1, grouped by p: the twiddles of the pass with s
// copies. p k s < n, so every entry is an entry of the root table.
template <class T>
const T* chain_builder<T>::stage_twiddles(std::size_t n, std::size_t radix, std::size_t m,
                                          std::size_t s) {
  const auto key = std::make_pair(n, s);
  if (const auto it = stage_tables_.find(key); it != stage_tables_.end()) {
    return it->second;
  }
  const auto& w = roots(n);
  std::vector<std::complex<T>> t;
  t.reserve((radix - 1) * m);
  for (std::size_t p = 0; p < m; ++p) {
    for (std::size_t k = 1; k < radix; ++k) {
      t.push_back(w[p * k * s]);
    }
  }
  const T* stored = pack(t);
  stage_tables_.emplace(key, stored);
  return stored;
}

// The radix-th roots exp(-2 pi i t / radix) = w_n^(t n / radix), for the generic pass.
template <class T> const T* chain_builder<T>::radix_roots(std::size_t n, std::size_t radix) {
  const auto key = std::make_pair(n, radix);
  if (const auto it = radix_roots_.find(key); it != radix_roots_.end()) {
    return it->second;
  }
  const auto& w = roots(n);
  std::vector<std::complex<T>> t;
  t.reserve(radix);
  for (std::size_t i = 0; i < radix; ++i) {
    t.push_back(w[i * (n / radix)]);
  }
  const T* stored = pack(t);
  radix_roots_.emplace(key, stored);
  return stored;
}

// w_n^k for k = 0 .. n/4: the twiddles of the real post- and pre-passes (n even).
template <class T> const T* chain_builder<T>::edge_twiddles(std::size_t n) {
  if (const auto it = edge_tables_.find(n); it != edge_tables_.end()) {
    return it->second;
  }
  const auto& w = roots(n);
  const std::vector<std::complex<T>> t(w.begin(),
                                       w.begin() + static_cast<std::ptrdiff_t>(n / 4 + 1));
  const T* stored = pack(t);
  edge_tables_.emplace(n, stored);
  return stored;
}

template <class T>
std::uint32_t chain_builder<T>::chunks_of(std::size_t units, std::size_t points_per_unit) const {
  const std::size_t per_chunk = std::max<std::size_t>(1, chunk_points_ / points_per_unit);
  const std::size_t chunks = std::max<std::size_t>(1, (units + per_chunk - 1) / per_chunk);
  return static_cast<std::uint32_t>(
      std::min<std::size_t>(chunks, std::numeric_limits<std::uint32_t>::max()));
}

namespace {

const char* direction_name(int sign) { return sign < 0 ? "forward" : "backward"; }

} // namespace

template <class T>
void chain_builder<T>::add_pass(std::size_t n, std::size_t radix, std::size_t s, int sign,
                                endpoint src, endpoint dst, std::vector<stage>& stages) {
  const std::size_t m = n / (radix * s);
  stage st;
  const signed_pass* fixed = nullptr;
  switch (radix) {
  case 2:
    fixed = &kernels_.chain.radix2;
    break;
  case 3:
    fixed = &kernels_.chain.radix3;
    break;
  case 4:
    fixed = &kernels_.chain.radix4;
    break;
  case 5:
    fixed = &kernels_.chain.radix5;
    break;
  default:
    break;
  }
  const signed_pass& pass = fixed != nullptr ? *fixed : kernels_.chain.generic;
  st.run = sign < 0 ? pass.forward : pass.backward;
  if (fixed == nullptr) {
    st.args.roots = radix_roots(n, radix);
  }
  st.args.twiddles = stage_twiddles(n, radix, m, s);
  st.args.radix = radix;
  st.args.m = m;
  st.args.s = s;
  st.args.n = n;
  st.args.src_stride = src.stride;
  st.args.dst_stride = dst.stride;
  st.src = src.where;
  st.dst = dst.where;
  st.units = m * s;
  // The generic pass costs radix^2 per butterfly against radix for the others.
  st.chunks = chunks_of(st.units, fixed != nullptr ? radix : radix * radix);
  st.name = (fixed != nullptr ? "radix-" : "generic radix-") + std::to_string(radix) + " " +
            direction_name(sign) + " (m " + std::to_string(m) + ", s " + std::to_string(s) +
            (fixed != nullptr ? ")" : "; O(radix^2) per butterfly)");
  stages.push_back(std::move(st));
}

template <class T>
void chain_builder<T>::add_edge(pass_fn run, const char* name, std::size_t n, std::size_t units,
                                endpoint src, endpoint dst, const T* twiddles,
                                std::vector<stage>& stages) {
  stage st;
  st.run = run;
  st.args.twiddles = twiddles;
  st.args.n = n;
  st.args.src_stride = src.stride;
  st.args.dst_stride = dst.stride;
  st.src = src.where;
  st.dst = dst.where;
  st.units = units;
  st.chunks = chunks_of(units, 2);
  st.name = name;
  stages.push_back(std::move(st));
}

template <class T>
slot chain_builder<T>::scratch_chain(std::size_t n, int sign, slot from,
                                     std::vector<stage>& stages) {
  std::size_t s = 1;
  slot at = from;
  for (const std::size_t radix : chain_radices(n)) {
    const slot next = at == slot::a ? slot::b : slot::a;
    add_pass(n, radix, s, sign, {at, 1}, {next, 1}, stages);
    at = next;
    s *= radix;
  }
  return at;
}

template <class T>
std::size_t chain_builder<T>::c2c_line(std::size_t n, int sign, endpoint src, endpoint dst,
                                       bool aliased, std::vector<stage>& stages) {
  const std::vector<std::size_t> radices = chain_radices(n);
  if (radices.empty()) {
    add_edge(kernels_.chain.copy, "copy", n, n, src, dst, nullptr, stages);
    return 0;
  }
  if (radices.size() == 1 && aliased) {
    add_pass(n, radices[0], 1, sign, src, {slot::a, 1}, stages);
    add_edge(kernels_.chain.copy, "copy", n, n, {slot::a, 1}, dst, nullptr, stages);
    return n;
  }
  std::size_t s = 1;
  endpoint at = src;
  for (std::size_t i = 0; i < radices.size(); ++i) {
    const bool last = i + 1 == radices.size();
    const endpoint next = last ? dst : endpoint{at.where == slot::a ? slot::b : slot::a, 1};
    add_pass(n, radices[i], s, sign, at, next, stages);
    at = next;
    s *= radices[i];
  }
  return radices.size() > 1 ? n : 0;
}

template <class T>
std::size_t chain_builder<T>::r2c_line(std::size_t n, endpoint src, endpoint dst,
                                       std::vector<stage>& stages) {
  const auto& k = kernels_.chain;
  if (n % 2 == 0) {
    const std::size_t h = n / 2;
    add_edge(k.r2c_pack_pairs, "r2c pack pairs", n, h, src, {slot::a, 1}, nullptr, stages);
    const slot z = scratch_chain(h, -1, slot::a, stages);
    add_edge(k.r2c_post, "r2c post", n, h / 2 + 1, {z, 1}, dst, edge_twiddles(n), stages);
    return h;
  }
  add_edge(k.r2c_pack, "r2c pack", n, n, src, {slot::a, 1}, nullptr, stages);
  const slot z = scratch_chain(n, -1, slot::a, stages);
  add_edge(k.r2c_store, "r2c store half", n, n / 2 + 1, {z, 1}, dst, nullptr, stages);
  return n;
}

template <class T>
std::size_t chain_builder<T>::c2r_line(std::size_t n, endpoint src, endpoint dst,
                                       std::vector<stage>& stages) {
  const auto& k = kernels_.chain;
  if (n % 2 == 0) {
    const std::size_t h = n / 2;
    add_edge(k.c2r_pre, "c2r pre", n, h / 2 + 1, src, {slot::a, 1}, edge_twiddles(n), stages);
    const slot z = scratch_chain(h, +1, slot::a, stages);
    add_edge(k.c2r_unpack_pairs, "c2r unpack pairs", n, h, {z, 1}, dst, nullptr, stages);
    return h;
  }
  add_edge(k.c2r_expand, "c2r expand", n, n / 2 + 1, src, {slot::a, 1}, nullptr, stages);
  const slot z = scratch_chain(n, +1, slot::a, stages);
  add_edge(k.c2r_store, "c2r store real", n, n, {z, 1}, dst, nullptr, stages);
  return n;
}

template class chain_builder<double>;
template class chain_builder<float>;

} // namespace fftn::inline v0::detail
