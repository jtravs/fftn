// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "strategies/chain.hpp"

#include "core/program.hpp"
#include "core/twiddles.hpp"
#include "isa/kernel_table.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

const char* direction_name(int sign) { return sign < 0 ? "forward" : "backward"; }

std::size_t direction_index(int sign) { return sign < 0 ? 0 : 1; }

std::size_t edge_index(edge_op op) { return static_cast<std::size_t>(op); }

const char* edge_name(edge_op op) {
  constexpr std::array<const char*, edge_op_count> names{"none", "scale", "factor"};
  return names.at(edge_index(op));
}

// " (in: scale, out: factor)" for the operations present, or nothing.
std::string edges_note(edge_op in, edge_op out) {
  std::string text;
  if (in != edge_op::none) {
    text += std::string("in: ") + edge_name(in);
  }
  if (out != edge_op::none) {
    text += (text.empty() ? "" : ", ") + std::string("out: ") + edge_name(out);
  }
  return text.empty() ? "" : "; " + text;
}

slot other_scratch(slot s) { return s == slot::a ? slot::b : slot::a; }

} // namespace

template <class T>
chain_builder<T>::chain_builder(const kernels_of<T>& kernels, const target_profile& profile,
                                std::vector<std::vector<T>>& tables, std::size_t chunk_points,
                                std::vector<std::size_t> forced)
    : kernels_(kernels), profile_(profile), tables_(tables),
      chunk_points_(std::max<std::size_t>(chunk_points, 1)), forced_(std::move(forced)) {}

template <class T> std::vector<std::size_t> chain_builder<T>::radices(std::size_t n) const {
  if (!forced_.empty()) {
    std::size_t product = 1;
    for (const std::size_t r : forced_) {
      product *= r;
    }
    if (product == n) {
      return forced_;
    }
  }
  return rule_radices(n, profile_);
}

template <class T> const std::vector<std::complex<T>>& chain_builder<T>::roots(std::size_t n) {
  auto it = roots_.find(n);
  if (it == roots_.end()) {
    it = roots_.emplace(n, unit_roots<T>(-1, n)).first;
  }
  return it->second;
}

// Converts interleaved twiddles to the target's storage, per lane or broadcast, and keeps
// them.
template <class T>
const T* chain_builder<T>::pack(const std::vector<std::complex<T>>& w, bool broadcast) {
  std::vector<T> plain(2 * w.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    plain[2 * i] = w[i].real();
    plain[2 * i + 1] = w[i].imag();
  }
  std::vector<T>& stored = tables_.emplace_back(w.size() * kernels_.tw_reals);
  if (broadcast) {
    kernels_.pack_bcast_twiddles(plain.data(), w.size(), stored.data());
  } else {
    kernels_.pack_twiddles(plain.data(), w.size(), stored.data());
  }
  return stored.data();
}

template <class T> std::size_t chain_builder<T>::blocks(std::size_t count) const {
  return (count + kernels_.complex_lanes - 1) / kernels_.complex_lanes;
}

// The grouped first pass's table (stockham.inl): rows k = 1 .. R-1 of w_n^(p k) for p < m,
// each padded with ones to a multiple of C entries, in per-lane storage.
template <class T> const T* chain_builder<T>::grouped_twiddles(std::size_t n, std::size_t radix) {
  const auto key = std::make_pair(n, radix);
  if (const auto it = grouped_tables_.find(key); it != grouped_tables_.end()) {
    return it->second;
  }
  const auto& w = roots(n);
  const std::size_t m = n / radix;
  const std::size_t padded = blocks(m) * kernels_.complex_lanes;
  std::vector<std::complex<T>> t;
  t.reserve((radix - 1) * padded);
  for (std::size_t k = 1; k < radix; ++k) {
    for (std::size_t p = 0; p < padded; ++p) {
      t.push_back(p < m ? w[p * k] : std::complex<T>(1, 0));
    }
  }
  const T* stored = pack(t, false);
  grouped_tables_.emplace(key, stored);
  return stored;
}

// The table of a lane or generic pass with s copies: w_n^(p k s) = w_(R m)^(p k) for p < m,
// k = 1 .. R-1, grouped by p, in broadcast storage. p k s < n, so every entry is an entry
// of the root table.
template <class T>
const T* chain_builder<T>::shared_twiddles(std::size_t n, std::size_t radix, std::size_t s) {
  const auto key = std::make_tuple(n, radix, s);
  if (const auto it = shared_tables_.find(key); it != shared_tables_.end()) {
    return it->second;
  }
  const auto& w = roots(n);
  const std::size_t m = n / (radix * s);
  std::vector<std::complex<T>> t;
  t.reserve((radix - 1) * m);
  for (std::size_t p = 0; p < m; ++p) {
    for (std::size_t k = 1; k < radix; ++k) {
      t.push_back(w[p * k * s]);
    }
  }
  const T* stored = pack(t, true);
  shared_tables_.emplace(key, stored);
  return stored;
}

// The radix-th roots exp(-2 pi i t / radix), t < radix, for the generic pass.
template <class T> const T* chain_builder<T>::radix_roots(std::size_t radix) {
  if (const auto it = radix_roots_.find(radix); it != radix_roots_.end()) {
    return it->second;
  }
  const T* stored = pack(roots(radix), true);
  radix_roots_.emplace(radix, stored);
  return stored;
}

// w_n^k for k = 0 .. n/4: the twiddles of the real post- and pre-passes (n even). A plan
// has one real axis and builds its line once, so nothing is shared.
template <class T> const T* chain_builder<T>::edge_twiddles(std::size_t n) {
  const auto& w = roots(n);
  const std::vector<std::complex<T>> t(w.begin(),
                                       w.begin() + static_cast<std::ptrdiff_t>(n / 4 + 1));
  return pack(t, true);
}

template <class T>
std::uint32_t chain_builder<T>::chunks_of(std::size_t units, std::size_t points_per_unit) const {
  const std::size_t per_chunk = std::max<std::size_t>(1, chunk_points_ / points_per_unit);
  const std::size_t chunks = std::max<std::size_t>(1, (units + per_chunk - 1) / per_chunk);
  return static_cast<std::uint32_t>(
      std::min<std::size_t>(chunks, std::numeric_limits<std::uint32_t>::max()));
}

template <class T>
void chain_builder<T>::add_pass(std::size_t n, const std::vector<std::size_t>& rad, std::size_t i,
                                std::size_t s, int sign, endpoint src, endpoint dst,
                                const line_edges& edges, std::vector<stage>& stages) {
  const std::size_t radix = rad[i];
  const std::size_t m = n / (radix * s);
  const std::size_t c = kernels_.complex_lanes;
  const bool first = i == 0;
  const bool last = i + 1 == rad.size();
  const edge_op in = first ? edges.in : edge_op::none;
  const edge_op out = last ? edges.out : edge_op::none;
  const std::size_t d = direction_index(sign);
  const pass_kernels<T>& k = kernels_.passes;
  stage st;
  std::string kind;
  if (radix > max_codelet_radix) {
    st.run = k.generic[d][edge_index(in)][edge_index(out)];
    st.args.roots = radix_roots(radix);
    st.args.twiddles = m > 1 ? shared_twiddles(n, radix, s) : nullptr;
    st.units = m * blocks(s);
    st.chunks = chunks_of(st.units, radix * radix * c);
    kind = "generic";
  } else if (first) {
    st.run = k.codelet[radix].grouped[d][edge_index(in)];
    st.args.twiddles = grouped_twiddles(n, radix);
    st.units = blocks(m);
    st.chunks = chunks_of(st.units, radix * c);
    kind = "grouped";
  } else if (last) {
    st.run = k.codelet[radix].last[d][edge_index(out)];
    st.units = blocks(s);
    st.chunks = chunks_of(st.units, radix * c);
    kind = "lane";
  } else {
    st.run = k.codelet[radix].lane[d];
    st.args.twiddles = shared_twiddles(n, radix, s);
    st.units = m * blocks(s);
    st.chunks = chunks_of(st.units, radix * c);
    kind = "lane";
  }
  st.args.radix = radix;
  st.args.m = m;
  st.args.s = s;
  st.args.n = n;
  st.args.scale = edges.scale;
  st.args.in_factor = in == edge_op::factor ? edges.in_factor : nullptr;
  st.args.out_factor = out == edge_op::factor ? edges.out_factor : nullptr;
  st.args.src_stride = src.stride;
  st.args.dst_stride = dst.stride;
  st.src = src.where;
  st.dst = dst.where;
  st.name = kind + " radix-" + std::to_string(radix) + " " + direction_name(sign) + " (m " +
            std::to_string(m) + ", s " + std::to_string(s) +
            (radix > max_codelet_radix ? "; O(radix^2) per butterfly" : "") + edges_note(in, out) +
            ")";
  stages.push_back(std::move(st));
}

template <class T>
void chain_builder<T>::add_leaf(std::size_t n, int sign, endpoint src, endpoint dst,
                                const line_edges& edges, std::vector<stage>& stages) {
  stage st;
  st.run = kernels_.passes.codelet[n]
               .leaf[direction_index(sign)][edge_index(edges.in)][edge_index(edges.out)];
  st.args.radix = n;
  st.args.n = n;
  st.args.scale = edges.scale;
  st.args.in_factor = edges.in == edge_op::factor ? edges.in_factor : nullptr;
  st.args.out_factor = edges.out == edge_op::factor ? edges.out_factor : nullptr;
  st.args.src_stride = src.stride;
  st.args.dst_stride = dst.stride;
  st.src = src.where;
  st.dst = dst.where;
  st.units = 1;
  st.chunks = 1;
  st.name = "leaf radix-" + std::to_string(n) + " " + direction_name(sign) +
            (edges.in != edge_op::none || edges.out != edge_op::none
                 ? " (" + edges_note(edges.in, edges.out).substr(2) + ")"
                 : "");
  stages.push_back(std::move(st));
}

template <class T>
void chain_builder<T>::add_copy(std::size_t n, endpoint src, endpoint dst, const line_edges& edges,
                                std::vector<stage>& stages) {
  stage st;
  st.run = kernels_.passes.copy[edge_index(edges.in)][edge_index(edges.out)];
  st.args.n = n;
  st.args.scale = edges.scale;
  st.args.in_factor = edges.in == edge_op::factor ? edges.in_factor : nullptr;
  st.args.out_factor = edges.out == edge_op::factor ? edges.out_factor : nullptr;
  st.args.src_stride = src.stride;
  st.args.dst_stride = dst.stride;
  st.src = src.where;
  st.dst = dst.where;
  st.units = n;
  st.chunks = chunks_of(n, 1);
  st.name = "copy" + (edges.in != edge_op::none || edges.out != edge_op::none
                          ? " (" + edges_note(edges.in, edges.out).substr(2) + ")"
                          : std::string());
  stages.push_back(std::move(st));
}

template <class T>
void chain_builder<T>::add_edge(pass_fn run, const char* name, std::size_t n, std::size_t units,
                                endpoint src, endpoint dst, const T* twiddles, double scale,
                                std::vector<stage>& stages) {
  stage st;
  st.run = run;
  st.args.twiddles = twiddles;
  st.args.n = n;
  st.args.scale = scale;
  st.args.src_stride = src.stride;
  st.args.dst_stride = dst.stride;
  st.src = src.where;
  st.dst = dst.where;
  st.units = units;
  st.chunks = chunks_of(units, 2);
  st.name = name;
  if (scale != 1.0) {
    st.name += " (out: scale)";
  }
  stages.push_back(std::move(st));
}

template <class T>
slot chain_builder<T>::scratch_chain(std::size_t n, int sign, slot from,
                                     std::vector<stage>& stages) {
  const std::vector<std::size_t> rad = radices(n);
  if (rad.empty()) {
    return from;
  }
  const line_edges none;
  if (rad.size() == 1 && n <= max_codelet_radix) {
    const slot to = other_scratch(from);
    add_leaf(n, sign, {from, 1}, {to, 1}, none, stages);
    return to;
  }
  std::size_t s = 1;
  slot at = from;
  for (std::size_t i = 0; i < rad.size(); ++i) {
    const slot next = other_scratch(at);
    add_pass(n, rad, i, s, sign, {at, 1}, {next, 1}, none, stages);
    at = next;
    s *= rad[i];
  }
  return at;
}

template <class T>
std::size_t chain_builder<T>::c2c_line(std::size_t n, int sign, endpoint src, endpoint dst,
                                       bool aliased, const line_edges& edges,
                                       std::vector<stage>& stages) {
  const std::vector<std::size_t> rad = radices(n);
  if (rad.empty()) {
    add_copy(n, src, dst, edges, stages);
    return 0;
  }
  if (rad.size() == 1 && n <= max_codelet_radix) {
    add_leaf(n, sign, src, dst, edges, stages);
    return 0;
  }
  const bool wide = kernels_.complex_lanes > 1;
  const bool copy_in = wide && src.stride != 1;
  // A single (generic) pass cannot run in place.
  const bool copy_out = (wide && dst.stride != 1) || (rad.size() == 1 && aliased && !copy_in);
  const line_edges none;
  endpoint at = src;
  if (copy_in) {
    add_copy(n, src, {slot::a, 1}, none, stages);
    at = endpoint{.where = slot::a, .stride = 1};
  }
  std::size_t s = 1;
  for (std::size_t i = 0; i < rad.size(); ++i) {
    const bool last = i + 1 == rad.size();
    const endpoint next =
        last && !copy_out ? dst
                          : endpoint{.where = at.where == slot::a ? slot::b : slot::a, .stride = 1};
    add_pass(n, rad, i, s, sign, at, next, edges, stages);
    at = next;
    s *= rad[i];
  }
  if (copy_out) {
    add_copy(n, at, dst, none, stages);
  }
  return n;
}

template <class T>
std::size_t chain_builder<T>::r2c_line(std::size_t n, endpoint src, endpoint dst, double scale,
                                       std::vector<stage>& stages) {
  const pass_kernels<T>& k = kernels_.passes;
  const std::size_t scaled = scale != 1.0 ? 1 : 0;
  if (n % 2 == 0) {
    const std::size_t h = n / 2;
    add_edge(k.r2c_pack_pairs, "r2c pack pairs", n, h, src, {slot::a, 1}, nullptr, 1.0, stages);
    const slot z = scratch_chain(h, -1, slot::a, stages);
    add_edge(k.r2c_post[scaled], "r2c post", n, h / 2 + 1, {z, 1}, dst, edge_twiddles(n), scale,
             stages);
    return h;
  }
  add_edge(k.r2c_pack, "r2c pack", n, n, src, {slot::a, 1}, nullptr, 1.0, stages);
  const slot z = scratch_chain(n, -1, slot::a, stages);
  add_edge(k.r2c_store[scaled], "r2c store half", n, n / 2 + 1, {z, 1}, dst, nullptr, scale,
           stages);
  return n;
}

template <class T>
std::size_t chain_builder<T>::c2r_line(std::size_t n, endpoint src, endpoint dst, double scale,
                                       std::vector<stage>& stages) {
  const pass_kernels<T>& k = kernels_.passes;
  const std::size_t scaled = scale != 1.0 ? 1 : 0;
  if (n % 2 == 0) {
    const std::size_t h = n / 2;
    add_edge(k.c2r_pre, "c2r pre", n, h / 2 + 1, src, {slot::a, 1}, edge_twiddles(n), 1.0, stages);
    const slot z = scratch_chain(h, +1, slot::a, stages);
    add_edge(k.c2r_unpack_pairs[scaled], "c2r unpack pairs", n, h, {z, 1}, dst, nullptr, scale,
             stages);
    return h;
  }
  add_edge(k.c2r_expand, "c2r expand", n, n / 2 + 1, src, {slot::a, 1}, nullptr, 1.0, stages);
  const slot z = scratch_chain(n, +1, slot::a, stages);
  add_edge(k.c2r_store[scaled], "c2r store real", n, n, {z, 1}, dst, nullptr, scale, stages);
  return n;
}

template class chain_builder<double>;
template class chain_builder<float>;

} // namespace fftn::inline v0::detail
