// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "strategies/chain.hpp"

#include "core/program.hpp"
#include "core/twiddles.hpp"
#include "isa/kernel_table.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"
#include "platform/aligned_allocator.hpp"
#include "strategies/bluestein.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
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
                                plan_tables<T>& tables, std::size_t chunk_points,
                                std::vector<std::size_t> forced, std::size_t bluestein_above,
                                real_fusion fusion)
    : kernels_(kernels), profile_(profile), tables_(tables),
      chunk_points_(std::max<std::size_t>(chunk_points, 1)), forced_(std::move(forced)),
      fusion_(fusion),
      bluestein_above_(std::max(bluestein_above != 0 ? bluestein_above : profile.bluestein_above,
                                dense_codelet_radix)) {
  // The test knob takes only radices some pass has: a codelet's (gen::radices) or the
  // generic pass's (any other radix, in practice the primes above 16).
  for (const std::size_t r : forced_) {
    if (r < 2 || (has_codelet(r) && kernels_.passes.codelet[codelet_slot(r)]->lane[0] == nullptr)) {
      throw std::invalid_argument("fftn::plan: build_settings.radices holds radix " +
                                  std::to_string(r) + ", which no pass takes");
    }
  }
}

template <class T> bool chain_builder<T>::forces(std::size_t n) const {
  if (forced_.empty()) {
    return false;
  }
  return checked_product(forced_) == n;
}

template <class T> std::vector<std::size_t> chain_builder<T>::radices(std::size_t n) const {
  if (!forces(n) && n >= 4 && (n & (n - 1)) == 0 && n * 2 * sizeof(T) > long_from_bytes_) {
    return long_pow2_radices(static_cast<std::size_t>(std::countr_zero(n)),
                             n * 2 * sizeof(T) <= long_radix8_bytes_);
  }
  if (!forces(n) && n == fit_row_) {
    if (std::vector<std::size_t> fitted = fitting_radices(n, profile_, fit_row_fits_);
        !fitted.empty()) {
      return fitted;
    }
  }
  std::vector<std::size_t> rad = forces(n) ? forced_ : rule_radices(n, profile_);
  // The trailing radices above the Bluestein threshold (the rules' largest primes) become one
  // Bluestein stage.
  std::size_t q = 1;
  while (!rad.empty() && rad.back() > bluestein_above_ && !has_codelet(rad.back())) {
    q *= rad.back();
    rad.pop_back();
  }
  // Experiment X36: a Bluestein stage with fewer copies than a register holds (1 < s < C)
  // runs every pass of its convolution on partial units; one stage over the whole line
  // (Bluestein needs no prime) runs them whole, where its convolution is no longer than the
  // copies' together. 1018 = 2 x 509 took 0.68 of its time on AVX-512 in double (0.40 in
  // float); 12279 = 3 x 4093 in double, whose whole convolution (25920) is longer than
  // 3 x 8192, took 1.17. X36b: with s = C the units are whole but the copies' chains are
  // lane passes of s = C (no grouped first pass), so s = C takes the whole line too while
  // the line is not long (long_convolution_bytes): EPYC 1018 0.73 of its time, Xeon 2036
  // 0.81; 262142 = 2 x 131071 in NEON float lost 1.23 (its whole convolution 552960 has a
  // power-of-two part of 4096).
  const std::size_t s_copies = q > 1 ? n / q : 0;
  const bool few = s_copies > 1 &&
                   (s_copies < kernels_.complex_lanes ||
                    (s_copies == kernels_.complex_lanes && n * 2 * sizeof(T) <= conv_from_bytes_));
  // X62: with s = C, on the profiles that say so, whatever the lengths (whole_lanes_).
  if (few && !forces(n) && rader_root(q, profile_) == 0 &&
      ((whole_lanes_ && s_copies == kernels_.complex_lanes) ||
       bluestein_m(n) <= s_copies * bluestein_m(q))) {
    return {n};
  }
  if (q > 1) {
    rad.push_back(q);
  }
  return rad;
}

template <class T> std::size_t chain_builder<T>::bluestein_m(std::size_t q) const {
  if (forced_conv_ >= 2 * q - 1) {
    std::size_t rest = forced_conv_;
    for (const std::size_t f : {std::size_t{2}, std::size_t{3}, std::size_t{5}, std::size_t{7}}) {
      while (rest % f == 0) {
        rest /= f;
      }
    }
    if (rest == 1) {
      return forced_conv_;
    }
  }
  return bluestein_length(q, profile_,
                          std::max<std::size_t>(conv_pow2_part_bytes_ / (2 * sizeof(T)), 1));
}

template <class T>
std::vector<std::size_t> chain_builder<T>::convolution_radices(std::size_t len) const {
  if (len >= 4 && (len & (len - 1)) == 0 && len * 2 * sizeof(T) > conv_from_bytes_) {
    return long_pow2_radices(static_cast<std::size_t>(std::countr_zero(len)),
                             len * 2 * sizeof(T) <= conv_radix8_bytes_);
  }
  return rule_radices(len, profile_);
}

template <class T> bool chain_builder<T>::bluestein_last(std::size_t n) const {
  const std::vector<std::size_t> rad = radices(n);
  return !rad.empty() && rad.back() > bluestein_above_ && !has_codelet(rad.back());
}

template <class T> std::string chain_builder<T>::radix_text(std::size_t n) const {
  if (std::ranges::find(batch_lines_, n) != batch_lines_.end()) {
    return std::to_string(n) + " (batch leaf, " + std::to_string(batch_->lanes) +
           " rows per register)";
  }
  if (std::ranges::find(pair_lines_, n) != pair_lines_.end()) {
    if (const pair_choice p = pair_of(n); p.entry != nullptr) {
      return std::to_string(p.entry->r1) + " x " + std::to_string(p.entry->m) +
             " in registers (pair leaf)";
    }
  }
  return radix_text(n, radices(n));
}

template <class T>
std::string chain_builder<T>::radix_text(std::size_t n, const std::vector<std::size_t>& rad) const {
  if (rad.empty()) {
    return "none (a copy)";
  }
  const auto list = [](const std::vector<std::size_t>& v) {
    std::string text;
    for (const std::size_t r : v) {
      text += (text.empty() ? "" : " x ") + std::to_string(r);
    }
    return text;
  };
  if (!bluestein_last(n)) {
    // Contiguous output assumed: a strided one on NEON double has no split run (split_from).
    const std::size_t from = split_from(rad, false, 1);
    return list(rad) + (from < rad.size() ? ", split from pass " + std::to_string(from + 1) : "");
  }
  const std::size_t q = rad.back();
  const std::vector<std::size_t> head(rad.begin(), rad.end() - 1);
  if (rader_root(q, profile_) != 0) {
    return (head.empty() ? "" : list(head) + " x ") + "rader " + std::to_string(q) + " (L " +
           std::to_string(q - 1) + " = " + list(convolution_radices(q - 1)) + ")";
  }
  const std::size_t m = bluestein_m(q);
  return (head.empty() ? "" : list(head) + " x ") + "bluestein " + std::to_string(q) + " (M " +
         std::to_string(m) + " = " + list(convolution_radices(m)) + ")";
}

template <class T> const std::vector<std::complex<T>>& chain_builder<T>::roots(std::size_t n) {
  auto it = roots_.find(n);
  if (it != roots_.end()) {
    return it->second;
  }
  // Entry m 2^k of the table of n 2^k is entry m of this one, bit for bit (core/twiddles.hpp),
  // so a table already built for such a length is sliced (a real row's chain of n / 2 after
  // its pairing twiddles of n), and one of n / 2 provides the even entries.
  for (const auto& [len, w] : roots_) {
    if (len > n && len % n == 0 && std::has_single_bit(len / n)) {
      const std::size_t step = len / n;
      std::vector<std::complex<T>> sliced(n);
      for (std::size_t m = 0; m < n; ++m) {
        sliced[m] = w[m * step];
      }
      return roots_.emplace(n, std::move(sliced)).first->second;
    }
  }
  if (n % 2 == 0) {
    if (const auto half = roots_.find(n / 2); half != roots_.end()) {
      return roots_.emplace(n, unit_roots<T>(-1, n, std::span<const std::complex<T>>(half->second)))
          .first->second;
    }
  }
  return roots_.emplace(n, unit_roots<T>(-1, n)).first->second;
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
  auto& stored = tables_.emplace_back(w.size() * kernels_.tw_reals);
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

// The table of a split lane pass with s copies (split.inl): the entries of shared_twiddles,
// w_n^(p k s) for p < m and k = 1 .. R-1, grouped by p, as tight interleaved values.
template <class T>
const T* chain_builder<T>::split_twiddles(std::size_t n, std::size_t radix, std::size_t s) {
  const auto key = std::make_tuple(n, radix, s);
  if (const auto it = split_tables_.find(key); it != split_tables_.end()) {
    return it->second;
  }
  const auto& w = roots(n);
  const std::size_t m = n / (radix * s);
  auto& t = tables_.emplace_back();
  t.reserve(2 * (radix - 1) * m);
  for (std::size_t p = 0; p < m; ++p) {
    for (std::size_t k = 1; k < radix; ++k) {
      t.push_back(w[p * k * s].real());
      t.push_back(w[p * k * s].imag());
    }
  }
  split_tables_.emplace(key, t.data());
  return t.data();
}

template <class T>
std::size_t chain_builder<T>::split_from(const std::vector<std::size_t>& rad, bool bluestein,
                                         std::ptrdiff_t dst_stride) const {
  const std::size_t passes = rad.size();
  // The last split pass stores runs of W elements: stride 1 unless W = 1 (with C = 1 and
  // W > 1, NEON double, the chain copies nothing for strides).
  const bool strided = dst_stride != 1 && kernels_.real_lanes > 1;
  if (!profile_.split || bluestein || strided || passes < 3) {
    return passes;
  }
  std::size_t from = passes;
  std::size_t s = rad[0];
  for (std::size_t i = 1; i < passes; ++i) {
    if (s % kernels_.split_block == 0) {
      from = i;
      break;
    }
    s *= rad[i];
  }
  if (passes - from < 2) {
    return passes;
  }
  for (std::size_t i = from; i < passes; ++i) {
    if (!has_split_codelet(rad[i])) {
      return passes;
    }
  }
  return from;
}

template <class T>
chain_builder<T>::pass_layout chain_builder<T>::layout_of(std::size_t i, std::size_t from,
                                                          std::size_t passes) {
  if (i < from) {
    return pass_layout::interleaved;
  }
  if (i + 1 == passes) {
    return pass_layout::split_last;
  }
  return i == from ? pass_layout::split_first : pass_layout::split_mid;
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

// w_n^k for k = 0 .. n/4: the twiddles of an r2c leaf and of the strided pairing passes
// (n even). A plan has one real axis and builds its line once, so nothing is shared.
template <class T> const T* chain_builder<T>::edge_twiddles(std::size_t n) {
  const auto& w = roots(n);
  const std::vector<std::complex<T>> t(w.begin(),
                                       w.begin() + static_cast<std::ptrdiff_t>(n / 4 + 1));
  return pack(t, true);
}

// A Bluestein stage's tables (strategies/bluestein.hpp), kept per (q, sign) so that every
// axis and line of the plan with the same Q uses one copy. The kernel spectrum: M tight
// complex values (an input factor, passes/bluestein.inl). The chirp: Q entries in broadcast
// storage when copies == 0 (passes that share it across the copies), otherwise one entry
// per element of a line of `copies` interleaved sequences, copies Q, in per-lane storage
// padded with ones to whole registers.
template <class T>
chain_builder<T>::bluestein_tables chain_builder<T>::bluestein_tables_of(std::size_t q, int sign,
                                                                         std::size_t copies) {
  const std::size_t m = bluestein_m(q);
  const auto kernel_key = std::make_pair(q, sign);
  auto kit = kernel_tables_.find(kernel_key);
  if (kit == kernel_tables_.end()) {
    // For double plans the kernel's root table of M is the plan's own.
    std::vector<std::complex<double>> roots_m;
    if constexpr (std::is_same_v<T, double>) {
      roots_m = roots(m);
    } else {
      roots_m = unit_roots<double>(-1, m);
    }
    const std::vector<std::complex<double>> spectrum =
        bluestein_kernel(bluestein_chirp<double>(sign, q), roots_m);
    auto& kernel = tables_.emplace_back(2 * m);
    for (std::size_t f = 0; f < m; ++f) {
      kernel[2 * f] = static_cast<T>(spectrum[f].real());
      kernel[2 * f + 1] = static_cast<T>(spectrum[f].imag());
    }
    kit = kernel_tables_.emplace(kernel_key, kernel.data()).first;
  }
  const auto chirp_key = std::make_tuple(q, sign, copies);
  auto cit = chirp_tables_.find(chirp_key);
  if (cit == chirp_tables_.end()) {
    const std::vector<std::complex<T>> w = bluestein_chirp<T>(sign, q);
    const T* stored = nullptr;
    if (copies == 0) {
      stored = pack(w, true);
    } else {
      const std::size_t c = kernels_.complex_lanes;
      std::vector<std::complex<T>> chirp(blocks(copies * q) * c, std::complex<T>(1, 0));
      for (std::size_t j = 0; j < q; ++j) {
        for (std::size_t t = 0; t < copies; ++t) {
          chirp[t + copies * j] = w[j];
        }
      }
      stored = pack(chirp, false);
    }
    cit = chirp_tables_.emplace(chirp_key, stored).first;
  }
  return {.chirp = cit->second, .kernel = kit->second};
}

template <class T> const T* chain_builder<T>::rader_index_of(std::size_t q, std::size_t g) {
  // g^m mod q for m < q - 1, as values of T: exact, since rader_root admits q < 2^16.
  auto it = rader_index_tables_.find(q);
  if (it == rader_index_tables_.end()) {
    auto& index = tables_.emplace_back(q - 1);
    std::uint64_t e = 1;
    for (std::size_t m = 0; m + 1 < q; ++m) {
      index[m] = static_cast<T>(e);
      e = e * g % q;
    }
    it = rader_index_tables_.emplace(q, index.data()).first;
  }
  return it->second;
}

template <class T>
const T* chain_builder<T>::rader_kernel_of(std::size_t q, std::size_t g, int sign) {
  // A prime takes either Rader or Bluestein (rader_root), so the key never collides.
  const auto key = std::make_pair(q, sign);
  auto it = kernel_tables_.find(key);
  if (it == kernel_tables_.end()) {
    const std::size_t len = q - 1;
    std::vector<std::complex<double>> roots_l;
    if constexpr (std::is_same_v<T, double>) {
      roots_l = roots(len);
    } else {
      roots_l = unit_roots<double>(-1, len);
    }
    const std::vector<std::complex<double>> spectrum = rader_kernel(sign, q, g, roots_l);
    auto& kernel = tables_.emplace_back(2 * len);
    for (std::size_t f = 0; f < len; ++f) {
      kernel[2 * f] = static_cast<T>(spectrum[f].real());
      kernel[2 * f + 1] = static_cast<T>(spectrum[f].imag());
    }
    it = kernel_tables_.emplace(key, kernel.data()).first;
  }
  return it->second;
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
                                const line_edges& edges, std::vector<stage>& stages,
                                std::size_t batch, pass_layout layout) {
  const std::size_t radix = rad[i];
  const std::size_t m = n / (radix * s);
  const std::size_t c = kernels_.complex_lanes;
  const bool first = i == 0;
  const bool last = i + 1 == rad.size();
  // A Bluestein convolution's chain starts with s copies, so its first pass is a lane pass;
  // so is a single pass, which only a lanes tile of one row has (a line's single codelet
  // pass is a leaf), since it carries the output edge.
  const bool grouped = first && s == 1 && !last;
  const edge_op in = first ? edges.in : edge_op::none;
  const edge_op out = last ? edges.out : edge_op::none;
  const std::size_t d = direction_index(sign);
  const pass_kernels<T>& k = kernels_.passes;
  stage st;
  std::string kind;
  std::string note;
  const bool generic = !has_codelet(radix);
  if (layout != pass_layout::interleaved) {
    // chain_builder::split_from chose this run: codelet radices, s a multiple of W, no
    // input edge, no shared factor.
    const std::size_t w = kernels_.real_lanes;
    const radix_passes& rp = *k.codelet[codelet_slot(radix)];
    if (layout == pass_layout::split_last) {
      st.run = rp.split_last[d][edge_index(out)];
      note = "; BL to interleaved";
    } else {
      st.run = rp.split_mid[d][layout == pass_layout::split_first ? 1 : 0];
      st.args.twiddles = split_twiddles(n / batch, radix, s / batch);
      note = layout == pass_layout::split_first ? "; interleaved to BL" : "; BL";
    }
    st.units = m * (s / w);
    st.chunks = chunks_of(st.units, radix * w);
    kind = "split lane";
  } else if (generic) {
    st.run = k.generic[d][edge_index(in)][edge_index(out)];
    st.args.roots = radix_roots(radix);
    st.args.twiddles = m > 1 ? shared_twiddles(n / batch, radix, s / batch) : nullptr;
    st.units = m * blocks(s);
    st.chunks = chunks_of(st.units, radix * radix * c);
    kind = "generic";
  } else if (grouped) {
    const radix_passes& rp = *k.codelet[codelet_slot(radix)];
    st.run = edges.in_head ? rp.grouped_head[d] : rp.grouped[d][edge_index(in)];
    st.args.src_cols = edges.in_head ? edges.in_offset : 0;
    st.args.twiddles = grouped_twiddles(n, radix);
    st.units = blocks(m);
    st.chunks = chunks_of(st.units, radix * c);
    kind = "grouped";
    note = edges.in_head ? "; pruned: reads legs 0-" + std::to_string(radix / 2 - 1) : "";
  } else if (last) {
    const radix_passes& rp = *k.codelet[codelet_slot(radix)];
    st.run = edges.out_head ? rp.last_head[d][out == edge_op::scale ? 1 : 0]
             : out == edge_op::factor && edges.out_factor_copies > 1
                 ? rp.last_shared_out
                 : rp.last[d][edge_index(out)];
    st.args.dst_cols = edges.out_head ? edges.out_offset : 0;
    st.args.factor_copies = edges.out_factor_copies;
    st.units = blocks(s);
    st.chunks = chunks_of(st.units, radix * c);
    kind = "lane";
    note = edges.out_head ? "; pruned: writes legs 0-" + std::to_string(radix / 2 - 1) : "";
  } else {
    // A first pass with copies is a Bluestein convolution's (batch copies, all of them
    // here); its input factor, the backward chain's kernel spectrum, holds one entry per
    // element of a copy, which every lane shares.
    st.run = in == edge_op::factor ? k.codelet[codelet_slot(radix)]->lane_shared_in
                                   : k.codelet[codelet_slot(radix)]->lane[d];
    st.args.twiddles = shared_twiddles(n / batch, radix, s / batch);
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
            (generic ? "; O(radix^2) per butterfly" : "") + note + edges_note(in, out) + ")";
  stages.push_back(std::move(st));
}

template <class T>
void chain_builder<T>::add_leaf(std::size_t n, int sign, endpoint src, endpoint dst,
                                const line_edges& edges, std::vector<stage>& stages) {
  stage st;
  const std::size_t d = direction_index(sign);
  const std::size_t in = edge_index(edges.in);
  const std::size_t out = edge_index(edges.out);
  // The narrow table has no row of 20 (a pass radix only, X38): the target's own leaf then.
  const bool narrow = narrow_ != nullptr && codelet_slot(n) < narrow_leaf_slots &&
                      narrow_->leaf[codelet_slot(n)][0][0][0] != nullptr;
  const radix_passes& own = *kernels_.passes.codelet[codelet_slot(n)];
  // Lines of stride 1 without a factor edge: the leaf with constant offsets (X28b).
  if (src.stride == 1 && dst.stride == 1 && edges.in != edge_op::factor &&
      edges.out != edge_op::factor) {
    const std::size_t i = edges.in == edge_op::scale ? 1 : 0;
    const std::size_t o = edges.out == edge_op::scale ? 1 : 0;
    st.leaf = narrow ? narrow_->unit[codelet_slot(n)][d][i][o] : own.leaf_unit[d][i][o];
  }
  if (st.leaf == nullptr) {
    st.leaf = narrow ? narrow_->leaf[codelet_slot(n)][d][in][out] : own.leaf[d][in][out];
  }
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
leaf_fn chain_builder<T>::batch_leaf_for(std::size_t n, int sign, endpoint src, endpoint dst,
                                         const line_edges& edges) const {
  if (batch_ == nullptr || !has_codelet(n) || forces(n) || src.stride != 1 || dst.stride != 1 ||
      edges.in == edge_op::factor || edges.out == edge_op::factor) {
    return nullptr;
  }
  return batch_->leaf[codelet_slot(n)][direction_index(sign)][edges.in == edge_op::scale ? 1 : 0]
                     [edges.out == edge_op::scale ? 1 : 0];
}

template <class T>
void chain_builder<T>::add_batch_leaf(leaf_fn leaf, std::size_t n, int sign, endpoint src,
                                      endpoint dst, const line_edges& edges,
                                      std::vector<stage>& stages) {
  stage st;
  st.leaf = leaf;
  st.args.radix = n;
  st.args.n = n;
  st.args.scale = edges.scale;
  st.args.src_stride = src.stride;
  st.args.dst_stride = dst.stride;
  st.src = src.where;
  st.dst = dst.where;
  st.units = 1;
  st.chunks = 1;
  st.name = "batch leaf radix-" + std::to_string(n) + " " + direction_name(sign) + ", " +
            std::to_string(batch_->lanes) + " rows per register" +
            (edges.in != edge_op::none || edges.out != edge_op::none
                 ? " (" + edges_note(edges.in, edges.out).substr(2) + ")"
                 : "");
  stages.push_back(std::move(st));
  if (std::ranges::find(batch_lines_, n) == batch_lines_.end()) {
    batch_lines_.push_back(n);
  }
}

template <class T>
chain_builder<T>::pair_choice chain_builder<T>::pair_for(std::size_t n, endpoint src, endpoint dst,
                                                         const line_edges& edges) const {
  if (pairs_ == nullptr || forces(n) || src.stride != 1 || dst.stride != 1 ||
      edges.in == edge_op::factor || edges.out == edge_op::factor) {
    return {};
  }
  return pair_of(n);
}

template <class T> chain_builder<T>::pair_choice chain_builder<T>::pair_of(std::size_t n) const {
  for (const pair_leaf_set<T>* set : {pairs_, more_pairs_}) {
    if (set == nullptr) {
      continue;
    }
    for (const pair_leaf_entry& e : set->entry) {
      if (e.n == 0 || e.n > n) {
        break;
      }
      if (e.n == n) {
        return {.entry = &e, .set = set};
      }
    }
  }
  return {};
}

// As grouped_twiddles, padded to the set's C and in its storage: the same table where the
// set's storage is the plan's own.
template <class T>
const T* chain_builder<T>::pair_twiddles(std::size_t n, std::size_t r1,
                                         const pair_leaf_set<T>& set) {
  if (set.lanes == kernels_.complex_lanes && set.pack_twiddles == kernels_.pack_twiddles) {
    return grouped_twiddles(n, r1);
  }
  const auto& w = roots(n);
  const std::size_t m = n / r1;
  const std::size_t padded = (m + set.lanes - 1) / set.lanes * set.lanes;
  std::vector<T> plain;
  plain.reserve(2 * (r1 - 1) * padded);
  for (std::size_t k = 1; k < r1; ++k) {
    for (std::size_t p = 0; p < padded; ++p) {
      const std::complex<T> v = p < m ? w[p * k] : std::complex<T>(1, 0);
      plain.push_back(v.real());
      plain.push_back(v.imag());
    }
  }
  const std::size_t count = plain.size() / 2;
  auto& stored = tables_.emplace_back(count * kernels_.tw_reals);
  set.pack_twiddles(plain.data(), count, stored.data());
  return stored.data();
}

template <class T>
void chain_builder<T>::add_pair_leaf(const pair_choice& p, int sign, endpoint src, endpoint dst,
                                     const line_edges& edges, std::vector<stage>& stages) {
  const pair_leaf_entry& e = *p.entry;
  stage st;
  st.leaf = e.leaf[direction_index(sign)][edges.in == edge_op::scale ? 1 : 0]
                  [edges.out == edge_op::scale ? 1 : 0];
  st.args.twiddles = pair_twiddles(e.n, e.r1, *p.set);
  st.args.radix = e.r1;
  st.args.m = e.m;
  st.args.n = e.n;
  st.args.scale = edges.scale;
  st.args.src_stride = src.stride;
  st.args.dst_stride = dst.stride;
  st.src = src.where;
  st.dst = dst.where;
  st.units = 1;
  st.chunks = 1;
  st.name = "pair leaf radix-" + std::to_string(e.r1) + " x " + std::to_string(e.m) +
            " in registers " + direction_name(sign) +
            (edges.in != edge_op::none || edges.out != edge_op::none
                 ? " (" + edges_note(edges.in, edges.out).substr(2) + ")"
                 : "");
  stages.push_back(std::move(st));
  if (std::ranges::find(pair_lines_, e.n) == pair_lines_.end()) {
    pair_lines_.push_back(e.n);
  }
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
slot chain_builder<T>::add_convolution(std::size_t s, std::size_t len,
                                       const std::vector<std::size_t>& rad, const T* kernel,
                                       slot conv, std::vector<stage>& stages) {
  // The convolution's passes must be codelet lane passes with at least two of them: the
  // kernel spectrum is shared through the last lane pass's output or the first backward
  // lane pass's input (factor_sharing), which the generic pass, a grouped pass with copies
  // and a single pass that is both first and last would not honour.
  const pass_kernels<T>& k = kernels_.passes;
  if (rad.size() < 2) {
    throw std::logic_error("fftn: a convolution of " + std::to_string(len) +
                           " points has fewer than two passes");
  }
  for (const std::size_t r : rad) {
    if (!has_codelet(r) || k.codelet[codelet_slot(r)]->lane[0] == nullptr) {
      throw std::logic_error("fftn: a convolution of " + std::to_string(len) +
                             " points has radix " + std::to_string(r) + ", which no codelet takes");
    }
  }
  // The kernel spectrum, len entries for all the copies: output element t + s f of the
  // forward chain (copy t) takes entry f. Its last pass applies it when C divides s, its
  // lanes then sharing one entry (factor_sharing::output; s = 1 is the plain per-element
  // factor). Otherwise the lanes of that pass straddle entries, and the first backward
  // pass, a lane pass whose lanes are the copies, multiplies its input element t + s f by
  // entry f instead (factor_sharing::input). Measured on the Mac (C = 1, task 2.3b): the
  // input form cost 5 % at 4096 x 17 against the output form, whose multiplications hide
  // under the last pass's memory traffic.
  const std::size_t c = kernels_.complex_lanes;
  line_edges forward;
  line_edges backward;
  if (s % c == 0 || s == 1) {
    forward.out = edge_op::factor;
    forward.out_factor = kernel;
    forward.out_factor_copies = s;
  } else {
    backward.in = edge_op::factor;
    backward.in_factor = kernel;
  }
  slot at = conv;
  // Experiment X36: one copy runs the split run a line of len points would (split_from), its
  // last forward pass applying the kernel spectrum as a per-element output factor; copies,
  // whose factor or lanes are shared, stay interleaved.
  const std::size_t split = s == 1 ? split_from(rad, false, 1) : rad.size();
  for (const int dir : {-1, +1}) {
    std::size_t copies = s;
    for (std::size_t i = 0; i < rad.size(); ++i) {
      const slot next = other_scratch(at);
      add_pass(s * len, rad, i, copies, dir, {at, 1}, {next, 1}, dir < 0 ? forward : backward,
               stages, s, layout_of(i, split, rad.size()));
      at = next;
      copies *= rad[i];
    }
  }
  return at;
}

template <class T>
endpoint chain_builder<T>::last_stage_to(std::size_t q, endpoint at, endpoint next) const {
  const bool scratch = at.where == slot::a || at.where == slot::b;
  if (rader_root(q, profile_) != 0 && scratch && next.where == other_scratch(at.where)) {
    return endpoint{.where = at.where, .stride = 1};
  }
  return next;
}

template <class T>
void chain_builder<T>::add_rader(std::size_t n, std::size_t q, std::size_t g, int sign,
                                 endpoint src, endpoint dst, const line_edges& edges, bool first,
                                 std::vector<stage>& stages) {
  const std::size_t s = n / q;
  const std::size_t len = q - 1;
  const std::vector<std::size_t> rad = convolution_radices(len);
  const std::size_t c = kernels_.complex_lanes;
  const edge_op in = first ? edges.in : edge_op::none;
  const pass_kernels<T>& k = kernels_.passes;
  const bool shared = s % c == 0;
  const T* kernel = rader_kernel_of(q, g, sign);
  const T* index = rader_index_of(q, g);
  // The convolution runs in the two scratch slots and ends where it starts (an even number
  // of passes), in neither the stage's source (rader_in gathers from it) nor its destination
  // (rader_out scatters into it); its other slot may be either, which is read or not yet
  // written by then. last_stage_to keeps the two apart.
  const auto scratch = [](slot w) { return w == slot::a || w == slot::b; };
  const slot conv = [&] {
    if (scratch(dst.where)) {
      return other_scratch(dst.where);
    }
    return scratch(src.where) ? other_scratch(src.where) : slot::a;
  }();
  if (conv == src.where) {
    throw std::logic_error("fftn: a Rader stage from scratch slot to the other scratch slot");
  }
  const std::string geometry = "Q " + std::to_string(q) + ", L " + std::to_string(len) + ", root " +
                               std::to_string(g) + ", copies " + std::to_string(s);
  const std::size_t units = shared ? s / c : s;
  stage st;
  st.run = k.rader_in[shared ? 1 : 0][edge_index(in)];
  st.args.m = len;
  st.args.s = s;
  st.args.n = n;
  st.args.roots = index;
  st.args.scale = edges.scale;
  st.args.in_factor = in == edge_op::factor ? edges.in_factor : nullptr;
  st.args.src_stride = src.stride;
  st.args.dst_stride = 1;
  st.src = src.where;
  st.dst = conv;
  st.units = units;
  st.chunks = chunks_of(units, q * (shared ? c : 1));
  st.name = std::string("rader in, gather ") + direction_name(sign) + " (" + geometry +
            edges_note(in, edge_op::none) + ")";
  stages.push_back(std::move(st));

  const slot at = add_convolution(s, len, rad, kernel, conv, stages);
  for (std::size_t i = stages.size() - 2 * rad.size(); i < stages.size(); ++i) {
    stages[i].name = "rader convolution: " + stages[i].name;
  }

  stage out;
  out.run = k.rader_out[shared ? 1 : 0][edge_index(edges.out)];
  out.args.m = len;
  out.args.s = s;
  out.args.n = n;
  out.args.roots = index;
  out.args.scale = edges.scale;
  out.args.out_factor = edges.out == edge_op::factor ? edges.out_factor : nullptr;
  out.args.src_stride = 1;
  out.args.dst_stride = dst.stride;
  out.src = at;
  out.dst = dst.where;
  out.units = units;
  out.chunks = chunks_of(units, q * (shared ? c : 1));
  out.name = std::string("rader out, scatter ") + direction_name(sign) + " (" + geometry +
             edges_note(edge_op::none, edges.out) + ")";
  stages.push_back(std::move(out));
  // The convolution's s L elements, then x_0 and the sum of each copy (rader_in).
  line_scratch_ = std::max(line_scratch_, s * (len + 2));
}

template <class T>
stage chain_builder<T>::bluestein_in_stage(std::size_t n, std::size_t q, std::size_t m, int sign,
                                           endpoint src, slot conv, edge_op in,
                                           const line_edges& edges, const bluestein_tables& tab) {
  const std::size_t s = n / q;
  const std::size_t c = kernels_.complex_lanes;
  const bool shared = s > 1 && s % c == 0;
  stage st;
  st.run = kernels_.passes.bluestein_in[shared ? 1 : 0][edge_index(in)];
  st.args.m = m;
  st.args.s = s;
  st.args.n = n;
  st.args.twiddles = tab.chirp;
  st.args.scale = edges.scale;
  st.args.in_factor = in == edge_op::factor ? edges.in_factor : nullptr;
  st.args.src_stride = src.stride;
  st.args.dst_stride = 1;
  st.src = src.where;
  st.dst = conv;
  st.units = blocks(s * m);
  st.chunks = chunks_of(st.units, c);
  st.name = std::string("bluestein in, chirp ") + direction_name(sign) + " (Q " +
            std::to_string(q) + ", M " + std::to_string(m) + ", copies " + std::to_string(s) +
            edges_note(in, edge_op::none) + ")";
  return st;
}

template <class T>
stage chain_builder<T>::bluestein_out_stage(std::size_t n, std::size_t q, std::size_t m, int sign,
                                            slot from, endpoint dst, const line_edges& edges,
                                            const bluestein_tables& tab) {
  const std::size_t s = n / q;
  const std::size_t c = kernels_.complex_lanes;
  const bool shared = s > 1 && s % c == 0;
  stage out;
  out.run = kernels_.passes.bluestein_out[shared ? 1 : 0][edge_index(edges.out)];
  out.args.m = m;
  out.args.s = s;
  out.args.n = n;
  out.args.twiddles = tab.chirp;
  out.args.scale = edges.scale;
  out.args.out_factor = edges.out == edge_op::factor ? edges.out_factor : nullptr;
  out.args.src_stride = 1;
  out.args.dst_stride = dst.stride;
  out.src = from;
  out.dst = dst.where;
  out.units = blocks(s * q);
  out.chunks = chunks_of(out.units, c);
  out.name = std::string("bluestein out, chirp ") + direction_name(sign) + " (Q " +
             std::to_string(q) + ", M " + std::to_string(m) + ", copies " + std::to_string(s) +
             edges_note(edge_op::none, edges.out) + ")";
  return out;
}

template <class T>
void chain_builder<T>::add_bluestein(std::size_t n, std::size_t q, int sign, endpoint src,
                                     endpoint dst, const line_edges& edges, bool first,
                                     std::vector<stage>& stages) {
  if (const std::size_t g = rader_root(q, profile_); g != 0) {
    add_rader(n, q, g, sign, src, dst, edges, first, stages);
    return;
  }
  const std::size_t s = n / q;
  const std::size_t m = bluestein_m(q);
  const std::vector<std::size_t> rad = convolution_radices(m);
  const std::size_t c = kernels_.complex_lanes;
  const edge_op in = first ? edges.in : edge_op::none;
  const bool shared = s > 1 && s % c == 0;
  const bluestein_tables tab = bluestein_tables_of(q, sign, shared ? 0 : s);
  // The convolution starts in the scratch line the source is not in, and ends there.
  const slot conv =
      src.where == slot::a || src.where == slot::b ? other_scratch(src.where) : slot::a;
  stages.push_back(bluestein_in_stage(n, q, m, sign, src, conv, in, edges, tab));
  const slot at = add_convolution(s, m, rad, tab.kernel, conv, stages);
  for (std::size_t i = stages.size() - 2 * rad.size(); i < stages.size(); ++i) {
    stages[i].name = "bluestein convolution: " + stages[i].name;
  }
  stages.push_back(bluestein_out_stage(n, q, m, sign, at, dst, edges, tab));
  line_scratch_ = std::max(line_scratch_, s * m);
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
  if (rad.size() == 1 && has_codelet(n)) {
    const slot to = other_scratch(from);
    add_leaf(n, sign, {from, 1}, {to, 1}, none, stages);
    return to;
  }
  const bool bluestein = bluestein_last(n);
  const std::size_t split = split_from(rad, bluestein, 1);
  std::size_t s = 1;
  slot at = from;
  for (std::size_t i = 0; i < rad.size(); ++i) {
    slot next = other_scratch(at);
    if (bluestein && i + 1 == rad.size()) {
      next = last_stage_to(rad[i], {at, 1}, {next, 1}).where;
      add_bluestein(n, rad[i], sign, {at, 1}, {next, 1}, none, i == 0, stages);
    } else {
      add_pass(n, rad, i, s, sign, {at, 1}, {next, 1}, none, stages, 1,
               layout_of(i, split, rad.size()));
    }
    at = next;
    s *= rad[i];
  }
  return at;
}

// c2c_line's routes before its passes, for lines of stride 1 without a factor: a batch leaf or
// a leaf (one codelet radix), a copy (n = 1), a pair leaf; else its passes, two or more.
template <class T> bool chain_builder<T>::head_first(std::size_t n) const {
  const std::vector<std::size_t> rad = radices(n);
  return rad.size() >= 2 && (batch_ == nullptr || !has_codelet(n)) &&
         pair_for(n, {}, {}, line_edges()).entry == nullptr &&
         has_pruned_codelet(rad.front());
}

template <class T> bool chain_builder<T>::head_last(std::size_t n) const {
  const std::vector<std::size_t> rad = radices(n);
  return rad.size() >= 2 && (batch_ == nullptr || !has_codelet(n)) &&
         pair_for(n, {}, {}, line_edges()).entry == nullptr &&
         has_pruned_codelet(rad.back()) && !bluestein_last(n);
}

template <class T>
std::size_t chain_builder<T>::c2c_line(std::size_t n, int sign, endpoint src, endpoint dst,
                                       bool aliased, const line_edges& edges,
                                       std::vector<stage>& stages) {
  // The pruned edge passes are the caller's choice from head_first and head_last, on lines of
  // stride 1, which then take neither leaves nor pair leaves.
  const bool heads = edges.in_head || edges.out_head;
  if (heads && ((edges.in_head && (!head_first(n) || src.stride != 1 ||
                                   edges.in != edge_op::none)) ||
                (edges.out_head && (!head_last(n) || dst.stride != 1 ||
                                    edges.out == edge_op::factor)))) {
    throw std::logic_error("fftn: internal: a pruned edge pass where c2c_line builds none");
  }
  line_scratch_ = 0;
  if (const leaf_fn batch = batch_leaf_for(n, sign, src, dst, edges); batch != nullptr) {
    add_batch_leaf(batch, n, sign, src, dst, edges, stages);
    return 0;
  }
  const std::vector<std::size_t> rad = radices(n);
  if (rad.empty()) {
    add_copy(n, src, dst, edges, stages);
    return 0;
  }
  if (rad.size() == 1 && has_codelet(n)) {
    // A codelet of up to 16 points with a pair leaf (8, 12, 16 on x86, X37) takes it: two
    // passes on whole registers rather than one on a quarter or half of each.
    if (const pair_choice p = n <= 16 ? pair_for(n, src, dst, edges) : pair_choice{};
        p.entry != nullptr) {
      add_pair_leaf(p, sign, src, dst, edges, stages);
      return 0;
    }
    add_leaf(n, sign, src, dst, edges, stages);
    return 0;
  }
  if (const pair_choice p = pair_for(n, src, dst, edges); p.entry != nullptr) {
    add_pair_leaf(p, sign, src, dst, edges, stages);
    return 0;
  }
  const bool wide = kernels_.complex_lanes > 1;
  const bool copy_in = wide && src.stride != 1;
  const bool bluestein = bluestein_last(n);
  // A single generic pass cannot run in place; a Bluestein stage can.
  const bool copy_out =
      (wide && dst.stride != 1) || (rad.size() == 1 && aliased && !copy_in && !bluestein);
  const line_edges none;
  endpoint at = src;
  if (copy_in) {
    add_copy(n, src, {slot::a, 1}, none, stages);
    at = endpoint{.where = slot::a, .stride = 1};
  }
  // A contiguous destination that is not the source serves as the second scratch line:
  // passes alternate between it and a, ending in it, so a chain touches three lines (source,
  // destination, a) rather than four, which keeps rows of 512 to 1024 complex doubles in
  // a 32 KiB L1 longer (dev/experiments/X18-x86-kernels.md).
  // Not with a pruned last pass, whose destination takes the line's first half only.
  const bool via_dst = !aliased && !copy_in && !copy_out && !bluestein && dst.stride == 1 &&
                       dst.where != slot::a && dst.where != slot::b && !edges.out_head;
  // Two passes touch two lines: the first writes the destination and the last runs there in
  // place. A last pass (m = 1) reads and writes the same R elements in each unit, and every
  // output of a codelet depends on every input, so each unit loads all before it stores. On
  // the EPYC a 512-point row refilled L1 517 times with three lines against FFTW's 181
  // (dev/experiments/X47-epyc-mid-2.md). Not after a generic pass, which may store as it goes.
  const bool two_lines = via_dst && rad.size() == 2 && has_codelet(rad.back());
  // A pruned last pass is interleaved: a split run then ends before it, in the pass that writes
  // the interleaved line (layout_of), if the passes before it make a run.
  const std::size_t split_end = rad.size() - (edges.out_head ? 1 : 0);
  const std::size_t split =
      edges.out_head
          ? split_from(std::vector<std::size_t>(rad.begin(), rad.end() - 1), false, 1)
          : split_from(rad, bluestein, copy_out ? 1 : dst.stride);
  std::size_t s = 1;
  for (std::size_t i = 0; i < rad.size(); ++i) {
    const bool last = i + 1 == rad.size();
    endpoint next{.where = at.where == slot::a ? slot::b : slot::a, .stride = 1};
    if ((last && !copy_out) || two_lines) {
      next = dst;
    } else if (via_dst) {
      next = (rad.size() - 1 - i) % 2 == 0 ? dst : endpoint{.where = slot::a, .stride = 1};
    }
    if (bluestein && last) {
      next = last_stage_to(rad[i], at, next);
      add_bluestein(n, rad[i], sign, at, next, edges, i == 0, stages);
    } else {
      add_pass(n, rad, i, s, sign, at, next, edges, stages, 1,
               i < split_end ? layout_of(i, split, split_end) : pass_layout::interleaved);
    }
    at = next;
    s *= rad[i];
  }
  if (copy_out) {
    add_copy(n, at, dst, none, stages);
  }
  // A single pass from the caller's memory to the caller's memory touches no scratch, nor do
  // two-line chains.
  return std::max(copy_in || copy_out || (rad.size() > 1 && !two_lines) ? n : 0, line_scratch_);
}

template <class T>
void chain_builder<T>::chain_tail(std::size_t n, const std::vector<std::size_t>& rad,
                                  std::size_t i0, std::size_t s, int sign, endpoint at,
                                  endpoint dst, const line_edges& edges,
                                  std::vector<stage>& stages) {
  const bool bluestein = bluestein_last(n);
  // The same split run as the whole chain's (c2c_line): a c2r line whose pairing is fused
  // into the first pass then runs the passes the separate route runs, so the two routes, and
  // the targets of a pair that choose them differently, give the same bits.
  std::size_t split = std::max(i0, split_from(rad, bluestein, dst.stride));
  if (rad.size() - split < 2) {
    split = rad.size();
  }
  for (std::size_t i = i0; i < rad.size(); ++i) {
    const bool last = i + 1 == rad.size();
    endpoint next = last ? dst : endpoint{.where = other_scratch(at.where), .stride = 1};
    if (bluestein && last) {
      next = last_stage_to(rad[i], at, next);
      add_bluestein(n, rad[i], sign, at, next, edges, i == 0, stages);
    } else {
      add_pass(n, rad, i, s, sign, at, next, edges, stages, 1, layout_of(i, split, rad.size()));
    }
    at = next;
    s *= rad[i];
  }
}

template <class T>
bool chain_builder<T>::depth_fits(std::size_t n, const std::vector<std::size_t>& tops) const {
  const std::size_t c = kernels_.complex_lanes;
  // The first pass's last block of sub-rows may be partial with one top (X51), not with two:
  // its stepping over blocks of the first top's columns needs C to divide F1 (depth.inl).
  if (tops.empty() || tops.size() > 2 || (tops.size() == 2 && tops[0] % c != 0)) {
    return false;
  }
  std::size_t m = n;
  for (const std::size_t f : tops) {
    if (f < 2 || m % f != 0 || !has_codelet(f) ||
        kernels_.passes.codelet[codelet_slot(f)]->depth_top[0][0] == nullptr) {
      return false;
    }
    m /= f;
  }
  if (m % c != 0) {
    return false;
  }
  const std::vector<std::size_t> rad = radices(m);
  return rad.size() >= 2 && std::ranges::all_of(rad, has_codelet) &&
         kernels_.passes.codelet[codelet_slot(rad[0])]->depth_first[0] != nullptr;
}

// The block B of a top pass over legs = M (depth.inl): the divisor of M that is a multiple of
// C with the fewest entries B + M / B in its two tables (C divides M: depth_fits).
template <class T> std::size_t chain_builder<T>::derived_block(std::size_t legs) const {
  const std::size_t c = kernels_.complex_lanes;
  std::size_t best = 0;
  for (std::size_t b = c; b <= legs; b += c) {
    if (legs % b == 0 && (best == 0 || b + legs / b < best + legs / best)) {
      best = b;
    }
  }
  return best;
}

// The top pass's two tables for a line of n = f M (depth.inl): rows r = 1 .. f - 1 of
// w_n^(r k_lo) for k_lo < block, per-lane, and w_n^(r k_hi block) for k_hi < M / block, r = 1
// .. f - 1, grouped by k_hi, broadcast. Every entry is an entry of the root table of n.
template <class T>
std::pair<const T*, const T*> chain_builder<T>::derived_twiddles(std::size_t n, std::size_t f,
                                                                 std::size_t block) {
  const auto& w = roots(n);
  const std::size_t legs = n / f;
  std::vector<std::complex<T>> lo;
  lo.reserve((f - 1) * block);
  for (std::size_t r = 1; r < f; ++r) {
    for (std::size_t k = 0; k < block; ++k) {
      lo.push_back(w[r * k]);
    }
  }
  std::vector<std::complex<T>> hi;
  hi.reserve((f - 1) * (legs / block));
  for (std::size_t k = 0; k < legs / block; ++k) {
    for (std::size_t r = 1; r < f; ++r) {
      hi.push_back(w[r * k * block]);
    }
  }
  const T* lo_stored = pack(lo, false);
  return {lo_stored, pack(hi, true)};
}

template <class T>
std::string chain_builder<T>::depth_text(std::size_t n,
                                         const std::vector<std::size_t>& tops) const {
  std::size_t m = n;
  std::string text = "depth-first ";
  for (const std::size_t f : tops) {
    text += std::to_string(f) + " x ";
    m /= f;
  }
  return text + std::to_string(m) + " (sub-rows " + radix_text(m) + ")";
}

// The passes, in order: the first pass of the sub-rows' chains (depth_first_pass) over the
// whole line, from the input to the output; for each sub-row (the plan's loop), passes 2 ..
// of its chain on its block, alternating with scratch a and ending in the block (an odd
// count takes b for one pass, a single pass a copy back); the top passes in place, the inner
// one (two tops) on each of F1 runs, the last from the work slot to the output. The sub-rows'
// chains are the chain of M the line would get, so they take its split run.
template <class T>
chain_builder<T>::depth_stages
chain_builder<T>::c2c_depth(std::size_t n, const std::vector<std::size_t>& tops, int sign,
                            double scale, slot work, slot src, slot dst, const T* out_factor) {
  depth_stages out;
  const std::size_t f1 = tops[0];
  const std::size_t f2 = tops.size() > 1 ? tops[1] : 1;
  const std::size_t subs = f1 * f2;
  const std::size_t m = n / subs;
  const std::size_t c = kernels_.complex_lanes;
  const std::size_t d = direction_index(sign);
  const std::vector<std::size_t> rad = radices(m);
  const std::size_t r2 = rad[0];
  const line_edges none;

  stage first;
  first.run = kernels_.passes.codelet[codelet_slot(r2)]->depth_first[d];
  first.args.twiddles = shared_twiddles(n, r2, subs); // w_n^(p k S) = w_M^(p k)
  first.args.radix = r2;
  first.args.m = m / r2;
  first.args.s = subs;
  first.args.n = n; // the pass moves the whole line (work_graph.cpp, stage_work)
  first.args.dst_cols = f1;
  first.args.dst_qstride = static_cast<std::ptrdiff_t>(f2 * m);
  first.src = src;
  first.dst = work;
  first.units = (m / r2) * ((subs + c - 1) / c);
  first.chunks = chunks_of(first.units, r2 * c);
  first.name = "depth-first first pass radix-" + std::to_string(r2) + " " + direction_name(sign) +
               " (" + std::to_string(subs) + " sub-rows of " + std::to_string(m) +
               ", each to its block)";
  out.first.push_back(std::move(first));

  const std::size_t passes = rad.size() - 1;
  const std::size_t split = split_from(rad, false, 1);
  endpoint at{.where = work, .stride = 1};
  std::size_t s = r2;
  for (std::size_t i = 1; i < rad.size(); ++i) {
    const std::size_t left = rad.size() - i; // passes from this one to the end
    endpoint next{.where = slot::a, .stride = 1};
    if (passes > 1 && passes % 2 == 1 && left == 2) {
      next = endpoint{.where = slot::b, .stride = 1};
    } else if (passes > 1 && (left == 1 || at.where != work)) {
      next = endpoint{.where = work, .stride = 1};
    }
    add_pass(m, rad, i, s, sign, at, next, none, out.sub, 1, layout_of(i, split, rad.size()));
    at = next;
    s *= rad[i];
  }
  if (at.where != work) {
    add_copy(m, at, {work, 1}, none, out.sub);
  }
  out.scratch = m;

  // A top pass of radix f over runs of f * legs elements (legs apart), the scale on the last.
  const auto top_pass = [&](std::size_t f, std::size_t legs, bool last, std::vector<stage>& to) {
    stage st;
    std::size_t e = last && scale != 1.0 ? 1 : 0;
    if (last && out_factor != nullptr) {
      e = 2;
    }
    const std::size_t block = derived_block(legs);
    const auto [lo, hi] = derived_twiddles(f * legs, f, block);
    st.run = kernels_.passes.codelet[codelet_slot(f)]->depth_top[d][e];
    st.args.twiddles = lo;
    st.args.roots = hi;
    st.args.s = block;
    st.args.radix = f;
    st.args.m = legs;
    st.args.n = f * legs;
    st.args.scale = scale;
    st.args.out_factor = last ? out_factor : nullptr;
    st.src = work;
    st.dst = last ? dst : work;
    st.units = legs / c;
    st.chunks = chunks_of(st.units, f * c);
    st.name = "depth-first top pass radix-" + std::to_string(f) + " " + direction_name(sign) +
              " (decimation in time, " + (st.dst == st.src ? "in place" : "to the output") +
              ", legs " + std::to_string(legs) + " apart" + "; twiddles derived over blocks of " +
              std::to_string(block) + (e == 1 ? "; out: scale" : "") +
              (e == 2 ? "; out: factor" : "") + ")";
    to.push_back(std::move(st));
  };
  if (f2 > 1) {
    top_pass(f2, m, false, out.mid);
  }
  top_pass(f1, f2 * m, true, out.top);
  return out;
}

template <class T>
bool chain_builder<T>::bluestein_depth_fits(std::size_t m,
                                            const std::vector<std::size_t>& tops) const {
  return depth_fits(m, tops) && checked_product(tops) % kernels_.complex_lanes == 0;
}

template <class T>
bool chain_builder<T>::c2r_depth_fits(std::size_t h, const std::vector<std::size_t>& tops) const {
  if (!depth_fits(h, tops) || checked_product(tops) % kernels_.complex_lanes != 0) {
    return false;
  }
  const std::size_t r2 = radices(h / checked_product(tops))[0];
  return kernels_.passes.codelet[codelet_slot(r2)]->depth_first_c2r != nullptr;
}

// Experiment X78: the c2r pairing in the first pass's loads saves the separate pass's write of h
// values to the workspace block and their second read, and in a team the phase between them.
template <class T>
chain_builder<T>::depth_stages
chain_builder<T>::c2r_depth(std::size_t n, const std::vector<std::size_t>& tops, double scale) {
  const std::size_t h = n / 2;
  depth_stages out = c2c_depth(h, tops, +1, scale, slot::out, slot::in, slot::out);
  stage& first = out.first.front();
  first.run = kernels_.passes.codelet[codelet_slot(first.args.radix)]->depth_first_c2r;
  const auto [w, om] = depth_c2r_twiddles(n, first.args.radix);
  first.args.in_factor = w;
  first.args.roots = om;
  first.name += "; the c2r pairing fused in, from the input's " + std::to_string(h + 1) + " bins";
  return out;
}

template <class T> std::size_t chain_builder<T>::bluestein_single(std::size_t n) const {
  const std::vector<std::size_t> rad = radices(n);
  return rad.size() == 1 && bluestein_last(n) && rader_root(n, profile_) == 0 ? n : 0;
}

// Experiment X50: the convolution of a Bluestein stage of one copy as two depth-first chains
// (c2c_depth), the forward one's blocks in the workspace block and its top pass into scratch
// a, where the backward one's first pass reads, so that each chain's sub-rows run in cache
// with a loop moving the block, as a row of M would (X46). Once the backward first pass has
// read scratch a, it is free for the sub-rows' passes (the forward sub-rows run before a is
// written).
template <class T>
chain_builder<T>::bluestein_depth_stages
chain_builder<T>::bluestein_depth(std::size_t q, const std::vector<std::size_t>& tops, int sign,
                                  endpoint src, endpoint dst, const line_edges& edges) {
  const std::size_t m = bluestein_m(q);
  const bluestein_tables tab = bluestein_tables_of(q, sign, 1);
  bluestein_depth_stages out;
  out.forward = c2c_depth(m, tops, -1, 1.0, slot::block, src.where, slot::a, tab.kernel);
  out.backward = c2c_depth(m, tops, +1, 1.0, slot::block, slot::a, slot::block);
  // Arm C: the chirp passes fused into the forward chain's first pass (reading the input) and
  // the backward chain's last top pass (writing the first Q outputs), which saves a pass over
  // M elements each way.
  stage& first = out.forward.first.front();
  first.run = kernels_.passes.codelet[codelet_slot(first.args.radix)]->depth_first_chirp;
  first.args.in_factor = tab.chirp;
  first.args.src_cols = q;
  first.name += "; chirp " + std::string(direction_name(sign)) + " in, from the input";
  stage& top = out.backward.top.back();
  const bool scaled = edges.out == edge_op::scale;
  top.run = kernels_.passes.codelet[codelet_slot(top.args.radix)]->depth_top_chirp[scaled ? 1 : 0];
  top.args.out_factor = tab.chirp;
  top.args.dst_cols = q;
  top.args.scale = edges.scale;
  top.dst = dst.where;
  top.name += "; chirp " + std::string(direction_name(sign)) + " out, the first " +
              std::to_string(q) + " to the output" + (scaled ? "; out: scale" : "");
  for (std::vector<stage>* part : {
           &out.forward.first,
           &out.forward.sub,
           &out.forward.mid,
           &out.forward.top,
           &out.backward.first,
           &out.backward.sub,
           &out.backward.mid,
           &out.backward.top,
       }) {
    for (stage& st : *part) {
      st.name = "bluestein convolution: " + st.name;
    }
  }
  out.scratch = std::max({m, out.forward.scratch, out.backward.scratch});
  return out;
}

template <class T>
std::string chain_builder<T>::bluestein_depth_text(std::size_t q,
                                                   const std::vector<std::size_t>& tops) const {
  const std::size_t m = bluestein_m(q);
  return "bluestein " + std::to_string(q) + " (M " + std::to_string(m) + " = " +
         depth_text(m, tops) + ")";
}

template <class T> real_route chain_builder<T>::route(std::size_t n, bool r2c, bool strided) const {
  if (n % 2 == 1) {
    return real_route::full;
  }
  if (strided) {
    return real_route::strided;
  }
  const std::size_t h = n / 2;
  if (h == 1) {
    return real_route::pairing_only;
  }
  if (fusion_.radix <= 1) {
    return real_route::separate;
  }
  const std::vector<std::size_t> rad = real_radices(h, r2c);
  // A leaf of h up to 16 or of the prime codelets (17, 19, 23: experiment X27); not of 32 or
  // 64, whose leaves lose wherever C > 1 (experiment X8).
  if (rad.size() == 1 && has_codelet(h) && (h <= dense_codelet_radix || h % 2 == 1)) {
    return real_route::leaf;
  }
  // r2c: the last pass must be a codelet's (not a Bluestein stage or a generic pass); c2r:
  // the first.
  if (rad.size() < 2) {
    return real_route::separate;
  }
  // r2c: a fused pairing would be the split run's last pass, so a half-length chain with a
  // split run takes the separate route, as a target that does not fuse does: the pair then
  // runs the same passes and gives the same bits, and the row keeps the split gain.
  if (r2c) {
    const std::vector<std::size_t> whole = radices(h);
    if (split_from(whole, bluestein_last(h), 1) < whole.size()) {
      return real_route::separate;
    }
  }
  const std::size_t edge_radix = r2c ? rad.back() : rad.front();
  const std::size_t half_copies = (h / edge_radix - 1) / 2; // copies (groups) 1 .. s/2
  const bool fits = edge_radix <= std::min(fusion_.radix, dense_codelet_radix) &&
                    half_copies >= fusion_.min_lanes * kernels_.complex_lanes &&
                    (!r2c || h * 2 * sizeof(T) <= fusion_.r2c_bytes);
  return fits ? real_route::fused : real_route::separate;
}

// The chain's radices for h with, where the pairing would fuse, the smallest radix within
// the limit moved to the pass it fuses into (last for r2c, first for c2r) when that pass's
// radix exceeds the limit; a Bluestein stage stays last, and forced radices stay as given.
template <class T>
std::vector<std::size_t> chain_builder<T>::real_radices(std::size_t h, bool r2c) const {
  std::vector<std::size_t> rad = radices(h);
  const bool bluestein = bluestein_last(h);
  if (forces(h) || rad.size() < 2 || fusion_.radix <= 1 || (r2c && bluestein)) {
    return rad;
  }
  const std::size_t limit = std::min(fusion_.radix, dense_codelet_radix);
  if ((r2c ? rad.back() : rad.front()) <= limit) {
    return rad;
  }
  // Experiment X39: an override that starts with a radix-20 pass (X38's codelet) is worth it
  // only first; moved behind the c2r pairing's small radix it lost (c2r 1000 and 10000 on the
  // Xeon 1.19-1.25), so a c2r line takes the costs' sequence instead, where one has a radix
  // the pairing can take.
  if (!r2c && rad.front() == 20 && !bluestein &&
      std::ranges::any_of(rad, [limit](std::size_t r) { return r <= limit; })) {
    if (std::vector<std::size_t> costs = rule_radices(h, profile_, false); costs.size() >= 2) {
      rad = std::move(costs);
      if (rad.front() <= limit) {
        return rad;
      }
    }
  }
  const std::size_t end = bluestein ? rad.size() - 1 : rad.size();
  std::size_t pick = end;
  for (std::size_t i = 0; i < end; ++i) {
    if (rad[i] <= limit && (pick == end || rad[i] < rad[pick])) {
      pick = i;
    }
  }
  if (pick == end) {
    return rad;
  }
  const std::size_t r = rad[pick];
  rad.erase(rad.begin() + static_cast<std::ptrdiff_t>(pick));
  rad.insert(r2c ? rad.end() : rad.begin(), r);
  return rad;
}

template <class T>
std::string chain_builder<T>::real_text(std::size_t n, bool r2c, bool strided) const {
  const std::string kind = r2c ? "r2c" : "c2r";
  const real_route how = route(n, r2c, strided);
  const std::string half =
      " through a half-length complex chain of " + std::to_string(n / 2) + ", radices " +
      (how == real_route::fused ? radix_text(n / 2, real_radices(n / 2, r2c)) : radix_text(n / 2));
  switch (how) {
  case real_route::full:
    return kind + " through a full-length complex chain (n odd), radices " + radix_text(n);
  case real_route::strided:
    return kind + half + " in scratch, the pairing a separate pass (strided line)";
  case real_route::pairing_only:
    return kind + " of 2 points: the pairing pass alone";
  case real_route::leaf:
    return kind + " in registers: a leaf of " + std::to_string(n / 2) + " with the pairing fused";
  case real_route::fused:
    return kind + half + ", the pairing fused into its " + (r2c ? "last" : "first") + " pass";
  case real_route::separate:
    break;
  }
  return kind + half + (r2c ? ", then a pairing pass" : ", after a pairing pass");
}

// -i w_n^k = w_n^(k + n/4), the pairing's twiddle (passes/real.inl): an exact rearrangement
// of a root-table entry, which also holds when 4 does not divide n.
template <class T> std::complex<T> chain_builder<T>::pairing_root(std::size_t n, std::size_t k) {
  const std::complex<T> w = roots(n)[k];
  return {w.imag(), -w.real()};
}

// The pairing twiddles of bins k = 1 .. (n/2 - 1) / 2, padded with ones to whole registers,
// per-lane storage: the separate pairing passes' table.
template <class T> const T* chain_builder<T>::pairing_twiddles(std::size_t n) {
  const std::size_t pairs = (n / 2 - 1) / 2;
  if (pairs == 0) {
    return nullptr;
  }
  std::vector<std::complex<T>> t(blocks(pairs) * kernels_.complex_lanes, std::complex<T>(1, 0));
  for (std::size_t k = 1; k <= pairs; ++k) {
    t[k - 1] = pairing_root(n, k);
  }
  return pack(t, false);
}

// The pairing twiddles of bins k = 0 .. n/4, broadcast storage: the leaves' table.
template <class T> const T* chain_builder<T>::leaf_pairing_twiddles(std::size_t n) {
  std::vector<std::complex<T>> t(n / 4 + 1);
  for (std::size_t k = 0; k < t.size(); ++k) {
    t[k] = pairing_root(n, k);
  }
  return pack(t, true);
}

// The tables of the depth-first first pass with the c2r pairing (depth.inl, run_in_c2r) of a row
// of n whose half length's first pass has radix r: the pairing twiddles v_e = w_n^(e + n/4) of
// e = 0 .. n/(2r) - 1, per-element storage, and the roots w_n^(n/(2r) j) for j = 1 .. r - 1,
// broadcast storage; v_e w_n^(n/(2r) j) is the pairing twiddle of e + n/(2r) j, so the pair
// covers every element of the half length, the upper half with the same expression.
template <class T>
std::pair<const T*, const T*> chain_builder<T>::depth_c2r_twiddles(std::size_t n, std::size_t r) {
  const std::size_t leg = n / 2 / r;
  std::vector<std::complex<T>> v(leg);
  for (std::size_t e = 0; e < leg; ++e) {
    v[e] = pairing_root(n, e);
  }
  const auto& w = roots(n);
  std::vector<std::complex<T>> om(r - 1);
  for (std::size_t j = 1; j < r; ++j) {
    om[j - 1] = w[leg * j];
  }
  const T* v_stored = pack(v, false);
  return {v_stored, pack(om, true)};
}

template <class T>
void chain_builder<T>::add_pairing(bool r2c, std::size_t n, endpoint src, endpoint dst,
                                   double scale, std::vector<stage>& stages) {
  const pass_kernels<T>& k = kernels_.passes;
  const std::size_t scaled = scale != 1.0 ? 1 : 0;
  const std::size_t units = 1 + blocks((n / 2 - 1) / 2);
  add_edge(r2c ? k.r2c_post[scaled] : k.c2r_pre[scaled], r2c ? "r2c pairing" : "c2r pairing", n,
           units, src, dst, pairing_twiddles(n), scale, stages);
  stages.back().chunks = chunks_of(units, 4 * kernels_.complex_lanes);
}

template <class T>
std::size_t chain_builder<T>::r2c_line(std::size_t n, endpoint src, endpoint dst, bool aliased,
                                       double scale, std::vector<stage>& stages) {
  line_scratch_ = 0;
  const pass_kernels<T>& k = kernels_.passes;
  const std::size_t scaled = scale != 1.0 ? 1 : 0;
  const std::size_t c = kernels_.complex_lanes;
  const std::size_t h = n / 2;
  // The reals of an even line of stride 1 are read in place as h complex values.
  switch (route(n, true, src.stride != 1 || dst.stride != 1)) {
  case real_route::strided: {
    add_edge(k.r2c_pack_pairs, "r2c pack pairs", n, h, src, {slot::a, 1}, nullptr, 1.0, stages);
    const slot z = scratch_chain(h, -1, slot::a, stages);
    add_edge(k.r2c_post_strided[scaled], "r2c pairing", n, h / 2 + 1, {z, 1}, dst, edge_twiddles(n),
             scale, stages);
    return std::max(h, line_scratch_);
  }
  case real_route::full: {
    add_edge(k.r2c_pack, "r2c pack", n, n, src, {slot::a, 1}, nullptr, 1.0, stages);
    const slot z = scratch_chain(n, -1, slot::a, stages);
    add_edge(k.r2c_store[scaled], "r2c store half", n, n / 2 + 1, {z, 1}, dst, nullptr, scale,
             stages);
    return std::max(n, line_scratch_);
  }
  case real_route::pairing_only:
    add_pairing(true, n, src, dst, scale, stages);
    return 0;
  case real_route::leaf: {
    stage st;
    st.run = k.codelet[codelet_slot(h)]->r2c_leaf[scaled];
    st.args.radix = h;
    st.args.n = h;
    st.args.twiddles = leaf_pairing_twiddles(n);
    st.args.scale = scale;
    st.src = src.where;
    st.dst = dst.where;
    st.units = 1;
    st.chunks = 1;
    st.name = "r2c leaf radix-" + std::to_string(h) + " with the pairing" +
              (scale != 1.0 ? " (out: scale)" : "");
    stages.push_back(std::move(st));
    return 0;
  }
  case real_route::fused: {
    const std::vector<std::size_t> rad = real_radices(h, true);
    const std::size_t radix = rad.back();
    const std::size_t s = h / radix;
    endpoint at = src;
    std::size_t copies = 1;
    for (std::size_t i = 0; i + 1 < rad.size(); ++i) {
      const endpoint next{.where = other_scratch(at.where), .stride = 1};
      add_pass(h, rad, i, copies, -1, at, next, line_edges(), stages);
      at = next;
      copies *= rad[i];
    }
    // The pairing twiddles of the bins q + s r: per-lane rows r for the copies q = 1 ..
    // (s - 1) / 2; broadcast, copy 0's, then copy s/2's (s even).
    const std::size_t qm = (s - 1) / 2;
    const std::size_t qpad = blocks(qm) * c;
    std::vector<std::complex<T>> lanes(radix * qpad, std::complex<T>(1, 0));
    std::vector<std::complex<T>> ends(2 * radix, std::complex<T>(1, 0));
    for (std::size_t r = 0; r < radix; ++r) {
      for (std::size_t q = 1; q <= qm; ++q) {
        lanes[r * qpad + q - 1] = pairing_root(n, q + s * r);
      }
      ends[r] = pairing_root(n, s * r);
      if (s % 2 == 0) {
        ends[radix + r] = pairing_root(n, s / 2 + s * r);
      }
    }
    stage st;
    st.run = k.codelet[codelet_slot(radix)]->r2c_last[scaled];
    st.args.radix = radix;
    st.args.m = 1;
    st.args.s = s;
    st.args.n = h;
    st.args.twiddles = qpad != 0 ? pack(lanes, false) : nullptr;
    st.args.roots = pack(ends, true);
    st.args.scale = scale;
    st.src = at.where;
    st.dst = dst.where;
    st.units = 1 + blocks(qm);
    st.chunks = chunks_of(st.units, 2 * radix * c);
    st.name = "lane radix-" + std::to_string(radix) + " forward (m 1, s " + std::to_string(s) +
              "; copies q and s - q together, with the r2c pairing" +
              (scale != 1.0 ? "; out: scale)" : ")");
    stages.push_back(std::move(st));
    return std::max(h, line_scratch_);
  }
  case real_route::separate:
    break;
  }
  // The complex transform of h writes the output's first h bins; the pairing then runs in
  // place there.
  const std::size_t need = c2c_line(h, -1, src, dst, aliased, line_edges(), stages);
  add_pairing(true, n, dst, dst, scale, stages);
  return need;
}

template <class T>
std::size_t chain_builder<T>::c2r_line(std::size_t n, endpoint src, endpoint dst, double scale,
                                       std::vector<stage>& stages, bool packed) {
  line_scratch_ = 0;
  const pass_kernels<T>& k = kernels_.passes;
  const std::size_t scaled = scale != 1.0 ? 1 : 0;
  const std::size_t c = kernels_.complex_lanes;
  const std::size_t h = n / 2;
  line_edges edges;
  if (scale != 1.0) {
    edges.out = edge_op::scale;
    edges.scale = scale;
  }
  switch (route(n, false, src.stride != 1 || dst.stride != 1)) {
  case real_route::strided: {
    add_edge(k.c2r_pre_strided, "c2r pairing", n, h / 2 + 1, src, {slot::a, 1}, edge_twiddles(n),
             1.0, stages);
    const slot z = scratch_chain(h, +1, slot::a, stages);
    add_edge(k.c2r_unpack_pairs[scaled], "c2r unpack pairs", n, h, {z, 1}, dst, nullptr, scale,
             stages);
    return std::max(h, line_scratch_);
  }
  case real_route::full: {
    add_edge(k.c2r_expand, "c2r expand", n, n / 2 + 1, src, {slot::a, 1}, nullptr, 1.0, stages);
    const slot z = scratch_chain(n, +1, slot::a, stages);
    add_edge(k.c2r_store[scaled], "c2r store real", n, n, {z, 1}, dst, nullptr, scale, stages);
    return std::max(n, line_scratch_);
  }
  case real_route::pairing_only:
    add_pairing(false, n, src, dst, scale, stages);
    stages.back().args.packed = packed;
    return 0;
  case real_route::leaf: {
    stage st;
    st.run = k.codelet[codelet_slot(h)]->c2r_leaf[scaled];
    st.args.packed = packed;
    st.args.radix = h;
    st.args.n = h;
    st.args.twiddles = leaf_pairing_twiddles(n);
    st.args.scale = scale;
    st.src = src.where;
    st.dst = dst.where;
    st.units = 1;
    st.chunks = 1;
    st.name = "c2r leaf radix-" + std::to_string(h) + " with the pairing" +
              (scale != 1.0 ? " (out: scale)" : "");
    stages.push_back(std::move(st));
    return 0;
  }
  case real_route::fused: {
    const std::vector<std::size_t> rad = real_radices(h, false);
    const std::size_t radix = rad.front();
    const std::size_t m = h / radix;
    // Per-lane rows of the groups p = 1 .. (m - 1) / 2: their twiddles w_h^(p k), those of
    // the groups m - p, and the pairing twiddles of the bins p + m j. Broadcast: the pairing
    // twiddles of groups 0 and m/2 (m even), then group m/2's w_h^((m/2) k).
    const auto& wh = roots(h);
    const std::size_t pm = (m - 1) / 2;
    const std::size_t ppad = blocks(pm) * c;
    std::vector<std::complex<T>> lanes((3 * radix - 2) * ppad, std::complex<T>(1, 0));
    std::vector<std::complex<T>> ends(3 * radix - 1, std::complex<T>(1, 0));
    for (std::size_t i = 0; i < pm; ++i) {
      for (std::size_t r = 1; r < radix; ++r) {
        lanes[(r - 1) * ppad + i] = wh[(1 + i) * r];
        lanes[(radix - 1 + r - 1) * ppad + i] = wh[(m - 1 - i) * r];
      }
      for (std::size_t j = 0; j < radix; ++j) {
        lanes[(2 * radix - 2 + j) * ppad + i] = pairing_root(n, 1 + i + m * j);
      }
    }
    for (std::size_t j = 0; j < radix; ++j) {
      ends[j] = pairing_root(n, m * j);
      if (m % 2 == 0) {
        ends[radix + j] = pairing_root(n, m / 2 + m * j);
      }
    }
    for (std::size_t r = 1; r < radix && m % 2 == 0; ++r) {
      ends[2 * radix + r - 1] = wh[m / 2 * r];
    }
    stage st;
    st.run = k.codelet[codelet_slot(radix)]->c2r_first;
    st.args.packed = packed;
    st.args.radix = radix;
    st.args.m = m;
    st.args.s = 1;
    st.args.n = h;
    st.args.twiddles = ppad != 0 ? pack(lanes, false) : nullptr;
    st.args.roots = pack(ends, true);
    st.src = src.where;
    st.dst = slot::a;
    st.units = 1 + blocks(pm);
    st.chunks = chunks_of(st.units, 2 * radix * c);
    st.name = "grouped radix-" + std::to_string(radix) + " backward (m " + std::to_string(m) +
              ", s 1; groups p and m - p together, with the c2r pairing)";
    stages.push_back(std::move(st));
    chain_tail(h, rad, 1, radix, +1, {slot::a, 1}, dst, edges, stages);
    return std::max(h, line_scratch_);
  }
  case real_route::separate:
    break;
  }
  // The pairing writes Z to the output's memory (h complex values: the n reals), where the
  // complex transform then runs in place; the input is preserved (unless it is that memory).
  add_pairing(false, n, src, dst, 1.0, stages);
  stages.back().args.packed = packed;
  return c2c_line(h, +1, dst, dst, true, edges, stages);
}

template class chain_builder<double>;
template class chain_builder<float>;

} // namespace fftn::inline v0::detail
