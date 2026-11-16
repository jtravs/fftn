// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "strategies/grid.hpp"

#include "core/program.hpp"
#include "core/twiddles.hpp"
#include "isa/kernel_table.hpp"
#include "passes/pass_args.hpp"
#include "strategies/chain.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

// No prime factor above 13: a chain of codelet passes (radices 2 to 16) covers it.
bool codelet_smooth(std::size_t n) {
  for (const std::size_t p : {2U, 3U, 5U, 7U, 11U, 13U}) {
    while (n % p == 0) {
      n /= p;
    }
  }
  return n == 1;
}

const char* direction_name(int sign) { return sign < 0 ? "forward" : "backward"; }

slot other_scratch(slot s) { return s == slot::a ? slot::b : slot::a; }

std::string radix_text(const std::vector<std::size_t>& radices) {
  std::string text;
  for (const std::size_t r : radices) {
    text += (text.empty() ? "" : " x ") + std::to_string(r);
  }
  return text;
}

std::size_t longest_factor(const grid_split& g) {
  std::size_t longest = 0;
  for (std::size_t i = 0; i < g.phases; ++i) {
    longest = std::max(longest, g.factors.at(i));
  }
  return longest;
}

// n / N1: the columns of phase 1.
std::size_t first_columns(const grid_split& g) {
  std::size_t rest = 1;
  for (std::size_t i = 1; i < g.phases; ++i) {
    rest *= g.factors.at(i);
  }
  return rest;
}

// The block of a split: the widest power of two from `least` to max_grid_block dividing N1
// and n / N1 with block x max Ni within the budget, or `least`.
std::size_t widest_block(const grid_split& g, std::size_t least, std::size_t budget) {
  const std::size_t longest = longest_factor(g);
  std::size_t block = least;
  for (std::size_t b = 2 * least; b <= max_grid_block; b *= 2) {
    if (g.factors[0] % b != 0 || first_columns(g) % b != 0 || b * longest > budget) {
      break;
    }
    block = b;
  }
  return block;
}

// Whether the factors of g can be a grid with blocks of `least`.
bool runnable(const grid_split& g, std::size_t lanes, std::size_t least) {
  for (std::size_t i = 0; i < g.phases; ++i) {
    if (g.factors.at(i) < 2 || !codelet_smooth(g.factors.at(i))) {
      return false;
    }
  }
  return g.factors[0] % least == 0 && g.factors[0] % lanes == 0 && first_columns(g) % least == 0;
}

std::size_t shortest_factor(const grid_split& g) {
  std::size_t shortest = g.factors[0];
  for (std::size_t i = 1; i < g.phases; ++i) {
    shortest = std::min(shortest, g.factors.at(i));
  }
  return shortest;
}

// Better: a wider block, then a shorter longest factor, then a longer shortest factor (the
// more even), then a shorter first factor (X12: 256 x 512 against 512 x 256, 1.22 and 1.60
// times FFTW's time).
bool better(const grid_split& a, const grid_split& b) {
  if (b.phases == 0) {
    return true;
  }
  if (a.block != b.block) {
    return a.block > b.block;
  }
  if (longest_factor(a) != longest_factor(b)) {
    return longest_factor(a) < longest_factor(b);
  }
  if (shortest_factor(a) != shortest_factor(b)) {
    return shortest_factor(a) > shortest_factor(b);
  }
  return a.factors[0] < b.factors[0];
}

// The radices of a phase's chain (all lane passes, s >= block >= C, so no pass is grouped
// and none has tails; planner v1's tables price the chain's grouped first pass and its
// tails, which a phase has not): the fewest passes, then the radices closest to equal,
// ascending (X12 measured 16 x 16 against v1's 2 x 8 x 16 for 256). Radices up to
// `largest` that have a codelet (grid_builder's phase_radix).
std::vector<std::size_t> phase_radices(std::size_t n, std::size_t largest) {
  // best[d]: the fewest passes to d, and the largest spread (largest / smallest radix) on
  // the way, over the divisors of n; a shortest path as in planner v1.
  struct path {
    std::size_t passes = 0;
    std::size_t low = 0; // smallest radix so far (0: none)
    std::size_t high = 0;
    std::size_t last = 0; // radix of the last pass
    std::size_t from = 0;
  };
  std::vector<std::size_t> ds;
  for (std::size_t d = 1; d <= n; ++d) {
    if (n % d == 0) {
      ds.push_back(d);
    }
  }
  std::vector<path> best(ds.size());
  const auto at = [&](std::size_t d) {
    return static_cast<std::size_t>(std::ranges::lower_bound(ds, d) - ds.begin());
  };
  std::vector<bool> reached(ds.size(), false);
  reached[0] = true;
  for (std::size_t i = 0; i < ds.size(); ++i) {
    if (!reached[i]) {
      continue;
    }
    for (std::size_t r = largest; r >= 2; --r) {
      if ((n / ds[i]) % r != 0 || !has_codelet(r)) {
        continue;
      }
      const std::size_t j = at(ds[i] * r);
      const path cand{
          .passes = best[i].passes + 1,
          .low = best[i].low == 0 ? r : std::min(best[i].low, r),
          .high = std::max(best[i].high, r),
          .last = r,
          .from = i,
      };
      const bool wins =
          !reached[j] || cand.passes < best[j].passes ||
          (cand.passes == best[j].passes && cand.high * best[j].low < best[j].high * cand.low);
      if (wins) {
        best[j] = cand;
        reached[j] = true;
      }
    }
  }
  std::vector<std::size_t> rad;
  for (std::size_t j = ds.size() - 1; j != 0; j = best[j].from) {
    rad.push_back(best[j].last);
  }
  std::ranges::sort(rad);
  return rad;
}

// Whether a phase with these radices reads and writes its rows directly (grid_fuse_radix).
bool fusible(const std::vector<std::size_t>& rad) {
  return std::ranges::all_of(rad, [](std::size_t r) { return r <= grid_fuse_radix; });
}

std::vector<std::size_t> divisors(std::size_t n) {
  std::vector<std::size_t> low;
  std::vector<std::size_t> high;
  for (std::size_t d = 1; d * d <= n; ++d) {
    if (n % d == 0) {
      low.push_back(d);
      if (d != n / d) {
        high.push_back(n / d);
      }
    }
  }
  low.insert(low.end(), high.rbegin(), high.rend());
  return low;
}

} // namespace

grid_split choose_grid(std::size_t n, std::size_t lanes, std::size_t budget, bool three_phases,
                       std::size_t forced_block, const std::vector<std::size_t>& forced) {
  // Two columns at least: a block of one would gather single elements a row apart.
  const std::size_t least = forced_block != 0 ? forced_block : std::max<std::size_t>(lanes, 2);
  if (lanes == 0 || least % lanes != 0 || forced.size() == 1 || forced.size() > max_grid_phases) {
    return {};
  }
  const auto finish = [&](grid_split g) {
    if (!runnable(g, lanes, least)) {
      return grid_split{};
    }
    g.block = forced_block != 0 ? forced_block : widest_block(g, least, budget);
    return g;
  };
  if (!forced.empty()) {
    grid_split g{.phases = forced.size()};
    std::size_t product = 1;
    for (std::size_t i = 0; i < forced.size(); ++i) {
      g.factors.at(i) = forced[i];
      product *= forced[i];
    }
    return product == n ? finish(g) : grid_split{};
  }
  const std::vector<std::size_t> ds = divisors(n);
  grid_split two;
  grid_split three;
  for (const std::size_t n1 : ds) {
    if (n1 < 2 || n1 == n) {
      continue;
    }
    const grid_split g2 = finish({.phases = 2, .factors = {n1, n / n1, 0}});
    if (g2.phases != 0 && better(g2, two)) {
      two = g2;
    }
    for (const std::size_t n2 : ds) {
      if (n2 < 2 || (n / n1) % n2 != 0 || n / n1 == n2) {
        continue;
      }
      const grid_split g3 = finish({.phases = 3, .factors = {n1, n2, n / n1 / n2}});
      if (g3.phases != 0 && better(g3, three)) {
        three = g3;
      }
    }
  }
  if (two.phases != 0 && (!three_phases || three.phases == 0)) {
    return two;
  }
  if (three.phases != 0 && forced_block == 0) {
    // Three phases are for rows far beyond L2, where X12 measured wide rows in the gathers
    // and scatters worth more than keeping a block's chain within L1.
    three.block = widest_block(three, least, 2 * budget);
  }
  return three;
}

std::size_t grid_pad(const grid_split& g, std::size_t element_bytes, std::size_t period) {
  constexpr std::size_t line = 64;
  constexpr std::size_t most_blocks = 8;
  if (g.phases < 2 || g.block == 0 || period < 2 * line) {
    return 0;
  }
  const std::size_t n2 = g.factors[1];
  const std::size_t stride = first_columns(g) / n2 * g.factors[0];
  if (g.phases == 3 && g.factors[2] % g.block != 0) {
    return 0;
  }
  const std::size_t sets = period / line;
  const std::size_t row_bytes = g.block * element_bytes;
  // The most of the rows' lines that share one set.
  std::vector<std::size_t> set_of;
  const auto load = [&](std::size_t pad) {
    set_of.clear();
    for (std::size_t r = 0; r < n2; ++r) {
      const std::size_t start = r * (stride + pad) * element_bytes;
      for (std::size_t l = start / line; l <= (start + row_bytes - 1) / line; ++l) {
        set_of.push_back(l % sets);
      }
    }
    std::ranges::sort(set_of);
    std::size_t most = 0;
    for (std::size_t i = 0; i < set_of.size();) {
      std::size_t j = i;
      while (j < set_of.size() && set_of[j] == set_of[i]) {
        ++j;
      }
      most = std::max(most, j - i);
      i = j;
    }
    return most;
  };
  std::size_t best = 0;
  std::size_t best_load = load(0);
  for (std::size_t k = 1; k <= most_blocks; ++k) {
    const std::size_t l = load(k * g.block);
    if (l < best_load) {
      best = k * g.block;
      best_load = l;
    }
  }
  return best;
}

template <class T>
grid_builder<T>::grid_builder(const kernels_of<T>& kernels, chain_builder<T>& chain,
                              plan_tables<T>& tables, bool fused, std::size_t phase_radix)
    : kernels_(kernels), chain_(chain), tables_(tables), fused_(fused),
      phase_radix_(phase_radix != 0 ? phase_radix : default_grid_phase_radix) {}

template <class T>
void grid_builder<T>::add_copy(const char* what, pass_fn run, std::size_t rows, std::size_t cols,
                               slot src, std::ptrdiff_t src_stride, slot dst,
                               std::ptrdiff_t dst_stride, double scale, const T* twiddles,
                               program& p) {
  stage st;
  st.run = run;
  st.args.s = cols;
  st.args.n = rows * cols;
  st.args.scale = scale;
  st.args.src_stride = src_stride;
  st.args.dst_stride = dst_stride;
  st.args.twiddles = twiddles;
  st.moving_twiddles = twiddles != nullptr;
  st.src = src;
  st.dst = dst;
  st.units = rows;
  st.chunks = 1;
  std::string scaled;
  if (scale != 1.0) {
    scaled = std::string(what) == "gather" ? "; in: scale" : "; out: scale";
  }
  st.name = std::string("grid ") + what + " (" + std::to_string(cols) + " columns x " +
            std::to_string(rows) + " rows, row strides " + std::to_string(src_stride) + " -> " +
            std::to_string(dst_stride) + scaled + ")";
  p.records.emplace_back(std::move(st));
}

// A phase's radices: forced ones, or the phase rule's.
template <class T> std::vector<std::size_t> grid_builder<T>::phase_radix_list(std::size_t n) const {
  return chain_.forces(n) ? chain_.radices(n) : phase_radices(n, phase_radix_);
}

template <class T>
slot grid_builder<T>::phase_chain(std::size_t n, const std::vector<std::size_t>& rad,
                                  std::size_t cols, int sign, const side_of& from,
                                  const side_of& to, double in_scale, double out_scale,
                                  const T* row_tw, program& p) {
  const std::size_t c = kernels_.complex_lanes;
  const std::size_t d = sign < 0 ? 0 : 1;
  std::size_t s = cols;
  slot at = from.where;
  for (std::size_t i = 0; i < rad.size(); ++i) {
    const std::size_t radix = rad[i];
    const std::size_t m = n * cols / (radix * s);
    const bool first = i == 0;
    const bool last = i + 1 == rad.size();
    const bool big_in = first && from.big;
    const bool big_out = last && to.big;
    const slot next =
        big_out ? to.where : other_scratch(at == slot::a || at == slot::b ? at : slot::b);
    stage st;
    st.args.radix = radix;
    st.args.m = m;
    st.args.s = s;
    st.args.n = n * cols;
    st.units = m * (s / c);
    st.chunks = 1;
    if (m > 1) {
      st.args.twiddles = chain_.lane_twiddles(n * cols, radix, s);
    }
    std::string ends;
    if (big_in || big_out) {
      phase_variant v = phase_plain;
      if (m > 1) {
        v = big_in && in_scale != 1.0 ? phase_twiddled_in_scale : phase_twiddled;
      } else if (big_in && in_scale != 1.0) {
        v = phase_plain_in_scale;
      } else if (big_out && row_tw != nullptr) {
        v = phase_plain_row_twiddles;
        st.args.twiddles = row_tw;
        st.moving_twiddles = true;
      } else if (big_out && out_scale != 1.0) {
        v = phase_plain_out_scale;
      }
      st.run = kernels_.passes.codelet[codelet_slot(radix)]->phase[d][v];
      // The scale of the variant chosen: a single pass is both a phase's read and its write.
      st.args.scale = 1.0;
      if (v == phase_twiddled_in_scale || v == phase_plain_in_scale) {
        st.args.scale = in_scale;
      } else if (v == phase_plain_out_scale) {
        st.args.scale = out_scale;
      }
      st.args.src_stride = big_in ? from.row : static_cast<std::ptrdiff_t>(s);
      st.args.src_cols = big_in ? cols : 0;
      st.args.dst_stride =
          big_out ? to.row * static_cast<std::ptrdiff_t>(s / cols) : static_cast<std::ptrdiff_t>(s);
      st.args.dst_cols = big_out ? cols : 0;
      st.args.dst_qstride = big_out ? to.row : 0;
      if (big_in) {
        ends += std::string("; reads rows ") + std::to_string(from.row) + " apart" +
                (in_scale != 1.0 ? ", in: scale" : "");
      }
      if (big_out) {
        ends += std::string("; writes rows ") + std::to_string(to.row) + " apart";
        if (row_tw != nullptr) {
          ends += ", twiddled";
        } else if (out_scale != 1.0) {
          ends += ", out: scale";
        }
      }
    } else {
      st.run = m > 1 ? kernels_.passes.codelet[codelet_slot(radix)]->lane[d]
                     : kernels_.passes.codelet[codelet_slot(radix)]->last[d][0];
    }
    st.src = big_in ? from.where : at;
    st.dst = next;
    st.name = "lane radix-" + std::to_string(radix) + " " + direction_name(sign) + " (m " +
              std::to_string(m) + ", s " + std::to_string(s) + ends + ")";
    p.records.emplace_back(std::move(st));
    at = next;
    s *= radix;
  }
  return at;
}

// Phase 1's twiddles w_n^(c k1), c = c0 + q, as two factors, each root computed alone
// (no table of the n roots) (twiddle_transpose): w_n^(q k1)
// for q < block, in the order of a block's tiles (per tile of C columns, per tile of C rows,
// C rows of C columns), and w_n^(c0 k1) per block, its N1 rows; plain interleaved.
template <class T>
std::pair<const T*, const T*> grid_builder<T>::first_twiddles(const grid_split& g) {
  const std::size_t n1 = g.factors[0];
  const std::size_t m = first_columns(g);
  const std::size_t n = n1 * m;
  const std::size_t lanes = kernels_.complex_lanes;
  auto& local = tables_.emplace_back(2 * g.block * n1);
  std::size_t e = 0;
  for (std::size_t q0 = 0; q0 < g.block; q0 += lanes) {
    for (std::size_t k0 = 0; k0 < n1; k0 += lanes) {
      for (std::size_t i = 0; i < lanes; ++i) {
        for (std::size_t l = 0; l < lanes; ++l) {
          const std::complex<T> v = unit_root<T>(-1, (q0 + l) * (k0 + i) % n, n);
          local[e++] = v.real();
          local[e++] = v.imag();
        }
      }
    }
  }
  // Taken before the next table is appended: that may move the outer vector (the inner
  // buffers stay where they are).
  const T* local_data = local.data();
  auto& rows = tables_.emplace_back(2 * (m / g.block) * n1);
  e = 0;
  for (std::size_t c0 = 0; c0 < m; c0 += g.block) {
    for (std::size_t k = 0; k < n1; ++k) {
      const std::complex<T> v = unit_root<T>(-1, c0 * k % n, n);
      rows[e++] = v.real();
      rows[e++] = v.imag();
    }
  }
  return {local_data, rows.data()};
}

// A later phase's w_(Ni M)^(c k): per column c < M, the Ni rows k (plain interleaved).
template <class T> const T* grid_builder<T>::row_twiddles(std::size_t ni, std::size_t m) {
  auto& t = tables_.emplace_back(2 * ni * m);
  std::size_t e = 0;
  for (std::size_t c = 0; c < m; ++c) {
    for (std::size_t k = 0; k < ni; ++k) {
      const std::complex<T> v = unit_root<T>(-1, c * k, ni * m);
      t[e++] = v.real();
      t[e++] = v.imag();
    }
  }
  return t.data();
}

// The factors of c2r_pair_gather's f_j = i w^-j (w = e^(-2 pi i / n), j = c + m1 r < n / 2):
// i w^-c for c < m1 and w^-(m1 r) for r < n1, plain interleaved.
template <class T>
std::pair<const T*, const T*> grid_builder<T>::c2r_pair_factors(std::size_t n1, std::size_t m1,
                                                                std::size_t n) {
  auto& columns = tables_.emplace_back(2 * m1);
  for (std::size_t q = 0; q < m1; ++q) {
    const std::complex<T> v = unit_root<T>(+1, q, n);
    columns[2 * q] = -v.imag();
    columns[2 * q + 1] = v.real();
  }
  // Taken before the next table is appended (first_twiddles).
  const T* columns_data = columns.data();
  auto& rows = tables_.emplace_back(2 * n1);
  for (std::size_t r = 0; r < n1; ++r) {
    const std::complex<T> v = unit_root<T>(+1, m1 * r, n);
    rows[2 * r] = v.real();
    rows[2 * r + 1] = v.imag();
  }
  return {columns_data, rows.data()};
}

template <class T> bool grid_builder<T>::uses_block(const grid_split& g, bool in_place) noexcept {
  return g.phases == 3 || in_place;
}

template <class T> std::size_t grid_builder<T>::block_elements(const grid_split& g) noexcept {
  return g.factors[0] * first_columns(g) + g.factors[1] * g.pad;
}

template <class T>
std::size_t grid_builder<T>::c2c_row(const grid_split& g, int sign, bool in_place, double scale,
                                     bool scale_on_input, program& p, std::size_t c2r_n) {
  const double in_scale = scale_on_input ? scale : 1.0;
  const double out_scale = scale_on_input ? 1.0 : scale;
  const std::size_t b = g.block;
  const std::size_t c = kernels_.complex_lanes;
  const std::size_t d = sign < 0 ? 0 : 1;
  const auto cb = static_cast<std::ptrdiff_t>(2 * sizeof(T));
  const auto blk = static_cast<std::ptrdiff_t>(b);
  const auto index = [](slot s) -> std::size_t {
    if (s == slot::in) {
      return 0;
    }
    return s == slot::out ? 1 : 2;
  };
  // The array each phase writes (buf[i] for phase i, buf[0] the input): the last phase
  // writes the output, the positions it read; every other phase writes an array other
  // than the one it reads, and phase 1 never the input's memory.
  std::array<slot, max_grid_phases + 1> buf{slot::in, slot::out, slot::out, slot::out};
  if (uses_block(g, in_place)) {
    buf[1] = slot::block;
  }
  const std::size_t n1 = g.factors[0];
  const std::size_t m1 = first_columns(g);
  const std::size_t n = n1 * m1;

  // Phase 1: blocks of the input's columns, written transposed. With a padded
  // intermediate (grid_pad), each of phase 2's rows holds m2 of phase 1's columns and then
  // the pad: three phases write each such row's blocks in a loop of their own; two (m2 = 1)
  // write every column of a block as one padded row.
  const std::size_t pad = g.pad;
  const std::size_t m2 = m1 / g.factors[1];
  const bool padded_rows = pad != 0 && m2 > 1;
  const std::size_t row1 = m2 == 1 ? n1 + pad : n1;
  const std::size_t first_loop = p.records.size();
  if (padded_rows) {
    loop rows{.kind = loop_kind::lines, .count = g.factors[1]};
    rows.step[index(buf[0])] = static_cast<std::ptrdiff_t>(m2) * cb;
    rows.step[index(buf[1])] = static_cast<std::ptrdiff_t>(n1 * m2 + pad) * cb;
    rows.step[3] = static_cast<std::ptrdiff_t>(m2 / b * n1) * cb;
    p.records.emplace_back(rows);
  }
  const std::size_t block_loop = p.records.size();
  loop l1{.kind = loop_kind::lines, .count = (padded_rows ? m2 : m1) / b};
  l1.step[index(buf[0])] = blk * cb;
  l1.step[index(buf[1])] = blk * static_cast<std::ptrdiff_t>(row1) * cb;
  l1.step[3] = static_cast<std::ptrdiff_t>(n1) * cb;
  p.records.emplace_back(l1);
  // A phase reads its rows in its first pass and (after phase 1) writes them in its last,
  // instead of through a gather and a scatter, when every radix of its chain is at most
  // grid_fuse_radix. A pass touches R rows at once, which lie a power of two apart for most
  // lengths and so share cache sets: with more rows than the sets' ways each row's lines are
  // evicted before the next unit, at the next columns, reuses them (X12: radix 16 at the
  // ends doubled the time), and fusing only one end of a phase gained nothing (X12, run
  // 20261114-131752-7e99), so a phase fuses both ends or neither.
  const std::vector<std::size_t> rad1 = phase_radix_list(n1);
  // A c2r row's phase 1 always gathers: its gather forms the paired spectrum (X29).
  const bool fuse1 = c2r_n == 0 && fused_ && fusible(rad1);
  const auto [local, rows] = first_twiddles(g);
  if (c2r_n != 0) {
    add_copy("gather", kernels_.passes.c2r_pair_gather, n1, b, buf[0],
             static_cast<std::ptrdiff_t>(m1), slot::a, blk, 1.0, rows, p);
    stage& st = std::get<stage>(p.records.back());
    const auto [columns, row_factors] = c2r_pair_factors(n1, m1, c2r_n);
    st.args.roots = rows;
    st.args.in_factor = columns;
    st.args.out_factor = row_factors;
    st.args.m = n1;
    st.args.n = c2r_n;
    st.name += "; c2r pairing";
  } else if (!fuse1) {
    add_copy("gather", kernels_.passes.block_copy[in_scale != 1.0 ? 1 : 0], n1, b, buf[0],
             static_cast<std::ptrdiff_t>(m1), slot::a, blk, in_scale, nullptr, p);
  }
  const slot z1 = phase_chain(
      n1, rad1, b, sign,
      fuse1 ? side_of{.where = buf[0], .big = true, .row = static_cast<std::ptrdiff_t>(m1)}
            : side_of{.where = slot::a},
      side_of{.where = slot::a}, fuse1 ? in_scale : 1.0, 1.0, nullptr, p);
  stage tt;
  tt.run = kernels_.passes.twiddle_transpose[d];
  tt.args.roots = local;
  tt.args.twiddles = rows;
  tt.args.m = n1;
  tt.args.dst_stride = static_cast<std::ptrdiff_t>(row1);
  tt.args.s = b;
  tt.args.n = n1 * b;
  tt.moving_twiddles = true;
  tt.src = z1;
  tt.dst = buf[1];
  tt.units = (b / c) * (n1 / c);
  tt.chunks = 1;
  tt.name = std::string("grid twiddled transpose ") + direction_name(sign) + " (" +
            std::to_string(b) + " columns x " + std::to_string(n1) + " rows -> rows of " +
            std::to_string(n1) + ")";
  p.records.emplace_back(std::move(tt));
  std::get<loop>(p.records[first_loop]).body_end = static_cast<std::uint32_t>(p.records.size());
  std::get<loop>(p.records[block_loop]).body_end = static_cast<std::uint32_t>(p.records.size());

  // Later phases: blocks of b adjacent l < L at each column c < M.
  std::size_t l_count = n1;
  for (std::size_t i = 1; i < g.phases; ++i) {
    const std::size_t ni = g.factors.at(i);
    const std::size_t m = n / (l_count * ni);
    const auto big_l = static_cast<std::ptrdiff_t>(l_count);
    // The rows this phase reads: the padded intermediate's for phase 2.
    const std::ptrdiff_t src_row =
        big_l * static_cast<std::ptrdiff_t>(m) + static_cast<std::ptrdiff_t>(i == 1 ? pad : 0);
    const slot src = buf.at(i);
    const slot dst = buf.at(i + 1);
    const std::size_t outer = p.records.size();
    if (m > 1) {
      loop lc{.kind = loop_kind::lines, .count = m};
      lc.step.at(index(src)) = big_l * cb;
      lc.step.at(index(dst)) = big_l * static_cast<std::ptrdiff_t>(ni) * cb;
      lc.step[3] = static_cast<std::ptrdiff_t>(ni) * cb;
      p.records.emplace_back(lc);
    }
    const std::size_t inner = p.records.size();
    loop lines{.kind = loop_kind::lines, .count = l_count / b};
    lines.step.at(index(src)) = blk * cb;
    lines.step.at(index(dst)) = blk * cb;
    p.records.emplace_back(lines);
    const std::vector<std::size_t> rad = phase_radix_list(ni);
    const bool fuse_in = fused_ && fusible(rad);
    const bool fuse_out = fuse_in;
    const T* row_tw = m > 1 ? row_twiddles(ni, m) : nullptr;
    if (!fuse_in) {
      add_copy("gather", kernels_.passes.block_copy[0], ni, b, src, src_row, slot::a, blk, 1.0,
               nullptr, p);
    }
    const slot z = phase_chain(
        ni, rad, b, sign,
        fuse_in ? side_of{.where = src, .big = true, .row = src_row} : side_of{.where = slot::a},
        fuse_out ? side_of{.where = dst, .big = true, .row = big_l} : side_of{.where = slot::a},
        1.0, fuse_out ? out_scale : 1.0, fuse_out ? row_tw : nullptr, p);
    if (!fuse_out && m > 1) {
      add_copy("twiddled scatter", kernels_.passes.twiddle_scatter[d], ni, b, z, blk, dst, big_l,
               1.0, row_tw, p);
    } else if (!fuse_out) {
      add_copy("scatter", kernels_.passes.block_copy[out_scale != 1.0 ? 1 : 0], ni, b, z, blk, dst,
               big_l, out_scale, nullptr, p);
    }
    const auto end = static_cast<std::uint32_t>(p.records.size());
    std::get<loop>(p.records[inner]).body_end = end;
    if (m > 1) {
      std::get<loop>(p.records[outer]).body_end = end;
    }
    l_count *= ni;
  }
  return b * longest_factor(g);
}

template <class T> std::string grid_builder<T>::note(const grid_split& g) const {
  std::string sizes;
  std::string phases;
  for (std::size_t i = 0; i < g.phases; ++i) {
    sizes += (i == 0 ? "" : " x ") + std::to_string(g.factors.at(i));
    const std::size_t f = g.factors.at(i);
    phases += (i == 0 ? "" : ", then ") + std::to_string(f) + " (radices " +
              radix_text(phase_radix_list(f)) + ")";
  }
  const std::string padded =
      g.pad != 0 ? "; phase 2's rows padded by " + std::to_string(g.pad) + " elements" : "";
  return "grid " + sizes + " (decimation in frequency: phases of " + phases +
         ", each in blocks of " + std::to_string(g.block) + " columns" + padded + ")";
}

template class grid_builder<double>;
template class grid_builder<float>;

} // namespace fftn::inline v0::detail
