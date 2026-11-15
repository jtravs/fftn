// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The grid strategy (DESIGN.md 8.5): a long c2c row of n = N1 N2 ... Nk (k = 2 or 3) as a
// Cooley-Tukey transform in decimation-in-frequency order, one phase per factor, each run
// depth-first over blocks of adjacent columns so that only one block at a time is worked on
// in cache (experiment X12).
//
// Phase i transforms down the columns of what the phase before wrote. With L = N1 ... N(i-1)
// (1 for the first phase) and M = N(i+1) ... Nk, its input element (l, c, r), l < L, c < M,
// r < Ni, is at l + L (c + M r): columns l + L c are adjacent, rows r are L M apart. The
// phase computes, for every column, the length-Ni transform over r to k, multiplies by
// w_(Ni M)^(c k), and writes element (l, k, c) to l + L (k + Ni c). After the last phase
// (M = 1, no twiddles) element (k1, ..., kk) is at k1 + N1 (k2 + N2 (...)), which is X in
// natural order: this is the decimation-in-frequency factorisation applied once per phase.
//
// Every phase works on blocks of `block` adjacent columns: a block is gathered row by row
// into tight scratch, transformed there by a chain of lane passes (lanes over the block's
// columns), and written back:
// - phase 1 (L = 1): the columns are the c, so a block's output is block rows of N1
//   adjacent elements; the twiddled transpose writes them (C x C tiles transposed in
//   registers), each element times its own twiddle w_n^(c k1), formed from two small tables
//   (src/passes/grid.inl);
// - later phases: a block is block adjacent l at one c (block divides N1), so it is written
//   row by row like the gather, rows L apart, each row times its twiddle w_(Ni M)^(c k), one
//   value per row; the last phase's rows carry the output edge's scale instead (phase 1's
//   gather the input edge's, with build_settings::scale_on_input).
// A phase reads one array and writes another, except the last, which writes the positions
// it read and so runs in place: phase 1 writes the output out of place when there are two
// phases, otherwise the workspace block, and phase 2 of three writes the output. Scratch
// lines a and b hold one block (block x max Ni). The loops over blocks are the program's
// (DESIGN.md 8.3), one iteration per block, independent; the twiddle tables move with them
// through the twiddle cursor (program.hpp).
//
// Why gathers: for most long lengths the factors are powers of two, so a block's rows lie a
// power of two apart and map to the same few cache sets. A pass that read the block in
// place would load one register from each of R rows per unit and evict them before the
// next unit, at the next columns, came back for the rest of each line. Copying whole rows
// into tight scratch and back touches every line once. Why three phases: the block must be
// narrow for a long phase to fit the cache, and narrow blocks make the gathers and scatters
// touch a cache line, and on large arrays a page, per few elements; a third phase costs one
// more pass over memory but keeps every phase short and its blocks wide.
//
// There is no fourth phase: the factors of three phases grow as n^(1/3), so their blocks
// narrow; on a 128 KiB L1 they stay 32 columns wide to 2^24 points (in double) and halve for
// every further factor of 8.
#pragma once

#include "core/program.hpp"
#include "isa/kernel_table.hpp"
#include "strategies/chain.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

inline constexpr std::size_t max_grid_phases = 3;
// The widest block the rules choose (experiment X12).
inline constexpr std::size_t max_grid_block = 32;
// The largest radix of a phase whose first and last passes read and write the arrays' rows
// directly (grid_builder, fused; experiment X12).
inline constexpr std::size_t grid_fuse_radix = 8;
// The largest radix of the phase rule (grid_builder): every codelet radix up to it.
inline constexpr std::size_t default_grid_phase_radix = 16;

// How a row of n is split: n = factors[0] x ... x factors[phases - 1], each phase run on
// blocks of `block` columns. phases == 0: no grid.
struct grid_split {
  std::size_t phases = 0;
  std::array<std::size_t, max_grid_phases> factors{};
  std::size_t block = 0;
};

// The grid of n for a target with `lanes` complex numbers per register, or none. The
// factors have no prime above 13 (each phase is a chain of codelet passes), the block (a
// multiple of lanes, at least 2) divides N1 and n / N1, and lanes divides N1. forced: the
// factors in phase order (two or three), or empty to choose; forced_block: the block, or 0
// to choose. The block of a split is the widest power of two from max(2, lanes) to
// max_grid_block that divides N1 and n / N1 and keeps block x max Ni within `budget` points
// (the scratch lines a and b of one block in L1). The choice, among the splits into two or
// (three_phases) into three: the widest block, then the shortest longest factor, then the
// longest shortest factor, then the shortest first factor; three phases may fill twice the
// budget (experiment X12). Two phases when three are not asked for or n has no such split,
// and three when n has no split into two. A pure function.
[[nodiscard]] grid_split choose_grid(std::size_t n, std::size_t lanes, std::size_t budget,
                                     bool three_phases, std::size_t forced_block,
                                     const std::vector<std::size_t>& forced);

template <class T> class grid_builder {
public:
  // The chain builder supplies the radices of the phases and their lane passes' twiddles;
  // the grid's own tables are appended to `tables` (as the chain's, they must outlive the
  // stages).
  // fused: phases read and write the arrays in their first and last passes rather than
  // through a gather and a scatter (experiment X12). phase_radix: the largest radix the
  // phase rule uses (codelet radices up to it; 0: default_grid_phase_radix).
  grid_builder(const kernels_of<T>& kernels, chain_builder<T>& chain,
               std::vector<std::vector<T>>& tables, bool fused, std::size_t phase_radix);

  // Appends the records of one c2c row from the in slot to the out slot, both stride 1,
  // and returns the complex elements of scratch it needs in each of a and b. in_place: in
  // and out are the same memory. Whether it uses the workspace block: block_elements().
  // scale: on the output (1: none), or on the input with scale_on_input (the input edge:
  // phase 1's gather applies it). c2r_n: 0, or the row is the half-length complex transform
  // of a c2r row of c2r_n reals and the in slot holds its n/2 + 1 bins, which phase 1's
  // gather pairs (c2r_pair_gather, experiment X29): no pairing pass precedes the grid.
  std::size_t c2c_row(const grid_split& g, int sign, bool in_place, double scale,
                      bool scale_on_input, program& p, std::size_t c2r_n = 0);

  // Whether c2c_row of g uses the block slot, n complex elements.
  [[nodiscard]] static bool uses_block(const grid_split& g, bool in_place) noexcept;

  // "grid 64 x 128 x 128 (...)" for describe().
  [[nodiscard]] std::string note(const grid_split& g) const;

private:
  // One end of a phase's chain: tight scratch, or rows of an array `row` apart (a gather's
  // or scatter's array, read or written directly by the first or last pass).
  struct side_of {
    slot where = slot::a;
    bool big = false;
    std::ptrdiff_t row = 0;
  };
  // The chain of a phase of n on a block of `cols` columns from `from` to `to`; returns the
  // slot holding the result. row_tw: the row twiddles of a later phase's last pass.
  slot phase_chain(std::size_t n, const std::vector<std::size_t>& rad, std::size_t cols, int sign,
                   const side_of& from, const side_of& to, double in_scale, double out_scale,
                   const T* row_tw, program& p);
  [[nodiscard]] std::vector<std::size_t> phase_radix_list(std::size_t n) const;
  void add_copy(const char* what, pass_fn run, std::size_t rows, std::size_t cols, slot src,
                std::ptrdiff_t src_stride, slot dst, std::ptrdiff_t dst_stride, double scale,
                const T* twiddles, program& p);
  std::pair<const T*, const T*> first_twiddles(const grid_split& g);
  std::pair<const T*, const T*> c2r_pair_factors(std::size_t n1, std::size_t m1, std::size_t n);
  const T* row_twiddles(std::size_t ni, std::size_t m);

  const kernels_of<T>& kernels_;
  chain_builder<T>& chain_;
  std::vector<std::vector<T>>& tables_;
  bool fused_;
  std::size_t phase_radix_;
};

extern template class grid_builder<double>;
extern template class grid_builder<float>;

} // namespace fftn::inline v0::detail
