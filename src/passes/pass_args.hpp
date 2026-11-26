// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The calling convention of pass kernels (DESIGN.md 8.3): what a stage hands its kernel,
// and the entry points of the passes as each target exports them. Plain pointers and
// integers only, so the struct can cross from baseline code into an ISA region
// (region.hpp). Included before any region opens.
#pragma once

#include "gen/codelets.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace fftn::inline v0::detail {

// One pass over one line: the stage's fixed geometry, set when the plan is built. The
// addresses of its source and destination are the kernel's arguments (pass_fn). Each kernel
// documents which fields it reads (src/passes/*.inl).
struct pass_args {
  // In elements of each side's type (real or complex); any sign. Scratch has stride 1.
  std::ptrdiff_t src_stride = 1;
  std::ptrdiff_t dst_stride = 1;
  // The stage's twiddles in the storage form its kind reads (src/passes/stockham.inl).
  const void* twiddles = nullptr;
  // A second table: the generic prime pass's radix-th roots of unity, in broadcast
  // storage; the fused real passes' pairing twiddles (src/passes/real.inl); a Rader stage's
  // index table (src/passes/rader.inl).
  const void* roots = nullptr;
  // Stockham geometry: radix, butterfly groups m and copies s (n = radix * m * s).
  std::size_t radix = 1;
  std::size_t m = 1;
  std::size_t s = 1;
  // The line's logical length, for the edge passes of real transforms.
  std::size_t n = 1;
  // Edge operations (DESIGN.md 12.1), read by the variants that have them: the real scale,
  // rounded once to the kernel's precision, and the complex factor arrays of the line's
  // input and output (one value per element of the line, tight). Fixed tables, or for a
  // plan's edges the execution's arrays, which the edge driver sets from the factor cursors
  // (core/program.hpp, stage::moving_factors).
  double scale = 1.0;
  const void* in_factor = nullptr;
  const void* out_factor = nullptr;
  // A grid phase's pass (src/passes/grid.inl, phase_pass): copy q of s is (q % cols, q / cols)
  // on each side, at (q % cols) + qstride (q / cols) + stride idx; 0 for cols means s. The
  // edge pass (core/edges.cpp) reads its factor's stride along the line, in the factor's
  // elements, from src_qstride.
  std::size_t src_cols = 0;
  std::size_t dst_cols = 0;
  std::ptrdiff_t src_qstride = 0;
  std::ptrdiff_t dst_qstride = 0;
  // A lane pass whose output factor is shared (factor_sharing::output): the consecutive
  // elements that share each entry, a multiple of the target's complex count.
  std::size_t factor_copies = 1;
  // The lanes strategy's passes (src/passes/lanes.inl): the reals between consecutive rows
  // on each side, the rows of a tile, and the rows of the block being run, which the driver
  // sets for every block (core/driver.cpp, program::lanes).
  std::ptrdiff_t src_dist = 0;
  std::ptrdiff_t dst_dist = 0;
  // 32 bits each, so that columns_inner fits in the 8 bytes the two took: a struct 8 bytes
  // longer made a single row of 128 points 1.17 times slower on the Xeon 6240 (X40; its
  // copies in the flat programs' stage arrays span more lines). B is at most a few thousand.
  std::uint32_t block = 0; // B: the rows of a tile
  std::uint32_t rows = 0;
  // lanes_first and lanes_last take their units with the block's runs of C rows innermost
  // instead of outermost: for the strided axis of a plane, whose lanes' rows are adjacent
  // columns, so that each array row's B columns are read or written at once, whole cache
  // lines, before the next (experiment X40).
  bool columns_inner = false;
  // The c2r passes that read a line's half spectrum (passes/real.inl: c2r_pre, c2r_first,
  // c2r_leaf) read it packed into h complex values: element 0 holds (Re X_0, Re X_h), the
  // others X_1 .. X_h-1; otherwise it is the h + 1 values X_0 .. X_h. The packed line fits the
  // n reals of the output, where a multi-axis c2r out of place builds it (experiment X60).
  // In the struct's tail padding: its size is unchanged.
  bool packed = false;
};

// Runs units [first, last) of a pass from src to dst (the line's first elements). A unit is
// the kernel's own (a block of butterflies, a line, an element); units are independent
// within a pass, so any split of [0, units) into ranges gives the same results bit for bit.
// The addresses are arguments, as a leaf's are, so that the driver calls the kernel on the
// plan's own arguments: a fresh copy of them per call cost 3-5 ns per stage on the Xeon and
// the EPYC, about 1.5 on the M5 Max (core/driver.cpp; experiment X28).
using pass_fn = void (*)(const pass_args* args, const void* src, void* dst, std::size_t first,
                         std::size_t last);

// A leaf (a whole transform in registers, DESIGN.md 8.2) over `count` lines: line u read at
// src + u src_dist and written at dst + u dst_dist bytes, with the edges of args. The addresses are
// arguments so that the driver can call it on a plan's own arguments, without the copy a pass_fn
// needs, and a batch of rows in one call (core/driver.cpp; dev/experiments/X20-call-overhead.md).
using leaf_fn = void (*)(const pass_args* args, const void* src, void* dst, std::ptrdiff_t src_dist,
                         std::ptrdiff_t dst_dist, std::size_t count);

// What an edge pass does to the values it loads (input edge) or stores (output edge), as
// the first two edge operations of DESIGN.md 12.1. Each is a separate instantiation of
// the pass; nothing in a kernel tests it at run time.
enum class edge_op : std::uint8_t {
  none,
  scale,  // times pass_args::scale
  factor, // times the line's factor array, element by element
};
inline constexpr std::size_t edge_op_count = 3;

// How a lane pass reads a factor table that holds one entry per element of a sequence
// whose copies are interleaved in its line (passes/stockham.inl, lane_pass): not at all,
// on its input or on its output. A separate instantiation each.
enum class factor_sharing : std::uint8_t { none, input, output };

// The codelets' radices are gen::radices (every radix 2 to 16, then 17, 19, 20, 23, 25, 32, 64);
// pass_kernels::codelet has one row per entry, in that order. codelet_slot(r) is r's row,
// or gen::radix_count when no codelet has radix r (a radix above 16 that is not listed is
// a generic pass).
[[nodiscard]] constexpr std::size_t codelet_slot(std::size_t r) noexcept {
  for (std::size_t i = 0; i < gen::radix_count; ++i) {
    if (gen::radices[i] == r) {
      return i;
    }
  }
  return gen::radix_count;
}
[[nodiscard]] constexpr bool has_codelet(std::size_t r) noexcept {
  return codelet_slot(r) < gen::radix_count;
}
// Whether a split codelet (src/gen/split_codelets.inl) of radix r exists.
[[nodiscard]] constexpr bool has_split_codelet(std::size_t r) noexcept {
  return std::ranges::any_of(gen::split_radices, [r](std::size_t q) { return q == r; });
}
// Whether the pruned codelets of radix r (dif_head, plain_head; task 5.1e) exist.
[[nodiscard]] constexpr bool has_pruned_codelet(std::size_t r) noexcept {
  return std::ranges::any_of(gen::pruned_radices, [r](std::size_t q) { return q == r; });
}
// Every radix from 2 to this one has a codelet; the planner and the grid choose pass radices
// up to it, and prime factors above it are the generic pass's or Bluestein's.
inline constexpr std::size_t dense_codelet_radix = 16;

// The passes of precision T as a target exports them (src/passes/entries.inl). The
// direction index is 0 forward (exponent -i), 1 backward (+i); edge indices are edge_op
// values. An entry is null where no codelet of that radix exists, and where both edges
// would scale (a plan has one scale).
//
// NOLINTBEGIN(modernize-avoid-c-arrays): filled inside ISA regions, which use no std templates
// The variants of a grid phase's pass: twiddled (m > 1) or plain (m = 1), with the input's
// scale, or on output the scale or one twiddle per row.
enum phase_variant : std::uint8_t {
  phase_twiddled,
  phase_twiddled_in_scale,
  phase_plain,
  phase_plain_in_scale,
  phase_plain_out_scale,
  phase_plain_row_twiddles,
  phase_variants,
};

struct radix_passes {
  // The chain's first pass (s = 1): lanes over butterflies; input edge.
  pass_fn grouped[2][edge_op_count];
  // Its middle passes (m > 1): lanes over copies, twiddled.
  pass_fn lane[2];
  // The backward middle pass with every input leg times a factor shared by the copies
  // (factor_sharing::input): the first backward pass of a Bluestein convolution with more
  // than one copy, taking the kernel spectrum (passes/bluestein.inl).
  pass_fn lane_shared_in;
  // The forward last pass with every output times a factor shared by factor_copies
  // consecutive copies (factor_sharing::output): the last forward pass of a Bluestein
  // convolution whose copy count C divides.
  pass_fn last_shared_out;
  // Its last pass (m = 1): lanes over copies, no twiddles; output edge.
  pass_fn last[2][edge_op_count];
  // The edge passes of a line with stored runs (task 5.1e), on the pruned codelets: the first
  // pass reading legs 0 .. R/2 - 1 only (dif_head), from src_cols bytes after its source; the
  // last pass storing legs 0 .. R/2 - 1 only (plain_head), from dst_cols bytes after its
  // destination [output edge none or scale]. Null for the radices without pruned codelets
  // (gen::pruned_radices).
  pass_fn grouped_head[2];
  pass_fn last_head[2][2];
  // A whole transform of n = radix in registers; input edge, output edge.
  leaf_fn leaf[2][edge_op_count][edge_op_count];
  // The same for lines of stride 1 on both sides (experiment X28b, passes/unit_leaf.inl);
  // edges none or scale ([input][output]). Null where both would scale, for radices above
  // 32, which are never a leaf by default, and on the x86 targets, whose plans take the
  // narrow leaves (kernel_table.hpp).
  leaf_fn leaf_unit[2][2][2];
  // Split-arithmetic lane passes (src/passes/split.inl), null where the radix has no split
  // codelet: a middle pass (twiddled) writing the BL layout, from the interleaved line (the
  // first pass of a split run, [1]) or from BL ([0]); and the last pass, from BL to the
  // interleaved line, with the output edge.
  pass_fn split_mid[2][2];
  pass_fn split_last[2][edge_op_count];
  // A grid phase's first or last pass reading or writing the caller's rows in place of a
  // gather or scatter (src/passes/grid.inl), by phase_variant.
  pass_fn phase[2][phase_variants];
  // Real transforms of n = 2h with the pairing of bins k and h - k fused (DESIGN.md 9.1,
  // passes/real.inl): the last pass of an r2c chain of h, [scaled]; the first pass of a
  // c2r chain of h; the leaves of h = radix, [scaled].
  pass_fn r2c_last[2];
  pass_fn c2r_first;
  pass_fn r2c_leaf[2];
  pass_fn c2r_leaf[2];
  // The lanes strategy's passes that read or write rows (src/passes/lanes.inl): a block's
  // first pass, its last pass [output edge none or scale], its chain of one pass [the same].
  pass_fn lanes_first[2];
  pass_fn lanes_last[2][2];
  pass_fn lanes_leaf[2][2];
  // The depth-first chain's passes (src/passes/depth.inl, experiment X46), by direction: the
  // first pass of every sub-row, each written to its own block, and the in-place top pass
  // [direction][output edge none, scale or factor (a Bluestein convolution's kernel spectrum,
  // experiment X50)]. Null for the radices the chain does not take (passes/entries.inl,
  // depth_radix).
  pass_fn depth_first[2];
  pass_fn depth_top[2][3];
  // A Bluestein convolution's forward first pass and backward top pass [output edge none or
  // scale] with the stage's chirp passes fused in (experiment X50 arm C).
  pass_fn depth_first_chirp;
  pass_fn depth_top_chirp[2];
  // A c2r row's half-length chain's first pass (backward) with the pairing fused in, from the
  // row's h + 1 bins (experiment X78).
  pass_fn depth_first_c2r;
  // The depth-first pair's forward first pass reading one stored run from logical 0 with zeros
  // beyond, and its backward top pass writing one stored run [output edge none or scale]
  // (DESIGN.md 12.2, task 5.2d).
  pass_fn depth_first_head;
  pass_fn depth_top_part[2];
};

template <class T> struct pass_kernels {
  // Indexed by codelet_slot(radix); never null. Pointers, because a target's rows are
  // defined in several translation units (src/isa/rows.inl, DESIGN.md 6.4).
  const radix_passes* codelet[gen::radix_count];
  // Any radix by direct evaluation (radices with no codelet: the primes above 16); input
  // edge, output edge.
  pass_fn generic[2][edge_op_count][edge_op_count];
  // y[i] = x[i] with the edges' operations; input edge, output edge.
  pass_fn copy[edge_op_count][edge_op_count];
  // The chirp multiplications around a Bluestein stage's convolution (passes/bluestein.inl):
  // into the zero-padded buffer, with the input edge; out of it, with the output edge.
  // Indexed [shared][edge]: shared reads the chirp once for all the copies (s a multiple
  // of C), otherwise one entry per element of the line.
  pass_fn bluestein_in[2][edge_op_count];
  pass_fn bluestein_out[2][edge_op_count];
  // The gather and scatter around a Rader stage's convolution (passes/rader.inl), indexed
  // [shared][edge] as the Bluestein passes: shared runs C copies per vector (s a multiple of
  // C), otherwise one copy at a time.
  pass_fn rader_in[2][edge_op_count];
  pass_fn rader_out[2][edge_op_count];
  // Real-to-complex edges (DESIGN.md 9.1): even n = 2h through a complex transform of h,
  // whose pairing runs as a separate pass where no pass of the transform takes it; odd n
  // through n. [scaled].
  pass_fn r2c_post[2];
  pass_fn r2c_pack;
  pass_fn r2c_store[2];
  // Complex-to-real edges, the mirror images (the pre-pass scales only when h = 1).
  pass_fn c2r_pre[2];
  pass_fn c2r_expand;
  pass_fn c2r_store[2];
  // Re X_h into a packed line (pass_args::packed; experiment X60).
  pass_fn c2r_nyquist;
  // Even real lines with a stride other than 1: packing, pairing at any stride, unpacking.
  pass_fn r2c_pack_pairs;
  pass_fn r2c_post_strided[2];
  pass_fn c2r_pre_strided;
  pass_fn c2r_unpack_pairs[2];
  // The grid's passes (src/passes/grid.inl): a block of columns copied row by row,
  // [prefetching rows ahead][scaled]; by direction, the twiddled transpose of the first phase
  // and the twiddled row copy of the later ones ([prefetching][direction]).
  pass_fn block_copy[2][2];
  pass_fn twiddle_transpose[2];
  pass_fn twiddle_scatter[2][2];
  // The gather of a c2r grid's first phase with the pairing in it (src/passes/real.inl).
  pass_fn c2r_pair_gather;
  // The r2c pairing in a grid's last phase (src/passes/real.inl): the gather of a pair of
  // column sets, their pairing and writing [scaled], and column 0's [scaled].
  pass_fn r2c_pair_gather;
  pass_fn r2c_pair_scatter[2];
  pass_fn r2c_pair_column[2];
  // The lanes strategy (src/passes/lanes.inl): a block of rows into a tile whose lanes are
  // the rows and back, and the pairing of real rows on the tile (r2c [scaled], c2r).
  pass_fn lanes_gather;
  pass_fn lanes_scatter;
  pass_fn lanes_r2c_pair[2];
  pass_fn lanes_c2r_pair;
};
// NOLINTEND(modernize-avoid-c-arrays)

} // namespace fftn::inline v0::detail
