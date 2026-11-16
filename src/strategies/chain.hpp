// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The chain strategy (DESIGN.md 8.5): a 1-D transform of one line as a Stockham autosort
// chain of radix-R passes, breadth-first, on the generated codelets. This builds the stages
// of one line; the plan (core/plan_core.cpp) wraps them in the loops over lines and
// transforms.
//
// - Radices: the rules planner's (planner/rules.hpp), or a sequence a test forces.
// - Pass kinds (DESIGN.md 8.2): a chain of one codelet radix (n itself) is one leaf;
//   otherwise the first pass is grouped, the middle ones are lane passes with twiddles and
//   the last a lane pass without; a radix with no codelet (a prime factor above 16) is a
//   generic pass in any position, except that the prime factors above the profile's
//   Bluestein threshold, which the rules put last, form one Bluestein stage at the end of
//   the chain (below). Codelet radices (32, 64) are never Bluestein's.
// - Placement: the first pass reads the line where it lies and the last writes it where it
//   goes; passes in between ping-pong between two scratch lines a and b of the line's
//   length, or between a and the destination when it is contiguous and not the source. On targets
//   with several complex numbers per register (C > 1), the passes load and store whole runs of
//   adjacent elements, so a line with a stride other than 1 is first copied to scratch (input) or
//   written to scratch and then copied (output); leaves and copies work at any stride. A single
//   generic pass cannot run in place, so when its source and destination are the same memory it
//   writes scratch and is copied.
// - Edge operations (DESIGN.md 12.1): the input edge's operation applies in the first
//   pass that is not a copy, the output edge's in the last, so neither costs a pass.
// - Real lines (DESIGN.md 9.1, passes/real.inl): even n = 2h reads the reals in place as
//   h complex values (a real line has stride 1) and runs a complex transform of h, with the
//   pairing of bins k and h - k fused into the chain's last pass (r2c) or first pass (c2r),
//   or into a leaf for h <= 16 and h = 17, 19, 23. Where that pass is not a codelet's (a
//   Bluestein stage or a generic pass) the pairing is a separate pass in place on the output (r2c,
//   after a c2c line written there) or from the input to the output (c2r, before a c2c line in
//   place there); the grid uses the same separate pass (core/plan_core.cpp). Odd n runs the
//   full-length complex chain on (x, 0) or on the expanded Hermitian spectrum. The scale
//   applies in the pass that writes the output.
// - Bluestein (DESIGN.md 8.5, strategies/bluestein.hpp): the last pass of a chain whose
//   remaining radix Q (the product of the large primes) has s = n / Q copies is the
//   convolution of s interleaved sequences of length M (copy innermost): a chirp pass
//   into scratch, zero-padded to s M; the chain of M, forward, with every pass's copy
//   count multiplied by s and its output element t + s f times entry f of the kernel
//   spectrum (M entries for all the copies: the last pass's output factor when C divides
//   s, broadcast to lanes that share it, else the backward chain's first pass's input,
//   whose lanes are the copies); the chain of M backward; and a chirp pass out, elementwise
//   because m = 1 makes the stage's input and output indices the same. The chirp passes
//   read Q chirp entries, broadcast, when C divides s, otherwise one per element (s Q).
//   Both tables are kept per (Q, sign), so every axis with that Q shares them. The
//   convolution ping-pongs between a and b, an even number of passes, so it ends where
//   it began; scratch lines are then s M long. It runs in place.
// - Split passes (DESIGN.md 6.3, src/passes/split.inl; experiment X1): where the profile
//   says so, a chain's last passes run in split arithmetic: from the first pass after the
//   first whose copy count s is a multiple of the target's split_block (vocabulary.inl;
//   the same on both targets of a pair, so they choose the same passes) to the end, if
//   that run has two passes or more, all of codelet radices with a split codelet, and the
//   line ends in no Bluestein stage. The run's first pass reads the interleaved line and
//   writes BL, the middle ones BL to BL, the last writes the interleaved line with the
//   output edge; so the first pass (grouped), the placement and the scratch are those of
//   an interleaved chain. A Bluestein convolution's chains stay interleaved.
// - Chunks: a stage of u units is cut into ranges of about chunk_points points each.
#pragma once

#include "core/program.hpp"
#include "isa/kernel_table.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

// Where a line is read or written: a slot and the stride between its elements there.
struct endpoint {
  slot where = slot::in;
  std::ptrdiff_t stride = 1;
};

// The edge operations of a line (DESIGN.md 12.1): what its first pass does to the values it
// loads and its last to the values it stores.
struct line_edges {
  edge_op in = edge_op::none;
  edge_op out = edge_op::none;
  double scale = 1.0;              // for edge_op::scale on either edge
  const void* in_factor = nullptr; // n complex values, for edge_op::factor
  const void* out_factor = nullptr;
  // When above 1, out_factor holds one entry per run of this many consecutive elements, a
  // multiple of C (a Bluestein convolution's kernel spectrum over its interleaved copies).
  std::size_t out_factor_copies = 1;
};

// When the pairing of an even real line fuses into a pass of its half-length chain
// (passes/real.inl, DESIGN.md 9.1): the pass's radix at most `radix` (the fused pass runs
// two butterflies at once; beyond the register file they spill), at least `min_lanes`
// register-loads of copies or groups per mirror half (so that the one-lane units for copy 0
// and s/2 and the partial unit do not dominate), and for r2c at most `r2c_bytes` of half
// spectrum (the fused last pass writes 4 R streams a power of two apart; beyond L1 they
// conflict). radix 1: never.
struct real_fusion {
  std::size_t radix = 8;
  std::size_t min_lanes = 4;
  std::size_t r2c_bytes = SIZE_MAX;
};

// How a real line runs (DESIGN.md 9.1): n odd through a full-length complex chain; a line
// with a stride other than 1 through scratch with the pairing at any stride; n = 2
// as the pairing alone; h = n/2 <= 16 as a leaf with the pairing; through a chain of h
// with the pairing fused into its last (r2c) or first (c2r) pass; or with the pairing as a
// separate pass.
enum class real_route : std::uint8_t { full, strided, pairing_only, leaf, fused, separate };

template <class T> class chain_builder {
public:
  // Tables are appended to `tables`, whose elements must outlive the stages (moving the
  // outer vector keeps every inner buffer where it is). forced: when its product is a
  // line's length, that line's radices in pass order, instead of the rules' (tests).
  // bluestein_above: the Bluestein threshold instead of the profile's, when not 0 (tests).
  // fusion: when the pairing of an even real line fuses into a pass (real_fusion).
  chain_builder(const kernels_of<T>& kernels, const target_profile& profile,
                std::vector<std::vector<T>>& tables, std::size_t chunk_points,
                std::vector<std::size_t> forced, std::size_t bluestein_above = 0,
                real_fusion fusion = {});

  // Leaves (n <= 16) from this set instead of the target's own: the 128-bit leaves of the
  // x86 targets (kernel_table.hpp, experiment X28), which round alike. Null: the target's.
  void use_narrow_leaves(const narrow_leaf_set<T>* leaves) noexcept { narrow_ = leaves; }

  // The radices of a chain for n, in pass order (empty for n = 1); the last is a Bluestein
  // stage's when bluestein_last(n).
  [[nodiscard]] std::vector<std::size_t> radices(std::size_t n) const;
  // Whether a test forces the radices of n (build_settings::radices multiply to n).
  [[nodiscard]] bool forces(std::size_t n) const;
  [[nodiscard]] bool bluestein_last(std::size_t n) const;
  // The radices for describe(): "4 x 4 x 16", with a Bluestein stage as
  // "bluestein 257 (M 520 = 8 x 5 x 13)"; of n, or the given ones of n.
  [[nodiscard]] std::string radix_text(std::size_t n) const;
  [[nodiscard]] std::string radix_text(std::size_t n, const std::vector<std::size_t>& rad) const;
  // The radices of the complex transform of h of an even real line (r2c or c2r).
  [[nodiscard]] std::vector<std::size_t> real_radices(std::size_t h, bool r2c) const;

  // Each appends the stages of one line to `stages` and returns the complex elements of
  // scratch the line needs in each of a and b (0 when it needs none).
  //
  // aliased: src and dst are the same memory for this line.
  std::size_t c2c_line(std::size_t n, int sign, endpoint src, endpoint dst, bool aliased,
                       const line_edges& edges, std::vector<stage>& stages);
  // Real lines: scale is applied to the output (1 means none). c2r never writes its source.
  std::size_t r2c_line(std::size_t n, endpoint src, endpoint dst, bool aliased, double scale,
                       std::vector<stage>& stages);
  std::size_t c2r_line(std::size_t n, endpoint src, endpoint dst, double scale,
                       std::vector<stage>& stages);
  // strided: the line has a stride other than 1 on either side.
  [[nodiscard]] real_route route(std::size_t n, bool r2c, bool strided) const;
  // The route of a real line for describe().
  [[nodiscard]] std::string real_text(std::size_t n, bool r2c, bool strided) const;
  // The separate pairing pass of an even real line of n (passes/real.inl, r2c_post or
  // c2r_pre), stride 1 on both sides; src may be dst. scale: on the output (for c2r, only
  // when n = 2, where no transform follows).
  void add_pairing(bool r2c, std::size_t n, endpoint src, endpoint dst, double scale,
                   std::vector<stage>& stages);

  // The lanes strategy (DESIGN.md 8.5, strategies/lanes.cpp, experiment X7): the stages of
  // one block of `block` rows (a multiple of C), rows src_dist and dst_dist elements apart
  // on each side: a gather into a tile in scratch whose lanes are the rows, the chain the row
  // would get alone with every pass's copy count multiplied by block (each lane computes its
  // row's arithmetic: the same codelets, twiddles and split run as c2c_line, r2c_line and
  // c2r_line), and a scatter. Every stage is one chunk. Each returns the complex elements of
  // scratch the block needs in each of a and b. Preconditions: lanes_fit; no input edge
  // operation (c2c); real rows of stride 1 on the real side.
  [[nodiscard]] bool lanes_fit(std::size_t n, bool real) const;
  std::size_t c2c_lanes(std::size_t n, int sign, endpoint src, std::ptrdiff_t src_dist,
                        endpoint dst, std::ptrdiff_t dst_dist, const line_edges& edges,
                        std::size_t block, std::vector<stage>& stages);
  std::size_t r2c_lanes(std::size_t n, endpoint src, std::ptrdiff_t src_dist, endpoint dst,
                        std::ptrdiff_t dst_dist, double scale, std::size_t block,
                        std::vector<stage>& stages);
  std::size_t c2r_lanes(std::size_t n, endpoint src, std::ptrdiff_t src_dist, endpoint dst,
                        std::ptrdiff_t dst_dist, double scale, std::size_t block,
                        std::vector<stage>& stages);

  // For the grid (strategies/grid.hpp), whose phases are lane passes of this builder's
  // kind: the table of a lane pass of `radix` with s copies in a line of n.
  const T* lane_twiddles(std::size_t n, std::size_t radix, std::size_t s) {
    return shared_twiddles(n, radix, s);
  }

private:
  const std::vector<std::complex<T>>& roots(std::size_t n);
  const T* pack(const std::vector<std::complex<T>>& w, bool broadcast);
  const T* grouped_twiddles(std::size_t n, std::size_t radix);
  const T* shared_twiddles(std::size_t n, std::size_t radix, std::size_t s);
  const T* radix_roots(std::size_t radix);
  const T* edge_twiddles(std::size_t n);
  struct bluestein_tables {
    const T* chirp;
    const T* kernel;
  };
  bluestein_tables bluestein_tables_of(std::size_t q, int sign, std::size_t copies);
  const T* rader_kernel_of(std::size_t q, std::size_t g, int sign);
  const T* rader_index_of(std::size_t q, std::size_t g);
  std::complex<T> pairing_root(std::size_t n, std::size_t k);
  const T* pairing_twiddles(std::size_t n);
  const T* leaf_pairing_twiddles(std::size_t n);
  [[nodiscard]] std::uint32_t chunks_of(std::size_t units, std::size_t points_per_unit) const;
  [[nodiscard]] std::size_t blocks(std::size_t count) const;
  // The passes of a chain of n from scratch slot `from` into scratch; returns the slot
  // holding the result.
  slot scratch_chain(std::size_t n, int sign, slot from, std::vector<stage>& stages);
  // How a pass's arithmetic and layouts go (DESIGN.md 6.3): interleaved, or a split pass
  // reading the interleaved line or BL and writing BL (split_first, split_mid), or the last
  // split pass, from BL to the interleaved line.
  enum class pass_layout : std::uint8_t { interleaved, split_first, split_mid, split_last };
  // The first pass of the split run of a chain with these radices whose last pass writes
  // with stride dst_stride, or rad.size() for none.
  [[nodiscard]] std::size_t split_from(const std::vector<std::size_t>& rad, bool bluestein,
                                       std::ptrdiff_t dst_stride) const;
  [[nodiscard]] static pass_layout layout_of(std::size_t i, std::size_t from, std::size_t passes);
  const T* split_twiddles(std::size_t n, std::size_t radix, std::size_t s);
  // Pass i of `rad` (n = the product), s copies, with its edges' operations.
  // batch: the line is that many interleaved lines of n / batch (a Bluestein convolution, a
  // lanes tile), whose twiddles are those of the chain of n / batch with s / batch copies.
  void add_pass(std::size_t n, const std::vector<std::size_t>& rad, std::size_t i, std::size_t s,
                int sign, endpoint src, endpoint dst, const line_edges& edges,
                std::vector<stage>& stages, std::size_t batch = 1,
                pass_layout layout = pass_layout::interleaved);
  void add_leaf(std::size_t n, int sign, endpoint src, endpoint dst, const line_edges& edges,
                std::vector<stage>& stages);
  // Passes i0 .. of rad (the radices of n) from `at`, s copies at i0, the last writing dst
  // with edges.out (a Bluestein stage when bluestein_last(n)); scratch in between.
  void chain_tail(std::size_t n, const std::vector<std::size_t>& rad, std::size_t i0, std::size_t s,
                  int sign, endpoint at, endpoint dst, const line_edges& edges,
                  std::vector<stage>& stages);
  void add_copy(std::size_t n, endpoint src, endpoint dst, const line_edges& edges,
                std::vector<stage>& stages);
  // The Bluestein stage of radix q as the last pass of a line of n (s = n / q copies), its
  // edges' operations applied when `first` (input) and always (output).
  void add_bluestein(std::size_t n, std::size_t q, int sign, endpoint src, endpoint dst,
                     const line_edges& edges, bool first, std::vector<stage>& stages);
  // The destination of a line's last stage, a Bluestein or Rader stage of q, that would
  // write `next` from `at`: a Rader stage permutes into its destination, so from one scratch
  // slot to the other, where its convolution would end in its destination, it writes back
  // over its source instead, which it has read by then (add_rader).
  [[nodiscard]] endpoint last_stage_to(std::size_t q, endpoint at, endpoint next) const;
  // In its place, for a prime q whose rader_root is g (planner/rules.hpp), a Rader stage.
  void add_rader(std::size_t n, std::size_t q, std::size_t g, int sign, endpoint src, endpoint dst,
                 const line_edges& edges, bool first, std::vector<stage>& stages);
  // The convolution chain of a Bluestein or Rader stage: forward and backward over s copies
  // of len points from scratch slot `conv` back to it, the kernel spectrum applied between.
  slot add_convolution(std::size_t s, std::size_t len, const std::vector<std::size_t>& rad,
                       const T* kernel, slot conv, std::vector<stage>& stages);
  void add_edge(pass_fn run, const char* name, std::size_t n, std::size_t units, endpoint src,
                endpoint dst, const T* twiddles, double scale, std::vector<stage>& stages);
  // The lanes strategy's pieces (strategies/lanes.cpp). A block's rows on one side: where,
  // and the reals between a row's elements and between rows; the tile is rows of stride 2B
  // and distance 2 in scratch.
  struct lane_rows {
    slot where = slot::in;
    std::ptrdiff_t stride = 2;
    std::ptrdiff_t dist = 0;
  };
  // The stages of the chain of n (radices rad, split run from pass `split`) on a block of
  // `block` rows from `src` to `dst`, either of which may be a tile; scratch tiles in a and
  // b in between. edges.out applies to the last pass.
  void lanes_body(std::size_t n, const std::vector<std::size_t>& rad, std::size_t split, int sign,
                  lane_rows src, lane_rows dst, const line_edges& edges, std::size_t block,
                  std::vector<stage>& stages);
  void add_lanes_copy(bool gather, std::size_t elements, lane_rows rows, slot tile,
                      std::size_t block, std::vector<stage>& stages);

  const kernels_of<T>& kernels_;
  const narrow_leaf_set<T>* narrow_ = nullptr;
  target_profile profile_;
  std::vector<std::vector<T>>& tables_;
  std::size_t chunk_points_;
  std::vector<std::size_t> forced_;
  real_fusion fusion_;
  std::size_t bluestein_above_;
  std::size_t line_scratch_ = 0; // scratch a Bluestein stage of the current line needs
  std::map<std::size_t, std::vector<std::complex<T>>> roots_;
  std::map<std::pair<std::size_t, std::size_t>, const T*> grouped_tables_;              // (n, R)
  std::map<std::tuple<std::size_t, std::size_t, std::size_t>, const T*> shared_tables_; // n, R, s
  std::map<std::tuple<std::size_t, std::size_t, std::size_t>, const T*> split_tables_;  // n, R, s
  std::map<std::size_t, const T*> radix_roots_;                                         // R
  std::map<std::pair<std::size_t, int>, const T*> kernel_tables_;                       // Q, sign
  std::map<std::tuple<std::size_t, int, std::size_t>, const T*> chirp_tables_; // Q, sign, copies
  std::map<std::size_t, const T*> rader_index_tables_;                         // Q
};

extern template class chain_builder<double>;
extern template class chain_builder<float>;

} // namespace fftn::inline v0::detail
