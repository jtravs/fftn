// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The chain strategy (DESIGN.md 8.5): a 1-D transform of one line as a Stockham autosort
// chain of radix-R passes, breadth-first, on the generated codelets. This builds the stages
// of one line; the plan (core/plan_core.cpp) wraps them in the loops over lines and
// transforms.
//
// - Radices: the rules planner's (planner/rules.hpp), or a sequence a test forces.
// - Pass kinds (DESIGN.md 8.2): n <= 16 is one leaf; otherwise the first pass is grouped,
//   the middle ones are lane passes with twiddles and the last a lane pass without; a radix
//   above 16 (a prime factor) is a generic pass in any position, except that the prime
//   factors above the profile's Bluestein threshold, which the rules put last, form one
//   Bluestein stage at the end of the chain (below).
// - Placement: the first pass reads the line where it lies and the last writes it where it
//   goes; passes in between ping-pong between two scratch lines a and b of the line's
//   length. On targets with several complex numbers per register (C > 1), the passes
//   load and store whole runs of adjacent elements, so a line with a stride other than 1
//   is first copied to scratch (input) or written to scratch and then copied (output);
//   leaves and copies work at any stride. A single generic pass cannot run in place, so
//   when its source and destination are the same memory it writes scratch and is copied.
// - Edge operations (DESIGN.md 12.1): the input edge's operation applies in the first
//   pass that is not a copy, the output edge's in the last, so neither costs a pass.
// - Real lines (DESIGN.md 9.1): even n packs the reals as n/2 complex values into scratch,
//   runs the half-length chain there and finishes with a separate post-pass (r2c), or
//   starts with a pre-pass and ends unpacking (c2r); odd n runs the full-length complex
//   chain on (x, 0) or on the expanded Hermitian spectrum. The scale applies in the edge
//   pass that writes the output. Fusing the edges into the first or last pass is phase 2.
// - Bluestein (DESIGN.md 8.5, strategies/bluestein.hpp): the last pass of a chain whose
//   remaining radix Q (the product of the large primes) has s = n / Q copies is the
//   convolution of s interleaved sequences of length M (copy innermost): a chirp pass
//   into scratch, zero-padded to s M; the chain of M, forward, with every pass's copy
//   count multiplied by s and the kernel spectrum as the last pass's output factor (one
//   value per element of the s M, each repeated s times); the chain of M backward; and a
//   chirp pass out, elementwise because m = 1 makes the stage's input and output indices
//   the same. The convolution ping-pongs between a and b, an even number of passes, so
//   it ends where it began; scratch lines are then s M long. It runs in place.
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
};

template <class T> class chain_builder {
public:
  // Tables are appended to `tables`, whose elements must outlive the stages (moving the
  // outer vector keeps every inner buffer where it is). forced: when its product is a
  // line's length, that line's radices in pass order, instead of the rules' (tests).
  // bluestein_above: the Bluestein threshold instead of the profile's, when not 0 (tests).
  chain_builder(const kernels_of<T>& kernels, const target_profile& profile,
                std::vector<std::vector<T>>& tables, std::size_t chunk_points,
                std::vector<std::size_t> forced, std::size_t bluestein_above = 0);

  // The radices of a chain for n, in pass order (empty for n = 1); the last is a Bluestein
  // stage's when bluestein_last(n).
  [[nodiscard]] std::vector<std::size_t> radices(std::size_t n) const;
  // Whether a test forces the radices of n (build_settings::radices multiply to n).
  [[nodiscard]] bool forces(std::size_t n) const;
  [[nodiscard]] bool bluestein_last(std::size_t n) const;
  // The radices for describe(): "4 x 4 x 16", with a Bluestein stage as
  // "bluestein 257 (M 520 = 8 x 5 x 13)".
  [[nodiscard]] std::string radix_text(std::size_t n) const;

  // Each appends the stages of one line to `stages` and returns the complex elements of
  // scratch the line needs in each of a and b (0 when it needs none).
  //
  // aliased: src and dst are the same memory for this line.
  std::size_t c2c_line(std::size_t n, int sign, endpoint src, endpoint dst, bool aliased,
                       const line_edges& edges, std::vector<stage>& stages);
  // scale: applied to the output (1 means none).
  std::size_t r2c_line(std::size_t n, endpoint src, endpoint dst, double scale,
                       std::vector<stage>& stages);
  std::size_t c2r_line(std::size_t n, endpoint src, endpoint dst, double scale,
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
  [[nodiscard]] std::uint32_t chunks_of(std::size_t units, std::size_t points_per_unit) const;
  [[nodiscard]] std::size_t blocks(std::size_t count) const;
  // The passes of a chain of n from scratch slot `from` into scratch; returns the slot
  // holding the result.
  slot scratch_chain(std::size_t n, int sign, slot from, std::vector<stage>& stages);
  // Pass i of `rad` (n = the product), s copies, with its edges' operations.
  // batch: the line is that many interleaved lines of n / batch (a Bluestein convolution),
  // whose twiddles are those of the chain of n / batch with s / batch copies.
  void add_pass(std::size_t n, const std::vector<std::size_t>& rad, std::size_t i, std::size_t s,
                int sign, endpoint src, endpoint dst, const line_edges& edges,
                std::vector<stage>& stages, std::size_t batch = 1);
  void add_leaf(std::size_t n, int sign, endpoint src, endpoint dst, const line_edges& edges,
                std::vector<stage>& stages);
  void add_copy(std::size_t n, endpoint src, endpoint dst, const line_edges& edges,
                std::vector<stage>& stages);
  // The Bluestein stage of radix q as the last pass of a line of n (s = n / q copies), its
  // edges' operations applied when `first` (input) and always (output).
  void add_bluestein(std::size_t n, std::size_t q, int sign, endpoint src, endpoint dst,
                     const line_edges& edges, bool first, std::vector<stage>& stages);
  void add_edge(pass_fn run, const char* name, std::size_t n, std::size_t units, endpoint src,
                endpoint dst, const T* twiddles, double scale, std::vector<stage>& stages);

  const kernels_of<T>& kernels_;
  target_profile profile_;
  std::vector<std::vector<T>>& tables_;
  std::size_t chunk_points_;
  std::vector<std::size_t> forced_;
  std::size_t bluestein_above_;
  std::size_t line_scratch_ = 0; // scratch a Bluestein stage of the current line needs
  std::map<std::size_t, std::vector<std::complex<T>>> roots_;
  std::map<std::pair<std::size_t, std::size_t>, const T*> grouped_tables_;              // (n, R)
  std::map<std::tuple<std::size_t, std::size_t, std::size_t>, const T*> shared_tables_; // n, R, s
  std::map<std::size_t, const T*> radix_roots_;                                         // R
};

extern template class chain_builder<double>;
extern template class chain_builder<float>;

} // namespace fftn::inline v0::detail
