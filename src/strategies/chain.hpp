// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The chain strategy (DESIGN.md 8.5): a 1-D transform of one line as a Stockham autosort
// chain of radix-R passes, breadth-first. This builds the stages of one line; the plan
// (core/plan_core.cpp) wraps them in the loops over lines and transforms.
//
// Walking-skeleton choices, all of them planner decisions to revisit in phases 1 and 2:
// - Radices in the order 4, 4, ..., 2, 3, ..., 5, ..., then the other primes ascending,
//   each through the generic O(p^2) pass.
// - The first pass reads the line where it lies (any stride), the last writes it where it
//   goes; passes in between ping-pong between two scratch lines a and b of the line's
//   length. When source and destination are the same memory, a one-pass chain goes
//   through a and a copy, since the generic pass cannot run in place.
// - Real lines (DESIGN.md 9.1): even n packs the reals as n/2 complex values into scratch,
//   runs the half-length chain there and finishes with a separate post-pass (r2c), or
//   starts with a pre-pass and ends unpacking (c2r); odd n runs the full-length complex
//   chain on (x, 0) or on the expanded Hermitian spectrum. Fusing the edges into the
//   first or last pass is phase 2.
// - Chunks: a stage of u units is cut into ranges of about chunk_points points each.
#pragma once

#include "core/program.hpp"
#include "isa/kernel_table.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {

// Where a line is read or written: a slot and the stride between its elements there.
struct endpoint {
  slot where = slot::in;
  std::ptrdiff_t stride = 1;
};

// The radices of a chain for n, in pass order; empty for n = 1.
[[nodiscard]] std::vector<std::size_t> chain_radices(std::size_t n);

template <class T> class chain_builder {
public:
  // Tables are appended to `tables`, whose elements must outlive the stages (moving the
  // outer vector keeps every inner buffer where it is).
  chain_builder(const kernels_of<T>& kernels, std::vector<std::vector<T>>& tables,
                std::size_t chunk_points);

  // Each appends the stages of one line to `stages` and returns the complex elements of
  // scratch the line needs in each of a and b (0 when it needs none).
  //
  // aliased: src and dst are the same memory for this line.
  std::size_t c2c_line(std::size_t n, int sign, endpoint src, endpoint dst, bool aliased,
                       std::vector<stage>& stages);
  std::size_t r2c_line(std::size_t n, endpoint src, endpoint dst, std::vector<stage>& stages);
  std::size_t c2r_line(std::size_t n, endpoint src, endpoint dst, std::vector<stage>& stages);

private:
  const std::vector<std::complex<T>>& roots(std::size_t n);
  const T* pack(const std::vector<std::complex<T>>& w);
  const T* stage_twiddles(std::size_t n, std::size_t radix, std::size_t m, std::size_t s);
  const T* radix_roots(std::size_t n, std::size_t radix);
  const T* edge_twiddles(std::size_t n);
  std::uint32_t chunks_of(std::size_t units, std::size_t points_per_unit) const;
  // The passes of a chain of n from `from` into scratch, ping-ponging between a and b;
  // returns the slot holding the result.
  slot scratch_chain(std::size_t n, int sign, slot from, std::vector<stage>& stages);
  void add_pass(std::size_t n, std::size_t radix, std::size_t s, int sign, endpoint src,
                endpoint dst, std::vector<stage>& stages);
  void add_edge(pass_fn run, const char* name, std::size_t n, std::size_t units, endpoint src,
                endpoint dst, const T* twiddles, std::vector<stage>& stages);

  const kernels_of<T>& kernels_;
  std::vector<std::vector<T>>& tables_;
  std::size_t chunk_points_;
  std::map<std::size_t, std::vector<std::complex<T>>> roots_;
  std::map<std::pair<std::size_t, std::size_t>, const T*> stage_tables_; // (n, s)
  std::map<std::pair<std::size_t, std::size_t>, const T*> radix_roots_;  // (n, radix)
  std::map<std::size_t, const T*> edge_tables_;
};

extern template class chain_builder<double>;
extern template class chain_builder<float>;

} // namespace fftn::inline v0::detail
