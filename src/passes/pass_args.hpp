// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The calling convention of pass kernels (DESIGN.md 8.3): what a stage hands its kernel,
// and the entry points of the passes as each target exports them. Plain pointers and
// integers only, so the struct can cross from baseline code into an ISA region
// (region.hpp). Included before any region opens.
#pragma once

#include <cstddef>
#include <cstdint>

namespace fftn::inline v0::detail {

// One pass over one line, as the driver resolves it at execution: the stage's fixed
// geometry plus the addresses of its source and destination. Each kernel documents which
// fields it reads (src/passes/*.inl).
struct pass_args {
  const void* src = nullptr;
  void* dst = nullptr;
  // In elements of each side's type (real or complex); any sign. Scratch has stride 1.
  std::ptrdiff_t src_stride = 1;
  std::ptrdiff_t dst_stride = 1;
  // The stage's twiddles in the storage form its kind reads (src/passes/stockham.inl).
  const void* twiddles = nullptr;
  // The generic prime pass: the radix-th roots of unity, in broadcast storage.
  const void* roots = nullptr;
  // Stockham geometry: radix, butterfly groups m and copies s (n = radix * m * s).
  std::size_t radix = 1;
  std::size_t m = 1;
  std::size_t s = 1;
  // The line's logical length, for the edge passes of real transforms.
  std::size_t n = 1;
  // Edge operations (DESIGN.md 12.1), read by the variants that have them: the real scale,
  // rounded once to the kernel's precision, and the complex factor arrays of the line's
  // input and output (one value per element of the line, tight). The arrays are fixed at
  // plan construction for now; phase 5 passes them at execution (exec_args).
  double scale = 1.0;
  const void* in_factor = nullptr;
  const void* out_factor = nullptr;
};

// Runs units [first, last) of a pass. A unit is the kernel's own (a block of butterflies,
// a line, an element); units are independent within a pass, so any split of [0, units)
// into ranges gives the same results bit for bit.
using pass_fn = void (*)(const pass_args* args, std::size_t first, std::size_t last);

// What an edge pass does to the values it loads (input edge) or stores (output edge), as
// the first two edge operations of DESIGN.md 12.1. Each is a separate instantiation of
// the pass; nothing in a kernel tests it at run time.
enum class edge_op : std::uint8_t {
  none,
  scale,  // times pass_args::scale
  factor, // times the line's factor array, element by element
};
inline constexpr std::size_t edge_op_count = 3;

// The codelets' radices index the tables below directly; gen::radices lists which exist.
inline constexpr std::size_t max_codelet_radix = 16;

// The passes of precision T as a target exports them (src/passes/entries.inl). The
// direction index is 0 forward (exponent -i), 1 backward (+i); edge indices are edge_op
// values. An entry is null where no codelet of that radix exists, and where both edges
// would scale (a plan has one scale).
//
// NOLINTBEGIN(modernize-avoid-c-arrays): filled inside ISA regions, which use no std templates
struct radix_passes {
  // The chain's first pass (s = 1): lanes over butterflies; input edge.
  pass_fn grouped[2][edge_op_count];
  // Its middle passes (m > 1): lanes over copies, twiddled.
  pass_fn lane[2];
  // Its last pass (m = 1): lanes over copies, no twiddles; output edge.
  pass_fn last[2][edge_op_count];
  // A whole transform of n = radix <= 16 in registers; input edge, output edge.
  pass_fn leaf[2][edge_op_count][edge_op_count];
};

template <class T> struct pass_kernels {
  // Indexed by the codelet's radix.
  radix_passes codelet[max_codelet_radix + 1];
  // Any radix by direct evaluation (the primes above 16); input edge, output edge.
  pass_fn generic[2][edge_op_count][edge_op_count];
  // y[i] = x[i] with the edges' operations; input edge, output edge.
  pass_fn copy[edge_op_count][edge_op_count];
  // The chirp multiplications around a Bluestein stage's convolution (passes/bluestein.inl):
  // into the zero-padded buffer, with the input edge; out of it, with the output edge.
  pass_fn bluestein_in[edge_op_count];
  pass_fn bluestein_out[edge_op_count];
  // Real-to-complex edges (DESIGN.md 9.1): even n = 2h through a complex transform of h,
  // odd n through n. The output edges take [scaled].
  pass_fn r2c_pack_pairs;
  pass_fn r2c_post[2];
  pass_fn r2c_pack;
  pass_fn r2c_store[2];
  // Complex-to-real edges, the mirror images.
  pass_fn c2r_pre;
  pass_fn c2r_unpack_pairs[2];
  pass_fn c2r_expand;
  pass_fn c2r_store[2];
};
// NOLINTEND(modernize-avoid-c-arrays)

} // namespace fftn::inline v0::detail
