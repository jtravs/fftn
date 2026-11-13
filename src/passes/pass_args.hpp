// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The calling convention of pass kernels (DESIGN.md 8.3): what a stage hands its kernel,
// and the entry points of the chain strategy's passes as each target exports them. Plain
// pointers and integers only, so the struct can cross from baseline code into an ISA
// region (region.hpp). Included before any region opens.
#pragma once

#include <cstddef>

namespace fftn::inline v0::detail {

// One pass over one line, as the driver resolves it at execution: the stage's fixed
// geometry plus the addresses of its source and destination. Each kernel documents which
// fields it reads (src/passes/chain.inl).
struct pass_args {
  const void* src = nullptr;
  void* dst = nullptr;
  // In elements of each side's type (real or complex); any sign. Scratch has stride 1.
  std::ptrdiff_t src_stride = 1;
  std::ptrdiff_t dst_stride = 1;
  // The stage's twiddles in the target's storage form (kernels.inl, pack_twiddles).
  const void* twiddles = nullptr;
  // The generic prime pass: the radix-th roots of unity, in the same storage form.
  const void* roots = nullptr;
  // Stockham geometry: radix, butterfly groups m and copies s (n = radix * m * s).
  std::size_t radix = 1;
  std::size_t m = 1;
  std::size_t s = 1;
  // The line's logical length, for the edge passes of real transforms.
  std::size_t n = 1;
};

// Runs units [first, last) of a pass. A unit is a butterfly for the Stockham passes and
// the kernel's own unit for edge passes; units are independent within a pass, so any split
// of [0, units) into ranges gives the same results bit for bit.
using pass_fn = void (*)(const pass_args* args, std::size_t first, std::size_t last);

// A pass in both directions: forward (exponent -i) and backward (+i).
struct signed_pass {
  pass_fn forward = nullptr;
  pass_fn backward = nullptr;
};

// The chain strategy's passes for precision T (src/passes/chain.inl). All null on a target
// that does not have them yet: in phase 0 only the scalar target does.
template <class T> struct chain_kernels {
  signed_pass radix2;
  signed_pass radix3;
  signed_pass radix4;
  signed_pass radix5;
  signed_pass generic; // any radix; used for the primes above 5
  pass_fn copy = nullptr;
  // Real-to-complex edges: even n = 2h through a complex transform of h, odd n through n.
  pass_fn r2c_pack_pairs = nullptr;
  pass_fn r2c_post = nullptr;
  pass_fn r2c_pack = nullptr;
  pass_fn r2c_store = nullptr;
  // Complex-to-real edges, the mirror images.
  pass_fn c2r_pre = nullptr;
  pass_fn c2r_unpack_pairs = nullptr;
  pass_fn c2r_expand = nullptr;
  pass_fn c2r_store = nullptr;
};

} // namespace fftn::inline v0::detail
