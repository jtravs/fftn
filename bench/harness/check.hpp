// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Case layout, deterministic input data and the correctness check of the fbench harness.
#pragma once

#include "backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace fbench {

// One side (input or output) of a case: logical shape and physical layout in elements.
struct side {
  bool complex = true;
  int rank = 1;
  std::array<std::int64_t, 3> shape{}; // logical; for half spectra the last is n/2+1
  std::int64_t last_phys = 0;          // physical length of the last dimension
  std::int64_t dist = 0;               // elements between transforms
  std::int64_t count = 0;              // logical elements per transform
  std::size_t bytes = 0;               // whole buffer, all transforms, in the case's precision
  // The interleaved layout (fbench_case::interleave): a block holds `block` transforms, each
  // element `stride` apart, and blocks lie `block_dist` apart. Otherwise `block` is the whole
  // batch and `stride` 1.
  std::int64_t stride = 1;
  std::int64_t block = 1;
  std::int64_t block_dist = 0;

  // Element offset of logical flat index `flat` (row-major) in transform `b`.
  [[nodiscard]] std::int64_t offset(std::int64_t b, std::int64_t flat) const {
    const std::int64_t last = shape.at(static_cast<std::size_t>(rank - 1));
    return (b / block) * block_dist + (b % block) * dist +
           stride * ((flat / last) * last_phys + flat % last);
  }
};

struct layout {
  std::int32_t kind = fbench_c2c;
  int rank = 1;
  std::array<std::int64_t, 3> dims{}; // logical transform lengths
  std::int64_t batch = 1;             // all transforms (blocks times `interleave`)
  std::int64_t interleave = 1;        // fbench_case::interleave
  std::int64_t outer = 1;             // fbench_case::outer
  int sign = -1;
  bool in_place = false;
  std::int32_t precision = fbench_f64;
  std::size_t real_bytes = sizeof(double); // of one real value in the case's precision
  std::int64_t n_total = 1;                // product of dims
  side in;
  side out;
  std::size_t buffer_bytes_in = 0;  // allocation for the input (the only one if in place)
  std::size_t buffer_bytes_out = 0; // allocation for the output (0 if in place)

  // Real values in the input allocation and in the buffer holding the output.
  [[nodiscard]] std::size_t reals_in() const { return buffer_bytes_in / real_bytes; }
  [[nodiscard]] std::size_t reals_out() const {
    return (in_place ? buffer_bytes_in : buffer_bytes_out) / real_bytes;
  }

  // The C description for the backend; points into this object.
  [[nodiscard]] fbench_case to_case() const;
};

// Builds a layout with FFTW's conventions: rows contiguous, transforms contiguous, r2c
// output of n/2+1 complex values in the last dimension, in-place real arrays padded to
// 2(n/2+1) reals. With `interleaved`, the layout of fbench_case::interleave instead (not for
// in-place real transforms): `batch` transforms per block, `outer` blocks. Throws
// std::invalid_argument for an impossible case.
layout make_layout(std::int32_t kind, int rank, const std::array<std::int64_t, 3>& dims,
                   std::int64_t batch, int sign, bool in_place, std::int32_t precision,
                   bool interleaved = false, std::int64_t outer = 1);

// The functions below read and write arrays of double whatever the case's precision:
// reals_in() or reals_out() values at the case's element offsets. The harness converts
// single-precision buffers to and from them outside the timer.

// Fills the input with deterministic values in [-1, 1) from `seed`; padding is zero. A
// c2r input is made Hermitian-consistent (self-conjugate entries real, conjugate pairs in
// the k = 0 and k = n/2 columns equal), so every library sees the same signal.
void fill_input(const layout& l, void* in, std::uint64_t seed);

// Multiplies every output value by (1 + eps r) with r = +-1 at random: the test-only
// perturbation that proves the check catches wrong output.
void perturb_output(const layout& l, void* out, double eps, std::uint64_t seed);

struct check_result {
  std::int64_t rows = 0;    // transforms checked (the first and the last)
  std::int64_t samples = 0; // output values compared with the reference, per row
  double err_rms = 0.0;     // ||y - ref|| / ||ref|| over the samples
  double err_max = 0.0;     // max |y - ref| / rms |ref| over the samples
  double parseval = 0.0;    // relative energy error over whole rows
  double tol = 0.0;         // bound on err_rms and parseval (err_max: 8 tol)
  bool pass = false;
};

// Compares the output with a double-double direct DFT of the pristine input at sampled
// outputs (all outputs when n * outputs fits `budget`), and checks Parseval's identity
// over the whole of each checked row. The tolerance takes the epsilon of the case's
// precision.
check_result check_output(const layout& l, const void* in, const void* out, std::int64_t budget,
                          std::uint64_t seed);

// Self-test of the reference and the check (the `selftest` command); returns an empty
// string on success, else what failed.
std::string selftest();

} // namespace fbench
