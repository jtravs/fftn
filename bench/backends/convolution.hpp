// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The convolution cases of the PIPELINE options (FFTN's backend and FFTW's; bench/README.md,
// "FFTN options"; experiment X67): how a case's buffers hold the lines, their stored runs and
// their fan-out copies, the kernel spectrum that delays a line by one element, and the user's
// multiply loops, written as plain C++ for the compiler to vectorise. Each backend wraps the
// loops in functions of its own cloned for AVX2 and AVX-512 (FBENCH_EDGE_CLONES), as a user
// building for their own machine would get them.
#pragma once

#include "harness/backend.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

// On x86-64 Linux the loops are cloned for AVX2 and AVX-512 (the bench is compiled for the
// baseline target, SSE2); elsewhere the baseline is already the machine's vector set (NEON).
#if defined(__x86_64__) && defined(__linux__)
#define FBENCH_EDGE_CLONES [[gnu::target_clones("default", "avx2", "avx512f")]]
#else
#define FBENCH_EDGE_CLONES
#endif

namespace fbench::convolution {

// A convolution case: the case's rows (c2c forward or r2c, out of place) as `lines` lines of
// `n` points, each with `fanout` copies (PIPE_FANOUT), and with the valid halves as stored runs
// (PIPE_RUNS=valid): the input run is the m = n/2 points from logical 0, stored at the start
// of the line's input row, and the output run the m points from logical m/2, stored at the
// start of each copy's output row. Line r reads the case's input row r F and copy k writes its
// output row r F + k (X67, amendment 5). Distances are in elements of the line's type (reals
// for an r2c case, whose output row of n/2 + 1 complex values holds the line's reals from its
// start); with the interleaved layout (c2c only, no runs or fan-out) the lines are `stride`
// elements per point, one element apart, in `outer` blocks.
struct layout {
  bool real = false;
  std::size_t n = 0;
  std::size_t bins = 0; // the spectrum's length: n, or n/2 + 1 for a real line
  std::size_t fanout = 1;
  bool valid = false;
  std::size_t m = 0; // the valid half (valid only)
  std::size_t lines = 0;
  std::ptrdiff_t in_dist = 0;   // between lines
  std::ptrdiff_t out_dist = 0;  // between lines
  std::ptrdiff_t copy_dist = 0; // between the copies of a line
  std::ptrdiff_t stride = 1;    // between a line's points (interleave)
  std::size_t outer = 1;
  std::ptrdiff_t outer_in_dist = 0;
  std::ptrdiff_t outer_out_dist = 0;

  // The output's logical first position (valid: m/2) and the points stored per line.
  [[nodiscard]] std::size_t out_logical() const { return valid ? m / 2 : 0; }
  [[nodiscard]] std::size_t in_count() const { return valid ? m : n; }
  [[nodiscard]] std::size_t out_count() const { return valid ? m : n; }
};

// The case as a convolution, or throws std::invalid_argument naming what does not fit: rows of
// a c2c forward or an r2c case, out of place; the interleaved layout for c2c without runs or
// fan-out; runs need n a multiple of 4 (the output run starts at m/2); a batch that F divides.
inline layout layout_of(const fbench_case& c, std::size_t fanout, bool valid) {
  const bool cplx = c.kind == fbench_c2c && c.sign < 0;
  if (c.rank != 1 || c.in_place != 0 || (!cplx && c.kind != fbench_r2c)) {
    throw std::invalid_argument("PIPELINE takes rows of c2c forward and r2c cases, out of place");
  }
  layout l;
  l.real = !cplx;
  l.n = static_cast<std::size_t>(c.dims[0]);
  l.bins = l.real ? l.n / 2 + 1 : l.n;
  l.fanout = fanout;
  l.valid = valid;
  const auto batch = static_cast<std::size_t>(c.batch);
  if (batch % fanout != 0) {
    throw std::invalid_argument("PIPE_FANOUT=" + std::to_string(fanout) +
                                " does not divide the case's batch");
  }
  if (c.interleave > 1) {
    if (l.real || valid || fanout > 1) {
      throw std::invalid_argument(
          "the interleaved layout takes c2c lines without PIPE_RUNS or PIPE_FANOUT");
    }
    l.stride = c.interleave;
    l.outer = static_cast<std::size_t>(c.outer);
    l.outer_in_dist = c.outer_in_dist;
    l.outer_out_dist = c.outer_out_dist;
  }
  if (valid) {
    if (l.n % 4 != 0) {
      throw std::invalid_argument("PIPE_RUNS=valid needs a length that 4 divides");
    }
    l.m = l.n / 2;
  }
  l.lines = batch / fanout;
  const auto f = static_cast<std::ptrdiff_t>(fanout);
  const std::ptrdiff_t out_row = l.real ? 2 * c.out_dist : c.out_dist;
  l.in_dist = f * c.in_dist;
  l.out_dist = f * out_row;
  l.copy_dist = out_row;
  return l;
}

// The kernel's spectrum over `bins` bins of a line of n: a delay by one element, times 1/n (what
// fbench's delayed check expects, harness/check.hpp), `copies` times one after another (a
// kernel array per fan-out copy, each `bins` apart).
template <class T>
std::vector<std::complex<T>> delay_kernel(std::size_t n, std::size_t bins, std::size_t copies = 1) {
  std::vector<std::complex<T>> k(bins * copies);
  for (std::size_t j = 0; j < bins; ++j) {
    const double angle =
        -2.0 * std::numbers::pi * static_cast<double>(j % n) / static_cast<double>(n);
    const auto v = std::complex<T>(std::polar(1.0 / static_cast<double>(n), angle));
    for (std::size_t c = 0; c < copies; ++c) {
      k[c * bins + j] = v;
    }
  }
  return k;
}

// One row of the user's multiply loop: each bin times the kernel's, in place (one restrict
// pointer to the row) or from `src` into a distinct `dst`, the complex arithmetic written out so
// that the compiler vectorises it. Without restrict the compilers version the loop on an overlap
// check, which an in-place call fails, and run the scalar copy (the second reader of task 5.2g).
template <class T>
[[gnu::always_inline]] inline void multiply_row_in_place(T* __restrict row, const T* __restrict k,
                                                         std::size_t bins) {
  for (std::size_t j = 0; j < bins; ++j) {
    const T vr = row[2 * j];
    const T vi = row[2 * j + 1];
    row[2 * j] = vr * k[2 * j] - vi * k[2 * j + 1];
    row[2 * j + 1] = vr * k[2 * j + 1] + vi * k[2 * j];
  }
}

template <class T>
[[gnu::always_inline]] inline void multiply_row_into(T* __restrict row, const T* __restrict from,
                                                     const T* __restrict k, std::size_t bins) {
  for (std::size_t j = 0; j < bins; ++j) {
    const T vr = from[2 * j];
    const T vi = from[2 * j + 1];
    row[2 * j] = vr * k[2 * j] - vi * k[2 * j + 1];
    row[2 * j + 1] = vr * k[2 * j + 1] + vi * k[2 * j];
  }
}

// The user's multiply loop over rows [first, first + count) of a spectrum, rows `dist` complex
// values apart: each bin of `src` times the kernel's (one for every row), into `dst`, which is
// `src` (in place) or an array of its own.
template <class T>
[[gnu::always_inline]] inline void multiply_rows_as(T* dst, const T* src, const T* k,
                                                    std::size_t bins, std::size_t dist,
                                                    std::size_t first, std::size_t count) {
  for (std::size_t r = first; r < first + count; ++r) {
    if (dst == src) {
      multiply_row_in_place(dst + 2 * r * dist, k, bins);
    } else {
      multiply_row_into(dst + 2 * r * dist, src + 2 * r * dist, k, bins);
    }
  }
}

// The same loop over interleaved lines in place (the case's interleaved layout): bin j of the
// `lines` adjacent lines of block b at b block_dist + j stride + line, all times the kernel's
// bin j, the lines the contiguous inner loop.
template <class T>
[[gnu::always_inline]] inline void
multiply_interleaved_as(T* spectrum, const T* __restrict k, std::size_t bins, std::size_t lines,
                        std::ptrdiff_t stride, std::size_t blocks, std::ptrdiff_t block_dist) {
  for (std::size_t b = 0; b < blocks; ++b) {
    for (std::size_t j = 0; j < bins; ++j) {
      T* __restrict row = spectrum + 2 * (static_cast<std::ptrdiff_t>(b) * block_dist +
                                          static_cast<std::ptrdiff_t>(j) * stride);
      const T kr = k[2 * j];
      const T ki = k[2 * j + 1];
      for (std::size_t r = 0; r < lines; ++r) {
        const T vr = row[2 * r];
        const T vi = row[2 * r + 1];
        row[2 * r] = vr * kr - vi * ki;
        row[2 * r + 1] = vr * ki + vi * kr;
      }
    }
  }
}

// The user's loops of FFTW's arm (which has no stored runs): the input run copied into the
// start of a zero-padded line (the zeros written once, when the arm is made), and the output run
// cropped from a whole line.
template <class T>
[[gnu::always_inline]] inline void copy_lines_as(T* __restrict dst, std::ptrdiff_t dst_dist,
                                                 const T* __restrict src, std::ptrdiff_t src_dist,
                                                 std::size_t lines, std::size_t count) {
  for (std::size_t r = 0; r < lines; ++r) {
    const auto rr = static_cast<std::ptrdiff_t>(r);
    T* __restrict to = dst + rr * dst_dist;
    const T* __restrict from = src + rr * src_dist;
    for (std::size_t i = 0; i < count; ++i) {
      to[i] = from[i];
    }
  }
}

} // namespace fbench::convolution
