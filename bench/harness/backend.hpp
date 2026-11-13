// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The interface between the fbench harness and one FFT library (DESIGN.md 16.1).
//
// Each library is linked into its own executable: libraries with the FFTW API export the
// same symbols, and two OpenMP runtimes in one process crash or crawl. The functions have C
// linkage and C-layout arguments, so a backend can be written in any language that exports
// C symbols; the harness (main.cpp) owns everything else: case parsing, buffers, data,
// timing, the correctness check and the output.
//
// Call order in one process: fbench_begin once; then for each case fbench_create, any
// number of fbench_execute, fbench_destroy; finally fbench_end. fbench_describe may be
// called at any time after fbench_begin.
#pragma once

#include <cstddef>
#include <cstdint>

// The values below are the ABI (plain 32-bit integers); the names are for C++ backends.

/// Transform kinds (fbench_case::kind).
inline constexpr std::int32_t fbench_c2c = 0;
inline constexpr std::int32_t fbench_r2c = 1;
inline constexpr std::int32_t fbench_c2r = 2;

/// Results of fbench_begin and fbench_create.
inline constexpr std::int32_t fbench_ok = 0;
/// The library cannot do this case; not an error (the report lists it).
inline constexpr std::int32_t fbench_unsupported = 1;
inline constexpr std::int32_t fbench_error = 2;

extern "C" {

/// One case: `batch` transforms of a row-major array of `rank` dimensions, FFTW's layout
/// conventions. Complex data is interleaved double. Lengths are in elements: complex
/// elements for complex arrays, doubles for real arrays.
struct fbench_case {
  std::int32_t kind;        ///< fbench_c2c, fbench_r2c or fbench_c2r
  std::int32_t rank;        ///< 1, 2 or 3
  const std::int64_t* dims; ///< `rank` logical lengths, slowest-varying first
  std::int64_t batch;       ///< number of transforms
  std::int32_t sign;        ///< -1 forward, +1 backward (r2c is -1, c2r is +1)
  std::int32_t in_place;    ///< 1 when the input and output buffers are the same
  std::int64_t in_last;     ///< physical length of the input's last dimension (padding)
  std::int64_t out_last;    ///< physical length of the output's last dimension
  std::int64_t in_dist;     ///< elements between consecutive input transforms
  std::int64_t out_dist;    ///< elements between consecutive output transforms
};

/// Starts the session. `options` is the arm's backend options as space-separated
/// KEY=VALUE pairs; an unknown key is an error (a typo must not be measured silently).
/// On failure, writes a message (NUL-terminated, at most message_size bytes).
std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size);

/// Ends the session (exports wisdom, releases global state).
void fbench_end();

/// The library's identity as the members of a JSON object, without braces: at least
/// "library", "version" and "isa" (the instruction set actually selected), and
/// "loaded_from" with "expected_root" when the library is a shared library, so that the
/// driver can refuse a binary that resolved the wrong one. Valid until fbench_end.
const char* fbench_describe();

/// Plans one case. `in` and `out` are buffers of the case's size and of the alignment
/// every later execute will use; they may be overwritten. On success stores the plan.
std::int32_t fbench_create(const fbench_case* c, void* in, void* out, void** plan, char* message,
                           std::size_t message_size);

/// Executes a plan on buffers of the same size and alignment as those given to create.
void fbench_execute(const void* plan, void* in, void* out);

/// Releases a plan.
void fbench_destroy(void* plan);
}
