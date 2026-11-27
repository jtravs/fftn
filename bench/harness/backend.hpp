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
/// Real-to-real: n reals in, n reals out per axis, the kind of each axis in
/// fbench_case::r2r_kinds (task 6.12).
inline constexpr std::int32_t fbench_r2r = 3;

/// Real-to-real kinds (fbench_case::r2r_kinds), with FFTW's definitions (unnormalised): in the
/// order of FFTN's fftn::r2r_kind, FFTW's REDFT00, REDFT10, REDFT01, REDFT11, RODFT00, RODFT10,
/// RODFT01, RODFT11 and DHT.
inline constexpr std::int32_t fbench_dct1 = 0;
inline constexpr std::int32_t fbench_dct2 = 1;
inline constexpr std::int32_t fbench_dct3 = 2;
inline constexpr std::int32_t fbench_dct4 = 3;
inline constexpr std::int32_t fbench_dst1 = 4;
inline constexpr std::int32_t fbench_dst2 = 5;
inline constexpr std::int32_t fbench_dst3 = 6;
inline constexpr std::int32_t fbench_dst4 = 7;
inline constexpr std::int32_t fbench_dht = 8;

/// Precisions (fbench_case::precision).
inline constexpr std::int32_t fbench_f64 = 0;
inline constexpr std::int32_t fbench_f32 = 1;

/// Results of fbench_begin and fbench_create.
inline constexpr std::int32_t fbench_ok = 0;
/// The library cannot do this case; not an error (the report lists it).
inline constexpr std::int32_t fbench_unsupported = 1;
inline constexpr std::int32_t fbench_error = 2;

extern "C" {

/// One case: `batch` transforms of a row-major array of `rank` dimensions, FFTW's layout
/// conventions. Complex data is interleaved real values of the case's precision. Lengths
/// are in elements: complex elements for complex arrays, real values for real arrays. The
/// harness gives a backend f32 cases only when its describe lists "f32" in "precisions".
struct fbench_case {
  std::int32_t kind;        ///< fbench_c2c, fbench_r2c, fbench_c2r or fbench_r2r
  std::int32_t rank;        ///< 1, 2 or 3
  const std::int64_t* dims; ///< `rank` logical lengths, slowest-varying first
  std::int64_t batch;       ///< number of transforms
  std::int32_t sign;        ///< -1 forward, +1 backward (r2c is -1, c2r is +1)
  std::int32_t in_place;    ///< 1 when the input and output buffers are the same
  std::int64_t in_last;     ///< physical length of the input's last dimension (padding)
  std::int64_t out_last;    ///< physical length of the output's last dimension
  std::int64_t in_dist;     ///< elements between consecutive input transforms
  std::int64_t out_dist;    ///< elements between consecutive output transforms
  std::int32_t precision;   ///< fbench_f64 or fbench_f32
  /// 1, or the interleaved layout (chosen axes of a strided array): the `batch` transforms lie
  /// one element apart, every element of a transform `interleave` (= `batch`) apart, so the
  /// array's fastest dimension is not transformed. Backends other than FFTN's, FFTW's, pocketfft's
  /// and oneMKL's answer fbench_unsupported.
  std::int64_t interleave;
  /// 1, or the number of blocks of `batch` interleaved transforms one after another, each
  /// `outer_in_dist` and `outer_out_dist` elements apart: a middle dimension of an array.
  std::int64_t outer;
  std::int64_t outer_in_dist;
  std::int64_t outer_out_dist;
  /// fbench_r2r: `rank` kinds (fbench_dct1 ... fbench_dht), slowest axis first; otherwise null.
  /// Both sides of a real-to-real case hold n reals per axis (in place too: no padding).
  const std::int32_t* r2r_kinds;
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
/// driver can refuse a binary that resolved the wrong one; and "supports", the list of kinds
/// ("c2c", "r2c", "c2r", "r2r") the backend claims, for which smoke fails on an unsupported case;
/// optionally "precisions", the list of precisions ("f64", "f32") it takes (default f64
/// only); and optionally "output":"delay" when the session times a convolution instead of the
/// case's transform (FFTN's pipeline options): each row's output is then its input delayed by one
/// element, y_j = x_((j - 1) mod n), which the harness checks instead of the DFT (check.hpp,
/// check_delay; rows of c2c and r2c cases only). Valid until fbench_end.
const char* fbench_describe();

/// Plans one case. `in` and `out` are buffers of the case's size and of the alignment
/// every later execute will use; they may be overwritten. On success stores the plan and
/// may write a note on what it chose into `message` (NUL-terminated; the harness reports
/// it with the case as "note"); on failure writes why.
std::int32_t fbench_create(const fbench_case* c, void* in, void* out, void** plan, char* message,
                           std::size_t message_size);

/// Executes a plan on buffers of the same size and alignment as those given to create.
void fbench_execute(const void* plan, void* in, void* out);

/// Releases a plan.
void fbench_destroy(void* plan);
}
