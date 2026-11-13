// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Transform lengths for the suites, and the switch for long runs.
//
// The gate runs every suite in each of its builds, sanitizers and coverage included, so the
// default lists keep each test to seconds. Setting the environment variable FFTN_TEST_LONG
// to 1 extends them (larger sampled lengths) for runs outside the gate.
//
//   naive_lengths()  every n <= 64 and a sample up to 2048 (up to 8192 in long runs): for
//                    O(n^2) subjects such as the naive DFT.
//   fast_lengths()   every n <= 512 and a sample of larger lengths up to 2^14 (2^18 in long
//                    runs): for FFTW and FFTN in the analytic and reference suites and
//                    for FFTN in the comparison with FFTW.
//   plane_shapes()   (ny, nx) pairs for batched 2-D planes.
//
// The samples mix powers of two, smooth composites, prime powers, primes and awkward
// composites (twice a prime, products of two large primes).
//
// Under ThreadSanitizer the lists are shorter (every n <= 64 for fast subjects, <= 32 for
// O(n^2) ones, and a few larger lengths). These suites are single-threaded, so TSan can
// find nothing in them that ASan and UBSan do not, and its instrumentation makes the
// quadruple-precision arithmetic of the expected values about ten times slower. The
// threading tests of later phases are unaffected.

#pragma once

#include "platform/environment.hpp"

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

#ifdef __SANITIZE_THREAD__
#define FFTN_TEST_UNDER_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define FFTN_TEST_UNDER_TSAN 1
#endif
#endif

namespace fftn::test {

#ifdef FFTN_TEST_UNDER_TSAN
inline constexpr bool under_thread_sanitizer = true;
#else
inline constexpr bool under_thread_sanitizer = false;
#endif

inline bool long_tests() { return fftn::detail::environment_variable("FFTN_TEST_LONG") == "1"; }

inline std::vector<std::size_t> every_length_up_to(std::size_t last) {
  std::vector<std::size_t> v;
  for (std::size_t n = 1; n <= last; ++n) {
    v.push_back(n);
  }
  return v;
}

inline void append(std::vector<std::size_t>& v, std::initializer_list<std::size_t> more) {
  v.insert(v.end(), more);
}

inline std::vector<std::size_t> naive_lengths() {
  if (under_thread_sanitizer) {
    auto v = every_length_up_to(32);
    append(v, {64, 97, 100, 128, 509, 1024});
    return v;
  }
  auto v = every_length_up_to(64);
  append(v, {72, 81, 96, 97, 100, 125, 128, 210, 243, 256});
  append(v, {289, 360, 509, 512, 625, 1000, 1024, 2002, 2048});
  if (long_tests()) {
    append(v, {3000, 4096, 4099, 6561, 8192});
  }
  return v;
}

inline std::vector<std::size_t> fast_lengths() {
  if (under_thread_sanitizer) {
    auto v = every_length_up_to(64);
    append(v, {97, 100, 128, 243, 509, 512, 1000, 1024, 4096, 4097, 16384});
    return v;
  }
  auto v = every_length_up_to(512);
  append(v, {625, 729, 1000, 1024, 1331, 2002, 2048, 2187, 3125});
  append(v, {4096, 4097, 4099, 6561, 8192, 10007, 11025, 16384});
  if (long_tests()) {
    append(v, {32768, 59049, 65536, 65537, 100000, 131072, 262144});
  }
  return v;
}

inline std::vector<std::pair<std::size_t, std::size_t>> plane_shapes() {
  std::vector<std::pair<std::size_t, std::size_t>> v{{1, 1}, {1, 8}, {8, 1}, {2, 2}, {3, 5}};
  v.insert(v.end(), {{4, 4}, {5, 7}, {8, 8}, {6, 10}, {16, 12}, {13, 17}});
  v.insert(v.end(), {{32, 32}, {64, 48}, {96, 96}});
  return v;
}

} // namespace fftn::test
