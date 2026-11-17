// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Choosing a fast length: the example quoted in docs/lengths.md. An application that may pad
// its grid takes the next fast length at or above the size it needs, and can ask how much a
// length costs per point on this machine. Exits with 0 when the results are as documented.

#include <fftn/fftn.hpp>

#include <cstddef>
#include <cstdio>
#include <exception>

namespace {

bool as_documented() {
  // snippet-begin: lengths
  // The grid needs at least 1001 points: pad it to the next fast length.
  const std::size_t n = fftn::next_fast_length(1001); // 1008 = 2^4 * 3^2 * 7
  const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, 1, fftn::direction::forward));

  // Real transforms: an even length whose half is fast.
  const std::size_t r = fftn::next_fast_length(1001, true);

  // This machine's estimate, per point, relative to the power of two at or above the length.
  const double prime = fftn::relative_cost<double>(4093);  // a large prime: several times
  const double padded = fftn::relative_cost<double>(4096); // a power of two: 1
  // snippet-end: lengths

  std::printf("next_fast_length(1001) = %zu, real %zu; relative cost 4093: %.2f, 4096: %.2f; %s\n",
              n, r, prime, padded, plan.describe().substr(0, plan.describe().find('\n')).c_str());
  return n == 1008 && r % 2 == 0 && r >= 1001 && padded == 1.0 && prime > 1.3;
}

} // namespace

int main() {
  try {
    return as_documented() ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "lengths example: %s\n", e.what());
    return 1;
  }
}
