// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/rules.hpp"

#include "isa/arch.hpp"
#include "isa/target.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

// The largest radix that is a single codelet (src/gen/codelets.hpp); above it a line is
// split into passes.
constexpr std::size_t largest_leaf = 16;

bool power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

std::size_t log2_of(std::size_t n) {
  std::size_t k = 0;
  while ((std::size_t{1} << k) < n) {
    ++k;
  }
  return k;
}

bool prime(std::size_t n) {
  if (n < 2) {
    return false;
  }
  for (std::size_t d = 2; d <= n / d; ++d) {
    if (n % d == 0) {
      return false;
    }
  }
  return true;
}

// Removes the factors p of n and returns how many there were.
std::size_t take(std::size_t& n, std::size_t p) {
  std::size_t e = 0;
  while (n % p == 0) {
    n /= p;
    ++e;
  }
  return e;
}

} // namespace

target_profile profile_of(target t) noexcept {
  switch (t) {
  case target::avx512:
  case target::neon:
  case target::neon_fcma:
    return {.registers = 32, .merge = t != target::avx512};
  case target::scalar:
    return {.registers = FFTN_ARCH_AARCH64 ? 32U : 16U, .merge = FFTN_ARCH_AARCH64 != 0};
  case target::x86_v2:
  case target::avx:
  case target::avx2:
    break;
  }
  return {.registers = 16, .merge = false};
}

// FFTL.jl's single-line rules (S = 1), with its looped prime kernel's role taken by the
// generic pass for every prime above 16. In order:
// - n <= 16, or n prime: one pass.
// - Powers of two from 32 to 64: 4, then n/4.
// - Powers of two up to 16384 on 32 registers: radix 8, with one or two 4s first for the
//   leftover factor (2-7 % over radix 4 on the M1; on 16 registers radix 8 lost 3-25 %).
// - Otherwise: the twos as 4s, one 8 when the exponent is odd and nothing else divides n
//   (on x86 also not 7), else one 2; with merging, 3s pair with 4s into 12 and with the
//   spare 2 into 6, and 5s with it into 10; remaining 3s as 9s and a 3; 5s; other primes.
//   The powers of two go first and the rest ascending, largest radix last (39 as 3 x 13 in
//   0.65-0.75 of the time of 13 x 3; median 1.00 against the best order over 306 lengths
//   on the M1, against 1.05-1.14 for the order before).
std::vector<std::size_t> rule_radices(std::size_t n, const target_profile& p) {
  if (n <= 1) {
    return {};
  }
  if (power_of_two(n) && n > largest_leaf && n <= 64) {
    return {4, n / 4};
  }
  if (p.registers >= 32 && power_of_two(n) && n > 64 && n <= 16384) {
    const std::size_t k = log2_of(n);
    std::vector<std::size_t> out;
    std::size_t rest = k;
    if (k % 3 == 2) {
      out.push_back(4);
      rest -= 2;
    } else if (k % 3 == 1) {
      out.insert(out.end(), {4, 4});
      rest -= 4;
    }
    out.insert(out.end(), rest / 3, 8);
    return out;
  }
  if (n <= largest_leaf || prime(n)) {
    return {n};
  }
  std::size_t m = n;
  std::size_t a = take(m, 2);
  std::size_t b = take(m, 3);
  std::size_t c = take(m, 5);
  std::size_t eight = 0;
  std::size_t two = 0;
  if (a % 2 == 1) {
    if (a >= 3 && b + c == 0 && (p.merge || n % 7 != 0)) {
      eight = 1;
      a -= 3;
    } else {
      two = 1;
      a -= 1;
    }
  }
  std::size_t fours = a / 2;
  std::vector<std::size_t> rest;
  while (p.merge && b > 0 && fours > 0) {
    rest.push_back(12);
    --b;
    --fours;
  }
  while (p.merge && b > 0 && two > 0) {
    rest.push_back(6);
    --b;
    --two;
  }
  for (; b >= 2; b -= 2) {
    rest.push_back(9);
  }
  if (b == 1) {
    rest.push_back(3);
  }
  for (; c > 0; --c) {
    if (p.merge && two > 0) {
      rest.push_back(10);
      --two;
    } else {
      rest.push_back(5);
    }
  }
  for (std::size_t q = 7; q <= m / q; q += 2) {
    rest.insert(rest.end(), take(m, q), q);
  }
  if (m > 1) {
    rest.push_back(m);
  }
  std::ranges::sort(rest);
  std::vector<std::size_t> out(eight, 8);
  out.insert(out.end(), fours, 4);
  out.insert(out.end(), two, 2);
  out.insert(out.end(), rest.begin(), rest.end());
  return out;
}

} // namespace fftn::inline v0::detail
