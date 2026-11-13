// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "dd.hpp"

#include <cstdint>

namespace fbench {
namespace {

// pi / 4 as a double-double.
constexpr dd quarter_pi{7.853981633974483096e-01, 3.061616997868382943e-17};

// sin and cos of phi in [0, pi/4] by their Taylor series; the 15th terms are below 1e-36.
void sin_cos(dd phi, dd& s, dd& c) {
  const dd phi2 = phi * phi;
  dd term = phi; // phi^(2k+1) / (2k+1)!, then phi^(2k) / (2k)! below
  s = term;
  for (int k = 1; k <= 15; ++k) {
    term = (term * phi2) / (static_cast<double>(2 * k) * static_cast<double>(2 * k + 1));
    s = (k % 2 == 1) ? s - term : s + term;
  }
  term = dd{1.0, 0.0};
  c = term;
  for (int k = 1; k <= 15; ++k) {
    term = (term * phi2) / (static_cast<double>(2 * k - 1) * static_cast<double>(2 * k));
    c = (k % 2 == 1) ? c - term : c + term;
  }
}

// num / den for integers below 2^53 (both exact doubles), as a double-double.
dd ratio(std::int64_t num, std::int64_t den) {
  return dd{static_cast<double>(num), 0.0} / static_cast<double>(den);
}

} // namespace

ddc unit_root(std::int64_t m, std::int64_t n, int sign) {
  // 2 pi m / n = octant * pi/4 + phi, with phi = pi/4 * rem / n in [0, pi/4).
  const std::int64_t eighths = 8 * m;
  const std::int64_t octant = eighths / n;
  const std::int64_t rem = eighths % n;
  // Odd octants use the complementary angle pi/4 - phi, so phi stays small and exact.
  const bool odd = octant % 2 == 1;
  const dd phi = quarter_pi * ratio(odd ? n - rem : rem, n);
  dd s;
  dd c;
  sin_cos(phi, s, c);
  dd cos_t;
  dd sin_t;
  switch (octant) {
  case 0:
    cos_t = c;
    sin_t = s;
    break; // phi
  case 1:
    cos_t = s;
    sin_t = c;
    break; // pi/2 - phi'
  case 2:
    cos_t = -s;
    sin_t = c;
    break; // pi/2 + phi
  case 3:
    cos_t = -c;
    sin_t = s;
    break; // pi - phi'
  case 4:
    cos_t = -c;
    sin_t = -s;
    break; // pi + phi
  case 5:
    cos_t = -s;
    sin_t = -c;
    break; // 3pi/2 - phi'
  case 6:
    cos_t = s;
    sin_t = -c;
    break; // 3pi/2 + phi
  default:
    cos_t = c;
    sin_t = -s;
    break; // 2pi - phi'
  }
  return {cos_t, sign < 0 ? -sin_t : sin_t};
}

} // namespace fbench
