// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "strategies/bluestein.hpp"

#include "core/twiddles.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

// The forward DFT of the n values x[0], x[stride], ... into out[0 .. n), out of place,
// decimation in time over the smallest prime factor of n; w is the root table of `total`
// points (exp(-2 pi i t / total)), and n divides total.
void plan_time_dft(const std::complex<double>* x, std::size_t stride, std::size_t n,
                   std::complex<double>* out, const std::vector<std::complex<double>>& w,
                   std::size_t total) {
  if (n < 2) {
    std::copy_n(x, n, out); // n is 1 here (it divides total); 0 would copy nothing
    return;
  }
  std::size_t p = 2;
  while (n % p != 0) {
    ++p;
  }
  const std::size_t m = n / p;
  for (std::size_t r = 0; r < p; ++r) {
    plan_time_dft(x + r * stride, stride * p, m, out + r * m, w, total);
  }
  // out[r m + k] holds bin k of the DFT of length m of residue r; bin k + t m of the whole
  // is sum_r W^(r (k + t m)) Y_r[k], W = exp(-2 pi i / n) = w^(total / n).
  const std::size_t step = total / n;
  std::vector<std::complex<double>> y(p);
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t r = 0; r < p; ++r) {
      y[r] = out[r * m + k];
    }
    for (std::size_t t = 0; t < p; ++t) {
      const std::size_t bin = k + t * m;
      std::complex<double> sum = y[0];
      for (std::size_t r = 1; r < p; ++r) {
        sum += y[r] * w[(r * bin % n) * step];
      }
      out[bin] = sum;
    }
  }
}

} // namespace

template <class T> std::vector<std::complex<T>> bluestein_chirp(int sign, std::size_t q) {
  // (q - j)^2 = j^2 + q (q - 2 j), so w_(q - j) = (-1)^q w_j exactly: half the entries are
  // computed, the others are copies or negations.
  std::vector<std::complex<T>> w(q);
  std::size_t square = 0; // j^2 mod 2q
  for (std::size_t j = 0; 2 * j <= q; ++j) {
    w[j] = unit_root<T>(sign, square, 2 * q);
    if (j != 0) {
      w[q - j] = q % 2 == 0 ? w[j] : -w[j];
    }
    square = (square + 2 * j + 1) % (2 * q); // (j + 1)^2 = j^2 + 2 j + 1
  }
  return w;
}

std::vector<std::complex<double>> bluestein_kernel(const std::vector<std::complex<double>>& w,
                                                   const std::vector<std::complex<double>>& roots) {
  const std::size_t q = w.size();
  const std::size_t m = roots.size();
  std::vector<std::complex<double>> b(m);
  b[0] = std::conj(w[0]);
  for (std::size_t l = 1; l < q; ++l) {
    b[l] = std::conj(w[l]);
    b[m - l] = std::conj(w[l]);
  }
  std::vector<std::complex<double>> spectrum(m);
  plan_time_dft(b.data(), 1, m, spectrum.data(), roots, m);
  const double inverse = 1.0 / static_cast<double>(m);
  for (std::complex<double>& v : spectrum) {
    v *= inverse;
  }
  return spectrum;
}

std::vector<std::complex<double>> rader_kernel(int sign, std::size_t q, std::size_t g,
                                               const std::vector<std::complex<double>>& roots) {
  const std::size_t len = roots.size();
  // g^-1 = g^(L - 1) mod q; q < 2^32, so the products fit in 64 bits.
  std::uint64_t inverse = 1;
  for (std::size_t i = 0; i + 1 < len; ++i) {
    inverse = inverse * g % q;
  }
  std::vector<std::complex<double>> b(len);
  std::uint64_t e = 1; // g^-l mod q
  for (std::size_t l = 0; l < len; ++l) {
    b[l] = unit_root<double>(sign, static_cast<std::size_t>(e), q);
    e = e * inverse % q;
  }
  std::vector<std::complex<double>> spectrum(len);
  plan_time_dft(b.data(), 1, len, spectrum.data(), roots, len);
  const double scale = 1.0 / static_cast<double>(len);
  for (std::complex<double>& v : spectrum) {
    v *= scale;
  }
  return spectrum;
}

template std::vector<std::complex<double>> bluestein_chirp<double>(int, std::size_t);
template std::vector<std::complex<float>> bluestein_chirp<float>(int, std::size_t);

} // namespace fftn::inline v0::detail
