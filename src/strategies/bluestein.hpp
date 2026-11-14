// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The plan-time tables of a Bluestein stage (DESIGN.md 8.5, 9.2): a transform of q points as
// a chirp multiplication, a cyclic convolution of length M >= 2q - 1 with a fixed chirp, and
// a second chirp multiplication. With w_j = exp(sign pi i j^2 / q), j k = (j^2 + k^2 -
// (k - j)^2) / 2 gives
//
//   X_k = sum_j x_j exp(sign 2 pi i j k / q) = w_k sum_j (x_j w_j) conj(w_(k - j)),
//
// a convolution of a_j = x_j w_j (zero-padded to M) with b_l = conj(w_l) for |l| < q
// (b at l and at M - l, zero between). The chain computes it as
// inverse_dft(dft(a) . dft(b) / M); the kernel spectrum dft(b) / M is fixed by q, M and
// the direction, so the plan holds it. (L. I. Bluestein, IEEE Trans. Audio Electroacoust.
// 18, 1970; L. R. Rabiner, R. W. Schafer and C. M. Rader, IEEE Trans. Audio Electroacoust.
// 17, 1969.)
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace fftn::inline v0::detail {

// w_j = exp(sign pi i j^2 / q) for j < q, each correctly rounded: j^2 is reduced modulo 2q
// in integers and the value is that entry of the root table of 2q (twiddles.hpp). sign is
// -1 or +1.
template <class T>
[[nodiscard]] std::vector<std::complex<T>> bluestein_chirp(int sign, std::size_t q);

// dft(b) / M for the b above, M = roots.size() values, from the double chirp w (q values)
// and the forward root table of M (unit_roots<double>(-1, M)): computed in double by a
// mixed-radix transform over the prime factors of M (direct sums of at most 7 terms), so
// its relative RMS error is that of one FFT of length M in double (DESIGN.md 15.2).
// Precondition: M >= 2q - 1, no prime factor of M above 7.
[[nodiscard]] std::vector<std::complex<double>>
bluestein_kernel(const std::vector<std::complex<double>>& w,
                 const std::vector<std::complex<double>>& roots);

} // namespace fftn::inline v0::detail
