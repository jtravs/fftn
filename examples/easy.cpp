// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The convenience layer: the examples quoted in docs/easy.md. A spectrum and its inverse by
// numpy's functions, a real 2-D transform of a strided array along chosen axes, the
// frequencies and the shift, and a context of one's own. Exits with 0 when the results are
// right.

#include <fftn/easy.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numbers>
#include <vector>

namespace {

bool one_dimensional() {
  // snippet-begin: one
  // A tone at bin 3 of 16 samples.
  std::vector<double> x(16);
  for (std::size_t j = 0; j < x.size(); ++j) {
    x[j] = std::cos(2 * std::numbers::pi * 3 * static_cast<double>(j) / 16);
  }

  // numpy.fft.rfft(x): 9 values, 8 at bin 3 (unnormalised forward, numpy's default).
  const std::vector<std::complex<double>> spectrum = fftn::easy::rfft(x);

  // numpy.fft.irfft(spectrum, n=16): the tone again, scaled by 1/16 on the way back.
  std::vector<double> back(16);
  fftn::easy::irfft(spectrum, back);

  // Unitary transforms instead: norm="ortho".
  std::vector<std::complex<double>> unitary(9);
  fftn::easy::rfft(x, unitary, fftn::easy::norm::ortho);
  // snippet-end: one
  bool ok = std::abs(spectrum[3] - 8.0) < 1e-12 && std::abs(unitary[3] - 2.0) < 1e-12;
  for (std::size_t j = 0; j < x.size(); ++j) {
    ok = ok && std::abs(back[j] - x[j]) < 1e-14;
  }
  return ok;
}

bool multi_dimensional() {
  // snippet-begin: views
  // A batch of 4 real images of 32 x 48, C-ordered: rfft2 transforms the last two dimensions.
  const std::size_t batch = 4;
  const std::size_t ny = 32;
  const std::size_t nx = 48;
  std::vector<float> images(batch * ny * nx, 1.0F);
  std::vector<std::complex<float>> spectra(batch * ny * (nx / 2 + 1));
  fftn::easy::rfft2(
      fftn::easy::view<const float, 3>(images.data(), {batch, ny, nx}),
      fftn::easy::view<std::complex<float>, 3>(spectra.data(), {batch, ny, nx / 2 + 1}));

  // The same along dimensions 0 and 2 of a padded array (rows 64 apart), the last listed
  // (2) halved, as numpy's axes=(0, 2) or axes=(0, -1).
  std::vector<float> padded(ny * 64, 1.0F);
  std::vector<std::complex<float>> half(ny * (nx / 2 + 1));
  fftn::easy::rfftn(fftn::easy::view<const float, 2>(padded.data(), {ny, nx}, {64, 1}),
                    fftn::easy::view<std::complex<float>, 2>(half.data(), {ny, nx / 2 + 1}),
                    {0, -1});
  // snippet-end: views
  // A constant image has all its weight at bin (0, 0).
  const auto points = static_cast<float>(ny * nx);
  return std::abs(spectra[0] - points) < 1e-2F && std::abs(spectra[1]) < 1e-3F &&
         std::abs(half[0] - points) < 1e-2F && std::abs(half[nx / 2 + 2]) < 1e-3F;
}

bool helpers() {
  // snippet-begin: helpers
  // numpy.fft.fftfreq(8, d=0.5) and the spectrum's zero frequency moved to the middle.
  std::vector<double> f(8);
  fftn::easy::fftfreq(8, f, 0.5); // 0, 0.25, 0.5, 0.75, -1, -0.75, -0.5, -0.25
  std::vector<double> centred(8);
  fftn::easy::fftshift(fftn::easy::view<const double, 1>(f.data(), {8}),
                       fftn::easy::view<double, 1>(centred.data(), {8})); // -1, ..., 0.75
  // snippet-end: helpers
  return f[4] == -1.0 && centred[0] == -1.0 && centred[4] == 0.0 && centred[7] == 0.75;
}

bool own_context() {
  // snippet-begin: context
  // A context of one's own: at most 8 plans, released together with the context.
  fftn::easy::context ctx;
  ctx.set_max_plans(8);
  std::vector<std::complex<double>> a(1000, 1.0);
  std::vector<std::complex<double>> b(1000);
  ctx.fft(a, b);  // builds and keeps the plan of 1000 points
  ctx.ifft(b, a); // another plan: the inverse
  ctx.fft(a, b);  // finds the first plan: allocates nothing
  // snippet-end: context
  return ctx.cached_plans() == 2 && std::abs(b[0] - 1000.0) < 1e-9 && std::abs(a[1] - 1.0) < 1e-12;
}

} // namespace

int main() {
  try {
    if (!one_dimensional() || !multi_dimensional() || !helpers() || !own_context()) {
      std::fputs("easy example: a result is wrong\n", stderr);
      return 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "easy example: %s\n", e.what());
    return 1;
  }
  return 0;
}
