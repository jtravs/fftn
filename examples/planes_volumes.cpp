// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Planes and volumes through the shape functions: a batch of complex planes forward and
// back, and a batch of real volumes to their half spectra and back in place: the example
// quoted in docs/conventions.md. Exits with 0 when both round trips give the input back.

#include <fftn/fftn.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

double planes_round_trip() {
  // snippet-begin: planes
  // Two planes of 6 rows of 10 complex values, one after another; back scaled by 1/60.
  const std::size_t ny = 6;
  const std::size_t nx = 10;
  const std::size_t count = 2;
  const fftn::c2c_plan<double> forward(
      fftn::planes_c2c<double>(ny, nx, count, fftn::direction::forward));
  const fftn::c2c_plan<double> backward(
      fftn::planes_c2c<double>(ny, nx, count, fftn::direction::backward),
      {.scale = 1.0 / static_cast<double>(ny * nx)});
  fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

  std::vector<std::complex<double>> x(ny * nx * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {std::cos(0.3 * static_cast<double>(i)), std::sin(0.7 * static_cast<double>(i))};
  }
  std::vector<std::complex<double>> spectrum(x.size());
  std::vector<std::complex<double>> back(x.size());
  forward.execute(x.data(), spectrum.data(), ws);
  backward.execute(spectrum.data(), back.data(), ws);
  // snippet-end: planes
  double error = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    error = std::max(error, std::abs(back[i] - x[i]));
  }
  return error;
}

double volumes_round_trip() {
  // snippet-begin: volumes
  // Three real volumes of 4 x 5 x 7 values, in place: each real row of 7 is padded to the
  // 2 (7/2 + 1) = 8 reals of its complex row, so a volume takes 4 x 5 x 4 complex values.
  const std::size_t nz = 4;
  const std::size_t ny = 5;
  const std::size_t nx = 7;
  const std::size_t count = 3;
  const std::size_t half = nx / 2 + 1;
  const fftn::r2c_plan<double> forward(fftn::volumes_r2c<double>(nz, ny, nx, count),
                                       {.in_place = true});
  const fftn::c2r_plan<double> backward(
      fftn::volumes_c2r<double>(nz, ny, nx, count),
      {.in_place = true, .scale = 1.0 / static_cast<double>(nz * ny * nx)});
  fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

  const std::size_t rows = nz * ny * count;
  std::vector<std::complex<double>> data(rows * half);
  auto* real = reinterpret_cast<double*>(data.data()); // the padded real rows
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < nx; ++j) {
      real[r * 2 * half + j] = std::cos(0.1 * static_cast<double>(r * nx + j));
    }
  }
  forward.execute(real, data.data(), ws);  // the half spectra, n/2 + 1 values per row
  backward.execute(data.data(), real, ws); // the real volumes again
  // snippet-end: volumes
  double error = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < nx; ++j) {
      error = std::max(error, std::abs(real[r * 2 * half + j] -
                                       std::cos(0.1 * static_cast<double>(r * nx + j))));
    }
  }
  return error;
}

} // namespace

int main() {
  try {
    const double planes = planes_round_trip();
    const double volumes = volumes_round_trip();
    std::printf("largest round-trip error: planes %.3g, volumes %.3g\n", planes, volumes);
    return planes < 1e-14 && volumes < 1e-14 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
