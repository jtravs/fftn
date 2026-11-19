// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Transforms along chosen dimensions of a strided array, without moving the data: the example
// quoted in docs/conventions.md. Exits with 0 when both round trips give the input back.

#include <fftn/fftn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

double planes_of_a_volume() {
  // snippet-begin: axes
  // An array of ny x nx x nw complex values, C-ordered: w varies fastest. The planes (y, x), one
  // per w, are transformed where they lie: their elements stand nw apart.
  const std::size_t ny = 12;
  const std::size_t nx = 10;
  const std::size_t nw = 8;
  const std::array<fftn::axis, 3> array{{{.n = ny}, {.n = nx}, {.n = nw}}};
  const std::array<std::size_t, 2> yx{0, 1};
  const auto points = static_cast<double>(ny * nx);
  const fftn::c2c_plan<double> forward(fftn::axes_c2c<double>(array, yx, fftn::direction::forward));
  const fftn::c2c_plan<double> backward(
      fftn::axes_c2c<double>(array, yx, fftn::direction::backward), {.scale = 1.0 / points});
  fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

  std::vector<std::complex<double>> x(ny * nx * nw);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {std::cos(0.3 * static_cast<double>(i)), std::sin(0.7 * static_cast<double>(i))};
  }
  std::vector<std::complex<double>> spectrum(x.size());
  std::vector<std::complex<double>> back(x.size());
  forward.execute(x.data(), spectrum.data(), ws);
  backward.execute(spectrum.data(), back.data(), ws);
  // snippet-end: axes
  double worst = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    worst = std::max(worst, std::abs(back[i] - x[i]));
  }
  return worst;
}

double real_columns() {
  // snippet-begin: axes-real
  // The columns of a real ny x nx array, each of ny reals nx apart, to half spectra of
  // ny/2 + 1 complex values stored the same way. The last axis listed is the halved one.
  const std::size_t ny = 16;
  const std::size_t nx = 6;
  const std::array<fftn::axis, 2> array{{{.n = ny}, {.n = nx}}};
  const std::array<std::size_t, 1> columns{0};
  const fftn::r2c_plan<double> forward(fftn::axes_r2c<double>(array, columns));
  const fftn::c2r_plan<double> backward(fftn::axes_c2r<double>(array, columns),
                                        {.scale = 1.0 / static_cast<double>(ny)});
  fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

  std::vector<double> x(ny * nx);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::cos(0.1 * static_cast<double>(i * i));
  }
  std::vector<std::complex<double>> spectrum((ny / 2 + 1) * nx);
  std::vector<double> back(x.size());
  forward.execute(x.data(), spectrum.data(), ws);
  backward.execute(spectrum.data(), back.data(), ws);
  // snippet-end: axes-real
  double worst = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    worst = std::max(worst, std::abs(back[i] - x[i]));
  }
  return worst;
}

} // namespace

int main() {
  try {
    const double error = std::max(planes_of_a_volume(), real_columns());
    if (error > 1e-12) {
      std::fprintf(stderr, "round trip error %g\n", error);
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
