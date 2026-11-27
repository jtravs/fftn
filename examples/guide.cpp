// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The first steps of docs/guide.md, also quoted by README.md: one complex transform, real rows
// there and back, and a refused shape. Exits with 0 when the results are what the guide says.

#include <fftn/fftn.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace {

// The largest distance of the first transform's output from n at bin 1 and 0 elsewhere.
double first_transform_error() {
  // snippet-begin: first
  // Eight complex samples of the tone exp(2 pi i j / 8).
  const std::size_t n = 8;
  std::vector<std::complex<double>> x(n);
  for (std::size_t j = 0; j < n; ++j) {
    x[j] = std::polar(1.0, 2 * std::numbers::pi * static_cast<double>(j) / static_cast<double>(n));
  }

  // The plan: one row of n points, forward. Build it once, execute it as often as needed.
  const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, 1, fftn::direction::forward));
  // Its scratch, allocated here: execute never allocates.
  fftn::workspace ws(plan);

  // y_k = sum_j x_j exp(-2 pi i j k / n): 8 at bin 1 and 0 elsewhere (unnormalised, as FFTW).
  std::vector<std::complex<double>> y(n);
  plan.execute(x.data(), y.data(), ws);

  // What the plan runs, in words: instruction set, strategy, passes.
  std::printf("%s\n", plan.describe().c_str());
  // snippet-end: first

  double error = 0;
  for (std::size_t k = 0; k < n; ++k) {
    const std::complex<double> expected = k == 1 ? static_cast<double>(n) : 0.0;
    error = std::max(error, std::abs(y[k] - expected));
  }
  return error;
}

// The largest error of real rows taken to their half spectra and back, normalised.
double real_round_trip_error() {
  const std::size_t n = 10;
  const std::size_t count = 4;
  std::vector<double> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::sin(0.37 * static_cast<double>(i * i));
  }

  // snippet-begin: real
  // Four real rows of n = 10 samples: r2c gives n/2 + 1 = 6 bins per row, c2r takes them back.
  const std::size_t bins = n / 2 + 1;
  const fftn::r2c_plan<double> forward(fftn::rows_r2c<double>(n, count));
  const fftn::c2r_plan<double> backward(fftn::rows_c2r<double>(n, count),
                                        {.scale = 1.0 / static_cast<double>(n)});
  fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

  std::vector<std::complex<double>> spectrum(bins * count);
  std::vector<double> back(n * count);
  forward.execute(x.data(), spectrum.data(), ws);
  backward.execute(spectrum.data(), back.data(), ws); // back == x, to rounding
  // snippet-end: real

  double error = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    error = std::max(error, std::abs(back[i] - x[i]));
  }
  return error;
}

// Whether plan construction refuses a row of no points with std::invalid_argument.
bool refuses_empty_rows() {
  bool refused = false;
  // snippet-begin: errors
  // Misuse is reported when a plan is constructed, never during execute: here a length of 0.
  try {
    const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(0, 1, fftn::direction::forward));
  } catch (const std::invalid_argument& e) {
    std::printf("refused: %s\n", e.what()); // the message names the offending argument
    refused = true;
  }
  // snippet-end: errors
  return refused;
}

} // namespace

int main() {
  try {
    const double first = first_transform_error();
    const double real = real_round_trip_error();
    const bool refused = refuses_empty_rows();
    std::printf("first transform error %.3g, real round trip error %.3g, refused %d\n", first, real,
                refused ? 1 : 0);
    return first < 1e-13 && real < 1e-14 && refused ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
