// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Plan edges (fftn/edges.hpp, provisional): a window on the input and a normalised spectrum
// added to an accumulator, in one plan, with the window passed at execution. The example quoted
// in docs/conventions.md ("Edges"). Exits with 0 when the result matches the same work done
// by a plain plan and loops.

#include <fftn/fftn.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numbers>
#include <vector>

namespace {

// The largest difference between the plan with edges and a plain plan followed by loops,
// relative to the largest output.
double edges_error() {
  const std::size_t n = 64;
  const std::size_t count = 4;
  std::vector<std::complex<double>> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {std::cos(0.3 * static_cast<double>(i)), std::sin(0.7 * static_cast<double>(i))};
  }

  // snippet-begin: edges
  // A Hann window, one row of it for every row of the batch (batch stride 0).
  std::vector<double> window(n);
  for (std::size_t j = 0; j < n; ++j) {
    const double phase = 2 * std::numbers::pi * static_cast<double>(j) / static_cast<double>(n);
    window[j] = 0.5 - 0.5 * std::cos(phase);
  }

  // Rows times the window, transformed, scaled by 1/n and added to the accumulator. The plan
  // records the edges' kinds and strides; the window's address comes with each execution.
  const fftn::c2c_plan<double> plan(
      fftn::rows_c2c<double>(n, count, fftn::direction::forward), {},
      {.in = {.mul = {.kind = fftn::factor_kind::real, .axis_strides = {1}, .batch_strides = {0}}},
       .out = {.scale = 1.0 / static_cast<double>(n), .accumulate = true}});
  fftn::workspace ws(plan);

  std::vector<std::complex<double>> sum(n * count); // the accumulator, zero to start with
  plan.execute(x.data(), sum.data(), ws, {.in_factor = window.data()});
  plan.execute(x.data(), sum.data(), ws, {.in_factor = window.data()}); // adds it again
  // snippet-end: edges

  // The same work by a plain plan and loops.
  std::vector<std::complex<double>> windowed(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    windowed[i] = x[i] * window[i % n];
  }
  const fftn::c2c_plan<double> plain(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  fftn::workspace plain_ws(plain);
  std::vector<std::complex<double>> y(n * count);
  plain.execute(windowed.data(), y.data(), plain_ws);
  double largest = 0;
  double error = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const std::complex<double> want = 2.0 * y[i] / static_cast<double>(n);
    largest = std::max(largest, std::abs(want));
    error = std::max(error, std::abs(sum[i] - want));
  }
  return error / largest;
}

} // namespace

int main() {
  try {
    const double error = edges_error();
    std::printf("plan with edges against a plain plan and loops: largest error %.3g\n", error);
    return error < 1e-14 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
