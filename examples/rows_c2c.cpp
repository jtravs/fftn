// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A batch of complex rows, forward out of place and back in place: the example quoted in
// docs/conventions.md. Exits with 0 when the results are what the conventions say.

#include <fftn/fftn.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <numbers>
#include <vector>

int main() {
  // snippet-begin: rows
  const std::size_t n = 12;
  const std::size_t count = 3;

  // Plans are built once and reused: three rows of 12 points, one after another.
  const fftn::c2c_plan<double> forward(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  const fftn::c2c_plan<double> backward(fftn::rows_c2c<double>(n, count, fftn::direction::backward),
                                        {.in_place = true});

  // Scratch for one execution at a time, allocated here and never during execute.
  fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

  // Row r holds the tone exp(+2 pi i (r + 1) j / n).
  std::vector<std::complex<double>> x(n * count);
  for (std::size_t r = 0; r < count; ++r) {
    for (std::size_t j = 0; j < n; ++j) {
      const double phase = 2 * std::numbers::pi * double((r + 1) * j) / double(n);
      x[r * n + j] = std::polar(1.0, phase);
    }
  }

  // Forward, out of place: row r of the spectrum is n at bin r + 1 and 0 elsewhere.
  std::vector<std::complex<double>> y(n * count);
  forward.execute(x.data(), y.data(), ws);

  // Backward, in place (the plan says so): FFTN does not normalise, so this gives n * x.
  backward.execute(y.data(), y.data(), ws);
  // snippet-end: rows

  double error = 0;
  for (std::size_t i = 0; i < n * count; ++i) {
    error = std::max(error, std::abs(y[i] / double(n) - x[i]));
  }
  std::printf("largest round-trip error: %.3g\n", error);
  return error < 1e-14 ? 0 : 1;
}
