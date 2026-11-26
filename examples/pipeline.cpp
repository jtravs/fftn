// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Row pipelines (fftn/pipeline.hpp, provisional): a circular convolution of every row with one
// kernel, the forward transform, the kernel's spectrum and the inverse transform in one sweep,
// with the kernel passed at execution. The example quoted in docs/conventions.md ("Pipelines").
// Exits with 0 when the result matches the same work done by two plain plans and a loop.

#include <fftn/fftn.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

// The largest difference between the pipeline and plain plans with a loop, relative to the
// largest output.
double pipeline_error() {
  const std::size_t n = 256;
  const std::size_t count = 8;
  std::vector<std::complex<double>> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {std::cos(0.3 * static_cast<double>(i)), std::sin(0.7 * static_cast<double>(i))};
  }
  // A kernel's spectrum (the forward transform of a short filter, here made up directly).
  std::vector<std::complex<double>> kernel(n);
  for (std::size_t k = 0; k < n; ++k) {
    kernel[k] = std::polar(1.0 / (1.0 + 0.01 * static_cast<double>(k)), 0.1 * static_cast<double>(k));
  }

  // snippet-begin: pipeline
  // Every row convolved with one kernel: forward transform, times the kernel's spectrum
  // (shared by the rows), backward transform, normalised by 1/n. The pipeline records the
  // kernel's kind and strides; its address comes with each execution.
  fftn::pipeline_desc desc = fftn::complex_convolution_rows<double>(n, count);
  desc.middle.scale = 1.0 / static_cast<double>(n);
  const fftn::pipeline pipe(desc);
  fftn::workspace ws(pipe);

  std::vector<std::complex<double>> y(n * count);
  pipe.execute(x.data(), y.data(), ws, {.mid_factor = kernel.data()});
  // snippet-end: pipeline

  // The same work by two plain plans and a loop.
  const fftn::c2c_plan<double> forward(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  const fftn::c2c_plan<double> backward(
      fftn::rows_c2c<double>(n, count, fftn::direction::backward));
  fftn::workspace plain_ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));
  std::vector<std::complex<double>> spectrum(n * count);
  std::vector<std::complex<double>> want(n * count);
  forward.execute(x.data(), spectrum.data(), plain_ws);
  for (std::size_t i = 0; i < spectrum.size(); ++i) {
    spectrum[i] *= kernel[i % n] / static_cast<double>(n);
  }
  backward.execute(spectrum.data(), want.data(), plain_ws);
  double largest = 0;
  double error = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    largest = std::max(largest, std::abs(want[i]));
    error = std::max(error, std::abs(y[i] - want[i]));
  }
  return error / largest;
}

} // namespace

int main() {
  try {
    const double error = pipeline_error();
    std::printf("pipeline against plain plans and a loop: largest error %.3g\n", error);
    return error < 1e-13 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
