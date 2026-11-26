// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Row pipelines (fftn/pipeline.hpp, provisional): a circular convolution of every row with one
// kernel, the forward transform, the kernel's spectrum and the inverse transform in one sweep,
// with the kernel passed at execution; for complex rows, and for real rows (r2c, the spectrum's
// n/2 + 1 bins, c2r); and the valid half of a zero-padded (linear) convolution through stored
// runs. The examples quoted in docs/conventions.md ("Pipelines"). Exits with 0 when the results
// match the same work done by two plain plans and a loop, and by the direct convolution.

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
    kernel[k] =
        std::polar(1.0 / (1.0 + 0.01 * static_cast<double>(k)), 0.1 * static_cast<double>(k));
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

// The largest difference between a real pipeline smoothing every row and the direct circular
// convolution, relative to the largest output.
double real_pipeline_error() {
  const std::size_t n = 200;
  const std::size_t count = 6;
  std::vector<double> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::cos(0.05 * static_cast<double>(i)) + 0.3 * std::sin(1.9 * static_cast<double>(i));
  }
  // A 5-point moving average, padded with zeros to the row's length.
  const std::size_t taps = 5;
  std::vector<double> filter(n, 0.0);
  std::fill_n(filter.begin(), taps, 1.0 / static_cast<double>(taps));

  // snippet-begin: real-pipeline
  // The filter's spectrum: its r2c transform, n/2 + 1 bins.
  const fftn::r2c_plan<double> r2c(fftn::rows_r2c<double>(n, 1));
  fftn::workspace r2c_ws(r2c);
  std::vector<std::complex<double>> kernel(n / 2 + 1);
  r2c.execute(filter.data(), kernel.data(), r2c_ws);

  // Every real row circularly convolved with the filter: r2c, times the spectrum, c2r,
  // normalised by 1/n, in one sweep.
  fftn::pipeline_desc desc = fftn::convolution_rows<double>(n, count);
  desc.middle.scale = 1.0 / static_cast<double>(n);
  const fftn::pipeline pipe(desc);
  fftn::workspace ws(pipe);

  std::vector<double> y(n * count);
  pipe.execute(x.data(), y.data(), ws, {.mid_factor = kernel.data()});
  // snippet-end: real-pipeline

  double largest = 0;
  double error = 0;
  for (std::size_t r = 0; r < count; ++r) {
    for (std::size_t j = 0; j < n; ++j) {
      double want = 0;
      for (std::size_t i = 0; i < taps; ++i) {
        want += filter[i] * x[r * n + (j + n - i) % n];
      }
      largest = std::max(largest, std::abs(want));
      error = std::max(error, std::abs(y[r * n + j] - want));
    }
  }
  return error / largest;
}

// The largest difference between a linear convolution through a pipeline's stored runs and the
// direct convolution, relative to the largest output.
double valid_half_error() {
  const std::size_t m = 300;
  const std::size_t count = 4;
  std::vector<double> x(m * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::sin(0.11 * static_cast<double>(i)) + 0.2 * std::cos(2.3 * static_cast<double>(i));
  }
  // A causal filter of 7 taps.
  const std::size_t taps = 7;
  const std::size_t n = 2 * m;
  std::vector<double> filter(n, 0.0);
  for (std::size_t i = 0; i < taps; ++i) {
    filter[i] = 1.0 / static_cast<double>(i + 1);
  }
  const fftn::r2c_plan<double> r2c(fftn::rows_r2c<double>(n, 1));
  fftn::workspace r2c_ws(r2c);
  std::vector<std::complex<double>> kernel(n / 2 + 1);
  r2c.execute(filter.data(), kernel.data(), r2c_ws);

  // snippet-begin: valid-half
  // Rows of m values convolved linearly with the filter: the line of n = 2m holds a row in
  // its first half and zeros beyond, which are not stored or read, and only the first m outputs
  // are stored. The rows are m apart on both sides (the runs' stored extents).
  const std::vector<fftn::stored_run> runs{
      {.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = m},
      {.axis = 0, .where = fftn::side::out, .logical = 0, .stored = 0, .count = m},
  };
  fftn::pipeline_desc desc = fftn::convolution_rows<double>(n, count, runs);
  desc.middle.scale = 1.0 / static_cast<double>(n);
  const fftn::pipeline pipe(desc);
  fftn::workspace ws(pipe);

  std::vector<double> y(m * count);
  pipe.execute(x.data(), y.data(), ws, {.mid_factor = kernel.data()});
  // snippet-end: valid-half

  double largest = 0;
  double error = 0;
  for (std::size_t r = 0; r < count; ++r) {
    for (std::size_t j = 0; j < m; ++j) {
      double want = 0;
      for (std::size_t i = 0; i < taps && i <= j; ++i) {
        want += filter[i] * x[r * m + j - i];
      }
      largest = std::max(largest, std::abs(want));
      error = std::max(error, std::abs(y[r * m + j] - want));
    }
  }
  return error / largest;
}

} // namespace

int main() {
  try {
    const double error = pipeline_error();
    std::printf("pipeline against plain plans and a loop: largest error %.3g\n", error);
    const double real_error = real_pipeline_error();
    std::printf("real pipeline against the direct convolution: largest error %.3g\n", real_error);
    const double half_error = valid_half_error();
    std::printf("valid half against the direct linear convolution: largest error %.3g\n",
                half_error);
    return error < 1e-13 && real_error < 1e-13 && half_error < 1e-13 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
