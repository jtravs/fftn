// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Row pipelines (fftn/pipeline.hpp, provisional): a circular convolution of every row with one
// kernel, the forward transform, the kernel's spectrum and the inverse transform in one sweep,
// with the kernel passed at execution; for complex rows, and for real rows (r2c, the spectrum's
// n/2 + 1 bins, c2r); the valid half of a zero-padded (linear) convolution through stored runs;
// a fan-out, several kernels on one forward transform; and a block callback between the transforms
// (task 5.3d). The examples quoted in docs/conventions.md ("Pipelines"). Exits with 0 when the
// results match the same work done by two plain plans and a loop, and by the direct convolution.

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

// The largest difference between a fan-out (one forward transform per row, three kernels, three
// outputs) and three pipelines of one kernel each, relative to the largest output.
double fanout_error() {
  const std::size_t n = 128;
  const std::size_t count = 4;
  const std::size_t copies = 3;
  std::vector<std::complex<double>> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {std::sin(0.2 * static_cast<double>(i)), std::cos(0.9 * static_cast<double>(i))};
  }
  // Three kernels' spectra, one after another.
  std::vector<std::complex<double>> kernels(n * copies);
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    kernels[i] =
        std::polar(1.0 / (1.0 + 0.02 * static_cast<double>(i % n)), 0.05 * static_cast<double>(i));
  }

  // snippet-begin: fanout
  // Every row transformed once, then convolved with three kernels: copy k multiplies the
  // row's spectrum by kernel k (n bins on) and writes a whole batch of rows (n * count
  // elements on).
  fftn::pipeline_desc desc = fftn::complex_convolution_rows<double>(n, count);
  desc.middle.scale = 1.0 / static_cast<double>(n);
  desc.fanout = copies;
  desc.fanout_factor_dist = static_cast<std::ptrdiff_t>(n);
  desc.fanout_out_dist = static_cast<std::ptrdiff_t>(n * count);
  const fftn::pipeline pipe(desc);
  fftn::workspace ws(pipe);

  std::vector<std::complex<double>> y(n * count * copies);
  pipe.execute(x.data(), y.data(), ws, {.mid_factor = kernels.data()});
  // snippet-end: fanout

  // The same by one pipeline per kernel.
  fftn::pipeline_desc one = fftn::complex_convolution_rows<double>(n, count);
  one.middle.scale = 1.0 / static_cast<double>(n);
  const fftn::pipeline single(one);
  fftn::workspace single_ws(single);
  std::vector<std::complex<double>> want(n * count * copies);
  for (std::size_t k = 0; k < copies; ++k) {
    single.execute(x.data(), want.data() + k * n * count, single_ws,
                   {.mid_factor = kernels.data() + k * n});
  }
  double largest = 0;
  double error = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    largest = std::max(largest, std::abs(want[i]));
    error = std::max(error, std::abs(y[i] - want[i]));
  }
  return error / largest;
}

// The largest difference between a pipeline whose middle is a block callback (a gain that saturates
// with each bin's power, which no factor can express) and two plain plans with the same loop
// between them, relative to the largest output.
double middle_callback_error() {
  const std::size_t n = 128;
  const std::size_t count = 4;
  std::vector<std::complex<double>> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {std::cos(0.4 * static_cast<double>(i)), std::sin(0.1 * static_cast<double>(i))};
  }
  const auto gain = [](std::complex<double> y) { return y / (1.0 + std::norm(y) / 1000.0); };

  // snippet-begin: middle-callback
  // Between the transforms, a callback on each row's spectrum (the whole row in one call, in
  // natural order), then the scale 1/n. The lambda's address arrives with each execution as
  // the callback's `user` pointer.
  auto saturate = [&gain](const fftn::block_info& info, std::complex<double>* bins) noexcept {
    for (std::size_t b = 0; b < info.length; ++b) {
      std::complex<double>& y = bins[static_cast<std::ptrdiff_t>(b) * info.element_stride];
      y = gain(y);
    }
  };
  fftn::pipeline_desc desc = fftn::spectral_pair_rows<double>(n, count, fftn::direction::forward);
  desc.middle.scale = 1.0 / static_cast<double>(n);
  desc.middle.call = {.fn = fftn::callback_of<decltype(saturate), std::complex<double>>()};
  const fftn::pipeline pipe(desc);
  fftn::workspace ws(pipe);

  std::vector<std::complex<double>> y(n * count);
  pipe.execute(x.data(), y.data(), ws, {.mid_user = &saturate});
  // snippet-end: middle-callback

  // The same work by two plain plans and a loop.
  const fftn::c2c_plan<double> forward(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  const fftn::c2c_plan<double> backward(
      fftn::rows_c2c<double>(n, count, fftn::direction::backward));
  fftn::workspace plain_ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));
  std::vector<std::complex<double>> spectrum(n * count);
  std::vector<std::complex<double>> want(n * count);
  forward.execute(x.data(), spectrum.data(), plain_ws);
  for (std::complex<double>& v : spectrum) {
    v = gain(v) / static_cast<double>(n);
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
    const double real_error = real_pipeline_error();
    std::printf("real pipeline against the direct convolution: largest error %.3g\n", real_error);
    const double half_error = valid_half_error();
    std::printf("valid half against the direct linear convolution: largest error %.3g\n",
                half_error);
    const double fan_error = fanout_error();
    std::printf("fan-out against one pipeline per kernel: largest error %.3g\n", fan_error);
    const double callback_error = middle_callback_error();
    std::printf("middle callback against plain plans and a loop: largest error %.3g\n",
                callback_error);
    return error < 1e-13 && real_error < 1e-13 && half_error < 1e-13 && fan_error < 1e-13 &&
                   callback_error < 1e-13
               ? 0
               : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
