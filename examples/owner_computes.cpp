// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Owner-computes (DESIGN.md 12.4; edges, pipelines and callbacks are provisional): in one OpenMP
// parallel region, a row pipeline smooths every real row, the application's own kernel then
// rescales each row by its largest value (a reduction over the whole row, which no edge or
// callback of FFTN's sees at once), and a plan with edges takes each row's windowed, normalised
// spectrum. Every thread runs the same rows in all three steps (`completion::own` and
// `transforms_of`), so a row stays in one core's cache from the first step to the last and the
// threads never wait for each other. The example quoted in docs/fusion.md. Built only when the
// compiler has OpenMP. Exits with 0 when the results are bit for bit those of the same steps run
// serially.

#include <fftn/fftn.hpp>
#include <fftn/omp.hpp>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numbers>
#include <vector>

namespace {

using real_pipeline = fftn::pipeline<double, std::complex<double>, double>;

// The application's own kernel: rows [first, first + count) of y, each divided by its largest
// magnitude.
void normalise_rows(double* y, std::size_t n, std::size_t first, std::size_t count) {
  for (std::size_t r = first; r < first + count; ++r) {
    double* row = y + r * n;
    double peak = 0;
    for (std::size_t j = 0; j < n; ++j) {
      peak = std::max(peak, std::abs(row[j]));
    }
    const double gain = peak > 0 ? 1.0 / peak : 1.0;
    for (std::size_t j = 0; j < n; ++j) {
      row[j] *= gain;
    }
  }
}

// snippet-begin: region
// The rows of x smoothed into y, each normalised by the application's own kernel, then their
// spectra into s: three steps in one parallel region, each thread on the same rows throughout.
void smooth_normalise_transform(const real_pipeline& smooth, const fftn::r2c_plan<double>& spectra,
                                std::size_t n, const double* x, const std::complex<double>* kernel,
                                const double* window, double* y, std::complex<double>* s) {
  // A workspace for each object, sized for the region's team: each serves one execution.
  fftn::workspace smooth_ws(smooth, omp_get_max_threads());
  fftn::workspace spectra_ws(spectra, omp_get_max_threads());

#pragma omp parallel
  {
    const fftn::team me = fftn::omp::this_team();
    // The rows this thread owns: the same in the pipeline and in the plan, which have as many.
    const fftn::transform_range mine = smooth.transforms_of(me);

    // Each step returns once this thread's own rows are written: no barrier between them.
    smooth.execute(x, y, me, smooth_ws, fftn::completion::own, {.mid_factor = kernel});
    normalise_rows(y, n, mine.first, mine.count);
    spectra.execute(y, s, me, spectra_ws, fftn::completion::own, {.in_factor = window});
  } // the region's end: every row of y and s is written
}
// snippet-end: region

bool owner_computes_matches_serial() {
  const std::size_t n = 2048;
  const std::size_t count = 24;
  std::vector<double> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const auto t = static_cast<double>(i);
    x[i] = std::sin(0.01 * t) * (1.0 + 0.5 * std::cos(0.0007 * t)) + 0.3 * std::sin(2.9 * t);
  }
  // A 9-point moving average's spectrum, and a Hann window.
  std::vector<double> filter(n, 0.0);
  std::fill_n(filter.begin(), 9, 1.0 / 9.0);
  const fftn::r2c_plan<double> r2c(fftn::rows_r2c<double>(n, 1));
  fftn::workspace r2c_ws(r2c);
  std::vector<std::complex<double>> kernel(n / 2 + 1);
  r2c.execute(filter.data(), kernel.data(), r2c_ws);
  std::vector<double> window(n);
  for (std::size_t j = 0; j < n; ++j) {
    window[j] = 0.5 - 0.5 * std::cos(2 * std::numbers::pi * static_cast<double>(j) /
                                     static_cast<double>(n));
  }

  // snippet-begin: objects
  // A pipeline smoothing every row (r2c, times the filter's spectrum, c2r, scaled by 1/n), and a
  // plan with edges taking each row's spectrum, windowed and scaled by 1/n. Both have `count`
  // rows, so a team deals them alike.
  fftn::pipeline_desc desc = fftn::convolution_rows<double>(n, count);
  desc.middle.scale = 1.0 / static_cast<double>(n);
  const fftn::pipeline smooth(desc);
  const fftn::r2c_plan<double> spectra(
      fftn::rows_r2c<double>(n, count), {},
      {.in = {.mul = {.kind = fftn::factor_kind::real, .axis_strides = {1}, .batch_strides = {0}}},
       .out = {.scale = 1.0 / static_cast<double>(n)}});

  std::vector<double> y(n * count);
  std::vector<std::complex<double>> s((n / 2 + 1) * count);
  smooth_normalise_transform(smooth, spectra, n, x.data(), kernel.data(), window.data(), y.data(),
                             s.data());
  // snippet-end: objects

  // The same steps, serially.
  fftn::workspace serial_ws(std::max(smooth.workspace_bytes(), spectra.workspace_bytes()));
  std::vector<double> y_serial(n * count);
  std::vector<std::complex<double>> s_serial((n / 2 + 1) * count);
  smooth.execute(x.data(), y_serial.data(), serial_ws, {.mid_factor = kernel.data()});
  normalise_rows(y_serial.data(), n, 0, count);
  spectra.execute(y_serial.data(), s_serial.data(), serial_ws, {.in_factor = window.data()});

  return std::ranges::equal(y, y_serial) && std::ranges::equal(s, s_serial);
}

} // namespace

int main() {
  try {
    const bool same = owner_computes_matches_serial();
    std::printf("owner-computes results %s serial\n", same ? "equal" : "differ from");
    return same ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
