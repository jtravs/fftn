// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// OpenMP: a plan executed by the threads of the caller's parallel region, each then working
// on the rows it owns, and by an executor that opens a region itself; the examples quoted in
// docs/threading.md. Built only when the compiler has OpenMP. Exits with 0 when the results
// are bit for bit those of serial execution.

#include <fftn/fftn.hpp>
#include <fftn/omp.hpp>

#include <omp.h>

#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numbers>
#include <vector>

namespace {

// snippet-begin: region
// Transforms the rows of x into y and divides each by n, in one parallel region.
void transform_and_scale(const fftn::c2c_plan<double>& plan, std::size_t n,
                         const std::complex<double>* x, std::complex<double>* y) {
  // One workspace for the region's team, sized before the region.
  fftn::workspace ws(plan, omp_get_max_threads());

#pragma omp parallel
  {
    const fftn::team me = fftn::omp::this_team();

    // Every thread of the region calls execute; with completion::own each returns once
    // its own rows are written and goes on to work on them, with no barrier in between.
    plan.execute(x, y, me, ws, fftn::completion::own);
    const fftn::transform_range mine = plan.transforms_of(me);
    for (std::size_t i = mine.first * n; i < (mine.first + mine.count) * n; ++i) {
      y[i] /= static_cast<double>(n);
    }
  }
}
// snippet-end: region

bool openmp_matches_serial() {
  const std::size_t n = 4096;
  const std::size_t count = 12;
  std::vector<std::complex<double>> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::polar(1.0, 2 * std::numbers::pi * static_cast<double>(i % 101) / 101.0);
  }
  const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  std::vector<std::complex<double>> serial(n * count);
  fftn::workspace serial_ws(plan);
  plan.execute(x.data(), serial.data(), serial_ws);

  std::vector<std::complex<double>> y(n * count);
  transform_and_scale(plan, n, x.data(), y.data());

  // snippet-begin: executor
  // FFTN opens the region: one region of the executor's constant width per execution, or
  // none when the job is too small to share.
  const fftn::omp::executor omp_lanes;
  fftn::workspace ws(plan, omp_lanes.concurrency());
  std::vector<std::complex<double>> z(n * count);
  plan.execute(x.data(), z.data(), omp_lanes, ws);
  // snippet-end: executor

  for (std::size_t i = 0; i < serial.size(); ++i) {
    if (y[i] != serial[i] / static_cast<double>(n) || z[i] != serial[i]) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  try {
    const bool same = openmp_matches_serial();
    std::printf("OpenMP results %s serial\n", same ? "equal" : "differ from");
    return same ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
