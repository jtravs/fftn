// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Block callbacks (fftn/callback.hpp, provisional): a nonlinear term applied to each output
// bin by a lambda while the row is in cache. The example quoted in docs/conventions.md ("Block
// callbacks"). Exits with 0 when the result matches the same work done by a plain plan and a
// loop.

#include <fftn/fftn.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

// The largest difference between the plan with a callback and a plain plan followed by a loop,
// relative to the largest output.
double callbacks_error() {
  const std::size_t n = 128;
  const std::size_t count = 3;
  std::vector<std::complex<double>> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {std::cos(0.3 * static_cast<double>(i)), std::sin(0.7 * static_cast<double>(i))};
  }

  // snippet-begin: callbacks
  // Each output bin k of transform t times exp(i g |y|^2), a phase that depends on the bin's
  // own value: pure per element, so any team or executor gives the same bits.
  const double g = 0.01;
  auto kerr = [g](const fftn::block_info& info, std::complex<double>* y) noexcept {
    for (std::size_t i = 0; i < info.length; ++i) {
      std::complex<double>& v = y[static_cast<std::ptrdiff_t>(i) * info.element_stride];
      v *= std::polar(1.0, g * std::norm(v));
    }
  };
  const fftn::c2c_plan<double> plan(
      fftn::rows_c2c<double>(n, count, fftn::direction::forward), {},
      {.out = {.call = {.fn = fftn::callback_of<decltype(kerr), std::complex<double>>()}}});
  fftn::workspace ws(plan);

  std::vector<std::complex<double>> y(n * count);
  // The lambda's address reaches the callback as its `user` pointer.
  plan.execute(x.data(), y.data(), ws, {.out_user = &kerr});
  // snippet-end: callbacks

  // The same work by a plain plan and a loop.
  const fftn::c2c_plan<double> plain(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  fftn::workspace plain_ws(plain);
  std::vector<std::complex<double>> want(n * count);
  plain.execute(x.data(), want.data(), plain_ws);
  double largest = 0;
  double error = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const std::complex<double> w = want[i] * std::polar(1.0, g * std::norm(want[i]));
    largest = std::max(largest, std::abs(w));
    error = std::max(error, std::abs(y[i] - w));
  }
  return error / largest;
}

} // namespace

int main() {
  try {
    const double error = callbacks_error();
    std::printf("plan with a callback against a plain plan and a loop: largest error %.3g\n",
                error);
    return error < 1e-13 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
