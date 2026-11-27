// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Split-complex data (provisional, task 6.10): the c2c transform of rows kept as two arrays, one
// of real parts and one of imaginary parts, and the r2c transform of real planes into split
// arrays. The example quoted in docs/conventions.md ("Split complex"). Exits with 0 when each
// result agrees with the interleaved plan's to rounding.

#include <fftn/fftn.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

bool rows_agree() {
  const std::size_t n = 100;
  const std::size_t count = 8;
  std::vector<double> re(n * count);
  std::vector<double> im(n * count);
  for (std::size_t i = 0; i < re.size(); ++i) {
    re[i] = std::cos(0.3 * static_cast<double>(i));
    im[i] = std::sin(0.7 * static_cast<double>(i));
  }

  // snippet-begin: rows
  using split = fftn::split_complex<double>;
  // The shape of rows_c2c, its strides and distances counting doubles in each array.
  fftn::shape<split, split> s;
  s.dir = fftn::direction::forward;
  s.axes = {{.n = n}};
  s.batches = {{.count = count}};
  const fftn::plan<split, split> p(s);
  fftn::workspace ws(p);

  std::vector<double> out_re(n * count);
  std::vector<double> out_im(n * count);
  p.execute({re.data(), im.data()}, {out_re.data(), out_im.data()}, ws);
  // snippet-end: rows

  // The interleaved plan of the same rows (the split plan converts into the interleaved form,
  // transforms it there and converts the result back; its strategy may differ, so its last bits).
  std::vector<std::complex<double>> x(n * count);
  std::vector<std::complex<double>> y(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = {re[i], im[i]};
  }
  const fftn::c2c_plan<double> q(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
  fftn::workspace qs(q);
  q.execute(x.data(), y.data(), qs);
  double error = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    error = std::fmax(error, std::abs(y[i] - std::complex<double>(out_re[i], out_im[i])));
  }
  return error <= 1e-12 * static_cast<double>(n);
}

bool planes_agree() {
  const std::size_t ny = 12;
  const std::size_t nx = 20;
  const std::size_t h = nx / 2 + 1;
  std::vector<double> x(ny * nx);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::cos(0.11 * static_cast<double>(i * i % 97));
  }

  // snippet-begin: r2c
  // A real plane into split arrays of ny rows of nx/2 + 1 bins each.
  fftn::shape<double, fftn::split_complex<double>> s;
  s.axes = {{.n = ny}, {.n = nx}};
  const fftn::plan<double, fftn::split_complex<double>> p(s);
  fftn::workspace ws(p);

  std::vector<double> re(ny * h);
  std::vector<double> im(ny * h);
  p.execute(x.data(), {re.data(), im.data()}, ws);
  // snippet-end: r2c

  const fftn::r2c_plan<double> q(fftn::planes_r2c<double>(ny, nx, 1));
  fftn::workspace qs(q);
  std::vector<std::complex<double>> y(ny * h);
  q.execute(x.data(), y.data(), qs);
  double error = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    error = std::fmax(error, std::abs(y[i] - std::complex<double>(re[i], im[i])));
  }
  return error <= 1e-12 * static_cast<double>(ny * nx);
}

} // namespace

int main() {
  try {
    const bool rows = rows_agree();
    const bool planes = planes_agree();
    std::printf("split rows %s, split r2c planes %s\n", rows ? "agree" : "DIFFER",
                planes ? "agree" : "DIFFER");
    return rows && planes ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
