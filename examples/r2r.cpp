// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Real-to-real plans (provisional, tasks 6.3 to 6.5): the discrete Hartley transform of rows and
// of planes, each its own inverse up to the number of points, the DCT-II of rows undone by the
// DCT-III, and the DCT-IV of rows, its own inverse up to 2n. The example quoted in
// docs/conventions.md ("Real-to-real transforms"). Exits with 0
// when each transform matches its direct sum and the scaled inverse undoes the first.

#include <fftn/fftn.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numbers>
#include <vector>

namespace {

// The largest error of the DHT of one row against its direct sum, and of the round trip,
// relative to the largest value.
double dht_error() {
  const std::size_t n = 48;
  const std::size_t count = 3;
  std::vector<double> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::cos(0.3 * static_cast<double>(i)) + 0.25 * std::sin(1.7 * static_cast<double>(i));
  }

  // snippet-begin: dht
  // H_k = sum_j x_j (cos + sin)(2 pi jk / n) of every row.
  const fftn::r2r_plan<double> dht(fftn::rows_r2r<double>(n, count, fftn::r2r_kind::dht));
  // The same transform scaled by 1/n: the DHT is its own inverse up to n.
  const fftn::r2r_plan<double> inverse(fftn::rows_r2r<double>(n, count, fftn::r2r_kind::dht),
                                       {.scale = 1.0 / static_cast<double>(n)});
  fftn::workspace ws(dht);

  std::vector<double> h(n * count);
  std::vector<double> back(n * count);
  dht.execute(x.data(), h.data(), ws);
  inverse.execute(h.data(), back.data(), ws); // back == x, to rounding
  // snippet-end: dht

  double error = 0;
  double largest = 0;
  for (std::size_t r = 0; r < count; ++r) {
    for (std::size_t k = 0; k < n; ++k) {
      double sum = 0;
      for (std::size_t j = 0; j < n; ++j) {
        const double phase =
            2 * std::numbers::pi * static_cast<double>(j * k % n) / static_cast<double>(n);
        sum += x[r * n + j] * (std::cos(phase) + std::sin(phase));
      }
      error = std::max(error, std::abs(h[r * n + k] - sum));
      largest = std::max(largest, std::abs(sum));
    }
  }
  double round_trip = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    round_trip = std::max(round_trip, std::abs(back[i] - x[i]));
  }
  return std::max(error / largest, round_trip);
}

// The DHT of planes: the separable product of the rows' and the columns' transforms.
double planes_error() {
  const std::size_t ny = 6;
  const std::size_t nx = 10;
  std::vector<double> x(ny * nx);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::sin(0.9 * static_cast<double>(i) + 0.2);
  }

  // snippet-begin: dht-planes
  const fftn::r2r_plan<double> planes(fftn::planes_r2r<double>(ny, nx, 1, fftn::r2r_kind::dht));
  fftn::workspace ws(planes);
  std::vector<double> h(ny * nx);
  planes.execute(x.data(), h.data(), ws);
  // snippet-end: dht-planes

  double error = 0;
  double largest = 0;
  for (std::size_t ky = 0; ky < ny; ++ky) {
    for (std::size_t kx = 0; kx < nx; ++kx) {
      double sum = 0;
      for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t xx = 0; xx < nx; ++xx) {
          const double py =
              2 * std::numbers::pi * static_cast<double>(y * ky % ny) / static_cast<double>(ny);
          const double px =
              2 * std::numbers::pi * static_cast<double>(xx * kx % nx) / static_cast<double>(nx);
          sum += x[y * nx + xx] * (std::cos(py) + std::sin(py)) * (std::cos(px) + std::sin(px));
        }
      }
      error = std::max(error, std::abs(h[ky * nx + kx] - sum));
      largest = std::max(largest, std::abs(sum));
    }
  }
  return error / largest;
}

// The DCT-II of rows (FFTW's REDFT10) and the DCT-III scaled by 1/(2n) that undoes it.
double dct_error() {
  const std::size_t n = 30;
  const std::size_t count = 2;
  std::vector<double> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::sin(0.4 * static_cast<double>(i)) - 0.5 * std::cos(2.3 * static_cast<double>(i));
  }

  // snippet-begin: dct
  // Y_k = 2 sum_j x_j cos(pi (j + 1/2) k / n) of every row.
  const fftn::r2r_plan<double> dct2(fftn::rows_r2r<double>(n, count, fftn::r2r_kind::dct2));
  // The DCT-III undoes it up to 2n.
  const fftn::r2r_plan<double> dct3(fftn::rows_r2r<double>(n, count, fftn::r2r_kind::dct3),
                                    {.scale = 1.0 / (2.0 * static_cast<double>(n))});
  fftn::workspace ws(dct2);

  std::vector<double> y(n * count);
  std::vector<double> back(n * count);
  dct2.execute(x.data(), y.data(), ws);
  dct3.execute(y.data(), back.data(), ws); // back == x, to rounding
  // snippet-end: dct

  double error = 0;
  double largest = 0;
  for (std::size_t r = 0; r < count; ++r) {
    for (std::size_t k = 0; k < n; ++k) {
      double sum = 0;
      for (std::size_t j = 0; j < n; ++j) {
        sum += 2 * x[r * n + j] *
               std::cos(std::numbers::pi * (static_cast<double>(j) + 0.5) * static_cast<double>(k) /
                        static_cast<double>(n));
      }
      error = std::max(error, std::abs(y[r * n + k] - sum));
      largest = std::max(largest, std::abs(sum));
    }
  }
  double round_trip = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    round_trip = std::max(round_trip, std::abs(back[i] - x[i]));
  }
  return std::max(error / largest, round_trip);
}

// The DCT-IV of rows (FFTW's REDFT11), its own inverse up to 2n.
double dct4_error() {
  const std::size_t n = 25;
  const std::size_t count = 2;
  std::vector<double> x(n * count);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::cos(0.7 * static_cast<double>(i)) + 0.3 * std::sin(1.1 * static_cast<double>(i));
  }

  // snippet-begin: dct4
  // Y_k = 2 sum_j x_j cos(pi (j + 1/2)(k + 1/2) / n) of every row.
  const fftn::r2r_plan<double> dct4(fftn::rows_r2r<double>(n, count, fftn::r2r_kind::dct4));
  // The same kind scaled by 1/(2n) undoes it.
  const fftn::r2r_plan<double> inverse(fftn::rows_r2r<double>(n, count, fftn::r2r_kind::dct4),
                                       {.scale = 1.0 / (2.0 * static_cast<double>(n))});
  fftn::workspace ws(dct4);

  std::vector<double> y(n * count);
  std::vector<double> back(n * count);
  dct4.execute(x.data(), y.data(), ws);
  inverse.execute(y.data(), back.data(), ws); // back == x, to rounding
  // snippet-end: dct4

  double error = 0;
  double largest = 0;
  for (std::size_t r = 0; r < count; ++r) {
    for (std::size_t k = 0; k < n; ++k) {
      double sum = 0;
      for (std::size_t j = 0; j < n; ++j) {
        sum += 2 * x[r * n + j] *
               std::cos(std::numbers::pi * (static_cast<double>(j) + 0.5) *
                        (static_cast<double>(k) + 0.5) / static_cast<double>(n));
      }
      error = std::max(error, std::abs(y[r * n + k] - sum));
      largest = std::max(largest, std::abs(sum));
    }
  }
  double round_trip = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    round_trip = std::max(round_trip, std::abs(back[i] - x[i]));
  }
  return std::max(error / largest, round_trip);
}

} // namespace

int main() {
  try {
    const double rows = dht_error();
    const double planes = planes_error();
    const double dct = dct_error();
    const double dct4 = dct4_error();
    std::printf("dht rows: %.2e, planes: %.2e; dct2 and dct3 rows: %.2e; dct4 rows: %.2e\n", rows,
                planes, dct, dct4);
    return rows < 1e-13 && planes < 1e-13 && dct < 1e-13 && dct4 < 1e-13 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
