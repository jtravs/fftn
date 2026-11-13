// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A small Kokkos program: real rows forward and back through the Kokkos-FFT spellings, then
// in-place batched planes through the adapter's own names. The example quoted in
// docs/kokkos.md. Exits with 0 when both round trips give back their input.

#include <fftn/kokkos.hpp>
#include <fftn/shape.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>

namespace {

using host = Kokkos::DefaultHostExecutionSpace;
using complex = Kokkos::complex<double>;

// The largest error of a round trip of `rows` real rows of n points.
double rows_round_trip(const host& exec) {
  const std::size_t rows = 4;
  const std::size_t n = 96;
  // snippet-begin: rows
  namespace kfft = fftn::kokkos::compat; // was: namespace kfft = KokkosFFT;
  using real_rows = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using spectra = Kokkos::View<complex**, Kokkos::LayoutRight, Kokkos::HostSpace>;

  const real_rows x("x", rows, n);
  const spectra y("y", rows, n / 2 + 1);
  const real_rows z("z", rows, n);

  // Plans along the last axis, made once: FFTN plans and workspaces are allocated here.
  const kfft::Plan<host, real_rows, spectra> forward(exec, x, y, kfft::Direction::forward, -1);
  const kfft::Plan<host, spectra, real_rows> backward(exec, y, z, kfft::Direction::backward, -1);

  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < n; ++j) {
      x(r, j) = std::cos(0.1 * static_cast<double>((r + 1) * j));
    }
  }
  kfft::execute(forward, x, y, kfft::Normalization::none);
  kfft::execute(backward, y, z); // Kokkos-FFT's default: backward scaled by 1/n
  // snippet-end: rows

  double error = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < n; ++j) {
      error = std::max(error, std::abs(z(r, j) - x(r, j)));
    }
  }
  return error;
}

// The largest error of an in-place round trip of `count` planes of ny x nx points.
double planes_round_trip(const host& exec) {
  const std::size_t count = 3;
  const std::size_t ny = 15;
  const std::size_t nx = 16;
  // snippet-begin: planes
  using planes = Kokkos::View<complex***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const planes v("v", count, ny, nx);
  const planes original("original", count, ny, nx);

  // In place (the same View twice): a 2-D transform of every plane, over the last two axes.
  const fftn::kokkos::axis_type<2> axes{-2, -1};
  const fftn::kokkos::plan<host, planes, planes, 2> forward(exec, v, v, fftn::direction::forward,
                                                            axes);
  const fftn::kokkos::plan<host, planes, planes, 2> backward(exec, v, v, fftn::direction::backward,
                                                             axes);

  for (std::size_t p = 0; p < count; ++p) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        v(p, iy, ix) = complex(static_cast<double>(p + iy), static_cast<double>(ix % 3));
      }
    }
  }
  Kokkos::deep_copy(original, v);
  forward.execute(v, v, fftn::kokkos::normalization::none);
  backward.execute(v, v); // scaled by 1/(ny nx)
  // snippet-end: planes

  double error = 0;
  for (std::size_t p = 0; p < count; ++p) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        error = std::max(error, Kokkos::abs(v(p, iy, ix) - original(p, iy, ix)));
      }
    }
  }
  return error;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Kokkos::ScopeGuard kokkos(argc, argv);
    const host exec;
    const double rows = rows_round_trip(exec);
    const double planes = planes_round_trip(exec);
    std::printf("largest round-trip errors: rows %.3g, planes %.3g\n", rows, planes);
    return rows < 1e-14 && planes < 1e-13 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
