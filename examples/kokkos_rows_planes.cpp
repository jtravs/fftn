// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// A small Kokkos program: real rows forward and back through the Kokkos-FFT spellings, in-place
// batched planes through the adapter's own names, then core plans on Kokkos' threads, through
// the executor and in a team kernel, planes through the one-shot functions, and a pipeline over a
// View's strided columns written from array_of. The example quoted in docs/kokkos.md and
// docs/fusion.md. Exits with 0 when every round trip gives back its input and the pipeline agrees
// with the plans.

#include <fftn/fftn.hpp>
#include <fftn/kokkos.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

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

// The largest error of a round trip of planes through the one-shot functions.
double one_shot_round_trip(const host& exec) {
  const std::size_t ny = 12;
  const std::size_t nx = 20;
  // snippet-begin: one-shot
  namespace kfft = fftn::kokkos::compat;
  const Kokkos::View<double**, Kokkos::HostSpace> a("a", ny, nx);
  const Kokkos::View<complex**, Kokkos::HostSpace> spectrum("spectrum", ny, nx / 2 + 1);
  const Kokkos::View<double**, Kokkos::HostSpace> back("back", ny, nx);
  for (std::size_t i = 0; i < ny; ++i) {
    for (std::size_t j = 0; j < nx; ++j) {
      a(i, j) = std::sin(0.3 * static_cast<double>(i * nx + j));
    }
  }
  // A plan per layout is built on the first call and kept in this thread's convenience context
  // (fftn/easy.hpp); the second round trip finds both.
  for (int round = 0; round < 2; ++round) {
    kfft::rfft2(exec, a, spectrum);
    kfft::irfft2(exec, spectrum, back); // backward scaled by 1/(ny nx), as Kokkos-FFT
  }
  // snippet-end: one-shot
  double error = 0;
  for (std::size_t i = 0; i < ny; ++i) {
    for (std::size_t j = 0; j < nx; ++j) {
      error = std::max(error, std::abs(back(i, j) - a(i, j)));
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
  const fftn::kokkos::view_plan<host, planes, planes, 2> forward(exec, v, v,
                                                                 fftn::direction::forward, axes);
  const fftn::kokkos::view_plan<host, planes, planes, 2> backward(exec, v, v,
                                                                  fftn::direction::backward, axes);

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

// The largest error of a round trip of core plans on the instance's threads: forward through
// the executor, backward in a team kernel.
double threaded_round_trip(const host& exec) {
  const std::size_t rows = 32;
  const std::size_t n = 1024;
  using rows_view = Kokkos::View<complex**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const rows_view x("x", rows, n);
  const rows_view y("y", rows, n);
  const rows_view z("z", rows, n);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < n; ++j) {
      x(r, j) = complex(std::cos(0.01 * static_cast<double>((r + 1) * j)), 0.5);
    }
  }
  // snippet-begin: executor
  namespace fk = fftn::kokkos;
  const auto forward = fk::make_plan(x, y, fftn::direction::forward, fk::axis_type<1>{-1});
  const fk::executor threads(exec); // the instance's threads; width 1 on Kokkos::Serial
  fftn::workspace ws(forward.workspace_bytes(threads.width()));
  forward.execute(fk::data_of(x), fk::data_of(y), threads, ws); // one parallel_for
  // snippet-end: executor

  // snippet-begin: team
  using team_policy = Kokkos::TeamPolicy<host>;
  const auto backward = fk::make_plan(y, z, fftn::direction::backward, fk::axis_type<1>{-1});
  const int team_size = exec.concurrency();
  fftn::workspace team_ws(backward.workspace_bytes(team_size));
  Kokkos::parallel_for(
      "inverse", team_policy(exec, 1, team_size), [&](const team_policy::member_type& member) {
        // Every member makes the call; work done before or after it shares this kernel.
        backward.execute(fk::data_of(y), fk::data_of(z), fk::team_of(member), team_ws);
      });
  // snippet-end: team
  exec.fence();

  double error = 0;
  const auto scale = 1.0 / static_cast<double>(n);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < n; ++j) {
      error = std::max(error, Kokkos::abs(z(r, j) * scale - x(r, j)));
    }
  }
  return error;
}

// The largest difference between a pipeline over a View's strided columns, on the instance's
// threads, and the same convolution by two plans and a loop, relative to the largest output.
double pipeline_error(const host& exec) {
  const std::size_t n = 64;
  const std::size_t count = 24;
  using columns = Kokkos::View<complex**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  const columns x("x", n, count);
  const columns y("y", n, count);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t c = 0; c < count; ++c) {
      x(j, c) =
          complex(std::cos(0.2 * static_cast<double>(j * (c + 1))), 0.1 * static_cast<double>(c));
    }
  }
  // A kernel's spectrum, made up directly.
  std::vector<std::complex<double>> kernel(n);
  for (std::size_t k = 0; k < n; ++k) {
    kernel[k] =
        std::polar(1.0 / (1.0 + 0.1 * static_cast<double>(k)), 0.3 * static_cast<double>(k));
  }

  // snippet-begin: pipeline
  namespace fk = fftn::kokkos;
  // Every column of x (n rows of `count` columns: lines of n points, `count` apart) convolved
  // with one kernel into y. array_of gives one axis per dimension of the Views, with their
  // extents and strides: dimension 0 is the line, dimension 1 the batch.
  const auto array = fk::array_of(x, y);
  fftn::pipeline_desc desc = fftn::complex_convolution_rows<double>(n, count);
  desc.line = array[0];
  desc.batches = {
      {.count = array[1].n, .in_dist = array[1].in_stride, .out_dist = array[1].out_stride},
  };
  desc.middle.scale = 1.0 / static_cast<double>(n);
  const fftn::pipeline pipe(desc);

  const fk::executor threads(exec);
  fftn::workspace ws(pipe.workspace_bytes(threads.width()));
  pipe.execute(fk::data_of(x), fk::data_of(y), threads, ws, {.mid_factor = kernel.data()});
  // snippet-end: pipeline

  // The same work by two plans of the Views and a loop.
  const columns spectrum("spectrum", n, count);
  const columns want("want", n, count);
  const auto forward = fk::make_plan(x, spectrum, fftn::direction::forward, fk::axis_type<1>{0});
  const auto backward =
      fk::make_plan(spectrum, want, fftn::direction::backward, fk::axis_type<1>{0});
  fftn::workspace plain_ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));
  forward.execute(fk::data_of(x), fk::data_of(spectrum), plain_ws);
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t c = 0; c < count; ++c) {
      spectrum(k, c) *= complex(kernel[k].real(), kernel[k].imag()) / static_cast<double>(n);
    }
  }
  backward.execute(fk::data_of(spectrum), fk::data_of(want), plain_ws);
  double largest = 0;
  double error = 0;
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t c = 0; c < count; ++c) {
      largest = std::max(largest, Kokkos::abs(want(j, c)));
      error = std::max(error, Kokkos::abs(y(j, c) - want(j, c)));
    }
  }
  return error / largest;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Kokkos::ScopeGuard kokkos(argc, argv);
    const host exec;
    const double rows = rows_round_trip(exec);
    const double planes = planes_round_trip(exec);
    const double threaded = threaded_round_trip(exec);
    const double one_shot = one_shot_round_trip(exec);
    const double pipeline = pipeline_error(exec);
    std::printf("largest round-trip errors: rows %.3g, planes %.3g, threaded %.3g, one-shot %.3g; "
                "pipeline against plans %.3g\n",
                rows, planes, threaded, one_shot, pipeline);
    return rows < 1e-14 && planes < 1e-13 && threaded < 1e-13 && one_shot < 1e-14 &&
                   pipeline < 1e-14
               ? 0
               : 1;
  } catch (const std::exception& e) {
    std::printf("failed: %s\n", e.what());
    return 1;
  }
}
