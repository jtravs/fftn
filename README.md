# FFTN

**The Fastest Fourier Transform in the North.**

> FFTN is an experiment under development. Nothing here is released yet, and the API may
> change freely until the first tagged release.

FFTN is a C++20 fast Fourier transform library for CPUs, aimed at simulation codes that run
the same transforms millions of times. It has three aims that existing open-source
libraries do not meet together:

- **Speed on one thread.** As fast as or faster than any open-source FFT on Apple
  M-series, x86-64 (AVX2 and AVX-512) and AArch64 Linux, and close to the vendor libraries
  on their own hardware.
- **Threading that cooperates.** FFTN creates no threads behind your back. It runs on the
  threads your application already has (OpenMP, Kokkos, your own pool), or on a pool you
  start explicitly, and gives the same bits however the work is shared out.
- **Fusion.** Scaling, windows, spectral multipliers, zero padding and your own pointwise
  kernels run inside or beside the transform's passes, while the data are in cache.

## A first transform

```cpp file=examples/guide.cpp region=first
// Eight complex samples of the tone exp(2 pi i j / 8).
const std::size_t n = 8;
std::vector<std::complex<double>> x(n);
for (std::size_t j = 0; j < n; ++j) {
  x[j] = std::polar(1.0, 2 * std::numbers::pi * static_cast<double>(j) / static_cast<double>(n));
}

// The plan: one row of n points, forward. Build it once, execute it as often as needed.
const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, 1, fftn::direction::forward));
// Its scratch, allocated here: execute never allocates.
fftn::workspace ws(plan);

// y_k = sum_j x_j exp(-2 pi i j k / n): 8 at bin 1 and 0 elsewhere (unnormalised, as FFTW).
std::vector<std::complex<double>> y(n);
plan.execute(x.data(), y.data(), ws);

// What the plan runs, in words: instruction set, strategy, passes.
std::printf("%s\n", plan.describe().c_str());
```

## Features

Built and tested; the parts marked provisional have not yet had their API review
([the guide](docs/guide.md#what-is-provisional) lists them):

- Complex and real transforms (c2c, r2c, c2r) of any length, in double and single
  precision, with FFTW's conventions.
- Batched 1-D rows, 2-D planes and 3-D volumes, with strides and distances of either sign,
  in place or out of place, along any chosen axes of a strided array.
- Immutable, shareable plans chosen by rules in microseconds to milliseconds, or by timing
  the candidates, with wisdom to keep the choice; no global state. `plan::describe()` says
  how each plan computes.
- Threads: teams of the caller's threads, executors (OpenMP, Kokkos, FFTN's own pool or
  yours), bitwise the same as serial execution.
- Fusion (provisional): scales, factors and accumulation on a plan's input and output, zero
  padding and cropping by stored runs, block callbacks of your own, and row pipelines
  (convolutions, spectral pairs, fan-out, groups) that read and store each line once.
- Real-to-real transforms (provisional): DCT and DST of types I to IV and the Hartley
  transform.
- Split-complex arrays (provisional), converted in the plan's workspace.
- numpy- and scipy-style one-call functions with cached plans (`fftn/easy.hpp`,
  provisional), a Kokkos adapter shaped like Kokkos-FFT, and a C API (provisional).
- No dependencies: a static library with light headers. GCC 14+, Clang 16+, Apple Clang;
  Windows through Clang. Run-time selection of the instruction set, from SSE-only x86-64
  and plain NEON up to AVX2, AVX-512 and NEON with FCMA, so one build runs well on old and
  new machines.

## How it works, briefly

Kernels are produced by a code generator that verifies each one against an exact DFT and
writes it against a small vector vocabulary implemented per instruction set. A transform
is a short program of passes chosen by a planner from a cost model of the machine, with
optional measurement. The program also defines how work is divided between threads, which
is why results do not depend on the thread count.

## Documents

- [docs/guide.md](docs/guide.md): the user's guide, from a first transform to every feature,
  with links to the pages below.
- [docs/conventions.md](docs/conventions.md): exactly what is computed and how memory is read
  and written.
- [docs/building.md](docs/building.md): building, installing and using FFTN from CMake,
  pkg-config or FetchContent.
- [docs/planning.md](docs/planning.md) and [docs/lengths.md](docs/lengths.md): planning
  effort, wisdom and fast lengths.
- [docs/threading.md](docs/threading.md): teams, executors, OpenMP and thread pools.
- [docs/fusion.md](docs/fusion.md): pointwise work fused with the transforms (edges, block
  callbacks, row pipelines) and owner-computes (provisional).
- [docs/easy.md](docs/easy.md): numpy's and scipy's FFT functions in one call each, with
  cached plans (provisional).
- [docs/kokkos.md](docs/kokkos.md): the Kokkos adapter.
- [docs/c_api.md](docs/c_api.md): the C API (provisional).
- The public headers in [include/fftn/](include/fftn/): the API reference.
- [dev/DESIGN.md](dev/DESIGN.md): goals, evidence, architecture and design decisions.
- [dev/PLAN.md](dev/PLAN.md): the phased implementation and experiment plan, with current
  status.
- [AGENTS.md](AGENTS.md): rules for contributors, human and automated.

## Status

The features above are built and tested on Linux, macOS and Windows. Performance work is
under way, measured against FFTW, Intel oneMKL, Apple Accelerate and others: the current
standing and what comes next are in [dev/PLAN.md](dev/PLAN.md) ("Status"), and the first
release will carry a report of its own.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
