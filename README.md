# FFTN

**The Fastest Fourier Transform in the North.**

> FFTN is an experiment under development. Nothing here is released, the API will change,
> and the performance figures below are targets, not results.

FFTN is a C++20 fast Fourier transform library for CPUs, aimed at simulation codes that run
the same transforms millions of times. It has three aims that existing open-source
libraries do not meet together:

- **Speed on one thread.** As fast as or faster than any open-source FFT on Apple
  M-series, x86-64 (AVX2 and AVX-512) and AArch64 Linux, and close to the vendor libraries
  on their own hardware.
- **Threading that cooperates.** FFTN creates no threads of its own. It runs on the
  threads your application already has (OpenMP, Kokkos, your own pool), scales with cores,
  and gives the same bits however the work is shared out.
- **Fusion.** Scaling, windows, spectral multipliers, zero padding and your own pointwise
  kernels run inside or beside the transform's passes, while the data are in cache.

## Planned features

- Complex and real transforms (c2c, r2c, c2r) of any length, in double and single
  precision.
- Batched 1-D rows, 2-D planes and 3-D volumes, with strides and distances, in place or
  out of place, along any axis.
- Fused convolutions and forward–inverse pairs for split-step and pseudo-spectral methods.
- Immutable, shareable plans that are cheap to create; no global state.
- No dependencies: a small static library with light headers. GCC 14+, Clang, Apple
  Clang; Windows through clang.
- Run-time selection of the instruction set, from SSE-only x86-64 and plain NEON up to
  AVX2, AVX-512 and NEON with FCMA, so one build runs well on old and new machines.

## How it works, briefly

Kernels are produced by a code generator that verifies each one against an exact DFT and
writes it against a small vector vocabulary implemented per instruction set. A transform
is a short program of passes chosen by a planner from a cost model of the machine, with
optional measurement. The program also defines how work is divided between threads, which
is why results do not depend on the thread count.

## Documents

- [dev/DESIGN.md](dev/DESIGN.md): goals, evidence, architecture and design decisions.
- [dev/PLAN.md](dev/PLAN.md): the phased implementation and experiment plan, with current
  status.
- [AGENTS.md](AGENTS.md): rules for contributors, human and automated.

## Status

Design stage. See [dev/PLAN.md](dev/PLAN.md) for what exists and what is next.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
