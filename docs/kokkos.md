# Kokkos

`fftn/kokkos.hpp` transforms host-accessible `Kokkos::View`s with FFTN. It is optional and
header-only: `fftn.hpp` does not include it, it compiles only where `Kokkos_Core.hpp` can be
included, and it is compiled in your translation units against your Kokkos, so the FFTN
library itself never depends on Kokkos. Link your target to both `fftn::fftn` and
`Kokkos::kokkos`. It is tested with Kokkos 5.2.1. The declarations, their preconditions and
their errors are documented in the header; this page explains how to use it. The
transforms themselves follow [the conventions](conventions.md).

The header offers two levels:

- **The Kokkos-FFT-shaped plan.** `fftn::kokkos::view_plan` has the constructor and
  execute shape of Kokkos-FFT 2.0's `KokkosFFT::Plan`, and `fftn::kokkos::execute(plan, in,
  out, norm)` is its free `KokkosFFT::execute`. The namespace `fftn::kokkos::compat` spells
  them as Kokkos-FFT does (`Plan`, `Direction`, `Normalization`, `axis_type`, `shape_type`,
  `execute`), so code written for Kokkos-FFT changes only its include and its namespace.
- **View-based construction for the core API.** `shape_of(in, out, dir, axes)` gives the
  `fftn::shape` of a pair of Views, `make_plan` the ordinary `fftn::plan`, and `data_of(view)`
  the pointer to execute it with. Use this level for concurrent executions of one plan (a
  workspace each) or to manage workspaces yourself.

Execution is serial, on the calling thread. Threaded execution on the caller's Kokkos
instance comes later.

## Example

Real rows forward and back, written as for Kokkos-FFT:

```cpp file=examples/kokkos_rows_planes.cpp region=rows
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
```

Batched planes in place, with the adapter's own names:

```cpp file=examples/kokkos_rows_planes.cpp region=planes
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
```

## Moving a Kokkos-FFT call site

For host Views, the include and the namespace change and nothing else:

```diff
-#include <KokkosFFT.hpp>
-namespace kfft = KokkosFFT;
+#include <fftn/kokkos.hpp>
+namespace kfft = fftn::kokkos::compat;

 using plan = kfft::Plan<Kokkos::DefaultHostExecutionSpace, view, view>;
 const plan forward(exec, in, out, kfft::Direction::forward, /*axis=*/-1);
 kfft::execute(forward, in, out, kfft::Normalization::none);
```

`kfft::axis_type<2>{-2, -1}` and the multi-axis constructor carry over the same way.
`kfft::Direction` is Kokkos-FFT's enumeration, with its values 0 and 1, which the plan
converts to `fftn::direction` (the exponent's sign, −1 and +1); `view_plan` takes either. An
application that also builds for GPUs keeps Kokkos-FFT for its device Views and chooses the
namespace per build.

## Views and axes

- **Values.** `float`, `double`, `Kokkos::complex` or `std::complex` of either, forming
  c2c, r2c or c2r in one precision. The output View must be writable. The Views' memory
  must be accessible from the host, and the plan's execution space must run on the host;
  both are checked at compile time.
- **Layouts.** `LayoutRight`, `LayoutLeft` and `LayoutStride` are read through the Views'
  extents and strides, so any of them works, subviews included. Strides of 0 (broadcast)
  are refused, except on dimensions of extent 1, which never step and whose strides are
  ignored.
- **Axes.** Indices into the Views' dimensions, negative ones counting from the end, one to
  three of them, none repeated. They are FFTN's transformed axes in the order given: for real
  transforms the last one listed is halved, as in Kokkos-FFT.
- **Extents.** Equal on both sides, except along the halved axis of a real transform, where
  the complex side has n/2 + 1 for the real side's n. Zero extents are refused.
- **Batch dimensions** are all the other dimensions. Those of extent 1 are dropped, and two
  that nest (one spans the other exactly on both sides, as in any contiguous View) merge
  into one. Any number may remain: a plan loops over each, with no temporary copy.
- **Placement.** Views that start at the same address are transformed in place; Views that
  overlap otherwise are refused. In place, the strides must agree as the
  [conventions](conventions.md#in-place) require: equal for c2c, and FFTW's padded layout for
  real transforms. The real View must show that padding itself: take it as the first n
  columns of a View of 2(n/2 + 1) reals over the complex memory (a subview of the padded rows
  does it, in either layout).

## Normalisation

FFTN's core does not normalise. The plan's `execute` follows Kokkos-FFT instead: its
`normalization` argument defaults to `backward` (backward transforms are multiplied by 1/N,
N the number of points of one transform), and `forward`, `ortho` (1/√N both ways) and
`none` are available. Scaling is a separate pass over the described output after the
transform: each real and imaginary part is multiplied by the coefficient, computed in
`double` and rounded once to the data's precision. With `none` the results are those of
the core plan bit for bit.

## Execution and workspaces

The plan makes its FFTN plan and a workspace when it is constructed; `execute` allocates
nothing (with no Kokkos Tools library loaded, which may allocate for the fence). `execute`
first fences the plan's execution-space instance, so that work queued on it before the call
is complete, then transforms on the calling thread and returns when the output is written.
Because it fences, it must not be called inside a Kokkos kernel: a transform inside a kernel
needs a core plan from `make_plan` and a workspace per thread, the only route that runs
serially inside a kernel for now.

Because a plan owns one workspace, **one plan runs one execution at a time**. A second
execution that starts while one is running throws `std::logic_error` rather than share the
workspace. For concurrent transforms of one shape, build one plan per thread, or build an
`fftn::plan` with `make_plan` and give each execution its own `fftn::workspace`.

`execute` checks the Views it is given against those the plan was built with (extents,
strides, and whether they share their first element), and refuses overlapping Views for an
out-of-place plan, throwing `std::invalid_argument` naming `in` or `out`. Any Views of the
planned layout may be transformed.

## Differences from Kokkos-FFT 2.0

- Serial execution on the calling thread, whatever the execution space; the FFTW back end of
  Kokkos-FFT threads FFTW on OpenMP and Threads builds.
- One execution per plan at a time (above); a concurrent one throws.
- No named Kokkos Tools regions around transforms; only the fence is labelled. `execute`
  must not be called inside a Kokkos kernel.
- `LayoutStride` Views are accepted; Kokkos-FFT takes `LayoutLeft` and `LayoutRight` only.
- In-place real transforms read the real side's padding from its View's strides. Kokkos-FFT
  assumes the padding whatever the real View's strides say, so a tight real View that it
  accepts in place is refused here.
- Cropping and padding are refused: the one-axis `n` and the multi-axis `s` may only repeat
  the Views' own lengths (or be 0 in `s`), and extents that disagree are an error rather
  than a reshape.
- Errors are `std::invalid_argument` naming the argument, at construction and at execution;
  Kokkos-FFT throws `std::runtime_error`.
- The plan's direction for real transforms is checked as in Kokkos-FFT (r2c forward, c2r
  backward).
- Not provided: the one-shot functions (`KokkosFFT::fft`, `rfft`, `fft2` and the rest),
  `DynPlan`, and device Views.
