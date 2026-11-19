# Kokkos

`fftn/kokkos.hpp` transforms host-accessible `Kokkos::View`s with FFTN. It is optional and
header-only: `fftn.hpp` does not include it, it compiles only where `Kokkos_Core.hpp` can be
included, and it is compiled in your translation units against your Kokkos, so the FFTN
library itself never depends on Kokkos. Link your target to both `fftn::fftn` and
`Kokkos::kokkos`. It is tested with Kokkos 5.2.1. The declarations, their preconditions and
their errors are documented in the header; this page explains how to use it. The
transforms themselves follow [the conventions](conventions.md).

The header offers three levels:

- **The Kokkos-FFT-shaped plan.** `fftn::kokkos::view_plan` has the constructor and
  execute shape of Kokkos-FFT 2.0's `KokkosFFT::Plan`, and `fftn::kokkos::execute(plan, in,
  out, norm)` is its free `KokkosFFT::execute`. The namespace `fftn::kokkos::compat` spells
  them as Kokkos-FFT does (`Plan`, `Direction`, `Normalization`, `axis_type`, `shape_type`,
  `execute`), so code written for Kokkos-FFT changes only its include and its namespace.
- **View-based construction for the core API.** `shape_of(in, out, dir, axes)` gives the
  `fftn::shape` of a pair of Views, `make_plan` the ordinary `fftn::plan`, and `data_of(view)`
  the pointer to execute it with. Use this level for concurrent executions of one plan (a
  workspace each) or to manage workspaces yourself.
- **Threads.** `fftn::kokkos::executor` runs a core plan on an execution-space instance, and
  `fftn::kokkos::team_of` makes the members of a `Kokkos::TeamPolicy` kernel an `fftn::team`.
  `view_plan` executes through the executor already.

The executor, `team_of` and the threaded `view_plan` are **provisional**: their shape has not
yet had its review and may change. The results on any instance width are bitwise those of serial
execution. Plans and executors run on `Kokkos::Serial`, `Kokkos::OpenMP` and `Kokkos::Threads`;
any other execution space is refused at compile time, and so is `Kokkos::Threads` with a Kokkos
other than 5.2.1, because telling whether a thread is inside one of its kernels takes Kokkos'
internal names.

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

## Threads

A `view_plan` runs on the threads of the execution-space instance it was built with, with no
further code. Any core plan does the same through the executor:

```cpp file=examples/kokkos_rows_planes.cpp region=executor
namespace fk = fftn::kokkos;
const auto forward = fk::make_plan(x, y, fftn::direction::forward, fk::axis_type<1>{-1});
const fk::executor threads(exec); // the instance's threads; width 1 on Kokkos::Serial
fftn::workspace ws(forward.workspace_bytes(threads.width()));
forward.execute(fk::data_of(x), fk::data_of(y), threads, ws); // one parallel_for
```

A transform worth sharing costs one `Kokkos::parallel_for` on the instance, labelled
`fftn::bulk` (Kokkos Tools show it, and a `view_plan` on OpenMP or Threads adds a region named
`fftn::kokkos::view_plan::execute`), then a fence. A small job runs on the calling thread
without one, and so does every job at width 1, which is every job on `Kokkos::Serial`.

FFTN allocates nothing while executing. Kokkos allocates nothing for a dispatch on Serial and
OpenMP; its Threads back end allocates short-lived strings on every `parallel_for`. Dispatches on
one instance take turns (the OpenMP back end holds the instance's mutex for each), and
`Kokkos::Threads` accepts them only from the thread that initialised it: from any other thread
the executor runs serially there.

Inside a kernel of the OpenMP or Threads back end nothing is dispatched: the executor reports one
lane and the execution runs serially on the calling thread (with a workspace per thread, as any
concurrent execution). The same holds for `view_plan::execute`, which then neither fences nor
opens a Tools region; each thread needs a plan of its own, because a plan runs one execution at
a time, and the exceptions `execute` documents end the program there, since they cannot leave
the kernel.

A `Kokkos::Serial` kernel cannot be detected. Inside one, use a plan or executor on
`Kokkos::Serial`, which never fences, dispatches or opens a region. A plan on OpenMP or Threads
must not be called from a Serial kernel: it would dispatch from there, and with a Kokkos Tools
library loaded Kokkos' global fence would wait for the kernel for ever.

### Team kernels

When a transform belongs to a kernel that does other work on the same threads, make the members
of a `Kokkos::TeamPolicy` the team, as `fftn::omp::this_team()` does for an OpenMP region:

```cpp file=examples/kokkos_rows_planes.cpp region=team
using team_policy = Kokkos::TeamPolicy<host>;
const auto backward = fk::make_plan(y, z, fftn::direction::backward, fk::axis_type<1>{-1});
const int team_size = exec.concurrency();
fftn::workspace team_ws(backward.workspace_bytes(team_size));
Kokkos::parallel_for(
    "inverse", team_policy(exec, 1, team_size), [&](const team_policy::member_type& member) {
      // Every member makes the call; work done before or after it shares this kernel.
      backward.execute(fk::data_of(y), fk::data_of(z), fk::team_of(member), team_ws);
    });
```

Every member of the team makes every call, from the kernel's body (not from inside `single` or
a nested range), with one workspace per team, sized for the team. With `completion::all`, the
default, each call returns when the whole output is written, and no barrier is needed.
With `completion::own` each member returns once its own transforms (`plan::transforms_of`) are
written, and `member.team_barrier()` is the synchronisation to use before reading the others'.
Measured (experiment [T3](../dev/experiments/T3-kokkos-dispatch.md)): a transform in a team
kernel costs what the executor's dispatch costs, within the runs' noise of a few percent, so use
one when other work shares the kernel, and the executor otherwise. `completion::own` with `team_barrier` is slower: no member
splits a transform, so a batch of few transforms runs on few threads, and the barrier itself
costs more than an OpenMP one (1.5 to 3 µs against 0.6 to 1.3 at 8 to 16 threads).

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
converts to `fftn::direction` (the exponent's sign, −1 and +1); `view_plan` takes either.
`kfft::Normalization` is Kokkos-FFT's enumeration too, converted to `normalization`. An
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

The plan makes its FFTN plan, an executor on its instance and a workspace for the instance's
width when it is constructed; `execute` allocates nothing itself (see
[Threads](#threads) for what Kokkos does, and a loaded Kokkos Tools library may allocate too).
On OpenMP and Threads, `execute` first fences the plan's execution-space instance, so that work
queued on it before the call is complete (`Kokkos::Serial` needs no fence: its kernels have
finished by the time the thread that dispatched them gets here), then transforms on the instance
and returns when the output is written. Scaling a large output is shared among the instance's threads. `execute` is
not `noexcept`: it throws on the checks below, and its fence and dispatch take the instance's
mutex. Inside a kernel it runs serially on the calling thread, without fencing
([Threads](#threads) gives the rules for each back end).

Because a plan owns one workspace, **one plan runs one execution at a time**. A second
execution that starts while one is running throws `std::logic_error` rather than share the
workspace. For concurrent transforms of one shape, build one plan per thread, or build an
`fftn::plan` with `make_plan` and give each execution its own `fftn::workspace`.

`execute` checks the Views it is given against those the plan was built with (extents,
strides, and whether they share their first element), and refuses overlapping Views for an
out-of-place plan, throwing `std::invalid_argument` naming `in` or `out`. Any Views of the
planned layout may be transformed.

## Differences from Kokkos-FFT 2.0

- Threaded execution goes through Kokkos on the plan's instance; the FFTW back end of Kokkos-FFT
  threads FFTW on OpenMP and Threads builds instead.
- One execution per plan at a time (above); a concurrent one throws.
- Inside a kernel, `execute` runs serially on the calling thread.
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
- The plan executes through `execute` (the member, or the free `execute`); Kokkos-FFT's
  public member `execute_impl(in, out, norm)` is not provided.
- Not provided: the one-shot functions (`KokkosFFT::fft`, `rfft`, `fft2` and the rest),
  `DynPlan`, and device Views.
