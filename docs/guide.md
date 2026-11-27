# User's guide

This guide walks through FFTN from a first transform to the features a simulation code
needs: plans, workspaces, layouts, planning effort and wisdom, threads, fusion, real-to-real
transforms, split-complex data, the convenience layer and the C API. It is a map: each
section says what the feature is for and when to use it, shows the first steps, and links to
the page that states the details. Three kinds of document stand behind it:

- the public headers, `include/fftn/*.hpp` and `include/fftn/c_api.h`, are the API
  reference: every declaration's preconditions and guarantees;
- [conventions.md](conventions.md) states exactly what is computed and how memory is read
  and written;
- the topic pages ([planning](planning.md), [lengths](lengths.md),
  [threading](threading.md), [fusion](fusion.md), [easy](easy.md), [kokkos](kokkos.md),
  [c_api](c_api.md), [building](building.md)) explain one subject each.

Every code block here and on those pages is quoted from a program under `examples/` that the
build compiles and the tests run.

FFTN is unreleased: every API may still change before the first tagged release. Parts marked
**provisional** here, as in their headers, have not yet had the maintainer's review and are
the likeliest to change ("What is provisional" below).

## Building and including

FFTN is a static library with no dependencies. Build and install it with CMake, then link
`fftn::fftn` ([building](building.md)); pkg-config and FetchContent work too. In C++ include
`<fftn/fftn.hpp>`, which brings every core header; the OpenMP adapter, the Kokkos adapter and
the convenience layer have headers of their own (`fftn/omp.hpp`, `fftn/kokkos.hpp`,
`fftn/easy.hpp`). Compile as C++20 or later. No instruction-set
flag is needed or wanted: one build carries kernels for every target of its architecture and
each plan takes the fastest one the processor supports.

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

Three objects do all the work:

- A **shape** (`fftn::shape`, built here by `rows_c2c`) says what to transform: the kind, the
  lengths of up to three transformed axes, any number of batch dimensions, and the strides
  and distances of both sides in memory.
- A **plan** (`fftn::plan`, with the aliases `c2c_plan`, `r2c_plan`, `c2r_plan` and
  `r2r_plan`) is built from a shape and `fftn::options`. All validation, choice and
  allocation happen here. A plan is immutable: copy it, share it, and execute it from several
  threads at once, each with its own workspace.
- A **workspace** (`fftn::workspace`) is the scratch of one execution at a time, at least
  `plan::workspace_bytes()` bytes. It may borrow the caller's memory. A workspace large enough
  for several plans serves them in turn.

`execute` never allocates, locks, throws or does I/O. Transforms follow FFTW's conventions:
the forward transform has the sign −1 in its exponent and nothing is normalised unless the
plan asks (`options::scale`), so backward after forward multiplies by n
([conventions](conventions.md#what-is-computed)). Both precisions are offered: write `float`
for `double` throughout.

`plan::describe()` tells a person how the plan computes: the instruction set, the strategy,
each pass, and anything that made the plan take a slower route than the rules would otherwise
choose. FFTN never computes a shape by a slower route without the description saying so.

Misuse is reported at construction, by `std::invalid_argument` naming the argument:

```cpp file=examples/guide.cpp region=errors
// Misuse is reported when a plan is constructed, never during execute: here a length of 0.
try {
  const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(0, 1, fftn::direction::forward));
} catch (const std::invalid_argument& e) {
  std::printf("refused: %s\n", e.what()); // the message names the offending argument
  refused = true;
}
```

## Rows, planes and volumes

`rows_c2c(n, count, dir)` is a batch of `count` rows, one after another; `planes_c2c` and
`volumes_c2c` are batches of C-ordered planes and volumes, and each has `_r2c`, `_c2r` and
`_r2r` forms (`include/fftn/shape.hpp`). Batching is the normal case: one plan executes the
whole batch, and the planner chooses how (whole rows per vector lane for short rows, chains
of passes, blocked grids for long ones). [Conventions](conventions.md#shapes-and-layouts)
shows planes and volumes, in place and out of place.

For any other layout, fill in the shape's fields yourself: strides and distances in elements,
negative ones included, 0 meaning the tight default. To transform some dimensions of a strided
array where it lies and keep the others as batch dimensions (a field of several components, a
plane of each slice), list the array's dimensions and the ones to transform with `axes_c2c`,
`axes_r2c`, `axes_c2r` or `axes_r2r` ([conventions](conventions.md#chosen-axes-of-a-strided-array)).

In place is a plan option (`options::in_place`); the layouts must then agree as
[conventions](conventions.md#in-place) states, which construction checks.

## Real transforms

r2c takes n real values to the n/2 + 1 complex bins that the Hermitian symmetry leaves, and
c2r takes them back:

```cpp file=examples/guide.cpp region=real
// Four real rows of n = 10 samples: r2c gives n/2 + 1 = 6 bins per row, c2r takes them back.
const std::size_t bins = n / 2 + 1;
const fftn::r2c_plan<double> forward(fftn::rows_r2c<double>(n, count));
const fftn::c2r_plan<double> backward(fftn::rows_c2r<double>(n, count),
                                      {.scale = 1.0 / static_cast<double>(n)});
fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

std::vector<std::complex<double>> spectrum(bins * count);
std::vector<double> back(n * count);
forward.execute(x.data(), spectrum.data(), ws);
backward.execute(spectrum.data(), back.data(), ws); // back == x, to rounding
```

In more dimensions the last axis is halved. In place, a real row is padded to the 2(n/2 + 1)
reals of its complex row, FFTW's layout. A c2r plan never writes its input out of place. Every
rule is in [conventions](conventions.md#what-is-computed).

## Choosing lengths

Every length works; lengths whose prime factors are small are fastest, and large primes go
through Rader's or Bluestein's algorithm, several times slower per point. A grid that may be
padded can take `fftn::next_fast_length(n)` (or its `length_kind::real` form for the halved
axis of real transforms), and `fftn::relative_cost` gives this machine's estimate for a
length ([lengths](lengths.md)).

## Planning effort and wisdom

By default a plan is chosen by rules and cost tables, in microseconds to milliseconds, and the
same shape always gets the same program on the same processor model. With
`options::effort = planning::measure` (or the cheaper `planning::quick`) the plan times the
candidates the rules decided between and keeps the fastest; that pays when a plan is executed
many times. `fftn::wisdom` records measured choices as text, so a later run rebuilds them
without timing, bit for bit ([planning](planning.md)). Pipelines take the same options
(provisional).

## Threads

FFTN creates no thread unless you construct an `fftn::thread_pool`, and never changes a
threading runtime's settings. A plan executes serially unless it is handed one of these
([threading](threading.md)):

- **A team** of threads you already run: inside an OpenMP region each thread calls
  `plan.execute(in, out, fftn::omp::this_team(), ws)`, and any other set of threads does the
  same with its own `fftn::team{rank, size}`. The workspace is shared and sized for the team
  (`workspace(plan, members)`). Every member makes every call.
- **An executor**: one thread calls `plan.execute(in, out, ex, ws)` and FFTN dispatches the
  work on `fftn::omp::executor`, an `fftn::thread_pool`, or an adapter of a few lines over
  your own pool.
- **`execute_transforms(first, count, ...)`**, a range of the batch's transforms, for
  schedulers of your own.

The results are bitwise those of serial execution, whatever the team's size or timing
([conventions](conventions.md#determinism)). With `completion::own` each member runs exactly
the transforms `transforms_of` deals it and returns as soon as they are written, so it can go
straight on to its own work on them ("Owner-computes" in [fusion](fusion.md#owner-computes)).
Kokkos programs have an executor on an execution-space instance and teams from a
`TeamPolicy` kernel ([kokkos](kokkos.md#threads); provisional).

## Fusion: edges, callbacks and pipelines

Provisional. Simulation codes spend much of their time in sweeps over the data around each
transform: a window, a scale, a spectral multiplier, zero padding, a nonlinear step. FFTN can
run that work inside or beside its passes while the data are in cache
([fusion](fusion.md), which says which tool fits which work):

- **Edges** (`plan_edges`): a scale, real or complex factors through strides of your choosing,
  and accumulation into the output, on the input and the output of a plan. The factors'
  addresses come with each execution (`exec_args`), so one plan serves every array of a layout.
- **Stored runs** (`shape::runs`): zero padding and cropping without storing the zeros or the
  outputs you do not want.
- **Block callbacks** (`edge::call`): any pointwise function of yours, called on blocks of the
  input or output between passes: modifying them, producing the input or consuming the output.
- **Pipelines** (`fftn::pipeline`): each line of one axis through a transform, a pointwise
  middle and the inverse transform, reading the line once and storing it once: convolutions
  (`complex_convolution_rows`, `convolution_rows`), spectral pairs, fan-out to several kernels,
  and groups of lines handed to one middle callback.
- **Owner-computes**: `completion::own` and `transforms_of`, for work that needs whole rows,
  several arrays or your own parallel code between transforms.

[Conventions](conventions.md#edges) states their rules (orders, layouts, refusals, accuracy).

## Real-to-real transforms

Provisional. `shape<T, T>` with one `fftn::r2r_kind` per axis computes the DCTs and DSTs of
types I to IV and the discrete Hartley transform, with FFTW's unnormalised definitions
(REDFT00 to RODFT11 and DHT), in rows, planes, volumes and chosen axes, the kinds mixed across
axes if you like (`rows_r2r`, `planes_r2r`, `volumes_r2r`, `axes_r2r`). Each kind's formula,
its inverse and the factor a round trip multiplies by are tabulated in
[conventions](conventions.md#real-to-real-transforms). scipy's functions with their norms are
in the convenience layer ([easy](easy.md#real-to-real-transforms)). The routes are built for
every kind and length and tested against an independent reference; tuning their speed against
other libraries is planned.

## Split-complex data

Provisional. Complex data kept as two real arrays, one of real parts and one of imaginary
parts, are transformed by naming the element `fftn::split_complex<T>` in the shape and passing
`fftn::split_ptr{re, im}` to `execute` (`include/fftn/split.hpp`): c2c, r2c with split output
and c2r with split input, in every rank and form of execution. The plan converts to and from
the interleaved form in its workspace around the transform, so the results are those of the
interleaved plan bit for bit; [conventions](conventions.md#split-complex) gives the layout and
what is not built yet (edges, stored runs, measured planning and wisdom on split plans). A
fused form that reads and writes the split arrays in the transform's own passes is planned.

## The convenience layer

Provisional. `<fftn/easy.hpp>` gives numpy.fft's functions (`fft`, `ifft`, `rfft`, `irfft`
and their 2-D and N-D forms, with numpy's norms, `fftfreq` and the shifts) and scipy.fft's
real-to-real ones (`dct`, `idct`, `dst`, `idst`, the DHT), one call each. Each call finds or
builds a plan and keeps it, with its scratch, for the next call of the same layout. Use it for
scripts, tests and code that calls a transform now and then; use plans where the layout is
known in advance or the call is hot ([easy](easy.md)).

## Kokkos

`<fftn/kokkos.hpp>`, optional and header-only, transforms host-accessible `Kokkos::View`s:
a plan shaped like Kokkos-FFT 2.0's (`view_plan`, and `fftn::kokkos::compat` with Kokkos-FFT's
own names, so a call site changes only its include and namespace), View-based construction of
core plans (`shape_of`, `make_plan`), and Kokkos-FFT's one-shot functions (provisional)
([kokkos](kokkos.md)). A helper giving a View's dimensions as the `axis` list of the `axes_*`
functions (`array_of`) is planned; its design awaits the maintainer.

## The C API

Provisional. `<fftn/c_api.h>`, valid C99 and C++, gives C and any language that can call C the
same plans: shapes of every kind and precision, options, workspaces, the four forms of
execution (serial, a range of transforms, a team, an executor), wisdom, the length queries,
errors through `fftn_error` with no exception crossing the boundary, the fusion objects (edges,
callbacks as C function pointers, pipelines) and split-complex data. A C plan computes what the
C++ plan of the same shape computes, bit for bit ([c_api](c_api.md)). A shared library
exporting only the C API can be built too ([building](building.md#options)).

## What is provisional

Everything is unreleased, but these parts have not yet had the maintainer's review; their
headers say so and they are the likeliest to change before the first release:

| Part | Header | Documented in |
|---|---|---|
| Edges, the vocabulary, stored runs, block callbacks | `edges.hpp`, `callback.hpp`, `shape.hpp` | [fusion](fusion.md), [conventions](conventions.md#edges) |
| Pipelines, their measured planning and wisdom | `pipeline.hpp`, `wisdom.hpp` | [fusion](fusion.md#pipelines), [planning](planning.md#pipelines) |
| Real-to-real kinds | `shape.hpp` | [conventions](conventions.md#real-to-real-transforms) |
| Split complex | `split.hpp` | [conventions](conventions.md#split-complex) |
| The convenience layer | `easy.hpp` | [easy](easy.md) |
| The Kokkos executor, `team_of`, the threaded `view_plan`, the one-shot functions | `kokkos.hpp` | [kokkos](kokkos.md) |
| The C API and its shared library | `c_api.h` | [c_api](c_api.md) |

The rest (shapes of the c2c, r2c and c2r kinds, plans, options, workspaces, teams, executors,
the thread pool, the OpenMP adapter, wisdom for plans, the length queries, `completion::own`
and `transforms_of` on plans) follows the agreed design.

## What is planned

Not built, and so not documented beyond a mention: tuning the speed of the real-to-real
transforms; split-complex data read and written in the transform's own passes; edges, stored
runs and measured planning on split-complex plans; `array_of` for Kokkos Views. FFTN's
progress and what comes next are in [dev/PLAN.md](../dev/PLAN.md).
