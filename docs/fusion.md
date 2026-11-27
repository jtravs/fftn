# Fusion

FFTN runs pointwise work while the data are in cache: in its own passes, between them, or on
the same core right after them. This page says which tool fits which work, how the tools fit
together and what is built. The exact rules (orders, layouts, refusals, accuracy) are in
[conventions](conventions.md): "Edges", "Block callbacks", "Stored runs" and "Pipelines". The
declarations, preconditions and guarantees are in the public headers, the API reference:
`include/fftn/edges.hpp`, `include/fftn/callback.hpp`, `include/fftn/pipeline.hpp`,
`include/fftn/shape.hpp` (stored runs) and the team overloads of `plan::execute`. The design
and its reasons are in [DESIGN section 12](../dev/DESIGN.md#12-fusion-and-pipelines). C
programs have the same objects through `fftn/c_api.h` ([c_api.md](c_api.md#fusion)).

**Provisional.** Everything on this page except `completion::own` and `transforms_of` on plans
is provisional, as its headers say: edges, the vocabulary, stored runs, block callbacks and
pipelines may change before the first release.

## Which tool

| The work | The tool | What it saves |
|---|---|---|
| A scale, a window, a filter or a spectral multiplier, the same layout every time | an **edge**'s vocabulary: scales, real and complex factors, accumulation | the separate sweep over the array; often the multiply itself, fused into a pass |
| Zeros on the input you do not want to store, outputs you do not want | **stored runs** of an axis | the zeros' memory and reads, the unwanted outputs' writes |
| Any other pointwise function: nonlinear, computed on the fly, a reduction into your own memory | a **block callback** on an edge | the sweep: your function runs on blocks still in cache |
| A transform, a pointwise middle and the inverse along one axis (a convolution, a spectral multiplier, a split-step's linear step) | a row **pipeline** | two sweeps: the spectrum never reaches memory |
| Work that needs a whole row at once, several of your arrays, or your own threading between transforms | **owner-computes**: `completion::own` and `transforms_of` | the barrier, and the cache: each row stays on the core that transformed it |

The tools combine: a pipeline takes the vocabulary on its prologue, middle and epilogue,
stored runs on its line and callbacks in all three places; a plan with edges and a pipeline
both take part in owner-computes.

## The vocabulary and edges

A plan takes an input edge and an output edge (`plan_edges`). Each holds a `scale`, a factor
(`factor`: real or complex values indexed by logical coordinates through strides of your
choosing, 0 broadcasting) and, on the output, `accumulate`. The plan records the factor's
kind and strides; each execution passes the array's address in `exec_args`, so one plan
serves every array of that layout:

```cpp file=examples/edges.cpp region=edges
// A Hann window, one row of it for every row of the batch (batch stride 0).
std::vector<double> window(n);
for (std::size_t j = 0; j < n; ++j) {
  const double phase = 2 * std::numbers::pi * static_cast<double>(j) / static_cast<double>(n);
  window[j] = 0.5 - 0.5 * std::cos(phase);
}

// Rows times the window, transformed, scaled by 1/n and added to the accumulator. The plan
// records the edges' kinds and strides; the window's address comes with each execution.
const fftn::c2c_plan<double> plan(
    fftn::rows_c2c<double>(n, count, fftn::direction::forward), {},
    {.in = {.mul = {.kind = fftn::factor_kind::real, .axis_strides = {1}, .batch_strides = {0}}},
     .out = {.scale = 1.0 / static_cast<double>(n), .accumulate = true}});
fftn::workspace ws(plan);

std::vector<std::complex<double>> sum(n * count); // the accumulator, zero to start with
plan.execute(x.data(), sum.data(), ws, {.in_factor = window.data()});
plan.execute(x.data(), sum.data(), ws, {.in_factor = window.data()}); // adds it again
```

Every plan takes scales, as one product with `options::scale`. Factors and accumulation are
taken by c2c, r2c and c2r plans of every rank in the strategy the rules or measured planning
choose: a complex factor of stride 1 on a c2c row's chain inside its first or last pass, any
other factor and accumulation as a pass of its own over the edge's side while it is in cache.
Where an edge keeps a plan from the strategy it would otherwise take, `plan::describe()` names
both and says why, and its line for the edges counts the passes they cost.

**Stored runs** (`shape::runs`) are the vocabulary's zero-padding and cropping: an axis stored
in part on a side, with the zeros outside the input's runs never read and the outputs outside
the output's runs never written. A plan with runs computes the whole line and stores only the
runs, on the chain; a one-axis c2c plan whose runs lie in the line's first half reads or
writes only that half, in the chain's first or last pass where its passes allow. Runs in place
and beside a factor are refused ([conventions](conventions.md#stored-runs)).

## Block callbacks

A block callback (`edge::call`) is a function of yours, `noexcept`, that the plan calls on
blocks of an edge's side between its passes: `modify` changes the block in place, `produce`
writes the input (which is then not read), `consume` takes the output (which is then not
stored). Each call gets the block's place (`block_info`: transform, coordinates, length,
lines and strides), the member running it and that member's scratch. `callback_of` turns a
`noexcept` lambda into a callback, its address passed in `exec_args`:

```cpp file=examples/callbacks.cpp region=callbacks
// Each output bin k of transform t times exp(i g |y|^2), a phase that depends on the bin's
// own value: pure per element, so any team or executor gives the same bits.
const double g = 0.01;
// A block holds info.lines rows (consecutive transforms), info.length elements of each.
auto kerr = [g](const fftn::block_info& info, std::complex<double>* y) noexcept {
  for (std::size_t k = 0; k < info.lines; ++k) {
    for (std::size_t i = 0; i < info.length; ++i) {
      std::complex<double>& v = y[static_cast<std::ptrdiff_t>(i) * info.element_stride +
                                  static_cast<std::ptrdiff_t>(k) * info.line_stride];
      v *= std::polar(1.0, g * std::norm(v));
    }
  }
};
const fftn::c2c_plan<double> plan(
    fftn::rows_c2c<double>(n, count, fftn::direction::forward), {},
    {.out = {.call = {.fn = fftn::callback_of<decltype(kerr), std::complex<double>>()}}});
fftn::workspace ws(plan);

std::vector<std::complex<double>> y(n * count);
// The lambda's address reaches the callback as its `user` pointer.
plan.execute(x.data(), y.data(), ws, {.out_user = &kerr});
```

Your code never runs inside FFTN's instruction-set regions, so compile it as you like. The
plan chooses the blocks: a row whole up to a size set per machine (half the L1 data cache; a
quarter of L2 on Intel cores of the Xeon Gold 6240's class), longer rows in element ranges,
several short rows together where the plan holds them so. A callback that
is pure per element keeps every determinism guarantee. Callbacks are taken by c2c, r2c and
c2r plans of every rank; on planes and volumes inside the line bodies of the first and last
phases, so they cost no pass of their own. Only a consumed output changes a strategy (some
strategies use the output as their working memory), and `describe()` then says so.
Beside stored runs a callback runs in the runs' pass, on each run of a row at its logical
place: `produce` writes only the runs (the plan writes the zeros) and `consume` takes only the
output's runs. Real-to-real plans refuse callbacks.

**What a callback costs, and how to write one** (measured by experiment
[X66](../dev/experiments/X66-block-callbacks.md) on an AMD EPYC 7763 and an Intel Xeon Gold 6240,
one thread unless stated):

- **The stage itself is cheap.** A callback that returns at once costs 1–3 % over the plain plan
  on rows of 1024 points and more and on planes; short rows in batches (256 points) pay 12–37 %,
  because a plan with a callback walks them row by row.
- **Against your own loop after the plan:** about even on c2c rows and on real rows of 4096
  points and more (a cubic term per element: 0.90–1.09 of the time of the plain plan plus a
  separate loop), a win on many short c2c rows (16 points: 0.55–0.94), a loss on short real rows
  in batches (256 points: 1.12–1.14 on the EPYC, 1.16–1.22 on the Xeon). Planes, and the free-space layout of a stride-1 batch of planes, pay more (1.1–1.6
  on the EPYC; 1.08–1.66 on the Xeon, its planes 0.61–0.77 of that since its blocks were widened):
  there the blocks are columns, a few at a time. Fusing a pointwise term into a callback is worth
  it for the code it saves you, not yet for speed on planes.
- **Walk the block in memory order.** On planes and volumes the output's blocks are columns:
  `element_stride` is the row's length and `line_stride` is 1. Put the lines in the inner loop
  there; walking each column in turn cost 1.06–5.1 times as much on these machines.
- **Precompute a factor that does not change.** A complex factor array (the vocabulary,
  `edge::mul`) took 0.07–0.40 of the time of the same `exp(i φ)` computed per element in a
  callback with the C library's `cos` and `sin` (0.11–0.70 at eight threads on the EPYC):
  computing a transcendental per element costs several times the transform itself. Compute in a
  callback what changes between executions or depends on the data.
- **Build arithmetic callbacks for the machine.** The same cubic callback compiled with AVX2 and
  FMA (`[[gnu::target("avx2,fma")]]`, or `target_clones`) took 0.90 of its baseline x86-64
  build's time over the catalogue on both machines (0.70–0.94 on most cases); a callback bound
  by scalar `cos` and `sin` gains nothing without a vectorised math library.
- **An input `modify` callback out of place copies your input first**, because the plan must
  not change it: up to 1.33 (EPYC) and 1.42 (Xeon) times the time of modifying your own array
  in place before the plan. If you may overwrite the input, run the plan in place (no copy: even
  with your own loop).

**A stage over several axes** (the free-space stage of [DESIGN 12.5](../dev/DESIGN.md), measured
by experiment [X68](../dev/experiments/X68-free-space-stage.md) on the same two machines):

- **Planes whose stride-1 dimension is a batch are slow at large sizes for now.** Planes (y, x) of
  a (y, x, ω) array with ω contiguous took 4–5 times the time of the same planes stored tight at
  384² × 750, more than the transposes they save; the spectral pair on the ω pencils, on the other
  hand, took a fifth to a third of the time of transposing, transforming the rows and transposing
  back. Until that is fixed, transpose into tight planes for large fields, or run the y axis alone
  and the x axis slab by slab while the slab is in cache.
- **A propagator: precompute it on few threads, compute it on many.** A complex factor array took
  0.70–0.76 of a callback computing `exp(i φ)` per element at one thread, but 1.43 of it at 64
  threads on the EPYC 7763, where reading an array as large as the field costs more than the cos
  and sin.

## Pipelines

A row pipeline (`pipeline_desc`, `pipeline`) takes each line of one axis through a transform,
a middle and the inverse transform, reading the line once and storing it once. Its prologue
and epilogue are edges; its middle multiplies the spectrum by a `scale` and a factor (a
convolution's kernel spectrum) and may call a callback on each line's whole spectrum:

```cpp file=examples/pipeline.cpp region=pipeline
// Every row convolved with one kernel: forward transform, times the kernel's spectrum
// (shared by the rows), backward transform, normalised by 1/n. The pipeline records the
// kernel's kind and strides; its address comes with each execution.
fftn::pipeline_desc desc = fftn::complex_convolution_rows<double>(n, count);
desc.middle.scale = 1.0 / static_cast<double>(n);
const fftn::pipeline pipe(desc);
fftn::workspace ws(pipe);

std::vector<std::complex<double>> y(n * count);
pipe.execute(x.data(), y.data(), ws, {.mid_factor = kernel.data()});
```

Built, all planned by the rules (`options::effort` is ignored, and `describe()` says so):

- **Kinds.** `pipeline<C, C, C>` with the forward transform first (a convolution:
  `complex_convolution_rows`) or the inverse first (a spectral pair: `spectral_pair_rows` with
  `direction::backward`); `pipeline<T, C, T>`, r2c then c2r, a real convolution over the
  n/2 + 1 bins (`convolution_rows`); `pipeline<C, T, C>`, c2r then r2c, half spectra in and
  out around a real middle.
- **Pairs.** Each line runs as the chain pair (for the real kinds, the real chain pair): the
  two transforms' chains on a line held in the workspace, a `<C, C, C>` pipeline's complex
  factors of stride 1 fused into the neighbouring pass, every other operation a pass over the
  line in cache. A depth-first pair for long `<C, C, C>` lines is built too, but no pipeline
  takes it yet: only an internal build setting of the tests reaches it, until the rules learn
  where it pays.
- **Stored runs** on the line, both sides (the valid half of a zero-padded convolution, a
  band of bins of a `<C, T, C>` half spectrum).
- **Fan-out:** several inverses per forward transform, each with its own middle factor and
  output, the spectrum kept in the workspace meanwhile.
- **Callbacks** on the prologue, the middle (once per line, the whole spectrum in one call)
  and the epilogue (once per copy of a fan-out).
- **Execution** serially, in pieces (`execute_transforms`), by a team or on an executor,
  bitwise alike.

`describe()` says which pair a pipeline runs, its program form (flat, unrolled or walked) and
where each operation goes. What is refused until built is listed in
[conventions](conventions.md#pipelines).

## Owner-computes

Some work cannot ride inside FFTN: it needs a whole row at once (a peak, a norm), reads
several of your arrays, or is simply your existing parallel code. Owner-computes keeps it on
the same core as the transform anyway. With `completion::own`, the member whose `fftn::team`
is `me` runs exactly the transforms `transforms_of(me)` deals it, whole: contiguous ranges in
rank order, differing by at most one transform, and with fewer transforms than members one
each for the first members. Nothing is claimed by another member or split, so each member returns once its
own transforms are written, and goes straight on to work on them. Plans (with or without
edges and callbacks) and pipelines deal the same way, so objects with the same number of
transforms give every member the same rows in every step, inside one parallel region and
with no barrier between the steps:

```cpp file=examples/owner_computes.cpp region=region
// The rows of x smoothed into y, each normalised by the application's own kernel, then their
// spectra into s: three steps in one parallel region, each thread on the same rows throughout.
void smooth_normalise_transform(const real_pipeline& smooth, const fftn::r2c_plan<double>& spectra,
                                std::size_t n, const double* x, const std::complex<double>* kernel,
                                const double* window, double* y, std::complex<double>* s) {
  // A workspace for each object, sized for the region's team: each serves one execution.
  fftn::workspace smooth_ws(smooth, omp_get_max_threads());
  fftn::workspace spectra_ws(spectra, omp_get_max_threads());

#pragma omp parallel
  {
    const fftn::team me = fftn::omp::this_team();
    // The rows this thread owns: the same in the pipeline and in the plan, which have as many.
    const fftn::transform_range mine = smooth.transforms_of(me);

    // Each step returns once this thread's own rows are written: no barrier between them.
    smooth.execute(x, y, me, smooth_ws, fftn::completion::own, {.mid_factor = kernel});
    normalise_rows(y, n, mine.first, mine.count);
    spectra.execute(y, s, me, spectra_ws, fftn::completion::own, {.in_factor = window});
  } // the region's end: every row of y and s is written
}
```

`real_pipeline` is `fftn::pipeline<double, std::complex<double>, double>`, and
`normalise_rows` the application's own kernel: each of the member's rows divided by its
largest magnitude, a reduction over the row that neither an edge nor a callback sees whole.
The objects, a real convolution and a plan with an input window:

```cpp file=examples/owner_computes.cpp region=objects
// A pipeline smoothing every row (r2c, times the filter's spectrum, c2r, scaled by 1/n), and a
// plan with edges taking each row's spectrum, windowed and scaled by 1/n. Both have `count`
// rows, so a team deals them alike.
fftn::pipeline_desc desc = fftn::convolution_rows<double>(n, count);
desc.middle.scale = 1.0 / static_cast<double>(n);
const fftn::pipeline smooth(desc);
const fftn::r2c_plan<double> spectra(
    fftn::rows_r2c<double>(n, count), {},
    {.in = {.mul = {.kind = fftn::factor_kind::real, .axis_strides = {1}, .batch_strides = {0}}},
     .out = {.scale = 1.0 / static_cast<double>(n)}});

std::vector<double> y(n * count);
std::vector<std::complex<double>> s((n / 2 + 1) * count);
smooth_normalise_transform(smooth, spectra, n, x.data(), kernel.data(), window.data(), y.data(),
                           s.data());
```

The rules of `completion::own` ([threading](threading.md#how-the-work-is-shared)):

- every member of the team makes the call, since nobody takes over another's transforms;
- the caller synchronises its threads before reading transforms other members own, and
  before a workspace serves another execution: hence a workspace per object here, each
  serving one execution in the region, whose end synchronises;
- the deal depends only on the number of transforms and the team's size, so it holds for a
  plan and a pipeline alike; give the objects that share a deal the same number of transforms.

Results are bitwise those of serial execution, as for every team execution
([conventions](conventions.md#determinism)); the tests check that each member runs exactly
the transforms dealt to it, with edges, callbacks and pipelines of every kind.
