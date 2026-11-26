# FFTN conventions

What FFTN computes and how it reads and writes memory, stated exactly. The declarations and
their preconditions are documented in the public headers (`include/fftn/`); this page
does not repeat them. FFTN follows FFTW's conventions throughout.

## Example

```cpp file=examples/rows_c2c.cpp region=rows
const std::size_t n = 12;
const std::size_t count = 3;

// Plans are built once and reused: three rows of 12 points, one after another.
const fftn::c2c_plan<double> forward(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
const fftn::c2c_plan<double> backward(fftn::rows_c2c<double>(n, count, fftn::direction::backward),
                                      {.in_place = true, .scale = 1.0 / static_cast<double>(n)});

// Scratch for one execution at a time, allocated here and never during execute.
fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

// Row r holds the tone exp(+2 pi i (r + 1) j / n).
std::vector<std::complex<double>> x(n * count);
for (std::size_t r = 0; r < count; ++r) {
  for (std::size_t j = 0; j < n; ++j) {
    const auto turns = static_cast<double>((r + 1) * j) / static_cast<double>(n);
    x[r * n + j] = std::polar(1.0, 2 * std::numbers::pi * turns);
  }
}

// Forward, out of place: row r of the spectrum is n at bin r + 1 and 0 elsewhere.
std::vector<std::complex<double>> y(n * count);
forward.execute(x.data(), y.data(), ws);

// Backward, in place and scaled by 1/n (the plan says so): this gives x back.
backward.execute(y.data(), y.data(), ws);
```

## What is computed

For one transform of logical lengths n₀ × … × n_{r−1} (r = 1, 2 or 3 axes, n₀ slowest):

- **c2c**, `shape<std::complex<T>, std::complex<T>>`:
  Y[k] = Σ_j X[j] · exp(s · 2πi · Σ_d j_d k_d / n_d), with s = −1 for
  `direction::forward` and s = +1 for `direction::backward`.
- **r2c**, `shape<T, std::complex<T>>`: the forward c2c transform of the real input, of
  which only the outputs with k_{r−1} ≤ n_{r−1}/2 are written: n_{r−1}/2 + 1 complex
  values along the last axis (integer division). The others follow from Hermitian
  symmetry. The imaginary parts of the bins that must be real are 0 to rounding.
- **c2r**, `shape<std::complex<T>, T>`: the backward transform of the Hermitian array
  whose stored half is the input, real by construction. For rows, the imaginary parts of
  bin 0 and, for even n, of bin n/2 are ignored. For more axes, the backward c2c
  transforms of the other axes are applied first and the rows of the last axis are then
  read as above.
- **r2r**, `shape<T, T>` (provisional, "Real-to-real transforms" below): each axis's own
  real transform (`shape::kinds`), applied axis by axis: the separable product.
- Nothing is normalised unless asked: backward after forward multiplies by
  n₀ · … · n_{r−1}. `options::scale` multiplies every output by a real factor (1/N for
  a normalised transform, N the number of points in one transform), inside the pass that
  writes the output: each output is then the unscaled one times the scale (rounded once
  to `T`), bit for bit, at no cost in passes.
- Lengths are any integer of at least 1. The product of the prime factors above 16 goes
  through one Bluestein stage (a convolution of a smooth length M of at least twice that
  product less one), which `plan::describe()` reports with its M; its scratch is two lines of 2 to
  4 times the length, and its error bound is that of `dev/DESIGN.md` section 15.2.

## Shapes and layouts

A shape lists its transformed axes (`shape::axes`, slowest first, one to three) and its
batch dimensions (`shape::batches`, slowest first, any number; none means one transform).
Transforms are numbered in batch order with the last batch dimension fastest;
`execute_transforms` uses these numbers. Every axis needs its length `n` and every batch dimension
its `count`: both default to 0, which plan construction refuses, so that `.batches{8}`
(eight empty batch dimensions, not one of eight) is caught; write `.batches = {{8}}`.

- **Units.** Strides and distances count elements of each side's own type: real values
  on the real side of r2c and c2r, complex values on the complex side. They may be
  negative. The pointers passed to `execute` address element 0 of transform 0, which with
  negative strides is not the lowest address.
- **Defaults.** A stride or distance of 0 means the tight C-ordered default, worked out
  for each zero independently of the other values: the last axis has stride 1; each
  other axis spans the axes after it; the innermost batch dimension is one whole array
  apart; each outer batch dimension spans the inner ones. Along the last axis an array
  spans n values, except the complex side of real transforms (n/2 + 1) and the real side
  of in-place real transforms (2(n/2 + 1), see below).
- **Unused values.** The strides of an axis of length 1 and the distances of a batch
  dimension of count 1 never step, so they are ignored, whatever their value.
- **Limits.** Every other stride and distance must be above `PTRDIFF_MIN`, and on each
  side the farthest element from element 0, in bytes, must be addressable with
  `std::ptrdiff_t`; plan construction refuses layouts that are not.
- **Real transforms out of place.** The real side has n values per row and the complex
  side n/2 + 1, each tight by default.
- **No overlap.** No two output elements may share an address. Plan construction checks
  a sufficient condition: ordered by absolute stride, each dimension's stride (or
  distance) exceeds the reach of all smaller ones, which every nested layout (rows,
  columns, padded rows, planes, reversed strides) satisfies. Inputs may overlap each
  other.

Planes and volumes have shape functions like rows': `planes_c2c`, `planes_r2c`,
`planes_c2r`, `volumes_c2c`, `volumes_r2c` and `volumes_c2r` (`fftn/shape.hpp`), each a
batch of `count` C-ordered transforms with the defaults above (the last axis halved on the
complex side of the real kinds):

```cpp file=examples/planes_volumes.cpp region=planes
// Two planes of 6 rows of 10 complex values, one after another; back scaled by 1/60.
const std::size_t ny = 6;
const std::size_t nx = 10;
const std::size_t count = 2;
const fftn::c2c_plan<double> forward(
    fftn::planes_c2c<double>(ny, nx, count, fftn::direction::forward));
const fftn::c2c_plan<double> backward(
    fftn::planes_c2c<double>(ny, nx, count, fftn::direction::backward),
    {.scale = 1.0 / static_cast<double>(ny * nx)});
fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

std::vector<std::complex<double>> x(ny * nx * count);
for (std::size_t i = 0; i < x.size(); ++i) {
  x[i] = {std::cos(0.3 * static_cast<double>(i)), std::sin(0.7 * static_cast<double>(i))};
}
std::vector<std::complex<double>> spectrum(x.size());
std::vector<std::complex<double>> back(x.size());
forward.execute(x.data(), spectrum.data(), ws);
backward.execute(spectrum.data(), back.data(), ws);
```

```cpp file=examples/planes_volumes.cpp region=volumes
// Three real volumes of 4 x 5 x 7 values, in place: each real row of 7 is padded to the
// 2 (7/2 + 1) = 8 reals of its complex row, so a volume takes 4 x 5 x 4 complex values.
const std::size_t nz = 4;
const std::size_t ny = 5;
const std::size_t nx = 7;
const std::size_t count = 3;
const std::size_t half = nx / 2 + 1;
const fftn::r2c_plan<double> forward(fftn::volumes_r2c<double>(nz, ny, nx, count),
                                     {.in_place = true});
const fftn::c2r_plan<double> backward(
    fftn::volumes_c2r<double>(nz, ny, nx, count),
    {.in_place = true, .scale = 1.0 / static_cast<double>(nz * ny * nx)});
fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

const std::size_t rows = nz * ny * count;
std::vector<std::complex<double>> data(rows * half);
auto* real = reinterpret_cast<double*>(data.data()); // the padded real rows
for (std::size_t r = 0; r < rows; ++r) {
  for (std::size_t j = 0; j < nx; ++j) {
    real[r * 2 * half + j] = std::cos(0.1 * static_cast<double>(r * nx + j));
  }
}
forward.execute(real, data.data(), ws);  // the half spectra, n/2 + 1 values per row
backward.execute(data.data(), real, ws); // the real volumes again
```

## Chosen axes of a strided array

To transform some dimensions of an array and leave its other dimensions as batches, without
moving the data, list one `axis` per dimension of the array and the dimensions to transform.
`axes_c2c`, `axes_r2c` and `axes_c2r` return the shape; every other dimension becomes a batch dimension:

```cpp file=examples/axes.cpp region=axes
// An array of ny x nx x nw complex values, C-ordered: w varies fastest. The planes (y, x), one
// per w, are transformed where they lie: their elements stand nw apart.
const std::size_t ny = 12;
const std::size_t nx = 10;
const std::size_t nw = 8;
const std::array<fftn::axis, 3> array{{{.n = ny}, {.n = nx}, {.n = nw}}};
const std::array<std::size_t, 2> yx{0, 1};
const auto points = static_cast<double>(ny * nx);
const fftn::c2c_plan<double> forward(fftn::axes_c2c<double>(array, yx, fftn::direction::forward));
const fftn::c2c_plan<double> backward(
    fftn::axes_c2c<double>(array, yx, fftn::direction::backward), {.scale = 1.0 / points});
fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

std::vector<std::complex<double>> x(ny * nx * nw);
for (std::size_t i = 0; i < x.size(); ++i) {
  x[i] = {std::cos(0.3 * static_cast<double>(i)), std::sin(0.7 * static_cast<double>(i))};
}
std::vector<std::complex<double>> spectrum(x.size());
std::vector<std::complex<double>> back(x.size());
forward.execute(x.data(), spectrum.data(), ws);
backward.execute(spectrum.data(), back.data(), ws);
```

- **The array.** A stride of 0 is the tight C-ordered default for that dimension of the array,
  each dimension spanning the ones after it; give strides, of either sign, for any other
  layout. The dimensions need not nest in the order they are listed.
- **The axes** are listed by index, one to three, in any order. Within a c2c plan the order
  changes neither the result nor (to rounding) how it is computed: the planner runs the axes
  in memory order, the fastest-varying first.
- **The batch dimensions** are the remaining dimensions, widest input stride first (ties in
  the array's order), those of length 1 dropped and neighbours that nest exactly merged. This
  fixes the transforms' numbering for `execute_transforms`.
- **Real transforms.** `array` gives the real side's lengths. The last axis listed is the
  halved one: n/2 + 1 complex values along it on the complex side, whose default strides are
  worked out with that length (the other sides of the layout rules above apply unchanged).
  `padded` asks for FFTW's padded real rows as the default strides. An in-place plan needs it
  (pass the same value as `options::in_place`) and needs the halved dimension to be the array's
  fastest; see below.

```cpp file=examples/axes.cpp region=axes-real
// The columns of a real ny x nx array, each of ny reals nx apart, to half spectra of
// ny/2 + 1 complex values stored the same way. The last axis listed is the halved one.
const std::size_t ny = 16;
const std::size_t nx = 6;
const std::array<fftn::axis, 2> array{{{.n = ny}, {.n = nx}}};
const std::array<std::size_t, 1> columns{0};
const fftn::r2c_plan<double> forward(fftn::axes_r2c<double>(array, columns));
const fftn::c2r_plan<double> backward(fftn::axes_c2r<double>(array, columns),
                                      {.scale = 1.0 / static_cast<double>(ny)});
fftn::workspace ws(std::max(forward.workspace_bytes(), backward.workspace_bytes()));

std::vector<double> x(ny * nx);
for (std::size_t i = 0; i < x.size(); ++i) {
  x[i] = std::cos(0.1 * static_cast<double>(i * i));
}
std::vector<std::complex<double>> spectrum((ny / 2 + 1) * nx);
std::vector<double> back(x.size());
forward.execute(x.data(), spectrum.data(), ws);
backward.execute(spectrum.data(), back.data(), ws);
```

For any other arrangement, or to choose the batch order yourself, write the `shape` directly:
its axes and batch dimensions take any strides and distances of either sign, nested in any
order, as "Shapes and layouts" above describes.

## Edges

Provisional (task 5.1, `fftn/edges.hpp`). A plan may take an input edge and an output edge
(`plan_edges`) of linear pointwise work, applied in its own passes:

- **Input edge**: each input element times the edge's factor and its `scale`, then the
  transform.
- **Output edge**: each output element times the edge's factor and its `scale`, then stored,
  or added to the output's value with `accumulate` (output edge only, and refused on an
  in-place plan).

A factor (`factor`) is an array of real or complex values of the plan's precision, indexed
by logical coordinates through its own strides: one per axis and one per batch dimension, in
its own elements, any sign, 0 broadcasting along that dimension (no batch strides at all
broadcast along every batch dimension). The plan records the
factor's kind and strides; each execution passes the array's address in `exec_args`,
through the overloads of `execute` and `execute_transforms` that take one, so one plan serves
every array of that layout. Executing such a plan without an array it needs (through an
overload without `exec_args`, or with a null pointer) ends the program. A complex factor is refused on a real side (the input of r2c,
the output of c2r). `options::scale` is a shortcut for the output edge's scale.

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

Results equal the same work done by separate loops within the transform's error bound plus
one rounding per multiplication and addition, not bit for bit: the scales are applied as
one product, which differs from separate multiplications only in rounding, and a factor may
be fused into a pass's arithmetic. The guarantees of
"Determinism" below hold for plans with edges too.

Scales are taken by every plan; so are factors and accumulation, in the strategy the rules
or measured planning choose (`options::effort`), with a few exceptions: there
`plan::describe()` names the strategy the same plan without the edges takes, the one it takes
instead and why. Its line for the edges also counts the separate passes they cost. A wisdom entry of a plan with edges serves the same edges only. Where that strategy is the chain of a c2c row, a
complex factor of stride 1 along the row is applied in its first or last pass. Every other
factor, and accumulation, is a pass of its own over the edge's side, before or after the
transform; the workspace may then hold one transform (or row) more: an input edge that
cannot be applied in the output, and accumulation, which keeps the result or the output's
prior values there.

## Block callbacks

Provisional (task 5.3a, `fftn/callback.hpp`). An edge may also take a block callback
(`edge::call`): a function the application compiles, which the plan calls on blocks of the
edge's side while they are in cache. Its type is `block_fn<T>`, `noexcept`, with `T` the side's
element type; `callback_of<F, T>()` makes one that calls a callable object (a `noexcept` lambda)
whose address arrives as the callback's `user` pointer (`exec_args::in_user`, `out_user`). A
callback `modify`s the block in place, `produce`s the input (the input is then not read and may
be null) or `consume`s the output (nothing is stored; the output may be null). Each call gets a
`block_info`: the transform, the logical coordinates of the block's first element, its length,
lines and strides, the member running it and that member's scratch (`callback::scratch_bytes`
of it, counted by `workspace_bytes()`).

```cpp file=examples/callbacks.cpp region=callbacks
// Each output bin k of transform t times exp(i g |y|^2), a phase that depends on the bin's
// own value: pure per element, so any team or executor gives the same bits.
const double g = 0.01;
auto kerr = [g](const fftn::block_info& info, std::complex<double>* y) noexcept {
  for (std::size_t i = 0; i < info.length; ++i) {
    std::complex<double>& v = y[static_cast<std::ptrdiff_t>(i) * info.element_stride];
    v *= std::polar(1.0, g * std::norm(v));
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

On the input edge the input is read (or produced), then a `modify` callback runs, then the
factor and the scales; on the output edge the factor and the scales apply first, then a
`modify` callback, then the store, the accumulation or `consume`. Each element (each stored
element, with stored runs) is passed exactly once per execution, in blocks whose shape the plan
chooses; blocks may run concurrently on different members and in any order. When the callback
is pure per element (its result depends only on the element's value and coordinates), the
guarantees of "Determinism" below hold. A callback may execute another plan through a serial
overload with a workspace of its own, such as `fftn::workspace{info.scratch}`. A plan with a
callback is executed through the overloads that take `exec_args`; the others end the program.
So far callbacks are taken by c2c rows only, on the chain whatever strategy the rules would
choose and by the rules whatever `options::effort` says (`plan::describe()` says so); other
plans refuse them at construction.

## Stored runs

Provisional (task 5.1d, `fftn::stored_run` in `fftn/shape.hpp`). An axis may be stored in
part on either side: `shape::runs` lists runs, each `count` elements stored from position
`stored` of the axis's memory that are the logical positions `logical` onwards of the
transformed line. On an axis with runs on a side, input elements outside the runs are zero and
are never read, output elements outside the runs are never written, and the default stride of
the next axis out spans the axis's stored extent (the end of its last run), not its length; the
overlap check measures the same extent. Examples: the band of bins of a long time line that an
application keeps, the valid half of a zero-padded convolution's row, the bins an output crops.
`axes_c2c`, `axes_r2c` and `axes_c2r` take runs too (their axes index the array), so that a
compact array of stored bins gets its strides by default.

Runs of one axis and side may not overlap in their logical or their stored positions; they may
be listed in any order. A plan with runs reads the runs into the whole line with zeros elsewhere,
computes the whole transform (skipping the lines of a multi-axis transform that are known to be
zero or are not wanted) and stores only the runs, on the chain whatever the rules would choose,
except that a one-axis c2c plan whose runs on a side lie in the line's first half reads or writes
only that half in the chain's first or last pass where the chain's passes allow (one run of
exactly the half, contiguous, is then read or written in place, with no separate pass),
planned by the rules whatever `options::effort` says; `plan::describe()` says so and counts the
passes. Not built yet, and refused at construction: runs on an in-place plan, and runs beside an
edge's factor (accumulation and scales are taken).

## Pipelines

Provisional (tasks 5.2b and 5.2c, `fftn/pipeline.hpp`). A row pipeline takes each line of one axis
through a transform, a pointwise middle and the inverse transform, reading the line once and
storing it once: the spectrum stays in the workspace, so the work is one sweep of memory
instead of three. A `pipeline_desc<In, Mid, Out>` describes it: the `line` (its length and the
strides of its input and output elements), the `batches`, and three places for the vocabulary
of edges:

- **prologue**: each input element times its factor and its `scale` (an `edge<In>`, without
  `accumulate`);
- **middle**: each bin of the spectrum times its factor and its `scale`: a convolution's
  kernel spectrum, one for every line (batch strides 0, or none) or one per line;
- **epilogue**: each output element times its factor and its `scale`, then stored or, with
  `accumulate` (refused in place), added to the output.

Factors are indexed by the logical position along the line (the bin, for the middle) and the
batch, through their own strides, as an edge's are. Each execution passes the arrays in
`exec_args` (`in_factor`, `mid_factor`, `out_factor`); a pipeline's execute calls all take
one, and an execution without an array the pipeline needs ends the program. Nothing is
normalised: a round trip multiplies by n unless a scale or the kernel divides it out. The scales
(the three places' and `options::scale`, the epilogue's shortcut) are applied as one product.

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

A `pipeline<T, C, T>` (C = `std::complex<T>`) is a real convolution: the r2c transform of each
real line, the middle on the spectrum's n/2 + 1 bins (bins 0 to n/2, its factor indexed by bin),
and the c2r transform, which ignores the imaginary parts of bin 0 and, for even n, of bin n/2, as a
c2r plan does. Its prologue and epilogue are on real values and take real factors only.
`convolution_rows` describes real rows convolved with one kernel spectrum:

```cpp file=examples/pipeline.cpp region=real-pipeline
// The filter's spectrum: its r2c transform, n/2 + 1 bins.
const fftn::r2c_plan<double> r2c(fftn::rows_r2c<double>(n, 1));
fftn::workspace r2c_ws(r2c);
std::vector<std::complex<double>> kernel(n / 2 + 1);
r2c.execute(filter.data(), kernel.data(), r2c_ws);

// Every real row circularly convolved with the filter: r2c, times the spectrum, c2r,
// normalised by 1/n, in one sweep.
fftn::pipeline_desc desc = fftn::convolution_rows<double>(n, count);
desc.middle.scale = 1.0 / static_cast<double>(n);
const fftn::pipeline pipe(desc);
fftn::workspace ws(pipe);

std::vector<double> y(n * count);
pipe.execute(x.data(), y.data(), ws, {.mid_factor = kernel.data()});
```

A linear (zero-padded) convolution of m values with a kernel of L taps is the circular one of a
length n of at least m + L − 1; until stored runs are built the caller stores the zeros.

Built so far: `pipeline<C, C, C>` with the forward transform first and `pipeline<T, C, T>`,
lines stored whole, one inverse per forward, planned by the rules (`options::effort` is
ignored, and `pipeline::describe()` says so). Refused at construction, naming the field, until
their steps: the inverse transform first (`spectral_pair_rows` with `direction::backward`), the
kind `<C, T, C>`, stored runs (`pipeline_desc::runs`) and `fanout` above 1. On `<C, C, C>`, a
complex factor of stride 1 along the line is applied in the neighbouring transform's pass; any
other factor, every factor of `<T, C, T>`, and accumulation, in a pass of its own over the line
while it is in the workspace (`describe()` counts them). Results equal the transform, the multiplications and the
inverse done separately within two transforms' error bound plus one rounding per
multiplication and addition, not bit for bit; the guarantees of "Determinism" below hold for
pipelines too. `pipeline::transforms_of` deals lines among a team's members as
`plan::transforms_of` deals transforms.

## Real-to-real transforms

Provisional (task 6.3, `fftn::r2r_kind` in `fftn/shape.hpp`). A real-to-real shape,
`shape<T, T>`, holds n reals per line on both sides, laid out as a c2c shape's complex values
are (both sides real, so strides count reals), and gives each axis its transform in
`shape::kinds`, one kind per axis in the order of `shape::axes`; `shape::dir` is ignored.
`rows_r2r`, `planes_r2r` and `volumes_r2r` give one kind to every axis, `axes_r2r` a kind per
listed dimension of a strided array. The definitions are FFTW's, unnormalised, for a line of n
reals x_j (sums over j = 0 … n − 1 unless shown):

| Kind | Y_k, k = 0 … n − 1 | FFTW | Inverse, up to N |
|---|---|---|---|
| `dct1` | x_0 + (−1)^k x_(n−1) + 2 Σ_(j=1…n−2) x_j cos(π j k / (n − 1)); n ≥ 2 | REDFT00 | `dct1`, N = 2(n − 1) |
| `dct2` | 2 Σ x_j cos(π (j + ½) k / n) | REDFT10 | `dct3`, N = 2n |
| `dct3` | x_0 + 2 Σ_(j=1…n−1) x_j cos(π j (k + ½) / n) | REDFT01 | `dct2`, N = 2n |
| `dct4` | 2 Σ x_j cos(π (j + ½)(k + ½) / n) | REDFT11 | `dct4`, N = 2n |
| `dst1` | 2 Σ x_j sin(π (j + 1)(k + 1) / (n + 1)) | RODFT00 | `dst1`, N = 2(n + 1) |
| `dst2` | 2 Σ x_j sin(π (j + ½)(k + 1) / n) | RODFT10 | `dst3`, N = 2n |
| `dst3` | (−1)^k x_(n−1) + 2 Σ_(j=0…n−2) x_j sin(π (j + 1)(k + ½) / n) | RODFT01 | `dst2`, N = 2n |
| `dst4` | 2 Σ x_j sin(π (j + ½)(k + ½) / n) | RODFT11 | `dst4`, N = 2n |
| `dht` | Σ x_j (cos + sin)(2π j k / n) | DHT | `dht`, N = n |

A transform followed by its inverse multiplies by N (the product of the axes' N for several
axes). The transform of several axes is the separable product, axis by axis, so the DHT of a
plane is the product of the rows' and the columns' DHTs, as FFTW's, not the transform whose
kernel is cas of the summed phases.

```cpp file=examples/r2r.cpp region=dht
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
```

```cpp file=examples/r2r.cpp region=dht-planes
const fftn::r2r_plan<double> planes(fftn::planes_r2r<double>(ny, nx, 1, fftn::r2r_kind::dht));
fftn::workspace ws(planes);
std::vector<double> h(ny * nx);
planes.execute(x.data(), h.data(), ws);
```

```cpp file=examples/r2r.cpp region=dct
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
```

Built so far: `dht`, `dct2`, `dct3`, `dst2` and `dst3`, any n and mixed across axes; `dct1`,
`dct4`, `dst1` and `dst4` are refused at construction, naming `shape.kinds[i]`, until task 6.5.
Refused too: `kinds` without one kind per axis, `kinds` on a shape of another kind, `dct1` on an
axis of one point, and stored runs. A real-to-real plan takes edges with scales, real factors
and accumulation (a complex factor is refused on either side), in place as c2c does, teams,
executors and `execute_transforms`; it is planned by the rules whatever `options::effort` says
(`plan::describe()` says so). Each axis is a phase over its lines; a line is read into the
workspace by a read stage, transformed there by a complex transform of half its length (n even)
or of its length (n odd), and written out by a store stage, which applies the output edge's
factor, the scales and accumulation. The DHT combines the bins of an r2c; DCT-II reorders the
line (Makhoul's order) for an r2c and twiddles its bins; DCT-III twiddles the line into the
spectrum of a c2r and reads the result back from Makhoul's order; the sine kinds are the cosine
ones with alternating signs and reversals. An axis of one point is the identity for `dht`,
`dct3` and `dst3` and doubles for `dct2` and `dst2` (Y_0 = 2 x_0). The results are within the
underlying transform's error bound plus a rounding per stage operation; the guarantees of
"Determinism" below hold.

## In place

A plan built with `options::in_place` is executed with the same address as input and
output, and overwrites its input. The layouts must agree, which construction checks:

- c2c and r2r: every input stride and distance equals the output's.
- r2c and c2r (FFTW's padded layout): the last axis is contiguous on both sides, and every
  other stride and every distance on the real side is twice the complex side's. A real
  row then occupies the first n of the 2(n/2 + 1) reals of its complex row; the last
  reals of the row are padding, overwritten by r2c and ignored by c2r. With default
  strides and distances this is what you get.

Out of place, input and output must not overlap at all.

## What is preserved

- The input of an out-of-place plan is never written, for every kind (c2r included: a
  c2r of two or more axes builds its intermediate in the output or the workspace instead).
- No memory outside the described output elements is written: not the gaps between
  strided elements or rows, not the padding of out-of-place rows.
- An in-place plan overwrites the elements it describes, padding included.

## Alignment

Arrays may have any alignment the element type allows (that of `T`, also for
`std::complex<T>` data). Workspaces over caller memory may have any alignment at all.
Alignment never changes results: every offset gives the same bits.

## Workspaces and execution

Each execution needs a `workspace` of at least `plan::workspace_bytes()` bytes, used by
no other execution meanwhile. Plans hold no scratch, so one plan may be executed from
several threads at once with a workspace each. Execution never allocates, locks, throws
or does I/O; everything is decided and allocated when the plan is constructed.

## Accuracy

With N the number of points in one transform, L = max(1, log₂ N) and ε the machine
epsilon of `T`, the test suites require a relative RMS error ‖Y − X‖₂ / ‖X‖₂ ≤ 3ε√L and a
largest error ‖Y − X‖∞ / ‖X‖∞ ≤ 3εL against a quadruple-precision reference, in both
precisions:

- rows of every kind at every length up to 512 and at sampled lengths up to 2¹⁴ (primes
  among them), compared at every output up to n = 64 and at 32 sampled outputs beyond;
- c2c planes up to 96 × 96, compared at every output up to 64 points along each axis and
  at sampled outputs beyond;
- complex and real arrays of two and three axes (up to 32 × 30 and 4 × 6 × 8 points) at
  every output, with default and strided layouts, and rows and planes with three and four
  batch dimensions in several memory orders, axes and batch dimensions interleaved; every
  ordered selection of one to three dimensions of arrays of up to four dimensions (up to
  8 x 9 x 16 and 2 x 3 x 4 x 5 points), the others as batch dimensions, under C, Fortran and
  rotated memory orders, on every instruction set the machine runs; their bounds add the
  rounding of the reference's values between axes and, for c2r, of the input spectrum.

The same rows and planes are compared with FFTW at every output, within the sum of the
two libraries' bounds. Measured on an Apple M-series Mac (Clang 23), the largest error in
any of these checks is about a fifth of its bound: error/tolerance 0.19 against the
reference, 0.14 against FFTW, 0.12 for the arrays of several axes, and 0.20 in the
identity checks (whose bounds include the rounding of their closed-form inputs).
Twiddle factors are correctly rounded, which is checked entry by entry for every length
the suites test.

These bounds assume IEEE arithmetic in the calling process as well as in the library.
FFTN's build sets its own floating-point flags with every supported compiler, Intel's
icpx included (whose default model is a fast one), and the public headers refuse
translation units compiled with fast-math. They cannot see the floating-point
environment, though: icpx's default model, when it compiles an optimised `main()`, turns
on flush-to-zero and denormals-are-zero for the whole process, and FFTN's results then
lose their denormal values like any other code's. Compile the file holding `main()` with
`-fno-fast-math` (or `-fp-model=precise`) to keep IEEE behaviour.

## Determinism

For a given plan, results are bitwise identical:

- across repeated executions, copies of the plan and concurrent executions;
- between serial execution and execution by a team of any size, grouping and timing
  ([threading](threading.md));
- for every transform of a batch, whatever its position, through `execute` or through
  `execute_transforms` with any `first`;
- at any alignment of the arrays and the workspace.

Results may differ in the last bits between instruction sets, between plans built with
different options (in place or not), between plans of different batch counts (a row of a
batch against a plan of that row alone), and between library versions. A plan's instruction
set is the fastest the processor and the build support (`plan::instruction_set()`);
`options::force_isa` picks another, for tests and benchmarks.
Measured planning (`options::effort`, [planning](planning.md)) may choose a different
program from run to run, and so different last bits; a plan built from a wisdom entry runs
the recorded program, bit for bit.
