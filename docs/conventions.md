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

[Fusion](fusion.md) says which of edges, block callbacks, stored runs, pipelines and
owner-computes fits which work; this section and the next three state their rules.

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

Provisional (tasks 5.3a to 5.3c, `fftn/callback.hpp`). An edge may also take a block callback
(`edge::call`): a function the application compiles, which the plan calls on blocks of the
edge's side while they are in cache. Its type is `block_fn<T>`, `noexcept`, with `T` the side's
element type; `callback_of<F, T>()` makes one that calls a callable object (a `noexcept` lambda)
whose address arrives as the callback's `user` pointer (`exec_args::in_user`, `out_user`). A
callback `modify`s the block in place, `produce`s the input (the input is then not read and may
be null) or `consume`s the output (nothing is stored; the output may be null). Each call gets a
`block_info`: the transform, the logical coordinates of the block's first element, its length,
lines and strides, the member running it and that member's scratch (`callback::scratch_bytes`
of it, counted by `workspace_bytes()`). A block holds a row whole up to a size the plan's machine
profile sets (half the L1 data cache by default), longer rows in element ranges; where the plan
holds several short rows together (the lanes) one call takes them all, line k being transform
`transform + k`. On a plane or volume a block is lines along one axis (`element_axis`), and where
the plan holds several together, adjacent lines along another axis (`line_axis`: a block of
adjacent columns) or consecutive transforms (`along_transforms`). On a real side the elements are real values (`T`); on the complex side of a
real transform they are the n/2 + 1 bins.

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

On the input edge the input is read (or produced), then a `modify` callback runs, then the
factor and the scales; on the output edge the factor and the scales apply first, then a
`modify` callback, then the store, the accumulation or `consume`. Each element (each stored
element, with stored runs) is passed exactly once per execution, in blocks whose shape the plan
chooses; blocks may run concurrently on different members and in any order. When the callback
is pure per element (its result depends only on the element's value and coordinates), the
guarantees of "Determinism" below hold. A callback may execute another plan through a serial
overload with a workspace of its own, such as `fftn::workspace{info.scratch}`. A plan with a
callback is executed through the overloads that take `exec_args`; the others end the program.
Callbacks are taken by c2c, r2c and c2r plans of every rank on the strategy the rules or measured
planning choose; on a plane or volume the input's runs inside the lines of the transform's first
phase and the output's inside those of its last, so a callback costs no pass over the array of
its own, and a side with a factor or accumulation runs its callback in that side's own pass; measured planning times its candidates with stand-ins that do
nothing in place of the callbacks, so construction never calls one. Only a consumed output keeps
a row from a strategy that writes the output as its working memory (the grid, the depth-first
chains, the lanes over adjacent transforms, and a real row's lanes whose input is in the
workspace block too), which then runs as the chain, as `plan::describe()` says. Beside stored
runs (rows, planes and volumes) a side's callback runs in that side's pass of the runs, on each
run of a row: a block never crosses a gap, its coordinates are logical, `produce` writes only
the runs and `consume` receives only the output's runs. Real-to-real plans refuse callbacks at
construction.

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

Provisional (tasks 5.2b to 5.2g and 5.3d to 5.3f, `fftn/pipeline.hpp`). A row pipeline takes each line of one axis
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
length n of at least m + L − 1. The line's stored runs (`pipeline_desc::runs`, axis 0, as in
"Stored runs" above; the helpers take them and set the rows' distances to their stored extents)
keep the zeros out of memory: the input's elements outside its runs are zero and not read, and
the output's outside its runs are not written. The valid half of such a convolution:

```cpp file=examples/pipeline.cpp region=valid-half
// Rows of m values convolved linearly with the filter: the line of n = 2m holds a row in
// its first half and zeros beyond, which are not stored or read, and only the first m outputs
// are stored. The rows are m apart on both sides (the runs' stored extents).
const std::vector<fftn::stored_run> runs{
    {.axis = 0, .where = fftn::side::in, .logical = 0, .stored = 0, .count = m},
    {.axis = 0, .where = fftn::side::out, .logical = 0, .stored = 0, .count = m},
};
fftn::pipeline_desc desc = fftn::convolution_rows<double>(n, count, runs);
desc.middle.scale = 1.0 / static_cast<double>(n);
const fftn::pipeline pipe(desc);
fftn::workspace ws(pipe);

std::vector<double> y(m * count);
pipe.execute(x.data(), y.data(), ws, {.mid_factor = kernel.data()});
```

A **fan-out** runs several inverse transforms (copies) on each line's spectrum: the forward
transform and the prologue once per line, then for each copy k the middle with its own factor,
`fanout_factor_dist` k elements past copy 0's (0: one factor for every copy), the inverse
transform and the epilogue (whose factor is every copy's), written `fanout_out_dist` k `Out`
elements past copy 0's output. The copies' outputs must overlap neither one another nor other
lines' (refused, naming `desc.fanout_out_dist`), and an in-place pipeline has one copy. The
spectrum stays in the workspace while the copies run:

```cpp file=examples/pipeline.cpp region=fanout
// Every row transformed once, then convolved with three kernels: copy k multiplies the
// row's spectrum by kernel k (n bins on) and writes a whole batch of rows (n * count
// elements on).
fftn::pipeline_desc desc = fftn::complex_convolution_rows<double>(n, count);
desc.middle.scale = 1.0 / static_cast<double>(n);
desc.fanout = copies;
desc.fanout_factor_dist = static_cast<std::ptrdiff_t>(n);
desc.fanout_out_dist = static_cast<std::ptrdiff_t>(n * count);
const fftn::pipeline pipe(desc);
fftn::workspace ws(pipe);

std::vector<std::complex<double>> y(n * count * copies);
pipe.execute(x.data(), y.data(), ws, {.mid_factor = kernels.data()});
```

Block callbacks ("Block callbacks" above) take three places in a pipeline: the prologue's
(`prologue.call`, `modify` or `produce`) on each input line before the prologue's factor and
scale, the middle's (`middle.call`, `modify` only) on each line's spectrum before the middle's
factor and scale, and the epilogue's (`epilogue.call`, `modify` or `consume`) on each output line
after the epilogue's factor and scale and before the store or accumulation. Their arguments are
`exec_args::in_user`, `mid_user` and `out_user`. The middle's callback is passed the whole
spectrum of a line in one call, once per line, so it may depend on the whole line (it is then
pure per line, as "Determinism" below needs): it may run another plan or pipeline on the line
through a serial overload. With fan-out the middle's runs once per line, before every copy, and
the epilogue's once per copy (`block_info::copy`); a consumed output receives each line's copies
in copy order, and its copies' places (`fanout_out_dist`) are not checked, since nothing is
stored. The scales never cross a middle callback: the prologue's applies before it, and the
middle's, the epilogue's and `options::scale` after it, as a second product.

```cpp file=examples/pipeline.cpp region=middle-callback
// Between the transforms, a callback on each row's spectrum (the whole row in one call, in
// natural order), then the scale 1/n. The lambda's address arrives with each execution as
// the callback's `user` pointer.
auto saturate = [&gain](const fftn::block_info& info, std::complex<double>* bins) noexcept {
  for (std::size_t b = 0; b < info.length; ++b) {
    std::complex<double>& y = bins[static_cast<std::ptrdiff_t>(b) * info.element_stride];
    y = gain(y);
  }
};
fftn::pipeline_desc desc = fftn::complex_convolution_rows<double>(n, count);
desc.middle.mul = {}; // no kernel: the callback is the whole middle
desc.middle.scale = 1.0 / static_cast<double>(n);
desc.middle.call = {.fn = fftn::callback_of<decltype(saturate), std::complex<double>>()};
const fftn::pipeline pipe(desc);
fftn::workspace ws(pipe);

std::vector<std::complex<double>> y(n * count);
pipe.execute(x.data(), y.data(), ws, {.mid_user = &saturate});
```

A **group** (`pipeline_desc::group`, task 5.3f) hands the middle callback every line of one batch
dimension together, such as the two polarisation components of a field: each line of a group
takes the first transform, then the callback runs once on the group's G spectra
(`block_info::group` G, member k's line `k * group_stride` elements after member 0's, whose line
number is `block_info::transform`), then each line takes the middle's factor and scale and the
second transform. A group is one task: `execute_transforms` and `transforms_of` count groups, as
many as `pipeline::transform_count()`, and the results are the same however the work is shared
when the callback is pure per group. Each workspace slot holds the group's G lines, so the
workspace grows with G. A group needs a middle callback, and is refused with fan-out (not built
yet).

```cpp file=examples/pipeline.cpp region=group
// Batch dimension 0 is the component (2), dimension 1 the row: with group 0 the middle
// callback receives both components' spectra of one row in one call, component 1 group_stride
// elements after component 0, and rotates each bin's pair.
auto rotate = [&angle](const fftn::block_info& info, std::complex<double>* bins) noexcept {
  for (std::size_t b = 0; b < info.length; ++b) {
    std::complex<double>& u = bins[static_cast<std::ptrdiff_t>(b) * info.element_stride];
    std::complex<double>& v = (&u)[info.group_stride];
    const double c = std::cos(angle(b));
    const double s = std::sin(angle(b));
    const std::complex<double> u0 = u;
    u = c * u0 - s * v;
    v = s * u0 + c * v;
  }
};
fftn::pipeline_desc desc = fftn::complex_convolution_rows<double>(n, rows);
desc.batches.insert(desc.batches.begin(), fftn::batch{.count = 2});
desc.group = 0;
desc.middle.mul = {};
desc.middle.scale = 1.0 / static_cast<double>(n);
desc.middle.call = {.fn = fftn::callback_of<decltype(rotate), std::complex<double>>()};
const fftn::pipeline pipe(desc);
fftn::workspace ws(pipe);

std::vector<std::complex<double>> y(x.size());
pipe.execute(x.data(), y.data(), ws, {.mid_user = &rotate});
```

A **spectral pair** runs the inverse transform first: `spectral_pair_rows(n, count,
direction::backward)` (or `pipeline_desc::first`) for complex lines, and `pipeline<C, T, C>`, whose
input and output lines are half spectra of n/2 + 1 bins (the c2r transform, ignoring the
imaginary parts of bin 0 and, for even n, of bin n/2, then the middle on a real line of n points,
which takes real factors only, then the r2c transform). A `<C, T, C>` pipeline's stored runs are
bands of the half spectrum's bins.

Built so far: `pipeline<C, C, C>` with either transform first, `pipeline<T, C, T>` and
`pipeline<C, T, C>`, with stored runs on either side, fan-out, block callbacks and groups, planned
by the rules or, with `options::effort`, by timing the pairs of transforms the description can take
([planning](planning.md), "Pipelines"; provisional). Refused at construction, naming the field, until their steps:
`fanout` above 1 in place, and stored runs on an in-place pipeline or beside a prologue or epilogue
factor; a middle callback in another mode than `modify`, and `consume` in place, are refused too.
On `<C, C, C>`, a complex factor of stride 1 along the line is applied in the neighbouring
transform's pass; any other factor, every factor of `<T, C, T>`, accumulation and the runs' reading
and storing, in a pass of its own over the line while it is in the workspace (`describe()` counts
them), except that runs within the line's first half are read or written by the transform's first or
last pass where its passes allow, as for a plan. A pipeline is built as the chain pair, except
where the rules take another pair of transforms measured faster on the machine (experiment X67):
on some processors long contiguous `<C, C, C>` lines take a depth-first pair, and a plane's strided
`<C, C, C>` lines a lanes pair ([fusion](fusion.md#pipelines) names the processors and limits);
`pipeline::describe()` names the pair. Results equal the transform, the multiplications and
the inverse done separately within two transforms' error bound plus one rounding per multiplication
and addition, not bit for bit; the guarantees of "Determinism" below hold for pipelines too.
`pipeline::transforms_of` deals lines among a team's members as `plan::transforms_of` deals
transforms.

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

Nothing is normalised unless asked, as for the other kinds: `options::scale` multiplies every
output by one factor, so a plan of the inverse kind with `scale` 1/N undoes the transform, and
1/√N on both makes the DHT, `dst1`, `dct4` and `dst4` orthonormal. The orthonormal `dct1`,
`dct2`, `dct3`, `dst2` and `dst3` need weights on the first or last points besides 1/√N (scipy's
`norm="ortho"`). A plan does not apply them itself. Along one axis they are real factors on its
edges, which is how the convenience layer's `norm::ortho` gives scipy's results there; over
several axes the convenience layer weights the first and last hyperplanes of each axis in a pass
of its own ([easy](easy.md#real-to-real-transforms), which lists the weights).

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

```cpp file=examples/r2r.cpp region=dct4
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
```

Every kind is built, any n (`dct1`: n ≥ 2) and mixed across axes. Refused: `kinds` without one
kind per axis, `kinds` on a shape of another kind, `dct1` on an axis of one point, and stored
runs. A real-to-real plan takes edges with scales, real factors and accumulation (a complex
factor is refused on either side), in place as c2c does, teams, executors and
`execute_transforms`; it is planned by the rules whatever `options::effort` says
(`plan::describe()` says so). Each axis is a phase over its lines; a line is read into the
workspace by a read stage, transformed there by a complex transform, and written out by a store
stage, which applies the output edge's factor, the scales and accumulation. The DHT combines the
bins of an r2c (a complex transform of n/2, or of n for odd n); DCT-II reorders the line
(Makhoul's order) for an r2c and twiddles its bins; DCT-III twiddles the line into the spectrum
of a c2r and reads the result back from Makhoul's order; DCT-IV pairs and twiddles the line for
a complex transform of n/2 (odd n: a zero-padded complex transform of 2n, about four times the
work); DCT-I and DST-I take the r2c of the line's even or odd extension, of 2(n − 1) or 2(n + 1)
reals (about twice the work of the kinds II and III); the sine kinds are the cosine ones with
alternating signs and reversals. `plan::describe()` names each axis's route. An axis of one point
is the identity for `dht`, `dct3` and `dst3`, doubles for `dct2`, `dst2` and `dst1`
(Y_0 = 2 x_0), and multiplies by √2 for `dct4` and `dst4`; these factors join the scales, so the
scales times them must lie within the precision's range. The results are within the underlying
transform's error bound plus a rounding per stage operation; the guarantees of "Determinism"
below hold.

## Split complex

Provisional (task 6.10, `fftn/split.hpp`). Complex data may be kept as two arrays of reals, one
of real parts and one of imaginary parts, as FFTW's split interface takes them. The element tag
`split_complex<T>` takes the place of `std::complex<T>` in the shape and the plan:
`shape<split_complex<T>, split_complex<T>>` (c2c), `shape<T, split_complex<T>>` (r2c, the
output split) and `shape<split_complex<T>, T>` (c2r, the input split), in rows, planes, volumes
and chosen axes alike. On a split side the strides and distances count reals and hold for both
arrays; the defaults are those of the interleaved shape, counted in reals. The element of a
side at offset o (the sum of its indices times their strides and distances, as for interleaved
data) has its real part at `re[o]` and its imaginary part at `im[o]`: the split r2c output of a
row of n reals is n/2 + 1 values in each array, and the two arrays of a tight batch of rows
each hold count × (n/2 + 1). An execution passes a
split side as `split_ptr<const T>` (input) or `split_ptr<T>` (output), `{re, im}`, in every
form: serial, `execute_transforms`, teams and executors (`plan::in_data` and `plan::out_data`
name the types); a `split_ptr<T>` converts to `split_ptr<const T>`. A null imaginary array ends
the program with a message naming `split_ptr::im`. Mixed forms (interleaved in, split out) are
not offered.

```cpp file=examples/split.cpp region=rows
using split = fftn::split_complex<double>;
// The shape of rows_c2c, its strides and distances counting doubles in each array.
fftn::shape<split, split> s;
s.dir = fftn::direction::forward;
s.axes = {{.n = n}};
s.batches = {{.count = count}};
const fftn::plan<split, split> p(s);
fftn::workspace ws(p);

std::vector<double> out_re(n * count);
std::vector<double> out_im(n * count);
p.execute({re.data(), im.data()}, {out_re.data(), out_im.data()}, ws);
```

```cpp file=examples/split.cpp region=r2c
// A real plane into split arrays of ny rows of nx/2 + 1 bins each.
fftn::shape<double, fftn::split_complex<double>> s;
s.axes = {{.n = ny}, {.n = nx}};
const fftn::plan<double, fftn::split_complex<double>> p(s);
fftn::workspace ws(p);

std::vector<double> re(ny * h);
std::vector<double> im(ny * h);
p.execute(x.data(), {re.data(), im.data()}, ws);
```

The two arrays of a side may lie anywhere relative to each other, interleaved in one buffer
included; an output's arrays must overlap neither each other nor the input's, which is a
precondition and not checked, since the pointers arrive with each execution. In place (c2c
only) each input array is the output array of its part (`in.re == out.re`, `in.im == out.im`);
r2c and c2r are refused in place, since their real array and two split arrays cannot each be
the same in and out.

The plan converts: a pass reads each transform's input from its two arrays into the
interleaved form in the workspace block, the transform runs there in place, and a pass writes
the result into the output's two arrays (`plan::describe()` names both passes; r2c and c2r have
one each). The values are those of the interleaved plan of the transform in the block, bit for
bit, so the accuracy and "Determinism" below are the interleaved kinds'. The workspace holds one
transform more. Not built yet: rows on other strategies than the chain (the lanes, the grid and
the depth-first chains; `describe()` names the strategy the plan on interleaved data takes),
edges, stored runs, measured planning and wisdom (split plans are planned by the rules, and
`describe()` says so; `wisdom::add` takes none). The C API has split plans too
([c_api.md](c_api.md), "Split complex").

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
