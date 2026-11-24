# Proposal: the phase 5 fusion API

**Proposal, not agreed.** Drafted 2026-11-24 for the maintainer, who may accept, change or reject
any part of it. Until then [DESIGN.md](../DESIGN.md) section 12 remains the design, and nothing
here may be implemented (AGENTS.md section 7: public API changes need the maintainer's agreement).
When a version is agreed, its parts move into DESIGN sections 5 and 12 and the decision log, and
this file is deleted, so that each fact lives in one place. Every header and example below is a
sketch: abbreviated, not compiled, and not a promise of exact names.

Contents:

1. [Summary](#1-summary)
2. [The problem, by bytes moved](#2-the-problem-by-bytes-moved)
3. [What the API must express](#3-what-the-api-must-express)
4. [Choices and alternatives](#4-choices-and-alternatives)
5. [The proposed API](#5-the-proposed-api)
6. [Examples](#6-examples)
7. [Should `axes_*` change?](#7-should-axes_-change)
8. [What it costs to build](#8-what-it-costs-to-build)
9. [Experiments](#9-experiments)
10. [Open questions for the maintainer](#10-open-questions-for-the-maintainer)

---

## 1. Summary

- **Two kinds of pointwise work, two mechanisms.** Linear, separable work (scale, real or complex
  factors, zero-padding and cropping, accumulation) is a small fixed vocabulary that FFTN compiles
  into its own passes, as `options::scale` and the internal `factor` edge are today. Everything
  else (a nonlinear term, a factor the application computes on the fly, a reduction) is a
  **block callback**: an ordinary function pointer compiled by the application with its own
  compiler and flags, which FFTN calls on a block of data while it is in cache, between two of its
  passes. User code never enters an ISA region; there are no per-element functors.
- **Two places to attach it.** Plans gain an input and an output edge (vocabulary and callbacks).
  A new `pipeline` owns a fixed form: *prologue → first transform → middle → second transform →
  epilogue*, the second transform the inverse of the first over the same axes, with an optional
  fan-out (one forward, several inverses). The convolution and the spectral pair of DESIGN 12.2
  are this one type with different middles. No general composition graph.
- **Code at construction, data at execution.** A plan or pipeline records which operations and
  which callback functions exist; the factor arrays and the callbacks' `void*` data arrive at each
  execution in an `exec_args`, so one plan serves every stage of a time step and stays immutable
  and safe to execute concurrently.
- **Zero-padding and cropping are part of the shape**, as runs of stored elements inside a longer
  logical line, because the overlap check and the tight default strides must know the stored
  extent. `valid` and `embed` of DESIGN 12.1 become one `embed`.
- **`axes_*` stay as agreed**, with one compatible extension when embeddings land: an optional
  trailing `embeds` argument, so that tight defaults use the stored extents.
- **Guarantees carry over:** execution does not allocate, lock or throw in FFTN's code
  (callbacks are `noexcept` by type); the output is bitwise identical for serial, team and
  executor execution for a given pipeline as long as each callback's result depends only on its
  block's values and coordinates; a pipeline may differ from the unfused composition in the last
  bits, within the accuracy contract.
- **Order of work** (section 8): vocabulary edges, then callbacks, then row pipelines (built from
  the Bluestein stage's fused convolution, which exists), then multi-axis pipelines (the
  free-space stage). Experiments F1–F5 (section 9) measure bytes saved against time on the Mac,
  the Xeon and the EPYC.

---

## 2. The problem, by bytes moved

The client application's FFT work sits between pointwise passes, and most of its non-FFT time is
data movement around the transforms (DESIGN section 2, E9). Four workloads, largest first. One
*sweep* reads and writes an array once; V is the array's size.

| Workload | Today: sweeps over memory | Fused: sweeps | What fusion removes |
|---|---|---|---|
| **Free-space stage.** A complex field stored as (y, x, ω), ω contiguous, `nω` stored bins of an `nt`-point time line. Stage input times a propagator, inverse 2-D on (y, x), pack (crop to `nt`, scale), inverse ω, a pointwise nonlinear term with a time window, forward ω, crop, forward 2-D, finish (times a prefactor, a window and the inverse propagator) into a stage register | about 8 (two crossings between the state's order and a plane-major scratch, the planes twice, pack, the rows twice, the nonlinear pass, crop): about 16 V of traffic, 28 GB a stage at 384² × 750 (V = 1.77 GB) | 3: [input, propagator, inverse y and x] [ω pencils: inverse, nonlinear term, forward] [forward x and y, finish]: 6 V, 11 GB | the two crossings, pack, crop, the separate nonlinear pass; in the client's 192² case pack, crop and fold are about 56 % of the window time |
| **Convolutions.** Real rows of 2m points whose upper half is zero, r2c, times a kernel spectrum (chosen per row), c2r of which the first m are kept; and one forward feeding K kernel-multiplied inverses | per row in units of m reals: drive written 2, r2c read 2 and written 2, multiply 4, c2r read 2 and written 2, first half read 1: 15 (and K times the inverse part) | 2 (read the m valid inputs, write the m wanted outputs); with prologue and epilogue callbacks the drive formation and the finish are absorbed too | the spectrum never reaches memory; pruned edges halve the first and last passes |
| **Split-step on rows.** In place: inverse, a factor per time sample, forward | 3 | 1 | two sweeps and two parallel regions per call |
| **Transform, then reduce.** Statistics and monitors read the transformed rows only to sum over them | transform 1, reduction ½ | ½ (read only) | the output is never stored |

Points that matter:

- The volume figures are the client's own estimate from its profile (the fused count is a
  design target, not a measurement). They are upper bounds on the saving: the propagator
  exponentials and the nonlinear term's arithmetic stay, now inside callbacks.
- Three sweeps assume each transform sweep is one pass over memory. A plane sweep over strided
  (y, x) runs over blocks of B adjacent ω pencils (X61's lanes); a block of 192² × 4 complex
  doubles is 2.4 MB, beyond the x86 L2s, so on the Xeon and the EPYC each plane sweep may be two
  passes from L3 while DRAM sees one. F4 measures it.
- Most row batches of the client are 0.1–5 MB, within L2 or L3: there fusion buys passes and
  parallel regions (a region costs 0.6–12 µs, a short row transform 6–40 µs) rather than DRAM
  bytes. The free-space stage and its volume are DRAM-bound.
- A plain 3-D transform of the volume, with nothing fused, was measured by the client and was no
  faster than planes plus rows (E9). The gain is in what moves with the transform, not in the
  transform.

---

## 3. What the API must express

From the four workloads, and from DESIGN 12:

1. Transforms over **strided axes in the array's own order** (no packing), any subset of one to
   three axes, as `axes_*` and the general `shape` already describe.
2. **Pointwise work at three places**: before the first pass (prologue), between an inverse and
   a forward on the same lines (middle), after the last pass (epilogue).
3. Work that is **linear and separable** (a real or complex factor indexed by some coordinates,
   broadcast along the others; a scale; accumulation) and work that is **not** (a cubic term,
   `exp(δL)` computed from the coordinates, a reduction).
4. **Zero-padding and cropping per axis**: stored elements at runs of positions inside a longer
   logical line (the stored band of `nω` bins inside `nt`; the valid first half of a
   delayed-response convolution's row), both on input (the rest is zero and is not read) and on output (the rest is not written).
5. **One forward, several inverses**, each with its own kernel, its outputs stored or consumed.
6. **Coordinates** for callbacks: which element of which line of the array each value is, in
   logical indices (time index for a window, (kx, ky, ω) for a propagator).
7. Sometimes several batch members **together** (two polarisation components that a nonlinear
   term couples).
8. All of this under the existing contracts: no allocation, lock or throw at execution; serial,
   team and executor execution with bitwise identical results; float and double; a plan that
   says what it did (`describe()`).

---

## 4. Choices and alternatives

### 4.1 Where fusion attaches

| | A. Hooks on plans only | B. General composition | C. Edges on plans and one fixed-form pipeline (recommended) |
|---|---|---|---|
| What it is | Plans gain input and output edges; applications chain plans themselves | A pipeline built from a list of stages (embed, transform, pointwise, transform, extract, ...), FFTX-style; the planner fuses the patterns it recognises | Plans gain edges; a `pipeline` is exactly prologue → T1 → middle (× K) → T2 → epilogue, T2 the inverse of T1 on the same axes |
| Free-space stage | No: the middle needs both ω transforms in one object, and separate plans order axes by stride, so a forward plan would not run ω first | Yes | Yes |
| Convolution, fan-out | The spectrum reaches memory (one more read and write of it per row) | Yes | Yes |
| Plane-at-a-time when everything fits in cache (one sweep instead of three) | No | Yes | Yes: the pipeline sees both transforms |
| Parallel regions per stage | One per plan | One | One |
| API surface | Smallest | Largest: a stage language, its validation, and rules for which sequences fuse | Moderate: one edge type, one pipeline type |
| Risk of silently slow descriptions | None | High: a description that does not match a fusible pattern runs unfused, and every such case must be found and reported | Low: every valid description has a fused program; `describe()` and `sweeps()` report it |
| Later needs (DCT pre- and post-passes, dealiased products, resampling) | Edges cover pre- and post-passes | All | Edges cover pre- and post-passes; a different T2 length (resampling) would be an extension (question 10) |

**Recommendation: C.** It covers all four workloads and DESIGN 12's convolution and spectral pair
with one type, keeps the planner's job bounded (one form, pinned axis order, known sweeps), and
its edges on plans are A, so A's cases need no pipeline. B's generality has no client behind it
yet (AGENTS section 1: no feature without a motivating use) and its main cost, deciding which
compositions fuse, is exactly the "never silently slower" rule's hardest case. If B is wanted
later, C's pipeline is one of its fusible patterns and nothing in C is lost.

### 4.2 How pointwise code is written

| | Per-element functors inlined into kernels | Per-element function pointers (the legacy GPU-callback style) | Fixed vocabulary only | Block callbacks (recommended, with the vocabulary) |
|---|---|---|---|---|
| Where user code runs | Inside FFTN's kernels | Called from FFTN's kernels per element | Nowhere | Called by FFTN's driver between passes, outside any ISA region |
| Feasible in a compiled library? | No. Kernels are compiled once, per ISA region, in the library's translation units (DESIGN 6.4, 14.2); a functor would have to be compiled inside every region with lambdas and standard templates, which AGENTS section 4 forbids for good reasons (wrong target, merged instantiations), or the kernels would move into headers, which experiment 14.1 rejected | Yes | Yes | Yes |
| Speed | Best in principle (no extra memory touch) | An indirect call per element, an estimated 1–2 ns, more than the transform's own cost per element | Best: fused into a pass's loads or stores | One indirect call per block (negligible) and one extra pass over the block in L1 or L2, estimated at a small fraction of a DRAM sweep (F2 measures it) |
| Expressiveness | Anything | Anything | Linear separable work only. A propagator `exp(δL)` would have to be precomputed per stage: field-sized arrays (one per δ of a Runge-Kutta tableau) and one more read sweep each | Anything pointwise or blockwise; whole lines guaranteed in the middle |
| User's compiler flags and vector code | n/a | Yes | n/a | Yes: the application vectorises its callback as it likes (its own dispatch, Kokkos SIMD, `#pragma omp simd`) |
| C, Fortran, Kokkos | No | Yes | Yes | Yes: a function pointer, a `void*` and a plain struct |

**Recommendation: the vocabulary for linear separable work, block callbacks for the rest.** This
is DESIGN 12.1 and 12.3 as written, made concrete. The block callback keeps the bytes saving,
which is where the gain is, without putting user code into ISA regions; the vocabulary keeps the
common linear cases at full speed inside the passes. A per-element functor would be faster by one
in-cache pass over the block at most, and it is not buildable in this library's structure.

### 4.3 Code at construction, data at execution

The plan records which vocabulary operations exist (they select kernel variants) and which
callback functions exist (they shape the program). The arrays (factors) and the callbacks' data
pointers arrive with each execution. Alternatives: data fixed at construction (cuFFT's
`callerInfo`) makes a plan per stage of a time step, or makes the application mutate shared
state between calls, which breaks concurrent execution of one plan. Data at execution costs one
small struct per call.

### 4.4 Zero-padding and cropping belong to the shape

An axis that is stored in part (`nω` bins of an `nt`-point line) changes the memory a transform
touches. The overlap check that plan construction runs and the tight default strides both depend
on the stored extent: with the logical extent, a compact (y, x, ω) array whose x stride is `nω`
would look as if its ω lines overlapped, and would be refused. So an embedding is part of the
shape, not an option. DESIGN 12.1's `valid` (a sub-range is non-zero, the rest not read) is the
case where stored and logical positions coincide; one `embed` covers both.

---

## 5. The proposed API

### 5.1 `fftn/edge.hpp`: the vocabulary and callbacks (new header)

```cpp
namespace fftn {

// A run of stored elements of an axis: `count` elements stored from position `stored` of the
// axis's memory are logical positions `logical` onwards of the transformed line.
struct run { std::size_t logical = 0, stored = 0, count = 0; };

// An axis stored in part. Input: the elements outside the runs are zero and are not read.
// Output: they are not written. An empty side stores the whole line (stored = logical).
// The axis's stored extent on a side is the end of its last run.
struct embed {
  std::size_t axis = 0;          // index into shape::axes (in axes_*: into the array)
  std::vector<run> in, out;
};

enum class factor_kind : unsigned char { none, real, complex };
// Times an array read at execution, in the plan's precision (T or std::complex<T>), indexed by
// logical coordinates through its own strides: one per axis and one per batch dimension, in
// elements of the array; 0 broadcasts along that dimension (a window: {1} along time, 0 elsewhere).
struct factor {
  factor_kind kind = factor_kind::none;
  std::vector<std::ptrdiff_t> axis_strides, batch_strides;
};

// Where a block lies, for callbacks. Coordinates are logical indices along the shape's axes
// (positions in the logical line on an embedded axis) and the transform's number in the batch.
struct block_info {
  std::size_t transform = 0;            // batch number of the block's transform
  std::span<const std::size_t> origin;  // index along each axis of element 0 of line 0
  std::size_t element_axis = 0;         // elements of a line advance along this axis
  std::size_t line_axis = 0;            // lines advance along this axis, or along_transforms
  std::size_t length = 0, lines = 0;    // elements per line, lines in the block
  std::ptrdiff_t element_stride = 0, line_stride = 0;  // within `data`, in elements
  std::size_t group = 1;                // batch members seen together (pipeline_desc::group)
  std::ptrdiff_t group_stride = 0;
  std::size_t copy = 0;                 // which inverse of a fan-out (pipeline_desc::fanout)
  int member = 0;                       // the team member running the block (0 serially)
};
inline constexpr std::size_t along_transforms = SIZE_MAX;

// A callback: noexcept is part of the type, so a function that may throw does not convert.
template <class T> using block_fn = void (*)(void* user, const block_info&, T* data) noexcept;
enum class callback_mode : unsigned char {
  modify,   // FFTN loads the block (input edge) or stores it afterwards (output edge)
  produce,  // input edge only: the callback fills the block; FFTN reads nothing from `in`
  consume,  // output edge only: the callback takes the block; FFTN stores nothing
};
template <class T> struct callback { block_fn<T> fn = nullptr; callback_mode mode = callback_mode::modify; };

// A block_fn that calls (*static_cast<F*>(user))(info, data): lets a functor or a lambda be a
// callback. Compiled in the caller's translation unit with the caller's flags.
template <class F, class T> void thunk(void* user, const block_info& b, T* d) noexcept;

// One edge of a transform. Input: [load or produce] -> callback -> vocabulary in the first pass.
// Output: vocabulary in the last pass -> callback -> [store, accumulate or consume].
template <class T> struct edge {
  double scale = 1.0;
  factor mul;
  bool accumulate = false;       // output only: out += result
  callback<T> call;
};

// The data of one execution, read during that call only.
struct exec_args {
  const void* in_factor = nullptr;   // the arrays of the edges' and the middle's factors,
  const void* mid_factor = nullptr;  // each addressing logical coordinate 0 (stored on edges)
  const void* out_factor = nullptr;
  void* in_user = nullptr;           // the `user` argument of each callback
  void* mid_user = nullptr;
  void* out_user = nullptr;
};
}
```

### 5.2 Additions to shapes and plans

```cpp
template <class In, class Out> struct shape {
  direction dir; std::vector<axis> axes; std::vector<batch> batches;
  std::vector<embed> embeds;       // new: at most one per axis
};
template <class T> shape<std::complex<T>, std::complex<T>>
    axes_c2c(std::span<const axis> array, std::span<const std::size_t> axes, direction,
             std::span<const embed> embeds = {});        // new trailing argument; r2c, c2r alike

// The edges of a plan: typed by each side's element, so not a member of the untyped options.
template <class In, class Out> struct plan_edges { edge<In> in; edge<Out> out; };

template <class In, class Out> class plan {
  explicit plan(const shape<In, Out>&, const options& = {}, const plan_edges<In, Out>& = {});
  // ... as now (options::scale stays, the shortcut for out.scale), plus each execute overload
  // with a trailing exec_args:
  void execute(const In*, Out*, workspace&, const exec_args&) const noexcept;
  void execute(const In*, Out*, const team&, workspace&, const exec_args&,
               completion = completion::all) const noexcept;
  void execute(const In*, Out*, executor_ref, workspace&, const exec_args&) const;
};
```

### 5.3 `fftn/pipeline.hpp` (new header)

```cpp
namespace fftn {

// Kinds by element types: <C, C, C> two complex transforms; <T, C, T> r2c then c2r (a real
// convolution); <C, T, C> c2r then r2c (a spectral pair with a real middle). C = std::complex<T>.
template <class In, class Mid, class Out> struct pipeline_desc {
  direction first = direction::forward;  // c2c: the first transform's; the second is the opposite
  std::vector<axis> axes;                // as in shape: logical lengths; in strides on the In
  std::vector<batch> batches;            //   side, out strides on the Out side
  std::vector<embed> embeds;
  std::size_t fused = last_axis;         // the axis the middle runs on: the first transform's last
                                         //   axis and the second's first
  edge<In> prologue;
  struct { double scale = 1.0; factor mul; callback<Mid> call; } middle;   // call: modify only
  edge<Out> epilogue;
  std::size_t fanout = 1;                // inverses per forward, each with its own middle factor
  std::ptrdiff_t fanout_out_dist = 0, fanout_factor_dist = 0;
  std::size_t group = no_group;          // a batch dimension the middle callback sees whole
};
inline constexpr std::size_t last_axis = SIZE_MAX, no_group = SIZE_MAX;

// The geometry, embeddings and first direction of a complex shape, as a pair.
template <class T> pipeline_desc<std::complex<T>, std::complex<T>, std::complex<T>>
    pair_of(const shape<std::complex<T>, std::complex<T>>&);

template <class In, class Mid, class Out> class pipeline {
public:
  explicit pipeline(const pipeline_desc<In, Mid, Out>&, const options& = {});
  void execute(const In* in, Out* out, workspace&, const exec_args& = {}) const noexcept;
  void execute_transforms(std::size_t first, std::size_t count, const In*, Out*, workspace&,
                          const exec_args& = {}) const noexcept;
  void execute(const In*, Out*, const team&, workspace&, const exec_args& = {},
               completion = completion::all) const noexcept;
  void execute(const In*, Out*, executor_ref, workspace&, const exec_args& = {}) const;
  transform_range transforms_of(const team&) const noexcept;
  std::size_t workspace_bytes(int lanes = 1) const noexcept;
  std::size_t sweeps() const noexcept;   // passes over the caller's memory the program makes
  std::string describe() const;          // the sweeps, what is fused, anything that is not
  isa instruction_set() const noexcept;
};
}
```

Points that matter:

- **The form.** T1 runs the axes with the fused axis last; the middle runs on whole lines of the
  fused axis (its logical length, embedding applied in scratch); T2 runs the fused axis first and
  the others after. Where everything fits a cache level the pipeline may run a transform at a
  time in one sweep; otherwise each outer-axis phase of T1 and T2 is a sweep, and the
  intermediate lives in `out` (a multi-sweep pipeline is in place on `out` after its first
  sweep). That is why an accumulating epilogue on a multi-sweep pipeline is refused (question 9):
  an epilogue callback in `consume` mode can accumulate into the application's own array instead.
- **Edges of a pipeline** are its outer edges: the prologue applies at T1's first pass, whatever
  axis that is, and the epilogue at T2's last pass. In a multi-sweep pipeline they run in the
  first and last sweeps.
- **Fan-out.** T2 runs `fanout` times on the same middle spectrum, copy k with its middle factor at
  `mid_factor + k fanout_factor_dist` and its output at `out + k fanout_out_dist` (or consumed by
  the epilogue callback, which sees `block_info::copy`). Single-sweep pipelines only (question 8).
- **Tables are shared** between T1 and T2 (same lengths, both directions), which realises the
  two-way plan of DESIGN 18.2 (a) inside the pipeline.
- **The spectral layout** (X9): the middle is in natural order. A vocabulary-only middle could
  later run in the plan's own order (an option and a `spectral_layout`, DESIGN 12.2); callbacks
  always see natural order, because their coordinates must be affine in the block.

### 5.4 Guarantees

- **Precision.** `float` and `double` as plans. Factor arrays are in the plan's precision;
  `scale` is converted once to it, as `options::scale`. Callbacks see the plan's precision.
- **Execute contract.** FFTN's own code does not allocate, lock, throw or do I/O. Callbacks are
  `noexcept` by type; a callback that allocates or locks makes the execution do so, which is the
  application's choice. A callback must not use the workspace it is called under, must not write
  outside its block except to the application's own memory, and must not assume an order of
  blocks.
- **Threading.** Every execution form of a plan exists for a pipeline. One team or executor call
  covers the whole pipeline: its sweeps are phases of the work graph (DESIGN 13.2), members join
  between them, and blocks are the chunks. A callback runs concurrently with others on distinct
  blocks, by whichever member claims the block; `block_info::member` names it, for per-member
  scratch. Owner-computes (`completion::own`, `transforms_of`) works as for plans.
- **Determinism.** Block boundaries are fixed by the pipeline, never by the team. So, for a given
  pipeline, the output is bitwise identical for serial execution, any team size, any executor
  and any schedule, provided each callback's result depends only on its block's values and
  coordinates (not on `member`, timing or call order). A pipeline may differ in the last bits
  from the same work done by separate plans and loops (fused multiplications, the pinned axis
  order's radices); docs/conventions.md "Determinism" would list it with the edge operations.
- **Accuracy.** The transform's bound plus one rounding per vocabulary multiplication; tests
  derive the tolerance before running (AGENTS section 5).
- **Errors.** Construction validates everything (runs inside the logical length and not
  overlapping; factor strides within range; callback modes on the right edge; `fused` and `group`
  in range; in-place layouts) and throws `std::invalid_argument` naming the field, for example
  `pipeline_desc.embeds[0].in[1]`.
- **Nothing silent.** `describe()` lists the sweeps and each fused operation, and says when part
  of the work could not be fused (an embedding done by copy, a callback on a copied tile).

### 5.5 How it relates to the rest

- **`axes_*` and `shape`**: pipelines take the same geometry (axes with logical lengths, batch
  dimensions, embeddings); `pair_of` converts a complex shape, so `axes_c2c` builds a pair's
  geometry. Callbacks' coordinates are along the shape's axes plus the transform number, whose
  order `axes_*` already documents (batch dimensions by input stride, widest first, merged where
  they nest).
- **Plans**: a plan with edges is a pipeline without a middle; both share the edge type and the
  `exec_args`. `options::scale` remains as the shortcut it is.
- **Executors and teams**: unchanged contracts; a pipeline is one execution.
- **`fftn/kokkos.hpp`**: the adapter gains `array_of(view)` (the `std::vector<axis>` of a View,
  which `shape_of` computes already) so pipelines can be built from Views, and documents `thunk`
  with host functors. Kokkos-FFT has no fusion, so `compat` needs nothing; device Views stay
  refused at compile time.
- **The C API** (phase 6) maps directly: a function pointer, a `void*` and plain structs.

---

## 6. Examples

Sketches, not compiled.

### 6.1 The free-space stage

One stage of an interaction-picture Runge-Kutta step. The field is (ny, nx, nω) complex, ω
fastest; its `nω` stored bins sit at two runs (positive and negative frequencies) of an `nt`-point
time line.

```cpp
using C = std::complex<double>;
struct stage_data { double delta; /* propagator parameters, window, prefactor */ };

// The application's own functions, compiled with its own flags; each runs on a block in cache.
void to_interaction(void* u, const fftn::block_info& b, C* d) noexcept; // times exp(+delta L(kx, ky, w))
void nonlinear(void* u, const fftn::block_info& b, C* d) noexcept;      // whole time lines: window(t) * N(u) / nt
void finish(void* u, const fftn::block_info& b, C* d) noexcept;         // times prefactor * window(w) * exp(-delta L)

const fftn::axis array[] = {{.n = ny}, {.n = nx}, {.n = nt}};     // logical lengths, tight defaults
const std::size_t dims[] = {0, 1, 2};                             // the last listed (w, t) is fused
const fftn::run band[] = {{.logical = 0, .stored = 0, .count = a},
                          {.logical = nt - b, .stored = a, .count = b}};   // a + b = nw
const fftn::embed crop[] = {{.axis = 2, .in = {band, band + 2}, .out = {band, band + 2}}};

auto d = fftn::pair_of(fftn::axes_c2c<double>(array, dims, fftn::direction::backward, crop));
d.prologue.call = {to_interaction};          // modify: on each block of `in` as it is loaded
d.middle.call = {nonlinear};                 // modify: whole t lines, natural order
d.epilogue.call = {finish};                  // modify: on each block before it is stored
const fftn::pipeline<C, C, C> stage(d);      // y, x then t backward; t then x, y forward

fftn::workspace ws(stage, executor.concurrency());
stage_data s{.delta = h / 2};
stage.execute(state, k2, executor, ws, {.in_user = &s, .mid_user = &s, .out_user = &s});
// `state` is preserved; k2 receives the stage and holds the intermediate between sweeps.
// stage.sweeps() == 3 where the planes do not fit a cache level.
```

The prologue sees, in the first sweep, blocks of B adjacent ω values along a y line (element axis
0, line axis 2); the middle sees pencils of `nt` time samples (element axis 2), the embedding
already applied; the epilogue sees the last sweep's blocks. Each callback finds every element's
logical (y, x, ω or t) index from `origin`, the two axes and the two strides. With two coupled
polarisation components stored as a batch dimension, `d.group = 0` gives the middle both
components of each pencil at once.

### 6.2 A convolution: one forward, several kernel-multiplied inverses

Real rows of 2m points whose upper half is zero, stored compactly (m reals a row); K kernel
spectra of m + 1 bins each; the first m samples of each result wanted.

```cpp
using C = std::complex<double>;
const std::size_t n = 2 * m;
const auto pm = static_cast<std::ptrdiff_t>(m), prows = static_cast<std::ptrdiff_t>(rows);
const fftn::run half[] = {{.logical = 0, .stored = 0, .count = m}};
fftn::pipeline_desc<double, C, double> d{
    .axes = {{.n = n, .in_stride = 1, .out_stride = 1}},
    .batches = {{.count = rows, .in_dist = pm, .out_dist = pm}},   // m stored reals a row
    .embeds = {{.axis = 0, .in = {half, half + 1}, .out = {half, half + 1}}},
    .middle = {.scale = 1.0 / static_cast<double>(n),
               .mul = {.kind = fftn::factor_kind::complex, .axis_strides = {1}, .batch_strides = {0}}},
    .fanout = K, .fanout_out_dist = prows * pm, .fanout_factor_dist = pm + 1,
};
const fftn::pipeline<double, C, double> conv(d);   // r2c, K x (multiply, c2r), one sweep
fftn::workspace ws(conv);
conv.execute(drives, histories, ws, {.mid_factor = kernel_spectra});
```

A kernel chosen per row is a second batch dimension with its own factor stride (rows as
(chunks, drives), `batch_strides = {0, pm + 1}`). A contraction over the K results (a
weighted sum) is an epilogue callback in `consume` mode that adds each copy's block into the
application's accumulator, so the K results are never stored.

### 6.3 Two more, briefly

- **Split-step on rows, in place**: `pair_of(rows_c2c<double>(nt, rows, direction::backward))`,
  `middle.mul` a real factor along time (`axis_strides = {1}`, broadcast over rows), `middle.scale
  = 1/nt`, `options::in_place`; a monitoring reduction is a middle callback that sums what it
  removes into a per-member slot.
- **Transform, then reduce**: a plan with `out_edge.call = {sum_power, callback_mode::consume}`;
  `out` is never written.

---

## 7. Should `axes_*` change?

**Keep them, with one compatible extension.** The names, the `padded` flag and the batch order
by input stride (agreed on 2026-11-24) need no change for fusion:

- The pipeline takes the same geometry a shape has, so `axes_c2c` builds a pair's geometry
  through `pair_of`; nothing in a pipeline needs a different array description.
- Callbacks need each transform's number to mean something in the application's terms; the
  documented batch order (widest input stride first, merged where nested) already gives it.
- Strided plane axes in the array's own order, the free-space stage's first requirement, are
  exactly what task 4.3 built (X61's lanes over adjacent ω).

The extension: once shapes carry embeddings (task 5.1), `axes_*` take an optional trailing
`std::span<const embed>`, with axis indices into the array, so that the tight default strides use
each side's stored extent. Without it a caller with a compact embedded axis must write every
stride by hand. A defaulted argument changes no existing call.

Alternatives considered: a kind-free geometry builder (`along_axes(array, dims)` returning axes
and batches, with the kind supplied later) would serve the real pipelines (`<T, C, T>`), whose
element types `shape` cannot express; but a real pipeline's sides are real and of the logical
length, so its geometry is simpler than a shape's and needs no builder for rows (question 11).
Reshaping `axes_*` for that alone is not worth it. **Recommendation: drop "provisional" from
`axes_*` when this proposal (or its successor) is accepted.**

---

## 8. What it costs to build

In agent sessions of the size of the phase 4 tasks; estimates.

| Step | What | Internals that change | Size |
|---|---|---|---|
| 5.1 | Vocabulary edges on plans; embeddings in shapes; `exec_args` | `shape.hpp` (`embed`, `run`, `axes_*` argument), new `edge.hpp`, `options.hpp`, `plan.hpp` overloads; `detail/core.hpp` (`shape_view` gains embeddings and edges, execute entries take `exec_args`); validation with stored extents; `pass_args` and the kernel tables grow edge variants (real factor, accumulate, pruned loads and stores; the library grows, which the maintainer has accepted for speed); factor cursors in loops (`moving_slots` grows); multi-axis plans skip zero sub-transforms of embedded axes (restricting the other axes' loops to stored runs) | 2 |
| 5.2 | Block callbacks on plan edges | a callback record in the program; the driver computes coordinates (today it moves byte offsets only); a copy into scratch where a modified input edge would otherwise be read in place; the work graph treats a callback as part of its chunk | 1 |
| 5.3 | Row pipelines: convolution, spectral pair, fan-out, the three kinds | reuse of the Bluestein stage's convolution (forward chain with an output factor in its last pass, backward chain with an input factor in its first, pruned edges, depth-first chains: DESIGN 8.5, X50); a loop over a scratch-resident line, the extension that Bluestein's copies need too (X36b); fan-out over a scratch spectrum | 2 |
| 5.4 | Multi-axis pipelines: the free-space stage | pinned axis order (`order_axes` overridden), sweeps as phases, the middle on blocks of fused-axis pencils (chain on contiguous pencils, lanes on strided), embeddings applied in the block, `group`, `sweeps()` | 3 |
| 5.5 | X9: natural or scrambled middle for vocabulary middles | strategies' last and first passes in the plan's order; `spectral_layout` | 1–2 |
| 5.6 | Documents, examples, Kokkos | `docs/fusion.md` with compiled examples (AGENTS section 4), `kokkos.hpp` `array_of`, fbench pipeline cases and their unfused baselines | 1 |

Tests per step: each vocabulary operation against "transform, then a separate pass" within the
derived tolerance, and against the quadruple-precision reference; canaries around embedded
edges; callbacks that record coverage (each element exactly once, coordinates right); bitwise
equality of serial, team and executor execution with a deterministic callback; the `noexcept`
requirement as a compile-time test; pipelines against the composition of reference transforms
and pointwise loops. The `sweeps()` claim of PLAN task 5.3 is verified with counters (F4).

---

## 9. Experiments

Each becomes an experiment file with its decision rule before its judging run; machines: the
M5 Max, the Xeon 6240 and the EPYC 7763, at one thread and at full width (6, 8 and 8 lanes),
with fbench's interleaved A/B and an A/A control.

| ID | Question | Arms and cases | Metric |
|---|---|---|---|
| F1 | Do vocabulary edges pay, size by size? | each operation fused against the same plan plus a separate pass; rows, planes, volumes from L1 to DRAM | time; a fused operation that loses beyond the band on a profile runs as a separate pass there, said by `describe()` |
| F2 | What does keeping user code outside the kernels cost, and which block size? | an identity and a cubic callback against the vocabulary doing the same multiplication; blocks of one line, B lines, a tile; the scratch copy of a modified input edge; the callback compiled for the baseline and for AVX2 | time per element; sets the block size per profile |
| F3 | Convolution pipeline (PLAN task 5.2) | against three separate calls and against FFTW r2c, a loop, c2r; real rows of 2m = 18432 to 131072, 1 to 64 rows, valid halves, fan-out 1, 4, 16 | time; acceptance at or below 0.8 of three separate calls for L2-resident sizes |
| F4 | Free-space stage (PLAN task 5.3) | (y, x, ω) at 128² × 864, 192² × 384, 256² × 864, 384² × 750 against today's route built from FFTN plans (planes, pack, rows, crop, planes, finish as loops with the same callbacks) and against three separate plans with edges (A of 4.1, to price what the pipeline object adds) | time and memory traffic: VTune's uncore counters on the Xeon, perf on the EPYC where the cluster allows it, time against a bandwidth model on the Mac; 384² × 750 needs about 4 GB |
| F5 | Scrambled middle (X9, PLAN task 5.4) | natural against the plan's order for vocabulary middles | time |

---

## 10. Open questions for the maintainer

1. **Structure**: accept C (edges on plans, one fixed-form pipeline) over A and B?
2. **Pointwise code**: accept the vocabulary plus block callbacks, and no per-element functors?
3. **Data at execution**: factor arrays and callback data in an `exec_args` at each execution
   (recommended), rather than fixed at construction?
4. **Embeddings in the shape**, `valid` merged into `embed`, and `axes_*` gaining the optional
   `embeds` argument?
5. **The fused axis**: the last listed by default, with `pipeline_desc::fused` to choose another
   (needed for multi-axis real pipelines, whose halved axis is last but cannot be fused)?
6. **`group`**: give the middle callback whole batch members together (coupled polarisation
   components) in the first version, or later?
7. **Callbacks' block shape**: FFTN chooses the layout and reports strides (recommended: it keeps
   lanes tiles without a transpose), or every block is contiguous lines (simpler for users, a
   transpose for strided axes)?
8. **Fan-out outputs**: one array with a distance (as sketched), or K pointers in `exec_args`?
   And single-sweep pipelines only?
9. **Multi-sweep pipelines keep their intermediate in `out`**: accept that an accumulating
   epilogue is then refused (a `consume` callback can accumulate), or add a caller-supplied
   intermediate array?
10. **Scope**: is a second transform of a different length (Fourier resampling) out of scope for
    phase 5?
11. **Real pipelines over several axes** (`<T, C, T>` over planes): the half spectrum
    intermediate needs a padded `out`; restrict multi-axis pipelines to c2c in phase 5?
12. **Order of work**: rows first (5.3, smaller, the convolutions) or the free-space stage
    first (5.4, the larger saving)? Section 8 assumes rows first, since 5.4 reuses 5.3's pair.
13. **Names**: `pipeline`, `pipeline_desc`, `edge`, `embed`, `run`, `factor`, `exec_args`,
    `block_info`, `thunk`.
14. **`options::scale`**: keep it as the shortcut for the output edge's scale (recommended), or
    move it into the edge?
