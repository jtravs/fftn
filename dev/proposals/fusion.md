# Proposal: the phase 5 fusion API

**Proposal, not agreed.** Drafted on 2026-11-24 for the maintainer, who may accept, change or reject
any part of it, and revised the same day after an independent critical read (see
[Revisions](#revisions)). Until a version is agreed, [DESIGN.md](../DESIGN.md) section 12 remains
the design, and nothing here may be implemented: AGENTS.md section 7 requires the maintainer's
agreement for public API changes. Once a version is agreed, its parts move into DESIGN sections 5
and 12 and into the decision log, and this file is deleted, so that each fact lives in one place.
Every header and example below is a sketch: abbreviated, not compiled, and not a commitment to
exact names.

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
10. [Decisions](#10-decisions)

---

## 1. Summary

- **Two kinds of pointwise work, two mechanisms.**
  - Linear, separable work (scale, real or complex factors, zero-padding and cropping,
    accumulation) uses a small fixed vocabulary that FFTN compiles into its own passes, as it
    already does for `options::scale` and the internal `factor` edge.
  - Everything else (a nonlinear term, a factor the application computes on the fly, a reduction)
    is a **block callback**. A block callback is an ordinary function pointer, compiled by the
    application with its own compiler and flags. FFTN calls it on a block of data while the block
    is in cache, between two of FFTN's passes.
  - User code never enters an ISA region, and there are no per-element functors.
- **Two objects in phase 5.**
  - *Edges on plans*: an input edge and an output edge, each taking vocabulary and callbacks.
  - *A row pipeline*: one transformed axis, which may be strided, under any number of batch
    dimensions. Its form is: prologue → transform → middle → inverse transform → epilogue, with
    an optional fan-out (one forward feeding several inverses).
  - The convolution and the spectral pair of DESIGN 12.2 are this one type with different
    middles.
- **The free-space stage is built from three of these objects** in sequence:
  1. an inverse plane plan with a prologue edge;
  2. a row spectral pair over the contiguous ω pencils;
  3. a forward plane plan with an epilogue edge.

  A multi-axis pipeline is built later only if experiment F4 shows that it moves fewer bytes than
  this three-object route at the client application's sizes (section 4.1).
- **Code at construction, data at execution.** A plan records which operations and callback
  functions exist. Factor arrays and the callbacks' `void*` data arrive with each call in an
  `exec_args`, so one plan serves every stage of a time step and stays immutable.
- **Zero-padding and cropping are part of the shape**, as runs of stored elements inside a longer
  logical line. `valid` and `embed` from DESIGN 12.1 become one feature.
- **`axes_*` stay as agreed.** Their only change is an optional `stored_runs` argument, added
  once embeddings exist. Decision M4 offers an alternative that avoids even that.
- **Guarantees.** FFTN's own code does not allocate, lock or throw at execution, and callbacks
  are `noexcept` by type. Results are bitwise identical across serial, team and executor
  execution for callbacks that are pure per element (on edges) or per line (in the middle).
  Callbacks must be correct for every block shape, because a team can be given narrower blocks
  than a serial run gets.
- **Cost and decisions.** The cost is 15–25 agent sessions, aligned with PLAN tasks 5.1–5.5
  (section 8). Section 10 lists nine decisions for the maintainer (M), each with a
  recommendation, and the decisions the lead may take (L).

---

## 2. The problem, by bytes moved

The client application's FFT work sits between pointwise passes, and most of its non-FFT time is
data movement around the transforms (DESIGN section 2, E9). Four workloads follow, largest first.
In the table, a *sweep* reads and writes an array once, and V is the array's size.

| Workload | Today: sweeps over memory | Fused: sweeps | What fusion removes |
|---|---|---|---|
| **Free-space stage.** A complex field stored as (y, x, ω), ω contiguous, holding `nω` stored bins of an `nt`-point time line. The stage multiplies its input by a propagator, runs an inverse 2-D transform on (y, x), packs (crops to `nt` and scales), runs an inverse transform along ω, applies a pointwise nonlinear term with a time window, runs a forward transform along ω, crops, runs a forward 2-D transform, and finishes by multiplying by a prefactor, a window and the inverse propagator into a stage register | 9 sweeps: two crossings between the state's order and a plane-major scratch, the planes twice, pack, the rows twice, the nonlinear pass and crop. That is about 18 V of traffic, or 32 GB a stage at 384² × 750 (V = 1.77 GB) | 5 sweeps (10 V, about 18 GB) at 384² with the three-object route: the y and x passes are separate sweeps there (see below). 3 sweeps (6 V, 11 GB) is the lower bound, reached where a block of adjacent ω pencils over a whole plane fits a cache level | the two crossings, pack, crop and the separate nonlinear pass. In the client's 192² case, pack, crop and fold are about 56 % of the window time |
| **Convolutions.** Real rows of 2m points whose upper half is zero: r2c, a multiply by a kernel spectrum (chosen per row), then c2r, of which the first m samples are kept. Also one forward feeding K kernel-multiplied inverses | Per row, in units of m reals: drive written 2, r2c read 2 and written 2, multiply 4, c2r read 2 and written 2, first half read 1. That is 15 units, and K times the inverse part | 2 units: read the m valid inputs and write the m wanted outputs. Prologue and epilogue callbacks also absorb the drive formation and the finish | the spectrum never reaches memory, and pruned edges halve the first and last passes |
| **Split-step on rows.** In place: inverse, a factor per time sample, forward | 3 | 1 | two sweeps and two parallel regions per call |
| **Transform, then reduce.** Statistics and monitors read the transformed rows only to sum over them | transform 1, reduction ½ | ½ (read only) | the output is never stored |

Points that matter:

- **When the y and x passes are separate sweeps.** The plane passes run over blocks of B adjacent
  ω pencils (X61's lanes, B at least one cache line of pencils). At 384² one block is
  384² × 4 × 16 B = 9.4 MB. That is larger than every per-core L2, and at full width larger than
  each core's share of L3 (EPYC 7763: 32 MB per 8 cores; Xeon 6240: 24.75 MB per 18). So y and x
  are two sweeps on each side of the ω pass, which gives 5 sweeps. At 192² a block is 2.4 MB, so
  3–5 sweeps depending on machine and width. F4 reports DRAM traffic at 384² at full width.
- **Upper bounds.** The volume figures are bounds on the saving, built from the client's own
  profile. The propagator exponentials and the nonlinear term's arithmetic remain, now inside
  callbacks, and they may be compute-bound (F2).
- **L2- and L3-resident rows.** Most of the client's row batches are 0.1–5 MB, which fits in L2 or
  L3. There, fusion buys fewer passes and fewer parallel regions rather than fewer DRAM bytes: a
  region costs 0.6–12 µs, and a short row transform 6–40 µs. The free-space stage, by contrast,
  is DRAM-bound.
- **A plain 3-D transform does not help.** The client measured a 3-D transform of the volume with
  nothing fused, and it was no faster than planes plus rows (E9). The gain comes from what moves
  with the transform, not from the transform itself.

---

## 3. What the API must express

From the four workloads, and from DESIGN 12:

1. Transforms over **strided axes in the array's own order**, with no packing: any subset of one
   to three axes, as `axes_*` and the general `shape` already describe.
2. **Pointwise work at three places**: before the first pass (prologue), between an inverse and a
   forward on the same lines (middle), and after the last pass (epilogue).
3. Work that is **linear and separable**, and work that is **not**:
   - linear and separable: a real or complex factor indexed by some coordinates and broadcast
     along the others, a scale, accumulation;
   - not: a cubic term, `exp(δL)` computed from the coordinates, a reduction.
4. **Zero-padding and cropping per axis**: stored elements at runs of positions inside a longer
   logical line. Examples are the stored band of `nω` bins inside `nt`, and the valid first half
   of a delayed-response convolution's row. This applies on input (the rest is zero and is not
   read) and on output (the rest is not written).
5. **One forward, several inverses**, each inverse with its own kernel, and each output stored or
   consumed.
6. **Coordinates** for callbacks, in logical indices: which element of which line each value is.
   Examples are the time index for a window, or (kx, ky, ω) for a propagator.
7. Sometimes several batch members **together**, such as two polarisation components that a
   nonlinear term couples.
8. All of this **under the existing contracts**:
   - no allocation, lock or throw at execution;
   - serial, team and executor execution;
   - float and double;
   - a plan that says what it did (`describe()`).

---

## 4. Choices and alternatives

### 4.1 Where fusion attaches

| | A. Edges on plans and row pipelines (recommended) | C. A plus a multi-axis pipeline | B. General composition |
|---|---|---|---|
| What it is | Plans gain edges. A pipeline is prologue → T → middle (× K) → inverse T → epilogue on one transformed axis (strided allowed) under any batch dimensions | A, and also pipelines over two or three axes with a pinned axis order: the fused axis runs last in the inverse and first in the forward | A pipeline built from a list of stages (embed, transform, pointwise, transform, extract, ...), FFTX-style; the planner fuses the patterns it recognises |
| Free-space stage | Three objects: the inverse (y, x) plan with a prologue edge (ω, stride 1, is the batch: X61's lanes), a row pair on the ω pencils, then the forward (y, x) plan with an epilogue edge. 5 sweeps at 384²; 3 sweeps where a block of pencils over a plane fits a cache level | One object. It could also group the axes differently: y alone, then an (x, ω) slab per y (x inverse, ω pairs, x forward), then y. That gives 3 DRAM sweeps where a slab (384 × 750 × 16 B = 4.6 MB) stays in a member's cache: at one thread yes, at full width not on the Xeon and the EPYC | Yes |
| Real-field family (planes complex, time real) | Yes: c2c planes, then the ω pencils as a `<C, T, C>` row pair | Needs real multi-axis pipelines (a padded intermediate) | Yes |
| Convolution, fan-out, split-step, reduction | Yes | Yes | Yes |
| Parallel regions per stage | Three (a few µs against a stage of hundreds of ms) | One | One |
| API surface and planner work | Moderate: one edge type, one row pipeline. Every valid description has a fused program | More: pinned order, sweeps as phases, axis grouping | Largest: a stage language, its validation, and rules for which sequences fuse; a description that matches no pattern runs unfused |
| Phase 5 steps | 5.1–5.3 (section 8) | and a multi-axis pipeline step of about 4–6 sessions | larger than C |

**Recommendation: A, with C's multi-axis pipeline kept as an option.** A covers all four workloads
and the real-field family, and it uses only objects that the convolution and split-step workloads
need anyway. What C adds is the axis grouping, which saves two DRAM sweeps (about 4 V, 7 GB a stage
at 384²), but only where a slab stays in cache, which F4 would have to show at the widths the
client uses. C's other benefits do not matter here:

- C fuses two fork/joins into one, which is worth microseconds;
- C allows "plane at a time when everything fits", which never applies, since the client's
  smallest volume is 226 MB.

B's generality has no client behind it (AGENTS section 1: no feature without a motivating use).
Its hardest part, deciding which compositions fuse, is also the hardest case for the rule that
nothing runs slower silently.

### 4.2 How pointwise code is written

| | Per-element functors inlined into kernels | Per-element function pointers (the legacy GPU-callback style) | Fixed vocabulary only | Block callbacks (recommended, with the vocabulary) |
|---|---|---|---|---|
| Where user code runs | Inside FFTN's kernels | Called from FFTN's kernels for each element | Nowhere | Called by FFTN's driver between passes, outside any ISA region |
| Feasible in a compiled library? | No. Kernels are compiled once per ISA region, in the library's translation units (DESIGN 6.4 and 14.2). A functor would have to be compiled inside every region, with lambdas and standard templates, which AGENTS section 4 forbids for good reasons (wrong target, merged instantiations). The only other route is moving the kernels into headers, which experiment 14.1 rejected | Yes | Yes | Yes |
| Speed | Best in principle: no extra memory touch | An indirect call per element, an estimated 1–2 ns, which is more than the transform's own cost per element | Best: fused into a pass's loads or stores | One indirect call per block (negligible) and one extra pass over the block in L1 or L2 (F2 measures it) |
| Expressiveness | Anything | Anything | Linear separable work only. A propagator `exp(δL)` must then be precomputed: one field-sized array per δ of the Runge-Kutta tableau, each costing a read sweep (F2 weighs this against computing it in a callback) | Anything pointwise or blockwise; the middle always gets whole lines |
| The application's flags and vector code | n/a | Yes | n/a | Yes: the application vectorises its callback as it likes |
| Usable from C, Fortran and Kokkos | No | Yes | Yes | Yes: a function pointer, a `void*` and a plain struct |

**Recommendation: the vocabulary for linear separable work, and block callbacks for everything
else.** This is DESIGN 12.1 and 12.3 as written, made concrete. A per-element functor would save
at most one in-cache pass over the block, and it cannot be built in this library's structure.

### 4.3 Code at construction, data at execution

The plan records which vocabulary operations exist, since they select kernel variants, and which
callback functions exist, since they shape the program. The factor arrays and the callbacks' data
pointers arrive with each execution.

The alternative is data fixed at construction, as with cuFFT's `callerInfo`. That forces either a
plan per stage of a time step, or shared state that the application mutates between calls, which
breaks concurrent execution of one plan.

### 4.4 Zero-padding and cropping belong to the shape

An axis stored in part (`nω` bins of an `nt`-point line) changes the memory a transform touches.
Plan construction's overlap check and the tight default strides both depend on the stored extent.

Measured by logical extent instead, a compact (y, x, ω) array whose x stride is `nω` would look as
if its ω lines overlapped, and it would be refused. So an embedding belongs to the shape, not to
the options. The `valid` operation of DESIGN 12.1 (a sub-range is non-zero, the rest is not read)
is the case where stored and logical positions coincide, so one feature covers both. How to
represent the runs is decision M4.

---

## 5. The proposed API

### 5.1 `fftn/edge.hpp`: the vocabulary and callbacks (new header)

```cpp
namespace fftn {

enum class side : unsigned char { in, out };
// Stored elements of one axis on one side: `count` elements stored from position `stored`
// of the axis's memory are logical positions `logical` onwards of the transformed line. On an
// axis with runs on a side, input elements outside them are zero and not read; output elements
// outside them are not written. The stored extent is the end of the last run.
struct stored_run { std::size_t axis = 0; side where = side::in;
                    std::size_t logical = 0, stored = 0, count = 0; };

enum class factor_kind : unsigned char { none, real, complex };
// Times an array read at execution, in the plan's precision (T or std::complex<T>), indexed by
// logical coordinates through its own strides: one per axis and one per batch dimension, in
// elements of the array; 0 broadcasts along that dimension (a window: {1} along time, 0 elsewhere).
struct factor {
  factor_kind kind = factor_kind::none;
  std::vector<std::ptrdiff_t> axis_strides, batch_strides;
};

// Where a block lies. Coordinates are logical indices along the shape's axes and the
// transform's number in the batch. The block's shape is FFTN's choice and may differ between
// serial and team execution (5.4): a callback must be correct for any block.
struct block_info {
  std::size_t transform = 0;            // batch number of line 0's transform
  std::span<const std::size_t> origin;  // index along each axis of element 0 of line 0
  std::size_t element_axis = 0;         // elements of a line advance along this axis
  std::size_t line_axis = 0;            // lines advance along this axis, or along_transforms
  std::size_t length = 0, lines = 0;    // elements per line, lines in the block
  std::ptrdiff_t element_stride = 0, line_stride = 0;  // within `data`, in elements
  std::size_t group = 1;                // batch members seen together (pipeline_desc::group)
  std::ptrdiff_t group_stride = 0;
  std::size_t copy = 0;                 // which inverse of a fan-out
  int member = 0;                       // 0 <= member < the lanes the workspace was sized for
  std::span<std::byte> scratch;         // the callback's own scratch for this member (5.4)
};
inline constexpr std::size_t along_transforms = SIZE_MAX;

// noexcept is part of the type, so a function that may throw does not convert.
template <class T> using block_fn = void (*)(void* user, const block_info&, T* data) noexcept;
enum class callback_mode : unsigned char {
  modify,   // FFTN loads the block (input edge) or stores it afterwards (output edge)
  produce,  // input edge only: the callback fills the block; FFTN reads nothing from `in`
  consume,  // output edge only: the callback takes the block; FFTN stores nothing
};
template <class T> struct callback {
  block_fn<T> fn = nullptr;
  callback_mode mode = callback_mode::modify;
  std::size_t scratch_bytes = 0;        // per member, in the workspace
};

// The block_fn that calls (*static_cast<F*>(user))(info, data), so a functor or a lambda can
// be a callback; instantiated in the caller's translation unit with the caller's flags.
template <class F, class T> constexpr block_fn<T> callback_of() noexcept;

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
  const void* in_factor = nullptr;   // the factor arrays, each addressing logical coordinate 0
  const void* mid_factor = nullptr;  // (on an edge: stored coordinate 0)
  const void* out_factor = nullptr;
  void* in_user = nullptr;           // each callback's `user`
  void* mid_user = nullptr;
  void* out_user = nullptr;
};
}
```

### 5.2 Additions to shapes and plans

```cpp
template <class In, class Out> struct shape {
  direction dir; std::vector<axis> axes; std::vector<batch> batches;
  std::vector<stored_run> runs;    // new; axis indices into `axes`
};
template <class T> shape<std::complex<T>, std::complex<T>>
    axes_c2c(std::span<const axis> array, std::span<const std::size_t> axes, direction,
             std::span<const stored_run> runs = {});   // new trailing argument (M4); r2c, c2r alike

// The edges of a plan, typed by each side's element type, so they cannot live in the untyped options.
template <class In, class Out> struct plan_edges { edge<In> in; edge<Out> out; };

template <class In, class Out> class plan {
  explicit plan(const shape<In, Out>&, const options& = {}, const plan_edges<In, Out>& = {});
  // ... as now, plus each execute overload with a trailing exec_args:
  void execute(const In*, Out*, workspace&, const exec_args&) const noexcept;
  void execute(const In*, Out*, const team&, workspace&, const exec_args&,
               completion = completion::all) const noexcept;
  void execute(const In*, Out*, executor_ref, workspace&, const exec_args&) const;
};
```

`options::scale` remains as a shortcut. Construction refuses it together with a non-unit
`out.scale`, naming both.

### 5.3 `fftn/pipeline.hpp`: row pipelines (new header)

```cpp
namespace fftn {

// Kinds by element types: <C, C, C> two complex transforms; <T, C, T> r2c then c2r (a real
// convolution); <C, T, C> c2r then r2c (a spectral pair whose middle is real). C = std::complex<T>.
template <class In, class Mid, class Out> struct pipeline_desc {
  direction first = direction::forward;  // c2c: the first transform's; the second is the opposite
  axis line;                             // the transformed axis: logical length, in/out strides
  std::vector<batch> batches;            // as in shape
  std::vector<stored_run> runs;          // axis 0 only
  edge<In> prologue;
  struct { double scale = 1.0; factor mul; callback<Mid> call; } middle;   // call: modify only
  edge<Out> epilogue;
  std::size_t fanout = 1;                // inverses per forward, each with its own middle factor
  std::ptrdiff_t fanout_out_dist = 0, fanout_factor_dist = 0;
  std::size_t group = no_group;          // a batch dimension the middle callback sees whole
};
inline constexpr std::size_t no_group = SIZE_MAX;

// DESIGN 12.2's two pipelines as helpers returning a pipeline_desc (decision M5).
template <class T> pipeline_desc<T, std::complex<T>, T> convolution_rows(std::size_t n, std::size_t count);
template <class T> pipeline_desc<std::complex<T>, std::complex<T>, std::complex<T>>
    spectral_pair_rows(std::size_t n, std::size_t count, direction first);

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
  std::string describe() const;          // what is fused, and anything that is not
  isa instruction_set() const noexcept;
};
}
```

Points that matter:

- **One sweep.** Each line, or block of adjacent lines when the axis is strided, is loaded once,
  transformed, given the middle, transformed back and stored once. The spectrum never reaches
  memory. So there is no intermediate array, and an accumulating epilogue raises no conflict.
- **The middle gets whole lines** of the logical length, with the runs already applied in scratch.
  Lines are in natural order, since callbacks' coordinates must be affine in the block.
- **Fan-out.** The inverse runs `fanout` times on the same middle spectrum. Copy k reads its
  middle factor at `mid_factor + k fanout_factor_dist` and writes its output at
  `out + k fanout_out_dist`, or hands it to the epilogue callback, which sees `block_info::copy`.
- **Tables are shared** between the two transforms, since they have the same length and opposite
  directions. That realises the two-way plan of DESIGN 18.2 (a) inside the pipeline.
- **Scrambled order (X9).** A middle made only of vocabulary could later run in the plan's own
  order, through an option and a `spectral_layout` (DESIGN 12.2). Callbacks always see natural
  order.

### 5.4 Guarantees

- **Precision.** `float` and `double`, as for plans. Factor arrays are in the plan's precision,
  `scale` is converted to it once (as `options::scale` is), and callbacks see the plan's
  precision.
- **Execute contract.** FFTN's own code does not allocate, lock, throw or do I/O. Callbacks are
  `noexcept` by type. A callback that allocates or locks makes the execution do so, which is the
  application's choice. A callback:
  - must not use the workspace it is called under, except its own `scratch`;
  - must not write outside its block, except to the application's own memory;
  - must not assume any order of blocks.

  Whether a callback may itself execute an FFTN plan is decision M7.
- **Threading.** Every execution form of a plan exists for a pipeline, and owner-computes
  (`completion::own`, `transforms_of`) works as it does for plans.
  - Callbacks run concurrently on distinct blocks, each on whichever member claims the block.
  - `member` lies in [0, lanes) for the workspace's lanes: 0 when serial, the rank in a team, and
    the task's index on an executor.
  - `scratch` is a per-member area of `callback::scratch_bytes` bytes, carved from the workspace
    and counted in `workspace_bytes()`. The workspace stays plain bytes, as DESIGN 18.2 requires.
- **Determinism.** Block shapes are not fixed across execution forms. DESIGN 13.2's team programs
  (X43, X43b) give a team narrower lanes blocks than a serial run. Therefore:
  - The output is bitwise identical for serial execution, any team, any executor and any schedule
    when each edge callback is **pure per element** (its result for an element depends only on that
    element's value and coordinates) and each middle callback is **pure per line**.
  - A callback that reduces must not keep partial sums per `member`, whose grouping depends on the
    schedule. It should keep them per line (indexed by the line's coordinates) and let the
    application sum them in a fixed order afterwards.
  - A pipeline may differ in the last bits from the same work done by separate plans and loops,
    because of fused multiplications. docs/conventions.md, "Determinism", would list that next to
    the edge operations.
- **Accuracy.** The transform's error bound plus one rounding per vocabulary multiplication. Tests
  derive the tolerance before they run (AGENTS section 5).
- **Errors.** Construction validates everything and throws `std::invalid_argument` naming the
  offending field, for example `pipeline_desc.runs[1]`. It checks that:
  - runs lie inside the logical length and do not overlap;
  - factor strides are in range;
  - callback modes are on the right edge;
  - `group` names a batch dimension;
  - `options::scale` is not combined with a non-unit output scale;
  - in-place layouts agree.
- **Nothing silent.** An edge that forces a slower strategy is built, not refused (decision L8),
  and `describe()` says so. One example: an input edge on a batch that would otherwise run as
  lanes.

### 5.5 How it relates to the rest

- **`axes_*` and `shape`**: plans with edges take shapes as now, so the free-space stage's plane
  plans come from `axes_c2c`.
  - A row pipeline's geometry is one `axis` plus batch dimensions. For the ω pencils of a
    (y, x, ω) array, the axis is ω with stride 1, and the batch is y and x merged (stride `nω`).
  - Callbacks' transform numbers follow the documented batch order: widest input stride first,
    merged where the dimensions nest.
- **Plans**: a plan with edges and a pipeline share the edge type and `exec_args`.
- **Executors and teams**: their contracts are unchanged, and a pipeline is one execution.
- **`fftn/kokkos.hpp`**: the adapter gains `array_of(view)`, which returns the
  `std::vector<axis>` that `shape_of` already computes, and documents `callback_of` with host
  functors. Kokkos-FFT has no fusion, so the `compat` layer needs nothing, and device Views stay
  refused at compile time.
- **The C API** (phase 6) maps onto this directly: a function pointer, a `void*` and plain structs.

---

## 6. Examples

These are sketches, not compiled code.

### 6.1 The free-space stage, as three objects

This is one stage of an interaction-picture Runge-Kutta step. The field is (ny, nx, nω) complex,
with ω fastest. Its `nω` stored bins sit in two runs (positive and negative frequencies) of an
`nt`-point time line.

```cpp
using C = std::complex<double>;
struct stage_data { double delta; /* propagator parameters, window, prefactor */ };

// The application's own functions, compiled with its own flags; each runs on a block in cache
// and is pure per element (edges) or per line (middle).
void to_interaction(void* u, const fftn::block_info& b, C* d) noexcept; // times exp(+delta L(kx, ky, w))
void nonlinear(void* u, const fftn::block_info& b, C* d) noexcept;      // whole time lines: window(t) * N(u) / nt
void finish(void* u, const fftn::block_info& b, C* d) noexcept;         // times prefactor * window(w) * exp(-delta L)

const fftn::axis array[] = {{.n = ny}, {.n = nx}, {.n = nw}};     // the stored field
const std::size_t planes[] = {0, 1};                              // w is the batch (stride 1)

// 1. Inverse planes, the propagator on loading. 3. Forward planes, the finish on storing.
const fftn::c2c_plan<double> inverse(fftn::axes_c2c<double>(array, planes, fftn::direction::backward),
                                     {}, {.in = {.call = {to_interaction}}});
const fftn::c2c_plan<double> forward(fftn::axes_c2c<double>(array, planes, fftn::direction::forward),
                                     {.in_place = true}, {.out = {.call = {finish}}});

// 2. The w pencils: nw stored bins inside nt, inverse, the nonlinear term, forward, crop.
auto p = fftn::spectral_pair_rows<double>(nt, ny * nx, fftn::direction::backward);
p.line = {.n = nt, .in_stride = 1, .out_stride = 1};
p.batches = {{.count = ny * nx, .in_dist = pnw, .out_dist = pnw}};   // pnw = nw as ptrdiff_t
p.runs = {{.axis = 0, .where = fftn::side::in,  .logical = 0,      .stored = 0, .count = a},
          {.axis = 0, .where = fftn::side::in,  .logical = nt - b, .stored = a, .count = b},
          {.axis = 0, .where = fftn::side::out, .logical = 0,      .stored = 0, .count = a},
          {.axis = 0, .where = fftn::side::out, .logical = nt - b, .stored = a, .count = b}};
p.middle.call = {nonlinear};
const fftn::pipeline<C, C, C> pencils(p, {.in_place = true});

fftn::workspace ws(std::max({inverse.workspace_bytes(lanes), pencils.workspace_bytes(lanes),
                             forward.workspace_bytes(lanes)}));
stage_data s{.delta = h / 2};
inverse.execute(state, k2, executor, ws, {.in_user = &s});   // state is preserved
pencils.execute(k2, k2, executor, ws, {.mid_user = &s});
forward.execute(k2, k2, executor, ws, {.out_user = &s});     // k2 holds the stage
```

The prologue sees blocks of B adjacent ω values along a y line, or a single line where the class
runs the plane as a chain. With two coupled polarisation components stored as a batch dimension,
`p.group` gives the middle both components of each pencil at once. For a real time domain the
pencils are a `<C, T, C>` pair.

### 6.2 A convolution: one forward, several kernel-multiplied inverses

The input is real rows of 2m points whose upper half is zero, stored compactly as m reals a row.
There are K kernel spectra of m + 1 bins each, and the first m samples of each result are wanted.

```cpp
using C = std::complex<double>;
const std::size_t n = 2 * m;
const auto pm = static_cast<std::ptrdiff_t>(m), prows = static_cast<std::ptrdiff_t>(rows);
auto d = fftn::convolution_rows<double>(n, rows);
d.batches = {{.count = rows, .in_dist = pm, .out_dist = pm}};
d.runs = {{.axis = 0, .where = fftn::side::in,  .count = m},
          {.axis = 0, .where = fftn::side::out, .count = m}};
d.middle = {.scale = 1.0 / static_cast<double>(n),
            .mul = {.kind = fftn::factor_kind::complex, .axis_strides = {1}, .batch_strides = {0}}};
d.fanout = K; d.fanout_out_dist = prows * pm; d.fanout_factor_dist = pm + 1;
const fftn::pipeline<double, C, double> conv(d);   // r2c, then K x (multiply, c2r): one sweep
fftn::workspace ws(conv);
conv.execute(drives, histories, ws, {.mid_factor = kernel_spectra});
```

- **A kernel chosen per row** is a second batch dimension with its own factor stride: rows as
  (chunks, drives), with `batch_strides = {0, pm + 1}`.
- **A contraction over the K results** (a weighted sum) is an epilogue callback in `consume` mode.
  It adds each copy's block into the application's accumulator in copy order, so the K results
  are never stored.

### 6.3 Two more, briefly

- **Split-step on rows, in place**: `spectral_pair_rows<double>(nt, rows, direction::backward)`,
  with `middle.mul` a real factor along time (`axis_strides = {1}`, broadcast over rows),
  `middle.scale = 1/nt`, and `options::in_place`. A monitoring reduction is a middle callback that
  writes what it removes into a slot per line, which the application sums afterwards (5.4).
- **Transform, then reduce**: a plan with `out = {.call = {sum_power, callback_mode::consume}}`.
  `out` is never written, and the partial sums go per line, as above.

---

## 7. Should `axes_*` change?

**Keep them.** The names, the `padded` flag and the batch order by input stride, agreed on
2026-11-24, need no change for fusion:

- The free-space stage's plane plans are exactly what `axes_c2c` builds: strided (y, x) in the
  array's own order, with ω as a stride-1 batch running as X61's lanes.
- Row pipelines need one axis and batch dimensions, which is simpler than an `axes_*` selection.
- Callbacks need each transform number to mean something in the application's terms, and the
  documented batch order already provides that.

**One addition, when embeddings land (task 5.1).** A caller with a compact embedded axis needs
default strides that use the stored extent. Decision M4 chooses between two ways to provide this:

- (a) An optional trailing `std::span<const stored_run>` argument on `axes_*`. A defaulted
  argument changes no existing call.
- (b) Runs as a field of `axis`, so that `axes_*` keep their signature.

The lead recommends (a): `axis` stays a plain value type crossing into the library inside
`shape_view` ("no std::vector crosses into the library", DESIGN 5). A vector member would break
that, and a borrowed pointer would dangle in a shape the application keeps.

**Drop "provisional" from `axes_*` when this proposal, or its successor, is accepted.**

---

## 8. What it costs to build

Steps follow PLAN's phase 5 task numbers. Sizes are in agent sessions of the size of the phase 4
tasks. They are estimates, and they are larger than the first draft's because of what exists
today:

- No pruned codelets exist yet: the generator plans them (gen/README.md).
- The internal factor edge works only on c2c chains of one axis (`plan_core.hpp`):
  - the lanes strategy refuses input edge operations;
  - grid and depth-first chains are not built with a factor edge;
  - batch leaves drop out where there is one.

| Task | What | Internals that change | Sessions |
|---|---|---|---|
| 5.1 | Vocabulary edges on plans (scale, real and complex factors, accumulate), stored runs in shapes, `exec_args` | **Headers:** `shape.hpp` (`stored_run`, `axes_*` per M4), new `edge.hpp`, `plan.hpp` overloads, and `detail/core.hpp` (`shape_view` gains runs and edges; execute entries take `exec_args`).<br>**Validation:** with stored extents.<br>**Passes:** `pass_args` and the kernel tables gain variants (real factor, accumulate, pruned loads and stores by run). The library grows, which the maintainer accepted for speed.<br>**Strategies:** factor edges in every strategy the rules choose (lanes, grid, depth-first chain, batch leaves, multi-axis column phases). Factor cursors in loops (`moving_slots` grows). Pruning in multi-axis plans first by restricting the other axes' loops to stored runs, then pruned codelets from the generator | 6–9 |
| 5.2 | Row pipelines in convolution form: `<T, C, T>` and `<C, C, C>`, forward first, kernel factor, runs, fan-out | **Bluestein reuse:** the Bluestein stage's convolution (a forward chain with an output factor in its last pass, a backward chain with an input factor in its first, pruned edges, depth-first chains; DESIGN 8.5, X50).<br>**Strided pencils:** a lanes pair over adjacent pencils.<br>**Programs:** a loop over a scratch-resident line, the extension that Bluestein's copies need too (X36b). Fan-out over a scratch spectrum | 3–5 |
| 5.3 | Block callbacks on plan edges and pipeline middles; `spectral_pair` (inverse first, `<C, T, C>`), `group`, per-member callback scratch; the three-object free-space stage and F4 | **Programs:** a callback record. The driver computes coordinates (today it moves byte offsets only).<br>**Copies:** a scratch copy where a modified input edge would otherwise be read in place.<br>**Work graph:** a callback is part of its chunk.<br>**Coverage:** callbacks in the team programs' blocks | 3–5 |
| 5.4 | X9: scrambled middle for vocabulary middles (optional) | **Strategies:** last and first passes in the plan's own order. A `spectral_layout` | 1–2 |
| 5.5 | Owner-computes with edges and pipelines: `transforms_of` for pipelines, an example and test that run application kernels on the same lines | **Work:** examples and tests, and `transforms_of` for pipelines | 1 |
| docs | `docs/fusion.md` with compiled examples (AGENTS section 4), `kokkos.hpp` `array_of`, fbench pipeline cases and their unfused baselines | | 1–2 |
| (C) | Multi-axis pipeline, only if F4 shows the axis grouping wins | Pinned axis order (`order_axes` overridden), sweeps as phases, slab grouping | 4–6 |

The total is 15–24 sessions without the multi-axis pipeline, and about 20–30 with it.

**Tests per step:**

- each vocabulary operation against "transform, then a separate pass", within the derived
  tolerance, and against the quadruple-precision reference;
- canaries around embedded edges;
- callbacks that record coverage: each element seen exactly once, with correct coordinates,
  under serial and team block shapes;
- bitwise equality of serial, team and executor results with pure callbacks;
- the `noexcept` requirement as a compile-time test;
- pipelines against the composition of reference transforms and pointwise loops.

---

## 9. Experiments

Each experiment gets an experiment file with its decision rule committed before the judging run.
The machines are the M5 Max, the Xeon 6240 and the EPYC 7763, at one thread and at full width (6,
8 and 8 lanes), using fbench's interleaved A/B with an A/A control.

| ID | Question | Arms and cases | Metric |
|---|---|---|---|
| F1 | Do vocabulary edges pay, size by size? And what does an edge cost when it forces a slower strategy? | Each operation fused, against the same plan plus a separate pass, over rows, planes and volumes from L1 to DRAM | Time. A fused operation that loses beyond the noise band on a profile runs as a separate pass there, and `describe()` says so |
| F2 | What does keeping user code outside the kernels cost, which block size should be used, and are callbacks compute-bound? | An identity, a cubic and an **`exp(δL)`-per-element** callback (about 110 million elements twice per stage at 384² × 750) against: the vocabulary doing the same multiplication from a **precomputed factor array** (one read sweep); blocks of one line, B lines and a tile; the scratch copy of a modified input edge; the callback compiled for the baseline and for AVX2 | Time per element, set against the time of one DRAM sweep. This sets the block size per profile, and tells whether computing or reading the propagator wins |
| F3 | Row convolution pipeline (PLAN task 5.2) | Against three separate calls, and against FFTW r2c followed by a loop and c2r. Real rows of 2m = 18432 to 131072 points, 1 to 64 rows, valid halves, fan-out 1, 4 and 16 | Time. Acceptance: at or below 0.8 of three separate calls for L2-resident sizes |
| F4 | Free-space stage (PLAN task 5.3) | (y, x, ω) at 128² × 864, 192² × 384, 256² × 864 and 384² × 750. The three-object route, against today's route built from FFTN plans (planes, pack, rows, crop, planes, and the finish as loops with the same callbacks). On each machine class, the plane plans' own route (X61 lanes or the chain) | Time and **DRAM traffic at 384² at full width**: VTune's uncore counters on the Xeon, perf on the EPYC where the cluster allows it, and time against a bandwidth model on the Mac. 384² × 750 needs about 4 GB. Also the sweep count a slab grouping would save, which decides C |
| F5 | Scrambled middle (X9, PLAN task 5.4) | Natural order against the plan's own order, for vocabulary middles | Time |

---

## 10. Decisions

### The maintainer's (M), each with the lead's recommendation

| # | Decision | Recommendation |
|---|---|---|
| M1 | Structure: edges on plans and row pipelines (A) in phase 5, with the free-space stage built from three objects. The multi-axis pipeline (C) only if F4 shows its axis grouping beats that route; no general composition (B) | Accept |
| M2 | Pointwise code: the vocabulary for linear separable work, block callbacks for the rest, no per-element functors | Accept |
| M3 | Factor arrays and callback data passed at each execution in an `exec_args`, not fixed at construction | Accept |
| M4 | Stored runs are part of the shape, and `valid` merges into them. Representation: (a) a flat list of `stored_run` records in `shape`, plus an optional `axes_*` argument; or (b) runs as a field of `axis`, with no change to `axes_*` | (a), for the plain-struct boundary (section 7) |
| M5 | One `pipeline` type replaces DESIGN 12.2's separate `convolution` and `spectral_pair` types, which survive as the helper functions `convolution_rows` and `spectral_pair_rows` (needs a decision-log entry) | Accept |
| M6 | Determinism with callbacks: bitwise identity across execution forms only for callbacks pure per element (edges) or per line (middle); every callback must be correct for any block shape; reductions keep partial sums per line, not per member | Accept: this narrows nothing that holds today, and it is what the team programs allow |
| M7 | May a callback execute another FFTN plan or pipeline, with a serial overload and its own workspace (a convolution nested in a middle)? | Allowed: serial overloads only, with a workspace taken from the callback's `scratch` (sized at construction from the nested plan's `workspace_bytes()`) or from the application's per-member array; never the outer workspace |
| M8 | Scope of phase 5: a second transform of a different length (Fourier resampling) and real pipelines over several axes are out | Out (the real-field family is covered by `<C, T, C>` rows) |
| M9 | Names: `pipeline`, `pipeline_desc`, `edge`, `stored_run`, `factor`, `exec_args`, `block_info`, `callback_of`, `plan_edges` | Accept, or rename |

### The lead's (L), taken as stated unless the maintainer overrides them

| # | Decision |
|---|---|
| L1 | Order: rows first (5.1, 5.2, then 5.3 with the free-space stage as three objects) |
| L2 | `group` is in the first row pipelines (coupled components) |
| L3 | FFTN chooses each block's layout and reports its strides; blocks are not forced into contiguous lines (that would need a transpose for strided axes) |
| L4 | Fan-out writes one output array with a distance between copies; consumed copies are seen in copy order per line |
| L5 | Accumulation is in the vocabulary on plan output edges and pipeline epilogues; row pipelines have no intermediate array, so nothing conflicts; a `consume` callback may accumulate into the application's own memory |
| L6 | `options::scale` stays, as the shortcut for the output edge's scale; construction refuses it together with a non-unit `out.scale` |
| L7 | `member` lies in [0, lanes) for the workspace's lanes: 0 when serial, the rank in a team, the task's index on an executor |
| L8 | An edge that forces a slower strategy is built, never refused, and `describe()` says so (AGENTS section 1); F1 records the cost |
| L9 | FFTN supplies per-member callback scratch, requested at construction (`callback::scratch_bytes`) and carved from the workspace |

---

## Revisions

2026-11-24, after an independent critical read of the first draft:

- **Structure.** The first draft rejected "edges on plans" for the free-space stage, wrongly. An
  inverse plane plan with a prologue edge, a row pair on the ω pencils and a forward plane plan
  with an epilogue edge do the stage in the same sweeps, using only objects the other workloads
  need. Phase 5 is now edges and row pipelines, and the multi-axis pipeline depends on F4. The
  first draft's questions on the fused axis, the multi-sweep intermediate and multi-axis real
  pipelines are gone.
- **Sweep counts.** Today's route is 9 sweeps (32 GB), not 8. The fused count of 3 is a lower
  bound: at 384² it is 5 sweeps (about 18 GB).
- **Determinism.** The determinism statement contradicted DESIGN 13.2, since team programs change
  block shapes. It is now limited to pure callbacks, and per-member reductions are discouraged.
- **Cost.** The estimate rose from about 10 to 15–24 sessions, given the factor edge's present
  limits and the missing pruned codelets. Steps now follow PLAN's numbering, owner-computes
  included.
- **Experiments.** F2 times an `exp`-per-element callback against a precomputed factor array.
- **Decisions.** They are split into the maintainer's (nine) and the lead's. New entries cover
  nested execution, `member`'s range, callback scratch, slower-strategy edges and replacing DESIGN
  12.2's two types. Renamed: `run` to `stored_run`, `thunk` to `callback_of`. The runs are now a
  flat list of plain records with a side.
