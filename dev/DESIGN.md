# FFTN design

FFTN: the Fastest Fourier Transform in the North. This document is the design as it is
intended now. It is a live document: when work changes the design, edit the section and
add a line to the [decision log](#19-decision-log). The implementation plan and current
status are in [PLAN.md](PLAN.md); the working rules are in [AGENTS.md](../AGENTS.md).

Status of this document: initial design, 2026-11-13, before any library code exists. Every
performance statement about FFTN below is a target or a hypothesis until an experiment in
[PLAN.md](PLAN.md) confirms it. Statements about other libraries are measurements from the
prototype's benchmark log or from the studies listed in [References](#20-references).

Contents:

1. [Goals](#1-goals)
2. [Evidence the design rests on](#2-evidence-the-design-rests-on)
3. [Principles](#3-principles)
4. [Architecture overview](#4-architecture-overview)
5. [Public API](#5-public-api)
6. [Vector vocabulary and instruction sets](#6-vector-vocabulary-and-instruction-sets)
7. [Codelet generator](#7-codelet-generator)
8. [Passes, programs and strategies](#8-passes-programs-and-strategies)
9. [Real transforms, any length, many dimensions](#9-real-transforms-any-length-many-dimensions)
10. [Memory-system rules](#10-memory-system-rules)
11. [Planner](#11-planner)
12. [Fusion and pipelines](#12-fusion-and-pipelines)
13. [Threading and execution](#13-threading-and-execution)
14. [Build and packaging](#14-build-and-packaging)
15. [Correctness, accuracy and determinism](#15-correctness-accuracy-and-determinism)
16. [Benchmark rig: fbench](#16-benchmark-rig-fbench)
17. [Documentation](#17-documentation)
18. [Risks and open questions](#18-risks-and-open-questions)
19. [Decision log](#19-decision-log)
20. [References](#20-references)

---

## 1. Goals

### 1.1 What FFTN is

A C++20 library of fast Fourier transforms for CPUs, built for simulation codes that run
the same transforms millions of times: split-step and pseudo-spectral solvers, convolution
kernels, spectral analysis inside a time loop. Such codes need three things that existing
open-source libraries do not give together: top single-thread speed on every current
machine family, threading that cooperates with the application's own threads, and the
ability to fuse the pointwise work around a transform into the transform's own passes.

### 1.2 Goals, in priority order

1. **Single-thread speed.** As fast as or faster than every open-source library on the
   core targets: Apple M-series, x86-64 Linux (AVX2 and AVX-512, AMD and Intel), AArch64
   Linux. Specifically faster than FFTW and FFTL.jl everywhere, and as close as possible
   to Intel MKL on Intel processors, or ahead of it. **Older machines are targets too**:
   every x86-64 processor at the x86-64-v2 level or above (no AVX required) and every
   AArch64 processor (plain NEON) gets tuned, generated, tested kernels, not a fallback
   (section 6.1).
2. **Threading with no pathological behaviour.** Scales with cores, never slower than
   serial, never fights the application's threads. Works naturally with OpenMP, Kokkos and
   custom thread pools. Kokkos is optional; where an application uses it, FFTN feels
   native and performs at its best.
3. **Fusion.** Pointwise work before, between and after transforms (scaling, windows,
   spectral multipliers, zero padding and cropping, user kernels) runs while the data are
   in cache, inside or beside the transform's passes.
4. **Batches and layouts as first-class shapes.** Rows with strides and distances, 2-D
   planes, 3-D volumes, in place and out of place, along any axis.
5. **Reasonable general coverage.** What a typical FFT user expects, where it is cheap to
   provide: any length, real and complex, float and double, convenience functions with
   the usual normalisation choices, DCT/DST, a C interface. Not parity with FFTW for its
   own sake.
6. **Rigour.** Clean code under strict rules, complete tests with analytic oracles and an
   independent reference, a benchmark rig that makes A/B decisions quickly and tracks
   performance over time against every major competitor.
7. **Easy to adopt.** No dependencies. A small static library with light public headers. GCC 14 and newer, current Clang, Apple Clang; Windows through clang.

### 1.3 Performance targets

Ratios are time ÷ reference time on one thread unless stated; below 1 is faster. Each is
measured by fbench over the standard catalogues (section 16) as a geometric mean, with the
additional rule that no more than 5 % of cases may be slower than the reference by more
than 5 % beyond noise: a case counts when the whole bootstrap interval of its time ÷ the
reference's lies above 1.05 × (1 + the run's A/A noise band), fbench's own test for a
claimed effect (section 16) moved from 1 to 1.05. A run whose band is wide (a loaded
machine) is lenient by that rule, so a borderline result is repeated on a quieter run.

| Machine family | Reference | Target |
|---|---|---|
| Apple M-series (NEON + FCMA) | FFTW (NEON build, MEASURE) | ≤ 0.85 |
| Apple M-series | FFTL.jl | ≤ 0.95 |
| AMD Zen 3 (AVX2) | FFTW (AVX2, MEASURE) | ≤ 0.90 |
| AMD Zen 4/5 (AVX-512) | FFTW (AVX-512, MEASURE) | ≤ 0.85 |
| Intel Xeon (AVX-512) | FFTW (AVX-512, MEASURE) | ≤ 0.85 |
| Intel Xeon (AVX-512) | Intel MKL | ≤ 1.00 |
| AArch64 Linux (Neoverse) | FFTW (NEON), Arm Performance Libraries | ≤ 0.90, ≤ 1.00 |
| Older x86-64 (v2 level: SSE only; or AVX without AVX2) | FFTW built with its SSE2 and AVX codelets | ≤ 0.95, best effort: timed with the old targets forced on newer machines, since no old hardware is available; correctness and dispatch are verified by emulating the old processors |
| Threaded, few rows on many cores | FFTW with its best threading | ≤ 0.60 |
| Threaded, many rows | a team of the best serial plans | ≤ 1.02 (never meaningfully worse than the trivial way to thread) |
| Any client catalogue, every thread count the client uses | the client's current provider | no case above 1.02 |

The prototype reached 0.93–1.03 against FFTW on these machines with hand-written kernels
and no planner, and 0.31–0.66 threaded on few-row shapes, so the targets are ambitious but
start from measured ground. Phase exits in [PLAN.md](PLAN.md) may set interim values on
the way (for example 1.10 against MKL at the end of phase 1); the table is the end state.

### 1.4 Non-goals

- GPUs and distributed memory (MPI).
- Bitwise agreement with FFTW or between instruction sets.
- Extended or arbitrary precision.
- Completeness for its own sake: no feature is added without a motivating use.

---

## 2. Evidence the design rests on

The design follows from measurements, mostly from a three-day prototype (about 5000 lines,
four machines, roughly 130 commits each with a measured effect) and from source studies of
FFTW, FFTL.jl, KFR, zldsp::fft, ducc0 and RustFFT. The findings that shape the design:

| # | Finding | Consequence |
|---|---|---|
| E1 | A small code base can match FFTW per machine family; no single kernel family wins everywhere. Interleaved complex with complex-arithmetic instructions wins short rows (Apple FCMA, Zen 3 AVX2). Split real/imaginary arithmetic wins some wide-SIMD and long-row cases. | One generator and one pass framework, several data layouts, chosen per pass by the planner (sections 6–8). |
| E2 | FFTL.jl leads on Apple short rows by 10–27 % through cheaper passes, not different radices: rotations by ±i tracked symbolically and folded into the next add, cheap N/8 twiddles, a flat low-overhead stage loop. Its flop counts equal FFTW's. | The generator tracks pending rotations; the stage driver is a flat array of records (sections 7, 8). |
| E3 | FFTW leads on rows of 16–40 thousand points that sit inside L2 (1.2–2.4× over MKL and the prototype on a Xeon) by using two or three passes with radix 16–64 codelets and working in the output buffer. The prototype's own attempts at radix 20–32 lost everywhere, through register spills and breadth-first order. | Large codelets only as the generator can make them (ordered depth-first, spills counted), in a few-pass, low-scratch strategy (section 8.5); experiment X3 tests it. |
| E4 | FFTW's SIMD model keeps one complex number per lane pair. Each twiddle multiply costs about three shuffles on x86; on NEON double it is one complex per register with no FCMLA. FFTW-NEON is only 1.5× its own scalar code. Stock FFTW also ships non-FMA codelets. | FMA-native codelets and layouts without shuffles are a direct route past FFTW (sections 6, 7). |
| E5 | Most performance problems were memory-system problems, and machine dependent: power-of-two stride conflicts, L2 capacity, cache-line splits, prefetchers. Fixes that help one machine hurt another. | Cache geometry is detected and drives planning; memory rules are explicit (section 10). |
| E6 | Fixed rules reached parity but left per-length misses of 5–25 %; a global calibration of knobs found nothing above noise. The gains are per shape. | A cost model with optional per-shape measurement, not global knobs (section 11). |
| E7 | Threading is where the largest wins are. FFTW's fork/join per pass makes short rows 13–19× slower than serial. Teams must stay inside one cache group; rows below about 2000 points must never be split; waking idle threads costs 10–50 µs. | Threading is part of the core design: whole rows first, split only long rows, only inside a cache group, wake only threads with work (section 13). |
| E8 | A hidden library thread pool fighting the application's runtime is the recurring failure (FFTW threads, MKL layers, FFTL's baked pool, two OpenMP runtimes). | No hidden threads. FFTN runs on threads the caller provides (section 13). |
| E9 | In the client application, transforms are 5–53 % of a step, and for its largest cases more than half of the remaining time is transposing and pointwise passes around the transforms. A plain 3-D FFT there was no faster than planes plus rows. | Fusion and layout-aware multi-dimensional transforms are worth more than a faster bare FFT (sections 9, 12). |
| E10 | Process-to-process timing spread (median 5.6 %, worst 33 %) exceeds within-process spread. FFTW's aligned codelets are silently lost with `FFTW_UNALIGNED`; stale wisdom slowed FFTW by 1.3×. | The benchmark rig replicates over processes, controls competitors' builds and plans, and reports confidence intervals (section 16). |
| E11 | FFTW's column batches (stride ≠ 1) run 4–5× slower per point than row batches, and its strided AVX-512 access uses gather/scatter. | Strided axes are vectorised across neighbouring lines with no transposes (section 9.3). |
| E12 | Per-file `-march` flags leak instructions into shared inline code; every library that compiles per ISA with global flags has this exposure. | ISA code only in attribute regions; no global ISA flags (sections 6, 14). |

---

## 3. Principles

1. **Plans describe shapes, not pointers.** A plan is immutable, holds no user pointers,
   accepts any alignment, and can be executed concurrently on different data. Scratch
   belongs to an execution context.
2. **Everything is decided at plan construction.** Instruction set, layout, factorisation,
   blocking and the division of work are fixed in the plan. Executing it does not branch on
   the machine, allocate, lock or throw.
3. **Generated, verified kernels.** Every codelet comes from one generator, is checked
   against an exact DFT before it is emitted, and is written against a small vector
   vocabulary that each instruction set implements with intrinsics.
4. **One pass framework.** A transform is a short program of passes. Batches, strided
   axes, blocking, real transforms and fusion are variations of the same passes, not
   separate engines.
5. **Memory first.** Passes over memory are counted and designed; cache sizes and
   associativity are inputs to the planner.
6. **Threads belong to the caller.** FFTN defines the work; the caller's runtime provides
   the threads. Results do not depend on how work is shared out.
7. **Measure, then decide.** Rules are defaults. Every rule has an experiment behind it,
   and the planner can measure where rules are known to miss.
8. **Fusion is a feature, not a special case.** The framework has places for pointwise
   work at the input edge, the output edge and between a transform pair.
9. **Small and deletable.** A path that loses is removed. Complexity must pay for itself
   in measured speed or in a client need.

---

## 4. Architecture overview

```
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ Convenience layer   fft/ifft/rfft/… over spans   kokkos.hpp   C API   FFTW shim │
 ├────────────────────────────────────────────────────────────────────────────────┤
 │ Core API            shape descriptors · plan<…> · workspace · pipelines        │
 ├───────────────────────────────┬────────────────────────────────────────────────┤
 │ Planner                       │ Execution                                      │
 │  candidates → cost model      │  work graph from the program's loops           │
 │  → optional measurement       │  engine (claiming, cache-group aware)          │
 │  → wisdom                     │  adapters: serial, OpenMP, Kokkos, pool, custom│
 ├───────────────────────────────┴────────────────────────────────────────────────┤
 │ Programs            flat arrays of stage records and loop records              │
 │ Strategies          chain · grid · lanes · bluestein  (program builders)       │
 ├────────────────────────────────────────────────────────────────────────────────┤
 │ Pass kernels        lane pass · grouped pass · leaf · edge variants            │
 │ Codelets            generated; one text, included per instruction set          │
 ├────────────────────────────────────────────────────────────────────────────────┤
 │ Vector vocabulary   scalar · x86-v2 · AVX · AVX2 · AVX-512 · NEON · FCMA · (SVE)│
 │ Platform            CPU features · cache topology · aligned memory · timers    │
 └────────────────────────────────────────────────────────────────────────────────┘
```

Repository layout (target; created as phases need it):

```
include/fftn/            public headers only: fftn.hpp, plan.hpp, execution.hpp,
                         pipeline.hpp, convenience.hpp, omp.hpp, kokkos.hpp, c_api.h
src/                     the library: platform/, planner/, engine/, strategies/, c_api.cpp
src/isa/                 regions, vocabularies, the kernel units of each instruction set
src/passes/              pass kernels (.inl, included per instruction set)
src/gen/                 generated codelets and their metadata tables (committed)
gen/fftn_gen/            the codelet generator (Python), with gen/README.md
docs/                    user documentation (Markdown)
examples/                compiled, tested examples that the documentation quotes
tests/                   unit, analytic, reference, contract, threading, property tests
bench/                   fbench: driver, backends (one binary per library), catalogues
tools/                   repository tooling (lint, coverage gate, capture, calibration)
scripts/                 the gate, remote runners
cmake/                   presets and helper modules
dev/                     DESIGN.md, PLAN.md, KNOBS.toml, experiments/
```

---

## 5. Public API

The API has three levels. The core level is pointers plus geometry and is what everything
else is built on. The public headers are its reference (signatures, preconditions,
guarantees) and `docs/conventions.md` states what it computes and how it lays out memory;
this section gives the shape of the API and the reasons for it. The sketches below are the
headers abbreviated; members marked *later* are the intent of later phases and do not exist
yet. Until the first tagged release the API is not frozen: a decision that proves wrong is
changed, with a line in the decision log, rather than kept for compatibility.

Headers: `fftn.hpp` includes the whole core API, one concept per header: `shape.hpp`
(`direction`, `axis`, `batch`, `shape` and the shape functions), `options.hpp`, `isa.hpp`,
`workspace.hpp`, `plan.hpp`, `wisdom.hpp`. The typed `plan<In, Out>` is a thin wrapper over a type-erased
core declared in `detail/core.hpp` and defined in `src/core/` (section 14.2): it passes its
shape across as a `shape_view` of plain structs (the kind and precision as values, the axes
and batch dimensions as spans borrowed for the call), so no `std::vector` crosses into
the library and nothing on the execute path sees one.

### 5.1 Conventions

- FFTW's conventions: forward is exponent −i, backward +i; unnormalised by default;
  r2c writes n/2 + 1 complex values per row; c2r reads n/2 + 1 and ignores the imaginary
  parts of the DC bin and, for even n, the Nyquist bin.
- **Input is never destroyed** unless the caller asks for in-place operation
  (`options::in_place`) or, later, grants permission (`options::may_destroy_input`), in
  which case the planner may use the input as scratch. Such plans will execute through
  overloads that take a non-const input pointer, so the permission is visible in the call.
  In-place plans use the ordinary `execute(const In*, Out*, workspace&)`: the caller passes
  one address twice (for real kinds, the same memory viewed as both types).
- Complex data are `std::complex<T>` (interleaved). Split real/imaginary arrays are a
  later layout option (section 5.3). Both `double` and `float` are supported. `double` is
  tuned first. `float` is instantiated and tested for correctness in CI from phase 1 (on
  the scalar target and at least one SIMD target per architecture) so the precision-generic
  code cannot drift; its remaining targets and its tuning follow in phase 6.
- Any alignment of at least `alignof(T)` is accepted. Kernels use unaligned loads
  everywhere and there are no aligned-only code paths, so alignment is not part of a plan
  and never changes results. Speed can still depend on it: a 64-byte access to data that
  is not 64-byte aligned straddles cache lines. Whether the edge passes should choose a
  peeled variant from the base address at execute time is part of experiment X4.

### 5.2 Shapes

A shape is a kind, the transformed axes and the batch dimensions, in the manner of FFTW's
guru interface but typed and with defaults.

```cpp
namespace fftn {

enum class direction : std::int8_t { forward = -1, backward = +1 };   // the exponent's sign

struct axis {                 // one transformed dimension
  std::size_t n = 0;          // logical length; must be given
  std::ptrdiff_t in_stride = 0, out_stride = 0;   // in elements; 0 = tight C-ordered default
};
struct batch {                // one loop dimension (not transformed)
  std::size_t count = 0;      // must be given
  std::ptrdiff_t in_dist = 0, out_dist = 0;       // in elements; 0 = tight C-ordered default
};

template <class In, class Out>
struct shape {                // kind given by In/Out: complex→complex, real→complex, …
  direction dir = direction::forward;             // c2c only
  std::vector<axis> axes;                         // one to three; last halved for real kinds
  std::vector<batch> batches;                     // any number; none = one transform
};

// The common cases, as functions returning concrete shapes:
template <class T> shape<std::complex<T>, std::complex<T>>
    rows_c2c(std::size_t n, std::size_t count, direction);
template <class T> shape<T, std::complex<T>> rows_r2c(std::size_t n, std::size_t count);
template <class T> shape<std::complex<T>, T> rows_c2r(std::size_t n, std::size_t count);
template <class T> shape<std::complex<T>, std::complex<T>>
    planes_c2c(std::size_t ny, std::size_t nx, std::size_t count, direction);
template <class T> shape<T, std::complex<T>>
    planes_r2c(std::size_t ny, std::size_t nx, std::size_t count);
template <class T> shape<std::complex<T>, T>
    planes_c2r(std::size_t ny, std::size_t nx, std::size_t count);
template <class T> shape<std::complex<T>, std::complex<T>>
    volumes_c2c(std::size_t nz, std::size_t ny, std::size_t nx, std::size_t count, direction);
template <class T> shape<T, std::complex<T>>
    volumes_r2c(std::size_t nz, std::size_t ny, std::size_t nx, std::size_t count);
template <class T> shape<std::complex<T>, T>
    volumes_c2r(std::size_t nz, std::size_t ny, std::size_t nx, std::size_t count);
// Chosen dimensions of a strided array (provisional, task 4.3): one `axis` per dimension of
// the array, the dimensions to transform by index, the others becoming batch dimensions.
template <class T> shape<std::complex<T>, std::complex<T>>
    axes_c2c(std::span<const axis> array, std::span<const std::size_t> axes, direction);
template <class T> shape<T, std::complex<T>>
    axes_r2c(std::span<const axis> array, std::span<const std::size_t> axes, bool in_place = false);
template <class T> shape<std::complex<T>, T>
    axes_c2r(std::span<const axis> array, std::span<const std::size_t> axes, bool in_place = false);
}
```

Points that matter:

- **Order and numbering.** Axes and batch dimensions are both listed slowest first, as the
  dimensions of a C array. Transforms are numbered with the last batch dimension fastest;
  `plan::execute_transforms` uses these numbers. A transform is a row, a plane or a volume.
- **Every kind at every rank.** The general `shape` is accepted for one to three axes and
  all three kinds, not only through the shape functions: the same code serves them all
  (axis by axis, a transform at a time, section 9.3). A c2r of two or more axes out of place
  keeps its intermediate in a workspace block, so its input is preserved (9.1). More than
  three transformed axes is a later tier (5.6).
- **Batch dimensions** are as many as the caller needs: a `Kokkos::View` of any rank, or a
  sub-array whose loops do not nest, maps onto one shape without a copy. Each becomes one
  loop of the program (8.3).
- **Chosen axes of an array** (`axes_c2c`, `axes_r2c`, `axes_c2r`; provisional until the
  maintainer has reviewed them). The general `shape` already describes any selection and order
  of the axes of a strided array, with the rest as batch dimensions; these functions write it
  from the array's description, working out the array's tight defaults and sorting the batch
  dimensions by input stride, widest first, merged where they nest (`fftn/kokkos.hpp`'s
  `shape_of` goes through the same code). For real kinds the last axis listed is halved.
- **Containers.** `axes` and `batches` are `std::vector`s: plan construction allocates
  anyway, so an inline container would buy nothing and would freeze the rank limits into
  the type. The limits are checked at construction like every other misuse. `axis::n` and
  `batch::count` default to 0, which construction refuses, because the direct-list form
  `.batches{8}` calls the vector's size constructor (eight empty batch dimensions) and must
  not pass as a shape the caller did not mean.
- **Directions.** `dir` is ignored by the real kinds (r2c is always forward, c2r always
  backward; `rows_c2r` sets it to backward for readability). A mismatch is not an error.
  `direction`'s underlying type is the smallest that holds its values.
- **Defaults and validation.** Zero strides and distances take the tight C-ordered
  defaults, each zero independently, so the common cases need no arithmetic
  (docs/conventions.md has the rules, including FFTW's padded rows for in-place real
  transforms). Construction refuses, with `std::invalid_argument` naming the argument: zero
  lengths and counts; no axis or more than three; output elements that may overlap; sizes
  whose byte reach does not fit `std::ptrdiff_t` and strides of `PTRDIFF_MIN`; in-place
  layouts other than identical ones (c2c) and FFTW's padded layout (real kinds). The
  overlap test is a sufficient condition (sorted by stride, each stride exceeds the reach of
  the smaller ones): it admits every nested layout and refuses a few exotic interleavings
  that do not actually collide. The strides of an axis of length 1 and the distances of a
  batch dimension of count 1 are ignored, since they never step: refusing them would reject
  harmless layouts, and using them would let unchecked values into the kernels' arithmetic.

### 5.3 Plans and execution

```cpp
namespace fftn {

enum class planning : unsigned char { rules, quick, measure };   // section 11.2

struct options {
  bool in_place = false;               // in == out at execute
  double scale = 1.0;                  // fused into an edge pass; 1/n for normalised use
  isa force_isa = isa::best;           // tests and benchmarks only
  planning effort = planning::rules;   // section 11.2
  double measure_seconds = 0;          // measured planning's time budget; 0: the effort's
  std::size_t measure_scratch_bytes = 0;  // its scratch limit; 0: 1 GiB
  const wisdom* use = nullptr;         // recorded programs to rebuild (section 11.3)
  // later, with their phases:
  // int max_lanes = 1;                   width hint: the most threads this plan will see
  // bool strict_layout = false;          ignore the width hint when choosing the program
  // bool may_destroy_input = false;
  // edge_ops in_ops, out_ops;            fused pointwise work (section 12)
  // table_cache* tables = nullptr;       share twiddle tables between plans (optional)
};

class wisdom {                         // caller-owned records of plans' programs (11.3)
public:
  wisdom(); explicit wisdom(std::string_view text);
  static wisdom load(const std::filesystem::path&); void save(const std::filesystem::path&) const;
  std::string text() const;
  template <class In, class Out> void add(const plan<In, Out>&); void add(const wisdom&);
  std::size_t size() const noexcept;
};

enum class isa : unsigned char { scalar, x86_v2, avx, avx2, avx512, neon, neon_fcma };
const char* isa_name(isa) noexcept;

class workspace {                      // scratch for one execution at a time; move-only
public:
  workspace() noexcept;                // empty: enough for plans that need 0 bytes
  explicit workspace(std::size_t bytes);
  template <class In, class Out> explicit workspace(const plan<In, Out>&, int lanes = 1);
  explicit workspace(std::span<std::byte> memory) noexcept;   // borrows caller memory
  std::size_t size() const noexcept;
  std::span<std::byte> memory() const noexcept;
};

struct team { int rank = 0; int size = 1; int group_size = 0; };   // one member of a team
enum class completion : unsigned char { all, own };
struct transform_range { std::size_t first = 0; std::size_t count = 0; };
int cache_group_size(int members) noexcept;   // a team's group_size, from the topology

using task_function = void (*)(void* arg, int index);   // = fftn_task of fftn/executor.h
template <class E> concept executor = requires(E& e, int n, task_function f, void* a) {
  { e.concurrency() } -> std::convertible_to<int>;    // optional: e.group_size()
  e.bulk(n, f, a);
};
class executor_ref {                   // non-owning; three function pointers (fftn_executor)
public:
  template <executor E> executor_ref(E&&) noexcept;         // implicit
  explicit executor_ref(const fftn_executor&) noexcept;     // a C pool
  int concurrency() const; int group_size() const;
  void bulk(int n, task_function, void* arg) const;
};
class thread_pool {                    // opt-in; std::threads; an executor
public:
  explicit thread_pool(int threads = 0);
  int concurrency() const noexcept; int group_size() const noexcept;
  void bulk(int n, task_function, void* arg) const noexcept;
};

template <class In, class Out>
class plan {
public:
  explicit plan(const shape<In, Out>&, const options& = {});   // copyable, no default

  // (1) Serial, on the calling thread.
  void execute(const In* in, Out* out, workspace&) const noexcept;

  // (2) A sub-range of the batch's transforms, serial: for callers that own the loop.
  void execute_transforms(std::size_t first, std::size_t count, const In* in, Out* out,
                          workspace&) const noexcept;

  // (3) Inside the caller's parallel region: every member of the team calls it (13.3).
  void execute(const In* in, Out* out, const team&, workspace&,
               completion = completion::all) const noexcept;
    transform_range transforms_of(const team&) const noexcept;   // a member's own (12.4)

  // (4) On an executor, anything that can run n tasks and return when done (13.6).
  void execute(const In* in, Out* out, executor_ref, workspace&) const;

  // later: a trailing exec_args (arrays for the edge operations, section 12) on every
  // overload, and twins taking In* for plans that may destroy their input.

  std::size_t workspace_bytes(int lanes = 1) const noexcept;
  std::string describe() const;        // the program, human readable: strategy, radices,
                                       // layouts, instruction set, scratch, threading shape
  isa instruction_set() const noexcept;
};

template <class T> using c2c_plan = plan<std::complex<T>, std::complex<T>>;
template <class T> using r2c_plan = plan<T, std::complex<T>>;
template <class T> using c2r_plan = plan<std::complex<T>, T>;
}
```

Points that matter:

- **Kind safety is static.** The element types select the kind, so calling a real plan
  with complex data does not compile. Nothing is checked at execute time. Kinds that the
  element types cannot tell apart (the real-to-real family of phase 6, split-complex
  arrays) will be selected by a kind tag in the shape, defaulted from the types.
- **Plans are values.** Copies share one immutable core (`std::shared_ptr<const>`), so
  copying is cheap and thread-safe. There is no default constructor and every plan is
  valid: the class declares its copy operations and no move operations, so moving a plan
  copies it (for the price of a reference-count increment) and leaves the source usable.
  The alternative, a moved-from state in which only assignment and destruction are
  allowed, would bring back an invalid plan that every member function had to rule out.
- **`workspace`** holds scratch and synchronisation state for one concurrent execution. It
  is sized from a plan and a lane count, owns its memory or borrows the caller's, and
  reports its size, so an application can account for every byte. Every core `execute`
  takes one; none allocates. `workspace_bytes()` includes 64 bytes of slack, so memory of
  any alignment works. Callers who do not want to manage workspaces will use the
  convenience layer (later, 5.4), which caches them.
- **Teams.** A team's members share one workspace of `workspace_bytes(lanes)`: a serial
  scratch slot per member and the team's synchronisation state (13.3), 128 bytes of slack
  for its alignment. The state persists between team executions; the workspace object
  holds one word besides its memory saying whether the memory holds valid state, so the
  first member of a first execution sets it up and the memory never needs initialising.
  Serial executions clear that word, and so does an execution with `completion::own` by a
  smaller team, whose scratch overlaps the larger team's state. The members of a
  workspace's executions follow one sequence of executions: a member may arrive late,
  after the others have moved on, but may not skip one (a serial execution, even of no
  transforms, is the reset); all pass the same workspace object; one rank's successive
  calls are ordered by happens-before; serial use and a change of team size wait until
  every member of the earlier executions has returned. A team is plain numbers (rank, size, members per cache
  group), so any threading system can form one; `cache_group_size(members)` estimates the
  grouping from the topology (13.6), and `fftn::omp::this_team()` fills a team in.
- **Executors.** Overload (4) takes an `executor_ref`, a non-owning reference to anything
  with `concurrency()` and `bulk(n, task, arg)`, which reaches the compiled library as three
  function pointers (the C contract's struct, 5.5), so it allocates nothing and needs no
  template in the library. FFTN chooses the team (13.6) and dispatches one task per lane
  that has work; every dispatched task runs exactly once, so the team-sequence
    precondition holds by construction, whatever order or concurrency the executor gives the
  tasks. The contract requires that the call of `bulk` happens before each task and each
  task's return before `bulk` returns, which orders one rank's successive calls, and that a
  throwing `bulk` throws only after every task it started has returned. It is not `noexcept`: FFTN's own code throws nothing, allocates nothing and takes
  no lock, but the executor's calls may, and an exception from them propagates after the
  workspace's team state is dropped (some members may not have run). The workspace is sized
  for `concurrency()` lanes.
- **Execution preconditions** (a large enough workspace, `execute_transforms` within the
  batch, one address for in-place calls) are documented and checked only by assertions in
  debug builds of the library; the serial and team overloads stay `noexcept` and check
  nothing in release builds.
- **Plan construction is cheap and safe**: microseconds to a few milliseconds with rules,
  bounded and optional with measurement; it never touches user buffers and uses no
  process-global mutable state. A plan owns its twiddle tables. Plans that should share
  tables will be given the same `table_cache` object, which the caller owns; there is no
  hidden global cache, and all memory is released when the plans and the cache are
  destroyed.
- **`isa`** lists the seven targets of section 6.1; `options::force_isa` and `isa::best`
  arrive with the second target that has the passes. Phase 0 plans report `isa::scalar`.
- **`describe()`** makes the planner's choice visible, including any slow route (for
  example Bluestein for a large prime). Nothing is silent. Its text is for people; its
  format is not stable.

### 5.4 Convenience layer

*Later* (phase 6, 5.6). A thin header for users who want the numpy-shaped interface: `fft`, `ifft`, `rfft`,
`irfft`, `fft2`, `fftn`, `rfftn`, `irfftn` over `std::span` and a small strided array view
(C++20 has no `mdspan`; the view converts from one where available), with `axes`,
`norm = backward | ortho | forward`, a cache of plans and per-thread workspaces keyed by
shape (the one place FFTN allocates behind the caller's back, and documented), and the helpers
`fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` (`next_fast_length` is already in the core API,
`fftn/lengths.hpp`, task 2.5). All of it is expressed through the core API; none of it adds
kernels.

### 5.5 Other interfaces

**`fftn/kokkos.hpp`** (optional, header-only, compiled in the user's translation units
against the user's Kokkos; `docs/kokkos.md` lists every difference from Kokkos-FFT). It
has three levels. **The threaded parts (`executor`, `team_of`, the threaded `view_plan`, task
3.5) are provisional**: written without the maintainer's review of their shape, flagged in
the header and the docs until it happens.

- *View-based construction returns ordinary core types:* `shape_of(in, out, dir, axes)` an
  `fftn::shape`, `make_plan` an `fftn::plan` (in place when the Views share their first
  element), `data_of(view)` the pointer. Views are read through their extents and strides,
  so `LayoutStride` works too; broadcast (zero) strides are refused, except on dimensions of
  extent 1, which are dropped; the other non-transformed dimensions become batch
  dimensions, listed largest input stride first (which fixes their transforms' numbering);
  those that nest are merged, and any number may remain, so no View needs the temporary
  transpose Kokkos-FFT makes. In-place real transforms take FFTW's padding from the real
  View's strides, which must show it. `n` and `s` are accepted only as the Views' own
  lengths: FFTN does not crop or pad. Reading a `Kokkos::complex<T>` array through a
  `std::complex<T>*` is not a language guarantee (the types are not similar); it relies on
  their identical layout, which `data_of` checks, and on the library touching the data only
  through `void*` and `T` scalars in its own translation units, as Kokkos-FFT does with
  FFTW's complex type.
- *`view_plan<ExecutionSpace, InView, OutView, Dim = 1>`* has the constructors (`axis` with
  optional `n`; `axis_type<Dim>` with optional lengths `s`) and `execute(in, out, norm)` of
  Kokkos-FFT 2.0's `Plan`, plus the free `execute(plan, in, out, norm)`, `fftn_plan()` and
  `describe()`; Kokkos-FFT's public `execute_impl` is not provided (call sites use
  `execute`). It is not named `plan` because `fftn::plan` and `fftn::kokkos::plan` would
  be ambiguous under using-directives for both namespaces. Its constructors are `explicit`,
  and it is neither copyable nor movable, as Kokkos-FFT's. It owns its workspace, made at
  construction for the width of the plan's instance, so one plan runs one execution at a
  time: an atomic busy flag makes a second concurrent execution throw `std::logic_error`
  rather than compute wrong results; concurrent executions use `make_plan` and a workspace
  each. `execute` checks the Views against the plan's extents, strides and placement and
  throws `std::invalid_argument` naming `in` or `out`; it allocates nothing itself (the
  labels Kokkos takes as `std::string` are built at construction). It fences the instance
  and then executes through the plan's `executor` (13.6), inside a Kokkos Tools region. A plan
  on `Kokkos::Serial` has neither fence nor region: its kernels finish before their dispatch
  returns, and it cannot tell whether the caller is inside one, where a fence, or the global
  fence Kokkos makes before a region callback, would wait for the caller's own kernel. Inside a
  kernel of OpenMP or Threads (detected whichever space the plan runs on) it neither fences,
  dispatches nor opens a region, and runs serially. Exceptions from `execute` end the program
  there. A plan on a threaded back end must not be called from a Serial kernel. Normalisation follows Kokkos-FFT
  (`backward` by default) as a separate pass after the transform, each part multiplied by the
  coefficient computed in `double` and rounded once to `T` (Kokkos-FFT multiplies real
  `float` outputs in `double`), shared among the instance's threads from 2^16 output
  elements, cut along the output's longest dimension (T3); the core stays unnormalised, and
  `options::scale` can later fuse it into the last pass. The direction of real plans is
  checked as in Kokkos-FFT (r2c forward, c2r backward).
- *`executor<ExecutionSpace>` and `team_of(member)`* put core plans on Kokkos' threads (13.6):
  the executor overload of `plan::execute` from outside a kernel, the team overload from the
  members of a `TeamPolicy` kernel. Either is serial inside a kernel, the executor by
  reporting one lane.
- *`fftn::kokkos::compat`* spells all of this as Kokkos-FFT does (`Plan`, `Direction`,
  `Normalization`, `axis_type`, `shape_type`, `execute`), so a call site changes its include
  and namespace only. It is a deliberate exception to the naming rule of AGENTS.md section
  4, confined to that namespace, and nothing in it differs from Kokkos-FFT on purpose:
  `Direction` and `Normalization` are Kokkos-FFT's enumerations (underlying `int`;
  `Direction` 0 and 1), converted to `fftn::direction` and the adapter's `normalization`
  inside the plan, and `shape_type`, which means only Kokkos-FFT's crop-and-pad lengths, exists
  only there. Keeping the adapter a faithful mirror keeps open an FFTN host back end inside
  Kokkos-FFT later. CTAD through the `Plan` alias needs Clang 19 or GCC; the class itself
  deduces everywhere.
- Host accessibility is a property of the types and is checked at compile time
  (`static_assert`), so a device View does not compile against the adapter, and plans and
  executors accept only `Kokkos::Serial`, `Kokkos::OpenMP` and `Kokkos::Threads` (HPX would pass
  the host checks and then fence inside its own kernels; Threads needs Kokkos 5.2.1, 13.6). An
  application that builds for devices selects the provider per build. Errors are `std::invalid_argument` with the prefix
  `fftn::kokkos:` and the argument's name (Kokkos-FFT throws `std::runtime_error`).
- *Not provided yet:* the one-shot functions (`KokkosFFT::fft` and the rest, which would make
  a plan per call and belong with the convenience layer of 5.4); `DynPlan`.

**`fftn/omp.hpp`** (optional, header-only, compiled in the user's translation units against
the user's OpenMP runtime, so the process has one OpenMP runtime; the library links none):
`fftn::omp::this_team()` for callers inside a parallel region and `fftn::omp::executor` for
FFTN to open one (13.6). It needs the compiler's OpenMP option and refuses to compile
without it. Neither changes a runtime setting.

**The executor contract in C** (`fftn/executor.h`, valid C99 and C++, types only): a struct
`fftn_executor` of the pool's pointer and three functions, `concurrency(self)`,
`bulk(self, n, task, arg)` and `group_size(self)` (which may be null), with
`fftn_task = void (*)(void* arg, int index)`. The C++ `executor_ref` holds exactly this
struct, filled with thunks for a C++ executor or copied from a C pool's, so the C API of
phase 6 will include the header and take the struct unchanged.

The other interfaces are later work:

- **C API** (`fftn/c_api.h`): opaque plans, the same shapes, and the three-function
  executor contract for foreign thread pools, which exists already (above).
- **FFTW3-compatible shim** (separate opt-in library and header, never installed as
  `fftw3.h`): the `fftw_plan_many_dft`, guru and execute subset, so C and Fortran users
  and existing benchmark tools can link FFTN. It is opt-in because a header that
  impersonates FFTW breaks applications that probe for FFTW features.

### 5.6 Coverage tiers

| Tier | Contents | Phase |
|---|---|---|
| Core | c2c, r2c, c2r; 1-D rows, 2-D planes, 3-D volumes; strides and distances; in place and out of place; any length; double; serial, team and executor execution; edge ops; convolution and transform-pair pipelines | 1–5 |
| General | float on every target, tuned where asked for; convenience layer; DCT/DST types I–IV and the discrete Hartley transform as composed plans; split-complex layout; C API; wisdom import and export | 6 |
| Later, on demand | FFTW shim; more than three transformed axes; SVE; Python binding | 7 |

---

## 6. Vector vocabulary and instruction sets

### 6.1 Targets

| Target | Register | Complex per register (interleaved) | Registers | Notes |
|---|---|---|---|---|
| `scalar` | 1 double | — | — | Portable fallback; reference for generator tests |
| `x86_v2` | 128 bit, SSE2–SSE4.2, no FMA | 1 | 16 | Every x86-64 CPU since about 2009 (Nehalem, Westmere, Atom-class, older AMD). First class |
| `avx` | 256 bit, no FMA, no AVX2 | 2 | 16 | Sandy Bridge, Ivy Bridge, AMD Bulldozer family (2011–2014). `avx2`'s shape without FMA; kept by experiment X14 (0.68–0.73 of `x86_v2`'s time, timed on Zen 3 and Cascade Lake) |
| `avx2` | 256 bit + FMA | 2 | 16 | x86-64-v3: Zen 1–3, Intel since Haswell (2013) |
| `avx512` | 512 bit (F, DQ, VL) | 4 | 32 | x86-64-v4: Zen 4/5, Intel server since Skylake-SP |
| `neon` | 128 bit | 1 | 32 | AArch64 baseline (Neoverse N1, Graviton 2, Ampere) |
| `neon_fcma` | 128 bit + FCMLA/FCADD | 1 | 32 | Apple M-series, Neoverse V1/N2 and newer |
| `sve` | scalable | — | 32 | Experiment only (phase 7); Graviton 3/4 are 256/128 bit |

**Coverage by age.** The lowest x86 target is the x86-64-v2 feature level, which is the
oldest level current Linux distributions and Python wheels still build for; processors
below it (before about 2009) run the `scalar` target correctly but untuned. Because no
translation unit carries an ISA flag (section 6.4), one binary built at any baseline
(v2, v3 or anything else) contains every target and picks the best one the processor
has: a v2-baseline build still uses AVX-512 on a new machine, and a v3-baseline build's
FFTN kernels are not the limit on an old one. Targets without FMA implement the
vocabulary's `fmadd` as multiply then add, and the generator's FMA shaping is keyed on
the target's capabilities in the codelet metadata, so those targets get the non-FMA
operation order rather than a de-fused FMA order. `x86_v2` has the same shape as plain
`neon` (one complex per register, no complex-arithmetic instructions beyond SSE3's
`addsub`), so the two share planner rules as a starting point.

The FMA x86 targets (`avx2`, `avx512`) share one set of 128-bit leaves (n ≤ 16 and the
primes 17, 19, 23): the leaves compiled in the `avx2` region with `avx2`'s operations on 128-bit registers
(`src/isa/vocab_avx2_narrow.inl`, experiments X28 and X28b). A leaf broadcasts its one line
into every lane, so wider registers only repeat the work, and on Cascade Lake a 512-bit
leaf also lowers the clock; the half-width operations round as `avx2`'s and `avx512`'s do,
so the bits are unchanged. Every leaf has a second instantiation for lines of stride 1 on
both sides, whose leg offsets are constants (`src/passes/unit_leaf.inl`, X28b); `avx2` and
`avx512` keep only the narrow one. Short composite rows on those two and on NEON are pair
leaves (section 8.2): a table per target and precision (`src/isa/pairs_<target>.cpp`) lists
the lengths and their factors r1 × m. On x86 in double the codelet lengths 8, 12 and 16 take
a pair leaf before the narrow leaf (whole registers; X37), and `avx512` plans take `avx2`'s
pair leaves for lengths their own table lacks (C = 4 does not divide them, or r1 < C would
store partial runs: 8, 12, 18, 22, 26, 30); each pair set carries its lane count and
twiddle packing, so such a leaf's table is built in its own storage. On AArch64 the
leaves' batch loops step by the batch distance in bytes (no division per call). `x86_v2`
and `avx`, which have no FMA and so cannot run `avx2` code, use their own leaves, as the
scalar target does, and no pair leaves.

The target is detected once per process (CPUID, `getauxval`/`sysctlbyname`, Windows
`IsProcessorFeaturePresent`) and chosen once per plan. A plan may be forced to a lower
target for tests and benchmarks; that never affects other plans or other libraries.

### 6.2 The vocabulary

Kernels and generated codelets are written against a small vocabulary of vector types
and functions, never against intrinsics. Each target defines the vocabulary, under the
same names, in its own namespace inside its attribute region; kernel text is written once
and included once per region (section 6.4). The vocabulary has two families of
operations, all force-inlined. Below, `cv`, `rv`, `C` and `W` are those per-target names.

**Complex-register operations** (interleaved layout: a `cv` holds `C` complex
numbers as re, im pairs):

| Operation | Meaning | NEON+FCMA | NEON | AVX2 / AVX-512 |
|---|---|---|---|---|
| `add(a, b)`, `sub(a, b)` | a ± b | 1 instruction | 1 | 1 |
| `addj(a, b)`, `subj(a, b)` | a ± i·b | `FCADD` (1) | swap + FMA by (∓1, ±1) (2) | AVX2: swap + `addsub`, or FMA by (1, −1) for `subj` (2); AVX-512, which has no 512-bit `addsub`: swap + `fmaddsub` or `fmsubadd` by ones (2) |
| `mul(a, w)`, `mulc(a, w)` | a·w, a·conj(w) | 2 × `FCMLA` | 1 shuffle + multiply + FMA with pre-arranged twiddles | 1 shuffle + multiply + FMA with pre-arranged twiddles |
| `madd(acc, a, w)` | acc + a·w | 2 × `FCMLA` onto acc | shuffle + 2 FMA onto acc | shuffle + 2 FMA onto acc |
| `scale(a, r)`, `fmadd_r(acc, a, r)` | real constant times a (plus acc) | 1 | 1 | 1 |
| `neg`, `conj`, `swap_ri` | sign and part manipulation | 1 | 1 | 1 |
| `cmul(a, b)` | a·b for two complex registers (factor arrays) | as `mul` | twiddle form built in registers, then `mul` | twiddle form built in registers, then `mul` |
| `load`, `store`, `bcast`, `load_tw`, `bcast_tw` | memory (`bcast_tw`: one stored twiddle in every lane) | | | |
| `transpose` | in-register C × C transpose of complex values | zips | zips | unpack / permute: AVX-512 double a 4 × 4 block of 128-bit units (8 `vshuff64x2`), float an 8 × 8 block of 64-bit units (8 unpacks, then two 4 × 4 transposes of 128-bit units: 16 `vshuff64x2`) |

**Real-register operations** (split layout: an `rv` holds `W` reals; a complex vector
is a pair of them): `add`, `sub`, `mul`, `fmadd`, `fmsub`, `fnmadd`, `neg`, `load`,
`store`, `bcast`, and `deinterleave` / `interleave` to convert from and to interleaved
memory (`LD2`/`ST2` on NEON, unpack pairs on x86).

Each target implements the vocabulary in one header of a few hundred lines. Adding a
target is adding that header; adding a radix never touches a target.

The names, generic over the real type `T` (`double` and `float`): `cv<T>` and `rv<T>` are
the register types, `C<T>` and `W<T>` their lane counts, `tw<T>` the twiddle form `mul`,
`mulc` and `madd` consume, `tw_reals<T>` the reals stored per twiddle (2, or 4 where
pre-arranged), and `fused` says whether `fmadd` and its relatives round once. They are
reached through specialisations of a per-target `vec<T>`, so a vector type is never a
template argument. The real-register family lives in a nested namespace `real`, because
`cv<T>` and `rv<T>` are one machine type on SIMD targets while `mul` and `bcast` mean
different things in the two families. The FMA names follow x86: `fmadd(a, b, c)` is
a·b + c, `fmsub` a·b − c, `fnmadd` c − a·b. `transpose` is a C × C block. An R × C form
for radices below C (avx512: R = 2 for `double`, R = 2 and 4 for `float`) is not built: the
grouped pass stores such radices, and every R that C does not divide, through its tile
(section 8.2). Planner v0's rules (FFTL.jl's) did not take C into account, and on
`avx512` most lengths started with a radix below 8 (32 as 4 · 8, 1024 as 4 · 4 · 8 · 8; of
the lengths 17 to 4096 that take more than one pass, 644 of 3522 started with 8 or more).
The grouped pass then stores through the tile and the next pass, with s = 4 < 8 copies in
`float`, runs entirely as tail units, element by element: correct (the tests cover it) but
slow. Planner v1 (section 11.2) counts those tails with C = 8 and so prefers a leading
radix that C divides, but its costs on `avx512` are the AVX2 table's, unmeasured there.
Whether `avx512` needs its own table or an R × C transpose is for an experiment comparing
`avx512` with `avx2` (X16, planned). The scalar target is never
fused, so it gives the same bits on every processor. The full contract is
`src/isa/vocabulary.inl`.

Each SIMD target rounds every operation exactly as another of its architecture does, so
the same passes give the same bits on both targets of a pair: plain `neon` takes FCMLA's
order in its products (the product with wr rounded, the wi term fused onto it), `avx512`
takes `avx2`'s FMA order, `x86_v2` and `avx` round each product and then sum them, in the
scalar target's pairing (so all three agree operation by operation), and `addj` and `subj`
round one addition everywhere
(`tests/core/passes_test.cpp` holds whole transforms to it). Forming a ± i·b on AVX-512 as
`fmaddsub` or `fmsubadd` by a vector of ones, which rounds once because the product is
exact, is an idiom FFTW's AVX-512 SIMD header uses too (the idea only; FFTN's code is its
own, AGENTS.md section 10). `neon` and `neon_fcma` share
a planner profile, so their plans agree bit for bit; `avx2` and `avx512` do not (16 and
32 registers), so for some lengths they choose different radices and so different bits;
`x86_v2` takes `avx`'s W as its `split_block`, as `avx2` takes `avx512`'s.
The operations the two NEON targets share are written once
(`src/isa/vocab_neon_common.inl`).

Twiddle storage on x86 and on plain NEON is chosen so `mul` needs no run-time
rearrangement: each twiddle is stored as the two vectors (wr, wr) and (−wi, wi) the FMA
form consumes. That is the "pre-arranged" form in the table, and one reason twiddle
layout is a per-target, per-pass decision (section 8.4). It doubles the bytes per stored
twiddle, which the planner weighs against the derived mode for long rows. On `avx512` a
register's C twiddles are stored as 2C reals (wr, wr, …) and then 2C reals (−wi, wi, …),
as on `avx2`; on plain NEON (C = 1 or 2) likewise. `neon_fcma` and scalar store (wr, wi).
Twiddles that every lane shares (lane passes) are stored once each in a broadcast form,
(wr, wi) on NEON+FCMA and scalar and one lane of the pre-arranged form, (wr, wr, −wi, wi),
on x86 and plain NEON, and read by `bcast_tw` (written by `pack_tw1`).

### 6.3 Data layouts

Three in-memory forms of complex data exist, and a pass declares which it reads and
writes:

| Layout | Form | Used for |
|---|---|---|
| `IL` | interleaved re, im (the caller's `std::complex`) | caller data; all intermediates of interleaved-arithmetic chains |
| `BL(W)` | blocks of W reals followed by W imaginaries | intermediates of split-arithmetic chains (no shuffles between passes) |
| `SP` | separate real and imaginary arrays | caller data in split-complex form (optional API layout) |

Arithmetic inside a codelet is either **complex-register** (interleaved) or
**split**. The trade-off, in registers: a register carries W reals either way, so density
is equal. Interleaved arithmetic keeps a radix-R codelet in R live registers, which lets
the radix go to 16 on 32-register targets and keeps passes few. Split arithmetic needs 2R
live registers but has no shuffles in multiplies and processes twice as many points per
codelet call. On NEON+FCMA the instruction counts per complex point are equal (a complex
multiply is two `FCMLA` interleaved, or two multiplies and two FMAs for a pair in split
form), so interleaved is expected to win there by needing no layout conversion and half
the live registers; the prototype and FFTL agree, but it is a measurement, not a law. On
x86 the answer differs by pass, length and core (finding E1). Both forms are generated
and experiment X1 in [PLAN.md](PLAN.md) settles the defaults for every target, FCMA
included. Where one form never wins on a target it is not instantiated there.

As built (task 1.5, `src/passes/split.inl`, `strategies/chain.hpp`): split arithmetic runs
only in a chain's last lane passes, a *split run*: from the pass after the first whose copy
count is a multiple of the target's `split_block` (W, and on `avx2` the W of `avx512`, so a
bitwise pair chooses the same passes) to the end, when that is two passes or more of radices
with split codelets. The run's first pass deinterleaves IL into BL(W), its middle passes go
BL to BL with no shuffle, its last interleaves into IL with the output edge; so BL never
leaves a chain and the public layout stays interleaved. The grouped first pass stays
complex-register: writing BL from it costs an 8 × 8 transpose of reals where IL costs a 4 × 4
one of complex pairs, the same one shuffle per register as the deinterleave, so the
conversion is paid at the run's ends either way and only middle passes gain. Split codelets
are forward only; a backward pass exchanges real and imaginary parts on loading and storing.
A plan takes split runs where its profile says so (`target_profile::split`: `avx2` and
`avx512`, both precisions, from X1) or where a test or fbench forces it
(`build_settings::layout`, fbench `LAYOUT`). Not yet done: the planner's costs are those of
interleaved passes, so it does not prefer a first radix that lets a run start (AVX2 picks 4 ×
16 × 16 for 1024, which has none); and the grid's phases stay interleaved.

### 6.4 ISA regions and safety

All target-specific code sits between `FFTN_ISA_BEGIN(target)` and `FFTN_ISA_END`, which
apply the target's attribute to every function in the region (`#pragma clang attribute
push` / `#pragma GCC target`) and open a per-target namespace so each target's functions
have distinct symbols. Kernel and codelet sources are `.inl` files without include
guards, written once and included inside each region. They are not templates over a
vocabulary type: a target attribute cannot depend on a template parameter, and that
design failed in every compiler configuration tried (section 14.1). The real type `T` is
an ordinary template parameter instantiated inside the region. Each region exports entry
points that take only pointers and integers; plan construction stores the chosen
target's entry points in the program. Rules that keep this sound (also in AGENTS.md):

- no translation unit is compiled with a global ISA flag;
- inside a region, only the vocabulary and built-in types: no standard-library templates
  (their instantiations would be compiled for the baseline, fail to inline, or be merged
  across targets by the linker);
- vector types never cross a function boundary that is not force-inlined;
- aggregates holding vector types define their special members inside the region;
- no lambdas inside a region (they do not inherit its target);
- a header is never included for the first time inside a region.

Every function inside a region has internal linkage (the `.inl` texts sit in an unnamed
namespace within the target's namespace); only data is exported: the target's function
table and the rows of its codelet table. Each translation unit therefore keeps its own copy
of every vocabulary function and kernel, and the linker cannot substitute a copy compiled
for one target for a caller of another. The library is compiled with `-Werror=psabi` under
GCC (experiment B1, section 14.1), always, not only in the gate.

Each target is compiled as nine translation units with the same region and the same kernel
text (`src/isa/kernel_text.inl`: twiddle packing, codelets, passes), each instantiating a
share of it. Eight row units (`src/isa/rows<k>_<target>.cpp`) instantiate the passes of the
codelet radices that `src/isa/rows.inl` gives them (radices 2–12, 13–16, 32, 64 once per
precision, and 17, 19 and 23 one each, about equal in compile time), each row as a constant
`codelet_row<T, I>::passes` of external linkage in the target's namespace (an explicit
instantiation). The main unit (`src/isa/<target>.cpp`) instantiates the other passes and the
function table, which points at the rows (declared there as explicit instantiations made
elsewhere: a row that no unit defines fails to link). Kernel units are optimised in every
build type. One unit per target needed 9–16 GB and 5–12 minutes to compile with Clang 23
under ASan and UBSan (Apple M5 Max; avx512 the most, GCC more still), which ran CI runners
out of memory; split, each unit needs 3.1–4.8 GB and at most about 3 minutes (radix 64 in
float, avx512), and the main units 0.3 GB.

Target strings: `x86_v2` is `"sse4.2,popcnt"` and `avx` `"avx,popcnt"`, the features their
run-time checks require (SSE4.2 implies SSE4.1, SSSE3, SSE3; AVX implies SSE4.2), with no
AVX in the first and no AVX2 or FMA in the second, which the disassembly test checks;
`avx2` is `"avx2,fma"`, the features its run-time check requires, and no AVX-512 feature; `avx512` is `"avx2,fma,avx512f,avx512dq,avx512vl"`, again the features
its check requires (BW and CD, which every such processor also has, are left out).
Compilers imply a few more from these names (Clang: POPCNT and CRC32 from the AVX2 chain,
F16C from `avx512f`); the checks do not test them, and every processor that passes a
check has them;
plain `neon` has no attribute, since Advanced SIMD is the AArch64 baseline, so its code
has what the build's baseline has and nothing more (on an Armv8.0 baseline no FCMA and no
pointer authentication, which the disassembly test checks); `neon_fcma` is
`"arch=armv8.3-a"`, because Clang's NEON header
declares the complex intrinsics for the whole Armv8.3 architecture (`"+fcma"` is
rejected). That string also allows other Armv8.3 instructions: the disassembly test
forbids RCPC and JSCVT in `neon_fcma` code. Pointer authentication is allowed there,
because branch protection may sign that code's returns with RETAA; `neon_fcma` is
therefore dispatched only on processors with both FCMA and pointer authentication (both
mandatory from Armv8.3), and a core with FCMA alone (A64FX) gets plain `neon`. The macros put
each target pragma last in its expansion: GCC drops the tokens that follow a
`_Pragma("GCC target ...")` within one macro expansion.

A dedicated test links every instruction set's translation unit into one binary and
checks the disassembly of each target's kernels (`tools/asm_check.py`) for the intended
instructions, for the absence of calls to out-of-line helpers, and for the absence of the
target's instructions in baseline code. What the build's own baseline already has is
allowed there: on Apple platforms the compiler targets Apple silicon, which has FCMA, even
under `-march=armv8-a` (only `-mcpu=generic` removes it).

---

## 7. Codelet generator

### 7.1 Role

A codelet is a straight-line DFT of small size R on R vector values: the arithmetic core
of every pass. Hand-written codelets limited the prototype (three kernel families, three
copies, composites built from generic twiddle multiplies). Every library that leads on
kernels generates them. FFTN's generator is a Python package (`gen/fftn_gen`), run by
developers; its output is committed C++, so users need no Python. It is a development
tool run on current machines, so it targets current Python (3.14 or newer, in the
project's `uv` environment) and uses what that offers: structural pattern matching over
the DAG's node types, frozen slotted dataclasses for hash-consed nodes, and `mpmath` for
high-precision verification. It is documented in `gen/README.md` (section 17).

### 7.2 Pipeline

1. **Build** the size-R DFT as a DAG of complex-valued nodes, choosing the algorithm by R:

   | R | Algorithm |
   |---|---|
   | 2, 4, 8, 16, 32, 64 | conjugate-pair split radix; twiddles that are multiples of N/4 are free, N/8 is (1 ± i)·√½ |
   | 6, 10, 12, 15, 20, 24, … (coprime factors) | Good–Thomas index mapping: no internal twiddle multiplies |
   | 3, 5, 7, 11, 13 | symmetric-pair form: sums and differences of x_j and x_{R−j}, then cosine/sine combinations, with the mean of each cosine row taken out first (m² − m + 1 multiplies instead of m² for m pairs, same additions) |
   | 9, 25, 27 | Cooley–Tukey with constant internal rotations |

2. **Track pending rotations.** Every node carries a pending factor (−i)^k. Combining two
   nodes whose rotations differ by 0, 1, 2 or 3 quarter turns emits `add`, `subj`, `sub`
   or `addj`. A multiplication by ±i or −1 therefore never costs an instruction. General
   constants are reduced modulo quarter turns first, which cuts the number of distinct
   constants and the register pressure.
3. **Simplify.** Constant folding, sign canonicalisation (push negations into consumers;
   treat a − b and b − a as one node), common-subexpression elimination by value numbering
   (evaluate each node at random inputs in high precision and hash), dead-code elimination
   for pruned inputs and outputs. Then the **real-coefficient form**: every node v is
   rebuilt as R(v) + i·I(v), the combinations of the inputs by the real and imaginary parts
   of its coefficients, so that multiplications by i (register swaps on x86) happen only
   at the outputs, once per output pair X_k, X_{n−k} (radix 16: 7 against 19), at no more
   operations (experiment X19).
4. **Lower** to one of two forms:
   - *complex-register*: nodes map directly onto the complex vocabulary, one operation per
     node. Each node's register holds its value times a rotation chosen backwards from the
     stores, so that no output needs an extra operation; a scale with a single user fuses
     into `fmadd_r`, a shared one stays one multiply;
   - *split*: each complex node expands to real and imaginary nodes, followed by a second
     simplification and **FMA shaping** (emit a ± b·c directly; factor rotations as
     c·(x − t·y) with t = s/c where the outer multiply can be absorbed).
5. **Order** the emission by greedy list scheduling: the ready operation that adds the
   fewest live values goes next, ties to the one that uses the most recently computed value,
   which finishes one sub-transform before starting the next; place each load just before
   its first use and each store next to the operation that produces its value. This keeps
   the number of live values near the minimum, which matters for radix ≥ 16 on 16
   registers and ≥ 32 on 32. (The recursive partition of Frigo's PLDI 1999 paper was tried
   first and kept more values live; the commit that holds it is named in
   `gen/fftn_gen/schedule.py`.) Past the register file the fewest live values is the wrong
   aim: a complex codelet that cannot fit 16 registers is also ordered by a list scheduler
   that tracks a model register file, and takes that order where a furthest-next-use model
   of the allocator counts at least a quarter fewer spill moves (radix 32 and 64; GCC
   allocates in source order on x86; experiment X54). A codelet whose radix exceeds
   the register count necessarily keeps some values on the stack; that is acceptable when
   the spills are few and ordered (FFTW's radix-32 and radix-64 codelets work this way),
   and ruinous when they are not: the prototype's hand-built radix 20, 24 and 32
   composites lost on every machine. The generator therefore records, per codelet and
   target, the spill count read from the compiled code, and the planner prices it.
6. **Verify.** Evaluate the emitted operation list in high-precision arithmetic against
   the exact DFT for random and basis inputs. A codelet that fails is not written.
7. **Emit** C++ against the vocabulary: one text for all targets per lowering, included
   in each target's region, with a per-target selection of which codelets to instantiate.
   The complex-register text is `src/gen/complex_codelets.inl`: `cdft<R>` with static
   member templates `plain`, `dit` and `dif` over `T` and `Sign`, which take the caller's
   accessor objects (`in.get(j)`, `out.put(k, v)`, `twid.get(k)`), so that pass kinds, edge
   variants and layouts change only the accessors (the calling convention is in
   `gen/README.md`).
   Alongside it, emit a metadata table (`src/gen/codelets.hpp`): radix, variant,
   direction, lowering, operation counts (adds, multiplies, FMAs, complex multiplies,
   vocabulary operations), peak live values, twiddle count. The planner's
   cost model, the kernel registry and the tests are driven from this table, so there are
   no hand-maintained lists of radices anywhere.

### 7.3 Variants

For each radix the generator can emit:

- **plain** (no twiddles): leaves and last passes;
- **DIF twiddle** (multiply outputs) and **DIT twiddle** (multiply inputs). Fusing the
  multiply into the first butterfly layer as `madd` is not done: it pays only where a
  product has a single user, and in the radix 2–16 codelets every twiddled input feeds a
  sum and a difference, where fusing both would repeat the multiply;
- **forward and backward**: generated separately for the complex-register form; for the
  split form the backward transform reuses the forward codelet with real and imaginary
  registers exchanged at load and store;
- **pruned**: declared-zero inputs and unwanted outputs removed by dead-code elimination
  (zero-padded convolutions, real-input and half-complex codelets);
- **fused-edge**: the real-transform post- or pre-processing folded into the codelet of
  the last or first pass (section 9.1).

Generated so far: the complex-register form of plain, DIT and DIF, forward and backward,
for the radices listed in `gen/fftn_gen/catalogue.py` (2–16 at task 1.1; 32 and 64 since
task 1.6, for leaves and big passes); the split form of plain and DIF, forward only, for
radices 2–16 (task 1.5; `gen/README.md`, "The split lowering"). Pruned and fused-edge
variants follow with task 2.1.

Reference values for regression: the generator's operation counts are compared with the
best published counts for each size (for example a size-16 complex DFT in 144 additions
and 24 multiplications in scalar split form). Reaching them means nothing is missing. The
table, the source of each entry (published, derived from its parts, or none with a
reason) and the recorded gaps are in `gen/fftn_gen/reference.py`; at task 1.1 the one gap
is 13, where the reference uses Rader's algorithm.

### 7.4 What the generator does not do

Instruction selection, register allocation and scheduling inside a straight-line block are
left to the compiler; current compilers do these well. The generator does what compilers
may not: algebra that changes floating-point results, and the choice of algorithm.

Pass loops (section 8) are hand-written templates that take a codelet as a parameter. If
experiments show that generating whole passes wins (the prototype lost to FFTL partly in
loop structure), the generator grows to emit them; the design does not depend on which.

---

## 8. Passes, programs and strategies

### 8.1 The index picture

A radix-R pass combines R input streams into R output streams. Three indices describe
every pass: `j` selects the stream (0 … R−1), `p` selects the butterfly group (and its
twiddles), `q` runs over independent copies that share twiddles. Which index the SIMD
lanes run over is the pass's defining choice.

### 8.2 Pass kinds

| Kind | Lanes run over | Loads | Twiddles | Where it is used |
|---|---|---|---|---|
| **Lane pass** | `q`: independent copies | R contiguous runs, a stream apart | one set per `p`, broadcast | every pass with at least C copies (s ≥ C): later passes of a chain; column phase of a grid; batches across rows; strided axes |
| **Grouped pass** | `p`: consecutive butterflies | DIF: R contiguous runs, with the R × C block transposed in registers before the store; DIT: one contiguous block of R·C values transposed after the load | per-lane, from a table contiguous in `p` | every pass with fewer than C copies (s < C): the leading passes of a chain; leaf phase of a grid |
| **Leaf** | — (whole transform in registers) | n values | constants | rows of n ≤ 16 … 64 |
| **Batch leaf** | rows: lane l holds row u + l | C consecutive elements of each of C rows, C × C transposed in registers (the last block overlapping where C does not divide n); outputs transposed back | constants | contiguous c2c batches of rows of a codelet length, from a per-length batch count (X53): `avx2`, `avx512` (avx2's two rows per register, its own four from 4 rows at 11–23 points) in double, NEON in float |
| **Pair leaf** | `p` then `q`: a grouped pass of whole units and a lane pass, both in registers | R1 contiguous runs per unit; the C × C transposes of the grouped pass's outputs are the lane pass's input | per-lane, as the grouped pass | contiguous c2c rows of n = R1 M (C dividing M) whose n / C registers fit the file: x86, 20–120 points (X28b), 8–18 and 22, 26 points in double (X37); NEON double 18–30 points, C = 1 (X37) |

Each kind exists in DIF and DIT form, in complex-register and split arithmetic, and with
**edge variants** that change only how values are loaded or stored: layout conversion
(IL ↔ BL), gathering rows into lanes with a tile transpose, the fused real post- or
pre-processing, multiplication by a factor array or scale, pruned loads and stores.
Variants are separate instantiations chosen at plan time. A generic pass never tests a
run-time flag in its loop (the prototype measured 0–40 % for doing so).

Which kind a pass is follows from its copy count s against the register's complex
count C. In a DIF chain s is the product of the radices already applied, so a leading
radix of at least C leaves only the first pass grouped; planner v1 prices tails per C
(section 11.2; section 6.2 gives the consequence on `avx512`); with C = 1
(NEON double) the distinction disappears. The copy counts that are not multiples of C
(tails) never go through a mask or select inside the main loop. As built (task 1.2) they
run the same instantiation on a padded tile: the last unit of a pass gathers its valid
elements into a small stack tile, element by element, computes all C lanes and stores
only the valid ones. Vector operations act on each lane independently, so a tile performs
exactly the same arithmetic per element as a full register; this is what keeps a row's
result independent of its position in a batch (section 15.5). On the x86 targets in GCC
builds (`FFTN_MASKED_TAILS`, section 14.3; experiment X23) a partial run of adjacent
elements is instead one masked load or store (`load_n`, `store_n`), the partial units run
after the loop over whole units (their masks and pointers kept out of it, with constant
unit counts inside it), and the grouped pass stores radices C does not divide through
C × C transposes padded with zero registers; the arithmetic, and so the bits, are the
tile's. Clang builds keep the tile: with the masked code Clang 23 lost up to 1.42 of the
tile's time on odd double lengths, for a reason not yet found. A narrower vocabulary
instantiation (FFTL's ladder of vector widths) is the alternative should tiles prove
slow; it needs a second width per target.

As built (task 1.2), in `src/passes/` (one text for every target, the codelet a template
parameter, everything else accessors; gen/README.md has the codelet calling convention):

- The **grouped** pass is a chain's first pass (s = 1): lanes over C consecutive
  butterflies p, per-lane twiddles, the R × C output block transposed in C × C blocks
  before contiguous stores when C divides R, and through a tile otherwise. On C = 1 it is
  the lane pass with s = 1. Passes with 1 < s < C (after a leading radix below C, which
  planner v0 chose for most lengths on `avx512`; planner v1 avoids it where tails cost more) are lane passes that run entirely as
  tails.
- The **lane** passes are the rest: lanes over copies q, one twiddle per leg shared by
  every lane (`bcast_tw`), the DIF codelet in the middle and the plain codelet in the last
  pass (m = 1).
- The **leaf** is one codelet call on one line, for n ≤ 16 and for a codelet radix up to
  the profile's leaf limit (32 on NEON and `x86_v2` in double, 16 elsewhere; experiments X8
  and X14): values are
  broadcast on loading and lane 0 is stored, so it takes any stride and runs in place, at
  the cost of C − 1 idle lanes; leaves vectorised across rows are the lanes strategy's
  (section 8.5, task 2.2) and, for short batches of contiguous rows, the **batch leaf**
  (`src/passes/batch_leaf.inl`, X53): the same codelet over C rows at once, each lane doing
  the leaf's arithmetic to its row, so every row of the plan gets the same arithmetic
  wherever it falls (a single row left over runs in the leaf's form, the same codelet); a
  row of a batch need not get what a plan of that row alone gives it (section 15.5). A leaf kernel takes its first line's addresses and the distances to the next
  line as arguments (`leaf_fn`), so the driver runs a batch of leaf rows as one call on the
  plan's own arguments (section 8.3, X20).
- The **generic** pass (any radix by direct evaluation) and the copy are lane-form passes
  on every target (the generic pass's lanes run over copies, so a single prime row has one
  valid lane; with the Bluestein threshold at 16 the rules no longer choose it, X17);
  the two chirp passes of a Bluestein stage (section 8.5) are elementwise, in blocks of C
  consecutive elements, with the chirp per lane or, when C divides the copy count,
  broadcast; the real-transform edges run one element at a time on every target.
  consecutive elements; the real transforms' pairing passes (section 9.1) run over C
  consecutive bins and their C partners, read and written downwards through a tile, except
  the strided-line edges, which run one element at a time.
- **Edge variants**: scale and a per-element complex factor, on loading (first passes,
  leaves, generic passes, copies) and on storing (last passes, the same three), as the
  first two edge operations of section 12.1; real edges take the scale on storing. Each
  is an instantiation: a pass kernel table per target and precision indexed by radix,
  direction and edge operations (`pass_kernels`, `src/passes/pass_args.hpp`), filled from
  the generated codelet table, so a new radix gets its passes without any change.
- Full runs are single vector loads and stores, so on C > 1 a pass touching the caller's
  memory needs stride 1; the chain copies other strides to or from scratch (section
  8.5).

### 8.3 Programs

A plan's executable form is a **program**: a flat array of plain records, prepared at
construction and interpreted by a small driver.

```cpp
struct stage {            // one pass over one buffer range
  pass_fn   run;          // the instantiated pass kernel: run(args, src, dst, first, last)
  const void* twiddles;
  std::uint32_t radix, m, s;        // geometry
  std::uint8_t  src, dst;           // buffer slots: caller in, caller out, scratch a/b, block
  std::uint32_t chunks;             // independent ranges of units this pass divides into
  …
};
struct loop {             // repeat the enclosed records
  loop_kind kind;         // rows | lane_blocks | sub_rows | planes | pencils
  std::size_t count, in_step, out_step;
  std::uint32_t body_begin, body_end;
  bool independent;       // iterations may run on different threads
};
```

Loops express everything that is not a single pass: a batch of rows, depth-first blocking
(run all passes on one cache-sized block before the next), the row phase of a grid, the
planes of a volume. Walking records costs a few nanoseconds per record and transform, which
is more than a tiny transform's arithmetic, so the commonest programs are not walked: a
program whose body is a few stages and no loop (rows) is marked *flat* at construction, and
the driver then runs a batch of leaves as one kernel call, one transform from a contiguous
copy of the stages (entered straight from `execute_core`, its slots passed as scalars), and
a batch with the slots moved by the batch step (experiments X20 and X28: at 8 points 7.7 →
4.3 ns per call, 7.1 → 1.7 ns per transform in a batch on the M5 Max); in a walked program
a loop over lines whose body is one leaf (the rows of a plane of short rows) is likewise one
call of the leaf over all its lines (X40: 8 × 6 to 32 × 32 planes 0.59–0.85 of the walk's
time on the Mac and the EPYC); and a multi-axis program of at most 256 kernel calls per
transform is not walked at all: its calls are unrolled at construction with the slots' byte
offsets (`program::calls`), and the driver calls them in turn (X41: planes of up to 1024
points 0.87–0.94 of the walk's time on the three machines; the walk had cost about 10 ns a
plane on the M5 Max). Every kernel, leaf
or pass, takes its source and destination as arguments and reads the rest from the plan's
own arguments: a fresh copy of the arguments per stage and transform cost 3–5 ns per stage
on x86, most of it the copy's store-to-load forwarding into the kernel (X28). Every kernel
still gets the arguments and units the walk would give it, so the bits do not change. Two
consequences:

- **Blocking and batching are data**, not separate engines.
- **The threading work graph is read off the program** (section 13.2). Independent loops
  give whole-row, whole-plane and block tasks. Within one row, each stage is a phase and
  its `chunks` ranges of units (blocks of butterflies or copies, as the pass kind defines
  them) are shared between threads, with the stage
  boundary as the ordering constraint; this is how the prototype split chain passes, and
  the blocked grid adds coarser chunks (column blocks, rows). The division of work is
  fixed by the plan, which is what makes results independent of the schedule.

### 8.4 Twiddles

- Tables are per pass, contiguous in the order the pass reads them, in the storage form
  the target's `mul` consumes.
- Storage mode is per pass. **Full** (every twiddle stored) is the only mode built at
  first. **Shared** (conjugate-pair and symmetry sharing: fewer loads) and **derived** (a
  few stored, the rest computed in registers: for long rows where twiddle traffic competes
  with data traffic) are added only if experiment X5 shows they pay. Built so far: the
  depth-first chain's top passes derive theirs from two small tables, one more complex
  multiplication per leg (section 8.5; X46: 0.69–0.84 of the stored table's time on the
  Xeon 6240 at 16384–262144 points, which deleted the stored form).
- Values are computed with exact integer octant reduction and double-double evaluation,
  rounded once; no recurrences. Tables are shared between plans through the twiddle cache.
- As built (task 1.2), full mode, per plan (deduplicated within a plan, not yet shared
  through `table_cache`): the grouped pass reads R − 1 rows of w^(pk), each padded with
  ones to a multiple of C, in the per-lane storage of `load_tw`; lane and generic passes
  read (R − 1) twiddles per group p in broadcast storage (`bcast_tw`), one stored twiddle
  per entry rather than C copies (as FFTL.jl's lane passes read theirs, broadcast from
  memory per vector); the generic pass's roots and the real edges' twiddles are broadcast
  storage too. Every entry is an entry of the length's root table (the roots of n, or of
  the radix for the generic pass), the exact rearrangement −i w = (Im w, −Re w) of one (the
  real pairing's twiddles, section 9.1), or a padding 1: the root tables are what the tests check
  entry by entry against the reference. The packing into storage forms is exact (copies
  and sign flips, `pack_tw` and `pack_tw1`), which the vocabulary tests check bit for bit
  through `load_tw` and `bcast_tw`; the packed tables of a plan are not compared with the
  reference directly.

### 8.5 Strategies

A strategy builds a program for a 1-D transform of length N from passes. There are five.
The planner picks among them per shape (section 11); the regimes named here are the
starting hypotheses.

**Chain** (Stockham autosort, breadth-first). N = R1·R2·…·Rk. Pass i reads one buffer and
writes the other; outputs land in sorted order, so there is no reordering step. The
leading passes (until the copy count reaches C) are grouped passes, the first reading the
caller's input; the rest are lane passes, the last writing the caller's output. Scratch: one buffer of N. Best where the row and its
scratch fit comfortably in cache and passes are cheap: short and medium rows everywhere,
which is most of the target workload. Radix order is a planner choice. Starting rules: radix by
register count; a leading radix of at least C where C > 1; otherwise small radices at the
ends (they touch unpadded caller memory in R streams) and larger ones in the middle.
These pull in different directions on wide registers, which is what experiment X2
resolves per target.

As built (task 1.2, `src/strategies/chain.cpp`): radices from planner v1 (section 11.2);
the first pass grouped, then lane passes, a radix above 16 a generic pass in any
position except the trailing ones above the profile's Bluestein threshold, which form one
Bluestein stage at the end (below), a single codelet radix (n ≤ 16, or up to the leaf limit)
a leaf, n = 1 a copy. The input edge's operation is applied by the first
pass that is not a copy and the output edge's by the last, so neither costs a pass. On
C > 1 a line whose stride is not 1 (a strided axis, a strided row) is copied into scratch
before the chain or out of it after: an extra pass that the lanes strategy removes for
batches of short rows (task 2.2) and for the strided axes of c2c planes and volumes (X40, section 9.3). A single generic pass cannot run in place, so it writes scratch
and is copied when its source is its destination. Scratch is two lines a and b; an
out-of-place line with a contiguous destination (not aliased, no copy, no Bluestein
stage) uses the destination in place of b, so a chain touches three lines, not four
(experiment X18: up to 15 % faster at 512–16384 on a 32 KiB L1). A chain of two such passes
whose last radix has a codelet runs its first pass into the destination and its last pass
in place there (a last pass reads and writes the same elements in each unit), so it touches
two lines and no scratch (experiment X47: 512 points 0.95 of the time on the EPYC 7763).

**Grid** (two-phase Cooley–Tukey; the family that contains FFTW's recursion, Bailey's
four-step and KFR's sandwich). N = N1·N2, the data viewed as N1 rows of N2. One phase is
a length-N1 transform down the columns, done by lane passes over contiguous lanes; these
passes run **in place** and can run depth-first over blocks of columns that fit in cache.
The other phase is N1 contiguous transforms of length N2, each a leaf or a short chain
inside L1. Twiddles between the phases fold into one of the adjacent passes. The two
phases' natural orders differ by a transpose, which folds into the row phase's strided
loads from the caller's input (DIT order: rows first, then columns in place in the
output) or its strided stores to the caller's output (DIF order: columns first in a work
buffer, then rows). Both foldings need distinct source and destination, so:

- out of place, DIT order needs no scratch of size N and DIF order needs one;
- **in place** (N1 ≠ N2 in general) the grid needs either one scratch of size N, or an
  explicit in-place permutation pass (a digit-reversal swap, which exists only for
  palindromic radix sequences, as in KFR's sandwich), or, inside a pipeline, no
  permutation at all (section 12.2). Experiment X12 chooses per regime.

Two regimes:

- *Few big passes* for rows that exceed L1 but must share L2 with their buffers (finding
  E3): N2 a leaf of 16–128, N1 one or two lane passes of radix 16–64, working in the
  output buffer with no scratch of size N.
- *Blocked* for rows beyond L2: column blocks sized to L1 or L2 run depth-first, rows are
  L1-sized chains. This is also the form that splits across threads: column blocks and
  rows are the independent chunks.

As built (task 1.7, `src/strategies/grid.cpp`, experiment X12): the decimation-in-frequency
form, generalised to two or three phases (n = N1 N2 or N1 N2 N3), every phase blocked; the
"few big passes" regime is not built separately. Phase i transforms down adjacent columns
of what phase i − 1 wrote: with L the product of the factors before it and M of those
after, element (l, c, r) at l + L (c + M r) becomes (l, k, c) at l + L (k + Ni c), times
w_(Ni M)^(c k); after the last phase the output is in natural order. Each phase loops over
blocks of `block` adjacent columns (program loops, one independent iteration per block): a
block is gathered row by row into tight scratch a, transformed there by a chain of lane
passes (s ≥ block ≥ C, so none is grouped and none has tails; the radices are the fewest
passes, most even, of radix up to 16, not planner v1's, whose tables price a grouped first
pass), and written back row by row. Experiment X3 allowed phase radices up to 64
(`build_settings::grid_phase_radix`): they lost at 4096–16384 (up to 1.8 times: single radix-64
passes for 64-point phases spill) and won at 32768–65536 on the Xeon (0.87–0.98
on both instruction sets) and the EPYC at 32768 (0.94–0.98); a phase radix that depends on
the length waits for X12b's grid thresholds, so the default stays 16. Phase 1's columns become rows, so it writes through a twiddled transpose
(C × C tiles transposed in registers) with each element's twiddle w_n^(c k) formed as
w_n^(q k) w_n^(c0 k) from two small tables (one rounding more than a stored twiddle; a
table of n values cost memory traffic and plan memory for no measured gain); later phases'
blocks lie within one column c, so their twiddle is one value per row, applied by the
scatter, and the last phase's scatter applies the scale. Gathers and scatters by default,
because a block's rows usually lie a power of two apart and share cache sets, and they
prefetch eight rows ahead, since rows a page apart defeat the hardware prefetchers, except
on `avx2`, where the rows asked for share the scratch block's 8-way L1 sets and evicting it
cost more than the prefetch saved (X26). A
phase whose radices are all at most 8 instead reads its rows in its first pass and writes
them (with the row twiddles or the scale) in its last (`phase_pass`, task 1.7b), while the
row fits `grid_fuse_bytes` (default a quarter of L2): eight rows per pass fit the L1 sets'
ways, radix 16 at the ends doubled the time, fusing one end only gained nothing, and
beyond L2 the copies' prefetching won (X12). Placement: the last phase writes the positions it read (in place in
the output); phase 1 writes the output out of place with two phases, otherwise the
workspace block (n complex values), which is also the in-place route (the scratch of N of
the list above). Scratch line b is staggered from a by seven 64-byte lines (rule 4 of
section 10; X4). Where the L2 is shared by a cluster (Apple) an intermediate already in
the workspace block is padded after each of phase 2's rows, so that the rows phase 2
gathers spread over the cache sets (`grid_pad`, profile `grid_pad_period`; X26b: 0.976
over the long rows); on x86 it gained nothing and is off. The loops step a twiddle cursor (program.hpp) so a block's stage reads its
own table. Planner: a c2c row of one axis, contiguous on both sides, without edge factors,
is a grid when its bytes exceed `grid_min_bytes`, in three phases beyond
`grid_three_bytes`, by default half of L2 and L2 where each core has its own L2 (shared by
no more CPUs than the L1: x86), and the L1 data cache and an eighth of L2 where a cluster
shares it (Apple; X12b: a private L2 serves a chain's passes well, a cluster's does not).
With a private L2 and at least 2 MiB of L3 per CPU sharing it (Zen's CCX), a row whose
length is not a power of two and has a power-of-two part of at most 1024 points (16 KiB)
stays a chain up to half of L3 (`chain_part_limit_bytes`; X29: 0.64–0.85 of the grid's
time on the EPYC 7763 at 46080 to 2 Mi points, steadily; the Xeon 6240, with 0.7 MiB of L3
per CPU, keeps L2/2 for every length). Power-of-two c2c rows stay chains beyond
`grid_min_bytes` up to `pow2_chain_bytes`: twice L3 on those Zen-like caches, four times L2
where a cluster shares it (Apple), never elsewhere (the Xeon); so does a real row's
half-length transform (X33: 0.58–0.97 of the grid route's time on the EPYC and the Mac). A
power-of-two chain longer than `grid_min_bytes` takes FFTL.jl's radix sequence instead of
the rules': radix 8 with one or two leading 4s up to `pow2_radix8_bytes` (any size on Zen,
L2/8 on Apple), radix 4 with a leading 8 for an odd exponent beyond (X32: 0.57–0.85 of the
grid's time on the EPYC from 32768 to 2²² points, 0.57–0.94 on the M5 Max from 16384; the
Xeon lost 1.06–1.50 with chains). The rules price radices in cache (X2) and choose radix-16
passes, whose 16 streams a power of two apart fall into one L2 set where the buffers lie on
transparent huge pages: X29's chains of 1–2 MiB on the EPYC were bimodal (0.7 to 1.8 of the
grid's time by process) for that reason, and chains of radices ≤ 8 are steady (X32).
Where a private L2 has less L3 per CPU behind it (the Xeon 6240), a row whose length has
an odd part (n over its largest power-of-two factor) of at least 15 stays in two phases at
any size (`grid_two_odd_part`; X34: two phases took 0.70–1.01 of three phases' cycles on
the 15 such rows tried, 71680 to 4 × 10^6 points, 100000 0.73; rows nearer a power of
two, odd part 3–9, lost up to 1.13 as two phases, and on the EPYC two phases lost
1.13–1.67 below 16 MiB).
On x86-64 the same machines (measured on the Xeon 6240) also make c2c powers of two grids
beyond a quarter of L2 (`pow2_grid_bytes`: 32768 points, a two-phase grid 0.89 of the chain's
time; 16384 lost as a grid in every split, and c2r 65536 lost 1.13, so real rows keep
`grid_min_bytes`), give two-phase blocks twice L1 (`grid_two_block_bytes`: blocks of 8 for
192–256-long phases, 0.94–0.97 at 46080 and 65536 in place) and cap three-phase blocks at
16 columns (`grid_three_block`: 131072 and 262144 0.93–0.96 against blocks of 32). Radix-32
and radix-64 grid phases there were rejected (X39: 0.985 over the rows they change, c2r 2²⁰
1.02–1.03). After X38's radix-20 codelet, grid phases there may take radix-20 passes
(`grid_phase_radix` 20: phases of 200 as 10 · 20), r2c rows' half-length powers of two follow
`pow2_grid_bytes` too (c2r keeps the chain), and c2c powers of two out of place stay in two
phases beyond half of L3 (`grid_two_pow2_bytes`; 2²⁰–2²⁴ 0.87–0.95 of three phases; in place
and c2r lost). `avx512` double sequences of 300–25000 points with 2² · 5 to spare start with
a radix-20 pass (overrides; a c2r line, whose pairing takes the first pass, keeps the costs'
sequence).
On x86-64 a private L2 of 2 MiB or more (Granite Rapids) takes none of those chains, whatever
its L3 per CPU (a VM's L3 per CPU is the socket's L3 over the VM's CPUs): the grid and three
phases beyond L2/2, two phases where the power-of-two part of n is at most 2048 complex
doubles (`grid_two_part_bytes`), and no odd-part rule (X35: 30 changed cases at 0.59–1.007
of the chains' and X34's time on a Granite Rapids VM; an Axion VM, Neoverse V2 with the same
L2, lost 1.15–1.28 and keeps the large-L3 rule).
Where a cluster shares L2, a row whose length is not a power of two stays a chain when the
power-of-two part of its length is at most L1/8 in bytes (1024 points in double on the
M5), at any size, or when the row is at most 2 · L1 (`chain_part_bytes`,
`chain_mixed_bytes`; X29: such chains took 0.49–0.72 of the grid's time to 4 Mi points,
because the cluster L2's conflict problem is a power-of-two problem: an odd factor in
every leg distance spreads a pass's legs over the L1 sets). A row past these thresholds is
a grid only if it has a split whose factors
have no prime above 13 and whose first factor and n / N1 are multiples of the block (at
least max(2, C)). A long row with a prime factor above 13 therefore stays a chain, with
its Bluestein or generic last pass (9.2): the grid takes only lengths whose every factor a
codelet chain covers, all below the Bluestein threshold. With `scale_on_input` (a test
knob) the scale moves to phase 1's gather. The split: the widest block, then the shortest longest factor, then the
most even, then the shorter first factor; the block is the widest power of two up to 32
with block × max Ni within `grid_block_bytes` (the L1; twice that with three phases, where
X12 found wide rows worth more than a chain inside L1). These are machine-profile fields
(`src/threading/schedule.hpp`) so X12 can tune them per machine; `build_settings` forces
the strategy, the factors and the block. Not built: a fourth phase (a phase's chain leaves
L1 for rows beyond about 2^24 points on a 128 KiB L1), DIT order, padded work arrays (so
that end passes of radix 16 could read rows without set conflicts), and
splitting one row's blocks across threads beyond what the work graph already shares (task
3.6).

**Depth-first chain** (decimation in time at the top; experiment X46, `src/passes/depth.inl`,
`chain_builder::c2c_depth`). N = F1 M (or F1 F2 M): X[k + M t] = Σ_r ω_F1^(r t) w_N^(r k)
Y_r[k], Y_r the length-M transform of the sub-row x[F1 j + r]. The sub-rows are F1
interleaved sequences, the layout a Stockham lane pass reads with s = F1 copies, so one pass
(lanes over C adjacent sub-rows, broadcast twiddles) runs the first pass of every sub-row's
chain and stores each sub-row's outputs, transposed in registers, into its own block of M
elements of the output (with two tops, sub-row d1 + F1 d2 into block d1 F2 + d2); a loop then
runs the rest of each sub-row's chain (the chain of M, from its second pass) on its block
with scratch lines a and b of M, in L1 or L2; one or two top passes (the DIT codelet, lanes
over adjacent k, legs M apart) combine the blocks in place. In place the blocks live in the
workspace block and the last top pass writes the output. So a row is touched in three sweeps
by two lines, where the breadth-first chain makes one sweep per pass over three. The top
pass's twiddles are derived (section 8.4): w_N^(r k) with k = k_hi B + k_lo as two
multiplications, by w_N^(r k_lo) (per lane, B entries a row) and w_N^(r k_hi B) (broadcast),
B the divisor of M with the fewest entries; the stored table would hold (F1 − 1) M entries of
32 bytes on x86, as many bytes as the row (X46: with it the depth-first chain still missed L2
0.3–0.7 times per point at 16384–25000 on the Xeon 6240, with the derived tables 0.00–0.01).
Requirements: C divides M, and F1 when there are two tops (with one top the first pass's
last block of sub-rows may be partial, as a lane pass's is: X51, so 25000 = 2³ · 5⁵ runs as
10 × 2500 on `avx512`'s 4 lanes), codelet radices with a depth-first pass for the tops and
the sub-row's first radix (2–16, 20, 32; 25 only as a first pass), a sub-row chain of two
passes or more. The rules take it where the profile's fields say (a byte range, a limit for
powers of two, a power-of-two part above a size, up to three tops tried in order, double
and optionally float), on targets with 32 vector registers (`avx512`, NEON) or, with tops up
to 8, on `avx2`:
- the Xeon 6240's profile (x86-64, private L2 under 2 MiB with little L3) from L2/4 to 64 L2
  (16384–2²² complex doubles) and in float from L2/4 to 8 L2 (32768–2²⁰ complex floats, the
  measured range) with one top of 16, else 20, else 10
  (X46: 16384–262144 0.67 of their time; X51: 25000 0.75, 524288–2²² 0.68–0.75, float
  0.61–0.78; FFTN ÷ FFTW 1.06–1.35 → 0.50–1.00, ÷ MKL 0.33–0.90);
- the Apple profile (a cluster shares L2) from L1 when the length's power-of-two part is
  above L1/8, with one top of 4, to 2 L1 for powers of two and to 16 L1 for the rest (X46:
  8192 0.73, 16384 0.84; X51: 20480–122880 0.69–0.86; from L1/2, batches of 4096 lost
  1.03–1.04; powers of two beyond 2 L1 gained nothing against X32's chains);
- the EPYC 7763's profile (x86-64, private L2 with a large L3) where that L2 is under 1 MiB
  (Zen 3; Zen 4 and Zen 5, 1 MiB and `avx512`, keep their plans: unmeasured), for lengths
  that are not powers of two, from L2/4 to 4 L2, power-of-two part above L1/4,
  one top of 4 (X51: 12288–122880 and 46080 0.63–0.79; tops of 16 lost 1.2–1.7 on `avx2`'s
  16 registers; powers of two tied or lost against X32's chains).
- x86-64 with a private L2 of 2 MiB or more (X55, Granite Rapids): the Xeon 6240's fields,
  from L2/4 to 64 L2 with tops 16, 20 and 10, double only (32768–2^22 and 46080, 100000:
  0.44–0.82 of the grids' time, 0.688 over the rows it changes; 2^21 tied). Sapphire and
  Emerald Rapids share the branch, unmeasured.
- x86-64 with a private L2 between 1 and 2 MiB (X58, an Ice Lake VM, whose large L3 per CPU
  otherwise puts it with the EPYC): the same fields, from L2/4 to 64 L2 (0.596 of the time over
  the rows it changes; long rows ÷ FFTW 1.13–1.53 → 0.64–0.79). Zen 4 and Zen 5 (exactly 1 MiB)
  stay out, unmeasured.
Not built: three or more top factors, edge factors; float on the Apple and
EPYC profiles (X51's Apple float screen: mixed rows 0.69–0.84, but 24576 and 196608 did
not gain). In place, the blocks live in the workspace block: n complex elements per team
member (X51's range on the Xeon 6240 makes that up to 64 MiB per member at 2²² points, 8 MiB
in float at 2²⁰), where the out-of-place chain needs only two sub-row lines of scratch.

**Lanes** (vectorise across rows). C rows, or C neighbouring lines of a strided axis,
become the lanes of every pass; the whole transform is lane passes. For row-major batches
the first pass gathers a tile of rows into lanes and the last scatters it; for a strided
axis the lanes are already adjacent and nothing is gathered. Lane blocks are sized so a
block and its scratch stay in L1 or L2. Best for short rows whose own first passes
vectorise badly (awkward lengths on wide registers) and for every strided axis.

As built (task 2.2, `src/strategies/lanes.cpp`, `src/passes/lanes.inl`, experiment X7), for
plans of one axis with a batch: a block of B rows of the innermost batch dimension is a
*tile*, element i of row r at i B + r, which is B interleaved copies of a line of n (the
layout of a Bluestein convolution's copies), so the row's own chain runs on it with every
pass's copy count multiplied by B: the same radices in the same order, the same split run,
the same twiddle values and edges. The row's grouped first pass and its leaf become lane
passes whose lanes do the same operations on the same values, so each row gets the bits the
chain gives it alone, whatever block it falls in (section 15.5; the tests compare every
byte with the chain's). Where a run of C rows is one vector access (C = 1, or rows one
complex value apart: interleaved rows) the block's first pass reads the rows and its last
pass writes them, rows outermost (`lanes_first`, `lanes_last`, `lanes_leaf` for a chain of
one pass); otherwise a gather builds the tile from C × C transposes in registers and a
scatter takes it apart (a first pass reading C rows through a stack tile per element was
slower than the gather on NEON float, X7). Real rows read their reals as pairs, run the
half-length chain on the tile and pair their bins on it with the other routes' arithmetic
(`lanes_r2c_pair`, `lanes_c2r_pair`, reading or writing the rows where they can); a real row
whose complex side is strided gets the bits of the contiguous row, not of the chain's
strided route. The program's body is one block's stages with `program::lanes` = B; the
driver runs each run of the innermost batch loop in blocks of B, then blocks of C through a
second body (`lane_tail`), the last padded with zero rows that are never written, and sets
`pass_args::rows` per block; so `execute_transforms` over any range, and a team's claims,
give the same bits. A lanes program's work graph is one unsplittable phase per transform.
B is the largest power of two times C up to 64 rows and the batch whose two scratch tiles
fit half the L1 data cache (`lanes_block`). The rules take the lanes for a batch of at
least C rows up to `target_profile::lanes_up_to` (X7, measured against the chain with X20's
batched leaves): 100 on AVX-512 and 32 on AVX2, in float only; in double and on NEON the
chain's batched leaf (one kernel call for a batch of leaf rows, section 8.3) runs short rows
as fast as the lanes, so the threshold is 0 and the lanes run there only when forced; never
with an input edge operation, forced radices or a real row strided on its real side, and by
the rules never for a real row strided on its complex side (the lanes give it the contiguous
row's bits, the chain's strided route others, and avx2 and avx512, whose thresholds differ,
must agree). The
lanes driver runs each block's stages through `run_records`-style argument copies. X24
tried X20's per-call preparation, with a lanes leaf run in one call: 1–7 % slower at batch
64 of 8–16 points, so it was not kept. X24 also tried edge passes that read and write rows
one 128-bit unit per row instead of a gather and scatter (FFTW's inserts on x86). That won
on AVX2 but cost up to 1.17 on AVX-512 at 24–256 points, so it was not kept either; even
with both, double c2c lost to the chain at 8 points. The strided axes of c2c planes and
volumes run as lanes over blocks of adjacent columns (section 9.3). Not built: lane blocks
across an outer batch dimension, a threshold per transform kind (X24 follow-up).

**Bluestein** and the **generic prime pass** cover lengths with large prime factors
(section 9.2).

As built (task 2.3, `src/strategies/chain.cpp`, `strategies/bluestein.hpp`,
`passes/bluestein.inl`): a **Bluestein stage** is the last pass of a chain, not a separate
strategy. The rules give n = n_s · q₁ · q₂ · … with the large primes last (section 11.2);
the product Q of those above the threshold is one stage with s = n/Q copies and m = 1. Its
Stockham input x[q + s j] and output y[q + s k] are then s interleaved sequences of length Q,
copy innermost, in the same positions, which is exactly the layout of s copies of a chain
of length M: a chain of M whose every pass has its copy count multiplied by s computes s
interleaved transforms (the passes treat copies as independent and share their twiddles).
So the stage is a program fragment built from the chain's own passes, nothing recursive:

1. *bluestein in*: z[t + s j] = x[t + s j] · w_j for j < Q, zero up to sM (w the chirp
   exp(±iπ j²/Q), each entry correctly rounded from the root table of 2Q), with the line's
   input edge when the stage is the first pass;
2. the chain of M forward over the s copies (its first pass grouped when s = 1, a lane pass
   otherwise), times the kernel spectrum dft(b)/M (b = conj(w) on both sides of zero):
   output element t + s f takes entry f. When C divides s (s = 1 included) its last pass
   applies it as an output factor (section 12.1), its lanes sharing one broadcast entry
   per leg for s > 1; otherwise those lanes straddle entries and the first pass of the
   backward chain, a lane pass whose lanes are the copies, multiplies its input instead
   (the input form cost 5 % at 4096 × 17 on the Mac, its multiplications landing in a
   compute-bound pass);
3. the chain of M backward;
4. *bluestein out*: y[t + s k] = z[t + s k] · w_k for k < Q, with the line's output edge.

Both chirp passes are elementwise over i = t + s j because m = 1, so they vectorise in
blocks of C whatever s is, and the out pass may write in place, which makes the stage run
in place (no copy back, unlike a single generic pass). The convolution ping-pongs between
the scratch lines a and b, an even number of passes, so it ends where it began; the scratch
lines grow to sM (at most 4n). The plan holds the kernel spectrum once (M entries,
computed at construction in double by a small mixed-radix transform over the root table of
M) and the chirp once (Q entries, broadcast to a unit's C elements, which share j, when C
divides s; otherwise one entry per element, sQ = n, because a unit's lanes then straddle two
values of j). Both are kept per (Q, direction), so the axes of a plane with the same Q
share them. Before task 2.3b both were stored per element (sQ and sM entries): 889 MB of
tables for n = 17 · 2²⁰ in double, now under a kilobyte. No allocation at execution, every chunk of every stage a fixed range as
for any pass. Not done: running the s copies in blocks of B, which would shrink the scratch
lines from sM to BM (1.2 GB per team member at n = 17 · 2²⁰) and keep each block's
convolution in cache, needs a loop over blocks whose source is a scratch line, which the
program's loops (scratch slots do not move, section 8.3) cannot express yet. M is the 7-smooth length in [2Q − 1, next power of two] whose chain
plus chirp passes cost least under the profile (`bluestein_length`, section 11.2), with a
power-of-two part of at most `machine_profile::bluestein_pow2_part_bytes` (experiment X36:
a quarter of L1 where the L2 is shared, as on Apple, whose chains of a power of two from
the L1's size up ran 1.4–1.7 times as long per point as 7-smooth lengths just above; the
L1's size on Zen-like caches, where the convolution's two scratch lines a power of two
apart made 4093–32749 take 1.4–5 times as long; unlimited on the Xeon 6240, where a limit
gained nothing); when that excludes the power of two, the candidates run up to 2 (2Q − 1).
A convolution's chain takes the rules' radices, except that one of a power of two takes
X32's long-row radices beyond `grid_min_bytes` where X32's chains apply (Apple, Zen) and
beyond four times the L2 elsewhere (`machine_profile::long_convolution_bytes`; Granite
Rapids takes the Xeon's rules, unmeasured), since a convolution is never a grid: the rules'
radix-16 passes over 2¹⁸ points ran 2.5 times as long on the M5 Max, while on the Xeon the
long-row radices lost at 1 MiB (X36). A single copy's convolution (s = 1) takes the split
run its length would as a line (`split_from`; x86). A Bluestein stage that would have
1 < s < C copies covers the whole line instead (Bluestein needs no prime) where that
convolution is no longer than the copies' together, so its passes run whole units (X36:
1018 = 2 · 509 on AVX-512 0.68, in float 0.40); so does one of s = C copies while the line
is not longer than `long_convolution_bytes` (X36b: its chains over s = C copies have no
grouped first pass; EPYC 1018 0.73). Running the copies one at a time, as FFTW does, needs a
loop over a scratch-resident line, and a grid convolution a second movable workspace slot:
neither is expressible yet (X36b). A Bluestein stage of one copy (a c2c row of one axis
whose chain is that stage, `chain_builder::bluestein_single`) instead runs its convolution
as two depth-first chains (section 8.5's depth-first chain; experiment X50,
`chain_builder::bluestein_depth`) in double where the profile's depth-first fields would make
a row of M one, with `machine_profile::depth_convolution_max_bytes` as the upper limit in
place of the rows' (the convolutions measured: the Xeon's 32 MiB, 2²¹ complex doubles, beyond
its rows' original 4 L2 since a convolution is never a grid and the alternative is the
breadth-first chain; Apple's 2 L1); longer convolutions keep the breadth-first chain,
unmeasured. The workspace block is the movable line: the forward chain's first pass reads the
input times the chirp (zero from Q on: the chirp pass in, fused) and writes its sub-row
blocks into the block, a loop over the blocks runs the sub-rows (with scratch a and b), and
its top pass writes into scratch a times the kernel spectrum (the top pass's factor edge);
the backward chain's first pass reads a into the block, its sub-rows run the same way (a is
dead by then), and its top pass stores only the first Q outputs, times the chirp with the
output edge's scale, to the output (the chirp pass out, fused). Scratch a holds M elements,
the block M. On the Xeon 6240 the depth-first chain took 0.54 of the breadth-first chain's
time at 2¹⁸ points and 0.56 at 2073600; the stage took 131071 to 0.70 of its time and
999983 to 0.79 before the chirp passes were fused (X50).
Why a pass and not a strategy that wraps the chain: the program and stage design already
expresses it, the passes before it (codelet and generic) keep their positions and edges,
r2c and c2r get it through their inner complex chain unchanged, and the threading work
graph sees ordinary stages. Restricting the stage to m = 1 is what makes its edges
elementwise; several large primes are therefore one stage of their product rather than
one per prime (one convolution of about 2Q points instead of passes of about 2qᵢ · Q/qᵢ
each, which is about the same work).

A **Rader stage** (experiment X27, `passes/rader.inl`) takes the Bluestein stage's place
when the product is a single prime q in the profile's Rader range whose q − 1 is 13-smooth
(`rader_root`, below): with g
the smallest primitive root, bins g⁻ᵏ are x₀ plus a cyclic convolution of length L = q − 1
of a_m = x(g^m) with b_l = w^(g⁻ˡ), and bin 0 is x₀ plus the sum of the inputs. *rader
in* gathers a (from an index table of g^m, values of T) into the convolution's slot and
keeps x₀ and the sum (taken pairwise) in two more rows; the convolution is the Bluestein
stage's chain with dft(b)/L as its kernel spectrum; *rader out* scatters x₀ + c_k to bin
g^m for m = −k. A unit is C copies when C divides s, else one copy in scalar arithmetic.
The convolution is about half a Bluestein stage's (257: 256 points against M ≥ 513), so
Rader took 0.4–0.8 of the time from 97 to 40961 on all three machines. The range
(`target_profile::rader_from`, `rader_below`): with C ≥ 4 a single small prime loses,
because Rader's gather and scatter of one row run in scalar code while the chirp passes
fill C lanes, so Rader starts at 16C there (64 for AVX2 float and AVX-512 double, 128 for
AVX-512 float); at 65537 it lost on the Mac (1.2: a megabyte gathered and scattered in
random order) and won on x86 (0.56–0.83), so NEON stops below 2¹⁶ and x86 below 2²⁴, where
the index table's floats stop being exact. Because the scatter permutes, the stage cannot write the slot its convolution ends
in: from one scratch slot to the other it writes back over its source (`last_stage_to`).

Strategies compose: a grid's row phase is a chain or a leaf; a plane transform is a lanes
program for the strided axis around a chain for the contiguous one.

---

## 9. Real transforms, any length, many dimensions

### 9.1 Real transforms

Baseline (even n): a complex transform of length n/2 plus a post-pass (r2c) or pre-pass
(c2r) that pairs bins k and n/2 − k. The pairing is **fused into the last pass** for r2c,
and c2r runs through the transposed (DIT) chain so its pre-processing fuses into a first
pass that has no twiddles. The prototype measured these at 15 % on Apple, and measured
that eligibility depends on radix and register count (fused with a last radix of 8 or 16
lost on 16 registers), so fusion is a per-plan choice informed by the generator's
live-value counts. The key performance indicator is the real overhead, time(r2c of 2M)
minus time(c2c of M), as a fraction of the complex time; the target is at most 20 %.

Odd n: full-length complex transform at first; real-input codelets (the generator's
pruned variants) are experiment X6, which also tests them for even n against the
half-length route.

As built (task 2.1, `src/passes/real.inl`, `chain_builder::r2c_line` and `c2r_line`,
experiment X6): an even line of stride 1 reads its reals in place as the h = n/2 complex
values z_j = x_2j + i x_2j+1, so no pass packs them. With v = −i w^k (w = e^(−2πi/n); stored as
(Im w^k, −Re w^k), an exact rearrangement of a root-table entry for every even n, where
w^(k + n/4) would need 4 to divide n), A = Z_k + conj Z_(h−k)
and B = Z_k − conj Z_(h−k), the r2c pairing is X_k = (A + vB)/2, X_(h−k) = conj(A − vB)/2,
and the c2r pairing Z_k = A + conj(v) B, Z_(h−k) = conj(A − conj(v) B) with A and B formed from
X: an addition, a subtraction and a product per pair; the halving carries the plan's scale.
Routes, by `real_route`:

- *Fused*: bin q + s k (copy q, leg k) of the r2c chain's last pass pairs with bin
  (s − q) + s (R − 1 − k), so the fused last pass runs copies q and s − q together (lanes up
  and down), and the c2r chain's first pass likewise groups p and m − p, forming each Z
  pair once on loading; copies (groups) 0 and s/2 are one unit with one element per lane.
  The transform needs no DIT chain: the decimation-in-frequency chain's first pass takes
  the c2r pairing as it is. Fusion pays only where the pass stays in registers and in L1
  (X6's table): the rule (`fusion_of`, core/plan_core.cpp) fuses into a codelet pass of
  radix at most 8 (4 on avx2, unmeasured), with at least 4 registers of copies per mirror
  half, and for r2c only while the half spectrum fits L1; the chain's radices are
  reordered to bring such a radix to that end. This is the "eligibility by live-value
  count" of the plan, in its simplest form: two butterflies of radix R hold 2R registers.
- *Leaf*: h ≤ 16, one codelet call with the pairing in registers.
- *Separate*: a pairing pass over the h + 1 values of the complex side, in place: after a
  c2c line written into the output (r2c) or before a c2c line run in place there (c2r,
  which keeps the input intact). Used where the edge pass is a Bluestein stage or a
  generic pass, where the rule declines, after an r2c grid that does not pair in its last
  phase, and before a c2r grid in place or in three phases.
- *Grid*: a real row of one axis whose half-length transform would be a grid as a c2c row
  (section 8.5) runs the grid on the reals read as complex values, through the grid
  builder. Where the L2 is shared (Apple; `c2r_paired_gather`), a c2r row out of place in
  two phases pairs its bins in phase 1's gather
  (`c2r_pair_gather`): each Z_j is formed from X_j and X_(h−j) of the input, which is only
  read, so the grid runs from the input to the output as a c2c row does, without the
  separate pass and the workspace block (X29: 0.92–0.94 of the separate route's time on
  the Mac; on the EPYC it took 1.11, so x86 keeps the pass). In place or in three phases
  the block stays, and the gather's second read of the input lost to the separate pass
  (1.02–1.06), so those rows keep it (c2r rereads the
  output slot for the input: `read_output_for_input`). An r2c row pairs its bins in the
  grid's last phase (X33; `r2c_pair_gather`, `r2c_pair_scatter`, `r2c_pair_column`): bin
  K = l + L k pairs with h − K = (L − l) + L (Nk − 1 − k), so the phase runs over pairs of
  column sets, the low set l = c·hb + 1 … c·hb + hb and the set of its mirrors (aligned),
  hb = block / 2; both are gathered into one block, transformed, paired in registers and
  written; column 0, whose bins pair among themselves, has a gather, chain and pairing of
  its own. Only where each set is at least a register and a cache line, the last phase is
  not fused (radices ≤ 8 read the arrays directly and keep the separate pass), and not on
  `avx2` (the EPYC gained nothing; build_settings::r2c_pair_last forces it): Mac r2c rows
  0.94–0.98 of the separate route's time, Xeon `avx512` 262144 and 2²⁰ 0.94–0.97.
- *Strided*: a line with a stride other than 1 (allowed out of place) packs its reals into
  scratch and pairs at any stride, one element at a time, as before task 2.1.
- *Full*: odd n through the full-length complex chain, as before (X6).

Measured on the Mac (X6, fbench run 20261114-135716-0600): real overhead 2–16 % for rows
beyond L1 (grid plus separate pairing) and 15–43 % within it (r2c 39–43 %, c2r 15–28 %);
the 20 % target within L1 needs X6's real-input codelets.

c2r preserves its input by default; with permission to destroy it the planner may take a
faster route. A multi-dimensional c2r that preserves its input needs scratch for the
half-spectrum of one plane (or pencil block) per thread, because the real output cannot
hold it; this is reported by `workspace_bytes`. In-place r2c and c2r use FFTW's padded
layout: each row of the last axis holds n + 2 reals (n/2 + 1 complex), the last two
being padding on the real side.

### 9.2 Any length

| Factor | Route |
|---|---|
| 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20, 25, 32, 64 | generated codelets (20 and 25 pass radices only, never leaves: experiments X38, X42; 25 has passes on the x86 and scalar targets only, where `avx2` and `avx512` double overrides take it, and none on NEON, where it lost) |
| 17, 19, 23 | generated codelets (symmetric-pair form), the leaf of n = p and a pass after the priced radices in composites (experiment X27: 0.15–0.42 of the Bluestein stage's time on the Mac) |
| other primes above 16 up to the profile's threshold | one generic looped pass, O(p²) per butterfly, vectorised across copies, in direct form with pairwise sums; the threshold is 16 on every profile (X17: Bluestein was faster at every prime from 17 up on the Mac, the Xeon and the EPYC), so the rules do not choose it now |
| a single prime with q − 1 13-smooth, in the profile's range (16C ≤ q for C ≥ 4; q < 2¹⁶ on NEON, 2²⁴ on x86) | Rader: a gather by powers of a primitive root, a cyclic convolution of q − 1 points with a precomputed kernel spectrum, a scatter (section 8.5; X27) |
| other large primes | Bluestein: the smooth part of N is split off first; the product of the large prime factors becomes one stage at the end of the chain that performs a chirp-z transform through a smooth-length convolution with a precomputed kernel spectrum (section 8.5) |

Rader and Bluestein share the convolution (`add_convolution`). The cost model prices
these routes.
`next_fast_length` lets applications choose grid sizes that are fast on FFTN rather than
on another library; it is a pure function of the library version (a fixed reference cost
table), so a grid chosen on one machine is the same on another. A separate query returns
the machine-specific estimate for callers who want it.

*As built (task 2.5, `fftn/lengths.hpp`, `src/planner/rules.cpp`).*
- **The cost of a row** (`row_cost`) is the planner's own estimate:
  - the cost table's cheapest sequence for the length's 13-smooth part, leaves included, the
    overrides left out because they use radices the table does not price;
  - 17, 19 and 23 codelet passes, priced as radix-16 passes scaled by r log₂ r;
  - a Rader or Bluestein stage, priced as its convolution's two transforms and two
    elementwise passes per copy.
- **`relative_cost<T>(n, isa)`** is that cost per point over the cost per point of the power
  of two at or above n, on the target's own table. It is 1 for a length the planner runs as
  one codelet (`rule_radices(n) == {n}`; 20 is always a chain, X38, and is priced as one).
  It does not price memory beyond L1, nor long rows' chain or grid (P3).
- **A fast length** is one whose relative cost on the reference table is at most 1.25. The
  reference is a frozen copy of AVX2's double table (X2's measured one), kept apart from
  the planner's tuned table so that a refit changes plans, never the fast lengths. No
  length with a prime factor above 23 is fast: such lengths are priced at 1.49 or more up
  to 20000. Nor is a composite with a factor 11, 17, 19 or 23: P4 measured them slower than
  the table believes, on two machines each.
- "Fast" is a property of the table, not a measured guarantee: in P4, 78 % of the picks ran
  within 1.3 of the power of two above on each x86 machine, and 140 (the pick for 131) ran
  at 1.76 on the Xeon.
- **`next_fast_length(n, kind)`** returns the smallest fast length at or above n, or with
  `length_kind::real` the smallest even one whose half is fast (2 for n ≤ 2). In 2-D and
  3-D only the halved axis takes `length_kind::real`. It tests only the candidates in [n, 2n] (products of 2, 3, 5,
  7 and 13, and the lone 11 to 23), so one query takes 5–200 µs on average up to 2⁴⁰ on
  the M5 Max, under 0.6 ms at worst. The largest step up to 200000 is 17 % (41 → 48).
- Both queries accept n ≤ 2⁴⁰ and throw `std::invalid_argument` above it: `row_cost` is
  exact up to there, and a prime near 2⁶² took more than 30 s to price.
- The fast lengths are pinned by tests (a list to 300 and a checksum to 2²⁰), since a change
  to them changes every caller's grid.
- Experiment P4 measured whether they are fast on the three machines.

### 9.3 Many dimensions

A multi-dimensional transform is a sequence of 1-D transforms, one per axis. What matters
is the order in which memory is touched:

- **Contiguous axis** (stride 1): rows through a chain or grid.
- **Strided axis**: the lanes strategy over neighbouring lines. No transposes, no gather
  or scatter instructions.
- **Plane-at-a-time.** For batched 2-D transforms each plane is finished (both axes)
  while it is in cache, and whole planes go to threads; the prototype measured this at
  1.4–1.9× over axis-at-a-time sweeps. When a plane does not fit in L2 it is blocked along
  the strided axis.
- **Volumes.** The two inner axes are done plane-at-a-time; the outer axis runs over
  pencil blocks (lanes strategy over blocks of neighbouring pencils sized to L2). Scratch
  is per thread and of plane or pencil-block size, never a second volume.
- **In place** is supported for every shape. The chain's last pass writes back to the
  caller's array from scratch; the grid's in-place routes are listed in section 8.5.
  In-place planes and volumes need per-thread scratch of row, plane or pencil-block size,
  never of the whole array.
- **Real multi-dimensional transforms** halve the last axis, as FFTW does.

Axis order within a plan is a planner choice unless a pipeline pins it (section 12.3). As
built (task 4.3): whatever order the shape lists them in, the axes run widest output stride
first as a C-ordered array lists them, so the contiguous axis is the one that runs as rows
(`order_axes`, `plan_core::axis_of`); a real transform's halved axis stays last. Two listings of
the same transform are the same program. Axes of extent 1 never step and sort first.

As built (task 4.1, experiment X40; `src/core/plan_core.cpp`, `column_block`,
`column_radices`, `add_column_phase`): a transform is finished before the next, axis by
axis from the last. The last axis runs line by line through the chain (or a leaf, a loop of
which is one leaf call, 8.3). In a c2c plan every other axis whose neighbouring columns are
adjacent on both sides of its phase (the last axis contiguous there) runs as the lanes
strategy (8.5) over blocks of B adjacent columns: the block's first pass reads its columns
from the array (C columns are one vector, so there is neither a gather nor a transpose), the
column's chain runs on a tile in scratch, and the last pass writes them back, in place or
not; the first and last passes (and the scatter after a split run) take every array row's B
columns at once (`pass_args::columns_inner`), so each line is read once. Each column is a
lane computing its own chain, so the columns get the bits of the chain route where the
radices agree. The radices are the row's unless its last pass would put more of its legs
into one L1 set than the L1 has ways (legs s = n / R column elements apart): then the
cheapest ranked sequence whose last pass does not (a last pass of radix 16 whose legs shared
a set made 128² and 256² planes 2.2 times slower on the M5 Max; reads in the first pass did
not matter). A leaf length (one pass of radix n), which the ranking cannot replace, whose
legs crowd a set runs as the most balanced two lane passes whose last pass fits (16 = 4 × 4;
X52: the M5 Max's 16 × 256 × 256, legs 1 MiB apart, took 0.55 of the leaf's time). In double a column of 17–32 points whose length has a codelet is that one lane
pass where its legs fit (X44: on x86 the row of 32 is a pair leaf, which the lanes cannot
take, and its radices as two lane passes made the EPYC's 32² column phase 1.9 times its
rows; one pass took the plane to 0.72 of that time, at FFTW's). B is 256 bytes of columns (16 complex doubles), at most the columns there are;
where a column's elements lie a page or more apart, at least four L1 lines (32 complex
doubles on Apple's 128-byte lines); where they lie less than a page apart it is halved while
the two tiles exceed `column_tile_bytes` (a quarter of L2; twice L1 where a cluster shares
L2 and a column's elements lie at least four to a page, X41: the M5 Max's 2048 × 36 0.89–0.91
of the time of 16 columns with 4, while 1000², nearly a page per element, lost 1.03), down to 64
bytes of columns (an x86 line, half an Apple line), and the columns run alone through the
chain if they still exceed it (X40's sweeps: wide blocks won where each element of a column
has its own page, 2048² on the EPYC 0.126 of the chain columns' time with 16 columns against
0.183 with 4, the M5 Max's 1024² 0.46 with 32 against 0.50 with 16; where rows share pages a
block whose tiles left L2 lost, Xeon 2048 × 36 0.79 with 4 columns and 1.09 with 16). On
the profile of a private L2 with a large L3 behind it (the EPYC 7763), the one-thread
program's c2c blocks then widen to 1 KiB of columns where both tiles fit `column_tile_bytes`
(`column_wide_bytes`; X52 arm W4b: 0.933 of the time over the 38 planes and volumes it
changes, 64 × 64 × 512 0.87–0.88), while teams keep these blocks in a team program (the Xeon's
8 lanes lost up to 1.46 with the wide ones, and its one thread 1.05 at 30 × 40 × 50). The columns past the last whole block run as one
narrower block (a loop's `start` offsets its nest), its idle lanes never written. A block is
a line of the phase's loop nest, so threads share blocks as they share rows (13.2); the
narrower block's nest is a phase of its own, of one chunk (joining it to the whole blocks'
phase gained 3–4 % at 8 lanes, X41, below the bar). The complex axes of r2c and c2r plans take the same phase with their
own block rule (`column_block`, `real`; X41): their rows of n/2 + 1 columns lie an odd
multiple of 16 bytes apart for even n, so a column alone crowds no L1 set and the lanes pay
only for the copies a strided column costs where C > 1: none on NEON double (the M5 Max lost
1.04 over the real planes with X40's rule), and on x86 blocks of at least two lines whose
tiles fit `column_tile_bytes`, X40's wide block only where each column element has its own
page and the column exceeds half of L1, the chain otherwise (EPYC: 0.89 over the 71 real
planes, 32 × 30 0.74, 36 × 2048 0.77, 2048² 0.73–0.85; 1024² stays on the chain, whose
4-column tiles lost 1.05). Not
built yet: blocks over the columns of a long axis beyond L2
(a two-phase split of the column, the grid's way), and padding or block ordering against the
rows' own power-of-two conflicts.

Volumes as built (task 4.2, experiment X43b; `add_volume_planes`): one thread runs a c2c
volume axis by axis over the whole volume, as X40 built it; a team of two or more, where the
profile's `team_volume_planes` says (x86-64 with a private L2 below 2 MiB: the EPYC 7763 and
the Xeon 6240), runs a team program (13.2) with the two inner axes plane at a time: a loop
over the n0 planes whose body is the plane's rows and its column phase, then the outer axis
over blocks of adjacent pencils (the same lanes column phase, `column_block`'s rule, the
pencils' elements a plane apart). At 8 lanes that took 0.775 (EPYC) and 0.771 (Xeon) of the
axes one by one: whole planes per member and one phase boundary fewer. One thread gained too
little to take it: EPYC 0.979 (its 32 MiB L3 per CCX holds volumes to 128³, so a third sweep
is cheap), Xeon 0.934 but 97³ 1.048, M5 Max 0.962. Only volumes of at least 8 planes take
plane order in a team: the plane phase has one line per plane, so 2 × 1024 × 1024 or
4 × 512 × 512 keep the axes one by one, whose phases share out by lines (teams of more
members than planes still idle some, a known limit). Either order computes every line alike,
so the bits do not depend on it. Scratch per member is two lines of at most a plane or a tile
of pencils (tests/core/volumes_test.cpp), never a second volume.

Strided axes over adjacent transforms (task 4.3, experiment X60; `adjacent` in `build`): when
a dimension that is not transformed varies fastest (the columns of a matrix, the planes (y, x)
of a (y, x, w) array, a middle axis), every transformed axis is strided and the innermost batch
dimension's transforms lie one element apart on both sides. Such a c2c plan is a lanes program
(8.3, `program::lanes`) whose body holds one phase per axis, each a loop over the lines of the
other axes around a lanes pass with one transform per lane, so a run of C transforms is one
vector access and every line of memory is touched once; the block is the narrowest
`column_block` of the axes (the transforms in the role of the columns), the radices each axis's
`column_radices`, and transforms left over go through the tail in blocks of C. Each lane
computes its transform alone, so blocks and execution ranges do not change the bits. A block is
one task of the team (the work graph's one phase), so whole transforms are shared out, never
their lines. On x86-64 with a private L2 below 2 MiB (`machine_profile::adjacent_team_chain`,
measured on the EPYC and the Xeon) a team runs the chain instead, which shares lines, and the
program's lanes take the chain's radices so that the bits agree: at 8 lanes the blocks lost up
to 1.74 (EPYC) and 2.4 (Xeon) of the chain's time on planes and volumes of few transforms. Within L1 (all the plan's transforms together, `machine_profile::
adjacent_min_bytes`) the chain keeps a plan with one complex value per register, and on wider
registers a plan whose axes are all of leaf lengths, 16 points or fewer: there a strided line
costs no more than a contiguous one and the chain's short passes win, while longer axes gain
from the lanes even within L1 on x86 (X60). Lengths without
lane passes (a Bluestein or Rader stage, generic passes) and the
real kinds keep one transform at a time through the chain: a strided halved axis has no lane
path yet.

Threaded column blocks (experiment X43): a team of two or more runs a team program
(`plan_core::team`) whose lanes column blocks hold at most `column_team_bytes` of columns
(128 bytes, 8 complex doubles against X40's 16, on x86-64 where a private L2 has a large L3
behind it, the EPYC 7763; 0 elsewhere: the Xeon 6240's 1 MiB L2 holds X40's blocks, and
narrower ones took 0.997 there, 512²–2048² 1.05–1.15), and whose columns
that one thread runs alone through the chain because a line of them exceeds the tile budget
(2048 × 36) run as blocks of one line with the chain's radices. At 8 lanes on the EPYC the
narrower blocks took 0.887 of the time over the c2c planes and volumes they change (1000²
0.66, 128² 0.75: tiles within a member's 512 KiB L2, and more blocks than members), the line
blocks 0.72 at 2048 × 36 (members no longer write neighbouring columns of one line), while
one thread lost up to 1.42 with the narrower blocks.

---

## 10. Memory-system rules

These are the rules the prototype paid for. Each is a default with a measurement behind
it; the planner applies them using detected cache geometry (size, line size and
associativity of L1, L2 and L3 per core class, page size). On Apple M-series the L1 is
128 KiB with 16 KiB pages and 128-byte lines, so the stride at which streams collide
differs from x86's 4 KiB; nothing below hard-codes either.

1. **Count streams against associativity.** A pass with R input streams and R output
   streams whose spacing is a multiple of the L1 set period conflicts once 2R exceeds the
   associativity. The planner checks this for every pass (FFTL checks inputs only) and
   answers with a smaller radix at that position, padding, or a buffered pass.
2. **Pad internal buffers** where rule 1 would otherwise bind: one block of padding per
   so many blocks breaks the power-of-two spacing. Padded addresses are computed
   incrementally (the prototype's first attempt, recomputing them per access, cost
   30–60 % on AVX2). Padding is applied only to buffers whose passes need it.
3. **Small radices where a pass touches caller memory.** The first and last passes read
   and write unpadded arrays; they get the smallest radices unless rule 1 is satisfied
   anyway.
4. **Stagger scratch buffers** by an odd number of cache lines so the ping-pong pair does
   not alias.
5. **Minimise scratch per thread.** Throughput workloads at many threads are bandwidth
   bound; the prototype's three buffers per thread cost 9–12 % at 64 threads. Prefer
   strategies that work in the output buffer when rows leave L2.
6. **Block by detected cache size, never by a constant.** Depth-first blocking helps only
   past L2 and hurts inside it (a fixed threshold lost 16–30 % on a machine with a larger
   L2).
7. **Whole accesses on whole lines.** With several complex values per register, a stride
   that makes every other access straddle a cache line costs more than a narrower pass
   (measured: +12 % for a masked AVX-512 pass). The planner prices line splits.
8. **Prefetch and non-temporal stores are per-machine choices**, off by default. Next-row
   software prefetch gained 25 % on a mesh-L3 Xeon for batches beyond the last-level
   cache and lost 3–6 % on Zen 3 and Apple. They are planner options backed by the
   machine profile, never unconditional.

---

## 11. Planner

### 11.1 Requirements

- Construction in microseconds to milliseconds by default. Applications create plans
  freely; nobody should need a wisdom subsystem to start quickly.
- Deterministic by default: the same shape, options, library version and CPU model give
  the same program.
- Never touches user buffers; no process-global mutable state.
- Catches the per-shape misses that fixed rules leave (finding E6) when asked to.

### 11.2 Efforts: rules and measurement

**`planning::rules`** (default). Enumerate candidate programs for the shape: strategy ×
factorisation and radix order × arithmetic layout × twiddle modes × blocking. Prune with
the memory rules. Rank with an additive **cost model**:

    cost(program) = Σ over stages  points × unit_cost(codelet, target, pass kind)
                                          × memory_factor(working set, streams, level)
                  + Σ over loops   iterations × overhead(loop kind)

`unit_cost` comes from a **machine profile**: a table of per-codelet, per-pass-kind costs
measured by microbenchmarks. Profiles for the known CPU families ship with the library
(generated by `fbench calibrate` on the benchmark machines); an unknown CPU uses the
nearest family by vendor, register width and cache geometry. `memory_factor` is a small
function of whether the working set fits L1, L2 or L3 and of the stream count. The best
candidate becomes the plan. Keys include the batch count, because the best program for
one row and for 256 rows differ (measured on the Xeon).

**Planner v1 (as built, experiment X2, `src/planner/rules.cpp`)** chooses the radices of a
chain and their order with the simplest cost model that reproduces X2's measurements: a
table of measured pass costs per profile and an exact search over the sequences. It
replaced planner v0 (task 1.2: FFTL.jl's single-line rules, T. Knopp, MIT licence, keyed on
the register count and on merged radices), which X2 found 1.25 times slower than the best
sequence in double and 1.44 times in single precision on AVX2 (geometric means over the
7-smooth lengths up to 1024). A profile is the precision's number of complex lanes C and a
cost table: for each pass kind and codelet radix R (2 to 16, and 32 and 64 where the table
prices them) a cost per unit and a cost per tail, and a cost per pass. For a line of n points:

1. n = 1: no pass. n ≤ 16, or n a codelet radix (32, 64) no larger than the profile's
   leaf limit (experiments X8 and X14: 32 on NEON and `x86_v2` in double, 16 elsewhere): one leaf of radix n.
2. A length the table overrides takes the table's radices (the lengths where rule 4 was
   measured more than 2 % slower than planner v0 or, since the planner refresh, more than
   2 % behind the measured winner in every case; 36 and 13 on AVX2, double's with radix
   20, 32 and 64 passes the costs cannot price: experiments X30 and X38). The rows of a
   one-axis c2c plan of contiguous rows of a length in the profile's `fit_overrides` (`avx2` double: 768 = 12 · 64,
   1024 = 16 · 64; X57) instead take that sequence while the plan's rows and output (rows alone
   in place) fit `machine_profile::fit_row_bytes`, three quarters of the L3 where a private L2
   has a large L3 behind it, and rule 4's sequence beyond; rows of planes and volumes keep
   this rule.
3. Otherwise write n = n_s q₁ q₂ … with n_s free of primes above 16 and q₁ ≤ q₂ ≤ … the
   prime factors above 16. Each qᵢ is one pass, in ascending order, after the codelet
   passes: a codelet pass for 17, 19 and 23 (experiment X27), otherwise a generic pass; if
   n_s = 1 they are the whole chain. (`rule_radices` returns this sequence.) Then the
   trailing qᵢ without a codelet above the profile's Bluestein threshold (16 on every
   profile, experiment X17, so all of them) are merged into one stage of their product Q
   at the end of the chain (section 8.5): a Rader stage when Q is a prime in the profile's
   Rader range with Q − 1 13-smooth (`rader_root`), otherwise a Bluestein stage, whose
   convolution length M is the
   7-smooth length from 2Q − 1 to the next power of two that minimises twice the cost of
   rule 4's sequence for M plus two elementwise passes of M points (priced as a last
   radix-2 pass per point), the smallest on a tie (`bluestein_length`), among those whose
   power-of-two part the machine profile admits (up to 2 (2Q − 1) when that excludes the
   power of two; experiment X36, section 8.5). A Bluestein stage with 1 < n / Q < C copies
   becomes one stage of n where M(n) ≤ (n / Q) M(Q) (X36), and so does one with
   n / Q = C copies while n's line is not long (X36b). A stage of one copy runs its
   convolution on two depth-first chains in double where a row of M would be a depth-first
   chain, M up to the profile's `depth_convolution_max_bytes` (X50, section 8.5).
4. The codelet passes are the sequence of radices from 2 to 16 (and 32 and 64 where the
   table's unit cost for them is not 0) multiplying to n_s with the
   lowest cost, the sum of its passes' costs. The pass of radix R after passes whose
   radices multiply to s has m = n/(R s) and:
   - kind *first* if s = 1, *last* if R s = n_s and no generic pass follows, *middle*
     otherwise (the chain's grouped pass, its lane pass with twiddles and its last lane
     pass, section 8);
   - units ⌈m/C⌉ for the first pass (it runs over m in blocks of C) and m⌈s/C⌉ for the
     others (over s in blocks of C, once per p < m);
   - tails, the units that are padded tiles: 1 for the first pass when C does not divide
     m, m for the others when C does not divide s, otherwise 0;
   - cost unit[kind][R] × units + tail[kind][R] × tails + pass.

   Among sequences of equal cost the lexicographically first wins (the smallest radix at
   the first pass where they differ). The costs are integers, so the choice is exact.

The tables come from X2's campaign on an EPYC 7763 (AVX2): one for C = 2 (double) and one
for C = 4 (float), fitted to the timings of every candidate sequence of every 7-smooth
length up to 1024 at batches 1 and 64 (the fit, the overrides and the reproduction rate are
in the experiment file). No 7-smooth length has 11 or 13, so their columns are radix 10's
and 12's scaled by the codelets' operation counts. What the tables learnt is in their
shape: tails cost several units, so the first radix wants to be a multiple of C and so
does every prefix product; and on 16 registers the radices 10 to 16 are cheap enough per
point that the best sequences have one pass fewer than v0's (2.7 against 3.6 on average up
to 1024). Tables are keyed by target and C (`measured_tables` in `rules.cpp`;
`bench/x2/fit.py` writes a target's rows from its X2 campaign), so a measured target is
added without touching the search. A target and C without a table of their own take
AVX2's table of the nearest C (the C = 2 table for C = 1, the C = 4 table for C = 8) with
their own C in rule 4, so the structure carries over but the costs are unmeasured there:
x86-64-v2, AVX and scalar until X2 runs on them. AVX-512 has its own tables since the
planner refresh (X2, after X1's split runs, X3, X18 and X19, on the Xeon 6240): in float (C =
8) costs fitted to searches over the 7-smooth lengths up to 1024, the multiples of 32 up to
2048 with radix 32 or 64, and the catalogue lengths 1200–32768, with radix 32 and 64
columns (X3) and overrides at every measured length whose choice was slow or never timed;
in double (C = 4) AVX2's C = 4 costs with overrides where their choice was more than 2 %
behind the measured winner in every case (mostly a first radix of 8, after which a split
run starts). Every other table leaves the radix 32 and 64 columns 0, so those passes stay
out of their chains: they tied or lost there (X3), except AVX2 double's overrides 256 =
8 · 32 and 512 = 16 · 32 (X30: two passes where the costs' choice ran three). The refresh kept the AVX2 and NEON
costs (refits gained about 1 %) and added overrides from new searches on the EPYC (up to
16384) and the Mac (up to 1024). NEON has its own tables (C = 1 and 2), fitted to X2's
campaign on an Apple M5 Max with `neon_fcma`; plain `neon` takes them too, unmeasured on a
core without FCMA. Above 1024 the costs extrapolate from the cost estimate's best 20
candidates per catalogue length.

The search is a shortest path over the divisors of n_s, from 1 to n_s, each step a radix:
at most a few thousand operations for lengths up to 2²⁴, so plans stay cheap and
deterministic. A
pure function of (n, profile); `tests/core/data/rule_radices.txt` records its output for
every 73-smooth n ≤ 20000 and C = 1, 2, 4 and 8, generated from this specification by an
implementation independent of the library's. `describe()` lists the radices and every
stage. Hooks for phase 2: the table is the first machine profile of the cost model above
(`unit_cost(codelet, target, pass kind)` with tails as a pass variant); calibration
replaces the fitted tables per machine, and the memory factor is still missing (every
measured length fits in L1). The chain builder accepts any sequence (the tests force
sequences through `build_settings::radices`); the memory rules of section 10 (set
conflicts) are not applied yet.

**`planning::measure`** (opt-in). Take the top candidates by the model, build each, time
whole programs on scratch data with warmed caches and several repetitions, and keep the
fastest. Whole plans are timed; no optimal-substructure assumption. The result is recorded
as wisdom.

*As built (task 2.4, `src/core/candidates.cpp`, `src/core/measure.cpp`).* A candidate is a
few build settings (strategy, radices, grid factors or phase count, Rader on or off), so
every candidate is a program the rules' code builds and can be written down as text. In the
order they are timed: the rules' choice; for one axis whose complex line (the c2c row, or an
even real row's half length) exceeds L1, the chain with the rules' radices, the grid in two
phases and, beyond `grid_min_bytes`, in three, and for powers of two FFTL.jl's long-row
radices in radix 8 and radix 4 (the choices X12b, X29, X32, X33 and X34 make by thresholds);
for batches the lanes and the chain (X7); for every line length the next cheapest radix
sequences by the profile's pass costs (`ranked_radices`, the shortest-path search keeping the
k best continuations); and Rader off and on where a prime factor has no codelet (X27).
Candidates that build the same program as an earlier one (by the fingerprint of
`describe()`) are dropped. Candidates are built in the model's order and admitted while
half the time budget is left and their tables, the workspace and the timing data fit
`options::measure_scratch_bytes` (fewer transforms of the outer batch dimensions when the
data would not); when only the rules' choice is admitted nothing is timed. The rest are
timed at the shape's size in interleaved rounds: a warm-up execution sets each one's
repetitions per sample, each round starts one candidate later and places the input and
output at other 64-byte offsets within a page (as fbench does), the statistic is each
candidate's median sample, and after the first rounds a candidate 25 % behind the fastest
is dropped and its tables freed. In place, the input is restored before each sample and the
repetitions are capped so that values cannot overflow. The budgets are targets: a large
transform overruns the time by about one plan's construction and a few executions per
candidate (P2), and the rules' plan and one transform's data are always allocated. The rules' choice stays
unless another candidate is at least 1 % faster; otherwise the fastest wins, the earlier on
a tie. `planning::quick` times at most 4 candidates with shorter samples (0.1 s budget),
`planning::measure` at most 16 (10 s); P1 and P2 set these. Measurement is serial, on the
calling thread, and times nothing combinatorial: the candidate list is linear in the
alternatives above.

The budget is generous by design (the maintainer, 2026-11-16): a transform that is cheap to
compute is either planned with `planning::rules` or with a quick measure, while a hard one
runs for minutes, hours or days, so longer planning with a bigger payoff is worth it. A
useful check is not ruled out to save planning time or memory, within reason: candidates
are timed at the real size where scratch of the user's data size is acceptable (a reduced
batch only for large batches and volumes, bounded by a stated scratch limit), and the time
spent per shape may run to seconds, of the order of FFTW's MEASURE (which, with PATIENT,
is what the maintainer's single-thread codes have used). What it must not become is
FFTW's PATIENT or EXHAUSTIVE: no combinatorial search over every factorisation, above all
not under threads. Effort levels (a quick measure, the default measure) and their budgets
are set by experiment P2; faster planning that finds the same plan is always better.

### 11.3 Wisdom and reproducibility

Wisdom is a small text record: shape key, CPU model, library version, program
description. It can be exported and imported as a string or file and is an input to plan
construction (`options::use`), never global state.

*As built (task 2.4, `src/core/wisdom.cpp`).* One tab-separated line per plan: a format
tag, the build (the library version and a build identifier, a hash of the library's sources
and of the configuration that changes its code, `cmake/FftnBuildId.cmake`, since the version
does not change between development commits), the processor model (Linux `/proc/cpuinfo`,
macOS `machdep.cpu.brand_string`, the Windows registry), the instruction set, the shape key
(kind, precision, direction, in place, the scale written exactly, each axis's length and
strides and each batch dimension's count and distances as given), the program (the
candidate's text in canonical form, naming only what took effect, a grid's factors written
out) and a check, the fingerprint of the program's `describe()`. An entry is found without
building anything (an index by those fields) and applies only if its program builds, rebuilds
to the same fingerprint and is named the same way; otherwise it is ignored (`describe()` says
so) and the effort decides. Lines of other formats or builds are kept verbatim and exported
again. A program description is also a valid
plan input, so a result can be pinned and reproduced exactly on the same instruction set.
Measured planning is the only source of run-to-run variation in program choice, and it is
opt-in.

### 11.4 What the planner deliberately lacks

No flag lattice, no exhaustive recursive search, no timing of sub-plans in isolation, no
alignment in the problem key, no planning modes beyond the efforts above: the rules, and
measurement at two budgets (`quick`, `measure`) that differ only in how many candidates
they time and for how long. If the model is poor on some machine, the remedy is a better profile, not more
planner.

---

## 12. Fusion and pipelines

Fusion has three tiers, from cheapest to most general. All three keep data in cache
between the pointwise work and the transform; they differ in who writes the pointwise
code.

### 12.1 Tier A: edge operations

Descriptor-driven pointwise work on a plan's input or output edge, compiled into the edge
variants of the first and last passes (section 8.2):

| Operation | Meaning |
|---|---|
| `scale` | multiply by a real constant (normalisation costs nothing) |
| `window` | multiply element i of every row by a real factor w[i] |
| `factor` | multiply element i by a complex factor f[i], per row or shared |
| `valid` | on input: only a sub-range is non-zero, the rest is not read; on output: only a sub-range is wanted, the rest is not written |
| `embed` | input or output elements sit at given positions of a longer logical row (crop maps, zero-filled gaps) |
| `accumulate` | out += result instead of out = result |

The plan records which operations are present (that selects kernel variants); the arrays
themselves are passed at execute time, so they may change every call. Task 1.2 built
`scale` (public: `options::scale`) and `factor` (internal, test-only, its arrays fixed at
construction until `exec_args` arrives) as edge variants of every pass kind. Pruned edges use the
generator's pruned codelets, confined to the edge passes.

### 12.2 Tier B: transform pairs

A **pipeline** owns two or more transforms and the work between them, and is free to
choose everything in between:

- **`convolution`**: forward, multiply by a kernel spectrum, inverse; real or complex;
  with zero-padded input (`valid` range), partial output, several kernels selected per
  row, and one forward feeding several inverses. Fully fused, the spectrum never reaches
  memory: the forward transform's last pass, the multiply and the inverse's first pass
  form one sweep.
- **`spectral_pair`**: inverse, pointwise operation, forward (or the reverse), as in a
  split-step method. The spectral multiplier is given in natural order by default.
- **Scrambled order.** Because a pipeline owns both transforms, the intermediate order is
  its own business. With in-place strategies that removes two permutation passes; in the
  blocked grid it removes the forward's last transpose and the inverse's first. A
  `spectral_layout` object exposes the order to callers who store their multipliers
  permuted. This is experiment X9; the estimate from prior art (MKL's scrambled mode,
  djbfft, PFFFT) is 10–25 % for L2-resident sizes and more beyond.

### 12.3 Tier C: in-cache callbacks

For work the descriptors cannot express, a pipeline calls a user function on blocks of
data while they are cache resident:

```cpp
// called for each block of each row (or pencil block), between the transforms
using block_fn = void (*)(void* user, block_info where, std::complex<T>* data);
```

`block_info` gives the row, the element range, and the worker slot (for per-thread user
scratch). The callback is an ordinary function compiled by the user's compiler with the
user's flags, working on natural-order data in L1 or L2. This is deliberately **not** a
per-element functor inlined into kernels: user code cannot be compiled into ISA regions
soundly, and block granularity captures the memory saving, which is where the gain is.

Multi-dimensional pipelines pin the axis order so the fused axis is the last one
transformed backward and the first transformed forward: for a volume, "inverse on the two
plane axes; then per pencil block: inverse along the pencil, callback, forward along the
pencil; then forward on the plane axes" is three sweeps of memory instead of eight.

Prologue and epilogue callbacks (fill a block of the input; consume a block of the output
without storing it, for reductions) use the same mechanism on the outer edges.

### 12.4 Owner-computes

With `completion::own` and `plan::transforms_of(team)`, an application's own pointwise
kernels can use the same transform-to-thread assignment as the transform, so rows stay in
one core's cache across the application's work and FFTN's, inside one parallel region. In
this mode transforms are dealt statically and whole (member r does exactly
`transforms_of(r)`: contiguous ranges in rank order differing by at most one transform, or,
with fewer transforms than members, one each for the first members; nothing is claimed by
others or split), which requires that every member of the team takes part.

---

## 13. Threading and execution

### 13.1 Kinds of parallel work

| Kind | Example | How it is shared |
|---|---|---|
| Many independent rows | 384 points × 7000 rows | whole rows per thread; no synchronisation inside a row |
| Few long rows | one row of 65536 | a row's independent chunks are shared by a small team in one cache group |
| Planes and volumes | 384 planes of 192² | whole planes per thread; pencil blocks per thread |
| Few rows on many threads | 4 rows on 64 cores | rows dealt to cache groups; long rows split within their group |

### 13.2 The plan fixes the work; execution decides who does it

A plan's program already contains its independent loops and each stage's chunk count
(section 8.3). From them the plan derives a **work graph**: row tasks, each either atomic
(a whole row or plane) or a sequence of phases (stages, or groups of stages inside a
blocked loop), each phase a fixed number of chunks whose boundaries are aligned to the
kernel's blocks. The chunking depends on the shape, the target and the plan's options
(including the `max_lanes` hint), never on the team that turns up. Every chunk runs the
same code whoever runs it.

As built (`src/threading/work_graph.hpp`), the tasks are the transforms of the batch. A
task's phases are its body's records in order: a stage (the passes of a 1-D line) is a
phase of the stage's own chunks, which share the task's scratch lines; a loop nest over the
lines of one axis (2-D and 3-D) is a phase whose chunks are runs of whole lines, each line
on the scratch of the member that runs it, the block (9.1) shared; a volume's loop over its
planes (9.3) is a phase whose lines are whole planes, each running the plane's own phases
on its member's scratch. A team of two or more members may run a *team program* instead of
the program (`plan_core::team`, experiments X43 and X43b): the same transform with narrower
lanes column blocks, columns that one thread runs alone through the chain run as blocks of
one line with the chain's radices, and a volume of at least 8 planes with its inner axes
plane at a time (9.3). Each column computes alike in any block and each line alike in either
order, so the team program computes every element as the program does; a team's workspace
covers both, since an executor that dispatches one member runs the program. Therefore:

> For a given plan the output is bitwise identical for serial execution, any team size,
> any executor and any timing.

The width hint may change the program (a long row planned for 16 lanes may use a grid
where a serial plan would use a chain), so results depend on the plan, not on its
execution. `options::strict_layout` makes the program ignore the hint for callers who
want plans of every width to agree.

### 13.3 The engine: claiming, not waiting

Each thread that enters an execution is a participant. It finds its cache group, then
claims work: whole-row tasks first, then chunks of the current phase of a split row in its
group. Phase order within a row is enforced by per-phase completion counters; a
participant with nothing to claim waits on a counter (spin for a calibrated budget, then
yield, then park on `std::atomic::wait`), never on a particular other thread.

Why this form rather than barriers between passes:

- **It is correct on any runtime.** One thread alone completes everything; latecomers
  find nothing left. OpenMP regions and dedicated pools guarantee that all threads run;
  Kokkos `RangePolicy`, TBB, C++26 `bulk` and oversubscribed machines do not, and a spin
  barrier there deadlocks or burns.
- **Nobody waits for the slowest thread to wake** (10–50 µs at 64 lanes, measured).
- **Cache-group teams form by themselves** when threads are unpinned: a participant looks
  up the group of the CPU it is on at entry.

Work is queued per cache group. A participant whose group has nothing left may take
**whole rows** from another group's queue, never chunks of a split row: a row is only
ever split among threads of one group. If unpinned threads all happen to sit in one
group, the other groups' rows are therefore still done, as whole rows.

As built (`src/threading/engine.hpp`, task 3.1):

- **Claiming.** A member claims its group's whole tasks in runs (a share of what is left,
  shrinking to single tasks at the end), then joins its group's split tasks (at most `cap`
  members each, members starting on different tasks) and claims their chunks in order,
  in runs within one phase (a share of the phase's chunks left, at most left / (2 × cap),
  shrinking to single chunks; X52); a chunk of phase p waits until the task's done counter shows phase p − 1
  complete. Those chunks were all claimed before it, by members that are running, so no
  wait depends on a member that has not arrived. The member that claims a task's first
  chunk owns it: the task's shared scratch is the owner's slot, and the owner leaves only
  when the task is done. A member whose group is exhausted takes other groups' whole tasks,
  and runs whole the split tasks of groups none of whose members has arrived (so a group
  that never turns up cannot strand them); then it waits for the execution's done count.
- **State.** Counters live in the workspace (5.3), every hot word on its own 128-byte line.
  Executions are numbered by each member's own 64-bit count of its calls on the workspace;
  every counter carries the low 40 bits of the number of the execution that last wrote it
  above a 24-bit count, so it reads as zero in another one (nothing is reset between
  executions; tags are compared only for equality). A separate untagged word holds the
  highest execution a member has returned from; a member that meets another execution's
  tag reads it, and finds there whether its own execution is over. That is how a member
  may arrive after the others have moved on to later executions without a barrier
  anywhere. Batches too large for 24-bit counts are claimed in units of several
  transforms.
- **Waiting.** Spin for a time budget, then yield, then block in `std::atomic::wait`; a
  notifier calls `notify_all` only when a waiter has registered as blocked. An
  oversubscribed team (more members than CPUs) does not spin.
- **`completion::own`** uses none of this: a static deal, whole transforms (12.4).

The prototype's precomputed schedule with spin barriers is kept as a candidate fast path
for the case where every lane is known present and pinned. Experiment T2 decides whether
it earns its place; both forms run the same chunks and give the same bits.

### 13.4 Scheduling rules

The scheduler is a pure function from (work graph, lane map, machine profile) to a
schedule, tested by simulation. Starting rules, each with its measurement:

1. Run inline when the whole job is smaller than a few times the dispatch cost.
2. Whole rows before split rows. With at least as many rows as lanes, rows go whole to
   lanes; leftover rows (9 long rows on 8 lanes) are shared by sub-teams when they are
   long enough to split, so the last row does not cost a whole extra round.
3. Split a row only inside one cache group, with at least about 1024 points per member,
   and only from about 6000 points in a small group, 8000 on an Apple performance
   cluster or a large mesh-connected one (experiment T5). A real plane or volume counts its
   complex points where each core has its own L2, and on a mesh-connected L3 a plane or
   volume is split only from 32768 points (X52 arm S: its phases move it between the members'
   caches).
4. Deal rows to cache groups; only lanes that have work do any. Under OpenMP that means
   surplus team members return at once from a region of constant width, not a narrower
   region: libgomp ends and restarts threads when the width changes.
5. By default use at most one cache group for a batch smaller than the group, and only the
   fastest core class on hybrid processors.

Thresholds come from the machine profile (dispatch cost, barrier cost, per-group scaling),
not from constants in code.

As built (`src/threading/schedule.hpp`, task 3.2): the scheduler runs at the start of every
team execution, in every member, and returns plain data (no allocation): whether the first
member to arrive takes everything (rule 1), how many groups are dealt tasks (as many as
the batch has a group's worth of tasks for, so rule 5's small batch stays in one group),
each group's contiguous range in proportion to its members, and per group which tasks are
split and among how many members (rules 2 and 3: leftovers after whole rounds, or all of a
group's tasks when it has fewer than members; at most members / split, points / 1024 and
the widest phase's chunks; only from the split threshold of the group's kind; only when the
cost model says t/k plus a phase synchronisation per phase beats t). An oversubscribed team
splits nothing. The machine profile (`profile_for`, from the topology) carries the values
experiments T1 and T5 measured (task 3.4; each field cites its runs): the split thresholds
by group kind, a Zen 3 CCX's, an Apple cluster's (macOS's topology) and a mesh's (groups
of more than 8 members), and on a CCX the phase hand-off cost and the claim runs T1 measured.
Its dispatch and work costs stay provisional until the dispatch cost has a per-lane term
(18.2). Rule 4 under an executor (task 3.3): only the lanes dealt
work are dispatched, a team of `dispatch` members (the dealt lanes, or one per task when
the batch is smaller and nothing splits), which get the same groups and no splits (the
per-group shares can differ, one member per task); an OpenMP executor still opens a region of
its constant width, whose surplus threads return at once. In the caller's team surplus
members help or wait for completion. Rule 5's fastest-class restriction needs a core class per member,
which `team` does not carry yet (18.2).

### 13.5 Topology

`fftn::topology` is explicit data: CPUs, cores (SMT siblings), L2 and L3 sharing groups,
NUMA nodes, core class, cache geometry per class, the allowed-CPU mask and any cgroup
quota. Sources: Linux sysfs (reading online CPUs, not the calling thread's affinity,
which an OpenMP runtime may already have narrowed), macOS `hw.perflevel*`, Windows
`GetLogicalProcessorInformationEx`, and an environment or programmatic override for
tests. A `lane_map` records which lanes sit in which groups and whether that is verified,
assumed or looked up per call. FFTN never pins or re-pins threads it does not own.

The readers are plain parsers: the Linux one reads through a small file interface (the
live root, or a sysfs snapshot: one text file per machine, format in
`src/platform/file_source.hpp`) and the macOS one a sysctl snapshot, so both run on every
machine against fixtures recorded with `tools/capture_platform.py` (named by CPU model) and
synthetic ones. The environment override `FFTN_TOPOLOGY` names such a recording (a sysfs
snapshot, told apart by its header line, or a sysctl file); the programmatic override is that everything consuming a topology takes it as an argument.
The process's topology is detected once, into a function-local constant. Unknown values
are zero, never guessed: macOS reports no associativity, and it does not say which logical
CPU belongs to which performance level, so there CPUs are numbered in class order (fastest
first) and the topology says the numbers are not the OS's.

### 13.6 Integration models

| Caller | How | Notes |
|---|---|---|
| Serial code | `plan.execute(in, out)` | no threads anywhere |
| Inside `#pragma omp parallel` | every thread calls `execute(in, out, fftn::omp::this_team(), ws)` | no fork/join; pointwise work and the transform share one region |
| OpenMP, FFTN opens the region | `execute(in, out, fftn::omp::executor{}, ws)` | header-only: compiled against the caller's runtime; inline when one lane has all the work |
| Kokkos | `fftn::kokkos::executor(exec_space_instance)`, View-based plans and `view_plan`; `team_of(member)` inside a `TeamPolicy` kernel | one `parallel_for` per transform on the caller's instance; correct on Serial, OpenMP and Threads back ends; serial when called inside a kernel |
| Custom pool | any type with `concurrency()` and `bulk(n, fn, arg)` (optionally `group_size()`), or a C pool's `fftn_executor` | the only requirement is "run n callables and return when done" |
| No runtime at hand | `fftn::thread_pool` | explicit, opt-in, `std::thread` based, grouped by cache topology; never created implicitly |

Rules that hold everywhere: FFTN creates no threads unless the caller constructs a
`thread_pool`; it never changes a runtime's settings (thread counts, wait policy,
binding); a call from inside a parallel region or kernel runs serially rather than
nesting; width 1 opens no region.

As built (task 3.3):

- **The executor overload** (`src/threading/executor_execute.cpp`) asks the scheduler once
  with the executor's `concurrency()` and `group_size()`. A job under rule 1's threshold
  (decided once at plan construction, `plan_core::runs_inline`, so the executor is not
  asked anything), width 1, or one lane dispatched: `execute_core` on the caller, no
  `bulk`. Otherwise one `bulk(dispatch, …)`, task i running the team overload as rank i of
  a team of `dispatch` members with `completion::all`. The members judge oversubscription by
  the executor's width, not their own number (`lane_map::threads`): its other threads
  compete for the CPUs too. Nesting is the executor's business:
  executors report `concurrency()` 1 from inside their own tasks or region, so a nested call
  takes the serial path.
- **`fftn::omp::this_team()`** is `{omp_get_thread_num(), omp_get_num_threads(),
  cache_group_size(size)}`; outside any region a team of one. Callers must make the call
  from the region's body, where every thread runs (not `single`, `masked`, a task or a
  worksharing loop); OpenMP offers no query to check it. In a nested region it is the inner
  team.
- **`fftn::omp::executor`** has a width and a grouping fixed at construction
  (`omp_get_max_threads()` by default). `bulk` opens `#pragma omp parallel num_threads(width)` every time; thread t runs
  tasks t, t + w, … (w the threads the region actually got), so surplus threads return at
  once and a runtime that gives fewer threads still runs every task. `concurrency()` is 1
  when `omp_in_parallel()`, and `bulk` then runs its tasks on the caller.
- **`fftn::thread_pool`** (`src/threading/thread_pool.cpp`) starts `threads - 1`
  `std::thread`s; the caller of `bulk` is the last lane. A job is published by storing its
  task, argument and size, then the unclaimed count (release), then bumping an epoch that
  idle threads wait on; a thread claims by decrementing the count while positive and reads
  the job only after a successful claim, which proves the job is not over. A late thread
  therefore finds nothing, or claims a task of a later job as that job's member: no
  per-job handshake waits for idle threads. Waiting is the engine's (spin 20 µs, yield
  50 µs, then `std::atomic::wait`, notifying only registered sleepers). A thread-local flag
  marks pool tasks: from inside one, any pool's `concurrency()` is 1 and its `bulk` runs
  serially; a second thread's `bulk` while a job holds the pool also runs serially on its
  own thread. Threads are not bound. Default size: the process's allowed CPUs (and CPU
  quota). `bulk` neither allocates nor locks.
- **Grouping** (`cache_group_size`, `src/threading/cache_groups.cpp`): logical CPUs per
  group are the smallest group of the last level shared by more than one core (L3, else L2,
  as Apple's clusters); a team is assumed packed onto consecutive cores, one thread per
  core while it has no more members than the machine has cores, then every logical CPU;
  0 (one group) when the team fits in one group or nothing is known. The pool and both
  OpenMP adapters use it.

- **`fftn::kokkos::executor`** (task 3.5, provisional) holds the instance, its
  `concurrency()` and the grouping. `bulk` is one `Kokkos::parallel_for` of a static
  `RangePolicy` with chunks of one, then a fence of the instance: Kokkos lets a
  `parallel_for` return before its work is done, though the host back ends tested do not.
  The label `fftn::bulk` must fit the small-string buffer, because `parallel_for` copies its
  label into a `std::string` on every call. Kokkos 5 has no query for "inside a kernel", so
  the adapter asks every enabled threaded back end: on OpenMP `omp_get_level() > 0` (not
  `omp_in_parallel()`: a `parallel_for` also aborts inside an inactive region), on Threads
  whether this is the thread that may dispatch and whether its pool entry is active (`Impl`
  names, so `Kokkos::Threads` is accepted only with Kokkos 5.2.1, checked at compile time).
  There `concurrency()` is 1 and `bulk` runs on the caller, so an execution inside a kernel is
  serial. A `Kokkos::Serial` kernel cannot be detected: on Serial nothing fences, dispatches or
  opens a region, and a plan or executor on a threaded back end has the precondition of not
  being used from one. **`team_of(member)`** is
  `{team_rank, team_size, cache_group_size(team_size)}`. T3 measured what these cost.

What "natural in Kokkos" means concretely: plans accept host `View`s of any layout
through their extents and strides; the executor takes the caller's execution-space
instance and its concurrency; `view_plan`'s executions appear as a Kokkos Tools region, their
dispatches as `parallel_for`s named `fftn::bulk` (a plan on Serial opens no region); a plan class matching Kokkos-FFT's `Plan`
interface makes adoption a type change; the adapter is for the host only (with a device space
or View it does not compile, so an application building for devices selects the provider per
build). Kokkos is never a dependency of the core.

### 13.7 Testing threads

Bitwise comparison of serial against every team size on synthetic topologies;
deterministic simulation of the scheduler with adversarial interleavings; ThreadSanitizer
on the built-in pool and, with an instrumented OpenMP runtime, on the OpenMP adapters;
stress under oversubscription and injected delays; a runtime matrix (libgomp, libomp,
Kokkos Serial/OpenMP/Threads, the built-in pool).

As built (`tests/threading/`): real-thread teams of 1 to 16 members in cache groups of 1,
2, 3, 5 and all, over every kind, ranks 1 to 3, in place, strided and multi-dimensional
batches, with transforms split into many chunks, against serial execution bit for bit
(under ThreadSanitizer in the gate's tsan stage); late, missing and one-at-a-time members;
workspace reuse; no allocation. A simulator runs the members' threads one at a time,
switching at every access to shared state, in seeded random orders (adversarial
interleavings, members that arrive late or never, several executions on one workspace) or
in virtual time under a cost model, where a model of static schedules with barriers (the
prototype's, and FFTW's waking of its threads for a short job) shows the shape of their
measured pathologies beside FFTN's handling of the same plans. The model's cost per point
is the scheduler's own profile value, so agreement there checks the logic, not the
constants: experiments T1 and T5 are the real check. T1's harness (`bench/t1/`) runs the
claiming engine itself on synthetic work graphs with empty or fixed-length chunks, so that
a phase hand-off is timed without any transform; T5 times real plans through fbench's
threaded arms (16.1).

The executors (task 3.3): over the same cases, bitwise against serial on an executor that
runs tasks in order, one in reverse, one in seeded random order on three threads, the
thread pool seen at widths 1 to 16 and several groupings, and a pool through the C
contract; under ThreadSanitizer in the tsan stage. What FFTN asks of an executor (no call at
width 1 or for an inline job, not even a query for an inline job; exactly `dispatch`
tasks), an executor that throws part
way, nested calls, no allocation; the pool alone (every task once, calls back to back and
after its threads have blocked, concurrent callers, nesting, shutdown); the grouping on
synthetic and recorded topologies. The OpenMP adapters (`tests/omp`, built when the
compiler has OpenMP): the executor at several widths and this_team() in regions of 1 to 8
threads with both completions, bitwise against serial; regions of constant width; serial
inside an active region and at width 1; runtime settings untouched. They are not run under
ThreadSanitizer: the OpenMP runtimes here are not instrumented, and TSan then reports the
join of every region as a race (checked with Homebrew LLVM 23's libomp); an Archer-enabled
libomp would lift that. The runtime matrix is libomp (Clang) on macOS and Linux and libgomp
(GCC) on Linux.

The Kokkos adapter (`tests/kokkos`, task 3.5) runs on `Kokkos::Serial` and on the one threaded
back end a configuration builds (`FFTN_KOKKOS_BACKEND`: Threads by default, OpenMP in CI's
GCC 16 job), at four threads and again on a pool of one: every kind, bitwise against serial;
one `parallel_for` per transform (counted through a Kokkos Tools callback), none for an inline
job; allocations no more than Kokkos' own per dispatch; kernels calling the executor and
`view_plan`; `team_of` with both completions and with several teams; host threads other than
the one that started Kokkos. The Threads back end runs under ThreadSanitizer (Clang's tsan
stage); OpenMP cannot, as above.

---

## 14. Build and packaging

### 14.1 What was established by experiment

A proof of concept (a radix-4 transform, two translation units, scalar, SSE2, AVX2,
AVX-512, NEON and NEON+FCMA kernels in one source, no ISA flags) was built and run with
Apple Clang 21, LLVM Clang 23 and GCC 16 on AArch64 and with Clang for x86-64 under
Rosetta. Findings:

- **The mechanism of section 6.4 works**: per-target namespaces inside target pragma
  regions, a force-inlined vocabulary, kernel text re-included per region, entry points
  that take only pointers and integers, a function table chosen once. Results matched the
  reference at every optimisation level and under sanitizers; the link had no duplicate
  symbols; the disassembly showed the intended instructions and no residual calls.
- **A kernel template parameterised on a vocabulary type does not work**: a target
  attribute cannot depend on a template parameter (failed in every configuration tried).
  Hence re-inclusion, not a policy template.
- **Per-file ISA flags are unsafe**: an illegal-instruction crash was reproduced, by link
  order, when a shared inline function was compiled in an AVX-512 unit. The region design
  was immune.
- **Compile time rules out codelets in public headers.** 47 thousand lines of generated
  radix-16 codelets cost 13–30 s per including translation unit at -O2 and megabytes of
  code; merely parsing them unused costs 1–2 s. Unoptimised kernels run 30–60× slower.
- Lambdas do not inherit a region's target; on x86, Clang rejects passing a vector by
  value to any function outside the region. Both are covered by the region rules.
- The portable target string for FCMA is `arch=armv8.3-a` (`+fcma` fails with Clang's
  NEON header).
- On x86-64 Linux (experiment [B1](experiments/B1-isa-regions-gcc-linux.md)) the mechanism
  holds with GCC 13 to 16 and Clang 18 to 23, at -O0, -O2, -O3 with LTO and under ASan and
  UBSan. But every GCC compiles a vector passed by value from a region into a baseline
  function with only a `-Wpsabi` warning, and the program then segfaults (Clang rejects
  the same code). GCC also warns `-Wignored-attributes` wherever a vector type is a
  template argument, and does not inline a lambda in a region at -O0. The library is
  therefore built with `-Werror=psabi` under GCC, and vector types are never template
  arguments in region code.
- Not yet run: Clang on Windows (experiment B2).

### 14.2 Decision: a compiled static library

FFTN is built as a library: CMake target `fftn::fftn`, static by default, with several
translation units per instruction set (section 6.4) so they compile in parallel. A header-only
configuration was considered and dropped (decision log): it is technically possible, but
it would force every non-kernel part to be written header-style for the life of the
project, weaken the floating-point guarantees (the implementation would inherit the
user's flags), and cost users minutes of compile time. If drop-in use is wanted later, a
generated amalgamation (one header, one source file) gives most of the convenience from
the same code.

What the experiments still require of the build:

- **Public headers are light.** They declare shapes, plans, workspaces and adapters and
  contain no kernels and no intrinsics, only standard C++20. `plan<In, Out>` is a thin
  typed wrapper over a type-erased core. Including FFTN costs little, and the headers
  pass through compilers that merely parse them (for example a GPU toolchain's host
  pass).
- **No ISA flags.** Every instruction set comes from attribute regions, so the library
  can be built at the baseline of any project or wheel and still use AVX-512 or FCMA at
  run time.
- **ABI warnings are errors.** With GCC the library is built with `-Werror=psabi`
  (section 14.1).
- **Its own floating-point flags.** The library sets `-fno-fast-math -ffp-contract=off`,
  refuses fast-math, and never inherits the including project's flags. All FMAs in kernels
  are explicit. Intel's icpx defaults to a fast model that the headers cannot detect (no
  `__FAST_MATH__`) and sets flush-to-zero in an optimised `main()`, so an icpx build of
  FFTN passes `-fno-fast-math` to every target, third-party ones included (experiment
  X22).
- **Always optimised.** Unoptimised kernels are 30–60× slower, so the library's kernels
  are built optimised whatever the surrounding project's build type.
- **Adapters are header-only** (`omp.hpp`, `kokkos.hpp`): they must be compiled against
  the application's own runtime, and they contain no kernels.
- **Two copies in one process** (two shared libraries or Python wheels that each embed
  FFTN) must not collide: all symbols sit in a versioned inline namespace and have hidden
  visibility by default.
- **Detection details**: AVX and AVX-512 are used only when the operating system saves
  their state (XGETBV), not on CPUID alone. macOS on Intel enables the AVX-512 state
  lazily, on a thread's first use, so there `hw.optional.avx512f` stands in for XGETBV.
  Windows on Arm has no processor-feature flag for FCMA or pointer authentication, and
  reading the ID registers (MRS) faults in user mode; both are decoded from the copies of
  ID_AA64ISAR1_EL1 and ID_AA64ISAR2_EL1 that Windows keeps in the registry (`CP 4031` and
  `CP 4032` of processor 0), and reported absent where those values are missing
  (experiment B2).

### 14.3 Toolchains and packaging

- C++20, extensions off. GCC 14 and newer; Clang 16 and newer; Apple Clang; clang on the
  MSVC ABI (clang-cl and clang++ with the MSVC standard library); Intel's icpx (2026.1
  tested, X22: supported, not recommended over GCC or Clang). MSVC's compiler is not
  a target. C++23 library features may be used only behind feature-test macros.
- CMake 3.25 or newer with presets; `FetchContent`-friendly with an offline override;
  installed package config (`find_package(fftn)`) and pkg-config; all symbols in
  `namespace fftn`, all macros prefixed `FFTN_`, all targets namespaced.
- Test-only dependencies (GoogleTest, FFTW, Boost.Multiprecision, and Kokkos for the
  optional adapter's tests and example) and benchmark-only dependencies are fetched pinned
  by URL and content hash (SHA-256), with an offline override, and treated as system code.
  The library itself links none of them; `fftn/kokkos.hpp` is compiled in its user's
  translation units against the user's Kokkos. Kokkos' back ends are chosen by forcing its
  `Kokkos_ENABLE_*` cache entries (Serial only); task 3.5's OpenMP and Threads test
  configurations need that reworked. FetchContent also brings Kokkos' install rules into the
  build, to be excluded when packaging lands (`EXCLUDE_FROM_ALL` on the declaration needs
  CMake 3.28). GCC ThreadSanitizer builds leave the Kokkos tests out by default: GCC's
  `-Wtsan` fires on Kokkos' fences once inlined into our units, beyond the reach of SYSTEM,
  and the adapter is serial; Clang's TSan builds run them. Windows on Arm leaves them out
  too: Kokkos 5.2.1 passes the x86 option `-mcx16` to every Clang build on Windows.
- Build options are the cache variables at the top of `CMakeLists.txt`, each described
  there. One changes the generated code: `FFTN_MASKED_TAILS` (`AUTO`, the default, is on
  with GCC and off with every other compiler; `ON`; `OFF`) selects masked partial runs on
  x86 (section 8.2). The release toolchain for x86-64 Linux is GCC, so release builds have
  them; results are bitwise the same either way. Another chooses which targets exist:
  `FFTN_OLD_X86_TARGETS` (`ON`, the default) builds `x86_v2` and `avx` (section 6.1), 52 % of
  an x86-64 library's text for processors from before 2013 (X14); `OFF` leaves them out,
  and on such processors dispatch falls back to the scalar target. Distributed builds keep
  the default so dispatch stays complete; CI builds both (`.github/workflows/ci.yml`, the
  post-merge job "clang-23 no old x86"). The option is part of the wisdom build identifier.
- A generated single-file amalgamation is a possible release artefact (phase 6); it is
  the same implementation concatenated, not a separate code path.

---

## 15. Correctness, accuracy and determinism

### 15.1 Oracles

1. **Analytic identities**, as the first oracle: impulses and shifted impulses, single
   complex bins and real tones, constant input, Parseval, the shift theorem, linearity,
   forward–backward round trips, conjugate symmetry of real transforms, separability of
   multi-dimensional transforms, convolution with a pure delay.
2. **A direct DFT in quadruple precision** (`tests/reference`), O(n²), for moderate n;
   for large n, comparison at sampled output bins. It uses Boost.Multiprecision's
   113-bit binary float, standalone and header-only, as a test-only dependency (with
   Boost.Config, which its standalone mode requires; nothing else of Boost). Native
   types do not cover the platforms: `long double` is plain double on Apple silicon and
   with the MSVC ABI, and `__float128` is rejected by Clang on Apple silicon (GCC accepts
   it there). A 1024-point reference DFT takes about 0.15 s on an Apple M5 Max. The
   library's own twiddle tables, which it builds with a few lines of double-double
   arithmetic, are checked against the same reference entry by entry (correctly rounded),
   so the reference shares no code with the library.
3. **FFTW**, as an independent implementation, for large sizes and every shape
   (`tests/fftw`): the library's outputs compared with FFTW's at every output, with the
   tolerances of both added. FFTW also runs oracles 1 and 2 as a subject, which checks
   the suites themselves. Test builds compile FFTW 3.3.11 from source in both precisions (plain C, about 20 s), or use
   an existing install (`FFTN_FFTW_ROOT`); `FFTN_TEST_WITH_FFTW=OFF` builds without it and
   the tests that need it skip.
4. **The generator's own check** of each codelet against the exact DFT at emission.

### 15.2 Tolerances

Derived before the test runs from machine epsilon, the length and the algorithm: a
transform of length n built from stable butterflies has relative RMS error of order
ε·√(log₂ n), and FFTN's bound for the test is a small multiple of that (the prototype
measured 4–5·10⁻¹⁶ against a tolerance 700 times looser; FFTN's tolerances are within
about 10× of the measured error). With L = max(1, log₂ n), the tests require a relative
RMS error ‖Y − X‖₂/‖X‖₂ ≤ 3ε√L and a largest error ‖Y − X‖∞/‖X‖∞ ≤ 3εL
(`tests/support/tolerance.hpp`, which records the derivation and the measurements: FFTW
and a careful naive DFT sit 3–10 times below the first and 7–40 times below the
second). These catch errors of order one and twiddle errors that grow with the index (a
base twiddle one ulp off, propagated by powers, from about 64 points on). A single table
entry one ulp off is below the rounding of every correct implementation and no output
tolerance sees it, so twiddle tables are checked against the reference entry by entry.
Bluestein paths have their own, looser, derived bound: three transforms of s M points in
sequence (forward and inverse at execution, the kernel spectrum at construction, in
double), adding in quadrature, so 3ε√(3L′) and 3√3·εL′ with L′ = log₂(sM) ≤ log₂ n + 2
(`bluestein_bound` in `tests/support/tolerance.hpp`, applied to every length with a prime
factor above 16 other than the codelets' 17, 19 and 23, and to Rader stages, whose
convolution is shorter). Measured on the Mac (task 2.3, primes and products up to 10403, double
and float): 0.42–0.83 ε√L RMS and 0.13–0.37 εL largest, inside even the stable bound.
Accuracy is also reported by fbench next to speed for every library.

### 15.3 What is tested

- **Every path, forced.** Parameterised suites run each oracle under every instruction
  set the machine has, each arithmetic layout, each strategy, each pass kind and edge
  variant, each twiddle mode. A new path arrives with its forcing parameter; the registry
  generated from codelet metadata lists what must be covered, and a test fails if a
  registered kernel was never exercised.
- **Lengths**: all n up to a few hundred; every product of the supported radices up to a
  bound; primes and awkward composites; powers of two up to 2²⁴ against FFTW.
- **Contracts**: memory outside the described rows is untouched (canary pages); any
  alignment and offset; in place and out of place; strides and distances, including
  negative strides; input preserved; partial batches; plans executed concurrently from
  several threads; no allocation in execute (an allocator hook counts).
- **Threading**: section 13.7.
- **Property tests**: random shapes, strides, batch counts and edge operations checked
  against the reference built from the same description.
- **Build contracts**: the public headers compile alone, in any order and from several
  translation units; disassembly checks per target (section 6.4); the public headers refuse `__FAST_MATH__`.

### 15.4 Gate and continuous integration

The gate (`scripts/check.sh`) runs: SPDX headers, document links, experimental-switch
registry, clang-format, ruff, shell syntax, generator reproducibility, public-API
documentation completeness, Release build with warnings as errors and the tests and
documented examples under every kernel variant, clang-tidy, ASan with UBSan (leak
detection on), ThreadSanitizer, coverage of hand-written code against a threshold that
may not fall, and an fbench smoke run.

CI matrix (phase 0 onward, growing): Linux x86-64 with GCC 14 and current GCC and Clang;
macOS arm64 with Apple Clang and Homebrew LLVM; Linux AArch64; Windows with clang on the
MSVC ABI. A pull request runs a short set before it may merge (current GCC on Linux x86-64,
one ASan + UBSan job there with Clang, Homebrew LLVM on macOS arm64, and lint); each push
to `main` then builds and tests one configuration per platform (both floors, GCC 14 and
Clang 16, among them) and runs one ThreadSanitizer job on Linux x86-64 with Clang, so a
break shows within the hour and is fixed on `main` (decided 2026-11-17, while there are no
users, for merge latency); the other compilers, the other sanitizer runs, clang-tidy and
coverage run weekly and on demand. AVX-512 paths run under Intel SDE and NEON-without-FCMA and SVE paths under
QEMU where the runner's CPU lacks them, so every kernel is executed in CI, not only on
developers' machines. Old processors are emulated as whole machines too (SDE's Nehalem
and Sandy Bridge models, QEMU's Neoverse N1), which checks detection and dispatch on
them as well as the kernels: a stray newer instruction anywhere faults. Emulation runs the
test programs that contain kernels or detection, not the long oracle suites (those run
natively), weekly and on demand.

The workflows in [`.github/workflows/`](../.github/workflows/) are the matrix: which gate
stages run on which compiler and why, the emulated models with the target each must
choose, and every pinned version.

### 15.5 Determinism and the numerical contract

- No fast-math, no flush-to-zero, no approximate reciprocals. FMAs are explicit; the
  library is built with `-ffp-contract=off` so the compiler introduces
  none (GCC 15 contracted two inlined copies of the same code differently in the
  prototype).
- **Bitwise identical** for a given plan across: serial and threaded execution, any team
  size, executor and schedule; repeated runs; batch position (row 0 and row 1000 of a
  batch get the same arithmetic, as do the transforms of `execute_transforms` for any
  `first`). The rule that guarantees the last: within a plan every row goes through the
  same sequence of codelets; lane tails and leftover rows use narrower instantiations or
  padded tiles of the same program, never a different route. Tests compare row k against
  row 0 for every k modulo the lane count.
- **May differ in the last bits** between: instruction sets; plans built with different
  options (width hint without `strict_layout`, edge operations, in place versus out of
  place); plans of different batch counts (a batch's row against a plan of that row alone:
  batch leaves run C rows side by side, X53); plans chosen by measurement on different runs,
  unless pinned by wisdom; library versions.
- **What the guarantees are worth** (maintainer, 2026-11-18): the accuracy contract is the
  requirement, identical last bits are not. The bitwise guarantees above are kept while
  they cost nothing measurable; a change that would break one is adopted only for a real
  measured gain, and narrows the published guarantee (`docs/conventions.md`,
  "Determinism") in the same change.
- Error grows as O(ε·√log n) RMS for all lengths with codelet factors; Bluestein lengths
  have the bound of section 15.2 (the same growth in log₂ of the convolution's length).

---

## 16. Benchmark rig: fbench

fbench answers two questions: "is B faster than A?" within minutes and with a stated
confidence, and "where do we stand against every competitor, and has anything moved?"
over time. It inherits the prototype rig's concepts and replaces its statistics.

### 16.1 Structure

- **One executable per library.** FFTW-API libraries export the same symbols and OpenMP
  runtimes collide, so each backend is its own binary behind a small C interface
  (`begin(options)`, `create(desc)`, `execute`, `destroy`, `describe`, `end`). A shared
  harness owns everything else: buffers, data, timing and the correctness check. Each
  binary reports the library version, build flags, commit and the instruction set actually
  selected; fbench refuses to compare arms that report identical configurations when they
  are meant to differ, refuses a binary that resolved the wrong shared library, and refuses
  to time sanitizer or Debug builds.
- **Arms** are written `label=library[:build][+OPTION=value…]`. An FFTN arm can be any
  commit (fbench builds it from `git archive`) or any plan option, and, for experiments,
  an internal construction knob the public API lacks (`RADICES`, experiment X2).
- **Precision**: a case is double or single precision; a backend takes single-precision
  cases only if it says so. The harness generates and checks in double, rounds a
  single-precision input to float before the library and the reference see it, and
  scales the tolerance by the precision's epsilon.
- **Catalogues** are data files of weighted cases:
  - *standard*: powers of two 2¹–2²⁴; 2-3-5-7-smooth lengths; primes and awkward
    composites; batches from 1 to thousands of rows; r2c and c2r; planes and volumes;
    in place and out of place; strided axes;
  - *client*: shapes captured from an application's trace with weights from its real FFT
    share (direction, placement and planes included); kept with the client, outside this
    repository, and given to fbench by path;
  - *threads*: the same shapes at 1, a cache group, a socket and all cores, in library
    mode and in team mode (a team of serial plans, the baseline threading must beat).
- **Tiers**: smoke (seconds, correctness only, in the gate), quick (2–3 minutes, A/B),
  tracking (a fixed set of 30–60 cases), full.

As built for threading (task 3.4): FFTN arms take `THREADS=N` (the executor overload on an
`fftn::thread_pool`, or on `fftn::omp::executor` in a second backend binary, `fftn_omp`,
linked with the compiler's OpenMP runtime; `CALL=team` there runs the team overload inside
a region), `GROUP_SIZE`, and every machine-profile field and the chunk size as options. The
backend reaches the plan core through the functions the public plan forwards to, and the
tuning options through its private build settings, as tests do: no library switch exists
for them. The `threads` catalogue groups shapes by the scheduling rule they exercise; the
library and team modes against competitors are still to come. Fixed costs that fbench
cannot isolate (dispatch, wake-ups, phase hand-offs, claims) are measured by a separate
harness for experiment T1 (`bench/t1/`), which also prints the scheduler's decisions for
catalogue cases.

### 16.2 Competitors

| Platform | Libraries |
|---|---|
| all | FFTW 3.3.11 (best SIMD and FMA options; MEASURE and PATIENT; fresh wisdom per campaign), pocketfft/ducc0, KFR, PFFFT (double), KissFFT (floor), RustFFT, zldsp::fft (powers of two) |
| macOS arm64 | + FFTL.jl (libfftl), Apple Accelerate, Arm Performance Libraries |
| Linux x86-64 | + Intel MKL (DFTI), Intel IPP, AOCL-FFTW, AOCL-FFTZ, FFTL.jl |
| Linux AArch64 | + Arm Performance Libraries, FFTL.jl, NVPL on Neoverse V2 |
| Windows | + MKL, IPP |

Competitors are built as their best users would build them. Convention adapters keep the
correctness check meaningful (packed formats, scale factors). Plan time is measured and
reported separately for every library.

### 16.3 Statistics

- **The unit of replication is a process round.** A round launches every arm once, in a
  counterbalanced order; within a process each case takes several short timed batches and
  contributes its median (and its minimum). Ratios are paired by round.
- **Report** the median ratio with a bootstrap 95 % confidence interval over rounds. An
  **identical-arm control** (A against A) runs automatically and sets the noise band: the
  95th percentile over cases of the control's |median log ratio|, at least 1 %. The
  verdict for each case is faster, slower or no evidence, by whether the interval clears
  that band. Rounds are added until the target cases resolve (a verdict, or an interval
  inside the band) or the budget ends. Arm orders follow a Williams design. Test-only arm
  options seed a known slowdown (`DELAY`) or a wrong output (`PERTURB`) to prove the
  statistics and the guard.
- **Same-process A/B** for two FFTN plan options: both plans alternate inside one
  process, which removes page-placement and address-layout effects.
- **Hot and cold are explicit.** Hot reuses buffers; cold rotates through a pool larger
  than the last-level cache, plan tables included where the library allows it (FFTW
  shares its twiddle tables between plans, so for FFTW only the data rotates). Buffers are
  first touched by the threads that will use them. Buffer offsets within a page are
  randomised between rounds, in multiples of 64 bytes so that alignment is kept, and are
  shared by all arms of a round so an aliasing accident cannot decide a result.
- **Load.** fbench records load, frequency and thermal state where the OS exposes them,
  per round the load average and other processes' current CPU use. Load does not
  invalidate a round: interleaving, repeated process rounds and the A/A control absorb it,
  so a busy machine gives a wider band, and claims are judged against the band of the same
  run. A round waits (boundedly) only while other processes use more than half the CPUs;
  a strict mode refuses rounds above the quiet thresholds. A process timeout loses the
  cases that process had not finished in that round, not the run. On macOS fbench
  requests the performance-core class.
- **Accuracy** (relative RMS and maximum error against a double-double direct DFT in the
  harness, at sampled outputs, plus Parseval over whole rows) is stored with every timing;
  a wrong output in any round is reported as such, never averaged away.
- **Competitors' state is per campaign:** FFTW wisdom starts empty in every run and is
  shared by that run's processes with the same binary and planner.
- Throughput is reported as ns per point and as mflops = 5·n·log₂n / t for complex
  transforms and 2.5·n·log₂n / t for real ones, the convention of FFTW's published
  benchmarks.

### 16.4 Tracking

Each run appends compact JSON lines (case, arm, statistic, host fingerprint, build
identity, run ID) to a results store kept outside the main branch. The tracking tier runs
on the benchmark machines on a schedule and after merges that touch kernels. Every result
is stored as a ratio to FFTW measured in the same session, which is robust to machine
drift. `fbench history` shows trends and flags changes that leave the noise band;
`fbench gaps` ranks losses against the best competitor by weight. Documents cite run IDs;
status tables are generated, never typed.

This store is internal. **Published benchmarks are separate and minimal**: one clean
report per tagged release, generated by one command from a fixed, documented set of
cases and machines, and committed with the release. Nothing else from the internal store
is published.

For CI, where wall-clock time is unusable, instruction counts of a fixed kernel set are
compared against the previous commit as a cheap regression tripwire.

### 16.5 Machines

Apple M-series (local), Xeon Gold 6240 and EPYC 7763 (Slurm, exclusive nodes), and rented
cloud instances for Zen 4/5, recent Intel and AArch64 (Neoverse N1, V1, V2). Runner
scripts ship pinned dependencies, copy sources by content, give each job its own build
directory and record the commit of every arm.

---

## 17. Documentation

Readers know what an FFT is. They want to know exactly what this library gives them. The
documentation is therefore small, exact and kept next to the code, with no documentation
infrastructure to maintain.

### 17.1 What exists

| Document | Content | Form |
|---|---|---|
| `README.md` | what FFTN is, a first example, links | Markdown |
| `docs/conventions.md` | every convention, stated exactly: the formula each transform computes, signs, normalisation, the layout of every input and output (r2c and c2r sizes and padding, multi-dimensional halving), strides and distances, in-place rules, what is and is not preserved, alignment, accuracy bounds, the determinism contract | Markdown |
| `docs/guide.md` | task-oriented: rows, planes, volumes, real transforms, choosing lengths, workspaces, planning effort and wisdom | Markdown with examples |
| `docs/planning.md` | planning effort (rules, quick and full measurement) and wisdom | Markdown with an example |
| `docs/lengths.md` | fast lengths (`next_fast_length`) and the per-length estimate (`relative_cost`) | Markdown with an example |
| `docs/kokkos.md` | the optional Kokkos adapter: View-based construction, the Kokkos-FFT-shaped plan, moving a Kokkos-FFT call site, the rules for Views and axes, normalisation, and every difference from Kokkos-FFT | Markdown with examples |
| `docs/threading.md` | each integration model with a complete example: serial, inside an OpenMP region, the OpenMP executor, Kokkos, a custom pool, the built-in pool; what FFTN never does to the caller's runtime | Markdown with examples |
| `docs/fusion.md` | edge operations, convolution, transform pairs, block callbacks | Markdown with examples |
| `docs/building.md` | CMake use, supported compilers and platforms, instruction sets and how one is chosen, options | Markdown |
| API reference | every public type, function, parameter, precondition and guarantee | comments in the public headers |
| `gen/README.md` | the generator: pipeline, the intermediate representation, how to add a radix, algorithm or lowering, the command line, and its subtleties | Markdown |
| `bench/README.md` | fbench: commands, catalogues, how to add a backend, how to read a verdict | Markdown |

### 17.2 Rules

- **The public headers are the API reference.** Every public declaration carries a
  comment stating what it does, its preconditions, and what it guarantees. The Markdown
  documents explain and link; they do not restate signatures.
- **Examples are compiled.** Every code example in `docs/` is a snippet of a file under
  `examples/`, which the build compiles and the tests run. A gate stage checks that the
  snippets in the documents match the files, so examples cannot rot.
- **Completeness is checked, not hoped for.** Doxygen is used as a linter only: a gate
  stage runs it over `include/fftn/` with undocumented public members as errors. It needs
  one small configuration file and produces nothing that is committed. Rendering an HTML
  reference from the same comments for a release is one command and optional; the
  comments are written to read well as plain text in the header first.
- **Internal code is commented sparingly.** No boilerplate docstrings. A comment is
  required wherever something is subtle, non-obvious, or was learned the hard way: it
  states the reason and cites the experiment or finding (for example "radix ≤ 8 here: 2R
  streams exceed L1 associativity, experiment X4"). The same rule applies to the
  generator's Python.
- **Documentation changes with the code**, in the same commit. A change to the public
  API or a convention without its documentation is not finished.

---

## 18. Risks and open questions

### 18.1 Technical risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| The generator does not reach FFTL's and FFTW's codelet quality quickly | every speed target depends on it | start from FFTL's small design (rotation tracking), verify against published operation counts, add algorithms one at a time; the walking skeleton keeps the rest of the project moving |
| Two arithmetic layouts double the kernel surface | compile time, test matrix, planner space | both come from one DAG and one set of pass templates; experiment X1 may show one layout suffices per target, and the loser is deleted there |
| MKL parity on Intel | MKL leads the prototype by 15–45 % on short smooth lengths | few-big-pass strategy, split arithmetic to avoid port-5 shuffles, per-shape measurement; treated as a target, not a guarantee |
| Compiler fragility of attribute regions (Windows clang still untested; B1 passed on Linux) | the whole dispatch mechanism rests on them | experiment B2; strict region rules; a disassembly test; whole-machine emulation of old CPUs in CI |
| The claiming engine costs more per phase than a barrier | long single rows on few threads | experiment T2; the barrier path is kept if it earns it |
| Cost model drifts on new CPUs | wrong plans by default | nearest-family profiles, `planning::measure`, `fbench calibrate` as a one-command remedy |
| Scope creep | the project stalls before the client milestones | phases 1–5 serve the core tier only; the general tier waits for phase 6 |

### 18.2 Open questions

Decisions taken are in the decision log; questions that arise during implementation are
listed here until settled.

**`FFTN_OLD_X86_TARGETS`: settled** (agreed by the maintainer, 2026-11-17: `ON` by default; see
the decision log).

**Deferred API ideas.** (a) A two-way plan: forward and backward programs of one shape over
one set of twiddle tables, with `forward()` and `backward()`, for real kinds the r2c and c2r
pair. (b) An object owning one plan and one workspace, executing without a workspace
argument. Both are additions over the current API (two plans sharing a `table_cache`; a
plan plus `workspace(plan)`) and stay cheap only while two properties hold: twiddle tables
do not depend on the direction, and a workspace is plain bytes sized by
`workspace_bytes()`, not tied to one plan. Do not break either without revisiting this.

**Generator (task 1.1).** (a) A vocabulary operation that multiplies by a constant complex
number: two `FCMLA` on NEON+FCMA, where the complex-register codelets now spend two
`scale` and one `addj` or `subj` on each generic constant root (four in radix 16, four in
9). It touches every target's vocabulary, so it is weighed with the split lowering's FMA
shaping (task 1.5). (b) Rader's algorithm inside the generator for 13 (and 17, 19, 23,
whose codelets are in the symmetric-pair form since X27): the symmetric-pair form costs
192 additions and 134 multiplies at 13 against Rader's 176 and 68.
(c) Settled in task 1.2: the vocabulary gained `bcast_tw` (decision log).

**Task 1.2 (for the maintainer).** Public API added in the shape of section 5.3, each a
choice to confirm: (a) `options::scale` (a `double`, default 1): multiplies every output,
fused into the pass that writes the output; each output equals the unscaled output times
the scale rounded once to `T`, bit for bit; plan construction refuses NaN, infinities and,
for `float` plans, magnitudes beyond `float`'s range or non-zero ones below its smallest
value (naming `options.scale`). Alternative: accept them. (b) `options::force_isa` with a new
enumerator `isa::best` (the default), appended after `neon_fcma` so the other values keep
theirs; `instruction_set()` never returns it and `isa_name` names it "best". A target the
build lacks or the processor cannot run is refused, naming `options.force_isa`.
Alternative: `std::optional<isa>`, which needs no sentinel enumerator. (c) The suites'
FFTN subjects are named per instruction set (`fftn_scalar_double_…`, `fftn_neon_fcma_…`,
`fftn_avx2_…`) instead of `fftn_…`, and run on every target the machine has. Internal
questions: (d) code size: each target's unit holds about 1,000 pass instantiations (1 MB
on AArch64, 2.5 MB of object on x86-64), most of them leaves (15 radices × 2 directions ×
2 precisions × 8 edge combinations); edge combinations could be pruned once the public
edge operations are known. (e) Planner v0's rules come from FFTL.jl (MIT licence). The first implementation
followed FFTL's function too closely (its structure and identifiers), which AGENTS
section 10 rules out without the maintainer's agreement; it was replaced by code written
from the specification in section 11.2, which reproduces the first implementation's
output exactly for every 73-smooth length up to 20000 on every profile (recorded test
data). To confirm: that rules from an MIT-licensed library, rewritten from a
specification and cited, need nothing further (an MIT notice is needed only if code were
copied). Planner v1 (experiment X2) has since replaced those rules by measured pass costs;
nothing of FFTL's remains in the planner.

**Task 2.4: the planning and wisdom API.** Settled: the maintainer agreed the API as
proposed (decision log, 2026-11-16, "The planning API (task 2.4, pull request 26) as
proposed"): three efforts, the two budget options, `fftn::wisdom` filled with `add` and passed
by `options::use`, stale or unbuildable entries ignored with a note in `describe()`, the 1 %
margin, serial timing. Still open, details below the API: the shape key uses strides and
distances as given (0 for a default), so two spellings of one layout are two keys; a plan for
a team is chosen by its serial time.

**Task 2.5: fast lengths and the per-length estimate.** Settled: agreed by the maintainer,
2026-11-17, with `bool real` replaced by `length_kind` (decision log).

**Threading API and contract (tasks 3.1 and 3.2), for the maintainer's review.** Built as
below because section 5.3 left the shapes open; each is easy to change before a release.

1. *`team` is `{rank, size, group_size}`*: cache groups are runs of `group_size`
   consecutive ranks. There is no per-member CPU lookup (a dynamic lane map for unpinned
   threads, 13.3), no core class (rule 5's fastest-class restriction) and no caller
   barrier hook. The OpenMP and pool adapters fill `group_size` from the topology (item 18);
   unequal groups (a hybrid chip's) cannot be described yet.
2. *Completion.* `all`: claiming; no member waits for one that has not arrived, and each
   member returns once the whole output is written; surplus members help or wait, they do
   not return at once. `own`: a static deal of whole transforms, never split; with fewer
   transforms than members the first members own one each and the rest none; the caller
   synchronises before reading other members' transforms or reusing the workspace. `own`
   does not use the team state, but a smaller team's `own` execution overwrites a larger
   team's state with scratch, so it clears the workspace's word and the next `all`
   execution sets the state up afresh.
3. *`rows_of` is `plan::transforms_of(const team&)`*, returning `transform_range{first,
   count}`: on the plan rather than the workspace, since the deal depends only on the
   transform count and the team, and named after `execute_transforms` (review item 5).
4. *Workspace sizing.* `workspace_bytes(lanes)` is 128 bytes of slack, the team state
   ((3 + ⌈lanes / 16⌉ + 4 lanes) lines of 128 bytes) and one serial scratch slot per
   member, rounded to 128 bytes, saturating at `SIZE_MAX` if that does not fit;
   `workspace_bytes(1)` is the serial size. A split
   transform's stage chunks use their owner's slot, so no slot is shared between tasks.
   This keeps three buffers' worth of scratch per member, against section 10's rule 5;
   sharing slots within a group is possible later.
5. *The workspace keeps team state between executions* and gains one hidden word (whether
   its memory holds valid state, for how many lanes); serial executions clear it. A
   workspace is still not tied to a plan, so the deferred ideas above stay open.
6. *The sequence precondition*: members follow one sequence of executions per workspace
   (late arrival allowed, skipping not), and serial use or a new team size waits until
   every member of earlier executions has returned. It is the price of late arrival without
   a barrier; the alternative, a barrier at the end of every execution, waits for the
   slowest member and deadlocks a runtime that runs members one after another.
   *Skipping*: a member knows its execution only by counting its own calls, so one that
   skips an execution counts wrong and returns from its next call before that call's
   output is written. Nothing can detect this; the only reset is a serial execution on the
   workspace (`execute_transforms(0, 0, …)` costs nothing else). A runtime that may not run
   every member (Kokkos `RangePolicy`, TBB) belongs with the executor overload of task 3.3,
   where FFTN dispatches the members itself.
7. *Oversubscription* is a team larger than `std::thread::hardware_concurrency()` taken at
   plan construction: no transform is split and waiters do not spin. The allowed-CPU mask
   and cgroup quota (13.5) are not consulted yet.
8. *Stealing split tasks.* A member of another group runs a split task whole, and only when
   no member of its group has arrived (13.3 left open how the stealing group runs it).
9. *Not built*: the prototype's static fast path (experiment T2 decides), hierarchical
   completion across groups.
10. *The epoch limit.* Epochs are 64-bit counts and never wrap; tags carry their low 40 bits
    and are compared only for equality. A wrong result needs a state word left unwritten
    for exactly a multiple of 2^40 executions and then reused at that execution: at least
    12.7 days of back-to-back one-microsecond executions on one workspace, and then only at
    one execution in 2^40. Counts have 24 bits: split tasks of 2^22 chunks or more run
    whole, batches of 2^22 transforms or more are claimed in units, and teams of 2^22
    members or more run inline.
11. *Ordering of a rank's calls.* A member reads and writes its call count without
    synchronisation, so one rank's successive calls must be made by one thread or ordered
    by happens-before (OpenMP threads and a `std::thread` per rank are; a task system that
    moves a rank between threads must order them). All members pass the same workspace
    object: two workspaces over the same borrowed memory have separate words.
12. *Lock-based waiting.* `std::atomic::wait` blocks through the platform (futex,
    `__ulock_wait`, `WaitOnAddress`) in libstdc++ on Linux, libc++ on macOS 11 and later and
    the MSVC STL from Windows 8. Where a standard library falls back to a mutex and
    condition variable (libstdc++ on platforms without a futex, the MSVC STL before
    Windows 8), a member that blocks takes a lock, though only after its spin and yield
    budgets and never on the path where work is done. Whether that is acceptable under the
    execute contract ("no locks") is the maintainer's call; the alternative is FFTN's own
        per-platform futex wrapper.

**Threading adapters (task 3.3), for the maintainer's review.** Built as below; each is easy
to change before a release.

13. *The executor concept*: `concurrency()` and `bulk(n, task, arg)`, optionally
    `group_size()`; tasks are a function pointer `void (*)(void*, int)` and an argument, so
    nothing allocates and the library needs no template. `executor_ref` is the C struct
    (item 20) with thunks, implicitly constructible from any executor (the one implicit
    constructor in the API, so that `execute(in, out, pool, ws)` reads naturally). Not
    provided: a `run_team` refinement for executors that guarantee concurrency (it would
    serve the static fast path, if experiment T2 keeps it), lane maps or a CPU per lane, and
    an `in_parallel()` query: nesting is signalled by `concurrency()` returning 1 (item 16).
14. *The executor overload is not `noexcept`*: FFTN's own code throws nothing, allocates
    nothing and locks nothing, but a user's executor may throw (a task system that fails to
    spawn); the exception propagates, the output is unspecified, and the workspace's team
    state is dropped so that members that did not run cannot count the next execution
    wrong. The alternative is `noexcept`, terminating on a throwing executor.
15. *Team size under an executor*: the scheduler is asked with `concurrency()` lanes and
    `dispatch` tasks are dispatched (the dealt lanes, at most one per task when nothing
    splits), as a team of that size (the same groups and no splits; shares may differ).
    The workspace must hold `workspace_bytes(concurrency())` (a precondition, asserted in
    debug builds); clamping the width to what the workspace holds was the alternative.
16. *Nesting through `concurrency()`*: an executor reports 1 from inside its own tasks, so a
    nested execution takes the serial path. `thread_pool` does so inside a task of any FFTN
    pool (a thread-local flag); `omp::executor` when `omp_in_parallel()`, that is inside an
    active region only (an enclosing region of one thread lets it open its own). A custom
    executor that does not is still correct, but a pool that blocks a busy thread while its
    nested tasks wait for a free one deadlocks, as any nested `parallel_for` on it would.
17. *`omp::this_team()`*: every thread of the region must make the call; inside `single`,
    `masked`, a task or a worksharing loop only some do, and OpenMP has no query that tells,
    so this is a documented precondition. In a nested region it describes the inner team.
18. *Grouping (`fftn::cache_group_size`, public, in `team.hpp`)*: the smallest group of the
    last level shared by several cores (L3, else L2), with the team assumed packed onto
    consecutive cores (one thread per core while it has no more members than the machine has
    cores). It is public because the header-only OpenMP adapter and callers' own teams need
    it. It is an assumption: unbound threads that Linux spreads over the groups can put a
    split transform across groups (correct, slower), while an estimate assuming spread
    threads would have stopped small teams on Apple's clusters (where macOS packs threads)
    from splitting at all. The research proposal of conservative thresholds for assumed
    groupings is not built (`team` cannot say a grouping is assumed); a lane map measured per
    call (13.3, 13.5) and experiments T1 and T5 should replace this.
19. *`thread_pool`*: lanes include the caller of `bulk`; threads are not bound (no placement
    modes, no QoS classes); no process-wide `shared()` instance; no fork handling; idle
    threads linger for fixed budgets (spin 20 µs, yield 50 µs), not configurable; a second
    thread's `bulk` while a job runs is run serially on that thread rather than queued or
    locked; `bulk` takes a function pointer only (no template convenience for lambdas); the
    default size is every allowed CPU, of every core class (rule 5 would take the fastest
    class only).
20. *The C contract* lives in `fftn/executor.h` now (types only, so no linkage
    specification), not in the phase 6 `fftn/c_api.h`, which will include it: `fftn_executor`
    {`self`, `concurrency`, `bulk`, `group_size` (may be null)} and `fftn_task`. Counts and
    indices are `int`.
21. *`omp::executor`'s width* is fixed at construction (`omp_get_max_threads()` by
    default), not read at each call: a later `omp_set_num_threads` does not change it, which
    keeps every region's width constant.
22. *Dispatch cost and the team's width* (T1, T5): the machine profile prices a team call as
    one constant, `dispatch_ns`, but it grows about linearly with the executor's width (on
    the EPYC 5.6 µs for 8 pool lanes and 26 µs for 64; 3.3 and 35 µs for the OpenMP
    executor, whose regions keep their full width). Rule 1, decided at plan construction,
    cannot see the width, and rule 3's cost model ignores that splitting a lone transform
    adds a dispatch that running it whole on the caller does not. T5's split thresholds
    absorb this at the widths measured (a CCX's crossover moved from 3072 points at 8 lanes
    to 6144 at 64). Proposed: a cost a + b × lanes, used by rule 1 at execution (or bounded
    by the CPU count at construction) and by rule 3 for a batch that would not otherwise be
    dispatched; then T1's measured `ns_per_work` (1.0 on the EPYC, not adopted because it
    exposed this) is tried again.

---

## 19. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-13 | C++20 for all code | Required for public headers by the first client; GCC 14 floor; no C++23 feature offers a material gain (assessed in the header-only study) |
| 2026-11-13 | No dependencies in the library; own vector vocabulary instead of a SIMD library | Full control of code generation, including FCMA, which the portable SIMD library tried never emits; the vocabulary is a few hundred lines per target; removes process-global dispatch state |
| 2026-11-13 | Generator in Python, output committed | No build-time dependency for users; graph rewriting and counting are not expressible in templates |
| 2026-11-13 | One pass framework with programs as flat records; strategies as program builders | Removes the prototype's three duplicated kernel families; makes blocking, batching and the threading work graph data |
| 2026-11-13 | Both complex-register and split arithmetic generated; planner chooses | Measured winners differ by target, length and pass position |
| 2026-11-13 | Planner: cost model by default, bounded measurement opt-in, wisdom as explicit data | Cheap deterministic planning; measured misses of fixed rules; no global state |
| 2026-11-13 | No hidden threads; claiming engine; plan fixes the division of work | Measured failure modes of library-owned pools; correctness on runtimes without a concurrency guarantee; bitwise schedule independence |
| 2026-11-13 | User fusion through block callbacks, not per-element functors in kernels | User code cannot be compiled into ISA regions soundly; block granularity captures the memory saving |
| 2026-11-13 | No FFTW fallback, ever; any length handled natively | A fallback hides gaps and drags in a second library and its threading |
| 2026-11-13 | Older machines are first-class: `x86_v2` (SSE, no AVX) and plain `neon` targets are generated, tuned and tested; an `avx` (no AVX2) target is decided by experiment X14 | Maintainer's steer: run well on every architecture and age the client application supports (x86-64-v2 and up, Neoverse N1 and up) |
| 2026-11-13 | Every core `execute` takes a workspace; twiddle sharing through a caller-owned `table_cache`; no hidden caches outside the convenience layer | Independent review: a cached workspace and a global twiddle cache contradicted the execute contract and the no-global-state rule |
| 2026-11-13 | Grid is in place only for its column passes; in-place transforms use scratch of N, a permutation pass or a pipeline; row splitting uses chunked stages | Independent review of the first draft |
| 2026-11-13 | Licence: Apache-2.0 | Permissive with a patent grant; compatible with GPL-3.0-or-later clients; the design takes only published ideas from GPL libraries and copies no code |
| 2026-11-13 | No header-only configuration; a compiled static library only, with an amalgamation possible later | The maintainer does not need it; it taxes all non-kernel code, weakens the floating-point guarantees and costs users minutes of compile time |
| 2026-11-13 | The width hint may change the program, with `strict_layout` to forbid it; float precision in phase 6; minimum Clang 16; cloud machines rented per experiment | Maintainer's agreement to the first draft's working assumptions |
| 2026-11-13 | Old x86-64 hardware: best effort. Old targets are timed forced on newer machines and verified by whole-machine emulation | No such hardware is available |
| 2026-11-13 | Published benchmarks: one generated report per tagged release; the tracking store stays internal | Keep the public record minimal and clean |
| 2026-11-13 | Minimum Clang 16 as the tested floor; development and release builds use Clang 23; older Clang is neither tested nor deliberately broken | The floor only affects third parties building from source with old toolchains |
| 2026-11-13 | Documentation: Markdown under `docs/`, the public headers as the API reference, compiled examples, Doxygen as a completeness linter only | Smallest thing that is exact and cannot rot; no documentation infrastructure |
| 2026-11-13 | Python 3.14 or newer for the generator and all tooling, through `uv` | Development runs on current machines; `uv` supplies Python on the clusters |
| 2026-11-13 | Reference DFT in quadruple precision through Boost.Multiprecision, test-only | Native quad types are missing with Clang on Apple silicon and on the MSVC ABI; an independent reference should not share the library's arithmetic |
| 2026-11-13 | Float is supported and kept alive from phase 1 by correctness tests in CI; tuned in phase 6 | It has users, and precision-generic code that is never instantiated drifts |
| 2026-11-13 | Bitwise schedule independence is kept | It costs nothing measurable (no cross-thread reductions) and makes threading tests exact; revisit only on experimental evidence |
| 2026-11-13 | fbench replicates over processes with paired intervals and an A/A control | Process-to-process spread dominated the prototype's noise |
| 2026-11-13 | fbench v0: noise band = 95th percentile of the A/A control's \|median log ratio\| (floor 1 %); double-double reference in the harness; wisdom fresh per run | Simulated, a band from the A/A intervals' reach was as safe but needed about twice the rounds for a 5 % effect; the quadruple-precision test reference needs Boost, which benchmark binaries should not; stale wisdom slowed FFTW 1.3× (E10) |
| 2026-11-13 | ISA regions confirmed on Linux x86-64; GCC builds use `-Werror=psabi`; no vector types as template arguments | Experiment B1: GCC silently miscompiles by-value vectors across a region boundary |
| 2026-11-13 | Platform readers are parsers over recorded sources (a one-file sysfs snapshot, a sysctl snapshot); `FFTN_TOPOLOGY` names a recording; detection results are function-local constants | The Linux, macOS and x86 logic is tested on any machine; one fixture file per machine rather than thousands; no mutable global state |
| 2026-11-13 | The planning clock is `std::chrono::steady_clock` (measured step 41 ns on Apple M5 Max) | Planning times whole executions of microseconds; a cycle counter adds per-CPU skew and frequency questions |
| 2026-11-13 | Vocabulary names `cv`, `rv`, `tw`, `C`, `W`, `tw_reals`, `fused` through `vec<T>`; the real-register family in namespace `real`; x86 FMA naming | Vector types must never be template arguments (B1); the two families share one machine type on SIMD targets |
| 2026-11-13 | Region text has internal linkage; only each target's function table is exported | Removes, by construction, the linker picking one target's copy of a shared inline function for another caller |
| 2026-11-13 | `neon_fcma` runs only where the processor has FCMA and pointer authentication; PAuth instructions are allowed in its code and nowhere else outside the hint space | Branch protection may emit RETAA in an Armv8.3 region (seen with GCC 16 and Clang 23); both features are mandatory from Armv8.3, so only rare FCMA-only cores lose the target |
| 2026-11-13 | Test tolerances: relative RMS error ≤ 3ε√L and largest error ≤ 3εL, L = max(1, log₂ n) | Measured FFTW and naive-DFT errors sit 3–10× (RMS) and 7–40× (largest) below; every seeded fault of task 0.4 is caught |
| 2026-11-13 | "One-ulp twiddle error" is tested two ways: entry-by-entry table checks for a single entry, output tolerances for an error propagated from the base twiddle | A single entry one ulp off changes outputs by less than correct rounding does, at any n; a propagated one grows like n |
| 2026-11-13 | FFTW, the test oracle, is built from source by the test build (or taken from `FFTN_FFTW_ROOT`), without UBSan | No system FFTW is assumed; FFTW 3.3.11's planners do null-pointer arithmetic that UBSan reports |
| 2026-11-13 | fbench competitors are optional backends, each configured by one variable (an install root, or a pinned fetch for header-only pocketfft) and never found on default paths; a library that handles some shapes only (Accelerate: rows of vDSP's lengths) declares those kinds `partial`, and smoke requires a pass of each; client catalogues live outside the repository | Handicapped system builds are a trap (Homebrew's FFTW); a backend that refused every case must not count as correct; the client's workload is private |
| 2026-11-13 | Walking skeleton (task 0.5): a program's stages read and write slots (caller in, caller out, scratch lines a and b, block); every transform is finished before the next, one axis at a time, one line at a time; each line is a Stockham DIF chain whose first pass reads the line where it lies and whose last writes it where it goes, with a and b in between; stages carry chunks of about 2048 points | The smallest structure that serves every kind, rank, stride and placement with one driver, and whose loops and chunks phase 3 can share out unchanged |
| 2026-11-13 | The generic prime pass sums each output pairwise (a binary counter of partial sums) | A sequential sum's error grows like p: it fails the RMS tolerance for primes of a few hundred, which the suites test |
| 2026-11-13 | The chain passes exist only on the scalar target in phase 0; the per-target table carries them as a member that is null elsewhere, and a plan takes the fastest target whose table has them | The skeleton's passes assume one complex per register; phase 1 replaces them with generated, lane-aware ones on every target |
| 2026-11-13 | Twiddle tables hold forward roots only, computed in double-double with Dekker's exact product (no FMA) and rounded once; backward passes multiply by conjugates | The tables are then the same on every processor and build; conjugation is exact, so backward tables are correctly rounded too |
| 2026-11-13 | A multi-dimensional c2r out of place keeps its intermediate in a workspace block of one transform's half spectrum (a whole volume's, for now) | The input is preserved by default (9.1); pencil-block scratch arrives with phase 4's volumes |
| 2026-11-13 | Kokkos core 5.2.1, Serial back end, is a test-only dependency: fetched pinned by URL and SHA-256 (the release's own checksum file, checked against the download), built as a CMake subproject, SYSTEM, in every gate stage that builds; the library still has no dependency, since `fftn/kokkos.hpp` is header-only and compiled in its user's translation units | The adapter's call sites must compile and run in the gate so they cannot rot (task 0.5b); a Serial-only Kokkos builds in seconds |
| 2026-11-13 | No `omp simd` on the scalar pass loops; old-ISA targets (`x86_v2`, `avx`, plain `neon`) get hand vocabularies from the start, not the scalar text compiled in their regions | Experiment X15: `omp simd` gained 5.5 % with Clang on the M5 but cost 2.0× with GCC 16 on Zen 3; the scalar text in the `avx2` region stayed 2.2–3.7× slower than the hand vocabulary per pass, and `-march=native` closed none of that gap |
| 2026-11-13 | Windows on Arm reads FCMA and pointer authentication from the registry copies of the ID registers | No `IsProcessorFeaturePresent` flag exists for either (SDK 10.0.26100) and MRS faults; on the CI's Cobalt 100 the registry fields agreed with executing FCMLA and XPACI (B2) |
| 2026-11-13 | Emulated CI runs the instruction-set test programs only, on `main`, weekly and on demand | It checks instruction-set safety, which branch pushes rarely change; the full suites under emulation took over an hour per processor model (maintainer, after the walking skeleton) |
| 2026-11-13 | API review (task 0.9): the maintainer took every recommendation of the review of DESIGN 18.2 items 1–35 (the list at commit db2183d). Accepted as built: 1, 3, 4, 6–21, 23–32. Changed: axes and batch dimensions become `std::vector` and `small_vector` goes (2); the shape takes more batch dimensions, so Views with more than two non-nesting loops work in the adapter (35); `execute_rows` becomes `execute_transforms` (5); the adapter's class becomes `fftn::kokkos::view_plan`, with `compat::Plan` as its alias (33); `shape_type` lives only in `compat` (34); `compat::Direction` takes Kokkos-FFT's values 0 and 1 (22, so that nothing in `compat` differs from Kokkos-FFT on purpose) | Plan construction allocates anyway, so an inline container buys nothing and freezes the rank limit into the type; names should be exact before anyone depends on them; the adapter should stay a faithful mirror of Kokkos-FFT, which keeps open an FFTN host backend inside Kokkos-FFT later |
| 2026-11-13 | FFTN is unreleased: until the first tagged release the API may change in any way, and a wrong choice is fixed rather than kept (README, AGENTS section 1, section 5) | Maintainer's decision at the API review |
| 2026-11-13 | Deferred, not rejected: a two-way plan (forward and backward over one table set) and an object owning a plan and its workspace | Small gain for more API surface; both can be added later without changing the API as long as twiddle tables stay direction-independent and a workspace stays plain bytes sized by `workspace_bytes()` (18.2) |
| 2026-11-13 | `compat::Direction` is its own enumeration, declared as Kokkos-FFT declares it (underlying `int`); `view_plan` has a twin of each constructor taking it, which converts it to `fftn::direction` and refuses any other value naming `dir` (task 0.10) | Keeps `compat::Plan` an alias of `view_plan`, as the review asked, rather than a second class; a cast-in integer is refused as an invalid `fftn::direction` is |
| 2026-11-13 | Codelet calling convention: `cdft<R>::plain`, `dit` and `dif`, static member templates over `T` and `Sign` (−1 forward, +1 backward), taking the caller's accessor objects `in.get(j)`, `out.put(k, v)` and `twid.get(k)`; twiddles are forward roots, multiplied by `mulc` backward (gen/README.md) | One codelet text then serves every pass kind, edge variant and layout, which differ only in their accessors; stores stay next to their producers inside the codelet; the accessors hold pointers, never vectors, so no vector crosses a non-inlined boundary |
| 2026-11-13 | Task 1.1 generates the complex-register lowering only, for radices 2–16: plain, DIT and DIF, forward and backward. They are compiled and tested in test-only region units (`tests/isa/codelets_<target>.cpp`) until task 1.2's passes instantiate them in the library | The split lowering, pruned and fused-edge variants are tasks 1.5 and 2.1; codelets nothing calls do not belong in the library's units |
| 2026-11-13 | The generator orders codelets by greedy list scheduling for fewest live values, not by the recursive partition of section 7.2's first draft | Measured over radices 2–16, all variants, both directions, each order with the same load and store placement (commit aa966ef holds the partition and the test): a total peak of 1004 live values for the greedy order, 1196 for the partition and 1098 for the builders' own creation order, and the greedy order no worse than either on any codelet; a test keeps holding it against creation order |
| 2026-11-13 | Operation counts are compared with the best published counts in unfused real arithmetic; a size the generator misses is recorded with its exact counts and reason (`gen/fftn_gen/reference.py`); at task 1.1 only 13 is (Rader's algorithm, 176 + 68 against 192 + 134). A size with no published count gets a reference derived from its parts, or none with a reason | A gap is never widened silently and a closed gap is noticed; the mean removal in the prime form beats the reference at 7, 11 and 14 |
| 2026-11-14 | Team execution (tasks 3.1, 3.2): `team{rank, size, group_size}`, `completion::all` by claiming and `completion::own` by a static deal of whole transforms, `plan::transforms_of`, `workspace_bytes(lanes)`; the team state lives in the workspace and persists between executions, numbered per member, with every counter tagged by the execution that last wrote it | Late members need no barrier and nothing is reset between executions; whole transforms keep owner-computes simple; open points for review in 18.2 |
| 2026-11-14 | Team state words are `std::atomic<std::uint64_t>` objects constructed in the workspace when the state is set up, not `std::atomic_ref` over bytes | `atomic_ref` is missing from libc++ before 19 (Clang 16 with libc++, older Apple Clang); constructed objects also have a lifetime the standard recognises |
| 2026-11-14 | Team state tags carry 40 bits of a 64-bit epoch, compared only for equality, with 24-bit counts; whether an execution is over is a separate untagged word holding the highest epoch a member has returned from | 32-bit epochs compared modulo 2^32 failed after 2^31 executions (hours at microsecond executions): never-written words read as later (second reader of tasks 3.1 and 3.2) |
| 2026-11-14 | The scheduler is computed by every member at the start of every execution, from the work graph, a lane map and a machine profile of provisional constants | It is a few dozen operations and needs no shared storage or cache; calibration replaces the constants (task 3.4) |
| 2026-11-14 | A lane pass reads its shared twiddles with a new vocabulary operation `bcast_tw` from tables holding one twiddle per entry (broadcast storage written by `pack_tw1`), not from tables replicating each twiddle C times (18.2 generator (c)) | C times less twiddle memory and traffic (up to 64 bytes per twiddle on AVX2 float otherwise); FFTL.jl measured broadcasting twiddles from memory per vector as its best lane-pass form. Reversible: replicate the entries with `pack_twiddles` and read them with `load_tw` |
| 2026-11-14 | Pass kernels keep the entry `run(args, first, last)`; a target exports them in a table indexed by codelet radix, direction and edge operation, filled from the generated codelet table; edge operations (scale, factor) are instantiations applied by accessors; a unit is a block of C butterflies (grouped), a group and block of C copies (lane, generic), a line (leaf) or an element (copy); tails run the same instantiation on a padded tile | One codelet text serves every kind and variant; no run-time flag in a loop; units aligned to C keep chunks whole vectors; the tile keeps each element's arithmetic identical wherever its row sits (15.5) |
| 2026-11-14 | Planner v0: radices and order from FFTL.jl's single-line rules, keyed on the target's register count and whether it merges radices (AArch64) | The plan's starting point (PLAN task 1.2); the rules were measured on M-series and x86 machines; X2 revisits them per target |
| 2026-11-14 | On targets with C > 1, a line whose stride is not 1 is copied to or from scratch around the chain; leaves and copies take any stride | Passes then load and store whole vectors only; strided axes get the lanes strategy in phase 4, which removes the copies |
| 2026-11-14 | `options::scale` is fused into the pass that writes the output, so the result is bitwise the unscaled result times the scale; `options::force_isa` with `isa::best` as in section 5.3 (pending the maintainer's agreement, 18.2) | A fused scale costs no pass (task 1.2's acceptance, tested); forcing is how the suites run every target |
| 2026-11-14 | `avx512` (task 1.3): 512-bit registers for both precisions (C = 4 `double`, 8 `float`), target string `"avx2,fma,avx512f,avx512dq,avx512vl"`, twiddles pre-arranged as on `avx2`, `addj` and `subj` by `fmaddsub` and `fmsubadd` by ones, `transpose` by `vshuff64x2` (4 × 4 blocks of 128-bit units, and for `float` after an unpack into such blocks), every operation rounding as on `avx2` | The targets table (6.1); 4 complex numbers per register and 32 registers are the target's point. Recorded, not decided: Skylake-SP and Cascade Lake lower their clock while 512-bit floating-point instructions run (the AVX-512 licence levels), which a 256-bit form of this vocabulary (EVEX at ymm width, 16 or 32 registers) would avoid at half the width; Zen 4 executes 512-bit operations as two 256-bit halves without a licence penalty. Which form wins on which machine is an fbench measurement on the Xeon and a Zen 4, not made here |
| 2026-11-14 | Plain `neon` (task 1.3): no attribute (Advanced SIMD is the AArch64 baseline); twiddles pre-arranged as on x86 (`tw_reals` 4); products in FCMLA's rounding order, so `neon` and `neon_fcma` give the same bits; the operations both NEON targets share live once, in `vocab_neon_common.inl`; chosen wherever FCMA or pointer authentication is missing (Neoverse N1, Cortex-A72, A64FX, Windows on Arm without the registry fields) | Three instructions per product (shuffle, multiply, FMA) with no sign fix-up, where plain (wr, wi) storage needs a sign flip of a shuffled value per product (four) or two shuffles more for `float`; twice the twiddle bytes, as on x86, for the planner and X5 to weigh. The same bits make `neon` a pure fallback of `neon_fcma`: with the same passes (and the two share a planner profile, so they choose the same ones) every output bit is equal; only what reports the target, such as `instruction_set()`, tells them apart |
| 2026-11-14 | No R × C `transpose` for radices below C (avx512); the grouped pass stores them through its tile | Correct as it stands. Performance risk recorded: planner v0 ignored C (planner v1 now prices tails per C, with AVX2 costs), so on `avx512` most lengths started with radix 4 (or 2 or 3), and in `float` (C = 8) their grouped pass stores through the tile and the second pass (s = 4) runs only tail units. Preferring a leading radix of at least C, or an R × C transpose, is a planner change and needs its experiment (X16, `avx512` against `avx2`), not a rule edited here |
| 2026-11-14 | fbench times single precision: input rounded to float, reference and check in double with float's epsilon; backends opt in with `precisions` in describe | Experiment X2 and the float targets need float timings; rounding first keeps the reference exact for the values the library sees, and opting in keeps competitors without float from failing smoke |
| 2026-11-14 | Threading adapters (task 3.3): the `executor` concept (`concurrency`, `bulk`, optional `group_size`) behind a non-owning `executor_ref` that is the C contract's struct (`fftn/executor.h`); the executor overload dispatches one task per lane dealt work, as a team; `fftn/omp.hpp` (`this_team`, an executor of constant width); the opt-in `thread_pool`; `cache_group_size` | One allocation-free path from any runtime to the claiming engine, with no template in the library; dispatching only lanes with work is rule 4, and every dispatched task runs, so no member skips an execution; open points for review in 18.2 |
| 2026-11-14 | The OpenMP tests are built wherever the compiler has OpenMP, except under ThreadSanitizer; the thread pool and custom executors carry the executor path's TSan coverage | The OpenMP runtimes at hand are not instrumented, and TSan reports the join of every region as a race (shown with Homebrew LLVM 23's libomp) |
| 2026-11-14 | fbench measures on loaded machines: load is recorded per round, not a reason to invalidate it; a round waits at most 60 s, and only while other processes use more than half the CPUs; `--strict-quiet` keeps the old rule; a process timeout loses that process's unfinished cases for the round, not the run | Maintainer's direction: the shared machines will never be much quieter, and interleaving, repeated rounds and the A/A control already measure the noise the load adds |
| 2026-11-14 | Planner v1 (experiment X2): the cheapest radix sequence under a per-profile table of measured pass costs (per unit and per padded tail, for first, middle and last passes, plus a cost per pass), found by a shortest path over the divisors of the length, with a short table of measured overrides; profiles carry the precision's lanes C; AVX2's tables (C = 2 and 4) serve every target by nearest C until measured there (section 11.2) | On the EPYC 7763 it reproduces 95.7 % of the measured best sequences and no case is more than 2 % slower than v0, against v0's geometric-mean deficit of 1.25 (double) and 1.44 (float) up to 1024; structural rules (fewest passes, a first radix that is a multiple of C) reached at most 72 %, a cost by operation counts 79 % |
| 2026-11-14 | Bluestein (task 2.3): the product of the prime factors above the profile's threshold is one stage at the end of the chain (m = 1): a chirp pass, the chain of M over the s copies forward with the kernel spectrum as its last pass's output factor, the chain backward, a chirp pass; M the cheapest 7-smooth length in [2Q − 1, next power of two]; the threshold is 16 on every profile (X17) | With m = 1 the chirp passes are elementwise and the s copies are a chain of M with s times the copies, so the stage reuses the chain's passes and slots with no recursion and runs in place; on the Mac it took the primes from 4.3–5184 times FFTW's time to below 1.3 times (section 8.5), and beat the generic pass at every prime from 17 |
| 2026-11-14 | A Bluestein stage stores its chirp (Q entries, or sQ when C does not divide s) and kernel spectrum (M entries) once, shared by the copies and by the axes of a plan; the kernel is the forward chain's last output factor when C divides s, else the backward chain's first input factor (task 2.3b, X17 follow-up) | Per-element tables took 889 MB at n = 17 · 2²⁰; shared, under 1 KB, at the same speed or faster (to 0.84 times with many copies); the input form alone was about 5 % slower where the last pass could take it |
| 2026-11-14 | NEON cost tables (C = 1, 2) from X2's Mac campaign; plain `neon` uses them | Against the AVX2 tables NEON had used, 3.6 % faster on the Mac over 496 cases from 17 to 1024 (124 faster, 9 slower, band 1.7 %, run 20261114-101523-3d56); the three lengths that were 1.25–1.40 times slower keep the old choice as overrides |
| 2026-11-14 | Planner v1's cost tables are keyed by target and C; a target and C without one take AVX2's table of the nearest C, as before. The X2 fit is a tracked tool (`bench/x2/fit.py`) | An `avx512` table (X16) and NEON tables (X2 on the Mac) drop in as rows; the search and every current choice (`rule_radices.txt`) are unchanged |
| 2026-11-14 | Real transforms (task 2.1, section 9.1): even lines read their reals in place as complex values; the pairing of bins k and h − k is fused into the r2c chain's last pass or the c2r chain's first pass (mirror copies or groups together, each pair formed once) when that pass's radix is at most 8 (4 on avx2) and for r2c the half spectrum fits L1, a leaf for h ≤ 16, and otherwise a separate pass in place on the complex side; long rows run the half-length transform as a grid with a separate pairing; the c2r pairing fuses into the DIF chain's first pass, so no DIT chain is built | Fused and separate pairing measured against each other on the Mac (X6): fusion wins by 10–15 points of overhead with radix 8 in L1 and loses with radix 16 or beyond L1 (register spills, conflict misses across 4R streams a power of two apart); the old route's pack, scalar post and unpack passes cost 50–165 % over the half-length complex time |
| 2026-11-14 | The grid strategy (task 1.7, section 8.5): decimation in frequency in two or three blocked phases, gathers and scatters around a chain of lane passes on tight scratch, the transpose folded into phase 1's store; chosen for c2c rows beyond the L1 data cache, three phases beyond an eighth of L2; loops carry a twiddle cursor | X12: on the Mac the chain ran 1.85–2.84 times FFTW's time from 2^14 to 2^22 points; the grid ran 0.47–0.71 of the chain's time there, 1.08–1.43 times FFTW's (fbench run 20261114-110844-8b03). Two phases lose beyond L2 because their blocks must narrow to keep a phase in L1; three lose while the row fits L2 |
| 2026-11-14 | The serial driver runs a stage's chunks as one call and divides only when a range starts mid-stage; a chain uses a contiguous, unaliased destination as its second scratch line (experiment X18) | 64-bit division costs 40–90 cycles on Cascade Lake (37 % of a 64-point transform there); three lines instead of four keep 512–1024-point rows in a 32 KiB L1 longer. Xeon 1.50 → 1.25 of FFTW, EPYC 1.37 → 1.29; the driver and scratch changes keep every bit, the AVX2 overrides for 4096 and 16384 (also reached by C ≤ 2 targets without a table of their own) change those lengths' radices and so their bits |
| 2026-11-14 | Grid phases whose radices are all at most 8 read and write their rows in their end passes rather than through a gather and a scatter, for rows up to `grid_fuse_bytes` (a quarter of L2); elsewhere the copies stay (task 1.7b) | X12 (Mac, runs 20261114-131752-7e99 and -132841-80e0): 2^18 from 1.03 to 0.92 times FFTW; radix 16 at the ends doubled the time (set conflicts), one fused end gained nothing, and beyond L2 the copies' prefetch won (2^20 1.46 fused against 1.35) |
| 2026-11-14 | The grid's thresholds follow the L2's sharing: with an L2 of each core's own the grid starts beyond L2/2 and three phases beyond L2 (were L1 and L2/8); with a cluster-shared L2 they stay L1 and L2/8 | X12b (Xeon Gold 6240 avx512 and avx2, EPYC 7763; Mac check): 0.83–0.90 of the old default's time on rows between L1 and L2, 0.91–0.96 over all its cases, long rows unchanged; on the Mac a chain at 2·L1 took 1.67 times the grid's time, so no rule in the sizes alone fits both |
| 2026-11-14 | The grid's scratch line b is staggered from a by seven 64-byte lines (rule 4 of section 10); the grid's intermediate array is not padded | X4 (Mac): the stagger gained 7–9 % at 2^22 and lost nowhere; padding the intermediate (one line per row) added nothing separable, and reading padded rows directly with radix 16 lost 5 % at 2^22 |
| 2026-11-14 | Codelets of radix 32 and 64 (task 1.6) in every variant, with all their passes instantiated; the pass table has one row per generated radix (`codelet_slot`), not one per radix value; a planner profile has a leaf limit, 16 by default, above which a codelet radix is a leaf only when the limit allows (`build_settings::leaf_limit`, fbench `LEAF_LIMIT`) | Leaves above 16 and big passes (experiments X8 and X3) need the codelets; their operation counts equal FFTW's n1_32, n1_64, t1_32 and t1_64. They roughly double the kernel code (B3: 33 % of the pass instructions are radix 64's, 14 % radix 32's; numbers in X8), kept for speed over build size: the grid and X3 may use them as passes |
| 2026-11-14 | Chains on `avx2` and `avx512` (both precisions) end in a split run: their last passes on split codelets in the BL layout, converting at the run's ends; NEON and scalar stay interleaved. A run starts where the copy count reaches `split_block`, `avx512`'s W on both x86 targets (task 1.5, experiment X1) | X1: split runs 0.89 (geomean of their cases) of the interleaved time on the Xeon's `avx512`, 0.87 on its `avx2`, 0.93 on the EPYC, nothing slower than band + 2 %; port-5 uops −12 % at 1024; on the Mac 10–25 % slower. The shared `split_block` keeps the bitwise pair agreeing with the same radices, at the cost of no run on AVX2 where the first radix is 4 (1024 = 4 × 16 × 16) |
| 2026-11-14 | Leaf limit 32 on NEON in double (C = 1), 16 on every other target and precision; no leaf of 64 (experiment X8) | The leaf of 32 was 16–22 % faster than planner v1's chain on the M5 Max in double; with C > 1 it idles lanes and lost on the Mac and the EPYC; 64 spills and lost everywhere. The Xeon's `avx512` float chain at 32 lost to the leaf only through its own tail problem (X16) |
| 2026-11-14 | Machine profile from T1 and T5 (task 3.4): split a lone transform from 6144 points on a Zen 3 CCX (the defaults), 8192 on an Apple cluster (`profile_for`, from macOS's topology); mesh groups keep 8192; leftovers keep 8192; at least 1024 points per member; on a CCX, phase hand-off 1100 ns and claim runs of left / members; dispatch, work cost and spin budgets stay provisional | T1/T5 per-field and combined adoption runs on the EPYC, the M5 and the Xeon (numbers in the experiment files); the measured dispatch and work costs made cases slower until the dispatch cost has a per-lane term (18.2, item 22) |
| 2026-11-14 | Flat programs and leaves called with addresses (experiment X20): rows of a few stages skip the record walk; a leaf is a `leaf_fn` taking its lines' addresses and distances, so a batch of leaf rows is one call and no leaf needs a copy of `pass_args` | On the M5 Max the walk and the 144-byte argument copy cost about 3.5 ns per transform and the entry about 1 ns per call, against a 2.5 ns 8-point leaf; the record walk stays for programs with loops (grids, planes), which the threading work graph reads unchanged |
| 2026-11-15 | Intel's icpx is a supported compiler; FFTN's targets are built with `-fno-fast-math -ffp-contract=off` everywhere, and an icpx build adds `-fno-fast-math` to every target and to the FFTW oracle (experiment X22) | icpx's default model reassociates, drops signed zeros and flushes denormals without defining `__FAST_MATH__`; with only `-ffp-contract=off`, 55 of 1608 tests failed (twiddles 1 ulp off, Parseval, vocabulary references), all pass with the flag. `-fp-model=precise -ffp-contract=off` draws an override warning that `-Werror` makes fatal. Not recommended over GCC or Clang: equal to Clang in double on the Xeon, 3 % behind it on the tracking rows; its float lead (0.80 of Clang on 7-smooth rows) comes from masked partial runs, which the vocabulary can write explicitly |
| 2026-11-15 | The lanes strategy for batches of short rows (task 2.2, section 8.5): blocks of B rows of the innermost batch dimension as a tile on which the row's own chain runs with B times its copies, its first and last passes reading and writing the rows where a run of C rows is one access, a gather and scatter of C × C transposes elsewhere; the rules choose it up to `lanes_up_to` per target and C (X7): float on x86 only (AVX-512 100, AVX2 32) | Every row keeps the chain's bits, so the choice changes no result. Against the chain before X20 the lanes won widely (0.13–0.59 of its time at 8–16 points on all three machines); against the chain with X20's batched leaves they still take 0.32–0.55 of its time at 8–16 points in float on x86, but lose at 8 points in double and on NEON (1.09–1.66), so the rule (no slower case up to the threshold) leaves them off there |
| 2026-11-15 | Each ISA region's kernel unit is split into a main unit and five row units (`rows.inl`, `table.inl`; pass kernels hold codelet pointers); per-push CI runs one configuration per platform plus Clang ASan + UBSan and TSan on Linux x86-64, and every other job (second GCC, Apple Clang, clang++ on Windows, the other sanitizer jobs, clang-tidy, coverage) runs weekly and on demand (`extended.yml`); CI on a pull request is the merge gate | One unit per region needed more memory than the 16 GB runners have once the radix 32 and 64 codelets landed (sanitizer builds were OOM-killed); macOS LLVM ASan alone took 61 minutes per push. Local gates are kept for the remote machines' toolchains when a change needs them |
| 2026-11-15 | Planner refresh: the measured radix tables are refitted after X1, X3, X18 and X19 on every machine; AVX-512 has its own tables in both precisions (float a fresh fit with 35 overrides, double AVX2 float's costs with 15 overrides); `avx512` stays the first target in double, and float's per-plan target choice is deferred (X16, Cascade Lake only) | A/B over the changed lengths: Xeon AVX-512 0.90 of before (float) and 0.94 (double), EPYC AVX2 0.94–0.95, Mac 0.98; the tracking tier unchanged within noise (0.994–1.001). Overrides measured slower inside another path (Bluestein lengths, r2c planes) are dropped |
| 2026-11-16 | Codelets of radix 17, 19 and 23 (generator catalogue, a row unit each per target); n = p is a leaf and those factors are codelet passes after the priced radices, no longer Bluestein stages (experiment X27) | 0.24–0.31 of the Bluestein route over rows, composites, batches and real rows on the Mac, the EPYC and the Xeon, both precisions; c2c 17 from 1.6–2.0 to 0.28–0.47 of FFTW; 25 % more library code (B3) |
| 2026-11-16 | A Rader stage replaces the Bluestein stage for a single prime q with q − 1 13-smooth inside the profile's range (from 16C when C ≥ 4; below 2¹⁶ on NEON and scalar, 2²⁴ on x86), sharing the Bluestein stage's convolution (X27) | Half the convolution length: 0.4–0.8 of Bluestein's time from 97 up on all three machines; small primes lose with C ≥ 4 (scalar gather of a single row) and 65537 loses on the Mac (random access over a megabyte) |
| 2026-11-15 | The grid's copies (`block_copy`, `twiddle_scatter`) do not prefetch rows ahead on `avx2`; `avx512` and NEON keep the prefetch of eight rows ahead (experiment X26) | EPYC 7763: target rows 0.920 of main's time without it (a block's rows lie a multiple of 4 KiB apart and share one or two 8-way L1 sets, so rows asked for ahead evict the scratch block); Xeon `avx512` 1.04 and Mac 1.01 slower without it. Cheaper copy loops (24-37 % fewer instructions) were rejected: within the band on the Mac and the Xeon, up to 1.56 slower on the EPYC |
| 2026-11-15 | Masked partial runs on x86 (X23's arm P5: `load_n`/`store_n`, partial units after the loops, padded transposes, `avx2` stores in pieces) are built behind `FFTN_MASKED_TAILS`, `AUTO` by default: on with GCC, the x86 release toolchain, and off with Clang and icpx, which keep the tile (sections 8.2, 14.3) | The maintainer adopted X23 for GCC after it was rejected for a Clang-only loss (Clang 23: `avx2` double 75 points 1.42), under the rule that decisions judge the release toolchains. Amendment 4 with GCC: float 0.53 of main's time on the Xeon's `avx512`, 0.77 on `avx2`, 0.72 on the EPYC, double 0.78–0.89, bits unchanged; off, the kernel objects are byte for byte main's. Its rule was missed by one whole-unit case on two machine-targets (float 256 1.047 on `avx512`, double 72 1.033 on the EPYC), kept as recorded exceptions (the lead's decision under the maintainer's direction to keep X23 for GCC; neither case has a partial unit) |
| 2026-11-15 | The x86 targets' leaves (n ≤ 16) are 128-bit: compiled in the `avx2` region with `avx2`'s operations at half width (`vocab_avx2_narrow.inl`), used by `avx2` and `avx512` plans alike (experiment X28) | A leaf broadcasts its one line into every lane, so width only repeats work; on Cascade Lake the 512-bit leaf ran at licence 1–2 (Xeon 8 points 65 → 47 cycles, 16 points 110 → 85); same operations per lane, so bits unchanged |
| 2026-11-15 | A pass kernel takes its source and destination as arguments (`pass_fn(args, src, dst, first, last)`), like a leaf; `pass_args` holds only the fixed geometry, and the driver calls kernels on the plan's own arguments; one transform of a flat program runs from a contiguous `flat_body`, entered from `execute_core` with its slots as scalars; `plan::execute` is one library call (X28) | The per-stage, per-transform copy of `pass_args` cost 3.7–6 ns per stage on Zen 3 (much of it store-to-load forwarding into the kernel), 1.5–2.7 on the M5 Max: single rows of 24–120 points 0.69–0.88 of their time on the Mac; only grid stages with moving twiddles and lanes blocks still copy |
| 2026-11-15 | Where the L2 is shared (Apple), a c2r row out of place whose half-length transform is a two-phase grid pairs its bins in phase 1's gather (`c2r_pair_gather`, `c2r_paired_gather`), reading the input only; x86 keeps the separate pass (EPYC: 1.11); in place and three-phase grids keep the separate pairing pass; the r2c pairing stays separate (section 9.1) | X29 (Mac): 0.92–0.94 of the separate route's time at 32768–131072 and 46080 out of place, where the workspace block goes; 1.02–1.06 where the block stays (the gather reads the input twice) |
| 2026-11-15 | Where a cluster shares L2 (Apple), a row whose length is not a power of two stays a chain when its power-of-two part is at most L1/8 in bytes, or the row at most 2 L1 (`chain_part_bytes`, `chain_mixed_bytes`; section 8.5) | X29 (M5 Max): chains took 0.49–0.72 of the grid's time for every such length tried (144 KiB to 62.5 MiB), the grid's L1-conflict problem being one of powers of two; tracking tier 0.983 (46080 0.63, 100000 0.60), nothing slower |
| 2026-11-16 | With a private L2 and less than 2 MiB of L3 per CPU (Xeon 6240), a row whose length has an odd part of at least 15 stays a two-phase grid at any size (`grid_two_odd_part`; section 8.5) | X34 (Xeon): two phases took 0.70–1.01 of three phases' cycles on the 15 such rows tried, 71680 to 4 × 10^6 points; fbench (run 20261116-125601-0542) 0.86 over the 14 changed rows, 100000 0.71, 10^6 0.72, none slower than 0.996; chains lost 1.05–1.45 there; the EPYC (two phases 1.13–1.67 below 16 MiB) and the Mac change no plan |
| 2026-11-16 | On x86-64, a private L2 of 2 MiB or more (Granite Rapids) gets its own grid profile whatever its L3 per CPU: no X29, X32 or X33 chains, the grid and three phases beyond L2/2, two phases where the power-of-two part is at most 2048 complex doubles (`grid_two_part_bytes`; section 8.5) | X35 (Granite Rapids VM, runs 20261116-192413-6457, -193919-29c3, -195152-5453): 30 changed cases 0.59–1.007 of main, tracking 0.967, none slower; first variants lost on 71680 (1.10) and c2r out of place (1.15–1.16); Axion (same L2) lost 1.15–1.28 and is excluded; the Mac, EPYC, Zen 5 and Xeon 6240 profiles unchanged |
| 2026-11-16 | With a private L2 and less than 2 MiB of L3 per CPU on x86-64 (Xeon 6240), c2c powers of two are grids beyond L2/4 (`pow2_grid_bytes`), two-phase grid blocks fill twice L1 (`grid_two_block_bytes`) and three-phase blocks are at most 16 columns (`grid_three_block`; section 8.5) | X39 (Xeon, run 20261116-220608-84fe): three-phase blocks 0.935 over the 4 changed rows (131072 0.93, real 262144 0.91–0.94); two-phase blocks and the 32768 grid 0.958 over 7 (32768 0.90, 46080 0.94, 65536 in place 0.97, none above 1.002); radix-64 phases rejected (0.985, c2r 2²⁰ up to 1.027); tracking tier 0.992, none slower than 1.028; other recorded machines unchanged (profile tests) |
| 2026-11-17 | On `avx512` in double, lengths 300–25000 with 2² · 5 to spare start with a radix-20 pass (overrides; c2r lines keep the costs' sequence); on the Xeon profile, radix-20 grid phases, r2c powers of two as grids beyond L2/4, and c2c powers of two out of place in two phases beyond L3/2 (section 8.5) | X39 arm D (Xeon, run 20261117-033154-673d, `--serial`): 0.905 over 26 changed cases, none above 1.010; the first form lost on c2r (1.19–1.25) and planes of 1000 (1.02–1.04) and was narrowed; the overrides reach Zen 5 and Granite Rapids unmeasured |
| 2026-11-16 | With a private L2 and at least 2 MiB of L3 per CPU (Zen), a row whose length is not a power of two and has a power-of-two part of at most 1024 points stays a chain up to half of L3 (section 8.5) | X29 (EPYC 7763): 0.64–0.85 of the grid's time, steadily, at 46080 to 2 Mi points; power-of-two chains of 1–2 MiB were bimodal (0.7–1.8 by process) and keep the grid; the Xeon 6240 (0.7 MiB of L3 per CPU) keeps L2/2, where every chain lost 1.1–1.6 |
| 2026-11-16 | Where the L2 is shared by a cluster (Apple), an intermediate the grid already keeps in the workspace block is padded after each of phase 2's rows (`grid_pad`, profile `grid_pad_period`); off on x86. The copies' prefetch keeps X26's target gate: keying it to the core was rejected (experiment X26b) | Mac: double long rows 0.976 of main (2^18 0.94); Xeon 1.006 and EPYC 0.993, where counters showed no fewer L1 refills. The prefetch on an Intel core under `avx2` took 2^22 to 0.84 but float 100000 to 1.27; X26's cheaper copy loops were rejected again on the padded grid |
| 2026-11-16 | Pair leaves on x86 (experiment X28b): a contiguous c2c row of n = r1 m with C dividing m and about n / C registers runs the chain r1 × m as one leaf, the grouped pass's C × C transposes held in registers as the last pass's input; a table of lengths per target and precision (`src/isa/pairs_<target>.cpp`); the lanes strategy runs the same chain on its tile | Single rows of 20–120 points 0.42–0.69 of the chain's time on the Xeon 6240 (`avx512`), 20–48 at 0.51–0.76 on the EPYC 7763, float 0.39–0.78; batches of 24 and 48 points 0.43–0.70; Xeon FFTN ÷ FFTW over single rows 0.98 → 0.65 with the other two X28b arms; +0.75 % library; the bits are the forced chain's |
| 2026-11-16 | Leaves with constant leg offsets for lines of stride 1 (`leaf_unit`, `src/passes/unit_leaf.inl`), on every target, and the x86 narrow leaves extended to 17, 19 and 23 (X28b) | 8–23-point single rows 0.75–0.92 of their time (EPYC, Xeon, M5 Max), bit for bit the same; 17–23 a further 0.85–0.99 on x86 from the 128-bit leaf; the x86 targets keep only the narrow unit-stride leaves (their own were 3.5 % of the library unused) |
| 2026-11-16 | More pair leaves (X37): `avx2` 8, 12, 16, 18, 22, 26 in double, also taken by `avx512` plans for 8, 12, 18, 22, 26, 30; `avx512` 16 = 4 × 4; codelet lengths up to 16 take a pair leaf before the narrow leaf; NEON pair leaves (C = 1) for 18–30 points with the rules' factors. On AArch64 only, leaves step their batch distance in bytes and `execute_core`'s general path is out of line | Xeon 18–30 points 0.40–0.54 of their time, 16 × 16–256 0.59 (÷ MKL 1.58–1.70 → 0.94–1.01); EPYC 18–26 0.49–0.54, 8–16 0.75–0.97; Mac 18–30 0.58–0.84 (20 points 1.58 → 1.04 of FFTW), single rows of 2–6 0.83–0.89 from the lighter path; library x86 −0.2 %, arm64 +0.2 %. `avx512`'s own 2 × 4 and 3 × 4 rejected (in-place batches 1.5–2.1: partial stores read back); the general path out of line rejected on x86 (EPYC single rows of 31–300 points 1.03–1.13) |
| 2026-11-16 | AVX2 double plans 256 as 8 · 32 and 512 as 16 · 32 (overrides of `avx2_double`) | X30: two passes against the three the costs chose (2 · 16 · 8, 4 · 8 · 16): EPYC 0.90 of the time at 256 and 0.87–0.95 at 512 (rows, batches of 64, the real transforms whose half-length chains they are); X30's other arms, x86 twiddles stored (wr, wi) and 1024 as 16 · 64, were rejected by their rule |
| 2026-11-16 | Plan tables (twiddles, kernels, indices) start on a 64-byte boundary (`aligned_vector`, `platform/aligned_allocator.hpp`) | X30b: `std::vector` put them 16 bytes into a line, so half the per-lane twiddle loads crossed one (up to 4135 per transform at 3072 on the EPYC); aligned, x30 double 0.986 (EPYC), 0.989 (Xeon `avx512`), 0.996 (`avx2`), float 0.97–0.98, the Mac neutral; bits unchanged. The half-size (wr, wi) storage per table kind was rejected on single cases (X30b) |
| 2026-11-16 | Long power-of-two c2c rows stay chains on Apple (to 4 L2) and on Zen-like caches (private L2, ≥ 2 MiB of L3 per CPU; to 2 L3), with FFTL.jl's radix sequences for power-of-two chains beyond `grid_min_bytes` (radix 8, radix 4 beyond L2/8 on Apple); the Xeon keeps the grid; real rows unchanged by X32 and included by X33 (section 8.5; idea from FFTL.jl's `src/factor.jl`) | X32: Mac tracking 0.968, the long powers of two 0.57–0.94 (FFTN ÷ FFTW 1.16 → 0.87, ÷ FFTL.jl 1.34 → 1.00); EPYC 0.57–0.85 of the grid; Xeon chains 1.06–1.50 (rejected there); X29's bimodality explained (radix-16 passes on huge pages) |
| 2026-11-16 | An r2c row's grid pairs its bins in its last phase, over pairs of mirror column sets (half a block each, at least a line), except on `avx2` and where the last phase is fused | X33: the separate pairing pass over the half spectrum was 0.03–0.06 of the row's time on the Mac and the Xeon (r2c 0.94–0.98 there), nothing on the EPYC (0.98–1.02); narrower or wider sets lost (EPYC 1.05–1.07, Xeon 1.08–1.09) |
| 2026-11-16 | A real row's half-length power of two stays a chain up to `pow2_chain_bytes`, as a c2c row of that length does (X32's rule, Apple and Zen-like caches) | X33: Mac r2c and c2r 32768–2²⁰ 0.69–0.97 of main, EPYC 65536–2²⁰ 0.58–0.76 (FFTN ÷ FFTW r2c 2²⁰ 1.41 → 0.82, c2r 1.35 → 0.84); the Xeon has no such chains |
| 2026-11-16 | `planning::measure` gets a generous budget: real-size timing where scratch of the data's size is acceptable, seconds per shape if needed (of the order of FFTW's MEASURE), with a quick level for cheap transforms; never PATIENT- or EXHAUSTIVE-like combinatorial search, above all under threads (section 11.2); task 2.4 comes forward, ahead of more threshold rules | The maintainer: cheap transforms are planned by rules or a quick measure, hard ones run for minutes to days and repay longer planning; their single-thread codes used FFTW PATIENT. Granite Rapids (X31b) showed hand thresholds on cache geometry misfiring on an unmeasured chip |
| 2026-11-16 | `planning::measure` and wisdom built (task 2.4, sections 11.2 and 11.3): candidates are build settings of the rules' own builders (the threshold decisions of X7, X12b, X27, X29, X32–X34 and X2's next cheapest radix sequences), timed whole in interleaved rounds at the shape's size, the rules' choice kept unless beaten by 1 %; wisdom lines checked by the program's fingerprint; public `planning`, `options::effort`, `measure_seconds`, `measure_scratch_bytes`, `use`, `fftn::wisdom` (open choices in 18.2) | The maintainer's direction of the same day: measured choice instead of more threshold rules (X31b); reusing the forcing knobs keeps every candidate a program the tests already exercise and makes wisdom a few words of text; the fingerprint check makes "rebuilds exactly the same plan" a test rather than an assumption |
| 2026-11-16 | C++20 stays (C++23 library features behind feature-test macros as before); no older standard is supported and no code is rewritten for one; compatibility with older toolchains and other languages comes from the C API (`fftn/c_api.h`) and possibly an FFTW-compatible interface, to be decided later | The maintainer prefers C++20 and newer; the public headers' C++20 use (nested inline namespaces, the executor concept, `std::span`, designated initialisers) is convenience, but a C interface serves old compilers, C, Fortran and FFTW users better than a C++17 subset |
| 2026-11-16 | The planning API (task 2.4, pull request 26) as proposed: efforts `planning::rules` (default), `quick` and `measure`; budgets as two plain options (`measure_seconds`, `measure_scratch_bytes`, 0 = the effort's default); `class wisdom` as a value filled with `add` and passed by `options::use`, never global state; a stale or unbuildable entry is ignored and `describe()` says so; measure keeps the rules' choice unless another is at least 1 % faster; serial timing only for now | The maintainer agreed the public surface beyond section 5.3's sketch (AGENTS.md section 7) after the second reader's review; the reviewer's defects are fixed before merge |
| 2026-11-16 | Bluestein and Rader convolutions follow the line rules (X36): long power-of-two convolutions take X32's long-row radices (from `grid_min_bytes` on Apple and Zen, beyond 4 L2 elsewhere); a Bluestein length's power-of-two part is limited per machine (`bluestein_pow2_part_bytes`: L1/4 on Apple, L1 on Zen, none on the Xeon); a single copy's convolution takes the split run; a stage of 1 < s < C copies covers the whole line where that convolution is no longer; half-form chirp passes rejected (section 8.5) | X36: the convolution took the in-cache table's radices and a power-of-two length at any size; Mac changed cases 0.49 (131071 2.61 → 0.88 of FFTW, 999983 1.99 → 0.75), EPYC 0.56 (4093 1.33 → 0.94, 65537 0.80), Xeon 1018 0.68 (÷ MKL 1.37 → 0.93) |
| 2026-11-16 | A radix-20 codelet (Good–Thomas 4 × 5), a pass radix only (rule 1 leaves it out, the narrow x86 leaf table has no row of it); `avx2` double overrides take it at 80, 160, 240, 640, 2000, 3000 and 3600, and re-fitted sequences at 384 and 20000; `radix_override` holds five radices (experiment X38, proposed in pull request 27) | FFTW's plans take radix 20 there; 0.868 of main on those lengths on the EPYC (0.81–0.96), tracking tier 1.000; library text +5.6 %; two unchanged-plan cases past the decision rule's limit, for the lead |
| 2026-11-17 | A Bluestein stage of s = C copies covers the whole line while the line is not longer than `long_convolution_bytes` (X36b; s < C already did, X36) | X36b: EPYC 1018 0.73 (1.53 → 1.12 of FFTW), Xeon 2036 0.81; NEON float 262142 lost 1.23 as one stage, hence the length limit; a grid convolution for the Xeon and per-copy stages need a second movable workspace slot, not built |
| 2026-11-17 | The strided axes of c2c planes and volumes run as lanes over blocks of adjacent columns (256 bytes of columns, four lines where each column element has its own page, halved to keep two tiles in L2/4 below a page, the chain beyond), with column radices whose last pass keeps its legs within the L1's ways; a loop over lines whose body is one leaf is one leaf call (section 9.3) | X40: the 37 c2c planes 0.613 (Mac), 0.453 (EPYC), 0.537 (Xeon) of the column-by-column chain's time; FFTN ÷ FFTW 1.44 → 0.88, 2.07 → 0.94, 1.49 → 0.80; ÷ MKL 1.87 → 1.00; no case slower |
| 2026-11-17 | `next_fast_length` and `relative_cost` public in `fftn/lengths.hpp` (task 2.5, section 9.2): a fast length is one AVX2's double cost table prices within 1.25 of the power of two above, per point, and not a composite with a factor 11, 17, 19 or 23 (experiment P4's rule: such composites missed 1.3 of the power of two on two machines each); the API choices are proposed by the lead, pending the maintainer (18.2) | DESIGN 9.2's design (a pure function of the version, a machine-specific query beside it); the tables are the planner's own, so the two agree with what plans do; P4 checks the lengths are fast on the three machines |
| 2026-11-17 | The fast lengths' reference table is a frozen copy of AVX2's double table (`reference_costs`), not the planner's; both length queries throw above 2⁴⁰; `next_fast_length` tests only the 2, 3, 5, 7, 13-smooth candidates in [n, 2n]; 20 is priced as the chain the planner builds (PR #41 review) | a refit of the planner's tables must not move users' grids (a checksum test pins the set to 2²⁰; unchanged by this change); the old trial of every integer took 1.8 s at 2³⁹ + 1, now under 0.6 ms; pricing above 2⁴⁰ was inexact or undefined |
| 2026-11-17 | The `x86_v2` (SSE4.2) and `avx` (AVX without AVX2 or FMA) targets exist (task 1.3): their own vocabularies without FMA, rounding as the scalar target, interleaved passes (no split runs), `avx2`'s pass-cost tables of the nearest C, no 128-bit or pair leaves; `avx` is kept; `x86_v2` takes a leaf of 32 in double (experiment X14) | X14: `avx` 0.68–0.73 of `x86_v2`'s time (EPYC 7763, Xeon 6240); `x86_v2` 0.96–0.98 of FFTW built with SSE2 only, `avx` 0.90–0.94 of FFTW with SSE2 and AVX; split runs without FMA gained 0.5–2.4 % (rule: 3 %); NEON's tables for `x86_v2` lost (1.01–1.07); 128-bit leaves for `avx` lost on two-pass rows of 100–120 points on the EPYC (1.08–1.14); the leaf of 32 took 32-point rows to 0.55–0.65 |
| 2026-11-17 | Task 2.5's API, agreed by the maintainer, 2026-11-17: `fftn/lengths.hpp`, included by `fftn.hpp`; `std::size_t next_fast_length(std::size_t n, length_kind kind = length_kind::complex)` with `enum class length_kind : unsigned char { complex, real }` (an enum replaces the proposed `bool real`, and a value outside it throws `std::invalid_argument`); `template <class T> double relative_cost(std::size_t n, isa target = isa::best)`, a ratio to the power of two above, complex in-cache rows only; edge values 1 for n = 0, 2 with `length_kind::real` for n ≤ 2 | the enum reads better at the call site than scipy's `bool`; a header of its own since grids are chosen before planning; a ratio since the tables' units are not time, and a shape-wide estimate needs the cost model P3 rejected |
| 2026-11-17 | "Fast" as defined by the frozen reference table (within 1.25 of the power of two above, per point; no prime factor above 23; no composite with 11, 17, 19 or 23), the 2⁴⁰ limit of both queries, and P4's x86 shortfall (78 % of picks within 1.3 against the 90 % planned), agreed by the maintainer, 2026-11-17 | machine-independent grids are the point of the query; `relative_cost` gives each machine's own view; P4's rule 1 target is lowered to 75 % in the experiment file, with the reason |
| 2026-11-17 | `FFTN_OLD_X86_TARGETS` (the `x86_v2` and `avx` targets of task 1.3) ON by default, agreed by the maintainer, 2026-11-17 | older x86 processors run a vector target rather than the scalar one |
| 2026-11-17 | The library's size (X27's codelets of 17, 19 and 23: +25 % of library text) accepted, agreed by the maintainer, 2026-11-17; no size budget that can fail a pull request: phase 6 adds an informational report (CI prints the library's text size per target on each pull request), experiment files record size changes, and build options (`FFTN_OLD_X86_TARGETS`, instantiating only the kernels that win) remain the levers if size ever matters | speed takes precedence; the codelets cut those cases to 0.24–0.31 of main's time, and the report keeps growth visible |
| 2026-11-17 | The recorded protocol caveats, agreed by the maintainer, 2026-11-17: X35's and X14's, the queue bypasses of X36, X38 and X39, and the recorded exception cases (X23b's two) | each is recorded in its experiment file with its reason |
| 2026-11-17 | CI for merge latency, decided by the maintainer, 2026-11-17: the ruleset on main requires four checks (lint; GCC 16 build and benchmark smoke and Clang 23 ASan + UBSan on Linux x86-64; Homebrew LLVM on macOS arm64) and no longer requires branches up to date; ThreadSanitizer, Clang 16, the build without the old x86 targets, AArch64 and Windows run after each merge on main (runs on main are not cancelled); emulation weekly and on demand only; the per-change sanitizer jobs leave out the old x86 targets | merges waited 30–90 minutes, and each merge restarted every other pull request's checks; with no users yet, a break on main found within the hour and fixed there costs less |
| 2026-11-17 | A fifth strategy, the depth-first chain (section 8.5): the sub-rows' first pass over the row, each sub-row's chain on its own block in cache, decimation-in-time top passes in place with derived twiddles (section 8.4); taken by the rules in double on `avx512` and NEON where the profile sets its range: the Xeon 6240's profile from L2/4 to 4 L2 with a top of 16 or 20, Apple's from L1 to 2 L1 with a top of 4 where the power-of-two part exceeds L1/8 (experiment X46) | Xeon 6240: 0.674 over the 12 changed rows (16384–262144, 20000, 46080, 100000, in place, backward, batches of 3), L2 misses per point at 16384–25000 0.8–1.3 → 0.00–0.01, FFTN ÷ FFTW 0.70–0.86 there (100000 1.02); stored top-pass tables lost to derived ones on every row (0.784); M5 Max 8192 0.73, 16384 0.84; the EPYC lost with every rule tried; library text +4.4 % on the Mac, the x86-64 archive +3.0 % |
| 2026-11-17 | The phase 1 tail rule (section 1.3) counts a case only when it is slower than 1.05 of the reference beyond noise (its whole interval above 1.05 × (1 + the A/A band)), decided by the maintainer, 2026-11-17 | a fixed 1.05 sat inside the noise of loaded runs (the Xeon's band was ±4.68 % on 2026-11-17) and of calls of a few nanoseconds (2 points in place); the geometric-mean targets are unchanged |
| 2026-11-17 | A radix-25 codelet (Cooley–Tukey 5 × 5), a pass radix only, with passes on the x86 and scalar targets only (none on NEON); `avx2` double overrides take it at 500, 3125, 5000, 10000, 15625, 25000 and `avx512` double at 100, 500, 3125, 10000, 15625, 25000; two-level codelets staged through memory for 16-register targets rejected (experiment X42) | EPYC 0.874 / 0.891 of the arm before over the changed lengths (FFTN ÷ FFTW 3125 0.77 → 0.65, 15625 0.79 → 0.66), Xeon about 0.86 / 0.842 (100 1.14 → 0.88 of FFTW, 3125 0.81 → 0.53); on the Mac radix 25 lost to radix-5 chains; the two-level form changed nothing with GCC (1.000) and slowed radix 25; library text +6.1 % on x86-64, +2.0 % on the Mac |
| 2026-11-17 | Multi-axis programs of at most 256 kernel calls per transform run from their calls unrolled at construction (section 8.3); the complex axes of real planes take the lanes column phase where C > 1 with their own block rule; where a cluster shares L2 a column block's tiles stay within twice L1 when its column elements lie four to a page (section 9.3) | X41: planes of up to 1024 points 0.87–0.94 of the walk's time (three machines); real planes 0.89 (EPYC) and 0.83 (Xeon) of the chain columns' time, the M5 Max lost 1.04 and keeps the chain; M5 Max 2048 × 36 0.90; joining the narrower block's phase (T1) rejected at 0.96 |
| 2026-11-17 | In double, a lanes column phase of 17–32 points with a codelet of its length is that one pass (section 9.3) | X44: on x86 the row of 32 is a pair leaf and its columns took two lane passes (EPYC 32² 0.72–0.75, Xeon 0.77–0.81 of the time; ÷ FFTW 1.39 → 1.00, ÷ MKL 1.37 → 1.07); nested runs in the column edge passes rejected (EPYC 0.936 but 4 × 6 × 8 1.03 in two runs, Xeon 0.962, Mac 0.997) |
| 2026-11-17 | An out-of-place chain of two passes runs its last pass in place on the destination (two lines, no scratch) (X47) | Zen 3 refilled L1 1.4–2.9 times as often as FFTW at 512–2048 points; with two lines a 512-point row refilled 172 times instead of 509: EPYC exit-check cases of 256–640 points 0.95–0.98, every other case unchanged (tracking EPYC 0.998, Xeon 0.997, Mac 0.999); bits unchanged |
| 2026-11-17 | Shape functions are plural, as the code has them: `rows_*`, `planes_*`, `volumes_*`, each returning a shape of `count` transforms; c2c, r2c and c2r for each (`planes_r2c`, `planes_c2r`, `volumes_c2c`, `volumes_r2c`, `volumes_c2r` in `fftn/shape.hpp`, section 5.2), shape builders only, laid out as `rows_r2c` and `rows_c2r` lay out theirs. Agreed by the maintainer, 2026-11-17 | task 4.2 proposed `volumes_c2c` (5.2 had said `volume_c2c`); the maintainer kept the plural and asked for the real kinds of both; the general `shape` already serves every kind at every rank |
| 2026-11-17 | A plan may hold a team program (`plan_core::team`) that every team of two or more runs: the same transform with narrower lanes column blocks (at most `column_team_bytes`: 128 on x86-64 with a private L2 and a large L3, the EPYC), chain-only columns as blocks of a line with the chain's radices, and (X43b) a volume's inner axes plane at a time; serial executions run the program. Bits are the program's by construction (a column's result does not depend on its block, a line's on the order of lines) and tested (sections 9.3, 13.2) | X43: at 8 lanes on the EPYC the narrower blocks took 0.887 of the time over the c2c planes and volumes they change and the line blocks 0.72 at 2048 × 36, while one thread lost up to 1.42 with them; the plan does not know at construction whether teams will run it |
| 2026-11-17 | c2c volumes: one thread keeps the axes one by one; a team runs the two inner axes plane at a time (the team program) where the profile's `team_volume_planes` says (x86-64, private L2 below 2 MiB); the outer axis over blocks of adjacent pencils by `column_block`'s rule (section 9.3) | X43b: one thread EPYC 0.979, Xeon 0.934 with 97³ 1.048, M5 Max 0.962 (rejected everywhere); 8 lanes EPYC 0.775 (VT), Xeon 0.771 (VTX) |
| 2026-11-17 | The `avx512` units are assembled with branches padded off 32-byte boundaries (`-Wa,-mbranches-within-32B-boundaries`, GCC on x86-64 only) | Bisect of the Xeon 6240's small-row loss: the plans did not change; the merge of X40 moved code and c2c 160, 180, 240 took 1.10 of their time (run 20261117-184544-8056), a JCC-erratum placement effect; padded, 160–384 0.89–0.94 and tracking 0.996 with nothing slower (20261117-204818-1541, -205228-e841); the flag on every unit slowed the EPYC 5–11 %, so `avx2` units stay unpadded; LESSONS 2.11. Unmeasured on Ice Lake and later and on Zen 4/5 (cloud tracking checks it) |
| 2026-11-18 | Complex codelets that cannot fit 16 registers (radix 32 and 64) are emitted in an order fitted to a model register file where a furthest-next-use model of the allocator counts at least a quarter fewer spill moves (section 7.2, step 5); `avx2` double 768 = 12 × 64 and float 512 = 8 × 64, 640 = 32 × 20, 768 = 12 × 64, 2048 = 32 × 64, 4096 = 8 × 32 × 16 (X54) | GCC 16 allocates in source order on x86: radix-64 lane unit stack stores and loads 149 + 178 → 87 + 107; EPYC plans with a radix-32/64 pass 0.975, tracking EPYC 0.994, Xeon 0.999, Mac 0.999; bit-identical. With it the two-pass plans win: 768 0.84 (÷ FFTW 1.08 → 0.91), float 0.79–0.95. 1024 = 16 × 64 (÷ FFTW 1.21 → 1.01 at batch 1) is not taken: 1.25 at 1024 × 1024, and an override cannot see the batch |
| 2026-11-18 | Batch leaves (X53): a contiguous c2c batch of a codelet length runs C rows per register (C × C block transposes in registers, a last row alone in the leaf's form) from a per-length batch count in the target profile; NEON float, avx2 and avx512 double (avx2's two rows per register in avx512 plans, its own four from four rows at 11–23 points) | Each lane does the leaf's arithmetic, so every row of a plan gets the same arithmetic (not necessarily a plan of one row's: section 15.5); 15 × 2 backward 1.21 → 0.88 of FFTW on the EPYC, 1.16 → 0.90 on the Xeon, NEON float batches 0.61 of the leaf; per-element gathers, padded groups for leftover rows, avx512's four rows on batches of two or three and chains through a tile all lost (experiment file) |
| 2026-11-18 | `avx2` double rows of 768 and 1024 in one-axis c2c plans take their two-pass radices (12 · 64, 16 · 64) only while rows and output fit `fit_row_bytes`, three quarters of the L3 on the EPYC class (section 11.2, rule 2; X57) | A two-pass plan's first pass reads 12–16 streams, which pays in cache and loses once the data leave the L3: on the EPYC 7763 the rule's moved cases took 0.88 of their time (1024 ÷ FFTW 1.19 → 0.99, × 256 1.08 → 1.01; 768 × 2048 0.75), tracking tier 0.989, planes and volumes unchanged. Aligning the `avx2` functions to 64 bytes was rejected: it fixed 72 and 120 (code placement) but left the mean at 0.999 and moved 1024 × 3 by 4–6 % |
| 2026-11-18 | A lanes column phase whose column has a leaf length and whose legs crowd one L1 set runs as the most balanced two lane passes whose last pass fits (16 = 4 × 4), except on `avx512` (section 9.3) | X52: M5 Max 16 × 256 × 256 0.54–0.57 of the leaf's time, EPYC 16³ 0.66–0.67 and 16 × 256 × 256 0.85–0.87; Xeon 6240 0.94–1.00 (0.966, under the rule's 0.95) |
| 2026-11-18 | A member claims a split task's chunks in runs within a phase (a share of the phase's chunks left, as for whole tasks) where the profile's claim divisor is 2 (Apple, the mesh); single chunks on the CCX (section 13) | X52: at 8 lanes Xeon 0.91 over the planes and volumes that split, Mac at 6 lanes 0.86–0.94; EPYC volumes 0.877 and planes 0.882 but real 256² 1.035 on re-run (rule 2 failed; arm G2 pending); bits unchanged |
| 2026-11-18 | A real plane or volume is split by its complex points (last axis n / 2 + 1) where each core has its own L2; on a mesh-connected L3 a plane or volume is split only from 32768 points (`machine_profile::split_complex_points`, `split_min_points_lines`; section 13, rule 3) | X52 arm S: at 8 lanes single planes of 9216–16384 points took up to 2.9 times their one-thread time on the Xeon 6240 and real 96²–100² 1.1–1.3 on the EPYC 7763; with S, Xeon 0.35–0.67 on those cases, EPYC real 96²–100² 0.72–0.90; the M5 Max (cluster L2) lost 1.29 with S1, so it keeps the real points |
| 2026-11-18 | On x86-64 with a private L2 and a large L3 (the EPYC 7763), the one-thread program's c2c column blocks widen to 1 KiB of columns where both tiles fit `column_tile_bytes`; teams keep the narrower blocks in a team program (section 9.3) | X52 arms W4 and W4b: one thread EPYC 0.933 over 38 changed cases, Xeon 0.95 but 30 × 40 × 50 1.05 on re-run (rejected there); W4 in teams lost up to 1.46 on the Xeon's 8 lanes |
| 2026-11-18 | Chunk runs on every profile, of at most left / (2 × cap) chunks (X52 arm G2), replacing their restriction to claim divisor 2 | X52: EPYC at 8 lanes 0.910 (volumes) and 0.896 (planes) over the cases that split, the real 256² that failed G at 1.04 and 0.98 on re-run; the Mac and the Xeon unchanged (their divisor is 2) |
| 2026-11-18 | The depth-first chain (section 8.5) takes one top factor that C does not divide (a partial first-pass block); the rules gain a third top, a limit for powers of two, single precision by profile and tops up to 8 on `avx2`; the Xeon 6240's profile runs it to 64 L2 with a third top of 10 and in float to 8 L2, the Apple profile to 16 L1 for lengths that are not powers of two, the EPYC 7763's profile (top 4, private L2 under 1 MiB) for lengths that are not powers of two from L2/4 to 4 L2 (X51) | X51 judged on each machine: Xeon 25000 0.75 (1.35 → 1.00 of FFTW), 524288–2²² 0.68–0.75, float 0.687; Mac mixed rows 0.735; EPYC mixed rows and 46080 0.730 (1.06–1.39 → 0.83–1.01 of FFTW); tracking tiers clean; Granite Rapids unmeasured, unchanged |
| 2026-11-18 | A measured planner rule extends by its cache key to unmeasured machines of the same instruction set target and register file, never across an instruction set, `avx2`/`avx512` or register count; every phase exit screens each family a rule reaches (AGENTS section 6) | Maintainer's decision. Granite Rapids lost 1.15–1.65 × FFTW on plans that kept it out of the Xeon's rule (X48) and gained 0.688 with it (X55); the moves that lost crossed an ISA (Axion with Granite Rapids' rules, 1.28; X35) or a register count (tops of 16 on `avx2`, X46) |
| 2026-11-18 | Last bits are not a goal: the bitwise guarantees of section 15.5 stay while they cost nothing measurable; a change that would break one needs a real measured gain and narrows the published guarantee with it; unpublished bit properties are not protected | Maintainer's decision ("rounding in other parts of calculations matter more"). First use: X53's batch leaves for 8, 12 and 25 (x86) and 20 and 32 (NEON float), 0.43–0.56 of the time in screens, give a batch's rows other last bits than a plan of one row; the published position guarantee holds (one leftover row runs the same leaf with the row in every lane, two or more a padded group) |
| 2026-11-18 | x86-64 with a private L2 of 2 MiB or more (Granite Rapids) takes the depth-first chain from L2/4 to 64 L2 in double, with the Xeon 6240's tops (section 8.5) | X55 on a Granite Rapids VM: 0.688 of the grids' time over the 14 rows it changes (46080 0.44, 65536 0.52, 131072–262144 0.63, 2^22 0.69), none slower, tracking tier 0.940; ÷ FFTW 1.15–1.65 → 0.70–0.83 on X48's failing long rows |
| 2026-11-19 | x86-64 with a private L2 between 1 and 2 MiB takes the depth-first chain; a private L2 of 2 MiB or more does not take the batch leaf of 12 (`machine_profile::batch_leaf_12`) | X58, from X56's screen of the families the rules reach: Ice Lake VM 0.596 over the changed rows, tracking 0.929; 12 × 64 and 12 × 1024 without the leaf 0.867 on Granite Rapids and 0.892 on Sapphire Rapids, tracking 0.998 and 1.000 |
| 2026-11-18 | A Bluestein stage of one copy runs its convolution as two depth-first chains where a row of M would be one, and on the Xeon profile beyond, up to 32 MiB, the longest measured (`depth_convolution_max_bytes`); the workspace block holds the chains' blocks, scratch a the forward output; the chirp passes are fused into the forward first pass and the backward last top pass (sections 8.5, 11.2; experiment X50) | Xeon 6240: 8179–999983 0.56–0.73 of `main`'s time (131071 0.61, 999983 0.71), FFTN ÷ MKL 131071 1.34 → 0.84, 999983 1.44 → 1.05, ÷ FFTW 0.65–0.83; M5 Max 4813–5119 0.86; the EPYC unchanged; X36b had judged a grid convolution to need a new slot, the block serves |
| 2026-11-19 | Whatever order a shape lists its axes in, a plan runs them widest stride first (section 9.3, `order_axes`), keeping a real transform's halved axis last | A plan built from `{x, y}` used to run y as a strided rows phase and could not use the column blocks that need the last axis contiguous; two listings of one transform are now one program, bit for bit |
| 2026-11-19 | A c2c plan whose axes are all strided and whose innermost batch dimension's transforms lie one element apart runs as lanes over blocks of adjacent transforms, every axis a lanes phase (section 9.3, X60); within L1 the chain stays for one complex per register or leaf-length axes | X60: 0.48 (M5 Max), 0.29 (EPYC 7763), 0.37 (Xeon 6240) of the chain's time over 86 interleaved cases, no case slower; ÷ FFTW 0.86 / 0.80 / 0.76 |
| 2026-11-19 | Provisional API (task 4.3, to be agreed by the maintainer): `axes_c2c`, `axes_r2c`, `axes_c2r` build the shape of chosen dimensions of a strided array from one `axis` per dimension (section 5.2); batch dimensions by decreasing input stride, merged where they nest | The general `shape` could already express every selection (property tests against the reference), but only by hand; DESIGN named `along_axes` as "later". The name follows `rows_c2c`, `planes_c2c`, `volumes_c2c`, one function per kind |
| 2026-11-19 | Windows CI installs its own LLVM (the release archive, pinned by URL and SHA-256, 23.1.5, the series the Linux jobs use) instead of using the runner image's | The x64 image moved to Visual Studio 18, whose standard library refuses any Clang older than 22 (STL1000), while its LLVM stayed at 20.1.8 (CI run 44409722437). The images update Visual Studio and LLVM separately, so this would recur; `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` would test an unsupported pair |
| 2026-11-19 | Provisional, for the maintainer's review: `fftn::kokkos::executor` (one `parallel_for` per transform on the caller's instance, then a fence), `team_of(member)`, and `view_plan` executing through the executor, serial inside a kernel, sharing large scaling passes among threads; Kokkos' tests build one threaded back end per configuration (`FFTN_KOKKOS_BACKEND`, Threads by default) | Task 3.5; experiment T3 for the costs; the adapter detects kernels per back end because Kokkos 5 removed `in_parallel` |

---

## 20. References

Internal studies (kept outside the public repository) cover: the prototype and its
benchmark log; FFTW 3.3.11; FFTL.jl; KFR; zldsp::fft; ducc0; RustFFT; threading and
Kokkos; header-only dispatch experiments; and a survey of other libraries and the
literature. The public sources the design draws on most:

- M. Frigo and S. G. Johnson, "The design and implementation of FFTW3", Proc. IEEE 93
  (2005); M. Frigo, "A fast Fourier transform compiler", PLDI 1999.
- T. Knopp, FFTL.jl (source and documentation).
- D. H. Bailey, "FFTs in external or hierarchical memory", J. Supercomputing 4 (1990).
- A. Blake, "Computing the fast Fourier transform on SIMD microprocessors", PhD thesis,
  University of Waikato (2012).
- F. Franchetti, M. Püschel et al., "SPIRAL: extreme performance portability", Proc. IEEE
  106 (2018); "FFT program generation for shared memory: SMP and multicore", SC 2006.
- D. Popovici, T. M. Low and F. Franchetti, "Large bandwidth-efficient FFTs on multicore
  and multi-socket systems", IPDPS 2018.
- E. Linzer and E. Feig, "Implementation of efficient FFT algorithms on fused
  multiply-add architectures", IEEE Trans. Signal Processing 41 (1993).
- M. Tasche and H. Zeuner, "Improved roundoff error analysis for precomputed twiddle
  factors", J. Comput. Anal. Appl. 4 (2002).
- F. Giesen, "Notes on FFTs: for implementers" (2023).
- J. Chen and J. Revels, "Robust benchmarking in noisy environments" (2016).
- Documentation of cuFFT callbacks, Intel MKL DFTI (scrambled ordering), FFTX
  (plan composition), Kokkos and Kokkos-FFT, KFR, zldsp::fft, ducc0 and RustFFT.
