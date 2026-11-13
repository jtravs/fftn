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
than 5 %.

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
src/isa/                 regions, vocabularies, one translation unit per instruction set
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
else is built on. Names and signatures here are the intended shape; details are settled
during phase 0 and recorded in the headers.

### 5.1 Conventions

- FFTW's conventions: forward is exponent −i, backward +i; unnormalised by default;
  r2c writes n/2 + 1 complex values per row; c2r reads n/2 + 1 and ignores the imaginary
  parts of the DC bin and, for even n, the Nyquist bin.
- **Input is never destroyed** unless the caller asks for in-place operation or grants
  permission (`options::may_destroy_input`), in which case the planner may use the input
  as scratch. Such plans execute through overloads that take a non-const input pointer,
  so the permission is visible in the call.
- Complex data are `std::complex<T>` (interleaved). Split real/imaginary arrays are a
  layout option (section 5.3). Both `double` and `float` are supported. `double` is tuned
  first. `float` is instantiated and tested for correctness in CI from phase 1 (on the
  scalar target and at least one SIMD target per architecture) so the precision-generic
  code cannot drift; its remaining targets and its tuning follow in phase 6.
- Any alignment of at least `alignof(T)` is accepted. Kernels use unaligned loads
  everywhere and there are no aligned-only code paths, so alignment is not part of a plan
  and never changes results. Speed can still depend on it: a 64-byte access to data that
  is not 64-byte aligned straddles cache lines. Whether the edge passes should choose a
  peeled variant from the base address at execute time is part of experiment X4.

### 5.2 Shapes

A shape is a kind, the transformed axes and the batch axes, in the manner of FFTW's guru
interface but typed and with defaults.

```cpp
namespace fftn {

enum class direction : int { forward = -1, backward = +1 };

struct axis {                 // one transformed dimension
  std::size_t n;              // logical length
  std::ptrdiff_t in_stride = 0, out_stride = 0;   // in elements; 0 = contiguous default
};
struct batch {                // one loop dimension (not transformed)
  std::size_t count = 1;
  std::ptrdiff_t in_dist = 0, out_dist = 0;       // in elements; 0 = tight default
};

template <class In, class Out>
struct shape {                // kind is implied by In/Out: complex→complex, real→complex, …
  direction dir = direction::forward;             // c2c only
  small_vector<axis, 3> axes;                     // last axis is the halved one for real kinds
  small_vector<batch, 2> batches;
};

// The common cases, as functions returning shapes:
template <class T> auto rows_c2c(std::size_t n, std::size_t count, direction);
template <class T> auto rows_r2c(std::size_t n, std::size_t count);
template <class T> auto rows_c2r(std::size_t n, std::size_t count);
template <class T> auto planes_c2c(std::size_t ny, std::size_t nx, std::size_t count, direction);
template <class T> auto volume_c2c(std::size_t n0, std::size_t n1, std::size_t n2, direction);
// From an array's extents: transform these axes of a C-ordered (or strided) array.
template <class T> auto along_axes(std::span<const std::size_t> extents,
                                   std::span<const int> axes, direction);
}
```

Zero strides and distances mean the tight C-ordered defaults, so the common cases need no
arithmetic. Construction validates that rows do not overlap, that size arithmetic cannot
overflow, and that in-place use is consistent.

### 5.3 Plans and execution

```cpp
namespace fftn {

struct options {
  planning effort = planning::rules;   // rules | measure (section 11)
  int max_lanes = 1;                   // width hint: the most threads this plan will see
  bool strict_layout = false;          // ignore the width hint when choosing the program
  bool in_place = false;               // in == out at execute
  bool may_destroy_input = false;
  double scale = 1.0;                  // fused into an edge pass; 1/n for normalised use
  edge_ops in_ops, out_ops;            // fused pointwise work (section 12)
  isa force_isa = isa::best;           // tests and benchmarks only
  const wisdom* use = nullptr;
  table_cache* tables = nullptr;       // share twiddle tables between plans (optional)
};

struct exec_args {                     // optional per-call data, one struct for all overloads
  edge_data in, out;                   // arrays for the plan's edge operations (section 12)
};

template <class In, class Out>
class plan {
public:
  explicit plan(const shape<In, Out>&, const options& = {});

  // (1) Serial, on the calling thread.
  void execute(const In* in, Out* out, workspace&, const exec_args& = {}) const noexcept;

  // (2) On an executor: anything that can run n callables (section 13).
  void execute(const In* in, Out* out, executor_ref, workspace&) const;

  // (3) Inside the caller's parallel region: every team member calls this.
  void execute(const In* in, Out* out, const team&, workspace&,
               completion = completion::all) const noexcept;

  // (4) A sub-range of the batch, serial: for callers that own the row loop.
  void execute_rows(std::size_t first, std::size_t count, const In* in, Out* out,
                    workspace&) const noexcept;

  // Every overload also accepts a trailing exec_args, and has a twin taking In* for
  // plans that may destroy their input.

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
  arrays) are selected by a kind tag in the shape; the tag defaults from the types.
- **`workspace`** holds scratch and synchronisation state for one concurrent execution. It
  is sized from a plan and a lane count, may be built over caller-supplied memory, and
  reports its size, so an application can account for every byte. Every core `execute`
  takes one; none allocates. Callers who do not want to manage workspaces use the
  convenience layer, which caches them.
- **Plan construction is cheap and safe**: microseconds to a few milliseconds with rules,
  bounded and optional with measurement; it never touches user buffers and uses no
  process-global mutable state. A plan owns its twiddle tables. Plans that should share
  tables are given the same `table_cache` object, which the caller owns; there is no
  hidden global cache, and all memory is released when the plans and the cache are
  destroyed.
- **`describe()`** makes the planner's choice visible, including any slow route (for
  example Bluestein for a large prime). Nothing is silent.

### 5.4 Convenience layer

A thin header for users who want the numpy-shaped interface: `fft`, `ifft`, `rfft`,
`irfft`, `fft2`, `fftn`, `rfftn`, `irfftn` over `std::span` and a small strided array view
(C++20 has no `mdspan`; the view converts from one where available), with `axes`,
`norm = backward | ortho | forward`, a cache of plans and per-thread workspaces keyed by
shape (the one place FFTN allocates behind the caller's back, and documented), and the helpers
`fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `next_fast_length`. All of it is
expressed through the core API; none of it adds kernels.

### 5.5 Other interfaces

- **`fftn/kokkos.hpp`** (optional, header-only): plans built from host-accessible
  `Kokkos::View`s of any layout, an executor over an execution-space instance, and a plan
  class with the constructor and execute shape of Kokkos-FFT's `Plan` so existing call
  sites need no change (section 13.6).
- **`fftn/omp.hpp`** (optional, header-only): the OpenMP adapters, compiled in the user's
  translation unit so the process has one OpenMP runtime.
- **C API** (`fftn/c_api.h`): opaque plans, the same shapes, a
  three-function executor contract for foreign thread pools.
- **FFTW3-compatible shim** (separate opt-in library and header, never installed as
  `fftw3.h`): the `fftw_plan_many_dft`, guru and execute subset, so C and Fortran users
  and existing benchmark tools can link FFTN. It is opt-in because a header that
  impersonates FFTW breaks applications that probe for FFTW features.

### 5.6 Coverage tiers

| Tier | Contents | Phase |
|---|---|---|
| Core | c2c, r2c, c2r; 1-D rows, 2-D planes, 3-D volumes; strides and distances; in place and out of place; any length; double; serial, team and executor execution; edge ops; convolution and transform-pair pipelines | 1–5 |
| General | float on every target, tuned where asked for; convenience layer; DCT/DST types I–IV and the discrete Hartley transform as composed plans; split-complex layout; C API; wisdom import and export; `next_fast_length` backed by the cost model | 6 |
| Later, on demand | FFTW shim; more than three transformed axes; SVE; Python binding | 7 |

---

## 6. Vector vocabulary and instruction sets

### 6.1 Targets

| Target | Register | Complex per register (interleaved) | Registers | Notes |
|---|---|---|---|---|
| `scalar` | 1 double | — | — | Portable fallback; reference for generator tests |
| `x86_v2` | 128 bit, SSE2–SSE4.2, no FMA | 1 | 16 | Every x86-64 CPU since about 2009 (Nehalem, Westmere, Atom-class, older AMD). First class |
| `avx` | 256 bit, no FMA, no AVX2 | 2 | 16 | Sandy Bridge, Ivy Bridge, AMD Bulldozer family (2011–2014). A variant of the `avx2` vocabulary; kept if experiment X14 shows a gain over `x86_v2` on such machines |
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

| Operation | Meaning | NEON+FCMA | AVX2 / AVX-512 |
|---|---|---|---|
| `add(a, b)`, `sub(a, b)` | a ± b | 1 instruction | 1 instruction |
| `addj(a, b)`, `subj(a, b)` | a ± i·b | `FCADD` (1) | swap + `addsub` (2); AVX-512 has no 512-bit `addsub`, so swap + sign flip + add, or `fmaddsub` with a ones vector |
| `mul(a, w)`, `mulc(a, w)` | a·w, a·conj(w) | 2 × `FCMLA` | 1 shuffle + 2 FMA with pre-arranged twiddles |
| `madd(acc, a, w)` | acc + a·w | 2 × `FCMLA` onto acc | as `mul` then add, or fused |
| `scale(a, r)`, `fmadd_r(acc, a, r)` | real constant times a (plus acc) | 1 | 1 |
| `neg`, `conj`, `swap_ri` | sign and part manipulation | 1 | 1 |
| `load`, `store`, `bcast`, `load_tw` | memory | | |
| `transpose(R, C)` | in-register R × C transpose of complex values | zips | unpack / permute |

**Real-register operations** (split layout: an `rv` holds `W` reals; a complex vector
is a pair of them): `add`, `sub`, `mul`, `fmadd`, `fmsub`, `fnmadd`, `neg`, `load`,
`store`, `bcast`, and `deinterleave` / `interleave` to convert from and to interleaved
memory (`LD2`/`ST2` on NEON, unpack pairs on x86).

Each target implements the vocabulary in one header of a few hundred lines. Adding a
target is adding that header; adding a radix never touches a target.

x86 twiddle storage is chosen so `mul` needs no run-time rearrangement: each twiddle is
stored as the two vectors (wr, wr) and (−wi, wi) the FMA form consumes. That is the
"pre-arranged" form in the table, and one reason twiddle layout is a per-target, per-pass
decision (section 8.4). It doubles the bytes per stored twiddle, which the planner weighs
against the derived mode for long rows.

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

A dedicated test links every instruction set's translation unit into one binary and
checks the disassembly of each target's kernels for the intended instructions and for the
absence of calls to out-of-line helpers.

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
   | 3, 5, 7, 11, 13 | symmetric-pair form: sums and differences of x_j and x_{R−j}, then cosine/sine combinations |
   | 9, 25, 27 | Cooley–Tukey with constant internal rotations |

2. **Track pending rotations.** Every node carries a pending factor (−i)^k. Combining two
   nodes whose rotations differ by 0, 1, 2 or 3 quarter turns emits `add`, `subj`, `sub`
   or `addj`. A multiplication by ±i or −1 therefore never costs an instruction. General
   constants are reduced modulo quarter turns first, which cuts the number of distinct
   constants and the register pressure.
3. **Simplify.** Constant folding, sign canonicalisation (push negations into consumers;
   treat a − b and b − a as one node), common-subexpression elimination by value numbering
   (evaluate each node at random inputs in high precision and hash), dead-code elimination
   for pruned inputs and outputs.
4. **Lower** to one of two forms:
   - *complex-register*: nodes map directly onto the complex vocabulary;
   - *split*: each complex node expands to real and imaginary nodes, followed by a second
     simplification and **FMA shaping** (emit a ± b·c directly; factor rotations as
     c·(x − t·y) with t = s/c where the outer multiply can be absorbed).
5. **Order** the emission depth-first by recursive partition of the DAG (finish one
   sub-transform before starting the next), and place each store next to the operation
   that produces its value. This keeps the number of live values near the minimum, which
   matters for radix ≥ 16 on 16 registers and ≥ 32 on 32. A codelet whose radix exceeds
   the register count necessarily keeps some values on the stack; that is acceptable when
   the spills are few and ordered (FFTW's radix-32 and radix-64 codelets work this way),
   and ruinous when they are not: the prototype's hand-built radix 20, 24 and 32
   composites lost on every machine. The generator therefore records, per codelet and
   target, the spill count read from the compiled code, and the planner prices it.
6. **Verify.** Evaluate the emitted operation list in high-precision arithmetic against
   the exact DFT for random and basis inputs. A codelet that fails is not written.
7. **Emit** C++ against the vocabulary: one text for all targets per lowering, included
   in each target's region, with a per-target selection of which codelets to instantiate.
   Alongside it, emit a metadata table: radix, variant, operation counts (adds,
   multiplies, FMAs, complex multiplies), peak live values, twiddle count. The planner's
   cost model, the kernel registry and the tests are driven from this table, so there are
   no hand-maintained lists of radices anywhere.

### 7.3 Variants

For each radix the generator can emit:

- **plain** (no twiddles): leaves and last passes;
- **DIF twiddle** (multiply outputs) and **DIT twiddle** (multiply inputs, with the
  multiply fused into the first butterfly layer as `madd` where the target has it);
- **forward and backward**: generated separately for the complex-register form; for the
  split form the backward transform reuses the forward codelet with real and imaginary
  registers exchanged at load and store;
- **pruned**: declared-zero inputs and unwanted outputs removed by dead-code elimination
  (zero-padded convolutions, real-input and half-complex codelets);
- **fused-edge**: the real-transform post- or pre-processing folded into the codelet of
  the last or first pass (section 9.1).

Reference values for regression: the generator's operation counts are compared with the
best published counts for each size (for example a size-16 complex DFT in 144 additions
and 24 multiplications in scalar split form). Reaching them means nothing is missing.

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

Each kind exists in DIF and DIT form, in complex-register and split arithmetic, and with
**edge variants** that change only how values are loaded or stored: layout conversion
(IL ↔ BL), gathering rows into lanes with a tile transpose, the fused real post- or
pre-processing, multiplication by a factor array or scale, pruned loads and stores.
Variants are separate instantiations chosen at plan time. A generic pass never tests a
run-time flag in its loop (the prototype measured 0–40 % for doing so).

Which kind a pass is follows from its copy count s against the register's complex
count C. In a DIF chain s is the product of the radices already applied, so the planner
prefers a leading radix of at least C (then only the first pass is grouped); with C = 1
(NEON double) the distinction disappears. The copy counts that are not multiples of C
(tails) run on a narrower instantiation of the same codelet, outside the main loop,
never through a mask or select inside it. Vector operations act on each lane
independently, so a narrower instantiation performs exactly the same arithmetic per
element; this is what keeps a row's result independent of its position in a batch
(section 15.5).

### 8.3 Programs

A plan's executable form is a **program**: a flat array of plain records, prepared at
construction and interpreted by a small driver.

```cpp
struct stage {            // one pass over one buffer range
  pass_fn   run;          // the instantiated pass kernel: run(args, first, last)
  const void* twiddles;
  std::uint32_t radix, m, s;        // geometry
  std::uint8_t  src, dst;           // buffer slots: caller in, caller out, scratch a/b, block
  std::uint32_t chunks;             // independent butterfly ranges this pass divides into
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
planes of a volume. The driver costs a few nanoseconds per record. Two consequences:

- **Blocking and batching are data**, not separate engines.
- **The threading work graph is read off the program** (section 13.2). Independent loops
  give whole-row, whole-plane and block tasks. Within one row, each stage is a phase and
  its `chunks` butterfly ranges are the units shared between threads, with the stage
  boundary as the ordering constraint; this is how the prototype split chain passes, and
  the blocked grid adds coarser chunks (column blocks, rows). The division of work is
  fixed by the plan, which is what makes results independent of the schedule.

### 8.4 Twiddles

- Tables are per pass, contiguous in the order the pass reads them, in the storage form
  the target's `mul` consumes.
- Storage mode is per pass. **Full** (every twiddle stored) is the only mode built at
  first. **Shared** (conjugate-pair and symmetry sharing: fewer loads) and **derived** (a
  few stored, the rest computed in registers: for long rows where twiddle traffic competes
  with data traffic) are added only if experiment X5 shows they pay.
- Values are computed with exact integer octant reduction and double-double evaluation,
  rounded once; no recurrences. Tables are shared between plans through the twiddle cache.

### 8.5 Strategies

A strategy builds a program for a 1-D transform of length N from passes. There are four.
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

**Lanes** (vectorise across rows). C rows, or C neighbouring lines of a strided axis,
become the lanes of every pass; the whole transform is lane passes. For row-major batches
the first pass gathers a tile of rows into lanes and the last scatters it; for a strided
axis the lanes are already adjacent and nothing is gathered. Lane blocks are sized so a
block and its scratch stay in L1 or L2. Best for short rows whose own first passes
vectorise badly (awkward lengths on wide registers) and for every strided axis.

**Bluestein** and the **generic prime pass** cover lengths with large prime factors
(section 9.2).

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

c2r preserves its input by default; with permission to destroy it the planner may take a
faster route. A multi-dimensional c2r that preserves its input needs scratch for the
half-spectrum of one plane (or pencil block) per thread, because the real output cannot
hold it; this is reported by `workspace_bytes`. In-place r2c and c2r use FFTW's padded
layout: each row of the last axis holds n + 2 reals (n/2 + 1 complex), the last two
being padding on the real side.

### 9.2 Any length

| Factor | Route |
|---|---|
| 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, (20, 25, 32, 64) | generated codelets |
| other primes up to about 100 | one generic looped pass in symmetric form, O(p²) per butterfly, vectorised across lanes |
| larger primes | Bluestein: the smooth part of N is split off first; each large prime factor becomes one pass that performs a chirp-z transform through a smooth-length convolution with a precomputed kernel spectrum |

Rader's algorithm is not planned; it shares a convolution object with Bluestein and can be
added if measurements on prime sizes justify it. The cost model prices these routes.
`next_fast_length` lets applications choose grid sizes that are fast on FFTN rather than
on another library; it is a pure function of the library version (a fixed reference cost
table), so a grid chosen on one machine is the same on another. A separate query returns
the machine-specific estimate for callers who want it.

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

Axis order within a plan is a planner choice unless a pipeline pins it (section 12.3).

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

### 11.2 Two efforts

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

**`planning::measure`** (opt-in). Take the top k candidates by the model (k about 4–8),
build each, time whole programs on scratch data for a bounded budget (default 10 ms per
shape in total) with warmed caches and several repetitions, and keep the fastest. Whole
plans are timed; no optimal-substructure assumption. Large batches and volumes are
measured on a reduced batch (a few rows, planes or pencil blocks) with scratch bounded by
a stated limit, so measuring never allocates memory on the scale of the user's data. The
result is recorded as wisdom.

### 11.3 Wisdom and reproducibility

Wisdom is a small text record: shape key, CPU model, library version, program
description. It can be exported and imported as a string or file and is an input to plan
construction (`options::use`), never global state. A program description is also a valid
plan input, so a result can be pinned and reproduced exactly on the same instruction set.
Measured planning is the only source of run-to-run variation in program choice, and it is
opt-in.

### 11.4 What the planner deliberately lacks

No flag lattice, no exhaustive recursive search, no timing of sub-plans in isolation, no
alignment in the problem key, no planning modes that differ in more than the two efforts
above. If the model is poor on some machine, the remedy is a better profile, not more
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
themselves are passed at execute time, so they may change every call. Pruned edges use the
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

With `completion::own` and `workspace::rows_of(lane)`, an application's own pointwise
kernels can use the same row-to-thread assignment as the transform, so rows stay in one
core's cache across the application's work and FFTN's, inside one parallel region. In
this mode rows are dealt statically (lane i does exactly `rows_of(i)`, nothing is
claimed by others), which requires that every lane of the team takes part.

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
same code whoever runs it. Therefore:

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
3. Split a row only inside one cache group, with at least about 512 points per member, and
   on large mesh-connected groups only from about 8000 points.
4. Deal rows to cache groups; only lanes that have work do any. Under OpenMP that means
   surplus team members return at once from a region of constant width, not a narrower
   region: libgomp ends and restarts threads when the width changes.
5. By default use at most one cache group for a batch smaller than the group, and only the
   fastest core class on hybrid processors.

Thresholds come from the machine profile (dispatch cost, barrier cost, per-group scaling),
not from constants in code.

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
| OpenMP, FFTN opens the region | `fftn::omp::executor` | header-only: compiled against the caller's runtime; inline when one lane has all the work |
| Kokkos | `fftn::kokkos::executor(exec_space_instance)` and View-based plans | one `parallel_for` per transform on the caller's instance; correct on Serial, OpenMP and Threads back ends; serial when called inside a kernel |
| Custom pool | any type with `concurrency()` and `bulk(n, fn, arg)` | the only requirement is "run n callables and return when done" |
| No runtime at hand | `fftn::thread_pool` | explicit, opt-in, `std::thread` based, grouped by cache topology; never created implicitly |

Rules that hold everywhere: FFTN creates no threads unless the caller constructs a
`thread_pool`; it never changes a runtime's settings (thread counts, wait policy,
binding); a call from inside a parallel region or kernel runs serially rather than
nesting; width 1 opens no region.

What "natural in Kokkos" means concretely: plans accept host `View`s of any layout
through their extents and strides; the executor takes the caller's execution-space
instance and its concurrency; transforms appear as named regions to Kokkos Tools; a plan
class matching Kokkos-FFT's `Plan` interface makes adoption a type change; device
back ends are untouched (the header compiles out or forwards under CUDA and HIP builds).
Kokkos is never a dependency of the core.

### 13.7 Testing threads

Bitwise comparison of serial against every team size on synthetic topologies;
deterministic simulation of the scheduler with adversarial interleavings; ThreadSanitizer
on the built-in pool and, with an instrumented OpenMP runtime, on the OpenMP adapters;
stress under oversubscription and injected delays; a runtime matrix (libgomp, libomp,
Kokkos Serial/OpenMP/Threads, the built-in pool).

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
  holds with GCC 13 to 16 and Clang 18 to 23. But every GCC compiles a vector passed by
  value from a region into a baseline function with only a `-Wpsabi` warning, and the
  program then crashes. The library is therefore built with `-Werror=psabi` under GCC, and
  vector types are never template arguments in region code (GCC drops their attributes).
- Not yet run: Clang on Windows (experiment B2).

### 14.2 Decision: a compiled static library

FFTN is built as a library: CMake target `fftn::fftn`, static by default, with one
translation unit per instruction set so they compile in parallel. A header-only
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
- **Its own floating-point flags.** The library sets `-ffp-contract=off`, refuses
  fast-math, and never inherits the including project's flags. All FMAs in kernels are
  explicit.
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
  Windows on Arm has no documented feature flag for FCMA; how to detect it there is part
  of experiment B2, and until then FCMA is reported absent there.

### 14.3 Toolchains and packaging

- C++20, extensions off. GCC 14 and newer; Clang 16 and newer; Apple Clang; clang on the
  MSVC ABI (clang-cl and clang++ with the MSVC standard library). MSVC's compiler is not
  a target. C++23 library features may be used only behind feature-test macros.
- CMake 3.25 or newer with presets; `FetchContent`-friendly with an offline override;
  installed package config (`find_package(fftn)`) and pkg-config; all symbols in
  `namespace fftn`, all macros prefixed `FFTN_`, all targets namespaced.
- Test-only dependencies (GoogleTest, FFTW, Boost.Multiprecision) and benchmark-only dependencies are fetched
  pinned by commit and treated as system code.
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
   113-bit binary float, standalone and header-only, as a test-only dependency. Native
   types do not cover the platforms: `long double` is plain double on Apple silicon and
   with the MSVC ABI, and `__float128` is rejected by Clang on Apple silicon (GCC accepts
   it there). A 1024-point reference DFT takes about 0.3 s. The library's own twiddle
   tables, which it builds with a few lines of double-double arithmetic, are checked
   against the same reference, so the reference shares no code with the library.
3. **FFTW**, as an independent implementation, for large sizes and every shape.
4. **The generator's own check** of each codelet against the exact DFT at emission.

### 15.2 Tolerances

Derived before the test runs from machine epsilon, the length and the algorithm: a
transform of length n built from stable butterflies has relative RMS error of order
ε·√(log₂ n), and FFTN's bound for the test is a small multiple of that (the prototype
measured 4–5·10⁻¹⁶ against a tolerance 700 times looser; FFTN's tolerances are within
about 10× of the measured error). Bluestein paths have their own, looser, derived bound.
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
MSVC ABI. AVX-512 paths run under Intel SDE and NEON-without-FCMA and SVE paths under
QEMU where the runner's CPU lacks them, so every kernel is executed in CI, not only on
developers' machines. Old processors are emulated as whole machines too (SDE's Nehalem
and Sandy Bridge models, QEMU's Neoverse N1), which checks detection and dispatch on
them as well as the kernels: a stray newer instruction anywhere faults.

### 15.5 Determinism and the numerical contract

- No fast-math, no flush-to-zero, no approximate reciprocals. FMAs are explicit; the
  library is built with `-ffp-contract=off` so the compiler introduces
  none (GCC 15 contracted two inlined copies of the same code differently in the
  prototype).
- **Bitwise identical** for a given plan across: serial and threaded execution, any team
  size, executor and schedule; repeated runs; batch position (row 0 and row 1000 of a
  batch get the same arithmetic, as do the rows of `execute_rows` for any `first`). The
  rule that guarantees the last: within a plan every row goes through the same sequence
  of codelets; lane tails and leftover rows use narrower instantiations or padded tiles
  of the same program, never a different route. Tests compare row k against row 0 for
  every k modulo the lane count.
- **May differ in the last bits** between: instruction sets; plans built with different
  options (width hint without `strict_layout`, edge operations, in place versus out of
  place); plans chosen by measurement on different runs, unless pinned by wisdom; library
  versions.
- Error grows as O(ε·√log n) RMS for all lengths with codelet factors; Bluestein lengths
  are documented separately.

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
  commit (fbench builds it from `git archive`) or any plan option.
- **Catalogues** are data files of weighted cases:
  - *standard*: powers of two 2¹–2²⁴; 2-3-5-7-smooth lengths; primes and awkward
    composites; batches from 1 to thousands of rows; r2c and c2r; planes and volumes;
    in place and out of place; strided axes;
  - *client*: shapes captured from an application's trace with weights from its real FFT
    share (direction, placement and planes included);
  - *threads*: the same shapes at 1, a cache group, a socket and all cores, in library
    mode and in team mode (a team of serial plans, the baseline threading must beat).
- **Tiers**: smoke (seconds, correctness only, in the gate), quick (2–3 minutes, A/B),
  tracking (a fixed set of 30–60 cases), full.

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
- **Validity.** fbench records load, frequency and thermal state where the OS exposes
  them and marks rounds on a busy machine invalid (load average, and other processes'
  current CPU use). On macOS it requests the performance-core class.
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

None at present. Decisions taken are in the decision log; questions that arise during
implementation are listed here until settled.

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
