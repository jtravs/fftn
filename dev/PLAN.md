# FFTN implementation and experiment plan

This is the live plan for building FFTN as designed in [DESIGN.md](DESIGN.md), under the
rules in [AGENTS.md](../AGENTS.md). Read "Status" first. Update "Status" in every working
session; update a phase when its tasks or findings change.

## Status

**As of 2026-11-13.** Phase: before phase 0. No library code exists.

Done:

- Research: source studies of FFTW, FFTL.jl, KFR, zldsp::fft, ducc0 and RustFFT, a review
  of the prototype and its benchmark rig, a threading and Kokkos study, a literature and
  landscape survey, header-only dispatch experiments.
- Initial documents: DESIGN.md, this plan, AGENTS.md, README.md.
- An independent review of the first draft; its findings are applied (see the decision
  log in DESIGN.md).
- Gate skeleton: `scripts/check.sh` with the repository-hygiene stages live (SPDX,
  document links, switch registry, format, ruff, shell); code stages switch on as their
  CMake presets appear.

Next actions (in order):

1. Maintainer review of DESIGN.md.
2. Phase 0, tasks 0.1–0.4 (skeleton, platform layer, ISA regions, test infrastructure).
   These are independent enough for two or three agents in parallel worktrees.
3. Phase 0, task 0.6 (fbench v0) can start at once, in parallel, against FFTW alone.

Open questions: see DESIGN.md section 18.2. Decided so far (decision log): Apache-2.0;
no header-only configuration; the first draft's working assumptions.

Claims: none.

Handoff notes: none yet.

## How this plan works

- **Phases** group work with a shared exit criterion. Phases 1–2 (kernels), 3 (threading)
  and the fbench work are separate tracks after phase 0 and can run in parallel: the
  threading engine is built and tested on the walking skeleton's programs (its chain
  passes already divide into chunks). Phase 3's performance exit is measured once the
  phase 2 kernels exist.
- **Tasks** have an ID, a short description and an acceptance test. Claim a task by adding
  `(claimed: name, date)` to its line.
- **Experiments** are listed in the index below and each gets a file under
  `dev/experiments/`, named `<ID>-<slug>.md` (template:
  [experiments/TEMPLATE.md](experiments/TEMPLATE.md)), before
  it is run. An experiment ends with a verdict and a design change or a deletion.
- **Exit criteria** are measured by fbench on the named machines. A phase is not finished
  because its tasks are ticked; it is finished when its criteria are met or the plan is
  changed, with reasons, in the decision log of DESIGN.md.
- Later phases are deliberately less detailed. Refine a phase when the one before it is
  half done.

## Phases

| Phase | Outcome | Depends on |
|---|---|---|
| 0 | Foundations: build, gate, CI, platform layer, ISA mechanism, tests, a correct but slow end-to-end library, fbench v0 | — |
| 1 | Generator and single-thread complex 1-D kernels at target speed | 0 |
| 2 | Real transforms, batches, any length, planner with cost model and measurement | 1 |
| 3 | Threading: engine, adapters, topology, scheduler | 0 (parallel with 1–2) |
| 4 | Planes, volumes, strided axes, in place | 2, 3 |
| 5 | Fusion: edge operations, convolution, transform pairs, callbacks | 4 |
| 6 | General coverage and release preparation | 5 |
| 7 | On demand: FFTW shim, SVE, Python, more axes | — |

---

### Phase 0: foundations

Goal: everything needed to develop safely and measure honestly, plus a walking skeleton:
the public API computing correct transforms of any length through deliberately simple
kernels. Speed is not a goal here.

| ID | Task | Acceptance |
|---|---|---|
| 0.1 | Repository skeleton: CMake (3.25+), presets `check-release`, `check-tidy`, `check-sanitize`, `check-tsan`, `check-coverage`; static library target with one translation unit per instruction set; `tools/coverage_gate.py`; a `uv` project (`pyproject.toml`, lock file, Python 3.14+) for all Python tooling; the Doxygen linter configuration and the example-snippet check | `scripts/check.sh` runs every stage on the Mac and on Linux GCC 14 |
| 0.2 | Platform layer: CPU feature detection (x86 CPUID, AArch64 Linux `getauxval`, macOS `sysctlbyname`, Windows), cache geometry and topology detection with overrides, aligned memory, timers | unit tests with synthetic topologies; correct output recorded for the Mac, the Xeon and the EPYC cluster |
| 0.3 | ISA regions and vocabularies: `FFTN_ISA_BEGIN/END`, per-target namespaces, vocabularies for `scalar`, `neon_fcma`, `avx2` (others in phase 1); dispatch at plan construction | all-targets link test; disassembly test per target; experiments B1, B2 |
| 0.4 | Test infrastructure: quadruple-precision reference DFT (Boost.Multiprecision, test-only), analytic oracle suites, derived tolerances, canary-page helper, allocation hook, FFTW as a test-only dependency; suites parameterised on precision (`double` and `float`) from the start | suites run against a naive DFT implementation and catch seeded faults (wrong sign, off-by-one bin, 1 ulp twiddle error at large n) |
| 0.5 | Walking skeleton: `shape`, `plan`, `workspace`, `describe()`; program and stage driver with chunked stages; a chain built from templated radix-2/3/4/5 butterflies plus the generic prime pass on the scalar vocabulary; c2c, r2c, c2r rows and in-place batched 2-D planes, any length, serial; every public declaration documented in its header; first `docs/conventions.md` and one compiled example | all oracles pass for every n ≤ 512 and sampled larger n; contracts tested; coverage threshold recorded in the gate; `apidoc` stage green |
| 0.5b | A first `fftn/kokkos.hpp`: View-based construction and the Kokkos-FFT-shaped plan class over the skeleton (serial execution only) | a small Kokkos program using rows and planes compiles and passes; gives the API review real call sites |
| 0.6 | fbench v0: driver with process rounds, paired bootstrap intervals and the identical-arm control; backends for FFTN and FFTW; the standard catalogue; quiet-machine check; JSON-lines store; remote runner for the Slurm machines | A/A reports "no evidence" on all cases; a seeded 5 % slowdown is detected; a perturbed output is flagged |
| 0.7 | fbench backends for FFTL.jl, MKL, pocketfft, KFR, Accelerate; client catalogue re-captured with directions, placement, planes and real weights | each backend passes the correctness guard on its platforms |
| 0.8 | CI: Linux x86-64 (GCC 14, current GCC, Clang), macOS arm64, Linux AArch64, Windows clang; gate stages split across jobs; SDE and QEMU runs for instruction sets the runners lack | green on an empty change |
| 0.9 | API review: the headers of 0.5 reviewed by the maintainer against DESIGN.md section 5 | decisions recorded in the decision log |

Exit: gate and CI green on all four platforms; fbench A/B trusted (its self-tests pass on
the Mac, the Xeon and the EPYC cluster); baseline numbers stored for the backends of 0.6
and 0.7. The remaining competitors of DESIGN.md section 16.2 are added as phases need
them, all before the phase 2 exit.

---

### Phase 1: generator and complex 1-D kernels

Goal: single-thread c2c rows at target speed on Apple M-series, Zen 3 and the Xeon.

| ID | Task | Acceptance |
|---|---|---|
| 1.1 | Generator core: DAG with pending rotations, split-radix, Good–Thomas, symmetric primes, simplification, value-numbering CSE, depth-first ordering, high-precision verification, C++ emission in complex-register form, metadata table; radices 2–16; `gen/README.md` (pipeline, intermediate representation, how to add a radix or lowering, command line, subtleties) | operation counts match the reference table; `gen` gate stage live; a second reader can add a radix from the README alone |
| 1.2 | Pass templates (lane, grouped, leaf) with tails and with the edge-variant hooks in place from the start (scale and factor on load and store, as the first two edge operations); twiddle tables (full mode) with exact generation; chain strategy; rules planner v0 (FFTL's rules as the starting point) | all tests pass with generated kernels on `scalar` and `neon_fcma`, in `double` and `float`; a fused scale costs no extra pass |
| 1.3 | Vocabularies `avx512`, `neon`, `x86_v2`, `avx`; in-register transposes per target; non-FMA operation order from the generator for targets without FMA; experiment X14 | tests pass under every target, natively or emulated, including whole-machine emulation of Nehalem, Sandy Bridge and Neoverse N1 |
| 1.4 | Experiments X2 (radix sets and order), X4 (padding and addressing), X8 (whole-transform leaves) | defaults set from results |
| 1.5 | Split lowering with FMA shaping; `BL` layout and conversion edge variants; experiment X1 | layout default per target decided |
| 1.6 | Radix 32 and 64 codelets; grid strategy, few-big-passes regime; twiddle modes; experiments X3, X5 | rows of 16–64 thousand points on the Xeon at or below FFTW |
| 1.7 | Grid strategy, blocked regime for rows beyond L2; experiment X12 | long rows (2¹⁷–2²⁴) at or below the best competitor |

Exit (fbench standard catalogue, c2c rows, one thread): the DESIGN.md section 1.3 targets
against FFTW and FFTL.jl on M-series, Zen 3 and Xeon; within 1.10 of MKL on the Xeon;
the `x86_v2`, `avx` and plain `neon` targets, forced on the same machines, at or below
FFTW restricted to the same instruction sets. The MKL figure is interim; the end target
is 1.00. Per-length choices at this stage come from the offline searches of X2 encoded as
rules; the cost model arrives in phase 2. One cloud check on a Zen 4 or 5 and one on an
AArch64 Linux machine (Neoverse) confirm that nothing is badly off there; tuning them is
phase 2 and 6 work.

---

### Phase 2: real transforms, batches, any length, planner

| ID | Task | Acceptance |
|---|---|---|
| 2.1 | r2c with the post fused into the last pass; c2r through the DIT chain with the pre fused; eligibility by live-value count; experiment X6 (real-input codelets) | real overhead ≤ 20 % of the half-length complex time on all three machines |
| 2.2 | Lanes strategy for row batches (tile gather and scatter) and strided lines; lane blocks sized by cache; experiment X7 | awkward short rows on AVX-512 at or below FFTW and MKL |
| 2.3 | Codelets 7, 11, 13 and composites; vectorised generic prime pass; Bluestein pass with smooth-part split | every n ≤ 4096 correct and priced; no FFTW fallback anywhere |
| 2.4 | Cost model, machine profiles and `fbench calibrate`; `planning::measure`; wisdom export and import; experiments P1, P2 | the model's first choice is within 5 % of the measured best for ≥ 90 % of catalogue cases |
| 2.5 | `next_fast_length` and published per-length cost | documented and tested |

Exit: all 1-D row shapes (c2c, r2c, c2r; 1 to thousands of rows; smooth, prime and
awkward lengths) meet the section 1.3 targets on one thread on M-series, Zen 3, the Xeon,
one Zen 4 or 5 and one AArch64 Linux machine; default plan construction under 1 ms for
smooth lengths.

---

### Phase 3: threading

Can start after 0.5: the engine works on the walking skeleton's programs, including
splitting a row across a chain's chunked stages.

| ID | Task | Acceptance |
|---|---|---|
| 3.1 | Work graph from program loops; `workspace` synchronisation state; the claiming engine; `completion::all` and `own`; `rows_of` | bitwise equality with serial for every team size on synthetic topologies |
| 3.2 | Scheduler as a pure function; simulation tests with adversarial interleavings | pathological cases from the prototype's log reproduced in simulation and handled |
| 3.3 | `fftn/omp.hpp` (team and executor), `fftn::thread_pool`, the custom-executor concept and C contract | TSan clean; runtime matrix (libgomp, libomp) green |
| 3.4 | Experiments T1 (dispatch and wait costs per machine), T2 (claiming against static barriers), T5 (split thresholds), T4 (run-to-run swings: first touch, NUMA balancing, false sharing) | machine profiles carry threading constants; default engine decided |
| 3.5 | `fftn/kokkos.hpp`: executor, View-based plans, Kokkos-FFT-shaped plan; experiment T3 | tests on Kokkos Serial, OpenMP and Threads |
| 3.6 | Intra-row splitting with the phase 1–2 kernels: chunked chain stages and the blocked grid (needs 1.7) | single long rows scale within a cache group on Zen 3 and the Xeon |

Exit: threaded targets of section 1.3 on Zen 3 (8 and 64 threads), Xeon (18) and M-series
(performance cores), measured with the phase 2 kernels; no fbench threads-catalogue case
slower than serial; never slower than a team of serial plans by more than 2 %.

---

### Phase 4: planes, volumes, strided axes

| ID | Task | Acceptance |
|---|---|---|
| 4.1 | Multi-axis shapes; plane-at-a-time programs; in place for all strategies | batched 2-D c2c at or below FFTW and MKL, one thread and threaded |
| 4.2 | Volumes with pencil blocks; per-thread scratch of plane or pencil-block size | 3-D c2c at or below FFTW and MKL; scratch bound tested |
| 4.3 | Arbitrary axis subsets and orders of strided arrays; real multi-dimensional transforms | property tests against the reference |

Exit: fbench planes and volumes catalogues meet the one-thread and threaded targets.

---

### Phase 5: fusion

| ID | Task | Acceptance |
|---|---|---|
| 5.1 | Edge operations: scale, window, factor, valid, embed, accumulate | each matches "transform then separate pass" within the derived tolerance (fusing a multiply into an FMA changes last bits), and is faster |
| 5.2 | `convolution` pipeline: fused, pruned, several kernels, one forward to several inverses | at or below 0.8 of three separate calls for L2-resident sizes |
| 5.3 | `spectral_pair` and in-cache callbacks, including multi-dimensional pipelines with pinned axis order, prologue and epilogue | a volume pipeline sweeps memory three times (verified by counters) |
| 5.4 | Experiment X9: scrambled intermediate order and `spectral_layout` | adopted or rejected with numbers |
| 5.5 | Owner-computes row sharing with application kernels | example and test |

Exit: pipelines meet their acceptance numbers, and the client catalogue meets the
DESIGN.md section 1.3 row for client catalogues.

---

### Phase 6: general coverage and release preparation

Float on every target and tuned where wanted (it is correct and tested from phase 1 on);
convenience layer; DCT/DST I–IV and DHT as composed plans; split-complex
layout; C API; Windows tuning; AArch64 Linux tuning on Neoverse N1, V1 and V2 (cloud);
Zen 4/5 and recent Intel (cloud); the documents of DESIGN.md section 17 complete and
reviewed as a whole (each phase adds its own part as it lands); the per-release public benchmark
report (one command, fixed cases and machines); packaging (CMake package,
pkg-config, a generated amalgamation if wanted); first tagged release.

### Phase 7: on demand

FFTW3-compatible shim; SVE (experiment X11); Python binding; more than three transformed
axes; Rader; SME on Apple M4 and later for batches of short transforms (speculative).

---

## Experiment index

Each row becomes `dev/experiments/<ID>-<slug>.md` before it runs. Status: planned,
running, done (with verdict).

| ID | Question | Phase | Status |
|---|---|---|---|
| B1 | Do ISA attribute regions hold on Linux x86-64 with GCC 14 and 15 (inlining, by-value vector ABI at -O0, two-TU link, disassembly)? | 0 | planned |
| B2 | The same on Windows with clang on the MSVC ABI | 0 | planned |
| B3 | Library build time and size as codelets and targets grow; which codelets are worth instantiating per target? | 1 | planned |
| X1 | Complex-register against split arithmetic, per target (FCMA included) and pass position (first, middle, last; short and long rows) | 1 | planned |
| X2 | Best radix sets and orders per target by exhaustive search over the standard lengths; how far are simple rules from the optimum? | 1 | planned |
| X3 | Few big passes (radix 16–64, in the output buffer) against chains for rows of 8–64 thousand points; spill counts per codelet and target. The prototype's radix 20–32 attempts lost everywhere: find out whether generated, ordered codelets change that | 1 | planned |
| X4 | Padding: incremental addressing cost; pad always, never, or only where the stream rule binds. Also: do edge passes gain from a variant chosen by the caller's base address (cache-line splits on wide registers)? | 1 | planned |
| X5 | Twiddle modes (full, shared, derived) by length and target | 1 | planned |
| X6 | Real-input codelets against half-length complex with fused post/pre, even and odd n | 2 | planned |
| X7 | Lanes against in-row kernels for batches of short rows, by length, target and row count | 2 | planned |
| X8 | Whole-transform leaves for n ≤ 64–256 against two- and three-stage chains | 1 | planned |
| X9 | Scrambled intermediate order for transform pairs | 5 | planned |
| X10 | Generated whole passes against templated pass loops | 1 | planned |
| X11 | SVE at 128 and 256 bits on Graviton 3/4 against NEON | 7 | planned |
| X12 | Beyond-L2 forms: DIF grid, DIT grid, in-place sandwich; block sizes by cache level; the in-place route (scratch of N, permutation pass, or palindromic swap) | 1 | planned |
| X14 | Is a separate `avx` target (256 bit, no FMA, no AVX2) worth keeping over `x86_v2` on Sandy Bridge-class machines? | 1 | planned |
| X13 | Next-row prefetch and non-temporal stores per machine for out-of-cache batches | 2 | planned |
| P1 | Cost-model accuracy: rank agreement with measurement across the catalogue per machine | 2 | planned |
| P2 | Measurement budget: how many candidates and how long before the choice stops improving | 2 | planned |
| T1 | Dispatch, wake and wait costs per machine and runtime (libgomp, libomp, Kokkos, pool) | 3 | planned |
| T2 | Claiming engine against precomputed schedule with spin barriers; keep one or both? | 3 | planned |
| T3 | Kokkos: cost of one `parallel_for` per transform; `team_barrier` behaviour | 3 | planned |
| T4 | Run-to-run swings with several concurrent teams: first touch, NUMA balancing, false sharing of synchronisation words | 3 | planned |
| T5 | Thresholds for splitting a row, by cache-group type | 3 | planned |
| T6 | Hybrid processors: performance cores only, or all | 3 | planned |

## Machines

| Machine | Use | Access |
|---|---|---|
| Apple M5 Max | NEON+FCMA development and timing when quiet | local |
| Xeon Gold 6240 (AVX-512, 2 × 18 cores) | AVX-512 and Intel timing; MKL | Slurm, exclusive |
| EPYC 7763 cluster (Zen 3, AVX2, 2 × 64 cores) | AVX2 timing; threading at 64 | Slurm, exclusive |
| Cloud: Zen 4/5, recent Intel, Neoverse N1/V1/V2 | checks per phase exit; phase 6 tuning | rented per experiment, with a ledger |
| Old x86-64 (v2 level, or AVX without AVX2) | best effort | no hardware: the `x86_v2` and `avx` targets are timed forced on the Xeon and Zen 3, and verified by emulating Nehalem and Sandy Bridge |
