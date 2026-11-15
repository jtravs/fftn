# FFTN implementation and experiment plan

This is the live plan for building FFTN as designed in [DESIGN.md](DESIGN.md), under the
rules in [AGENTS.md](../AGENTS.md). Read "Status" first. Update "Status" in every working
session; update a phase when its tasks or findings change.

## Status

**As of 2026-11-14, evening.** Phase 0 done except the Xeon client-catalogue baseline.
Merged today: 1.3a, planner v1 with NEON tables (X2), X8 (radix 32/64 codelets; a 32-point
leaf on NEON double), 1.7 and 1.7b with X4 (grid strategy, fused phase passes, staggered
scratch), 2.1 (fused real transforms), 2.3 and 2.3b (Bluestein, shared tables), X18 (x86
driver and scratch fixes), X16 rule 1, X10 (rejected). Where FFTN stands (FFTN ÷ FFTW):
Mac smooth c2c rows 0.75–1.03, primes from 97 0.64–1.28, long rows 0.92–1.37, real
256–4096 0.82–0.96; Xeon (AVX-512) about 1.17–1.29, EPYC about 1.32 (x86 kernels are the
open problem, see X18/X10/X19). Lessons that generalise are collected in
[LESSONS.md](LESSONS.md).

In flight (claims): X19 (generator ±i rotations; reviewed, merging after regeneration on
main), 1.5/X1 (split lowering on x86), 1.6/X3 (radix 32/64 passes; radix 32 wins on AVX-512
float), X12b (grid thresholds per machine: the grid starts too early on x86); T1/T5 done
on branch `exp/t1-t5-constants` (threading constants; awaiting review). Small follow-ups recorded: X16's
precision split on Cascade Lake (avx512 for double, avx2 for float), X2's refit above 1024.

Done in phase 0 (details in the merge commits and the decision log):

- 0.1 skeleton, 0.2 platform layer, 0.3 ISA regions and vocabularies, 0.4 test
  infrastructure, 0.6 fbench v0, B1.
- 0.5 walking skeleton: the public API, programs and stage driver, the chain strategy on
  the scalar target, every kind at ranks 1–3, any length; checked against all three oracles
  at every n ≤ 512. 0.5b: `fftn/kokkos.hpp` (View-based plans, Kokkos-FFT-shaped
  `view_plan`, `compat` spellings). Both had second readers.
- 0.7 fbench backends (FFTW, MKL, pocketfft, KFR, FFTL, Accelerate) pass their guards.
- 0.8 CI: 13 jobs on Linux x86-64/AArch64, macOS, Windows x64/Arm; emulated runs (SDE,
  QEMU) of the instruction-set tests on `main`, weekly and on demand. B2 done.
- 0.9 API review: every recommendation taken; 0.10 made the changes (vectors, any number of
  batch dimensions, `execute_transforms`, `view_plan`, `compat` enums). FFTN is unreleased:
  the API changes freely until the first tagged release.
- X15 done: no `omp simd` on scalar loops; old-ISA targets need hand vocabularies.

Phase 0 exit: gate and CI green on all platforms; fbench self-tests trusted on the three
machines; competitor baselines against FFTW at MEASURE (PATIENT on the quick tier) for the
EPYC (tracking, client catalogue), the Xeon (tracking) and the Mac (tracking, client
catalogue). Still to do: the Xeon client catalogue.

Next actions: merge X19; land X1 and X3 (x86 kernels and passes) and X12b; then
tiny-size overhead (fixed cost per call), X6 (real-input codelets), the X2 refit, and the
phase 1 exit cloud checks, done (X21: nothing badly off; follow-ups are `best_target` per
precision and length, c2c planes on Zen 5, single rows of 65536 and 100000 points on x86).
In parallel: T1 and T5 are done (profile constants set; their follow-up is a dispatch cost
that grows with the team's width, DESIGN 18.2 item 22), then the rest of 3.4 (T2, T4) and
3.5 (Kokkos executor).

Open questions: see DESIGN.md section 18.2.

Baseline runs (fbench run IDs): EPYC 7763-class node (GCC 16): tracking
20261113-230721-7c95 failed (PATIENT timeout), re-run without PATIENT; pocketfft re-run with
`-march=native` after a first run built it without; PATIENT vs MEASURE on the quick tier done.
Xeon Gold 6240 (GCC 14): tracking 20261114-004852-f498. Mac (NEON FFTW): client catalogue
20261114-030832-198b; FFTL.jl (libfftl at 53171e8, apple-m1) tracking 20261114-095234-2dcf:
1.11 against FFTW (FFTN 2.20 on the same run: ahead on smooth rows ≤ 4096, far behind on
primes, long rows and real transforms, which phases 1.7 and 2 build).

Handoff notes: remote gates go to DMOG first (`gate-dmog.sbatch`, `--mem=96G`, and 160G with
4 hours for sanitizer builds of the radix-64 codelets; LSan off there, ptrace is forbidden on
its nodes). The Xeon is shared: small allocations (`-c 8`), never exclusive. Timing uses
fbench's load-tolerant rounds everywhere; add rounds rather than waiting for quiet. Build
once with the Mac's GCC 16 before a remote round trip (GCC-only warnings).

Task 1.5 (branch `task/1.5-split`, split-lowering session): the generator's split lowering
(radices 2–16, plain and DIF, FMA-shaped, tangent rotations), split lane passes on the BL
layout converting at a split run's ends, `target_profile::split` and `build_settings::layout`
(fbench `LAYOUT`). X1's verdict: split runs on `avx2` and `avx512` (0.87–0.93 of the
interleaved time where a run exists), interleaved on NEON. Next: a split-aware planner (costs
of split passes; a first radix that lets a run start: AVX2's 1024 = 4 × 16 × 16 has none),
split runs in Bluestein convolutions and grid phases, a split grouped first pass with an 8 × 8
real transpose (port 5 is still 58 % busy on `avx512`, FFTW 41–44 %).

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
| 1.5 | Split lowering with FMA shaping; `BL` layout and conversion edge variants; experiment X1 | layout default per target decided: split runs by default on `avx2` and `avx512`, interleaved on NEON and scalar (X1, merged 2026-11-14) |
| 1.6 | Radix 32 and 64 codelets; grid strategy, few-big-passes regime; twiddle modes; experiments X3, X5 | rows of 16–64 thousand points on the Xeon at or below FFTW (X3 done: radix 32/64 priced for AVX-512 float only; X5 open) |
| 1.7 | Grid strategy, blocked regime for rows beyond L2; experiment X12 | long rows (2¹⁷–2²⁴) at or below the best competitor |

Exit (fbench standard catalogue, c2c rows, one thread): the DESIGN.md section 1.3 targets
against FFTW and FFTL.jl on M-series, Zen 3 and Xeon; within 1.10 of MKL on the Xeon;
the `x86_v2`, `avx` and plain `neon` targets, forced on the same machines, at or below
FFTW restricted to the same instruction sets. The MKL figure is interim; the end target
is 1.00. Per-length choices at this stage come from the offline searches of X2 encoded as
rules; the cost model arrives in phase 2. Cloud checks (Google Cloud or AWS) on a Zen 4 or
5, on a recent Intel Xeon (Sapphire Rapids or newer, with MKL as a competitor: its AVX-512
behaves differently from the Cascade Lake Xeon's) and on an AArch64 Linux machine
(Neoverse) confirm that nothing is badly off there; tuning them is phase 2 and 6 work.

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
| [B1](experiments/B1-isa-regions-gcc-linux.md) | Do ISA attribute regions hold on Linux x86-64 with GCC 14 and 15 (inlining, by-value vector ABI at -O0, two-TU link, disassembly)? | 0 | done: regions hold; GCC needs `-Werror=psabi` |
| [B2](experiments/B2-isa-regions-windows-clang.md) | The same on Windows with clang on the MSVC ABI (x64 and Arm64); how to detect FCMA and pointer authentication on Windows on Arm | 0 | done: regions hold; FCMA and PAuth read from the registry (task 0.8) |
| B3 | Library build time and size as codelets and targets grow; which codelets are worth instantiating per target? | 1 | planned |
| [X1](experiments/X1-split-lowering.md) | Complex-register against split arithmetic, per target (FCMA included) and pass position (first, middle, last; short and long rows) | 1 | done: split runs on avx2 and avx512 (0.87–0.93 where a run exists), interleaved on NEON |
| [X2](experiments/X2-radix-sets.md) | Best radix sets and orders per target by exhaustive search over the standard lengths; how far are simple rules from the optimum? | 1 | running: EPYC done (planner v1 adopted); Mac done (own NEON tables, 3.6 % faster); lengths above 1024 and the Xeon pending |
| [X3](experiments/X3-big-radices.md) | Few big passes (radix 16–64, in the output buffer) against chains for rows of 8–64 thousand points; spill counts per codelet and target. The prototype's radix 20–32 attempts lost everywhere: find out whether generated, ordered codelets change that | 1 | done: adopted for AVX-512 float only (planner table with radix 32/64 columns: Xeon float chains 0.977 of before, 256–1024 0.76–0.94); rejected for AVX-512 double, AVX2, NEON and as grid phases up to 16384; radix 32 phases from 32768 a follow-up after X12b |
| [X4](experiments/X4-padding.md) | Padding: incremental addressing cost; pad always, never, or only where the stream rule binds. Also: do edge passes gain from a variant chosen by the caller's base address (cache-line splits on wide registers)? | 1 | running: for the grid on the Mac, staggered scratch lines adopted (2^22 −8 %), padding the intermediate rejected; the chain's lines, x86 and edge-pass variants open |
| X5 | Twiddle modes (full, shared, derived) by length and target | 1 | planned |
| [X6](experiments/X6-real-codelets.md) | Real-input codelets against half-length complex with fused post/pre, even and odd n | 2 | planned (needs the generator's pruned variants); the half-length route's fusion rule measured on the Mac (task 2.1) |
| X7 | Lanes against in-row kernels for batches of short rows, by length, target and row count | 2 | planned |
| [X8](experiments/X8-leaves.md) | Whole-transform leaves for n ≤ 64–256 against two- and three-stage chains | 1 | done: leaf limit 32 on NEON in double (16–22 % faster), 16 elsewhere; no leaf of 64 |
| X9 | Scrambled intermediate order for transform pairs | 5 | planned |
| [X10](experiments/X10-pass-loops.md) | Generated whole passes against templated pass loops; on x86 first: addressing, register budget and shuffles of the pass loops (after X18) | 1 | done: rejected (loop addressing cuts instructions, not time; the cost is port-5 shuffles, X19 next) |
| X11 | SVE at 128 and 256 bits on Graviton 3/4 against NEON | 7 | planned |
| [X12](experiments/X12-beyond-l2.md) | Beyond-L2 forms: DIF grid, DIT grid, in-place sandwich; block sizes by cache level; the in-place route (scratch of N, permutation pass, or palindromic swap) | 1 | running: DIF grid in two or three phases adopted for rows beyond L1 on the Mac (0.47–0.71 of the chain's time, run 20261114-110844-8b03); copies fused into end passes for small-radix phases within L2/4 (2^18: 1.03 → 0.92, run 20261114-132841-80e0); x86 runs pending |
| [X12b](experiments/X12b-grid-thresholds.md) | The grid's thresholds (chain or grid, two or three phases) as rules of L1 and L2 that suit the Xeon, the EPYC and the Mac | 1 | done: with a private L2 (x86) the grid starts beyond L2/2 and three phases beyond L2; the Mac (cluster-shared L2) unchanged; x86 0.91–0.96 of the old default over the X12b cases, 4096 points on the Xeon 1.42 → 1.05 times FFTW |
| X14 | Is a separate `avx` target (256 bit, no FMA, no AVX2) worth keeping over `x86_v2` on Sandy Bridge-class machines? | 1 | planned |
| X13 | Next-row prefetch and non-temporal stores per machine for out-of-cache batches | 2 | planned |
| [X15](experiments/X15-compiler-vectorised-scalar.md) | What does a current compiler get from the scalar pass loops (`omp simd`, the scalar text in the `avx2` region, `-march=native` as a ceiling), against a hand vocabulary? | 0 | done: omp simd rejected (compiler-dependent); old targets need hand vocabularies |
| [X16](experiments/X16-avx512-table.md) | `avx512` against `avx2` on AVX-512 machines: does `avx512` need its own cost table, or an R × C transpose for leading radices below C? | 1 | running (Xeon campaign) |
| [X17](experiments/X17-bluestein-threshold.md) | Above which prime does Bluestein beat the generic prime pass, per target? | 2 | done on the Mac: Bluestein from 17 up (threshold 16); x86 to confirm |
| [X18](experiments/X18-x86-kernels.md) | Why does FFTN trail FFTW on x86 (Xeon, EPYC), and what closes the gap? | 1 | done: driver divisions and a four-line chain fixed (Xeon 1.50 → 1.25 of FFTW, EPYC 1.37 → 1.29); next X10 addressing, radix 32/64, then X1 |
| [X19](experiments/X19-generator-rotations.md) | Do codelets with fewer multiplications by ±i (the real-coefficient form: one register swap per output pair, radix 16 7 against 19) make x86 faster without slowing NEON? | 1 | done: adopted (x86 0.976 of main: Xeon avx512 0.980, avx2 0.976, EPYC 0.971; Mac 0.991); port-5 uops −4 to −16 % |
| [X21](experiments/X21-phase1-cloud.md) | Phase 1 exit cloud checks: is anything badly off on a recent Intel Xeon (with MKL), Zen 5 and a Neoverse, and does `avx512` over `avx2` hold off Cascade Lake? | 1 | done: nothing badly off (FFTN ÷ FFTW over the tracking c2c rows: Emerald Rapids 1.17, Zen 5 1.12, Axion 0.81; FFTN ÷ MKL 1.19); avx512 wins in double everywhere, in float not on Zen 5, and loses 3× at 60 points in float; USD 1.36 |
| P1 | Cost-model accuracy: rank agreement with measurement across the catalogue per machine | 2 | planned |
| P2 | Measurement budget: how many candidates and how long before the choice stops improving | 2 | planned |
| [T1](experiments/T1-dispatch-wait-costs.md) | Dispatch, wake and wait costs per machine and runtime (libgomp, libomp, Kokkos, pool) | 3 | done: hand-off cost and claim runs adopted on Zen 3; dispatch and work costs wait for a per-lane dispatch term |
| T2 | Claiming engine against precomputed schedule with spin barriers; keep one or both? | 3 | planned |
| T3 | Kokkos: cost of one `parallel_for` per transform; `team_barrier` behaviour | 3 | planned |
| T4 | Run-to-run swings with several concurrent teams: first touch, NUMA balancing, false sharing of synchronisation words | 3 | planned |
| [T5](experiments/T5-split-thresholds.md) | Thresholds for splitting a row, by cache-group type | 3 | done: 6144 points (Zen 3 CCX), 8192 (Apple cluster, mesh); 1024 points per member |
| T6 | Hybrid processors: performance cores only, or all | 3 | planned |

## Machines

| Machine | Use | Access |
|---|---|---|
| Apple M5 Max | NEON+FCMA development and timing when quiet | local |
| Xeon Gold 6240 (AVX-512, 2 × 18 cores) | AVX-512 and Intel timing; MKL | Slurm, exclusive |
| EPYC 7763 cluster (Zen 3, AVX2, 2 × 64 cores) | AVX2 timing; threading at 64 | Slurm, exclusive |
| Cloud: Zen 4/5, recent Intel, Neoverse N1/V1/V2 | checks per phase exit; phase 6 tuning | rented per experiment, with a ledger |
| Old x86-64 (v2 level, or AVX without AVX2) | best effort | no hardware: the `x86_v2` and `avx` targets are timed forced on the Xeon and Zen 3, and verified by emulating Nehalem and Sandy Bridge |
