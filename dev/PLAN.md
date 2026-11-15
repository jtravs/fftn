# FFTN implementation and experiment plan

This is the live plan for building FFTN as designed in [DESIGN.md](DESIGN.md), under the
rules in [AGENTS.md](../AGENTS.md). Read "Status" first. Update "Status" in every working
session; update a phase when its tasks or findings change.

## Status

**As of 2026-11-15, evening.** Work continues on the one-dimensional double leads (claims
below); this section is also the handoff. Lessons
that generalise are in [LESSONS.md](LESSONS.md), decisions in DESIGN.md section 19, open
questions in DESIGN.md section 18.2.

### Where FFTN stands

Tracking tier, one thread, FFTN's time ÷ FFTW's (geometric mean; below 1 is faster), all
of `main` as of 2026-11-15 morning: Mac M5 Max 1.06 (FFTN 0.56 of Accelerate's time on the
cases Accelerate supports), Xeon 6240 1.15 (MKL: FFTN 1.27 of its time), EPYC 7763 1.33. Runs
20261115-094953-38c5 (Mac), 20261115-114105-ea43 (Xeon, MKL), 20261115-101319-f98f (EPYC).

One-dimensional double only (46 cases; the planes and volumes excluded): Mac 0.97, EPYC
1.21. By group, Mac / EPYC: powers of two ≤ 64 1.14 / 1.38; 128–4096 0.91 / 1.22; ≥ 16384
1.15 / 1.20; mixed radix 0.95 / 1.29 (46080 and 100000: 1.18–1.23 / 1.67–1.98); primes
0.92 / 1.20 (17: 1.57 / 1.95; 4093: 1.24 / 1.55; 65537 EPYC 1.65; 97–1009 win on both);
batches 0.94 / 1.12; r2c and c2r 1.00 / 1.25 (65536 and 2²⁰: 1.30–1.39 / 1.41–1.62).
Against FFTL.jl (libfftl 53171e8, Mac run 20261115-170717-c72c, 1-D double): FFTN 0.86
of its time overall; FFTL ahead only on rows ≥ 16384 (1.35) and 2431 (2.57); FFTN far ahead
on single rows ≤ 64 (FFTL's fixed cost per call: 0.07–0.51) and batches (0.70).
EPYC (libfftl x86-64-v3, run 20261115-171147-bc20, no round under load): FFTN 0.81 of
FFTL's time overall (FFTL 1.49 of FFTW's); the same pattern: FFTL ahead on rows ≥ 16384
(1.51; 262144 2.04, 100000 2.00, r2c and c2r 2²⁰ 2.0–2.2) and 2431 (2.93), FFTN far ahead
on rows ≤ 64 (0.12), batches (0.40) and 128–4096 (0.75).
Planes and volumes are far behind (phase 4 has not started): c2c 1024² 2.1 / 4.1, 64³
3.1 / 10.9 (EPYC: likely cache aliasing of power-of-two strides).

### Tasks

Done: phase 0 (0.1–0.10, 0.5b, B1, B2, X15); phase 1 tasks 1.1, 1.2, 1.5 (X1), 1.6 (X3),
1.7 and 1.7b (X12 on the Mac, X12b), with X8, X10 (rejected), X18, X19, X20, X21 (cloud,
USD 1.36), X22 (icpx); phase 2 tasks 2.1, 2.3 and 2.3b (X17), the planner refresh (X2 refit
on every machine, X16); phase 3 tasks 3.1–3.3 with T1 and T5. X24 rejected. CI restructured
(per-push slim, weekly extended; CI on a pull request is the merge gate).

Open, by phase:

- 0: the Xeon client-catalogue baseline.
- 1: 1.3's `x86_v2` and `avx` vocabularies and X14; X2 on the Mac above 1024; X4 on x86
  and for edge passes; X12 on x86; X5. The phase 1 exit is not met: x86 trails FFTW.
- 2: 2.2's acceptance in double c2c on AVX-512 (the chain 1.01–1.26 of FFTW at 8–16
  points; X24 rejected both planned fixes); 2.4 (cost model, `planning::measure`, wisdom;
  not started); 2.5 (`next_fast_length` exists internally, not public); X6, X13, P1, P2.
- 3: T2, T3, T4, T6; 3.5 (Kokkos executor); 3.6 to be checked against its acceptance.
- 4–7: not started.

In flight (lead session, 2026-11-15 evening): X23b (masked partial runs adopted for GCC
builds, `FFTN_MASKED_TAILS`), X26 (lead 1, the grid's copies), X27 (lead 2, 17-point and
other prime codelets), X28 (lead 4, single short rows), X29 (lead 5, the real footprint in
the chain-or-grid rule). Lead 3 (x86 pass-loop instructions) starts after X23b lands. X25 (the x86 gap's causes, from counters and FFTW's
printed plans) is done; its verdicts are folded into the leads below. X23 (masked partial
runs on x86) is done and rejected by its rule: Clang 23 loses on odd double lengths, while
GCC 14 gains much (float 0.60 of main on `avx512`); the masked code is in the branch's
history (arms 1132477, 0baf6a3, 489b268), and finding Clang's loss is a follow-up (lead 7).

### Leads, one-dimensional double first

**Start here when work resumes (the maintainer's choice, 2026-11-15): leads 1–5, the
one-dimensional double gap.** Each needs an experiment file (AGENTS.md section 6) before it
runs; the numbers above are the baseline to beat. The rest (6–9) follow. X25's verdict
(counters on both x86 machines) says which change each lead needs first.

1. Long rows (≥ 16384, both machines; mixed 46080 and 100000 worst on the EPYC at 1.67–1.98;
   r2c and c2r of 65536 and 2²⁰ at 1.3–1.6): X12 on x86 (the grid's phases and blocks
   there), the real pairing fused into grid phases, radix 32 phases from 32768 (X3
   follow-up). X25: on both x86 machines it is the grid's copy loops, not memory
   (`block_copy` 40–52 % of FFTN's instructions on the Xeon, about 33 instructions per
   64-byte row with blocks of 4 columns; FFTN has fewer L2 misses than FFTW): first a copy
   specialised for one-vector rows, wider or tiled blocks, fewer separate copies.
   FFTL.jl is the reference to study here: on the Mac it beats both FFTN (FFTN
   1.35 of its time at ≥ 16384; 2²⁰ 1.60, 100000 1.53, 46080 1.45) and FFTW (0.86).
2. Prime lengths and prime factors above 16: 17 takes Bluestein (1.57 / 1.95 of FFTW), and
   inside composites each such factor is one generic pass (2431 = 11 · 13 · 17: FFTN 2.57
   of FFTL's time on the Mac, 1.25 of FFTW's on the EPYC): a generated 17-point codelet
   (and 19, 23 if cheap), or Rader where p − 1 is smooth (65537, 257, 97); 4093 and 65537
   on the EPYC.
3. The x86 kernels at 128–4096 (EPYC 1.05–1.35): a split-aware planner (costs of split
   passes; a first radix that lets a run start: AVX2's 1024 = 4 × 16 × 16 has none), split
   runs in Bluestein convolutions and grid phases, a split grouped first pass with an 8 × 8
   real transpose (port 5 is 58 % busy on `avx512`, FFTW 41–44 %). X25: the Xeon is at
   par here (port 5 57–73 % busy against FFTW's 34–55 %); the EPYC executes 1.38–1.60 times
   FFTW's instructions at 256–1024 for the same floating-point work, so the AVX2 pass loops
   need 30–40 % fewer instructions; at 4096 FFTN runs four passes to FFTW's three.
4. Single short rows (≤ 64 points: Mac 1.14, EPYC 1.38; 60 points on the EPYC 1.42): the
   remaining fixed cost per call after X20. X25: at 8 points 23 of the 25-cycle gap on the
   Xeon (14 of 15 on the EPYC) is fixed cost; FFTW runs single 8- and 16-point rows with
   128-bit codelets, and FFTN's 512-bit leaf also lowers the Xeon's clock. First: 128-bit
   leaves for single rows of 8–16 points, a flat driver path for two-pass chains (60 points
   on the EPYC: 37 of 79 cycles are the driver), and the partial-unit radix-4 pass at 60
   on the Xeon (57 % of the time).
5. Real transforms on x86 (EPYC 1.25): X6 real-input codelets. X25: r2c 65536 on the Xeon
   is memory (its 32768-point chain at IPC 0.84 with 3.4 times FFTW's L2 misses): count
   the whole real footprint in the chain-or-grid rule; on the EPYC it is the grid's
   instructions (lead 1).
6. Double c2c batches of 8–16 points on AVX-512 (task 2.2): the chain against FFTW and MKL;
   a lanes threshold per transform kind (r2c and c2r of 16–64 points under AVX-512 already
   take 0.56–0.96 of the chain's time). X25: batched leaves saturate one unit (Xeon ports
   0 and 5 at 82–87 %, EPYC FP pipes 2 and 3 up to 95 %).
7. Float: 60 points on `avx512` 2.6 times `avx2` (Xeon and cloud); `best_target` per
   precision and length; X23's masked partial runs (GCC 14: float 0.60 of main on `avx512`,
   0.80 on `avx2`) once Clang 23's loss on odd double lengths (`avx2` 75: 1.42) is found.
8. Planes and volumes (phase 4): the EPYC's 64³ c2c (10.9 times FFTW).
9. Threading: a dispatch cost that grows with the team's width (DESIGN 18.2 item 22).

Decisions the maintainer has not yet taken are listed in DESIGN.md section 18.2.

### Working notes for whoever resumes

- CI on a pull request is the gate (`ci.yml` per push, `extended.yml` weekly and on demand,
  `emulation.yml` for SDE and QEMU). Build and test what you touch locally; do not spend
  local time on full gates.
- Remote timing: `uv run --frozen python scripts/remote/fbench_remote.py submit
  <xeon|epyc|xeon-clang> ARMS -- fbench-args` (profiles in `~/.config/fftn/remote.toml`,
  outside the repository; the Xeon profile sets the MKL root). `fetch` runs from the main
  checkout. Each arm is built once per machine into a shared cache by a short,
  wider build job before the timing job (`bench/README.md`, "Remote runs"); `--serial`
  runs timing jobs one at a time, `prune` clears old cache entries. Small shared allocations (`-c 8`), never exclusive; at most three of one
  agent's jobs on the Xeon at once; fbench's load-tolerant rounds, adding rounds rather
  than waiting for quiet. Remote gates go to DMOG first (`--mem=96G`; 160G and 4 hours for
  sanitizer builds; LSan off there, ptrace is forbidden on its nodes).
- On the Mac compare against the NEON build of FFTW, and build at most 6 jobs in parallel
  (the Mac is shared).
- Every experiment's verdict goes into its file, the index below and LESSONS.md.
- The arm branches that the X10, X15 and X24 files cite (`exp/x10-pass-loops`,
  `exp/x15-arm-*`, `exp/x24-lb-only`) exist only in the maintainer's local clone; they
  are not pushed.

Baseline runs (fbench run IDs): EPYC tracking 20261113-230721-7c95 (PATIENT timeout,
re-run with MEASURE); Xeon (GCC 14) tracking 20261114-004852-f498; Mac client catalogue
20261114-030832-198b; FFTL.jl (libfftl 53171e8) tracking 20261114-095234-2dcf.

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
| 2.2 | Lanes strategy for row batches (tile gather and scatter) and strided lines; lane blocks sized by cache; experiment X7 | awkward short rows on AVX-512 at or below FFTW and MKL (built 2026-11-15; X7 after X20: lanes only in float on x86, 0.32–0.55 of the chain at 8–16 points; double c2c on AVX-512 still 1.1–1.3 of FFTW with the chain, not met; strided axes of planes are phase 4's) |
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
| [X2](experiments/X2-radix-sets.md) | Best radix sets and orders per target by exhaustive search over the standard lengths; how far are simple rules from the optimum? | 1 | running: planner v1 adopted (EPYC); own NEON tables; refreshed on every machine after X1/X3/X18/X19 (AVX-512 own tables, float 0.957 of before on the Xeon; AVX2 and NEON overrides, 0.94–0.95 on changed EPYC lengths, 0.98 on the Mac's); Mac above 1024 pending |
| [X3](experiments/X3-big-radices.md) | Few big passes (radix 16–64, in the output buffer) against chains for rows of 8–64 thousand points; spill counts per codelet and target. The prototype's radix 20–32 attempts lost everywhere: find out whether generated, ordered codelets change that | 1 | done: adopted for AVX-512 float only (planner table with radix 32/64 columns: Xeon float chains 0.977 of before, 256–1024 0.76–0.94); rejected for AVX-512 double, AVX2, NEON and as grid phases up to 16384; radix 32 phases from 32768 a follow-up after X12b |
| [X4](experiments/X4-padding.md) | Padding: incremental addressing cost; pad always, never, or only where the stream rule binds. Also: do edge passes gain from a variant chosen by the caller's base address (cache-line splits on wide registers)? | 1 | running: for the grid on the Mac, staggered scratch lines adopted (2^22 −8 %), padding the intermediate rejected; the chain's lines, x86 and edge-pass variants open |
| X5 | Twiddle modes (full, shared, derived) by length and target | 1 | planned |
| [X6](experiments/X6-real-codelets.md) | Real-input codelets against half-length complex with fused post/pre, even and odd n | 2 | planned (needs the generator's pruned variants); the half-length route's fusion rule measured on the Mac (task 2.1) |
| [X7](experiments/X7-lanes.md) | Lanes against in-row kernels for batches of short rows, by length, target and row count | 2 | done: re-decided after X20's batched leaves: lanes up to `lanes_up_to` in float on x86 only (AVX-512 100, AVX2 32; 0.32–0.55 of the chain at 8–16 points); 0 in double and on NEON |
| [X8](experiments/X8-leaves.md) | Whole-transform leaves for n ≤ 64–256 against two- and three-stage chains | 1 | done: leaf limit 32 on NEON in double (16–22 % faster), 16 elsewhere; no leaf of 64 |
| X9 | Scrambled intermediate order for transform pairs | 5 | planned |
| [X10](experiments/X10-pass-loops.md) | Generated whole passes against templated pass loops; on x86 first: addressing, register budget and shuffles of the pass loops (after X18) | 1 | done: rejected (loop addressing cuts instructions, not time; the cost is port-5 shuffles, X19 next) |
| X11 | SVE at 128 and 256 bits on Graviton 3/4 against NEON | 7 | planned |
| [X12](experiments/X12-beyond-l2.md) | Beyond-L2 forms: DIF grid, DIT grid, in-place sandwich; block sizes by cache level; the in-place route (scratch of N, permutation pass, or palindromic swap) | 1 | running: DIF grid in two or three phases adopted for rows beyond L1 on the Mac (0.47–0.71 of the chain's time, run 20261114-110844-8b03); copies fused into end passes for small-radix phases within L2/4 (2^18: 1.03 → 0.92, run 20261114-132841-80e0); x86 runs pending |
| [X12b](experiments/X12b-grid-thresholds.md) | The grid's thresholds (chain or grid, two or three phases) as rules of L1 and L2 that suit the Xeon, the EPYC and the Mac | 1 | done: with a private L2 (x86) the grid starts beyond L2/2 and three phases beyond L2; the Mac (cluster-shared L2) unchanged; x86 0.91–0.96 of the old default over the X12b cases, 4096 points on the Xeon 1.42 → 1.05 times FFTW |
| X14 | Is a separate `avx` target (256 bit, no FMA, no AVX2) worth keeping over `x86_v2` on Sandy Bridge-class machines? | 1 | planned |
| X13 | Next-row prefetch and non-temporal stores per machine for out-of-cache batches | 2 | planned |
| [X15](experiments/X15-compiler-vectorised-scalar.md) | What does a current compiler get from the scalar pass loops (`omp simd`, the scalar text in the `avx2` region, `-march=native` as a ceiling), against a hand vocabulary? | 0 | done: omp simd rejected (compiler-dependent); old targets need hand vocabularies |
| [X16](experiments/X16-avx512-table.md) | `avx512` against `avx2` on AVX-512 machines: does `avx512` need its own cost table, or an R × C transpose for leading radices below C? | 1 | done (Cascade Lake only): own table in float (0.955 of before); `avx512` first in double (0.80–0.87 of `avx2`); float loses at batch 64 (1.02) because of tails (lengths 8 does not divide 1.20–1.23, the rest 0.78–0.85): per-plan choice and tail work go to phase 2 |
| [X17](experiments/X17-bluestein-threshold.md) | Above which prime does Bluestein beat the generic prime pass, per target? | 2 | done: Bluestein from 17 up (threshold 16) on the Mac, the Xeon (avx512, avx2) and the EPYC (avx2), double and float |
| [X18](experiments/X18-x86-kernels.md) | Why does FFTN trail FFTW on x86 (Xeon, EPYC), and what closes the gap? | 1 | done: driver divisions and a four-line chain fixed (Xeon 1.50 → 1.25 of FFTW, EPYC 1.37 → 1.29); next X10 addressing, radix 32/64, then X1 |
| [X19](experiments/X19-generator-rotations.md) | Do codelets with fewer multiplications by ±i (the real-coefficient form: one register swap per output pair, radix 16 7 against 19) make x86 faster without slowing NEON? | 1 | done: adopted (x86 0.976 of main: Xeon avx512 0.980, avx2 0.976, EPYC 0.971; Mac 0.991); port-5 uops −4 to −16 % |
| [X20](experiments/X20-call-overhead.md) | Where does a tiny serial execute's time go (entry, driver, kernel), and what does a leaner driver close at n ≤ 64? | 1 | done: adopted (flat programs, leaves called with addresses, a batch of leaf rows in one call): Mac 0.45 of main over the x20 cases, 8 × 4096 3.08 → 0.97 of FFTW; Xeon 0.57 (one case slower at that commit, gone after two entry fixes per the harness; fbench confirmation pending) |
| [X21](experiments/X21-phase1-cloud.md) | Phase 1 exit cloud checks: is anything badly off on a recent Intel Xeon (with MKL), Zen 5 and a Neoverse, and does `avx512` over `avx2` hold off Cascade Lake? | 1 | done: nothing badly off (FFTN ÷ FFTW over the tracking c2c rows: Emerald Rapids 1.17, Zen 5 1.12, Axion 0.81; FFTN ÷ MKL 1.19); avx512 wins in double everywhere, in float not on Zen 5, and loses 3× at 60 points in float; USD 1.36 |
| [X22](experiments/X22-icpx.md) | Intel's icpx: does FFTN build and pass with it, which floating-point flags keep the numerical contract, and is it faster than GCC and Clang on the Xeon? | 1 | done: supported with `-fno-fast-math` (set by CMake); not recommended (Clang's equal in double, 3 % behind on tracking rows); float 0.80 of Clang on 7-smooth rows from masked partial runs, a portable vocabulary follow-up |
| [X23](experiments/X23-masked-tails.md) | Do masked partial runs in the `avx512` and `avx2` vocabularies close most of icpx's float lead with GCC and Clang (X22), and become the default on x86? | 1 | done: rejected. B as first written slowed whole units (up to 1.15); amended B5 (partial units after the loops, constant counts) wins with GCC 14 (float 0.60 of main on `avx512`) but loses with Clang 23 on odd double lengths (`avx2` 75: 1.42), so the masked accesses are deleted; the partial-run test stays. Follow-up: Clang's odd-length loss |
| [X24](experiments/X24-lanes-x86.md) | Cheaper lanes on x86: with X20's per-call preparation in the lanes driver and first and last passes that build registers from 128-bit loads of each row (no gather or scatter), do the lanes beat the batched-leaf chain in double at 8–64 points and reach FFTW at 8–16? | 2 | done: both changes rejected (per-call driver: 1–7 % slower at 8–16 × 64; per-row edge passes: up to 1.17 on AVX-512 at 24–256 points). `lanes_up_to` in double stays 0 on x86; 2.2's double c2c acceptance on AVX-512 not met (lanes 1.31–1.53 of FFTW, chain 1.01–1.26) |
| [X25](experiments/X25-x86-diagnosis.md) | Why does FFTN trail FFTW in 1-D double on x86 (EPYC 1.21, Xeon 1.06): kernel instruction stream, memory hierarchy or fixed cost per call, per case group? Counters and FFTW's plans, no library change | 1 | done: long rows (both machines) and EPYC 256–1024 are the instruction stream (the grid's `block_copy` 40–52 % of FFTN's instructions; AVX2 loops 1.4–1.6 times FFTW's instructions for the same FP work, at IPC up to 4.7); ≤ 16 points the fixed cost per call (FFTW uses 128-bit codelets); memory only for Xeon r2c 65536 (chain of 32768) and EPYC 4096; Xeon 128–4096 at par, port 5 the limit |
| [X28](experiments/X28-short-rows.md) | Single short rows (≤ 120 points, batch 1): what is left after X20, and do a lighter entry, 128-bit leaves on x86 or a flat path for two-pass chains close it? (PLAN lead 4) | 1 | running |
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
| Xeon Gold 6240 (AVX-512, 2 × 18 cores) | AVX-512 and Intel timing; MKL; native AVX-512 tests | Slurm, shared: small jobs (`-c 8`), never exclusive |
| EPYC 7763 cluster (Zen 3, AVX2, 2 × 64 cores) | AVX2 timing; threading at 64; remote gates (GCC 14–16, Clang 23) | Slurm (DMOG), shared: `-c 8`–16, whole nodes only for threading runs |
| Cloud: Zen 4/5, recent Intel, Neoverse N1/V1/V2 | checks per phase exit; phase 6 tuning | rented per experiment, with a ledger |
| Old x86-64 (v2 level, or AVX without AVX2) | best effort | no hardware: the `x86_v2` and `avx` targets are timed forced on the Xeon and Zen 3, and verified by emulating Nehalem and Sandy Bridge |
