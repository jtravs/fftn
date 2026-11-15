# X25: why FFTN trails FFTW in one-dimensional double on x86

- **Status:** running
- **Owner:** x25 session, 2026-11-15
- **Phase / task:** 1 (exit criteria); PLAN leads 1–5; follows X18, X10, X19, X1, X12b

## Question

Why does FFTN still trail FFTW in one-dimensional double precision on x86 (tracking tier,
FFTN ÷ FFTW: EPYC 1.21, Xeon 1.06, against the Mac's 0.97), and is each case group's gap
in the kernels' instruction stream, the memory hierarchy, or the fixed cost per call? The
answer orders and sharpens PLAN's leads 1–5 before any of them gets an experiment of its
own. A diagnosis: no library change.

## Hypothesis

From the baselines (runs 20261115-101319-f98f EPYC, 20261115-114105-ea43 Xeon) and X18,
X10, X19, X1:

- **128–4096 points** (EPYC 1.22, Xeon 1.07; Mac 0.91, so x86-specific): the instruction
  stream. On the Xeon port 5 (shuffles) is the bottleneck (X10: 73 % busy against FFTW's
  55 %, 58 % after X1's split runs); on Zen 3, with two shuffle pipes and four-wide
  dispatch, the expectation is instruction count and front-end throughput rather than one
  pipe. These rows fit in L1 or L2, so cache misses should explain nothing.
- **≤ 64 points** (EPYC 1.38, Xeon 1.51): fixed cost per call (FFTN's entry and driver
  against FFTW's plan-call path), plus leaves that use one lane per register (X18 item 3).
- **≥ 16384, 46080, 100000, r2c 65536** (EPYC 1.20–1.98, Xeon 1.03–2.06): memory
  hierarchy (more passes over L2 or memory than FFTW's plan), and for 46080 and 100000 the
  grid's phases with factors FFTN handles less well than FFTW's plan.

## Change under test

None: a diagnosis. Two small tools are added to fbench on the way, neither of which
changes what fbench measures by default:

- the FFTW backend's option `PLAN_NOTE=1` writes FFTW's own plan (`fftw_sprint_plan`, the
  public API) into the case's note, next to FFTN's radices;
- the harness option `--pmu EVENTS` (Linux) counts hardware events in user mode for the
  executes of one case (`perf_event_open`), after its timed batches, and reports them per
  call. On DMOG, which has no `perf` but allows user-mode counting
  (`perf_event_paranoid` = 2), this is how the EPYC's counters are read.

FFTW's source code is not read (whether to read GPL source is a maintainer decision,
postponed): FFTW's plans come from its public API, its time and counters from running it.

## Method

- Cases (double, one thread, hot, out of place, forward): c2c rows 8, 16, 64, 60, 120,
  256, 1024, 4096, 65536, 100000, 46080; r2c 65536.
- Machines: Xeon Gold 6240 (Cascade Lake, `avx512`; GCC 14; FFTW 3.3.11 with AVX-512 and
  MKL 2026.1), EPYC 7763 (Zen 3, `avx2`; DMOG; GCC 16; FFTW 3.3.11 with AVX2). Slurm
  `-c 8`, shared, never exclusive. FFTW at MEASURE as fbench runs it.
- Per case, FFTN and FFTW (and MKL on the Xeon):
  1. The plans: FFTN's `describe()` radices (fbench note) and FFTW's `fftw_sprint_plan`.
  2. Counters per transform: cycles, instructions, IPC; uops by port (Xeon: ports 0, 1, 5
     and the load and store ports) or by FP pipe (Zen 3: pipes 0–3); L1D refills and L2
     misses; front-end stalls (Xeon: uops not delivered; Zen 3: dispatch stalls).
  3. Static analysis where counters cannot attribute: uiCA (Cascade Lake) and llvm-mca
     (`znver3`) on FFTN's hot loops.
- Attribution per group: instructions per transform against IPC (what share of the cycle
  gap each explains); the busiest port or pipe and its occupancy against FFTW's; loads and
  stores per point; L1 and L2 misses per point; for ≤ 64 points the cycles outside the
  kernels (counters of a batch of 1 against the per-row counts at batch 64).

What each outcome would mean:

- **Kernel instruction stream** (more instructions at equal IPC, or a saturated port or
  pipe that FFTW's code does not saturate): codelet and pass changes (lead 3; X1's split
  runs extended, fewer shuffles, larger radices).
- **Memory hierarchy** (L1 or L2 misses per point well above FFTW's, IPC lower at similar
  instruction counts): plan and blocking changes (lead 1; X12 on x86, fewer passes over L2).
- **Fixed per-call cost** (cycles at batch 1 well above batch 64 per row, instructions
  outside the kernels): driver work (lead 4).

## Results

Pending.

## Verdict

Pending.
