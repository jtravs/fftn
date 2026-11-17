# X44: the one-thread plane cases task 4.1 still misses

- **Status:** running
- **Owner:** X44 agent, 2026-11-17
- **Phase / task:** 4.1

## Question

Where does FFTN's one-thread time go in the four plane cases X41 left over the reference
(EPYC `avx2` c2c 32², 1.13–1.36 of FFTW; Mac NEON c2c 32²–64², 1.02–1.09 of FFTW; Xeon
`avx512` c2c planes from 16² up, 1.06 of MKL; real planes on the Xeon, 1.12–1.13 of MKL), and
which change to the column radices, the column kernels or the real-plane path removes it?
The answer decides task 4.1's one-thread plane programs.

## Hypothesis

- **EPYC 32².** X41's screen (run 20261117-115010-a798, x41 catalogue) shows a 32² plane at
  2.11 ns per point against 0.73 for its 32 rows as a 1-D batch, so its column phase costs
  about 1.9 times its rows, while the 16² and 64² columns cost 1.0 and 1.3 times theirs. The
  three column chains differ: 16² one radix-16 lane pass, 64² 8 × 8, 32² 2 × 16 (the row's
  radices, which column_radices keeps because a radix-2 last pass crowds no L1 set). The
  radix-2 pass is a whole trip through the 16-column tile for one butterfly per pair, and the
  radix-16 pass on 16 `ymm` registers spills. Expected: a column chain of 4 × 8, 8 × 4 or one
  radix-32 pass, or a narrower block, takes the column phase to about its rows' cost (32²
  near 1.5 ns per point, at FFTW's 1.48).
- **Mac 32²–64².** On the Mac a plane costs the sum of its two batches (X41). Whether FFTW's
  plane costs less than twice its own row batch, and where FFTN's column phase sits against
  its rows, decides between a column-chain change and nothing short of a fused plane kernel.
- **Xeon against MKL, c2c from 16².** Not diagnosed. Hardware counters (instructions,
  cycles, L1 and L2 misses, port 5) of FFTN's and MKL's planes and of FFTN's rows as batches
  first, then arms for what they show (rows, columns, copies or fixed cost).
- **Real planes against MKL on the Xeon** (and the EPYC and Mac real planes, near FFTW):
  not diagnosed; the same split into the real rows, the complex columns and the pairing.

## Change under test

- **D (diagnosis, no default change):** a test knob, `build_settings::column_radices`
  (fbench `COLUMN_RADICES`), forces a column phase's radices; with `COLUMN_BLOCK` and
  `COLUMNS=chain` it splits a plane's time into its rows (the x44 catalogue's 1-D batches)
  and its column phase under each column chain and block. Counters through `fbench run
  --pmu` on Linux (DMOG nodes, the Xeon) and perf or VTune on the Xeon.
- **Arms:** named in Results as the diagnosis supports them, one commit each, each with its
  target cases and any narrowing of the rule below written into this file and committed
  before its judging run. Candidates: column radices chosen by register budget or by the
  number of columns; a narrower column block for short columns; a whole-plane path for
  planes that fit in L1 (rows, then columns from one tile); a real plane's half-spectrum
  columns, padding or pairing pass.

Bits: an arm that changes a column phase's radices moves results within the derived
tolerances (each column still a lane computing its own chain; the tests compare with the
chain's columns within twice the tolerance). A block change keeps bits.

## Method

- Cases: the standard catalogue's 108 plane cases (c2c, r2c, c2r; tier full; filter
  `^(c2c|r2c|c2r)-[0-9]+x[0-9]+-b[0-9]+-(op|ip)$`) and the x44 catalogue (c2c and real
  planes of 16²–256² with their rows as 1-D batches); for an arm that touches a kernel 1-D
  rows also use, the standard catalogue's tracking tier's one-dimensional cases.
- Machines, compilers, instruction sets, thread modes: EPYC 7763 (GCC 16, `avx2`), Xeon
  6240 (GCC 14, `avx512`, with oneMKL), Mac M5 Max (Homebrew Clang, `neon_fcma`, FFTW NEON
  build). One thread. fbench A/B, interleaved rounds, A/A control. Judging runs on the EPYC
  and the Xeon go through `fbench_remote.py submit --serial` (8-CPU shared allocations); the
  Mac runs locally, one fbench at a time. Screens may use any queue gap but never decide.
- Metric: per case the median paired ratio arm ÷ base with its 95 % interval; the geometric
  mean over the arm's target cases; FFTN ÷ FFTW and ÷ MKL for the record (task 4.1's
  acceptance is judged on those, per case).
- Decision rule, written before any judging run, per arm and machine-target (release
  toolchains: GCC on x86-64, Clang on the Mac): adopt the arm if (1) the geometric mean of
  arm ÷ base over its target cases (the plane cases of the case set whose program it
  changes, unless its paragraph names a narrower set before its run) is at most 0.95, and
  (2) no case of the 108 planes or of the x44 catalogue is slower than base by more than the
  larger of 2 % and the run's A/A band (a case just past it on a loaded machine is re-run
  before it decides), and (3) 1-D plans are unchanged, or, where the arm changes a kernel or
  plan that 1-D rows use, no one-dimensional tracking case is slower than base beyond the
  same limit. A win on some machine-targets and a loss on others is adopted per target. A
  rule changed after seeing results means a new, named arm judged by fresh runs.

## Results

(Pending.)

## Verdict

(Pending.)
