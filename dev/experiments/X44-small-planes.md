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

### D: diagnosis (screens; none decides)

Screens of the x44 catalogue against `main`'s library (ddf8737: `main` 310d8d6 with the
knob), forced column radices and blocks, FFTW (and MKL) in the same run; ratios are paired
medians against the rule's plan (A), ns per point from A.

- **EPYC, run 20261117-143058-da8f** (8 rounds, all loaded, band ±1.5 %). 32²: A 2.10 ns
  per point, its rows as a batch 0.73, so its column phase (2 × 16) costs 1.9 times its
  rows. One radix-32 lane pass (`COLUMN_RADICES=32`) 0.694, 4 × 8 0.873, 8 × 4 0.908, 16 × 2
  1.048, blocks of 4 or 8 columns 1.00; FFTW 0.694. Real 32² (complex columns of 32): radix
  32 0.80–0.82. 64² columns: radix 64 in one pass 1.33, 4 × 16 1.40 (8 × 8 stays). Columns
  through the chain 1.25–3.2. Run 20261117-143626-3be6 (band ±1.5 %), 64²–256² columns
  under 8 × 8, 16 × 8, 8 × 16, 4 × 32, 32 × 4, 4 × 8 × 8, 8 × 4 × 8, 8 × 32, 16 × 16: nothing
  beyond 0.96–0.98 (real 64² with 8 × 8 0.97–0.98, real 256² with 16 × 16 0.96 but c2c 256²
  1.33). The EPYC's real planes trail FFTW by 4–7 % at 64²–256² although FFTN's real rows
  beat FFTW's (r2c 64 × 64 rows: FFTW 1.14 of FFTN's time): FFTW's column phase of a real
  plane costs about 0.98 ns per complex point against FFTN's 1.38 at 64².
- **Xeon, run 20261117-143847-64ba** (8 rounds, none loaded, band ±2.9 %; with MKL). The
  column phase costs 1.3–1.95 times the rows (16² 1.3, 32² 1.45, 64² 1.95, 128² 1.28, 256²
  1.46), MKL's about 1.1–1.2 times its own; MKL ÷ FFTN on the c2c planes 0.73–0.92 (32²
  0.725, 16² 0.775, 256² 0.83, 128² 0.88, 64² 0.92) and on the rows as batches 0.91–1.02
  (c2c), 0.73–0.90 (r2c), 0.76–0.99 (c2r). Radix 32 in one pass: c2c 32² 0.772, real 32²
  0.86–0.87; 4 × 8 (the rule's), 8 × 4, 4 × 16, blocks of 8: 0.97–1.03.
- **Mac, runs 20261117-142835-4a2c and 20261117-143226-dcc7** (8 and 10 rounds, all loaded,
  bands ±4.3 % and ±3.0 %). A plane costs the sum of its two batches (32²: 1.06 against
  0.52 + 0.52); FFTW's plane costs less than twice its own batch because its columns are
  one codelet over the columns (`PLAN_NOTE`: `n1fv_32` and `n1fv_64` vector loops). FFTN's
  32² columns are already one radix-32 pass. At 64² FFTW's radix-64 codelet is the
  difference: FFTN's rows and columns of 64 as one radix-64 pass lose (1.06–1.14; rows
  through `LEAF_LIMIT=64` 1.19), and 4 × 16, 16 × 4, blocks of 8 or 32 change nothing beyond
  the band. The Mac's 32²–64² gap (FFTW 0.87–0.95 of FFTN's time in these runs) needs a
  faster radix-64 codelet on NEON, the generator's work (X42), not a plane program.

### P: columns of 32 as one lane pass (double)

On x86 the row of 32 is a pair leaf (two passes in registers), which a column phase cannot
take, so its columns got the row's radices as two lane passes over the tile (2 × 16 on
`avx2`, 4 × 8 on `avx512`). Arm P: a column of 17–32 points in double whose length has a
codelet with lane passes runs as that one pass, where its 32 legs fit the L1's ways
(`column_radices`); in practice columns of 32 (17, 19, 23 already had one pass; NEON's 32
already had one). Float is unchanged (not measured). It changes the c2c planes and volumes
with columns of 32 and the real planes whose complex columns have 32 points, on x86. Target
cases: the planes of the standard catalogue's plane and volume rows (filter
`^(c2c|r2c|c2r)-[0-9]+x[0-9]+(x[0-9]+)?-b[0-9]+-(op|ip)$`, 126 cases) whose program P changes;
the rule's case set (2) is those 126 (the x44 catalogue's planes are among them except 16²,
which P does not change, and its rows are 1-D). Rule as above, on the EPYC and the Xeon;
the Mac's plans are unchanged (checked by describe). Bits: the columns of 32 change route,
within the tolerances.

## Verdict

(Pending.)
