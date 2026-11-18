# X44: the one-thread plane cases task 4.1 still misses

- **Status:** done (P adopted on x86; N rejected)
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
- **Xeon counters** (`fbench-<lib> run --pmu`, user mode, one process per library and
  event group, FFTW from its own wisdom; Slurm job 16351, ddf8737; per point). FFTN's rows as
  batches run MKL's instructions and cycles (16–64 points: 3.4–3.9 instructions, 1.6–2.2
  cycles both). The planes do not: 32² 12.2 instructions and 4.66 cycles against MKL's 7.9
  and 3.32; 256² 20.0 and 14.1 against 14.0 and 11.4; 512² 24.1 and 19.7 against 15.8 and
  14.8. Subtracting the rows, FFTN's column phase runs about twice MKL's instructions, with
  1.3–1.5 more scalar uops per point (port 1, which only scalar work uses beside 512-bit
  vectors), about 1.3 more loads, and at 256²–512² two to three times MKL's L1 refills; at
  32² the front end failed to deliver 2.2 uops per point (MKL 0.09). `perf record` of 64²
  and 256² (Slurm, `srun`): the column edge passes `lanes_first` and `lanes_last` each take
  as many instructions as the row phase's whole pair leaf; their hot instructions are scalar
  address arithmetic, loop state spilled to the stack and to `xmm` registers, the per-unit
  run counter, and each run of C columns reloading its group's R − 1 broadcast twiddles
  (two loads each). FFTW's printed plans: one `n1fv` codelet over the columns at 16², 32²
  and 64², buffered columns from 128².
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

P against `main` (B: ed598e1), the 126 cases, FFTW in the same run (P÷B paired):

- EPYC, run 20261117-153628-5fcb (Slurm 511991; 12 rounds, 4 loaded, band ±1.5 %): 0.793
  over the 17 changed cases (c2c 32² 0.72, 32² × 16 0.75, 32 × 30 0.73, 32³ 0.84; real 32²
  and 32 × 30 0.81–0.83, real 32³ 0.80–0.83); 0.970 over all 126. FFTN ÷ FFTW at 32²: 1.39
  → 1.00 (op and ip), 32² × 16 1.31 → 0.97. Worst unchanged case c2c 36 × 2048 in place
  1.032 (its program is unchanged), just past the 2 % limit: re-run (Slurm 512068).

- Xeon, run 20261117-165710-e283 (Slurm 16371; 12 rounds, 5 loaded, band ±4.3 %; arms B, P,
  N, FFTW, MKL in one run; the filter's 132 cases): 0.870 over the 17 changed cases (c2c 32²
  0.77–0.81, 32² × 16 0.78, 32 × 30 0.85–0.87, 32³ 0.88; real 0.86–0.93); worst unchanged
  case r2c 36 × 2048 1.036, inside the band. FFTN ÷ MKL at 32²: 1.36–1.38 → 1.06–1.09, 32² ×
  16 1.30 → 1.06. Rule met: P adopted on `avx512`.

### N: the column edge passes' runs in an inner loop

From the Xeon's counters (D): in `lanes_first` and `lanes_last` with `columns_inner`, a whole
block of columns (every run of C full) over whole groups or copies runs its B / C runs in an
inner loop at one group or copy, so the leg addresses and the loop state are set up once
per group instead of once per run; `lanes_first` also loads a group's twiddles once for
its runs at radices up to 8 (2b3056c, with db0f483). Same operations on the same values:
bits unchanged; 1-D plans never set `columns_inner`, so their programs and kernels' paths
are unchanged. Counter screen on the Xeon against P (cycles per point, one process each,
three alternations): 64² 0.84, 128² 0.89, 96² 0.91, 100² 0.93, 36 × 2048 0.89, 512² 0.92,
real 64² 0.95–0.97, the rest 0.98–1.04 (noise about ±4 %); the Mac (run
20261117-152334-8e57, 8 rounds, band ±3.5 %) 0.995 over the 108 planes, nothing beyond the
band. Target cases: the cases of the 126 (planes and volumes) whose plans have a lanes
column phase with at least one whole block. Base: P. Rule as above per machine-target (the
Xeon and the EPYC by `fbench_remote.py submit --serial`, the Mac locally); where N is not
adopted on a target, the fast path is compiled out for it.

N against P on the EPYC, run 20261117-164127-b5e3 (Slurm 512049; 12 rounds, all loaded, band
±1.6 %; the filter's 132 plane and volume cases): 0.936 over the 100 with a lanes column
phase (64³ 0.84, 48³ 0.85, 32³ 0.86, 96² 0.87, 64² 0.87–0.88, 64² × 16 0.87, 100³ 0.87),
0.951 over all; FFTN ÷ FFTW 0.981 → 0.931 over the 132. Past the 2 % limit: c2c 4 × 6 × 8
1.034, 32 × 30 × 2 in place 1.031, 32² in place 1.021, c2r 32² 1.020 (columns of 32 or of
4–8 points, single-pass leaves that N does not touch): re-run with P against B's 36 × 2048
in one job (Slurm 512179).

N against P on the Xeon, the same run 20261117-165710-e283: 0.962 over the 110 cases with a
lanes column phase, 0.969 over all; c2c planes 0.92 (64² 0.83–0.84, 64² × 16 0.87, 256² in
place 0.87, 128² 0.88–0.89, 100² 0.88, 36 × 2048 0.89), real planes 0.96–0.97, volumes 0.96
(64³ 0.86); worst c2r 8 × 6 × 2 in place 1.038, inside the band. Rule (1) fails on `avx512`
(0.962 against 0.95): N is not adopted there as written. Over MKL the c2c planes would go
from 0.95 (P) to 0.905 and 64² to 0.90–0.93; a narrower arm for `avx512` would need its own
rule and fresh runs (a decision for the maintainer, below).

Re-run on the EPYC, run 20261117-192713-da02 (Slurm 512179; 12 rounds, all loaded, band
±1.0 %; B, P, N on the six cases past the limit): P ÷ B 36 × 2048 0.999 and 1.001, 32 × 30 ×
2 in place 0.752, 32² in place 0.738, c2r 32² 0.811 (P's rule met on the EPYC); N ÷ P 32 ×
30 × 2 0.991, 32² 1.011, c2r 32² 1.012, 36 × 2048 0.911–0.913, but c2c 4 × 6 × 8 1.029 again,
past the 2 % limit: rule (2) fails on `avx2`, and N is rejected there too.

N against P on the Mac, run 20261117-154001-6cda (10 rounds, all loaded, band ±1.9 %; the 126
cases and six more volumes of the filter): 0.997 over all, 256² 0.97, 64³ 0.94, 48³ 0.96;
r2c 5 × 7 1.06 (a plan without a column phase). Rule (1) fails on the Mac: N is not adopted
on NEON, where the fast path is compiled out.

## Verdict

- **P adopted** (columns of 17–32 points as one lane pass, double): Xeon 0.870 over the 17
  changed cases; EPYC 0.793 (its one unchanged case past the limit, c2c 36 × 2048 in place
  1.032, re-run: 0.999); NEON plans unchanged. EPYC 32² ÷ FFTW 1.39 → 1.00, Xeon
  32² ÷ MKL 1.37 → 1.07. DESIGN 9.3 and the decision log.
- **N rejected** on every target: on `avx2` it met rule (1) (0.936) but c2c 4 × 6 × 8 was
  1.034 and, re-run, 1.029 against a ±1.0 % band (rule (2)); on `avx512` 0.962 against 0.95
  (its c2c planes 0.92); on NEON 0.997. Its code is removed (764b74a). LESSONS 4.18.
- No arm for the Mac: its 32²–64² gap is FFTW's radix-64 codelet (`n1fv_64`), against which
  FFTN's radix-64 pass loses in rows and columns alike (1.06–1.19); a faster generated
  radix-64 codelet on NEON is the generator's work (X42).
- Column radices and blocks beyond P: no forced radix sequence or block width moved a plane
  of 64²–512² by more than 2–4 % on the EPYC or the Xeon (D; after N, on the Xeon, radix 64
  in one pass 1.17 at 64²).

Not met (task 4.1, one thread), from run 20261117-165710-e283 with the adopted code (P;
N's numbers are not the adopted code's): the Xeon against MKL on c2c planes 64² (1.07–1.08), 128² (1.14–1.16),
256² (1.21–1.23), 100² (1.39), 480 × 640 (1.32–1.33), 512² (1.32–1.34), 1024² (1.24–1.32),
16² × 2 (1.24–1.25), 36 × 2048 and 2048 × 36 (1.14–1.23), 32² (1.06–1.09), though their
geometric mean is 0.95 (MKL is slow on others); real planes against MKL (geometric means
r2c 1.005, c2r 1.023; up to 1.6 at 16² × 2 and 1.3–1.5 at 256²–1024²), where the real rows
as 1-D batches already take 1.1–1.4 times MKL's time (the pairing pass's scalar work, X6's
question); EPYC real planes 64²–256² 4–7 % behind FFTW. Next: the column edge kernels'
per-call set-up (a radix-16 column leaf spills 32 leg addresses for four runs at 16²), the
middle lane passes' per-unit set-up (in `lane_pass`, shared with rows), and N again
without its loss at tiny volumes (a minimum block or column count) if the maintainer wants
one.
