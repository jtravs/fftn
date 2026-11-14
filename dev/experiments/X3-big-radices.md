# X3: few big passes (radix 32 and 64) against radix-16 chains and grids

- **Status:** running
- **Owner:** task 1.6 session (r3264), 2026-11-14
- **Phase / task:** 1.6

## Question

Do passes of radix 32 and 64 (generated, ordered codelets, X8) beat the planner's radix ≤ 16
choice for rows of 512 to 65536 points, as chain passes or as grid phases, on any target and
precision? The answer decides whether planner v1's cost tables get radix 32 and 64 columns
(DESIGN.md 11.2) and whether the grid's phase rule may use them (DESIGN.md 8.5).

## Hypothesis

FFTW plans 1024 as 32 × 32 and 4096 as 64 × 64 on the Xeon (X18 item 4), and FFTN's radix
≤ 16 needs three passes at 512–4096 and four at 16384. Fewer passes mean fewer trips over
memory and less per-pass overhead, and the arithmetic per point and bit is nearly the same
(vocabulary operations per point per log₂ R of the twiddled codelet: 1.55 at 16, 1.62 at
32, 1.67 at 64). Against that, X8 counted stack spills on every target: radix 64 stores
67–681 values to the stack per pass, radix 32's grouped and lane passes 76–222, worst on
AVX2 (16 registers). The prototype's radix 20–32 passes lost everywhere. Expected:

- AVX-512 (32 registers of 512 bits): radix 32 passes win at 512–2048 and in grid phases;
  radix 64 at best ties (X18's expectation: 512–2048 and 16384 to about 1.0 of FFTW).
- AVX2 (16 registers): both lose, radix 64 badly.
- NEON (32 registers of 128 bits, C = 1 in double): radix 32 may win in double, where the
  32-point leaf already beat the chain by 16–22 % (X8).

## Change under test

No library default changes in this first measurement: every arm is the same build of
commit M (this branch's first commit), with internal build settings forced through the
fbench FFTN backend (bench/README.md, "FFTN options"; none of them is a switch of the
library, so `dev/KNOBS.toml` is untouched):

- A: the public plan (planner v1's chain, or the grid beyond L1).
- `cR1.R2...`: the chain forced to a radix sequence containing 32 or 64 (`RADICES`), with
  no more passes than the fewest a radix-16 chain needs: the best four by X2's cost
  estimate padded to each C of the target (bench/x3/run.py).
- From 4096 up: `chain` (the planner's radix-16 chain, `STRATEGY=chain`), `grid` (the grid
  with the radix-16 phase rule, `STRATEGY=grid`), `g64` (the grid's phase rule allowed
  radices up to 64: `GRID_PHASE_RADIX=64`, a build setting added for this experiment),
  and `gN1.N2[.N3]` (that phase rule on the two two-phase splits and the three-phase split
  with the fewest passes, `GRID_FACTORS`).

If a big radix wins, the second step is the planner change (commit T): cost-table columns
for 32 and 64 fitted with bench/x2/fit.py from an X2-style search campaign that includes
them, and/or the phase rule's largest radix, measured A/B against M.

## Method

- Cases: bench/catalogues/x3.toml: c2c rows of 512, 1024, 2048, 4096, 8192, 16384, 32768
  and 65536 points, forward, out of place, batches 1 and 64, double and float (32 cases).
- Driver: `bench/x3/run.py --isa T`, one fbench run per length (interleaved process rounds,
  6 to 30 valid rounds by fbench's stopping rule, 1200 s budget, the A/A control).
- Machines, compilers, instruction sets, one thread:
  - Xeon Gold 6240 (Cascade Lake), GCC 14.4, Slurm `-c 8` on the shared node: `avx512`
    and `avx2`;
  - EPYC 7763 (DMOG node16–21), GCC 16, Slurm `-c 8 --hint=nomultithread`: `avx2`;
  - Apple M5 Max, Homebrew Clang, the shared Mac: `neon_fcma`.
  The Xeon and the Mac are loaded by other work; every claim is judged against its run's
  A/A band.
- Metric: fbench's median paired ratio arm ÷ A per case with its 95 % interval.
- Decision rule, written before running:
  1. *Screen (this measurement).* A big radix is worth a planner change for a target and
     precision when its best big-radix arm beats A by more than 3 % with the interval clear
     of the band in at least two of the eight lengths at one batch or more, as a chain
     (lengths where A is a chain) or as a grid (lengths where A is a grid). Otherwise
     radix 32 and 64 stay out of the planner for that target and precision, and the
     result is recorded.
  2. *Adoption (commit T against M, interleaved A/B on the same machine).* Adopt the
     change for a target and precision if it improves the geometric mean by at least 2 %
     over the affected lengths (chains: the X2 catalogue's lengths that are multiples of
     32; grids: the X3 lengths from the grid threshold up) at batches 1 and 64, and no
     case is slower than the run's band plus 2 %. Elsewhere planner v1's choices stay
     exactly as they are (`tests/core/data/rule_radices.txt` changes only where 32 or 64
     is chosen).

## Results

Interim (2026-11-14, commit a0a09bf/7e4c438; `bench/x3/run.py report RUN...` gives the
tables). Ratios are arm ÷ A, bands ±1.0–1.4 %.

- **Xeon 6240, `avx512`** (GCC 14; runs 20261114-175401-8e3c, -175657-851c, -180137-b6e4,
  -180554-c5b7, -181331-5e3c): float chains with radix 32 win: 512 32.16 0.82 / 0.85
  (b1 / b64), 1024 32.32 0.92 / 0.94, 2048 8.8.32 0.98, 4096 8.16.32 0.97–0.98. Double
  ties at 512–2048 (0.95–1.00). At 4096 and 8192, where A is a grid, the radix-16 chain
  is 0.68–0.81 of the grid and chains with radix 32 0.56–0.79 (8192 f32 16.16.32 0.56):
  the grid starts too early on this 32 KiB L1. Grids with radix 32/64 phases lose
  (1.07–1.8) except 8192 b64 double (64 × 128, 0.95). 16384–65536 and `avx2` pending
  (job 15568).
- **EPYC 7763, `avx2`** (GCC 16; runs -175446-3f09, -175550-be92, -175649-0c4a,
  -180042-48bd, -180740-effd): big chains lose or tie at 512–2048 (1.00–1.07); 4096 f32
  b64 8.32.16 0.79 against the planner's 16.16.16 (radix 16 is slow on 16 registers,
  X18). At 4096 double the radix-16 chain is 0.75–0.78 of the default grid, at 8192
  0.87–0.93 of the three-phase grid. Grid 64 × 128 with a radix-64 phase 0.95–0.97 at
  8192 double, against a three-phase grid. 16384–65536 pending (job 509113).
- **M5 Max, `neon_fcma`** (runs 20261114-174600-8f9b, -174623-cfd1, -174822-ee44,
  -175029-6d2e, -175434-9b3c, -180103-8956, -180803-ddb8, -181715-180b): chains with 32
  win only at 512 f32 (16.32, 0.94) and 2048 b64 double (0.95); elsewhere tie or lose;
  radix-64 grid phases lose 1.5–2.0 at 4096–8192 and never change the plan above it.
  **Artifact:** at 32768 and 65536 double, A and A~ ran 1.2–1.4 times slower than the
  arms `grid` and `g64`, whose plans are identical to A's; re-run alone
  (20261114-182537-c7af) A, `STRATEGY=rules`, `STRATEGY=grid` and no ISA option agree
  within 1 %. So comparisons against A at those lengths are invalid; the same-plan
  `chain`/`grid` arms are the reference there.

Long rows and the Xeon's `avx2` (runs 20261114-182424-264a, -183721-1230, -185515-f896 for
`avx512` 16384–65536; -191534-f69f to -201742-9125 for `avx2`; EPYC 20261114-181711-ff80,
-182751-37e4, -184128-5d86; bands ±1.0–1.5 %):

- **Chains beyond the grid threshold.** On the Xeon a chain with radix 32 beats the default
  grid at every length up to 16384 in float (0.57–0.65 at 16384 on `avx512`, against 0.69–0.71
  for the radix-16 chain) and in double (0.84–0.92 at 16384), on both instruction sets; at
  32768–65536 they mostly win at batch 1 (0.76–0.98) and lose at batch 64 in double and at
  65536 (1.04–1.41). On the EPYC big chains lose at
  16384–65536 (1.09–3.7) while the radix-16 chain beats the grid at 16384 double (0.88–0.90).
  Where the grid starts is X12b's question; X3 only records that radix 32 widens the chain's
  lead where the chain is the better strategy on the Xeon.
- **Grid phases with radix 32 and 64 at 32768–65536.** The phase rule allowed radix 64
  (`g64`, or the splits 32 × 32 × 32 and 32 × 32 × 64 with single-pass phases) beats the
  radix-16 grid on the Xeon at 32768 and 65536 in both precisions and on both instruction
  sets (`avx512` 0.89–0.98, `avx2` 0.87–0.95), and on the EPYC at 32768 (0.94–0.98) but not
  at 65536 in float (1.04). At 4096–16384 the same rule loses (1.0–1.8): there it puts a
  64-point phase where two radix-8 passes were cheaper. A phase radix that depends on the
  length (32 from 32768) would pass rule 1 on the Xeon; it is left as a follow-up for after
  X12b moves the grid threshold, since both decide which plans those lengths get.
- **Xeon `avx2` float** also gains from radix 32 chains at 512 (16.32 0.86) and 1024 (32.32
  0.94), unlike the EPYC on the same instruction set (1.00–1.07): a table per target cannot
  serve both, and AVX-512 is the Xeon's default target (in float it is 1.6 times faster
  than `avx2` there: 0.83 against 1.35 ns per point at 512).

Screen (rule 1): passes for `avx512` float (512, 1024, 4096 by more than 3 %); fails for
`avx512` double at chain lengths, for the EPYC's `avx2` and for the Mac. Step 2 for
`avx512` float: campaign xeon-avx512-big (search.py `--radices 2-16,32,64 --require
32,64`, 26 multiples of 32 from 64 to 2048), Slurm job 15573 after 15568.

## Verdict

Pending.
