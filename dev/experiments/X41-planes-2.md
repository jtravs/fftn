# X41: the rest of task 4.1: real planes, small planes, long columns, threaded planes

- **Status:** running
- **Owner:** task 4.1 agent (second part), 2026-11-17
- **Phase / task:** 4.1

## Question

After X40 (lanes over blocks of adjacent columns; c2c planes 0.80–0.94 of FFTW), which
changes take the cases X40 left over FFTW and MKL to or below them, one thread and
threaded? Each sub-item below is its own question and its own arms; the answers decide the
multi-axis programs task 4.1 ends with.

## Hypotheses and changes under test

Sub-items, in the order they are worked; at most about three arms each.

- **R: the complex axes of real planes.** r2c and c2r planes still run their strided
  complex axes column by column (DESIGN.md 9.3 "Not built yet"); on C > 1 each column is
  copied element by element. These axes are c2c transforms over adjacent columns (the
  last, halved axis contiguous), exactly X40's case: give them X40's column phase
  (`column_block`, `add_column_phase`). Expected: real planes take about what c2c planes
  took in X40 (0.45–0.6 of main's time at 256² and beyond). Arm R: the column phase on
  the complex axes of r2c (after the real rows) and c2r (before them, from the input or
  the block to the block, or in place on the output). Bits: as X40 (each column is a
  lane computing its own chain; results move within the tolerances where the column
  radices differ from the chain's).
- **S: small planes in batches.** 8 × 6, 5 × 7, 16 × 16 in batches of 2, 32 × 32: a fixed
  cost per plane (records walked, kernels called per stage per plane). Diagnosis first
  (time per plane against the leaf and column calls it makes), then arms such as one call
  per stage over every plane of a batch where the planes are adjacent.
- **L: columns beyond L2** (Mac 2048 × 36 1.12–1.16 of FFTW): a block's tiles leave L2;
  a two-phase split of the column (the grid's way) or narrower blocks with the chain's
  radices, after a diagnosis.
- **T: threaded planes.** Column blocks are lines of the work graph (each member's
  scratch holds its tiles). Baseline: FFTN's thread pool against FFTW's threads and
  MKL's OpenMP threading at 8 lanes on the EPYC (whole cores, one CCX: 8 cores sharing an
  L3, `--distribution=block:block`) and the Xeon (8 cores of one socket); fix what loses.
  fbench gains `THREADS` for FFTW (`fftw_plan_with_nthreads`, libfftw3_threads, no OpenMP
  runtime) and an `mkl_omp` backend (oneMKL's GNU OpenMP threading layer) for this.
- **M: the Xeon against MKL at 16²–1000²** (1.08–1.38): diagnosis with counters only if
  time remains.

1-D plans are unchanged by every arm (their programs are identical; the tests' 1-D cases
and the tracking tier's 1-D cases are the collateral check).

## Method

- Cases: the standard catalogue, tier full. R: `^(r2c|c2r)-[0-9]+x[0-9]+-b[0-9]+-(op|ip)$`
  (the real planes: smoke planes in batches of 2, the planes group in and out of place,
  r2c plane batches of 16). S, L, M: the 37 c2c planes of X40,
  `^c2c-[0-9]+x[0-9]+-b[0-9]+-(op|ip)$`. T: the same planes and the threads catalogue's
  planes, plane batches and volumes, at 8 lanes (`THREADS=8`).
- Machines: Mac M5 Max (Homebrew Clang, `neon_fcma`, FFTW NEON build), EPYC 7763 (GCC 16,
  `avx2`), Xeon 6240 (GCC 14, `avx512`, with MKL). fbench A/B, interleaved rounds, A/A
  control; remote timing through `fbench_remote.py submit --serial` on shared 8-CPU
  allocations.
- Metric: per case the median paired ratio arm ÷ base, 95 % interval; the geometric mean
  over the cases; FFTN ÷ FFTW and ÷ MKL for the record (and for task 4.1's acceptance).
- Decision rule, written before running, per sub-item and machine-target (release
  toolchains: Clang on the Mac, GCC on x86-64): adopt an arm if (1) the geometric mean of
  arm ÷ base over the sub-item's target cases is at most 0.95 (R: the real planes whose
  program changes; S: the c2c planes of at most 1024 points; L: the planes whose column
  block changes; T: the threaded cases), and (2) no case of the sub-item's case set (and,
  for S, L and T, none of the 37 c2c planes at one thread) is slower than base by more
  than the larger of 2 % and the run's A/A band (a case just past it on a loaded machine
  is re-run before it decides), and (3) 1-D plans are unchanged. A win on some targets and
  a loss on others is adopted per target. Task 4.1's acceptance is judged separately on
  FFTN ÷ FFTW and ÷ MKL per case, one thread and threaded.

## Results

### R: the complex axes of real planes

Arm R (862b140) against `main` (4d00f0a), the 71 real planes, FFTW (and MKL) in the same
runs; R ÷ main is the paired ratio:

- Mac, run 20261117-102848-aba7 (10 rounds, all loaded, band ±2.8 %): geometric mean
  1.039, 20 cases slower, worst r2c 512² 1.22, 1024² 1.13–1.16, 480 × 640 1.12–1.14. On
  NEON double (C = 1) the chain reads a column at its stride without a copy, and a real
  plane's complex rows (n/2 + 1 columns, an odd multiple of 16 bytes apart for even n) do
  not crowd an L1 set as X40's power-of-two c2c columns did: the lanes' tiles only add
  traffic. Rule (1) fails: R rejected on the Mac.
- EPYC, run 20261117-103320-609d (10 rounds, none loaded, band ±1.0 %): geometric mean
  0.899, 47 faster (32 × 32 0.74, 36 × 2048 0.76–0.79, c2r 2048² 0.76–0.78), but 1024²
  1.09–1.11 (its 16-column tiles, 512 KiB, leave the 512 KiB L2; the chain's 16 KiB line
  stays in L1) and c2r 5 × 7 1.04. Rule (2) fails: R rejected as written.
- Xeon, run 20261117-103328-552d (6 of 10 rounds loaded, band ±18.5 %): geometric mean
  0.838, no verdict possible at that band (re-run below with arm R2).

Arm R2 (a new arm, its rule written before its runs; this paragraph was committed with it):
the same column phase for real planes, but only where C > 1, for columns of at least 32
points, with the tiles held to `column_tile_bytes` (a quarter of L2) at any stride, and if
a line of columns still exceeds that, X40's block where the column alone exceeds half of
L1, else the chain (`column_block`, `real`). The Mac's plans are `main`'s. Rule for R2: as
the decision rule above, over the real planes whose program R2 changes, on the EPYC and
the Xeon, against `main` in fresh runs.

R2 against the same build with every column through the chain (W: 3e405ec with
`COLUMNS=chain`, which isolates the column choice), EPYC run 20261117-105625-db9e (10
rounds, 3 loaded, band ±1.1 %): geometric mean 0.885 over the 71 (1.00 where R2 keeps the
chain), 32 × 30 0.74–0.75, 36 × 2048 0.75–0.78, 2048² 0.73–0.87; but 1024² 1.05–1.06 with 4
columns (rule (2) fails again) and c2r 2048 × 36 1.11 with a wide interval (19 columns, X40's
block of 16). Arm R3 (the third and last arm of R; rule written before its runs and
committed with it): as R2, but a block from the tile budget needs at least two L1 lines of
columns, and X40's wide block applies only where the stride is a page or more (each column
element on its own page) and the column exceeds half of L1; else the chain. On the EPYC
that keeps 1024² (8208-byte stride, a 16 KiB column) and 2048 × 36 (304-byte stride) on the
chain and 2048² on X40's block. Rule for R3: as for R2, on the EPYC and the Xeon.

## Verdict
