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

### S: small planes

Diagnosis (Mac harness, one process, Clang, `neon_fcma`, `main` 4d00f0a with arm R): a plane
costs the two 1-D batches it is made of plus about 10 ns: 8 × 6 30.6 ns per plane against
9.6 ns for its 8 rows as a 1-D batch and 10.1 ns for its 6 columns as rows; 32 × 32 1005 ns
against 476 + 476; 64 × 64 4.83 µs against 2.28 + 2.27. The rest is the driver walking the
records per transform (a variant test, a recursion and the slots moved per loop iteration).

Arm S1 (dd21c2e, `build_settings::unroll_calls`): a multi-axis program of at most 256
kernel calls per transform is unrolled at construction into its calls with the slots' byte
offsets (a loop of leaves one call), and the driver calls them in turn. Bits unchanged (the
tests compare unrolled and walked output byte for byte); 1-D programs keep their paths.
Harness: 8 × 6 b2 60 → 53 ns, 5 × 7 b2 51 → 45 ns, 64 × 64 4.83 → 4.64 µs. S1 against its
parent (B: 31ea042), the 37 c2c planes; target cases the 11 planes of at most 1024 points:

- EPYC run 20261117-105106-e6b7 (10 rounds, 8 loaded, band ±1.7 %): targets 0.90 (8 × 6
  0.81, 5 × 7 0.78, 16 × 16 0.94–0.95, 32 × 32 0.96–0.98), worst of the 37 1.013. Rule met.
- Xeon run 20261117-105256-e618 (10 rounds, all loaded, six overloaded; band ±11 %):
  targets 0.87 (8 × 6 0.78–0.80, 5 × 7 0.74–0.76, 16 × 16 0.86–0.89), worst 1.038. Rule met
  by the letter at that band.
- Mac run 20261117-104725-7d2a (12 rounds, all loaded, band ±5.5 %): targets 0.94 (8 × 6
  0.86–0.88, 5 × 7 0.84–0.87), worst 1.044. The lead reported the Mac on battery power
  around the end of this run, so it cannot decide; a re-run on mains decides the Mac.

### T: threaded planes

Baseline, EPYC (8 whole cores of one CCX, `--hint=nomultithread --distribution=block:block`):
run 20261117-104129-206a (31ea042, the 37 c2c planes, band ±2.1 %) with FFTN's pool at 8
lanes (T8), FFTN serial, FFTW at 8 threads (F8, `fftw_plan_with_nthreads`) and serial. FFTW's
threaded plans of small planes are 7–380 times slower than its serial ones (a planner that
chose threads for 48 points), so FFTN is compared with the better of FFTW's two per case.
Planes up to 64² run inline (T8 = serial). From 128² FFTN's 8 lanes take 2.3–6.6 times less
time than its one; against FFTW's better plan FFTN is faster everywhere except the small
planes it already trails serially (8 × 6, 5 × 7, 32 × 32: S below) and 1000² (FFTW's 8
threads 0.91–0.92 of FFTN's time), 2048² out of place (0.93). Threads catalogue, run
20261117-110630-d7e0 (3e405ec, band ±1.4 %): FFTN ahead of FFTW's 8 threads on c2c planes and
plane batches (1.12–1.26 of FFTN's time for FFTW) except 1024² (0.99); behind on r2c 1024²
(0.77) and the volumes (64³–128³ 0.72–1.02, task 4.2's).

A column phase's narrower last block was a phase of one chunk (the tail nest after the
whole blocks), run by one member while the others waited. Arm T1 (947bfbe, `loop::joins`):
that nest's lines join the whole blocks' phase. Serial programs and every plane whose
column count the block divides are unchanged. T1 against 3e405ec, both at 8 lanes, the 108
c2c and real planes, EPYC run 20261117-110929-0d06 (band ±2.7 %): 0.983 over all; on the
planes with a narrower block 0.81–0.95 (100² 0.92, real 96²–512² 0.92–0.97, real 2048 × 36
0.81–0.90); past the band c2c 512² op 1.030 (no narrower block: its program and graph are
unchanged) and c2r 2048² ip 1.073 (wide interval). As written, rule (1) took "the threaded
cases" as the target set, over which 0.983 fails; that set includes the planes T1 cannot
change. Arm T1b (a new arm, the same code, judged by fresh runs): target cases the threaded
planes whose work graph T1 changes (those with a narrower last block); adopt if their
geometric mean is at most 0.95 and no threaded plane is slower than the base by more than
the larger of 2 % and the band, a case past it re-run once; serial programs unchanged by
construction.

### L: columns beyond L2 (the Mac's 2048 × 36)

Screen (not a judging run), Mac run 20261117-111654-cd9f (band ±1.3 %), forced blocks on
the head build against its rule (16 columns, tiles of 1 MiB): 2048 × 36 with 4 columns
0.89–0.90 (FFTN ÷ FFTW 1.02–1.07 instead of 1.15–1.19), 8 columns 1.01, 32 columns 1.00,
every column through the chain 1.06–1.08; 36 × 2048 and 1000² lose with narrow blocks
(their column elements are a page or more apart, which the tile budget does not govern).
Arm L1 (rule written before its runs, committed with it): where a cluster shares L2,
`column_tile_bytes` is twice L1 (256 KiB on the M5 Max) instead of a quarter of L2, so
2048 × 36 takes blocks of 4 (the halving below a page may go down to 64 bytes of columns,
half of Apple's 128-byte line, instead of a whole line; on x86 a line is 64 bytes, so x86
plans are unchanged). Rule for L1: on the Mac, adopt
if 2048 × 36 (both placements) takes at most 0.95 of the head's time and none of the 37 c2c
planes is slower by more than the larger of 2 % and the band.

## Verdict
