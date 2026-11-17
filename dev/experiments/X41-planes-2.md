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

## Verdict
