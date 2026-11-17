# X40: where batched 2-D c2c loses, and plane programs that do not

- **Status:** done (first pass: arms L and B adopted; task 4.1 continues)
- **Owner:** task 4.1 agent, 2026-11-17
- **Phase / task:** 4.1

## Question

Where does FFTN's time go on batched 2-D c2c planes against FFTW's and MKL's, and which
change to the plane program removes it? The answer chooses the first arms of task 4.1.

## Hypothesis

Before this experiment a plane is axis by axis, line by line (DESIGN.md 9.3 "as built" was
the chain for every line): the rows through the chain, then every column alone, its
elements `nx` apart. On targets with several complex values per register (C > 1: `avx2`,
`avx512`) a strided column is copied element by element into scratch, run there and copied
back (chain.hpp); on NEON double (C = 1) the chain's passes read and write the column at
its stride. Either way a column touches one element per cache line and the next column
reads the same lines again; when `nx` is a power of two of 4 KiB or more those lines all
fall into a few L1 and L2 sets, so the second column finds none of them (PLAN suspected
aliasing on the EPYC). Expected: the column phase is most of the time, the row phase near
the 1-D batch's speed, and FFTW runs the columns as vector loops of codelets across
neighbouring columns (rank-2 plans split into a rank-1 row plan and a vrank-1 column plan).

The fix DESIGN.md 9.3 already names: the strided axis through the lanes strategy over
blocks of neighbouring columns (C columns per vector, so no copy and no transpose), each
block's chain in a scratch tile, with the block's first pass reading and last pass writing
all B columns of one row before the next row (whole cache lines, each line read once).

## Change under test

- Arm D (diagnosis, no library change): the baseline (`main` 841c414) against FFTW with its
  printed plans (`PLAN_NOTE=1`) and MKL; and on the Mac, a scratch harness timing a plane's
  row phase (the 1-D batch of rows) and column phase (a 1-D plan of one strided axis with a
  batch of adjacent columns) apart.
- Arm L (lanes columns): in a c2c plan of two or three axes, every axis but the last whose
  neighbouring columns are adjacent on both sides (the last axis has stride 1 in the phase's
  source and destination) runs as the lanes strategy over blocks of B adjacent columns:
  `lanes_first` reads the block's B columns, the chain runs on a tile in scratch,
  `lanes_last` writes them back (in place or not). The block's units run its column groups
  innermost, so the first and last passes take each row's B columns at once. B, the tail
  (the last `nx mod B` columns, in blocks of C padded with idle lanes) and the scratch follow
  the lanes strategy's rules (`lanes_block`), with B at least a cache line's columns. The
  rows and 1-D plans are unchanged (their programs are identical; the tests' 1-D cases and
  the tracking tier's 1-D cases are the collateral check).
- Further arms only as the diagnosis and arm L's result support (block ordering or padding
  against power-of-two strides; plane-at-a-time blocking for planes beyond L2).

Bits: arm L changes the column phase's arithmetic route (each column's chain runs as a
lane of a tile instead of a copied line), so results move within the derived tolerances;
determinism across teams holds (a block is a line of the lines loop).

## Method

- Cases: the standard catalogue's c2c 2-D planes, tier full, filter
  `^c2c-[0-9]+x[0-9]+-b[0-9]+-(op|ip)$` (37 cases: the smoke planes in batches of 2, the
  planes group in and out of place, the plane batches of 16).
- Machines, compilers, instruction sets: Mac M5 Max (Apple Clang via Homebrew LLVM,
  `neon_fcma`, FFTW NEON build), EPYC 7763 (GCC 16, `avx2`), Xeon 6240 (GCC 14, `avx512`,
  with MKL). One thread. fbench A/B, interleaved rounds, A/A control; remote timing through
  `fbench_remote.py submit --serial`.
- Metric: per case the median paired ratio arm ÷ main, 95 % interval; the geometric mean
  over the cases; FFTN ÷ FFTW and ÷ MKL for the record.
- Decision rule, written before running: adopt an arm on a machine-target (release
  toolchain) if (1) the geometric mean of arm ÷ main over the 37 cases is at most 0.95, and
  (2) no case is slower than main by more than the larger of 2 % and the run's A/A band (a
  case just past it on a loaded machine is re-run before it decides), and (3) 1-D plans are
  unchanged (describe and program identical; checked by the tests and by construction). A
  win on some targets and a loss on others is adopted per target.

## Results

### Diagnosis (arm D)

Baselines, `main` 841c414, the 37 cases, one thread: FFTN ÷ FFTW (geometric mean) Mac 1.46
(run 20261117-042807-7868, band ±3.3 %), EPYC 2.05 (20261117-043112-f117, every round on a
loaded shared node, band ±1.4 %), Xeon 1.50 and ÷ MKL 1.85 (20261117-043928-a79e, band
±1.2 %). The gap grows with the plane: EPYC 64² 1.9, 256² 2.6, 1024² 4.1, 2048² 6.8–7.6.

- **Where the time goes** (Mac scratch harness, one 1024² plane): the rows, as a 1-D batch,
  1.5 ns/pt; the columns, as a 1-D plan of one strided axis, 8.9 ns/pt; the plane 9.7 against
  FFTW's 3.8. The rows run near FFTW's speed; the columns are the gap, as hypothesised.
- **FFTW's plans** (`PLAN_NOTE=1`): rank-2 split into the rows (`dft-vrank>=1` of a
  Cooley–Tukey row plan) and the columns. Small planes run the columns as a strided codelet
  over all columns (`dft-direct-64-x64`; EPYC: `dft-buffered-64-x64`, copied to rows first);
  larger ones as `dft-buffered-<n>-x8` to `-x32`: 8–32 adjacent columns copied into a buffer
  as rows, transformed there, copied back (2048² on the EPYC: blocks of 16); 256² on NEON and
  480 × 640 on `avx2` an in-place square transpose (`indirect-transpose`) around row
  transforms; 2048 × 36 a strided Cooley–Tukey over the 36 columns. So FFTW's columns are
  blocks of adjacent columns, which the lanes give without the copies.
- **Passes over memory**: two sweeps (rows, then columns) in both libraries; FFTN's column
  sweep re-read every line once per column it holds.
- **Cache sets** (Mac harness, after arm L's first version): with the rules' radices 16 x 16
  (256) and 8 x 16 (128) ending in a radix-16 pass whose 16 legs land in one L1 set (a
  power-of-two stride times s), 128² and 256² planes took 4.3 and 4.7 ns/pt; ending in radix 8
  (8 x 4 x 8, 16 x 8) 1.9 and 2.1. The first pass's reads in one set did not matter (16 x 8 at
  128 was as fast as 4 x 4 x 8). This is the aliasing PLAN suspected, on the write side.
- **`planning::measure`** does not apply: its candidates are those of one-axis plans
  (core/candidates.cpp); multi-axis plans are built by the rules alone.

### Arm L: lanes columns, and the block

Arm L (25ec492) against `main`, with the block forced to 4, 8, 16, 32 columns (arms L4–L32;
L is the rule of the first commit: a line of columns doubled while two tiles fit half of L1):
Mac run 20261117-045615-6c45 (band ±2.5 %): L 0.682, L16 0.670, L32 0.665 of main's time;
EPYC 20261117-045905-1429 (band ±1.2 %): L 0.547, L4 0.558, L8 0.517, L16 0.500, L32 0.511;
Xeon 20261117-045900-987f (band ±2.4 %): L 0.650, L4 0.676, L8 0.632, L16 0.610, L32 0.614.
Wider blocks won where every element of a column lies on its own page (2048²: EPYC L4 0.183,
L16 0.126; Xeon L 0.79, L32 0.62; Mac 1024² L 0.62, L32 0.46). Where rows share pages, tiles
beyond L2 lost: Xeon 2048 × 36 L4 0.79, L16 1.09, L32 1.20; EPYC 2048 × 36 1.02–1.06 with any
block (the one rule failure: the chain's single column fits L2 there). Small planes lost
with a tail block: Xeon 8 × 6 L 1.10–1.12 (4 + a padded 2), L8 0.93. Arm L2 (a2e1484) sets
the block from these (four lines of columns, at most the columns there are; halved to keep
two tiles within L2/4 where the stride is under a page, the chain beyond); arm L3 (6ccd962)
keeps L2's x86 plans and takes 16 instead of 32 columns on Apple below a page (L2 lost
1.023–1.027 at 100² on the Mac, run 20261117-051455-fdc4).

### Arm B: a loop of leaves is one leaf call

B (eecbd9c) against L (25ec492): Mac 0.919 (run 20261117-050548-5e55, band ±1.5 %), EPYC
0.931 (20261117-050839-4acf, band ±2.7 %), Xeon 0.907 (20261117-051127-5d41, band ±2.9 %):
planes of short rows 0.57–0.85 (8 × 6, 5 × 7, 16 × 16, 32 × 32 on NEON). B changes no
program, only how a loop of leaves is called; the cases past the band (EPYC 36 × 2048 op
1.106, Xeon 1024² ip 1.048, 480 × 640 ip 1.041) have no leaf loop and were within the band in
the confirmation runs.

### Confirmation: the adopted program against `main`

Arms N (`main` 841c414) and X (L3 with B on the Mac, 6ccd962; L2 with B on x86, a2e1484, whose
x86 plans are L3's), FFTW and MKL in the same runs: Mac 20261117-052127-ef6a (10 rounds, none
loaded, band ±1.8 %), EPYC 20261117-051803-7358 (10 rounds, all on a loaded node, band
±1.9 %), Xeon 20261117-051743-62dd (10 rounds, none loaded, band ±1.5 %). X÷main is the
paired ratio; main÷F, X÷F and the MKL columns are FFTN's time over the competitor's.

| case | Mac X÷main | Mac main÷F | Mac X÷F | EPYC X÷main | EPYC main÷F | EPYC X÷F | Xeon X÷main | Xeon main÷F | Xeon X÷F | Xeon main÷M | Xeon X÷M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8x6-b2-op | 0.452 | 2.65 | 1.20 | 0.384 | 3.18 | 1.22 | 0.542 | 3.11 | 1.68 | 0.65 | 0.35 |
| 8x6-b2-ip | 0.453 | 2.69 | 1.22 | 0.383 | 3.24 | 1.24 | 0.533 | 3.01 | 1.61 | 0.64 | 0.34 |
| 16x16-b2-op | 0.627 | 1.41 | 0.89 | 0.498 | 1.94 | 0.97 | 0.412 | 1.98 | 0.82 | 3.23 | 1.33 |
| 16x16-b2-ip | 0.638 | 1.40 | 0.90 | 0.495 | 1.93 | 0.96 | 0.409 | 1.97 | 0.81 | 3.29 | 1.35 |
| 5x7-b2-op | 0.411 | 2.74 | 1.13 | 0.476 | 2.73 | 1.30 | 0.519 | 2.51 | 1.30 | 0.55 | 0.28 |
| 5x7-b2-ip | 0.413 | 2.75 | 1.13 | 0.470 | 2.72 | 1.28 | 0.512 | 2.56 | 1.31 | 0.55 | 0.28 |
| 32x30-b2-op | 0.792 | 1.14 | 0.91 | 0.656 | 1.54 | 1.01 | 0.503 | 1.36 | 0.68 | 1.43 | 0.72 |
| 32x30-b2-ip | 0.793 | 1.01 | 0.80 | 0.627 | 1.35 | 0.84 | 0.489 | 1.22 | 0.60 | 1.68 | 0.82 |
| 32x32-op | 0.684 | 1.55 | 1.06 | 0.644 | 2.20 | 1.42 | 0.417 | 1.92 | 0.80 | 3.31 | 1.38 |
| 32x32-ip | 0.685 | 1.59 | 1.09 | 0.618 | 2.24 | 1.38 | 0.396 | 1.91 | 0.76 | 3.39 | 1.34 |
| 64x64-op | 0.972 | 1.11 | 1.08 | 0.529 | 1.86 | 0.99 | 0.386 | 1.70 | 0.66 | 2.82 | 1.09 |
| 64x64-ip | 0.955 | 1.11 | 1.06 | 0.521 | 1.89 | 0.99 | 0.379 | 1.69 | 0.64 | 2.96 | 1.12 |
| 128x128-op | 0.506 | 1.77 | 0.90 | 0.488 | 1.98 | 0.97 | 0.478 | 1.46 | 0.70 | 2.54 | 1.22 |
| 128x128-ip | 0.502 | 1.75 | 0.88 | 0.476 | 1.90 | 0.90 | 0.468 | 1.42 | 0.66 | 2.52 | 1.18 |
| 256x256-op | 0.445 | 1.75 | 0.78 | 0.307 | 2.54 | 0.78 | 0.514 | 1.48 | 0.76 | 2.35 | 1.21 |
| 256x256-ip | 0.445 | 1.78 | 0.79 | 0.304 | 2.34 | 0.71 | 0.487 | 1.40 | 0.68 | 2.51 | 1.22 |
| 512x512-op | 0.429 | 1.92 | 0.82 | 0.300 | 3.23 | 0.97 | 0.544 | 1.57 | 0.86 | 2.44 | 1.33 |
| 512x512-ip | 0.426 | 1.79 | 0.76 | 0.329 | 2.72 | 0.89 | 0.523 | 1.45 | 0.76 | 2.58 | 1.35 |
| 1024x1024-op | 0.474 | 2.12 | 1.00 | 0.237 | 4.05 | 0.96 | 0.619 | 1.50 | 0.93 | 1.96 | 1.21 |
| 1024x1024-ip | 0.450 | 2.25 | 1.01 | 0.224 | 4.24 | 0.95 | 0.734 | 1.42 | 1.04 | 1.83 | 1.34 |
| 2048x2048-op | 0.558 | 1.75 | 0.98 | 0.126 | 7.14 | 0.90 | 0.673 | 1.54 | 1.03 | 1.36 | 0.91 |
| 2048x2048-ip | 0.544 | 1.88 | 1.02 | 0.117 | 6.90 | 0.81 | 0.692 | 1.53 | 1.06 | 1.43 | 0.99 |
| 96x96-op | 0.959 | 0.66 | 0.63 | 0.561 | 1.40 | 0.79 | 0.422 | 1.22 | 0.51 | 2.27 | 0.96 |
| 96x96-ip | 0.981 | 0.62 | 0.61 | 0.553 | 1.34 | 0.74 | 0.415 | 1.12 | 0.47 | 2.30 | 0.96 |
| 100x100-op | 0.999 | 0.76 | 0.76 | 0.793 | 1.15 | 0.91 | 0.756 | 1.06 | 0.80 | 1.75 | 1.33 |
| 100x100-ip | 1.008 | 0.70 | 0.71 | 0.784 | 1.08 | 0.84 | 0.751 | 1.01 | 0.75 | 1.77 | 1.33 |
| 480x640-op | 0.445 | 1.63 | 0.72 | 0.481 | 1.53 | 0.74 | 0.627 | 1.34 | 0.84 | 2.03 | 1.27 |
| 480x640-ip | 0.442 | 1.53 | 0.68 | 0.490 | 1.43 | 0.70 | 0.613 | 1.30 | 0.80 | 2.17 | 1.33 |
| 1000x1000-op | 0.884 | 0.74 | 0.65 | 0.861 | 1.01 | 0.87 | 0.769 | 0.97 | 0.74 | 1.22 | 0.94 |
| 1000x1000-ip | 0.891 | 0.71 | 0.63 | 0.843 | 0.98 | 0.82 | 0.862 | 0.91 | 0.78 | 1.24 | 1.07 |
| 36x2048-op | 0.384 | 1.62 | 0.62 | 0.202 | 3.53 | 0.71 | 0.540 | 1.28 | 0.69 | 2.10 | 1.13 |
| 36x2048-ip | 0.367 | 1.68 | 0.61 | 0.312 | 1.98 | 0.62 | 0.499 | 1.23 | 0.61 | 2.39 | 1.19 |
| 2048x36-op | 0.929 | 1.24 | 1.16 | 0.967 | 1.06 | 1.03 | 0.765 | 1.03 | 0.79 | 1.51 | 1.15 |
| 2048x36-ip | 0.949 | 1.18 | 1.12 | 0.977 | 1.00 | 0.98 | 0.734 | 0.95 | 0.69 | 1.65 | 1.21 |
| 32x32-b16-op | 0.689 | 1.58 | 1.09 | 0.643 | 2.10 | 1.35 | 0.417 | 1.82 | 0.76 | 3.16 | 1.32 |
| 64x64-b16-op | 0.955 | 1.13 | 1.08 | 0.523 | 1.79 | 0.94 | 0.475 | 1.45 | 0.69 | 2.28 | 1.08 |
| 128x128-b16-op | 0.513 | 1.81 | 0.93 | 0.484 | 2.00 | 0.97 | 0.528 | 1.47 | 0.78 | 2.28 | 1.20 |
| geometric mean | 0.613 | 1.44 | 0.88 | 0.453 | 2.07 | 0.94 | 0.537 | 1.49 | 0.80 | 1.87 | 1.00 |

Collateral: 1-D plans are unchanged by construction (the column phase applies only to axes
other than the last of a multi-axis c2c plan; arm B only to loops over lines, which 1-D
programs walk only in grids, whose bodies are never a single leaf). Tracking tier, `main`
against X (61 cases): Mac 20261117-052850-a99d (band ±1.6 %): 6 faster (the c2c planes, 64³
0.319), 55 no evidence, worst r2c 2048 × 36 1.034; EPYC 20261117-052446-7d2b (band ±2.6 %,
every round loaded): 6 faster (64³ 0.088), 55 no evidence, worst r2c 65536 1.038; Xeon
20261117-053128-6752 (band ±2.1 %): 8 faster (64³ 0.460), 51 no evidence, 2 slower, c2c 128
(one row, an unchanged flat program) 1.125 and r2c 64² (unchanged program) 1.048: re-run
below.

## Verdict

**Adopted on all three machine-targets (rule met).** Mac `neon_fcma` (Clang): 0.613 of
main's time, no case slower beyond the band (worst 100² ip 1.008); EPYC `avx2` (GCC 16):
0.453, worst 2048 × 36 ip 0.977; Xeon `avx512` (GCC 14): 0.537, worst 1000² ip 0.862. 1-D
plans unchanged. FFTN ÷ FFTW over the 37 planes: Mac 1.44 → 0.88, EPYC 2.07 → 0.94, Xeon
1.49 → 0.80; ÷ MKL 1.87 → 1.00. DESIGN.md 9.3 ("as built"), 8.3 and 8.5 updated, decision
log entry, lessons 3.22 and 3.23.

Task 4.1's acceptance (at or below FFTW and MKL, one thread and threaded) is met on the
geometric means against FFTW and on the mean against MKL, but not case by case:

- planes of a few dozen points in batches of 2 (8 × 6, 5 × 7): 1.13–1.30 of FFTW on the Mac
  and the EPYC, 1.6–1.7 on the Xeon (fixed cost per plane: the row phase walks its records
  per transform, the columns are a padded block; MKL is 3 times slower there);
- 32 × 32 on the EPYC 1.35–1.42 of FFTW (rows of 32 are a two-pass chain per row, walked
  row by row; FFTW runs both axes as one codelet call over all lines);
- the Xeon against MKL: 1.08–1.38 from 16² to 1000² (MKL keeps a plane in cache and makes
  fewer passes; not diagnosed further here);
- long columns beyond L2: Mac 2048 × 36 1.12–1.16, Xeon 2048² 1.03–1.06 and 1024² ip 1.04.

Next (task 4.1's remainder): rows of a plane as one batched call per stage (or the lanes)
for short rows; a two-phase split of long columns (the grid's way) where a block's tiles
leave L2; the strided complex axes of r2c and c2r planes through the same column phase;
threaded planes (blocks are already lines of the work graph, each member's scratch holds its
tiles: measure against FFTW's and MKL's threads). Task 4.2's volumes take the same column
phase for both strided axes (64³ is in the tracking tier), with blocks of pencils sized by
L2 and per-member scratch of a block.
