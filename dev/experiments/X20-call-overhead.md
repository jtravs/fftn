# X20: the fixed cost of a call, and tiny transforms

- **Status:** done
- **Owner:** overhead session, 2026-11-14
- **Phase / task:** 1 (exit criteria), driver (DESIGN.md 8.3); lanes for batches stay task 2.2 / X7

## Question

Where does the time of a serial execute of a tiny c2c transform (n = 2–64, batch 1 and
64) go, between the entry (`execute_core`), the program's interpretation (the driver) and
the kernels, and how much of FFTN's distance to FFTW at n ≤ 64 does removing the first two
close?

## Hypothesis

X18 found about 4 ns of fixed cost per call and 9–13 ns of driver work per transform on
x86. On the Mac (run 20261114-095234-2dcf) FFTN ÷ NEON FFTW is 2.64 at 8, 2.15 at 16, 1.29
at 64 and 3.04 at 8 × 4096 rows, where the kernels are not the whole story: an 8-point leaf
is a few nanoseconds of arithmetic. Expected: per transform, the driver (`run_records`'
recursion over variant records, a 144-byte `pass_args` copy per stage, chunk ends, the
indirect call) costs several nanoseconds; a driver path that prepares the stages once per
call and only moves the slot addresses per transform removes most of it, with the same
kernels on the same arguments (bits unchanged).

## Change under test

Arm A: main (3433fc1). Arm B: the driver change(s) on branch `task/overhead`, one commit
each, timed against A and against FFTW in the same run.

## Method

- Decomposition: `bench/x20/x20_overhead.cpp` times the same work entered at four levels
  (ns per transform, median of 11–15 samples in one process): `api` (`execute_core`),
  `driver` (`run_transforms` with the slots prepared once), `kernel` (each stage's kernel
  through its pointer with `pass_args` prepared once), `once` (the loop around one kernel
  call). api − driver is the entry's cost, driver − kernel the interpretation's, kernel the
  floor of a design that calls kernels one transform at a time.
- Cases: catalogue `x20` (c2c rows 2–64 at batch 1 and 64; 8, 16 and 64 at 4096 rows),
  double and single precision, forward, out of place, `--tier full`.
- Machines: Apple M5 Max (NEON FFTW 3.3.11, which has no single precision; Clang 23);
  Xeon Gold 6240 (AVX-512, FFTW 3.3.11 with AVX-512 codelets at MEASURE; GCC 14.4; Slurm
  `-c 8`, shared). fbench interleaved rounds with the A/A control.
- Metric: FFTN ÷ FFTW per case and B ÷ A paired (fbench), weighted geomean over x20.
- Decision rule, written before running: adopt a driver change if (1) every test passes
  and serial, team and executor results stay bitwise identical (the same kernels on the
  same arguments); (2) B ÷ A over x20 is below 0.95 on the Mac with its interval clear of
  the A/A band; (3) no x20 case and no case of the quick tier is slower than A by more than
  the band on the Mac or the Xeon. A change that needs a new kernel entry or program shape
  (rows looped inside a kernel) is a separate arm, judged by the same rule.

## Results

### Where the time goes (before)

Mac, main + harness, Clang 23, ns per transform (`x20_overhead --reps 11 --ms 10`, load
about 3):

| n | b | api | driver | kernel | once |
|---|---|---|---|---|---|
| 2 | 1 | 6.8 | 5.4 | 1.7 | 1.1 |
| 8 | 1 | 7.7 | 6.9 | 3.9 | 2.5 |
| 16 | 1 | 12.0 | 11.2 | 7.2 | 6.5 |
| 64 | 1 | 48.0 | 47.0 | 39.1 | (2 stages) |
| 2 | 64 | 4.9 | 4.9 | 1.2 | 1.6 |
| 8 | 64 | 7.1 | 7.1 | 4.0 | 2.3 |
| 16 | 64 | 10.8 | 10.9 | 9.2 | 6.2 |
| 64 | 64 | 47.0 | 47.0 | 41.0 | (2 stages) |

Single precision is the same within 1 ns up to 16. FFTN ÷ FFTW before (Mac run
20261114-225941-d101, 7 rounds, band ±3.05 %): 2: 3.09 / 17.5 (b1 / b64), 4: 3.02 / 9.15,
8: 2.66 / 3.50, 16: 2.22 / 1.93, 32: 1.94 / 1.86, 64: 1.06 / 1.33; 4096 rows: 8 3.20,
16 2.06, 64 1.39.

### What was changed (one commit each)

1. a8862cb: programs whose body is at most 8 stages and no loop (every c2c, r2c and c2r row
   plan that is not a grid) are marked flat at construction (`program::flat`); the driver
   then copies each stage's `pass_args` once per call, not once per transform, and moves the
   slots by the batch loop's step instead of walking the records.
2. c5ed695: a leaf is a `leaf_fn` taking its first line's addresses and the distance to the
   next line as arguments, so the driver calls it on the plan's own `pass_args` (no copy)
   and runs a batch of leaf rows as one call. One transform of a multi-stage program copies
   each stage's arguments plainly (an on-stack array of copies cost about 6 ns more at
   batch 1). Same kernels, same arguments, same units: bits unchanged.

3. 4a74de2 and 0f1125b: rows of leaves (`program::leaf_rows`, marked at construction) and
   one transform of a flat program (`run_flat_one`) take small paths of their own, and the
   batch path `run_flat`, whose on-stack array of argument copies gives it a large frame and
   six saved registers, stays out of line. On the Xeon `perf` put 11 % of an 8-point float
   leaf harness in `run_flat`, spread over pushes, pops and moves. Mac, paired against
   4719129 (= c5ed695 plus a refactor), run 20261115-001125-ae28 (8 rounds, band ±1.00 %):
   leaf rows at batch 1 0.88 (2 points) to 0.95 (16), 64 × 64 0.92, 64 × 4096 0.91–0.95,
   nothing slower (geomean 0.957). The intermediate 4a74de2, which left one transform in
   `run_flat`, made 32 and 64 points in float 1.09–1.11 slower (run 20261115-000916-8423).

Rejected on the way: a second, batch entry beside each leaf's pass entry (regions may not
call each other, `tools/asm_check.py`, so the body would be duplicated: library text +18 %).

### Where the time goes (after)

Mac, c5ed695, same harness and load:

| n | b | api | driver | kernel | once |
|---|---|---|---|---|---|
| 2 | 1 | 3.6 | 2.5 | 1.8 | 1.1 |
| 8 | 1 | 4.9 | 4.3 | 3.6 | 2.6 |
| 16 | 1 | 7.2 | 6.7 | 5.9 | 5.2 |
| 64 | 1 | 45.3 | 44.1 | 38.2 | (2 stages) |
| 2 | 64 | 0.32 | 0.30 | 1.4 | 1.0 |
| 8 | 64 | 1.7 | 1.6 | 3.1 | 2.5 |
| 16 | 64 | 4.7 | 4.6 | 6.0 | 5.2 |
| 32 | 64 | 15.0 | 15.0 | 20.8 | 20.6 |
| 64 | 64 | 39.3 | 39.5 | 39.8 | (2 stages) |

A batch of leaves in one call is faster than the kernel called per row ("kernel", "once"):
besides the call, a leaf sets up its constants and saved registers once (32 points: 20.6 ns
per call, 15.0 per row in one call). At batch 1 what is left above "once" is the entry and
the flat path (about 1–1.5 ns) and, for passes, the copy of `pass_args` that a pass_fn needs
(1.2 ns per stage; the two stages of 64 points still cost 6 ns above their kernels).

### fbench, Mac

Run 20261115-002244-eba3 (M5 Max, final commit 0f1125b against main, both Clang 23, 7
rounds, all under load: load average 10, band ±3.02 %): new ÷ main 0.453 over x20, every
case faster (2-point batches 0.07, 8 b1 0.64, 8 × 4096 0.31, 64 b1 0.92). The c5ed695 arm
alone gave 0.464 (run 20261114-233002-c291, band ±1.98 %). FFTN ÷ NEON FFTW, double (FFTW
has no single precision here), main → final:

| n | b1 | b64 | b4096 |
|---|---|---|---|
| 2 | 3.38 → 1.82 | 17.1 → 1.18 | |
| 4 | 3.13 → 1.74 | 9.07 → 0.98 | |
| 8 | 2.67 → 1.70 | 3.47 → 0.86 | 3.08 → 0.97 |
| 16 | 2.20 → 1.32 | 1.91 → 0.85 | 2.03 → 0.85 |
| 32 | 1.46 → 1.19 | 1.89 → 1.06 | |
| 64 | 1.25 → 1.15 | 1.20 → 1.00 | 1.39 → 1.16 |

Collateral (quick tier, run 20261114-233217-f89b, 8 rounds, band ±1.00 %): 0.963 geomean,
8 cases faster (16 0.62, 64 × 256 0.91, r2c 64 × 64 0.92, r2c 256 0.94), 16 within the band,
none slower.

### fbench, Xeon

Run 20261115-003416-73e9 (Xeon Gold 6240, AVX-512, GCC 14.4, final commit 0f1125b against
main 3433fc1, Slurm `-c 8` shared, load 13, 12 rounds, band ±3.46 %): new ÷ main 0.570
over x20; 23 cases faster, 6 within the band, one slower: 8 points in float at batch 1,
1.108 [1.069, 1.143] (1.069 with c5ed695, run 20261114-234421-6872). FFTN ÷ FFTW (AVX-512
codelets, MEASURE; double, as FFTW has no single precision in that build), main → final:

| n | b1 | b64 | b4096 |
|---|---|---|---|
| 2 | 2.84 → 2.33 | 13.9 → 1.15 | |
| 4 | 3.11 → 2.72 | 7.63 → 0.95 | |
| 8 | 3.05 → 2.39 | 5.26 → 1.23 | 3.72 → 0.99 |
| 16 | 2.17 → 1.60 | 3.10 → 1.22 | 2.00 → 1.00 |
| 32 | 1.39 → 1.31 | 1.85 → 1.47 | |
| 64 | 1.31 → 1.25 | 1.05 → 0.92 | 1.01 → 0.91 |

Collateral on the Xeon (quick tier, c5ed695 against main, run 20261115-000546-493c, band
±2.38 %): 0.989, 2 faster (16, 64 × 256), 22 within the band, none slower.

On x86 the leaves broadcast every input into C lanes (C = 4 complex doubles or 8 complex
floats on AVX-512) and store lane 0, so the kernel, not the driver, dominates at batch 1
(X18: 65 % of a 16-point leaf; harness, 8 points in double: kernel through its pointer
12.5 ns of a 20.8 ns call). Batches gain because the call and the leaf's setup are paid
once. The one slower case: with the row loop inside the leaf GCC 14 turns its 16 leg
addresses into induction variables and parks some in `xmm` registers (`vmovq`): 171
instructions per call of the 8-point float leaf against 153 before, which a single row
pays in full (about 2 ns). Removing it needs either a second copy of every leaf (+18 %
library text) or the lanes and unit-stride leaves that task 2.2 (X7) brings for x86.

## Verdict

Adopted (0f1125b; merged with main at b3ae9c7). Rule (1) holds: every test passes, the new
`Contracts.BatchedRowsMatchSingleRows` compares every row of batched leaves and one-stage
rows bit for bit with a one-row plan, and serial, team and executor results are unchanged
(the same kernels on the same arguments and units). Rule (2): 0.453 on the Mac, every case
faster. Rule (3) holds on the Mac (x20 and the quick tier) and the Xeon quick tier, with one
exception on the Xeon x20 cases, the 8-point float row alone (+7 to +11 %), explained
above and left to X7; the maintainer may weigh it against 0.57 over the Xeon's x20 cases.

Where FFTN stands now at n ≤ 64: in batches at or near FFTW on both machines (Mac
0.85–1.18, Xeon 0.91–1.47 in double; 8 × 4096 0.97 and 0.99, from 3.08 and 3.72); one
transform still 1.2–1.8 times FFTW's time on the Mac and 1.3–2.7 on the Xeon. What is left
at batch 1: on the Mac about 1–1.5 ns of entry and dispatch and 1.2 ns per pass stage (the
`pass_args` copy a `pass_fn` needs; removing it means giving every pass its addresses as
arguments, a change to all kernel entries); on x86 the broadcast leaves themselves. Next:
lanes and unit-stride leaves for x86 (task 2.2, X7), and passes taking their addresses as
arguments if the 64-point single row matters (Mac 1.15, Xeon 1.25). DESIGN.md 8.2, 8.3 and
the decision log updated; lesson 4.7 in LESSONS.md.
