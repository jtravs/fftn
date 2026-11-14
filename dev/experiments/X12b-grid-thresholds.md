# X12b: The grid's thresholds per machine

- **Status:** done (adopted)
- **Owner:** grid-thresholds agent, 2026-11-14
- **Phase / task:** 1.7 (follow-up of X12 and X3)

## Question

Which cache rules should set the grid's thresholds (`grid_min_bytes`: chain or grid;
`grid_three_bytes`: two or three phases; DESIGN.md 8.5) so that each machine gets its best
measured values? The defaults (the L1 data cache; an eighth of L2) were set on the Mac,
whose L1 (128 KiB) and L2 (16 MiB) are four and sixteen times the x86 machines'.

## Hypothesis

On x86 both switches come too early. X3 (Xeon Gold 6240, 32 KiB L1, 1 MiB L2): at 4096 and
8192 points the radix-16 chain took 0.68–0.81 of the default grid's time; on the EPYC 7763
(32 KiB, 512 KiB) 0.75–0.78 at 4096 double and 0.87–0.93 at 8192. On the EPYC (X12's run
20261114-135001-ba70) three phases from 1 MiB instead of 64 KiB took 0.969 of the
default's time, never switching to three phases 1.226. Expected: the grid should start
near a few times L1 or a fraction of L2 on x86 (a 32 KiB L1 holds a 2048-point chain's
lines, but L2 serves the chain's passes well up to about L2/4), and three phases near L2;
on the Mac the defaults stay best. A rule of both caches, such as max(L1, L2/k), may fit
all three machines.

## Change under test

Profile fields only, set per arm with fbench options (bench/README.md, "FFTN options"); no
library switch. Arm A: the default profile. Arms B–E: `GRID_MIN_BYTES` = 2·L1, L2/4, L2/2,
L2. Arms F–H: `GRID_THREE_BYTES` = L2/2, L2, 2·L2. Arm I: FFTW (the machine's SIMD build at
MEASURE), the reference for FFTN ÷ FFTW. Raising `grid_min_bytes` also moves the r2c
pairing's fusion limit (`fusion_of`, which uses the same field), so real rows are in the
case set.

| machine | L1d | L2 | `GRID_MIN_BYTES` A–E | `GRID_THREE_BYTES` A, F–H |
|---|---|---|---|---|
| Xeon Gold 6240 | 32 KiB | 1 MiB | 32K, 64K, 256K, 512K, 1M | 128K, 512K, 1M, 2M |
| EPYC 7763 | 32 KiB | 512 KiB | 32K, 64K, 128K, 256K, 512K | 64K, 256K, 512K, 1M |
| Apple M5 Max | 128 KiB | 16 MiB | the rule's values (quick check) | the rule's values |

## Method

- Cases: `bench/catalogues/x12b.toml`: c2c rows of 2048 to 4194304 points (powers of two),
  one transform, double and single; c2c 4096 and 16384 points × 64 rows, both precisions;
  r2c rows of 8192, 32768 and 131072 points (double). Out of place, hot.
- Machines: the Xeon (Slurm, 8 CPUs of a shared node, GCC 14.4, `avx512` and a second run
  with every FFTN arm `ISA=avx2`); the EPYC (DMOG Slurm, 8 cores of a shared node, GCC 16,
  `avx2`); the Mac (shared, Homebrew Clang, `neon_fcma`), after the x86 runs, with the
  candidate rule's values. Serial.
- Command (Xeon values shown):

  ```bash
  scripts/remote/fbench_remote.py submit xeon A=fftn:<c> B=fftn:<c>+GRID_MIN_BYTES=65536 \
    C=...+GRID_MIN_BYTES=262144 D=...=524288 E=...=1048576 F=fftn:<c>+GRID_THREE_BYTES=524288 \
    G=...=1048576 H=...=2097152 I=fftw:<c> -- --catalogue x12b --tier full
  ```

- Metric: per case, the median paired ratio X ÷ A with its 95 % interval; per arm the
  weighted geomean over the cases the field can change (`GRID_MIN_BYTES`: rows of more
  than L1 bytes up to the largest value tried, the batched and the real cases;
  `GRID_THREE_BYTES`: rows of more than L2/8 bytes); the A/A band.
- Decision rule, written before running:
  1. Per machine and instruction set, a value beats the default when its geomean's interval
     lies below 1 − band and no affected case is slower than 1 + max(band, 3 %). The best
     value is the lowest geomean (the default if none beats it).
  2. The new defaults are rules in the fastest core class's L1 and L2 sizes, chosen among
     simple forms (a multiple of L1, a fraction of L2, the larger of the two), that give on
     every machine and instruction set a value whose geomean is within max(band, 1 %) of
     that machine's best; failing that, the form with the smallest worst-case excess.
  3. The Mac's defaults change only if its check finds a value better by rule 1.
  4. Report FFTN ÷ FFTW for the double b1 rows before (arm A ÷ I) and after (the new
     rule's arm ÷ I) on each x86 machine.

## Results

All runs serial, hot, on shared nodes (fbench rounds interleaved, A/A control): the Xeon
through Slurm with 8 CPUs (GCC 14.4), the EPYC on DMOG with 8 cores (GCC 16.2), the Mac
under load (Homebrew Clang 23). Ratios are medians of paired ratios to arm A (the old
default); "rows L1–L2" is the geomean over the rows of more than L1 and at most L2 bytes,
the batched and the real cases (the cases `GRID_MIN_BYTES` can change); "rows > L2/8" over
the c2c b1 rows above the old three-phase switch.

Screens (one field moved at a time):

| arm | Xeon avx512 L1–L2 | > L2/8 | Xeon avx2 L1–L2 | > L2/8 | EPYC L1–L2 | > L2/8 |
|---|---|---|---|---|---|---|
| min 2·L1 | 0.929 | 1.001 | 0.936 | 0.996 | 0.936 | 1.002 |
| min L2/4 | 0.865 | 0.983 | 0.855 | 0.975 | 0.935 | 0.999 |
| min L2/2 | **0.863** | 0.979 | 0.843 | 0.959 | 0.907 | 0.990 |
| min L2 | 0.874 | 0.998 | **0.831** | 0.952 | **0.895** | 0.980 |
| three L2/2 | 1.013 | 1.000 | 0.991 | 0.994 | 0.969 | 0.987 |
| three L2 | 0.999 | **0.984** | 0.987 | 0.996 | 0.961 | **0.982** |
| three 2·L2 | 1.004 | 0.999 | 0.988 | 1.014 | 0.961 | 0.999 |
| band | ±1.81 % | | ±1.59 % | | ±1.00 % | |

Runs 20261114-184254-3ad8 (Xeon avx512, 42 rounds, 32 loaded),
20261114-184157-25c5 (Xeon avx2, 42 rounds), 20261114-184230-4a9c (EPYC, 58 rounds). The
chain's gain is largest at 4096 points in double (0.74–0.78 of the grid's time on every
x86 arm, as X3 found) and 8192 on the Xeon (0.72–0.84). Two phases up to L2 won at exactly
L2 in double (65536 points on the Xeon: 0.875–0.88) but lost in single (131072 points,
avx2: 1.18); two phases at 256 KiB rows, which moving only the three-phase switch makes,
lost on the Xeon (16384 × 64 rows: 1.16–1.29), so that switch pays only where the grid
starts late as well.

The rule (below) against the old default and against `GRID_MIN_BYTES` = L2, then FFTW:
runs 20261114-192954-0959 (Xeon avx512, 43 rounds), -192954-bf31 (Xeon avx2, 42),
20261114-192856-cdd2 (EPYC, 62). Weighted geomean over all 31 cases, rule ÷ old: 0.921,
0.909, 0.957; min = L2 with three = L2: 0.927, 0.906, 0.956 (bands ±2.44, ±1.62,
±1.00 %). Slower with the rule: r2c 32768 on avx512 (1.08; faster on avx2 0.88 and the
EPYC 0.91), c2c 131072 single on avx2 (1.19), c2c 16384 single on the EPYC (1.03).

FFTN ÷ FFTW (double, one row; FFTW 3.3.11 MEASURE, the avx512 build on the Xeon for both
arms), before → after:

| points | Xeon avx512 | Xeon avx2 | EPYC |
|---|---|---|---|
| 4096 | 1.42 → 1.05 | 2.31 → 1.76 | 1.49 → 1.16 |
| 8192 | 1.47 → 1.05 | 2.29 → 1.89 | 1.72 → 1.60 |
| 16384 | 1.49 → 1.38 | 2.50 → 2.07 | 1.56 → 1.40 |
| 65536 | 1.33 → 1.18 | 1.77 → 1.54 | 1.42 → 1.43 |
| 262144 | 1.33 → 1.33 | 1.68 → 1.69 | 1.34 → 1.34 |
| 1048576 | 0.82 → 0.81 | 0.87 → 0.86 | 1.08 → 1.08 |
| 4194304 | 0.87 → 0.87 | 0.97 → 0.96 | 1.12 → 1.11 |

Rows of 2^18 points and more are unchanged (their plans are the same three-phase grids).

Mac check (run 20261114-190906-3fb4, c2c b1 rows, 16 rounds all loaded, band ±2.23 %):
grid from 2·L1 1.67 at 16384 (double) and 1.63 at 32768 (single), the rows it turns into
chains; the x86 rule applied to the Mac (min 8 MiB, three 16 MiB) 1.2–2.1 from 16384 to
524288 points; three phases from L2/16 no different (1.010); from L2/2 0.89–0.90 at 4 MiB
rows but 1.04 at 524288 double (not adopted by rule 1; X12's development timings had three
phases ahead at 2^18).

## Verdict

Adopted (commit 345755a, `set_grid_sizes` in `src/threading/schedule.cpp`, tested on the
recorded topologies). No rule in the cache sizes alone fits: the Mac has the larger L1 and
L2 yet needs the grid earlier (beyond L1), the x86 machines later (beyond L2/2). What
separates them is the L2's sharing: x86's is each core's own (shared by as many CPUs as the
L1), the M5's is shared by a cluster of six. With a private L2 the grid starts beyond L2/2
and three phases beyond L2; with a shared L2 the defaults stay L1 and L2/8. By rule 2 the
rule's combined geomean is within the band of the best form tried on every x86 run (min L2
was 0.4 % better on avx2 and the EPYC, 0.7 % worse on avx512). The fusion limit
(`grid_fuse_bytes`, L2/4) and `grid_block_bytes` (L1) are unchanged; with the grid starting
at L2/2 on x86, fused phases no longer occur there. Open: the r2c pairing's fusion limit
follows `grid_min_bytes` (r2c 32768 on avx512 lost 8 %); three phases from L2/4 on the Mac;
the remaining gap to FFTW at 8192–65536 points on avx2 and the EPYC is the chain's own
(X3's radix-32 chains). Lesson 3.8 in `dev/LESSONS.md`.
