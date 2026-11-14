# X12b: The grid's thresholds per machine

- **Status:** running
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

(pending)

## Verdict

(pending)
