# X30: the x86 chains at 128–4096 points: passes, twiddle footprint, loop overhead

- **Status:** done
- **Owner:** x30 session, 2026-11-15
- **Phase / task:** 1 (exit criteria); PLAN lead 3; follows X25, X23 amendment 4, X3, X10

## Question

What closes the EPYC's gap to FFTW in double at 128–4096 points (tracking tier 1.05–1.35 of
FFTW's time), and the Xeon's `avx2` there, without losing on the Xeon's `avx512`? The answer
changes the `avx2` double radix table and the x86 vocabularies' twiddle storage.

## Hypothesis

X25 found the EPYC executing 1.38–1.60 times FFTW's instructions at 256–1024 for the same
FP-pipe work, and read the excess as address arithmetic in the AVX2 pass loops. A first
look for this experiment (main 0e64ecd, GCC 16.2, EPYC 7763 node17, before any arm; counts
from `fbench-* run --pmu`, single processes, medians of three) splits that excess
differently:

1. **Passes, not loops, at 256 and 512.** The `avx2` double table has no radix 32 or 64
   (X3 rejected them on the EPYC before X19 and X1), so 256 runs 2 × 16 × 8 and 512 runs
   4 × 8 × 16: three passes where FFTW runs two (16 × 16 or 8 × 32 at 256, 32 × 16 at
   512). Forced two-pass chains execute about FFTW's instructions (256: 16.16 3814 and 8.32
   3742 against FFTW's 3579 and main's 5047; 512: 16.32 7945 against 7659 and 9802) and take
   0.85 (256, batch 1 and 64) and 0.78–0.97 (512) of main's time. 1024, 2048 and 4096 have
   no such clear winner in the screen (single processes, ±5–10 %).
2. **Twiddle footprint from 512 up.** FFTN's per-lane twiddle storage holds (wr, wr) and
   (−wi, wi) per twiddle, 32 bytes in double where FFTW stores 16. A grouped radix-16 first
   pass over 512 points reads 15 KiB of twiddles beside 24 KiB of lines, more than the
   32 KiB L1: the 16.32 chain refills 589 L1 lines per transform against FFTW's 3, and at
   1024 main refills 1970 against 521. Storing twiddles interleaved (wr, wi), 16 bytes, and
   reading them with the duplicating loads (`vmovddup` in double, `vmovsldup`/`vmovshdup`
   in float, which use the load ports only) and `vfmaddsub` (no sign fix-up) gives the
   same products with the same roundings, so bit-identical results, and halves the
   per-lane and broadcast tables of `avx2` and `avx512`.
3. **Loop overhead is the smaller part.** In GCC 16's code the radix-16 lane loops spend
   49–53 of 191 (plain) and 263 (twiddled) instructions per unit on integer work and 29–31
   on stack references; the grouped radix-4 loop is 36 instructions with none. Removing all
   of it would cut at most 20 % of a radix-16 pass. X10 removed part of it (fixed pointer
   steps, leg tables) for 1–3 %.

Also from the first look: the IPC there is 2.0–2.5 (256: main 2.51, FFTW 2.33; cycles
from the same counter group as the instructions), not X25's 4.65 (X25's EPYC cycle counts
are about 1.75 times lower than these for the same cases and times; not resolved here).
So the EPYC is not at its dispatch limit at 256–1024, and fewer instructions alone need
not mean less time (lesson 1.3).

## Change under test

Arms are commits of branch `exp/x30-x86-pass-loops`, each against A = main (0e64ecd, which
has X23b's masked partial runs under `FFTN_MASKED_TAILS=AUTO` and X26):

- **P (planner table).** `avx2_double_overrides` in `src/planner/rules.cpp` gains the
  two-pass sequences chosen by a selection run (below) for the powers of two 256–4096
  where one is at most 0.97 of main's sequence at batch 1 and 64. Only the table changes.
  This changes plans and so bits (within the tests' tolerances); it affects `avx2` double
  on every machine (the EPYC and the Xeon's `avx2`).
- **T (twiddle storage).** `vocab_avx2.inl` and `vocab_avx512.inl`: `tw_reals` 2, storage
  (wr, wi), `load_tw` by duplicating loads, `bcast_tw` by two scalar broadcasts, `mul`,
  `mulc`, `madd` and `cmul` through `vfmaddsub`/`vfmsubadd`. Double's `vmovddup` of the
  imaginary parts reads one real past a unit, so twiddle tables get one real of padding
  (`chain_builder::pack`). Bit-identical to A in both `FFTN_MASKED_TAILS` states.
- **Third arm**, chosen from what P and T's counters show (candidates: the last pass of a
  two-pass chain in place in the destination, two lines instead of three; per-unit loop
  overhead in the lane passes), written here before it is measured.
  *Chosen (before any arm was timed):* **Q = PT plus 1024 = 16 × 64** (d7036bb). A third
  counter screen (main and T builds, forced sequences, three single processes each, EPYC
  node17, GCC 16) showed that T's halved tables make the two-pass 1024 consistently
  faster: 16 × 64 with T at 0.81 of main's cycles at batch 1 and 0.92 at batch 64 (0.90
  and 0.88 of the time), against 0.84 and 0.94 without T, where the first two screens had
  disagreed. The in-place last pass was dropped: the generated plain codelets interleave
  their loads and stores, so a lane pass cannot run in place. Loop overhead stays a
  follow-up (lane radix-32 and radix-64 loops are dominated by stack traffic, 142 and 407
  stack references per unit, a generator matter).

No switch: arms are commits, `dev/KNOBS.toml` is untouched.

*Amendment (before any arm was timed):* X28 merged into main while the runs were queued
(d8ca0f1, new pass interface and 128-bit x86 leaves), so the queued runs were cancelled
unrun (EPYC 509855, 509857; Xeon 15853, 15855) and the arms rebuilt on main be593d1:
A = be593d1, T = 9e6e1a9 (T and `tw_tail` in X28's narrow vocabulary), P = b7845ca, Q =
a7253b5 (the branch merged with main: P, T and 1024 = 16 × 64). PT is dropped (Q holds
P and T).

## Method

- **Selection for P** (before P's commit): fbench on the EPYC, `RADICES` arms per length
  against the planner's sequence, cases c2c double batch 1 and 64: 256 (16.16, 8.32), 512
  (16.32, 8.64), 1024 (16.64, 32.32, 16.8.8), 2048 (32.64), 4096 (4.16.64, 8.8.64). A
  length gets an override when a sequence is at most 0.97 of the planner's at both batches
  with intervals clear of the band; the better of two such sequences is taken by batch 1.
  *Amendment (before P was timed):* the selection runs were queued behind four other
  experiments' serial timing jobs on the EPYC for hours, so they were cancelled unrun (Slurm
  509812–509820). P's sequences come from the two counter screens instead, where they were
  the fastest at both batches in both screens: 256 = 8 × 32, 512 = 16 × 32 (1024, 2048 and
  4096 had no sequence ahead of the planner's in both screens and keep theirs). P and PT
  (P on T) are timed as arms beside T.
- **Cases**: `bench/catalogues/x30.toml` (c2c rows of 128–4096 in double and float, batches
  of 64, r2c and c2r of 512–4096, planes 256² and 512²); collateral: the standard
  catalogue's tracking tier.
- **Machines** (one thread, fbench interleaved process rounds, A/A control, `--serial`):
  EPYC 7763 (DMOG, GCC 16.2, `avx2`): the main machine. Xeon Gold 6240 (GCC 14.4): `avx512`
  and `ISA=avx2` in one job per case set. Clang 23 on the Xeon recorded if time allows, not
  deciding (AGENTS.md section 6).
- **Metric**: fbench's median paired ratio arm ÷ A per case, geometric means per
  precision, against the run's A/A band; instructions, cycles and L1 refills per call from
  `run --pmu` for the arms' target cases.
- **Decision rule**, written before any arm is timed. Per arm and per machine-target where
  the arm changes code (P: EPYC `avx2`, Xeon `avx2`; T: those and Xeon `avx512`), judged on
  GCC:
  - (a) the double geometric mean over x30's double cases is at most 0.98 with its interval
    clear of the band on the EPYC (P, T) or on the Xeon's `avx512` (T), and at most
    1 + band on every other machine-target;
  - (b) no case of x30 (either precision) or of the tracking tier is slower than
    1 + band + 2 %; a case past this on a loaded machine is re-run once and the re-run
    decides;
  - (c) x30's float geometric mean is at most 1 + band.
  T may be adopted per target (`avx2`, `avx512`: separate vocabularies). An arm that fails
  is reported with its numbers and deleted; nothing further is iterated past three arms.

## Results

### Checks before timing

- **Bits.** T against main (0e64ecd), GCC 16, `avx2` and `scalar`: the check's RMS and
  maximum errors are equal to the last digit on 18 cases (c2c 17–100000 points, both
  directions, in place, batches, a plane, r2c and c2r up to 65536), as T's roundings say
  they must be. P and Q change plans, so bits, within the tests' tolerances.
- **Tests.** DMOG node, GCC 16, Release, `-Werror`, both `FFTN_MASKED_TAILS` states: T,
  P and Q pass (1366 tests each; P after `Rules.RadicesMultiplyToTheLength` learned that
  a radix may be 32 or 64). Xeon, GCC 14 (`avx512` and `avx2`, masked on): T passes all
  1688, including `Passes.TargetsThatRoundAlikeGiveTheSameBits`. CI on pull request 13.

### Counter diagnostics of T (not judged; single processes, medians of three, EPYC node)

Cycles per transform, double, batch 1, the planner's sequences of main (so before P):

| n | main | T | T, tables 64-byte aligned | T aligned, i by an in-lane shuffle | main, tables aligned |
|---|---|---|---|---|---|
| 512 | 2618 | 2622 | 2567 | 2475 | 2531 |
| 1024 | 5995 | 5739 | 5623 | 5540 | 5789 |
| 2048 | 14383 | 16401 | 16390 | 15554 | 14024 |
| 4096 | 33879 | 37789 | 38813 | 36373 | 33158 |

T loses at 2048 (8 × 16 × 16) and 4096 (8 × 8 × 8 × 8), whose grouped radix-8 first pass
has 256 and 512 butterflies. Two causes: the imaginary parts' load at p + 1 crosses a
cache line in half the units (the EPYC's misaligned-load counter, PMCx047, goes from 20
to 1250 per transform at 2048), and the duplicating loads cannot be folded into the
multiplies as the pre-arranged form's loads were (the grouped radix-8 loop goes from 87 to
100 instructions per unit, radix 16 from 228 to 258). Aligning the tables does not remove
the crossings (the i load starts 8 bytes in); building i by `vpermilpd` from the aligned
load does, and halves the loss. Main's tables start where `std::vector` puts them:
aligning them to 64 bytes alone took 0.97–0.98 of main's cycles at 1024–4096 (main
crosses lines on 863–3046 loads per transform there). These diagnostic builds are not
arms (branches `diag/x30-*`, not pushed).

### EPYC, tracking tier (run 20261116-020626-31eb)

EPYC 7763 (DMOG), GCC 16.2, `avx2`, one thread, 80 rounds, every round under load (other
processes 64 cores of 128), A/A band ±1.00 %. Arm ÷ A (be593d1); "over" = slower than
1.03, the rule's limit (b):

| Arm | Geomean | Faster / slower | Over (worst) | Notable |
|---|---|---|---|---|
| T | 0.997 [0.996, 0.999] | 16 / 17 | 9: 97 1.081, 100 × 1024 1.084, c2r 256 1.059, 1009 1.057, 2048 × 36 1.055, r2c 256 1.040, 128 1.037, 64 1.036, 120 1.033 | 1024 0.957, 4096 0.961, 16384 0.912, 10403 0.878, 65537 0.956, c2r 2²⁰ 0.951 |
| P | 0.997 [0.996, 0.998] | 4 / 1 | 0 (r2c 1024² 1.014) | 256 0.913, c2c 256² 0.965, r2c 256² 0.967, c2r 1024² 0.954 |
| Q | 0.998 [0.997, 1.000] | 18 / 20 | 12: T's and 1024 × 256 1.094, 1024 1.032 | 256 0.886 |

T and Q fail (b) on the EPYC, so neither can be adopted for `avx2`; both were reverted on
the branch (eb329f5, 2acd6b5). T's pattern is the counters' trade: it wins where tables
are large against L1 (1024, 4096, 16384, Bluestein's chirps at 10403 and 65537, long
real rows) and loses on short rows and small grouped passes, where the unfolded
duplicating loads cost more than the bytes save. 16 × 64 at 1024 loses for batches
(1024 × 256: 1.094), whatever the single-row screens said.

### EPYC, x30 catalogue (run 20261116-050953-a80c)

Same machine and build as above, 80 rounds, every round loaded (other processes 64–112
cores; 14 rounds started overloaded after the wait), A/A band ±1.00 %. Geometric means of
the median paired ratios per case (26 double, 16 float cases); fbench's weighted geomean
over both precisions in brackets:

| Arm | Double | Float | Both | Over 1.03 |
|---|---|---|---|---|
| T | 0.991 | 0.941 | 0.972 [0.970, 0.974] | 4: 3072 1.052, 128 1.040, 192 1.035, c2r 512 1.034 |
| P | **0.975** | 1.000 | 0.985 [0.984, 0.986] | 0 (worst c2r 2048 1.003) |
| Q | 0.974 | 0.942 | 0.962 [0.959, 0.964] | 5: 1024 × 64 1.094, 3072 1.052, 128 1.042, 1024 1.042, 192 1.035 |

P per case (double): 256 0.905, 256 × 64 0.874, 512 0.946, 512 × 64 0.924, r2c 512
0.903, r2c 1024 0.954, c2r 1024 0.942, c2r 512 0.989, planes 256² 0.956 and 512² 0.985;
every other case within 0.99–1.01 (they keep their plans). In float P changes nothing
(geomean 1.000, as it must: only `avx2_double` changed).

### Xeon (GCC 14.4), tracking tier (run 20261116-041211-2a5e)

Xeon Gold 6240, Slurm `-c 8` on the shared node, arms A, T with `ISA=avx512` and A, T, P,
Q with `ISA=avx2`; 80 rounds, 2 of them loaded, band ±1.00 % (A512's control), so the limit
is 1.03. Arm ÷ its own target's A, geometric means over the 61 cases:

| Arm | Geomean | Over the limit |
|---|---|---|
| T `avx512` | 0.973 | 17 1.103, 97 1.100, 16 1.066, 4096 1.047, 257 1.040 |
| T `avx2` | 0.977 | 17 1.058, 16 1.057 |
| P `avx2` | 0.999 | r2c 1024² 1.034 [0.997, 1.061] (256 0.932) |
| Q `avx2` | 0.977 | 16 1.067, 17 1.054, r2c 1024² 1.045, c2c 1024² 1.041, 1009 1.039 |

P's one case past the limit is 0.4 points over by its median, with an interval that
contains 1 (fbench: no evidence) and an A/A ratio of 1.02 for that case alone; on the EPYC
the same case is 1.014. Its rows of 1024 real points run the half-length chain of 512, which
P makes 16 × 32. A re-run of it beside its neighbours (A256, P256; Slurm 15965,
`x30s-xeon-rerun`) was queued when this was written.

The Xeon's x30 run (Slurm 15903, `x30s-xeon-x30`) was still queued behind other
experiments' serial jobs when this was written; whoever fetches it adds it here. It cannot
change T's or Q's verdict (both already fail (b)), and for P on the Xeon's `avx2` the
tracking tier shows no case past the limit.

Not measured: Clang (recorded only by the rule, no runs made); the Xeon's x30 cases.

## Verdict

- **P adopted** for `avx2` double (the table's overrides 256 = 8 × 32, 512 = 16 × 32).
  On the EPYC: (a) x30's double geomean 0.975 (≤ 0.98; every target case's interval clear
  of the band); (b) no case of x30 or of the tracking tier past 1.03 (worst r2c 1024² 1.014
  in the tracking tier); (c) float 1.000. On the Xeon's `avx2` (which shares the table):
  tracking geomean 0.999, one case 0.4 points past the limit by its median, r2c 1024²
  1.034 with an interval [0.997, 1.061] that contains 1; its re-run (Slurm 15965) and the
  Xeon's x30 run (15903) were still queued. If the re-run confirms it, the choice is the
  lead's: a recorded exception (as X23 amendment 4 did), or 512's override left out. Plans
  change, so bits change at 256 and 512 (within the tests' tolerances).
- **T rejected** on every machine-target by (b), although its geomeans are better: EPYC
  tracking 0.997 with nine cases past 1.03 (up to 1.084), x30 0.991 double / 0.941 float
  with four; Xeon `avx512` tracking 0.973 with five (up to 1.103), `avx2` 0.977 with two.
  Reverted (2acd6b5). The pattern (wins where twiddle tables overflow L1, losses on short
  rows and small grouped passes; Results above) is lesson 3.10. A follow-up that keeps
  both sides: the (wr, wi) storage only for tables larger than about half of L1, or only
  for Bluestein's chirps and the long rows' twiddles, with the i load from an aligned
  address (`vpermilpd`), and twiddle tables aligned to 64 bytes (alone 0.97–0.98 of main's
  cycles at 1024–4096 in the counter diagnostics).
- **Q rejected**: 1024 = 16 × 64 is slower for batches (EPYC 1024 × 64 1.094, 1024 × 256
  1.094) and for the single row (1.032–1.042). Reverted (eb329f5).

Against the lead's target: P removes the third pass at 256 and 512 (EPYC 256 from 1.38 to
about 1.25 of FFTW's time, from the baseline ratio and P's 0.905; not re-measured against
FFTW here). The instruction excess X25 reported at 256–1024 was mostly that third pass:
two-pass chains execute about FFTW's instruction count (256: 3742–3814 against 3579). What
remains at 1024–4096 is not instructions: the EPYC runs at IPC 2–2.5 there (these counters;
X25's cycle counts were about 1.75 times lower and are unexplained), with L1 refills 1.4–3.8
times FFTW's from three lines plus twiddle tables twice FFTW's size. Follow-ups: the
footprint (T restricted as above, aligned tables), and the radix-32/64 lane loops' spills
(142 and 407 stack references per unit in GCC 16's code), a generator matter.

### Addendum (lead, 2026-11-16): the Xeon `avx2` re-run of r2c 1024²

Re-run quiet (run 20261116-072340-b100, band ±1.00 %, no round under load): P ÷ main
over the re-run's cases 0.998, r2c 1024² 1.057 [1.033, 1.071], a real loss on the Xeon
under `avx2` (its half-length rows are 512 = 16 × 32). The same case is 1.014 on the EPYC,
the `avx2` target's measured release machine, where the 512 override gains (512 0.946,
r2c 512 0.903, c2r 1024 0.942). Decision: keep the override and record the case as an
exception; like X26's prefetch, the effect belongs to the core (Intel cores without
AVX-512), and joins the follow-up for per-core rules. The Xeon x30 run of arm T
(20261116-064327-9cea: 0.981, seven cases slower up to 1.046) confirms T's rejection.

**Reach, by vendor (the maintainer, 2026-11-24; DESIGN 13.5):** P's 256 = 8 × 32 and 512 = 16 ×
32 apply to `avx2` plans for AMD's cores only; every other vendor plans 256 = 2 × 16 × 8 and 512
by the costs, as before X30.
