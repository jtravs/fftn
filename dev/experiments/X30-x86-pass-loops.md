# X30: the x86 chains at 128–4096 points: passes, twiddle footprint, loop overhead

- **Status:** running
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

(fbench runs pending)

## Verdict

(pending)
