# X38: the EPYC's rows of 80–20000 points in double (the exit check's cause 4)

- **Status:** running
- **Owner:** x38 session, 2026-11-16
- **Phase / task:** 1 (exit criteria: DESIGN 1.3's tail rule); PLAN lead 3, the phase 1
  exit check's cause (4); follows X2, X25, X30, X30b

## Question

What takes the EPYC's (Zen 3, `avx2`, GCC 16) double rows of 240–2048 points, and the
mixed rows near them, under 1.05 of FFTW's time? The answer changes the `avx2` double radix
table in `src/planner/rules.cpp` and possibly the `avx2` pass loops.

## Hypothesis

The phase 1 exit check (EPYC run 20261116-161226-416e, standard catalogue, c2c rows, FFTN ÷
FFTW) has these cases past 1.05: 240 1.217, 384 1.233, 640 1.215, 1024 1.186 (× 3 1.202,
× 16 1.101, × 256 1.082, backward 1.195), 768 1.131, 2048 1.140, 512 1.112, 80 1.087, 100
1.087, 160 1.057, 256 1.056, 2000 1.095, 3000 1.082, 3600 1.099, 20000 1.135. FFTN's plans
there (the run's notes): 240 = 2 × 10 × 12, 384 = 2 × 16 × 12, 640 = 4 × 10 × 16, 768 =
4 × 16 × 12, 1024 = 4 × 16 × 16, 2048 = 8 × 16 × 16 (split from pass 2), 3600 = 2 × 10 ×
15 × 12, 20000 = 2 × 10⁴. Three things point at the plans rather than the loops:

1. **The table cannot choose radix 32 or 64.** `avx2_double` has no column for them (X3
   rejected them on the EPYC before X19, X1, X28 and X30b changed the passes), yet X30's
   two-pass 256 = 8 × 32 and 512 = 16 × 32 took 0.905 and 0.946 of the three-pass time,
   and executed about FFTW's instructions. FFTW plans 1024 as 32 × 32 on the EPYC (X25).
   Two-pass plans exist for every target length up to 2048 with a 32 or 64 pass (384 =
   12 × 32, 640 = 10 × 64, 768 = 12 × 64, 1024 = 32 × 32, 2048 = 32 × 64); X30 measured only
   1024 = 16 × 64 (lost in batches, 1.09).
2. **A first radix of 2 is priced nearly free.** The fitted first-pass unit cost of radix 2
   is 28 against 114 for radix 4, so 240, 384, 3600 and 20000 start with a radix-2 pass.
   The table was fitted on the EPYC (X2, 2026-11-14/15), but before X28 (new pass
   interface, 128-bit leaves), X23b (masked partial runs), X30 and X30b (aligned tables);
   costs may have moved.
3. **No split run at 1024.** A chain's split run (X1) starts after the first pass whose
   copy count is a multiple of 8 on `avx2`; 4 × 16 × 16 has none, 8 × 8 × 16 or 8 × 16 × 8
   would (lead 3's "first radix that lets a run start"). The costs do not price the split
   arithmetic.

What the plans cannot explain: X30 found 1024–4096 at IPC 2–2.5 with 1.4–3.8 times FFTW's
L1 refills; if the measured best sequences still leave 1024 well above FFTW, the remainder
is in the passes (twiddle footprint, X30b's suspicion of the FMADDSUB product).

## Change under test

Arms are commits of branch `exp/x38-epyc-mid`, against A = `origin/main` (e8c2639):

- **S (re-fitted sequences).** `avx2_double_overrides` gains (or changes) an entry for
  each length of the target set whose measured best sequence on the EPYC (selection below)
  is at most 0.97 of the planner's at batch 1 and batch 64. Only the table changes; plans,
  and so bits, change at those lengths (within the tests' tolerances).
- **Second and third arms**, chosen from the diagnostic below (FFTW's plans, counters per
  case for FFTN and FFTW, and what S leaves), written here before any is timed. Candidates:
  a `avx2` double cost column for radix 32 and 64 (so lengths outside the target set can
  take them, priced from the selection's timings); split-aware sequences for 1024 and
  2048; the pre-arranged twiddle product built from compact storage for large tables
  (X30b's follow-up); radix 3/5 placement in 240/384/640.

No switch: arms are commits, `dev/KNOBS.toml` is untouched.

*Amendment 1 (after the diagnostic and S's selection, before any arm was timed).* S is
384 = 12 × 32, 3600 = 15 × 16 × 15 and 20000 = 8 × 5 × 10 × 5 × 10 (f745178;
`radix_override` holds five radices). The selection left the planner's sequence best at 80,
100, 160, 256, 512, 1024 and 2048 among every sequence FFTN can run, and 240 = 15 × 16 just
missed the rule at batch 64 (0.973). FFTW's plans (diagnostic) take a radix-20 pass at 160,
240, 2000, 3000, 3600 and 20000 (240 = 12 × 20, 2000 = 10 × 20 × 10, 20000 = 20 × 5 × 20 ×
10), which FFTN cannot: it has no codelet of 20, and without one 2000 and 3000 need four
passes and 20000 five. The second arm is therefore:

- **R (a radix-20 codelet, on S).** The generator gains radix 20 (Good–Thomas 4 × 5, 208
  additions and 48 multiplications, the derived reference; 22 live values at peak), a
  pass radix only: never a whole-transform leaf (rule 1 of DESIGN 11.2 excludes it, so no
  plan of 20 points changes on any target). Plans use it only through `avx2` double
  overrides chosen like S's (the search driver with candidates that contain a 20, lengths
  80, 100, 160, 240, 640, 2000, 3000, 3600, 20000; the same rule against the planner's
  sequence, and against S's where S has one). Library text grows (+4.8 % in the Mac's
  static library, all targets); bits change only at the overridden lengths.

The third arm is not planned: 1024, 2048 and 512 keep the planner's sequences and their
gap is in the passes (FFTN executes 1.27 times FFTW's instructions at 1024 with three
passes against two; its radix-32 and 64 codelets peak at 35 and 67 live values in 16
registers), a generator matter beyond this experiment.

## Method

- **Diagnostic** (main only, before any arm): an fbench run on the EPYC with FFTW's plans
  printed (`PLAN_NOTE=1`) for the target set (standard catalogue, filter on the target
  lengths), and `run --pmu` (cycles, instructions, L1 fills) on FFTN and FFTW (shared
  wisdom) for the single rows of the target lengths.
- **Selection for S**: experiment X2's driver (`bench/x2/search.py all --isa avx2 --radices
  2-16,32,64`) on the EPYC for the target lengths 80, 100, 160, 240, 384, 640, 768, 1024,
  2000, 2048, 3000, 3600, 20000 (256 and 512 are X30's and are included as a check): every
  candidate sequence up to 1024 points, the cost estimate's best 20 above; screening,
  refinement of the best four per case, and a confirmation of the winner against the
  planner (both precisions, batches 1 and 64). Its campaign state names each stage's run.
  A length gets S's override when its double winner is the same sequence at both batches
  and at most 0.97 of the planner's at both in the confirmation, with the intervals clear
  of the band (else the batch-1 winner if it passes at both batches).
- **Judging runs** (one thread, fbench interleaved process rounds, A/A control,
  `--serial`):
  - EPYC 7763 (DMOG, GCC 16.2, `avx2`), the deciding machine: arms A, the arm(s), and
    FFTW, on the exit check's cases (standard catalogue, filter `^c2c-[0-9]+-b`, tier
    `full`), which give both arm ÷ A and FFTN ÷ FFTW before and after; then the tracking
    tier (all kinds: r2c, c2r and planes run the changed chains as half-length rows and
    axes) for collateral.
  - Xeon Gold 6240 (GCC 14.4) with `ISA=avx2`: the tracking tier, A and the arm(s),
    recorded; the `avx2` table is shared, the decision is the EPYC's (rules by core are
    deferred, PLAN Status lead (d)).
  - Clang 23 on the EPYC or the Xeon: recorded if time allows, not deciding.
- **Metric**: fbench's median paired ratio arm ÷ A per case and geometric means over the
  target cases whose plan changed; the run's A/A band.
- **Decision rule**, written before any arm is timed. An arm is adopted for `avx2` double
  if, on the EPYC with GCC:
  - (a) the geometric mean of arm ÷ A over the target cases whose plan it changes is at
    most 0.97, with the interval clear of the band;
  - (b) no case of the exit check's set or of the tracking tier is slower than
    1 + band + 2 %; a case past this on a loaded machine is re-run once and the re-run
    decides;
  - (c) a length whose cases are slower than 1 + band at any batch or direction in the
    judging runs loses its override before adoption (A's plan returns there, the same
    code as A, so no new run is needed for it).
  On the Xeon's `avx2` the tracking geometric mean must be at most 1 + band; single cases
  past 1 + band + 2 % there are recorded as exceptions (X30's precedent) with the core
  follow-up, not reasons to reject. An arm that fails is reported with its numbers and
  deleted; nothing is iterated past three arms.

## Results

### Diagnostic: plans and counters (main e8c2639, before any arm)

EPYC 7763 (DMOG node17), GCC 16.2, `avx2`, one thread. FFTW 3.3.11 (AVX2) at MEASURE,
one plan shared through wisdom by every counting process (Slurm 511064). FFTN ÷ FFTW
time from fbench run 20261116-180425-c571 (8 rounds, all loaded, band ±1.76 %); counters
per transform, user mode, medians of three single processes (`run --pmu`; L1 fills
PMCx044, FP-pipe uops PMCx000 pipes 0–3). FFTW's plans as `fftw_sprint_plan` prints them,
written here in FFTN's convention, executed sub-transform first (FFTW's DIT: the `n2fv`/
`n1fv` codelet runs first, the outer `t1fv`/`t2fv` radix last):

| n | FFTN ÷ FFTW | FFTN plan | FFTW plan (first → last) | instructions N / F | IPC N / F | L1 fills N / F | FP uops N / F |
|---|---|---|---|---|---|---|---|
| 80 | 1.10 | 8 × 10 | 10 → 8 | 1271 / 1026 | 4.34 / 3.88 | 0 / 0 | 509 / 521 |
| 100 | 1.10 | 10 × 10 | 10 → 10 | 1605 / 1340 | 4.19 / 3.76 | 0 / 0 | 709 / 725 |
| 160 | 1.04 | 10 × 16 | 8 → 20 | 2247 / 2108 | 3.92 / 3.74 | 0 / 0 | 1147 / 1148 |
| 240 | 1.20 | 2 × 10 × 12 | 12 → 20 | 4834 / 3194 | 4.71 / 3.65 | 0 / 0 | 1941 / 1842 |
| 256 | 1.06 | 8 × 32 | 64 → 4 | 3609 / 3371 | 3.91 / 3.46 | 0 / 0 | 1868 / 1875 |
| 384 | 1.24 | 2 × 16 × 12 | 12 → 32 | 7205 / 5167 | 4.13 / 3.68 | 180 / 23 | 3175 / 3103 |
| 512 | 1.11 | 16 × 32 | 64 → 8 | 7808 / 7040 | 3.61 / 3.63 | 517 / 200 | 4185 / 4369 |
| 640 | 1.21 | 4 × 10 × 16 | 64 → 10 | 12817 / 9357 | 3.89 / 3.21 | 825 / 543 | 6028 / 5779 |
| 768 | 1.19 | 4 × 16 × 12 | 12 → 64 | 14465 / 11920 | 3.58 / 3.59 | 1096 / 473 | 7104 / 7239 |
| 1024 | 1.20 | 4 × 16 × 16 | 64 → 16 | 19444 / 15276 | 3.44 / 3.30 | 1767 / 1263 | 9830 / 9561 |
| 2000 | 1.09 | 8 × 10 × 5 × 5 (split) | 10 → 20 → 10 | 47526 / 44268 | 3.24 / 3.34 | 5067 / 3170 | 28959 / 25046 |
| 2048 | 1.10 | 8 × 16 × 16 (split) | 64 → 32 | 40413 / 34650 | 2.98 / 2.83 | 4737 / 3066 | 25522 / 21568 |
| 3000 | 1.07 | 8 × 5 × 5 × 15 (split) | 20 → 10 → 15 | 74849 / 72957 | 3.30 / 3.42 | 7876 / 4175 | 44494 / 41043 |
| 3600 | 1.11 | 2 × 10 × 15 × 12 | 12 → 20 → 15 | 105346 / 85637 | 3.78 / 3.45 | 9481 / 5679 | 46647 / 49199 |
| 20000 | 1.16 | 2 × 10⁴ (split) | 10 → 20 → 5 → 20 | 705050 / 605458 | 3.05 / 2.98 | 62453 / 54158 | 369057 / 346490 |

FFTW runs two passes from 160 to 2048 (radix 20, 32 or 64 in one of them) and three or
four with radix-20 passes above; FFTN runs three at 240–1024. The FP work is the same
(FP-pipe uops within ±5 %, 2000–2048 +16–18 %); FFTN executes 1.1–1.5 times FFTW's
instructions at IPC at or above FFTW's (3.0–4.7: these counters agree with X25's, not X30's
2–2.5), and refills L1 1.4–2.3 times as often from 384 up. In the hot loop of the radix-16
twiddled lane pass (GCC 16, `objdump`), a unit of C = 2 copies is 258 instructions: 132
FP, 32 loads and stores of data, 30 twiddle broadcasts, 18 spills and 46 integer (address
arithmetic, the loop): per point and pass FFTN's passes are cheaper than FFTW's (1024: 6.3
against 7.5 instructions), so the excess is the third pass.

### Selection for S (campaign x38-epyc-avx2)

X2's driver on the EPYC (GCC 16.2, `ISA=avx2`, Slurm 511061, commit f99d31b = main's
library), every candidate with radices 2–16, 32 and 64 up to 1024 points (3573 sequences
in all), the cost estimate's best 20 above; screening 20261116-180527-1341 onwards,
confirmations against the planner 20261116-182906-6cba (240) to 20261116-183523-cbce
(20000). Double, planner ÷ winner (above 1: the winner is faster), with the winner:

| n | batch 1 | batch 64 |
|---|---|---|
| 80, 100, 160, 256, 512, 1024, 2048 | planner best | planner best |
| 240 | 15 × 16 1.044 | 15 × 16 1.028 |
| 384 | 12 × 32 1.161 | 12 × 32 1.098 |
| 640 | 10 × 8 × 8 0.998 | planner best |
| 768 | 12 × 8 × 8 0.993 | 6 × 8 × 16 1.006 |
| 2000 | 10 × 5 × 4 × 10 1.020 | 8 × 5 × 5 × 10 1.009 |
| 3000 | 10 × 5 × 4 × 15 1.010 | 6 × 5 × 10 × 10 1.004 |
| 3600 | 15 × 16 × 15 1.078 | 15 × 16 × 15 1.049 |
| 20000 | 8 × 5 × 10 × 5 × 10 1.076 (its batch-64 ratio 1.078) | 8 × 10 × 5 × 5 × 10 1.073 |

The two-pass plans of FFTW's shape lost to the planner's three passes in the screening
(2 rounds; time ÷ the planner's, batch 1 / 64): 1024 16 × 64 1.014 / 1.033, 32 × 32
1.050 / 1.099, 64 × 16 1.245 / 1.210; 2048 32 × 64 1.138 / 1.290, 64 × 32 1.194 / 1.262;
640 10 × 64 1.032 / 1.007, 64 × 10 1.312 / 1.189; 768 12 × 64 0.991 / 1.067, 64 × 12
1.208 / 1.198. A radix-64 (or 32) last pass about breaks even with the two passes it
replaces; as the first (grouped) pass it costs 1.2–1.3. FFTN's radix-32 and radix-64
codelets peak at 35 and 67 live values in 16 registers. So by S's rule: 384, 3600, 20000.

### Selection for R (campaign x38-r20)

The same driver on the radix-20 build (88f8c00 = S plus the codelet, no new overrides;
Slurm 511091, not queued behind the serial jobs: a screen), candidates containing a 20,
against the planner's sequence (S's at 3600 and 20000); confirmations 20261116-185510-eb00
and before. Planner ÷ winner, double:

| n | winner | batch 1 | batch 64 |
|---|---|---|---|
| 80 | 4 × 20 | 1.058 | 1.075 |
| 100 | (planner best: 10 × 10) | – | – |
| 160 | 8 × 20 | 1.061 | 1.035 |
| 240 | 12 × 20 | 1.198 | 1.197 |
| 640 | 20 × 32 | 1.130 | 1.114 |
| 2000 | 10 × 10 × 20 | 1.224 | 1.157 |
| 3000 | 10 × 15 × 20 | 1.195 | 1.121 |
| 3600 | 12 × 15 × 20 | 1.112 | 1.097 (against S's 15 × 16 × 15) |
| 20000 | 20 × 5 × 10 × 20 / 10 × 5 × 20 × 20 | 1.081 | 0.866 / 1.017: neither passes at both batches |

R (cd22279) takes the seven passing lengths; 20000 keeps S's sequence. These are FFTW's
radix-20 plans in all but 640 (FFTW 64 → 10) and 3600 (FFTW 12 → 20 → 15, the same
radices).

(Timing runs to come.)

## Verdict

(to come)
