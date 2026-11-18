# X54: radix-32 and radix-64 passes without most of their spills on `avx2`

- **Status:** running
- **Owner:** x54 session, 2026-11-18
- **Phase / task:** 1 (phase 1 exit check, the tail rule; cause 4); follows X38, X42, X47

## Question

Can FFTN's radix-32 and radix-64 codelets be ordered so that GCC spills far less of them on
`avx2`'s 16 registers, and does that let two-pass plans of 768, 1024 and 2048 points (FFTW's
shape) win at every batch on the EPYC 7763, so that the planner takes them? The answer
changes the generator's emission order (`gen/fftn_gen/schedule.py`, all targets' codelet
text) and the `avx2` double overrides in `src/planner/rules.cpp`.

## Hypothesis

The exit check on main 8c18dd2 (draft PR 67) fails on the EPYC by 12 cases beyond noise;
its mid rows are most of them: 1024 1.21 of FFTW (backward 1.20), 1024 × 3 1.22,
1024 × 16 1.13, 64 × 16 1.14, 2048 1.11, 768 1.09, 512 1.07. X47 placed the gap of 768–2048
in the third pass of the planner's three-pass chains; a two-pass plan would avoid it if its
radix-64 (or 32) pass kept FFTW's IPC, but GCC 16's radix-64 lane unit spends 327 of 1161
instructions on stack moves (X47), and after X47's two-line chains 1024 as 16 × 64 is 0.957
of the planner's time at batch 1 and 1.10 at batch 1024.

Why the codelets spill as much as they do: the generator orders a codelet for the fewest
live values (`schedule.py`), which is the right aim while a codelet fits the register file
and the wrong one past it: radix 32 and 64 peak at 35 and 67 live values, and what then
costs is how many values the allocator must store and reload. GCC on x86 allocates
registers in source order (it schedules only after allocation), so the order the generator
emits is close to the order GCC allocates. A model of the allocator (Belady's
furthest-next-use eviction in 16 registers, an evicted input re-loaded from its source, a
computed value stored once and reloaded at each later use) counts 292 stack moves for the
radix-64 plain codelet as ordered now and 93 for radix 32; GCC 16's lane units have 327 and
101 (149 + 178 and 49 + 52 stores and loads, plus 65 and 29 stack operands folded into
arithmetic), so the model tracks GCC. A list scheduler that scores each ready operation by
its operands outside a model register file of the values touched most recently (`fitted
order`) orders the same operations for 146 moves (radix 64) and 40 (radix 32) in the model;
compiled with GCC 16 (`avx2`, the lane passes' whole-unit loops, `objdump`), stack stores
and loads fall from 149 + 178 to 87 + 107 a unit for radix-64 plain (folded 65 → 38;
instructions 1161 → 1092), 155 + 171 to 85 + 100 for radix-64 DIF (1484 → 1407), 49 + 52 to
30 + 34 for radix-32 plain. X42's two-level codelets (staged through a fenced array) also
cut spills but added 4–12 % arithmetic and left GCC's instruction count unchanged; this
changes the order only, so every operation and rounding is the same and the results are
bit-identical.

Expected: radix-32 and radix-64 passes cheaper by a few per cent on `avx2` (Zen 3 stores
one 256-bit value a cycle: a radix-64 plain unit stored 64 outputs and 149 spills, now 87),
plans that already take them (256, 384, 512, 640 on `avx2`; 32-point columns of planes,
X44) a little faster, and the two-pass plans of 768–2048 close enough to win at both
batches of X38's selection rule. On the 32-register targets (`avx512`, NEON) the model
counts 191 → 76 moves for radix 64 and 4 → 8 for radix 32: small either way.

## Change under test

Arms are commits of branch `exp/x54-radix64-avx2` against O = its base (main 4d75c48):

- **A (fitted order).** `schedule()` takes `fitted_order()` for a complex codelet whose
  fewest-live order keeps more than 16 values live, where `spill_moves()` at 16 registers
  counts at least a quarter and at least 16 fewer moves (the best of model register files
  of 12–16 and two weights); that is radix 32 and 64, plain and DIF, both directions (DIT
  is not improved and no pass calls it). Every target compiles the new text; plans do not
  change; bits are identical.
- **B (two-pass `avx2` double overrides, on A).** X2's search driver on the EPYC with A's
  build, candidates with radix 32 or 64 (`--radices 2-16,20,25,32,64 --require 32,64`),
  lengths 512, 640, 768, 1024, 1536, 2048, 3072, 4096; a length gets an override by X38's
  selection rule: its double winner the same sequence at batch 1 and 64 and at most 0.97
  of the planner's at both in the confirmation, intervals clear of the band (else the
  batch-1 winner if it passes at both batches). `avx2` double only; no other target's plans
  change.

No switch: arms are commits, `dev/KNOBS.toml` is untouched.

## Method

- Static check (done for A, above): GCC 16 `objdump` of the `avx2` row units of radix 32 and
  64 (`rows3_avx2.cpp`, `rows4_avx2.cpp`) for O and A, stack moves per whole-unit loop.
- Screens (not deciding): forced sequences (`RADICES`) of 512, 768, 1024, 2048 at batches 1
  and 1024 on O's and A's builds, the planner's plan as baseline; `run --pmu` counters.
- Judging runs, `fbench_remote.py submit --serial`, one thread, interleaved process rounds,
  A/A control:
  - EPYC 7763 (DMOG, GCC 16.2, `avx2`), deciding for A and B: the exit check's cases
    (standard catalogue, filter `^c2c-[0-9]+-b`, tier `full`) with arms O, A (and B), FFTW;
    then the tracking tier, O against A (and B).
  - Xeon Gold 6240 (GCC 14.4, `avx512`): the tracking tier, O against A (`avx512` compiles
    the same codelet text). B changes no `avx512` plan and is not run there.
  - Apple M5 Max (Clang, `neon_fcma`, NEON FFTW): the tracking tier, O against A, when the
    Mac is free (one fbench run at a time).
- Metric: fbench's median paired ratio arm ÷ the arm before, per case, with its interval;
  geometric means over each arm's target cases; FFTN ÷ FFTW before and after on the named
  cases.
- **Decision rule, written before any judging run.** Judged on GCC on x86-64 Linux and
  Clang on macOS:
  - **A** (A ÷ O). (a) On the EPYC, over the exit check's cases whose FFTN plan in the run
    (the store's note) contains a radix-32 or radix-64 codelet, the geometric mean is at
    most 1 − band with its interval below 1. (b) No case of the EPYC's judged set or
    tracking tier is slower than 1 + band + 2 %. (c) On the Xeon and the Mac the tracking
    tier's geometric mean is at most 1 + band and no case is past 1 + band + 2 %. A case past
    a limit on a loaded machine is re-run once (`--serial` on x86) and the re-run decides. All
    targets compile one codelet text, so A passing (a) and (b) but failing (c) on one
    target means per-target texts (the order is free to differ: bits are the same); that
    is recorded and put to the lead, not built here. If A fails (a) but
    passes (b) and (c), it stays only as B's carrier and only if B is adopted (as X42's
    radix-25 codelet carried its overrides); otherwise it is deleted.
  - **B** (B ÷ A). (a) On the EPYC the geometric mean over the exit check's cases of the
    overridden lengths (every batch and direction in the set) is at most 0.97 with its
    interval clear of the band. (b) No case of the judged set or the EPYC's tracking tier
    is slower than 1 + band + 2 % (re-run once as above). (c) A length with any case slower
    than 1 + band loses its override before adoption.
  - An arm that fails is reported with its numbers and removed; at most three arms.

*Amendment 1 (after B's search, before any judging run of B or C).* The search on A's
build (below) found, in float, five lengths whose winner passes X38's rule at both batches
by 6–26 %. A third arm, written now and judged by fresh runs:

- **C (two-pass `avx2` float overrides, on B).** `avx2_float_overrides` gain 512 = 8 · 64,
  640 = 32 · 20, 768 = 12 · 64, 2048 = 32 · 64, 4096 = 8 · 32 · 16 (the campaign's float
  winners that are one sequence at both batches; 1024's two batches chose different
  sequences and is left out). `avx2` float only.
- **Rule for C** (C ÷ B), on the EPYC: (a) the geometric mean over the x2 catalogue's float
  cases of the five lengths (batches 1 and 64; the standard catalogue has none of them in
  float) is at most 0.97 with its interval clear of the band; (b) and (c) as for B, on the
  same judged set and the EPYC's tracking tier.

## Results

### Screens: forced two-pass sequences, O and A (EPYC, not deciding)

EPYC 7763 (DMOG), GCC 16.2, `avx2`, Slurm 512588–512590 (not `--serial`), 12 rounds, every
round under load (other processes 66 cores), band ±1.00 %; time ÷ O's planner plan:

| case | O, two passes | A, two passes | A, planner's plan |
|---|---|---|---|
| 1024 b1 / b1 bwd / b3 (16 · 64) | 0.961 / 0.959 / 0.959 | **0.837 / 0.906 / 0.840** | 0.993 / 0.996 / 1.001 |
| 1024 b16 / b256 / b1024 | 1.002 / 1.007 / 1.102 | 0.908 / 0.916 / 1.011 | 1.005 / 0.999 / 0.994 |
| 1024 in place | 0.959 | 0.851 | 1.002 |
| 1024 as 32 · 32, b1 / b1024 | | 0.889 / 1.087 | |
| 768 b1 / b64 (12 · 64) | 0.984 / 1.028 | **0.856 / 0.930** | 0.999 / 0.998 |
| 2048 b1 / b64 (32 · 64) | 1.097 / 1.106 | 1.047 / 1.090 | 0.998 / 1.006 |

Runs 20261118-033045-25ab (1024), -6964 (768), -bbe2 (2048). A takes the radix-64 last pass
from about break-even to 0.84–0.93 of the planner's plan; it changes no plan by itself
(0.993–1.006 where the plans have no radix-32/64 pass). 2048's radix-32 grouped first pass
still loses.

### Search for B (campaign x54-big-epyc)

X2's driver on A's build (d8b5376), EPYC 7763, `ISA=avx2`, Slurm 512591 (a screen, not
`--serial`), 333 candidates with radix 32 or 64 over 512, 640, 768, 1024, 1536, 2048, 4096;
confirmations 20261118-034240-7fbe onwards. Planner ÷ winner (above 1: the winner is faster):

| n | double b1 | double b64 | float b1 | float b64 |
|---|---|---|---|---|
| 512 | planner best | 8 · 64 1.029 | 8 · 64 1.262 | 8 · 64 1.198 |
| 640 | planner best | planner best | 32 · 20 1.079 | 32 · 20 1.084 |
| 768 | **12 · 64 1.162** | **12 · 64 1.089** | 12 · 64 1.097 | 12 · 64 1.063 |
| 1024 | **16 · 64 1.181** | **16 · 64 1.076** | 16 · 64 1.165 | 32 · 32 1.104 |
| 1536 | 8 · 6 · 32 1.031 | 8 · 6 · 32 1.020 | planner best | planner best |
| 2048 | planner best | planner best | 32 · 64 1.127 | 32 · 64 1.105 |
| 4096 | planner best | planner best | 8 · 32 · 16 1.088 | 8 · 32 · 16 1.068 |

By X38's rule B takes 768 = 12 · 64 and 1024 = 16 · 64 in double; C (amendment 1) the five
float lengths with one winner.

### Counters (EPYC, not deciding)

`fbench-<lib> run --pmu`, Zen 3 events in groups of five, user mode, medians of three
processes, O's and B's cache builds (planner plans) and FFTW 3.3.11 (MEASURE, shared
wisdom); Slurm 512619 (node load about 47). Per transform:

| case | plan | cycles | instructions | IPC | L1 fills | stores | FP-register-file stall, % |
|---|---|---|---|---|---|---|---|
| 768 | O 4 · 16 · 12 | 3985 | 14439 | 3.62 | 1080 | 1441 | 27.8 |
| 768 | B 12 · 64 | **3369** | 11478 | 3.41 | 848 | 1443 | 27.9 |
| 768 | FFTW | 3650 | 11920 | 3.27 | 443 | 1665 | 15.5 |
| 1024 | O 4 · 16 · 16 | 5521 | 19418 | 3.52 | 1766 | 2054 | 32.5 |
| 1024 | O, forced 16 · 64 | 5387 | 16926 | 3.14 | 1257 | 2697 | 26.2 |
| 1024 | B 16 · 64 | **4720** | 16429 | 3.48 | 1248 | 2230 | 28.0 |
| 1024 | FFTW | 4731 | 15276 | 3.23 | 1224 | 2412 | 21.8 |
| 1024 × 16 | O / B / FFTW | 94051 / 89204 / 87033 | 310476 / 262637 / 244251 | 3.30 / 2.94 / 2.81 | | | 33.1 / 32.3 / 22.5 |
| 1024 × 1024 | O / B / FFTW | 7.89 M / 10.15 M / 9.89 M | 19.9 M / 16.8 M / 15.6 M | 2.52 / 1.66 / 1.58 | | | 44.9 / 54.1 / 43.4 |
| 2048 | B forced 32 · 64 | 14335 | 36442 | 2.54 | 3804 | 4891 | 42.4 |

With the fitted order the 16 · 64 plan of 1024 keeps its instruction count (16 429 against
16 926 for the old order) and regains the IPC of the three-pass plan (3.48 against 3.14):
467 fewer stores a transform, 0.876 of the old order's cycles, level with FFTW's at batch 1
and 768 at 0.92 of FFTW's. At 1024 × 1024 (16 MiB, beyond L2) the two-pass plan takes 1.29
times O's cycles in this screen, as FFTW's own two-pass plan does (1.25): a memory effect of
the plan shape, not of the codelet; the judging run decides (rule (c) for B).

### EPYC, the exit check's cases (run 20261118-043810-5f74)

EPYC 7763 (DMOG node17), GCC 16.2, `avx2`, one thread, Slurm 512602 (`--serial`), arms O
(4d75c48), A (d8b5376), B (5d2b71c), C (a91736b), FFTW 3.3.11 (MEASURE); standard catalogue,
filter `^c2c-[0-9]+-b`, tier `full` (201 cases); 17 rounds, every one under load (other
processes 43 cores, median), A/A band ±2.10 %, so the limit of (b) is 1.041.

- **A ÷ O**, over the 12 cases whose plan has a radix-32 or radix-64 codelet (256 at
  batches 1, 16, 256, 1024; 384; 512 at 1, 16, 256, 1024; 640; 257 and 65535, whose Rader
  convolutions of 256 run 8 · 32): **0.975 [0.970, 0.980]**; 384 0.962, 512 0.968, 640
  0.968, 256 × 16 / × 256 / × 1024 0.968 / 0.956 / 0.961, 257 0.974, 512 × 16 0.977, the
  rest 0.986–0.999. All 201: 1.002 [0.997, 1.007]. Past 1.041: 524288 1.067 [1.032, 1.122]
  and 80 1.050 [1.041, 1.076], whose programs A does not change (80 = 4 · 20; a six-pass
  chain): re-run below.
- **B ÷ A**, over the double cases of 768 and 1024 (9): **0.913**; 768 0.844, 1024 0.845,
  1024 backward 0.893, 1024 × 3 0.841, 1024 × 3 in place 0.861, 1024 in place 0.857,
  1024 × 16 0.933, 1024 × 256 0.952, and **1024 × 1024 1.252** (B ÷ O 1.281 [1.231, 1.307]),
  where the batch is 16 MiB and FFTW's own two-pass plan is as slow (FFTW ÷ O 1.31; the
  counters above). Past 1.041 elsewhere: 80 1.042 (unchanged program). B ÷ O over all 201:
  0.992 [0.990, 0.996].
- **C ÷ B**: the exit check has no float case of C's lengths; 0.98–1.03 on the double cases
  (unchanged programs).

FFTN ÷ FFTW in this run, O → A → B (the named cases): 1024 1.211 → 1.197 → **1.012**; 1024
backward 1.200 → 1.209 → 1.079; 1024 × 3 1.218 → 1.219 → 1.026; 1024 × 16 1.116 → 1.105 →
1.031; 768 1.082 → 1.076 → **0.908**; 512 1.002 → 0.970 → 0.966; 2048 1.120 → 1.116 →
1.120; 64 × 16 1.136 → 1.140 → 1.134 (no radix-32/64 plan); 1024 × 1024 0.763 → 0.782 →
0.978.

### Mac, tracking tier, A ÷ O (run 20261118-032911-79d7)

Apple M5 Max, Homebrew Clang, `neon_fcma`, arms O (4d75c48) and A (d8b5376) built from the
commits, 60 rounds, 46 under load (other processes up to 6 cores), band ±1.00 % (limit
1.030): **0.999 [0.997, 1.001]**, one case faster (r2c 64 × 64 0.979), none slower; the
highest ratio 1.007 (128 × 4096). A passes (c) on the Mac.

### Xeon Gold 6240, tracking tier, A ÷ O (run 20261118-053949-7e40)

GCC 14.4, `avx512`, Slurm 16476 (`--serial`), 50 rounds, none under load, band ±1.35 %
(limit 1.034): **0.999 [0.996, 1.004]**, no case faster or slower; the highest ratios 4093
1.014 and 60 1.013. A passes (c) on the Xeon.

### Library size

x86-64 Linux, GCC 16.2, `size` of `libfftn.a` (the fbench cache builds): O 90 427 634 bytes
of text, A 90 146 912 (−0.31 %).

## Verdict

(running)
