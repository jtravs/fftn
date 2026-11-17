# X42: a radix-25 codelet and register-fitting radix-32/64 codelets (phase 1 tail)

- **Status:** running
- **Owner:** x42 session, 2026-11-17
- **Phase / task:** 1 (phase 1 exit check, the tail rule; causes 4 and 5); follows X3, X8,
  X27, X38, X39

## Question

Do (1) a radix-25 pass codelet and (2) radix-25, 32 and 64 codelets that keep their live
values within 16 registers make FFTW's plan shapes pay in FFTN: 20000, 25000, 100000, 5000,
15625 on `avx512` (Xeon) and `avx2` (EPYC), and two-pass 512–2048 on `avx2`? The answers
change the generator's catalogue, the `avx2` and `avx512` double radix tables in
`src/planner/rules.cpp`, and which codelet text the x86 targets compile.

## Hypothesis

- **Radix 25.** FFTW plans 25000 = 25 · 5 · 10 · 20, 20000 = 20 · 5 · 10 · 20 and
  100000 = 25 · 20 · 10 · 20 on the Xeon (`t3fv_25`; X39); FFTN, with radix 20 but not 25,
  needs five passes at 25000 (20 · 10 · 5 · 5 · 5) and X39's pass-count proxy puts
  25 · 10 · 10 · 10 at about 1.05 of FFTW against 1.34. The flat radix-25 codelet (Cooley–Tukey
  5 × 5: 352 additions, 184 multiplications, the derived reference; 222 vector operations
  plain, 246 DIF) peaks at 29 live values: it fits AVX-512's and NEON's 32 registers, not
  AVX2's 16.
- **Register fitting.** FFTN's radix-32 and 64 codelets peak at 35 and 67 live values; on
  AVX2's 16 registers the compiler spills them where its allocator chooses (X3 counted 67–681
  stack stores per pass), and X38 measured FFTW's two-pass plans 1.01–1.29 times FFTN's
  three radix-16 passes at 640–2048 on the EPYC, while FFTW itself is 1.13–1.22 times faster
  than FFTN there. A two-level form, R = n1 × n2 with the n1 n2 values staged once through a
  local array behind a compiler fence (each level ordered on its own: radix 64 as 8 × 8 peaks
  at 13 live values, 32 as 8 × 4 at 13, 25 as 5 × 5 at 15), costs 4–12 % more vector
  operations (DIF: 25 275 against 246, 32 253 against 243, 64 657 against 591) and exactly
  R stores and R loads, but no unplanned spill. Expected: radix-32 and 64 passes on `avx2`
  become cheaper than the two radix-4/8/16 passes they replace, so two-pass 512–2048 (FFTW's
  shape) win, by up to the 13–22 % FFTW leads.

## Change under test

Arms are commits of branch `exp/x42-codelets` against A = `origin/main` (4d00f0a):

- **B (radix-25 codelet).** The generator gains radix 25, flat, on every target, a pass radix
  only (never a leaf, as radix 20 in X38): no plan changes by itself (bits identical).
- **B2 (radix-25 sequences).** `avx512` and `avx2` double overrides with a radix-25 pass,
  selected by X2's search driver (`--radices 2-16,20,25,32,64 --require 25`) on the Xeon
  (`ISA=avx512`) and the EPYC (`ISA=avx2`) for the lengths with 5³ or more: 625, 1250, 2500,
  3125, 5000, 6250, 10000, 12500, 15625, 20000, 25000, 50000 (100000 is a grid on both
  machines; its phase of 500 is a candidate only if a phase radix of 25 is screened first).
  A length gets an override when its double winner is the same sequence at batch 1 and 64
  and at most 0.97 of the planner's at both in the confirmation (X38's selection rule).
- **C (two-level codelets on x86).** The generator writes a two-level form of the plain and
  DIF codelets of 25 (5 × 5), 32 (8 × 4) and 64 (8 × 8), which targets whose
  `codelet_registers` is 16 compile instead of the flat form. `avx2` and `avx512` round
  alike and must give the same bits with the same passes (DESIGN 6.2), so both take it
  (`avx512` takes `avx2`'s 16, as it takes its `split_block`); NEON and scalar keep the flat
  forms. Plans do not change; plans with a 25, 32 or 64 pass on x86 change bits (within the
  tests' tolerances).
- **D (re-fitted `avx2` double sequences, 512–4096).** X2's search driver on the EPYC with
  C's build, candidates with radix 32 or 64 (`--radices 2-16,20,25,32,64 --require 32,64`),
  lengths 512, 640, 768, 1024, 1536, 2048, 3072, 4096; overrides by B2's selection rule.

Order of the commits (amended before any timing): B, C, then B2 and D, whose sequences are
searched on C's build, since that is the codelet text both x86 targets then run; C is
judged against B on plans that already take a radix-32 pass on `avx2` (256, 384, 512, 640)
and on forced radix-25/32/64 sequences, B2 and D against C. (C's revert, 07fa979, removed
this paragraph with C's commit; it is restored here.)

No switch: arms are commits, `dev/KNOBS.toml` is untouched. Library text size is recorded
per arm (Mac static library, Clang; x86-64 Linux, GCC 16).

## Method

- Screens (not deciding): `fbench-fftn run --pmu` and fbench runs with forced `RADICES` of
  the two-level and flat builds, the search driver's screening and refinement stages.
- Judging runs: `fbench_remote.py submit --serial`, one thread, interleaved process rounds
  with the A/A control:
  - Xeon Gold 6240 (GCC 14.4, `avx512`, FFTW 3.3.11 AVX-512, MKL): B2 and C against the arm
    before, the exit check's cases (standard catalogue, filter `^c2c-[0-9]+-b`, tier `full`,
    restricted to the affected lengths), then the tracking tier.
  - EPYC 7763 (DMOG, GCC 16.2, `avx2`): B2, C and D against the arm before, the same.
  - Apple M5 Max (Clang, `neon_fcma`): B (no plan change; collateral only) and, if a NEON
    override is tried, the same rule.
- Metric: fbench's median paired ratio arm ÷ arm before per case with its interval; the
  geometric mean over the cases whose plan (or, for C, whose codelet text) changes; FFTN ÷
  FFTW and ÷ MKL before and after.
- **Decision rule, written before any arm is timed.** Judged on the release toolchains (GCC
  on x86-64 Linux, Clang on macOS), per target, in double:
  - B2 and D (overrides), on the target's deciding machine (`avx512`: Xeon; `avx2`: EPYC):
    (a) the geometric mean over the changed target cases is at most 0.97 with its interval
    clear of the band; (b) no case of the judged set or of the tracking tier is slower than
    1 + band + 2 % (a case past it on a loaded machine is re-run once, `--serial`, and the
    re-run decides); (c) a length whose cases are slower than 1 + band at any batch or
    direction loses its override before adoption.
  - C (codelet text, both x86 targets together): adopted if on the EPYC (`avx2`) the
    geometric mean of C ÷ B over the judged cases whose plans contain a radix-25, 32 or 64
    pass is at most 1 − band with its interval below 1, and on the Xeon (`avx512`) that
    geometric mean is at most 1 + band with no such case slower than 1 + band + 2 %;
    collateral as (b) on both. If C wins on `avx2` and loses on `avx512` past that limit,
    the pair contract decides: C is rejected as written and the per-target form (the
    contract changed for codelets above 23) is put to the lead, not adopted here.
  - An arm that fails is reported with its numbers and removed; at most three arms per part.

*Amendment 1 (after the screens and searches below, before any judging run of an override
arm).* The screens showed the two-level radix-25 codelet slower than the flat one on both
x86 targets, so B2's sequences are searched on a build without C (515b0c1, campaigns
x42-r25f-epyc and x42-r25f-xeon; the campaigns on C's build, x42-r25-epyc and
x42-r25-xeon, are recorded but not used). B2 keeps the selection rule as written (one
winner at both batches). A further arm, written now and judged by fresh runs:

- **B3 (radix-25 sequences, either batch's winner).** B2 plus each length of the searched set
  whose confirmation run has a sequence at most 0.97 of the planner's at batch 1 and at
  batch 64 though the two batches' winners differ: the sequence with the lower geometric
  mean of the two ratios. Judged against B2 by the same rule (a)–(c).

## Results

### Screens: flat against two-level codelets (EPYC, `avx2`, GCC 16.2)

Forced sequences, the planner's plan as baseline, flat (B, 7954244) and two-level (C,
3fd2203) builds in one run each, 12 rounds, band ±1.0 % (runs 20261117-110218-defb,
-20d7, -7067, -b9c8; screens: four ran at once on shared nodes, so not deciding). Time ÷
the planner's, batch 1 / 64, flat → two-level:

| n | sequence | flat | two-level |
|---|---|---|---|
| 512 | 16 · 32 (the planner's) | 0.99 / 0.99 | 1.00 / 0.99 |
| 512 | 8 · 64 | 1.17 / 1.10 | 1.10 / 1.04 |
| 1024 | 16 · 64 | 0.99 / 1.07 | 0.99 / 1.04 |
| 1024 | 32 · 32 | 1.04 / 1.11 | 1.05 / 1.09 |
| 1024 | 64 · 16 | 1.21 / 1.24 | 1.24 / 1.26 |
| 2048 | 32 · 64 | 1.16 / 1.23 | 1.17 / 1.23 |
| 25000 | 10 · 10 · 10 · 25 | **0.92 / 0.97** | 0.97 / 1.01 |
| 25000 | 25 · 10 · 10 · 10 | 1.10 / 1.33 | 1.16 / 1.45 |

Counters (`fbench-fftn run --pmu`, user mode, median of three processes, Slurm 511713 and
511720, node17-class), per point at batch 1: 1024 as 4 · 16 · 16 (the planner's) 5.47
cycles, 19.0 instructions, IPC 3.46; 16 · 64 flat 5.55 cycles, 16.6 instructions, IPC 2.98;
two-level 5.39, 16.7, 3.10; 32 · 32 flat 5.91 / 17.6, two-level 5.90 / 18.2. GCC 16 already
folds the flat codelets' spills into memory operands: the two-level form leaves the
instruction count unchanged (± 3 %) where Clang's lane pass loses two thirds of its stack
moves (gen/README.md). The two-pass plans execute 7–13 % fewer instructions than three
passes but at a lower IPC (2.5–3.1 against 2.9–3.5), and no FP pipe is the limit (busiest
pipe, fp2, 52 % of cycles at 1024 16 · 64). 25000: 10 · 10 · 10 · 25 flat 10.0 cycles per
point against 10.9 for the planner's, two-level 11.1 (more instructions: 40.6 against
38.9 at 25 · 10 · 10 · 10).

Xeon Gold 6240, `avx512`, GCC 14.4, run 20261117-111807-dd3c (12 rounds, 10 under load,
band ±1.0 %): 25000 as 10 · 10 · 10 · 25 flat 1.07 / 0.81, two-level 1.10 / 0.84; as
25 · 10 · 10 · 10 flat 1.09 / 1.09, two-level 1.20 / 1.21.

Mac (M5 Max, `neon_fcma`, Clang), runs 20261117-112402-d7a2 and -112417-040a (8 rounds,
band ±1.0 %): radix 25 does not win: 25000 as 10 · 10 · 10 · 25 1.00, others 1.02–1.25;
3125 as 25 · 5 · 25 1.01, 5 · 25 · 25 1.03–1.05, against the planner's radix-5 chains. So
radix 25's rows exist on `avx2` and `avx512` only (0aaf15b; Mac library text back to
main's).

### C judged (EPYC, `avx2`, `--serial`)

Run 20261117-113412-4c22 (Slurm 511719, `--serial`; 30 rounds, none under load, band
±1.00 %), arms B (7954244) and C (3fd2203), the exit check's cases whose `avx2` plans take
a radix-32 pass (256, 384, 512, 640 at their batches; 10 cases): C ÷ B weighted geomean
**1.000** [0.998, 1.003]; 384 0.977, 512 × 16 1.015, the rest 0.996–1.008. Rule (a) asks for
at most 0.99 with the interval below 1: **C is rejected** and reverted (07fa979; the Xeon
leg was not needed). Library text: C changed x86-64 GCC 16 `libfftn.a` by −0.03 % (57.22 →
57.20 MB) and the Mac's not at all.

### Searches (X2's driver)

- x42-big-epyc (C's build, `avx2`, candidates with 32 or 64; screening from
  20261117-110824-0d5b): no double length passes the selection rule; best 768 = 12 · 64 1.030
  at batch 1 only, 1024 = 16 · 64 1.017 at batch 1 only; 512, 640, 2048, 4096: the planner's
  sequence is best. So arm D has no override and is not built.
- x42-r25f-epyc (flat build, `avx2`, candidates with 25; confirmations 20261117-112854-1cfd
  to -113026-0983), planner ÷ winner at batch 1 / 64: 500 20 · 25 1.093 / 1.039, 3125
  5 · 25 · 25 1.201 / 1.195, 10000 20 · 20 · 25 1.066 / 1.057, 15625 25 · 25 · 25 1.176 /
  1.158 (one winner: B2); 5000 10 · 25 · 20 / 10 · 20 · 25 and 25000 20 · 5 · 10 · 25 /
  10 · 5 · 20 · 25 (winners differ: B3 takes 10 · 20 · 25, 0.870 / 0.901, and
  10 · 5 · 20 · 25, 0.932 / 0.948); 20000 no sequence passes at both batches; 100, 1000,
  2000, 3000: the planner's sequence is best.
- x42-r25f-xeon (flat build, `avx512`, confirmations 20261117-121144-f413 to -122306-e8c2;
  bands up to ±7.3 % on the shared Xeon): one winner at both batches, intervals clear of the
  band (B2): 100 4 · 25 1.297 / 1.270, 500 20 · 25 1.247 / 1.204; winners differ (B3): 3125
  25 · 5 · 25 (0.644 / 0.683), 10000 20 · 20 · 25 (0.897 / 0.923), 15625 5 · 5 · 25 · 25
  (0.839 / 0.904), 25000 4 · 25 · 10 · 25 (0.968 / 0.734); 5000 20 · 25 · 10 0.914 / 0.952
  with the batch-64 interval in the band (neither arm); 1000 and 20000 pass at one batch
  only (20000 and 25000 at batch 64 0.70–0.74: radix 25 last where the batch leaves L2).
- The campaigns on C's build (x42-r25-epyc, x42-r25-xeon) chose 3125, 5000, 15625 on the
  EPYC and only 100 on the Xeon; not used.

### B2 and B3 on `avx2` (EPYC, `--serial`)

Run 20261117-115206-d594 (Slurm 511780; 30 rounds, 5 under load, band ±1.00 %), arms N
(1137965: main's plans with the radix-25 rows), B2 (a44c8e5), B3 (10e9487), FFTW; the
standard catalogue's rows of the overridden lengths (all at batch 1) and the r2c and c2r
rows of 10000 (complex half 5000):

| case | B2 ÷ N | B3 ÷ N | FFTN ÷ FFTW, N → B3 |
|---|---|---|---|
| c2c 500 | **0.900** | 0.900 | 0.98 → 0.88 |
| c2c 3125 | **0.831** | 0.836 | 0.77 → 0.65 |
| c2c 5000 | 1.010 | **0.866** | 0.96 → 0.83 |
| c2c 10000 | **0.932** | 0.928 | 0.92 → 0.85 |
| c2c 15625 | **0.836** | 0.836 | 0.79 → 0.66 |
| c2c 25000 | 0.994 | **0.907** | 0.98 → 0.89 |
| r2c 10000 | 1.002 | **0.895** | 1.01 → 0.90 |
| c2r 10000 | 1.004 | **0.906** | 0.99 → 0.89 |

B2 over its four changed cases: geomean **0.874**, none slower; B3 against B2 over its four
(5000, 25000, r2c and c2r 10000): **0.891**; B3 against N over all eight 0.884 [0.879,
0.889]. Tracking tier, N against B3 (run 20261117-115432-0b0b, Slurm 511781; 24 rounds, all
under load, band ±4.22 %): 0.987 [0.751, 1.003], r2c and c2r 1000 (complex half 500) 0.93
and 0.90 faster, 59 no evidence; c2r 65536 1.074 past 1 + band + 2 % with an unchanged
plan, re-run (20261117-123726-548b, Slurm 511810; 40 rounds, band ±1.00 %): c2r 65536
0.993, c2c 65536 0.997. **B2 and B3 pass (a)–(c) on `avx2`.** (The EPYC's c2c rows of
these lengths were already at or under FFTW's time; 20000, its exit-check case at 1.07,
gained nothing from radix 25.)

## Verdict

(pending)
