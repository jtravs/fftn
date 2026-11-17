# X46: a depth-first long chain (rows between L2/4 and a few times L2)

- **Status:** done (A0, A, B adopted on the Xeon profile; C rejected, C2 adopted on the Apple profile; EPYC unchanged)
- **Owner:** x46 session, 2026-11-17
- **Phase / task:** 1 (phase 1 exit check, the tail rule; cause 5); follows X3, X26, X30b,
  X39, X42

## Question

Does a depth-first chain, in which every sub-row runs to completion while it is in L1 or L2,
close the rows between about a quarter of L2 and a few times L2, where FFTN's breadth-first
Stockham chain overfills L2 and the two-phase grid's copies cost too much (Xeon 6240 16384
1.21 of FFTW and 1.16 of MKL, 32768 1.22 / 1.13, 65536 ÷ MKL 1.15–1.23, 20000 1.13, 25000
about 1.18, 100000 1.28; Mac 8192–65536 1.05–1.10)? The answer adds a strategy (a program
of the existing pass kinds plus two new pass loops) and a profile rule for when to take it.

## Hypothesis

LESSONS 3.22: on the Xeon 6240 (1 MiB private L2) the chains of 16384–32768 complex doubles
(256–512 KiB) missed L2 0.7–1.7 times per point at an IPC near 1 with fewer instructions
than FFTW, which plans 16384 as recursive decimation in time (8 · 64 · 32) whose 2048-point
sub-transforms finish in L1 and misses L2 0.0–0.3 times per point. The breadth-first chain
keeps three lines live through four passes (input, output, scratch a: 768 KiB at 16384) plus
the first pass's twiddle table (about n entries); the grid's two phases each gather and
scatter (30–50 instructions per point against the chain's 12).

A two-level decimation in time keeps one line hot after its first pass. With n = R1 · M:

1. *depth-first first pass* (one pass over the row): for every residue r < R1, the first
   pass of the length-M chain of x[R1 j + r], with lanes over C adjacent residues (the input
   is R1 interleaved sub-rows, the layout a Stockham lane pass reads with s = R1 copies) and
   its outputs transposed in registers into each sub-row's own block of the output,
   out[r M + R2 p + k];
2. *sub-rows* (a loop of R1 independent iterations): the rest of each sub-row's chain on its
   block of M (16–32 KiB) and one or two scratch lines of M, ending in its block: every pass
   but the first in L1 or L2;
3. *top pass* (one pass over the row, in place): the radix-R1 decimation-in-time butterflies
   X[k + M t] = Σ_r ω_R1^(r t) w_n^(r k) Y_r[k], lanes over adjacent k, legs M apart, the
   twiddles per lane (the grouped first pass's table for (n, R1)).

So the row is touched in three sweeps (input to output, the sub-rows' blocks once each, the
top pass in place) by two lines instead of three, and every middle pass is cache-resident.
Expected on the Xeon: 16384 and 32768 near FFTW (0.80–0.90 of the current plans), L2 misses
per point down to FFTW's; 65536–100000 maybe, with M in L2; the EPYC (512 KiB L2) and the
Mac (a 16 MiB L2 shared by the cluster) probably differ, so the rule is per profile.

## Change under test

Arms are commits of branch `exp/x46-depth-first` against A = its base (X42's head merged
with `main`, b27970f):

- **A0 (the strategy, no default change).** Two pass loops on every target (the depth-first
  first pass and the top pass, `src/passes/depth.inl`), the strategy as a program of them and
  the chain's lane passes (`src/strategies/depth.*`), `strategy_choice::depth` and its split
  (`build_settings::depth_radix`) for tests and fbench (`STRATEGY=depth`, `DEPTH_RADIX=R1`).
  No plan changes; the strategy is tested in both precisions on every target.
- **A (the rule).** A profile field: c2c rows contiguous out of place whose bytes lie in a
  range keyed on the private L2 take the depth-first chain, R1 and M chosen by a rule fitted
  on the screen (step 1) and written here before the judging run.
- **B (derived twiddles for the top pass)**, only if the counters show the top pass's table
  (about n complex values) is a large part of its L2 traffic: w_n^(r k) formed in registers
  from w_n^k and a few stored powers, measured on the changed rows only (LESSONS 3.10).
- **C (the factor rule X39 left)**: 100000 as 100 × 1000 on the Xeon profile (five passes
  against six, 0.92–0.94 in X39's screen), if the depth-first chain does not take 100000.

Bits: the depth-first chain is another factorisation, so a changed plan changes the
rounding of its rows; tests hold it to the derived tolerances in both precisions and to
bitwise equality between serial and threaded runs.

## Method

- Machines: Xeon Gold 6240 (`avx512`, GCC 14.4, FFTW 3.3.11 with AVX-512, oneMKL 2026.1;
  Slurm `-c 8` shared; timing through `fbench_remote.py submit --serial`), then the EPYC 7763
  (`avx2`, GCC 16, DMOG `-c 8`) and the M5 Max (`neon_fcma`, Homebrew Clang, NEON FFTW).
- Step 1, screen (no default change): on the Xeon, user-mode counters (`fbench-fftn run
  --pmu`: cycles, instructions, `l2_rqsts.miss`; min over processes) and times of the rules'
  plan against forced depth-first splits (`STRATEGY=depth`, `DEPTH_RADIX`), FFTW and MKL, on
  8192, 16384, 20000, 25000, 32768, 46080, 65536, 100000 and 131072. The rule of arm A is
  written from it.
- Step 2, each arm against the one before, fbench interleaved rounds with the A/A control, on
  the standard catalogue's rows the rule changes (`--tier full`), then the standard tracking
  tier.
- Metric: median paired ratios with 95 % intervals; weighted geomean over the cases whose
  plan an arm changes; FFTN ÷ FFTW and ÷ MKL before and after.
- Decision rule, written before running: an arm is adopted on a machine's profile (the
  release toolchain there: GCC on x86-64, Clang on macOS) if over the cases whose plan it
  changes the weighted geomean B ÷ A is below 1 − band with its interval below 1, no such
  case is slower than 1 + max(band, 3 %), and no tracking-tier case is slower than
  1 + max(band, 3 %) (a case just past the limit on a loaded run is re-run once). A profile
  on which the arm is not judged keeps the old plans (the field off; the profile tests check
  the recorded machines). At most three arms.

## Results

### Step 1, the screen (Xeon, counters and times)

Builds: base b27970f (X42 merged with `main`; the same plans as `main` on these rows), A0
41c7903 and 85b4ee0 (forced only). Campaign `~/awork/fftn-x46` on the Xeon, `fbench-fftn run
--pmu cycles,instructions,l2miss=0x3f24` (user mode, `l2_rqsts.miss`), min over three
processes; Slurm 16388 (first screen) and 16391 (second, with derived twiddles), `-c 8`
shared, node load 3–12. Times per point and the ratio to the base plan in the same job;
D = stored top-pass twiddles, E = derived (arm B's form). The fbench screen of the first
(run 20261117-165423-78a6, Slurm 16386/16387, 10 quiet rounds, band ±1.86 %) agreed with the
counters within 0.03 on every case.

| row (base plan) | base ns/pt, L2 misses/pt | best D (ratio, misses) | E16 or E20 (ratio, misses) | FFTW, MKL ns/pt |
|---|---|---|---|---|
| 8192 (chain) | 2.02, 0.00 | 8 × 1024 0.90–0.99 | E16 1.06; E8 0.94 | 2.32, 1.99 |
| 16384 (chain 16·8·8·16) | 3.17, 0.83 | 8 × 8 × 256 0.79, 16 0.81, 8 0.77 | E16 **0.69**, 0.01 | 2.64, 2.44 |
| 20000 (chain 20·20·5·10) | 3.33, 0.96 | 20 × 1000 0.86–0.87 | E20 **0.73**, 0.00 | 2.82, 3.91 |
| 25000 (chain) | 3.94, 1.26 | no split: C = 4 divides no top and its sub-row | — | 3.02, 4.03 |
| 32768 (grid 128 × 256) | 4.19, 0.27 | 4 × 8 × 1024 0.89, 8 0.88 | E16 **0.67**, 0.29 (E32 0.63) | 3.14, 3.60 |
| 46080 (grid 192 × 240) | 4.32, 0.52 | 16 0.86–0.89 | E16 **0.63**, 0.56 | 3.48, 4.64 |
| 65536 (grid 256 × 256) | 5.10, 0.86 | 16 0.83–0.86 | E16 **0.70**, 0.83 | 4.75, 4.82 |
| 65536 in place | 5.71, 1.02 | 16 0.88 | E16 **0.69**, 1.21 | 4.67, 4.79 |
| 65536 backward | 5.15, 0.87 | 16 0.85 | E16 **0.69**, 0.85 | 4.65, 4.81 |
| 100000 (grid 200 × 500) | 5.04, 0.78 | 20 × 5000 1.01 | E20 **0.77**, 0.99 | 3.77, 5.86 |
| 131072 (grid 32 × 64 × 64) | 5.90, 1.30 | 16 0.81–0.85 | E16 **0.71**, 1.06 | 5.50, 6.62 |
| 262144 (grid 64³) | 6.43, 1.31 | 16 0.84 | E16 **0.71**, 1.22 | 5.61, 7.19 |

- The breadth-first chain's misses (0.8–1.3 per point at 16384–25000) fall to FFTW's level
  only with derived twiddles (0.00–0.01): the top pass's stored table, (R1 − 1) M entries in
  the pre-arranged storage of 32 bytes each (384 KiB at 16384 = 4 × 4096, 480 KiB for the
  current chain's first pass), was the rest of the L2 traffic. With them the depth-first
  chain runs at 0.63–0.77 of the base plan from 16384 to 262144 points, below FFTW's time
  except at 100000 (1.02), and below MKL's everywhere.
- Top factors: one top of 16 (20 where C does not divide n / 16) was within 0.05 of the
  best split tried at every row; tops of 4 left sub-rows too long (D4 at 65536–131072
  0.99–1.01), two tops added a pass over the row without fewer misses.
- 8192 (L2/8) gains nothing that the counters resolve; 25000 has no split on `avx512` (it
  has three factors of 2 and needs four).
- The EPYC (`avx2`, screen run 20261117-165514-af0c, every round under load, band ±1.27 %,
  stored twiddles): tops of 16 lose (1.19–1.71: 16 legs on 16 registers), tops of 4 and 8
  win on some rows (46080 0.81, 65536 0.91, 16384 0.93) and lose on others (25000 1.13–1.60,
  32768 1.04): no rule from this screen; the Mac (local screen, stored twiddles) gains at
  8192 (0.65) and loses or ties beyond. Neither profile changes in this experiment.

### The rule (arms A and B), written before the judging run

- **A (commit after 85b4ee0).** Profile fields `depth_min_bytes`, `depth_max_bytes`, set to
  L2/4 and 4 L2 where X39's fields are (x86-64, a private L2 with less than 2 MiB of L3 per
  CPU: the Xeon 6240): a c2c row of one axis in that range, `avx512`, double, contiguous on
  both sides (in place through the workspace block), no factor edge, is a depth-first chain
  with one top factor, 16 where C divides n / 16, else 20, where it fits; stored twiddles.
  Float and other targets keep their plans (unmeasured). In bytes against the L2 so that the
  rule carries to other profiles once measured there (Granite Rapids' 2 MiB L2 profile does
  not take it here).
- **B (the commit after A).** The same plans with derived top-pass twiddles; the stored
  form is deleted if B is adopted.
- **C (the Apple profile, a new arm named after the screen above and before its judging
  run).** A local screen on the M5 Max (harness, two processes per variant, with derived
  twiddles): a top of 4 took 0.88 (6144), 0.64 (8192), 0.91 (9216), 0.96 (10240), 0.85
  (12288), 0.90 (14336), 0.83 (16384) and 0.84 (16384 × 3) of the rules' plan, 0.92–0.95 at
  3072–5120, but 1.05–1.09 at 10000, 12000 and 16000 (power-of-two parts of 16–128 points)
  and 1.06 at 32768 (X32's chain); 24576 and 49152, grids there, took 0.71–0.72. The rule:
  where a cluster shares L2, rows of L1/2 to 2 L1 (4096–16384 complex doubles on the M5 Max)
  whose power-of-two part is above L1/8 (1024 points) take one top of 4, on NEON in double
  (the rule's target condition becomes "32 vector registers": `avx512`, `neon`,
  `neon_fcma`). Judged on the Mac (Clang, `neon_fcma`) by the decision rule against B, on the
  standard catalogue's rows it changes (c2c 4096 × 1, 16, 256, 1024 and in place, 8192,
  16384) and the tracking tier.
- **C judged (Mac, run 20261117-174232-3f7a; Homebrew Clang 23, `neon_fcma`; 30 rounds, all
  under load, band ±1.00 %)**, C = 6ea41f4 against N = f78509c (B: the Mac's plans as `main`):
  the 7 changed cases 0.938 [0.930, 0.944]; 8192 0.753 [0.743, 0.775], 16384 0.886, 4096 0.942,
  4096 in place 0.958, 4096 × 16 0.993, but 4096 × 256 **1.031** and × 1024 **1.039** (beyond
  1 + max(band, 3 %); batches of 4096 run from memory). **C fails the rule.**
- **C2 (named after C's run, before its own).** The same with the range L1 to 2 L1 (8192–16384
  complex doubles on the M5 Max): rows of 4096 keep their plans. Judged by the same rule
  against B on the same cases and the tracking tier.
- Judging run: Xeon, `--serial`, arms N = 85b4ee0, A, B, F = FFTW, M = MKL, the standard
  catalogue's rows the rule changes (c2c 16384, 20000, 32768, 46080, 65536 out of place,
  in place, backward and × 3 in and out of place, 100000, 131072, 262144), then the tracking
  tier N against B. The decision rule above applies to A against N and to B against A.

### Step 2, the judging runs

**Xeon, A and B** (run 20261117-191049-bba3; Slurm 16394 build, 16395 timing, `--serial`;
GCC 14.4, `avx512`, double, one thread; 30 rounds, 5 under load, band ±1.00 %): arms N =
85b4ee0 (the plans of `main`), A = 326017c, B = f78509c, F = FFTW 3.3.11 (MEASURE), M = oneMKL
2026.1. The 13 standard rows of the range; 25000 has no split and keeps its plan (A 1.008,
B 0.999, no evidence).

| row | A ÷ N | B ÷ N | B ÷ A | FFTN ÷ FFTW, N → B | FFTN ÷ MKL, N → B |
|---|---|---|---|---|---|
| 16384 | 0.777 | 0.654 | 0.842 | 1.29 → 0.84 | 1.31 → 0.85 |
| 20000 | 0.851 | 0.698 | 0.820 | 1.25 → 0.86 | 0.88 → 0.60 |
| 32768 | 0.895 | 0.636 | 0.711 | 1.32 → 0.86 | 1.11 → 0.72 |
| 46080 | 0.883 | 0.606 | 0.686 | 1.18 → 0.72 | 0.96 → 0.58 |
| 65536 | 0.828 | 0.658 | 0.795 | 1.06 → 0.70 | 1.07 → 0.71 |
| 65536 backward | 0.836 | 0.671 | 0.803 | 1.07 → 0.72 | 1.07 → 0.72 |
| 65536 in place | 0.933 | 0.697 | 0.747 | 1.14 → 0.80 | 1.13 → 0.79 |
| 65536 × 3 | 0.809 | 0.662 | 0.818 | 1.06 → 0.70 | 1.09 → 0.72 |
| 65536 × 3 in place | 0.916 | 0.699 | 0.763 | 1.14 → 0.79 | 1.13 → 0.79 |
| 100000 | 0.986 | 0.752 | 0.763 | 1.34 → 1.02 | 0.88 → 0.67 |
| 131072 | 0.806 | 0.676 | 0.839 | 1.09 → 0.73 | 0.90 → 0.61 |
| 262144 | 0.817 | 0.685 | 0.838 | 1.07 → 0.74 | 0.87 → 0.60 |

Over the 12 changed rows: A ÷ N 0.859 (every interval below 1; the largest 100000 0.986
[0.971, 0.995]); B ÷ A 0.784 (at most 0.842); B ÷ N 0.674 (weighted geomean over the 13 rows
0.694 [0.689, 0.700]). Tracking tier, N against B (run 20261117-192031-e85d, Slurm 16396/16397,
`--serial`; 24 rounds, every one under load, band ±6.66 %): 6 faster, all of them changed rows
(46080 0.547, 262144 0.554, 65536 0.638, in place 0.646, 100000 0.679, 16384 0.685), 55 no
evidence, weighted geomean 0.939 [0.888, 0.987]; the largest median ratios were rows whose
plans do not change (c2c 16 1.093 [0.974, 1.293], with B's own median time below N's; r2c 256
1.064; c2c 1024 backward 1.036; 2²⁰ 1.037). RERUN

**Mac, C and C2** (Homebrew Clang 23, `neon_fcma`, NEON FFTW 3.3.11): C above. C2 = d64e468
against N = f78509c, run 20261117-175324-c4e7 (30 rounds, 1 under load, band ±1.00 %): 8192
0.728 [0.706, 0.738], 16384 0.837 [0.812, 0.874]; the five rows of 4096 unchanged (0.993–1.005,
no evidence); FFTN ÷ FFTW 8192 1.14 → 0.83, 16384 0.98 → 0.82. Tracking tier N against C2, run
20261117-175507-1075 (24 rounds, 3 under load, band ±1.00 %): 1 faster (16384 0.848), 58 no
evidence, 2 slower by 1.016 (4093 and 10403, Bluestein rows whose plans do not change),
weighted geomean 0.998 [0.996, 0.999].

**EPYC** (no plan changes; collateral of the shared code): TRACKE

## Verdict

- **Diagnosis.** On the Xeon 6240 the rows from L2/4 up missed L2 because three lines and a
  first pass's twiddle table of about n entries (32 bytes each in x86's pre-arranged storage)
  overfill the 1 MiB L2. A depth-first chain keeps two lines and finishes every middle pass
  on a sub-row in cache; with derived top-pass twiddles its misses at 16384–25000 fall to
  FFTW's (0.00–0.01 per point).
- **A0 (the strategy, forced only): built**, every target, both precisions, in and out of
  place, tested against the reference and bit for bit across team sizes.
- **A (the Xeon profile's rule, stored twiddles): adopted** (0.859 over 12 changed rows,
  none slower), and then superseded by B.
- **B (derived top-pass twiddles): adopted** (0.784 against A on the same rows, at most
  0.842); the stored form is deleted (87fbb85). The Xeon's changed rows take 0.674 of
  `main`'s time; FFTN ÷ FFTW there 1.06–1.34 → 0.70–0.86 (100000 1.02), ÷ MKL 0.58–0.85.
- **C (Apple, L1/2 to 2 L1): rejected** by two batch rows of 4096 (1.031, 1.039).
  **C2 (Apple, L1 to 2 L1): adopted** (8192 0.728, 16384 0.837, tracking none slower than
  1.016 on rows it does not change).
- **The EPYC 7763 keeps its plans**: no rule passed the screens (tops of 16 lose 1.2–1.7 on
  16 registers; tops of 4 and 8 win at 20000 and 46080 only). Granite Rapids' profile (2 MiB
  L2) does not take the rule; the fields are in bytes against L2 so that it can once measured.
- **Arm C as planned (100000 as 100 × 1000)** was not built: the depth-first chain takes
  100000 to 0.752 of its time.
- **Library size**: text +4.4 % on the Mac (Homebrew Clang 23, `size` of `libfftn.a`, 20.83 → 21.75 MB at b27970f → 87fbb85); on x86-64 (GCC 14.4) the archive grew 4.7 % with both top-pass forms (b27970f → 85b4ee0), less after the stored form was deleted (not measured).
- **Left open.** 25000 (1.33 of FFTW: three factors of 2, so C = 4 divides no top and its
  sub-row; a masked or tiled first pass for C not dividing the top would cover it); 100000 at
  FFTW's time; float rules (unmeasured); three or more tops for rows beyond 4 L2 (the
  sub-rows of 524288 would be 512 KiB); the EPYC's 46080 and 20000 (0.83 and 0.94 with a top
  of 4); Granite Rapids and Zen 5 runs; rows of 24576 and 49152 on the Mac (grids there,
  0.71–0.72 in the screen) and 65536–131072 (0.90–0.92).
