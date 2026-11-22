# X62: the EPYC's remaining phase 1 exit cases

- **Status:** done (K3 with L adopted, odd batches from 5 rows; C adopted; K and K2 rejected)
- **Owner:** x62 session, 2026-11-22
- **Phase / task:** 1 (improvement after the exit: the tail of [X56](X56-exit-4.md)'s EPYC check);
  follows [X53](X53-two-row-leaves.md), [X54](X54-radix64-avx2.md), [X57](X57-epyc-1024.md)

## Question

The exit check on `main` fa155c9 (run 20261119-005944-cb48, every round under load, band
±1.67 %) left six c2c cases beyond noise against FFTW on the EPYC 7763: 64 × 2 backward 1.30,
64 × 16 1.13, 15 × 2 backward 1.13, 1018 1.10, 2048 1.09, 120 1.08 (and 1024 backward 1.08, 100
1.07 inside that band). Which of them are real losses on today's `main`, what does each come
from (planning, kernel cost or code placement), and which can a single change move?

## Diagnosis

### Re-time on `main` 021c117 (EPYC, not deciding)

Run 20261122-175008-ef4f (Slurm 515447 build, 515448 timing, node16; 27 rounds, all under load,
other processes 63 cores median; band ±1.00 %), arms FFTN, its identical copy and FFTW 3.3.11
(MEASURE, `PLAN_NOTE=1`). FFTN ÷ FFTW, with each library's plan:

| case | FFTN ÷ FFTW | FFTN's plan | FFTW's plan |
|---|---|---|---|
| 64 × 2 backward | **1.294** | 4 × 16 per row | `n1bv_64` over both rows (two rows per register) |
| 64 × 16 | **1.178** | 4 × 16 per row | `n1fv_64` over the rows |
| 64 × 256 | 1.045 | 4 × 16 per row | `n1fv_64` over the rows |
| 1018 | **1.125** | Bluestein of 1018, M 2048 = 8 × 16 × 16 | 2 × Bluestein of 509, M 1024 = 16 → `n2fv_64` |
| 2048 | **1.105** | 8 × 16 × 16, split from pass 2 | 32 (`t1fv_32`) → 64 (`n2fv_64`) |
| 120 | **1.077** | 10 × 12 | 20 (`t2fv_20`) → 6 |
| 100 | **1.076** | 10 × 10 | 10 → 10 |
| 72 | **1.078** | 6 × 12 | 6 → 12 |
| 1024 backward | 1.059 | 16 × 64 | 16 → 64 |
| 1024 | 1.021 | 16 × 64 | 16 → 64 |
| 15 × 2 backward | **0.893** | batch leaf of 15 | `n1bv_15` over both rows |
| 64 | 0.970 | 4 × 16 | 2 → 32 |

- **15 × 2 backward is not a loss**: 0.89 of FFTW here, as in run 20261118-194340-02a4 (0.91);
  the exit check's 1.13 was its scattered rounds (X56).
- **64 × B is one kernel**: FFTW runs every batch of 64 as its 64-point codelet over two rows
  per register; FFTN runs each row as 4 × 16. X53's batch leaf of 64 is that shape, but no rule
  takes it (after X54 it ran 64 × 2 at 0.91 of the chain, 64 × 3 at 1.06–1.08 and 64 × 1024 at
  1.035: not enough, and in need of a batch range).
- **1018 and 2048 are the same cost**: 1018's Bluestein convolution is two transforms of 2048
  points in FFTN's 8 × 16 × 16, the plan that trails FFTW at 2048. X54 and X57 screened every
  two-pass plan of 2048 with radix 32 or 64 and none won (its radix-32 grouped first pass stalls
  on the FP register file, X47).
- **72, 100 and 120 move together** (1.08 each) with plans X38 and X57 kept: X57 found 72 and 120
  to be code placement (aligning the same functions to 64 bytes took them to 0.88 and 0.93 of
  their time).

### Screens (EPYC, not deciding)

Same node class, `fbench_remote.py submit` without `--serial`, catalogue `x62` (this experiment's:
64-point rows at 2–4096 rows both directions and in place, 32-point rows, Bluestein lengths, 2048,
the short rows), every round under heavy load (other processes 91–94 cores of 128).

- Run 20261122-175239-5260 (Slurm 515457, 20 rounds, band ±1.00 %), forced options on `main`:
  - `BATCH_LEAVES=own` (X53's leaf of 64 with block transposes) ÷ the chain: 64 at 2–128 rows
    0.89–0.93, 256 0.95–0.98, 512 1.00–1.02, 1024 and 4096 1.03–1.07; odd counts 3 rows 1.07–1.09,
    5 and 17 rows 0.93–1.02. The leaf of 32 loses everywhere (1.02–1.23), as in X53.
  - `BLUESTEIN_POW2_PART_BYTES=8192` (a convolution's power-of-two part at most 512 points instead
    of the L1's 2048): 1018, 1013, 1019, 1021 take M = 2160 = 12 × 15 × 12 instead of 2048 = 8 × 16 ×
    16 and 0.92–0.94 of their time (1018 ÷ FFTW 1.10 → 1.02).
  - `LAYOUT=interleaved`: 2048 no evidence (1.056 [0.997, 1.113]); the Bluestein lengths 1.17–1.20
    slower. Nothing to take.
- Run 20261122-180330-a81f (Slurm 515460 build, 515461 timing, 20 rounds, every one started
  overloaded, band ±1.49 %): arm K (below) forced, `BATCH_LEAVES=own`, against `main`'s chain and
  `main`'s forced leaf. K ÷ the chain at 64 points: 2–1024 rows 0.69–0.88 (in place 0.71–0.85), 3
  rows 0.88, 17 rows 0.74–0.79, 5 rows 0.84 forward and 1.09 backward, 4096 rows 1.03 (no
  evidence); K ÷ `main`'s forced leaf about 0.85. 32 points 0.93–1.17 (3 rows 1.17): no gain to
  build a rule on.

- Run 20261122-181832-866a (Slurm 515470, 16 rounds, every one started overloaded, band ±1.00
  %), Bluestein limits on `main` ÷ `main`, with the plans they gave:

  | case | `main`'s M | 16 KiB (1024 points): M, time | 8 KiB (512 points): M, time |
  |---|---|---|---|
  | 1013, 1019, 1021 | 2048 | 2160, 0.92–0.93 | 2160, 0.93 |
  | 1018 = 2 × 509 | 2048 (whole line) | two copies of 509 at 1024, **1.37** | 2160 (whole line), 0.94 |
  | 509 | 1024 | 1024, 1.00 | 1080, **1.05** |
  | 131071 | 276480 | 268800, 0.98 | 268800, 0.98 |
  | 999983 | 2073600 | 2073600, 1.00 | 2032128, 0.95 |
  | 2027, 2039, 4093, 10403 | (unchanged) | 0.99–1.00 | 0.99–1.00 |

  A power-of-two part of the L1's size (2048 points) is what loses: half the L1 moves only the
  M of 2048 and 276480 and gains on each. At 1018 it also flips X36b's whole-line rule (the whole
  line's 2160 is longer than two copies' 1024), and the copies, lane passes over s = C = 2 copies,
  cost 1.37 (X36b measured them at 16.3 against 10.0 ns per point).

## Hypothesis

(K) A batch leaf that holds all R inputs and outputs in registers is a copy through the stack once
R passes the register file: at R = 64 on 16 registers the block transposes store all 64 transposed
inputs and outputs and reload them. Moving each element when the codelet asks for it (FFTW's
form, which X53's arm B2 tried before X54 with broadcasts that GCC turned into shuffles) leaves the
codelet's own order (X54's fitted order) to decide the spills. (L) With that leaf, a batch of
64-point rows beats the 4 × 16 chain per row at every batch the screens show up to 1024 rows,
which takes 64 × 2 backward and 64 × 16 under FFTW's time. (C) On Zen 3 a Bluestein convolution
whose power-of-two part is the L1's size (2048 complex doubles) loses to a 7-smooth length a
little longer (X36's effect one step further), so the limit belongs at half the L1; and a stage
over C copies should keep the whole line even when its convolution is the longer, since the
copies' lane passes cost about 1.6 times as much per point.

## Change under test

Arms are commits of branch `exp/x62-epyc-tail` on `main` 021c117 (the base, arm 0):

- **K** (d91de0d): `passes/batch_leaf.inl` on avx2 in double (C = 2), radices 32 and 64 only:
  each input from its two rows by `load_halves` (a 128-bit load and `vinsertf128` from memory),
  each output by `store_halves`; same codelet, same arithmetic per lane, same bits. No rule takes
  these leaves, so no plan changes anywhere: K is judged forced (`BATCH_LEAVES=own`).
- **L** (005864d, on K): the rules take avx2's batch leaf of 64 in double from 2 rows to 1024 rows
  (`batch_rows_avx2_f64`, and a new upper limit `target_profile::batch_max_rows`). Reach: every
  `avx2` double plan (the ISA target and its 16 registers, AGENTS.md section 6), as X53's table;
  no `avx512`, NEON or float plan changes (the `avx512` profile takes no leaf of 64).

- **C** (94ed7f3, on L): on Zen 3's class only (`zen3_like` in `set_grid_sizes`: x86-64, a
  private L2 under 1 MiB, at least 2 MiB of L3 per CPU; Zen 4, Zen 5 and the Ice Lake VMs share
  the Bluestein branch and keep the L1's size), `bluestein_pow2_part_bytes` is half the L1 and
  `bluestein_whole_lanes` (new) lets a stage over s = C copies take the whole line, while the line
  is not long, whatever the two convolutions' lengths. Both precisions, every ISA target that
  machine runs (it has only `avx2` and the older ones; 16 registers). Reach by key: Zen 2 and Zen 3
  machines. One change in purpose (the Bluestein length on Zen 3); the second field exists so
  that the first does not split 1018 into copies.

## Method

- Judging, EPYC 7763 (DMOG, GCC 16.2, `avx2`, one thread), `fbench_remote.py submit --serial`,
  interleaved process rounds, A/A control:
  1. K: catalogue `x62`, filter `^c2c-(64|32)-`, arms 0 and K both with `BATCH_LEAVES=own`.
  2. L: catalogue `x62`, filter `^c2c-64-`, arms 0, L and FFTW.
  3. L and C: the exit check's cases (standard catalogue, filter `^c2c-[0-9]+-b`, tier `full`),
     arms L, 0, C, FFTW (L the baseline: L ÷ 0 is read as the inverse of 0 ÷ L).
  4. L and C: tracking tier, arms L, 0, C.
  5. C: catalogue `x62`, the Bluestein lines, arms L, C, FFTW.
- **The moved cases** of L are the c2c cases of one axis of 64 contiguous points in one batch
  dimension of 2 to 1024 rows, whose plan note reads `batch leaf radix-64` under L and `4 x 16`
  under 0. No other plan can change.
- **Decision rule, written before any judging run:**
  - K (run 1) is adopted if its forced leaf of 64 is faster than 0's forced leaf beyond the band on
    the geometric mean over the 64-point cases, with no 64-point case slower beyond the band. 32 is
    reported, not judged (no rule takes it; it stays in K because it shares the code path, and is
    removed if any 32-point case is slower than 0's forced leaf beyond band + 2 %).
  - L is adopted if (a) over the moved cases of runs 2 and 3 the geometric mean of L ÷ 0 is at most
    0.90 with its interval below 1 − band; (b) no moved case is slower than 1 + band (one re-run,
    `--serial`, decides a case past it; if it stands, the limits move to exclude it before
    adoption); (c) the plan notes of 0 and L agree on every case outside the moved set in runs 3
    and 4; (d) tracking tier, L ÷ 0: geometric mean at most 1 + band and no unmoved case past
    1 + band + 2 % unless its plan is unchanged and its code byte-identical (placement).
  - 64 × 2 backward and 64 × 16 against FFTW are reported, not judged.
  - C, judged against L (*amendment, written after runs 1 and 2 had started and before any run
    of C; runs 3 and 4 were resubmitted with C as a fourth arm, L as the baseline*): run 5,
    catalogue `x62`, its Bluestein lines (c2c double and float, r2c and c2r), arms L, C, FFTW; runs
    3 and 4 with arms L, 0, C (and FFTW in 3). **The moved cases** of C are those whose plan note
    differs between L and C. C is adopted if (a) over the moved cases of runs 3 and 5 the geometric
    mean of C ÷ L is at most 0.97 with its interval below 1 − band; (b) no moved case is slower
    than 1 + band (one `--serial` re-run decides a case past it; if it stands, C is rejected);
    (c) tracking tier (run 4), C ÷ L: geometric mean at most 1 + band and no unmoved case past
    1 + band + 2 % unless its plan is unchanged and its code byte-identical. 1018 against FFTW is
    reported, not judged.
  - The Xeon and the Mac are not run: their default plans cannot change (no `avx2` double plan), and
    K's code is in an `avx2`-region unit that their plans do not call.

*Amendment 2, written after run 1 and before any run of K2.* Run 1 found K's forced leaf of 64
over 5 rows backward at 1.227 of `main`'s forced leaf, every other 64-point case faster, so K
fails its rule as written. Counters (`fbench-fftn run --pmu`, K's build forced, Slurm 515515):
the last row of an odd batch, run in the leaf's form, cost 650–750 cycles after two or more pairs
backward (5, 7, 17 rows) against 280 after one pair (3 rows) and 310 forward, at the same
instruction count and whatever the buffers' page offsets (0/0, 0/2048, 1024/64). Arm **K2**
(6d9f92f, on C): that last row runs in both lanes through the pairs' accessors instead. Same
arithmetic per lane. Rule, judged by run 6 (catalogue `x62`, filter `^c2c-64-`, arms 0, K2, FFTW,
the rules' plans, `--serial`): K2 replaces K if no 64-point case of 2–1024 rows is slower than
0 (the chain) beyond the band, 64 × 5 backward is faster than 0 beyond the band, and the geometric
mean over those cases is at most 0.90. If it does, run 6 also stands for L's rule (a) and (b) on
the moved cases (L with K2's leaf against 0), with run 3 for the exit check's.

*Amendment 3, written after run 6 and before any run of K3.* Run 6 (below) rejects K2: 64 × 3 and
64 × 5 backward stayed slower than the chain (1.09, 1.22). Op-cache counters on K2's build (Slurm
515591: PMCx28F hits and misses) found 190–300 misses a call at 5, 7 and 17 rows backward and
none at 3 or 4 rows or forward: the pairs' inlined copy of the 64-point codelet and the last
row's second copy evict each other, as their addresses fall. Arm **K3** (dcaaeef, on K2): every
group, the last row included (in both lanes), runs through one inlined copy. Same rule as K2's,
judged by run 7 (catalogue `x62`, filter `^c2c-64-`, arms 0, K3, FFTW, `--serial`); if K3 passes,
run 7 stands for L's rule (a) and (b) on the moved cases (L with K3's leaf against 0).

## Results

EPYC 7763 (DMOG node16, shared, `-c 8`), GCC 16.2, `avx2`, one thread, `--serial`; every round
under load. Ratios are fbench's median paired ratios with 95 % intervals.

### Run 1: K forced (20261122-183544-2e14, Slurm 515490, 12 rounds, band ±1.00 %)

K ÷ 0 with `BATCH_LEAVES=own` on both: **0.890 [0.886, 0.896]** over the 44 cases. 64 points:
0.81–0.91 at every count from 2 to 4096 rows, in place 0.87–0.91, odd counts 0.81–0.97, except
**64 × 5 backward 1.227 [1.222, 1.230]** (the screen's 1.09 again). 32 points 0.88–0.97. So K
misses its rule on one case: amendment 2 and arm K2.

### Run 2: L (20261122-183953-183e, Slurm 515493, 24 rounds, every one started overloaded,
other processes 103 cores median, band ±1.00 %)

L ÷ 0 at 64 points: even counts 2–1024 rows 0.76–0.96 (64 × 2 backward 0.765, 64 × 16 0.794),
in place 0.82; 4096 rows unmoved 1.00. **Odd counts slower: 3 rows 1.34–1.42, 5 rows 1.21–1.39**,
17 rows 0.90–0.91. The odd counts are the leaf's last row (amendment 2), much worse on this
saturated node than in run 1. By L's rule (b) these cases go to the re-run, which amendment 2
makes run 6 (K2's leaf).

### Run 3: the exit check's cases, L, 0 and C (20261122-203524-faba, Slurm 515537, 15 rounds, every
one started overloaded, other processes 91 cores median, band ±1.73 %)

The first submission (Slurm 515496) was cancelled before it started, to add arm C; the second
(515503) after its first round, to cap the run at 16 rounds.

- L moves 64 × 2 backward (L ÷ 0 0.772), 64 × 16 (0.792), 64 × 256 (0.885); every other plan
  note agrees. Geometric mean over the 201 cases, 0 ÷ L 1.002 [0.993, 1.008].
- C moves 1018 (C ÷ L 0.933 [0.929, 0.942]) and 131071 (0.981 [0.965, 0.986]); every other plan
  note agrees with L's. C ÷ L over the 201 cases 1.001 [0.995, 1.006].
- Unmoved cases that differ between the three binaries are tiny or long rows whose ratios
  scatter in this run: the row of 2 points is 0.84 of 0's time and of C's in L's binary
  (intervals [1.005, 1.224] and [0.997, 1.206]), 65537 1.08 [0.85, 1.28] in C's.
- Against FFTW (DESIGN 1.3, 193 c2c rows, band ±1.73 %): **L 0.77 with 4 beyond noise** (2048
  1.11, 1024 backward 1.09, 20000 1.09, 120 1.08), against 9 for 0 and 4 for C in the same run
  (medians; those two arms are not paired with FFTW by fbench): 64 × 2 backward 1.28 → 0.98, 64 ×
  16 1.15 → 0.91, 1018 1.05 → 0.98 (C). Still beyond noise: 2048, 1024 backward, 120, 20000.

### Run 4: tracking tier, L and C (20261122-191204-85fb, Slurm 515505, 24 rounds, band ±1.16 %)

- L ÷ 0: **0.996** (0 ÷ L 1.004 [1.002, 1.008]); one case moved, 64 × 256 at 0.876; every other
  plan note agrees; no unmoved case past ±1.3 %.
- C ÷ L: **1.002 [0.998, 1.006]**; no plan note differs (the tier has no Bluestein line); the
  highest unmoved case 1.017 (65537), inside 1 + band + 2 %.

### Run 5: C on the Bluestein lines (20261122-193757-2397, Slurm 515507, 24 rounds, every one
started overloaded, band ±1.00 %)

C ÷ L over the 12 moved cases (plan notes differ): **0.937**, every one faster, none slower:

| case | L's M → C's M | C ÷ L | FFTN ÷ FFTW, L → C |
|---|---|---|---|
| 1018 (× 1, × 3, × 16) | 2048 → 2160 (whole line) | 0.929–0.936 | 1.044–1.047 → 0.971–0.980 |
| 1013, 1019, 1021 | 2048 → 2160 | 0.927–0.937 | |
| 131071 | 276480 → 268800 | 0.988 [0.983, 0.993] | 0.797 → 0.789 |
| float 2036 (4 × 509, s = C = 4) | 4096 → 4320 | 0.922 | |
| r2c and c2r 2036, 2026 (half lengths 1018, 1013) | 2048 → 2160 | 0.935–0.941 | |

Unmoved and level: 509, 2027, 2039, 4093, 10403, 999983, float 1018 and 1013 (0.991–1.003).

### Run 6: K2 (20261122-200901-fd20, Slurm 515528, 24 rounds, every one started overloaded,
band ±1.07 %)

K2 ÷ 0 (the rules' plans): 0.858 [0.855, 0.862] over the 32 cases; even counts from 2 to 1024
rows 0.74–0.97, in place 0.80–0.97, 17 rows 0.85–0.89, 3 and 5 rows forward 0.91–0.96, but
**3 rows backward 1.092 and 5 rows backward 1.215**: K2 fails its rule (amendment 3).

### Run 7: K3 (20261122-213538-371a, Slurm 515596, 24 rounds, every one started overloaded,
other processes 114 cores median, band ±1.00 %)

K3 ÷ 0 (the rules' plans): **0.848 [0.845, 0.851]** over the 32 cases. Even counts from 2 to
1024 rows 0.76–0.93, in place 0.79–0.82, 17 rows 0.81–0.84, **5 rows 0.938 backward and 0.964
forward**, 4096 rows unmoved (1.00). **3 rows 1.050 forward, 1.048 in place, 1.011 backward**: the
last row is now a whole pair, which three rows do not repay. Op-cache counters on K3's build
(Slurm 515598): no misses at any count (5 rows backward 1220 → 958 cycles a call, 7 rows 1638 →
1268, 17 rows 3222 → 2922).

FFTN ÷ FFTW in this run, 0 → K3 with L's rule: 64 × 2 backward 1.29 → 0.98, 64 × 16 1.16 → 0.92,
64 × 256 1.07 → 0.91, 64 × 2 in place 1.28 → 1.01, 64 × 16 in place 1.26 → 1.03.

### The limit move

64 × 3 was slower than the chain in runs 2 (1.34–1.42 with K's leaf), 6 (1.09 backward with
K2's) and 7 (1.05 forward, 1.05 in place with K3's). By L's rule (b) the limits move to exclude
it: commit 427ce39 adds `target_profile::batch_ragged_rows`, from which a batch that C does not
divide takes the leaves (avx2 double, 64: 5 rows), so 64 × 3 keeps the 4 × 16 chain, the plan of
arm 0. No other case changes plan.

### Confirmation on the branch head

Run on 427ce39 (K3, L with the limits, C) against 0: see "Not done" if it is missing.

## Verdict

- **K: rejected** (run 1): 64 × 5 backward 1.227 of `main`'s forced leaf. **K2: rejected** (run 6):
  3 and 5 rows backward 1.09 and 1.22 of the chain. Both were steps to K3, kept in the history.
- **K3 with L: adopted**, with the limits moved as L's rule (b) provides (odd batches from 5
  rows). Over the cases the moved limits leave to the leaf (64 points, 2 and 4–1024 rows, odd from
  5), run 7 gives none slower than the chain (the slowest 0.964), 64 × 5 backward 0.938 (faster
  beyond the band) and a geometric mean of 0.820 over those 27 cases (0.848 over all 32, 64 × 3
  and 4096 rows included): K3's rule (amendment 3) on those cases, and L's (a) and (b) with run 3
  (64 × 2 backward 0.772, 64 × 16 0.792, 64 × 256 0.885). L's (c): only those plans move in runs 3
  and 4. L's (d): tracking tier 0.996, the one moved case 0.876, no unmoved case past ±1.3 % (run
  4). The case for the maintainer: K3's rule as written also covered 64 × 3, which it missed
  (1.05); it is adopted on the cases L's limits keep, under L's clause, not on the letter of K3's
  rule.
- **C: adopted** (runs 3, 4, 5): over its 14 moved case results C ÷ L 0.92–0.99 (geometric mean
  0.94), none slower; tracking tier 1.002 with no plan moved. 1018 ÷ FFTW 1.05 → 0.98.
- **2048: kernel cost, not moved.** The same three-pass plan; FFTW's two-pass 32 → 64 plan has no
  FFTN counterpart that wins (X54, X57). `LAYOUT=interleaved` gave no evidence (screen).
- **72, 100, 120: code placement, left.** Same plans as since X38, ratios 1.07–1.09 that move
  together and with the build (X57: the same functions aligned to 64 bytes took 72 and 120 to
  0.88 and 0.93 of their time); no arm here touches their code, and run 3 shows them unchanged by
  every arm (0.994–1.025).
- **15 × 2 backward: not a loss** (0.89–0.90 of FFTW in the re-time and run 3).
- **1024 backward** (1.06–1.09 of FFTW): the 16 × 64 plan of X57, unchanged; inside or near the
  band in every run, not examined further.
- The EPYC against FFTW over the exit check's 193 rows, run 3: L 4 cases beyond noise (2048,
  1024 backward, 20000, 120) where `main` had 9 in the same run; with C, 1018 leaves the list too.
- Lessons 4.14 (addendum) and 4.24 in `dev/LESSONS.md`; DESIGN sections 8.2, 8.5 and 11.2 and the
  decision log updated.

## Not done

- The Xeon and the Mac were not run: no default plan there changes (K3's code is reached only
  from `avx2` double plans of 32 or 64 points; L's and C's rules are keyed to `avx2` double and to
  Zen 3's profile).
- C reaches Zen 2 by its key, unmeasured; Zen 4 and 5 keep the L1's limit, unmeasured with half.
- The batch leaf of 32 (K3's code too) stays untaken: forced, it ran 0.88–0.97 of `main`'s forced
  leaf but 0.93–1.17 of the planner's plans in the screen.
- Float's leaf of 64 on avx2 (C = 4) keeps the block transposes.
