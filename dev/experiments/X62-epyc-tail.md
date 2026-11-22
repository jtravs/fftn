# X62: the EPYC's remaining phase 1 exit cases

- **Status:** running
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

## Results

## Verdict
