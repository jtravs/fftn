# X56: the phase 1 exit check on `main` after X49–X57

- **Status:** done: every mean target and the tail rule are met on every machine checked, the
  three local ones and six cloud cores (Ice Lake after X58)
- **Owner:** exit-final session, 2026-11-18 and 19
- **Phase / task:** the phase 1 exit ([PLAN.md](../PLAN.md) "Status", "Phase 1 exit check
  (2026-11-18 evening)"); repeats the local part of the check of 2026-11-18 on `main` 8c18dd2
  and [X48](X48-exit-3.md)'s cloud part, after X49–X57

## Question

Does `main` after X53 and X57 meet the phase 1 exit (DESIGN.md 1.3: the geometric-mean target,
and at most 5 % of cases slower than 1.05 of the reference beyond noise), and which cases still
fail?

## Hypothesis

The means stay met. X49 (fbench's backend), X50 (primes), X51 and X55 (depth-first chains), X53
(two-row leaves) and X54 (radix-32/64 order) each aimed at cases on the earlier list, so the
lists should shrink. When X53 was handed over the estimate for the EPYC against FFTW was about
9 of 193, at the limit; it measured 11, and [X57](X57-epyc-1024.md) then predicted 4 or 5 from
its own run.

## Change under test

None (a measurement). Two commits, because `main` moved during the check:

- **EPYC: `main` fa155c9** (X53 and X57 merged).
- **Mac and Xeon: 71307e0**, the head of X53's branch. Its library, generator and benchmark
  sources are identical to those of X53's merge 0508037. The only library change from there to
  fa155c9 is X57, whose fit overrides exist for `avx2` double alone, so `neon_fcma` and `avx512`
  plans are the same (`fit_overrides_of`, `src/planner/rules.cpp`). These two machines were not
  re-timed; the binaries differ from fa155c9's, so code placement can differ.

Arms, fbench Release builds at those commits: `N` = FFTN with the identical-arm control `N~`;
`F` = FFTW 3.3.11 with its SIMD codelets, MEASURE, fresh wisdom (the NEON build on the Mac); `L`
= FFTL.jl's library on the Mac and the EPYC; `M` = oneMKL on the Xeon. The references per machine
are those of the earlier check.

## Method

- **Cases:** `--catalogue standard --tier full --filter '^c2c-[0-9]+-b'`: 201 c2c rows, 193 of
  them double rows that every arm timed. One thread, hot, median statistic. X49's backend fix is
  in these commits, so no separate re-time of the tiny cases is needed.
- **Machines:** Mac M5 Max (Clang 23.1.4, `neon_fcma`), EPYC 7763 on DMOG (GCC 16.2.0, `avx2`,
  Slurm, 8 CPUs of a shared node), Xeon Gold 6240 (GCC 14.4.0, `avx512`, MKL 2026.0.1, Slurm
  `--serial`, 8 CPUs of the shared node).
- **Metric:** FFTN ÷ reference over the 193 rows: geometric mean, and the cases whose whole
  bootstrap interval lies above 1.05 × (1 + the run's A/A band).
- **Decision rule** (DESIGN 1.3, fixed before these runs; this file was written after the first
  three): a machine passes against a reference when the mean is at or below its target and at
  most 9 of the 193 cases count beyond noise.
- **Mac validity:** the run started on AC power with no other fbench running; a watcher checked
  the power source every 15 s and would have stopped the run on battery or on a gap (sleep).

## Results

### Local machines

| Machine | Commit | Run | Valid rounds | Under load | Other processes (median) | Band (A/A) |
|---|---|---|---|---|---|---|
| EPYC | fa155c9 | 20261119-005944-cb48 | 26 | 26 of 26 | 40 cores of 128 | ±1.67 % |
| Mac | 71307e0 | 20261118-194015-01cf | 53 | 34 of 53 | 1.6 cores | ±1.00 % |
| Xeon | 71307e0 | 20261118-194333-fbac | 21 | 4 of 21 | 0.5 cores (33 at most) | ±1.91 % |

The Mac stayed on AC power from 19:40 to 20:42 UTC with no sleep or wake in the power log. Slurm
jobs: EPYC build 513180, timing 513181 (node17); Xeon build 16532, timing 16533.

The earlier figures in brackets are the check on `main` 8c18dd2 with its tiny cases replaced by
their re-times on 4d75c48 (each case judged against the band of its own run).

| Machine | Reference | Geomean (target) | Beyond noise (limit 9) | Plain > 1.05 |
|---|---|---|---|---|
| EPYC | FFTW | **0.788** (0.90) [0.793] | **6, 3.1 %** [12] | 11 |
| EPYC | FFTL | **0.456** [0.450] | **1, 0.5 %** [1] | 7 |
| Mac | FFTW | **0.737** (0.85) [0.749] | **3, 1.6 %** [3] | 11 |
| Mac | FFTL | **0.522** (0.95) [0.525] | **4, 2.1 %** [4] | 6 |
| Xeon | FFTW | **0.658** (0.85) [0.689] | **1, 0.5 %** [5] | 2 |
| Xeon | MKL | **0.688** (1.00) [0.694] | **2, 1.0 %** [2] | 9 |

The EPYC's band is wider than the 1.00 % of its run on 71307e0, and a wide band is lenient (DESIGN
1.3). At the 1.00 % floor it would count 8 cases against FFTW (adding 1024 backward and 100) and
2 against FFTL (adding 65536 in place 1.11): the verdict is the same.

Cases beyond noise (FFTN ÷ reference, the earlier value in brackets):

- **EPYC ÷ FFTW (6):** 64 × 2 backward 1.30 [1.28], 64 × 16 1.13 [1.14], 15 × 2 backward 1.13
  [1.23], 1018 1.10 [1.10], 2048 1.09 [1.11]; new: 120 1.08 [1.02]. Out: 1024 (1.21 → 1.01), 1024 × 3
  (1.22 → 1.02), 1024 × 16 (1.13 → 1.01), 768 (1.09 → 0.97), 512 (1.07 → 1.03), and within this
  run's band 1024 backward (1.20 → 1.08) and 100 (1.09 → 1.07).
- **EPYC ÷ FFTL (1):** 16807 1.13 [1.08].
- **Mac ÷ FFTW (3):** 2 × 3 in place 1.42 [1.36], 2²⁴ 1.16 [1.10, new], 7 × 3 in place 1.09
  [1.09, new: the earlier band was wider]. Out: 32 × 4096 (1.10 → 1.06), 64 × 2 backward
  (1.07 → 0.95).
- **Mac ÷ FFTL (4):** 2²³ 1.92 [1.56], 2²⁴ 1.85 [1.68], 1024 × 1024 1.16 [1.17], 512 × 1024
  1.08 [1.09]; the same four cases.
- **Xeon ÷ FFTW (1):** 160 1.12 [1.16]. Out: 131071 (1.29 → 0.80), 25000 (1.25 → 0.99), 15 × 2
  backward (1.16 → 0.90), 240 (1.13 → 1.02).
- **Xeon ÷ MKL (2):** new: 512 × 16 1.15 [1.16, inside the earlier run's wider interval], 1024 × 3
  in place 1.10 [1.08]. Out: 999983 (1.41 → 1.04), 131071 (1.32 → 0.83).

### The EPYC before X57 (71307e0)

Run 20261118-194340-02a4 (Slurm 513012, 513013, node21; 28 rounds, all under load, other
processes 48 cores, band ±1.00 %): ÷ FFTW 0.776 with **11 beyond noise (5.7 %, over the limit)**:
64 × 2 backward 1.28, 1024 × 3 1.22, 1024 1.21, 1024 backward 1.18, 64 × 16 1.14, 1024 × 16 1.12,
1018 1.10, 2048 1.10, 120 1.08, 1024 × 256 1.08, 100 1.07; ÷ FFTL 0.451 with 2 (65536 × 3 in place
1.19, 16807 1.15). That result led to X57.

Between the two runs FFTN's 1024 went 1.796 → 1.508 ns per point and 1024 × 3 1.840 → 1.550, with
FFTW unchanged (1.496 → 1.499, 1.512 → 1.517).

### Against the earlier check

These compare separate runs under different load, so they locate questions and prove nothing
(AGENTS.md section 6). Geometric mean of each arm's time, this run ÷ the earlier one, over the
193 rows:

| Machine | FFTN | FFTW | Third reference |
|---|---|---|---|
| EPYC | 0.988 | 0.994 | FFTL 0.976 |
| Mac | 0.910 | 0.925 | FFTL 0.914 |
| Xeon | 0.916 | 0.959 | MKL 0.924 |

- **EPYC, tiny cases:** FFTN's rows of a few points scattered from round to round on this node
  while FFTW's did not. 15 × 2 backward keeps its plan (the batch leaf) yet went from 0.91 of
  FFTW on 71307e0 to 1.13: its rounds run 0.78–1.27 ns per point (0.77–1.11 on 71307e0), the
  identical-arm control shows the same spread, and FFTW stays near 0.88. 2, 10 and 26 points
  behave alike, which raises the mean a little over 71307e0's 0.776. The cause was not traced;
  the plans are the same, and the fastest rounds match 71307e0's.
- **EPYC, 72 and 120:** slower than at 8c18dd2 with FFTW unchanged; X57 found this to be code
  placement. The 65536 family's rise against FFTL since 8c18dd2 is FFTL's (× 3 in place: FFTN 4.83
  → 4.73 on 71307e0, FFTL 5.03 → 3.99 ns per point).
- **Mac:** every arm is about 8–9 % faster; the earlier run was loaded (44 of 46 rounds, other
  processes up to 16 cores), so most of this is the quieter machine. 2²³ and 2²⁴ rose against
  both references while FFTN itself got faster (2²³ 9.31 → 8.37 ns per point): FFTL went 5.88 →
  4.35 and FFTW 10.00 → 6.98.
- **Xeon:** X50 shows in the primes (131071 43.8 → 25.7, 999983 84.7 → 57.6 ns per point) and X51
  in 25000 (4.16 → 2.98). Part of the mean's gain against FFTW is FFTW's own: its MEASURE plans
  differ between runs (97 points 13.9 → 26.8 ns per point, 128 × 3 1.70 → 0.91), which is why the
  MKL column is the cleaner comparison here.

### Cloud

Google Cloud Spot VMs in us-central1-a, one thread per core, GCC 14.2, FFTW 3.3.11 built with
every SIMD option, oneMKL 2026.0.2 (wheel) on Intel; standard catalogue, `^c2c-[0-9]+-b`, 193
rows, every arm pinned to CPU 2, at most one round in twelve under load. The lead ran these:
sub-agents cannot create cloud resources. Ledger: [cloud-ledger-2026-11.md](cloud-ledger-2026-11.md),
"X56 and X58" (USD 7.07 for both experiments; the campaign at USD 12.19 of 20).

First round at 71307e0 (X53's code; X57 changes `avx2` double only, which none of these cores runs):

| Core (VM) | Run | Band | ÷ FFTW: geomean, beyond noise | ÷ oneMKL |
|---|---|---|---|---|
| Zen 5, EPYC 9B45 (`c4d-standard-16`) | 20261118-205528-8c25 | ±1.07 % | 0.67, 5 (2.6 %): 2^24 1.18, 100000 1.16, 2048 1.13, 2^23 1.12, 4096 1.09 | — |
| Axion, Neoverse V2 (`c4a-standard-16`) | 20261118-194313-91a3 | ±1.00 % | 0.67, 1 (0.5 %): 65537 1.08 | — |
| Granite Rapids, Xeon 6985P-C (`c4-standard-16`) | 20261118-194721-9d5d | ±2.11 % | 0.71, 9 (4.7 %) | 0.69, 5 (2.6 %) |
| Sapphire Rapids, Xeon 8481C (`c3-standard-8`) | 20261118-195508-bd24 | ±1.99 % | 0.70, 8 (4.1 %) | 0.69, 7 (3.6 %) |
| Ice Lake (`n2-standard-16`) | 20261118-194932-98a7 | ±3.31 % | 0.71, **12 (6.2 %)**: long rows 1.14–1.51 | 0.74, **14 (7.3 %)** |

Zen 5's first VM was preempted after 54 minutes and replaced. Ice Lake failed: its VM reports a
large L3 per CPU, so the rules filed it with the EPYC and it missed the depth-first chain. The
same round showed X53's batch leaf of 12 losing on Granite and Sapphire Rapids (12 × 64 at
1.18–1.22 of FFTW). [X58](X58-intel-keys.md) gave both their own keys, and the three Intel cores
were checked again on its merge candidate 11817ba (library as `main` 602d0af):

| Core | Run | Band | ÷ FFTW | ÷ oneMKL |
|---|---|---|---|---|
| Ice Lake | 20261119-024441-0417 | ±4.23 % | 0.67, **0** | 0.71, 4 (2.1 %): 999983 1.43, 131071 1.23, 10^6 1.22, 2^23 1.13 |
| Sapphire Rapids | 20261119-024650-416a | ±2.36 % | 0.70, 7 (3.6 %): 131071 1.20, 1024 backward 1.18, 160 1.15, 240 1.11, 384 1.10, 15 × 2 backward 1.10, 1024 1.09 | 0.69, 9 (4.7 %, at the limit): 999983 1.61, 131071 1.23, 1024 × 3 in place 1.20, 1024 in place 1.16, 32 × 256 1.15, 1024 × 16 1.13, 32 × 16 1.13, 1024 × 3 1.12, 1000 in place 1.11 |
| Granite Rapids | 20261119-021835-b712 | ±2.41 % | 0.72, 7 (3.6 %): 131071 1.32, 160 1.26, 240 1.15, 1024 backward 1.14, 1024 1.14, 1024 × 16 1.11, 1024 × 3 1.10 | 0.69, 4 (2.1 %): 131071 1.34, 999983 1.30, 1024 × 3 in place 1.17, 1024 × 16 1.11 |

The `avx512` branch padding (PR 62) was timed against a build without it on the same commit
(rows of 32–2048 points): Granite Rapids 1.000 [0.996, 1.002] (run 20261118-201122-73d9),
Sapphire Rapids 0.999 (20261118-202454-3c31), Ice Lake 1.003 (20261118-202315-75ad), Zen 5
1.037 [1.028, 1.044] with 240 points 1.35 slower without it (20261118-211132-2bd7). The padding
stays: free on recent Intel, a gain on Zen 5 by placement.

## Verdict

Measurement only; no library change. On the local machines every mean target is met and **the
tail rule is met on all three**: EPYC at fa155c9 (FFTW 3.1 %, FFTL 0.5 %; 8 and 2 cases at the
1.00 % band floor), Mac (FFTW 1.6 %, FFTL 2.1 %) and Xeon (FFTW 0.5 %, MKL 1.0 %) at X53's code.
On the cloud cores too: Zen 5, Axion and Granite Rapids at X53's code, and after X58 Ice Lake
(FFTW 0 cases, oneMKL 2.1 %), Sapphire Rapids (3.6 %, oneMKL 4.7 %) and Granite Rapids (3.6 %,
2.1 %). **The phase 1 exit is met** on all nine machines checked.

What is left beyond noise on the EPYC against FFTW: 64 × 2 backward 1.30, 64 × 16 1.13, 15 × 2
backward 1.13 (0.91 in the run before, see above), 1018 1.10, 2048 1.09, and 120 1.08 (code
placement, X57).

Not done: the Mac and the Xeon were not re-timed on fa155c9 (their plans are unchanged); the
scatter of the EPYC's tiny cases was not traced; the cloud cores were not re-timed after X57
(no `avx2` double there) or, apart from the three Intel ones, after X58; Emerald Rapids and Zen 4
were not checked. Cases common to several cores and worth an experiment: 131071 and 999983
against oneMKL (X50's convolutions are off outside the Xeon 6240 and Apple), 160 and 240 on
recent Intel, the 1024 family on recent Intel against both references.
