# X56: the phase 1 exit check on the code X53 brings to `main` (71307e0)

- **Status:** running (local machines done: the Mac and the Xeon pass; the EPYC meets its mean
  target and misses the tail rule against FFTW, 11 of 193 where 9 are allowed; the cloud
  section is still to come)
- **Owner:** exit-final session, 2026-11-18
- **Phase / task:** the phase 1 exit ([PLAN.md](../PLAN.md) "Status", "Phase 1 exit check
  (2026-11-18 evening)"); repeats the local part of the check of 2026-11-18 on `main` 8c18dd2
  and [X48](X48-exit-3.md)'s cloud part, after X49–X55

## Question

Does the code `main` holds once X53 has merged meet the phase 1 exit (DESIGN.md 1.3: the
geometric-mean target, and at most 5 % of cases slower than 1.05 of the reference beyond
noise), and which cases still fail?

## Hypothesis

The means stay met. X49 (fbench's backend), X50 (primes), X51 and X55 (depth-first chains), X53
(two-row leaves) and X54 (radix-32/64 order) each aimed at cases on the earlier list, so the
lists should shrink. The estimate made when X53 was handed over was about 9 of 193 on the EPYC
against FFTW, exactly at the limit.

## Change under test

None (a measurement). Commit 71307e0, the head of `exp/x53b-batch-lengths` (pull request 74,
which contains pull request 68). Its library code is what `main` becomes once X53 merges: `main`
664e377 has only `dev/PLAN.md` beyond the merge base 37f386b.

Arms, fbench Release builds at 71307e0: `N` = FFTN with the identical-arm control `N~`; `F` =
FFTW 3.3.11 with its SIMD codelets, MEASURE, fresh wisdom (the NEON build on the Mac); `L` =
FFTL.jl's library on the Mac and the EPYC; `M` = oneMKL on the Xeon. The references per machine
are those of the earlier check.

## Method

- **Cases:** `--catalogue standard --tier full --filter '^c2c-[0-9]+-b'`: 201 c2c rows, 193 of
  them double rows that every arm timed. One thread, hot, median statistic. X49's backend fix is
  in this commit, so no separate re-time of the tiny cases is needed.
- **Machines:** Mac M5 Max (Clang 23.1.4, `neon_fcma`), EPYC 7763 on DMOG (GCC 16.2.0, `avx2`,
  Slurm, 8 CPUs of a shared node), Xeon Gold 6240 (GCC 14.4.0, `avx512`, MKL 2026.0.1, Slurm
  `--serial`, 8 CPUs of the shared node).
- **Metric:** FFTN ÷ reference over the 193 rows: geometric mean, and the cases whose whole
  bootstrap interval lies above 1.05 × (1 + the run's A/A band).
- **Decision rule** (DESIGN 1.3, fixed before these runs; this file was written after them): a
  machine passes against a reference when the mean is at or below its target and at most 9 of
  the 193 cases count beyond noise.
- **Mac validity:** the run started on AC power with no other fbench running; a watcher checked
  the power source every 15 s and would have stopped the run on battery or on a gap (sleep).

## Results

### Local machines

| Machine | Run | Valid rounds | Under load | Other processes (median) | Band (A/A) |
|---|---|---|---|---|---|
| Mac | 20261118-194015-01cf | 53 | 34 of 53 | 1.6 cores | ±1.00 % |
| EPYC | 20261118-194340-02a4 | 28 | 28 of 28 | 48 cores of 128 | ±1.00 % |
| Xeon | 20261118-194333-fbac | 21 | 4 of 21 | 0.5 cores (33 at most) | ±1.91 % |

The Mac stayed on AC power from 19:40 to 20:42 UTC with no sleep or wake in the power log. Slurm
jobs: EPYC build 513012, timing 513013; Xeon build 16532, timing 16533.

The earlier figures in brackets are the check on `main` 8c18dd2 with its tiny cases replaced by
their re-times on 4d75c48 (each case judged against the band of its own run).

| Machine | Reference | Geomean (target) | Beyond noise (limit 9) | Plain > 1.05 |
|---|---|---|---|---|
| Mac | FFTW | **0.737** (0.85) [0.749] | **3, 1.6 %** [3] | 11 |
| Mac | FFTL | **0.522** (0.95) [0.525] | **4, 2.1 %** [4] | 6 |
| EPYC | FFTW | **0.776** (0.90) [0.793] | **11, 5.7 %** [12] | 13 |
| EPYC | FFTL | **0.451** [0.450] | **2, 1.0 %** [1] | 8 |
| Xeon | FFTW | **0.658** (0.85) [0.689] | **1, 0.5 %** [5] | 2 |
| Xeon | MKL | **0.688** (1.00) [0.694] | **2, 1.0 %** [2] | 9 |

Cases beyond noise (FFTN ÷ reference, the earlier value in brackets):

- **Mac ÷ FFTW (3):** 2 × 3 in place 1.42 [1.36], 2²⁴ 1.16 [1.10, new], 7 × 3 in place 1.09
  [1.09, new: the earlier band was wider]. Out: 32 × 4096 (1.10 → 1.06), 64 × 2 backward
  (1.07 → 0.95).
- **Mac ÷ FFTL (4):** 2²³ 1.92 [1.56], 2²⁴ 1.85 [1.68], 1024 × 1024 1.16 [1.17], 512 × 1024
  1.08 [1.09]; the same four cases.
- **EPYC ÷ FFTW (11):** 64 × 2 backward 1.28 [1.28], 1024 × 3 1.22 [1.22], 1024 1.21 [1.21],
  1024 backward 1.18 [1.20], 64 × 16 1.14 [1.14], 1024 × 16 1.12 [1.13], 1018 1.10 [1.10], 2048
  1.10 [1.11], 100 1.07 [1.09]; new: 120 1.08 [1.02], 1024 × 256 1.08 [1.06]. Out: 15 × 2
  backward (1.23 → 0.91), 768 (1.09 → 0.97), 512 (1.07 → 1.03).
- **EPYC ÷ FFTL (2):** 16807 1.15 [1.08]; new: 65536 × 3 in place 1.19 [0.98].
- **Xeon ÷ FFTW (1):** 160 1.12 [1.16]. Out: 131071 (1.29 → 0.80), 25000 (1.25 → 0.99), 15 × 2
  backward (1.16 → 0.90), 240 (1.13 → 1.02).
- **Xeon ÷ MKL (2):** new: 512 × 16 1.15 [1.16, inside the earlier run's wider interval], 1024 × 3
  in place 1.10 [1.08]. Out: 999983 (1.41 → 1.04), 131071 (1.32 → 0.83).

### Against the earlier check

These compare separate runs under different load, so they locate questions and prove nothing
(AGENTS.md section 6). Geometric mean of each arm's time, this run ÷ the earlier one, over the
193 rows:

| Machine | FFTN | FFTW | Third reference |
|---|---|---|---|
| Mac | 0.910 | 0.925 | FFTL 0.914 |
| EPYC | 0.971 | 0.992 | FFTL 0.969 |
| Xeon | 0.916 | 0.959 | MKL 0.924 |

- **Mac:** every arm is about 8–9 % faster; the earlier run was loaded (44 of 46 rounds, other
  processes up to 16 cores), so most of this is the quieter machine. 2²³ and 2²⁴ rose against
  both references while FFTN itself got faster (2²³ 9.31 → 8.37 ns per point): FFTL went 5.88 →
  4.35 and FFTW 10.00 → 6.98.
- **EPYC:** 120 (1.154 → 1.221 ns per point) and 72 (1.162 → 1.250) are slower with FFTW
  unchanged (1.124 → 1.120, 1.173 → 1.179). This looks like a change in FFTN between 8c18dd2 and
  71307e0; the cause is not known and needs a same-run A/B. 72 is 1.06 of FFTW, within noise. The
  65536 family's rise against FFTL is FFTL's: FFTN 4.83 → 4.73, FFTL 5.03 → 3.99 ns per point
  (× 3 in place).
- **Xeon:** X50 shows in the primes (131071 43.8 → 25.7, 999983 84.7 → 57.6 ns per point) and X51
  in 25000 (4.16 → 2.98). Part of the mean's gain against FFTW is FFTW's own: its MEASURE plans
  differ between runs (97 points 13.9 → 26.8 ns per point, 128 × 3 1.70 → 0.91), which is why the
  MKL column is the cleaner comparison here.

### Cloud

To be added.

## Verdict

Measurement only; no library change. Local machines at 71307e0: every mean target is met. The
tail rule is met on the **Mac** (FFTW 1.6 %, FFTL 2.1 %) and the **Xeon** (FFTW 0.5 %, MKL 1.0 %)
and against FFTL on the EPYC (1.0 %). **The EPYC fails it against FFTW: 11 of 193 (5.7 %), two
cases over the limit**, against the 9 estimated.

What stands in the way on the EPYC:

1. **1024 in five forms** (single, backward, × 3, × 16, × 256; 1.08–1.22): the radix choice that
   sees the data size, left open by [X54](X54-radix64-avx2.md).
2. **64 × 2 backward 1.28:** X53 took 15 × 2 backward under FFTW, not this one.
3. **64 × 16 1.14, 2048 1.10, 1018 1.10, 100 1.07.**
4. **120 1.08,** newly over, with the same-run question above.

Not done: the same-run A/B of 8c18dd2 against 71307e0 on the EPYC's 72 and 120; the cloud part.
