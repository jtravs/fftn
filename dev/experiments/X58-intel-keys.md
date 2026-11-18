# X58: profile keys for recent Intel (Ice Lake VMs, batch leaves of 12)

- **Status:** running
- **Owner:** lead, 2026-11-18
- **Phase / task:** 1 (phase 1 exit check); follows X53, X55 and X56

## Question

X56 screened the Intel families the rules reach. Two results call for keys of their own
(AGENTS section 6, "a family that loses gets its own key"):

1. An Ice Lake VM (Google Cloud N2, private L2 1.25 MiB, 6.9 MiB of L3 per CPU) fails the exit
   check: 12 of 193 cases beyond noise against FFTW, 14 against oneMKL, nearly all rows of
   32768 points and more. Its L3 per CPU puts it in the large-L3 branch of `set_grid_sizes`,
   which has no depth-first chain (the Xeon 6240's branch needs little L3 per CPU).
2. X53's batch leaf of 12 (`avx512`, double, from 8 rows) loses on Sapphire Rapids: without
   batch leaves 12 × 64 takes 0.857 and 12 × 1024 0.926 of the time. It wins on Ice Lake (1.233
   and 1.076 without) and on the Xeon 6240 (0.954 and 0.902 with).

## Hypothesis

1. The depth-first rule with the Xeon 6240's tops closes Ice Lake's long rows: forced tops of
   16 took 0.50–0.68 of the time from 32768 to 2^22 points, and 20 took 0.49 at 100000.
2. Leaving out the batch leaf of 12 where the private L2 is 2 MiB or more recovers 12 × 64 and
   12 × 1024 on Sapphire and Granite Rapids.

## Change under test

**G**, two profile changes in `set_grid_sizes`:

- x86-64 with a private L2 above 1 MiB and below 2 MiB: depth-first chain in double from L2/4
  to 64 L2, tops 16, 20 and 10. Zen 4 and Zen 5 (exactly 1 MiB) stay out.
- x86-64 with a private L2 of 2 MiB or more: `machine_profile::batch_leaf_12` false.

## Method

Baseline N: `main` 0508037. Spot VMs, GCC 14.2, `avx512`, every arm pinned to CPU 2, FFTW and
oneMKL as in X55. Ice Lake (N2): the rows of 16384 points and more, then the tracking tier.
Sapphire Rapids (C3) and Granite Rapids (C4): c2c rows of 12 points, then the tracking tier.

## Decision rule (written before any judging run)

Per change and machine family, as in X46 and AGENTS section 6, over the cases whose plan the
change alters: G is adopted if the weighted geomean of G ÷ N is below 1 − band with its
interval below 1, no such case is slower than 1 + max(band, 3 %), and no tracking-tier case
whose plan changes is slower than 1 + max(band, 3 %). A tracking case past the limit with the
same plan does not decide (placement). Each change is judged and kept separately.

## Results

## Verdict
