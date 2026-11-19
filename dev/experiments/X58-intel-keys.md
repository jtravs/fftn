# X58: profile keys for recent Intel (Ice Lake VMs, batch leaves of 12)

- **Status:** done: both changes adopted
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

Spot VMs of Google Cloud, GCC 14.2, `avx512`, every arm pinned to CPU 2, no round under load.
G is 5f8b14a, N is `main` 0508037. A first Ice Lake run was stopped: it added 2^23 points,
where FFTW's MEASURE planning took over 100 minutes. Two VMs were preempted before they
finished, and their replacements are the runs below.

**Depth-first chain, Ice Lake VM** (`n2-standard-16`, one thread per core, private L2 1.25 MiB,
55 MiB of L3 for 8 CPUs). Forced tops on N, screen run in X56's `n2b` VM: a top of 16 took
0.50–0.68 of the time from 32768 to 2^22 points, a top of 20 0.49 at 100000.

| Run | Cases | G ÷ N | Band |
|---|---|---|---|
| 20261119-002713-2377 | 18 rows of 16384 points and more | 0.596 [0.589, 0.603], 16 faster, none slower | ±4.30 % |
| 20261119-003220-1982 | tracking tier (61) | 0.929 [0.922, 0.937], 7 faster, none slower | ±3.60 % |

FFTN ÷ FFTW, N → G: 32768 1.20 → 0.78, 46080 1.43 → 0.70, 65536 (every form) 1.41–1.53 →
0.72–0.79, 100000 1.44 → 0.71, 131072 1.32 → 0.70, 262144 1.34 → 0.68, 2^22 1.13 → 0.64. ÷ oneMKL:
1.08–1.71 → 0.55–0.90 on the same rows. 16384 and 20000 lie below L2/4 and keep their plans.

**No batch leaf of 12** (`batch_leaf_12` false on a private L2 of 2 MiB or more). The rule
changes 12 × 64 and 12 × 1024 only.

| Machine | Run | 12 × 64 | 12 × 1024 | Changed cases | Band | Tracking tier |
|---|---|---|---|---|---|---|
| Granite Rapids (`c4-standard-16`) | 20261119-002918-a348 | 0.865 | 0.870 | 0.867 | ±1.00 % | 0.998, none slower (20261119-013614-c63c, ±2.25 %) |
| Sapphire Rapids (`c3-standard-8`) | 20261119-003032-02b8 | 0.854 | 0.932 | 0.892 | ±1.00 % | 1.000, none slower (20261119-020423-7938, ±2.06 %) |

With batch leaves off altogether on N (X56's `n2b` and `c3b` VMs): Ice Lake 12 × 64 1.233 and
12 × 1024 1.076, so it keeps the leaf; Sapphire Rapids 0.857 and 0.926, so it drops it.

## Verdict

**Both changes adopted**, each meeting the rule on every family it reaches and was measured
on. Not measured: Emerald Rapids (2 MiB of L2, takes the second change), Ice Lake on bare metal
(its small L3 per core already takes the Xeon 6240's branch), client cores with an L2 between 1
and 2 MiB, float on Ice Lake (stays off). Why the batch leaf of 12 loses with 2 MiB of L2 and
wins with 1–1.25 MiB was not traced. Cloud cost of this experiment and its screens: about USD
3.0 ([ledger](cloud-ledger-2026-11.md), X56 and X58).
