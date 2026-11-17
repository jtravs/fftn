# P3: a calibrated cost model for long rows

- **Status:** done, first pass (rejected as a replacement for the thresholds; see Verdict)
- **Owner:** task/2.4-cost-model session, 2026-11-17
- **Phase / task:** 2.4 (the cost model of DESIGN 11.2, machine profiles, `fbench calibrate`)

## Question

Can DESIGN 11.2's additive cost model, with a memory factor per cache level fitted per machine
by `fbench calibrate`, choose between a long row's structures as well as measurement does?
The structures are the chain (the rules' radices or FFTL.jl's long-row radices), the grid in
two phases and the grid in three. If it can, the model replaces the long-row thresholds of
X12b, X29, X32, X33 and X34 (`grid_min_bytes`, `grid_three_bytes`, `chain_part_bytes`,
`pow2_chain_bytes`, `grid_two_odd_part`): hand thresholds misfire on a machine they were not
measured on (Granite Rapids, X31b), and P1 found the Xeon's long rows off by 6–14 %.

## Hypothesis

A row's time is dominated by its passes over memory, and each pass costs per point according
to the cache level its working set occupies.

The model: one row of n points, a program of passes of radices R. A chain's pass costs
n (α + β log₂ R) m(level of its working set, 32 n bytes for double), multiplied by κ for a
power-of-two row with a radix of 16 or more beyond L1 (X29's and X32's set conflicts). A grid
phase's passes run on blocks in L1, at factor 1. Its gathers and scatters cost γ(level) per
point, one each per phase unless fused (X12). α, β, m(L2), m(L3), m(DRAM), κ and γ per level
are fitted per machine and precision, nine numbers.

If the hypothesis holds, such a model reproduces the measured best structure within 5 % on
most long rows of the three machines, because the thresholds it replaces are its crossovers
(L1, L2/2, L2, half of L3 and twice L3 by machine).

## Change under test

Data collection first, no library change. fbench gets `GRID_PHASES=2|3` (`build_settings::
grid_phases`). Arms on one commit:
- `R` = `fftn` (the rules);
- `C` = `fftn+STRATEGY=chain` (the chain with the rules' radices, FFTL.jl's on the machines
  whose profile uses them);
- `G2` = `fftn+STRATEGY=grid+GRID_PHASES=2`;
- `G3` = `fftn+STRATEGY=grid+GRID_PHASES=3`.

Each case's note gives the program (radices and phases) that the model prices.

## Method

- Cases: `bench/catalogues/p3.toml`: c2c rows of double and float, powers of two from 8192
  to 2²², mixed lengths with small and large odd parts (12288 to 4 · 10⁶), and r2c and c2r
  rows of 32768 to 2²²; one transform each, out of place.
- Machines: the Mac (Clang 23, `neon_fcma`), the EPYC 7763 (GCC 16, `avx2`) and the Xeon 6240
  (GCC 14, `avx512`). One thread, fbench's A/A control.
- Fit (`bench/calibrate/calibrate.py`, standard-library Python): per machine and precision,
  least squares on log time over every case and arm, the programs parsed from the notes.
  The fit is checked by leaving one length out at a time: each length's choice is predicted
  by a model fitted without it.
- Metric: for each case, the measured time of the structure the model chooses (left out)
  over the fastest of the four arms, and the same for the rules (`R`).
- Decision rule, written before running:
  1. The model's choice is within 5 % of the fastest arm on at least 90 % of the cases on
     every machine.
  2. Its geomean over the catalogue is no worse than the rules' by more than 1 % on any
     machine.
  3. With Granite Rapids' cache sizes given to the model, it chooses the Xeon's grids for
     the rows where X31b measured them faster. This is the misfire that motivates P3.

  If all three hold, the model is built into the planner as the long-row choice (a
  following commit with its own A/B), and the thresholds it replaces are deleted. If not,
  the fit's residuals say which term is missing, and the thresholds stay.

## Results

### Mac, first fit (run 20261117-015214-834a, commit 059b383; 7 of 10 rounds loaded, ±3.64 %)

The model as written in the hypothesis, with the conflict term widened so that it applies to
any row whose power-of-two part spans 16 KiB, separately for radix 16 and up (`k`) and
below (`k8`), as X29 and X32 imply. Leave-one-out:

| Precision | Cases | Fit rms (log) | Model within 5 % | Model geomean ÷ fastest | Rules ÷ fastest |
|---|---|---|---|---|---|
| double | 32 | 0.198 | 23 (71.9 %) | 1.071 | 1.031 |
| float | 20 | 0.180 | 11 (55.0 %) | 1.096 | 1.066 |

- Rule 1 fails, and so does rule 2: the model's choices are worse than the rules' on the Mac.
- The misses are mixed lengths with a large power-of-two part (12288, 61440, 245760,
  393216: the model chooses the chain where grids win by 1.1–1.7) and powers of two from 8192
  to 131072 (it chooses three phases where the chain or two phases win).
- The fitted factors are degenerate. L2 prices below L1 (`m1` 0.70). The L3 terms are
  unconstrained, since the Mac has no level between its L2 and memory.
- A fit error of about 20 % cannot separate structures that differ by 5–30 %.

The fallback, the profile's own long-row fields fitted to the same timings (`calibrate.py
thresholds`: `grid_min_bytes`, `chain_part_bytes`, `chain_mixed_bytes`, `pow2_chain_bytes`,
`grid_three_bytes`, `grid_two_odd_part`; a coordinate search from every `grid_min`, leaving
one case out), is no better on these data:
- double: 26 of 32 (81 %) within 5 %, geomean 1.084, against the rules' 28 of 32 and 1.031;
- float: 15 of 20 (75 %), 1.090, against the rules' 13 of 20 and 1.066.

With six fields and one case per length, leaving a case out often leaves its region without
a sample: a calibration catalogue needs several lengths per region.

A per-pass model with a factor per level does not capture what decides a grid's cost on the
Mac: the block widths, the copies' sets and the padding. Before adding terms, the EPYC and
Xeon runs (Slurm 511481/511482, 16209/16210), on machines with three cache levels, will show
whether the form fails generally. If it does, the practical replacement is calibration of the
crossovers themselves: `fbench calibrate` measuring, per machine, where chain, two and three
phases cross, and writing the profile's thresholds. That keeps the planner deterministic and
removes the hand-set constants that misfired on Granite Rapids.

### EPYC (run 20261117-020528-aad3, Slurm 511481/511482, commit 059b383; 10 of 10 rounds loaded, ±3.49 %)

| Precision | Cases | Model within 5 % (geomean) | Calibrated fields within 5 % (geomean) | Rules within 5 % (geomean) |
|---|---|---|---|---|
| double | 32 | 32 (100 %; 1.000) | 31 (96.9 %; 1.046) | 26 (81 %; 1.053) |
| float | 20 | 19 (95 %; 1.017) | 18 (90 %; 1.036) | 14 (70 %; 1.055) |

On the EPYC the model meets rules 1 and 2, where the rules themselves do not. Its fit is
loose (rms 0.29 in log time), but the choices come out right: on Zen 3 the chain wins
almost everywhere to beyond L3. The calibrated fields keep chains to 16 MiB, against the
rules' X29 and X32 exceptions. The one float miss is 393216 (the chain chosen, 1.41).

### Xeon (run 20261117-030022-5a80, Slurm 16209/16210, commit 059b383; 2 of 10 rounds loaded, ±3.21 %)

| Precision | Cases | Model within 5 % (geomean) | Calibrated fields within 5 % (geomean) | Rules within 5 % (geomean) |
|---|---|---|---|---|
| double | 32 | 23 (71.9 %; 1.038) | 27 (84.4 %; 1.026) | 25 (78 %; 1.017) |
| float | 20 | 16 (80 %; 1.027) | 14 (70 %; 1.043) | 17 (85 %; 1.013) |

The model's misses on the Xeon:
- real rows, chains chosen where grids win: c2r 2²⁰ 1.54, c2r 100000 1.14, r2c 65536 1.13;
- the boundaries between two and three phases: 262144, 393216 and 524288 in float, up to 1.21.

Rule 3, the Xeon's fit with Granite Rapids' caches:
- grids for the long powers of two (2¹⁷ to 2²²) and two phases for 100000, 10⁶ and 1024000,
  as X31b measured;
- chains at 46080, 61440 and 65536, where X31b found grids faster (0.85–0.88).

So rule 3 is half met.

## Verdict

First pass: **rejected** as a replacement for the long-row thresholds; the thresholds stay.
- **Rule 1** (90 % within 5 %) holds only on the EPYC (100 % and 95 %). It fails on the Mac
  (72 %, 55 %) and the Xeon (72 %, 80 %).
- **Rule 2** (no worse than the rules by 1 %) fails on both too: the model's geomean is
  1.07–1.10 of the fastest on the Mac against the rules' 1.03–1.07, and 1.03–1.04 on the Xeon
  against 1.01.
- **Rule 3** is half met.
- **The calibrated fields**, the fallback, also fail rule 1 on the Mac and the Xeon. They
  beat the rules only on the EPYC (97 % and 90 %).

What the fits say:
- **A per-pass model with one memory factor per level is too coarse.** What decides between
  structures beyond L1 is the copies' and passes' cache sets and the block widths, which
  change with the factorisation, not with the level alone. Its fit errors of 15–29 % in log
  time are as large as the differences it must rank.
- **The EPYC's success is easy to explain.** There the chain wins almost everywhere to
  16 MiB, so any model that learns "chains are cheap" chooses well.
- **The P3 catalogue is too sparse for calibration:** one length per region. Leaving a case
  out removes the only sample of its region.

What follows:
- The rules stay deterministic and stay the default.
- Misfires on unmeasured machines (Granite Rapids) are `planning::measure`'s and wisdom's to
  correct: P1 recovered 9 of 9 of the forced misfire.
- A second pass of P3, if wanted, needs a denser catalogue (several lengths per region and
  structure) and a model of the copies' set conflicts by stride (X26, X26b), before
  `fbench calibrate` writes any profile field.
- The tools stay for that pass: `GRID_PHASES`, the `p3` catalogue and `calibrate.py` (fit,
  thresholds, gnr). No library change.
