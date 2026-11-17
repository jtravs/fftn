# P3: a calibrated cost model for long rows

- **Status:** running (data collection and an offline fit first)
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

Pending.

## Verdict

Pending.
