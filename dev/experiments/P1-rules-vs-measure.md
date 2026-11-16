# P1: how often the rules' choice is within 5 % of the measured choice

- **Status:** running (first pass)
- **Owner:** task/2.4-measure session, 2026-11-16
- **Phase / task:** 2.4 (acceptance: the model's first choice within 5 % of the measured best
  for at least 90 % of catalogue cases)

## Question

How often does `planning::rules` build a program within 5 % of the one `planning::measure`
picks, per machine, and where it misses, which rule misfires? The answer decides whether the
rules can stay the default as they are, and which thresholds should be refitted or handed to
measurement. A second question: on a machine whose cache geometry misleads the rules (Granite
Rapids in X31b), does measurement recover the right program?

## Hypothesis

The rules were fitted on the Mac, the EPYC 7763 and the Xeon 6240, so on those three the rules'
choice should be within 5 % on at least 90 % of the tracking tier; the misses should cluster
on long rows near the chain-or-grid thresholds (X29, X32, X34) where X31b showed the
crossovers move with the core. With the Zen-like profile fields Granite Rapids' VM gets
forced on the Xeon (X31b: X29's and X32's chains, X34 off), the rules lose on long rows and
measurement should find the Xeon's own grids again.

## Change under test

None to the library's default: measurement is opt-in (`options::effort`). Arms, all one
commit (task/2.4-measure):

- `N` = `fftn` (rules; with fbench's identical-arm control `N~`).
- `M` = `fftn+PLANNING=measure`; `Q` = `fftn+PLANNING=quick` (P2 uses it).
- Xeon only, the misfire profile `G` = `fftn+CHAIN_PART_BYTES=16384+CHAIN_PART_LIMIT_BYTES=264241152+POW2_CHAIN_BYTES=1056964608+POW2_RADIX8_BYTES=never+GRID_TWO_ODD_PART=never`
  (what `set_grid_sizes` gives Granite Rapids' VM: 504 MiB of L3 over 8 CPUs, so X29's
  chains to half of L3, X32's radix-8 chains to twice L3, X34 off), and `GM` = `G` with
  `PLANNING=measure`.

Every candidate measurement times is a program the rules can also build (build_settings
knobs); no new kernels.

## Method

- Cases: the standard catalogue, tracking tier (61 cases), one thread, hot, median.
- Machines: the Mac M5 Max (Clang 23, `neon_fcma`), the EPYC 7763 (DMOG, GCC 16, `avx2`),
  the Xeon 6240 (GCC 14, `avx512`); shared allocations (`-c 8`), load recorded per round.
- Metric: per case the paired ratio `N ÷ M` (rules' time over measured choice's) with fbench's
  95 % interval and the A/A band; and from M's notes, the rules' time over the chosen one's
  as measured inside planning (`rules/chosen`), and the chosen program per round.
- Decision rules, written before running:
  1. **Acceptance (task 2.4):** on each machine, the share of cases whose `N ÷ M` point
     estimate is at most 1.05 must be at least 90 %. A case above 1.05 is re-run before it
     counts against the rules.
  2. **Measurement must not lose:** a case where M is slower than N with fbench's verdict
     "slower" beyond the band (`N ÷ M` below 0.97) is a bug in the measure loop (its
     statistic, its samples or its tie rule) and is fixed before P2 sets budgets.
  3. **Misfire recovery (Xeon):** among the cases where `G` is slower than `N` by more than
     5 %, `GM` must be within 5 % of `N` in at least 90 %.
  4. Rules missing by more than 5 % on two machines or more are listed as follow-ups
     (a refit or a measured-only choice); no default changes in this experiment.

## Results

First pass pending (runs below).

## Verdict

Pending.
