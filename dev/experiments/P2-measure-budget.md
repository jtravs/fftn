# P2: the measurement budget

- **Status:** running (first pass)
- **Owner:** task/2.4-measure session, 2026-11-16
- **Phase / task:** 2.4 (`planning::measure`, DESIGN.md 11.2)

## Question

How many candidates, and how much planning time, does `planning::measure` need before its
choice stops improving, and what should `planning::quick` time? The answer sets the two
efforts' budgets (`quick_budget`, `measure_budget` in `src/core/measure.cpp`): faster planning
that finds the same plan is better (DESIGN.md 11.2).

## Hypothesis

The winners come from a handful of candidates: the rules' choice, the other strategy
(chain or grid, two or three phases) and FFTL.jl's long-row radices for long rows, the
second radix sequence for short ones. So the full effort's 16 candidates are more than it
needs (ranks above 8 never win), and the quick effort's 4 catch most of the gains at a
tenth of the planning time. The measure loop's statistic (each candidate's fastest of 3–9
samples of about 1 ms) picks the same program in most rounds; where candidates are within
2 % of each other the choice may flip, harmlessly.

## Change under test

None to the library's default. Arms (one commit, task/2.4-measure), shared with
[P1](P1-rules-vs-measure.md): `N` = `fftn` (rules), `Q` = `fftn+PLANNING=quick`, `M` =
`fftn+PLANNING=measure`. Later passes add `MEASURE_SECONDS` arms and candidate caps.

## Method

- Cases, machines, compilers and run settings as P1 (tracking tier, the Mac, the EPYC, the
  Xeon; one thread).
- Metrics: per case the paired execution ratio `Q ÷ M`; planning time per case and arm
  (fbench's `plan_s`, the plan's construction time, recorded apart from execution); from
  M's and Q's notes the chosen program, its rank among the distinct candidates (0 = the
  rules') and the candidates timed, per round.
- Decision rules, written before running:
  1. **Quick:** keep 4 candidates if `Q ÷ M` over the tier is within 1.02 (geomean) on
     every machine and Q's total planning time is at most a tenth of M's; otherwise raise
     quick's candidates to the smallest count that covers 90 % of M's choices by rank.
  2. **Measure's candidates:** if no choice of M has rank above 8 on any machine (every
     round), cap the full effort at 8 candidates; otherwise keep 16.
  3. **Time:** report the median and largest `plan_s` per arm; the 10 s cap stays unless a
     case's planning reaches it (then the samples per candidate shrink first).
  4. **Stability:** report the share of cases whose M choice is the same program in every
     round; below 80 % on a machine, the sampling (samples, sample length, statistic) is
     revisited before the budgets are fixed.

## Results

First pass pending.

## Verdict

Pending.
