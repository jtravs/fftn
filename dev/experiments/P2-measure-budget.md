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

First pass, the runs of [P1](P1-rules-vs-measure.md): the Mac (20261116-181442-4eed and
20261116-183405-3034), the EPYC (20261116-184705-6935) and the Xeon (20261116-183511-33b2).
Every machine was loaded.

| | Mac (run 1, run 2) | EPYC | Xeon |
|---|---|---|---|
| `Q ÷ M` execution geomean | 1.005, 1.003 | 0.999 | 1.014 |
| worst `Q ÷ M` | c2c 480 × 640 1.22 | c2c 64³ 1.61 | c2c 128 × 4096 1.22 |
| M's planning, median per case and largest | 0.044 s, 7.4 s (2²²) | 0.048 s, 12.5 s (2²²) | 0.050 s, 12.6 s (2²²) |
| Q's share of M's planning time, summed over the tier | 27 % | 27 % | 23 % |
| ranks M chose (rank: count of case-rounds) | 0: 521, 1: 67, 2: 59, 3: 37, 4: 20, 5: 13, 6: 15 | 0: 611, 1: 44, 2: 14, 3: 44, 4: 6, 5: 11, 6: 2 | 0: 535, 1: 68, 2: 54, 3: 28, 4: 20, 5: 14, 6: 12, 7: 1 |
| cases whose M choice is one program in all 12 rounds | 32 and 26 of 61 | 40 of 61 | 35 of 61 |

- **Building plans, not timing them, sets the planning time of large transforms.** The
  rules' own plan takes 0.07–0.43 s to build at 2²⁰ and 0.5–1.7 s at 2²². So Q's time is
  dominated by its few builds of the largest rows, and M's 12.5 s at 2²² exceeds its 10 s
  cap. Every candidate admitted is timed at least once, and the final plan is built after
  the cap.
- **No winner ranked above 7.** The candidates ranked 0–3, which `quick` times, hold 76–84 %
  of M's choices that were not the rules'; ranks up to 5 hold 93–98 %.
- **The choice is unstable under load.** Only 43–66 % of cases keep one program over 12
  rounds. Where the choice changes, the alternatives were within the noise (the rules'
  choice over the chosen one 1.00–1.05 in most rounds). In some rounds the candidate timed
  first read 1.2–1.3 times its time (Mac: c2c 60, 1024 backward, c2r 256). Rotating the
  first candidate per round (ffb9046, the Mac's second run) did not raise stability under
  that load (26 of 61).

## Second pass (method)

Commit 73eb719 changes the sampling, as rule 4 required, and nothing else of the budgets:
- each round places the input and output at other 64-byte offsets within a page
  (`placement_of`, as fbench does);
- each candidate's statistic is its median sample, not its fastest;
- only live candidates compete for the choice;
- every allocation counts against the scratch limit;
- nothing is timed when only the rules' choice is admitted.

The second reader's fixes are in the same commit. Arms, cases and decision rules are as in
the first pass: `N`, `Q`, `M` at 73eb719, tracking tier, the Mac (local), the Xeon (Slurm
16140/16141) and the EPYC (511174/511175). The same runs are P1's re-run of the Xeon misses.

Mac result (run 20261116-200843-f54c, band ±1.93 %):
- `Q ÷ M` 1.008;
- the cases with one program over 12 rounds are still 32 of 61. The choices that change are
  near-ties: no case reads `N ÷ M` below 0.97, against 0.92 in the first pass;
- Q's planning summed over the tier fell from 4.5 to 2.4 s per round, and 2²² from 1.6 to
  0.8 s, because nothing is timed when only one candidate is admitted;
- M's median planning time is 0.042 s per case, and 2²² takes 6.4 s.

Xeon (20261116-215950-1f7c, ±4.10 %) and EPYC (20261116-231702-f43c, ±1.82 %):
- `Q ÷ M` 1.010 and 1.006;
- one program over 12 rounds in 23 and 40 of 61 cases;
- winners up to rank 6 and 5;
- M's planning 0.050 and 0.048 s median per case, 10.2 and 6.3 s at the largest;
- Q's summed over the tier 2.8 and 4.7 s per round (first pass 7.9 and 9.1 s).

Rule 4 still fails on every machine. The instability that remains is between candidates within
noise of each other; a rule judging stability should count a change of choice only when the
programs differ by more than the band. That is a proposal for a third pass, not a change to
this pass's rule.

## Verdict

First pass; the budgets are not changed yet.

1. **Rule 4 (stability) failed on every machine** (43–66 %, against 80 %). By the rule, the
   sampling is revisited before the budgets are fixed. Two changes are in:
   - candidates rotate their order per round (ffb9046);
   - candidates are built and warmed in turn, with admission stopping at half the budget
     (4333fff; this was P1's 2²² miss).

   Next: a placement that does not favour one program (scratch offsets as fbench uses, or
   different ones per round), and a second pass on a quiet allocation.
2. **The counts the first pass points to, once stable:**
   - `quick` 6 candidates: rule 1's time condition failed (23–27 %, against 10 %) and ranks
     0–5 cover 93–98 % of the choices;
   - `measure` at most 8: rule 2 holds, since no winner ranked above 7.

   The execution side of rule 1 held: `Q ÷ M` was 0.999–1.014 everywhere.
3. **Rule 3 (time):** the cap was overrun at 2²² by the final build and the per-candidate
   minimum (12.5 s for a 10 s cap). Building candidates in turn (4333fff) bounds admission,
   not the final build. Faster plan construction at 2²⁰–2²² (0.5–1.7 s for one plan) would
   cut both efforts' time most.
