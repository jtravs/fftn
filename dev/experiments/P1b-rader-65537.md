# P1b: Rader or Bluestein at 65537 on x86

- **Status:** done (rules unchanged: Rader stays at 65537 on x86)
- **Owner:** task/2.4-measure-2 session, 2026-11-17
- **Phase / task:** 2.4, a follow-up of [P1](P1-rules-vs-measure.md); PLAN lead (e)

## Question

P1's second pass on the EPYC found `planning::measure` running c2c 65537 at 0.57 of the rules'
time (rules 1.75 of measure's), by choosing a Bluestein stage over the rules' Rader stage in
10 of 12 rounds. X27 had adopted Rader on x86 up to 2²⁴, and PLAN's lead (e) noted EPYC
65537 at 1.63 of FFTW against X27's 1.09. After X36 changed the Rader convolution, does
Bluestein still beat Rader at 65537? If so, should the x86 targets' Rader range stop below
2¹⁶, as NEON's does?

## Hypothesis

Rader's permutations are random accesses over a megabyte at 65537 (LESSONS 4.11). On Zen 3,
with a 512 KiB L2, they lose to Bluestein's contiguous passes, as they did on the Mac. The
Xeon, with a 1 MiB L2, may differ.

## Change under test

The rules' Rader range, forced through fbench's `RADER` option (`build_settings::rader`):
`N` = `fftn` (Rader where the profile's range allows), `B` = `fftn+RADER=0` (Bluestein
everywhere). Same commit.

## Method

- Cases: X27's catalogue, primes 97, 257 and 65537, in double and float
  (`--catalogue x27 --filter 'c2c-(97|257|65537)-b1-op(-f32)?$'`). Only 65537 is above 2¹⁶;
  97 and 257 are the controls inside every range.
- Machines: the EPYC 7763 (GCC 16, `avx2`) and the Xeon 6240 (GCC 14, `avx512` and
  `ISA=avx2`). One thread, fbench's A/A control, shared allocations.
- Metric: `B ÷ N` per case with fbench's paired interval.
- Decision rule, written before running: the x86 targets' Rader range stops below 2¹⁶
  (`rader_below_of`), as NEON's does, if `B ÷ N` at 65537 is below 0.95, with the interval
  clear of the band, in both precisions on the EPYC and on both Xeon targets. If only the
  EPYC gains, the range is a property of the core, not of the target. Rules by core are
  deferred by the maintainer, so the rules stay, and the case is left to
  `planning::measure` and wisdom (recorded as such). 97 and 257 must not change.

## Results

EPYC (run 20261117-024156-2814, Slurm 511470/511484, commit 611f929; 8 of 8 rounds loaded,
A/A band ±1.00 %): `B ÷ N`, Bluestein's time over the rules' Rader:

| Case | double | float |
|---|---|---|
| 97 (control) | 1.517 [1.224, 1.581] | 1.357 [1.353, 1.373] |
| 257 (control) | 1.680 [1.646, 1.744] | 1.083 [1.056, 1.105] |
| 65537 | 1.493 [1.473, 1.499] | 1.178 [1.159, 1.187] |

At 65537 Rader is faster in this run, by 1.49 in double, the opposite of what measure found
in P1's second pass (Bluestein 0.57 of Rader's time). Three runs on the EPYC now disagree:
- P1's second pass (20261116-231702-f43c): Bluestein 1.75 times faster under measure;
- P2's verification (20261117-015335-5209): measure's Bluestein 1.41 times slower than
  the rules' Rader;
- here: Bluestein 1.49 times slower.

65537 on the EPYC is bimodal by process. The likely cause is the page backing of Rader's
random gathers over a megabyte, its index and kernel tables included (LESSONS 4.11; the page
kinds of P2's third pass). The Xeon's run (Slurm 16212) is still queued. The decision does
not wait for it, since the rule needs a gain on every machine.

## Verdict

Rules unchanged. The rule (Bluestein below 0.95 at 65537 on the EPYC and both Xeon targets)
fails on the EPYC: there Rader is faster at 65537 in this run (1.49 and 1.18). PLAN lead (e)'s
EPYC 65537 numbers (1.09 against 1.63 of FFTW) are the same bimodality. A Rader stage whose
gathers do not depend on the page size (blocked permutations, or tables laid out in the
order of access) is the follow-up, not a smaller range.
