# X8: whole-transform leaves above 16

- **Status:** running (plumbing checked on the Mac; the campaign on the three machines is
  pending)
- **Owner:** the lead session (written with the radix 32 and 64 codelets, 2026-11-14)
- **Phase / task:** 1.4 (with the codelets of task 1.6)

## Question

For a length n ≤ 64 that has a codelet (32 and 64 now), is one whole-transform leaf (one
codelet call per line) faster than the two- or three-pass chain planner v1 picks, per
target, precision and batch? The answer sets the planner profile's leaf limit
(`target_profile::leaf_limit`, `src/planner/rules.hpp`) per target and precision, and with
B3 whether radix 64 (and 32) stay in the library at all.

## Hypothesis

A leaf does no twiddle loads, no scratch round trip and has one pass overhead instead of
two or three, so it should win where its values fit in registers and every lane is busy.
Two things work against it:

- **Idle lanes.** The leaf (DESIGN.md 8.2) broadcasts each value on loading and stores
  lane 0, so it uses one of C lanes: C = 1 on `scalar` and on NEON in double, 2 on NEON in
  float and on AVX2 in double, 4 and 8 above. The chain uses all C lanes. Leaves
  vectorised across rows are phase 4's lanes strategy, not this experiment.
- **Spills.** In complex-register form radix 32 keeps 34 values live at its peak and
  radix 64 keeps 66 (all of split radix's U, Z and Z' before the first output), against
  16 registers on AVX2 and 32 on NEON and AVX-512. Counted in the disassembly (below),
  radix 64 spills on every target.

So we expect the leaf to win at 32 where C = 1 (NEON double), to lose wherever C ≥ 2, and
radix 64 to lose or tie everywhere. Radix 32 on AVX-512 barely spills but idles three or
seven lanes.

## Change under test

Arm A: planner v1's chain (no option). Arm B: the leaf, either per length through the
radix test knob (`RADICES=32` for n = 32, `RADICES=64` for n = 64: one codelet radix
equal to the length is a leaf) or for all lengths at once through the planner's leaf limit
(`LEAF_LIMIT=64`, `build_settings::leaf_limit`). Both are internal build settings set by
the fbench backend (bench/README.md, "FFTN options"), so no switch in `dev/KNOBS.toml`.
The two give the same plans for these cases; `LEAF_LIMIT` is one run instead of two.

## Method

- Cases: the X2 catalogue's c2c rows of length 32 and 64, forward, out of place, batch 1
  and 64, double and float: 8 cases (`--catalogue x2 --tier full --filter
  '^c2c-(32|64)-b(1|64)-op'`). Composite leaves (20, 24, 48 and the like) are not in it:
  no codelet exists for them; generating them is the generator's catalogue line plus B3's
  cost.
- Machines and targets: Mac M-series (`neon_fcma` and `neon`), the EPYC 7763-class node
  (`avx2`), the Xeon Gold 6240 (`avx512` and `avx2`); one thread, Release, the gate's
  compilers (Homebrew Clang on the Mac, GCC 14 or 16 on Linux).
- Metric: fbench's median paired ratio B ÷ A per case with its 95 % interval; the weighted
  geometric mean per target and precision; the A/A band from the control arm.
- Decision rule, written before running. For each target and precision, raise the leaf
  limit to the largest L in {32, 64} such that, over the cases with n ≤ L above 16 (batch
  1 and 64 together):
  1. the geometric mean of B ÷ A is at most 0.95, with the A/A band below 5 %, and
  2. no case is slower: no interval lies wholly above 1.02.
  Otherwise the limit stays 16. The limit goes into `profile_of` per target and
  precision. If no target raises it to 64, the radix 64 leaf is dropped from the pass
  table; whether the radix 64 codelets stay at all is X3's (few big passes) question.

Commands (the lead runs them; `F="uv run --frozen python bench/fbench.py"`, the build in
`build/check-release` from this branch):

```bash
X8="^c2c-(32|64)-b(1|64)-op"
# Mac
$F run A=fftn+ISA=neon_fcma B=fftn+ISA=neon_fcma+LEAF_LIMIT=64 --catalogue x2 --tier full --filter "$X8"
$F run A=fftn+ISA=neon B=fftn+ISA=neon+LEAF_LIMIT=64 --catalogue x2 --tier full --filter "$X8"
# EPYC (DMOG)
$F run A=fftn+ISA=avx2 B=fftn+ISA=avx2+LEAF_LIMIT=64 --catalogue x2 --tier full --filter "$X8"
# Xeon
$F run A=fftn+ISA=avx512 B=fftn+ISA=avx512+LEAF_LIMIT=64 --catalogue x2 --tier full --filter "$X8"
$F run A=fftn+ISA=avx2 B=fftn+ISA=avx2+LEAF_LIMIT=64 --catalogue x2 --tier full --filter "$X8"
# Per length, forced (the same comparison; a check of the leaf-limit arm where a result
# is close to the rule's thresholds)
$F run A=fftn+ISA=avx2 B=fftn+ISA=avx2+RADICES=32 --catalogue x2 --tier full --filter '^c2c-32-b(1|64)-op'
$F run A=fftn+ISA=avx2 B=fftn+ISA=avx2+RADICES=64 --catalogue x2 --tier full --filter '^c2c-64-b(1|64)-op'
```

## Codelets and their cost (task 1.6, B3)

Operation counts (`uv run python -m gen.fftn_gen --table`) equal FFTW 3.3's published
counts exactly: plain 32 is 372 additions and 84 multiplications (n1_32), plain 64 is 912
and 248 (n1_64), and the twiddle variants 434 and 208 (t1_32) and 1038 and 500 (t1_64);
both plain counts are the split-radix 4n log₂ n − 6n + 8. Peak live values: 34 (radix 32)
and 66 (radix 64).

Stack spills, counted in the disassembly of the Release objects (Homebrew Clang 23.1,
`-O3`; x86-64 objects cross-compiled on the Mac): vector stores to the stack and reloads in
the forward pass with no edge operation, double / float. On AArch64 eight of the stores
and reloads in each function are the prologue's saves of d8–d15, not spills.

| target | leaf 16 | leaf 32 | leaf 64 | grouped 32 | grouped 64 | middle lane 64 |
|---|---|---|---|---|---|---|
| `neon_fcma` | 0 / 0 | 24 / 24 | 78 / 71 | 76 / 135 | 221 / 505 | 167 / 343 |
| `neon` | 0 / 8 | 27 / 36 | 78 / 89 | 75 / 189 | 221 / 517 | 173 / 378 |
| `avx2` | 4 / 4 | 31 / 41 | 128 / 140 | 165 / 222 | 459 / 574 | 347 / 681 |
| `avx512` | 0 / 0 | 9 / 8 | 67 / 61 | 109 / 137 | 424 / 321 | 440 / 349 |

(Stores only; reloads are similar.) Radix 64 spills heavily on every target; radix 32's
grouped and lane passes spill on every target too, its leaf mostly on AVX2 and NEON. That
is the complex-register form's limit, which the split form (task 1.5) and a
register-aware order may move.

Library size and build time (B3), before (radices 2–16) and after (2–16, 32, 64), with
every pass of 32 and 64 instantiated (grouped, lane, last, leaf, all edge variants, both
directions and precisions): text bytes of each kernel unit and its serial compile time on
the Mac (shared and loaded, so times are rough):

| unit | text before | text after | ratio | compile before | compile after |
|---|---|---|---|---|---|
| `scalar` (AArch64) | 1 047 372 | 2 023 904 | 1.93 | 11 s | 34 s |
| `neon` | 990 744 | 1 872 556 | 1.89 | 6 s | 12 s |
| `neon_fcma` | 824 516 | 1 558 316 | 1.89 | 8 s | 16 s |
| `scalar` (x86-64) | 1 693 801 | 3 437 585 | 2.03 | 10 s | 34 s |
| `avx2` | 2 298 740 | 4 289 460 | 1.87 | 15 s | 33 s |
| `avx512` | 4 196 521 | 7 150 305 | 1.70 | 22 s | 46 s |

Of the pass instructions in each unit, radix 64's are 28–37 % and radix 32's 14 %. The
library's Release build on the Mac (6 jobs) went from 18 s to 36 s. The generator takes
12 s instead of about 1 s and its tests 39 s instead of about 5 s (radix 64's builds
dominate). The edge variants multiply it: a radix has 16 leaf instantiations per
precision, 14 more for the chain's passes.

## Results

Plumbing check only (the Mac, shared and loaded: every round under load, A/A band
±1.26 %), run 20261114-092904-8375, `A=fftn B=fftn+LEAF_LIMIT=64`, `neon_fcma`: the leaf
of 32 was about 25 % faster than the chain 2 × 16 in double (C = 1) at batch 1 and 64 and
3–6 % slower in float (C = 2); the leaf of 64 was 9–17 % slower in double and 1.7–1.8
times slower in float. This agrees with the hypothesis but decides nothing: the campaign
above is the result.

## Verdict

Pending the runs above.
