# X2: radix sets and orders per target

- **Status:** planned
- **Owner:** unassigned (written with task 1.2, 2026-11-14)
- **Phase / task:** 1.4

## Question

Which radix sequence (which radices, in which order) should the chain use for each length
on each target, and how far from the best are planner v0's rules?

## Hypothesis

Planner v0 adopts FFTL.jl's single-line rules as its starting point (DESIGN.md 11.2,
specified there and recorded in `tests/core/data/rule_radices.txt`). They were derived on
FFTL's own kernels: brute-force searches over radix multisets and orders for n ≤ 1024
(7-smooth) on the M1, checked on held-out lengths, plus interleaved A/B on Zen 2 and 3,
Tiger Lake and EPYC 9354. FFTN's passes differ (generated codelets, tails on a tile,
broadcast twiddles, copies around strided lines), so we expect the rules to be near the
best for most lengths and clearly off for some: lengths whose leading radix is below C on
wide registers (AVX2 float, AVX-512), lengths with primes above 16, and lengths where a
radix 14, 15 or 16 codelet, which the rules never choose, would remove a pass. The rules'
AVX-512 radix-16 gate (DESIGN.md 11.2) is not in v0 and is part of this question for
task 1.3's `avx512`.

## Change under test

Arm A: planner v0's rules (the commit current when the experiment starts). Arm B: for each
length, the best sequence found by the search below, forced through
`build_settings::radices` (an internal test knob, so no switch in `dev/KNOBS.toml`); the
adopted outcome is new rules, or a per-target table, that reproduce the winners.

## Method

- Cases: c2c rows, one transform and a batch of 64, every 7-smooth n ≤ 1024 and the
  standard catalogue's 1-D lengths above it, forward, double and float.
- Search: for each length, every multiset of codelet radices (2–16) whose product is n,
  in every distinct order, pruned to at most 8 passes; above 1024, the 20 best sequences of
  a cost estimate (pass count, then codelet operation counts from `src/gen/codelets.hpp`)
  plus v0's choice. Each candidate timed by fbench in one session per machine, then the
  winner and v0's choice confirmed by interleaved A/B with an A/A control.
- Machines, compilers, instruction sets, thread modes: Apple M5 Max (neon_fcma, Clang 23,
  quiet-machine check), Xeon Gold 6240 (avx2, and avx512 once task 1.3 lands; GCC 15),
  EPYC 7763 (avx2; GCC 15); one thread; scalar forced on one machine for its rules.
- Metric: per length, time(v0) ÷ time(best), the paired ratio of the confirmation A/B with
  its 95 % interval; per target, the geometric mean and the distribution over lengths.
- Decision rule, written before running: per target, replace the rule (or encode a table
  for the lengths concerned) wherever the best sequence beats v0 by more than 3 % with the
  interval clear of the A/A band; keep v0 where no candidate does. Accept new rules only if
  they reproduce at least 90 % of the measured winners and no length becomes slower than
  v0 by more than 2 % on any target measured. Rules that need cache geometry go to the cost
  model of phase 2 instead.

## Results

None yet.

## Verdict

Pending.
