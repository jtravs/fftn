# X54: radix-32 and radix-64 passes without most of their spills on `avx2`

- **Status:** running
- **Owner:** x54 session, 2026-11-18
- **Phase / task:** 1 (phase 1 exit check, the tail rule; cause 4); follows X38, X42, X47

## Question

Can FFTN's radix-32 and radix-64 codelets be ordered so that GCC spills far less of them on
`avx2`'s 16 registers, and does that let two-pass plans of 768, 1024 and 2048 points (FFTW's
shape) win at every batch on the EPYC 7763, so that the planner takes them? The answer
changes the generator's emission order (`gen/fftn_gen/schedule.py`, all targets' codelet
text) and the `avx2` double overrides in `src/planner/rules.cpp`.

## Hypothesis

The exit check on main 8c18dd2 (draft PR 67) fails on the EPYC by 12 cases beyond noise;
its mid rows are most of them: 1024 1.21 of FFTW (backward 1.20), 1024 × 3 1.22,
1024 × 16 1.13, 64 × 16 1.14, 2048 1.11, 768 1.09, 512 1.07. X47 placed the gap of 768–2048
in the third pass of the planner's three-pass chains; a two-pass plan would avoid it if its
radix-64 (or 32) pass kept FFTW's IPC, but GCC 16's radix-64 lane unit spends 327 of 1161
instructions on stack moves (X47), and after X47's two-line chains 1024 as 16 × 64 is 0.957
of the planner's time at batch 1 and 1.10 at batch 1024.

Why the codelets spill as much as they do: the generator orders a codelet for the fewest
live values (`schedule.py`), which is the right aim while a codelet fits the register file
and the wrong one past it: radix 32 and 64 peak at 35 and 67 live values, and what then
costs is how many values the allocator must store and reload. GCC on x86 allocates
registers in source order (it schedules only after allocation), so the order the generator
emits is close to the order GCC allocates. A model of the allocator (Belady's
furthest-next-use eviction in 16 registers, an evicted input re-loaded from its source, a
computed value stored once and reloaded at each later use) counts 292 stack moves for the
radix-64 plain codelet as ordered now and 93 for radix 32; GCC 16's lane units have 327 and
101 (149 + 178 and 49 + 52 stores and loads, plus 65 and 29 stack operands folded into
arithmetic), so the model tracks GCC. A list scheduler that scores each ready operation by
its operands outside a model register file of the values touched most recently (`fitted
order`) orders the same operations for 146 moves (radix 64) and 40 (radix 32) in the model;
compiled with GCC 16 (`avx2`, the lane passes' whole-unit loops, `objdump`), stack stores
and loads fall from 149 + 178 to 87 + 107 a unit for radix-64 plain (folded 65 → 38;
instructions 1161 → 1092), 155 + 171 to 85 + 100 for radix-64 DIF (1484 → 1407), 49 + 52 to
30 + 34 for radix-32 plain. X42's two-level codelets (staged through a fenced array) also
cut spills but added 4–12 % arithmetic and left GCC's instruction count unchanged; this
changes the order only, so every operation and rounding is the same and the results are
bit-identical.

Expected: radix-32 and radix-64 passes cheaper by a few per cent on `avx2` (Zen 3 stores
one 256-bit value a cycle: a radix-64 plain unit stored 64 outputs and 149 spills, now 87),
plans that already take them (256, 384, 512, 640 on `avx2`; 32-point columns of planes,
X44) a little faster, and the two-pass plans of 768–2048 close enough to win at both
batches of X38's selection rule. On the 32-register targets (`avx512`, NEON) the model
counts 191 → 76 moves for radix 64 and 4 → 8 for radix 32: small either way.

## Change under test

Arms are commits of branch `exp/x54-radix64-avx2` against O = its base (main 4d75c48):

- **A (fitted order).** `schedule()` takes `fitted_order()` for a complex codelet whose
  fewest-live order keeps more than 16 values live, where `spill_moves()` at 16 registers
  counts at least a quarter and at least 16 fewer moves (the best of model register files
  of 12–16 and two weights); that is radix 32 and 64, plain and DIF, both directions (DIT
  is not improved and no pass calls it). Every target compiles the new text; plans do not
  change; bits are identical.
- **B (two-pass `avx2` double overrides, on A).** X2's search driver on the EPYC with A's
  build, candidates with radix 32 or 64 (`--radices 2-16,20,25,32,64 --require 32,64`),
  lengths 512, 640, 768, 1024, 1536, 2048, 3072, 4096; a length gets an override by X38's
  selection rule: its double winner the same sequence at batch 1 and 64 and at most 0.97
  of the planner's at both in the confirmation, intervals clear of the band (else the
  batch-1 winner if it passes at both batches). `avx2` double only; no other target's plans
  change.

No switch: arms are commits, `dev/KNOBS.toml` is untouched.

## Method

- Static check (done for A, above): GCC 16 `objdump` of the `avx2` row units of radix 32 and
  64 (`rows3_avx2.cpp`, `rows4_avx2.cpp`) for O and A, stack moves per whole-unit loop.
- Screens (not deciding): forced sequences (`RADICES`) of 512, 768, 1024, 2048 at batches 1
  and 1024 on O's and A's builds, the planner's plan as baseline; `run --pmu` counters.
- Judging runs, `fbench_remote.py submit --serial`, one thread, interleaved process rounds,
  A/A control:
  - EPYC 7763 (DMOG, GCC 16.2, `avx2`), deciding for A and B: the exit check's cases
    (standard catalogue, filter `^c2c-[0-9]+-b`, tier `full`) with arms O, A (and B), FFTW;
    then the tracking tier, O against A (and B).
  - Xeon Gold 6240 (GCC 14.4, `avx512`): the tracking tier, O against A (`avx512` compiles
    the same codelet text). B changes no `avx512` plan and is not run there.
  - Apple M5 Max (Clang, `neon_fcma`, NEON FFTW): the tracking tier, O against A, when the
    Mac is free (one fbench run at a time).
- Metric: fbench's median paired ratio arm ÷ the arm before, per case, with its interval;
  geometric means over each arm's target cases; FFTN ÷ FFTW before and after on the named
  cases.
- **Decision rule, written before any judging run.** Judged on GCC on x86-64 Linux and
  Clang on macOS:
  - **A** (A ÷ O). (a) On the EPYC, over the exit check's cases whose FFTN plan in the run
    (the store's note) contains a radix-32 or radix-64 codelet, the geometric mean is at
    most 1 − band with its interval below 1. (b) No case of the EPYC's judged set or
    tracking tier is slower than 1 + band + 2 %. (c) On the Xeon and the Mac the tracking
    tier's geometric mean is at most 1 + band and no case is past 1 + band + 2 %. A case past
    a limit on a loaded machine is re-run once (`--serial` on x86) and the re-run decides. All
    targets compile one codelet text, so A passing (a) and (b) but failing (c) on one
    target means per-target texts (the order is free to differ: bits are the same); that
    is recorded and put to the lead, not built here. If A fails (a) but
    passes (b) and (c), it stays only as B's carrier and only if B is adopted (as X42's
    radix-25 codelet carried its overrides); otherwise it is deleted.
  - **B** (B ÷ A). (a) On the EPYC the geometric mean over the exit check's cases of the
    overridden lengths (every batch and direction in the set) is at most 0.97 with its
    interval clear of the band. (b) No case of the judged set or the EPYC's tracking tier
    is slower than 1 + band + 2 % (re-run once as above). (c) A length with any case slower
    than 1 + band loses its override before adoption.
  - An arm that fails is reported with its numbers and removed; at most three arms.

## Results

(running)

## Verdict

(running)
