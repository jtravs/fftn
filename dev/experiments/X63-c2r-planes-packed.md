# X63: out-of-place c2r planes with the intermediate packed in the output

- **Status:** running (2026-11-22): arm built, decision rule written before the judging runs
- **Owner:** X63 agent, 2026-11-22 (wave 1; the maintainer approved the follow-up)
- **Phase / task:** 4.1 (planes), task 2.1's c2r; follows [X60](X60-c2r-two-arrays.md)

## Question

X60's packed intermediate won at one thread on the Xeon 6240 (0.82) and the EPYC 7763 (0.91),
and lost at 8 lanes: the Xeon's volumes (96³–128³ 1.05–1.09, its teams run volumes plane at a
time) and the EPYC's 8-MiB planes (1000², 1024² 1.04–1.13). On the Xeon the planes gained at 8
lanes too (0.85 over 12, none slower beyond noise, in both runs). Does the packed route for
planes alone, on each x86 class above a threshold that X60's screens support, pass at one thread
and at 8 lanes?

## Hypothesis

The packed route removes the third array (input + block + output), which costs where it passes
a cache level that input and output fit (X60's diagnosis: the Xeon's c2r 1024² out of place took
2.0 times its in-place time). Its cost is the output's power-of-two row strides, which lose inside
the caches. Volumes add a team cost on the Xeon that planes do not show. On the EPYC at 8 lanes
the team shares the CCX's 32 MiB L3, so the gain needs a half spectrum well past the 8 MiB where
X60 lost: at 32 MiB (2048²) the packed route took 0.76–0.83 of the block's time at 8 lanes.

## Change under test

- **Arm A** (1f0b17b, branch `exp/x63-c2r-planes`). X60's packed route,
  restored from a983841 with the second reader's fixes and tests, restricted to planes:
  `c2r_packed` (plan_core.cpp) takes only two axes, so volumes keep the block even when forced.
  `machine_profile::c2r_packed_bytes` is SIZE_MAX by default and `c2r_packed_isa` names the one
  target a class's threshold applies to:
  - the Xeon 6240's class (x86-64, a private L2 under 2 MiB, under 2 MiB of L3 per CPU): the L2
    (1 MiB), `avx512` plans only, X60's threshold for that class (from its screen; every plane
    above it gained at one thread and at 8 lanes in X60's runs);
  - Zen 3's class (the EPYC 7763: a private L2 under 1 MiB, a large L3 per CPU): half the CCX's
    L3 (16 MiB), `avx2` plans only. X60's 8-lane screen and runs lost at 8 MiB (1000², 1024²) and
    gained at 32 MiB (2048²); its one-thread runs gained from 7.6 MiB. No size between 8 and 32
    MiB was timed at 8 lanes, so the threshold is the lowest that X60's data allow for both;
  - Apple and every other class: the block (X60's Apple losses, 512 × 1024 and 640 × 1280, were
    planes).
- **Arm M**: `main` (c6ebf02, X60's record merged; the block everywhere).

Bits differ from the block route in the last places, which `docs/conventions.md` allows
("between plans built with different options"); they are the same for every team size (tested).

## Method

- Cases: `bench/catalogues/x63.toml`: 34 out-of-place c2r planes (X60's screen and held-out
  planes and 12 sizes no run has timed) and two volumes. The header lists each case's half
  spectrum.
- Machines, compilers, instruction sets:
  - EPYC 7763 on DMOG: GCC 16, `avx2`, `fbench_remote.py submit epyc` (Slurm binds 8 CPUs);
  - Xeon 6240: GCC 14.4, `avx512`, `fbench_remote.py submit xeon --serial`, the job files
    edited to run fbench under `numactl -m` and `taskset` on eight idlest whole cores of the
    idlest socket and its memory, one CPU per core, chosen once at the job's start (X59; X60's
    Slurm 16804 and 16807).
- Thread modes: one thread and 8 lanes (`THREADS=8`) on each machine.
- Metric: R ÷ M per case, the median of paired process rounds with its 95 % interval; the band is
  fbench's identical-arm control of M in the same run.

## Decision rule (written 2026-11-22, before any judging run)

- **Judged cases** (the cases arm A moves, the planes whose half spectrum exceeds the class's
  threshold; every other case keeps M's plan):
  - Xeon 6240 (`avx512`, above 1 MiB): the 31 planes of `x63.toml` other than 256², 36 × 2048
    and 2048 × 36.
  - EPYC 7763 (`avx2`, above 16 MiB): 1536², 1600², 2000², 1500 × 2000, 3000 × 1000, 2048²,
    4096 × 1024, 1024 × 4096, 2560² (9 planes).
- **Four judged runs**: each class at one thread and at 8 lanes. A class passes only if both its
  runs pass. A run passes if
  1. the geometric mean of R ÷ M over its judged cases (equal weights) is below 0.97, and
  2. no judged case is slower beyond noise: none has its whole 95 % interval above
     1.02 × (1 + band).
- **The 8-lane criterion is the same as the one-thread criterion**: (1) and (2) above, over the
  same judged cases, with that run's own band.
- **Re-run clause.** A run may be re-run once, with the same cases and arms, only when (a) it
  fails criterion (2) through cases whose intervals' lower ends all lie within 0.010 of the limit,
  and (b) at least one of its rounds was loaded. The re-run replaces the run for both criteria,
  and its result is final. Criterion (1) never allows a re-run: a run whose geometric mean is
  0.97 or above fails, however close, and a re-run that a hair-past case prompted is judged on
  (1) as well.
- **Collateral check**: the standard catalogue's tracking tier, M against A, one thread, on
  each machine. A case outside the judged set slower beyond 1.05 × (1 + band) blocks adoption on
  that machine's class unless its plan is unchanged (describe() identical); then it is recorded
  as placement. In each judged run the cases outside the judged set (planes at or below the
  threshold, the two volumes) must show no case slower beyond 1.05 × (1 + band) either.
- **Action, per class**: a class that passes keeps arm A's threshold; a class that fails gets
  the block (its value back to SIZE_MAX). If no class passes, the packed route is deleted and
  only this file, its PLAN row and a LESSONS note remain.

## Results

(Judging runs to follow.)

## Verdict

(After the judging runs.)
