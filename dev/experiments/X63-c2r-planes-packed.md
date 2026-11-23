# X63: out-of-place c2r planes with the intermediate packed in the output

- **Status:** done (2026-11-22): adopted on the Xeon 6240 class (`avx512`) and Zen 3 (`avx2`)
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

## Amendment 1: single-precision planes (written 2026-11-23, after the second reader, before their runs)

The rule moves float planes too (`c2r_packed` counts a float half spectrum at half the bytes),
but every judged case above is double precision. This amendment adds them; nothing above
changes.

- Arms: M = `main` c6ebf02 and A = 1f0b17b (the library of the PR's head; later commits change
  tests, comments and documents only), each run with the identical-arm control of M.
- Cases: `bench/catalogues/x63f.toml`, 39 float out-of-place c2r planes (`x63.toml`'s and five
  larger) and two float volumes. Judged: the float planes whose half spectrum (rows × (n/2 + 1)
  × 8 bytes) exceeds the class's threshold:
  - Xeon 6240 (`avx512`, above 1 MiB): 34 planes, all but 256², 384², 448², 36 × 2048 and
    2048 × 36;
  - EPYC 7763 (`avx2`, above 16 MiB): 2048², 4096 × 1024, 1024 × 4096, 3000 × 2000, 2560²,
    2048 × 4096, 4096 × 2048, 3200², 4096² (9 planes).
- Four judged float runs: each class at one thread and at 8 lanes (the Xeon pinned as above).
  Criteria (1) and (2), the 8-lane criterion and the re-run clause are those of the rule above,
  word for word, applied to each float run on its own; the collateral check within each run
  likewise (the planes at or below the threshold and the volumes).
- Action: a class whose float runs both pass keeps the route for float planes. A class whose
  float runs fail keeps it for double only: the planner then packs float planes on that class
  never (a profile field that limits the route to double, with a test), and the double verdict
  above stands.

## Results (2026-11-22)

A ÷ M (M = `main` c6ebf02, A = 1f0b17b) over the judged cases; band from the identical-arm
control of M in the same run. No run needed the re-run clause.

| run | machine, threads | load | band | judged | geometric mean | worst judged case | verdict |
|---|---|---|---|---|---|---|---|
| 20261122-222557-06db | EPYC 7763, 1 | ~25 other busy cores, every round | ±1.00 % | 9 | **0.837** | 2560² 0.987 [0.972, 1.064] | passes |
| 20261122-222557-5494 | EPYC 7763, 8 lanes | ~25 other busy cores, every round | ±2.51 % | 9 | **0.767** | 1024 × 4096 0.865 [0.834, 0.891] | passes |
| 20261122-222653-8557 | Xeon 6240, 1 | 4 of 20 rounds loaded | ±2.34 % | 31 | **0.807** | 512² 0.958 [0.940, 0.968] | passes |
| 20261122-224320-1482 | Xeon 6240, 8 lanes, pinned | 6 of 20 rounds loaded | ±2.20 % | 31 | **0.824** | 512² 0.984 [0.973, 1.011] | passes |

- **EPYC 7763** (GCC 16, `avx2`, DMOG; Slurm 515713 and 515715, builds 515712, 515714): at one
  thread 1536²–2048² 0.91–0.94, 4096 × 1024 0.46, 1024 × 4096 0.72, 2560² 0.99 (no evidence);
  at 8 lanes 0.72–0.87, every judged case faster beyond noise. The planes at or below 16 MiB
  (X60's 1000², 1024², which lost at 8 lanes in X60, among them) and the two volumes kept M's
  plan: nothing slower beyond 1.05 × (1 + band) in either run.
- **Xeon 6240** (GCC 14.4, `avx512`; Slurm 16822 and 16823, build 16821; the 8-lane run confined
  to node 0, CPUs 2–5, 10, 14, 16, 17): one thread 0.69–0.96 (2048 × 320 0.69, 1000² 0.72,
  3000 × 1000 0.73), 8 lanes 0.61–0.98 (2048 × 320 0.61, 1024² 0.72, 960² 0.75); the widest planes
  gained too (320 × 2048 0.89 and 0.94, 1024 × 4096 0.79 at both). 256², 36 × 2048, 2048 × 36 and
  the volumes (64³, 128³), which keep the block, 0.98–1.02.
- **Held-out sizes** (the 12 of `x63.toml` no run had timed, inside the judged sets): EPYC 1600²,
  2000², 1500 × 2000, 3000 × 1000, 4096 × 1024, 1024 × 4096 and 2560² 0.46–0.99 at one thread and
  0.72–0.87 at 8 lanes; Xeon all twelve 0.61–0.94 in both runs.
- **Collateral, tracking tier, A ÷ M, one thread**: EPYC run 20261122-222557-0c83 (Slurm 515717,
  ±1.98 %) 1 faster (c2r 1000, a row A does not change), 60 no evidence; Xeon run 20261122-225712-05cd (Slurm 16824, ±3.37 %, 15 of 16 rounds loaded) 2 faster (c2r 1024² 0.70 and 480 × 640 0.85, both judged planes), 59 no evidence. Nothing slower.

## Verdict (2026-11-22)

- **Adopted on the Xeon 6240's class (`avx512` plans)**: c2r planes out of place pack their
  intermediate from a half spectrum above the L2 (1 MiB). Both judged runs pass (0.807 at one
  thread, 0.824 at 8 lanes pinned, over 31 planes; nothing slower), held-out sizes included.
- **Adopted on Zen 3's class (`avx2` plans)**: from a half spectrum above half the CCX's L3
  (16 MiB). Both judged runs pass (0.837 and 0.767 over 9 planes, all held out but 1536² and
  2048²).
- **Volumes keep the block** everywhere (X60's 8-lane losses on the Xeon), and so do Apple (X60's
  wide planes) and every unmeasured class.
- **Reach** (corrected after the second reader, who ran `set_grid_sizes` on synthetic
  classes). Unmeasured classes that the keys take:
  - **Zen 2** (server and desktop: 512 KiB of private L2, 16 MiB of L3 per 4-core CCX, at least
    2 MiB per CPU) matches Zen 3's key and packs on `avx2` from L3/2 = 8 MiB. That the threshold
    follows the CCX's L3 rather than a fixed byte count is the physical argument for it: the
    packed route pays where input, block and output pass the L3 that one CCX's cores share (and
    at 8 lanes the team shares one CCX's L3), so on a CCX half the size the turn should come at
    half the bytes. Zen 2 has the same `avx2` target and 16 registers, so the reach rule allows
    it, but it is unmeasured.
  - **The Xeon 6240's key** (x86-64, a private L2 under 2 MiB, under 2 MiB of L3 per CPU) packs
    on `avx512` from the L2 on Ice Lake-SP, Zen 4c with SMT on (1 MiB of L2; with SMT off its L3
    per CPU reaches 2 MiB and it keeps the block), Zen 4 VMs that report a small L3 per CPU,
    Intel client parts with AVX-512 (Ice Lake and Tiger Lake laptops, Rocket Lake), and any
    class that reports no L3.
  - Classes that keep the block: Zen 3 APUs such as Cezanne (they take the Xeon key, 1 MiB of L3
    per CPU, but their target is `avx2`, which `c2r_packed_isa` refuses), Google Cloud's Ice Lake
    N2 VMs (their large reported L3 per CPU files them with neither key), Zen 4 and Zen 5 with
    their large L3, Granite and Sapphire Rapids (2 MiB of L2), Apple, other AArch64.
  - **For the next phase exit's screen**: Zen 2 (`avx2`, 8 MiB), Ice Lake-SP, Zen 4c with SMT,
    Zen 4 VMs with a small reported L3, Intel client AVX-512 parts and a machine that reports no
    L3 (`avx512`, from the L2).
- **Not measured**: whether a threshold below 16 MiB would also pass on Zen 3 (X60 timed nothing
  between 8 and 32 MiB at 8 lanes); the Mac (Apple keeps the block, by X60).
- **Code.** `c2r_packed` (plan_core.cpp) takes planes only; `machine_profile::c2r_packed_bytes`
  is SIZE_MAX by default and `c2r_packed_isa` holds each class to its measured target. DESIGN 9.1
  and the decision log describe the route; `docs/conventions.md` promises only that the input is
  not written.
- **Tests**: `Columns.C2rPackedMatchesTheBlock`, `Columns.C2rPackedStridedLayouts` (fails with
  `in.stride[r - 1]` dropped from column h's start or the rows' block step zeroed; checked by
  mutation), `Columns.C2rPacksBeyondTheProfilesBytes` (thresholds, the target restriction,
  volumes never pack), team case `c2r_planes_strided_6x16x3`, `Schedule.GridSizesFromCaches`.
  Mac Homebrew Clang 23 Release, all 2549 tests; Mac GCC 16 build with `-Werror` and its
  targeted suites.
