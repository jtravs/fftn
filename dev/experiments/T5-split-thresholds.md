# T5: thresholds for splitting a transform, by cache-group type

- **Status:** done (2026-11-14)
- **Owner:** lead, 2026-11-14 (protocol and tooling prepared with a sub-agent)
- **Phase / task:** 3.4

## Question

From what size does sharing one transform among the members of a cache group beat
running it whole, with how many points per member, from what size are leftover
transforms worth splitting, and how many points should a chunk hold, on each kind of cache
group (a Zen 3 CCX of 8 cores, the Xeon's 18-core mesh, an Apple performance cluster)?
The answer sets the machine profile's `split_min_points`, `split_min_points_mesh`,
`mesh_group_lanes`, `min_points_per_member` and `remainder_split_min_points`
(`src/threading/schedule.hpp`, rules 2 and 3 of DESIGN.md 13.4) and the plan's
`chunk_points` (`src/core/plan_core.hpp`).

## Hypothesis

The provisional values are the prototype's (`schedule.hpp` cites each): a single row paid
for a team from about 2048 points inside a Zen 3 CCX and about 8000 on the Xeon's 18-core
mesh; at least 512 points per member (3072 points on 18 lanes took 13 µs against 5.7 µs
serial); leftover rows gained 0.71–0.90 from sub-teams from 8192 points and lost 1.04–1.21
at 3072. FFTN's engine claims chunks and counts phases instead of meeting at barriers, so
a phase hand-off should cost less than the prototype's barrier, which would lower the
thresholds; its SIMD passes (task 1.2) do less work per point than the walking skeleton
but not less than the prototype's kernels, which leaves them where they were. Expected:
thresholds within a factor of two of the provisional ones, lower on the CCX and the Apple
cluster than on the mesh. `chunk_points` (2048) is expected to matter little between 1024
and 8192 points.

These thresholds belong to the transforms split today: the chain's chunked stages and the
lines of planes and volumes. Task 3.6 (the blocked grid for long rows) changes the phases
of long rows, so T5 is re-run on those shapes when 3.6 lands.

## Change under test

One threshold at a time, set through the FFTN backend's profile options (`bench/README.md`,
"FFTN options"; they reach the library only through the private plan core's build
settings, as a test's do, so `dev/KNOBS.toml` lists no switch). Each sweep is one fbench
run per machine and width, with the identical-arm control:

- **S1, split threshold.** Arm N never splits (`SPLIT_MIN_POINTS=never
  SPLIT_MIN_POINTS_MESH=never`); arm Y splits whenever rule 3's member cap allows
  (`SPLIT_MIN_POINTS=1 SPLIT_MIN_POINTS_MESH=1 SYNC_NS=0`, so that the cost model does not
  veto it); arm D the defaults. Arm Y is a measuring arm, not a candidate default.
- **S2, points per member.** With splitting forced as in Y, `MIN_POINTS_PER_MEMBER` 256,
  512 (A), 1024, 2048 and 4096: the member cap of each length. The cap is also bounded by
  the widest phase's chunks, which `chunk_points` (2048) sets: a 4096-point row has two
  chunks per stage and so never more than two members (`bench/t1/t1.py schedule` shows
  it). S2 therefore runs with `CHUNK_POINTS=256` in every arm, and S4 decides the chunk
  size separately.
- **S3, leftover transforms.** Arm N `REMAINDER_SPLIT_MIN_POINTS=never`, arm Y
  `REMAINDER_SPLIT_MIN_POINTS=1 SYNC_NS=0`, arm D the defaults.
- **S4, chunk size.** `CHUNK_POINTS` 512, 1024, 2048 (A), 4096 and 8192, at one thread
  (`THREADS=1`: whether chunking costs a serial execution anything) and at full width.

## Method

- Cases: the `threads` catalogue (`bench/catalogues/threads.toml`), quick tier, with
  filters per sweep: S1 and S2 on single rows (c2c 2¹⁰–2²⁰ and the smooth lengths among
  them, r2c 2¹²–2²⁰), few rows (batches of 2–4) and single planes and volumes; S3 on the
  `rows-remainder` group (batches of lanes + 1, 2048–65536 points); S4 on single rows and
  planes. The `rows-many` and `small` groups are the collateral check: they must not
  change.
- Executor: `fftn_omp` (the compiler's OpenMP runtime, `OMP_PLACES=cores
  OMP_PROC_BIND=close`), the way the client application runs; the pool once per machine
  at full width as a check. Widths: 8, 16, 32 and 64 threads on the EPYC 7763 (an
  exclusive Slurm node, one socket; groups of 8), 18 and 36 on the Xeon Gold 6240 (groups
  of 18, the mesh), the performance cores on the Apple M5 Max (one cluster), and one thread
  as the serial reference in every run (`S=fftn_omp+THREADS=1`).
- Compilers: GCC 15 (libgomp) on Linux; Clang 23 (libomp) on the Mac and, when available,
  on the Xeon. Release builds.
- Metric: per case, the median paired ratio over process rounds of each arm to A (or of Y
  to N), with its 95 % bootstrap interval and the run's noise band.

**Decision rule, written before running.** Per cache-group type:

1. **Split threshold** (S1): the smallest catalogue length from which Y ÷ N is below
   1 − band, with the interval clear of the band, for that length and every longer one.
   It sets `split_min_points` (CCX, Apple cluster) or `split_min_points_mesh` (groups of
   more than `mesh_group_lanes` members). If the crossover lies within one catalogue
   length step of the provisional value, the value stays.
2. **Points per member** (S2): the value whose weighted geomean over the S1 cases at and
   above the threshold is lowest; among values within the noise band of the lowest, the
   largest (fewer members per transform, spare lanes left for other work).
3. **Leftover threshold** (S3): as rule 1, on the remainder cases, for
   `remainder_split_min_points`.
4. **Chunk size** (S4): the default changes only if a value beats 2048 by more than the
   band on the weighted geomean at full width and costs nothing at one thread (every case
   within the band).
5. `mesh_group_lanes` stays at 8 unless a measured machine has groups of 9–17 members whose
   crossover sits with the other kind.
6. **Adoption.** The new values together (arm B) against the provisional ones (arm A) on
   the whole quick tier at every measured width: adopted if the weighted geomean ratio is
   at or below 1.00 (interval clear of the band or inside it), no case is slower than A
   by more than the band plus 2 %, and no case is slower than serial by more than 2 % (the
   phase 3 exit). The adopted values go into the profile with a comment citing this file
   and the run IDs.

## Results

**First campaign (DMOG job 508943, commit 148b0d6).** Its sweeps ran before the scheduler
fixes 5a1cd72 and fb91770 and before the grid strategy (task 1.7) and the radix-32/64
codelets changed every row above 2048 points (Zen 3's L1 is 32 KiB, so c2c rows from
4096 points are grids now). Its never-split arm N was handicapped: a lone unsplittable row
was still dispatched to a whole group, so N ran 1.05–1.31 times slower than serial on long
rows (run 20261114-111309-26b4) and every Y ÷ N in it overstates splitting. None of its
fbench results is used; all four sweeps were re-run at ec63bb3 (DMOG job 509111; the DMOG
and Xeon run records of this campaign carry the label 62958fc, an ancestor, set by mistake).

Runs (fbench, `fftn_omp` unless stated, quick tier; band = the run's A/A noise band):

| Run | Machine, width | Sweep | Band |
|---|---|---|---|
| 20261114-175237-ee55 | EPYC 7763, 8 lanes (one CCX), GCC 15.2 | S1 | 1.16 % |
| 20261114-180642-f917 | EPYC 7763, 64 lanes (8 CCXs) | S1 | 1.86 % |
| 20261114-182042-1f0c | EPYC 7763, 8 lanes | S3 | 1.00 % |
| 20261114-182347-49bf | EPYC 7763, 64 lanes | S3 | 1.84 % |
| 20261114-182645-10c9 | EPYC 7763, 8 lanes, `CHUNK_POINTS=256` | S2 | 1.08 % |
| 20261114-184234-4a75 | EPYC 7763, 1 thread | S4 | 1.92 % |
| 20261114-190017-fd3d | EPYC 7763, 8 lanes | S4 | 1.39 % |
| 20261114-180925-7c39 | Xeon 6240, 18 lanes (one socket, one mesh group), GCC 14.4, shared (load 3–18) | S1 | 1.81 % |
| 20261114-174818-ee78 | M5 Max, 6 lanes (one cluster), `fftn` (pool) and one `fftn_omp` arm, Clang 23.1, loaded (load 7–9) | S1 | 3.69 % |

The EPYC runs had cores 0–63 of one socket (`-c 64 --distribution=block:block`), the
OpenMP threads `OMP_PLACES=cores OMP_PROC_BIND=close`; loads 1.6–5.

**S1, split threshold.** Y ÷ N (splitting whenever rule 3's cap allows, against never),
selected lengths:

| Case | EPYC 8 | EPYC 64 | Xeon 18 | Mac 6 |
|---|---|---|---|---|
| c2c 3072 | 0.94 | 1.68 | 2.67 | 1.04 |
| c2c 4096 | 0.78 | 1.33 | 2.14 | 1.17 |
| r2c 4096 | 1.52 | 2.37 | 3.99 | 2.01 |
| c2c 6144 | 0.53 | 0.82 | 1.17 | 1.00 |
| c2c 8192 | 0.42 | 0.61 | 1.00 | 0.84 |
| c2c 10240 | 0.33 | 0.47 | 0.88 | 0.42 |
| c2c 16384 | 0.23 | 0.31 | 0.62 | 0.42 |
| r2c / c2r 16384 | 0.47 / 0.48 | 0.63 / 0.65 | 1.11 / 1.17 | 0.79 / 0.83 |
| c2c 4096 × 2 rows | 0.68 | 0.81 | 1.00 | 1.41 |
| c2c 20480 and longer, planes, volumes | 0.14–0.28 | 0.15–0.34 | 0.08–0.48 | 0.17–0.47 |

Rule 1 (the smallest length from which every S1 case of that length and every longer one
gains beyond the band): EPYC 3072 at 8 lanes but 6144 at 64 (a lone row's split pays the
64-thread executor's region, 35 µs, which the unsplit row, run on the caller, does not);
both widths together, 6144. Xeon mesh: c2c gains from 10240, but r2c and c2r of 16384
lose 1.11–1.17, so 20480. Mac cluster: 8192 (6144 inside the band, 4096 and r2c 4096
lose). Each is more than one catalogue step from the provisional value (2048 and 8192),
so all three were candidates for rule 6: `split_min_points` 6144 (CCX), 8192 (Apple
cluster), `split_min_points_mesh` 20480 (which rule 6 rejected, below). Few rows of 4096 points, which pay the dispatch anyway,
gained from splitting on the EPYC (0.68 at 8 lanes, 0.81 at 64) and now run whole; the
cost model does not see that a lone row split pays a dispatch that the same row whole does
not, which is what makes the threshold depend on the executor's width (follow-up below).
Planes and volumes split by lines gain 3–13 times at every width.

**S2, points per member** (8 lanes, splitting forced, 256-point chunks so that the member
cap binds): the weighted geomean over the 35 S1 cases of 6144 points and more, against
512: 256 1.003, 1024 1.001, 2048 1.028, 4096 1.141. 512 is lowest; 1024 is within the
band (1.08 %) and the largest such, so `min_points_per_member` becomes 1024 (rule 2). With
the default 2048-point chunks the widest phase caps members before 1024 points per member
does on every catalogue length, so the change has no effect on the catalogue's plans.

**S3, leftover transforms** (Y ÷ N on batches of lanes + 1): at 8 lanes leftovers of 4096
points gain 0.85 (9 rows) to 0.98 (65 rows), of 8192 and more 0.60–0.96, of 2048 nothing
(one chunk per stage, never split); at 64 lanes nothing outside the band but 8192 × 65
(0.89). The crossover, 4096, is one catalogue step from the provisional 8192, so
`remainder_split_min_points` stays 8192 (rule 3).

**S4, chunk size** (512 to 8192 points against 2048): at one thread every arm is within
the band (geomeans 0.997–1.004; one noisy case, c2r of 262144 points, 1.24 at 4096 with
an interval reaching 1.005): chunking costs a serial execution nothing. At 8 lanes (one CCX; the protocol's full width
adds nothing, since a transform is split only inside one group; provisional thresholds) 2048 is
best: 512 1.069, 1024 1.009, 4096 1.077, 8192 1.222. Smaller chunks widen the split of
rows of 3072–6144 points (0.71–0.82 at 512 and 1024) but cut planes' lines too fine
(256² and 512² planes 1.9–2.7); larger chunks help 1024² planes (0.73–0.76 at 4096 and
8192) and starve rows of 8192–16384 points of members (1.5–2.6). Rows and lines would like
different chunk sizes; one value serves both best at 2048.

**Adoption (rule 6).** Whole quick tier (82 cases), 15 rounds, against A (provisional) and
serial (S), DMOG jobs 509162 and 509163 at ec63bb3. Arms: B = T5's values (split 6144,
1024 points per member); F = B with T1's adopted fields (`sync_ns` 1100,
`claim_divisor` 1), the profile as committed; F4 = F with a 4096-point threshold; F6 = F
with `inline_multiple` 6 (T1).

| Run (band) | B | F | F4 | F6 |
|---|---|---|---|---|
| 20261114-185256-4009, 8 lanes (2.36 %) | 1.017 [1.012, 1.023] | 1.013 [1.008, 1.018] | 1.000 [0.991, 1.008] | 1.012 |
| 20261114-185256-431c, 64 lanes (2.61 %) | 1.002 [0.993, 1.010] | 0.982 [0.976, 0.987] | 0.976 [0.970, 0.983] | 0.982 |
| 20261114-191255-5a9c, pool, 64 lanes (7.39 %) | | 0.971 [0.962, 0.987] | | |

Cases beyond the band plus 2 %, against A: F (and B, F6) lose on rows of 4096 points that A
split profitably: few rows (2–4) 1.44–1.46 at 8 lanes and 1.23–1.29 at 64, a lone row
1.27 at 8, and a lone 3072-point row 1.06 at 8. F4 avoids the first two but leaves a lone
4096-point row 1.38 times slower than serial at 64 lanes (A: 1.37, and 3072 points 1.74).
Against serial, F is never slower beyond 2 % at either width with the OpenMP executor
(worst c2c 256 × 8 at 64 lanes, 1.026, inside the band); F4 is, at 64 lanes. With the pool
at 64 lanes F removes the lone 3072- and 4096-point rows' losses (A: 1.48 and 1.23 times
slower than serial) but r2c and c2r of 16384 points stay 1.23 and 1.17 times slower than
serial, as under A: the pool's 64-lane dispatch (26 µs) against a 56 µs transform, the
same width effect; its worst case against A is c2c 4096 × 1024 at 1.108 [1.057, 1.199]
(band 7.4 %).

Mesh (Xeon, run 20261114-182732-001f, 18 lanes, band 2.40 %): the 20480-point threshold
from S1 (arm M) lost 1.047 [1.032, 1.060], with planes and few rows of 16384 points
1.5–2.4 and leftovers of 8192–16384 points 1.4 times slower unsplit; rejected, 8192
stays. Apple cluster (run 20261114-181503-ace8, 6 lanes, pool, loaded, band 3.92 %): the
8192 threshold alone (SP) 0.995 [0.973, 1.044], no case slower beyond the band; adopted.

## Verdict

Done. Values in `src/threading/schedule.hpp` (each field cites its runs) and
`profile_for` (commit bbfc4b3):

- `split_min_points` 6144 on a Zen 3 CCX (the default; was 2048) and 8192 on an Apple
  performance cluster. No threshold passes every condition of rule 6 at both widths: the
  crossover moves with the executor's width (3072 points at 8 lanes, 6144 at 64), because
  splitting a lone row adds a dispatch that the same row run whole on the caller does not
  pay, and the cost model does not know it. 6144 was chosen over 4096 because it is the
  only value with no case slower than serial at either width with the OpenMP executor (the
  phase 3 exit), at the price of rows of 4096 points (few rows up to 1.46 slower than A's
  split at 8 lanes, still faster than serial). With the pool at 64 lanes, real transforms
  of 16384 points remain 1.2 times slower than serial, as before. This choice puts the phase 3 exit before rule 6's per-case
  condition, which the maintainer should confirm; the follow-up below removes the conflict.
- `split_min_points_mesh` stays 8192: S1's 20480 failed rule 6 on the Xeon.
- `min_points_per_member` 1024 (was 512), rule 2; no effect on today's plans with
  2048-point chunks.
- `remainder_split_min_points` stays 8192 (crossover 4096, one catalogue step).
- `chunk_points` stays 2048 (rule 4): no value beats it at 8 lanes, and none costs anything
  at one thread.
- `mesh_group_lanes` stays 8 (rule 5): no measured machine has groups of 9–17.

Follow-up (shared with T1, DESIGN 18.2 item 22): a dispatch cost a + b × lanes, used by
rule 3 when splitting a lone transform causes a dispatch; then a single threshold near
3072 points should serve every width, few rows of 4096 points split again, and real
transforms on the mesh (points counted as n, half a complex transform's work) can be
priced rather than thresholded. Re-run S1 and rule 6 on the EPYC, the Xeon and a quiet Mac
then. Lessons 5.1 and 5.2 in `dev/LESSONS.md`.
