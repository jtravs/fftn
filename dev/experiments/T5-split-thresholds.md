# T5: thresholds for splitting a transform, by cache-group type

- **Status:** planned
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
  512 (A), 1024, 2048 and 4096: the member cap of each length.
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

None yet.

## Verdict

Pending.
