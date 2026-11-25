# X83: single transforms on the Xeon 6240's mesh, team width

- **Status:** running
- **Owner:** x83 session, 2026-11-25
- **Phase / task:** 3 (scheduling rules, DESIGN 13.4; follow-up of [X77](X77-real-rows-threaded.md))

## Question

On the Xeon 6240, single real rows of 18432-32768 points gain little or nothing from a team of 4
(X77's diagnosis: time at 4 threads ÷ time at 1 thread 0.76-1.01 under today's plans, against
0.38-0.52 on the EPYC 7763), and the client check had 16384-point rows at 4-8 threads 1.07-1.16
of FFTW. Where does a team of 2 or 4 stop paying for one transform of 4096-65536 points on the
Xeon's mesh, and which scheduling rule (a split threshold, or a cap on the members per point
count) makes FFTN use fewer members where a team is not worth it? The maintainer agreed
(2026-11-25) to fix this in FFTN rather than in the client's width rule.

## Hypothesis

The scheduler splits a single transform from 6144 points among up to points / 1024 members
(`split_min_points`, `min_points_per_member`; DESIGN 13.4 rule 3). T5 calibrated those on the
EPYC's CCX; on the Xeon its S1 put the crossover for rows at 20480 points, but that threshold was
judged over a catalogue whose planes needed the low one (they now have `split_min_points_lines`).
On the mesh, each phase boundary moves a transform's lines between private L2s through an L3
slice far away (T1: 2846 ns per phase boundary among 18 members, against 1141 on the CCX), and
today's depth-first chains have few phases with only 16-20 sub-rows each. So a team should pay
only from a larger size per member than on the CCX, and the cost model's `sync_ns` (700 on the
mesh) understates it.

## Change under test

Scheduling only: the fields compared are read by the scheduler (`share_of`,
src/threading/schedule.cpp) and by nothing that builds a plan, so every arm runs the same plans
and computes the same bits (DESIGN 13.2); at one thread nothing can move. The rule is written in
"Decision rules" below from the diagnosis, before the judging runs.

## Cases

Catalogue `x83` (bench/catalogues/x83.toml), 52 single transforms: the client catalogue's single
transforms of 4096-65536 points (c2c 16384 forward and 4608, 5120, 16384, 18432 backward; r2c and
c2r 6400, 9216, 10240, 18432, 32768; c2r 36864) and the standard catalogue's single rows of that
range (21 c2c rows out of place, 65536 backward, r2c and c2r 4096, 10000, 16384, 65536, and c2c,
r2c, c2r 4096 and 65536 in place).

## Method

- **Xeon 6240** (2 × 18 cores, Cascade Lake, `avx512`), GCC 14.4 Release, double, fbench hot mode,
  statistic median, process rounds, A/A control, `--min-band 0.01`. Slurm through
  `fbench_remote.py submit --serial` with X74's pinning setups (the client check's profiles):
  `xeoncc8` (`-c 16`, `numactl -m <node> -C <cpus>` on the 8 idlest whole cores of the least busy
  socket) for widths up to 8, `xeoncc16` (`-c 32`, 16 cores) for 16; threads unpinned within that
  set; never `--exclusive`. The Xeon's load (`uptime`, `squeue`) is recorded before each job.
- FFTN arms are `fftn_omp` (the `omp::executor` on libgomp), `THREADS=k` for a team of k.

### Diagnosis (informative, before the rule)

One run on 8b0a300 (`main` a696bfd plus the profile field `few_points_per_member`, unset, and
fbench's `FEW_POINTS_PER_MEMBER` option that sets it), arms:

- **S**: serial (`fftn_omp`, no executor), the baseline;
- **T2**, **T4**, **T8**: `THREADS=2`, `4`, `8` (today's schedules);
- **G2**: `THREADS=4+GROUP_SIZE=2`: an executor of 4 lanes reporting groups of 2, so a single
  transform goes to one group and is split between 2 members while the other 2 are not
  dispatched (DESIGN 13.4, rule 4): a team of 4 that uses 2;
- **P4a**, **P4b**, **P4c**: `THREADS=4+FEW_POINTS_PER_MEMBER=4096`, `8192`, `16384`: a single
  transform split among at most points / 4096 (8192, 16384) members, all 4 dispatched (the others
  wait); against G2 on the cases P4b splits between 2 (16384-24575 points), what the waiting
  members cost;
- **P8b**, **P8c**: `THREADS=8+FEW_POINTS_PER_MEMBER=8192`, `16384`.

Metric: each arm's time ÷ S per case. A case's crossover is the smallest team width that is not
faster than a narrower one beyond the band.

## Diagnosis results

Run 20261125-104559-e6f5 (Slurm 17105, Xeon socket 1 cores 18-25, `xeoncc8`, 20 rounds, none
loaded, load 3-4, band ±1.00 %; A/A median deviation 0.3 %). Every output correct. Time ÷ S
(one thread), medians over rounds:

| case | S µs | T2 | T4 | G2 | P4b | T8 |
|---|---|---|---|---|---|---|
| r2c 6400 | 6.9 | 1.02 | 2.46 | 1.02 | 1.02 | 2.69 |
| c2r 6400 | 6.5 | 1.67 | 1.82 | 1.72 | 1.02 | 2.02 |
| r2c 9216 | 10.0 | 1.93 | 1.86 | 1.98 | 1.01 | 1.94 |
| c2r 9216 | 9.4 | 1.59 | 1.43 | 1.62 | 1.02 | 1.55 |
| r2c 10000 | 11.2 | 1.83 | 1.76 | 1.87 | 1.02 | 1.89 |
| c2r 10000 | 12.1 | 1.50 | 1.35 | 1.52 | 1.02 | 1.47 |
| r2c 10240 | 11.5 | 1.86 | 1.72 | 1.91 | 1.01 | 1.76 |
| c2r 10240 | 10.9 | 1.54 | 1.32 | 1.56 | 1.02 | 1.44 |
| r2c 16384 | 19.9 | 1.54 | 1.17 | 1.58 | 1.61 | 1.23 |
| c2r 16384 | 21.9 | 1.21 | 0.90 | 1.24 | 1.25 | 0.93 |
| c2c 6561 | 19.4 | 1.24 | 0.93 | 1.25 | 1.01 | 0.94 |
| c2c 7200 | 12.5 | 1.66 | 1.22 | 1.69 | 1.00 | 1.32 |
| c2c 8192 | 15.7 | 1.50 | 1.10 | 1.51 | 1.01 | 1.20 |
| c2c 8191 | 74.4 | 1.24 | 1.11 | 1.26 | 1.00 | 1.14 |
| c2c 10000 | 17.8 | 1.33 | 1.13 | 1.38 | 1.01 | 1.00 |
| c2c 12000 | 24.5 | 1.33 | 1.07 | 1.33 | 1.00 | 0.82 |
| c2c 15625 | 62.6 | 1.06 | 0.84 | 1.09 | 1.01 | 0.57 |
| c2c 16807 | 72.9 | 0.92 | 0.69 | 0.92 | 0.96 | 0.51 |
| r2c 32768 | 41.2 | 1.22 | 0.93 | 1.26 | 0.93 | 0.67 |
| c2r 32768 | 46.3 | 0.88 | 0.63 | 0.92 | 0.63 | 0.45 |
| c2r 36864 | 49.2 | 0.93 | 0.69 | 0.97 | 0.68 | 0.52 |
| c2c 32768 | 84.8 | 0.81 | 0.60 | 0.85 | 0.60 | 0.40 |
| c2c 65536 | 217 | 0.61 | 0.41 | 0.63 | 0.41 | 0.29 |

The cases not listed are not split by today's schedule at any width (each team arm 0.99-1.02 of
S): rows below 6144 points, and the rows whose plan is a depth-first chain (lines in the work
graph, so `split_min_points_lines`, 32768 on the Xeon, applies): c2c 16384-25000 and the real
rows of 18432. So X77's single real rows of 18432 already run on one member, and those of
32768-36864 now gain from 4 members (0.63-0.93) and more from 8 (0.45-0.67).

- **A team of 2 loses almost everywhere below 32768 points**, also where 4 and 8 gain (r2c
  32768 1.22; c2c 15625 1.06), and G2 (2 of 4 dispatched) is no better than T2 (2 of 2), nor P4b
  (2 of 4 working, 2 waiting) than G2: the waiting members cost nothing measurable. So capping
  members per point count (P4a-P4c, P8b-P8c) only loses: P4c and P8c run 32768-46080 on 2-4
  members at 0.82-1.41 of S against T4's 0.53-0.93.
- **Real rows of 6144-12287 points on a chain lose 1.32-2.69 at 4 and 8 members**, the c2r
  rows and r2c rows alike. Their chain is a half-length complex chain plus a pairing: the work of
  n / 2 complex points in 2-3 chunks per phase (work graph read on the Xeon with the library's own
  scheduler), which the split thresholds compare as n points.
- **c2c rows of 6144-16384 points have no size threshold**: 7200, 8191, 8192, 10000 and 12000
  lose at 4 (1.07-1.22), but 6561 (0.93), 15625 (0.84) and 16807 (0.69) gain at 4, and 10000 and
  12000 gain at 8. The rows that gain have odd radices (9, 5 and 25, 7) and more work per
  point than the work graph counts. No threshold on points passes the "none slower" criterion
  (6561 at 4 would take 1.08, 15625 at 8 1.74), so this experiment does not change them.

## The rule (arm B)

`machine_profile::split_complex_rows`, set in the Xeon 6240 class's branch of `set_grid_sizes`
(Intel's cores only: `intel && private_l2 && !large_l2 && !large_l3`, `x86_64`): a real row (r2c
or c2r, rank 1) whose work graph has no phase of lines is split by its complex points, n / 2 + 1,
as real planes already are (`split_complex_points`, X52). The plan and its work are unchanged;
only the split thresholds read the points. Rows on the depth-first chain or a grid keep their
real points (`split_min_points_lines` was set on them, and X75's 32768 gains at 4).

Cases it moves (their split or cap differs, computed on the Xeon with the library's scheduler on
both arms' work graphs):

- widths 4, 6 and 8 (a group of at most 8: threshold 6144): r2c and c2r 6400, 9216, 10000 and
  10240 become whole (8 cases per width);
- width 16 (a mesh group: threshold 8192): r2c and c2r 9216, 10000 and 10240 become whole, and
  r2c 16384's cap goes from 9 to 8 members (7 cases).

Nothing else in catalogue `x83` moves at any width, and at one thread nothing can (the scheduler
runs only for teams).

## Decision rules

Written before the judging runs, committed with the rule.

- Arms: **O** = `main` 8fdcb06, **B** = O plus the rule (this branch's rule commit), both
  `fftn_omp+THREADS=k`; A/A control O~. Four runs, `--serial`, catalogue `x83` full, 16-20
  rounds: **J4**, **J6**, **J8** (`xeoncc8`, k = 4, 6, 8) and **J16** (`xeoncc16`, k = 16). J4
  also has the one-thread arms **O1** and **B1** (`fftn_omp`, serial).
- Metric: B ÷ O per case, median over valid rounds with its 95 % paired interval; band = the
  run's A/A band (`--min-band 0.01`).
- **Adopt B** on the Xeon 6240's class if, over the 31 moved (case, width) pairs above, the
  geometric mean of B ÷ O is below 1 − band (the largest band of the four runs) and no moved
  pair's whole interval lies above 1 + max(band of its run, 3 %).
- **One thread**: no case of B1 ÷ O1 in J4 may have its whole interval above 1 + max(band, 3 %)
  (both run the same plans by construction; a failure would mean a build or placement artefact,
  investigated before adoption).
- Cases outside the moved set run the same plan and the same schedule in both arms; they are
  reported as the collateral check and do not decide.
- The near-miss re-run and the narrowing clauses of AGENTS section 6 apply (narrowing by length
  or width).
- Reach: the Xeon 6240's class on Intel's cores (Ice Lake-SP by the cache key, per AGENTS
  section 6); AMD's cores the key reaches and unknown vendors keep the real points (the
  maintainer's policy of 2026-11-25, task L).

## Results

## Verdict
