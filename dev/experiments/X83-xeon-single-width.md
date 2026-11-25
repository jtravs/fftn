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

One run on `main` a696bfd, arms:

- **S**: serial (`fftn_omp`, no executor), the baseline;
- **T2**, **T4**, **T8**: `THREADS=2`, `4`, `8`;
- **G2**: `THREADS=4+GROUP_SIZE=2`: an executor of 4 lanes reporting groups of 2, so a single
  transform goes to one group and is split between 2 members while the other 2 are not
  dispatched (DESIGN 13.4, rule 4): a team of 4 that uses 2.

Metric: each arm's time ÷ S per case. A case's crossover is the smallest team width that is not
faster than a narrower one beyond the band.

## Decision rules

(Written after the diagnosis, before the judging runs; see below.)

## Results

## Verdict
