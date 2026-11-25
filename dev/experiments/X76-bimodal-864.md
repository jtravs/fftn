# X76: what picks the mode of the bimodal 864 × 384 backward batch on the Xeon 6240

- **Status:** running
- **Owner:** x76 session, 2026-11-25
- **Phase / task:** 3 (threading; X74's follow-up)

## Question

In X74 the c2c batch of 384 rows of 864 points, out of place, backward, at 16 threads on one
socket of the Xeon 6240 was bimodal between processes in both arms: arrival-order claiming
(`anchored_runs` off, the class's default) took about 67 or 100 µs, anchored claiming about 49 or
110–130 µs. Its slow mode decided X74's deciding re-run (1.156), so the class kept arrival-order
claiming. What picks the mode? The answer decides whether the class can take anchoring after a
fix, or with the batch excluded, or not at all. Informative: this experiment changes no default.

## Hypothesis

Candidates named by the brief, with what X74's stores already say (runs 20261125-001105-3d4f and
-011737-dee1, one line per process per round):

- **Buffer offsets within a page.** fbench draws the input and output offsets once per round and
  every arm of the round uses them. Processes of one round with the same offsets fell in
  different modes (dee1 round 14: M 69, M~ 73, A 107 µs; 3d4f rounds 3 and 5 have the same
  offsets, all slow in one and all fast in the other), so the offsets within a page do not decide.
- **The plan.** Every process printed the same plan, radices 8 × 9 × 12, and the schedule follows
  from the plan and the profile, both fixed: one cache group, 384 whole transforms, no split.
- Left open: the threads' placement within the `numactl` set (unpinned: two lanes can share a CPU,
  or a lane can share one with another user's process), page placement (the Xeon's transparent
  huge pages are `madvise`, so the buffers are 4 KiB pages; their physical colours in the 1 MiB
  16-way L2 differ from process to process), frequency (the AVX-512 licence, the socket's other
  load), and SMT siblings busy with other users' work.

In both arms the fast mode is about what the work costs where each member's rows stay in its L2
(anchored: 24 rows in and out, 663 KiB per member, 49 µs ≈ 1/16 of the one-thread time), and the
slow mode moves the batch's 10.6 MB at about 100 GB/s, a socket's memory bandwidth: as if the rows
came from memory or from other cores' caches in every execution.

## Change under test

None in the library. A probe (`bench/x76/x76_probe.cpp`, target `x76_probe`, Linux only) runs one
process of the case as fbench does (plan by rules, input written by the main thread, output first
touched by the team, fbench's calibrated batches of 2 ms after 20 ms of warm-up) through
`fftn::omp::executor`, with any fbench FFTN tuning option (`ANCHORED_RUNS=0|1`), and records beside
the time what might pick it:

- the plan's description;
- the buffers' addresses modulo 4 KiB and 2 MiB and the huge-page kilobytes of their mappings;
- per lane: its thread, every CPU it was seen on (sampled every 10 batches), its migrations and
  context switches over the timing, and user-mode counters per call (cycles, reference cycles,
  instructions, L2 lines in, loads that missed L3, loads that hit a modified line in another
  core, cycles in the AVX-512 licence);
- the busy fraction of the process's CPUs' SMT siblings and of the rest of their socket.

`fbench_ns` is the median of the first 7 batches after the warm-up, what one fbench round
records; the probe goes on timing for a second, so a mode that changes within a process shows.

## Method

- **Xeon 6240** (2 × 18 cores, Cascade Lake, `avx512`, GCC 14.4, Release, double), Slurm
  `-c 32`, never `--exclusive`; inside the job `numactl -m <node> -C <cpus>` on the 16 idlest
  whole cores of the least busy socket (`pick.py`, as X74's X16). Threads unpinned in that set
  unless a variant says otherwise. The load (`uptime`, `squeue`) is recorded before each job.
- Fresh processes, arms interleaved (A, M, A, M, ...), at least 20 per arm per variant.
- Variants, each a hypothesis test by intervention:
  - **U**: as X74 (unpinned, 4 KiB pages);
  - **P**: lanes pinned one per CPU (`OMP_PROC_BIND=close OMP_PLACES=cores`);
  - **H**: buffers on 2 MiB pages (`huge=1`), unpinned;
  - more as the findings ask, recorded below before they run.
- Metric: per process, the mode (fast or slow by the gap between the modes), and what it
  correlates with.

## Results

## Verdict
