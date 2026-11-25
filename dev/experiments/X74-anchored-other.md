# X74: whole transforms anchored to members on the Xeon 6240's class and Apple

- **Status:** done
- **Owner:** x74 session, 2026-11-25
- **Phase / task:** 3 (threading; X73's follow-up)

## Question

X73 (`machine_profile::anchored_runs`) anchors each member of a team to a fixed range of its
group's whole transforms, on Zen 3's class only. Does anchoring help, or at least not hurt, on the
Xeon 6240's class (8 and 16 threads) and on Apple (the M5 Max's 6 performance cores)? The answer
decides whether the field is set on those keys.

## Hypothesis

X73's gain came from rows moving between core complexes' L3s from one execution to the next. The
Xeon 6240 has one mesh L3 per socket and Apple one L2 per cluster, so within a socket or cluster a
row cannot change L3; what anchoring can still keep is the private L2 (Xeon: 1 MiB per core) or L1
of the core that ran the row last, and only while threads stay on their cores. So the gains should
be small: the batches whose rows fit the team's private L2s (Xeon at 8 threads: 8 MiB; at 16: 16
MiB) may gain a little, larger batches tie, and Apple ties (an L1 of 128 KiB holds little of a
batch, and the cluster's L2 is shared anyway). A loss could come from a member scanning ranges
whose owner is late (the others then claim in halves of a range instead of in shares of the whole
group), so the risk is on small batches whose execution is short.

## Change under test

`machine_profile::anchored_runs` on the Xeon 6240's class and on Apple. For the runs, one build
(this branch's commit with the rules committed: library code identical to X73's head 86e8ce2,
PR 108) with the fbench option that X73 added:

- Arm **M**: `fftn_omp:<commit>+THREADS=k+ANCHORED_RUNS=0` (the classes' default today: arrival-order
  runs).
- Arm **A**: `fftn_omp:<commit>+THREADS=k+ANCHORED_RUNS=1` (anchored ranges, as Zen 3 has them).
- Arm **F**: `fftw:<commit>+THREADS=k` (FFTW 3.3.11 at MEASURE, its own threads): reported, not
  judged.

The option sets the same profile field that the rule would set in `set_grid_sizes`, so A is the
code the rule would select. If the rule is adopted, the commit that sets it changes only that
condition and its test.

**Keys** (AGENTS section 6, by cache key within one ISA target and register file), the same as
X61's (`adjacent_lanes`):
- the Xeon 6240's class: x86-64, a private L2 below 2 MiB, less than 2 MiB of L3 per CPU
  (`private_l2 && !large_l2 && !large_l3`; Cascade Lake and Skylake-SP, and an Ice Lake-SP whose
  L3 per CPU is below 2 MiB);
- Apple: AArch64 with a cluster-shared L2 (`apple_like`).

Zen 4 and Zen 5, the Ice Lake VMs (large L3 per CPU), the 2 MiB-L2 key and other AArch64 stay as
they are: unmeasured.

**What it can move**: threaded executions (`completion::all`) of plans whose work graph has whole
transforms for a group of at least two members: batches of rows, planes and volumes. Single
transforms (batch 1) run inline or split and do not move. On both classes the team is one cache
group at the widths measured (the Xeon's L3 serves 18 cores, Apple's L2 6), so every member deals
from one group's ranges.

## Method

- Cases: the client catalogue's tracking tier (52 shapes, weights summing to 0.997; private, never
  committed), as X73; the **moved cases** are the 33 with batch ≥ 2 (weight 0.55); the 19 single
  transforms are the collateral check. Plus the standard catalogue's five tracking batches
  (`--filter '^c2c-(8-b4096|64-b256|128-b4096|1024-b256|100-b1024)-op'`), all moved.
- fbench, hot mode, statistic median, process rounds, A/A control M~ (M first), `--min-band 0.01`;
  the load of every round is in the store.
- **Xeon 6240** (2 × 18 cores, Cascade Lake, `avx512`, GCC 14.4, Release, double): Slurm through
  `fbench_remote.py submit --serial`, shared allocations (`-c 16` for 8 threads, `-c 32` for 16),
  never `--exclusive`; inside the job `numactl -m <node> -C <cpus>` on the idlest whole cores of
  the least busy socket (the client check's `xeoncc8` / `xeoncc16` profiles). Threads unpinned
  within that set.
  - **X8**: 8 threads, arms M, A, F, the client catalogue; **X8b**: the same on the five standard
    batches.
  - **X16**: 16 threads, the same; **X16b**: the same on the five standard batches.
- **Mac M5 Max** (6 performance + 12 efficiency cores, Clang 23, `neon_fcma`, Release, double, the
  NEON FFTW), on AC power (`pmset -g batt` before and after each run; a run that overlaps battery
  or sleep is invalid), one fbench at a time, threads unpinned (macOS cannot pin):
  - **C6**: 6 threads, arms M, A, F, the client catalogue; **C6b**: the same on the five standard
    batches.
- Each run has its own band; each run is judged against its own band.
- Metric: A ÷ M per case, median over rounds with its 95 % paired interval; the client catalogue's
  weighted geometric mean (`fbench show`); the five standard batches' unweighted geometric mean.

### Decision rules (written before any judging run)

Judged per class: the Xeon 6240's class on X8, X8b, X16 and X16b; Apple on C6 and C6b.

- **No loss**: in every run of the class, no moved case slower than 1 + band (its whole 95 %
  interval above 1 + band).
- **Means**: in every run of the class, the client catalogue's weighted geometric mean A ÷ M at
  most 1.00 (X8, X16, C6), and the five standard batches' geometric mean A ÷ M at most 1.00 (X8b,
  X16b, C6b).
- **Collateral**: no single transform slower than 1 + max(band, 3 %) (whole interval above) in any
  run; such a case, having the same plan and code path in both arms, is re-timed, not judged as a
  loss.
- **Adopt** on a class if all of the above hold there: `anchored_runs` is set on that class's key.
  The brief asks whether anchoring helps or at least does not hurt; a class that passes with no
  case faster beyond the band is adopted as "does not hurt" (one claiming rule on every measured
  class with a cache group of several cores, and the anchoring X73 found protective for unpinned
  threads, which these pinned runs do not test), and the verdict says so.
- **Otherwise** the class keeps arrival-order runs, and the verdict is recorded.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  gets one `--serial` re-run of that run (on the Mac, one re-run of that run), which decides.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose on a class, the change may be
  narrowed before adoption, to anchoring only for whole transforms of at least 64 KiB (one side),
  or only at 16 threads on the Xeon. A deciding run on the final code is then declared here before
  it starts.

### Informative (decides nothing)

- **X16s**: 16 threads on the Xeon over both sockets (8 idlest whole cores of each, memory placed by
  first touch, no `-m`), arms M, A, F, the client catalogue: the closest pinned stand-in for an
  unpinned team that spans sockets, where rows could change L3 as on Zen 3. Run only if the Xeon
  is free after the judging runs.

## Results

One build, d78f543 (the rules' commit; library code that of X73's head), arms M
(`ANCHORED_RUNS=0`), A (`ANCHORED_RUNS=1`) and F (FFTW, not judged), with the A/A control M~.
Stores in `bench-out/runs/` of the submitting worktree.

| Run | Store | Machine, placement, load | Rounds | Band | A ÷ M |
|---|---|---|---|---|---|
| C6 | 20261124-232852-22a0 | M5 Max, AC, load 4.0 (max 9.0), 2 rounds loaded | 24 | ±1.07 % | client catalogue 0.996 [0.993, 0.999] |
| C6 re-run (deciding) | 20261124-233956-4feb | M5 Max, AC, load 3.2, none loaded | 24 | ±1.36 % | client catalogue 0.996 [0.994, 0.999] |
| C6b | 20261124-233921-232b | M5 Max, AC, load 3.0, none loaded | 16 | ±1.00 % | standard batches 0.929 [0.925, 0.934] |
| X8 | 20261124-233329-4c73 (Slurm 17007) | Xeon, node 1, 8 cores; load 21 (max 36), 5 rounds loaded | 20 | ±3.56 % | client catalogue 0.951 [0.937, 0.960] |
| X8b | 20261125-004755-33c1 (17011) | Xeon, 8 cores; load 4.2, none loaded | 30 | ±1.00 % | standard batches 0.695 [0.689, 0.721] |
| X16 | 20261125-001105-3d4f (17009) | Xeon, node 1, 16 cores; load 5.2, 7 rounds loaded | 20 | ±1.00 % | client catalogue 0.954 [0.950, 0.971] |
| X16 re-run (deciding) | 20261125-011737-dee1 (17027) | Xeon, node 1, 16 cores; load 4.0 (another job's build: max 66), 3 rounds loaded | 20 | ±1.01 % | client catalogue 0.949 [0.943, 0.961] |
| X16b | 20261125-004848-5947 (17013) | Xeon, 16 cores; load 5.7, none loaded | 29 | ±2.08 % | standard batches 0.581 [0.561, 0.584] |

The Mac stayed on AC power through every run (checked each minute; 80 %, "AC attached; not
charging"), no sleep. Xeon: GCC 14.4, `avx512`; threads unpinned within `numactl -m 1 -C <cpus>`.

**Apple (M5 Max, 6 threads).**
- C6: one moved case past the band, the in-place batch 750 × 4469 backward, 1.017 [1.014, 1.028]
  against 1 + band = 1.0107: missed by 0.3 %, less than the band, so the run was re-run (near-miss
  clause). The re-run decides: no moved case slower beyond the band (750 × 4469 in place backward
  1.022 [1.013, 1.028], forward 1.014 [1.007, 1.029], inside 1 + band = 1.0136), weighted mean
  0.996. Faster beyond the band: the real batches 6144 × 66 and 6144 × 48 (0.97–0.98).
- C6b: all five standard batches faster beyond the band, 0.870 (8 × 4096) to 0.969 (128 × 4096);
  FFTN ÷ FFTW 8 × 4096 1.67 → 1.48, 64 × 256 1.53 → 1.43, 100 × 1024 1.13 → 1.05.
- Collateral: the single transforms 0.98–1.01, none beyond the band.
- In both C6 runs the in-place 750 × 4469 pair sat 1.4–2.2 % above M, inside the deciding run's
  band: a possible small cost, below what the runs resolve.

**Xeon 6240's class (8 and 16 threads).**
- X8: no moved case slower beyond the band; 8 faster: the real batches 6144 × 48 and × 66, 6400 × 7,
  10240 × 28, 864 × 384 r2c, 1728 × 128 c2c backward (0.60–0.89). FFTN ÷ FFTW r2c 864 × 384
  1.11 → 0.63 (gap 8 of the client check), r2c 6144 × 48 0.96 → 0.70. Singles 0.98–1.01.
- X8b: all five standard batches faster beyond the band, 0.496 (8 × 4096) to 0.952 (128 × 4096);
  FFTN ÷ FFTW 8 × 4096 1.72 → 0.85, 100 × 1024 1.00 → 0.64.
- X16: two moved cases past the band, the in-place 750 × 4469 pair, 1.019–1.021 with lower ends
  1.013 against 1 + band = 1.010: missed by 0.3 %, so the run was re-run (near-miss clause).
- X16 re-run, deciding: the 750 × 4469 pair passes (1.009 [1.003, 1.025], 1.014 [1.006, 1.020]),
  but the c2c batch 864 × 384 backward is slower beyond the band, **1.156 [1.077, 1.240]** (in the
  first X16 1.099 [0.773, 1.191], at 8 threads 0.983 [0.942, 1.049]). Its times are bimodal from
  process to process in both arms (M about 67 or 100 µs, A about 49 or 110–130 µs): A is faster in
  the fast mode and slower in the slow mode, which dominated the re-run. Weighted mean 0.949, 8
  moved cases faster (as at 8 threads, 0.73–0.94).
- X16b: all five standard batches faster beyond the band, 0.385 (8 × 4096) to 0.919 (1024 × 256);
  FFTN ÷ FFTW 8 × 4096 2.54 → 0.97, 64 × 256 1.64 → 0.79, 100 × 1024 1.49 → 0.85.
- Collateral: the single transforms 0.98–1.02, none beyond 1 + 3 %.

The gains are largest on the shortest executions (standard batches of 8 and 64 points, 0.4–0.6 of
the time), where each member's rows fit its own L1 or L2: with arrival-order runs each row lands on
another core in every execution and its lines come over the mesh from the core that last wrote
them (and every claim contends for one counter's line). These runs do not separate the two causes.

## Verdict

- **Apple: adopted**, by the rule's letter after one near-miss re-run (C6, deciding re-run
  20261124-233956-4feb) and C6b. `anchored_runs` is set on the `apple_like` key (3d74695); the
  schedule test checks the recorded M5 Max, and that Axion, the Ice Lake VM and Granite Rapids stay
  off. It helps: the standard batches 0.87–0.97, the client catalogue 0.996.
- **Xeon 6240's class: not adopted.** The deciding re-run of X16 has one moved case slower beyond
  the band (c2c 864 × 384 backward 1.156 [1.077, 1.240]). Neither declared narrowing removes it
  and keeps the change: anchoring only at 16 threads keeps the loss, and only whole transforms of
  at least 64 KiB excludes almost every case that gained (all five standard batches, the real
  batches of 864–6400 points). The class keeps arrival-order runs.
- **For the maintainer:** everything else on the Xeon gained, much: X8 and X8b pass every
  criterion (client catalogue 0.951, standard batches 0.695, nothing slower), X16b 0.581, X16 0.949;
  8 × 4096 at 16 threads goes from 2.54 to 0.97 of FFTW's time. Two ways to take it that the rules
  as written did not provide, each needing a declared deciding run: anchoring on the class for teams
  of at most 8 lanes (X8 and X8b pass as they stand), or the whole class after a diagnosis of the
  bimodal 864 × 384 backward batch (its fast mode is faster anchored).

**X16s** (informative, decides nothing; run 20261125-015820-c820, Slurm 17028; 16 threads on the 8
idlest cores of each socket, memory by first touch; load 5.3, none loaded; 20 rounds, ±2.91 %):
client catalogue 0.880 [0.875, 0.891], 21 moved cases faster beyond the band, none slower. Across
sockets X73's mechanism appears: the six out-of-place c2c batches 0.67–0.80 (FFTN ÷ FFTW 0.86–0.97
→ 0.61–0.70), their in-place twins 0.87–0.93. The c2c batch 864 × 384 backward again took more,
1.131 (inside this run's band). So a team that spans sockets, as an unpinned one on the Xeon can,
gains most from anchoring; that batch is the one case against it in every 16-thread run.
