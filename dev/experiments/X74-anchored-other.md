# X74: whole transforms anchored to members on the Xeon 6240's class and Apple

- **Status:** running
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

## Verdict
