# X79: the team state on pages of its own

- **Status:** running
- **Owner:** x79 session, 2026-11-25
- **Phase / task:** 3 (threading; X76's proposal)

## Question

X76 found that the team state (DESIGN 5.3, 13.3), which every member writes several times per
execution, shares its 4 KiB page with whatever the heap put before the workspace: after earlier
cases in an fbench process that is the plan's own twiddle table, and the twiddle lines on the
state's page are then loaded from other cores' modified lines on every execution (c2c 864 × 384
backward at 16 threads on the Xeon 6240: × 1.5–2.3, in both claiming rules). Does giving the
state pages of its own inside the workspace remove that slow mode without costing anything
elsewhere? And, on the fixed code, does anchoring (X73's `anchored_runs`) now pass X74's rules
on the Xeon 6240's class? The first answer decides the fix; the second goes to the maintainer.

## Hypothesis

With the state starting a page and padded to whole pages, nothing the team does not write
(twiddles before the workspace, the first member's scratch slot after it) shares a page with
the hot counters, so the per-page prefetch for ownership that X76 suspects can take no line
anyone else reads. The 864 × 384 batch should then be unimodal at its alone time in every
fbench process (X76 item 5: `WORKSPACE=page` did exactly this from outside the library), and
every other case should be unchanged within the band: the change moves no plan and no
kernel, only where the state and the slots sit in the workspace, which grows by at most one
page of slack and the padding of the state's last page.

## Change under test

`src/threading/`: the team state is aligned to a page (the alignment unit `team_page`) rather
than to 128 bytes, and its size is rounded up to whole pages, so that the members' slots start
on the page after it; `workspace_bytes(lanes)` grows by the extra slack and the padding.
DESIGN 5.3 and 13.3 follow.

**Page size.** 4 KiB on every target except Apple's AArch64 systems, where it is 16 KiB, the
page size of their operating systems: a prefetcher confined to a physical page there can cover
16 KiB, so a 4 KiB unit would not keep the state's page to itself. The cost is memory only (at
most 32 KiB more per team workspace on Apple, 8 KiB elsewhere); the padding is never touched.
Other AArch64 systems with 16 or 64 KiB pages keep 4 KiB, unmeasured.

Arms (one build each, both on this branch, so the bench code is the same and only the library
differs):

- **M**: the rules' commit (the library of `main` 1710a9b): the state at the first 128-byte
  boundary.
- **B**: the fix's commit, the classes' default claiming (arrival order on the Xeon 6240's
  class, anchored on Zen 3 and Apple).
- **BA** (Xeon only): the fix's commit with `ANCHORED_RUNS=1`, the code the anchoring rule
  would select on the Xeon 6240's class (as X74's arm A).
- **F**: FFTW 3.3.11, MEASURE, its own threads: reported, not judged.

**What it can move**: every threaded execution that dispatches a team (`completion::all`):
batches, and single transforms that split. Inline executions and single-thread runs do not
take the team path. Since the slot addresses move in every team execution, every case of a
threaded run is in the target set.

## Method

- fbench, hot mode, statistic median, process rounds, the A/A control M~ (M first),
  `--min-band 0.01`; the load of every round is in the store, and the Xeon's and DMOG's load
  (`uptime`, `squeue`) is recorded before each job.
- Cases: the client catalogue's tracking tier (52 shapes; private, never committed), and the
  standard catalogue's five tracking batches
  (`--filter '^c2c-(8-b4096|64-b256|128-b4096|1024-b256|100-b1024)-op'`), as X74.
- **Xeon 6240** (2 × 18 cores, Cascade Lake, `avx512`, GCC 14.4, Release, double): Slurm
  through `fbench_remote.py submit --serial`, shared allocations (`-c 16` for 8 threads, `-c 32`
  for 16), never `--exclusive`; inside the job `numactl -m <node> -C <cpus>` on the idlest whole
  cores of the least busy socket (X74's `xeoncc8` / `xeoncc16` setups); threads unpinned within
  that set. Arms M, B, BA, F; at least 20 rounds.
  - **X16**: 16 threads, the client catalogue. **X16b**: 16 threads, the five standard batches.
  - **X8**: 8 threads, the client catalogue. **X8b**: 8 threads, the five standard batches.
  - **X16a**: 16 threads, the case c2c 864 × 384 backward alone (`--filter
    '^c2c-864-b384-op-bwd$'`), arms M, B, BA, at least 12 rounds: each process runs nothing
    else, so its time is the case's alone time (X76 table 1). Its arm-B median is R.
  - **X16r**: 16 threads, the client catalogue, arms B, BA, F (B first, its own control B~),
    at least 20 rounds: the deciding X16 run for anchoring (declared here, before any run).
- **EPYC 7763** (DMOG, Zen 3, `avx2`, GCC 16, Release, double): Slurm, a shared 16-core
  allocation (`epyccc16`: `--hint=nomultithread --distribution=block:block`), never
  `--exclusive`, threads unpinned. Arms M, B, F; at least 20 rounds.
  - **E16**: 16 threads, the client catalogue. **E16b**: 16 threads, the five standard batches.
- **Mac M5 Max** (Clang 23, `neon_fcma`, Release, double, the NEON FFTW), on AC power
  (`pmset -g batt` before and after each run; a run that overlaps battery or sleep is invalid),
  one fbench at a time, threads unpinned. Arms M, B, F. If the Mac is on battery the Mac runs
  are skipped and the verdict says so.
  - **C6**: 6 threads, the client catalogue. **C6b**: 6 threads, the five standard batches.
- Metrics: B ÷ M per case (the fix), BA ÷ B per case (anchoring on the fix), medians over rounds
  of the paired per-round ratios with their 95 % bootstrap intervals; the client catalogue's
  weighted geometric mean; the five standard batches' unweighted geometric mean. Each run is
  judged against its own band (the 95th percentile of |log(M~ ÷ M)|, or B~ ÷ B in X16r, at
  least 1 %). BA ÷ B in runs whose first arm is M is computed from the store's paired rounds
  in the same way, against that run's band.
- Per process: the time of c2c 864 × 384 backward in every arm (the store's `median_ns` of each
  round), for the unimodality criterion.

### Decision rules for the fix (written before any judging run)

Judged on X16, X16b, X8, X8b, E16, E16b, C6 and C6b (C6 and C6b only if the Mac is on AC).

- **No loss**: in every run, no case with B ÷ M slower than 1 + band (its whole 95 % interval
  above 1 + band).
- **Means**: in every run, the client catalogue's weighted geometric mean B ÷ M at most 1.00
  (X16, X8, E16, C6), and the five standard batches' geometric mean B ÷ M at most 1.00 (X16b,
  X8b, E16b, C6b).
- **Unimodal** (Xeon, 16 threads): in X16, every process's time for c2c 864 × 384 backward in
  arm B within 1.15 × R (R: X16a's median over arm B's processes; X76's alone time was 66–71
  µs). A process above 1 ms (over 14 × R) is X76's third mode (item 6: a member descheduled for
  whole time slices, seen in both arms and in quiet processes), not the slow mode: it is
  reported, and does not count against this criterion unless arm B has more such processes
  than arm M in the same run, in which case the criterion fails.
- **Adopt the fix** if all of the above hold.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  (for the unimodal criterion: a process above 1.15 × R by less than the band) gets one
  `--serial` re-run of that run (on the Mac, one re-run), which decides.
- **Otherwise** the fix is not adopted; the PR carries this file with the negative verdict and
  the code left out.
- The fix moves no plan; a loss that persists would be a placement effect of the slots, and the
  narrowing clause does not apply to a layout change of every team execution.

### Decision rules for anchoring on the Xeon 6240's class after the fix (X74's, on the fix)

X74's rules unchanged, with A = BA and M = B, judged on X8, X8b, X16 and X16b:

- **No loss**: in every run, no moved case (batch ≥ 2: the client catalogue's 33 batches and
  all five standard batches) with BA ÷ B slower than 1 + band (whole interval above).
- **Means**: the client catalogue's weighted geometric mean BA ÷ B at most 1.00 (X8, X16), the
  five standard batches' geometric mean at most 1.00 (X8b, X16b).
- **Collateral**: no single transform slower than 1 + max(band, 3 %) (whole interval above);
  such a case, with the same plan and code path in both arms, is re-timed, not judged a loss.
- **Deciding X16 run**: X16r, declared here before any run, decides the X16 criteria; X16 is
  reported beside it. The near-miss clause applies once more to X16r, X8, X8b and X16b.
- **Narrowing** (as X74): to anchoring only for whole transforms of at least 64 KiB (one side),
  or only at 16 threads on the Xeon; a deciding run on the final code is then declared here
  before it starts.
- **Adopt** on the class if all hold: `anchored_runs` set on the Xeon 6240's key. Whatever the
  verdict, the flag change goes in a separate draft PR (`exp/x79-anchored-xeon`, stacked on the
  fix) for the maintainer, who has not yet chosen how to take anchoring on that class.

## Results

(to come)

## Verdict

(to come)
