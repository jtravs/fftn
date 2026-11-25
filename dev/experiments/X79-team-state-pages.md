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

Builds: M = 63c2737 (rules), B = 9b9875a (fix). Stores in `bench-out/runs/` of the submitting
worktree.

**Near-miss re-runs** (the rules' clause, committed in 63c2737):
- C6 (Mac) missed the means criterion by less than its band (weighted B ÷ M 1.001 against 1.00,
  band ±1.11 %): one Mac re-run, C6r, decides (started at once, 06:38Z; recorded here after).
- E16 (EPYC, node20 at load 60–72, band ±6.56 %) missed the means criterion by less than its band
  (weighted 1.001): one `--serial` re-run, E16r, decides; it runs on node18 (idle at
  submission). E16b had started on the loaded node20 and was cancelled after 26 s (Slurm 518352,
  no round recorded); it runs as E16b on node18 behind E16r.

**The fix, B ÷ M.**

| Run | Store (Slurm) | Machine, placement, load | Rounds | Band | B ÷ M | Slower beyond the band |
|---|---|---|---|---|---|---|
| C6 | 20261125-062634-36e3 | M5 Max, AC, load 6.0 (max 9.6) | 24 | ±1.11 % | catalogue 1.001 [0.999, 1.004] | none (mean missed by 0.1 %) |
| C6r (deciding) | 20261125-063803-394a | M5 Max, AC, load 4.7 | 24 | ±1.19 % | catalogue 0.999 [0.996, 1.003] | none |
| C6b | 20261125-063705-5a4f | M5 Max, AC, load 5.3 | 30 | ±1.00 % | batches 0.998 [0.990, 1.006] | none |
| E16 | 20261125-063226-7c0d (518350) | EPYC, node20, load 60 (max 72), all rounds loaded | 24 | ±6.56 % | catalogue 1.001 [0.984, 1.028] | none (mean missed by 0.1 %) |
| E16r (deciding) | 20261125-065709-cdb5 (518353) | EPYC, node18, load 5.6, none loaded | 24 | ±1.19 % | catalogue 1.002 [0.998, 1.005] | r2c 10240 × 1 1.028 [1.014, 1.043], c2r 864 × 384 1.027 [1.013, 1.041] |
| E16b | 20261125-071839-c3f7 (518355) | EPYC, node18, load 7.6, none loaded | 30 | ±1.00 % | batches 1.021 [1.015, 1.029] | c2c 8 × 4096 1.079 [1.070, 1.089] |
| X16a | 20261125-064327-ae34 (17084) | Xeon, node 1, 16 cores, load 2.2 | 16 | ±1.00 % | the case alone: M 69.3, B 68.85 (= R), BA 51.2 µs | — |
| X16 | 20261125-064345-8d9a (17086) | Xeon, node 1, 16 cores, load 4.1, none loaded | 24 | ±1.00 % | catalogue 0.996 [0.988, 1.003] | none; 8 faster |
| X16b | 20261125-073659-69cb (17087) | Xeon, node 1, 16 cores, load 8.0, none loaded | 30 | ±1.00 % | batches 0.983 [0.976, 1.001] | none |
| X8 | X8_ROW |
| X8b | X8B_ROW |

The Mac stayed on AC power throughout ("AC attached; not charging", 80 %, checked every 15 s
by the run wrapper; no sampling gap). Xeon: GCC 14.4, `avx512`, threads unpinned in `numactl -m 1
-C <16 or 8 idlest whole cores>`. EPYC: GCC 16, `avx2`.

- **Xeon 6240, 16 threads: the slow mode is gone.** c2c 864 × 384 backward in X16, arm B, every
  process 65.9–70.7 µs (1.15 × R = 79.2 µs), none in the third mode; arm M in the same
  processes 65.8–105.5 µs plus one at 6000 µs, M~ 67–105.4. B ÷ M 0.739 [0.710, 0.957], its
  real twins c2r 0.823, r2c 0.965 (B: c2r 63.5–66.4 µs with one process at 5772 µs, the third
  mode; r2c 60.9–64.1; M 62–137). Anchored on the fix (BA) the case is 48.4–52.0 µs in every
  process. The unimodal criterion holds.
- **Mac M5 Max, 6 threads:** nothing moves (C6r 0.999, C6b 0.998), as expected where the slot
  sizes are already large and nothing was measured slow.
- **EPYC 7763, 16 threads: the fix costs on the shortest batch.** c2c 8 × 4096 out of place
  1.079 in E16b, and in E16r two cases past the band by 0.2–0.3 % (r2c 10240 × 1, c2r 864 × 384)
  with the weighted mean 1.002. The informative run below says the cost comes from the state's
  place at the start of a page, not from the padding.

**Informative, EPYC (decides nothing):** run 20261125-072055-b650 (Slurm 518374, node19, load
2.3, band ±1.00 %, 40 rounds), the five standard batches, arms M, MP = M with the bench option
`WORKSPACE=page` (the old library; the workspace, and so the state, at the start of a page of the
backend's own memory; the slots right after the state's words), B, BP = B with `WORKSPACE=page`:
c2c 8 × 4096 MP ÷ M 1.049 [1.018, 1.068], B ÷ M 1.077 [1.052, 1.088], BP ÷ M 1.060; 64 × 256
1.013, 1.030, 1.009; the other three within 1.013. So on Zen 3 a team state at the start of a
page, put there by the library or by the benchmark, makes the 16-thread batch of 8 × 4096 (about
21 µs per execution, every member's claims and counters on the state's lines) 5–8 % slower; on
the Xeon 6240 the same placement removed a × 1.4–2.3 slow mode. Run 20261125-075639-0bf8 (Slurm
518383, node19, load 3.4, band ±1.00 %, 40 rounds) tried the state 2 KiB into a page of its own
(arm O, a local diagnosis commit a50d5fa, not pushed: the fix with the state's words moved by 2
KiB in their page): 8 × 4096 O ÷ M 1.077 [1.048, 1.093], B ÷ M 1.066, the five batches' mean
1.025 and 1.024. So it is not the offset within the page: on Zen 3 the state's page of its own
costs the shortest batch, wherever its words sit in it. The mechanism is not known.

**Anchoring on the fix, BA ÷ B** (X74's rules; band of each run):

| Run | Store | BA ÷ B | Moved cases slower beyond the band |
|---|---|---|---|
| X16 (reported) | 20261125-064345-8d9a | catalogue 0.939; 14 faster | in-place 750 × 4469 pair 1.018 [1.012, 1.038], 1.022 [1.012, 1.027] |
| X16r (deciding) | X16R_ROW |
| X16b | 20261125-073659-69cb | batches 0.563 [all five faster: 0.380 (8 × 4096) – 0.755 (1024 × 256)] | none |
| X8 | X8A_ROW |
| X8b | X8BA_ROW |

In X16 the anchored arm took c2c 864 × 384 backward to 0.730 of B and its real twins to
0.607–0.717; the in-place 750 × 4469 pair is X74's again (1.019–1.021 there).

## Verdict

(to come)
