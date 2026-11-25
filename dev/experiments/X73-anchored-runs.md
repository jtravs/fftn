# X73: whole transforms anchored to members (Zen 3's class, threaded batches)

- **Status:** done
- **Owner:** x73 session, 2026-11-24
- **Phase / task:** 3 (threading; gap 3 of the client catalogue's check of 2026-11-24)

## Question

On the EPYC 7763 at 16 threads, out-of-place c2c row batches of the client catalogue (384 × 7373,
384 × 7372, 750 × 4469, forward and backward) took 1.27–1.49 of FFTW's time on an idle node
(run 20261124-111422-12f5, node21, band ±3.65 %), while the same batches in place took 0.90–0.95.
Why, and does a scheduling rule keyed to the class close it without slowing anything at 8 threads?

## Diagnosis (before the change)

From run 20261124-111422-12f5 and the client check's 8-thread run 20261124-103014-bb67: FFTN's
out-of-place batches did not scale from 8 to 16 threads (0.88–0.94 → 0.81–0.93 ns per point),
while FFTN in place (0.73–0.78 → 0.41–0.46) and FFTW out of place (0.89–0.95 → 0.58–0.71) did.
The plans are the same in both placements (384 = 12 × 32).

A diagnostic harness (outside the repository: one process per measurement, 40 executions after
three warm-ups, median; `fftn::omp::executor` of 16 lanes against FFTW 3.3.11's threads at MEASURE;
GCC 16, `avx2`, double; DMOG jobs 518034 on node17, 16 cores of one NUMA node, and 518035 on
node19, 16 cores over two NUMA nodes at load 330) separated the causes. node17, 384 × 7373 out of
place, four rounds:

| How the batch was run | ms per execution |
|---|---|
| FFTN, executor (claims in arrival order), threads unpinned (the client's way) | 2.26–2.53 |
| FFTN, the same with `OMP_PROC_BIND=close OMP_PLACES=cores` | 1.19 |
| FFTN, `completion::own` (each member its fixed share), unpinned | 1.20 |
| FFTW, threads unpinned | 1.23–1.25 |
| FFTN, executor, output pages placed by a static first touch | 2.35–2.53 |

750 × 4469 behaved alike (3.0–3.2 against 1.84–1.87, FFTW 1.89–1.90), and so did in place (384 ×
7373: 0.82–0.91 against 0.46 for `completion::own`). Every page of input and output was on the
job's NUMA node (`move_pages`), and interleaving the pages or placing them by first touch did not
help: it is not NUMA placement or write streams. What differs is which core runs which rows.
`completion::all` claims a group's whole transforms in runs from one counter, in the order the
members arrive, so a row lands on a different member in every execution. Unpinned, the team's
cache groups (ranks 0–7 and 8–15, assumed to be one core complex each) do not match the core
complexes the threads run on, so a row also moves between core complexes, and the batch, 45 MiB
each way, no longer stays in the two complexes' 64 MiB of L3 from one execution to the next: each
execution fetches lines another complex holds dirty. Pinned, the ranks of a group share an L3 and
the rows stay in it; with a fixed share per member (FFTW's way, and `completion::own`) they stay
whatever the placement.

## Hypothesis

If each member claims a fixed range of its group's whole transforms first, the same rows run on
the same thread in every execution, and an unpinned thread rarely changes core between executions,
so a batch beyond one complex's L3 stays in the caches of the cores that run it. Out of place at
16 threads the six batches should fall to about FFTW's time or below (the `completion::own`
times), in place too, and 8 threads should be no slower (one cache group: the rows already share
an L3 when the 8 cores are one complex, and gain when they are not).

## Change under test

`machine_profile::anchored_runs` (`src/threading/schedule.hpp`), set in `set_grid_sizes` on the
`zen3_like` key only (x86-64, private L2 under 1 MiB, at least 2 MiB of L3 per CPU: the EPYC
7763's class), off everywhere else. With it, `team_engine::whole_tasks` (`src/threading/engine.hpp`,
`anchored_tasks`) cuts a group's whole transforms into one contiguous range per member dealt, in
order of rank; a member of the group starts with its own range, any other member at the range of
its rank, and each goes round the ranges claiming runs of half what is left of a range until none
is left. Each range has its own tagged counter (`team_layout::anchored`, one line per lane; the
team state grows by one 128-byte line per lane). Split transforms, chunks, `completion::own`,
plans and results are unchanged (every transform computes alike whoever runs it, so the bits are
those of serial execution, which the threading tests check with the anchored cases added).
The fbench option `ANCHORED_RUNS=0|1` sets the field for diagnosis.

**Reach** (AGENTS section 6): the `zen3_like` key, which also matches Zen 2 (16 MiB of L3 per CCX)
and an Intel core with a private L2 under 1 MiB and 2 MiB of L3 per CPU (DESIGN 13.5). Zen 4 and
Zen 5 (1 MiB L2), the Xeon 6240's class, the 2 MiB-L2 key and Apple keep arrival-order runs.

**What it can move**: threaded executions (executor or team, `completion::all`) of plans whose
work graph has whole transforms for a group of at least two members: batches of rows, planes and
volumes. Single transforms (batch 1) have no whole transforms dealt among members (they run inline
or split) and do not move.

## Method

- Machine: DMOG, AMD EPYC 7763 (2 × 64 cores, NPS4: 16 cores and two CCXs of 32 MiB L3 per NUMA
  node), Slurm partition `nodes`, nodes 16–21, shared allocations of 64 cores (`-c 64 --mem=180G
  --hint=nomultithread`, a socket's worth, never `--exclusive`), threads unpinned as the client
  runs them. GCC 16.2, Release, `avx2`, double. fbench, hot mode, statistic median, 16–30 rounds,
  `--min-band 0.01`; the load of every round is in the store.
- Arms: **M** = `fftn_omp:1b1d129` (main), **A** = `fftn_omp:<this branch's head with the rule
  committed>`, both `+THREADS=k`; **F** = `fftw` with `THREADS=k` (MEASURE): reported, not judged.
  M first, with the A/A control M~.
- Cases: the client catalogue's tracking tier (52 shapes; private, never committed), plus the
  standard catalogue's five tracking batches (`c2c-(8-b4096|64-b256|128-b4096|1024-b256|100-b1024)-op`).
- **Run J16**: threads 16, arms M, A, F, the client catalogue; **J16b**: the same on the five
  standard batches. **Run J8** and **J8b**: the same at 8 threads. All four in one job, in that
  order; each run has its own band. "J16" below means J16 and J16b, each judged against its own
  band (the weighted means are the client catalogue's, from J16 and J8).
- **Moved cases**: those with batch ≥ 2 (33 of the client catalogue's 52, weight 0.55, and the five
  standard batches). The 19 single transforms are the collateral check (same plan, same code path).
- Metric: A ÷ M per case, median over rounds with its 95 % paired interval; the client catalogue's
  weighted geometric mean (`fbench show`).

### Decision rules (written before any judging run)

- **Targets (J16)**: the six out-of-place c2c batches (384 × 7373, 384 × 7372, 750 × 4469; forward
  and backward): each A ÷ M below 1 − band with its interval's upper end below 1, and their
  geometric mean at most 0.90.
- **No loss at 16 (J16)**: no moved case slower than 1 + band (whole interval above 1 + band); the
  client catalogue's weighted geometric mean A ÷ M at most 1.00.
- **No loss at 8 (J8)**: no moved case slower than 1 + band (whole interval above 1 + band); the
  client catalogue's weighted geometric mean A ÷ M at most 1 + band.
- **Collateral**: no single transform slower than 1 + max(band, 3 %) (whole interval above) in
  either run; such a case, having the same plan and code path, is re-timed, not judged as a loss.
- **Adopt** if all of the above hold: `anchored_runs` stays on for the `zen3_like` key.
- **Near misses** (AGENTS section 6, 2026-11-23): a criterion missed by less than the run's band
  gets one `--serial` re-run of that run, which decides.
- **Narrowing** (AGENTS section 6, 2026-11-23): if moved cases lose, the change may be narrowed
  before adoption to anchoring only where the team has more than one cache group (16 threads on
  the class, not 8), or only for whole transforms of at least 64 KiB (one side). A deciding run on
  the final code is then declared here before it starts.

### Not measured

The thread pool (`fftn::thread_pool`) hands task indices to whichever thread claims first, so its
ranks do not stay on threads and anchoring gains it little; the client runs on OpenMP. Kokkos'
`RangePolicy` with chunks of one gives rank i to thread i, as OpenMP does. Pinned threads gain
less (the diagnosis' pinned arm was already at the anchored time). Other classes keep the old runs
until measured.

## Results

Arms M = `fftn_omp:1b1d129`, A = `fftn_omp:84bc1cb` (the change, rules committed), F = FFTW;
GCC 16.2 Release, `avx2`, double; Slurm allocations of 64 cores, threads unpinned. Main's runs
vary two- to fivefold between processes when rows wander between core complexes (the diagnosis),
so the A/A bands of M are wide.

| Run | Store | Node, load | Rounds | Band | A ÷ M, client catalogue (weighted) |
|---|---|---|---|---|---|
| J16 | 20261124-223028-0605 (job 518094) | node20, load 47–58, 25 of 30 rounds loaded | 30 | ±18.80 % | 0.786 [0.700, 0.851] |
| J8 | 20261124-223028-7f3c (job 518096) | node21, load 8–16 | 30 | ±13.74 % | 0.838 [0.762, 0.918] |
| J16b | 20261124-230837-32f5 (job 518098) | node20, load 65–67, all rounds loaded | 30 | ±3.36 % | (standard batches) 0.687 |
| J8b | 20261124-230837-066b (job 518100) | node21, load 21–28 | 30 | ±1.26 % | (standard batches) 0.731 |
| I16 (informative) | 20261124-230018-b00b (job 518159) | node21, `-c 16` as the client check, load 12–38 | 24 | ±8.35 % | (19 cases) 0.853 |

- **Targets (J16)**: A ÷ M 0.317–0.383 for the six out-of-place batches (each interval's upper end
  at most 0.45, below 1 − band = 0.81), geometric mean 0.344 (rule: at most 0.90). At 8 threads the
  same six took 0.290–0.342. The in-place twins: 0.60–0.69 at 16, 0.75–0.99 at 8.
- **No loss at 16 (J16)**: no moved case slower than 1 + band; the slowest is the in-place plane
  batch 192² × 384 backward at 0.987 [0.876, 1.029]. 22 of 33 moved cases faster beyond the band.
- **No loss at 8 (J8)**: no moved case slower than 1 + band; the slowest are 10240 × 3 backward
  1.061 [0.898, 1.186] and the in-place plane batches 384² × 750 1.025–1.030 (intervals from 0.998
  and 1.001, inside the band). 14 of 33 faster beyond the band.
- **Standard batches (J16b, J8b)**: 8 × 4096, 64 × 256, 128 × 4096, 1024 × 256 and 100 × 1024 took
  0.47–0.93 at 16 threads (64 × 256 0.927 [0.902, 1.086], the rest faster beyond the band) and
  0.62–0.88 at 8 (all faster beyond the band). None slower.
- **Collateral**: the single transforms 0.82–1.01 at 16 and 0.88–1.08 at 8, none with its interval
  above 1 + 3 % (r2c 32768 at 8: 1.077 [0.949, 1.331], the band's noise).
- **I16** (informative, the client check's own allocation of 16 cores, `epyccc16`-like): main ÷
  FFTW reproduced the gap on the six out-of-place batches (1.14–1.40), and A ÷ M 0.505–0.595 took
  them to 0.64–0.71 of FFTW's time; in place 0.66–0.79 (÷ FFTW 0.51–0.77); the plane batches
  384² × 750 and 192² × 384 0.994–0.999, the real 6144 × 66 batches 0.95–0.96.
- FFTN ÷ FFTW (medians of the same runs): the six out-of-place targets 0.99–1.51 → 0.34–0.45 at
  16 threads and 1.57–1.91 → 0.40–0.47 at 8; their in-place twins 0.72–0.97 → 0.45–0.56 and
  0.57–0.70 → 0.51–0.64. Over the client catalogue with every shape at full width (FFTW at the
  same width, `client_check.py`): 16 threads 0.546 → 0.421, 8 threads 0.626 → 0.541.

## Verdict

**Adopted** on the `zen3_like` key, by the rules' letter: every criterion holds in J16, J16b, J8
and J8b, with no re-run and no narrowing. `machine_profile::anchored_runs` stays on for the class
as committed in 43c79da (judged as 84bc1cb, whose library code is the same); the threading tests
gained anchored cases (team, executor, OpenMP and interleaving suites) and a simulation with an
absent and a late member (78c9eeb). DESIGN 13.3 and the decision log describe it; LESSONS 5.18.

Gap 3 of the client check (out-of-place c2c batches at 16 threads, 1.27–1.49 of FFTW) is closed on
the EPYC 7763: 0.64–0.71 of FFTW in the check's own allocation, 0.34–0.45 in a socket-sized one.

Open, for later experiments:

- Other classes keep arrival-order runs: the Xeon 6240 (one mesh L3 per socket: rows cannot move
  between L3s within a socket, but unpinned teams spanning sockets could), Zen 4 and Zen 5, the
  2 MiB-L2 key and Apple are unmeasured. The mechanism is not specific to Zen 3; a screen on each
  class decides whether the key widens.
- `fftn::thread_pool` gives task indices to whichever thread claims first, so its ranks do not
  stay on threads and anchoring cannot help it as it helps OpenMP; binding its ranks to its
  threads is a follow-up.
- Batches with fewer whole transforms than members dealt get ranges of zero or one transform;
  they complete correctly, and the runs judged lost nothing, but a member may scan empty ranges.
