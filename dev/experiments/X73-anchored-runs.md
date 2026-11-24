# X73: whole transforms anchored to members (Zen 3's class, threaded batches)

- **Status:** running
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

## Verdict
