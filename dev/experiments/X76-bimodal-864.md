# X76: what picks the mode of the bimodal 864 × 384 backward batch on the Xeon 6240

- **Status:** done (informative; no default changed)
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

Candidates named by the brief, with what X74's stores already said (runs 20261125-001105-3d4f and
-011737-dee1, one line per process per round):

- **Buffer offsets within a page.** fbench draws the input and output offsets once per round and
  every arm of the round uses them. Processes of one round with the same offsets fell in
  different modes (dee1 round 14: M 69, M~ 73, A 107 µs; 3d4f rounds 3 and 5 have the same
  offsets, all slow in one and all fast in the other), so the offsets within a page do not decide.
- **The plan.** Every process printed the same plan, radices 8 × 9 × 12, and the schedule follows
  from the plan and the profile, both fixed: one cache group, 384 whole transforms, no split.
- Left open: the threads' placement within the `numactl` set, page placement (the Xeon's
  transparent huge pages are `madvise`, so the buffers are 4 KiB pages whose physical colours in
  the 1 MiB 16-way L2 differ between processes), frequency (the AVX-512 licence, the socket's
  other load), and SMT siblings busy with other users' work.

In both arms the fast mode is about what the work costs where each member's rows stay in its L2
(anchored: 24 rows in and out, 663 KiB per member, 49 µs ≈ 1/16 of the one-thread time).

## Change under test

None in the library. Two bench-only tools:

- **`x76_probe`** (`bench/x76/x76_probe.cpp`, Linux only) runs one process of the case as fbench
  does (plan by rules, input written by the main thread, output first touched by the team,
  fbench's calibrated batches of 2 ms after 20 ms of warm-up) through `fftn::omp::executor`, with
  any fbench FFTN tuning option (`ANCHORED_RUNS=0|1`), and records beside the time: the plan and
  where its stages keep their twiddles; the buffers' and the workspace's addresses and huge-page
  kilobytes; per lane its thread, the CPUs it was seen on, its migrations, context switches,
  page faults and user-mode counters per call over the typical batches (cycles, reference
  cycles, instructions, L2 lines in, loads that missed L3, loads that hit a line modified in
  another core: `MEM_LOAD_L3_HIT_RETIRED.XSNP_HITM`); the busy fraction of the CPUs' SMT siblings
  and of the rest of the socket. Options change one thing at a time: the buffers' offsets,
  2 MiB pages (`huge=1`), lanes pinned (`OMP_PROC_BIND`), a freed block before anything
  (`prelude=`, which raises glibc's mmap threshold so that the workspace comes from the heap),
  another case planned and run first (`before=`), the free pages scattered (`fragment=`), the
  workspace at a chosen offset in memory of the probe's own (`ws_offset=`).
- **`WORKSPACE=page`**, an option of the fbench FFTN backend: each plan's workspace in memory
  of the backend's own that starts a page, instead of `fftn::workspace(bytes)` on the heap.

## Method

- **Xeon 6240** (2 × 18 cores, Cascade Lake, `avx512`, GCC 14.4, Release, double), Slurm
  `-c 32`, never `--exclusive`; inside each job `numactl -m <node> -C <cpus>` on the 16 idlest
  whole cores of the least busy socket (`pick.py`, as X74's X16), threads unpinned in that set
  unless a variant pins them. The load (`uptime`, `squeue`) was recorded at the start of each job
  and the load average beside every process. Another agent's builds and timing ran on the Xeon
  at times (load 2–79); processes it stalled show as executions of 5–9 ms (below).
- Fresh processes, arms interleaved (A = `ANCHORED_RUNS=1`, M = `ANCHORED_RUNS=0`), 12–24 per
  arm per variant; the fbench harness itself (`fbench-fftn_omp run`) on case files: the case
  alone, after one other case, twice, or among the client catalogue's 52 tracking cases in a
  shuffled order, as an fbench process runs them.
- `perf c2c` (user mode, PEBS load latency and stores) on the probe and on the harness, the
  harness's recording delayed past its first case.
- Builds: commits of this branch (`x76_probe`, `fbench-fftn_omp`, library code that of `main`
  e2a4413). Job outputs in `~/awork/fftn-x76` on the Xeon (`r*-<job>.jsonl`, `c2c-<job>/`).

## Results

Times are medians per process (µs, the probe's `fbench_ns` or median over a second, the
harness's `median_ns`). A = anchored, M = arrival order.

**1. Alone, the case is never slow** (Slurm 17052, 144 processes; 17054, 104; 17065, 192).

| Variant | A | M |
|---|---|---|
| U: as X74 (unpinned, 4 KiB pages, offsets 0) | 48.5–51.3 (24 of 24 fast) | 66.5–69.7 (24 of 24) |
| P: lanes pinned one per CPU | 50.1–52.3 | 68.0–70.5 |
| H: buffers on 2 MiB pages | 48.5–51.0 | 66.5–70.0 |
| O: random input and output offsets | 49 (13 of 13) | 66–69 (13 of 13) |
| fbench harness, case alone | 48–51 (13 of 13) | 66–69 (13 of 13) |
| 96 random input, output and workspace offsets | 48.5–50.3 | 66.4–68.5 |

Counters per call, A: 10–18 k L2 lines in over all lanes (a member's rows stay in its L2), no
loads from memory, about 300 HITM loads; M: 165 k lines in (the whole batch, 10.6 MB, moves
between L2s in every execution: rows land on another member), 730 HITM loads. Effective clock
2.5–2.6 GHz in every variant; two lanes seldom shared a CPU (in at most one of 13 samples of a
process), and the SMT siblings were idle. Every process was fast, whatever its migrations, so
placement, page colours, offsets, frequency and siblings do not make the slow mode on their own.

**2. In an fbench process the case is slow after other cases** (Slurm 17054, the harness on the
client catalogue's tracking cases shuffled, 12 orders): M 95–107 in 12 of 12 processes, A
105–128 in 9 of 12 (49–64 in 3). The case's real twins are bimodal in the same processes (r2c
864 × 384: 37–49 or 94–144, in both arms of an order alike), while X74's stores show the other
batched cases of such processes at 0.97–1.05 of their medians. One earlier case is enough (Slurm
17055, 12 rounds):

| Before the case | A | M |
|---|---|---|
| nothing | 49–53 | 66–71 |
| r2c 9216 (one row, its buffers 74 KB) | 115–123 | 95–100 |
| c2c 1728 (one row) | 97–105 | 81–93 |
| the case itself (second copy) | 69–73 | 77–80 |
| r2c 864 × 384 | 67–70 | 76–78 |
| c2c 750 × 4469 (buffers 54 MB) | 48–53 | 66–69 |
| c2c 384² × 750 planes | 69–74 | 76–79 |

**3. What an earlier case leaves is the heap's layout, and the workspace's place in it.** An
earlier case frees blocks (its workspace for 16 lanes, the harness's copies of its data) of
between about 0.5 and 32 MiB, and glibc raises its mmap threshold to the size of a freed mapped
block, so the case's workspace (444 KiB for 16 lanes) then comes from the heap, right after the
plan's own allocations, instead of from a mapping of its own. In the probe (Slurm 17062, 17068,
16 and 12 processes per arm): `prelude=1048576`, `before=c2c:1728:1` and `before=r2c:9216:1` all
put the workspace on the heap and all made the case slower in every process, A 57–64 against
50–55, M 71–75 against 68–72 (that run's load 65–79: another job's build);
`before=c2c:750:4469`, whose large blocks are above the threshold's 32 MiB cap and whose
workspace is smaller than this one's, left it in its own mapping and fast. Neither the workspace's offset in a page (64
offsets, Slurm 17064: A 48.4–51.3, M 66.7–70.5 at every offset) nor scattered free pages
(`fragment=256 MiB`, Slurm 17068) nor 2 MiB buffers changed anything; the heap did.

**4. The loads that pay are the plan's twiddles on the team state's page** (`perf c2c`, Slurm
17069, 17070, 17074). The team state (DESIGN 5.3, 13.3) starts at the first 128-byte boundary of
the workspace, and its first lines (tasks done, finished, every lane's `calls` word, the group's
claim counter, the anchored ranges) are written by every member several times per execution.
On the heap the workspace follows the plan's radix-8 twiddle table directly, so the table's
last lines share the state's 4 KiB page:

- probe, workspace on the heap (A 60 µs): table 0x2ef1f4c0–0x2ef2533f, workspace at 0x2ef25350,
  state at 0x2ef25380. Of 1157 HITM load samples, 892 fall in page 0x2ef25000, on the 13 table
  lines below the state, all loaded by the radix-8 codelet (`grouped_pass<double, 8>`); the
  table's other six pages, loaded as often (10–12 k samples each), have none. With the
  workspace in its own mapping (A 55 µs under `perf`) there are 322 HITM samples, on the state's
  own lines.
- harness after r2c 9216 (A 119 µs): 7619 of 8223 HITM samples on one page, 0x27273000, whose
  first 3.5 KiB (56 lines) are twiddles loaded by the codelet and whose last lines are the
  state, stored by `whole_tasks`. Stores on the twiddle lines: none sampled.
- Per lane in the probe (A, workspace on the heap and buffers on 2 MiB pages, Slurm 17068):
  about 75 HITM loads per call against 19 with the workspace in its own mapping, on every lane,
  and 175–189 k cycles against 144–160 k.

So the slow mode is a page shared by the team's hot counters and data every member reads for
every row. Its size follows how much of the table lands on the state's page (13 lines: +22 %;
56 lines: × 2.3), which the heap's history decides: hence "modes" that follow the case order,
and the real twins' modes, which follow it too. The mechanism inside the core is not verified
(prefetchers cannot be switched off without root). The likely one: Intel's L2 streamer tracks
streams per 4 KiB page and is triggered by RFOs too, so a core running the atomic operations on
the state also prefetches the page's other lines for ownership, taking them from the cores that
share them; their next twiddle loads then snoop that core. Lines on other pages are untouched,
as observed.

**5. The page fix, applied from outside, removes the slow mode** (Slurm 17076, harness,
`WORKSPACE=library` against `page`, the same orders and offsets):

| Process | A, library | A, page | M, library | M, page |
|---|---|---|---|---|
| r2c 9216, then the case (12) | 115–123 | 52.0–53.7 | 102–109 | 69.4–71.5 |
| the case twice, second (12) | 68–70 | 50–52 | 78–80 | 68–71 |
| the 52 tracking cases shuffled (6) | 107–120, one 54 | 50–55 | 82–102, one 77 | 69–70 |
| the same processes: c2r 864 × 384 | 50–143 | 47–51 | 78–126 | 65–67 |
| the same processes: r2c 864 × 384 | 38–134 | 38–39 | 63–128, one 1245 | 62–63 |

Each cell leaves out the executions of the third mode (item 6; one or two processes per cell,
in both workspace variants). With the workspace on a page of its own the case and its real twins
are unimodal at their alone times in every process, in both arms; anchored over arrival order
0.74 (52 against 70 µs).

**6. A third mode, 5–9 ms per execution,** appears now and then in single processes of both
arms, most often while another job loaded the machine (load 14–79 in Slurm 17055 and 17062) but
also in quiet processes (the first probe run, load 3), and in X74's stores (6000–39000 µs in 8 of
its X16 re-run's 60 FFTN processes). Its 2 ms batches of 16–40 calls then take 0.1–0.3 s: a member
descheduled or blocked for whole time slices while the others wait for its rows. Not specific
to this case, and not investigated further here.

## Proposal (for the maintainer; drafted rules, nothing adopted)

**Fix.** Give the team state pages of its own inside the workspace: align it to 4 KiB rather
than 128 bytes and round its size up to whole pages, so that neither the allocation before the
workspace nor the first member's scratch slot shares its pages (`workspace_bytes` grows by at
most 8 KiB). This is a change to the threading engine's layout (`src/threading/team_execute.cpp`,
`team_state.hpp`, DESIGN 5.3 and 13.3): it needs a second reader, a test that the state starts a
page at any workspace address, and an experiment, since it moves every team execution:

- Arms: `main` against the fix, both with arrival-order claiming on the Xeon class (its default);
  then a second arm pair with `ANCHORED_RUNS=1` on the fix (X74's question again).
- Runs, written before they start: Xeon 6240 at 16 and 8 threads, X74's X16 and X8 setups
  (client catalogue tracking tier and the five standard batches, at least 20 rounds); EPYC 7763
  at 16 threads (Zen 3, anchored there already); Mac M5 Max at 6 threads (anchored), on AC power.
- **Adopt the fix** if, in every run, no case is slower than 1 + band (whole interval above), the
  client catalogue's weighted mean is at most 1.00, and on the Xeon at 16 threads c2c 864 × 384
  backward is unimodal: every process within 1.15 of the median of the case alone (X76's 66–71
  µs, M) or the near-miss re-run says so.
- **Then anchoring on the Xeon 6240 class:** re-run X74's rules on the fix (A against M, X8, X8b,
  X16, X16b), with the deciding X16 re-run declared before it starts. Adopt `anchored_runs` on the
  class if they pass; by X74 everything else already gained (X16 0.949, X16b 0.581) and X76
  shows the 864 batch at 0.74 of its arrival-order time when neither mode is broken.

**Without the fix,** anchoring on the class should not be judged on this batch: its slow mode
is a property of the heap's history in the fbench process, not of the claiming rule, and the
same page sharing slows the arrival-order arm too (M 95–107 against 67). Excluding the batch
(the narrowing clause) would leave the cause in place for every plan whose twiddles happen to
precede its workspace.

## Verdict

Informative; no default changed and no library change. The mode is picked by the heap: after
earlier cases in an fbench process the workspace is allocated from the heap directly after the
plan's twiddle table, and the twiddles that share the team state's 4 KiB page are then loaded
from other cores' modified lines on every execution (`perf c2c`: up to 93 % of the HITM samples
on that one page). Thread placement, page colours, buffer offsets, frequency and SMT siblings
do not pick it (each varied alone, all processes fast). A workspace starting a page
(`WORKSPACE=page`) removes the slow mode in both arms. The proposal above drafts the library fix
and the rules for it and for anchoring on the Xeon class after it.
