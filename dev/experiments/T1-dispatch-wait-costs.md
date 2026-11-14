# T1: dispatch, wake and wait costs per machine and runtime

- **Status:** done (2026-11-14)
- **Owner:** lead, 2026-11-14 (protocol and tooling prepared with a sub-agent)
- **Phase / task:** 3.4

## Question

What does each fixed cost of a team execution cost on each machine and threading runtime
(an executor's dispatch and return, the engine's own fixed cost, a phase hand-off inside
a cache group, one claim, a wake-up from each stage of waiting), and therefore which values
should the machine profile (`src/threading/schedule.hpp`) carry for `ns_per_work`,
`dispatch_ns`, `inline_multiple`, `sync_ns`, `spin_ns`, `yield_ns` and `claim_divisor`?
The split thresholds are experiment [T5](T5-split-thresholds.md); `cpus` is detected.

## Hypothesis

The provisional values come from the prototype (DESIGN.md 13.4 and the comments in
`schedule.hpp`): fork/join 0.5–9.8 µs (libgomp: 0.69/1.70 µs barrier/fork-join at 6
threads, 2.51/9.80 µs at 64 on Zen 3; 1.15/3.17 µs at 18 on the Xeon), so `dispatch_ns`
2000; a phase boundary of 0.4 µs on an Apple cluster, 0.7 µs in a Zen 3 CCX and 1.1 µs on
the 18-core Xeon mesh, so `sync_ns` 700; waking parked threads 10–50 µs per call at 64
lanes, and a spin budget from the 5–100 µs range, so `spin_ns` 20000 and `yield_ns` 50000;
`ns_per_work` 0.5, the walking skeleton's scalar passes.

Expected: FFTN's pool dispatches hot (threads still spinning) in about the prototype's
fork/join time, growing with the team (one contended claim word, one epoch word);
dispatch to parked threads costs 10–50 µs at 64 lanes and is dominated by the futex
wake-ups. libgomp and libomp regions cost the prototype's figures; libomp on a hybrid Mac
with its default block time does not spin (14 µs per region measured before). The engine's
phase hand-off (a claim, a done-counter add, a wait on it) costs less than the prototype's
barrier because no member waits for all others: about 0.3–1 µs inside a group, more on the
mesh. `ns_per_work` is now several times below 0.5 (the SIMD passes of task 1.2), which
makes rule 1 run more jobs inline and the cost model split less.

## Change under test

T1 is a calibration: its arms are machines, runtimes and executors, measured with a
dedicated harness, and its outcome is a set of profile values. The change those values
make is then checked as one A/B per machine on the threads catalogue: arm A the provisional
profile, arm B the measured one, both through the FFTN backend's profile options
(`bench/README.md`, "FFTN options"), which reach the library only through the private plan
core's build settings, as a test's do. No library switch is added, so `dev/KNOBS.toml` lists
none.

## Method

**Harness.** `bench/t1/t1_costs.cpp`, built as `<build>/bench/t1_costs` and, where
the compiler has OpenMP, `t1_costs_omp` (one OpenMP runtime per binary, the compiler's);
run by `bench/t1/t1.py`, which launches every binary once per round in rotating
order (process rounds, at least 10), checks the machine is quiet before each round, and
stores the run as JSON lines under `bench-out/t1/`. Probes (each reports, per team size,
the median and the 10th/90th percentiles over repetitions inside a process; the driver
reports the median over rounds and the spread between rounds):

| Probe | What it times | Informs |
|---|---|---|
| `work` | serial execution of c2c rows 2⁸–2¹⁸ (1 Mi points per batch), divided by the plan's work units | `ns_per_work` |
| `bulk` | an executor's `bulk` of one task per lane: time until every lane runs (`start_last`), from the last task's end to `bulk`'s return (`return`), the round trip; executors `thread_pool` and `omp::executor`; after no gap (hot), 40 µs (the pool's yield stage) and 2 ms (parked) | `dispatch_ns`; the pool's linger |
| `team` | a real plan of one 64-point row per lane through the executor overload (and, in `t1_costs_omp`, the team overload inside a running region), minus a serial row; scheduled with `dispatch_ns` 0 so that it is dispatched | `dispatch_ns`, `inline_multiple` |
| `phase` | the claiming engine itself on a synthetic split task of 64 phases of k chunks among k members (harness threads, or the threads of a region), chunks empty or 1 µs long: time per phase beyond the chunks' own | `sync_ns` |
| `claim` | k threads claiming from one counter with the engine's compare-exchange loop | `claim_divisor` |
| `wake` | notify-to-running latency between two threads for a pure spin, a yield loop, `std::atomic::wait`, and the engine's policy (`wait_for`) after the waiter has waited 1–1000 µs; and the cost of `notify_all` | `spin_ns`, `yield_ns` |

- Team sizes 2, 4, 8, 16, 32 and 64 (up to the CPUs in use); groups by cache
  (`fftn::cache_group_size`). Harness threads pinned one per core where the OS allows it
  (`--pin`, Linux); OpenMP threads by `OMP_PLACES=cores OMP_PROC_BIND=close`; the pool
  unpinned, as users get it. `wake` and `claim` pairs inside one cache group and across
  groups when pinned.
- Runtimes wait differently by default, and that is part of what is measured: libgomp
  spins `GOMP_SPINCOUNT` iterations unless `OMP_WAIT_POLICY` says otherwise; libomp spins
  for `KMP_BLOCKTIME` (200 ms) on Linux but not at all on a hybrid Mac. Each OpenMP run is
  made with the defaults, and once more on the Mac with `KMP_BLOCKTIME=200`; the harness
  records these variables.
- Machines, compilers, runtimes: EPYC 7763 (Zen 3, CCX of 8 cores), an exclusive Slurm
  node, one socket's 64 cores, GCC 15 (libgomp), and libomp if a Clang build is available;
  Xeon Gold 6240 (18-core mesh per socket), one socket and both, GCC 15 (libgomp) and
  Clang 23 (libomp), only when quiet or exclusive; Apple M5 Max, the performance cores,
  Clang 23 (libomp), only when the quiet check passes. Release builds, as users get them.
  Kokkos (named in the index row) waits for task 3.5's executor, and its probe is added
  then.
- Metric: per probe, team size and machine, the median over process rounds of the
  in-process medians, with the range over rounds. A value is trusted when its range over
  rounds is within ±20 % of its median; otherwise the round count is doubled once, and the
  value is reported as unstable if that does not help.

**Decision rule, written before running.** Per machine class (Zen 3 CCX, Xeon mesh, Apple
cluster; the profile is chosen by the class, as rule 3 already chooses by group kind):

1. `ns_per_work` := the median over the `work` shapes; reported per instruction set.
2. `dispatch_ns` := the median `team` overhead of the pool, hot, at the full width
   measured (64, 18, the performance cores): the cost a back-to-back call pays at the
   width rule 1 judges. If that overhead at full width exceeds the one at a cache group's
   width by more than 2×, the verdict opens a follow-up for a per-lane term (a design
   change), and the full-width value is used meanwhile.
3. `sync_ns` := the median `phase` overhead per phase with 1 µs chunks at k = a cache
   group's members, pinned.
4. `spin_ns` := the median wake latency of a parked waiter (`policy` after 1 ms), rounded
   up to a whole microsecond and clamped to 5–100 µs: spinning about as long as blocking
   costs bounds the waste at twice the optimum. `yield_ns` := 2.5 × `spin_ns` (the
   provisional ratio) if the yield stage's latency is below the parked latency by more
   than its range over rounds, else 0 (no yield stage).
5. `inline_multiple` and `claim_divisor` are not derived from one probe: each is tried at
   the provisional value, half and double it (claim divisor 1, 2, 4) on the threads
   catalogue below, and changed only if a value beats the provisional one there.
6. **Adoption.** On each machine, fbench on the `threads` catalogue, quick tier, at the
   full width with the `fftn_omp` backend (pinned) and once with the pool: arm A
   provisional, arm B every field from 1–5, plus one arm per field changed alone. B is
   adopted if its weighted geomean ratio is at or below 1.00 with its interval clear of
   the noise band or inside it, and no case is slower than A by more than the band plus
   2 %, nor slower than serial (`THREADS=1`) by more than 2 % (the phase 3 exit). A field
   whose own arm is within the noise band on every case keeps its provisional value and
   is recorded as measured without effect. The adopted values go into the profile with a
   comment citing this file and the run IDs.

## Results

Three machines, harness runs stored as JSON lines with the host fingerprint (`t1.py show
<run>` prints every probe; the run directories are on the machines named). The second
campaign built ec63bb3 (`main` then); its DMOG and Xeon run records carry the label
62958fc (`FBENCH_SOURCE_COMMIT`, set by mistake to an ancestor), the Mac's say ec63bb3:

| Run | Machine, build, placement | Commit | Probes |
|---|---|---|---|
| 20261114-080613-96e1 | EPYC 7763 node, exclusive, cores 0–63 (one socket, 8 CCXs); GCC 15.2 (libgomp), AVX2; `--pin`, `OMP_PLACES=cores OMP_PROC_BIND=close`; 10 rounds (DMOG job 508943) | 148b0d6 | all |
| 20261114-175237-317d | same node type, `-c 64 --distribution=block:block` (cores 0–63), GCC 15.2; 10 rounds (DMOG job 509110) | ec63bb3 | `work`, `team` |
| 20261114-174647-f139, -174721-6158 | Apple M5 Max, the 6 fastest cores (one cluster), Clang 23.1 (libomp), NEON+FCMA; 6 rounds, loaded Mac (load 7); the second with `KMP_BLOCKTIME=200` | ec63bb3 | all |
| 20261114-175414-29ab | Xeon Gold 6240, 18 cores of socket 0, shared (load 3–20), GCC 14.4 (libgomp), AVX-512; 6 rounds (Slurm job 15569) | ec63bb3 | all |

**What the first campaign still says.** Job 508943 ran at 148b0d6, before the scheduler
fixes 5a1cd72 (dispatch no more members than tasks when nothing splits) and fb91770 (rule 1
decided at plan construction). The engine, the pool and the waiting code
(`engine.hpp`, `team_execute.cpp`, `thread_pool.cpp`, `wait.hpp`) did not change between
148b0d6 and ec63bb3, so its `bulk`, `phase`, `claim` and `wake` probes stand. Its `work`
probe does not (radix-32/64 codelets and the grid strategy changed the passes, and with
them the work units), and its `team` probe goes through the executor path that fb91770
changed: both were re-run (job 509110). Its fbench sweeps are T5's and were all re-run
there. Two of its probe groups were never valid: inside `t1_costs_omp`, every probe that
does not use OpenMP threads (the pool's `bulk` and `team`: 13 ms per call at 2 lanes,
260 ms at 64; the crew's `phase`, `claim` and `team`) ran on one core. libgomp with
`OMP_PROC_BIND=close` binds the initial thread to the first place, and threads created
afterwards inherit that one-CPU mask. Job 509110 confirmed it: the pool's hot round trip
in `t1_costs_omp` is 32 ms at 8 lanes with the binding and 4.7 µs without
(`OMP_PROC_BIND`, `OMP_PLACES` unset), the same as in `t1_costs`. Only the OpenMP probes of
`t1_costs_omp` are used below.

Medians over rounds (ns); "spread" is the range over rounds where it exceeds T1's ±20 %:

| Probe | EPYC 7763 (Zen 3) | Xeon 6240 (mesh) | M5 Max (cluster) |
|---|---|---|---|
| `work`, ns per work unit (6 shapes; range) | 1.00 (0.69–1.39) | 1.23 (1.02–2.56) | 0.57 (0.45–0.66) |
| `team` pool, hot, at 2 / group width / full width | 1690 / 5630 (8) / 25 970 (64, spread 10×) | 1070 / 11 180 (18) | 600 / 2030 (6) |
| `team` omp::executor, hot, 2 / group / full | 1710 / 3270 / 34 800 | 1470 / 8580 | 5140 / 26 310 (libomp default); 660 / 2380 (`KMP_BLOCKTIME=200`) |
| `team` OpenMP region (team overload), group / full | 2000 / 24 370 | 4220 | 554 (1170 with block time) |
| `bulk` round trip, pool hot / parked, group / full | 4000 / 17 270; 14 070 / 59 410 | 8100 / 108 500 (18) | 1420 / 21 850 (6) |
| `bulk` round trip, omp::executor hot, group / full | 2410 / 17 540 | 8390 | 20 250 (default); 1625 (block time) |
| `phase`, per phase, 1 µs chunks, k = group (pinned crew) | 1141 (k = 8); 5284 (16), 22 380 (64) | 2846 (k = 18); 1063 (8) | 461 (k = 6) |
| `phase`, same, in an OpenMP region | 1465 (8) | 2775 (18) | 941 (6; 463 with block time) |
| `claim`, ns per claim, k = group | 319 (8), 454 (64) | 221 (18) | 36 (6) |
| `wake` spin / yield latency | 96 / 617 (near) | 123 / 197 | 83 / 125 |
| `wake` policy after 30–60 µs (yield stage) | 930–1080 | 250–280 | 166 |
| `wake` policy after 100–300 µs (blocked) | 5020–5050 near, 7050–7160 far | 2450–3280 | 5310–6650 |
| `wake` policy after 1 ms (blocked) | 5405 near, 8466 far | 10 320 near, 13 450 far (spread 60–73 %) | 6354 |
| `wake` cost of `notify_all` to a blocked waiter | 1400 near, 2340 far | 2280–2720 | 1000 |

Values per rule, written as the rule says:

1. `ns_per_work`: EPYC 1.00 (AVX2), Xeon 1.23 (AVX-512), Mac 0.57 (NEON+FCMA). The six
   shapes spread by a factor of two on every machine (long rows cost more per unit than
   batches of short ones), so the cost model's single figure is a rough one.
2. `dispatch_ns` (pool, hot, at full width): EPYC 25 970 (its range over rounds is 10×:
   one round in ten took 270 µs; the median is the first campaign's 23 970 with a 6 %
   spread), Xeon 11 180 (spread 27 %, shared machine), Mac 2030. The EPYC's full-width
   cost is 4.6× its cache group's (5630 at 8 lanes) and the omp::executor's 10.6× (34 800
   against 3270): far beyond the rule's 2×. The follow-up for a per-lane term is opened
   (below); the Xeon's and the Mac's group is their full width, ratio 1.
3. `sync_ns` (k = a group's members, 1 µs chunks, pinned): EPYC 1141 → 1100; Xeon 2846 →
   2800; Mac 461 → 500.
4. `spin_ns` (parked latency after 1 ms, rounded up, clamped 5–100 µs): EPYC 9000 (far;
   6000 near), Xeon 14 000 (unstable), Mac 7000. The yield stage wakes in well under the
   parked latency on all three (0.17–1.1 µs against 5–13 µs), so `yield_ns` = 2.5 ×
   `spin_ns`: 22 500, 35 000, 17 500.
5. `inline_multiple` (1.5, 6) and `claim_divisor` (1, 4): arms of the adoption runs.


**Adoption (rule 6).** fbench, threads catalogue, quick tier (82 cases), 15 rounds, EPYC
node as above, DMOG jobs 509117 and 509118. Weighted geomean against A (provisional
profile), with the worst cases:

| Run, executor, width (band) | Arm | Geomean | Worst cases (ratio to A) |
|---|---|---|---|
| 20261114-181442-11ed, OpenMP, 64 (3.06 %) | B (1–4 together) | 1.062 [1.044, 1.070] | rows of 6144–10240 points 1.17–2.07 (inlined) |
| | NS (`ns_per_work` 1.0) | 1.011 | c2c 256 × 8 3.08 (now dispatched to the 64-thread region), r2c 4096 2.38 (now split) |
| | DISP (`dispatch_ns` 26 000) | 1.151 | rows of 6144–20480 points and 128² planes 1.17–4.2 (inlined) |
| | SYNC (`sync_ns` 1100) | 0.989 [0.975, 0.995] | 128³ 1.034 [0.964, 1.149]; nothing slower beyond the band |
| | SPIN (9000 / 22 500) | 1.004 | 8192 × 9 1.107, 128³ 1.083, 16384 × 3 1.074 |
| | IM 1.5 / IM 6 | 1.003 / 0.979 [0.967, 0.986] | IM 1.5: c2c 256 × 8 3.09; IM 6: c2r 262144 1.066 [0.959, 1.143] |
| | CD 1 / CD 4 | 0.972 [0.959, 0.979] / 1.011 | CD 1: c2c 4096 × 1024 1.034 [1.024, 1.055], batches of 1024 short rows 0.70–0.83; CD 4: those batches 1.20–1.39 |
| 20261114-181442-218b, pool, 64 (5.47 %) | B / DISP | 1.036 [1.027, 1.044] / 1.081 | DISP: 16384-point rows and 128² planes 1.83–3.9 |
| 20261114-182917-95c2, OpenMP, 8 (1.71 %) | B / DISP | 1.182 / 1.291 | c2c 4096 × 4 4.9, 128² 5.6, 16384-point rows 4.4 (inlined) |
| | B without DISP | 1.007 [0.999, 1.016] | r2c 4096 1.57 (split once `ns_per_work` is 1.0) |

The serial arm shows what the provisional profile does wrong at 64 lanes: a lone row of
3072 or 4096 points ran 1.37–1.75 times slower than serial with the OpenMP executor and
1.24–1.53 with the pool, r2c and c2r of 16384 points 1.26 and 1.20 with the pool. T5's
thresholds keep those rows on one thread.

## Verdict

Done, per machine class:

- **Zen 3 CCX (the defaults):** `sync_ns` 1100 and `claim_divisor` 1 adopted (their own
  arms beat A with nothing slower beyond the band; with T5's thresholds, the final profile F
  in T5's runs 20261114-185256-4009 and -431c). `inline_multiple` stays 3: IM 6 beat A at
  64 lanes only through lone 3072- and 4096-point rows, which T5's threshold now keeps on
  one thread, and on top of T5's values it changed nothing (1.012 against F's 1.013 at 8
  lanes, 0.982 against 0.982 at 64): measured without effect. `ns_per_work` (1.0
  measured), `dispatch_ns` (26 000 measured at 64 lanes, 5600 at a group's 8) and
  `spin_ns`/`yield_ns` (9000/22 500) keep their provisional values (0.5, 2000,
  20 000/50 000): each made cases slower beyond the band. The first two fail for one reason:
  rule 1 and the split cost model price a dispatch as one constant, while it grows about
  linearly with the executor's width (0.36 µs per lane for the pool, 0.56 for the OpenMP
  executor on the EPYC). With the measured full-width cost, jobs a cache group runs 2–5 times faster were inlined; with the
  measured work per unit and a group's dispatch cost, small batches went out to 64-thread
  regions and r2c rows of 4096 points were split. Rule 2's follow-up is therefore opened
  as a design change (below), and `ns_per_work` is re-tried with it.
- **Apple M5 cluster and the Xeon mesh:** measured (table above) but not adopted: the
  all-fields arms failed on both (M5 run 20261114-181503-ace8, loaded Mac, band 3.9 %:
  1.003 with batches of 8192-point rows 1.09–1.14; Xeon run 20261114-182732-001f, arm T:
  1.006 with c2c 1024 × 8 1.42, inlined by `dispatch_ns` 11 000). Their profiles keep the
  provisional `sync_ns` 700 and `claim_divisor` 2 (`profile_for`); per-field arms on a quiet
  Mac and on the Xeon are a follow-up once the dispatch model has a width term.
- **Spin budget:** the rule's "spin about as long as blocking costs" is wrong for FFTN's
  use: back-to-back calls arrive within tens of microseconds, so a 9 µs spin made members
  block and pay the 5–9 µs wake between executions. The budget should cover the gap
  between calls; 20 µs stays.

Follow-ups: (1) a dispatch cost of the form a + b × lanes in the machine profile, used by
rule 1 (with the executor's concurrency, which plan construction does not know: either
decide rule 1 at execution again, cheaply, or bound it by the CPU count) and by rule 3's
cost model when splitting a lone transform causes a dispatch that running it whole would
not (T5 shows the threshold moving from 3072 to 6144 points with the executor's width);
then re-try `ns_per_work` from this file. (2) `fftn::thread_pool` started after a bound
OpenMP runtime inherits a one-CPU mask (above): the pool could widen its threads' masks to
the process's allowed CPUs (its own threads; DESIGN 13.5 forbids only re-pinning threads
FFTN does not own). (3) Kokkos's executor probe when task 3.5 lands. Lessons 5.1, 5.5–5.8
in `dev/LESSONS.md`.
