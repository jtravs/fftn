# T1: dispatch, wake and wait costs per machine and runtime

- **Status:** planned
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

**Harness.** `bench/threading/t1_costs.cpp`, built as `<build>/bench/t1_costs` and, where
the compiler has OpenMP, `t1_costs_omp` (one OpenMP runtime per binary, the compiler's);
run by `bench/threading/t1.py`, which launches every binary once per round in rotating
order (process rounds, at least 10), checks the machine is quiet before each round, and
stores the run as JSON lines under `bench-out/t1/`. Probes (each reports, per team size,
the median and the 10th/90th percentiles over repetitions inside a process; the driver
reports the median over rounds and the spread between rounds):

| Probe | What it times | Informs |
|---|---|---|
| `work` | serial execution of c2c rows 2⁸–2¹⁸ (1 Mi points per batch), divided by the plan's work units | `ns_per_work` |
| `bulk` | an executor's `bulk` of one task per lane: time until every lane runs (`start_last`), from the last task's end to `bulk`'s return (`return`), the round trip; executors `thread_pool` and `omp::executor`; after no gap (hot), 40 µs (the pool's yield stage) and 2 ms (parked) | `dispatch_ns`; the pool's linger |
| `team_execute` | a real plan of one 64-point row per lane through the executor overload (and, in `t1_costs_omp`, the team overload inside a running region), minus a serial row; scheduled with `dispatch_ns` 0 so that it is dispatched | `dispatch_ns`, `inline_multiple` |
| `phase` | the claiming engine itself on a synthetic split task of 64 phases of k chunks among k members (harness threads, or the threads of a region), chunks empty or 1 µs long: time per phase beyond the chunks' own | `sync_ns` |
| `claim` | k threads claiming from one counter with the engine's compare-exchange loop | `claim_divisor` |
| `wake` | notify-to-running latency between two threads for a pure spin, a yield loop, `std::atomic::wait`, and the engine's policy (`wait_for`) after the waiter has waited 1–1000 µs; and the cost of `notify_all` | `spin_ns`, `yield_ns` |

- Team sizes 2, 4, 8, 16, 32 and 64 (up to the CPUs in use); groups by cache
  (`fftn::cache_group_size`). Harness threads pinned one per core where the OS allows it
  (`--pin`, Linux); OpenMP threads by `OMP_PLACES=cores OMP_PROC_BIND=close`; the pool
  unpinned, as users get it. `wake` and `claim` pairs inside one cache group and across
  groups when pinned.
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
2. `dispatch_ns` := the median `team_execute` overhead of the pool, hot, at the full width
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

None yet.

## Verdict

Pending.
