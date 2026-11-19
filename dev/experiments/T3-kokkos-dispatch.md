# T3: one Kokkos parallel_for per transform; team_barrier

- **Status:** done (2026-11-19)
- **Owner:** task 3.5 agent, 2026-11-19
- **Phase / task:** 3.5

## Question

Does `fftn::kokkos::executor`, which dispatches one `Kokkos::parallel_for` per transform
followed by a fence, cost anything that matters beside the OpenMP region it wraps, or must
the OpenMP back end bypass Kokkos (the prototype's route)? And what should the documentation
recommend for transforms inside a `TeamPolicy` kernel, where `team_barrier` is the
synchronisation Kokkos offers?

## Hypothesis

On the OpenMP back end a `parallel_for` is a `#pragma omp parallel for` under the instance's
mutex (Kokkos 5.2 sources), so the executor should add well under 1 µs to a dispatch of
several µs: invisible on transforms of 100 µs and more. A `TeamPolicy` region does more set-up
(team data) than a `RangePolicy` one. `team_barrier` is Kokkos' `HostBarrier`, which spins and
then yields and sleeps, so it should be near an OpenMP barrier when the team is balanced and
worse after a member arrives late; FFTN's `completion::all` needs no barrier at all, so
`completion::own` plus `team_barrier` should not beat it.

## Change under test

No switch and no library change between arms: the harness runs the routes side by side in one
process, on the same plan and data, interleaved repetition by repetition.

| Arm | What runs |
|---|---|
| `kokkos` | `plan.execute(in, out, fftn::kokkos::executor, ws)`: the implementation under test, and the base of every ratio |
| `kokkos2` | the same again with its own workspace: the identical-arm control |
| `omp` | `plan.execute(in, out, fftn::omp::executor, ws)` on the same OpenMP runtime: the bypass (OpenMP builds) |
| `serial` | the serial overload, for the speed-up; timed in its own pair with `kokkos` (see the correction below) |
| `team_all` | one `TeamPolicy(exec, 1, P)` kernel in which every member calls the team overload, `completion::all` |
| `team_own_barrier` | the same with `completion::own`, then `member.team_barrier()` |
| `team_four_in_a_region` | four executions in one team kernel, time per execution |
| `omp_team` | the threads of a plain `#pragma omp parallel` as the team (`fftn::omp::this_team()`): the team route without Kokkos' team set-up (added after the first run, recorded, not judged) |

The `dispatch` probe times empty dispatches (`kokkos`, `kokkos2`, `parallel_for` without the
fence, `team_region`, `omp`); the `scale` probe, `view_plan` with `normalization::none` and
with `backward` at 2^14 to 2^18 elements; the `barrier` probe, `team_barrier` beside an OpenMP
barrier, with one member arriving 0, 10 and 100 µs late.

## Method

- Harness: `bench/t3/t3_kokkos.cpp` (target `t3_kokkos`), one process per round, combined by
  `bench/t3/t3.py`. Each case repeats its arms in rotating order for 1 s; a case's result is
  the median over 12 process rounds of the in-process median, and the median over rounds of
  the paired ratio to `kokkos`, with its range over the rounds.
- Transforms (double, out of place): rows 64×1024, 64×4096, 512×4096, 8×65536; planes
  16×192², 4×512², 1×1024²; volumes 2×64³.
- Machines (shared Slurm allocations, never exclusive):
  - DMOG EPYC 7763, `-c 16 --hint=nomultithread --distribution=block:block`, GCC 16 (libgomp),
    P = 8 and 16;
  - Xeon Gold 6240, `-c 16` (8 cores with their hyperthreads), GCC 14.4 (libgomp), P = 8.
  - `OMP_PLACES=cores OMP_PROC_BIND=close`; Kokkos 5.2.1 built by the test configuration
    with `FFTN_KOKKOS_BACKEND=OpenMP`, Release. The Threads back end is run too, where the
    only comparators are its own arms.
- The cases the change can move are all of them (the route is the arm), so the rule names
  every case that dispatches and is long enough for a dispatch to be small against it.

**Decision rule, written before running.** A/A band of a case: the larger of 1 % and the
distance of `kokkos2`'s ratio from 1 (the median over rounds, widened by its range).

1. **Route.** On both machines, at every width, for every transform of at least 100 µs per
   execution (`kokkos` arm): the OpenMP arm must not be faster than `kokkos` by more than the
   band plus 2 % (`omp` ratio ≥ 0.98 − band). If so, one `parallel_for` per transform stands.
   If a case fails it, the executor gets an OpenMP-specific path that opens the region itself,
   and this experiment is run again. Shorter transforms are recorded, not judged: they are
   where a dispatch costs a noticeable share by nature, and rule 1 of the scheduler decides
   whether they dispatch at all.
2. **Team kernels (documentation).** `docs/kokkos.md` recommends `completion::all`. It adds
   `completion::own` with `team_barrier` as equally good only if `team_own_barrier` is within
   2 % of `team_all` (plus the band) on every transform of rule 1; otherwise it says that the
   barrier route costs more. It recommends running several transforms in one team kernel only
   if `team_four_in_a_region` beats `kokkos` by at least 3 % beyond the band on a majority of
   the transforms of rule 1.
3. **Scaling threshold.** `view_plan` shares its scaling from 2^16 elements. That stays if on
   both machines, at every width, the shared scaling of 2^16 elements (`scaled` minus `none`)
   costs no more than twice the serial scaling of 2^15 elements. Otherwise the threshold moves
   to the smallest size measured where shared scaling wins.

**Correction after the first run, before the second (the rules are unchanged).** In the first
run the serial arm was part of the cycle of arms, so the arm after it (`team_all`, always,
since the rotation keeps the cyclic order) found the data in one core's cache and paid for
moving it: `team_all` came out 1.1 to 1.3 of `kokkos` where four executions in one region came
out 0.98. The `serial` arm now runs in a separate pair, and the second run replaces the first
for the `transform` probe; the `dispatch`, `scale` and `barrier` probes of the first run stand.
Rule 1 is judged on the second run too (it passed on the first).

## Results

Runs: first run DMOG job 513179 and Xeon job 16558 (commit 8ea7e0c: `dispatch`, `scale` and
`barrier` probes used from these); second run DMOG job 513206 and Xeon job 16579 (commit
c3a6311: the `transform` probe). Raw JSON lines of every round are in
`/mnt/data/shared/jt52/fftn-t3` (`out-epyc`, `rerun-epyc`) and `~/awork/fftn-t3` on the Xeon.
12 process rounds of 1 s per case. DMOG's node 21 was loaded by other work (load 42 to 47) and
its CPUs came scattered over the socket; the Xeon too ran beside other jobs (load about 35), so
every ratio is judged against the identical-arm band of its own case.

**Dispatch, empty tasks, ns** (median over rounds; all at one team width):

| Machine, back end, P | `kokkos` | `parallel_for` alone | `omp` | `team_region` |
|---|---|---|---|---|
| EPYC, OpenMP, 8 | 2887 | 2721 | 2645 | 3321 |
| EPYC, OpenMP, 16 | 4213 | 4027 | 3767 | 6322 |
| EPYC, Threads, 8 | 2454 | 2556 | | 5320 |
| EPYC, Threads, 16 | 3271 | 3011 | | 7329 |
| Xeon, OpenMP, 8 | 2134 | 2056 | 1920 | 3778 |
| Xeon, Threads, 8 | 1846 | 1700 | | 3794 |

The executor costs 0.2 to 0.45 µs more than the OpenMP region it wraps, of which the fence is
about 0.1 to 0.2 µs. A `TeamPolicy` region costs 0.6 to 4.3 µs more than a `RangePolicy` one.

**Transforms of at least 100 µs, time ÷ `kokkos`** (median over the cases, then the range over
the cases; the second run):

| Machine, back end, P | cases | `omp` | `kokkos2` (A/A) | `team_all` | `team_own_barrier` | `team_four_in_a_region` | speed-up over serial |
|---|---|---|---|---|---|---|---|
| EPYC, OpenMP, 8 | 7 | 1.004 (0.915–1.058) | 0.994 (0.982–1.036) | 1.012 (0.979–1.039) | 1.144 (0.954–6.28) | 1.003 (0.574–1.044) | 3.3–5.9 |
| EPYC, OpenMP, 16 | 6 | 1.006 (0.950–1.062) | 0.986 (0.941–1.017) | 1.013 (0.696–1.080) | 1.640 (1.001–6.01) | 1.039 (0.609–1.291) | 3.1–7.5 |
| Xeon, OpenMP, 8 | 7 | 1.004 (0.997–1.024) | 0.994 (0.992–1.003) | 1.004 (0.999–1.016) | 1.141 (1.103–7.52) | 1.000 (0.980–1.018) | 5.2–6.9 |
| EPYC, Threads, 8 | 7 | | 1.000 (0.941–1.029) | 1.005 (0.932–1.039) | 1.174 (1.037–5.34) | 1.003 (0.798–1.044) | 3.0–6.8 |
| EPYC, Threads, 16 | 7 | | 0.988 (0.974–1.003) | 1.061 (0.990–1.142) | 1.157 (0.967–5.85) | 1.013 (0.959–1.106) | 2.6–7.4 |
| Xeon, Threads, 8 | 7 | | 0.989 (0.983–1.004) | 0.996 (0.985–1.005) | 1.179 (0.999–6.65) | 0.986 (0.960–1.000) | 4.8–6.9 |

`omp_team` (the team overload in a plain OpenMP region) came out at 1.005, 1.038 and 1.000 of
`kokkos` for the three OpenMP rows: the team route costs what the executor route costs, with or
without Kokkos. The widest entries are the EPYC's single long rows (8 × 65536), whose repeats
are bimodal there (their A/A band is 0.165 and 0.081); on the Xeon every arm of that case is
within 1 %.

**`team_barrier`, ns per barrier** (time per barrier as the last member sees it):

| Machine, P | balanced | beside `#pragma omp barrier` | one member 10 µs late | 100 µs late |
|---|---|---|---|---|
| EPYC, OpenMP, 8 | 1478 | 915 | 10798 (OpenMP 11270) | 100274 (101334) |
| EPYC, OpenMP, 16 | 3000 | 1309 | 11356 (11232) | 100370 (101446) |
| Xeon, OpenMP, 8 | 2494 | 611 | 11119 (10318) | 100444 (100360) |
| EPYC, Threads, 8 and 16 | 1217, 1798 | | 11933, 12476 | 104830, 105466 |
| Xeon, Threads, 8 | 1162 | | 11470 | 101691 |

A late member adds no penalty beyond its lateness on the OpenMP back end, up to 100 µs; on
Threads about 2 to 5 µs at 100 µs late. In this test the balanced barrier costs 1.6 to 4 times an
OpenMP barrier.

**Scaling** (`scaled` minus `none`, first run): serial below 2^16 elements, 22 to 26 µs at 2^14
and 44 to 53 µs at 2^15 on the EPYC, 25 and 49 µs on the Xeon with OpenMP (42 and 82 with Threads);
shared from 2^16: 27 to 49 µs on the EPYC, 33 µs on the Xeon, against the 88 to 107 and 97 to
164 µs that twice the 2^15 figure gives for one thread.

## Verdict

1. **Route: one `parallel_for` per transform stands.** No transform of at least 100 µs on either
   machine, at any width, has the direct OpenMP route faster than the band allows (`omp` ≥ 0.98 −
   band): the EPYC's closest case is 8 × 65536 at 8 lanes (0.915, within its band of 0.165); on
   the Xeon the lowest is 0.997. No OpenMP-specific path is added.
2. **Team kernels.** `team_all` matches the executor (median over cases 0.996 to 1.061), so
   `docs/kokkos.md` recommends a team kernel when other work shares it, and the executor
   otherwise. `completion::own` with `team_barrier` fails the 2 % test on every machine (a batch
   of few transforms then runs on few members, 5 to 8 times slower for one plane), so the docs
   say it costs more and recommend `completion::all`. Running four transforms in one kernel beats
   `kokkos` by 3 % beyond the band on at most one case per run: not recommended as a speed-up.
3. **Scaling threshold.** Shared scaling of 2^16 elements cost less than twice the serial
   scaling of 2^15 on every machine, width and back end: the threshold stays at 2^16.

Not measured: more than 16 threads, Apple machines (the Mac was not timed), and libomp in these
runs. Lessons 1.13 and 5.14 in `dev/LESSONS.md`.
