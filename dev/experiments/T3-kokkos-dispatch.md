# T3: one Kokkos parallel_for per transform; team_barrier

- **Status:** running
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

To be filled in from the runs.

## Verdict

Pending.
