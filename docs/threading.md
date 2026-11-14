# Threading

FFTN runs on the caller's threads. It creates no thread unless the program constructs an
`fftn::thread_pool`, never changes a threading runtime's settings, and executes serially
unless the caller hands it a team or an executor. The declarations, preconditions and
guarantees are in the public headers (`include/fftn/team.hpp`, `include/fftn/executor.hpp`,
`include/fftn/thread_pool.hpp`, `include/fftn/omp.hpp`, and the overloads of
`plan::execute` in `include/fftn/plan.hpp`); this page says how the pieces fit together.

There are two ways to share an execution among threads:

- **A team**: the caller's threads are already running (an OpenMP parallel region, a pool's
  workers, `std::thread`s), and each of them calls `execute` with its own `fftn::team`.
  Nothing is dispatched; the program's own work before and after the transform runs in the
  same region.
- **An executor**: one thread calls `execute` with an executor, anything that can run n
  tasks and return when they are done, and FFTN dispatches the team itself: an OpenMP
  region (`fftn::omp::executor`), FFTN's pool (`fftn::thread_pool`), or an adapter of a few
  lines over the program's own pool.

## A team of the caller's threads

Every member of a team calls the same plan's `execute` with its own `fftn::team` (its rank,
the team's size and, optionally, how many consecutive ranks share a cache) and one shared
workspace, sized for the team:

```cpp file=examples/team_threads.cpp region=team
const fftn::c2c_plan<double> plan(fftn::rows_c2c<double>(n, count, fftn::direction::forward));
const int members = 4;

// One workspace for the whole team: scratch for each member and the team's shared state.
fftn::workspace ws(plan, members);

std::vector<std::complex<double>> y(n * count);
std::vector<std::thread> team;
team.reserve(members);
for (int rank = 0; rank < members; ++rank) {
  team.emplace_back([&, rank] {
    const fftn::team me{.rank = rank, .size = members};

    // Every member calls execute; it returns when the whole output is written.
    plan.execute(x.data(), y.data(), me, ws);

    // With completion::own a member returns as soon as its own rows are written, and
    // can go straight on to work on them: the rows of transforms_of(me).
    plan.execute(x.data(), y.data(), me, ws, fftn::completion::own);
    const fftn::transform_range mine = plan.transforms_of(me);
    for (std::size_t i = mine.first * n; i < (mine.first + mine.count) * n; ++i) {
      y[i] /= static_cast<double>(n);
    }
  });
}
for (std::thread& t : team) {
  t.join();
}
```

`fftn::cache_group_size(members)` gives a `group_size` for a team whose consecutive ranks
run on consecutive cores, from the machine's topology. The threads of an OpenMP parallel
region form a team the same way (`fftn::omp::this_team()`, below); so do the members of a
Kokkos team.

## How the work is shared

The plan fixes how its work divides: whole transforms, and, for long transforms, phases of
chunks that may run on different threads (the passes of a long row, the lines of a plane).
Who runs which piece is decided during the execution, by the members that turn up:

- With `completion::all` (the default) members claim work: whole transforms first, then
  chunks of the transforms the plan splits, and only among members of one cache group.
  A member that finds nothing left waits for the rest by spinning briefly, then yielding,
  then blocking; it never waits for a member that has not arrived. The members present
  complete the execution between them, even one alone, and members that arrive late, even
  after the others have moved on to later executions, find nothing left. Arriving late is
  allowed; staying away is not (see below).
- With `completion::own` each member transforms exactly the transforms that
  `transforms_of` gives it, whole, and returns once they are written. Nothing is claimed or
  shared, so every member must take part, and the caller synchronises its threads before
  reading other members' transforms or reusing the workspace.

Small jobs run on the first member to arrive; batches smaller than a cache group stay in
one group; a transform is split only when it is long enough to pay for the
synchronisation. The output does not depend on any of this: a team execution gives the
same bits as serial execution, whatever the team's size, grouping or timing
([conventions](conventions.md#determinism)).

## Every member makes every call

A member knows which execution it is in by counting its own calls on the workspace, so the
members of a team must all make the same sequence of team calls on it, each call in turn,
however late. **A member must not skip a call.** One that did would count wrong and could
return from its next call before that call's output is written. If a member may miss an
execution (a runtime that does not run every member, an early exit), reset the workspace
before its next team call: any serial execution resets it, and
`plan.execute_transforms(0, 0, in, out, ws)` does nothing else. There is no other reset.

Two further rules follow from the count:

- All members pass the same `fftn::workspace` object. Two workspaces over the same borrowed
  memory are two workspaces, each with its own record of the team's state.
- One rank's successive calls are made by one thread, or ordered by happens-before: a
  mutex, a barrier, a thread join. The threads of an OpenMP region and a `std::thread` per
  rank satisfy this as they are.

## Workspaces for teams

`plan.workspace_bytes(lanes)` is the workspace a team of up to `lanes` members needs:
scratch for each member and the team's synchronisation state. The state lives in the
workspace from one team execution to the next, which is how a member that arrives late
knows that its execution is already over. The workspace sets the state up itself, on the
first team execution, after a serial execution and after a change of team size, so its
memory never needs initialising. The condition is that no member of an earlier execution
is still inside FFTN when the workspace changes hands (for example, when the parallel
region has ended). An execution with `completion::own` by a smaller team than the one
whose state the workspace holds overwrites that state, so the next team execution sets it
up afresh.

## OpenMP

`fftn/omp.hpp` is optional and header-only: `fftn.hpp` does not include it, and it compiles
only with the compiler's OpenMP option. It is compiled in the caller's translation units,
against the caller's OpenMP runtime, so the process has one OpenMP runtime: the library
links none. Two runtimes in one process (libgomp and libomp, say) crash or crawl.

Inside a parallel region, `fftn::omp::this_team()` is the calling thread as a member of the
region's team (rank `omp_get_thread_num()`, size `omp_get_num_threads()`, cache groups from
`fftn::cache_group_size`):

```cpp file=examples/omp_rows.cpp region=region
// Transforms the rows of x into y and divides each by n, in one parallel region.
void transform_and_scale(const fftn::c2c_plan<double>& plan, std::size_t n,
                         const std::complex<double>* x, std::complex<double>* y) {
  // One workspace for the region's team, sized before the region.
  fftn::workspace ws(plan, omp_get_max_threads());

#pragma omp parallel
  {
    const fftn::team me = fftn::omp::this_team();

    // Every thread of the region calls execute; with completion::own each returns once
    // its own rows are written and goes on to work on them, with no barrier in between.
    plan.execute(x, y, me, ws, fftn::completion::own);
    const fftn::transform_range mine = plan.transforms_of(me);
    for (std::size_t i = mine.first * n; i < (mine.first + mine.count) * n; ++i) {
      y[i] /= static_cast<double>(n);
    }
  }
}
```

Every thread of the region must make the call, so call it from the region's body, not from
inside `single`, `masked`, `critical`, a task or a worksharing loop, where only some threads
run; there, use the serial overloads.

Outside a region, `fftn::omp::executor` lets FFTN open one:

```cpp file=examples/omp_rows.cpp region=executor
// FFTN opens the region: one region of the executor's constant width per execution, or
// none when the job is too small to share.
const fftn::omp::executor omp_lanes;
fftn::workspace ws(plan, omp_lanes.concurrency());
std::vector<std::complex<double>> z(n * count);
plan.execute(x.data(), z.data(), omp_lanes, ws);
```

Its width is `omp_get_max_threads()` at construction, or the number it is given. Every
region it opens has that width, through the region's `num_threads` clause: when only some
lanes have work, the others return at once rather than the region being narrower, because
libgomp ends and restarts its threads whenever the width of successive regions changes.
Called from inside an active parallel region, it runs the execution serially on the calling
thread instead of nesting a region. Neither adapter changes the runtime's settings.

## Executors

An executor is anything with `concurrency()`, the most tasks it can usefully run at once,
and `bulk(n, task, arg)`, which runs `task(arg, i)` for every `i` from 0 to n - 1 and returns
when all have returned (`fftn::executor` in `include/fftn/executor.hpp`). An optional
`group_size()` says how many consecutive tasks share a cache. The tasks may run in any order,
on any threads, any number at a time, even one after another on the calling thread: FFTN's
tasks never wait for one that has not started, so a task system without a concurrency
guarantee (TBB, Kokkos `RangePolicy`, a busy pool) is as correct as a dedicated one.

`plan.execute(in, out, ex, ws)` asks the scheduler how many of the executor's lanes the plan
can use. A job too small to share, and every job when `concurrency()` is 1, runs on the
calling thread and the executor is not called. Otherwise FFTN calls `bulk` once, with one
task per lane that has work, which run as a team ([How the work is shared](#how-the-work-is-shared)).
The workspace is sized for the executor's concurrency, and the result is the same bits as
serial execution, whatever the executor does with the order of its tasks.

### FFTN's thread pool

For a program with no threading runtime of its own, `fftn::thread_pool` is a fixed set of
`std::thread`s. FFTN never creates one by itself:

```cpp file=examples/executor_rows.cpp region=pool
// The program starts the pool; FFTN never starts threads by itself.
const fftn::thread_pool pool(4);

// A workspace for as many lanes as the executor has.
fftn::workspace ws(plan, pool.concurrency());

std::vector<std::complex<double>> y(n * count);
plan.execute(x.data(), y.data(), pool, ws); // returns when y is written
```

The thread that calls `bulk` is one of the pool's lanes, so a pool of four lanes starts three
threads. Idle threads spin for some tens of microseconds after a job, then yield, then block
until the next one. They are not bound to CPUs; lanes are grouped as
`fftn::cache_group_size` says. The program may run its own work on the pool with `bulk`.

### The program's own pool

An adapter of a few lines makes any pool an executor:

```cpp file=examples/executor_rows.cpp region=adapter
// The program's own pool: anything that can run n jobs and wait for them.
struct my_pool {
  int threads = 4;

  [[nodiscard]] int size() const { return threads; }
  void parallel_for(int n, const std::function<void(int)>& job) const {
    std::vector<std::thread> team;
    team.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
      team.emplace_back([&, t] {
        for (int i = t; i < n; i += threads) {
          job(i);
        }
      });
    }
    for (std::thread& member : team) {
      member.join();
    }
  }
};

// The adapter FFTN needs: concurrency() and bulk(n, task, arg).
struct my_pool_executor {
  const my_pool& pool;
  [[nodiscard]] int concurrency() const { return pool.size(); }
  void bulk(int n, fftn::task_function task, void* arg) const {
    pool.parallel_for(n, [=](int i) { task(arg, i); });
  }
};
```

```cpp file=examples/executor_rows.cpp region=custom
const my_pool mine;
std::vector<std::complex<double>> z(n * count);
plan.execute(x.data(), z.data(), my_pool_executor{mine}, ws);
```

FFTN's own calls never allocate or lock, but an adapter like this one may: what the executor
does is the executor's business. If its `bulk` throws, the exception reaches the caller of
`execute`, the output is unspecified, and the workspace is ready for its next execution.

A pool written in C, or in any language that can call C, describes itself with the C
contract of `include/fftn/executor.h`: a struct of the pool's pointer and three functions,
`concurrency`, `bulk` and an optional `group_size`, passed to `execute` as
`fftn::executor_ref(contract)`.

### Nesting

An executor called from inside one of its own tasks (or, for OpenMP, inside an active
parallel region) reports a concurrency of 1, so the execution runs serially on the calling
thread instead of waiting for threads that are busy. `fftn::thread_pool` does this for calls
from inside any `fftn::thread_pool` task, and `fftn::omp::executor` inside any active region.
