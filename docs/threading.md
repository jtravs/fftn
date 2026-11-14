# Threading

FFTN runs on the caller's threads. It never creates a thread, never changes a threading
runtime's settings, and executes serially unless the caller hands it a team. The
declarations, preconditions and guarantees are in the public headers (`include/fftn/team.hpp`
and the team overload of `plan::execute` in `include/fftn/plan.hpp`); this page says how the
pieces fit together.

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

The threads of an OpenMP parallel region form a team the same way, with
`omp_get_thread_num()` and `omp_get_num_threads()`; so do the members of a Kokkos team.

## How the work is shared

The plan fixes how its work divides: whole transforms, and, for long transforms, phases of
chunks that may run on different threads (the passes of a long row, the lines of a plane).
Who runs which piece is decided during the execution, by the members that turn up:

- With `completion::all` (the default) members claim work: whole transforms first, then
  chunks of the transforms the plan splits, and only among members of one cache group.
  A member that finds nothing left waits for the rest by spinning briefly, then yielding,
  then blocking; it never waits for a member that has not arrived. One member alone
  completes the execution, and members that arrive late, even after the others have moved
  on to later executions, find nothing left.
- With `completion::own` each member transforms exactly the transforms that
  `transforms_of` gives it, whole, and returns once they are written. Nothing is claimed or
  shared, so every member must take part, and the caller synchronises its threads before
  reading other members' transforms or reusing the workspace.

Small jobs run on the first member to arrive; batches smaller than a cache group stay in
one group; a transform is split only when it is long enough to pay for the
synchronisation. The output does not depend on any of this: a team execution gives the
same bits as serial execution, whatever the team's size, grouping or timing
([conventions](conventions.md#determinism)).

## Workspaces for teams

`plan.workspace_bytes(lanes)` is the workspace a team of up to `lanes` members needs:
scratch for each member and the team's synchronisation state. The state lives in the
workspace from one team execution to the next, which is how a member that arrives late
knows that its execution is already over. The workspace sets the state up itself, on the
first team execution, after a serial execution and after a change of team size, so its
memory never needs initialising; the condition is only that no member of an earlier
execution is still inside FFTN when the workspace changes hands (for example, when the
parallel region has ended).
