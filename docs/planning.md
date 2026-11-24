# Planning effort and wisdom

A plan chooses the program that computes its transform when it is constructed. By default
(`fftn::planning::rules`) the planner's rules and cost tables decide, in microseconds to
milliseconds, and the same shape, options, library version and processor model always give
the same program. The rules depend on the processor as well as on the instruction set: on its
caches, and on who made it, since some rules were measured on one vendor's processors only and
apply only there (an `avx2` plan on an AMD processor can differ from one on an Intel
processor). `plan::describe()` names the vendor the plan was made for. The declarations, preconditions and guarantees are in the public headers
(`options::effort`, `options::measure_seconds`, `options::measure_scratch_bytes` and
`options::use` in `include/fftn/options.hpp`, `fftn::wisdom` in `include/fftn/wisdom.hpp`);
this page says how the pieces fit together.

## Measured planning

Rules decide some choices by thresholds on the cache sizes (whether a long row runs as one
chain of passes or as a blocked grid, in two phases or three; which radices a long power of
two takes; whether a batch of short rows runs one row per vector lane) and others by a table
of measured pass costs. Where a machine differs from the ones the thresholds were measured
on, the rules can miss the fastest program. `fftn::planning::measure` builds the rules'
choice and the alternatives the rules decided between, times each whole program at the
transform's size on scratch memory it allocates itself, and keeps the fastest; the rules'
choice stays unless another program is at least 1 % faster. `fftn::planning::quick` does
the same with fewer candidates and shorter timings.

Measuring costs time at construction: from milliseconds for small transforms to seconds for
large ones (`options::measure_seconds` bounds it), and scratch memory of the data's size
(`options::measure_scratch_bytes` bounds it by timing fewer transforms of a large batch).
It pays when a plan is executed many times. Both bounds are targets: a transform whose plan
takes long to build (around a second at 2²² points) overruns the time by about that much,
and the rules' plan and one transform's data are always allocated. The chosen program can
differ from run to run when two candidates are close in speed; `plan::describe()` says how
the program was chosen, how many distinct candidates were timed (and on how many transforms,
when fewer than the batch), the chosen program, and how much slower the rules' choice was.

## Wisdom

A measured choice is kept with `fftn::wisdom`: `add` records a plan's program, `text` and
`save` write the records out, and a plan built with `options::use` rebuilds a recorded
program without timing anything, with bitwise identical results:

```cpp file=examples/planning.cpp region=measure
const auto shape = fftn::rows_c2c<double>(4096, 16, fftn::direction::forward);

// Time the candidate programs once (a fraction of a second here) and keep the fastest.
const fftn::c2c_plan<double> measured(shape, {.effort = fftn::planning::measure});

// Record the choice; text() is what save() writes to a file.
fftn::wisdom w;
w.add(measured);
const std::string text = w.text();

// Later, perhaps in another run on the same machine: the same program, without timing.
const fftn::wisdom loaded(text);
const fftn::c2c_plan<double> again(shape, {.use = &loaded});
```

An entry applies only to the shape and options it was recorded for (lengths, counts,
strides and distances as given, in place or not, the scale), to the same build of the library
(its version and an identifier of its sources, so a development build never takes another
build's entries), processor model and instruction set, and only if its program rebuilds
exactly: an entry that would
give another program (on a machine whose caches differ, say) is ignored, and the plan's
effort decides. Wisdom is plain data owned by the program; FFTN keeps none of its own.
