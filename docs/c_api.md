# The C API

`fftn/c_api.h` gives C programs, and any language that can call C, the plans of the C++ API:
every kind and precision, workspaces, the four forms of execution, wisdom and the length
queries, the fusion objects (edges, block callbacks and row pipelines: "Fusion" below) and
split-complex data ("Split complex" below). It is provisional (steps 6.8 and 6.9, and the split
form after 6.10): it follows the proposals of DESIGN 5.9 and of 18.2 ("Phase 6", Q17–Q22 and
Q28–Q31), which the maintainer has not yet settled, so names and types may change before the
first release. The declarations, preconditions and guarantees are in
[include/fftn/c_api.h](../include/fftn/c_api.h); this page says how the pieces fit.

A C plan wraps the same plan core as `fftn::plan`, so it computes exactly what the C++ plan of
the same shape and options computes, bit for bit (the tests compare the two for every kind and
precision). The C++ documents apply: [conventions.md](conventions.md) for the meaning of every
shape field and the transforms' definitions, [threading.md](threading.md) for teams and
executors, [planning.md](planning.md) for planning effort and wisdom,
[lengths.md](lengths.md) for the length queries.

## Example

A batch of three complex rows, forward out of place and back in place, scaled by 1/n, from
[examples/c_api.c](../examples/c_api.c), a C99 program that the build runs as a test:

```c file=examples/c_api.c region=plans
// Three rows of 12 complex values, one after another: interleaved real and imaginary parts.
const fftn_axis axis = {N, 0, 0}; // n, then input and output strides (0: tight)
const fftn_batch batch = {COUNT, 0, 0};
fftn_shape shape = {0};
shape.kind = FFTN_C2C;
shape.precision = FFTN_DOUBLE;
shape.direction = FFTN_FORWARD;
shape.rank = 1;
shape.axes = &axis;
shape.batch_rank = 1;
shape.batches = &batch;

fftn_error error;
fftn_plan* forward = fftn_plan_create(&shape, NULL, &error); // NULL: the default options
if (forward == NULL) {
  fprintf(stderr, "fftn: %s\n", error.message);
  return 1;
}

// Backward, in place, scaled by 1/n so that it inverts the forward transform.
fftn_options options = fftn_options_default();
options.in_place = 1;
options.scale = 1.0 / N;
shape.direction = FFTN_BACKWARD;
fftn_plan* backward = fftn_plan_create(&shape, &options, &error);
if (backward == NULL) {
  fprintf(stderr, "fftn: %s\n", error.message);
  fftn_plan_release(forward);
  return 1;
}

// One workspace serves both plans, one execution at a time.
size_t bytes = fftn_plan_workspace_bytes(forward, 1);
if (fftn_plan_workspace_bytes(backward, 1) > bytes) {
  bytes = fftn_plan_workspace_bytes(backward, 1);
}
fftn_workspace* ws = fftn_workspace_create(bytes, &error);
```

```c file=examples/c_api.c region=execute
fftn_execute(forward, x, y, ws);  // y = the spectra of the rows of x
fftn_execute(backward, y, y, ws); // in place: y = x again, up to rounding

char text[512];
fftn_plan_describe(forward, text, sizeof text); // how the plan computes, for people
printf("%s\n", text);

fftn_workspace_destroy(ws);
fftn_plan_release(backward);
fftn_plan_release(forward);
```

## Building and linking

The C functions are compiled into the static library `fftn::fftn` (`libfftn`), which is C++. A
C program includes `<fftn/c_api.h>` and links with the C++ compiler driver, or adds the C++
standard library itself (`-lstdc++` with GCC, `-lc++` with Clang on macOS):

- With CMake, link `fftn::fftn` as from C++; CMake links a C target that uses it with the C++
  compiler. The project must enable both languages (`project(app LANGUAGES C CXX)`).
- With pkg-config, compile with `cc -std=c99 $(pkg-config --cflags fftn)` and link with
  `c++ ... $(pkg-config --libs fftn)`.

[examples/package](../examples/package/CMakeLists.txt) builds the example both ways in CI.

**A shared library.** Python's `ctypes`, Julia's `ccall` and some Fortran builds can load only
a shared library. With `-DFFTN_BUILD_C_SHARED=ON` the build also makes `libfftn_c`
(`fftn_c.dll` on Windows), installed and exported as `fftn::fftn_c`, which holds the whole
library inside and exports the `fftn_` functions and nothing else. The C++ API stays static.

Every C symbol the library defines starts with `fftn_`; a test checks both libraries.

## Types

Each C type stands for the C++ type of the same name, field by field:

| C | C++ |
|---|---|
| `fftn_shape` with `fftn_kind`, `fftn_precision` | `fftn::shape<In, Out>` (the kind and precision are its element types) |
| `fftn_axis`, `fftn_batch`, `fftn_stored_run`, `fftn_side` | `fftn::axis`, `fftn::batch`, `fftn::stored_run`, `fftn::side` |
| `fftn_r2r_kind` (`FFTN_DCT1` … `FFTN_DHT`) | `fftn::r2r_kind` |
| `FFTN_FORWARD`, `FFTN_BACKWARD` | `fftn::direction` |
| `fftn_options`, `fftn_isa`, `fftn_planning` | `fftn::options`, `fftn::isa`, `fftn::planning` |
| `fftn_plan` (opaque, reference counted) | `fftn::plan<In, Out>` (copies share the tables) |
| `fftn_workspace` (opaque) | `fftn::workspace` |
| `fftn_team`, `fftn_completion`, `fftn_transform_range` | `fftn::team`, `fftn::completion`, `fftn::transform_range` |
| `fftn_executor` (`fftn/executor.h`) | `fftn::executor_ref` |
| `fftn_wisdom` (opaque) | `fftn::wisdom` |
| `fftn_length_kind` | `fftn::length_kind` |
| `fftn_layout` (`fftn_shape::layout`), `fftn_split_ptr`, `fftn_split_const_ptr` ("Split complex" below) | `fftn::split_complex<T>`, `fftn::split_ptr<T>`, `fftn::split_ptr<const T>` |

- **One plan type** serves every kind and precision, and data are `void*`. Complex values are
  interleaved pairs, so arrays of C99 `double _Complex`, C++ `std::complex<double>` and
  Fortran `complex(c_double_complex)` pass unchanged. The kind and precision of the arrays are
  the caller's to get right: nothing checks them.
- **The shape borrows its arrays** (axes, batches, kinds, runs) for the call of
  `fftn_plan_create` only. Zero-initialise the struct and fill what the shape needs:
  `r2r_kinds` holds one kind per axis for `FFTN_R2R` and is NULL otherwise, and `direction` is
  read for `FFTN_C2C` only (r2c is always forward, c2r always backward).
- **Options** start from `fftn_options_default()`, which gives `fftn::options{}`; NULL options
  mean the same.
- **Plans** are reference counted: `fftn_plan_ref` adds a reference and returns the plan,
  `fftn_plan_release` drops one and the last frees it. Both are safe from several threads.
- **Workspaces** are opaque because a C++ workspace keeps a word of team state beside its
  memory. `fftn_workspace_create` allocates; `fftn_workspace_borrow` wraps the caller's memory
  (any alignment) and allocates only the handle. Either way, nothing is allocated during an
  execution.
- **Strings** (`fftn_plan_describe`, `fftn_wisdom_text`) are copied as `snprintf` does: call
  with a size of 0 to learn the length, then with a buffer one byte longer.

## Errors

No exception crosses into C, and there is no thread-local "last error". A function that
constructs something returns NULL (or a status) and, given a non-NULL `fftn_error*`, fills it
with a code and the message of the C++ exception it caught:

```c file=examples/c_api.c region=error
const fftn_axis empty = {0, 0, 0}; // a length of 0, which plans refuse
fftn_shape shape = {0};
shape.kind = FFTN_R2C;
shape.precision = FFTN_FLOAT;
shape.rank = 1;
shape.axes = &empty;

fftn_error error;
if (fftn_plan_create(&shape, NULL, &error) == NULL) {
  // error.code is FFTN_ERROR_INVALID_ARGUMENT; the message names shape.axes[0].n.
  printf("refused (%d): %s\n", error.code, error.message);
}
```

The codes are `FFTN_ERROR_INVALID_ARGUMENT` (the C++ `std::invalid_argument`, or a C field
holding a value that is not one of its enumerators, or a NULL array with a count),
`FFTN_ERROR_OUT_OF_MEMORY`, `FFTN_ERROR_FILE` (a wisdom file that cannot be read or written),
`FFTN_ERROR_EXECUTOR` (a C++ executor behind the C struct threw) and `FFTN_ERROR_INTERNAL`.
On success the error is set to `FFTN_OK` with an empty message. Messages longer than 255
characters are cut.

The executions (`fftn_execute`, `fftn_execute_transforms`, `fftn_execute_team`) return nothing
and check nothing, as in C++: they never allocate, lock or fail. `fftn_execute_on` returns a
status, since the pool behind the executor may fail.

## Threads

`fftn_execute_team` is the team overload of `plan::execute`: every member of a team of the
caller's threads calls it with its own `fftn_team` and one shared workspace, sized with
`fftn_plan_workspace_bytes(plan, size)`; `fftn_transforms_of` gives the transforms a member
owns. `fftn_execute_on` runs the plan on any pool described by an `fftn_executor`
(`fftn/executor.h`): its `concurrency` and `bulk` functions, and optionally `group_size`. The
rules are those of [threading.md](threading.md), and the results are bitwise those of a serial
execution. The tests drive a pool written in C through it.

## Wisdom and lengths

`fftn_wisdom_create`, `fftn_wisdom_from_text`, `fftn_wisdom_load`, `fftn_wisdom_add`,
`fftn_wisdom_text`, `fftn_wisdom_save`, `fftn_wisdom_size` and `fftn_wisdom_destroy` are the
members of `fftn::wisdom`; a plan reads a wisdom through `fftn_options::use` during its
construction only. `fftn_next_fast_length` and `fftn_relative_cost` are the length queries; they
return 0 with the error filled where the C++ functions throw.

## Fusion

Edges, block callbacks and row pipelines ([fusion.md](fusion.md)) have a C form too
(provisional, step 6.9): the same objects, computing the same bits as the C++ ones (the tests
compare them for every edge operation, callback mode and pipeline kind, in every form of
execution).

| C | C++ |
|---|---|
| `fftn_factor`, `fftn_factor_kind` (strides as a pointer and a count) | `fftn::factor`, `fftn::factor_kind` |
| `fftn_edge`, `fftn_plan_edges` | `fftn::edge<T>`, `fftn::plan_edges<In, Out>` |
| `fftn_exec_args` | `fftn::exec_args` |
| `fftn_callback`, `fftn_callback_mode`, `fftn_block_fn`, `fftn_block_info` | `fftn::callback<T>`, `fftn::callback_mode`, `fftn::block_fn<T>`, `fftn::block_info` |
| `fftn_pipeline_desc` with `fftn_pipeline_kind` (`FFTN_PIPELINE_CCC`, `_TCT`, `_CTC`), `fftn_middle` | `fftn::pipeline_desc<In, Mid, Out>` and its `middle` |
| `fftn_pipeline` (opaque, reference counted) | `fftn::pipeline<In, Mid, Out>` |

- **Start from the defaults.** `fftn_edge_default()` and `fftn_pipeline_desc_default()` give the
  C++ defaults: a zeroed struct would have scales of 0, which are valid scales, a fan-out of 0
  and a group of batch dimension 0. Stride arrays, batches and runs are borrowed for the call
  that builds the object only.
- **Plans with edges** come from `fftn_plan_create_edges` and run through the `_args` forms of
  the executions (`fftn_execute_args`, `fftn_execute_transforms_args`, `fftn_execute_team_args`,
  `fftn_execute_on_args`), which take the execution's `fftn_exec_args` (NULL for none). A plan
  with factors or callbacks executed without them ends the program, as in C++.
- **Pipelines** (`fftn_pipeline_create`) have the queries and the four executions of a plan
  under `fftn_pipeline_`; the executions always take an `fftn_exec_args`.

```c file=examples/c_api_fusion.c region=edges
// Real rows times a window (the input edge's real factor, one line for every row), r2c, and
// added to the output (the output edge's accumulation).
const fftn_axis axis = {N, 0, 0};
const fftn_batch batch = {COUNT, 0, 0};
const ptrdiff_t along_row = 1;
const ptrdiff_t across_rows = 0; // the same window for every row
fftn_shape shape = {0};
fftn_plan_edges edges;
fftn_exec_args args = {0};
fftn_error error;
fftn_plan* plan;
fftn_workspace* ws;
shape.kind = FFTN_R2C;
shape.precision = FFTN_DOUBLE;
shape.rank = 1;
shape.axes = &axis;
shape.batch_rank = 1;
shape.batches = &batch;
edges.in = fftn_edge_default(); // not {0}: a scale of 1
edges.in.mul.kind = FFTN_FACTOR_REAL;
edges.in.mul.axis_stride_count = 1;
edges.in.mul.axis_strides = &along_row;
edges.in.mul.batch_stride_count = 1;
edges.in.mul.batch_strides = &across_rows;
edges.out = fftn_edge_default();
edges.out.accumulate = 1;

plan = fftn_plan_create_edges(&shape, NULL, &edges, &error);
if (plan == NULL) {
  fprintf(stderr, "fftn: %s\n", error.message);
  return 1;
}
ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, 1), &error);
if (ws == NULL) {
  fprintf(stderr, "fftn: %s\n", error.message);
  fftn_plan_release(plan);
  return 1;
}
args.in_factor = window; // the window's address comes with each execution
fftn_execute_args(plan, rows, sum, ws, &args);
fftn_execute_args(plan, rows, sum, ws, &args); // sum holds twice the windowed spectra
fftn_workspace_destroy(ws);
fftn_plan_release(plan);
```

### Callbacks in C

A callback is an ordinary C function. It receives the user pointer of the execution's
`fftn_exec_args` as given and the block's place in an `fftn_block_info`, which holds what C++'s
`block_info` holds, with the origin and the scratch as a pointer and a count:

```c file=examples/c_api_fusion.c region=callback
#define LANES 1 // the members an execution has: this program executes serially

// What the epilogue's callback is given: the largest magnitude each member has seen, one slot
// per member, since a team's members (up to the lanes the workspace was sized for) run blocks
// at the same time; and a flag for a failure.
typedef struct peaks {
  double largest[LANES];
  int failed;
} peaks;

// A block callback in C: it reads the block where the plan holds it, between the pipeline's
// last pass and the store, and writes only its member's slot of the application's memory.
static void record_peak(void* user, const fftn_block_info* info, void* data) {
  peaks* p = (peaks*)user;
  const double* x = (const double*)data; // the epilogue of real rows: doubles
  size_t k;
  size_t i;
  if (info->member < 0 || info->member >= LANES) {
    p->failed = 1; // a callback cannot stop the execution: it records the failure and returns
    return;
  }
  for (k = 0; k < info->lines; ++k) {
    for (i = 0; i < info->length; ++i) {
      const double v =
          fabs(x[(ptrdiff_t)i * info->element_stride + (ptrdiff_t)k * info->line_stride]);
      if (v > p->largest[info->member]) {
        p->largest[info->member] = v;
      }
    }
  }
}
```

```c file=examples/c_api_fusion.c region=pipeline
// Real rows convolved with one kernel: r2c, times the kernel's spectrum (the middle's
// factor, one line for every row), c2r, scaled by 1/n; then the C callback on each block.
const fftn_batch batch = {COUNT, 0, 0};
const ptrdiff_t along_bins = 1;
peaks seen = {{0}, 0};
fftn_pipeline_desc desc = fftn_pipeline_desc_default(); // not {0}: scales of 1, fan-out 1
fftn_exec_args args = {0};
fftn_pipeline* conv;
fftn_workspace* ws;
desc.kind = FFTN_PIPELINE_TCT;
desc.line.n = N;
desc.batch_rank = 1;
desc.batches = &batch;
desc.middle.scale = 1.0 / N;
desc.middle.mul.kind = FFTN_FACTOR_COMPLEX;
desc.middle.mul.axis_stride_count = 1;
desc.middle.mul.axis_strides = &along_bins; // no batch strides: shared by every row
desc.epilogue.call.fn = record_peak;        // FFTN_CALLBACK_MODIFY, the default mode

conv = fftn_pipeline_create(&desc, NULL, &error);
if (conv == NULL) {
  fprintf(stderr, "fftn: %s\n", error.message);
  return 1;
}
ws = fftn_workspace_create(fftn_pipeline_workspace_bytes(conv, LANES), &error);
if (ws == NULL) {
  fprintf(stderr, "fftn: %s\n", error.message);
  fftn_pipeline_release(conv);
  return 1;
}

// The data of this execution: the kernel's spectrum and the callback's argument.
args.mid_factor = spectrum;
args.out_user = &seen;
fftn_pipeline_execute(conv, rows, out, ws, &args);
if (seen.failed) {
  fprintf(stderr, "the callback failed\n"); // the output is then discarded
}
printf("largest output value: %g\n", seen.largest[0]); // serial: member 0 only

fftn_workspace_destroy(ws);
fftn_pipeline_release(conv);
```

The contract is [fusion.md](fusion.md)'s and `fftn/callback.hpp`'s: each element once per
execution, in blocks of the plan's choosing, on several threads at once and in any order; a
pure callback gives bitwise-identical results in every form of execution. Two points are C's
own:

- **A callback cannot fail the execution.** It returns nothing: one that fails records it in
  the application's memory (through its user pointer: an atomic flag, or a slot per
  `info->member`) and returns, and the caller checks the record after the execution and
  discards the output.
- **Nothing may leave a callback but its return.** `longjmp` out of a callback is undefined (a
  team's members would wait for ever), and a C++ exception thrown by a callback written in C++
  ends the program (`std::terminate`) at the boundary: no exception crosses FFTN.

## Split complex

Complex values kept as two arrays of reals ([conventions.md](conventions.md), "Split complex")
have a C form too (provisional: DESIGN 5.9, "Split complex", and 18.2 Q28–Q31). A shape whose
`layout` is `FFTN_SPLIT` is the C++ shape with `fftn::split_complex<T>` on its complex sides
(`FFTN_INTERLEAVED`, 0, is the default, so a zeroed shape is interleaved), for `FFTN_C2C`,
`FFTN_R2C` and `FFTN_C2R`; its plan is executed through the four forms with a `_split` suffix,
which take the arrays as an `fftn_split_const_ptr` (input) and an `fftn_split_ptr` (output), the
C forms of `split_ptr<const T>` and `split_ptr<T>`:

```c file=examples/c_api_split.c region=split
// Real rows into split spectra: the shape of interleaved r2c rows with the layout FFTN_SPLIT; on
// the split side strides and distances count reals, the same numbers for both arrays.
const fftn_axis axis = {N, 0, 0};
const fftn_batch batch = {COUNT, 0, 0};
fftn_shape shape = {0};
fftn_options options = fftn_options_default();
fftn_error error;
shape.kind = FFTN_R2C;
shape.precision = FFTN_DOUBLE;
shape.rank = 1;
shape.axes = &axis;
shape.batch_rank = 1;
shape.batches = &batch;
shape.layout = FFTN_SPLIT;
fftn_plan* forward = fftn_plan_create(&shape, NULL, &error);

// And back: c2r from the split spectra, scaled by 1/n.
shape.kind = FFTN_C2R;
options.scale = 1.0 / N;
fftn_plan* backward = forward != NULL ? fftn_plan_create(&shape, &options, &error) : NULL;
if (backward == NULL) {
  fprintf(stderr, "fftn: %s\n", error.message);
  fftn_plan_release(forward);
  return EXIT_FAILURE;
}
size_t bytes = fftn_plan_workspace_bytes(forward, 1);
if (fftn_plan_workspace_bytes(backward, 1) > bytes) {
  bytes = fftn_plan_workspace_bytes(backward, 1);
}
fftn_workspace* ws = fftn_workspace_create(bytes, &error);
if (ws != NULL) {
  // A real side passes its array as re; its im is not read.
  const fftn_split_const_ptr real_in = {rows, NULL};
  const fftn_split_ptr spectra = {re, im};
  const fftn_split_const_ptr spectra_in = {re, im};
  const fftn_split_ptr real_out = {back, NULL};
  fftn_execute_split(forward, real_in, spectra, ws);
  fftn_execute_split(backward, spectra_in, real_out, ws); // back = rows, up to rounding

  char text[512];
  fftn_plan_describe(forward, text, sizeof text); // names the conversion pass
  printf("%s\n", text);
  fftn_workspace_destroy(ws);
}
fftn_plan_release(backward);
fftn_plan_release(forward);
```

- **The arrays.** `re` and `im` are `void*` in the plan's precision, laid out by the shape's
  strides and distances, which count reals on a split side and hold for both arrays. They may lie
  anywhere relative to each other. A real side (r2c's input, c2r's output) passes its array as
  `re`; its `im` is not read. In place (c2c only) each input array is the output array of its
  part.
- **The executions.** `fftn_execute_split`, `fftn_execute_transforms_split`,
  `fftn_execute_team_split` and `fftn_execute_on_split` are the four forms, with the threading
  rules above and results bitwise equal to the C++ split plan's (the tests compare them for every
  kind, rows to volumes, in both precisions). A split plan given to a form without the suffix,
  or an interleaved plan given to a `_split` form, ends the program with a message, as does a
  NULL `im` on a split side.
- **What split plans do not take,** as in C++: edges (`fftn_plan_create_edges` refuses split
  shapes with edges), stored runs, r2c or c2r in place, and wisdom (`fftn_wisdom_add` refuses a
  split plan; split plans are planned by the rules, which `fftn_plan_describe` says when other
  planning was asked for). `FFTN_SPLIT` with `FFTN_R2R` is refused, as is a `layout` that is
  not an `fftn_layout`. Pipelines have no split form.
- **`fftn_shape` grew a field** (`layout`, last, an `fftn_layout`, which is an `int` in C) on
  2026-11-27, as Q21 allows before 1.0. A zeroed shape is unchanged in meaning, but C programs
  must be rebuilt against the new header, and mirrors of the struct in other languages (Python's
  `ctypes`, Julia's `ccall`, Fortran's `iso_c_binding`) must add the field. C++ code that names
  every field in a designated initialiser and leaves out `layout` gets a
  `-Wmissing-field-initializers` warning.

## Not yet in C

The structs carry no size or version fields until 1.0 (Q21): `FFTN_VERSION_MAJOR`,
`FFTN_VERSION_MINOR` and `FFTN_VERSION_PATCH` (`fftn/version.h`, included by `fftn/c_api.h`) are
the check, and `fftn_version()` gives the linked library's version. Edges on split-complex data
are not built in C++ yet, so not in C either.
