# The C API

`fftn/c_api.h` gives C programs, and any language that can call C, the plans of the C++ API:
every kind and precision, workspaces, the four forms of execution, wisdom and the length
queries. It is provisional (step 6.8): it follows the proposals of DESIGN 5.9 and of 18.2
("Phase 6", Q17–Q22), which the maintainer has not yet settled, so names and types may change
before the first release. The declarations, preconditions and guarantees are in
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

## Not yet in C

Edges, `exec_args`, block callbacks and pipelines come to C in step 6.9, once their C++ API is
confirmed; the split-complex layout with 5.8. The structs carry no size or version fields
until 1.0 (Q21): `FFTN_VERSION_MAJOR`, `FFTN_VERSION_MINOR` and `FFTN_VERSION_PATCH`
(`fftn/version.h`, included by `fftn/c_api.h`) are the check, and `fftn_version()` gives the
linked library's version.
