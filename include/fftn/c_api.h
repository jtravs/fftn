/* SPDX-FileCopyrightText: 2026 John C. Travers and contributors
 * SPDX-License-Identifier: Apache-2.0 */

/**
 * @file
 * The C API: plans of every kind and precision, workspaces, the four forms of execution,
 * wisdom and the length queries, for C (C99 and later) and for any language that calls C.
 *
 * Provisional (step 6.8, dev/DESIGN.md 5.9): built on the proposals of dev/DESIGN.md 18.2,
 * "Phase 6", Q17-Q22, which the maintainer has not yet settled; names and types may change
 * before the first release. docs/c_api.md describes its use and quotes a compiled example.
 *
 * The header is valid C99 and C++. It wraps the same plan core as the C++ API (fftn/plan.hpp),
 * so a plan built here computes exactly what the C++ plan of the same shape and options
 * computes, bit for bit. Each C function states the C++ call it stands for; that call's
 * preconditions and guarantees hold, and are not repeated here.
 *
 * - **Data** are passed as `void*`: complex values are interleaved pairs (real part first),
 *   so arrays of C99 `double _Complex`, C++ `std::complex<double>` and Fortran
 *   `complex(c_double_complex)` pass unchanged. The caller passes arrays of the plan's kind
 *   and precision; nothing checks it.
 * - **Errors.** No exception crosses into C. A function that constructs something returns
 *   NULL, or a status, and fills the caller's fftn_error, if one is given, with a code and the
 *   message of the C++ exception it caught. There is no thread-local "last error".
 * - **Executions** (fftn_execute, fftn_execute_transforms, fftn_execute_team) check nothing,
 *   return nothing and never allocate, lock or fail, as in C++; fftn_execute_on returns a
 *   status because a C++ executor behind the struct may throw.
 * - **Strings** (fftn_plan_describe, fftn_wisdom_text) are copied into the caller's buffer as
 *   snprintf does: at most `size - 1` characters and a terminating NUL, and the return value
 *   is the full length, without the NUL.
 * - **Linking.** The functions are compiled into the C++ static library `fftn::fftn`; a C
 *   program links it with the C++ compiler driver, or adds the C++ standard library itself.
 */
#ifndef FFTN_C_API_H
#define FFTN_C_API_H

#include "fftn/executor.h" /* IWYU pragma: export */
#include "fftn/version.h"  /* IWYU pragma: export */

/* NOLINTNEXTLINE(modernize-deprecated-headers): the header is C as well as C++ */
#include <stddef.h>
/* NOLINTNEXTLINE(modernize-deprecated-headers): the header is C as well as C++ */
#include <stdint.h>

/* NOLINTBEGIN(modernize-use-using,modernize-redundant-void-arg,performance-enum-size): C */

#ifdef FFTN_C_EXPORTS
#ifdef _WIN32
#define FFTN_C_API __declspec(dllexport)
#else
#define FFTN_C_API __attribute__((visibility("default")))
#endif
#elif defined(FFTN_C_SHARED) && defined(_WIN32)
#define FFTN_C_API __declspec(dllimport)
#else
/**
 * Marks the functions of the C API. Empty, except in the optional shared library `fftn_c`
 * (dev/DESIGN.md 18.2, Q20), where it exports them, and in its Windows users, where it imports
 * them (the `fftn::fftn_c` target defines `FFTN_C_SHARED` for them).
 */
#define FFTN_C_API
#endif

#ifdef __cplusplus
#define FFTN_C_NOEXCEPT noexcept
#else
/** Marks the functions of the C API, which never throw: `noexcept` in C++, nothing in C. */
#define FFTN_C_NOEXCEPT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** The status of a function that can fail, and the code in fftn_error. */
typedef enum fftn_status {
  FFTN_OK = 0,                     /**< success */
  FFTN_ERROR_INVALID_ARGUMENT = 1, /**< an argument is not valid (std::invalid_argument) */
  FFTN_ERROR_OUT_OF_MEMORY = 2,    /**< memory ran out (std::bad_alloc) */
  FFTN_ERROR_INTERNAL = 3,         /**< any other failure inside FFTN */
  FFTN_ERROR_FILE = 4,             /**< a wisdom file could not be read or written */
  FFTN_ERROR_EXECUTOR = 5,         /**< the executor of fftn_execute_on failed */
} fftn_status;

/**
 * What went wrong, filled by a constructing function given a non-NULL pointer to one: on
 * success the code FFTN_OK and an empty message, on failure the code and the message of the
 * C++ exception, cut to fit with a terminating NUL. The caller owns it.
 */
typedef struct fftn_error {
  int code;          /**< an fftn_status */
  char message[256]; /**< the message, NUL-terminated; names the offending argument */
} fftn_error;

/** The kind of a transform: what fftn::shape's element types say in C++. */
typedef enum fftn_kind {
  FFTN_C2C, /**< complex to complex */
  FFTN_R2C, /**< real to complex (always forward) */
  FFTN_C2R, /**< complex to real (always backward) */
  FFTN_R2R, /**< real to real, one fftn_r2r_kind per axis */
} fftn_kind;

/** The precision of a transform's values. */
typedef enum fftn_precision {
  FFTN_DOUBLE, /**< double, and complex values of two doubles */
  FFTN_FLOAT,  /**< float, and complex values of two floats */
} fftn_precision;

/** The direction of a complex transform, the values of fftn::direction. */
enum {
  FFTN_FORWARD = -1,  /**< exponent -2 pi i jk / n */
  FFTN_BACKWARD = +1, /**< exponent +2 pi i jk / n */
};

/**
 * The kind of one axis of a real-to-real transform, fftn::r2r_kind: FFTW's definitions,
 * unnormalised (docs/conventions.md, "Real-to-real transforms"). Every kind is built.
 */
typedef enum fftn_r2r_kind {
  FFTN_DCT1, /**< DCT-I, FFTW's REDFT00 */
  FFTN_DCT2, /**< DCT-II, REDFT10 */
  FFTN_DCT3, /**< DCT-III, REDFT01 */
  FFTN_DCT4, /**< DCT-IV, REDFT11 */
  FFTN_DST1, /**< DST-I, RODFT00 */
  FFTN_DST2, /**< DST-II, RODFT10 */
  FFTN_DST3, /**< DST-III, RODFT01 */
  FFTN_DST4, /**< DST-IV, RODFT11 */
  FFTN_DHT,  /**< the discrete Hartley transform */
} fftn_r2r_kind;

/** A side of a transform, fftn::side. */
typedef enum fftn_side {
  FFTN_IN,  /**< the input */
  FFTN_OUT, /**< the output */
} fftn_side;

/** One transformed dimension, fftn::axis: strides in elements of each side's own type. */
typedef struct fftn_axis {
  size_t n;             /**< the logical length; at least 1 */
  ptrdiff_t in_stride;  /**< input stride; 0 for the tight default */
  ptrdiff_t out_stride; /**< output stride; 0 for the tight default */
} fftn_axis;

/** One batch dimension, fftn::batch: distances in elements of each side's own type. */
typedef struct fftn_batch {
  size_t count;       /**< the number of transforms; at least 1 */
  ptrdiff_t in_dist;  /**< input distance; 0 for the tight default */
  ptrdiff_t out_dist; /**< output distance; 0 for the tight default */
} fftn_batch;

/** A stored run, fftn::stored_run (provisional in C++, task 5.1d). */
typedef struct fftn_stored_run {
  size_t axis;     /**< the axis, an index into fftn_shape::axes */
  fftn_side where; /**< the side whose memory holds the run */
  size_t logical;  /**< the first logical position of the run */
  size_t stored;   /**< the first stored position of the run */
  size_t count;    /**< the number of elements; at least 1 */
} fftn_stored_run;

/**
 * A transform shape, fftn::shape in C: the arrays are borrowed for the call that takes the
 * shape and are not kept. Every field has the meaning of the C++ field of the same name
 * (docs/conventions.md); a zeroed struct with its kind, axes and batches filled is a valid
 * start.
 */
typedef struct fftn_shape {
  fftn_kind kind;                 /**< the kind of transform */
  fftn_precision precision;       /**< the precision of its values */
  int direction;                  /**< FFTN_FORWARD or FFTN_BACKWARD; read for FFTN_C2C only */
  size_t rank;                    /**< the number of axes: 1 to 3 */
  const fftn_axis* axes;          /**< rank axes, slowest first */
  size_t batch_rank;              /**< the number of batch dimensions; 0 for one transform */
  const fftn_batch* batches;      /**< batch_rank batch dimensions, slowest first */
  const fftn_r2r_kind* r2r_kinds; /**< FFTN_R2R: rank kinds, one per axis; otherwise NULL */
  size_t run_count;               /**< the number of stored runs; 0 for none */
  const fftn_stored_run* runs;    /**< run_count stored runs */
} fftn_shape;

/** The instruction sets, fftn::isa, for fftn_options::force_isa and fftn_isa_name. */
typedef enum fftn_isa {
  FFTN_ISA_SCALAR,    /**< portable code with no vector instructions */
  FFTN_ISA_X86_V2,    /**< x86-64, SSE2 to SSE4.2 */
  FFTN_ISA_AVX,       /**< x86-64, AVX without AVX2 or FMA */
  FFTN_ISA_AVX2,      /**< x86-64, AVX2 with FMA */
  FFTN_ISA_AVX512,    /**< x86-64, AVX-512 F, DQ and VL */
  FFTN_ISA_NEON,      /**< AArch64, Advanced SIMD */
  FFTN_ISA_NEON_FCMA, /**< AArch64, Advanced SIMD with FCMA */
  FFTN_ISA_BEST,      /**< not a target: let the plan choose (the default) */
} fftn_isa;

/** How plan construction chooses the program, fftn::planning. */
typedef enum fftn_planning {
  FFTN_PLANNING_RULES,   /**< the planner's rules (the default) */
  FFTN_PLANNING_QUICK,   /**< a short measurement */
  FFTN_PLANNING_MEASURE, /**< the full measurement */
} fftn_planning;

/** An opaque set of recorded programs, fftn::wisdom. */
typedef struct fftn_wisdom fftn_wisdom;

/**
 * Options for building a plan, fftn::options, with the same meaning field by field. Start from
 * fftn_options_default(), which gives fftn::options{}.
 */
typedef struct fftn_options {
  int in_place;                 /**< nonzero: the input and output are the same array */
  double scale;                 /**< a real factor applied to every output value; 1 */
  int force_isa;                /**< an fftn_isa; FFTN_ISA_BEST lets the plan choose */
  int effort;                   /**< an fftn_planning; FFTN_PLANNING_RULES */
  double measure_seconds;       /**< measured planning's time budget; 0 for the effort's */
  size_t measure_scratch_bytes; /**< measured planning's memory limit; 0 for 1 GiB */
  const fftn_wisdom* use;       /**< wisdom to plan from, read during construction; or NULL */
} fftn_options;

/** One member of a team of the caller's threads, fftn::team. */
typedef struct fftn_team {
  int rank;       /**< this member's number, from 0 to size - 1 */
  int size;       /**< the number of members */
  int group_size; /**< members per cache group; 0 puts all in one group */
} fftn_team;

/** When a member's call of fftn_execute_team returns, fftn::completion. */
typedef enum fftn_completion {
  FFTN_COMPLETION_ALL, /**< when the whole output is written, on every member */
  FFTN_COMPLETION_OWN, /**< when this member's own transforms are written */
} fftn_completion;

/** A range of a plan's transforms, fftn::transform_range. */
typedef struct fftn_transform_range {
  size_t first; /**< the first transform of the range */
  size_t count; /**< the number of transforms; 0 for none */
} fftn_transform_range;

/** What a length is for, in fftn_next_fast_length: fftn::length_kind. */
typedef enum fftn_length_kind {
  FFTN_LENGTH_COMPLEX, /**< a complex length, or a real transform's axis that is not halved */
  FFTN_LENGTH_REAL,    /**< the halved axis of a real transform */
} fftn_length_kind;

/** An opaque plan of any kind and precision, fftn::plan: reference counted. */
typedef struct fftn_plan fftn_plan;

/** An opaque workspace, fftn::workspace: scratch memory and the team's state word. */
typedef struct fftn_workspace fftn_workspace;

/** The options fftn::options{} gives: no in place, scale 1, the best ISA, the rules. */
FFTN_C_API fftn_options fftn_options_default(void) FFTN_C_NOEXCEPT;

/**
 * Builds the plan of `shape` with `options` (NULL: fftn_options_default()), as the
 * constructor of fftn::plan does, and returns it with one reference; NULL on failure.
 *
 * Fails with FFTN_ERROR_INVALID_ARGUMENT, and the message naming the argument, where the C++
 * constructor throws std::invalid_argument, and also when `shape` is NULL, a field holds a
 * value that is not one of its enumerators (kind, precision, direction of FFTN_C2C, an
 * r2r_kinds entry, a run's where, options->force_isa, options->effort), an array is NULL
 * while its count is not 0, or r2r_kinds is NULL for FFTN_R2R or not NULL for another kind.
 * Fails with FFTN_ERROR_OUT_OF_MEMORY when memory runs out. `error` may be NULL.
 */
FFTN_C_API fftn_plan* fftn_plan_create(const fftn_shape* shape, const fftn_options* options,
                                       fftn_error* error) FFTN_C_NOEXCEPT;

/**
 * Adds a reference to `plan` and returns it, as a C++ copy shares the plan's tables; NULL for
 * NULL. Never fails. Safe to call from several threads at once.
 */
FFTN_C_API fftn_plan* fftn_plan_ref(fftn_plan* plan) FFTN_C_NOEXCEPT;

/**
 * Drops a reference to `plan`; the last frees it. Nothing for NULL. Safe to call from several
 * threads at once. Precondition: no execution of the plan is running on its last reference.
 */
FFTN_C_API void fftn_plan_release(fftn_plan* plan) FFTN_C_NOEXCEPT;

/** fftn::plan::workspace_bytes(lanes): the bytes of workspace for teams of up to `lanes`. */
FFTN_C_API size_t fftn_plan_workspace_bytes(const fftn_plan* plan, int lanes) FFTN_C_NOEXCEPT;

/**
 * fftn::plan::describe() copied into `buffer` as snprintf does (`buffer` may be NULL when
 * `size` is 0); returns the description's full length. Returns 0, with an empty string in a
 * buffer of at least one byte, when memory runs out.
 */
FFTN_C_API size_t fftn_plan_describe(const fftn_plan* plan, char* buffer,
                                     size_t size) FFTN_C_NOEXCEPT;

/** fftn::plan::instruction_set(): the fftn_isa the plan's kernels use, never FFTN_ISA_BEST. */
FFTN_C_API int fftn_plan_instruction_set(const fftn_plan* plan) FFTN_C_NOEXCEPT;

/**
 * A workspace that owns `bytes` bytes of uninitialised memory (0: none), fftn::workspace(bytes);
 * NULL with FFTN_ERROR_OUT_OF_MEMORY when memory runs out. The handle and the memory are
 * allocated here, never during an execution.
 */
FFTN_C_API fftn_workspace* fftn_workspace_create(size_t bytes, fftn_error* error) FFTN_C_NOEXCEPT;

/**
 * A workspace over the caller's `bytes` bytes at `memory`, of any alignment, which must stay
 * valid while the workspace is in use and is never freed by it, as fftn::workspace(span).
 * Allocates the handle only. NULL with FFTN_ERROR_INVALID_ARGUMENT when `memory` is NULL and
 * `bytes` is not 0, and with FFTN_ERROR_OUT_OF_MEMORY when memory runs out.
 */
FFTN_C_API fftn_workspace* fftn_workspace_borrow(void* memory, size_t bytes,
                                                 fftn_error* error) FFTN_C_NOEXCEPT;

/**
 * Frees the workspace, and its memory if it owns it. Nothing for NULL. Precondition: no
 * execution is using it.
 */
FFTN_C_API void fftn_workspace_destroy(fftn_workspace* workspace) FFTN_C_NOEXCEPT;

/**
 * fftn::plan::execute(in, out, ws): every transform of the batch, serially on the calling
 * thread. Precondition: `workspace` is not NULL and holds fftn_plan_workspace_bytes(plan, 1).
 */
FFTN_C_API void fftn_execute(const fftn_plan* plan, const void* in, void* out,
                             fftn_workspace* workspace) FFTN_C_NOEXCEPT;

/**
 * fftn::plan::execute_transforms(first, count, in, out, ws): `count` transforms from
 * transform `first`, serially; `in` and `out` address transform 0, as for fftn_execute.
 */
FFTN_C_API void fftn_execute_transforms(const fftn_plan* plan, size_t first, size_t count,
                                        const void* in, void* out,
                                        fftn_workspace* workspace) FFTN_C_NOEXCEPT;

/**
 * The team overload of fftn::plan::execute: every member of team `team` calls it with its own
 * rank and the same other arguments, the same workspace among them (docs/threading.md).
 * Precondition: `workspace` holds fftn_plan_workspace_bytes(plan, team.size).
 */
FFTN_C_API void fftn_execute_team(const fftn_plan* plan, const void* in, void* out, fftn_team team,
                                  fftn_workspace* workspace, fftn_completion done) FFTN_C_NOEXCEPT;

/** fftn::plan::transforms_of(team): the transforms `team`'s member owns. */
FFTN_C_API fftn_transform_range fftn_transforms_of(const fftn_plan* plan,
                                                   fftn_team team) FFTN_C_NOEXCEPT;

/**
 * The executor overload of fftn::plan::execute: every transform of the batch, on the pool that
 * `executor` describes (fftn/executor.h, docs/threading.md). Returns FFTN_OK when the output
 * is written; FFTN_ERROR_INVALID_ARGUMENT when `executor`, its concurrency or its bulk, or
 * `workspace` is NULL (nothing is executed); FFTN_ERROR_OUT_OF_MEMORY or FFTN_ERROR_EXECUTOR
 * when a C++ executor behind the struct threw, the output then unspecified and the workspace
 * ready for its next execution. Precondition: `workspace` holds
 * fftn_plan_workspace_bytes(plan, concurrency).
 */
FFTN_C_API int fftn_execute_on(const fftn_plan* plan, const void* in, void* out,
                               const fftn_executor* executor,
                               fftn_workspace* workspace) FFTN_C_NOEXCEPT;

/** A wisdom with no entries, fftn::wisdom(); NULL with FFTN_ERROR_OUT_OF_MEMORY on failure. */
FFTN_C_API fftn_wisdom* fftn_wisdom_create(fftn_error* error) FFTN_C_NOEXCEPT;

/**
 * The entries in the NUL-terminated `text`, fftn::wisdom(text); NULL on failure, with
 * FFTN_ERROR_INVALID_ARGUMENT and the C++ message naming the first bad line, or for a NULL
 * `text`.
 */
FFTN_C_API fftn_wisdom* fftn_wisdom_from_text(const char* text, fftn_error* error) FFTN_C_NOEXCEPT;

/**
 * The entries of the file at `path` (a NUL-terminated path in the platform's narrow
 * encoding), fftn::wisdom::load; NULL on failure, with FFTN_ERROR_FILE when the file cannot be
 * read and FFTN_ERROR_INVALID_ARGUMENT when a line is not an entry or `path` is NULL.
 */
FFTN_C_API fftn_wisdom* fftn_wisdom_load(const char* path, fftn_error* error) FFTN_C_NOEXCEPT;

/**
 * Writes the wisdom's text to the file at `path`, replacing it, fftn::wisdom::save. Returns
 * FFTN_OK, or FFTN_ERROR_FILE when the file cannot be written and FFTN_ERROR_INVALID_ARGUMENT
 * for a NULL `path`, with `error` filled.
 */
FFTN_C_API int fftn_wisdom_save(const fftn_wisdom* wisdom, const char* path,
                                fftn_error* error) FFTN_C_NOEXCEPT;

/**
 * fftn::wisdom::text() copied into `buffer` as snprintf does; returns the text's full length,
 * or 0 when memory runs out (a wisdom's text is never empty).
 */
FFTN_C_API size_t fftn_wisdom_text(const fftn_wisdom* wisdom, char* buffer,
                                   size_t size) FFTN_C_NOEXCEPT;

/**
 * Records the program of `plan`, fftn::wisdom::add(plan). Returns FFTN_OK, or
 * FFTN_ERROR_OUT_OF_MEMORY with `error` filled.
 */
FFTN_C_API int fftn_wisdom_add(fftn_wisdom* wisdom, const fftn_plan* plan,
                               fftn_error* error) FFTN_C_NOEXCEPT;

/** fftn::wisdom::size(): the number of entries. */
FFTN_C_API size_t fftn_wisdom_size(const fftn_wisdom* wisdom) FFTN_C_NOEXCEPT;

/** Frees the wisdom. Nothing for NULL. Plans built from it do not keep it. */
FFTN_C_API void fftn_wisdom_destroy(fftn_wisdom* wisdom) FFTN_C_NOEXCEPT;

/**
 * fftn::next_fast_length(n, kind): the smallest length at or above `n` that FFTN transforms
 * fast (docs/lengths.md). Returns 0, with FFTN_ERROR_INVALID_ARGUMENT, where the C++ function
 * throws (n above 2^40, or `kind` not an fftn_length_kind).
 */
FFTN_C_API size_t fftn_next_fast_length(size_t n, fftn_length_kind kind,
                                        fftn_error* error) FFTN_C_NOEXCEPT;

/**
 * fftn::relative_cost<T>(n, target), with T given by `precision` and `target` an fftn_isa:
 * the planner's estimate of a complex row's time per point relative to the power of two at or
 * above `n`. Returns 0, with FFTN_ERROR_INVALID_ARGUMENT, where the C++ function throws, and
 * for a `precision` that is not an fftn_precision.
 */
FFTN_C_API double fftn_relative_cost(size_t n, fftn_precision precision, int target,
                                     fftn_error* error) FFTN_C_NOEXCEPT;

/** fftn::isa_name: the instruction set's name ("avx2", "best", "unknown" for others). */
FFTN_C_API const char* fftn_isa_name(int target) FFTN_C_NOEXCEPT;

/** fftn::version(): the linked library's version as "major.minor.patch". */
FFTN_C_API const char* fftn_version(void) FFTN_C_NOEXCEPT;

/* ---- Fusion: edges, block callbacks and pipelines (step 6.9) ----
 *
 * Provisional (step 6.9, dev/DESIGN.md 5.9 and 12): the C form of fftn/edges.hpp,
 * fftn/callback.hpp and fftn/pipeline.hpp, whose C++ API the maintainer accepted as built on
 * 2026-11-26. dev/DESIGN.md 5.9 names the edge types and gives no C block for the rest, so the
 * choices listed in its "As built (step 6.9)" are provisional too. The C++ headers state what
 * each operation computes, its preconditions and its guarantees; docs/fusion.md says which tool
 * fits which work, and docs/c_api.md ("Fusion") quotes a compiled C example. */

/** What an edge's factor multiplies by, fftn::factor_kind. */
typedef enum fftn_factor_kind {
  FFTN_FACTOR_NONE,    /**< no factor */
  FFTN_FACTOR_REAL,    /**< real values in the plan's precision */
  FFTN_FACTOR_COMPLEX, /**< complex values (interleaved pairs) in the plan's precision */
} fftn_factor_kind;

/**
 * A factor of an edge, fftn::factor: the strides are borrowed arrays, a pointer and a count,
 * read during the call that builds the plan or pipeline only. The array itself arrives with
 * each execution (fftn_exec_args). NULL strides only with a count of 0; a batch stride count of
 * 0 broadcasts along every batch dimension, as the C++ empty vector does.
 */
typedef struct fftn_factor {
  fftn_factor_kind kind;         /**< FFTN_FACTOR_NONE: no factor (the strides are ignored) */
  size_t axis_stride_count;      /**< the entries of axis_strides: one per axis */
  const ptrdiff_t* axis_strides; /**< the array's stride along each axis, in its elements */
  size_t batch_stride_count;     /**< the entries of batch_strides: one per batch dimension, or 0 */
  const ptrdiff_t* batch_strides; /**< the array's stride along each batch dimension */
} fftn_factor;

/**
 * fftn::along_transforms: fftn_block_info::line_axis when the lines of a block are consecutive
 * transforms of the batch.
 */
#define FFTN_ALONG_TRANSFORMS SIZE_MAX

/**
 * Where a block lies, fftn::block_info, handed to a C callback with each call and valid during
 * the call only. Element i of line k is at `data` advanced by `i * element_stride + k *
 * line_stride` elements of the edge's type (a complex value is one element: two reals).
 */
typedef struct fftn_block_info {
  size_t transform;         /**< the batch number of line 0's transform */
  size_t rank;              /**< the entries of origin: the shape's axes (1 for a pipeline) */
  const size_t* origin;     /**< the logical index of element 0 of line 0 along each axis */
  size_t element_axis;      /**< the axis (an index into the shape's axes) elements advance along */
  size_t line_axis;         /**< the axis lines advance along, or FFTN_ALONG_TRANSFORMS */
  size_t length;            /**< the elements per line */
  size_t lines;             /**< the lines of the block, at least 1 */
  ptrdiff_t element_stride; /**< the distance between consecutive elements of a line */
  ptrdiff_t line_stride;    /**< the distance between consecutive lines (0 for one line) */
  size_t group;             /**< the members of a pipeline's group handed together; 1 otherwise */
  ptrdiff_t group_stride;   /**< the distance between consecutive members of a group, or 0 */
  size_t copy;              /**< which copy of a pipeline's fan-out an epilogue block is; else 0 */
  int member;               /**< the member running the call: 0 serially, the rank, the task */
  void* scratch;            /**< this member's scratch, on a 128-byte boundary; NULL for none */
  size_t scratch_bytes;     /**< the bytes at scratch: fftn_callback::scratch_bytes */
} fftn_block_info;

/**
 * A block callback, fftn::block_fn in C: `user` is the edge's pointer in fftn_exec_args
 * (in_user, mid_user, out_user) as given, NULL included; `data` addresses element 0 of line 0,
 * of the edge's element type (the plan's precision; complex values interleaved).
 *
 * fftn/callback.hpp's contract holds: coverage (each element once per execution, in blocks of
 * the plan's choosing, possibly on several threads at once and in any order), purity for
 * bitwise-identical results across execution forms, no use of the workspace but `scratch`, no
 * writes outside the block but to the application's own memory, and nested serial executions
 * only (fftn_execute, fftn_execute_args, fftn_execute_transforms and its _args form, or a
 * pipeline's serial forms), each with a workspace of its own.
 *
 * **Failure.** A callback returns nothing and cannot stop the execution: one that fails records
 * the failure in the application's memory (through `user`, written by any member, so an atomic
 * flag or one slot per `member`) and returns, and the caller checks it after the execution,
 * whose output is then the application's to discard. A callback must return normally: leaving
 * it by `longjmp` is undefined (team members would wait for it for ever), and a C++ exception
 * that reaches FFTN from it ends the program (`std::terminate`), since none crosses FFTN.
 */
typedef void (*fftn_block_fn)(void* user, const fftn_block_info* info, void* data);

/** What a callback does with its block, fftn::callback_mode. */
typedef enum fftn_callback_mode {
  FFTN_CALLBACK_MODIFY,  /**< changes the edge's values in place */
  FFTN_CALLBACK_PRODUCE, /**< input edges only: fills the block; the input is not read */
  FFTN_CALLBACK_CONSUME, /**< output edges only: takes the block; the output is not written */
} fftn_callback_mode;

/** The callback of an edge, fftn::callback: a NULL fn is none, with the other fields 0. */
typedef struct fftn_callback {
  fftn_block_fn fn;        /**< the function, or NULL */
  fftn_callback_mode mode; /**< FFTN_CALLBACK_MODIFY by default */
  size_t scratch_bytes;    /**< scratch per member, counted in the workspace's bytes */
} fftn_callback;

/**
 * One edge, fftn::edge: a scale, a factor, accumulation (output edges only) and a callback.
 * Start from fftn_edge_default(): a zeroed struct has a scale of 0, which is a valid scale.
 */
typedef struct fftn_edge {
  double scale;       /**< a real factor; 1 */
  fftn_factor mul;    /**< the factor's kind and strides; none */
  int accumulate;     /**< nonzero: out += result (output edges only); 0 */
  fftn_callback call; /**< a block callback; none */
} fftn_edge;

/** A plan's two edges, fftn::plan_edges. */
typedef struct fftn_plan_edges {
  fftn_edge in;  /**< the input edge */
  fftn_edge out; /**< the output edge */
} fftn_plan_edges;

/**
 * The data of one execution, fftn::exec_args: the factor arrays and the callbacks' user
 * pointers. Every pointer may be NULL where nothing reads it; a zeroed struct is no data.
 */
typedef struct fftn_exec_args {
  const void* in_factor;  /**< the input edge's (or a pipeline prologue's) factor array */
  const void* mid_factor; /**< a pipeline middle's factor array */
  const void* out_factor; /**< the output edge's (or a pipeline epilogue's) factor array */
  void* in_user;          /**< the input edge's (or prologue's) callback argument */
  void* mid_user;         /**< a pipeline middle's callback argument */
  void* out_user;         /**< the output edge's (or epilogue's) callback argument */
} fftn_exec_args;

/** fftn::edge{}: scale 1, no factor, no accumulation, no callback. */
FFTN_C_API fftn_edge fftn_edge_default(void) FFTN_C_NOEXCEPT;

/**
 * Builds the plan of `shape` with `options` (NULL: the defaults) and `edges` (NULL: none), as
 * the constructor of fftn::plan that takes plan_edges does, and returns it with one reference;
 * NULL on failure. With NULL edges it is fftn_plan_create.
 *
 * Fails as fftn_plan_create does, and with FFTN_ERROR_INVALID_ARGUMENT where the C++ constructor
 * refuses the edges (the message names the C++ field), where a factor's kind or a callback's
 * mode is not one of its enumerators, and where a stride array is NULL with a count that is not
 * 0. The edges' stride arrays are borrowed for the call only; their callbacks are kept.
 */
FFTN_C_API fftn_plan* fftn_plan_create_edges(const fftn_shape* shape, const fftn_options* options,
                                             const fftn_plan_edges* edges,
                                             fftn_error* error) FFTN_C_NOEXCEPT;

/**
 * fftn::plan::execute(in, out, ws, args): fftn_execute with the data of the execution (`args`;
 * NULL is a zeroed fftn_exec_args). A plan with factors or callbacks must be executed through
 * the _args forms: the others, like an execution without an array the plan needs, end the
 * program with a message on stderr, as in C++.
 */
FFTN_C_API void fftn_execute_args(const fftn_plan* plan, const void* in, void* out,
                                  fftn_workspace* workspace,
                                  const fftn_exec_args* args) FFTN_C_NOEXCEPT;

/** fftn::plan::execute_transforms with exec_args: fftn_execute_transforms with `args`. */
FFTN_C_API void fftn_execute_transforms_args(const fftn_plan* plan, size_t first, size_t count,
                                             const void* in, void* out, fftn_workspace* workspace,
                                             const fftn_exec_args* args) FFTN_C_NOEXCEPT;

/**
 * The team fftn::plan::execute with exec_args: fftn_execute_team with `args`, the same on every
 * member (each member's callbacks receive the user pointers of its own call).
 */
FFTN_C_API void fftn_execute_team_args(const fftn_plan* plan, const void* in, void* out,
                                       fftn_team team, fftn_workspace* workspace,
                                       fftn_completion done,
                                       const fftn_exec_args* args) FFTN_C_NOEXCEPT;

/** The executor fftn::plan::execute with exec_args: fftn_execute_on with `args`. */
FFTN_C_API int fftn_execute_on_args(const fftn_plan* plan, const void* in, void* out,
                                    const fftn_executor* executor, fftn_workspace* workspace,
                                    const fftn_exec_args* args) FFTN_C_NOEXCEPT;

/**
 * The kinds of row pipeline, fftn::pipeline<In, Mid, Out>'s element types, with C the complex
 * type of the precision and T its real type.
 */
typedef enum fftn_pipeline_kind {
  FFTN_PIPELINE_CCC, /**< <C, C, C>: two complex transforms, of directions first and -first */
  FFTN_PIPELINE_TCT, /**< <T, C, T>: r2c, a complex middle of n/2 + 1 bins, then c2r */
  FFTN_PIPELINE_CTC, /**< <C, T, C>: c2r, a real middle of n points, then r2c */
} fftn_pipeline_kind;

/** The middle of a pipeline, fftn::pipeline_desc::middle: no accumulation. */
typedef struct fftn_middle {
  double scale;       /**< a real factor; 1 */
  fftn_factor mul;    /**< the factor by bin: one axis stride, batch strides or none */
  fftn_callback call; /**< a callback on each whole line of the spectrum (modify only) */
} fftn_middle;

/** fftn::no_group: fftn_pipeline_desc::group without a group. */
#define FFTN_NO_GROUP SIZE_MAX

/**
 * A row pipeline's description, fftn::pipeline_desc, field by field; the arrays (batches,
 * runs, the factors' strides) are borrowed for the call of fftn_pipeline_create only. Start
 * from fftn_pipeline_desc_default(): a zeroed struct has scales of 0, a fan-out of 0 and a
 * group of batch dimension 0.
 */
typedef struct fftn_pipeline_desc {
  fftn_pipeline_kind kind;      /**< the kind; FFTN_PIPELINE_CCC */
  fftn_precision precision;     /**< the precision; FFTN_DOUBLE */
  int first;                    /**< FFTN_PIPELINE_CCC: the first direction; FFTN_FORWARD */
  fftn_axis line;               /**< the line: its length n and strides (0: tight) */
  size_t batch_rank;            /**< the batch dimensions; 0 for a single line */
  const fftn_batch* batches;    /**< batch_rank batch dimensions, slowest first */
  size_t run_count;             /**< the stored runs of the line; 0 for none */
  const fftn_stored_run* runs;  /**< run_count runs, each of axis 0 */
  fftn_edge prologue;           /**< the input edge (no accumulation) */
  fftn_middle middle;           /**< what multiplies the spectrum between the transforms */
  fftn_edge epilogue;           /**< the output edge */
  size_t fanout;                /**< inverse transforms (copies) per line; 1 */
  ptrdiff_t fanout_out_dist;    /**< the distance between copies' outputs, in Out elements */
  ptrdiff_t fanout_factor_dist; /**< the distance between copies' middle factors */
  size_t group; /**< a batch dimension the middle callback sees whole; FFTN_NO_GROUP */
} fftn_pipeline_desc;

/** An opaque row pipeline of any kind and precision, fftn::pipeline: reference counted. */
typedef struct fftn_pipeline fftn_pipeline;

/**
 * fftn::pipeline_desc{} in C: FFTN_PIPELINE_CCC, FFTN_DOUBLE, first FFTN_FORWARD, scales 1,
 * no factors, callbacks, batches or runs, a fan-out of 1, FFTN_NO_GROUP. The line's length is
 * left 0, to set.
 */
FFTN_C_API fftn_pipeline_desc fftn_pipeline_desc_default(void) FFTN_C_NOEXCEPT;

/**
 * Builds the pipeline `desc` with `options` (NULL: the defaults), as the constructor of
 * fftn::pipeline does, and returns it with one reference; NULL on failure.
 *
 * Fails with FFTN_ERROR_INVALID_ARGUMENT, and the message naming the field (the C++ names, such
 * as desc.middle.mul.kind), where the C++ constructor throws std::invalid_argument, and also when
 * `desc` is NULL, a field holds a value that is not one of its enumerators (kind, precision,
 * first of FFTN_PIPELINE_CCC, a run's where, a factor's kind, a callback's mode), or an array is
 * NULL while its count is not 0; with FFTN_ERROR_OUT_OF_MEMORY when memory runs out.
 */
FFTN_C_API fftn_pipeline* fftn_pipeline_create(const fftn_pipeline_desc* desc,
                                               const fftn_options* options,
                                               fftn_error* error) FFTN_C_NOEXCEPT;

/** Adds a reference to `pipeline` and returns it; NULL for NULL. As fftn_plan_ref. */
FFTN_C_API fftn_pipeline* fftn_pipeline_ref(fftn_pipeline* pipeline) FFTN_C_NOEXCEPT;

/** Drops a reference; the last frees the pipeline. Nothing for NULL. As fftn_plan_release. */
FFTN_C_API void fftn_pipeline_release(fftn_pipeline* pipeline) FFTN_C_NOEXCEPT;

/** fftn::pipeline::workspace_bytes(lanes). */
FFTN_C_API size_t fftn_pipeline_workspace_bytes(const fftn_pipeline* pipeline,
                                                int lanes) FFTN_C_NOEXCEPT;

/** fftn::pipeline::describe() copied into `buffer` as fftn_plan_describe does. */
FFTN_C_API size_t fftn_pipeline_describe(const fftn_pipeline* pipeline, char* buffer,
                                         size_t size) FFTN_C_NOEXCEPT;

/** fftn::pipeline::instruction_set(): an fftn_isa, never FFTN_ISA_BEST. */
FFTN_C_API int fftn_pipeline_instruction_set(const fftn_pipeline* pipeline) FFTN_C_NOEXCEPT;

/**
 * The transforms of the batch as fftn_pipeline_execute_transforms and fftn_pipeline_transforms_of
 * count them, fftn::pipeline::transform_count(): the lines, or with a group the groups.
 */
FFTN_C_API size_t fftn_pipeline_transform_count(const fftn_pipeline* pipeline) FFTN_C_NOEXCEPT;

/**
 * fftn::pipeline::execute(in, out, ws, args): every line, serially. `args` NULL is a zeroed
 * fftn_exec_args; an execution without an array the pipeline needs ends the program, as in C++.
 */
FFTN_C_API void fftn_pipeline_execute(const fftn_pipeline* pipeline, const void* in, void* out,
                                      fftn_workspace* workspace,
                                      const fftn_exec_args* args) FFTN_C_NOEXCEPT;

/** fftn::pipeline::execute_transforms: `count` transforms from `first`, serially. */
FFTN_C_API void fftn_pipeline_execute_transforms(const fftn_pipeline* pipeline, size_t first,
                                                 size_t count, const void* in, void* out,
                                                 fftn_workspace* workspace,
                                                 const fftn_exec_args* args) FFTN_C_NOEXCEPT;

/** The team fftn::pipeline::execute with an explicit completion, as fftn_execute_team_args. */
FFTN_C_API void fftn_pipeline_execute_team(const fftn_pipeline* pipeline, const void* in, void* out,
                                           fftn_team team, fftn_workspace* workspace,
                                           fftn_completion done,
                                           const fftn_exec_args* args) FFTN_C_NOEXCEPT;

/** fftn::pipeline::transforms_of(team): the transforms `team`'s member owns. */
FFTN_C_API fftn_transform_range fftn_pipeline_transforms_of(const fftn_pipeline* pipeline,
                                                            fftn_team team) FFTN_C_NOEXCEPT;

/**
 * The executor fftn::pipeline::execute, with the statuses and preconditions of
 * fftn_execute_on.
 */
FFTN_C_API int fftn_pipeline_execute_on(const fftn_pipeline* pipeline, const void* in, void* out,
                                        const fftn_executor* executor, fftn_workspace* workspace,
                                        const fftn_exec_args* args) FFTN_C_NOEXCEPT;

#ifdef __cplusplus
} /* extern "C" */
#endif

/* NOLINTEND(modernize-use-using,modernize-redundant-void-arg,performance-enum-size) */

#endif /* FFTN_C_API_H */
