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

#ifdef __cplusplus
} /* extern "C" */
#endif

/* NOLINTEND(modernize-use-using,modernize-redundant-void-arg,performance-enum-size) */

#endif /* FFTN_C_API_H */
