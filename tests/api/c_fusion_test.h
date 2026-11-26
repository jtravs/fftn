/* SPDX-FileCopyrightText: 2026 John C. Travers and contributors
 * SPDX-License-Identifier: Apache-2.0 */

/* The C half of the fusion C API's tests (c_fusion_test.c, compiled as C99): the C++ half
 * (c_fusion_test.cpp) describes each plan with edges and each pipeline in the C structs, calls
 * these functions to build and execute them from C, and compares the results with the C++
 * objects of the same descriptions, whose callbacks call the same C functions. */
#ifndef FFTN_TESTS_C_FUSION_TEST_H
#define FFTN_TESTS_C_FUSION_TEST_H

#include "c_api_test.h"

#include "fftn/c_api.h"

/* NOLINTNEXTLINE(modernize-deprecated-headers): shared with C */
#include <stddef.h>

/* NOLINTBEGIN(modernize-use-using,performance-enum-size): C */
#ifdef __cplusplus
extern "C" {
#endif

/* The element type of the block a callback receives. */
typedef enum cfuse_type {
  CFUSE_DOUBLE,  /* double */
  CFUSE_FLOAT,   /* float */
  CFUSE_CDOUBLE, /* complex double: two doubles */
  CFUSE_CFLOAT,  /* complex float: two floats */
} cfuse_type;

/* What a callback is given as its user pointer. The callbacks are pure per element (per group
 * for a group's middle): nonlinear and dependent on the element's coordinates. */
typedef struct cfuse_user {
  cfuse_type type;
  /* consume: the application's array the block's elements go to, element (coordinates c,
   * transform t, copy k) at `t * dist + sum_d c[d] * strides[d] + k * copy_dist`, in elements of
   * the type. */
  void* sink;
  ptrdiff_t strides[3];
  ptrdiff_t dist;
  ptrdiff_t copy_dist;
  /* A group's middle: the transform numbers between consecutive members of the group. */
  size_t group_step;
} cfuse_user;

/* modify: x -> x (1 + s/2) + x |x|^2 / 4 with s from the coordinates; with a group of two or
 * more, then a rotation that mixes members 0 and 1 (a coupled callback). */
void cfuse_modify(void* user, const fftn_block_info* info, void* data);
/* produce: values that depend on the coordinates only. */
void cfuse_produce(void* user, const fftn_block_info* info, void* data);
/* consume: each element into the user's sink at its coordinates. */
void cfuse_consume(void* user, const fftn_block_info* info, void* data);

/* Builds the plan of `shape`, `options` and `edges` in C, runs it once in `form` on `in` and
 * `out` with `args` (by `lanes` members or pool threads, `group_size` per cache group), and
 * releases it. `transforms` is the batch's count, for the pieces. Returns FFTN_OK or the first
 * failing status, with `error` filled when construction failed. */
int cfuse_run_plan(const fftn_shape* shape, const fftn_options* options,
                   const fftn_plan_edges* edges, ctest_form form, int lanes, int group_size,
                   size_t transforms, const void* in, void* out, const fftn_exec_args* args,
                   fftn_error* error);

/* The same for the pipeline `desc`; the pieces and the team's coverage follow
 * fftn_pipeline_transform_count. */
int cfuse_run_pipeline(const fftn_pipeline_desc* desc, const fftn_options* options, ctest_form form,
                       int lanes, int group_size, const void* in, void* out,
                       const fftn_exec_args* args, fftn_error* error);

/* Refusals made in C (enumerations out of range, NULL arrays with counts, zeroed descriptions)
 * and the defaults. Returns 0, or the number of the first check that failed. */
int cfuse_refusals(void);

#ifdef __cplusplus
}
#endif
/* NOLINTEND(modernize-use-using,performance-enum-size) */

#endif /* FFTN_TESTS_C_FUSION_TEST_H */
