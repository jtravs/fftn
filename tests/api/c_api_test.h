/* SPDX-FileCopyrightText: 2026 John C. Travers and contributors
 * SPDX-License-Identifier: Apache-2.0 */

/* The C half of the C API's tests (c_api_test.c, compiled as C99): the C++ half
 * (c_api_test.cpp) describes each case, calls these functions and compares their results with
 * the C++ plans of the same shapes. */
#ifndef FFTN_TESTS_C_API_TEST_H
#define FFTN_TESTS_C_API_TEST_H

#include "fftn/c_api.h"

/* NOLINTNEXTLINE(modernize-deprecated-headers): shared with C */
#include <stddef.h>

/* NOLINTBEGIN(modernize-use-using,performance-enum-size): C */
#ifdef __cplusplus
extern "C" {
#endif

/* How the C side executes a plan. */
typedef enum ctest_form {
  CTEST_SERIAL,   /* fftn_execute */
  CTEST_PIECES,   /* fftn_execute_transforms over pieces of the batch */
  CTEST_TEAM_ALL, /* fftn_execute_team, FFTN_COMPLETION_ALL, one thread per member */
  CTEST_TEAM_OWN, /* fftn_execute_team, FFTN_COMPLETION_OWN */
  CTEST_EXECUTOR, /* fftn_execute_on a C thread pool */
} ctest_form;

/* A shape of up to three axes and one batch dimension, with its options. */
typedef struct ctest_case {
  fftn_kind kind;
  fftn_precision precision;
  int direction;
  size_t rank;
  fftn_axis axes[3];
  fftn_batch batch;
  fftn_r2r_kind r2r[3];
  size_t run_count;
  fftn_stored_run runs[2];
  int in_place;
  double scale;
} ctest_case;

/* Builds the case's plan in C, runs it once in `form` (by `lanes` members or pool threads,
 * `group_size` per cache group) on `in` and `out`, and releases everything. Returns FFTN_OK,
 * or the status of the first call that failed, with `error` filled when construction failed. */
int ctest_run(const ctest_case* c, ctest_form form, int lanes, int group_size, const void* in,
              void* out, fftn_error* error);

/* Refusals made in C: enumerations out of range, NULL arrays, NULL executors, bad lengths.
 * Returns 0, or the number of the first check that failed. */
int ctest_refusals(void);

/* Wisdom in C: a plan of 64-point complex rows recorded, written as text, read back, saved to
 * `path`, loaded, and used. Copies the first wisdom's text into `text` (snprintf-like) and
 * returns 0, or the number of the first check that failed. */
int ctest_wisdom(const char* path, char* text, size_t size);

/* The length queries from C, with their status in `error`. */
size_t ctest_next_fast_length(size_t n, fftn_length_kind kind, fftn_error* error);
double ctest_relative_cost(size_t n, fftn_precision precision, fftn_error* error);

#ifdef __cplusplus
}
#endif
/* NOLINTEND(modernize-use-using,performance-enum-size) */

#endif /* FFTN_TESTS_C_API_TEST_H */
