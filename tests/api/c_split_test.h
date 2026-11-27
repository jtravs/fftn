/* SPDX-FileCopyrightText: 2026 John C. Travers and contributors
 * SPDX-License-Identifier: Apache-2.0 */

/* The C half of the split-complex C API's tests (c_split_test.c, compiled as C99): the C++ half
 * (c_split_test.cpp) describes each shape, calls these functions to build and execute its plan
 * from C, and compares the results with the C++ split plans of the same shapes. */
#ifndef FFTN_TESTS_C_SPLIT_TEST_H
#define FFTN_TESTS_C_SPLIT_TEST_H

#include "c_api_test.h"

#include "fftn/c_api.h"

/* NOLINTNEXTLINE(modernize-deprecated-headers): shared with C */
#include <stddef.h>

/* NOLINTBEGIN(modernize-use-using): C */
#ifdef __cplusplus
extern "C" {
#endif

/* Builds the plan of `shape` (whose layout is FFTN_SPLIT) with `options` in C, runs it once in
 * `form` through the _split executions on `in` and `out` (by `lanes` members or pool threads,
 * `group_size` per cache group), and releases it. `transforms` is the batch's count, for the
 * pieces and the team's coverage. Returns FFTN_OK or the first failing status, with `error`
 * filled when construction failed. */
int csplit_run(const fftn_shape* shape, const fftn_options* options, ctest_form form, int lanes,
               int group_size, size_t transforms, fftn_split_const_ptr in, fftn_split_ptr out,
               fftn_error* error);

/* The refusals of split shapes made in C or passed through from C++ (FFTN_R2R, stored runs, in
 * place r2c, edges, wisdom, NULL executors) and what split plans accept (NULL edges, measured
 * effort, wisdom to use). Returns 0, or the number of the first check that failed. */
int csplit_refusals(void);

#ifdef __cplusplus
}
#endif
/* NOLINTEND(modernize-use-using) */

#endif /* FFTN_TESTS_C_SPLIT_TEST_H */
