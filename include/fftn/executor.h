/* SPDX-FileCopyrightText: 2026 John C. Travers and contributors
 * SPDX-License-Identifier: Apache-2.0 */

/**
 * @file
 * The executor contract in C: how FFTN runs its work on a thread pool it does not know.
 *
 * Valid C (C99 and later) and C++: a pool written in any language that can call C fills
 * in an fftn_executor, and C++ code passes it to the executor overload of
 * fftn::plan::execute through fftn::executor_ref. The header declares types only, no
 * functions, so it needs no linkage specification.
 */
#ifndef FFTN_EXECUTOR_H
#define FFTN_EXECUTOR_H

/**
 * One piece of work handed to an executor: FFTN calls the executor's bulk function with
 * one of these, and the executor calls it once for each index from 0 to n - 1, with the
 * `arg` it was given. FFTN's tasks never throw and never block on a task that has not
 * started.
 */
typedef void (*fftn_task)(void* arg, int index); /* NOLINT(modernize-use-using): C */

/**
 * A thread pool as FFTN sees it: three functions and the pool's own pointer, passed back
 * to each of them. FFTN calls them from the thread that called execute; it does not keep
 * the struct after execute returns.
 */
/* NOLINTNEXTLINE(modernize-use-using): C has no alias declarations */
typedef struct fftn_executor {
  /** The pool, passed as the first argument of each function below. */
  void* self;

  /**
   * The most tasks the pool can usefully run at once when called from the calling thread
   * now; at least 1. A pool that is called from one of its own tasks returns 1, so that a
   * nested call runs serially rather than waiting for threads that are busy. Required.
   */
  int (*concurrency)(void* self);

  /**
   * Runs task(arg, i) once for every i in [0, n), and returns when all of them have
   * returned. The tasks may run in any order, on any threads, any number at a time, the
   * calling thread among them, even one after another on the caller. Required.
   */
  void (*bulk)(void* self, int n, fftn_task task, void* arg);

  /**
   * How many consecutive task indices share a cache (the group_size of fftn::team); 0
   * puts all in one group. May be NULL, which means 0. It affects speed only, never
   * results.
   */
  int (*group_size)(void* self);
} fftn_executor;

#endif /* FFTN_EXECUTOR_H */
