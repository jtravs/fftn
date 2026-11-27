/* SPDX-FileCopyrightText: 2026 John C. Travers and contributors
 * SPDX-License-Identifier: Apache-2.0 */

/* The C half of the split-complex C API's tests, compiled as C99 with the strict warnings
 * (tests/api/CMakeLists.txt): split plans built, executed through the four _split forms and
 * released from C; the refusals of split shapes. */
#include "c_split_test.h"

#include "c_api_test.h"

#include "fftn/c_api.h"

#include <stddef.h>
#include <string.h>

/* What a team member or a pool's task needs. */
typedef struct csplit_job {
  const fftn_plan* plan;
  fftn_split_const_ptr in;
  fftn_split_ptr out;
  fftn_workspace* ws;
  int lanes;
  int group_size;
  fftn_completion done;
} csplit_job;

static void csplit_member(void* job, int rank) {
  const csplit_job* j = (const csplit_job*)job;
  fftn_team team;
  team.rank = rank;
  team.size = j->lanes;
  team.group_size = j->group_size;
  fftn_execute_team_split(j->plan, j->in, j->out, team, j->ws, j->done);
}

static int csplit_on_pool(void* job, const fftn_executor* executor) {
  const csplit_job* j = (const csplit_job*)job;
  return fftn_execute_on_split(j->plan, j->in, j->out, executor, j->ws);
}

/* Whether the members' own transforms cover the batch once, in rank order. */
static int csplit_covered(const csplit_job* j, size_t transforms) {
  size_t covered = 0;
  int r;
  for (r = 0; r < j->lanes; ++r) {
    fftn_team team;
    fftn_transform_range own;
    team.rank = r;
    team.size = j->lanes;
    team.group_size = j->group_size;
    own = fftn_transforms_of(j->plan, team);
    if (own.count != 0 && own.first != covered) {
      return 0;
    }
    covered += own.count;
  }
  return covered == transforms;
}

int csplit_run(const fftn_shape* shape, const fftn_options* options, ctest_form form, int lanes,
               int group_size, size_t transforms, fftn_split_const_ptr in, fftn_split_ptr out,
               fftn_error* error) {
  csplit_job j;
  int status = FFTN_OK;
  fftn_plan* plan = fftn_plan_create(shape, options, error);
  if (plan == NULL) {
    return error != NULL ? error->code : FFTN_ERROR_INTERNAL;
  }
  memset(&j, 0, sizeof j);
  j.plan = plan;
  j.in = in;
  j.out = out;
  j.lanes = lanes;
  j.group_size = group_size;
  j.ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, lanes), error);
  if (j.ws == NULL) {
    fftn_plan_release(plan);
    return error != NULL ? error->code : FFTN_ERROR_INTERNAL;
  }
  switch (form) {
  case CTEST_SERIAL:
    fftn_execute_split(plan, in, out, j.ws);
    break;
  case CTEST_PIECES:
    /* One transform, an empty piece, then the rest. */
    fftn_execute_transforms_split(plan, 0, 1, in, out, j.ws);
    fftn_execute_transforms_split(plan, 1, 0, in, out, j.ws);
    fftn_execute_transforms_split(plan, 1, transforms - 1, in, out, j.ws);
    break;
  case CTEST_TEAM_ALL:
  case CTEST_TEAM_OWN:
    j.done = form == CTEST_TEAM_OWN ? FFTN_COMPLETION_OWN : FFTN_COMPLETION_ALL;
    if (!csplit_covered(&j, transforms) || ctest_parallel(lanes, csplit_member, &j) != 0) {
      status = FFTN_ERROR_INTERNAL;
    }
    break;
  case CTEST_EXECUTOR:
    status = ctest_with_pool(lanes, group_size, csplit_on_pool, &j);
    break;
  default:
    status = FFTN_ERROR_INVALID_ARGUMENT;
  }
  fftn_workspace_destroy(j.ws);
  fftn_plan_release(plan);
  return status;
}

/* ---- Refusals ---- */

/* An executor that runs its tasks on the calling thread, for the refusals: with a NULL workspace
 * fftn_execute_on_split must refuse before calling it. */
static int csplit_serial_concurrency(void* self) {
  (void)self;
  return 1;
}

static void csplit_serial_bulk(void* self, int n, fftn_task task, void* arg) {
  int i;
  (void)self;
  for (i = 0; i < n; ++i) {
    task(arg, i);
  }
}

/* Whether `e` holds `code` and a non-empty message containing `needle`. */
static int csplit_error_is(const fftn_error* e, int code, const char* needle) {
  return e->code == code && e->message[0] != '\0' && strlen(e->message) < sizeof e->message &&
         strstr(e->message, needle) != NULL;
}

/* Whether creating a plan of `s` with `o` (and `edges`, when not NULL) fails with
 * FFTN_ERROR_INVALID_ARGUMENT and a message containing `needle`. */
static int csplit_refused(const fftn_shape* s, const fftn_options* o, const fftn_plan_edges* edges,
                          const char* needle) {
  fftn_error e;
  fftn_plan* p;
  e.code = -1;
  e.message[0] = '\0';
  p = edges != NULL ? fftn_plan_create_edges(s, o, edges, &e) : fftn_plan_create(s, o, &e);
  if (p != NULL) {
    fftn_plan_release(p);
    return 0;
  }
  return csplit_error_is(&e, FFTN_ERROR_INVALID_ARGUMENT, needle);
}

int csplit_refusals(void) {
  const fftn_axis axis = {16, 0, 0};
  const fftn_batch batch = {4, 0, 0};
  const fftn_r2r_kind dct = FFTN_DCT2;
  const fftn_stored_run run = {0, FFTN_IN, 0, 0, 4};
  fftn_shape s;
  fftn_options o = fftn_options_default();
  fftn_plan_edges edges;
  fftn_error e;
  fftn_plan* p;
  fftn_wisdom* w;
  fftn_workspace* ws;
  double re[128] = {0}; /* the input in the first half, the output in the second */
  double im[128] = {0};
  fftn_split_const_ptr in;
  fftn_split_ptr out;
  fftn_executor none;
  fftn_executor serial;

  memset(&s, 0, sizeof s);
  s.kind = FFTN_C2C;
  s.precision = FFTN_DOUBLE;
  s.direction = FFTN_FORWARD;
  s.rank = 1;
  s.axes = &axis;
  s.batch_rank = 1;
  s.batches = &batch;
  s.layout = FFTN_SPLIT;

  /* 1: real-to-real has no complex side, refused in C naming the field. */
  s.kind = FFTN_R2R;
  s.r2r_kinds = &dct;
  if (!csplit_refused(&s, NULL, NULL, "shape.layout")) {
    return 1;
  }
  s.kind = FFTN_C2C;
  s.r2r_kinds = NULL;

  /* A layout that is not an fftn_layout is refused, not read as split, on either side of the
   * range (stored through memcpy: the field's type need not hold it). */
  {
    const int bad[2] = {2, -1};
    int i;
    for (i = 0; i < 2; ++i) {
      memcpy(&s.layout, &bad[i], sizeof bad[i]);
      if (!csplit_refused(&s, NULL, NULL, "not an fftn_layout")) {
        return 1;
      }
    }
    s.layout = FFTN_SPLIT;
  }

  /* 2: stored runs, refused by the core. */
  s.run_count = 1;
  s.runs = &run;
  if (!csplit_refused(&s, NULL, NULL, "runs")) {
    return 2;
  }
  s.run_count = 0;
  s.runs = NULL;

  /* 3: r2c and c2r in place, refused by the core. */
  o.in_place = 1;
  s.kind = FFTN_R2C;
  if (!csplit_refused(&s, &o, NULL, "in_place")) {
    return 3;
  }
  s.kind = FFTN_C2R;
  if (!csplit_refused(&s, &o, NULL, "in_place")) {
    return 4;
  }
  s.kind = FFTN_C2C;
  o.in_place = 0;

  /* 5: edges, even the defaults; NULL edges build the split plan. */
  edges.in = fftn_edge_default();
  edges.out = fftn_edge_default();
  if (!csplit_refused(&s, NULL, &edges, "edges")) {
    return 5;
  }
  p = fftn_plan_create_edges(&s, NULL, NULL, &e);
  if (p == NULL || e.code != FFTN_OK) {
    return 6;
  }

  /* 7: wisdom does not record split plans; the wisdom stays empty. */
  w = fftn_wisdom_create(NULL);
  if (w == NULL) {
    fftn_plan_release(p);
    return 7;
  }
  e.code = -1;
  if (fftn_wisdom_add(w, p, &e) != FFTN_ERROR_INVALID_ARGUMENT ||
      !csplit_error_is(&e, FFTN_ERROR_INVALID_ARGUMENT, "split") || fftn_wisdom_size(w) != 0) {
    fftn_wisdom_destroy(w);
    fftn_plan_release(p);
    return 8;
  }

  /* 9: the executor form refuses a NULL executor, function or workspace, executing nothing. */
  ws = fftn_workspace_create(fftn_plan_workspace_bytes(p, 1), NULL);
  in.re = re;
  in.im = im;
  out.re = re + 64;
  out.im = im + 64;
  memset(&none, 0, sizeof none);
  serial = none;
  serial.concurrency = csplit_serial_concurrency;
  serial.bulk = csplit_serial_bulk;
  if (ws == NULL || fftn_execute_on_split(p, in, out, NULL, ws) != FFTN_ERROR_INVALID_ARGUMENT ||
      fftn_execute_on_split(p, in, out, &none, ws) != FFTN_ERROR_INVALID_ARGUMENT ||
      fftn_execute_on_split(p, in, out, &serial, NULL) != FFTN_ERROR_INVALID_ARGUMENT ||
      fftn_execute_on_split(p, in, out, &serial, ws) != FFTN_OK) {
    fftn_workspace_destroy(ws);
    fftn_wisdom_destroy(w);
    fftn_plan_release(p);
    return 9;
  }
  fftn_workspace_destroy(ws);
  fftn_plan_release(p);

  /* 10: measured effort and wisdom to use are accepted; split plans are planned by the rules. */
  o.effort = FFTN_PLANNING_MEASURE;
  o.use = w;
  p = fftn_plan_create(&s, &o, &e);
  fftn_wisdom_destroy(w);
  if (p == NULL || e.code != FFTN_OK) {
    return 10;
  }
  fftn_plan_release(p);
  return 0;
}
