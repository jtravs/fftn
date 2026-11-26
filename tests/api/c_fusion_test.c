/* SPDX-FileCopyrightText: 2026 John C. Travers and contributors
 * SPDX-License-Identifier: Apache-2.0 */

/* The C half of the fusion C API's tests (step 6.9), compiled as C99 with the strict warnings
 * (tests/api/CMakeLists.txt): plans with edges and pipelines built, executed in every form and
 * released from C, with callbacks written in C; the refusals made in C. */
#include "c_fusion_test.h"

#include "c_api_test.h"

#include "fftn/c_api.h"

#include <stddef.h>
#include <string.h>

/* ---- The callbacks ---- */

static int cfuse_complex(cfuse_type t) { return t == CFUSE_CDOUBLE || t == CFUSE_CFLOAT; }

/* The value at element `index` (in elements of the type) of `data`. */
static void cfuse_load(cfuse_type t, const void* data, ptrdiff_t index, double* re, double* im) {
  switch (t) {
  case CFUSE_DOUBLE:
    *re = ((const double*)data)[index];
    *im = 0;
    break;
  case CFUSE_FLOAT:
    *re = ((const float*)data)[index];
    *im = 0;
    break;
  case CFUSE_CDOUBLE:
    *re = ((const double*)data)[2 * index];
    *im = ((const double*)data)[2 * index + 1];
    break;
  case CFUSE_CFLOAT:
    *re = ((const float*)data)[2 * index];
    *im = ((const float*)data)[2 * index + 1];
    break;
  }
}

static void cfuse_store(cfuse_type t, void* data, ptrdiff_t index, double re, double im) {
  switch (t) {
  case CFUSE_DOUBLE:
    ((double*)data)[index] = re;
    break;
  case CFUSE_FLOAT:
    ((float*)data)[index] = (float)re;
    break;
  case CFUSE_CDOUBLE:
    ((double*)data)[2 * index] = re;
    ((double*)data)[2 * index + 1] = im;
    break;
  case CFUSE_CFLOAT:
    ((float*)data)[2 * index] = (float)re;
    ((float*)data)[2 * index + 1] = (float)im;
    break;
  }
}

/* The coordinates of element i of line k of group member g: along each axis, the transform and
 * the copy; returns a key that mixes them. */
static size_t cfuse_place(const cfuse_user* u, const fftn_block_info* info, size_t i, size_t k,
                          size_t g, size_t c[3], size_t* transform) {
  size_t d;
  size_t t = info->transform + g * u->group_step;
  c[0] = c[1] = c[2] = 0;
  for (d = 0; d < info->rank && d < 3; ++d) {
    c[d] = info->origin[d];
  }
  c[info->element_axis] += i;
  if (info->line_axis == FFTN_ALONG_TRANSFORMS) {
    t += k;
  } else {
    c[info->line_axis] += k;
  }
  *transform = t;
  return 1 + c[0] + 3 * c[1] + 5 * c[2] + 7 * t + 11 * info->copy;
}

static ptrdiff_t cfuse_at(const fftn_block_info* info, size_t i, size_t k, size_t g) {
  return (ptrdiff_t)i * info->element_stride + (ptrdiff_t)k * info->line_stride +
         (ptrdiff_t)g * info->group_stride;
}

void cfuse_modify(void* user, const fftn_block_info* info, void* data) {
  const cfuse_user* u = (const cfuse_user*)user;
  size_t g;
  size_t k;
  size_t i;
  for (g = 0; g < info->group; ++g) {
    for (k = 0; k < info->lines; ++k) {
      for (i = 0; i < info->length; ++i) {
        size_t c[3];
        size_t t;
        const double s = 1.0 / (double)cfuse_place(u, info, i, k, g, c, &t);
        const ptrdiff_t at = cfuse_at(info, i, k, g);
        double re;
        double im;
        double m;
        cfuse_load(u->type, data, at, &re, &im);
        m = re * re + im * im;
        re = re * (1 + 0.5 * s) + 0.015625 * re * m;
        im = im * (1 - 0.25 * s) + 0.015625 * im * m;
        cfuse_store(u->type, data, at, re, im);
      }
    }
  }
  /* A coupled callback: members 0 and 1 of a group rotated into each other. */
  if (info->group >= 2) {
    for (k = 0; k < info->lines; ++k) {
      for (i = 0; i < info->length; ++i) {
        double are;
        double aim;
        double bre;
        double bim;
        cfuse_load(u->type, data, cfuse_at(info, i, k, 0), &are, &aim);
        cfuse_load(u->type, data, cfuse_at(info, i, k, 1), &bre, &bim);
        cfuse_store(u->type, data, cfuse_at(info, i, k, 0), 0.6 * are - 0.8 * bre,
                    0.6 * aim - 0.8 * bim);
        cfuse_store(u->type, data, cfuse_at(info, i, k, 1), 0.8 * are + 0.6 * bre,
                    0.8 * aim + 0.6 * bim);
      }
    }
  }
}

void cfuse_produce(void* user, const fftn_block_info* info, void* data) {
  const cfuse_user* u = (const cfuse_user*)user;
  size_t k;
  size_t i;
  for (k = 0; k < info->lines; ++k) {
    for (i = 0; i < info->length; ++i) {
      size_t c[3];
      size_t t;
      const size_t key = cfuse_place(u, info, i, k, 0, c, &t);
      cfuse_store(u->type, data, cfuse_at(info, i, k, 0), 0.5 - 0.03 * (double)(key % 17),
                  0.25 - 0.02 * (double)((3 * key) % 13));
    }
  }
}

void cfuse_consume(void* user, const fftn_block_info* info, void* data) {
  const cfuse_user* u = (const cfuse_user*)user;
  size_t k;
  size_t i;
  for (k = 0; k < info->lines; ++k) {
    for (i = 0; i < info->length; ++i) {
      size_t c[3];
      size_t t;
      ptrdiff_t to;
      double re;
      double im;
      (void)cfuse_place(u, info, i, k, 0, c, &t);
      to = (ptrdiff_t)t * u->dist + (ptrdiff_t)c[0] * u->strides[0] +
           (ptrdiff_t)c[1] * u->strides[1] + (ptrdiff_t)c[2] * u->strides[2] +
           (ptrdiff_t)info->copy * u->copy_dist;
      cfuse_load(u->type, data, cfuse_at(info, i, k, 0), &re, &im);
      cfuse_store(u->type, u->sink, to, re, cfuse_complex(u->type) ? im : 0);
    }
  }
}

/* ---- Running plans and pipelines in every form ---- */

/* What a team member or a pool's task needs: one of the two objects and the execution. */
typedef struct cfuse_job {
  const fftn_plan* plan;
  const fftn_pipeline* pipeline;
  const void* in;
  void* out;
  fftn_workspace* ws;
  const fftn_exec_args* args;
  int lanes;
  int group_size;
  fftn_completion done;
} cfuse_job;

static void cfuse_member(void* job, int rank) {
  const cfuse_job* j = (const cfuse_job*)job;
  fftn_team team;
  team.rank = rank;
  team.size = j->lanes;
  team.group_size = j->group_size;
  if (j->plan != NULL) {
    fftn_execute_team_args(j->plan, j->in, j->out, team, j->ws, j->done, j->args);
  } else {
    fftn_pipeline_execute_team(j->pipeline, j->in, j->out, team, j->ws, j->done, j->args);
  }
}

static int cfuse_on_pool(void* job, const fftn_executor* executor) {
  const cfuse_job* j = (const cfuse_job*)job;
  return j->plan != NULL
             ? fftn_execute_on_args(j->plan, j->in, j->out, executor, j->ws, j->args)
             : fftn_pipeline_execute_on(j->pipeline, j->in, j->out, executor, j->ws, j->args);
}

static void cfuse_pieces(const cfuse_job* j, size_t first, size_t count) {
  if (j->plan != NULL) {
    fftn_execute_transforms_args(j->plan, first, count, j->in, j->out, j->ws, j->args);
  } else {
    fftn_pipeline_execute_transforms(j->pipeline, first, count, j->in, j->out, j->ws, j->args);
  }
}

/* Runs the job in `form`; `transforms` is the object's count of transforms. */
static int cfuse_run(cfuse_job* j, ctest_form form, size_t transforms) {
  switch (form) {
  case CTEST_SERIAL:
    if (j->plan != NULL) {
      fftn_execute_args(j->plan, j->in, j->out, j->ws, j->args);
    } else {
      fftn_pipeline_execute(j->pipeline, j->in, j->out, j->ws, j->args);
    }
    return FFTN_OK;
  case CTEST_PIECES:
    /* One transform, an empty piece, then the rest. */
    cfuse_pieces(j, 0, 1);
    cfuse_pieces(j, 1, 0);
    cfuse_pieces(j, 1, transforms - 1);
    return FFTN_OK;
  case CTEST_TEAM_ALL:
  case CTEST_TEAM_OWN:
    j->done = form == CTEST_TEAM_OWN ? FFTN_COMPLETION_OWN : FFTN_COMPLETION_ALL;
    return ctest_parallel(j->lanes, cfuse_member, j) != 0 ? FFTN_ERROR_INTERNAL : FFTN_OK;
  case CTEST_EXECUTOR:
    return ctest_with_pool(j->lanes, j->group_size, cfuse_on_pool, j);
  }
  return FFTN_ERROR_INVALID_ARGUMENT;
}

int cfuse_run_plan(const fftn_shape* shape, const fftn_options* options,
                   const fftn_plan_edges* edges, ctest_form form, int lanes, int group_size,
                   size_t transforms, const void* in, void* out, const fftn_exec_args* args,
                   fftn_error* error) {
  cfuse_job j;
  int status;
  fftn_plan* plan = fftn_plan_create_edges(shape, options, edges, error);
  if (plan == NULL) {
    return error != NULL ? error->code : FFTN_ERROR_INTERNAL;
  }
  memset(&j, 0, sizeof j);
  j.plan = plan;
  j.in = in;
  j.out = out;
  j.args = args;
  j.lanes = lanes;
  j.group_size = group_size;
  j.ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, lanes), error);
  if (j.ws == NULL) {
    fftn_plan_release(plan);
    return error != NULL ? error->code : FFTN_ERROR_INTERNAL;
  }
  status = cfuse_run(&j, form, transforms);
  fftn_workspace_destroy(j.ws);
  fftn_plan_release(plan);
  return status;
}

int cfuse_run_pipeline(const fftn_pipeline_desc* desc, const fftn_options* options, ctest_form form,
                       int lanes, int group_size, const void* in, void* out,
                       const fftn_exec_args* args, fftn_error* error) {
  cfuse_job j;
  int status;
  fftn_pipeline* pipeline = fftn_pipeline_create(desc, options, error);
  fftn_pipeline* second;
  size_t transforms;
  if (pipeline == NULL) {
    return error != NULL ? error->code : FFTN_ERROR_INTERNAL;
  }
  /* A second reference keeps the pipeline alive after the first is dropped. */
  second = fftn_pipeline_ref(pipeline);
  fftn_pipeline_release(pipeline);
  pipeline = second;
  transforms = fftn_pipeline_transform_count(pipeline);
  memset(&j, 0, sizeof j);
  j.pipeline = pipeline;
  j.in = in;
  j.out = out;
  j.args = args;
  j.lanes = lanes;
  j.group_size = group_size;
  if (form == CTEST_TEAM_ALL || form == CTEST_TEAM_OWN) {
    /* The members' own transforms cover the batch once, in rank order. */
    size_t covered = 0;
    int r;
    for (r = 0; r < lanes; ++r) {
      fftn_team team;
      fftn_transform_range own;
      team.rank = r;
      team.size = lanes;
      team.group_size = group_size;
      own = fftn_pipeline_transforms_of(pipeline, team);
      if (own.count != 0 && own.first != covered) {
        fftn_pipeline_release(pipeline);
        return FFTN_ERROR_INTERNAL;
      }
      covered += own.count;
    }
    if (covered != transforms) {
      fftn_pipeline_release(pipeline);
      return FFTN_ERROR_INTERNAL;
    }
  }
  j.ws = fftn_workspace_create(fftn_pipeline_workspace_bytes(pipeline, lanes), error);
  if (j.ws == NULL) {
    fftn_pipeline_release(pipeline);
    return error != NULL ? error->code : FFTN_ERROR_INTERNAL;
  }
  status = cfuse_run(&j, form, transforms);
  fftn_workspace_destroy(j.ws);
  fftn_pipeline_release(pipeline);
  return status;
}

/* ---- Refusals and defaults ---- */

static int cfuse_set_int(void* field, int value) {
  memcpy(field, &value, sizeof value);
  return value;
}

/* Whether building a plan of `s` with `edges` fails with FFTN_ERROR_INVALID_ARGUMENT, NULL
 * returned and `error` filled with a message that contains `names`. */
static int cfuse_plan_refused(const fftn_shape* s, const fftn_plan_edges* edges,
                              const char* names) {
  fftn_error e;
  fftn_plan* p;
  e.code = -1;
  e.message[0] = '\0';
  p = fftn_plan_create_edges(s, NULL, edges, &e);
  if (p != NULL) {
    fftn_plan_release(p);
    return 0;
  }
  return e.code == FFTN_ERROR_INVALID_ARGUMENT && strstr(e.message, names) != NULL;
}

static int cfuse_pipeline_refused(const fftn_pipeline_desc* d, const char* names) {
  fftn_error e;
  fftn_pipeline* p;
  e.code = -1;
  e.message[0] = '\0';
  p = fftn_pipeline_create(d, NULL, &e);
  if (p != NULL) {
    fftn_pipeline_release(p);
    return 0;
  }
  return e.code == FFTN_ERROR_INVALID_ARGUMENT && strstr(e.message, names) != NULL;
}

static void cfuse_nothing(void* user, const fftn_block_info* info, void* data) {
  (void)user;
  (void)info;
  (void)data;
}

int cfuse_refusals(void) {
  const fftn_axis axis = {16, 0, 0};
  const fftn_batch batch = {4, 0, 0};
  const ptrdiff_t one = 1;
  fftn_shape shape;
  fftn_plan_edges good;
  fftn_plan_edges edges;
  fftn_pipeline_desc d;
  fftn_pipeline_desc dg;
  fftn_edge e = fftn_edge_default();
  fftn_error error;
  fftn_plan* plan;
  fftn_pipeline* pipeline;
  fftn_workspace* ws;
  fftn_executor nothing;

  /* The defaults: fftn::edge{} and fftn::pipeline_desc{}. */
  if (e.scale != 1.0 || e.accumulate != 0 || e.mul.kind != FFTN_FACTOR_NONE ||
      e.mul.axis_strides != NULL || e.call.fn != NULL || e.call.mode != FFTN_CALLBACK_MODIFY ||
      e.call.scratch_bytes != 0) {
    return 1;
  }
  dg = fftn_pipeline_desc_default();
  if (dg.kind != FFTN_PIPELINE_CCC || dg.precision != FFTN_DOUBLE || dg.first != FFTN_FORWARD ||
      dg.fanout != 1 || dg.middle.scale != 1.0 || dg.prologue.scale != 1.0 ||
      dg.epilogue.scale != 1.0 || dg.line.n != 0 || dg.batches != NULL || dg.runs != NULL ||
      dg.middle.mul.kind != FFTN_FACTOR_NONE || dg.middle.call.fn != NULL ||
      dg.group != FFTN_NO_GROUP) {
    return 2;
  }

  memset(&shape, 0, sizeof shape);
  shape.kind = FFTN_C2C;
  shape.precision = FFTN_DOUBLE;
  shape.direction = FFTN_FORWARD;
  shape.rank = 1;
  shape.axes = &axis;
  shape.batch_rank = 1;
  shape.batches = &batch;
  good.in = fftn_edge_default();
  good.out = fftn_edge_default();

  /* Default edges build, and so do NULL edges; the error is cleared. */
  error.code = -1;
  plan = fftn_plan_create_edges(&shape, NULL, &good, &error);
  if (plan == NULL || error.code != FFTN_OK || error.message[0] != '\0') {
    return 3;
  }
  fftn_plan_release(plan);
  plan = fftn_plan_create_edges(&shape, NULL, NULL, &error);
  if (plan == NULL) {
    return 4;
  }
  fftn_plan_release(plan);

  /* Enumerations out of range, refused in C. */
  edges = good;
  cfuse_set_int(&edges.in.mul.kind, 3);
  if (!cfuse_plan_refused(&shape, &edges, "edges.in.mul.kind")) {
    return 5;
  }
  edges = good;
  edges.out.call.fn = cfuse_nothing;
  cfuse_set_int(&edges.out.call.mode, -1);
  if (!cfuse_plan_refused(&shape, &edges, "edges.out.call.mode")) {
    return 6;
  }
  /* NULL strides with a count. */
  edges = good;
  edges.in.mul.kind = FFTN_FACTOR_REAL;
  edges.in.mul.axis_stride_count = 1;
  if (!cfuse_plan_refused(&shape, &edges, "edges.in.mul.axis_strides")) {
    return 7;
  }
  edges = good;
  edges.out.mul.kind = FFTN_FACTOR_COMPLEX;
  edges.out.mul.axis_stride_count = 1;
  edges.out.mul.axis_strides = &one;
  edges.out.mul.batch_stride_count = 1;
  if (!cfuse_plan_refused(&shape, &edges, "edges.out.mul.batch_strides")) {
    return 8;
  }
  /* Refusals of the C++ constructor come through with its message. */
  edges = good;
  edges.in.accumulate = 1;
  if (!cfuse_plan_refused(&shape, &edges, "accumulate")) {
    return 9;
  }
  edges = good;
  edges.out.call.mode = FFTN_CALLBACK_CONSUME; /* a mode without a function */
  if (!cfuse_plan_refused(&shape, &edges, "mode")) {
    return 10;
  }
  edges = good;
  edges.in.call.fn = cfuse_nothing;
  edges.in.call.mode = FFTN_CALLBACK_CONSUME; /* consume on an input */
  if (!cfuse_plan_refused(&shape, &edges, "mode")) {
    return 11;
  }
  /* The shape is still checked, under this function's name. */
  if (!cfuse_plan_refused(NULL, &good, "fftn_plan_create_edges: shape")) {
    return 12;
  }

  /* Pipelines. */
  if (!cfuse_pipeline_refused(NULL, "desc")) {
    return 13;
  }
  memset(&d, 0, sizeof d); /* a zeroed description: line.n and fanout 0 */
  if (!cfuse_pipeline_refused(&d, "desc")) {
    return 14;
  }
  d = dg;
  d.line.n = 16;
  error.code = -1;
  pipeline = fftn_pipeline_create(&d, NULL, &error);
  if (pipeline == NULL || error.code != FFTN_OK || fftn_pipeline_transform_count(pipeline) != 1) {
    return 15;
  }
  ws = fftn_workspace_create(fftn_pipeline_workspace_bytes(pipeline, 1), &error);
  if (ws == NULL) {
    fftn_pipeline_release(pipeline);
    return 16;
  }
  /* An executor without functions is refused before anything runs. */
  memset(&nothing, 0, sizeof nothing);
  if (fftn_pipeline_execute_on(pipeline, NULL, NULL, &nothing, ws, NULL) !=
          FFTN_ERROR_INVALID_ARGUMENT ||
      fftn_pipeline_execute_on(pipeline, NULL, NULL, NULL, ws, NULL) !=
          FFTN_ERROR_INVALID_ARGUMENT) {
    fftn_workspace_destroy(ws);
    fftn_pipeline_release(pipeline);
    return 17;
  }
  fftn_workspace_destroy(ws);
  if (fftn_pipeline_ref(NULL) != NULL) {
    fftn_pipeline_release(pipeline);
    return 18;
  }
  fftn_pipeline_release(NULL);
  fftn_pipeline_release(pipeline);
  d = dg;
  d.line.n = 16;
  cfuse_set_int(&d.kind, 9);
  if (!cfuse_pipeline_refused(&d, "desc.kind")) {
    return 19;
  }
  d = dg;
  d.line.n = 16;
  d.first = 0;
  if (!cfuse_pipeline_refused(&d, "desc.first")) {
    return 20;
  }
  /* The real kinds do not read first. */
  d.kind = FFTN_PIPELINE_TCT;
  pipeline = fftn_pipeline_create(&d, NULL, &error);
  if (pipeline == NULL) {
    return 21;
  }
  fftn_pipeline_release(pipeline);
  d = dg;
  d.line.n = 16;
  cfuse_set_int(&d.precision, 2);
  if (!cfuse_pipeline_refused(&d, "desc.precision")) {
    return 22;
  }
  d = dg;
  d.line.n = 16;
  d.batch_rank = 1;
  if (!cfuse_pipeline_refused(&d, "desc.batches")) {
    return 23;
  }
  d = dg;
  d.line.n = 16;
  d.run_count = 1;
  if (!cfuse_pipeline_refused(&d, "desc.runs")) {
    return 24;
  }
  d = dg;
  d.line.n = 16;
  cfuse_set_int(&d.middle.mul.kind, 7);
  if (!cfuse_pipeline_refused(&d, "desc.middle.mul.kind")) {
    return 25;
  }
  d = dg;
  d.line.n = 16;
  d.middle.call.fn = cfuse_nothing;
  d.middle.call.mode = FFTN_CALLBACK_PRODUCE; /* the C++ constructor's refusal */
  if (!cfuse_pipeline_refused(&d, "desc.middle.call.mode")) {
    return 26;
  }
  d = dg;
  d.line.n = 16;
  d.prologue.accumulate = 1;
  if (!cfuse_pipeline_refused(&d, "accumulate")) {
    return 27;
  }
  /* A group needs a batch dimension that exists and a middle callback (the C++ refusals). */
  d = dg;
  d.line.n = 16;
  d.batch_rank = 1;
  d.batches = &batch;
  d.group = 0;
  if (!cfuse_pipeline_refused(&d, "desc.group")) {
    return 28;
  }
  d.middle.call.fn = cfuse_nothing;
  d.group = 1;
  if (!cfuse_pipeline_refused(&d, "desc.group")) {
    return 29;
  }
  d.group = 0;
  pipeline = fftn_pipeline_create(&d, NULL, &error);
  if (pipeline == NULL || fftn_pipeline_transform_count(pipeline) != 1) {
    fftn_pipeline_release(pipeline);
    return 30;
  }
  fftn_pipeline_release(pipeline);
  return 0;
}
