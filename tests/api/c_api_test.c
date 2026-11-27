/* SPDX-FileCopyrightText: 2026 John C. Travers and contributors
 * SPDX-License-Identifier: Apache-2.0 */

/* The C half of the C API's tests, compiled as C99 with the strict warnings (tests/api/
 * CMakeLists.txt): plans built, executed in every form and released from C, a C thread pool
 * driven through the executor contract, wisdom and the refusals. On POSIX systems team members
 * and the pool's workers are threads; elsewhere (Windows, which has no pthreads in C99) they
 * run one after another on the calling thread, in reverse order, which the contracts allow. */
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200112L
#endif

#include "c_api_test.h"

#include "fftn/c_api.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(_WIN32)
#define CTEST_THREADS 1
#include <pthread.h>
#else
#define CTEST_THREADS 0
#endif

enum { CTEST_MAX_LANES = 8 };

/* ---- A thread pool written in C, seen by FFTN only through fftn_executor ---- */

typedef struct ctest_pool {
  int lanes;
#if CTEST_THREADS
  pthread_t workers[CTEST_MAX_LANES];
  int started;
  pthread_mutex_t lock;
  pthread_cond_t wake;
  pthread_cond_t finished_all;
  fftn_task task;
  void* arg;
  int n;
  int next;
  int finished;
  unsigned long job;
  int stop;
#endif
} ctest_pool;

#if CTEST_THREADS
/* Runs tasks of the current job until none is left; called with the lock held, returns with
 * it held. */
static void ctest_pool_claim(ctest_pool* p) {
  while (p->next < p->n) {
    const int i = p->next++;
    fftn_task task = p->task;
    void* arg = p->arg;
    pthread_mutex_unlock(&p->lock);
    task(arg, i);
    pthread_mutex_lock(&p->lock);
    if (++p->finished == p->n) {
      pthread_cond_broadcast(&p->finished_all);
    }
  }
}

static void* ctest_pool_worker(void* self) {
  ctest_pool* p = (ctest_pool*)self;
  unsigned long seen = 0;
  pthread_mutex_lock(&p->lock);
  for (;;) {
    while (!p->stop && p->job == seen) {
      pthread_cond_wait(&p->wake, &p->lock);
    }
    if (p->stop) {
      break;
    }
    seen = p->job;
    ctest_pool_claim(p);
  }
  pthread_mutex_unlock(&p->lock);
  return NULL;
}

static int ctest_pool_inside(const ctest_pool* p) {
  int i;
  for (i = 0; i < p->started; ++i) {
    if (pthread_equal(pthread_self(), p->workers[i])) {
      return 1;
    }
  }
  return 0;
}
#endif

/* fftn_executor's concurrency: the lanes, the caller among them; 1 inside a task. */
static int ctest_pool_concurrency(void* self) {
  const ctest_pool* p = (const ctest_pool*)self;
#if CTEST_THREADS
  if (ctest_pool_inside(p)) {
    return 1;
  }
#endif
  return p->lanes;
}

/* fftn_executor's bulk: the workers and the caller claim the tasks; returns when all have
 * returned (the lock orders each task before the return). */
static void ctest_pool_bulk(void* self, int n, fftn_task task, void* arg) {
  ctest_pool* p = (ctest_pool*)self;
#if CTEST_THREADS
  pthread_mutex_lock(&p->lock);
  p->task = task;
  p->arg = arg;
  p->n = n;
  p->next = 0;
  p->finished = 0;
  ++p->job;
  pthread_cond_broadcast(&p->wake);
  ctest_pool_claim(p);
  while (p->finished < p->n) {
    pthread_cond_wait(&p->finished_all, &p->lock);
  }
  pthread_mutex_unlock(&p->lock);
#else
  int i;
  (void)p;
  for (i = n; i > 0; --i) {
    task(arg, i - 1);
  }
#endif
}

static int ctest_pool_group_size(void* self) {
  (void)self;
  return 2;
}

static int ctest_pool_start(ctest_pool* p, int lanes) {
  memset(p, 0, sizeof *p);
  p->lanes = lanes;
#if CTEST_THREADS
  pthread_mutex_init(&p->lock, NULL);
  pthread_cond_init(&p->wake, NULL);
  pthread_cond_init(&p->finished_all, NULL);
  for (; p->started < lanes - 1; ++p->started) {
    if (pthread_create(&p->workers[p->started], NULL, ctest_pool_worker, p) != 0) {
      return 1;
    }
  }
#endif
  return 0;
}

static void ctest_pool_stop(ctest_pool* p) {
#if CTEST_THREADS
  int i;
  pthread_mutex_lock(&p->lock);
  p->stop = 1;
  pthread_cond_broadcast(&p->wake);
  pthread_mutex_unlock(&p->lock);
  for (i = 0; i < p->started; ++i) {
    pthread_join(p->workers[i], NULL);
  }
  pthread_cond_destroy(&p->finished_all);
  pthread_cond_destroy(&p->wake);
  pthread_mutex_destroy(&p->lock);
#else
  (void)p;
#endif
}

/* ---- Team members ---- */

typedef struct ctest_member {
  const fftn_plan* plan;
  const void* in;
  void* out;
  fftn_workspace* ws;
  fftn_team team;
  fftn_completion done;
} ctest_member;

static void ctest_member_run(const ctest_member* m) {
  fftn_execute_team(m->plan, m->in, m->out, m->team, m->ws, m->done);
}

#if CTEST_THREADS
static void* ctest_member_thread(void* member) {
  ctest_member_run((const ctest_member*)member);
  return NULL;
}
#endif

/* Every member of a team of `lanes`; returns nonzero when a thread could not start or the
 * members' own transforms do not cover the batch in rank order. */
static int ctest_team(const fftn_plan* plan, const void* in, void* out, int lanes, int group_size,
                      fftn_completion done, fftn_workspace* ws, size_t transforms) {
  ctest_member members[CTEST_MAX_LANES];
  size_t covered = 0;
  int r;
  for (r = 0; r < lanes; ++r) {
    ctest_member m;
    m.plan = plan;
    m.in = in;
    m.out = out;
    m.ws = ws;
    m.team.rank = r;
    m.team.size = lanes;
    m.team.group_size = group_size;
    m.done = done;
    members[r] = m;
    {
      const fftn_transform_range own = fftn_transforms_of(plan, m.team);
      if (own.count != 0 && own.first != covered) {
        return 1;
      }
      covered += own.count;
    }
  }
  if (covered != transforms) {
    return 2;
  }
#if CTEST_THREADS
  {
    pthread_t threads[CTEST_MAX_LANES];
    int started = 0;
    int failed = 0;
    for (r = 1; r < lanes; ++r) {
      if (pthread_create(&threads[started], NULL, ctest_member_thread, &members[r]) != 0) {
        failed = 1;
        break;
      }
      ++started;
    }
    if (!failed) {
      ctest_member_run(&members[0]);
    }
    for (r = 0; r < started; ++r) {
      pthread_join(threads[r], NULL);
    }
    return failed ? 3 : 0;
  }
#else
  for (r = lanes; r > 0; --r) {
    ctest_member_run(&members[r - 1]);
  }
  return 0;
#endif
}

/* ---- Threads and pools for the fusion tests (c_fusion_test.c) ---- */

typedef struct ctest_rank {
  void (*fn)(void* arg, int rank);
  void* arg;
  int rank;
} ctest_rank;

#if CTEST_THREADS
static void* ctest_rank_thread(void* r) {
  const ctest_rank* rank = (const ctest_rank*)r;
  rank->fn(rank->arg, rank->rank);
  return NULL;
}
#endif

int ctest_parallel(int lanes, void (*fn)(void* arg, int rank), void* arg) {
  ctest_rank ranks[CTEST_MAX_LANES];
  int r;
  if (lanes < 1 || lanes > CTEST_MAX_LANES) {
    return 1;
  }
  for (r = 0; r < lanes; ++r) {
    ranks[r].fn = fn;
    ranks[r].arg = arg;
    ranks[r].rank = r;
  }
#if CTEST_THREADS
  {
    pthread_t threads[CTEST_MAX_LANES];
    int started = 0;
    int failed = 0;
    for (r = 1; r < lanes; ++r) {
      if (pthread_create(&threads[started], NULL, ctest_rank_thread, &ranks[r]) != 0) {
        failed = 1;
        break;
      }
      ++started;
    }
    if (!failed) {
      fn(arg, 0);
    }
    for (r = 0; r < started; ++r) {
      pthread_join(threads[r], NULL);
    }
    return failed;
  }
#else
  for (r = lanes; r > 0; --r) {
    fn(arg, r - 1);
  }
  return 0;
#endif
}

int ctest_with_pool(int lanes, int group_size, int (*fn)(void* arg, const fftn_executor* executor),
                    void* arg) {
  ctest_pool pool;
  fftn_executor executor;
  int status;
  if (lanes < 1 || lanes > CTEST_MAX_LANES) {
    return FFTN_ERROR_INVALID_ARGUMENT;
  }
  if (ctest_pool_start(&pool, lanes) != 0) {
    status = FFTN_ERROR_INTERNAL;
  } else {
    executor.self = &pool;
    executor.concurrency = ctest_pool_concurrency;
    executor.bulk = ctest_pool_bulk;
    executor.group_size = group_size != 0 ? ctest_pool_group_size : NULL;
    status = fn(arg, &executor);
  }
  ctest_pool_stop(&pool);
  return status;
}

/* ---- Plans of a case ---- */

static fftn_shape ctest_shape(const ctest_case* c) {
  fftn_shape s;
  memset(&s, 0, sizeof s);
  s.kind = c->kind;
  s.precision = c->precision;
  s.direction = c->direction;
  s.rank = c->rank;
  s.axes = c->axes;
  s.batch_rank = 1;
  s.batches = &c->batch;
  s.r2r_kinds = c->kind == FFTN_R2R ? c->r2r : NULL;
  s.run_count = c->run_count;
  s.runs = c->run_count != 0 ? c->runs : NULL;
  return s;
}

int ctest_run(const ctest_case* c, ctest_form form, int lanes, int group_size, const void* in,
              void* out, fftn_error* error) {
  const fftn_shape shape = ctest_shape(c);
  fftn_options options = fftn_options_default();
  fftn_plan* plan;
  fftn_plan* second;
  fftn_workspace* ws;
  int status = FFTN_OK;
  if (lanes < 1 || lanes > CTEST_MAX_LANES) {
    return FFTN_ERROR_INVALID_ARGUMENT;
  }
  options.in_place = c->in_place;
  options.scale = c->scale;
  plan = fftn_plan_create(&shape, &options, error);
  if (plan == NULL) {
    return error != NULL ? error->code : FFTN_ERROR_INTERNAL;
  }
  /* A second reference keeps the plan alive after the first is dropped. */
  second = fftn_plan_ref(plan);
  fftn_plan_release(plan);
  plan = second;
  ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, lanes), error);
  if (ws == NULL) {
    fftn_plan_release(plan);
    return error != NULL ? error->code : FFTN_ERROR_INTERNAL;
  }
  switch (form) {
  case CTEST_SERIAL:
    fftn_execute(plan, in, out, ws);
    break;
  case CTEST_PIECES: {
    /* The batch in pieces of one, then the rest, and an empty piece that changes nothing. */
    const size_t count = c->batch.count;
    fftn_execute_transforms(plan, 0, 0, in, out, ws);
    fftn_execute_transforms(plan, 0, 1, in, out, ws);
    fftn_execute_transforms(plan, 1, count - 1, in, out, ws);
    break;
  }
  case CTEST_TEAM_ALL:
  case CTEST_TEAM_OWN:
    if (ctest_team(plan, in, out, lanes, group_size,
                   form == CTEST_TEAM_OWN ? FFTN_COMPLETION_OWN : FFTN_COMPLETION_ALL, ws,
                   c->batch.count) != 0) {
      status = FFTN_ERROR_INTERNAL;
    }
    break;
  case CTEST_EXECUTOR: {
    ctest_pool pool;
    fftn_executor executor;
    if (ctest_pool_start(&pool, lanes) != 0) {
      status = FFTN_ERROR_INTERNAL;
    } else {
      executor.self = &pool;
      executor.concurrency = ctest_pool_concurrency;
      executor.bulk = ctest_pool_bulk;
      executor.group_size = group_size != 0 ? ctest_pool_group_size : NULL;
      status = fftn_execute_on(plan, in, out, &executor, ws);
    }
    ctest_pool_stop(&pool);
    break;
  }
  default:
    status = FFTN_ERROR_INVALID_ARGUMENT;
  }
  fftn_workspace_destroy(ws);
  fftn_plan_release(plan);
  return status;
}

/* ---- Refusals ---- */

/* Whether creating a plan of `s` with `o` fails with `code`, NULL returned and `error` filled
 * with a message. */
static int ctest_refused(const fftn_shape* s, const fftn_options* o, int code) {
  fftn_error e;
  fftn_plan* p;
  e.code = -1;
  e.message[0] = '\0';
  p = fftn_plan_create(s, o, &e);
  if (p != NULL) {
    fftn_plan_release(p);
    return 0;
  }
  return e.code == code && e.message[0] != '\0' && strlen(e.message) < sizeof e.message;
}

static int ctest_set_int(void* field, int value) {
  memcpy(field, &value, sizeof value);
  return value;
}

int ctest_refusals(void) {
  const fftn_axis axis = {16, 0, 0};
  const fftn_batch batch = {4, 0, 0};
  const fftn_r2r_kind kinds[1] = {FFTN_DHT};
  fftn_stored_run run;
  fftn_shape good;
  fftn_shape s;
  fftn_options o = fftn_options_default();
  fftn_error e;
  fftn_plan* p;
  memset(&good, 0, sizeof good);
  good.kind = FFTN_C2C;
  good.precision = FFTN_DOUBLE;
  good.direction = FFTN_FORWARD;
  good.rank = 1;
  good.axes = &axis;
  good.batch_rank = 1;
  good.batches = &batch;

  /* A valid shape succeeds and clears the error; NULL options are the defaults. */
  e.code = -1;
  memcpy(e.message, "stale", sizeof "stale");
  p = fftn_plan_create(&good, NULL, &e);
  if (p == NULL || e.code != FFTN_OK || e.message[0] != '\0') {
    return 1;
  }
  fftn_plan_release(p);
  fftn_plan_release(NULL);
  if (fftn_plan_ref(NULL) != NULL) {
    return 2;
  }
  if (!ctest_refused(NULL, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 3;
  }
  /* Without an fftn_error the failure is still a NULL plan. */
  if (fftn_plan_create(NULL, NULL, NULL) != NULL) {
    return 4;
  }
  s = good;
  ctest_set_int(&s.kind, 7);
  if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 5;
  }
  s = good;
  ctest_set_int(&s.precision, -1);
  if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 6;
  }
  s = good;
  s.direction = 0;
  if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 7;
  }
  /* The direction of the real kinds is not read. */
  s = good;
  s.kind = FFTN_R2C;
  s.direction = 0;
  p = fftn_plan_create(&s, NULL, &e);
  if (p == NULL) {
    return 8;
  }
  fftn_plan_release(p);
  s = good;
  s.axes = NULL;
  if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 9;
  }
  s = good;
  s.batches = NULL;
  if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 10;
  }
  /* No batch dimensions: one transform. */
  s = good;
  s.batch_rank = 0;
  s.batches = NULL;
  p = fftn_plan_create(&s, NULL, &e);
  if (p == NULL) {
    return 11;
  }
  fftn_plan_release(p);
  s = good;
  s.run_count = 1;
  if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 12;
  }
  memset(&run, 0, sizeof run);
  run.count = 4;
  ctest_set_int(&run.where, 3);
  s = good;
  s.run_count = 1;
  s.runs = &run;
  if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 13;
  }
  s = good;
  s.r2r_kinds = kinds;
  if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 14;
  }
  s = good;
  s.kind = FFTN_R2R;
  if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 15;
  }
  {
    fftn_r2r_kind bad[1];
    ctest_set_int(&bad[0], 42);
    s.r2r_kinds = bad;
    if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
      return 16;
    }
  }
  o.force_isa = 99;
  if (!ctest_refused(&good, &o, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 17;
  }
  o = fftn_options_default();
  o.effort = -1;
  if (!ctest_refused(&good, &o, FFTN_ERROR_INVALID_ARGUMENT)) {
    return 18;
  }
  /* The C++ refusals come through with their messages (c_api_test.cpp compares them). */
  s = good;
  {
    const fftn_axis zero = {0, 0, 0};
    s.axes = &zero;
    if (!ctest_refused(&s, NULL, FFTN_ERROR_INVALID_ARGUMENT)) {
      return 19;
    }
  }

  /* Workspaces. */
  if (fftn_workspace_borrow(NULL, 8, &e) != NULL || e.code != FFTN_ERROR_INVALID_ARGUMENT) {
    return 20;
  }
  {
    fftn_workspace* ws = fftn_workspace_borrow(NULL, 0, &e);
    if (ws == NULL || e.code != FFTN_OK) {
      return 21;
    }
    fftn_workspace_destroy(ws);
    fftn_workspace_destroy(NULL);
  }

  /* fftn_execute_on refuses a missing executor or function and executes nothing. */
  p = fftn_plan_create(&good, NULL, NULL);
  if (p == NULL) {
    return 22;
  }
  {
    fftn_workspace* ws = fftn_workspace_create(fftn_plan_workspace_bytes(p, 4), NULL);
    fftn_executor ex;
    double data[2 * 16 * 4];
    memset(data, 0, sizeof data);
    memset(&ex, 0, sizeof ex);
    if (ws == NULL) {
      fftn_plan_release(p);
      return 23;
    }
    if (fftn_execute_on(p, data, data + 2 * 16 * 2, NULL, ws) != FFTN_ERROR_INVALID_ARGUMENT ||
        fftn_execute_on(p, data, data, &ex, ws) != FFTN_ERROR_INVALID_ARGUMENT) {
      fftn_workspace_destroy(ws);
      fftn_plan_release(p);
      return 24;
    }
    ex.concurrency = ctest_pool_concurrency;
    ex.bulk = ctest_pool_bulk;
    if (fftn_execute_on(p, data, data, &ex, NULL) != FFTN_ERROR_INVALID_ARGUMENT) {
      fftn_workspace_destroy(ws);
      fftn_plan_release(p);
      return 25;
    }
    fftn_workspace_destroy(ws);
  }
  /* describe() as snprintf: the full length whatever the buffer. */
  {
    char small[8];
    const size_t full = fftn_plan_describe(p, NULL, 0);
    if (full == 0 || fftn_plan_describe(p, small, sizeof small) != full ||
        strlen(small) != sizeof small - 1) {
      fftn_plan_release(p);
      return 26;
    }
  }
  if (fftn_plan_instruction_set(p) == FFTN_ISA_BEST) {
    fftn_plan_release(p);
    return 27;
  }
  fftn_plan_release(p);

  /* Lengths and names. */
  if (fftn_next_fast_length(1001, FFTN_LENGTH_COMPLEX, &e) != 1008 || e.code != FFTN_OK) {
    return 28;
  }
  {
    fftn_length_kind bad;
    ctest_set_int(&bad, 7);
    if (fftn_next_fast_length(16, bad, &e) != 0 || e.code != FFTN_ERROR_INVALID_ARGUMENT) {
      return 29;
    }
  }
  if (SIZE_MAX >> 41 != 0 &&
      (fftn_next_fast_length((size_t)1 << 41, FFTN_LENGTH_COMPLEX, &e) != 0 ||
       e.code != FFTN_ERROR_INVALID_ARGUMENT)) {
    return 30;
  }
  {
    fftn_precision bad;
    ctest_set_int(&bad, 9);
    if (fftn_relative_cost(16, bad, FFTN_ISA_BEST, &e) != 0.0 ||
        e.code != FFTN_ERROR_INVALID_ARGUMENT) {
      return 31;
    }
  }
  if (fftn_relative_cost(16, FFTN_DOUBLE, 42, &e) != 0.0 || e.code != FFTN_ERROR_INVALID_ARGUMENT) {
    return 32;
  }
  if (fftn_relative_cost(4096, FFTN_FLOAT, FFTN_ISA_SCALAR, &e) != 1.0 || e.code != FFTN_OK) {
    return 33;
  }
  if (strcmp(fftn_isa_name(FFTN_ISA_AVX2), "avx2") != 0 ||
      strcmp(fftn_isa_name(FFTN_ISA_BEST), "best") != 0 ||
      strcmp(fftn_isa_name(42), "unknown") != 0 || strcmp(fftn_isa_name(-1), "unknown") != 0) {
    return 34;
  }
  if (fftn_version() == NULL || fftn_version()[0] == '\0') {
    return 35;
  }
  return 0;
}

/* ---- Wisdom ---- */

static fftn_plan* ctest_rows_plan(const fftn_wisdom* use) {
  const fftn_axis axis = {64, 0, 0};
  const fftn_batch batch = {4, 0, 0};
  fftn_shape s;
  fftn_options o = fftn_options_default();
  memset(&s, 0, sizeof s);
  s.kind = FFTN_C2C;
  s.precision = FFTN_DOUBLE;
  s.direction = FFTN_FORWARD;
  s.rank = 1;
  s.axes = &axis;
  s.batch_rank = 1;
  s.batches = &batch;
  o.use = use;
  return fftn_plan_create(&s, &o, NULL);
}

int ctest_wisdom(const char* path, char* text, size_t size) {
  /* Every handle is released at `done`, whichever check fails (the destroys take NULL). */
  fftn_error e;
  fftn_wisdom* w = fftn_wisdom_create(&e);
  fftn_wisdom* again = NULL;
  fftn_wisdom* loaded = NULL;
  fftn_plan* p = NULL;
  fftn_plan* q = NULL;
  char buffer[4096];
  size_t length;
  int failed = 0;
  if (w == NULL || e.code != FFTN_OK || fftn_wisdom_size(w) != 0) {
    failed = 1;
    goto done;
  }
  p = ctest_rows_plan(NULL);
  if (p == NULL || fftn_wisdom_add(w, p, &e) != FFTN_OK || fftn_wisdom_size(w) != 1) {
    failed = 2;
    goto done;
  }
  /* Adding the same plan again replaces its entry. */
  if (fftn_wisdom_add(w, p, NULL) != FFTN_OK || fftn_wisdom_size(w) != 1) {
    failed = 3;
    goto done;
  }
  length = fftn_wisdom_text(w, NULL, 0);
  if (length == 0 || length >= sizeof buffer ||
      fftn_wisdom_text(w, buffer, sizeof buffer) != length || strlen(buffer) != length) {
    failed = 4;
    goto done;
  }
  fftn_wisdom_text(w, text, size);
  again = fftn_wisdom_from_text(buffer, &e);
  if (again == NULL || fftn_wisdom_size(again) != 1) {
    failed = 5;
    goto done;
  }
  if (fftn_wisdom_save(again, path, &e) != FFTN_OK || e.code != FFTN_OK) {
    failed = 6;
    goto done;
  }
  loaded = fftn_wisdom_load(path, &e);
  if (loaded == NULL || fftn_wisdom_size(loaded) != 1) {
    failed = 7;
    goto done;
  }
  /* A plan from the loaded wisdom transforms as the recorded one did. */
  q = ctest_rows_plan(loaded);
  if (q == NULL) {
    failed = 8;
  } else {
    double in[2 * 64 * 4];
    double out_p[2 * 64 * 4];
    double out_q[2 * 64 * 4];
    fftn_workspace* ws =
        fftn_workspace_create(fftn_plan_workspace_bytes(p, 1) > fftn_plan_workspace_bytes(q, 1)
                                  ? fftn_plan_workspace_bytes(p, 1)
                                  : fftn_plan_workspace_bytes(q, 1),
                              NULL);
    size_t i;
    for (i = 0; i < sizeof in / sizeof in[0]; ++i) {
      in[i] = (double)(i % 13) - 6.0;
    }
    if (ws == NULL) {
      failed = 9;
    } else {
      fftn_execute(p, in, out_p, ws);
      fftn_execute(q, in, out_q, ws);
      if (memcmp(out_p, out_q, sizeof out_p) != 0) {
        failed = 10;
      }
      fftn_workspace_destroy(ws);
    }
  }
  /* Refusals: a line that is not an entry, a NULL text or path, a file that does not exist,
   * and a file that cannot be written. */
  if (!failed && (fftn_wisdom_from_text("not wisdom\n", &e) != NULL ||
                  e.code != FFTN_ERROR_INVALID_ARGUMENT)) {
    failed = 11;
  }
  if (!failed &&
      (fftn_wisdom_from_text(NULL, &e) != NULL || e.code != FFTN_ERROR_INVALID_ARGUMENT)) {
    failed = 12;
  }
  if (!failed && (fftn_wisdom_load("/nonexistent-fftn-dir/none.wisdom", &e) != NULL ||
                  e.code != FFTN_ERROR_FILE)) {
    failed = 13;
  }
  if (!failed && (fftn_wisdom_load(NULL, &e) != NULL || e.code != FFTN_ERROR_INVALID_ARGUMENT)) {
    failed = 14;
  }
  if (!failed && (fftn_wisdom_save(w, "/nonexistent-fftn-dir/none.wisdom", &e) != FFTN_ERROR_FILE ||
                  e.code != FFTN_ERROR_FILE)) {
    failed = 15;
  }
  if (!failed && (fftn_wisdom_save(w, NULL, &e) != FFTN_ERROR_INVALID_ARGUMENT)) {
    failed = 16;
  }
done:
  fftn_plan_release(q);
  fftn_plan_release(p);
  fftn_wisdom_destroy(loaded);
  fftn_wisdom_destroy(again);
  fftn_wisdom_destroy(w);
  fftn_wisdom_destroy(NULL);
  return failed;
}

/* ---- Lengths ---- */

size_t ctest_next_fast_length(size_t n, fftn_length_kind kind, fftn_error* error) {
  return fftn_next_fast_length(n, kind, error);
}

double ctest_relative_cost(size_t n, fftn_precision precision, fftn_error* error) {
  return fftn_relative_cost(n, precision, FFTN_ISA_BEST, error);
}
