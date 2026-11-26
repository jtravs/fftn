// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The C API for fusion (fftn/c_api.h, step 6.9), quoted in docs/c_api.md: a plan with a window
// on its input edge and accumulation on its output edge, and a pipeline that convolves real rows
// with a kernel, a C callback on its epilogue recording each member's largest value. A C99
// program; exits with 0 when the convolution agrees with a direct one and the edges with a loop.

#include <fftn/c_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 64    // points per row
#define COUNT 4 // rows
#define TAPS 5  // the kernel's taps

// snippet-begin: callback
// What the epilogue's callback is given: the largest magnitude each member has seen.
typedef struct peaks {
  double largest[8]; // one slot per member: members run blocks at the same time
} peaks;

// A block callback in C: it reads the block where the plan holds it, between the pipeline's
// last pass and the store, and writes only its member's slot of the application's memory.
static void record_peak(void* user, const fftn_block_info* info, void* data) {
  peaks* p = (peaks*)user;
  const double* x = (const double*)data; // the epilogue of real rows: doubles
  size_t k;
  size_t i;
  for (k = 0; k < info->lines; ++k) {
    for (i = 0; i < info->length; ++i) {
      const double v =
          fabs(x[(ptrdiff_t)i * info->element_stride + (ptrdiff_t)k * info->line_stride]);
      if (v > p->largest[info->member]) {
        p->largest[info->member] = v;
      }
    }
  }
}
// snippet-end: callback

// The rows' input and the kernel, zero beyond its taps.
static void fill(double* rows, double* kernel) {
  size_t j;
  for (j = 0; j < (size_t)N * COUNT; ++j) {
    rows[j] = sin(0.1 * (double)j) + 0.25 * cos(0.37 * (double)j);
  }
  for (j = 0; j < N; ++j) {
    kernel[j] = j < TAPS ? 1.0 / TAPS : 0.0; // a moving average
  }
}

static int convolve(void) {
  static double rows[N * COUNT];
  static double out[N * COUNT];
  static double kernel[N];
  static double spectrum[2 * (N / 2 + 1)]; // the kernel's r2c transform: interleaved pairs
  fftn_error error;
  size_t j;
  size_t t;
  int failed = 0;
  fill(rows, kernel);

  {
    // The kernel's spectrum, by an ordinary r2c plan.
    const fftn_axis axis = {N, 0, 0};
    fftn_shape shape = {0};
    fftn_plan* r2c;
    fftn_workspace* ws;
    shape.kind = FFTN_R2C;
    shape.precision = FFTN_DOUBLE;
    shape.rank = 1;
    shape.axes = &axis;
    r2c = fftn_plan_create(&shape, NULL, &error);
    if (r2c == NULL) {
      fprintf(stderr, "fftn: %s\n", error.message);
      return 1;
    }
    ws = fftn_workspace_create(fftn_plan_workspace_bytes(r2c, 1), &error);
    fftn_execute(r2c, kernel, spectrum, ws);
    fftn_workspace_destroy(ws);
    fftn_plan_release(r2c);
  }

  {
    // snippet-begin: pipeline
    // Real rows convolved with one kernel: r2c, times the kernel's spectrum (the middle's
    // factor, one line for every row), c2r, scaled by 1/n; then the C callback on each block.
    const fftn_batch batch = {COUNT, 0, 0};
    const ptrdiff_t along_bins = 1;
    peaks seen = {{0}};
    fftn_pipeline_desc desc = fftn_pipeline_desc_default(); // not {0}: scales of 1, fan-out 1
    fftn_exec_args args = {0};
    fftn_pipeline* conv;
    fftn_workspace* ws;
    desc.kind = FFTN_PIPELINE_TCT;
    desc.line.n = N;
    desc.batch_rank = 1;
    desc.batches = &batch;
    desc.middle.scale = 1.0 / N;
    desc.middle.mul.kind = FFTN_FACTOR_COMPLEX;
    desc.middle.mul.axis_stride_count = 1;
    desc.middle.mul.axis_strides = &along_bins; // no batch strides: shared by every row
    desc.epilogue.call.fn = record_peak;        // FFTN_CALLBACK_MODIFY, the default mode

    conv = fftn_pipeline_create(&desc, NULL, &error);
    if (conv == NULL) {
      fprintf(stderr, "fftn: %s\n", error.message);
      return 1;
    }
    ws = fftn_workspace_create(fftn_pipeline_workspace_bytes(conv, 1), &error);

    // The data of this execution: the kernel's spectrum and the callback's argument.
    args.mid_factor = spectrum;
    args.out_user = &seen;
    fftn_pipeline_execute(conv, rows, out, ws, &args);
    printf("largest output value: %g\n", seen.largest[0]); // serial: member 0 only

    fftn_workspace_destroy(ws);
    fftn_pipeline_release(conv);
    // snippet-end: pipeline

    // The direct circular convolution agrees, and so does the callback's record.
    {
      double largest = 0;
      for (t = 0; t < COUNT; ++t) {
        for (j = 0; j < N; ++j) {
          double sum = 0;
          size_t m;
          for (m = 0; m < TAPS; ++m) {
            sum += kernel[m] * rows[t * N + (j + N - m) % N];
          }
          if (fabs(sum - out[t * N + j]) > 1e-12) {
            failed = 1;
          }
          if (fabs(out[t * N + j]) > largest) {
            largest = fabs(out[t * N + j]);
          }
        }
      }
      if (largest != seen.largest[0]) {
        failed = 1;
      }
    }
  }
  return failed;
}

static int windowed_sum(void) {
  static double rows[N * COUNT];
  static double kernel[N];
  static double window[N];
  static double sum[2 * (N / 2 + 1) * COUNT]; // spectra of the rows, accumulated
  size_t j;
  fill(rows, kernel);
  for (j = 0; j < N; ++j) {
    window[j] = 0.5 - 0.5 * cos(2 * 3.14159265358979323846 * (double)j / N); // Hann
  }
  {
    // snippet-begin: edges
    // Real rows times a window (the input edge's real factor, one line for every row), r2c, and
    // added to the output (the output edge's accumulation).
    const fftn_axis axis = {N, 0, 0};
    const fftn_batch batch = {COUNT, 0, 0};
    const ptrdiff_t along_row = 1;
    const ptrdiff_t across_rows = 0; // the same window for every row
    fftn_shape shape = {0};
    fftn_plan_edges edges;
    fftn_exec_args args = {0};
    fftn_error error;
    fftn_plan* plan;
    fftn_workspace* ws;
    shape.kind = FFTN_R2C;
    shape.precision = FFTN_DOUBLE;
    shape.rank = 1;
    shape.axes = &axis;
    shape.batch_rank = 1;
    shape.batches = &batch;
    edges.in = fftn_edge_default(); // not {0}: a scale of 1
    edges.in.mul.kind = FFTN_FACTOR_REAL;
    edges.in.mul.axis_stride_count = 1;
    edges.in.mul.axis_strides = &along_row;
    edges.in.mul.batch_stride_count = 1;
    edges.in.mul.batch_strides = &across_rows;
    edges.out = fftn_edge_default();
    edges.out.accumulate = 1;

    plan = fftn_plan_create_edges(&shape, NULL, &edges, &error);
    if (plan == NULL) {
      fprintf(stderr, "fftn: %s\n", error.message);
      return 1;
    }
    ws = fftn_workspace_create(fftn_plan_workspace_bytes(plan, 1), &error);
    args.in_factor = window; // the window's address comes with each execution
    fftn_execute_args(plan, rows, sum, ws, &args);
    fftn_execute_args(plan, rows, sum, ws, &args); // sum holds twice the windowed spectra
    fftn_workspace_destroy(ws);
    fftn_plan_release(plan);
    // snippet-end: edges
  }
  {
    // Bin 0 of each row is twice the windowed row's sum.
    size_t t;
    for (t = 0; t < COUNT; ++t) {
      double s = 0;
      for (j = 0; j < N; ++j) {
        s += window[j] * rows[t * N + j];
      }
      if (fabs(sum[2 * (N / 2 + 1) * t] - 2 * s) > 1e-12 * N) {
        return 1;
      }
    }
  }
  return 0;
}

int main(void) { return convolve() != 0 || windowed_sum() != 0 ? EXIT_FAILURE : EXIT_SUCCESS; }
