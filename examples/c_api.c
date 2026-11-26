// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The C API (fftn/c_api.h), quoted in docs/c_api.md: a batch of complex rows forward out of
// place and back in place, scaled by 1/n, and a refused shape. A C99 program; exits with 0 when
// the round trip gives the input back and the refusal names the offending field.

#include <fftn/c_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 12    // points per row
#define COUNT 3 // rows

static int round_trip(void) {
  // snippet-begin: plans
  // Three rows of 12 complex values, one after another: interleaved real and imaginary parts.
  const fftn_axis axis = {N, 0, 0}; // n, then input and output strides (0: tight)
  const fftn_batch batch = {COUNT, 0, 0};
  fftn_shape shape = {0};
  shape.kind = FFTN_C2C;
  shape.precision = FFTN_DOUBLE;
  shape.direction = FFTN_FORWARD;
  shape.rank = 1;
  shape.axes = &axis;
  shape.batch_rank = 1;
  shape.batches = &batch;

  fftn_error error;
  fftn_plan* forward = fftn_plan_create(&shape, NULL, &error); // NULL: the default options
  if (forward == NULL) {
    fprintf(stderr, "fftn: %s\n", error.message);
    return 1;
  }

  // Backward, in place, scaled by 1/n so that it inverts the forward transform.
  fftn_options options = fftn_options_default();
  options.in_place = 1;
  options.scale = 1.0 / N;
  shape.direction = FFTN_BACKWARD;
  fftn_plan* backward = fftn_plan_create(&shape, &options, &error);
  if (backward == NULL) {
    fprintf(stderr, "fftn: %s\n", error.message);
    fftn_plan_release(forward);
    return 1;
  }

  // One workspace serves both plans, one execution at a time.
  size_t bytes = fftn_plan_workspace_bytes(forward, 1);
  if (fftn_plan_workspace_bytes(backward, 1) > bytes) {
    bytes = fftn_plan_workspace_bytes(backward, 1);
  }
  fftn_workspace* ws = fftn_workspace_create(bytes, &error);
  // snippet-end: plans
  if (ws == NULL) {
    fprintf(stderr, "fftn: %s\n", error.message);
    fftn_plan_release(backward);
    fftn_plan_release(forward);
    return 1;
  }

  double x[2 * N * COUNT];
  double y[2 * N * COUNT];
  for (int i = 0; i < 2 * N * COUNT; ++i) {
    x[i] = sin(0.37 * i);
  }

  // snippet-begin: execute
  fftn_execute(forward, x, y, ws);  // y = the spectra of the rows of x
  fftn_execute(backward, y, y, ws); // in place: y = x again, up to rounding

  char text[512];
  fftn_plan_describe(forward, text, sizeof text); // how the plan computes, for people
  printf("%s\n", text);

  fftn_workspace_destroy(ws);
  fftn_plan_release(backward);
  fftn_plan_release(forward);
  // snippet-end: execute

  double largest = 0;
  for (int i = 0; i < 2 * N * COUNT; ++i) {
    largest = fmax(largest, fabs(y[i] - x[i]));
  }
  printf("largest round-trip error: %.3g\n", largest);
  return largest < 1e-14 ? 0 : 1;
}

static int refusal(void) {
  // snippet-begin: error
  const fftn_axis empty = {0, 0, 0}; // a length of 0, which plans refuse
  fftn_shape shape = {0};
  shape.kind = FFTN_R2C;
  shape.precision = FFTN_FLOAT;
  shape.rank = 1;
  shape.axes = &empty;

  fftn_error error;
  if (fftn_plan_create(&shape, NULL, &error) == NULL) {
    // error.code is FFTN_ERROR_INVALID_ARGUMENT; the message names shape.axes[0].n.
    printf("refused (%d): %s\n", error.code, error.message);
  }
  // snippet-end: error
  return error.code == FFTN_ERROR_INVALID_ARGUMENT && strstr(error.message, "axes[0]") != NULL ? 0
                                                                                               : 1;
}

int main(void) { return round_trip() != 0 || refusal() != 0 ? EXIT_FAILURE : EXIT_SUCCESS; }
