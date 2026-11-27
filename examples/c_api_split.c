// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// Split complex in the C API (fftn/c_api.h, "Split complex"; provisional), quoted in
// docs/c_api.md: real rows into split spectra (one array of real parts, one of imaginary parts)
// and back, scaled by 1/n. A C99 program; exits with 0 when the round trip gives the rows back.

#include <fftn/c_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16          // points per row
#define H (N / 2 + 1) // bins per row
#define COUNT 4       // rows

int main(void) {
  double rows[N * COUNT];
  double back[N * COUNT];
  double re[H * COUNT]; // the spectra's real parts
  double im[H * COUNT]; // and their imaginary parts, in an array of their own
  int i;
  for (i = 0; i < N * COUNT; ++i) {
    rows[i] = sin(0.3 * i) + (double)(i % 5);
  }

  // snippet-begin: split
  // Real rows into split spectra: the shape of interleaved r2c rows with the layout FFTN_SPLIT; on
  // the split side strides and distances count reals, the same numbers for both arrays.
  const fftn_axis axis = {N, 0, 0};
  const fftn_batch batch = {COUNT, 0, 0};
  fftn_shape shape = {0};
  fftn_options options = fftn_options_default();
  fftn_error error;
  shape.kind = FFTN_R2C;
  shape.precision = FFTN_DOUBLE;
  shape.rank = 1;
  shape.axes = &axis;
  shape.batch_rank = 1;
  shape.batches = &batch;
  shape.layout = FFTN_SPLIT;
  fftn_plan* forward = fftn_plan_create(&shape, NULL, &error);

  // And back: c2r from the split spectra, scaled by 1/n.
  shape.kind = FFTN_C2R;
  options.scale = 1.0 / N;
  fftn_plan* backward = forward != NULL ? fftn_plan_create(&shape, &options, &error) : NULL;
  if (backward == NULL) {
    fprintf(stderr, "fftn: %s\n", error.message);
    fftn_plan_release(forward);
    return EXIT_FAILURE;
  }
  size_t bytes = fftn_plan_workspace_bytes(forward, 1);
  if (fftn_plan_workspace_bytes(backward, 1) > bytes) {
    bytes = fftn_plan_workspace_bytes(backward, 1);
  }
  fftn_workspace* ws = fftn_workspace_create(bytes, &error);
  if (ws != NULL) {
    // A real side passes its array as re; its im is not read.
    const fftn_split_const_ptr real_in = {rows, NULL};
    const fftn_split_ptr spectra = {re, im};
    const fftn_split_const_ptr spectra_in = {re, im};
    const fftn_split_ptr real_out = {back, NULL};
    fftn_execute_split(forward, real_in, spectra, ws);
    fftn_execute_split(backward, spectra_in, real_out, ws); // back = rows, up to rounding

    char text[512];
    fftn_plan_describe(forward, text, sizeof text); // names the conversion pass
    printf("%s\n", text);
    fftn_workspace_destroy(ws);
  }
  fftn_plan_release(backward);
  fftn_plan_release(forward);
  // snippet-end: split
  if (ws == NULL) {
    fprintf(stderr, "fftn: %s\n", error.message);
    return EXIT_FAILURE;
  }

  double largest = 0;
  for (i = 0; i < N * COUNT; ++i) {
    largest = fmax(largest, fabs(back[i] - rows[i]));
  }
  printf("largest round-trip error: %.3g\n", largest);
  return largest < 1e-13 ? EXIT_SUCCESS : EXIT_FAILURE;
}
