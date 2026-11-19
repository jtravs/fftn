// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for Intel oneMKL's DFTI (double precision), built only when
// FFTN_BENCH_MKL_ROOT names a oneMKL installation. oneMKL is proprietary (Intel Simplified
// Software License): this executable is for our own measurements and is never distributed.
//
// Built as its best users build it, from Intel's link-line advisor for one thread: the LP64
// interface, the sequential threading layer and the core library, dynamically
// (mkl_intel_lp64, mkl_sequential, mkl_core), so the timed code is oneMKL's own dispatch to
// the best code path for the CPU. Every case is one descriptor: the batch is
// DFTI_NUMBER_OF_TRANSFORMS, the layout is given by DFTI_INPUT_/OUTPUT_STRIDES and _DISTANCE
// for the case's one direction (FFTW's layout, padded in place for real data: the real side
// in reals, the complex side in complex values), real transforms use
// DFTI_CONJUGATE_EVEN_STORAGE = DFTI_COMPLEX_COMPLEX (FFTW's n/2+1 values), and neither
// direction is scaled (DFTI's defaults), so nothing is converted.
//
// describe reports the code path oneMKL selected (MKLVersion::Processor) and the variables
// that can change it (MKL_ENABLE_INSTRUCTIONS, MKL_CBWR), which would otherwise handicap
// or favour it silently.
//
// Options: none, except in fbench-mkl_omp (FBENCH_MKL_OMP), the same backend linked with
// oneMKL's GNU OpenMP threading layer (mkl_gnu_thread and the compiler's libgomp, the link
// line the advisor gives for GCC) instead of mkl_sequential, for threaded comparisons
// (experiment X41): THREADS=N, oneMKL's thread count (mkl_set_num_threads; default 1).

#include "backends/common.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <mkl_dfti.h>
#include <mkl_service.h>
#include <mkl_types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

namespace {

// The descriptors' DFTI_THREAD_LIMIT: 1, or THREADS in fbench-mkl_omp.
int& thread_limit() {
  static int limit = 1;
  return limit;
}

std::string& description() {
  static std::string text;
  return text;
}

struct plan_record {
  DFTI_DESCRIPTOR_HANDLE handle = nullptr;
  bool forward = true;
  bool in_place = false;
  // The interleaved layout's blocks (fbench_case::outer): one descriptor runs one block, so
  // the blocks are a loop around it, by these distances in bytes.
  std::int64_t outer = 1;
  std::ptrdiff_t in_step = 0;
  std::ptrdiff_t out_step = 0;

  plan_record() = default;
  plan_record(const plan_record&) = delete;
  plan_record& operator=(const plan_record&) = delete;
  plan_record(plan_record&&) = delete;
  plan_record& operator=(plan_record&&) = delete;
  ~plan_record() {
    if (handle != nullptr) {
      DftiFreeDescriptor(&handle);
    }
  }
};

std::string environment(const char* name) {
  const char* value = std::getenv(name); // NOLINT(concurrency-mt-unsafe): read once, at begin
  return value == nullptr ? "" : value;
}

// DFTI strides: an offset, then one stride per dimension, in elements of the domain's type.
std::array<MKL_LONG, 4> strides(const fbench_case& c, std::int64_t last) {
  std::array<MKL_LONG, 4> s{};
  const auto e = fbench::backends::element_strides(c, last);
  for (std::size_t d = 0; d < e.size(); ++d) {
    s.at(d + 1) = static_cast<MKL_LONG>(e[d] * c.interleave);
  }
  return s;
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  int threads = 1;
  for (const auto& [key, value] : fbench::backends::parse_options(options)) {
#ifdef FBENCH_MKL_OMP
    if (key == "THREADS") {
      char* end = nullptr;
      const long t = std::strtol(value.c_str(), &end, 10);
      if (value.empty() || end != value.c_str() + value.size() || t < 1 || t > 4096) {
        fbench::backends::say(message, message_size, "THREADS is a whole number from 1 to 4096");
        return fbench_error;
      }
      threads = static_cast<int>(t);
      continue;
    }
#endif
    fbench::backends::say(message, message_size, "unknown MKL option " + key);
    return fbench_error;
  }
  thread_limit() = threads;
#ifdef FBENCH_MKL_OMP
  mkl_set_num_threads(threads);
  const char* threading = "gnu_thread";
#else
  const char* threading = "sequential";
#endif
  MKLVersion v{};
  MKL_Get_Version(&v);
  const std::string version = std::to_string(v.MajorVersion) + "." +
                              std::to_string(v.MinorVersion) + "." +
                              std::to_string(v.UpdateVersion) + " build " + v.Build;
  description() = fbench::json()
                      .str("library", "mkl")
                      .str("version", version)
                      .str("isa", v.Processor)
                      .str("platform", v.Platform)
                      .str("precision", "double")
                      .raw("supports", R"(["c2c","r2c","c2r"])")
                      .str("threading", threading)
                      .integer("threads", threads)
                      .integer("max_threads", MKL_Get_Max_Threads())
                      .str("MKL_ENABLE_INSTRUCTIONS", environment("MKL_ENABLE_INSTRUCTIONS"))
                      .str("MKL_CBWR", environment("MKL_CBWR"))
                      .str("loaded_from", fbench::backends::defining_file("DftiComputeForward"))
                      .str("expected_root", FBENCH_MKL_ROOT)
                      .inner();
  return fbench_ok;
}

void fbench_end() {}

const char* fbench_describe() { return description().c_str(); }

std::int32_t fbench_create(const fbench_case* c, void* /*in*/, void* /*out*/, void** plan,
                           char* message, std::size_t message_size) {
  auto r = std::make_unique<plan_record>();
  r->forward = c->sign < 0;
  r->in_place = c->in_place != 0;
  r->outer = c->outer;
  {
    // Bytes between blocks: complex elements on the complex sides, reals on the real one.
    const auto in_element = static_cast<std::ptrdiff_t>(c->kind == fbench_r2c ? sizeof(double)
                                                                              : 2 * sizeof(double));
    const auto out_element = static_cast<std::ptrdiff_t>(
        c->kind == fbench_c2r ? sizeof(double) : 2 * sizeof(double));
    r->in_step = static_cast<std::ptrdiff_t>(c->outer_in_dist) * in_element;
    r->out_step = static_cast<std::ptrdiff_t>(c->outer_out_dist) * out_element;
  }
  const DFTI_CONFIG_VALUE domain = c->kind == fbench_c2c ? DFTI_COMPLEX : DFTI_REAL;
  std::array<MKL_LONG, 3> dims{};
  for (int d = 0; d < c->rank; ++d) {
    dims.at(static_cast<std::size_t>(d)) = static_cast<MKL_LONG>(c->dims[d]);
  }
  // The layout of the compute call's own input and output (a descriptor serves one direction
  // here), in elements of each side's type: real for the r2c input and the c2r output.
  const std::array<MKL_LONG, 4> in_strides = strides(*c, c->in_last);
  const std::array<MKL_LONG, 4> out_strides = strides(*c, c->out_last);
  const auto in_dist = static_cast<MKL_LONG>(c->in_dist);
  const auto out_dist = static_cast<MKL_LONG>(c->out_dist);

  MKL_LONG status = c->rank == 1
                        ? DftiCreateDescriptor(&r->handle, DFTI_DOUBLE, domain, 1, dims[0])
                        : DftiCreateDescriptor(&r->handle, DFTI_DOUBLE, domain,
                                               static_cast<MKL_LONG>(c->rank), dims.data());
  const auto set = [&](DFTI_CONFIG_PARAM param, auto value) {
    if (status == DFTI_NO_ERROR) {
      status = DftiSetValue(r->handle, param, value);
    }
  };
  // Placement before the layout: tight real distances are inconsistent with the default
  // in-place placement, and oneMKL checks each setting as it is made.
  set(DFTI_PLACEMENT, r->in_place ? DFTI_INPLACE : DFTI_NOT_INPLACE);
  set(DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(c->batch));
  if (c->kind != fbench_c2c) {
    set(DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
  }
  // DFTI_INPUT_/OUTPUT_ STRIDES and DISTANCE: every supported oneMKL has them (2025.2 and
  // 2026.1 have no DFTI_FWD_/BWD_STRIDES); newer headers mark them deprecated in favour of the
  // direction-independent names, so that warning is silenced here only.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  set(DFTI_INPUT_STRIDES, in_strides.data());
  set(DFTI_OUTPUT_STRIDES, out_strides.data());
  set(DFTI_INPUT_DISTANCE, in_dist);
  set(DFTI_OUTPUT_DISTANCE, out_dist);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
  set(DFTI_THREAD_LIMIT, static_cast<MKL_LONG>(thread_limit()));
  if (status == DFTI_NO_ERROR) {
    status = DftiCommitDescriptor(r->handle);
  }
  if (status != DFTI_NO_ERROR) {
    fbench::backends::say(message, message_size, std::string("MKL: ") + DftiErrorMessage(status));
    return fbench_error;
  }
  *plan = r.release();
  return fbench_ok;
}

// NOLINTNEXTLINE(misc-const-correctness): the signature is the interface's
void fbench_execute(const void* plan, void* in, void* out) {
  const auto* r = static_cast<const plan_record*>(plan);
  auto* x = static_cast<char*>(in);
  auto* y = static_cast<char*>(out);
  for (std::int64_t b = 0; b < r->outer; ++b) {
    if (r->forward && r->in_place) {
      DftiComputeForward(r->handle, x);
    } else if (r->forward) {
      DftiComputeForward(r->handle, x, y);
    } else if (r->in_place) {
      DftiComputeBackward(r->handle, x);
    } else {
      DftiComputeBackward(r->handle, x, y);
    }
    x += r->in_step;
    y += r->out_step;
  }
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
}
}
