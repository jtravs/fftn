// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for Intel oneMKL's DFTI (double precision), built only when
// FFTN_BENCH_MKL_ROOT names a oneMKL installation (2024.1 or newer: the direction-
// independent DFTI_FWD_/DFTI_BWD_ layout parameters). oneMKL is proprietary (Intel Simplified
// Software License): this executable is for our own measurements and is never distributed.
//
// Built as its best users build it, from Intel's link-line advisor for one thread: the LP64
// interface, the sequential threading layer and the core library, dynamically
// (mkl_intel_lp64, mkl_sequential, mkl_core), so the timed code is oneMKL's own dispatch to
// the best code path for the CPU. Every case is one descriptor: the batch is
// DFTI_NUMBER_OF_TRANSFORMS, the layout is given by DFTI_FWD_/BWD_STRIDES and _DISTANCE
// (FFTW's layout, padded in place for real data), real transforms use
// DFTI_CONJUGATE_EVEN_STORAGE = DFTI_COMPLEX_COMPLEX (FFTW's n/2+1 values), and neither
// direction is scaled (DFTI's defaults), so nothing is converted.
//
// describe reports the code path oneMKL selected (MKLVersion::Processor) and the variables
// that can change it (MKL_ENABLE_INSTRUCTIONS, MKL_CBWR), which would otherwise handicap
// or favour it silently.
//
// Options: none.

#include "backends/common.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <mkl.h>
#include <mkl_dfti.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

namespace {

std::string& description() {
  static std::string text;
  return text;
}

struct plan_record {
  DFTI_DESCRIPTOR_HANDLE handle = nullptr;
  bool forward = true;
  bool in_place = false;

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
    s.at(d + 1) = static_cast<MKL_LONG>(e[d]);
  }
  return s;
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  if (!fbench::backends::parse_options(options).empty()) {
    fbench::backends::say(message, message_size, std::string("MKL has no options: ") + options);
    return fbench_error;
  }
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
                      .str("threading", "sequential")
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
  const DFTI_CONFIG_VALUE domain = c->kind == fbench_c2c ? DFTI_COMPLEX : DFTI_REAL;
  std::array<MKL_LONG, 3> dims{};
  for (int d = 0; d < c->rank; ++d) {
    dims.at(static_cast<std::size_t>(d)) = static_cast<MKL_LONG>(c->dims[d]);
  }
  // The forward domain is the input of r2c and of a forward c2c; for c2r it is the output.
  const bool input_is_forward = c->kind == fbench_r2c || (c->kind == fbench_c2c && r->forward);
  const std::int64_t fwd_last = input_is_forward ? c->in_last : c->out_last;
  const std::int64_t bwd_last = input_is_forward ? c->out_last : c->in_last;
  const auto fwd_dist = static_cast<MKL_LONG>(input_is_forward ? c->in_dist : c->out_dist);
  const auto bwd_dist = static_cast<MKL_LONG>(input_is_forward ? c->out_dist : c->in_dist);
  const std::array<MKL_LONG, 4> fwd_strides = strides(*c, fwd_last);
  const std::array<MKL_LONG, 4> bwd_strides = strides(*c, bwd_last);

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
  set(DFTI_FWD_STRIDES, fwd_strides.data());
  set(DFTI_BWD_STRIDES, bwd_strides.data());
  set(DFTI_FWD_DISTANCE, fwd_dist);
  set(DFTI_BWD_DISTANCE, bwd_dist);
  set(DFTI_THREAD_LIMIT, MKL_LONG{1});
  if (status == DFTI_NO_ERROR) {
    status = DftiCommitDescriptor(r->handle);
  }
  if (status != DFTI_NO_ERROR) {
    fbench::backends::say(message, message_size,
                          std::string("MKL: ") + DftiErrorMessage(status));
    return fbench_error;
  }
  *plan = r.release();
  return fbench_ok;
}

void fbench_execute(const void* plan, void* in, void* out) {
  const auto* r = static_cast<const plan_record*>(plan);
  if (r->forward && r->in_place) {
    DftiComputeForward(r->handle, in);
  } else if (r->forward) {
    DftiComputeForward(r->handle, in, out);
  } else if (r->in_place) {
    DftiComputeBackward(r->handle, in);
  } else {
    DftiComputeBackward(r->handle, in, out);
  }
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
}
}
