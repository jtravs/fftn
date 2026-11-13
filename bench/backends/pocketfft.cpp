// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for pocketfft (the C++ header-only version, BSD-3-Clause), built only when
// FFTN_BENCH_POCKETFFT is on; bench/CMakeLists.txt fetches the header pinned by SHA-256.
//
// Built the way its best users build it: the high-level interface (c2c, r2c, c2r over an
// array with byte strides, the one NumPy and SciPy wrap) with the plan cache SciPy enables
// (POCKETFFT_CACHE_SIZE 16), one thread, and the build's vector extension (on x86-64 the
// backend is compiled for the build machine, see bench/README.md). pocketfft computes its
// twiddles when a length is first planned: fbench_create fetches every axis's plan from
// the cache, so that cost is plan time, and the timed executes find the plans cached.
// pocketfft's conventions are FFTW's (forward sign -1, no scaling, r2c writes n/2+1
// values), so nothing is converted.
//
// Options: none.

#define POCKETFFT_CACHE_SIZE 16
#include <pocketfft_hdronly.h>

#include "backends/common.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace pf = pocketfft::detail;

std::string& description() {
  static std::string text;
  return text;
}

struct plan_record {
  std::int32_t kind = fbench_c2c;
  bool forward = true;
  pf::shape_t shape;     // batch, then the logical dimensions (r2c/c2r: the real array's)
  pf::stride_t in;       // byte strides
  pf::stride_t out;
  pf::shape_t axes;      // every dimension but the batch
  // Keeps the cached plans alive for as long as the case exists.
  std::vector<std::shared_ptr<pf::pocketfft_c<double>>> complex_plans;
  std::shared_ptr<pf::pocketfft_r<double>> real_plan;
};

pf::stride_t byte_strides(const fbench_case& c, std::int64_t last, std::int64_t dist,
                          std::size_t element) {
  pf::stride_t s;
  s.push_back(static_cast<std::ptrdiff_t>(dist) * static_cast<std::ptrdiff_t>(element));
  for (const std::int64_t e : fbench::backends::element_strides(c, last)) {
    s.push_back(static_cast<std::ptrdiff_t>(e) * static_cast<std::ptrdiff_t>(element));
  }
  return s;
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  if (!fbench::backends::parse_options(options).empty()) {
    fbench::backends::say(message, message_size,
                          std::string("pocketfft has no options: ") + options);
    return fbench_error;
  }
  description() = fbench::json()
                      .str("library", "pocketfft")
                      .str("version", "cpp@" FBENCH_POCKETFFT_COMMIT)
                      .str("isa", fbench::backends::compiled_isa())
                      .str("precision", "double")
                      .raw("supports", R"(["c2c","r2c","c2r"])")
                      .integer("vlen", static_cast<std::int64_t>(pf::VLEN<double>::val))
                      .str("arch_flags", FBENCH_POCKETFFT_ARCH_FLAGS)
                      .inner();
  return fbench_ok;
}

void fbench_end() {}

const char* fbench_describe() { return description().c_str(); }

std::int32_t fbench_create(const fbench_case* c, void* /*in*/, void* /*out*/, void** plan,
                           char* message, std::size_t message_size) {
  try {
    auto r = std::make_unique<plan_record>();
    r->kind = c->kind;
    r->forward = c->sign < 0;
    r->shape.push_back(static_cast<std::size_t>(c->batch));
    for (int d = 0; d < c->rank; ++d) {
      r->shape.push_back(static_cast<std::size_t>(c->dims[d]));
      r->axes.push_back(static_cast<std::size_t>(d + 1));
    }
    constexpr std::size_t real_size = sizeof(double);
    constexpr std::size_t complex_size = sizeof(std::complex<double>);
    const std::size_t in_element = c->kind == fbench_r2c ? real_size : complex_size;
    const std::size_t out_element = c->kind == fbench_c2r ? real_size : complex_size;
    r->in = byte_strides(*c, c->in_last, c->in_dist, in_element);
    r->out = byte_strides(*c, c->out_last, c->out_dist, out_element);
    // Plan every axis now (the cache keeps them): the last axis of a real transform is a
    // real plan, every other axis a complex one.
    const int complex_axes = c->kind == fbench_c2c ? c->rank : c->rank - 1;
    for (int d = 0; d < complex_axes; ++d) {
      r->complex_plans.push_back(
          pf::get_plan<pf::pocketfft_c<double>>(static_cast<std::size_t>(c->dims[d])));
    }
    if (c->kind != fbench_c2c) {
      r->real_plan =
          pf::get_plan<pf::pocketfft_r<double>>(static_cast<std::size_t>(c->dims[c->rank - 1]));
    }
    *plan = r.release();
    return fbench_ok;
  } catch (const std::exception& e) {
    fbench::backends::say(message, message_size, std::string("pocketfft: ") + e.what());
    return fbench_error;
  }
}

void fbench_execute(const void* plan, void* in, void* out) {
  const auto* r = static_cast<const plan_record*>(plan);
  using cd = std::complex<double>;
  if (r->kind == fbench_c2c) {
    pocketfft::c2c(r->shape, r->in, r->out, r->axes, r->forward, static_cast<const cd*>(in),
                   static_cast<cd*>(out), 1.0, 1);
  } else if (r->kind == fbench_r2c) {
    pocketfft::r2c(r->shape, r->in, r->out, r->axes, pocketfft::FORWARD,
                   static_cast<const double*>(in), static_cast<cd*>(out), 1.0, 1);
  } else {
    pocketfft::c2r(r->shape, r->in, r->out, r->axes, pocketfft::BACKWARD,
                   static_cast<const cd*>(in), static_cast<double*>(out), 1.0, 1);
  }
}

void fbench_destroy(void* plan) { const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan)); }
}
