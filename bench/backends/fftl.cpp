// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTL.jl through libfftl, its C library, built only when
// FFTN_BENCH_FFTL_ROOT names a libfftl directory (include/fftw3.h, include/fftl.h and
// build/lib/libfftl). FFTL is MIT-licensed.
//
// libfftl is FFTL.jl compiled ahead of time with juliac: a self-contained shared library
// that embeds the Julia runtime (no Julia installation is needed to run it) and implements
// FFTW's C interface (its own fftw3.h, written from FFTW's manual). So the calls here are
// FFTW's advanced ("many") interface on the harness's buffers, as an FFTW user would make
// them, and FFTW's conventions hold. What differs from FFTW (libfftl's fftw3.h): planner
// flags and wisdom do nothing (FFTL plans by rules), any alignment is fine, and the worker
// pool is fixed when the library is built; plans use one thread unless
// fftw_plan_with_nthreads says otherwise, and this backend never does.
//
// The instruction set is fixed when libfftl is built (JULIA_CPU_TARGET; libfftl/build.sh
// picks apple-m1 on Apple silicon and x86-64-v3 on x86-64) and cannot be read back from the
// library, so describe reports the target given to CMake as FFTN_BENCH_FFTL_CPU_TARGET.
//
// Options: none.

#include "backends/common.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <fftl.h>
#include <fftw3.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

std::string& description() {
  static std::string text;
  return text;
}

struct plan_record {
  fftw_plan plan = nullptr;
  std::int32_t kind = fbench_c2c;
};

std::string fftl_text(std::size_t (*get)(char*, std::size_t)) {
  std::vector<char> text(get(nullptr, 0) + 1);
  get(text.data(), text.size());
  return text.data();
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  if (!fbench::backends::parse_options(options).empty()) {
    fbench::backends::say(message, message_size, std::string("FFTL has no options: ") + options);
    return fbench_error;
  }
  description() = fbench::json()
                      .str("library", "fftl")
                      .str("version", fftl_text(fftl_version))
                      .str("isa", FBENCH_FFTL_CPU_TARGET)
                      .str("precision", "double")
                      .raw("supports", R"(["c2c","r2c","c2r"])")
                      .integer("threads", fftw_planner_nthreads())
                      .str("loaded_from", fbench::backends::defining_file("fftw_execute"))
                      .str("expected_root", FBENCH_FFTL_ROOT)
                      .inner();
  return fbench_ok;
}

void fbench_end() { fftw_cleanup(); }

const char* fbench_describe() { return description().c_str(); }

std::int32_t fbench_create(const fbench_case* c, void* in, void* out, void** plan, char* message,
                           std::size_t message_size) {
  if (c->interleave > 1) {
    fbench::backends::say(message, message_size, "no interleaved layout");
    return fbench_unsupported;
  }
  // The interface takes int; the harness keeps every length below 2^31.
  std::array<int, 3> n{};
  std::array<int, 3> in_embed{};
  std::array<int, 3> out_embed{};
  for (int d = 0; d < c->rank; ++d) {
    const auto i = static_cast<std::size_t>(d);
    n.at(i) = static_cast<int>(c->dims[i]);
    in_embed.at(i) = n.at(i);
    out_embed.at(i) = n.at(i);
  }
  const auto last = static_cast<std::size_t>(c->rank - 1);
  in_embed.at(last) = static_cast<int>(c->in_last);
  out_embed.at(last) = static_cast<int>(c->out_last);
  const auto batch = static_cast<int>(c->batch);
  const auto in_dist = static_cast<int>(c->in_dist);
  const auto out_dist = static_cast<int>(c->out_dist);
  // Flags change nothing in FFTL (it plans by rules and never touches the arrays).
  const unsigned flags = FFTW_ESTIMATE;
  fftw_plan p = nullptr;
  if (c->kind == fbench_c2c) {
    p = fftw_plan_many_dft(c->rank, n.data(), batch, static_cast<fftw_complex*>(in),
                           in_embed.data(), 1, in_dist, static_cast<fftw_complex*>(out),
                           out_embed.data(), 1, out_dist, c->sign, flags);
  } else if (c->kind == fbench_r2c) {
    p = fftw_plan_many_dft_r2c(c->rank, n.data(), batch, static_cast<double*>(in), in_embed.data(),
                               1, in_dist, static_cast<fftw_complex*>(out), out_embed.data(), 1,
                               out_dist, flags);
  } else {
    p = fftw_plan_many_dft_c2r(c->rank, n.data(), batch, static_cast<fftw_complex*>(in),
                               in_embed.data(), 1, in_dist, static_cast<double*>(out),
                               out_embed.data(), 1, out_dist, flags);
  }
  if (p == nullptr) {
    fbench::backends::say(message, message_size,
                          "libfftl returned no plan: " + fftl_text(fftl_last_error));
    return fbench_error;
  }
  auto record = std::make_unique<plan_record>();
  record->plan = p;
  record->kind = c->kind;
  *plan = record.release();
  return fbench_ok;
}

void fbench_execute(const void* plan, void* in, void* out) {
  const auto* r = static_cast<const plan_record*>(plan);
  if (r->kind == fbench_c2c) {
    fftw_execute_dft(r->plan, static_cast<fftw_complex*>(in), static_cast<fftw_complex*>(out));
  } else if (r->kind == fbench_r2c) {
    fftw_execute_dft_r2c(r->plan, static_cast<double*>(in), static_cast<fftw_complex*>(out));
  } else {
    fftw_execute_dft_c2r(r->plan, static_cast<fftw_complex*>(in), static_cast<double*>(out));
  }
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
  fftw_destroy_plan(r->plan);
}
}
