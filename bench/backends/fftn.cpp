// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTN, through its public API: one plan and one workspace per case.
//
// Every case of rank 1 to 3 and every kind is supported, in double precision, with FFTW's
// embedding (in_last, out_last) and distances. FFTN has no plan options yet, so any option
// is refused. describe reports the instruction set FFTN's plans choose on this machine
// (every plan of the walking skeleton chooses the same one).

#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <fftn/fftn.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace {

using cplx = std::complex<double>;

std::string& description() {
  static std::string text;
  return text;
}

struct plan_record {
  std::optional<fftn::c2c_plan<double>> c2c;
  std::optional<fftn::r2c_plan<double>> r2c;
  std::optional<fftn::c2r_plan<double>> c2r;
  fftn::workspace ws;
};

// FFTW's many-interface layout: row-major arrays whose last dimension is embedded in
// `last` elements, transforms `dist` elements apart.
template <class In, class Out> fftn::shape<In, Out> shape_of(const fbench_case& c) {
  fftn::shape<In, Out> s;
  s.dir = c.sign < 0 ? fftn::direction::forward : fftn::direction::backward;
  const auto rank = static_cast<std::size_t>(c.rank);
  for (std::size_t d = 0; d < rank; ++d) {
    s.axes.push_back({.n = static_cast<std::size_t>(c.dims[d])});
  }
  std::ptrdiff_t in_stride = 1;
  std::ptrdiff_t out_stride = 1;
  for (std::size_t i = rank; i > 0; --i) {
    const std::size_t d = i - 1;
    s.axes[d].in_stride = in_stride;
    s.axes[d].out_stride = out_stride;
    const bool last = d + 1 == rank;
    in_stride *= last ? c.in_last : c.dims[d];
    out_stride *= last ? c.out_last : c.dims[d];
  }
  s.batches.push_back({
      .count = static_cast<std::size_t>(c.batch),
      .in_dist = static_cast<std::ptrdiff_t>(c.in_dist),
      .out_dist = static_cast<std::ptrdiff_t>(c.out_dist),
  });
  return s;
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  if (!std::string_view(options).empty()) {
    std::snprintf(message, message_size, "unknown FFTN option: FFTN has none yet (%s)", options);
    return fbench_error;
  }
  const fftn::c2c_plan<double> probe(fftn::rows_c2c<double>(8, 1, fftn::direction::forward));
  fbench::json j;
  j.str("library", "fftn")
      .str("version", fftn::version())
      .str("isa", fftn::isa_name(probe.instruction_set()))
      .str("precision", "double")
      .raw("supports", R"(["c2c","r2c","c2r"])")
      .str("strategy", "chain")
      // Experiment X15: the switches the library was built with (cmake/FftnExperiments.cmake).
      .str("x15_switches", FFTN_X15_LIBRARY_SWITCHES);
#if defined(FFTN_X_NATIVE) && FFTN_X_NATIVE
  j.str("distributable", "no: X15 arm E, the whole library built for this machine");
#endif
  description() = j.inner();
  return fbench_ok;
}

void fbench_end() {}

const char* fbench_describe() { return description().c_str(); }

std::int32_t fbench_create(const fbench_case* c, void* /*in*/, void* /*out*/, void** plan,
                           char* message, std::size_t message_size) {
  *plan = nullptr;
  try {
    auto record = std::make_unique<plan_record>();
    const fftn::options opts{.in_place = c->in_place != 0};
    std::size_t bytes = 0;
    if (c->kind == fbench_c2c) {
      bytes = record->c2c.emplace(shape_of<cplx, cplx>(*c), opts).workspace_bytes();
    } else if (c->kind == fbench_r2c) {
      bytes = record->r2c.emplace(shape_of<double, cplx>(*c), opts).workspace_bytes();
    } else {
      bytes = record->c2r.emplace(shape_of<cplx, double>(*c), opts).workspace_bytes();
    }
    record->ws = fftn::workspace(bytes);
    *plan = record.release();
    return fbench_ok;
  } catch (const std::exception& e) {
    std::snprintf(message, message_size, "%s", e.what());
    return fbench_error;
  }
}

// NOLINTNEXTLINE(misc-const-correctness): the harness fixes this C signature
void fbench_execute(const void* plan, void* in, void* out) {
  // The workspace is scratch, so a const plan record may lend it out.
  auto* r = const_cast<plan_record*>(static_cast<const plan_record*>(plan));
  if (r->c2c) {
    r->c2c->execute(static_cast<const cplx*>(in), static_cast<cplx*>(out), r->ws);
  } else if (r->r2c) {
    r->r2c->execute(static_cast<const double*>(in), static_cast<cplx*>(out), r->ws);
  } else if (r->c2r) {
    r->c2r->execute(static_cast<const cplx*>(in), static_cast<double*>(out), r->ws);
  }
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
}
}
