// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTN: one plan and one workspace per case.
//
// Every case of rank 1 to 3 and every kind is supported, in double and single precision,
// with FFTW's embedding (in_last, out_last) and distances. Plans are built and executed
// through the plan core behind fftn::plan (include/fftn/detail/core.hpp), with the calls
// plan's constructor and execute make. FFTN has no plan options yet, so any option is
// refused. describe reports the instruction set FFTN's plans choose on this machine; every
// case's note is its plan's per-axis line (the radices it uses).

#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <fftn/fftn.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <string_view>

namespace {

std::string& description() {
  static std::string text;
  return text;
}

struct plan_record {
  std::shared_ptr<const fftn::detail::plan_core> core;
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

template <class In, class Out>
std::shared_ptr<const fftn::detail::plan_core> core_of(const fbench_case& c,
                                                       const fftn::options& opts) {
  const fftn::shape<In, Out> s = shape_of<In, Out>(c);
  return fftn::detail::make_plan_core(fftn::detail::view_of(s), opts);
}

template <class T>
std::shared_ptr<const fftn::detail::plan_core> core_of(const fbench_case& c,
                                                       const fftn::options& opts) {
  using cplx = std::complex<T>;
  if (c.kind == fbench_c2c) {
    return core_of<cplx, cplx>(c, opts);
  }
  if (c.kind == fbench_r2c) {
    return core_of<T, cplx>(c, opts);
  }
  return core_of<cplx, T>(c, opts);
}

// The plan's per-axis lines of its describe, joined by "; ".
std::string axis_lines(const std::string& describe) {
  std::string out;
  std::size_t at = 0;
  while (at < describe.size()) {
    const std::size_t end = std::min(describe.find('\n', at), describe.size());
    const std::string_view line = std::string_view(describe).substr(at, end - at);
    if (line.starts_with("axis ")) {
      out += (out.empty() ? "" : "; ") + std::string(line);
    }
    at = end + 1;
  }
  return out;
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  if (!std::string_view(options).empty()) {
    std::snprintf(message, message_size, "unknown FFTN option: FFTN has none yet (%s)", options);
    return fbench_error;
  }
  const fftn::c2c_plan<double> probe(fftn::rows_c2c<double>(8, 1, fftn::direction::forward));
  description() = fbench::json()
                      .str("library", "fftn")
                      .str("version", fftn::version())
                      .str("isa", fftn::isa_name(probe.instruction_set()))
                      .raw("supports", R"(["c2c","r2c","c2r"])")
                      .raw("precisions", R"(["f64","f32"])")
                      .str("strategy", "chain")
                      .inner();
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
    record->core =
        c->precision == fbench_f32 ? core_of<float>(*c, opts) : core_of<double>(*c, opts);
    const std::string described = fftn::detail::describe(*record->core);
    record->ws = fftn::workspace(fftn::detail::workspace_bytes(*record->core, 1));
    std::snprintf(message, message_size, "%s", axis_lines(described).c_str());
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
  // What plan<In, Out>::execute does (include/fftn/plan.hpp).
  fftn::detail::execute_core(*r->core, 0, fftn::detail::transform_count(*r->core), in, out, r->ws);
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
}
}
