// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTN, through its public API: one plan and one workspace per case.
//
// Every case of rank 1 to 3 and every kind is supported, in double precision, with FFTW's
// embedding (in_last, out_last) and distances. One option: ISA=<name> forces every plan to
// that instruction set (options::force_isa; names as fftn::isa_name writes them, "best" the
// default), so that arms of one build can compare targets (A=fftn+ISA=avx2
// B=fftn+ISA=avx512). A target this build or processor lacks is refused when the session
// begins. describe reports the instruction set the plans run ("isa") and whether it was
// forced ("isa_forced").

#include "backends/common.hpp"
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

// The instruction set of the session's plans (ISA=), set once by fbench_begin.
fftn::isa& forced_isa() {
  static fftn::isa target = fftn::isa::best;
  return target;
}

// The enumerator named `name` (fftn::isa_name), or nullopt.
std::optional<fftn::isa> isa_from_name(std::string_view name) {
  for (auto i = static_cast<unsigned>(fftn::isa::scalar);
       i <= static_cast<unsigned>(fftn::isa::best); ++i) {
    const auto t = static_cast<fftn::isa>(i);
    if (name == fftn::isa_name(t)) {
      return t;
    }
  }
  return std::nullopt;
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
  forced_isa() = fftn::isa::best;
  for (const auto& [key, val] : fbench::backends::parse_options(options)) {
    const std::optional<fftn::isa> target = isa_from_name(val);
    if (key != "ISA" || !target) {
      fbench::backends::say(message, message_size,
                            "FFTN options: ISA=scalar|avx2|avx512|neon|neon_fcma|...|best (got " +
                                key + "=" + val + ")");
      return fbench_error;
    }
    forced_isa() = *target;
  }
  std::optional<fftn::c2c_plan<double>> probe;
  try {
    probe.emplace(fftn::rows_c2c<double>(8, 1, fftn::direction::forward),
                  fftn::options{.force_isa = forced_isa()});
  } catch (const std::exception& e) { // a target this build or processor lacks
    fbench::backends::say(message, message_size, e.what());
    return fbench_error;
  }
  description() = fbench::json()
                      .str("library", "fftn")
                      .str("version", fftn::version())
                      .str("isa", fftn::isa_name(probe->instruction_set()))
                      .boolean("isa_forced", forced_isa() != fftn::isa::best)
                      .str("precision", "double")
                      .raw("supports", R"(["c2c","r2c","c2r"])")
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
    const fftn::options opts{.in_place = c->in_place != 0, .force_isa = forced_isa()};
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
