// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for FFTN: one plan and one workspace per case.
//
// Every case of rank 1 to 3 and every kind is supported, in double and single precision,
// with FFTW's embedding (in_last, out_last) and distances. Plans are built and executed
// through the plan core behind fftn::plan (include/fftn/detail/core.hpp), with the calls
// plan's constructor and execute make, so that one option can reach a construction knob
// the public API does not offer:
//
//   RADICES=r1.r2...  the chain's radices in pass order (build_settings::radices in
//                     src/core/plan_core.hpp, an internal test knob), for c2c rows whose
//                     length they multiply to; any other case is an error, and so is a
//                     radix no pass of the target takes. Experiment X2 times radix
//                     sequences with it (dev/experiments/X2-radix-sets.md).
//
// Without options the plans are exactly the public API's. describe reports the instruction
// set FFTN's plans choose on this machine and the forced radices, if any; every case's note
// is its plan's per-axis line (the radices it uses).

#include "harness/backend.hpp"
#include "harness/json.hpp"

#include "core/plan_core.hpp"

#include <fftn/fftn.hpp>

#include <algorithm>
#include <charconv>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct session {
  std::string description;
  std::vector<std::size_t> radices; // forced, in pass order; empty: the rules'
  std::string radices_text;         // as given, r1.r2...
};

session& state() {
  static session s;
  return s;
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
std::shared_ptr<const fftn::detail::plan_core>
core_of(const fbench_case& c, const fftn::options& opts,
        const fftn::detail::build_settings& settings) {
  const fftn::shape<In, Out> s = shape_of<In, Out>(c);
  return fftn::detail::make_plan_core(fftn::detail::view_of(s), opts, settings);
}

template <class T>
std::shared_ptr<const fftn::detail::plan_core>
core_of(const fbench_case& c, const fftn::options& opts,
        const fftn::detail::build_settings& settings) {
  using cplx = std::complex<T>;
  if (c.kind == fbench_c2c) {
    return core_of<cplx, cplx>(c, opts, settings);
  }
  if (c.kind == fbench_r2c) {
    return core_of<T, cplx>(c, opts, settings);
  }
  return core_of<cplx, T>(c, opts, settings);
}

// "4 x 8 x 8", as the plan's describe lists radices.
std::string radix_list(const std::vector<std::size_t>& radices) {
  std::string text;
  for (const std::size_t r : radices) {
    text += (text.empty() ? "" : " x ") + std::to_string(r);
  }
  return text;
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

// Parses r1.r2... (each an integer of at least 2); throws std::invalid_argument.
std::vector<std::size_t> parse_radices(std::string_view text) {
  std::vector<std::size_t> out;
  std::size_t at = 0;
  while (at <= text.size()) {
    const std::size_t end = std::min(text.find('.', at), text.size());
    const std::string_view token = text.substr(at, end - at);
    std::size_t r = 0;
    const auto res = std::from_chars(token.data(), token.data() + token.size(), r);
    if (token.empty() || res.ec != std::errc{} || res.ptr != token.data() + token.size() || r < 2) {
      throw std::invalid_argument("RADICES is r1.r2... with every radix at least 2, not '" +
                                  std::string(text) + "'");
    }
    out.push_back(r);
    at = end + 1;
  }
  return out;
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  session& s = state();
  s.radices.clear();
  s.radices_text.clear();
  const std::string_view all(options);
  std::size_t at = 0;
  try {
    while (at < all.size()) {
      const std::size_t end = std::min(all.find(' ', at), all.size());
      const std::string_view token = all.substr(at, end - at);
      at = end + 1;
      if (token.empty()) {
        continue;
      }
      const std::size_t eq = token.find('=');
      const std::string_view key = token.substr(0, eq);
      if (key != "RADICES" || eq == std::string_view::npos) {
        throw std::invalid_argument("unknown FFTN option " + std::string(token) +
                                    " (FFTN takes RADICES=r1.r2...)");
      }
      s.radices_text = token.substr(eq + 1);
      s.radices = parse_radices(s.radices_text);
    }
    const fftn::c2c_plan<double> probe(fftn::rows_c2c<double>(8, 1, fftn::direction::forward));
    fbench::json d;
    d.str("library", "fftn")
        .str("version", fftn::version())
        .str("isa", fftn::isa_name(probe.instruction_set()))
        .raw("supports", R"(["c2c","r2c","c2r"])")
        .raw("precisions", R"(["f64","f32"])")
        .str("strategy", "chain");
    if (!s.radices.empty()) {
      d.str("radices", s.radices_text);
    }
    s.description = d.inner();
  } catch (const std::exception& e) {
    std::snprintf(message, message_size, "%s", e.what());
    return fbench_error;
  }
  return fbench_ok;
}

void fbench_end() {}

const char* fbench_describe() { return state().description.c_str(); }

std::int32_t fbench_create(const fbench_case* c, void* /*in*/, void* /*out*/, void** plan,
                           char* message, std::size_t message_size) {
  *plan = nullptr;
  const session& s = state();
  try {
    fftn::detail::build_settings settings;
    if (!s.radices.empty()) {
      std::size_t product = 1;
      for (const std::size_t r : s.radices) {
        product *= r;
      }
      if (c->kind != fbench_c2c || c->rank != 1) {
        throw std::invalid_argument("RADICES=" + s.radices_text + " is for c2c rows only");
      }
      if (product != static_cast<std::size_t>(c->dims[0])) {
        throw std::invalid_argument("RADICES=" + s.radices_text + " multiplies to " +
                                    std::to_string(product) + ", not the length " +
                                    std::to_string(c->dims[0]));
      }
      settings.radices = s.radices;
    }
    auto record = std::make_unique<plan_record>();
    const fftn::options opts{.in_place = c->in_place != 0};
    record->core = c->precision == fbench_f32 ? core_of<float>(*c, opts, settings)
                                              : core_of<double>(*c, opts, settings);
    const std::string described = fftn::detail::describe(*record->core);
    if (!s.radices.empty() &&
        described.find("radices " + radix_list(s.radices) + "\n") == std::string::npos) {
      throw std::logic_error("the plan does not use RADICES=" + s.radices_text + ":\n" + described);
    }
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
