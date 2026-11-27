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
// values), so nothing is converted. Real-to-real cases (task 6.12) go through its dct and dst
// (types 1 to 4, unnormalised, fct 1 and ortho off: FFTW's REDFT and RODFT definitions, as SciPy
// documents for the same calls) and r2r_separable_fht (FFTW's separable DHT, sum of x cas; its
// r2r_separable_hartley is the transform with cos - sin): one call over every axis when the
// axes share a kind, else one call per axis, the first from the input to the output and the
// others in place there (as a user mixing kinds calls it); their plans are cached and fetched at
// plan time too. The interleaved layout (fbench_case::interleave) is
// pocketfft's byte strides: the batch one element apart, each axis `interleave` times its tight
// stride, and the blocks (`outer`) a dimension of their own. Double and single precision.
//
// Options: none.

#define POCKETFFT_CACHE_SIZE 16
#include <pocketfft_hdronly.h>

#include "backends/common.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <algorithm>
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
  bool single = false;
  bool forward = true;
  std::vector<std::int32_t> r2r; // the kind of each axis of an r2r case
  pf::shape_t shape; // [outer,] batch, then the logical dimensions (r2c/c2r: the real array's)
  pf::stride_t in;   // byte strides
  pf::stride_t out;
  pf::shape_t axes; // every dimension but the batch (and the blocks)
  // Keeps the cached plans alive for as long as the case exists.
  std::vector<std::shared_ptr<void>> plans;
};

pf::stride_t byte_strides(const fbench_case& c, std::int64_t last, std::int64_t dist,
                          std::int64_t outer_dist, std::size_t element) {
  const auto bytes = [element](std::int64_t elements) {
    return static_cast<std::ptrdiff_t>(elements) * static_cast<std::ptrdiff_t>(element);
  };
  pf::stride_t s;
  if (c.outer > 1) {
    s.push_back(bytes(outer_dist));
  }
  s.push_back(bytes(dist));
  for (const std::int64_t e : fbench::backends::element_strides(c, last)) {
    s.push_back(bytes(e * c.interleave));
  }
  return s;
}

// The plan pocketfft's dct, dst or Hartley transform fetches for an axis of n.
template <class T> std::shared_ptr<void> r2r_plan(std::int32_t kind, std::size_t n) {
  switch (kind) {
  case fbench_dct1:
    return pf::get_plan<pf::T_dct1<T>>(n);
  case fbench_dst1:
    return pf::get_plan<pf::T_dst1<T>>(n);
  case fbench_dct4:
  case fbench_dst4:
    return pf::get_plan<pf::T_dcst4<T>>(n);
  case fbench_dht:
    return pf::get_plan<pf::pocketfft_r<T>>(n);
  default:
    return pf::get_plan<pf::T_dcst23<T>>(n);
  }
}

template <class T> void plan_axes(const fbench_case& c, plan_record& r) {
  constexpr std::size_t real_size = sizeof(T);
  constexpr std::size_t complex_size = sizeof(std::complex<T>);
  const bool real_in = c.kind == fbench_r2c || c.kind == fbench_r2r;
  const bool real_out = c.kind == fbench_c2r || c.kind == fbench_r2r;
  r.in = byte_strides(c, c.in_last, c.in_dist, c.outer_in_dist, real_in ? real_size : complex_size);
  r.out = byte_strides(c, c.out_last, c.out_dist, c.outer_out_dist,
                       real_out ? real_size : complex_size);
  if (c.kind == fbench_r2r) {
    for (int d = 0; d < c.rank; ++d) {
      r.plans.push_back(
          r2r_plan<T>(r.r2r.at(static_cast<std::size_t>(d)), static_cast<std::size_t>(c.dims[d])));
    }
    return;
  }
  // Plan every axis now (the cache keeps them): the last axis of a real transform is a
  // real plan, every other axis a complex one.
  const int complex_axes = c.kind == fbench_c2c ? c.rank : c.rank - 1;
  for (int d = 0; d < complex_axes; ++d) {
    r.plans.push_back(pf::get_plan<pf::pocketfft_c<T>>(static_cast<std::size_t>(c.dims[d])));
  }
  if (c.kind != fbench_c2c) {
    r.plans.push_back(
        pf::get_plan<pf::pocketfft_r<T>>(static_cast<std::size_t>(c.dims[c.rank - 1])));
  }
}

template <class T> void run(const plan_record& r, void* in, void* out) {
  using cx = std::complex<T>;
  const T one = 1;
  switch (r.kind) {
  case fbench_c2c:
    pocketfft::c2c(r.shape, r.in, r.out, r.axes, r.forward, static_cast<const cx*>(in),
                   static_cast<cx*>(out), one, 1);
    return;
  case fbench_r2c:
    pocketfft::r2c(r.shape, r.in, r.out, r.axes, pocketfft::FORWARD, static_cast<const T*>(in),
                   static_cast<cx*>(out), one, 1);
    return;
  case fbench_c2r:
    pocketfft::c2r(r.shape, r.in, r.out, r.axes, pocketfft::BACKWARD, static_cast<const cx*>(in),
                   static_cast<T*>(out), one, 1);
    return;
  default:
    break;
  }
  const auto call = [&](std::int32_t kind, const pf::shape_t& axes, const T* x, T* y,
                        const pf::stride_t& sx) {
    if (kind == fbench_dht) {
      pocketfft::r2r_separable_fht(r.shape, sx, r.out, axes, x, y, one, 1);
    } else if (kind <= fbench_dct4) {
      pocketfft::dct(r.shape, sx, r.out, axes, kind - fbench_dct1 + 1, x, y, one, false, 1);
    } else {
      pocketfft::dst(r.shape, sx, r.out, axes, kind - fbench_dst1 + 1, x, y, one, false, 1);
    }
  };
  const auto* x = static_cast<const T*>(in);
  auto* y = static_cast<T*>(out);
  if (std::ranges::all_of(r.r2r, [&](std::int32_t k) { return k == r.r2r.front(); })) {
    call(r.r2r.front(), r.axes, x, y, r.in);
    return;
  }
  for (std::size_t d = 0; d < r.axes.size(); ++d) {
    call(r.r2r[d], {r.axes[d]}, d == 0 ? x : y, y, d == 0 ? r.in : r.out);
  }
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
                      .raw("supports", R"(["c2c","r2c","c2r","r2r"])")
                      .raw("precisions", R"(["f64","f32"])")
                      .integer("vlen", static_cast<std::int64_t>(pf::VLEN<double>::val))
                      .integer("vlen_f32", static_cast<std::int64_t>(pf::VLEN<float>::val))
                      .str("arch_flags", FBENCH_POCKETFFT_ARCH_FLAGS)
                      .str("fp_contract", "fast")
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
    r->single = c->precision == fbench_f32;
    r->forward = c->sign < 0;
    if (c->kind == fbench_r2r) {
      r->r2r.assign(c->r2r_kinds, c->r2r_kinds + c->rank);
    }
    if (c->outer > 1) {
      r->shape.push_back(static_cast<std::size_t>(c->outer));
    }
    r->shape.push_back(static_cast<std::size_t>(c->batch));
    const std::size_t first_axis = r->shape.size();
    for (int d = 0; d < c->rank; ++d) {
      r->shape.push_back(static_cast<std::size_t>(c->dims[d]));
      r->axes.push_back(first_axis + static_cast<std::size_t>(d));
    }
    if (r->single) {
      plan_axes<float>(*c, *r);
    } else {
      plan_axes<double>(*c, *r);
    }
    *plan = r.release();
    return fbench_ok;
  } catch (const std::exception& e) {
    fbench::backends::say(message, message_size, std::string("pocketfft: ") + e.what());
    return fbench_error;
  }
}

// NOLINTNEXTLINE(misc-const-correctness): the signature is the interface's
void fbench_execute(const void* plan, void* in, void* out) {
  const auto* r = static_cast<const plan_record*>(plan);
  if (r->single) {
    run<float>(*r, in, out);
  } else {
    run<double>(*r, in, out);
  }
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
}
}
