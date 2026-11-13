// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for Apple Accelerate (vDSP), macOS only, built when FFTN_BENCH_ACCELERATE
// is on.
//
// vDSP has DFTs of rows only, of lengths 2^k and f 2^k for small f; anything else (planes,
// volumes, other lengths) is reported unsupported. Two interfaces, chosen with API=:
//   interleaved (default)  vDSP_DFT_Interleaved_CreateSetupD / _ExecuteD (macOS 12): works
//                          on interleaved complex data, FFTW's layout, so nothing is
//                          converted. Lengths f 2^k, f in {2, 3, 5, 9, 15, 25}, k >= 2
//                          (complex length; a real transform of n uses n/2).
//   dft                    vDSP_DFT_zop / zrop_CreateSetupD and vDSP_DFT_ExecuteD, Apple's
//                          recommended split-complex routines: each row is converted from
//                          and to interleaved data with vDSP_ctozD / vDSP_ztocD around the
//                          transform. Lengths 2^k and f 2^k, f in {3, 5, 15} (k >= 3 for
//                          complex, k >= 4 for real).
// The default was chosen by an fbench A/B of the two (bench/README.md).
//
// What is timed, beyond the transform: rows are executed one by one (vDSP has no batched
// DFT). Real transforms use vDSP's packed format, {2 X[0], 2 X[n/2]} in the first complex
// value and 2 X[k] after it, so r2c unpacks the Nyquist value and scales by 1/2, and c2r
// packs X[n/2] into the imaginary part of X[0] (in the input, as FFTW's c2r may destroy its
// input; its inverse then equals FFTW's unscaled c2r). With API=dft the conversions to and
// from split complex are timed too: an application holding interleaved data pays them.
//
// Options: API=interleaved|dft.

#include "backends/common.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <Accelerate/Accelerate.h>
#include <sys/sysctl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

enum class api { interleaved, dft };

struct backend_state {
  api interface = api::interleaved;
  std::string description;
};

backend_state& state() {
  static backend_state s;
  return s;
}

std::string sysctl_text(const char* name) {
  std::array<char, 256> text{};
  std::size_t size = text.size();
  if (sysctlbyname(name, text.data(), &size, nullptr, 0) != 0) {
    return "unknown";
  }
  return text.data();
}

struct plan_record {
  std::int32_t kind = fbench_c2c;
  api interface = api::interleaved;
  std::size_t n = 0; // logical length
  std::size_t batch = 1;
  std::size_t in_dist = 0;  // in doubles
  std::size_t out_dist = 0; // in doubles
  vDSP_DFT_Interleaved_SetupD interleaved = nullptr;
  vDSP_DFT_SetupD split = nullptr;
  // Split-complex scratch for API=dft (one row); the harness executes one plan at a time.
  mutable std::vector<double> re;
  mutable std::vector<double> im;

  plan_record() = default;
  plan_record(const plan_record&) = delete;
  plan_record& operator=(const plan_record&) = delete;
  plan_record(plan_record&&) = delete;
  plan_record& operator=(plan_record&&) = delete;
  ~plan_record() {
    if (interleaved != nullptr) {
      vDSP_DFT_Interleaved_DestroySetupD(interleaved);
    }
    if (split != nullptr) {
      vDSP_DFT_DestroySetupD(split);
    }
  }
};

DSPDoubleComplex* as_complex(double* p) { return reinterpret_cast<DSPDoubleComplex*>(p); }

// r2c in vDSP's packed format at `out` (h complex values) becomes FFTW's n/2+1 values.
void unpack_real(double* out, std::size_t h) {
  const double nyquist = out[1];
  out[1] = 0.0;
  out[2 * h] = nyquist;
  out[2 * h + 1] = 0.0;
  const double half = 0.5;
  vDSP_vsmulD(out, 1, &half, out, 1, 2 * h + 2);
}

void run_interleaved(const plan_record& r, double* in, double* out) {
  const std::size_t h = r.n / 2;
  for (std::size_t b = 0; b < r.batch; ++b) {
    double* x = in + b * r.in_dist;
    double* y = out + b * r.out_dist;
    if (r.kind == fbench_c2r) {
      x[1] = x[2 * h]; // X[n/2] into the packed slot
    }
    vDSP_DFT_Interleaved_ExecuteD(r.interleaved, as_complex(x), as_complex(y));
    if (r.kind == fbench_r2c) {
      unpack_real(y, h);
    }
  }
}

void run_split(const plan_record& r, double* in, double* out) {
  const DSPDoubleSplitComplex z{r.re.data(), r.im.data()};
  const std::size_t h = r.kind == fbench_c2c ? r.n : r.n / 2;
  for (std::size_t b = 0; b < r.batch; ++b) {
    double* x = in + b * r.in_dist;
    double* y = out + b * r.out_dist;
    vDSP_ctozD(as_complex(x), 2, &z, 1, h);
    if (r.kind == fbench_c2r) {
      z.imagp[0] = x[2 * h];
    }
    vDSP_DFT_ExecuteD(r.split, z.realp, z.imagp, z.realp, z.imagp);
    if (r.kind == fbench_r2c) {
      const double nyquist = z.imagp[0];
      z.imagp[0] = 0.0;
      const double half = 0.5;
      vDSP_vsmulD(z.realp, 1, &half, z.realp, 1, h);
      vDSP_vsmulD(z.imagp, 1, &half, z.imagp, 1, h);
      vDSP_ztocD(&z, 1, as_complex(y), 2, h);
      y[2 * h] = half * nyquist;
      y[2 * h + 1] = 0.0;
    } else {
      vDSP_ztocD(&z, 1, as_complex(y), 2, h);
    }
  }
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  backend_state& s = state();
  for (const auto& [key, val] : fbench::backends::parse_options(options)) {
    if (key == "API" && (val == "interleaved" || val == "dft")) {
      s.interface = val == "dft" ? api::dft : api::interleaved;
    } else {
      fbench::backends::say(message, message_size,
                            "Accelerate options: API=interleaved|dft (got " + key + ")");
      return fbench_error;
    }
  }
  // Accelerate lives in the dyld shared cache, so its image path does not exist on disk:
  // it is reported as loaded_image, not as loaded_from (which the harness checks on disk).
  s.description = fbench::json()
                      .str("library", "accelerate")
                      .str("version", "macOS " + sysctl_text("kern.osproductversion") + " (" +
                                          sysctl_text("kern.osversion") + ")")
                      .str("isa", "chosen by Accelerate")
                      .str("cpu", sysctl_text("machdep.cpu.brand_string"))
                      .str("precision", "double")
                      .raw("supports", "[]")
                      .raw("partial", R"(["c2c","r2c","c2r"])")
                      .str("api", s.interface == api::dft ? "dft" : "interleaved")
                      .str("loaded_image", fbench::backends::defining_file("vDSP_DFT_ExecuteD"))
                      .inner();
  return fbench_ok;
}

void fbench_end() {}

const char* fbench_describe() { return state().description.c_str(); }

std::int32_t fbench_create(const fbench_case* c, void* /*in*/, void* /*out*/, void** plan,
                           char* message, std::size_t message_size) {
  if (c->rank != 1) {
    fbench::backends::say(message, message_size, "vDSP has no general multidimensional DFT");
    return fbench_unsupported;
  }
  auto r = std::make_unique<plan_record>();
  r->kind = c->kind;
  r->interface = state().interface;
  r->n = static_cast<std::size_t>(c->dims[0]);
  r->batch = static_cast<std::size_t>(c->batch);
  // Distances in doubles.
  r->in_dist = static_cast<std::size_t>(c->in_dist) * (c->kind == fbench_r2c ? 1 : 2);
  r->out_dist = static_cast<std::size_t>(c->out_dist) * (c->kind == fbench_c2r ? 1 : 2);
  const bool real = c->kind != fbench_c2c;
  if (real && r->n % 2 != 0) {
    fbench::backends::say(message, message_size, "vDSP's real DFTs need an even length");
    return fbench_unsupported;
  }
  const vDSP_DFT_Direction direction = c->sign < 0 ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;
  if (r->interface == api::interleaved) {
    r->interleaved = vDSP_DFT_Interleaved_CreateSetupD(
        nullptr, real ? r->n / 2 : r->n, direction,
        real ? vDSP_DFT_Interleaved_RealtoComplex : vDSP_DFT_Interleaved_ComplextoComplex);
  } else {
    r->split = real ? vDSP_DFT_zrop_CreateSetupD(nullptr, r->n, direction)
                    : vDSP_DFT_zop_CreateSetupD(nullptr, r->n, direction);
    const std::size_t h = real ? r->n / 2 : r->n;
    r->re.resize(h);
    r->im.resize(h);
  }
  if (r->interleaved == nullptr && r->split == nullptr) {
    fbench::backends::say(message, message_size,
                          "vDSP has no DFT of length " + std::to_string(r->n));
    return fbench_unsupported;
  }
  *plan = r.release();
  return fbench_ok;
}

void fbench_execute(const void* plan, void* in, void* out) {
  const auto* r = static_cast<const plan_record*>(plan);
  if (r->interface == api::interleaved) {
    run_interleaved(*r, static_cast<double*>(in), static_cast<double*>(out));
  } else {
    run_split(*r, static_cast<double*>(in), static_cast<double*>(out));
  }
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
}
}
