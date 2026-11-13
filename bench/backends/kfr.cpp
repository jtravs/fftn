// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// fbench backend for KFR's DFT (double precision), built only when FFTN_BENCH_KFR_ROOT
// names an installation. KFR is GPL-2.0-or-later (or commercial): this executable is for our
// own measurements and is never distributed (AGENTS.md section 10).
//
// Built as its best users build it: KFR's own CMake build in Release with Clang, with its
// run-time dispatch over several x86 instruction sets (KFR_ENABLE_MULTIARCH, the default on
// x86; AArch64 has the single NEON target), installed and linked as kfr_dft
// (bench/README.md has the commands). The plans are KFR's: dft_plan for complex rows,
// dft_plan_real for real rows (packing CCs, which is FFTW's n/2+1 layout), and
// dft_plan_md / dft_plan_md_real for planes and volumes, each with a scratch buffer of the
// plan's temp_size allocated at plan time. KFR has no batched or strided interface, so a
// batch is a loop over transforms. Signs and scaling are FFTW's (forward e^{-i...}, no
// scaling), so nothing is converted.
//
// What is timed beyond KFR's own code: KFR's multidimensional real transforms take
// contiguous real arrays, so an in-place real plane or volume (whose real rows are padded to
// 2(n/2+1), FFTW's layout) goes through a contiguous copy of the real array (r2c: before,
// c2r: after). Rows, complex transforms and out-of-place real ones run on the caller's
// arrays directly.
//
// Options: none.

#include "backends/common.hpp"
#include "harness/backend.hpp"
#include "harness/json.hpp"

#include <kfr/dft.hpp>
#include <kfr/version.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

using cd = kfr::complex<double>;

std::string& description() {
  static std::string text;
  return text;
}

struct plan_record {
  std::int32_t kind = fbench_c2c;
  bool inverse = false;
  int rank = 1;
  std::size_t batch = 1;
  std::size_t in_dist = 0;  // in elements of the input type
  std::size_t out_dist = 0; // in elements of the output type
  std::size_t n_last = 0;   // logical length of the last dimension
  std::size_t rows = 1;     // real rows of one transform (product of the leading dims)
  std::size_t padded = 0;   // physical length of a real row (in-place real: 2(n/2+1))
  std::unique_ptr<kfr::dft_plan<double>> complex_row;
  std::unique_ptr<kfr::dft_plan_real<double>> real_row;
  std::unique_ptr<kfr::dft_plan_md<double>> complex_md;
  std::unique_ptr<kfr::dft_plan_md_real<double>> real_md;
  mutable kfr::univector<kfr::u8> temp;
  mutable std::vector<double> contiguous; // in-place real planes and volumes only
};

// Copies `rows` rows of `n` doubles between arrays whose rows are `from_stride` and
// `to_stride` doubles apart.
void copy_rows(const double* from, std::size_t from_stride, double* to, std::size_t to_stride,
               std::size_t rows, std::size_t n) {
  for (std::size_t i = 0; i < rows; ++i) {
    std::memcpy(to + i * to_stride, from + i * from_stride, n * sizeof(double));
  }
}

void run_c2c(const plan_record& r, const cd* in, cd* out) {
  for (std::size_t b = 0; b < r.batch; ++b) {
    if (r.rank == 1) {
      r.complex_row->execute(out + b * r.out_dist, in + b * r.in_dist, r.temp.data(), r.inverse);
    } else {
      r.complex_md->execute(out + b * r.out_dist, in + b * r.in_dist, r.temp.data(), r.inverse);
    }
  }
}

void run_r2c(const plan_record& r, const double* in, cd* out) {
  for (std::size_t b = 0; b < r.batch; ++b) {
    const double* x = in + b * r.in_dist;
    cd* y = out + b * r.out_dist;
    if (r.rank == 1) {
      r.real_row->execute(y, x, r.temp.data(), kfr::cdirect_t{});
    } else if (r.padded == r.n_last) {
      r.real_md->execute(y, x, r.temp.data(), kfr::cdirect_t{});
    } else {
      copy_rows(x, r.padded, r.contiguous.data(), r.n_last, r.rows, r.n_last);
      r.real_md->execute(y, r.contiguous.data(), r.temp.data(), kfr::cdirect_t{});
    }
  }
}

void run_c2r(const plan_record& r, const cd* in, double* out) {
  for (std::size_t b = 0; b < r.batch; ++b) {
    const cd* x = in + b * r.in_dist;
    double* y = out + b * r.out_dist;
    if (r.rank == 1) {
      r.real_row->execute(y, x, r.temp.data(), kfr::cinvert_t{});
    } else if (r.padded == r.n_last) {
      r.real_md->execute(y, x, r.temp.data(), kfr::cinvert_t{});
    } else {
      r.real_md->execute(r.contiguous.data(), x, r.temp.data(), kfr::cinvert_t{});
      copy_rows(r.contiguous.data(), r.n_last, y, r.padded, r.rows, r.n_last);
    }
  }
}

} // namespace

extern "C" {

std::int32_t fbench_begin(const char* options, char* message, std::size_t message_size) {
  if (!fbench::backends::parse_options(options).empty()) {
    fbench::backends::say(message, message_size, std::string("KFR has no options: ") + options);
    return fbench_error;
  }
  description() = fbench::json()
                      .str("library", "kfr")
                      .str("version", kfr::version_string)
                      .str("isa", kfr::cpu_runtime())
                      .str("archs", FBENCH_KFR_ARCHS)
                      .str("precision", "double")
                      .raw("supports", R"(["c2c","r2c","c2r"])")
                      .str("expected_root", FBENCH_KFR_ROOT)
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
    r->inverse = c->sign > 0;
    r->rank = c->rank;
    r->batch = static_cast<std::size_t>(c->batch);
    r->in_dist = static_cast<std::size_t>(c->in_dist);
    r->out_dist = static_cast<std::size_t>(c->out_dist);
    r->n_last = static_cast<std::size_t>(c->dims[c->rank - 1]);
    r->padded = static_cast<std::size_t>(c->kind == fbench_r2c   ? c->in_last
                                         : c->kind == fbench_c2r ? c->out_last
                                                                 : c->in_last);
    std::vector<kfr::index_t> dims;
    for (int d = 0; d < c->rank; ++d) {
      dims.push_back(static_cast<kfr::index_t>(c->dims[d]));
      if (d < c->rank - 1) {
        r->rows *= static_cast<std::size_t>(c->dims[d]);
      }
    }
    const kfr::shape<kfr::dynamic_shape> size(dims.begin(), dims.end());
    std::size_t temp_size = 0;
    if (c->kind == fbench_c2c && c->rank == 1) {
      r->complex_row = std::make_unique<kfr::dft_plan<double>>(r->n_last);
      temp_size = r->complex_row->temp_size;
    } else if (c->kind == fbench_c2c) {
      r->complex_md = std::make_unique<kfr::dft_plan_md<double>>(size);
      temp_size = r->complex_md->temp_size;
    } else if (c->rank == 1) {
      r->real_row =
          std::make_unique<kfr::dft_plan_real<double>>(r->n_last, kfr::dft_pack_format::CCs);
      temp_size = r->real_row->temp_size;
    } else {
      r->real_md = std::make_unique<kfr::dft_plan_md_real<double>>(size);
      temp_size = r->real_md->temp_size;
      if (r->padded != r->n_last) {
        r->contiguous.resize(r->rows * r->n_last);
      }
    }
    r->temp.resize(temp_size);
    *plan = r.release();
    return fbench_ok;
  } catch (const std::exception& e) {
    fbench::backends::say(message, message_size, std::string("KFR: ") + e.what());
    return fbench_error;
  }
}

void fbench_execute(const void* plan, void* in, void* out) {
  const auto* r = static_cast<const plan_record*>(plan);
  if (r->kind == fbench_c2c) {
    run_c2c(*r, static_cast<const cd*>(in), static_cast<cd*>(out));
  } else if (r->kind == fbench_r2c) {
    run_r2c(*r, static_cast<const double*>(in), static_cast<cd*>(out));
  } else {
    run_c2r(*r, static_cast<const cd*>(in), static_cast<double*>(out));
  }
}

void fbench_destroy(void* plan) {
  const std::unique_ptr<plan_record> r(static_cast<plan_record*>(plan));
}
}
