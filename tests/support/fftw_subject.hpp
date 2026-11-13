// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// FFTW 3.3.11 as a subject (DESIGN.md 15.1, oracle 3): an independent implementation for
// large sizes and every shape. Plans are made with FFTW_ESTIMATE and FFTW_UNALIGNED through
// the guru64 interface, then executed on the caller's arrays with the new-array execute
// functions; c2r plans preserve their input, as FFTN's do by default.
//
// FFTW is GPL: only test executables link it (cmake/FftnTestDeps.cmake). Its planner is not
// thread-safe, so a subject must be prepared from one thread at a time. When the build has
// no FFTW (FFTN_TEST_WITH_FFTW=OFF), fftw_subject() returns nothing.

#pragma once

#include "support/transform_under_test.hpp"

#include <optional>

#if FFTN_TEST_HAVE_FFTW
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#endif

namespace fftn::test {

#if FFTN_TEST_HAVE_FFTW

namespace detail {

template <class T> struct fftw_api;

template <> struct fftw_api<double> {
  using plan_t = fftw_plan;
  using complex_t = fftw_complex;
  static plan_t dft(int rank, const fftw_iodim64* dims, int howmany_rank,
                    const fftw_iodim64* howmany, complex_t* in, complex_t* out, int sign,
                    unsigned flags) {
    return fftw_plan_guru64_dft(rank, dims, howmany_rank, howmany, in, out, sign, flags);
  }
  static plan_t r2c(int rank, const fftw_iodim64* dims, int howmany_rank,
                    const fftw_iodim64* howmany, double* in, complex_t* out, unsigned flags) {
    return fftw_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany, in, out, flags);
  }
  static plan_t c2r(int rank, const fftw_iodim64* dims, int howmany_rank,
                    const fftw_iodim64* howmany, complex_t* in, double* out, unsigned flags) {
    return fftw_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany, in, out, flags);
  }
  static void execute(plan_t p, complex_t* in, complex_t* out) { fftw_execute_dft(p, in, out); }
  static void execute(plan_t p, double* in, complex_t* out) { fftw_execute_dft_r2c(p, in, out); }
  static void execute(plan_t p, complex_t* in, double* out) { fftw_execute_dft_c2r(p, in, out); }
  static void destroy(plan_t p) { fftw_destroy_plan(p); }
};

template <> struct fftw_api<float> {
  using plan_t = fftwf_plan;
  using complex_t = fftwf_complex;
  static plan_t dft(int rank, const fftw_iodim64* dims, int howmany_rank,
                    const fftw_iodim64* howmany, complex_t* in, complex_t* out, int sign,
                    unsigned flags) {
    return fftwf_plan_guru64_dft(rank, dims, howmany_rank, howmany, in, out, sign, flags);
  }
  static plan_t r2c(int rank, const fftw_iodim64* dims, int howmany_rank,
                    const fftw_iodim64* howmany, float* in, complex_t* out, unsigned flags) {
    return fftwf_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany, in, out, flags);
  }
  static plan_t c2r(int rank, const fftw_iodim64* dims, int howmany_rank,
                    const fftw_iodim64* howmany, complex_t* in, float* out, unsigned flags) {
    return fftwf_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany, in, out, flags);
  }
  static void execute(plan_t p, complex_t* in, complex_t* out) { fftwf_execute_dft(p, in, out); }
  static void execute(plan_t p, float* in, complex_t* out) { fftwf_execute_dft_r2c(p, in, out); }
  static void execute(plan_t p, complex_t* in, float* out) { fftwf_execute_dft_c2r(p, in, out); }
  static void destroy(plan_t p) { fftwf_destroy_plan(p); }
};

// Scratch arrays to plan with: FFTW_ESTIMATE never touches them, but a plan records whether
// it is in place, so they must alias exactly when the real call's arrays do.
template <class T> struct plan_arrays {
  std::vector<std::complex<T>> in;
  std::vector<std::complex<T>> out;
  std::complex<T>* in_ptr = nullptr;
  std::complex<T>* out_ptr = nullptr;

  plan_arrays(std::ptrdiff_t in_lo, std::ptrdiff_t in_hi, std::ptrdiff_t out_lo,
              std::ptrdiff_t out_hi, placement place) {
    const std::ptrdiff_t lo = std::min(in_lo, out_lo);
    const std::ptrdiff_t hi = std::max(in_hi, out_hi);
    in.resize(static_cast<std::size_t>(hi - lo + 1));
    in_ptr = in.data() - lo;
    if (place == placement::in_place) {
      out_ptr = in_ptr;
    } else {
      out.resize(in.size());
      out_ptr = out.data() - lo;
    }
  }
};

// The lowest and highest element offsets of a side, in units of the side's own element.
inline std::pair<std::ptrdiff_t, std::ptrdiff_t>
span_of(std::initializer_list<std::pair<std::size_t, std::ptrdiff_t>> dims) {
  std::ptrdiff_t lo = 0;
  std::ptrdiff_t hi = 0;
  for (const auto& [n, stride] : dims) {
    if (n == 0) {
      continue;
    }
    const std::ptrdiff_t reach = static_cast<std::ptrdiff_t>(n - 1) * stride;
    (reach < 0 ? lo : hi) += reach;
  }
  return {lo, hi};
}

inline fftw_iodim64 iodim(std::size_t n, std::ptrdiff_t is, std::ptrdiff_t os) {
  return {.n = static_cast<std::ptrdiff_t>(n), .is = is, .os = os};
}

template <class Plan, class Destroy> auto own(Plan p, Destroy destroy) {
  if (p == nullptr) {
    throw std::runtime_error("FFTW could not plan this layout");
  }
  return std::shared_ptr<std::remove_pointer_t<Plan>>(p, destroy);
}

constexpr unsigned plan_flags = FFTW_ESTIMATE | FFTW_UNALIGNED;

// Call objects are cached per direction and layout for the subject's lifetime: the suites
// run many inputs through few layouts, and FFTW's planning, not its execution, would
// dominate their run time under the sanitizers. Not thread-safe, like FFTW's planner.
inline std::string key_of(int sign, const rows& r) {
  std::ostringstream k;
  k << sign << ' ' << r.n << ' ' << r.count << ' ' << r.in_stride << ' ' << r.in_dist << ' '
    << r.out_stride << ' ' << r.out_dist << ' ' << static_cast<int>(r.place);
  return k.str();
}

inline std::string key_of(int sign, const planes& p) {
  std::ostringstream k;
  k << sign << ' ' << p.ny << ' ' << p.nx << ' ' << p.count << ' ' << p.in_y_stride << ' '
    << p.in_x_stride << ' ' << p.in_dist << ' ' << p.out_y_stride << ' ' << p.out_x_stride << ' '
    << p.out_dist << ' ' << static_cast<int>(p.place);
  return k.str();
}

template <class Call, class Layout, class Factory> auto cached(Factory make) {
  const auto calls = std::make_shared<std::map<std::string, Call>>();
  return [calls, make = std::move(make)](direction d, const Layout& layout) -> Call {
    const std::string key = key_of(sign_of(d), layout);
    if (const auto it = calls->find(key); it != calls->end()) {
      return it->second;
    }
    return calls->emplace(key, make(d, layout)).first->second;
  };
}

} // namespace detail

template <class T> std::optional<subject<T>> fftw_subject() {
  using api = detail::fftw_api<T>;
  using fc = api::complex_t;
  using cplx = std::complex<T>;
  subject<T> s;
  s.name = "fftw";
  s.c2c = [](direction d, const rows& r) -> c2c_call<T> {
    const auto [ilo, ihi] = detail::span_of({{r.n, r.in_stride}, {r.count, r.in_dist}});
    const auto [olo, ohi] = detail::span_of({{r.n, r.out_stride}, {r.count, r.out_dist}});
    const detail::plan_arrays<T> a(ilo, ihi, olo, ohi, r.place);
    const fftw_iodim64 dim = detail::iodim(r.n, r.in_stride, r.out_stride);
    const fftw_iodim64 many = detail::iodim(r.count, r.in_dist, r.out_dist);
    const auto plan =
        detail::own(api::dft(1, &dim, 1, &many, reinterpret_cast<fc*>(a.in_ptr),
                             reinterpret_cast<fc*>(a.out_ptr), sign_of(d), detail::plan_flags),
                    &api::destroy);
    return [plan](const cplx* in, cplx* out) {
      api::execute(plan.get(), reinterpret_cast<fc*>(const_cast<cplx*>(in)),
                   reinterpret_cast<fc*>(out));
    };
  };
  s.r2c = [](const rows& r) -> r2c_call<T> {
    const std::size_t h = r.n / 2 + 1;
    // Offsets of the real side, converted to complex units for the planning arrays.
    const auto [ilo, ihi] = detail::span_of({{r.n, r.in_stride}, {r.count, r.in_dist}});
    const auto [olo, ohi] = detail::span_of({{h, r.out_stride}, {r.count, r.out_dist}});
    const detail::plan_arrays<T> a(ilo / 2 - 1, ihi / 2 + 1, olo, ohi, r.place);
    const fftw_iodim64 dim = detail::iodim(r.n, r.in_stride, r.out_stride);
    const fftw_iodim64 many = detail::iodim(r.count, r.in_dist, r.out_dist);
    const auto plan = detail::own(api::r2c(1, &dim, 1, &many, reinterpret_cast<T*>(a.in_ptr),
                                           reinterpret_cast<fc*>(a.out_ptr), detail::plan_flags),
                                  &api::destroy);
    return [plan](const T* in, cplx* out) {
      api::execute(plan.get(), const_cast<T*>(in), reinterpret_cast<fc*>(out));
    };
  };
  s.c2r = [](const rows& r) -> c2r_call<T> {
    const std::size_t h = r.n / 2 + 1;
    const auto [ilo, ihi] = detail::span_of({{h, r.in_stride}, {r.count, r.in_dist}});
    const auto [olo, ohi] = detail::span_of({{r.n, r.out_stride}, {r.count, r.out_dist}});
    const detail::plan_arrays<T> a(ilo, ihi, olo / 2 - 1, ohi / 2 + 1, r.place);
    const fftw_iodim64 dim = detail::iodim(r.n, r.in_stride, r.out_stride);
    const fftw_iodim64 many = detail::iodim(r.count, r.in_dist, r.out_dist);
    const unsigned preserve = r.place == placement::out_of_place ? (FFTW_PRESERVE_INPUT) : 0U;
    const auto plan =
        detail::own(api::c2r(1, &dim, 1, &many, reinterpret_cast<fc*>(a.in_ptr),
                             reinterpret_cast<T*>(a.out_ptr), detail::plan_flags | preserve),
                    &api::destroy);
    return [plan](const cplx* in, T* out) {
      api::execute(plan.get(), reinterpret_cast<fc*>(const_cast<cplx*>(in)), out);
    };
  };
  s.c2c_planes = [](direction d, const planes& p) -> c2c_call<T> {
    const auto [ilo, ihi] =
        detail::span_of({{p.ny, p.in_y_stride}, {p.nx, p.in_x_stride}, {p.count, p.in_dist}});
    const auto [olo, ohi] =
        detail::span_of({{p.ny, p.out_y_stride}, {p.nx, p.out_x_stride}, {p.count, p.out_dist}});
    const detail::plan_arrays<T> a(ilo, ihi, olo, ohi, p.place);
    const std::array<fftw_iodim64, 2> dims{
        detail::iodim(p.ny, p.in_y_stride, p.out_y_stride),
        detail::iodim(p.nx, p.in_x_stride, p.out_x_stride),
    };
    const fftw_iodim64 many = detail::iodim(p.count, p.in_dist, p.out_dist);
    const auto plan =
        detail::own(api::dft(2, dims.data(), 1, &many, reinterpret_cast<fc*>(a.in_ptr),
                             reinterpret_cast<fc*>(a.out_ptr), sign_of(d), detail::plan_flags),
                    &api::destroy);
    return [plan](const cplx* in, cplx* out) {
      api::execute(plan.get(), reinterpret_cast<fc*>(const_cast<cplx*>(in)),
                   reinterpret_cast<fc*>(out));
    };
  };
  s.c2c = detail::cached<c2c_call<T>, rows>(std::move(s.c2c));
  s.c2c_planes = detail::cached<c2c_call<T>, planes>(std::move(s.c2c_planes));
  const auto r2c = detail::cached<r2c_call<T>, rows>(
      [make = std::move(s.r2c)](direction /*d*/, const rows& r) { return make(r); });
  const auto c2r = detail::cached<c2r_call<T>, rows>(
      [make = std::move(s.c2r)](direction /*d*/, const rows& r) { return make(r); });
  s.r2c = [r2c](const rows& r) { return r2c(direction::forward, r); };
  s.c2r = [c2r](const rows& r) { return c2r(direction::backward, r); };
  return s;
}

#else

template <class T> std::optional<subject<T>> fftw_subject() { return std::nullopt; }

#endif

} // namespace fftn::test
