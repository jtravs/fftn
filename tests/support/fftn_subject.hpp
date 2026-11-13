// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// FFTN's own plans as a subject: every factory builds a plan from the layout through the
// public API, and the call object executes it with a workspace of its own. Building the
// call object is plan construction; invoking it is execution, which the harness watches.

#pragma once

#include "support/transform_under_test.hpp"

#include <fftn/fftn.hpp>

#include <complex>
#include <memory>

namespace fftn::test {

namespace detail {

inline fftn::direction to_fftn(direction d) {
  return d == direction::forward ? fftn::direction::forward : fftn::direction::backward;
}

template <class In, class Out> fftn::shape<In, Out> shape_of(direction d, const rows& r) {
  fftn::shape<In, Out> s;
  s.dir = to_fftn(d);
  s.axes.push_back({.n = r.n, .in_stride = r.in_stride, .out_stride = r.out_stride});
  s.batches.push_back({.count = r.count, .in_dist = r.in_dist, .out_dist = r.out_dist});
  return s;
}

template <class In, class Out> auto call_of(const fftn::shape<In, Out>& s, placement place) {
  const auto p = std::make_shared<const fftn::plan<In, Out>>(
      s, fftn::options{.in_place = place == placement::in_place});
  const auto ws = std::make_shared<fftn::workspace>(*p);
  return [p, ws](const In* in, Out* out) { p->execute(in, out, *ws); };
}

} // namespace detail

template <class T> subject<T> fftn_subject() {
  using cplx = std::complex<T>;
  subject<T> s;
  s.name = "fftn";
  s.c2c = [](direction d, const rows& r) -> c2c_call<T> {
    return detail::call_of(detail::shape_of<cplx, cplx>(d, r), r.place);
  };
  s.r2c = [](const rows& r) -> r2c_call<T> {
    return detail::call_of(detail::shape_of<T, cplx>(direction::forward, r), r.place);
  };
  s.c2r = [](const rows& r) -> c2r_call<T> {
    return detail::call_of(detail::shape_of<cplx, T>(direction::backward, r), r.place);
  };
  s.c2c_planes = [](direction d, const planes& p) -> c2c_call<T> {
    fftn::shape<cplx, cplx> sh;
    sh.dir = detail::to_fftn(d);
    sh.axes.push_back({.n = p.ny, .in_stride = p.in_y_stride, .out_stride = p.out_y_stride});
    sh.axes.push_back({.n = p.nx, .in_stride = p.in_x_stride, .out_stride = p.out_x_stride});
    sh.batches.push_back({.count = p.count, .in_dist = p.in_dist, .out_dist = p.out_dist});
    return detail::call_of(sh, p.place);
  };
  return s;
}

} // namespace fftn::test
