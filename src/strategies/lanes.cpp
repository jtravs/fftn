// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The lanes strategy (DESIGN.md 8.5, experiment X7): batches of short rows with one row per
// vector lane. A block of B rows is gathered into a tile (element i of row r at i B + r),
// the row's chain runs on the tile as lane passes with every copy count multiplied by B, and
// the tile is scattered back (src/passes/lanes.inl). The chain is the one the row gets alone
// (chain.cpp): the same radices in the same order, the same split run, the same twiddle
// values and edge operations; a pass that was the row's grouped first pass or its leaf is a
// lane pass here, whose lanes perform the same operations on the same values (the grouped
// pass's per-lane twiddles and the leaf's broadcast loads give each lane what a lane pass
// gives it). So every row's result is bit for bit the one the chain computes for it alone,
// whatever block it falls in (DESIGN.md 15.5), which the tests check.

#include "core/program.hpp"
#include "isa/kernel_table.hpp"
#include "passes/pass_args.hpp"
#include "strategies/chain.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

slot other(slot s) { return s == slot::a ? slot::b : slot::a; }

} // namespace

template <class T> bool chain_builder<T>::lanes_fit(std::size_t n, bool real) const {
  std::size_t len = n;
  if (real) {
    if (n % 2 != 0 || n < 4) {
      return false;
    }
    len = n / 2;
    const real_route how = route(n, true, false);
    if (how != real_route::leaf && how != real_route::fused && how != real_route::separate) {
      return false;
    }
    if (route(n, false, false) == real_route::pairing_only) {
      return false;
    }
  }
  if (len < 2 || bluestein_last(len)) {
    return false;
  }
  for (const std::size_t r : radices(len)) {
    if (!has_codelet(r) || kernels_.passes.codelet[codelet_slot(r)].lane[0] == nullptr) {
      return false;
    }
  }
  return true;
}

template <class T>
void chain_builder<T>::add_lanes_gather(std::size_t elements, endpoint src,
                                        std::ptrdiff_t stride_reals, std::ptrdiff_t dist_reals,
                                        std::size_t block, std::vector<stage>& stages) {
  stage st;
  st.run = kernels_.passes.lanes_gather;
  st.args.n = elements;
  st.args.s = block;
  st.args.src_stride = stride_reals;
  st.args.src_dist = dist_reals;
  st.src = src.where;
  st.dst = slot::a;
  st.units = 1;
  st.name = "lanes gather (" + std::to_string(block) + " rows of " + std::to_string(elements) +
            " into a tile, one row per lane)";
  stages.push_back(std::move(st));
}

template <class T>
slot chain_builder<T>::lanes_chain(std::size_t n, const std::vector<std::size_t>& rad,
                                   std::size_t split, int sign, slot at, const line_edges& edges,
                                   std::size_t block, std::vector<stage>& stages) {
  std::size_t s = 1;
  for (std::size_t i = 0; i < rad.size(); ++i) {
    const slot next = other(at);
    add_pass(n * block, rad, i, s * block, sign, {at, 1}, {next, 1}, edges, stages, block,
             layout_of(i, split, rad.size()));
    stages.back().chunks = 1;
    at = next;
    s *= rad[i];
  }
  return at;
}

template <class T>
void chain_builder<T>::add_lanes_scatter(std::size_t elements, slot at, endpoint dst,
                                         std::ptrdiff_t stride_reals, std::ptrdiff_t dist_reals,
                                         std::size_t block, std::vector<stage>& stages) {
  stage st;
  st.run = kernels_.passes.lanes_scatter;
  st.args.n = elements;
  st.args.s = block;
  st.args.dst_stride = stride_reals;
  st.args.dst_dist = dist_reals;
  st.src = at;
  st.dst = dst.where;
  st.units = 1;
  st.name = "lanes scatter (the tile to " + std::to_string(block) + " rows of " +
            std::to_string(elements) + ")";
  stages.push_back(std::move(st));
}

template <class T>
std::size_t chain_builder<T>::c2c_lanes(std::size_t n, int sign, endpoint src,
                                        std::ptrdiff_t src_dist, endpoint dst,
                                        std::ptrdiff_t dst_dist, const line_edges& edges,
                                        std::size_t block, std::vector<stage>& stages) {
  const std::vector<std::size_t> rad = radices(n);
  // c2c_line's split run: a line with a stride other than 1 is copied on C > 1, so only C = 1
  // sees the destination's stride.
  const std::size_t split =
      split_from(rad, false, kernels_.complex_lanes > 1 ? 1 : dst.stride);
  add_lanes_gather(n, src, 2 * src.stride, 2 * src_dist, block, stages);
  const slot at = lanes_chain(n, rad, split, sign, slot::a, edges, block, stages);
  add_lanes_scatter(n, at, dst, 2 * dst.stride, 2 * dst_dist, block, stages);
  return n * block;
}

template <class T>
std::size_t chain_builder<T>::r2c_lanes(std::size_t n, endpoint src, std::ptrdiff_t src_dist,
                                        endpoint dst, std::ptrdiff_t dst_dist, double scale,
                                        std::size_t block, std::vector<stage>& stages) {
  const std::size_t h = n / 2;
  const real_route how = route(n, true, false);
  // r2c_line's radices and split run: the fused route reorders the radices and has no split
  // run; the separate route is c2c_line's; a leaf is one pass.
  const std::vector<std::size_t> rad = how == real_route::fused ? real_radices(h, true) : radices(h);
  const std::size_t split = how == real_route::separate ? split_from(rad, false, 1) : rad.size();
  // The reals of a row read as h complex pairs.
  add_lanes_gather(h, src, 2, src_dist, block, stages);
  const slot z = lanes_chain(h, rad, split, -1, slot::a, line_edges(), block, stages);
  stage st;
  st.run = kernels_.passes.lanes_r2c_pair[scale != 1.0 ? 1 : 0];
  st.args.n = n;
  st.args.s = block;
  st.args.twiddles = leaf_pairing_twiddles(n);
  st.args.scale = scale;
  st.src = z;
  st.dst = other(z);
  st.units = 1;
  st.name = std::string("lanes r2c pairing (") + std::to_string(block) + " rows" +
            (scale != 1.0 ? "; out: scale)" : ")");
  stages.push_back(std::move(st));
  add_lanes_scatter(h + 1, other(z), dst, 2 * dst.stride, 2 * dst_dist, block, stages);
  return (h + 1) * block;
}

template <class T>
std::size_t chain_builder<T>::c2r_lanes(std::size_t n, endpoint src, std::ptrdiff_t src_dist,
                                        endpoint dst, std::ptrdiff_t dst_dist, double scale,
                                        std::size_t block, std::vector<stage>& stages) {
  const std::size_t h = n / 2;
  const real_route how = route(n, false, false);
  // c2r_line's radices and split run (chain_tail's and c2c_line's are the same here).
  const std::vector<std::size_t> rad =
      how == real_route::fused ? real_radices(h, false) : radices(h);
  const std::size_t split = rad.size() > 1 ? split_from(rad, false, 1) : rad.size();
  line_edges edges;
  if (scale != 1.0) {
    edges.out = edge_op::scale;
    edges.scale = scale;
  }
  add_lanes_gather(h + 1, src, 2 * src.stride, 2 * src_dist, block, stages);
  stage st;
  st.run = kernels_.passes.lanes_c2r_pair;
  st.args.n = n;
  st.args.s = block;
  st.args.twiddles = leaf_pairing_twiddles(n);
  st.src = slot::a;
  st.dst = slot::b;
  st.units = 1;
  st.name = "lanes c2r pairing (" + std::to_string(block) + " rows)";
  stages.push_back(std::move(st));
  const slot z = lanes_chain(h, rad, split, +1, slot::b, edges, block, stages);
  // The h complex values of a row written as its n reals.
  add_lanes_scatter(h, z, dst, 2, dst_dist, block, stages);
  return (h + 1) * block;
}

// chain.hpp declares the class's instantiations extern (chain.cpp makes them), which covers
// only the members defined there: every member defined here is instantiated here.
template void chain_builder<double>::add_lanes_gather(std::size_t, endpoint, std::ptrdiff_t,
                                                   std::ptrdiff_t, std::size_t,
                                                   std::vector<stage>&);
template slot chain_builder<double>::lanes_chain(std::size_t, const std::vector<std::size_t>&,
                                              std::size_t, int, slot, const line_edges&,
                                              std::size_t, std::vector<stage>&);
template void chain_builder<double>::add_lanes_scatter(std::size_t, slot, endpoint, std::ptrdiff_t,
                                                    std::ptrdiff_t, std::size_t,
                                                    std::vector<stage>&);
template void chain_builder<float>::add_lanes_gather(std::size_t, endpoint, std::ptrdiff_t,
                                                   std::ptrdiff_t, std::size_t,
                                                   std::vector<stage>&);
template slot chain_builder<float>::lanes_chain(std::size_t, const std::vector<std::size_t>&,
                                              std::size_t, int, slot, const line_edges&,
                                              std::size_t, std::vector<stage>&);
template void chain_builder<float>::add_lanes_scatter(std::size_t, slot, endpoint, std::ptrdiff_t,
                                                    std::ptrdiff_t, std::size_t,
                                                    std::vector<stage>&);
template bool chain_builder<double>::lanes_fit(std::size_t, bool) const;
template bool chain_builder<float>::lanes_fit(std::size_t, bool) const;
template std::size_t chain_builder<double>::c2c_lanes(std::size_t, int, endpoint, std::ptrdiff_t,
                                                      endpoint, std::ptrdiff_t, const line_edges&,
                                                      std::size_t, std::vector<stage>&);
template std::size_t chain_builder<float>::c2c_lanes(std::size_t, int, endpoint, std::ptrdiff_t,
                                                     endpoint, std::ptrdiff_t, const line_edges&,
                                                     std::size_t, std::vector<stage>&);
template std::size_t chain_builder<double>::r2c_lanes(std::size_t, endpoint, std::ptrdiff_t,
                                                      endpoint, std::ptrdiff_t, double,
                                                      std::size_t, std::vector<stage>&);
template std::size_t chain_builder<float>::r2c_lanes(std::size_t, endpoint, std::ptrdiff_t,
                                                     endpoint, std::ptrdiff_t, double, std::size_t,
                                                     std::vector<stage>&);
template std::size_t chain_builder<double>::c2r_lanes(std::size_t, endpoint, std::ptrdiff_t,
                                                      endpoint, std::ptrdiff_t, double,
                                                      std::size_t, std::vector<stage>&);
template std::size_t chain_builder<float>::c2r_lanes(std::size_t, endpoint, std::ptrdiff_t,
                                                     endpoint, std::ptrdiff_t, double, std::size_t,
                                                     std::vector<stage>&);

} // namespace fftn::inline v0::detail
