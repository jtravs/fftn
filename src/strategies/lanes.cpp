// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The lanes strategy (DESIGN.md 8.5, experiment X7): batches of short rows with one row per
// vector lane. A block of B rows is laid out as a tile (element i of row r at i B + r), the
// row's chain runs on it as lane passes with every copy count multiplied by B, and the tile
// goes back to the rows (src/passes/lanes.inl). Where a run of C rows is one vector access
// (C = 1, or rows one complex value apart) the first pass reads the rows and the last writes
// them; otherwise a gather (C x C transposes) builds the tile and a scatter takes it apart.
//
// The chain is the one the row gets alone (chain.cpp): the same radices in the same order,
// the same split run, the same twiddle values and edge operations; a pass that was the row's
// grouped first pass or its leaf is a lane pass here, whose lanes perform the same
// operations on the same values (the grouped pass's per-lane twiddles and the leaf's
// broadcast loads give each lane what a lane pass gives it). So every row's result is bit
// for bit the one the chain computes for it alone, whatever block it falls in (DESIGN.md
// 15.5), which the tests check. Real rows pair their bins on the tile with the arithmetic
// of every other route's pairing; a real row whose complex side is strided gets the bits
// of the contiguous row (the chain's strided route computes differently).

#include "core/program.hpp"
#include "isa/kernel_table.hpp"
#include "passes/pass_args.hpp"
#include "strategies/chain.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

slot other(slot s) { return s == slot::a ? slot::b : slot::a; }

bool is_scratch(slot s) { return s == slot::a || s == slot::b; }

std::size_t direction_of(int sign) { return sign < 0 ? 0 : 1; }

const char* direction_text(int sign) { return sign < 0 ? "forward" : "backward"; }

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
  }
  if (len < 2 || bluestein_last(len)) {
    return false;
  }
  return std::ranges::all_of(radices(len), [this](std::size_t r) {
    return has_codelet(r) && kernels_.passes.codelet[codelet_slot(r)]->lane[0] != nullptr;
  });
}

template <class T>
void chain_builder<T>::add_lanes_copy(bool gather, std::size_t elements, lane_rows rows, slot tile,
                                      std::size_t block, std::vector<stage>& stages, bool columns) {
  stage st;
  st.args.columns_inner = columns;
  st.run = gather ? kernels_.passes.lanes_gather : kernels_.passes.lanes_scatter;
  st.args.n = elements;
  st.args.block = static_cast<std::uint32_t>(block);
  if (gather) {
    st.args.src_stride = rows.stride;
    st.args.src_dist = rows.dist;
    st.src = rows.where;
    st.dst = tile;
    st.name = "lanes gather (" + std::to_string(block) + " rows of " + std::to_string(elements) +
              " into a tile, one row per lane)";
  } else {
    st.args.dst_stride = rows.stride;
    st.args.dst_dist = rows.dist;
    st.src = tile;
    st.dst = rows.where;
    st.name = "lanes scatter (the tile to " + std::to_string(block) + " rows of " +
              std::to_string(elements) + ")";
  }
  st.units = 1;
  stages.push_back(std::move(st));
}

// A destination in scratch is a tile, in whichever scratch slot the chain ends: the caller
// reads the slot from the last stage.
template <class T>
void chain_builder<T>::lanes_body(std::size_t n, const std::vector<std::size_t>& rad,
                                  std::size_t split, int sign, lane_rows src, lane_rows dst,
                                  const line_edges& edges, std::size_t block,
                                  std::vector<stage>& stages, bool columns) {
  const std::size_t c = kernels_.complex_lanes;
  const std::size_t d = direction_of(sign);
  const std::size_t scaled = edges.out == edge_op::scale ? 1 : 0;
  const auto tile_of = [&](slot where) {
    return lane_rows{.where = where, .stride = static_cast<std::ptrdiff_t>(2 * block), .dist = 2};
  };
  // A run of C rows is one vector access.
  const auto direct = [&](const lane_rows& r) { return c == 1 || r.dist == 2; };
  const auto common = [&](stage& st, std::size_t radix, const lane_rows& from,
                          const lane_rows& to) {
    st.args.radix = radix;
    st.args.n = n * block;
    st.args.block = static_cast<std::uint32_t>(block);
    st.args.scale = edges.scale;
    st.args.src_stride = from.stride;
    st.args.src_dist = from.dist;
    st.args.dst_stride = to.stride;
    st.args.dst_dist = to.dist;
    st.src = from.where;
    st.dst = to.where;
  };
  // `at`: the scratch slot the next tile goes to, never the source's.
  slot at = is_scratch(src.where) ? other(src.where) : slot::a;
  if (!direct(src)) {
    add_lanes_copy(true, n, src, at, block, stages);
    src = tile_of(at);
    at = other(at);
  }
  const std::size_t last = rad.size() - 1;
  // A destination in scratch is the tile the last pass writes (other than its source).
  if (is_scratch(dst.where)) {
    // One pass writes `at`; with more, the last pass writes the slot its source is not in,
    // after the passes have alternated from `at`.
    dst = tile_of(rad.size() % 2 == 0 ? other(at) : at);
  }
  // The last pass writes the destination itself unless it is a split pass or a run of the
  // destination's rows is no vector access.
  const bool to_dst = split > last && direct(dst);
  if (rad.size() == 1) {
    const lane_rows to = to_dst ? dst : tile_of(at);
    stage st;
    st.run = kernels_.passes.codelet[codelet_slot(n)]->lanes_leaf[d][scaled];
    common(st, n, src, to);
    st.units = block / c;
    st.name = "lanes leaf radix-" + std::to_string(n) + " " + direction_text(sign) + " (" +
              std::to_string(block) + " rows" + (scaled != 0 ? "; out: scale)" : ")");
    stages.push_back(std::move(st));
    if (!to_dst) {
      add_lanes_copy(false, n, dst, at, block, stages, columns);
    }
    return;
  }
  {
    // The first pass, reading the rows (or a tile).
    stage st;
    const std::size_t m = n / rad[0];
    st.run = kernels_.passes.codelet[codelet_slot(rad[0])]->lanes_first[d];
    common(st, rad[0], src, tile_of(at));
    st.args.m = m;
    st.args.s = block;
    st.args.columns_inner = columns;
    st.args.twiddles = shared_twiddles(n, rad[0], 1);
    st.units = m * (block / c);
    st.name = "lanes first radix-" + std::to_string(rad[0]) + " " + direction_text(sign) + " (m " +
              std::to_string(m) + ", " + std::to_string(block) + " rows)";
    stages.push_back(std::move(st));
  }
  std::size_t s = rad[0];
  for (std::size_t i = 1; i < (to_dst ? last : rad.size()); ++i) {
    const slot next = other(at);
    add_pass(n * block, rad, i, s * block, sign, {at, 1}, {next, 1},
             i == last ? edges : line_edges(), stages, block, layout_of(i, split, rad.size()));
    stages.back().chunks = 1;
    at = next;
    s *= rad[i];
  }
  if (to_dst) {
    stage st;
    st.run = kernels_.passes.codelet[codelet_slot(rad[last])]->lanes_last[d][scaled];
    common(st, rad[last], tile_of(at), dst);
    st.args.m = 1;
    st.args.s = s * block;
    st.args.columns_inner = columns;
    st.units = s * (block / c);
    st.name = "lanes last radix-" + std::to_string(rad[last]) + " " + direction_text(sign) +
              " (s " + std::to_string(s) + ", " + std::to_string(block) + " rows" +
              (scaled != 0 ? "; out: scale)" : ")");
    stages.push_back(std::move(st));
  } else if (!is_scratch(dst.where)) {
    add_lanes_copy(false, n, dst, at, block, stages, columns);
  }
}

template <class T>
std::size_t chain_builder<T>::c2c_lanes(std::size_t n, int sign, endpoint src,
                                        std::ptrdiff_t src_dist, endpoint dst,
                                        std::ptrdiff_t dst_dist, const line_edges& edges,
                                        std::size_t block, std::vector<stage>& stages, bool columns,
                                        const std::vector<std::size_t>& given) {
  // Where c2c_line takes a pair leaf (X28b), its chain r1 x m with interleaved passes, so
  // that each lane computes what the row alone would.
  const pair_leaf_entry* pair = given.empty() ? pair_for(n, src, dst, edges).entry : nullptr;
  const std::vector<std::size_t> rad = [&]() -> std::vector<std::size_t> {
    if (!given.empty()) {
      return given;
    }
    if (pair != nullptr) {
      return {pair->r1, pair->m};
    }
    return radices(n);
  }();
  // c2c_line's split run: a line with a stride other than 1 is copied on C > 1, so only C = 1
  // sees the destination's stride.
  const std::size_t split =
      pair != nullptr ? rad.size()
                      : split_from(rad, false, kernels_.complex_lanes > 1 ? 1 : dst.stride);
  // A tile in scratch: element i of row r at i B + r.
  const lane_rows from = is_scratch(src.where)
                             ? lane_rows{.where = src.where,
                                         .stride = static_cast<std::ptrdiff_t>(2 * block),
                                         .dist = 2}
                             : lane_rows{src.where, 2 * src.stride, 2 * src_dist};
  lanes_body(n, rad, split, sign, from, {dst.where, 2 * dst.stride, 2 * dst_dist}, edges, block,
             stages, columns);
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
  // Where the separate route's c2c_line takes a pair leaf (X28b), its chain, interleaved.
  const pair_leaf_entry* pair =
      how == real_route::separate ? pair_for(h, {src.where, 1}, {dst.where, 1}, line_edges()).entry
                                  : nullptr;
  const std::vector<std::size_t> rad = [&]() -> std::vector<std::size_t> {
    if (pair != nullptr) {
      return {pair->r1, pair->m};
    }
    return how == real_route::fused ? real_radices(h, true) : radices(h);
  }();
  const std::size_t split =
      how == real_route::separate && pair == nullptr ? split_from(rad, false, 1) : rad.size();
  // The reals of a row read as h complex pairs; the transform ends in a tile.
  lanes_body(h, rad, split, -1, {src.where, 2, src_dist}, {slot::a, 0, 0}, line_edges(), block,
             stages);
  const slot z = stages.back().dst;
  const lane_rows out{dst.where, 2 * dst.stride, 2 * dst_dist};
  // The pairing writes the rows where a run of C of them is one vector store, else a tile.
  const bool direct = kernels_.complex_lanes == 1 || out.dist == 2;
  const lane_rows to = direct ? out
                              : lane_rows{
                                    .where = other(z),
                                    .stride = static_cast<std::ptrdiff_t>(2 * block),
                                    .dist = 2,
                                };
  stage st;
  st.run = kernels_.passes.lanes_r2c_pair[scale != 1.0 ? 1 : 0];
  st.args.n = n;
  st.args.block = static_cast<std::uint32_t>(block);
  st.args.twiddles = leaf_pairing_twiddles(n);
  st.args.scale = scale;
  st.args.dst_stride = to.stride;
  st.args.dst_dist = to.dist;
  st.src = z;
  st.dst = to.where;
  st.units = 1;
  st.name = std::string("lanes r2c pairing (") + std::to_string(block) + " rows" +
            (scale != 1.0 ? "; out: scale)" : ")");
  stages.push_back(std::move(st));
  if (!direct) {
    add_lanes_copy(false, h + 1, out, other(z), block, stages);
  }
  return (h + 1) * block;
}

template <class T>
std::size_t chain_builder<T>::c2r_lanes(std::size_t n, endpoint src, std::ptrdiff_t src_dist,
                                        endpoint dst, std::ptrdiff_t dst_dist, double scale,
                                        std::size_t block, std::vector<stage>& stages) {
  const std::size_t h = n / 2;
  const real_route how = route(n, false, false);
  // c2r_line's radices and split run (chain_tail's and c2c_line's are the same here).
  line_edges edges;
  if (scale != 1.0) {
    edges.out = edge_op::scale;
    edges.scale = scale;
  }
  // Where the separate route's c2c_line takes a pair leaf (X28b), its chain, interleaved.
  const pair_leaf_entry* pair = how == real_route::separate
                                    ? pair_for(h, {dst.where, 1}, {dst.where, 1}, edges).entry
                                    : nullptr;
  const std::vector<std::size_t> rad = [&]() -> std::vector<std::size_t> {
    if (pair != nullptr) {
      return {pair->r1, pair->m};
    }
    return how == real_route::fused ? real_radices(h, false) : radices(h);
  }();
  const std::size_t split =
      rad.size() > 1 && pair == nullptr ? split_from(rad, false, 1) : rad.size();
  // The pairing reads the rows where a run of C of them is one vector load, else a tile.
  lane_rows from{src.where, 2 * src.stride, 2 * src_dist};
  slot z = slot::a;
  if (kernels_.complex_lanes > 1 && from.dist != 2) {
    add_lanes_copy(true, h + 1, from, slot::a, block, stages);
    from = {.where = slot::a, .stride = static_cast<std::ptrdiff_t>(2 * block), .dist = 2};
    z = slot::b;
  }
  stage st;
  st.run = kernels_.passes.lanes_c2r_pair;
  st.args.n = n;
  st.args.block = static_cast<std::uint32_t>(block);
  st.args.twiddles = leaf_pairing_twiddles(n);
  st.args.src_stride = from.stride;
  st.args.src_dist = from.dist;
  st.src = from.where;
  st.dst = z;
  st.units = 1;
  st.name = "lanes c2r pairing (" + std::to_string(block) + " rows)";
  stages.push_back(std::move(st));
  // The h complex values of a row written as its n reals.
  lanes_body(h, rad, split, +1,
             {.where = z, .stride = static_cast<std::ptrdiff_t>(2 * block), .dist = 2},
             {dst.where, 2, dst_dist}, edges, block, stages);
  return (h + 1) * block;
}

// chain.hpp declares the class's instantiations extern (chain.cpp makes them), which covers
// only the members defined there: every member defined here is instantiated here.
template bool chain_builder<double>::lanes_fit(std::size_t, bool) const;
template bool chain_builder<float>::lanes_fit(std::size_t, bool) const;
template void chain_builder<double>::add_lanes_copy(bool, std::size_t, lane_rows, slot, std::size_t,
                                                    std::vector<stage>&, bool);
template void chain_builder<float>::add_lanes_copy(bool, std::size_t, lane_rows, slot, std::size_t,
                                                   std::vector<stage>&, bool);
template void chain_builder<double>::lanes_body(std::size_t, const std::vector<std::size_t>&,
                                                std::size_t, int, lane_rows, lane_rows,
                                                const line_edges&, std::size_t, std::vector<stage>&,
                                                bool);
template void chain_builder<float>::lanes_body(std::size_t, const std::vector<std::size_t>&,
                                               std::size_t, int, lane_rows, lane_rows,
                                               const line_edges&, std::size_t, std::vector<stage>&,
                                               bool);
template std::size_t chain_builder<double>::c2c_lanes(std::size_t, int, endpoint, std::ptrdiff_t,
                                                      endpoint, std::ptrdiff_t, const line_edges&,
                                                      std::size_t, std::vector<stage>&, bool,
                                                      const std::vector<std::size_t>&);
template std::size_t chain_builder<float>::c2c_lanes(std::size_t, int, endpoint, std::ptrdiff_t,
                                                     endpoint, std::ptrdiff_t, const line_edges&,
                                                     std::size_t, std::vector<stage>&, bool,
                                                     const std::vector<std::size_t>&);
template std::size_t chain_builder<double>::r2c_lanes(std::size_t, endpoint, std::ptrdiff_t,
                                                      endpoint, std::ptrdiff_t, double, std::size_t,
                                                      std::vector<stage>&);
template std::size_t chain_builder<float>::r2c_lanes(std::size_t, endpoint, std::ptrdiff_t,
                                                     endpoint, std::ptrdiff_t, double, std::size_t,
                                                     std::vector<stage>&);
template std::size_t chain_builder<double>::c2r_lanes(std::size_t, endpoint, std::ptrdiff_t,
                                                      endpoint, std::ptrdiff_t, double, std::size_t,
                                                      std::vector<stage>&);
template std::size_t chain_builder<float>::c2r_lanes(std::size_t, endpoint, std::ptrdiff_t,
                                                     endpoint, std::ptrdiff_t, double, std::size_t,
                                                     std::vector<stage>&);

} // namespace fftn::inline v0::detail
