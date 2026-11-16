// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

// The chain builder's leaf tables from outside the target (strategies/chain.hpp): the
// 128-bit leaves of the x86 targets (use_narrow_leaves, experiment X28) and the pair leaves
// (use_pair_leaves, X28b). Only x86 targets have such tables (kernel_table.hpp), and
// passes_test.cpp checks their arithmetic there; these tests give the builder tables of
// marker leaves, on every target, and check which lines take which entry and edge variant,
// with what arguments, and which lines keep the chain: a stride other than 1, a factor
// edge, forced radices and a length without an entry. The lanes strategy takes a pair
// leaf's radices, so that each lane computes what the row alone would.

#include "core/program.hpp"
#include "fftn/isa.hpp"
#include "isa/kernel_table.hpp"
#include "isa/target.hpp"
#include "passes/pass_args.hpp"
#include "planner/rules.hpp"
#include "platform/aligned_allocator.hpp"
#include "strategies/chain.hpp"
#include "support/fftn_subject.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using fftn::detail::edge_op;
using fftn::detail::endpoint;
using fftn::detail::leaf_fn;
using fftn::detail::line_edges;
using fftn::detail::pass_args;
using fftn::detail::slot;
using fftn::detail::stage;

// The variant of the last marker leaf called: 1000 direction (0 forward) + 100 kind (1 a
// narrow unit leaf, 2 a narrow strided leaf, 3 a pair leaf) + 10 input edge + output edge.
int last_marker = -1;

template <int Marker>
void marker_leaf(const pass_args* /*args*/, const void* /*src*/, void* /*dst*/,
                 std::ptrdiff_t /*src_dist*/, std::ptrdiff_t /*dst_dist*/, std::size_t /*count*/) {
  last_marker = Marker;
}

// The marker leaf of kind K for direction D and edges I, O (none 0, scale 1, factor 2).
template <int K, int D, int I, int O> constexpr leaf_fn marker() {
  return &marker_leaf<(1000 * D) + (100 * K) + (10 * I) + O>;
}

int marker_of(int kind, int sign, edge_op in, edge_op out) {
  return (1000 * (sign < 0 ? 0 : 1)) + (100 * kind) + (10 * static_cast<int>(in)) +
         static_cast<int>(out);
}

// Which marker a stage's leaf is.
int called(const stage& st) {
  last_marker = -1;
  if (st.leaf != nullptr) {
    st.leaf(&st.args, nullptr, nullptr, 0, 0, 0);
  }
  return last_marker;
}

template <class T> fftn::detail::narrow_leaf_set<T> narrow_markers() {
  fftn::detail::narrow_leaf_set<T> set{};
  for (std::size_t slot_index = 0; slot_index < fftn::detail::narrow_leaf_slots; ++slot_index) {
    auto& leaf = set.leaf[slot_index];
    auto& unit = set.unit[slot_index];
    leaf[0][0][0] = marker<2, 0, 0, 0>();
    leaf[0][0][1] = marker<2, 0, 0, 1>();
    leaf[0][1][0] = marker<2, 0, 1, 0>();
    leaf[0][2][0] = marker<2, 0, 2, 0>();
    leaf[0][0][2] = marker<2, 0, 0, 2>();
    leaf[1][0][0] = marker<2, 1, 0, 0>();
    leaf[1][0][1] = marker<2, 1, 0, 1>();
    leaf[1][1][0] = marker<2, 1, 1, 0>();
    leaf[1][2][0] = marker<2, 1, 2, 0>();
    leaf[1][0][2] = marker<2, 1, 0, 2>();
    unit[0][0][0] = marker<1, 0, 0, 0>();
    unit[0][0][1] = marker<1, 0, 0, 1>();
    unit[0][1][0] = marker<1, 0, 1, 0>();
    unit[1][0][0] = marker<1, 1, 0, 0>();
    unit[1][0][1] = marker<1, 1, 0, 1>();
    unit[1][1][0] = marker<1, 1, 1, 0>();
  }
  return set;
}

// A pair leaf entry of n = r1 m whose leaves are markers.
fftn::detail::pair_leaf_entry pair_markers(std::size_t r1, std::size_t m) {
  fftn::detail::pair_leaf_entry e{.n = r1 * m, .r1 = r1, .m = m, .leaf = {}};
  e.leaf[0][0][0] = marker<3, 0, 0, 0>();
  e.leaf[0][0][1] = marker<3, 0, 0, 1>();
  e.leaf[0][1][0] = marker<3, 0, 1, 0>();
  e.leaf[1][0][0] = marker<3, 1, 0, 0>();
  e.leaf[1][0][1] = marker<3, 1, 0, 1>();
  e.leaf[1][1][0] = marker<3, 1, 1, 0>();
  return e;
}

// A builder for target t as plan_core.cpp makes one, with the given forced radices and
// pairing fusion.
template <class T> struct builder {
  fftn::detail::target target;
  fftn::detail::plan_tables<T> tables;
  fftn::detail::chain_builder<T> chain;

  explicit builder(fftn::isa t, std::vector<std::size_t> forced = {},
                   fftn::detail::real_fusion fusion = {})
      : target(static_cast<fftn::detail::target>(static_cast<unsigned char>(t))),
        chain(fftn::detail::of<T>(*fftn::detail::find_kernels(target)),
              fftn::detail::profile_of(
                  target, fftn::detail::of<T>(*fftn::detail::find_kernels(target)).complex_lanes),
              tables, 2048, std::move(forced), 0, fusion) {}
};

line_edges edges_of(edge_op in, edge_op out, const void* factor) {
  line_edges e;
  e.in = in;
  e.out = out;
  e.scale = 0.375;
  e.in_factor = in == edge_op::factor ? factor : nullptr;
  e.out_factor = out == edge_op::factor ? factor : nullptr;
  return e;
}

std::string names(const std::vector<stage>& stages) {
  std::string text;
  for (const stage& st : stages) {
    text += st.name + "\n";
  }
  return text;
}

constexpr edge_op none = edge_op::none;
constexpr edge_op scale = edge_op::scale;
constexpr edge_op factor = edge_op::factor;

template <class T> void narrow_leaves(fftn::isa t) {
  SCOPED_TRACE(std::string(fftn::isa_name(t)) + (sizeof(T) == 8 ? " double" : " float"));
  const fftn::detail::narrow_leaf_table table{
      .f64 = narrow_markers<double>(),
      .f32 = narrow_markers<float>(),
  };
  const fftn::detail::narrow_leaf_set<T>& set = fftn::detail::of<T>(table);
  EXPECT_EQ(&set, sizeof(T) == 8 ? static_cast<const void*>(&table.f64)
                                 : static_cast<const void*>(&table.f32));
  builder<T> b(t);
  b.chain.use_narrow_leaves(&set);
  const std::vector<std::complex<T>> ones(64, std::complex<T>(1));
  for (const std::size_t n : {std::size_t{4}, std::size_t{8}, std::size_t{16}, std::size_t{23}}) {
    for (const int sign : {-1, 1}) {
      for (const auto [in, out] : {
               std::pair{none, none},
               std::pair{none, scale},
               std::pair{scale, none},
               std::pair{factor, none},
               std::pair{none, factor},
           }) {
        SCOPED_TRACE("n = " + std::to_string(n) + ", sign " + std::to_string(sign) + ", edges " +
                     std::to_string(static_cast<int>(in)) + std::to_string(static_cast<int>(out)));
        const line_edges e = edges_of(in, out, ones.data());
        const bool unit_edges = in != factor && out != factor;
        // Stride 1 on both sides: the unit leaf, unless a factor edge needs the strided one.
        std::vector<stage> stages;
        EXPECT_EQ(b.chain.c2c_line(n, sign, {slot::in, 1}, {slot::out, 1}, false, e, stages), 0U);
        ASSERT_EQ(stages.size(), 1U) << names(stages);
        EXPECT_EQ(called(stages[0]), marker_of(unit_edges ? 1 : 2, sign, in, out));
        EXPECT_EQ(stages[0].args.n, n);
        EXPECT_EQ(stages[0].args.radix, n);
        EXPECT_EQ(stages[0].args.scale, 0.375);
        EXPECT_EQ(stages[0].args.in_factor, in == factor ? ones.data() : nullptr);
        EXPECT_EQ(stages[0].args.out_factor, out == factor ? ones.data() : nullptr);
        // A stride on either side: the strided leaf, with the strides.
        stages.clear();
        b.chain.c2c_line(n, sign, {slot::in, 3}, {slot::out, -2}, false, e, stages);
        ASSERT_EQ(stages.size(), 1U) << names(stages);
        EXPECT_EQ(called(stages[0]), marker_of(2, sign, in, out));
        EXPECT_EQ(stages[0].args.src_stride, 3);
        EXPECT_EQ(stages[0].args.dst_stride, -2);
        EXPECT_EQ(stages[0].src, slot::in);
        EXPECT_EQ(stages[0].dst, slot::out);
      }
    }
  }
  // Above the narrow slots, the target's own leaf (forced: the rules may split 32).
  constexpr std::size_t above = 32;
  ASSERT_GE(fftn::detail::codelet_slot(above), fftn::detail::narrow_leaf_slots);
  const auto& own = *fftn::detail::of<T>(*fftn::detail::find_kernels(b.target))
                         .passes.codelet[fftn::detail::codelet_slot(above)];
  builder<T> one(t, {above});
  one.chain.use_narrow_leaves(&set);
  std::vector<stage> stages;
  one.chain.c2c_line(above, -1, {slot::in, 1}, {slot::out, 1}, false, line_edges(), stages);
  ASSERT_EQ(stages.size(), 1U) << names(stages);
  EXPECT_NE(stages[0].leaf, nullptr);
  EXPECT_TRUE(stages[0].leaf == own.leaf_unit[0][0][0] || stages[0].leaf == own.leaf[0][0][0]);
}

TEST(Chain, NarrowLeavesReplaceTheTargetsLeaves) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    narrow_leaves<double>(t);
    narrow_leaves<float>(t);
  }
}

template <class T> void pair_leaves(fftn::isa t) {
  SCOPED_TRACE(std::string(fftn::isa_name(t)) + (sizeof(T) == 8 ? " double" : " float"));
  // Entries ascend and end at the first n = 0, as the targets' tables do.
  fftn::detail::pair_leaf_table table{};
  for (auto* set : {&table.f64.entry, &table.f32.entry}) {
    (*set)[0] = pair_markers(16, 8);
    (*set)[1] = pair_markers(8, 32);
  }
  const fftn::detail::pair_leaf_set<T>& set = fftn::detail::of<T>(table);
  EXPECT_EQ(&set, sizeof(T) == 8 ? static_cast<const void*>(&table.f64)
                                 : static_cast<const void*>(&table.f32));
  builder<T> b(t);
  b.chain.use_pair_leaves(&set);
  const std::vector<std::complex<T>> ones(1024, std::complex<T>(1));
  for (const auto [n, r1, m] : {
           std::tuple{std::size_t{128}, std::size_t{16}, std::size_t{8}},
           std::tuple{std::size_t{256}, std::size_t{8}, std::size_t{32}},
       }) {
    ASSERT_GE(b.chain.radices(n).size(), 2U) << "n = " << n << " is a single codelet's leaf";
    for (const int sign : {-1, 1}) {
      for (const auto [in, out] :
           {std::pair{none, none}, std::pair{none, scale}, std::pair{scale, none}}) {
        SCOPED_TRACE("n = " + std::to_string(n) + ", sign " + std::to_string(sign) + ", edges " +
                     std::to_string(static_cast<int>(in)) + std::to_string(static_cast<int>(out)));
        std::vector<stage> stages;
        const line_edges e = edges_of(in, out, nullptr);
        EXPECT_EQ(b.chain.c2c_line(n, sign, {slot::in, 1}, {slot::out, 1}, false, e, stages), 0U);
        ASSERT_EQ(stages.size(), 1U) << names(stages);
        const stage& st = stages[0];
        EXPECT_EQ(called(st), marker_of(3, sign, in, out));
        EXPECT_EQ(st.args.n, n);
        EXPECT_EQ(st.args.radix, r1);
        EXPECT_EQ(st.args.m, m);
        EXPECT_EQ(st.args.scale, 0.375);
        EXPECT_NE(st.args.twiddles, nullptr);
        EXPECT_EQ(st.src, slot::in);
        EXPECT_EQ(st.dst, slot::out);
        const std::string name = "pair leaf radix-" + std::to_string(r1) + " x " +
                                 std::to_string(m) + " in registers " +
                                 (sign < 0 ? "forward" : "backward");
        EXPECT_EQ(st.name.rfind(name, 0), 0U) << st.name;
        EXPECT_EQ(st.name.find("scale") != std::string::npos, in == scale || out == scale)
            << st.name;
      }
    }
    EXPECT_EQ(b.chain.radix_text(n),
              std::to_string(r1) + " x " + std::to_string(m) + " in registers (pair leaf)");
    // Where the pair leaf does not apply, the chain runs: a stride on either side, a factor
    // edge.
    for (const auto& [src, dst, e] : {
             std::tuple{endpoint{.where = slot::in, .stride = 2},
                        endpoint{.where = slot::out, .stride = 1}, line_edges()},
             std::tuple{endpoint{.where = slot::in, .stride = 1},
                        endpoint{.where = slot::out, .stride = 2}, line_edges()},
             std::tuple{
                 endpoint{.where = slot::in, .stride = 1},
                 endpoint{.where = slot::out, .stride = 1},
                 edges_of(factor, none, ones.data()),
             },
             std::tuple{
                 endpoint{.where = slot::in, .stride = 1},
                 endpoint{.where = slot::out, .stride = 1},
                 edges_of(none, factor, ones.data()),
             },
         }) {
      std::vector<stage> stages;
      b.chain.c2c_line(n, -1, src, dst, false, e, stages);
      EXPECT_EQ(names(stages).find("pair leaf"), std::string::npos) << names(stages);
    }
    // Radices a test forces win over the pair leaf.
    builder<T> forced(t, {r1, m});
    forced.chain.use_pair_leaves(&set);
    std::vector<stage> stages;
    forced.chain.c2c_line(n, -1, {slot::in, 1}, {slot::out, 1}, false, line_edges(), stages);
    EXPECT_EQ(names(stages).find("pair leaf"), std::string::npos) << names(stages);
    EXPECT_GE(stages.size(), 2U) << names(stages);
  }
  // Lengths without an entry: below the first, between two, past the last.
  for (const std::size_t n : {std::size_t{64}, std::size_t{192}, std::size_t{512}}) {
    std::vector<stage> stages;
    b.chain.c2c_line(n, -1, {slot::in, 1}, {slot::out, 1}, false, line_edges(), stages);
    EXPECT_EQ(names(stages).find("pair leaf"), std::string::npos) << n << ":\n" << names(stages);
    EXPECT_EQ(b.chain.radix_text(n).find("pair leaf"), std::string::npos);
  }
}

TEST(Chain, PairLeavesTakeTheirLengths) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    pair_leaves<double>(t);
    pair_leaves<float>(t);
  }
}

// The lanes strategy over rows whose length has a pair leaf: the leaf's radices r1 x m, as
// the radices forced to r1 x m give, for c2c rows and for real rows of twice the length on
// the separate route (no pairing fused into a pass).
template <class T> void pair_lanes(fftn::isa t) {
  SCOPED_TRACE(std::string(fftn::isa_name(t)) + (sizeof(T) == 8 ? " double" : " float"));
  fftn::detail::pair_leaf_table table{};
  table.f64.entry[0] = pair_markers(16, 8);
  table.f32.entry[0] = pair_markers(16, 8);
  constexpr std::size_t n = 128;
  const fftn::detail::real_fusion separate{.radix = 1};
  builder<T> b(t, {}, separate);
  b.chain.use_pair_leaves(&fftn::detail::of<T>(table));
  builder<T> forced(t, {16, 8}, separate);
  ASSERT_EQ(b.chain.route(2 * n, true, false), fftn::detail::real_route::separate);
  const std::size_t block =
      fftn::detail::of<T>(*fftn::detail::find_kernels(b.target)).complex_lanes * 2;
  const auto dist = static_cast<std::ptrdiff_t>(4 * n);
  for (auto const* x : {&b, &forced}) {
    EXPECT_TRUE(x->chain.lanes_fit(n, false));
    EXPECT_TRUE(x->chain.lanes_fit(2 * n, true));
  }
  std::vector<stage> with;
  std::vector<stage> without;
  b.chain.c2c_lanes(n, 1, {slot::in, 1}, dist, {slot::out, 1}, dist, line_edges(), block, with);
  forced.chain.c2c_lanes(n, 1, {slot::in, 1}, dist, {slot::out, 1}, dist, line_edges(), block,
                         without);
  EXPECT_EQ(names(with), names(without));
  EXPECT_NE(names(with).find("lanes first radix-16"), std::string::npos) << names(with);
  with.clear();
  without.clear();
  b.chain.r2c_lanes(2 * n, {slot::in, 1}, dist, {slot::out, 1}, dist, 1.0, block, with);
  forced.chain.r2c_lanes(2 * n, {slot::in, 1}, dist, {slot::out, 1}, dist, 1.0, block, without);
  EXPECT_EQ(names(with), names(without));
  EXPECT_NE(names(with).find("lanes first radix-16"), std::string::npos) << names(with);
  with.clear();
  without.clear();
  b.chain.c2r_lanes(2 * n, {slot::in, 1}, dist, {slot::out, 1}, dist, 0.5, block, with);
  forced.chain.c2r_lanes(2 * n, {slot::in, 1}, dist, {slot::out, 1}, dist, 0.5, block, without);
  EXPECT_EQ(names(with), names(without));
  EXPECT_NE(names(with).find("lanes first radix-16"), std::string::npos) << names(with);
}

TEST(Chain, LanesTakeAPairLeafsRadices) {
  for (const fftn::isa t : fftn::test::runnable_isas()) {
    pair_lanes<double>(t);
    pair_lanes<float>(t);
  }
}

} // namespace
