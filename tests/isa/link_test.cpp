// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// All-targets link test (DESIGN.md 6.4): every target's unit and every target's probe unit
// are linked into this one executable, which links only because their symbols are
// distinct. Here: each compiled target has its own table, with its own entry points, and
// none is shared between targets. The disassembly test (isa.asm_check) checks the same
// objects for the instructions in each target's code.

#include "isa/arch.hpp"
#include "isa/kernel_table.hpp"
#include "isa/probe_table.hpp"
#include "isa/target.hpp"
#include "isa/test_targets.hpp"
#include "passes/pass_args.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <vector>

namespace {

using fftn::detail::compiled_in;
using fftn::detail::find_kernels;
using fftn::detail::find_probes;
using fftn::detail::kernel_table;
using fftn::detail::target;
using fftn::detail::target_name;

TEST(link, every_compiled_target_has_its_own_table) {
  std::set<std::uintptr_t> seen;
  int tables = 0;
  for (const target t : isa_test::all_targets) {
    const kernel_table* k = find_kernels(t);
    ASSERT_EQ(k != nullptr, compiled_in(t)) << target_name(t);
    ASSERT_EQ(find_probes(t) != nullptr, compiled_in(t)) << target_name(t);
    if (k == nullptr) {
      continue;
    }
    ++tables;
    EXPECT_EQ(k->id, t);
    EXPECT_EQ(find_probes(t)->id, t);
    // Addresses, only compared.
    const std::vector<std::uintptr_t> entries = {
        reinterpret_cast<std::uintptr_t>(k->f64.pack_twiddles),
        reinterpret_cast<std::uintptr_t>(k->f64.pack_bcast_twiddles),
        reinterpret_cast<std::uintptr_t>(
            k->f64.passes.codelet[fftn::detail::codelet_slot(4)]->lane[0]),
        reinterpret_cast<std::uintptr_t>(k->f64.passes.copy[0][0]),
        reinterpret_cast<std::uintptr_t>(k->f32.pack_twiddles),
        reinterpret_cast<std::uintptr_t>(k->f32.pack_bcast_twiddles),
        reinterpret_cast<std::uintptr_t>(
            k->f32.passes.codelet[fftn::detail::codelet_slot(4)]->lane[0]),
        reinterpret_cast<std::uintptr_t>(k->f32.passes.copy[0][0]),
        reinterpret_cast<std::uintptr_t>(find_probes(t)->f64),
        reinterpret_cast<std::uintptr_t>(find_probes(t)->f32),
    };
    for (const std::uintptr_t e : entries) {
      EXPECT_NE(e, 0U) << target_name(t);
      EXPECT_TRUE(seen.insert(e).second) << target_name(t) << " shares an entry point";
    }
  }
  EXPECT_EQ(tables, FFTN_ARCH_X86_64 != 0 ? 5 : FFTN_ARCH_AARCH64 != 0 ? 3 : 1);
}

TEST(link, lane_counts_and_fusion_match_each_target) {
  const kernel_table& s = *find_kernels(target::scalar);
  EXPECT_FALSE(s.fused);
  EXPECT_EQ(s.f64.complex_lanes, 1U);
  EXPECT_EQ(s.f32.complex_lanes, 1U);
  EXPECT_EQ(s.f64.real_lanes, 1U);
  EXPECT_EQ(s.f64.tw_reals, 2U);
  if (const kernel_table* v = find_kernels(target::x86_v2)) {
    EXPECT_FALSE(v->fused);
    EXPECT_EQ(v->f64.complex_lanes, 1U);
    EXPECT_EQ(v->f32.complex_lanes, 2U);
    EXPECT_EQ(v->f64.real_lanes, 2U);
    EXPECT_EQ(v->f32.real_lanes, 4U);
    EXPECT_EQ(v->f64.tw_reals, 4U);    // pre-arranged, as on the other x86 targets
    EXPECT_EQ(v->f64.split_block, 4U); // avx's W: the pair rounds alike
  }
  if (const kernel_table* a = find_kernels(target::avx)) {
    EXPECT_FALSE(a->fused);
    EXPECT_EQ(a->f64.complex_lanes, 2U);
    EXPECT_EQ(a->f32.complex_lanes, 4U);
    EXPECT_EQ(a->f64.real_lanes, 4U);
    EXPECT_EQ(a->f32.real_lanes, 8U);
    EXPECT_EQ(a->f32.tw_reals, 4U);
    EXPECT_EQ(a->f32.split_block, 8U);
  }
  if (const kernel_table* a = find_kernels(target::avx2)) {
    EXPECT_TRUE(a->fused);
    EXPECT_EQ(a->f64.complex_lanes, 2U);
    EXPECT_EQ(a->f32.complex_lanes, 4U);
    EXPECT_EQ(a->f64.real_lanes, 4U);
    EXPECT_EQ(a->f32.real_lanes, 8U);
    EXPECT_EQ(a->f64.tw_reals, 4U); // pre-arranged (wr, wr), (-wi, wi)
  }
  if (const kernel_table* a = find_kernels(target::avx512)) {
    EXPECT_TRUE(a->fused);
    EXPECT_EQ(a->f64.complex_lanes, 4U);
    EXPECT_EQ(a->f32.complex_lanes, 8U);
    EXPECT_EQ(a->f64.real_lanes, 8U);
    EXPECT_EQ(a->f32.real_lanes, 16U);
    EXPECT_EQ(a->f32.tw_reals, 4U); // pre-arranged (wr, wr), (-wi, wi)
  }
  if (const kernel_table* n = find_kernels(target::neon)) {
    EXPECT_TRUE(n->fused);
    EXPECT_EQ(n->f64.complex_lanes, 1U);
    EXPECT_EQ(n->f32.complex_lanes, 2U);
    EXPECT_EQ(n->f64.real_lanes, 2U);
    EXPECT_EQ(n->f32.real_lanes, 4U);
    EXPECT_EQ(n->f64.tw_reals, 4U); // pre-arranged, as on x86
  }
  if (const kernel_table* f = find_kernels(target::neon_fcma)) {
    EXPECT_TRUE(f->fused);
    EXPECT_EQ(f->f64.complex_lanes, 1U);
    EXPECT_EQ(f->f32.complex_lanes, 2U);
    EXPECT_EQ(f->f64.real_lanes, 2U);
    EXPECT_EQ(f->f32.real_lanes, 4U);
    EXPECT_EQ(f->f32.tw_reals, 2U);
  }
}

} // namespace
