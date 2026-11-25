// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// The driver of plans whose edges have factors (core/edges.hpp, DESIGN.md 12.1): the program
// interpreter of driver.hpp plus the cursors of the factor arrays, which arrive with each
// execution (exec_args). It is a separate driver so that every other plan runs on driver.hpp
// exactly as before edges existed: plans without edges were 2-6 % slower on the EPYC 7763 when
// one driver moved the cursors for every plan (rows of 8, 100 x 1024, real 2048 x 36).
//
// In such a plan a stage that reads a cursor (stage::moving_factors) carries guards as its
// kernels, and its edge_record the kernels themselves: executed through driver.hpp (an overload
// without exec_args) the plan ends the program (contract_failure) instead of reading through a
// null pointer. Only the walked form of the program runs here: its flat, unrolled and leaf-row
// forms run the same kernels on the same arguments, so the bits are the same.
#pragma once

#include "core/driver.hpp"
#include "core/program.hpp"
#include "fftn/edges.hpp"
#include "passes/pass_args.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fftn::inline v0::detail {

// What the edge driver keeps per record of a plan whose edges have factors.
struct edge_record {
  // A stage that reads a factor cursor: its kernel (the stage's own are guards).
  pass_fn run = nullptr;
  leaf_fn leaf = nullptr;
  // A loop: the bytes it moves the input's and the output's factor cursors per iteration.
  std::array<std::ptrdiff_t, 2> factor_step{};
};

// The cursors of the edges' factor arrays: addresses as integers, so that moving one that is 0
// (an edge without a factor) needs no test.
struct factor_cursors {
  std::uintptr_t in = 0;
  std::uintptr_t out = 0;
};

// The kernels a stage that reads a factor cursor carries for driver.hpp: they end the program
// (contract_failure).
void factor_guard_pass(const pass_args* args, const void* src, void* dst, std::size_t first,
                       std::size_t last);
void factor_guard_leaf(const pass_args* args, const void* src, void* dst, std::ptrdiff_t src_dist,
                       std::ptrdiff_t dst_dist, std::size_t count);

// The cursors' starting values from an execution's data: `needed` holds moving_in_factor and
// moving_out_factor for the arrays the plan reads; a needed array that is null ends the program.
[[nodiscard]] factor_cursors cursors_of(const exec_args& args, std::uint8_t needed) noexcept;

// The cursors of transform t: f moved by the batch loops.
[[nodiscard]] factor_cursors edge_transform_cursors(const program& p,
                                                    const std::vector<edge_record>& e,
                                                    std::size_t t, factor_cursors f) noexcept;

// As run_transforms, run_stage_chunks and run_lines (driver.hpp), with the cursors: base's
// slots and f's cursors are those of transform 0 (run_edge_transforms) or of the transform the
// piece belongs to (the others). `tail`: the records of a lanes program's tail
// (program::lane_tail), one per record.
void run_edge_transforms(const program& p, const std::vector<edge_record>& e,
                         const std::vector<edge_record>& tail, std::size_t first, std::size_t count,
                         const slot_addresses& base, factor_cursors f) noexcept;
void run_edge_stage_chunks(const program& p, const std::vector<edge_record>& e, std::size_t index,
                           const slot_addresses& at, factor_cursors f, std::size_t first,
                           std::size_t last) noexcept;
void run_edge_lines(const program& p, const std::vector<edge_record>& e, std::size_t begin,
                    std::size_t nest, std::size_t body_end, std::size_t first, std::size_t last,
                    const slot_addresses& at, factor_cursors f) noexcept;

} // namespace fftn::inline v0::detail
