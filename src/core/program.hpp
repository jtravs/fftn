// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Programs (DESIGN.md 8.3): a plan's executable form, a flat array of stage and loop
// records prepared at construction and interpreted by a small driver (driver.hpp).
//
// A stage is one pass over one line. Its source and destination are slots: the caller's
// input and output arrays, two scratch lines a and b, and a block holding one transform's
// half spectrum (multi-dimensional c2r out of place, DESIGN.md 9.1). The driver keeps a
// cursor into the in, out and block slots; loops move it. Scratch lines do not move: every
// line uses them from their start.
//
// A program is
//
//   batch loops (one per batch dimension, outermost first), then the body: one transform.
//
// The body is, for each transformed axis in turn (a phase), loops over the lines of the
// other axes enclosing the stages of one line. Every loop's iterations are independent,
// and every stage divides into `chunks` independent ranges of units fixed by the plan:
// the work graph of phase 3 reads both off (DESIGN.md 13.2). Serial execution runs every
// chunk in order.
#pragma once

#include "passes/pass_args.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace fftn::inline v0::detail {

enum class slot : std::uint8_t { in, out, a, b, block };

// The slots a loop moves, as indices into loop::step.
inline constexpr std::size_t moving_slots = 3; // in, out, block

struct stage {
  pass_fn run = nullptr;
  pass_args args; // src and dst are filled in by the driver from the slots
  slot src = slot::in;
  slot dst = slot::out;
  std::size_t units = 0;    // butterflies, or the edge pass's own units
  std::uint32_t chunks = 1; // independent ranges [units c / chunks, units (c+1) / chunks)
  std::string name;         // for describe()
};

enum class loop_kind : std::uint8_t {
  batch, // whole transforms
  lines, // the lines of one axis within a transform
};

struct loop {
  loop_kind kind = loop_kind::batch;
  std::size_t count = 1;
  std::array<std::ptrdiff_t, moving_slots> step{}; // bytes per iteration: in, out, block
  std::uint32_t body_end = 0;                      // records [this + 1, body_end) are the body
};

using record = std::variant<stage, loop>;

struct program {
  std::vector<record> records;
  std::size_t batch_loops = 0; // the leading records that are the batch loops
  std::size_t transforms = 1;  // product of the batch loops' counts
};

} // namespace fftn::inline v0::detail
