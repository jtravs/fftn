// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/driver.hpp"

#include "core/program.hpp"
#include "passes/pass_args.hpp"

#include <array>
#include <cstddef>
#include <variant>

namespace fftn::inline v0::detail {
namespace {

const void* source_of(slot s, const slot_addresses& at) noexcept {
  switch (s) {
  case slot::in:
    return at.in;
  case slot::out:
    return at.out;
  case slot::a:
    return at.a;
  case slot::b:
    return at.b;
  case slot::block:
    break;
  }
  return at.block;
}

// The input is never a destination (program.hpp).
void* destination_of(slot s, const slot_addresses& at) noexcept {
  switch (s) {
  case slot::out:
  case slot::in:
    return at.out;
  case slot::a:
    return at.a;
  case slot::b:
    return at.b;
  case slot::block:
    break;
  }
  return at.block;
}

slot_addresses moved(const slot_addresses& at, const std::array<std::ptrdiff_t, moving_slots>& step,
                     std::size_t times) noexcept {
  const auto k = static_cast<std::ptrdiff_t>(times);
  // A slot a loop does not move may be null (no block without a workspace), and no offset,
  // not even zero, is applied to a null pointer.
  slot_addresses next = at;
  if (step[0] != 0) {
    next.in += step[0] * k;
  }
  if (step[1] != 0) {
    next.out += step[1] * k;
  }
  if (step[2] != 0) {
    next.block += step[2] * k;
  }
  next.tw += step[3] * k;
  return next;
}

void run_records(const program& p, std::size_t begin, std::size_t end,
                 const slot_addresses& at) noexcept {
  for (std::size_t i = begin; i < end; ++i) {
    const record& r = p.records[i];
    if (const stage* st = std::get_if<stage>(&r)) {
      run_stage_chunks(*st, at, 0, st->chunks);
      continue;
    }
    const loop& l = *std::get_if<loop>(&r);
    for (std::size_t c = 0; c < l.count; ++c) {
      run_records(p, i + 1, l.body_end, moved(at, l.step, c));
    }
    i = l.body_end - 1;
  }
}

// Moves `at` to iteration `index` of the loops [begin, begin + depth), the last varying
// fastest.
slot_addresses moved_in_nest(const program& p, std::size_t begin, std::size_t depth,
                             std::size_t index, const slot_addresses& at) noexcept {
  slot_addresses here = at;
  std::size_t rest = index;
  for (std::size_t i = begin + depth; i > begin && rest != 0; --i) {
    const loop& l = *std::get_if<loop>(&p.records[i - 1]);
    // No division while the index fits the loop (always, for one batch loop): it runs once
    // per transform, and divisions are slow on some x86 cores (run_stage_chunks).
    if (rest < l.count) {
      return moved(here, l.step, rest);
    }
    here = moved(here, l.step, rest % l.count);
    rest /= l.count;
  }
  return here;
}

// A stage of a lanes block (program.hpp): whole, with the block's rows.
void run_lane_stage(const stage& st, const slot_addresses& at, std::size_t rows) noexcept {
  pass_args args = st.args;
  args.src = source_of(st.src, at);
  args.dst = destination_of(st.dst, at);
  args.rows = rows;
  st.run(&args, 0, st.units);
}

// Transforms [first, first + count) of a lanes program: within each run of the innermost
// batch loop, blocks of p.lanes rows through the body, then blocks of p.lane_unit through
// the tail. Every lane computes its row alone (strategies/lanes.cpp), so how the rows fall
// into blocks does not change any result.
void run_lane_blocks(const program& p, std::size_t first, std::size_t count,
                     const slot_addresses& base) noexcept {
  const loop& inner = *std::get_if<loop>(&p.records[p.batch_loops - 1]);
  const std::size_t end = first + count;
  std::size_t t = first;
  while (t < end) {
    slot_addresses at = transform_addresses(p, t, base);
    const std::size_t in_loop = inner.count - (t < inner.count ? t : t % inner.count);
    std::size_t run = end - t < in_loop ? end - t : in_loop;
    t += run;
    for (; run >= p.lanes; run -= p.lanes) {
      for (std::size_t i = p.batch_loops; i < p.records.size(); ++i) {
        run_lane_stage(*std::get_if<stage>(&p.records[i]), at, p.lanes);
      }
      at = moved(at, inner.step, p.lanes);
    }
    while (run != 0) {
      const std::size_t rows = run < p.lane_unit ? run : p.lane_unit;
      for (const stage& st : p.lane_tail) {
        run_lane_stage(st, at, rows);
      }
      at = moved(at, inner.step, rows);
      run -= rows;
    }
  }
}

} // namespace

void run_stage_chunks(const stage& st, const slot_addresses& at, std::size_t first,
                      std::size_t last) noexcept {
  pass_args args = st.args;
  args.src = source_of(st.src, at);
  args.dst = destination_of(st.dst, at);
  if (st.moving_twiddles) {
    args.twiddles = static_cast<const std::byte*>(args.twiddles) + at.tw;
  }
  // Chunks [first, last) are adjacent ranges of units, and a unit's result does not depend
  // on the range it is run in, so one call over their union gives the same bits as one call
  // per chunk. The ends are found without dividing in the common case of the whole stage:
  // a 64-bit division costs 40 to 90 cycles on Cascade Lake, and four per chunk were a third
  // of a 64-point transform there (dev/experiments/X18-x86-kernels.md).
  const std::size_t begin = first == 0 ? 0 : chunk_start(st.units, st.chunks, first);
  const std::size_t end = last == st.chunks ? st.units : chunk_start(st.units, st.chunks, last);
  if (begin < end) {
    st.run(&args, begin, end);
  }
}

slot_addresses transform_addresses(const program& p, std::size_t t,
                                   const slot_addresses& base) noexcept {
  return moved_in_nest(p, 0, p.batch_loops, t, base);
}

void run_transforms(const program& p, std::size_t first, std::size_t count,
                    const slot_addresses& base) noexcept {
  if (p.lanes != 0) {
    run_lane_blocks(p, first, count, base);
    return;
  }
  for (std::size_t t = first; t < first + count; ++t) {
    run_records(p, p.batch_loops, p.records.size(), transform_addresses(p, t, base));
  }
}

void run_lines(const program& p, std::size_t begin, std::size_t nest, std::size_t body_end,
               std::size_t first, std::size_t last, const slot_addresses& at) noexcept {
  for (std::size_t line = first; line < last; ++line) {
    run_records(p, begin + nest, body_end, moved_in_nest(p, begin, nest, line, at));
  }
}

} // namespace fftn::inline v0::detail
