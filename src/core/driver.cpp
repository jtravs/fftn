// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/driver.hpp"

#include "core/program.hpp"
#include "passes/pass_args.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

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

// The bytes a loop moves slot s per iteration (0 for the scratch lines, which never move).
std::ptrdiff_t step_of(const loop& l, slot s) noexcept {
  switch (s) {
  case slot::in:
    return l.step[0];
  case slot::out:
    return l.step[1];
  case slot::block:
    return l.step[2];
  case slot::a:
  case slot::b:
    break;
  }
  return 0;
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
    const slot_addresses from = moved(at, l.start, 1);
    // A loop over the lines of an axis whose body is one leaf (the rows of a plane of short
    // rows) is one call of the leaf over all its lines, as run_flat runs a batch of leaf rows
    // (X20); the leaf computes each line as it would alone (experiment X40).
    if (l.body_end == i + 2) {
      const stage* only = std::get_if<stage>(&p.records[i + 1]);
      if (only != nullptr && only->leaf != nullptr && step_of(l, only->src) != 0 &&
          step_of(l, only->dst) != 0) {
        only->leaf(&only->args, source_of(only->src, from), destination_of(only->dst, from),
                   step_of(l, only->src), step_of(l, only->dst), l.count);
        i = l.body_end - 1;
        continue;
      }
    }
    for (std::size_t c = 0; c < l.count; ++c) {
      run_records(p, i + 1, l.body_end, moved(from, l.step, c));
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
  args.rows = static_cast<std::uint32_t>(rows);
  st.run(&args, source_of(st.src, at), destination_of(st.dst, at), 0, st.units);
}

// Records [begin, end) of a lanes program (its body or its tail), a block of `rows` rows.
void run_lane_records(const std::vector<record>& records, std::size_t begin, std::size_t end,
                      const slot_addresses& at, std::size_t rows) noexcept {
  for (std::size_t i = begin; i < end; ++i) {
    const record& r = records[i];
    if (const stage* st = std::get_if<stage>(&r)) {
      run_lane_stage(*st, at, rows);
      continue;
    }
    const loop& l = *std::get_if<loop>(&r);
    const slot_addresses from = moved(at, l.start, 1);
    for (std::size_t c = 0; c < l.count; ++c) {
      run_lane_records(records, i + 1, l.body_end, moved(from, l.step, c), rows);
    }
    i = l.body_end - 1;
  }
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
    // `at` moves only to a block that follows, so no address beyond the batch is formed.
    while (run != 0) {
      const bool whole = run >= p.lanes;
      std::size_t rows = p.lanes;
      if (!whole) {
        rows = run < p.lane_unit ? run : p.lane_unit;
      }
      if (whole) {
        run_lane_records(p.records, p.batch_loops, p.records.size(), at, rows);
      } else {
        run_lane_records(p.lane_tail, 0, p.lane_tail.size(), at, rows);
      }
      run -= rows;
      if (run != 0) {
        at = moved(at, inner.step, rows);
      }
    }
  }
}

// Runs leaf stage s, its one line, on the slots at.
void run_leaf(const stage& s, const slot_addresses& at) noexcept {
  s.leaf(&s.args, source_of(s.src, at), destination_of(s.dst, at), 0, 0, 1);
}

// Runs stage s on the slots at: run_stage_chunks over every chunk, without its chunk ends.
void run_whole(const stage& s, const slot_addresses& at) noexcept {
  if (s.leaf != nullptr) {
    run_leaf(s, at);
    return;
  }
  if (s.units != 0) {
    s.run(&s.args, source_of(s.src, at), destination_of(s.dst, at), 0, s.units);
  }
}

// Transforms [first, first + count) of a flat program (program.hpp). run_records walks the
// records and finds each stage's chunk ends for every transform: about 3.5 ns per transform
// on the M5 Max, more than an 8-point leaf's own time (dev/experiments/X20-call-overhead.md).
// Here a batch of leaves is one call; a batch (this function) moves the slots by the batch
// loop's step and calls every kernel on the plan's own arguments. Every kernel gets the
// arguments and units run_records gives it (the whole stage: chunk ends 0 and units), so
// the bits are the same. One transform takes run_flat_one and rows of leaves
// run_leaf_rows, both kept out of this function: its frame and saved registers cost about
// 6 ns at one transform on the M5 Max and 2-3 ns of a leaf row on the Xeon (X20).
[[gnu::noinline]] void run_flat(const program& p, std::size_t first, std::size_t count,
                                const slot_addresses& base) noexcept {
  const std::size_t stages = p.records.size() - p.batch_loops;
  const loop* only = p.batch_loops == 1 ? std::get_if<loop>(p.records.data()) : nullptr;
  std::array<const stage*, flat_stages_max> st{};
  for (std::size_t k = 0; k < stages; ++k) {
    st[k] = std::get_if<stage>(&p.records[p.batch_loops + k]);
  }
  // One batch loop (rows): move by its step; several: from the transform's index.
  slot_addresses at = transform_addresses(p, first, base);
  for (std::size_t t = first; t < first + count; ++t) {
    if (t != first) {
      at = only != nullptr ? moved(at, only->step, 1) : transform_addresses(p, t, base);
    }
    for (std::size_t k = 0; k < stages; ++k) {
      run_whole(*st[k], at);
    }
  }
}

// One transform of a flat program on its slots: each stage of the program's flat_body on
// its own arguments. The slots come as scalars and are chosen by index, not through a
// slot_addresses (X28).
void run_flat_one(const program& p, const std::byte* in, std::byte* out, std::byte* a, std::byte* b,
                  std::byte* block) noexcept {
  // Indexed by slot (program.hpp); the input is never a destination.
  const std::array<const void*, 5> sources = {in, out, a, b, block};
  const std::array<void*, 5> destinations = {out, out, a, b, block};
  for (const flat_stage& s : p.flat_body) {
    const void* src = sources[static_cast<std::size_t>(s.src)];
    void* dst = destinations[static_cast<std::size_t>(s.dst)];
    if (s.leaf != nullptr) {
      s.leaf(&s.args, src, dst, 0, 0, 1);
      continue;
    }
    if (s.units != 0) {
      s.run(&s.args, src, dst, 0, s.units);
    }
  }
}

// One transform of a program with calls (program.hpp): each call on the transform's slots
// moved by its offsets. Every kernel gets the arguments and units run_records gives it (a
// stage whole, a loop of leaves as one call), so the bits are the same (X41).
void run_calls(const program& p, const slot_addresses& at) noexcept {
  // Indexed by slot (program.hpp); the input is never a destination.
  const std::array<const std::byte*, 5> sources = {at.in, at.out, at.a, at.b, at.block};
  const std::array<std::byte*, 5> destinations = {at.out, at.out, at.a, at.b, at.block};
  const pass_args* args = p.call_args.data();
  for (const call& c : p.calls) {
    const std::byte* src = sources[static_cast<std::size_t>(c.src)];
    std::byte* dst = destinations[static_cast<std::size_t>(c.dst)];
    // A slot that does not move has offset 0 and may be null (no block without a workspace).
    if (c.src_off != 0) {
      src += c.src_off;
    }
    if (c.dst_off != 0) {
      dst += c.dst_off;
    }
    if (c.leaf != nullptr) {
      c.leaf(&args[c.args], src, dst, c.src_step, c.dst_step, c.units);
    } else {
      c.run(&args[c.args], src, dst, 0, c.units);
    }
  }
}

// The bytes the slots in, out and block have moved (moved()'s step order).
using offsets = std::array<std::ptrdiff_t, 3>;

std::ptrdiff_t offset_of(slot s, const offsets& off) noexcept {
  switch (s) {
  case slot::in:
    return off[0];
  case slot::out:
    return off[1];
  case slot::block:
    return off[2];
  case slot::a:
  case slot::b:
    break;
  }
  return 0;
}

// Appends the calls of records [begin, end) at offsets `off` (run_records' order and
// grouping); false when the program has moving twiddles or more than calls_max calls.
bool unroll_records(const program& p, std::size_t begin, std::size_t end, const offsets& off,
                    std::vector<std::size_t>& arg_of, std::vector<call>& out) {
  for (std::size_t i = begin; i < end; ++i) {
    const record& r = p.records[i];
    if (const stage* st = std::get_if<stage>(&r)) {
      if (st->moving_twiddles || out.size() == calls_max) {
        return false;
      }
      if (st->leaf == nullptr && st->units == 0) {
        continue; // run_whole runs nothing
      }
      out.push_back({
          .run = st->run,
          .leaf = st->leaf,
          .units = st->leaf != nullptr ? 1 : st->units,
          .src_off = offset_of(st->src, off),
          .dst_off = offset_of(st->dst, off),
          .args = static_cast<std::uint32_t>(arg_of[i]),
          .src = st->src,
          .dst = st->dst,
      });
      continue;
    }
    const loop& l = *std::get_if<loop>(&r);
    const offsets from = {off[0] + l.start[0], off[1] + l.start[1], off[2] + l.start[2]};
    if (l.body_end == i + 2) {
      const stage* only = std::get_if<stage>(&p.records[i + 1]);
      if (only != nullptr && only->leaf != nullptr && step_of(l, only->src) != 0 &&
          step_of(l, only->dst) != 0) {
        if (out.size() == calls_max) {
          return false;
        }
        out.push_back({
            .leaf = only->leaf,
            .units = l.count,
            .src_off = offset_of(only->src, from),
            .dst_off = offset_of(only->dst, from),
            .src_step = step_of(l, only->src),
            .dst_step = step_of(l, only->dst),
            .args = static_cast<std::uint32_t>(arg_of[i + 1]),
            .src = only->src,
            .dst = only->dst,
        });
        i = l.body_end - 1;
        continue;
      }
    }
    if (l.step[3] != 0) {
      return false;
    }
    for (std::size_t c = 0; c < l.count; ++c) {
      const auto k = static_cast<std::ptrdiff_t>(c);
      if (!unroll_records(
              p, i + 1, l.body_end,
              {from[0] + l.step[0] * k, from[1] + l.step[1] * k, from[2] + l.step[2] * k}, arg_of,
              out)) {
        return false;
      }
    }
    i = l.body_end - 1;
  }
  return true;
}

// Sets p.calls and p.call_args when the body unrolls (program.hpp).
void mark_calls(program& p) {
  p.calls.clear();
  p.call_args.clear();
  std::vector<std::size_t> arg_of(p.records.size(), 0);
  std::vector<pass_args> args;
  for (std::size_t i = p.batch_loops; i < p.records.size(); ++i) {
    if (const stage* st = std::get_if<stage>(&p.records[i])) {
      arg_of[i] = args.size();
      args.push_back(st->args);
    }
  }
  std::vector<call> calls;
  if (unroll_records(p, p.batch_loops, p.records.size(), {}, arg_of, calls)) {
    p.calls = std::move(calls);
    p.call_args = std::move(args);
  }
}

} // namespace

void run_flat_first(const program& p, const void* in, void* out, std::byte* a, std::byte* b,
                    std::byte* block) noexcept {
  run_flat_one(p, static_cast<const std::byte*>(in), static_cast<std::byte*>(out), a, b, block);
}

void mark_flat(program& p, bool unroll) {
  p.calls.clear();
  p.call_args.clear();
  if (p.lanes != 0) {
    // Blocks of transforms (run_lane_blocks), never a transform at a time.
    p.flat = false;
    p.leaf_rows = false;
    return;
  }
  const std::size_t stages = p.records.size() - p.batch_loops;
  p.flat = stages <= flat_stages_max;
  for (std::size_t i = p.batch_loops; i < p.records.size(); ++i) {
    // Only the lines loops of a grid move twiddles (program.hpp); run_flat does not.
    const stage* st = std::get_if<stage>(&p.records[i]);
    p.flat = p.flat && st != nullptr && !st->moving_twiddles;
  }
  p.flat_body.clear();
  if (p.flat) {
    for (std::size_t i = p.batch_loops; i < p.records.size(); ++i) {
      const stage& st = *std::get_if<stage>(&p.records[i]);
      p.flat_body.push_back({
          .run = st.run,
          .leaf = st.leaf,
          .units = st.units,
          .src = st.src,
          .dst = st.dst,
          .args = st.args,
      });
    }
  }
  const auto moves = [](slot x) { return x == slot::in || x == slot::out; };
  const stage* only =
      p.flat && stages == 1 && p.batch_loops == 1 ? std::get_if<stage>(&p.records[1]) : nullptr;
  p.leaf_rows = only != nullptr && only->leaf != nullptr && moves(only->src) && moves(only->dst);
  if (p.leaf_rows) {
    // Only in and out move with the batch; the destination is never the input.
    const loop& l = *std::get_if<loop>(p.records.data());
    p.rows_from_in = only->src == slot::in;
    p.rows_leaf = only->leaf;
    p.rows_src_step = p.rows_from_in ? l.step[0] : l.step[1];
    p.rows_dst_step = l.step[1];
    p.rows_args = only->args;
  }
  if (!p.flat && unroll) {
    mark_calls(p);
  }
}

void run_stage_chunks(const stage& st, const slot_addresses& at, std::size_t first,
                      std::size_t last) noexcept {
  if (st.leaf != nullptr) {
    // One unit, the line: run when the chunks include it.
    if (first == 0 && last != 0) {
      run_leaf(st, at);
    }
    return;
  }
  // Chunks [first, last) are adjacent ranges of units, and a unit's result does not depend
  // on the range it is run in, so one call over their union gives the same bits as one call
  // per chunk. The ends are found without dividing in the common case of the whole stage:
  // a 64-bit division costs 40 to 90 cycles on Cascade Lake, and four per chunk were a third
  // of a 64-point transform there (dev/experiments/X18-x86-kernels.md).
  const std::size_t begin = first == 0 ? 0 : chunk_start(st.units, st.chunks, first);
  const std::size_t end = last == st.chunks ? st.units : chunk_start(st.units, st.chunks, last);
  if (begin >= end) {
    return;
  }
  const void* src = source_of(st.src, at);
  void* dst = destination_of(st.dst, at);
  if (st.moving_twiddles) {
    pass_args args = st.args;
    args.twiddles = static_cast<const std::byte*>(args.twiddles) + at.tw;
    st.run(&args, src, dst, begin, end);
    return;
  }
  st.run(&st.args, src, dst, begin, end);
}

slot_addresses transform_addresses(const program& p, std::size_t t,
                                   const slot_addresses& base) noexcept {
  return moved_in_nest(p, 0, p.batch_loops, t, base);
}

void run_transforms(const program& p, std::size_t first, std::size_t count,
                    const slot_addresses& base) noexcept {
  if (count == 0) {
    return; // forms no address past the batch (the fast paths below would)
  }
  if (p.lanes != 0) {
    run_lane_blocks(p, first, count, base);
    return;
  }
  if (p.leaf_rows) {
    run_leaf_rows(p, first, count, base.in, base.out);
    return;
  }
  if (p.flat) {
    if (count == 1) {
      const slot_addresses at = transform_addresses(p, first, base);
      run_flat_one(p, at.in, at.out, at.a, at.b, at.block);
    } else {
      run_flat(p, first, count, base);
    }
    return;
  }
  if (!p.calls.empty()) {
    for (std::size_t t = first; t < first + count; ++t) {
      run_calls(p, transform_addresses(p, t, base));
    }
    return;
  }
  for (std::size_t t = first; t < first + count; ++t) {
    run_records(p, p.batch_loops, p.records.size(), transform_addresses(p, t, base));
  }
}

void run_lines(const program& p, std::size_t begin, std::size_t nest, std::size_t body_end,
               std::size_t first, std::size_t last, const slot_addresses& at) noexcept {
  slot_addresses from = at;
  for (std::size_t i = begin; i < begin + nest; ++i) {
    from = moved(from, std::get_if<loop>(&p.records[i])->start, 1);
  }
  for (std::size_t line = first; line < last; ++line) {
    run_records(p, begin + nest, body_end, moved_in_nest(p, begin, nest, line, from));
  }
}

} // namespace fftn::inline v0::detail
