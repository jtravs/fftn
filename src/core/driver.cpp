// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/driver.hpp"

#include "core/program.hpp"
#include "passes/pass_args.hpp"

#include <array>
#include <cstddef>
#include <memory>
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

// One stage's arguments on run_flat's stack, left uninitialised until copied from the stage:
// default-initialising flat_stages_max of them would cost more than a tiny transform.
union stage_args {
  pass_args args;
  stage_args() noexcept {} // NOLINT(modernize-use-equals-default): must not initialise args
};

// Runs stage s on the slots at: run_stage_chunks over every chunk, without its chunk ends.
// a is a copy of s.args, used only by a pass_fn.
void run_whole(const stage& s, pass_args& a, const slot_addresses& at) noexcept {
  if (s.leaf != nullptr) {
    s.leaf(&s.args, source_of(s.src, at), destination_of(s.dst, at), 0, 0, 1);
    return;
  }
  a.src = source_of(s.src, at);
  a.dst = destination_of(s.dst, at);
  if (s.units != 0) {
    s.run(&a, 0, s.units);
  }
}

// Transforms [first, first + count) of a flat program (program.hpp). run_records walks the
// records, copies each stage's pass_args (144 bytes) and finds its chunk ends for every
// transform: about 3.5 ns per transform on the M5 Max, more than an 8-point leaf's own time
// (dev/experiments/X20-call-overhead.md). Here a batch of leaves is one call on the plan's
// own arguments; one transform copies each stage's arguments once; a batch copies them once
// for all its transforms and moves the slots by the batch loop's step. Every kernel gets the
// arguments and units run_records gives it (the whole stage: chunk ends 0 and units), so the
// bits are the same. The three forms are each the fastest of the ones X20 measured for their
// case: an array of copies costs about 6 ns more than plain copies at one transform.
void run_flat(const program& p, std::size_t first, std::size_t count,
              const slot_addresses& base) noexcept {
  const std::size_t stages = p.records.size() - p.batch_loops;
  const loop* only = p.batch_loops == 1 ? std::get_if<loop>(&p.records[0]) : nullptr;
  if (stages == 1 && only != nullptr) {
    const stage& s = *std::get_if<stage>(&p.records[1]);
    const auto moves = [](slot x) { return x == slot::in || x == slot::out; };
    if (s.leaf != nullptr && moves(s.src) && moves(s.dst)) {
      const slot_addresses at = moved(base, only->step, first);
      const std::ptrdiff_t src_dist = s.src == slot::in ? only->step[0] : only->step[1];
      s.leaf(&s.args, source_of(s.src, at), destination_of(s.dst, at), src_dist, only->step[1],
             count);
      return;
    }
  }
  if (count == 1) {
    const slot_addresses at = transform_addresses(p, first, base);
    for (std::size_t i = p.batch_loops; i < p.records.size(); ++i) {
      const stage& s = *std::get_if<stage>(&p.records[i]);
      if (s.leaf != nullptr) {
        s.leaf(&s.args, source_of(s.src, at), destination_of(s.dst, at), 0, 0, 1);
        continue;
      }
      pass_args a = s.args;
      run_whole(s, a, at);
    }
    return;
  }
  std::array<const stage*, flat_stages_max> st{};
  std::array<stage_args, flat_stages_max> args;
  for (std::size_t k = 0; k < stages; ++k) {
    st[k] = std::get_if<stage>(&p.records[p.batch_loops + k]);
    std::construct_at(&args[k].args, st[k]->args);
  }
  // One batch loop (rows): move by its step; several: from the transform's index.
  slot_addresses at = transform_addresses(p, first, base);
  for (std::size_t t = first; t < first + count; ++t) {
    if (t != first) {
      at = only != nullptr ? moved(at, only->step, 1) : transform_addresses(p, t, base);
    }
    for (std::size_t k = 0; k < stages; ++k) {
      run_whole(*st[k], args[k].args, at);
    }
  }
}

} // namespace

void mark_flat(program& p) noexcept {
  const std::size_t stages = p.records.size() - p.batch_loops;
  p.flat = stages <= flat_stages_max;
  for (std::size_t i = p.batch_loops; i < p.records.size(); ++i) {
    // Only the lines loops of a grid move twiddles (program.hpp); run_flat does not.
    const stage* st = std::get_if<stage>(&p.records[i]);
    p.flat = p.flat && st != nullptr && !st->moving_twiddles;
  }
}

void run_stage_chunks(const stage& st, const slot_addresses& at, std::size_t first,
                      std::size_t last) noexcept {
  if (st.leaf != nullptr) {
    // One unit, the line: run when the chunks include it.
    if (first == 0 && last != 0) {
      st.leaf(&st.args, source_of(st.src, at), destination_of(st.dst, at), 0, 0, 1);
    }
    return;
  }
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
  if (p.flat) {
    run_flat(p, first, count, base);
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
