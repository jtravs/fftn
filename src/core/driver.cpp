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
  for (std::size_t i = begin + depth; i > begin; --i) {
    const loop& l = *std::get_if<loop>(&p.records[i - 1]);
    here = moved(here, l.step, rest % l.count);
    rest /= l.count;
  }
  return here;
}

} // namespace

void run_stage_chunks(const stage& st, const slot_addresses& at, std::size_t first,
                      std::size_t last) noexcept {
  pass_args args = st.args;
  args.src = source_of(st.src, at);
  args.dst = destination_of(st.dst, at);
  for (std::size_t c = first; c < last; ++c) {
    // Fixed by the plan, so a split across threads runs exactly these ranges.
    st.run(&args, chunk_start(st.units, st.chunks, c), chunk_start(st.units, st.chunks, c + 1));
  }
}

slot_addresses transform_addresses(const program& p, std::size_t t,
                                   const slot_addresses& base) noexcept {
  return moved_in_nest(p, 0, p.batch_loops, t, base);
}

void run_transforms(const program& p, std::size_t first, std::size_t count,
                    const slot_addresses& base) noexcept {
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
