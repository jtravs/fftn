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

void run_stage(const stage& st, const slot_addresses& at) noexcept {
  pass_args args = st.args;
  args.src = source_of(st.src, at);
  args.dst = destination_of(st.dst, at);
  for (std::size_t c = 0; c < st.chunks; ++c) {
    // Chunk c is [units c / chunks, units (c + 1) / chunks): fixed by the plan, so a split
    // across threads (phase 3) runs exactly these ranges.
    const std::size_t first = st.units / st.chunks * c + st.units % st.chunks * c / st.chunks;
    const std::size_t next = c + 1;
    const std::size_t last = st.units / st.chunks * next + st.units % st.chunks * next / st.chunks;
    st.run(&args, first, last);
  }
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
      run_stage(*st, at);
      continue;
    }
    const loop& l = *std::get_if<loop>(&r);
    for (std::size_t c = 0; c < l.count; ++c) {
      run_records(p, i + 1, l.body_end, moved(at, l.step, c));
    }
    i = l.body_end - 1;
  }
}

} // namespace

void run_transforms(const program& p, std::size_t first, std::size_t count,
                    const slot_addresses& base) noexcept {
  const std::size_t body = p.batch_loops;
  for (std::size_t t = first; t < first + count; ++t) {
    // Transform t's position in the batch loops, the innermost varying fastest.
    slot_addresses at = base;
    std::size_t rest = t;
    for (std::size_t i = body; i > 0; --i) {
      const loop& l = *std::get_if<loop>(&p.records[i - 1]);
      at = moved(at, l.step, rest % l.count);
      rest /= l.count;
    }
    run_records(p, body, p.records.size(), at);
  }
}

} // namespace fftn::inline v0::detail
