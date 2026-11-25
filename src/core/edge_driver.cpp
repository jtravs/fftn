// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: Apache-2.0

#include "core/edge_driver.hpp"

#include "core/driver.hpp"
#include "core/edges.hpp"
#include "core/program.hpp"
#include "fftn/edges.hpp"
#include "passes/pass_args.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace fftn::inline v0::detail {
namespace {

constexpr const char* missing_array = "fftn: a plan whose edges have factors was executed "
                                      "without the factor arrays it needs (exec_args)";

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

// driver.cpp's moved(): a slot a loop does not move may be null, and takes no offset.
slot_addresses moved(const slot_addresses& at, const std::array<std::ptrdiff_t, moving_slots>& step,
                     std::size_t times) noexcept {
  const auto k = static_cast<std::ptrdiff_t>(times);
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

factor_cursors moved(factor_cursors f, const std::array<std::ptrdiff_t, 2>& step,
                     std::size_t times) noexcept {
  const auto k = static_cast<std::ptrdiff_t>(times);
  f.in += static_cast<std::uintptr_t>(step[0] * k);
  f.out += static_cast<std::uintptr_t>(step[1] * k);
  return f;
}

// Chunks [first, last) of stage st (record e), as driver.cpp's run_stage_chunks runs them, with
// the cursors in the arguments of a stage that reads them and its kernels from e.
void run_stage(const stage& st, const edge_record& e, const slot_addresses& at, factor_cursors f,
               std::size_t first, std::size_t last) noexcept {
  if (st.moving_factors == 0) {
    run_stage_chunks(st, at, first, last);
    return;
  }
  pass_args args = st.args;
  if (st.moving_twiddles) {
    args.twiddles = static_cast<const std::byte*>(args.twiddles) + at.tw;
  }
  if ((st.moving_factors & moving_in_factor) != 0) {
    args.in_factor = reinterpret_cast<const void*>(f.in);
  }
  if ((st.moving_factors & moving_out_factor) != 0) {
    args.out_factor = reinterpret_cast<const void*>(f.out);
  }
  const void* src = source_of(st.src, at);
  void* dst = destination_of(st.dst, at);
  if (e.leaf != nullptr) {
    if (first == 0 && last != 0) {
      e.leaf(&args, src, dst, 0, 0, 1);
    }
    return;
  }
  const std::size_t begin = chunk_start(st.units, st.chunks, first);
  const std::size_t end = chunk_start(st.units, st.chunks, last);
  if (begin < end) {
    e.run(&args, src, dst, begin, end);
  }
}

void run_records(const program& p, const std::vector<edge_record>& e, std::size_t begin,
                 std::size_t end, const slot_addresses& at, factor_cursors f) noexcept {
  for (std::size_t i = begin; i < end; ++i) {
    const record& r = p.records[i];
    if (const stage* st = std::get_if<stage>(&r)) {
      run_stage(*st, e[i], at, f, 0, st->chunks);
      continue;
    }
    // Every line on its own: a loop of one leaf gives each line the bits of the one call
    // driver.cpp makes of it (experiment X40).
    const loop& l = *std::get_if<loop>(&r);
    const slot_addresses from = moved(at, l.start, 1);
    for (std::size_t c = 0; c < l.count; ++c) {
      run_records(p, e, i + 1, l.body_end, moved(from, l.step, c), moved(f, e[i].factor_step, c));
    }
    i = l.body_end - 1;
  }
}

// Iteration `index` of the loops [begin, begin + depth), the last varying fastest.
std::pair<slot_addresses, factor_cursors> moved_in_nest(const program& p,
                                                        const std::vector<edge_record>& e,
                                                        std::size_t begin, std::size_t depth,
                                                        std::size_t index, const slot_addresses& at,
                                                        factor_cursors f) noexcept {
  slot_addresses here = at;
  std::size_t rest = index;
  for (std::size_t i = begin + depth; i > begin && rest != 0; --i) {
    const loop& l = *std::get_if<loop>(&p.records[i - 1]);
    here = moved(here, l.step, rest % l.count);
    f = moved(f, e[i - 1].factor_step, rest % l.count);
    rest /= l.count;
  }
  return {here, f};
}

} // namespace

void factor_guard_pass(const pass_args* /*args*/, const void* /*src*/, void* /*dst*/,
                       std::size_t /*first*/, std::size_t /*last*/) {
  contract_failure(missing_array);
}

void factor_guard_leaf(const pass_args* /*args*/, const void* /*src*/, void* /*dst*/,
                       std::ptrdiff_t /*src_dist*/, std::ptrdiff_t /*dst_dist*/,
                       std::size_t /*count*/) {
  contract_failure(missing_array);
}

factor_cursors cursors_of(const exec_args& args, std::uint8_t needed) noexcept {
  const bool in = (needed & moving_in_factor) != 0;
  const bool out = (needed & moving_out_factor) != 0;
  if ((in && args.in_factor == nullptr) || (out && args.out_factor == nullptr)) {
    contract_failure(missing_array);
  }
  return {
      .in = in ? reinterpret_cast<std::uintptr_t>(args.in_factor) : 0,
      .out = out ? reinterpret_cast<std::uintptr_t>(args.out_factor) : 0,
  };
}

factor_cursors edge_transform_cursors(const program& p, const std::vector<edge_record>& e,
                                      std::size_t t, factor_cursors f) noexcept {
  std::size_t rest = t;
  for (std::size_t i = p.batch_loops; i > 0 && rest != 0; --i) {
    const loop& l = *std::get_if<loop>(&p.records[i - 1]);
    f = moved(f, e[i - 1].factor_step, rest % l.count);
    rest /= l.count;
  }
  return f;
}

void run_edge_transforms(const program& p, const std::vector<edge_record>& e, std::size_t first,
                         std::size_t count, const slot_addresses& base, factor_cursors f) noexcept {
  for (std::size_t t = first; t < first + count; ++t) {
    run_records(p, e, p.batch_loops, p.records.size(), transform_addresses(p, t, base),
                edge_transform_cursors(p, e, t, f));
  }
}

void run_edge_stage_chunks(const program& p, const std::vector<edge_record>& e, std::size_t index,
                           const slot_addresses& at, factor_cursors f, std::size_t first,
                           std::size_t last) noexcept {
  run_stage(*std::get_if<stage>(&p.records[index]), e[index], at, f, first, last);
}

void run_edge_lines(const program& p, const std::vector<edge_record>& e, std::size_t begin,
                    std::size_t nest, std::size_t body_end, std::size_t first, std::size_t last,
                    const slot_addresses& at, factor_cursors f) noexcept {
  slot_addresses from = at;
  for (std::size_t i = begin; i < begin + nest; ++i) {
    from = moved(from, std::get_if<loop>(&p.records[i])->start, 1);
  }
  for (std::size_t line = first; line < last; ++line) {
    const auto [here, cursors] = moved_in_nest(p, e, begin, nest, line, from, f);
    run_records(p, e, begin + nest, body_end, here, cursors);
  }
}

} // namespace fftn::inline v0::detail
